"""Decision layers on top of the TCN meta-labeler: a neural-linear Thompson-sampling bandit, conformal abstention,
and an offline-RL (implicit Q-learning) exit policy.

**Neural-linear Thompson bandit** (Riquelme, Tucker & Snoek 2018). The network turns each candidate into features; a
Bayesian linear regression on top predicts R and keeps a full posterior. Each decision samples a plausible model from
the posterior (Thompson sampling): TAKE if the sampled R > ``take_r``, HALF if > 0, else SKIP. Every candidate's
shadow outcome (taken or not) updates the posterior, so a wrong SKIP is learned like a losing TAKE, and old evidence
fades with a half-life (the same forgetting as the tabular bandit). Unlike the tabular buckets it generalises across
similar situations instead of needing 20 outcomes per bucket.

**Conformal abstention** (split conformal prediction). On a rolling window of recent out-of-sample predictions, the
q-quantile of (predicted − realised R) gives a margin; a trade is allowed only if predicted R − margin > 0, i.e. it
is still profitable after the model's typical over-optimism. Distribution-free and calibrated by construction.

**Implicit Q-learning exits** (Kostrikov, Nair & Levine 2021). Offline RL that learns *when to close a position* from
logged price paths only (no live exploration): the state at each bar of an open trade (R so far, bars held, distance
to VWAP, last returns, time to close), actions {hold, exit}. Q(exit) is the R banked now after costs; Q(hold) is
learned by expectile regression toward the next state's value, which keeps it conservative about unseen actions.
The policy exits when Q(exit) ≥ Q(hold); the hard stop always stays in place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- neural-linear Thompson bandit
class NeuralLinearTS:
    def __init__(self, dim: int, prior: float = 1.0, noise: float = 1.0, half_life_days: float | None = 20.0,
                 take_r: float = 0.10, seed: int = 0):
        self.A = np.eye(dim) * prior                      # precision
        self.b = np.zeros(dim)
        self.noise, self.hl, self.take_r = noise, half_life_days, take_r
        self.rng = np.random.default_rng(seed)
        self.n = 0.0

    @staticmethod
    def features(df: pd.DataFrame, side: str) -> np.ndarray:
        e, p = df["e" + side].to_numpy(float), df["p" + side].to_numpy(float) if "p" + side in df else 0.5
        other = df["eS" if side == "L" else "eL"].to_numpy(float)
        return np.column_stack([np.ones(len(df)), e, p, e - other, np.abs(e), e * p])

    def end_of_day(self) -> None:
        if self.hl:
            f = 0.5 ** (1.0 / self.hl)
            self.A = np.eye(len(self.b)) * (1 - f) + self.A * f   # fade toward the prior
            self.b *= f
            self.n *= f

    def update(self, X: np.ndarray, r: np.ndarray) -> None:
        ok = np.isfinite(r)
        X, r = X[ok], np.clip(r[ok], -1.5, 2.5)
        self.A += X.T @ X / self.noise
        self.b += X.T @ r / self.noise
        self.n += len(r)

    def sample_r(self, X: np.ndarray) -> np.ndarray:
        cov = np.linalg.inv(self.A)
        theta = self.rng.multivariate_normal(cov @ self.b, cov)
        return X @ theta

    def mean_r(self, X: np.ndarray) -> np.ndarray:
        return X @ np.linalg.solve(self.A, self.b)

    def decide(self, X: np.ndarray) -> np.ndarray:
        s = self.sample_r(X)
        return np.where(s > self.take_r, "TAKE", np.where(s > 0, "HALF", "SKIP"))


# --------------------------------------------------------------------------- conformal abstention
class ConformalGate:
    """Allows a trade only if predicted R minus the q-quantile of recent over-prediction is still > 0."""

    def __init__(self, q: float = 0.8, window_days: int = 10):
        self.q, self.window = q, window_days
        self.res: list[np.ndarray] = []

    def margin(self) -> float:
        if not self.res:
            return float("inf")
        r = np.concatenate(self.res[-self.window:])
        return float(np.quantile(r, self.q)) if len(r) > 30 else float("inf")

    def allow(self, pred: np.ndarray) -> np.ndarray:
        return pred - self.margin() > 0

    def update(self, pred: np.ndarray, realised: np.ndarray) -> None:
        ok = np.isfinite(pred) & np.isfinite(realised)
        self.res.append((pred - realised)[ok])


# --------------------------------------------------------------------------- implicit Q-learning exits
EXIT_FEATS = ["r_now", "held", "vwap_dist", "ret1", "ret3", "to_close", "side"]


def exit_paths(blocks, picks: pd.DataFrame, cfg: dict, horizon: int = 12) -> pd.DataFrame:
    """For chosen entries (ticker, ts, side), every bar of the open trade: state, R if closed now after costs, and
    whether the hard stop / target / time barrier ended it. ``picks`` needs ticker, ts (signal bar), side ('L'/'S')."""
    from .dataset import _costs
    g = cfg["guardrails"]
    key = {(b.ticker, b.day): b for b in blocks}
    rows = []
    for tid, p in enumerate(picks.itertuples()):
        b = key.get((p.ticker, pd.Timestamp(p.ts).normalize()))
        if b is None:
            continue
        i = int(np.searchsorted(b.ts, pd.Timestamp(p.ts)))
        if i + 1 >= len(b.close) or not np.isfinite(b.dist[i]):
            continue
        sd = 1 if p.side == "L" else -1
        o = b.close                                       # closes; entry approximated at next bar via seq
        fill = float(b.close[i])                          # signal close ≈ next open (gap-free approximation)
        d = float(b.dist[i])
        stop, tgt = fill - sd * d, fill + sd * g["target_r"] * d
        qty = max(1.0, np.floor(cfg["pool"] * g["risk_per_trade"] / d))
        for j in range(1, horizon + 1):
            k = i + j
            if k >= len(o):
                break
            px = float(o[k])
            r_now = sd * (px - fill) / d - _costs(np.array([fill]), np.array([px]), np.array([qty]), sd, cfg["costs"])[0] / (d * qty)
            hit_stop = (sd > 0 and px <= stop) or (sd < 0 and px >= stop)
            hit_tgt = (sd > 0 and px >= tgt) or (sd < 0 and px <= tgt)
            done = hit_stop or hit_tgt or j == horizon or (b.ctx[k, 0] * 375 + 9 * 60 + 15 >= 15 * 60 + 15)
            rows.append({"trade": tid, "j": j, "r_now": r_now, "held": j / horizon,
                         "vwap_dist": float(b.seq[k, 4]) * sd, "ret1": float(b.seq[k, 0]) * sd,
                         "ret3": float(b.seq[max(k - 2, 0):k + 1, 0].sum()) * sd, "to_close": 1 - float(b.ctx[k, 0]),
                         "side": float(sd), "done": bool(done)})
            if done:
                break
    return pd.DataFrame(rows)


class IQLExit:
    """Fitted Q-iteration with expectile value regression over logged paths (actions: hold / exit)."""

    def __init__(self, expectile: float = 0.7, gamma: float = 1.0, iters: int = 8, seed: int = 0):
        self.tau, self.gamma, self.iters, self.seed = expectile, gamma, iters, seed
        self.q_hold = None

    def fit(self, paths: pd.DataFrame) -> "IQLExit":
        import lightgbm as lgb
        p = paths.sort_values(["trade", "j"]).reset_index(drop=True)
        X = p[EXIT_FEATS].to_numpy(float)
        nxt_same = (p["trade"].shift(-1) == p["trade"]).to_numpy() & ~p["done"].to_numpy()
        r_exit = p["r_now"].to_numpy(float)
        v = r_exit.copy()                                     # initial V = bank now
        for _ in range(self.iters):
            v_next = np.r_[v[1:], 0.0]
            target = np.where(nxt_same, self.gamma * v_next, r_exit)      # holding through the last bar = forced exit
            m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                                  objective="quantile", alpha=self.tau, random_state=self.seed, verbose=-1)
            m.fit(X, target)
            q_hold = m.predict(X)
            v = np.maximum(r_exit, np.where(nxt_same, q_hold, r_exit))
        self.q_hold = m
        return self

    def exit_now(self, state: pd.DataFrame) -> np.ndarray:
        return state["r_now"].to_numpy(float) >= self.q_hold.predict(state[EXIT_FEATS].to_numpy(float))

    def evaluate(self, paths: pd.DataFrame) -> dict:
        """Realised R per trade: fixed barriers (ride to the end) vs exiting when the policy says so."""
        p = paths.sort_values(["trade", "j"])
        ex = self.exit_now(p) | p["done"].to_numpy()
        p = p.assign(ex=ex)
        first = p[p["ex"]].groupby("trade").first()["r_now"]
        fixed = p.groupby("trade").last()["r_now"]
        j = pd.concat([fixed.rename("fixed"), first.rename("policy")], axis=1).dropna()
        d = j["policy"] - j["fixed"]
        return {"trades": int(len(j)), "fixed avg R": float(j["fixed"].mean()), "policy avg R": float(j["policy"].mean()),
                "diff": float(d.mean()), "t": float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else np.nan,
                "avg bars held (policy)": float(p[p["ex"]].groupby("trade")["j"].first().mean())}
