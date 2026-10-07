"""Intraday agent v2: deep-learning entries, run live alongside v1 as a separate paper account.

Each finished 5-minute bar: the TCN meta-labeler scores every in-play stock's bars (long and short, E[R] after costs);
candidates are walked in time order and taken when the predicted R clears a threshold calibrated on recent days
(about ``target_trades`` a day), optionally sized by the neural-linear bandit and gated by conformal abstention, as
``report/results/intraday_v2_decision.json`` selected. Stops are the label geometry (ATR-based, wide enough that costs
are ≤ 0.2R), target 2R, time exit after 60 minutes, optional learned (IQL) exits. Same Account, costs, guardrails and
tax as v1, in its own folder (data/intraday/v2/) so the two engines can be compared day by day.

After the close: the day's bars are archived (history grows past Yahoo's 60 days), the bandit / conformal gate learn
every candidate's outcome, and the TCN is refit once a week on all archived days.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from . import DIR
from .agent import in_play_matrix
from .dl import dataset as DS
from .dl import tcn as T
from .dl.policies import ConformalGate, NeuralLinearTS
from .setups import Signal
from .sim import Account

V2 = DIR / "v2"
MODEL = DIR / "dl" / "tcn_latest.pkl"


def save_model(m: T.Model, trained_days: list[str]) -> None:
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL, "wb") as f:
        pickle.dump({"state": m.net.state_dict(), "cal": m.cal, "n_train": m.n_train, "days": trained_days,
                     "when": time.strftime("%Y-%m-%d %H:%M")}, f)


def load_model() -> tuple[T.Model | None, dict]:
    if not MODEL.exists():
        return None, {}
    with open(MODEL, "rb") as f:
        d = pickle.load(f)
    net = T.TCN()
    net.load_state_dict(d["state"])
    return T.Model(net, d["cal"], d["n_train"], []), {k: d[k] for k in ("days", "when", "n_train")}


def refit_if_due(cfg: dict, every_days: int = 5, force: bool = False) -> str:
    """Refit the TCN on every archived day when the model is older than ``every_days`` trading days."""
    _, meta = load_model()
    bars = DS.load_bars()
    days = sorted(bars["ts"].dt.normalize().unique())
    trained = set(meta.get("days", []))
    new_days = [d for d in days if str(pd.Timestamp(d).date()) not in trained]
    if not force and meta and len(new_days) < every_days:
        return f"model current ({len(new_days)} new day(s) since {meta.get('when')})"
    blocks = DS.build(bars, cfg)
    m = T.fit(blocks)
    save_model(m, [str(pd.Timestamp(d).date()) for d in days])
    return f"refit on {len(days)} days ({m.n_train:,} samples)"


class EngineV2:
    def __init__(self, cfg: dict, root=V2, seed: int = 0):
        self.cfg = json.loads(json.dumps(cfg))                     # own copy
        self.cfg["guardrails"]["time_exit_bars"] = DS.H
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = self._load("state.json", {"pool": float(cfg["pool"]), "start_pool": float(cfg["pool"]),
                                                "days": [], "recent_e": []})
        self.model, self.meta = load_model()
        dec = {}
        try:
            from ..config import ROOT
            dec = json.loads((ROOT / "report" / "results" / "intraday_v2_decision.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        best = dec.get("best", "")
        self.use_bandit, self.use_conformal = "bandit" in best, "conformal" in best
        self.policy = best or "model: top-5 threshold (causal)"
        self.bandit = self._load_pickle("bandit.pkl") or NeuralLinearTS(6, seed=seed)
        self.gate = self._load_pickle("gate.pkl") or ConformalGate()
        self.target_trades = 5

    # -------------------------------------------------------------- persistence
    def _load(self, name, default):
        p = self.root / name
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
        except (ValueError, OSError):
            return default

    def _save(self, name, obj):
        p = self.root / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
        tmp.replace(p)

    def _load_pickle(self, name):
        p = self.root / name
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            return None

    def _dump(self, name, obj):
        with open(self.root / name, "wb") as f:
            pickle.dump(obj, f)

    # -------------------------------------------------------------- predictions → signals
    def threshold(self) -> float:
        if "R > 0" in self.policy:                                  # measured best: trade only positive predicted R
            return 0.0
        e = np.asarray(self.state.get("recent_e", []), float)
        per_day = len(e) / max(len(self.state.get("recent_days", [])) or 1, 1)
        if len(e) < 200:
            return 0.0                                              # no history yet: only positive predicted R
        return float(np.quantile(e, 1 - min(1.0, self.target_trades / max(per_day, 1))))

    def candidates(self, bars: dict[str, pd.DataFrame], ctx: dict[str, dict], nifty: pd.DataFrame | None) -> pd.DataFrame:
        """Model predictions for every finished bar of the stocks in play (from the bar they joined)."""
        if self.model is None:
            return pd.DataFrame()
        tick, S = in_play_matrix(bars, ctx, self.cfg["universe"])
        joined: dict[str, int] = {}
        for k in range(2, S.shape[1] if S.size else 0):
            col = S[:, k]
            for i in np.flatnonzero(np.isfinite(col))[np.argsort(-col[np.isfinite(col)])]:
                if tick[i] not in joined and len(joined) < self.cfg["universe"]["in_play_top"]:
                    joined[tick[i]] = k
        blocks, start = [], {}
        for t, k0 in joined.items():
            b = bars[t].copy()
            b.attrs["ticker"] = t
            blk = DS.stock_day(b, ctx.get(t, {}), nifty, self.cfg)
            if blk is not None:
                blocks.append(blk)
                start[t] = k0
        if not blocks:
            return pd.DataFrame()
        pr = T.predict_blocks(self.model, blocks)
        pr = pr[[b >= start.get(t, 0) for t, b in zip(pr["ticker"], pr["bar"])]]
        mins = pd.to_datetime(pr["ts"]).dt.hour * 60 + pd.to_datetime(pr["ts"]).dt.minute
        pr = pr[(mins >= 9 * 60 + 30) & (mins <= DS.LAST_ENTRY_MIN) & np.isfinite(pr["dist"].to_numpy(float))]
        pr["side"] = np.where(pr["eL"] >= pr["eS"], "L", "S")
        pr["e"] = np.where(pr["side"] == "L", pr["eL"], pr["eS"])
        pr["p"] = np.where(pr["side"] == "L", pr["pL"], pr["pS"])
        return pr

    def decide(self, cand: pd.DataFrame, frozen: dict) -> list[Signal]:
        """Time-ordered decisions, frozen once made (live re-runs the day every bar)."""
        thr, margin = self.threshold(), self.gate.margin() if self.use_conformal else -np.inf
        sigs = []
        for r in cand.sort_values("ts").itertuples():
            key = f"{r.ticker}|{r.ts}|{r.side}"
            if key not in frozen:
                take = r.e >= thr and r.e - margin > 0
                size = 1.0
                if take and self.use_bandit:
                    a = self.bandit.decide(NeuralLinearTS.features(pd.DataFrame([r._asdict()]), r.side))[0]
                    size = 1.0 if a == "TAKE" else 0.5 if a == "HALF" else 0.0
                frozen[key] = {"take": bool(take and size > 0), "size": size, "e": float(r.e), "p": float(r.p)}
            d = frozen[key]
            if not d["take"]:
                continue
            side = "long" if r.side == "L" else "short"
            stop = r.close - r.dist if side == "long" else r.close + r.dist
            s = Signal(r.ticker, pd.Timestamp(r.ts) + pd.Timedelta(minutes=5), int(r.bar), "dl", side, float(r.close),
                       float(stop), {"setup": "dl", "side": side, "e": d["e"], "p": d["p"]})
            sigs.append(s)
        return sigs

    def run(self, bars, ctx, nifty, frozen: dict, close_at_end: bool):
        cand = self.candidates(bars, ctx, nifty)
        sigs = self.decide(cand, frozen) if len(cand) else []
        lookup = {(s.ticker, str(s.ts)): frozen.get(f"{s.ticker}|{s.ts - pd.Timedelta(minutes=5)}|{'L' if s.side == 'long' else 'S'}", {})
                  for s in sigs}

        def decide_fn(s):
            d = lookup.get((s.ticker, str(s.ts)), {})
            act = "TAKE" if d.get("size", 1.0) >= 1 else "HALF"
            return act, d.get("p", float("nan")), f"v2: predicted {d.get('e', 0):+.2f}R after costs"
        acct = Account(self.cfg, float(self.state["pool"]))
        trades, log = acct.run_day(sigs, bars, decide_fn, close_at_end=close_at_end)
        return cand, trades, log, acct

    # -------------------------------------------------------------- end of day
    def settle(self, day: str, bars, ctx, nifty, frozen: dict) -> dict:
        cand, trades, log, _ = self.run(bars, ctx, nifty, frozen, close_at_end=True)
        net = sum(t.net for t in trades)
        pool0 = float(self.state["pool"])
        self.state["pool"] = pool0 + net
        lab = cand.assign(r=np.where(cand["side"] == "L", cand["rL"], cand["rS"])).dropna(subset=["r"]) if len(cand) else cand
        if len(lab):
            X = np.vstack([NeuralLinearTS.features(lab.iloc[[i]], s)[0] for i, s in enumerate(lab["side"])])
            self.bandit.update(X, lab["r"].to_numpy(float))
            self.gate.update(lab["e"].to_numpy(float), lab["r"].to_numpy(float))
        self.bandit.end_of_day()
        self.state["recent_e"] = (list(self.state.get("recent_e", [])) + [float(x) for x in cand.get("e", [])])[-20000:]
        self.state["recent_days"] = (list(self.state.get("recent_days", [])) + [day])[-5:]
        rec = {"day": day, "pool_start": pool0, "pool_end": self.state["pool"], "trades": len(trades), "net": round(net, 2),
               "gross": round(sum(t.gross for t in trades), 2), "costs": round(sum(t.costs for t in trades), 2),
               "candidates": int(len(cand)), "avg_r": float(np.mean([t.r for t in trades])) if trades else None,
               "candidate_avg_r": float(lab["r"].mean()) if len(lab) else None}
        self.state["days"] = [d for d in self.state["days"] if d["day"] != day] + [rec]
        self._save("state.json", self.state)
        self._dump("bandit.pkl", self.bandit)
        self._dump("gate.pkl", self.gate)
        tp = self.root / "trades.csv"
        new = pd.DataFrame([t.to_dict() | {"day": day} for t in trades]).drop(columns=["cost_detail"], errors="ignore")
        if len(new):
            old = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
            old = old[old["day"] != day] if len(old) else old
            pd.concat([old, new], ignore_index=True).to_csv(tp, index=False)
        self._save(f"log_{day}.json", log)
        return rec
