"""Gymnasium environment for rebalancing an Indian equity portfolio.

Timeline of one step (``rebalance_every`` trading days):
  1. At the close of day t the agent picks target weights over N stocks + cash.
  2. The weights are projected onto the feasible set (stock/sector caps,
     minimum cash) and optionally reduced to ``max_stocks`` holdings.
  3. Trades are executed at the close price of t. India delivery costs,
     slippage and (in tax-aware mode) capital-gains tax are paid from cash.
  4. Holdings drift with daily prices until the next rebalance. Cash earns
     the T-bill proxy rate, and SIP inflows arrive on the first trading day
     of each month.

The reward is computed on the flow-adjusted step return (NAV return), so SIP
contributions never count as performance.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .constraints import project
from .costs import IndiaCostModel
from .data import MarketData
from .features import FeatureSet
from .profile import PROFILE_DIM, InvestorProfile
from .rewards import RewardFn
from .tax import TaxLedger


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, md: MarketData, fs: FeatureSet, cfg: dict, split: str = "train",
                 profile: InvestorProfile | None = None, action_mode: str = "logits",
                 full_episode: bool = False, cost_scale: float = 1.0, tax_aware: bool | None = None,
                 random_init_weights: bool | None = None, seed: int | None = None):
        super().__init__()
        self.md, self.fs, self.cfg = md, fs, cfg
        self.ecfg = cfg["env"]
        self.fixed_profile = profile
        self.action_mode = action_mode
        self.full_episode = full_episode
        self.tax_aware = self.ecfg["tax_aware"] if tax_aware is None else tax_aware
        self.random_init = (not full_episode) if random_init_weights is None else random_init_weights
        self.costs = IndiaCostModel(cfg["costs"], scale=cost_scale)
        self.reward_fn = RewardFn(cfg["reward"])
        self.sector_ids = md.sector_ids
        self.rng = np.random.default_rng(seed)

        self.prices = md.close.to_numpy(dtype=np.float64)
        rets = md.close.pct_change().fillna(0.0)
        self.sigma = rets.rolling(20, min_periods=5).std().bfill().to_numpy()
        self.adv = (md.close * md.volume).rolling(20, min_periods=5).mean().bfill().to_numpy()
        self.dates = md.dates
        self.N = len(md.tickers)
        self.reb = int(self.ecfg["rebalance_every"])
        self.L = int(cfg["features"]["lookback"])
        self.stride = int(cfg["features"]["lookback_stride"])
        self.cash_daily = (1 + self.ecfg["cash_rate"]) ** (1 / 252) - 1

        lo_date, hi_date = cfg["splits"][split]
        mask = md.split_mask(lo_date, hi_date)
        idx = np.flatnonzero(mask)
        warm = fs.valid_from + (self.L - 1) * self.stride
        self.lo = max(int(idx[0]), warm)
        self.hi = int(idx[-1])

        F, M = fs.n_stock_features, fs.n_market_features
        self.observation_space = spaces.Dict({
            "stock": spaces.Box(-np.inf, np.inf, (self.L, self.N, F), np.float32),
            "market": spaces.Box(-np.inf, np.inf, (self.L, M), np.float32),
            "weights": spaces.Box(0.0, 1.0, (self.N + 1,), np.float32),
            "profile": spaces.Box(0.0, 1.0, (PROFILE_DIM,), np.float32),
        })
        if action_mode == "logits":
            self.action_space = spaces.Box(-5.0, 5.0, (self.N + 1,), np.float32)
        else:
            self.action_space = spaces.Box(0.0, 1.0, (self.N + 1,), np.float32)

    # ------------------------------------------------------------------ helpers
    def value(self, t: int | None = None) -> float:
        t = self.t if t is None else t
        return float(self.cash + self.shares @ self.prices[t])

    def current_weights(self) -> np.ndarray:
        v = self.value()
        return np.append(self.shares * self.prices[self.t], self.cash) / v

    def observation(self) -> dict:
        rows = self.t - self.stride * np.arange(self.L - 1, -1, -1)
        return {
            "stock": self.fs.stock[rows],
            "market": self.fs.market[rows],
            "weights": self.current_weights().astype(np.float32),
            "profile": self.profile.vector(self.horizon_left),
        }

    def _to_weights(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        if self.action_mode == "logits":
            return softmax(a)
        a = np.clip(a, 0.0, None)
        return a / a.sum() if a.sum() > 0 else np.eye(self.N + 1)[-1]

    def feasible(self, w: np.ndarray) -> np.ndarray:
        p = self.profile
        k = p.max_stocks
        if k is not None and k < self.N:
            keep = np.argsort(w[:self.N])[::-1][:k]
            mask = np.zeros(self.N + 1, bool)
            mask[keep] = True
            mask[self.N] = True
            w = np.where(mask, w, 0.0)
        return project(w, self.sector_ids, self.ecfg["max_stock_weight"],
                       self.ecfg["max_sector_weight"], p.min_cash)

    def _apply_trade_band(self, w_target: np.ndarray) -> np.ndarray:
        """No-trade band: leave a stock untouched if its change is below ``trade_band``.

        A continuous policy never outputs exactly the current weights. Without
        the band, it would trade every stock every week and pay the flat DP
        charge on each sale (29 stocks x Rs 16 x 52 weeks ~ Rs 24k/yr). Cash
        absorbs the difference. If that breaks a limit (cash floor, sector
        cap), the full projected target is used instead.
        """
        band = self.ecfg.get("trade_band", 0.0)
        if band <= 0:
            return w_target
        w_cur = self.current_weights()
        n = self.N
        skip = np.abs(w_target[:n] - w_cur[:n]) < band
        if not skip.any():
            return w_target
        w = w_target.copy()
        w[:n][skip] = w_cur[:n][skip]
        w[n] = 1.0 - w[:n].sum()
        sec = np.bincount(self.sector_ids, weights=w[:n])
        ok = (w[n] >= self.profile.min_cash - 1e-9 and np.all(w[:n] <= self.ecfg["max_stock_weight"] + 1e-9)
              and np.all(sec <= self.ecfg["max_sector_weight"] + 1e-9))
        return w if ok else w_target

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ep = int(self.ecfg["episode_steps"])
        if self.full_episode:
            self.t = self.lo
            self.max_steps = (self.hi - self.lo) // self.reb
        else:
            last = self.hi - ep * self.reb
            self.t = int(self.rng.integers(self.lo, max(self.lo + 1, last)))
            self.max_steps = ep
        self.profile = self.fixed_profile or InvestorProfile.sample(self.rng, self.cfg["profile_sampling"])
        self.horizon_left = float(self.profile.horizon_years)
        self.sip = self.profile.sip_monthly or self.ecfg["sip_monthly"]
        v0 = float(self.profile.initial)
        if self.random_init:
            w0 = self.feasible(self.rng.dirichlet(np.ones(self.N + 1)))
        else:
            w0 = np.eye(self.N + 1)[-1]
        self.shares = w0[:self.N] * v0 / self.prices[self.t]
        self.cash = w0[self.N] * v0
        self.ledger = TaxLedger(self.N, self.cfg["costs"]["tax"])
        for i in np.flatnonzero(self.shares > 0):
            self.ledger.buy(i, self.dates[self.t], self.shares[i], self.prices[self.t, i])
        self.nav, self.peak = 1.0, 1.0
        self.steps = 0
        self.reward_fn.reset()
        self.history = {"date": [self.dates[self.t]], "nav": [1.0], "value": [v0], "flows": [0.0]}
        self.trades = []
        return self.observation(), {}

    def _execute(self, w_target: np.ndarray) -> tuple[float, float, float, float]:
        """Trade to the target weights at today's close.

        Returns (cost, tax_paid, tax_liability, turnover). The ledger always
        tracks the liability, so after-tax results can be reported for every
        policy. Tax is deducted from cash only in tax-aware mode.
        """
        t, P = self.t, self.prices[self.t]
        V = self.value()
        cur_val = self.shares * P
        w_cur = np.append(cur_val, self.cash) / V
        turnover = 0.5 * float(np.abs(w_target - w_cur).sum())
        if turnover < 1e-9:
            return 0.0, 0.0, 0.0, 0.0
        min_trade = self.ecfg.get("min_trade_frac", 0.001) * V
        V_eff = V
        for _ in range(3):  # fixed point: costs (and tax) are paid out of the value being allocated
            delta = w_target[:self.N] * V_eff - cur_val
            # Drop dust trades. Otherwise shrinking every position by the cost
            # amount would sell a sliver of each untouched stock and pay a DP
            # charge on each one.
            delta[np.abs(delta) < min_trade] = 0.0
            buy, sell = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
            cost = self.costs.cost(buy, sell, self.sigma[t], self.adv[t])
            tax = max(self._tax_preview(sell / P), 0.0) if self.tax_aware else 0.0
            V_eff = V - cost.total - tax
        # Costs and tax come out of cash. If cash would go negative, trim the buys.
        deficit = -(self.cash + sell.sum() - buy.sum() - cost.total - tax)
        if deficit > 0 and buy.sum() > 0:
            delta = np.where(delta > 0, delta * max(0.0, 1 - deficit / buy.sum()), delta)
            buy = np.maximum(delta, 0.0)
            cost = self.costs.cost(buy, sell, self.sigma[t], self.adv[t])
        if not np.any(delta):
            return 0.0, 0.0, 0.0, 0.0
        d_shares = delta / P
        before = self.ledger.total_paid
        for i in np.flatnonzero(d_shares < -1e-12):
            self.ledger.sell(i, self.dates[t], -d_shares[i], P[i])
        for i in np.flatnonzero(d_shares > 1e-12):
            self.ledger.buy(i, self.dates[t], d_shares[i], P[i])
        liability = float(self.ledger.total_paid - before)
        tax_paid = liability if self.tax_aware else 0.0
        self.shares = self.shares + d_shares
        self.cash = V - cost.total - tax_paid - float(self.shares @ P)
        return cost.total, tax_paid, liability, turnover

    def _tax_preview(self, sell_shares: np.ndarray) -> float:
        """Tax that selling ``sell_shares`` would trigger, without mutating the ledger."""
        import copy
        led = copy.deepcopy(self.ledger)
        before = led.total_paid
        for i in np.flatnonzero(sell_shares > 1e-12):
            led.sell(i, self.dates[self.t], sell_shares[i], self.prices[self.t, i])
        return led.total_paid - before

    def step(self, action):
        V_start = self.value()
        if action is None:  # hold: no trade, no cost
            w_t, cost, tax, liab, turn = self.current_weights(), 0.0, 0.0, 0.0, 0.0
        else:
            w_t = self._apply_trade_band(self.feasible(self._to_weights(action)))
            cost, tax, liab, turn = self._execute(w_t)
        self.trades.append({"date": self.dates[self.t], "weights": w_t, "cost": cost,
                            "tax": tax, "tax_liability": liab, "turnover": turn, "value": V_start})

        # Let holdings drift day by day until the next rebalance. Day 1's
        # return is measured from the pre-trade value, so cost/tax drag counts.
        flows, nav_ret, prev_v = 0.0, 1.0, V_start
        end = min(self.t + self.reb, self.hi)
        for d in range(self.t + 1, end + 1):
            self.cash *= 1 + self.cash_daily
            v = float(self.cash + self.shares @ self.prices[d])
            nav_ret *= v / prev_v
            self.nav *= v / prev_v
            flow = self.sip if (self.sip > 0 and self.dates[d].month != self.dates[d - 1].month) else 0.0
            self.cash += flow
            flows += flow
            prev_v = v + flow
            self.history["date"].append(self.dates[d])
            self.history["nav"].append(self.nav)
            self.history["value"].append(prev_v)
            self.history["flows"].append(flow)
        self.t = end
        r = nav_ret - 1.0
        self.peak = max(self.peak, self.nav)
        dd = self.nav / self.peak - 1.0
        glide = self.profile.glide(self.horizon_left)
        reward = self.reward_fn(r, dd, self.profile.risk_aversion, self.profile.dd_tol, glide)
        self.horizon_left = max(0.0, self.horizon_left - self.reb / 252)
        self.steps += 1
        # Running out of data ends the episode; the 1-year episode limit is a
        # truncation (PPO bootstraps the value there).
        terminated = self.t + self.reb > self.hi
        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"return": r, "cost": cost, "tax": tax, "turnover": turn, "drawdown": dd,
                "nav": self.nav, "value": self.value(), "flows": flows}
        return self.observation(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ results
    def daily_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history).set_index("date")

    def trades_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.trades).set_index("date")
        w = pd.DataFrame(np.vstack(df.pop("weights").to_numpy()), index=df.index,
                         columns=self.md.tickers + ["CASH"])
        return df.join(w)
