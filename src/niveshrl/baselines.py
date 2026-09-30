"""Classical allocation strategies, run through the same env as the RL agents.

Each policy is a callable ``policy(env) -> weights | None``. Returning
``None`` means "hold, don't trade this week". Every strategy therefore pays
the same India cost model and obeys the same caps.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class MonthlyPolicy:
    """Rebalance on the first rebalance step of each calendar month, hold otherwise."""

    name = "base"

    def __init__(self):
        self._last_month = None

    def __call__(self, env):
        d = env.dates[env.t]
        key = (d.year, d.month)
        if key == self._last_month:
            return None
        self._last_month = key
        return self.target(env)

    def target(self, env) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _history(self, env, window: int = 252) -> pd.DataFrame:
        lo = max(0, env.t - window)
        return env.md.close.iloc[lo:env.t + 1]


class EqualWeight(MonthlyPolicy):
    name = "Equal weight"

    def target(self, env):
        n = env.N
        return np.append(np.full(n, 1.0 / n), 0.0)


class Momentum(MonthlyPolicy):
    name = "Momentum top-10"

    def __init__(self, k: int = 10, lookback: int = 120, skip: int = 5):
        super().__init__()
        self.k, self.lookback, self.skip = k, lookback, skip

    def target(self, env):
        p = env.prices
        t = env.t
        mom = p[t - self.skip] / p[t - self.lookback] - 1
        top = np.argsort(mom)[::-1][:self.k]
        w = np.zeros(env.N + 1)
        w[top] = 1.0 / self.k
        return w


class InverseVol(MonthlyPolicy):
    name = "Risk parity (inv-vol)"

    def target(self, env):
        r = self._history(env, 60).pct_change().dropna()
        iv = 1.0 / (r.std().to_numpy() + 1e-8)
        return np.append(iv / iv.sum(), 0.0)


class _PyPfOpt(MonthlyPolicy):
    def _ef(self, env):
        from pypfopt import EfficientFrontier, expected_returns, risk_models

        hist = self._history(env, 252)
        mu = expected_returns.mean_historical_return(hist, frequency=252)
        S = risk_models.CovarianceShrinkage(hist, frequency=252).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0, env.ecfg["max_stock_weight"]))
        sec_map = dict(zip(env.md.tickers, env.md.sectors))
        secs = sorted(set(env.md.sectors))
        ef.add_sector_constraints(sec_map, {s: 0.0 for s in secs},
                                  {s: env.ecfg["max_sector_weight"] for s in secs})
        return ef

    def _to_array(self, env, w: dict) -> np.ndarray:
        arr = np.array([w.get(tk, 0.0) for tk in env.md.tickers])
        arr = np.clip(arr, 0, None)
        return np.append(arr, max(0.0, 1.0 - arr.sum()))


class Markowitz(_PyPfOpt):
    name = "Markowitz max-Sharpe"

    def target(self, env):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ef = self._ef(env)
                w = ef.max_sharpe(risk_free_rate=env.ecfg["cash_rate"])
            except Exception:
                # max-Sharpe is infeasible when no asset beats the risk-free
                # rate; fall back to minimum variance.
                ef = self._ef(env)
                w = ef.min_volatility()
        return self._to_array(env, w)


class MinVariance(_PyPfOpt):
    name = "Minimum variance"

    def target(self, env):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = self._ef(env).min_volatility()
        return self._to_array(env, w)


class HRP(MonthlyPolicy):
    name = "HRP"

    def target(self, env):
        r = self._history(env, 252).pct_change().dropna()
        return np.append(hrp_weights(r.cov().to_numpy(), r.corr().to_numpy()), 0.0)


def hrp_weights(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    Written out here because PyPortfolioOpt's HRPOpt breaks on current SciPy.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    w = np.ones(len(cov))
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            a, b = c[: len(c) // 2], c[len(c) // 2:]

            def cvar(idx):
                sub = cov[np.ix_(idx, idx)]
                ivp = 1 / np.diag(sub)
                ivp /= ivp.sum()
                return ivp @ sub @ ivp

            va, vb = cvar(a), cvar(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


ALL_BASELINES = [EqualWeight, Momentum, InverseVol, Markowitz, MinVariance, HRP]
