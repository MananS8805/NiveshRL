"""Performance metrics over a daily NAV series (flow-adjusted)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def daily_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def cagr(nav: pd.Series) -> float:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(ANN))


def sharpe(r: pd.Series, rf: float = 0.065) -> float:
    ex = r - ((1 + rf) ** (1 / ANN) - 1)
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def sortino(r: pd.Series, rf: float = 0.065) -> float:
    ex = r - ((1 + rf) ** (1 / ANN) - 1)
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return float(ex.mean() / dd * np.sqrt(ANN)) if dd > 0 else 0.0


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def max_drawdown(nav: pd.Series) -> float:
    return float(drawdown(nav).min())


def alpha_beta(r: pd.Series, bench_r: pd.Series) -> tuple[float, float]:
    df = pd.concat([r, bench_r], axis=1).dropna()
    if len(df) < 20:
        return 0.0, 1.0
    b = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1] / df.iloc[:, 1].var()
    a = (df.iloc[:, 0].mean() - b * df.iloc[:, 1].mean()) * ANN
    return float(a), float(b)


def information_ratio(r: pd.Series, bench_r: pd.Series) -> float:
    active = (r - bench_r).dropna()
    sd = active.std()
    return float(active.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def summarize(nav: pd.Series, bench_nav: pd.Series | None = None, trades: pd.DataFrame | None = None,
              rf: float = 0.065) -> dict:
    r = daily_returns(nav)
    out = {
        "CAGR": cagr(nav), "Vol": ann_vol(r), "Sharpe": sharpe(r, rf), "Sortino": sortino(r, rf),
        "MaxDD": max_drawdown(nav),
    }
    out["Calmar"] = out["CAGR"] / abs(out["MaxDD"]) if out["MaxDD"] < 0 else np.nan
    wk = nav.resample("W").last().pct_change().dropna()
    out["HitRate"] = float((wk > 0).mean()) if len(wk) else np.nan
    if bench_nav is not None:
        br = daily_returns(bench_nav.reindex(nav.index).ffill())
        out["Alpha"], out["Beta"] = alpha_beta(r, br)
        out["InfoRatio"] = information_ratio(r, br)
    if trades is not None and len(trades):
        years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
        out["Turnover/yr"] = float(trades["turnover"].sum() / years)
        out["Costs(Rs)"] = float(trades["cost"].sum())
        out["TaxLiability(Rs)"] = float(trades.get("tax_liability", pd.Series(0.0)).sum())
    return out


def block_bootstrap_sharpe_diff(r_a: pd.Series, r_b: pd.Series, n_boot: int = 2000,
                                block: int = 20, rf: float = 0.065, seed: int = 0) -> dict:
    """Moving-block bootstrap CI for Sharpe(a) - Sharpe(b) on paired daily returns."""
    df = pd.concat([r_a, r_b], axis=1).dropna().to_numpy()
    n = len(df)
    rng = np.random.default_rng(seed)
    rfd = (1 + rf) ** (1 / ANN) - 1
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        s = df[idx] - rfd
        sh = s.mean(0) / s.std(0, ddof=1) * np.sqrt(ANN)
        diffs[k] = sh[0] - sh[1]
    point = sharpe(pd.Series(df[:, 0]), rf) - sharpe(pd.Series(df[:, 1]), rf)
    return {"diff": float(point), "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)), "p_le_0": float((diffs <= 0).mean())}
