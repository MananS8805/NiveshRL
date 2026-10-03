"""'Who will move tomorrow': predicting each stock's next-day trading range (high − low) ÷ close.

Range is far more predictable than direction: volatility clusters. Target = the cross-sectional percentile of
tomorrow's range. Features (all known at today's close): today's range, 5- and 20-day average range, ATR %, |gap|,
|return|, volume ÷ 20-day average, NR7 / NR4 flags, today's range ÷ its 20-day average, 60-day volatility, VIX level and
change, NIFTY's absolute move, day of week. LightGBM, walk-forward by year on a rolling 5-year window. Baselines:
today's range rank and the 20-day average range rank. Universe: point-in-time members when the panel has them.

Use: a 'who will move' list for intraday traders (TODAY) and a feature for the intraday agent. It says nothing about
direction.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import Panel

FEATS = ["range_1d", "adr5", "adr20", "atr_pct", "gap_abs", "ret_abs", "vol_ratio", "nr7", "nr4", "range_vs_adr",
         "vol60", "vix", "vix_chg", "nifty_abs", "dow"]


def build(p: Panel) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Long frame (date, ticker) of features, the target percentile and the raw next-day range."""
    c, h, l, o, v = p.close, p.high, p.low, p.open, p.volume
    prev = c.shift(1)
    rng = (h - l) / c
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], keys=range(3)).groupby(level=1).max().reindex(c.index)
    f = {"range_1d": rng, "adr5": rng.rolling(5, min_periods=4).mean(), "adr20": rng.rolling(20, min_periods=15).mean(),
         "atr_pct": tr.rolling(14, min_periods=10).mean() / c, "gap_abs": (o / prev - 1).abs(),
         "ret_abs": (c / prev - 1).abs(), "vol_ratio": v / v.rolling(20, min_periods=10).mean().replace(0, np.nan),
         "nr7": ((h - l) <= (h - l).rolling(7, min_periods=7).min()).astype(float).where(c.notna()),
         "nr4": ((h - l) <= (h - l).rolling(4, min_periods=4).min()).astype(float).where(c.notna()),
         "vol60": c.pct_change(fill_method=None).rolling(60, min_periods=40).std() * np.sqrt(252)}
    f["range_vs_adr"] = f["range_1d"] / f["adr20"]
    b = p.bench.reindex(c.index)
    mkt = {"vix": p.vix.reindex(c.index) / 100, "vix_chg": np.log(p.vix.reindex(c.index)).diff(5),
           "nifty_abs": (b / b.shift(1) - 1).abs(), "dow": pd.Series(c.index.dayofweek, index=c.index, dtype=float)}
    nxt = (h.shift(-1) - l.shift(-1)) / c
    member = p.member.reindex(index=c.index, columns=c.columns).fillna(False) if p.member is not None else None
    if member is not None:
        nxt = nxt.where(member)
        f = {k: x.where(member) for k, x in f.items()}
    target = nxt.rank(axis=1, pct=True)
    X = pd.DataFrame({k: x.stack(future_stack=True) for k, x in f.items()})
    for k, s in mkt.items():
        X[k] = s.reindex(X.index.get_level_values(0)).to_numpy()
    y = target.stack(future_stack=True).reindex(X.index)
    raw = nxt.stack(future_stack=True).reindex(X.index)
    ok = X[["range_1d", "adr20", "atr_pct"]].notna().all(axis=1) & np.isfinite(X["range_vs_adr"])
    return X[ok], y[ok], raw[ok]


def walk_forward(X: pd.DataFrame, y: pd.Series, first_year: int = 2015, window: int = 5, seed: int = 0) -> pd.Series:
    import lightgbm as lgb
    dates = X.index.get_level_values(0)
    out = []
    for yr in range(first_year, dates.max().year + 1):
        tr = (dates.year >= yr - window) & (dates.year < yr) & y.notna().to_numpy()
        te = dates.year == yr
        if tr.sum() < 10_000 or te.sum() == 0:
            continue
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(X[FEATS][tr], y[tr])
        out.append(pd.Series(m.predict(X[FEATS][te]), index=X.index[te], name="pred"))
        print(f"  [range] {yr}: train {int(tr.sum()):,}  test {int(te.sum()):,}", flush=True)
    return pd.concat(out)


def evaluate(pred: pd.Series, X: pd.DataFrame, y: pd.Series, raw: pd.Series, top: int = 20) -> pd.DataFrame:
    """Daily Spearman IC vs the actual next-day range, and the average next-day range of the top ``top`` picks, for the
    model and the two naive baselines."""
    df = pd.DataFrame({"model": pred, "today's range": X["range_1d"].reindex(pred.index),
                       "20-day avg range": X["adr20"].reindex(pred.index), "y": y.reindex(pred.index),
                       "raw": raw.reindex(pred.index)}).dropna()
    rows = {}
    g = df.groupby(level=0)
    avg_all = g["raw"].mean()
    for col in ("model", "today's range", "20-day avg range"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic = g.apply(lambda d: d[col].rank().corr(d["y"].rank()))
        topr = g.apply(lambda d: d.nlargest(top, col)["raw"].mean())
        rows[col] = {"IC mean": float(ic.mean()), "IC t-stat": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
                     f"Top {top} next-day range": float(topr.mean()), "All stocks next-day range": float(avg_all.mean()),
                     f"Top {top} ≥ 2% range": float(g.apply(lambda d: (d.nlargest(top, col)["raw"] >= 0.02).mean()).mean()),
                     "Days": int(len(ic))}
    return pd.DataFrame(rows).T


def fit_latest(X: pd.DataFrame, y: pd.Series, years: int = 5, seed: int = 0):
    """Model trained on the last ``years`` of labelled rows (for today's forecast in the daily pipeline)."""
    import lightgbm as lgb
    dates = X.index.get_level_values(0)
    tr = (dates >= dates.max() - pd.DateOffset(years=years)) & y.notna().to_numpy()
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.7,
                          subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(X[FEATS][tr], y[tr])
    return m
