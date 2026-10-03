"""Rule-based chart patterns on daily bars, with a pivot, a breakout state, and each pattern's *measured* edge.

Detectors (vectorised over the whole panel; each uses data up to that day only):
- **vcp** — volatility contraction: three consecutive 15-day windows whose high–low range shrinks (each smaller than the
  one before), the last under 10%, volume drying up, price above its 200-day average.
- **flat_base** — 25+ days (5 weeks) inside a ≤ 15% range, price within 5% of the range top, above the 200-day average.
- **high_box** — within 3% of the 52-week high and the last 20 days inside a ≤ 12% box (Darvas-style).
- **nr7** — today's range is the narrowest of the last 7 days (a quiet day).
- **pocket_pivot** — an up day whose volume beats every down day's volume of the prior 10 days, closing above the 10- and
  50-day averages.
- **squeeze** — Bollinger Bands (20, 2σ) inside Keltner Channels (20, 1.5 × ATR): unusually low volatility.
Pivot = the 20-day high (the base top). Distance = close ÷ pivot − 1. State: early base (< −10%), building (−10 to −3%),
near pivot (−3 to 0%), breakout (0 to +3%), extended (> +8%), failed (broke out within 10 days, now > 2% back under).
Rules describe price structure; they do not predict. ``measure_edges`` reports each pattern's forward 20-day excess
return and swing-rule outcome vs random stocks on the same days (point-in-time members), so the app shows evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel

PATTERNS = ["vcp", "flat_base", "high_box", "nr7", "pocket_pivot", "squeeze"]
LABELS = {"vcp": "VCP (tightening)", "flat_base": "Flat base", "high_box": "52-week-high box", "nr7": "NR7 (quiet day)",
          "pocket_pivot": "Pocket pivot", "squeeze": "Bollinger squeeze"}


def detect(p: Panel) -> dict[str, pd.DataFrame]:
    c, h, l, v = p.close, p.high, p.low, p.volume
    sma200 = c.rolling(200, min_periods=150).mean()
    above200 = c > sma200
    out = {}
    w = 15
    r1 = (h.rolling(w).max() / l.rolling(w).min() - 1).shift(2 * w)
    r2 = (h.rolling(w).max() / l.rolling(w).min() - 1).shift(w)
    r3 = h.rolling(w).max() / l.rolling(w).min() - 1
    vdry = v.rolling(w).mean() < v.rolling(w).mean().shift(2 * w)
    out["vcp"] = (r1 > r2) & (r2 > r3) & (r3 < 0.10) & vdry & above200
    hi25, lo25 = h.rolling(25).max(), l.rolling(25).min()
    out["flat_base"] = (hi25 / lo25 - 1 <= 0.15) & (c >= 0.95 * hi25) & above200
    hi252 = c.rolling(252, min_periods=200).max()
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    out["high_box"] = (c >= 0.97 * hi252) & (hi20 / lo20 - 1 <= 0.12)
    rng = h - l
    out["nr7"] = rng <= rng.rolling(7, min_periods=7).min()
    up = c > c.shift(1)
    down_vol = v.where(c < c.shift(1))
    max_down10 = down_vol.shift(1).rolling(10, min_periods=1).max()
    out["pocket_pivot"] = up & (v > max_down10) & (c > c.rolling(10).mean()) & (c > c.rolling(50).mean())
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], keys=range(3)).groupby(level=1).max().reindex(c.index)
    atr20 = tr.rolling(20).mean()
    out["squeeze"] = ((mid + 2 * sd) < (mid + 1.5 * atr20)) & ((mid - 2 * sd) > (mid - 1.5 * atr20))
    return {k: x.fillna(False).where(c.notna(), False) for k, x in out.items()}


def pivot_state(p: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot distance and breakout state per day and stock."""
    c, h = p.close, p.high
    pivot = h.rolling(20).max().shift(1)                 # the base top before today
    dist = c / pivot - 1
    broke = (dist > 0) & (dist.shift(1) <= 0)
    recent_break = broke.rolling(10, min_periods=1).max().astype(bool)
    state = pd.DataFrame("building", index=c.index, columns=c.columns)
    state = state.mask(dist < -0.10, "early base").mask((dist >= -0.03) & (dist < 0), "near pivot")
    state = state.mask((dist >= 0) & (dist <= 0.03), "breakout").mask(dist > 0.08, "extended")
    state = state.mask((dist > 0.03) & (dist <= 0.08), "confirmed")
    state = state.mask(recent_break & (dist < -0.02), "failed").where(c.notna())
    return dist, state


def snapshot(p: Panel, day: pd.Timestamp | None = None) -> pd.DataFrame:
    """Today's patterns, pivot distance and state per stock (screener columns)."""
    d = day or p.close.index[-1]
    pats = detect(p)
    dist, state = pivot_state(p)
    df = pd.DataFrame({k: pats[k].loc[d].astype(float) for k in PATTERNS})
    df["pattern_count"] = df[PATTERNS].sum(axis=1)
    df["pivot_dist"] = dist.loc[d]
    df["breakout_state"] = state.loc[d]
    return df


def measure_edges(p: Panel, step: int = 5, horizon: int = 20, start: str = "2014-02-01") -> pd.DataFrame:
    """Forward ``horizon``-day return minus the median eligible stock, for stocks flagged by each pattern vs all eligible
    stocks, every ``step`` days (point-in-time members when the panel has them). t-stat over sample days."""
    pats = detect(p)
    c = p.close
    fwd = c.shift(-horizon) / c - 1
    elig = c.notna() & fwd.notna()
    if p.member is not None:
        elig &= p.member.reindex(index=c.index, columns=c.columns).fillna(False)
    days = [d for d in c.index[::step] if d >= pd.Timestamp(start)]
    ex = fwd.sub(fwd.where(elig).mean(axis=1), axis=0)          # vs the average member (returns are skewed: mean > median)
    rows = {}
    for k in PATTERNS:
        flag = pats[k] & elig
        per_day = ex.where(flag).loc[days].mean(axis=1).dropna()
        n_obs = int(flag.loc[days].values.sum())
        hit = float((ex.where(flag).loc[days] > 0).sum().sum() / max(n_obs, 1))
        rows[LABELS[k]] = {"signals": n_obs, "avg 20d excess": float(per_day.mean()),
                           "t-stat": float(per_day.mean() / per_day.std() * np.sqrt(len(per_day))) if len(per_day) > 2 else np.nan,
                           "beat median": hit, "days": len(per_day)}
    return pd.DataFrame(rows).T
