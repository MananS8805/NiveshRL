"""Composite analyst score (0-100) from Yahoo Finance consensus data.

    score = 40% consensus + 30% target upside + 20% revision momentum + 10% coverage

- consensus: Yahoo's recommendationMean, 1 (strong buy) ... 5 (strong sell), mapped linearly to 100 ... 0.
- target upside: mean target / price - 1, mapped from -20% (0) to +40% (100), clipped.
- revision momentum: change in the share of buy/strong-buy ratings over 3 months, -20 pts (0) to +20 pts (100).
- coverage: number of analysts, 0 -> 0, 25+ -> 100 (more analysts = more reliable consensus).

Missing components are skipped and the remaining weights renormalised; with no
coverage at all the score is NaN. Third-party opinion, shown for context only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"consensus": 0.4, "upside": 0.3, "revision": 0.2, "coverage": 0.1}


def components(df: pd.DataFrame, price: pd.Series | None = None) -> pd.DataFrame:
    px = price.reindex(df.index) if price is not None else pd.to_numeric(df.get("currentPrice"), errors="coerce")
    mean = pd.to_numeric(df.get("recommendationMean"), errors="coerce")
    tgt = pd.to_numeric(df.get("targetMeanPrice"), errors="coerce")
    n = pd.to_numeric(df.get("numberOfAnalystOpinions"), errors="coerce")
    now = pd.to_numeric(df.get("buy_share_now"), errors="coerce")
    then = pd.to_numeric(df.get("buy_share_3m"), errors="coerce")
    upside = tgt / px - 1
    out = pd.DataFrame({
        "consensus": ((5 - mean) / 4 * 100).clip(0, 100),
        "upside": ((upside + 0.2) / 0.6 * 100).clip(0, 100),
        "revision": (((now - then) * 100 + 20) / 40 * 100).clip(0, 100),
        "coverage": (n / 25 * 100).clip(0, 100),
    }, index=df.index)
    out["target_upside"] = upside
    return out


def score(df: pd.DataFrame, price: pd.Series | None = None) -> pd.Series:
    c = components(df, price)
    w = pd.Series(WEIGHTS)
    vals = c[list(WEIGHTS)]
    has = vals.notna()
    num = (vals.fillna(0) * w).sum(axis=1)
    den = (has * w).sum(axis=1)
    s = num / den.replace(0, np.nan)
    n = pd.to_numeric(df.get("numberOfAnalystOpinions"), errors="coerce")
    return s.where(n.fillna(0) > 0)
