"""Monthly cross-sectional features for stock ranking (Takeuchi & Lee, 2013).

At each month-end t, for every eligible stock:

- **12 monthly features** ``m2..m13``: cumulative return from month-end
  t-k to month-end t-1, for k = 2..13. The most recent month is skipped,
  following the usual momentum convention ("12-1"). Its information comes
  in through the daily block instead.
- **20 daily features** ``d1..d20``: cumulative return over the last d
  trading days ending at t.
- **January dummy** ``jan``: 1 if the month being predicted (t+1) is January.
- **Target**: the return over month t+1. The label is 1 if it beats that
  month's cross-sectional median.

Each feature is z-scored across stocks within the month, so the models learn
*relative* patterns and the market's overall level drops out. That matches
the ranking objective.

Sequence models get the same information in a different shape: the 13
simple monthly returns ending at t (a 13-step sequence) plus the 20 daily
features and the January dummy as static context.

Eligibility at t means a price at t, at t-13 month-ends, and at least 18 of
the last 21 daily prices. Training and evaluation also need the realised
t+1 return. A stock with no t+1 price (e.g. suspended) is dropped from that
month, not counted as a zero return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Panel

MONTHLY = [f"m{k}" for k in range(2, 14)]
DAILY = [f"d{d}" for d in range(1, 21)]
STATIC = DAILY + ["jan"]
FEATURES = MONTHLY + DAILY + ["jan"]          # 33
SEQ_LEN = 13


@dataclass
class RankData:
    frame: pd.DataFrame        # index (date, ticker): FEATURES (z-scored), seq_0..seq_12, ret_next, label
    month_ends: pd.DatetimeIndex

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.frame.index.get_level_values(0).unique()

    def X(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[FEATURES].to_numpy(np.float32)

    def seq(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        s = rows[[f"seq_{j}" for j in range(SEQ_LEN)]].to_numpy(np.float32)[..., None]
        return s, rows[STATIC].to_numpy(np.float32)


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date (level 0), clipped to +-5."""
    g = df.groupby(level=0)
    z = (df - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z.clip(-5, 5).fillna(0.0)


def build_rank_data(p: Panel) -> RankData:
    close = p.close
    me = p.month_ends()
    pos = close.index.get_indexer(me)
    rows = []
    for i in range(13, len(me)):
        t = me[i]
        P_t = close.iloc[pos[i]]
        P_m = [close.iloc[pos[i - k]] for k in range(0, 14)]          # month-end prices t, t-1, ..., t-13
        win = close.iloc[max(0, pos[i] - 20): pos[i] + 1]              # last 21 daily prices
        ok = P_t.notna() & P_m[13].notna() & (win.notna().sum() >= 18)
        for k in range(1, 13):
            ok &= P_m[k].notna()
        if ok.sum() < 20:
            continue
        tick = ok[ok].index
        f = {}
        for k in range(2, 14):
            f[f"m{k}"] = P_m[1][tick] / P_m[k][tick] - 1
        wf = win[tick].ffill()
        for d in range(1, 21):
            f[f"d{d}"] = wf.iloc[-1] / wf.iloc[-1 - d] - 1
        for j in range(SEQ_LEN):                                        # simple monthly returns, oldest first
            f[f"seq_{SEQ_LEN - 1 - j}"] = P_m[j][tick] / P_m[j + 1][tick] - 1
        if i + 1 < len(me):
            nxt = close.iloc[pos[i + 1]][tick]
            f["ret_next"] = nxt / P_t[tick] - 1
        else:
            f["ret_next"] = pd.Series(np.nan, index=tick)
        nm = me[i + 1] if i + 1 < len(me) else t + pd.offsets.MonthEnd(1)
        df = pd.DataFrame(f)
        df["jan"] = float(nm.month == 1)
        df.index = pd.MultiIndex.from_product([[t], df.index], names=["date", "ticker"])
        rows.append(df)
    frame = pd.concat(rows)
    zcols = MONTHLY + DAILY + [f"seq_{j}" for j in range(SEQ_LEN)]
    frame[zcols] = _zscore(frame[zcols].replace([np.inf, -np.inf], np.nan))
    med = frame.groupby(level=0)["ret_next"].transform("median")
    frame["label"] = np.where(frame["ret_next"].isna(), np.nan, (frame["ret_next"] > med).astype(float))
    return RankData(frame=frame, month_ends=me)
