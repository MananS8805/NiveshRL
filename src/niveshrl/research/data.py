"""NIFTY 200 price panel for research.

Unlike the RL environment, the panel keeps NaNs: a stock simply isn't
eligible before it listed (or while its data is missing). Every model and
the backtester check eligibility per date.

Source of the universe: NSE's official ``ind_nifty200list.csv`` (current
constituents + industry). Using today's constituents back to 2006 carries
survivorship bias, as documented in the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT

UNIVERSE_CSV = ROOT / "data" / "ind_nifty200list.csv"
PANEL_PATH = ROOT / "data" / "nifty200_panel.parquet"
CONTEXT_PATH = ROOT / "data" / "nifty200_context.parquet"
PIT_PANEL_PATH = ROOT / "data" / "nifty200_pit_panel.parquet"   # current + former members (survivorship-free tests)
START = "2006-01-01"


@dataclass
class Panel:
    close: pd.DataFrame          # adjusted close, trading days x tickers (NaN = not listed / missing)
    volume: pd.DataFrame
    sectors: pd.Series           # ticker -> NSE industry
    names: pd.Series             # ticker -> company name
    bench: pd.Series             # NIFTY 50
    vix: pd.Series               # India VIX
    open: pd.DataFrame | None = None   # adjusted with the same factor as close (None on old caches)
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    member: pd.DataFrame | None = None  # point-in-time index membership (True = in the index that day); None = all

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change(fill_method=None)

    def month_ends(self) -> pd.DatetimeIndex:
        """Last trading day of each month."""
        idx = self.close.index
        s = pd.Series(idx, index=idx)
        return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    df["ticker"] = df["Symbol"].str.strip() + ".NS"
    return df.rename(columns={"Company Name": "name", "Industry": "sector"})[["ticker", "name", "sector"]]


def _download(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                       threads=True, group_by="column")


SPLIT_RATIOS = np.array([1 / 10, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 2, 2, 3, 4, 5, 10])
ACTIONS_CSV = ROOT / "configs" / "corporate_actions.csv"
GENUINE_CSV = ROOT / "configs" / "genuine_moves.csv"


def adjust_corporate_actions(close: pd.DataFrame, min_move: float = 0.30, tol: float = 0.025,
                             wide_tol: float = 0.08) -> tuple[pd.DataFrame, list]:
    """Back-adjust unadjusted splits/bonuses and reviewed demergers.

    yfinance leaves some NSE splits/bonuses unadjusted (e.g. VOLTAS x10 in 2006,
    MOTILALOFS 3:1 bonus in 2024). A daily move above ``min_move`` whose price
    ratio lands within ``tol`` of a standard ratio is treated as one. Demergers
    don't have standard ratios, so they come from the reviewed
    configs/corporate_actions.csv. In both cases prices *before* the event are
    rescaled so that day's return is zero. Genuine crashes (YES Bank 2020: ratio
    0.44) don't match a ratio and are left alone.
    """
    close = close.copy()
    events = []
    genuine = set()
    if GENUINE_CSV.exists():                         # reviewed real crashes that merely look like a ratio
        g = pd.read_csv(GENUINE_CSV, comment="#")
        genuine = {(pd.Timestamp(d).date(), t) for d, t in zip(g["date"], g["ticker"])}
    ratio = close / close.shift(1)
    for t in close.columns:
        r = ratio[t]
        cand = r[(r - 1).abs() > min_move].dropna()
        for d, x in cand.items():
            rel = np.abs(SPLIT_RATIOS - x) / SPLIT_RATIOS
            limit = wide_tol if (x >= 1.9 or x <= 0.52) else tol
            if rel.min() < limit and (d.date(), t) not in genuine:
                events.append((d.date(), t, f"ratio {x:.3f}"))
    manual = pd.read_csv(ACTIONS_CSV, comment="#") if ACTIONS_CSV.exists() else pd.DataFrame(columns=["date", "ticker"])
    for _, row in manual.iterrows():
        if row["ticker"] in close.columns:
            events.append((pd.Timestamp(row["date"]).date(), row["ticker"], "reviewed"))
    for d, t, _ in events:
        d = pd.Timestamp(d)
        if d not in close.index:
            continue
        prev = close[t].loc[:d].iloc[:-1].last_valid_index()
        if prev is None or pd.isna(close.at[d, t]):
            continue
        f = close.at[d, t] / close.at[prev, t]
        close.loc[:prev, t] *= f
    return close, events


def clean_close(close: pd.DataFrame, volume: pd.DataFrame, jump: float = 0.25) -> tuple[pd.DataFrame, list]:
    """Same data hygiene as the RL pipeline, per stock and NaN-tolerant.

    - One-day bad prints (a jump above ``jump`` that reverses the next day)
      are masked and forward-filled. The reversal tolerance widens for very
      large moves (DABUR 2007: -53% then +99%).
    - Frozen runs (same price, zero volume, 10+ days) are set to NaN, so the
      stock is ineligible there instead of looking riskless.
    """
    r = close.pct_change(fill_method=None)
    two_day = ((1 + r) * (1 + r.shift(-1)) - 1).abs()
    spike = (r.abs() > jump) & ((two_day < 0.05) | ((r.abs() > 0.4) & (two_day < 0.10)))
    repaired = [(d.date(), t) for d, t in spike.stack()[lambda s: s].index]
    close = close.mask(spike).ffill(limit=1).where(close.notna() | spike)
    stale = (close.diff() == 0) & (volume.reindex_like(close).fillna(0) == 0)
    run = stale.apply(lambda s: s.astype(int).groupby((~s).cumsum()).cumsum())
    frozen = run >= 10
    # Mask the whole run, not only its tail: extend backwards 9 days.
    frozen = frozen.astype(float)[::-1].rolling(10, min_periods=1).max()[::-1].astype(bool)
    close = close.mask(frozen & stale)
    return close, repaired


def load_panel(refresh: bool = False, end: str | None = None, universe: str = "current") -> Panel:
    """``universe="current"``: today's NIFTY 200 (the app's default; survivorship-biased in backtests).
    ``universe="pit"``: today's members plus every former member from the archived snapshots, with a
    point-in-time ``member`` mask (see research/constituents.py) for survivorship-free tests."""
    uni = load_universe()
    path = PANEL_PATH
    member_h = None
    if universe == "pit":
        from . import constituents as cs
        member_h = cs.history()
        extra = cs.all_tickers(member_h).reset_index().rename(columns={"index": "ticker"})
        extra = extra[~extra["ticker"].isin(uni["ticker"])]
        uni = pd.concat([uni, extra[["ticker", "name", "sector"]]], ignore_index=True)
        path = PIT_PANEL_PATH
    if refresh or not path.exists():
        raw = _download(uni["ticker"].tolist(), START, end)
        pd.concat({"close": raw["Close"], "volume": raw["Volume"], "open": raw["Open"], "high": raw["High"],
                   "low": raw["Low"]}, axis=1).to_parquet(path)
        if universe == "current" or not CONTEXT_PATH.exists():
            ctx = _download(["^NSEI", "^INDIAVIX"], START, end)["Close"]
            ctx.columns = ["bench" if c == "^NSEI" else "vix" for c in ctx.columns]
            ctx.to_parquet(CONTEXT_PATH)
    raw = pd.read_parquet(path)
    ctx = pd.read_parquet(CONTEXT_PATH)
    close, volume = raw["close"], raw["volume"]
    close = close.loc[:, close.notna().sum() > 250]           # drop tickers with < ~1 year of data
    volume = volume[close.columns]
    close = close.dropna(how="all")
    volume = volume.reindex(close.index)
    raw_close = close
    close, _ = clean_close(close, volume)
    close, _ = adjust_corporate_actions(close)
    ohl = {}
    if {"open", "high", "low"} <= set(raw.columns.get_level_values(0)):
        # Same per-day factor as the cleaned close (corporate actions, bad prints), so O/H/L/C agree.
        factor = (close / raw_close).replace([np.inf, -np.inf], np.nan).ffill()
        for k in ("open", "high", "low"):
            ohl[k] = (raw[k].reindex(index=close.index, columns=close.columns) * factor).where(close.notna())
        ohl["high"] = np.maximum(ohl["high"], close).where(close.notna())
        ohl["low"] = np.minimum(ohl["low"], close).where(close.notna())
    meta = uni.drop_duplicates("ticker").set_index("ticker").reindex(close.columns)
    member = None
    if universe == "pit":
        from . import constituents as cs
        member = cs.member_mask(close.index, list(close.columns), member_h, backfill=True)
    return Panel(close=close, volume=volume, sectors=meta["sector"].fillna("Other"),
                 names=meta["name"].fillna(pd.Series(close.columns, index=close.columns)),
                 bench=ctx["bench"].reindex(close.index).ffill(),
                 vix=ctx["vix"].reindex(close.index).ffill(), member=member, **ohl)
