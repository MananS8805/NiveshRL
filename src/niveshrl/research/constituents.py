"""Point-in-time NIFTY 200 membership, to remove survivorship bias from backtests.

Every backtest that uses *today's* index members applied to the past only ever sees stocks that
survived and stayed in the index; the ones that collapsed, got demoted or delisted are missing, which
flatters every strategy (momentum most). This module rebuilds who was actually in the index on each
date from archived copies of NSE's official constituent file (``ind_nifty200list.csv``, earlier
``ind_cnx200list.csv``) kept by the Internet Archive (``scripts/fetch_constituents.py`` downloads them
into ``data/constituents/``).

Rule (no look-ahead): a stock is a member on date *d* if it is in the latest snapshot dated on or
before *d*. Between snapshots membership is therefore *stale*, never anticipated. Before the first
snapshot nothing is known, so point-in-time tests start at the first snapshot.

Renames are matched by ISIN: a former member whose ISIN now trades under a different symbol (e.g.
MOTHERSUMI -> MOTHERSON, ZOMATO -> ETERNAL) maps to today's Yahoo ticker.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT

DIR = ROOT / "data" / "constituents"
CURRENT = ROOT / "data" / "ind_nifty200list.csv"


def snapshot_files() -> list[tuple[pd.Timestamp, Path]]:
    out = []
    for f in sorted(DIR.glob("nifty200_*.csv")):
        m = re.search(r"(\d{8})", f.name)
        if m:
            out.append((pd.Timestamp(m.group(1)), f))
    return out


def history(include_current: bool = True, current_date: str | None = None) -> pd.DataFrame:
    """All snapshots as rows: date, symbol, isin, company, industry."""
    rows = []
    snaps = [(d, pd.read_csv(f)) for d, f in snapshot_files()]
    if include_current and CURRENT.exists():
        d = pd.Timestamp(current_date) if current_date else pd.Timestamp(CURRENT.stat().st_mtime, unit="s").normalize()
        snaps.append((d, pd.read_csv(CURRENT)))
    for d, df in snaps:
        df = df.rename(columns=lambda c: c.strip())
        df = df[~df["Symbol"].str.upper().str.startswith("DUMMY")]          # NSE placeholder rows
        rows.append(pd.DataFrame({"date": d, "symbol": df["Symbol"].str.strip(), "isin": df["ISIN Code"].str.strip(),
                                  "company": df["Company Name"].str.strip(), "industry": df["Industry"].str.strip()}))
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "isin", "company", "industry"])
    h = pd.concat(rows, ignore_index=True)
    return h.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)


ALIASES = ROOT / "configs" / "ticker_aliases.csv"


def ticker_map(h: pd.DataFrame) -> dict[str, str]:
    """Symbol in any snapshot -> Yahoo ticker today. Renames are resolved via the newest symbol for the same
    ISIN, then the reviewed aliases in configs/ticker_aliases.csv (renamed stocks no longer in the index)."""
    newest = h.sort_values("date").groupby("isin")["symbol"].last()
    out = {s: newest.get(i, s) + ".NS" for s, i in zip(h["symbol"], h["isin"])}
    if ALIASES.exists():
        al = pd.read_csv(ALIASES, comment="#")
        for s, y in zip(al["symbol"], al["yahoo"]):
            if s in out and out[s] == s + ".NS":
                out[s] = y
    return out


def all_tickers(h: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every Yahoo ticker that was ever a member, with its latest known name and industry."""
    h = history() if h is None else h
    tm = ticker_map(h)
    h = h.assign(ticker=h["symbol"].map(tm))
    last = h.sort_values("date").groupby("ticker").last()
    return last[["company", "industry"]].rename(columns={"company": "name", "industry": "sector"})


def member_mask(index: pd.DatetimeIndex, tickers: list[str], h: pd.DataFrame | None = None,
                backfill: bool = False) -> pd.DataFrame:
    """Boolean frame (dates x tickers): True where the ticker was in the latest snapshot on or before the date.
    Dates before the first snapshot are False (unknown), or, with ``backfill=True``, use the first snapshot:
    only for *training* history before point-in-time tests begin (it carries a little survivorship into
    the oldest training years, never into test-period selection)."""
    h = history() if h is None else h
    tm = ticker_map(h)
    snaps = sorted(h["date"].unique())
    members = {d: set(h.loc[h["date"] == d, "symbol"].map(tm)) for d in snaps}
    out = np.zeros((len(index), len(tickers)), dtype=bool)
    col = {t: j for j, t in enumerate(tickers)}
    pos = np.searchsorted(np.array(snaps, dtype="datetime64[ns]"), index.values, side="right") - 1
    if backfill:
        pos = np.maximum(pos, 0)
    for k, d in enumerate(snaps):
        rows = pos == k
        if not rows.any():
            continue
        cols = [col[t] for t in members[d] if t in col]
        out[np.ix_(rows, cols)] = True
    return pd.DataFrame(out, index=index, columns=tickers)


def first_date(h: pd.DataFrame | None = None) -> pd.Timestamp | None:
    h = history() if h is None else h
    return h["date"].min() if len(h) else None
