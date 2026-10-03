"""The liquid NSE universe for intraday trading.

NSE's full equity list (EQUITY_L.csv, EQ series only: BE/T2T stocks cannot be traded intraday) is filtered on
the last ~30 trading days of Yahoo daily bars: 20-day average traded value ≥ ₹10 crore and price ≥ ₹50
(configs/intraday.yaml). Saves data/intraday/universe.csv with turnover, average daily range (ADR), ATR and
the previous day's OHLC per stock; refreshed every 7 days.
"""
from __future__ import annotations

import io
import time
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from . import DIR, config

EQUITY_L = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
PATH = DIR / "universe.csv"


def nse_equities() -> pd.DataFrame:
    req = urllib.request.Request(EQUITY_L, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"]
    return pd.DataFrame({"ticker": df["SYMBOL"].str.strip() + ".NS", "name": df["NAME OF COMPANY"].str.strip(),
                         "isin": df["ISIN NUMBER"].str.strip()})


def _daily(tickers: list[str], period: str = "45d", batch: int = 200, progress=None) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        raw = yf.download(chunk, period=period, interval="1d", auto_adjust=True, progress=False, threads=True,
                          group_by="column")
        for k in ("Open", "High", "Low", "Close", "Volume"):
            out.setdefault(k, []).append(raw[k] if isinstance(raw.columns, pd.MultiIndex) else raw[[k]].rename(columns={k: chunk[0]}))
        if progress:
            progress(min(1.0, (i + batch) / len(tickers)))
        time.sleep(1)
    return {k: pd.concat(v, axis=1) for k, v in out.items()}


def build(progress=None) -> pd.DataFrame:
    cfg = config()["universe"]
    eq = nse_equities()
    bars = _daily(eq["ticker"].tolist(), progress=progress)
    keep_rows = bars["Close"].notna().sum(axis=1) >= 0.5 * bars["Close"].shape[1]   # drop stray partial dates
    c, h, l, o, v = (bars[k][keep_rows] for k in ("Close", "High", "Low", "Open", "Volume"))
    turnover = (c * v).rolling(20, min_periods=10).mean().iloc[-1] / 1e7
    adr = ((h - l) / c).rolling(20, min_periods=10).mean().iloc[-1]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()]).groupby(level=0).max().reindex(c.index)
    atr = tr.rolling(14, min_periods=10).mean().iloc[-1]
    rng = h - l
    nr7 = (rng.iloc[-1] <= rng.iloc[-7:].min()).astype(float)
    df = pd.DataFrame({"turnover_cr": turnover, "adr": adr, "atr": atr, "close": c.iloc[-1], "prev_high": h.iloc[-1],
                       "prev_low": l.iloc[-1], "prev_close": c.iloc[-1], "avg_volume": v.rolling(20, min_periods=10).mean().iloc[-1],
                       "nr7": nr7})
    df = df.join(eq.set_index("ticker")[["name"]], how="left")
    keep = (df["turnover_cr"] >= cfg["min_turnover_cr"]) & (df["close"] >= cfg["min_price"]) & df["adr"].notna()
    df = df[keep].sort_values("turnover_cr", ascending=False)
    df["asof"] = c.index[-1].date().isoformat()
    df["built"] = datetime.now().isoformat(timespec="seconds")
    DIR.mkdir(parents=True, exist_ok=True)
    df.index.name = "ticker"
    df.to_csv(PATH)
    return df


def load(max_age_days: int | None = None, progress=None) -> pd.DataFrame:
    """The cached universe, rebuilt when missing or older than ``refresh_days``."""
    age = max_age_days if max_age_days is not None else config()["universe"]["refresh_days"]
    if PATH.exists():
        df = pd.read_csv(PATH, index_col=0)
        built = pd.Timestamp(df["built"].iloc[0]) if len(df) else pd.Timestamp(0)
        if (pd.Timestamp.now() - built).days < age:
            return df
    return build(progress)


def adr_rank(df: pd.DataFrame) -> pd.Series:
    """Rank by average daily range: the strongest single predictor of tomorrow's range."""
    return df["adr"].rank(ascending=False)


__all__ = ["build", "load", "nse_equities", "adr_rank", "np"]
