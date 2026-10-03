"""Replay the agent over history: Yahoo's last ~60 days of 5-minute bars for the whole liquid universe.

Each day uses only that day's bars plus context from earlier days (previous day's high/low/close, 20-day average
daily range, ATR, NR7, average volume, all computed from the 5-minute history itself), and the agent's learning
state from the days before. State is kept in data/intraday/replay/ (separate from the live account).
The first ``warmup`` days only build context. Run: ``python scripts/intraday_replay.py``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import DIR, config
from .agent import Agent

BARS = DIR / "bars5m.parquet"
IST = "Asia/Kolkata"


def download(tickers: list[str], days: int = 60, batch: int = 100, progress=None) -> pd.DataFrame:
    """Long table: ticker, ts (IST, bar start), open, high, low, close, volume."""
    import yfinance as yf
    frames = []
    allt = list(tickers) + ["^NSEI"]
    for i in range(0, len(allt), batch):
        chunk = allt[i:i + batch]
        raw = yf.download(chunk, period=f"{days}d", interval="5m", auto_adjust=False, progress=False, threads=True,
                          group_by="ticker")
        for t in chunk:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            d = d.dropna(subset=["Close"])
            if d.empty:
                continue
            idx = d.index.tz_convert(IST) if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert(IST)
            frames.append(pd.DataFrame({"ticker": t, "ts": idx.tz_localize(None), "open": d["Open"].to_numpy(),
                                        "high": d["High"].to_numpy(), "low": d["Low"].to_numpy(),
                                        "close": d["Close"].to_numpy(), "volume": d["Volume"].to_numpy()}))
        if progress:
            progress(min(1.0, (i + batch) / len(allt)))
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["ts"].dt.hour * 60 + out["ts"].dt.minute).between(9 * 60 + 15, 15 * 60 + 25)]
    DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(BARS)
    return out


def daily_context(bars: pd.DataFrame) -> pd.DataFrame:
    """Per (day, ticker): context known *before* that day opens (all shifted by one day)."""
    bars = bars.assign(day=bars["ts"].dt.normalize())
    d = bars.groupby(["ticker", "day"]).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"),
                                             c=("close", "last"), v=("volume", "sum"))
    g = d.groupby(level=0)
    prev_c = g["c"].shift(1)
    tr = pd.concat([d["h"] - d["l"], (d["h"] - prev_c).abs(), (d["l"] - prev_c).abs()], axis=1).max(axis=1)
    ctx = pd.DataFrame({
        "prev_high": g["h"].shift(1), "prev_low": g["l"].shift(1), "prev_close": prev_c,
        "adr": ((d["h"] - d["l"]) / d["c"]).groupby(level=0).transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1)),
        "atr": tr.groupby(level=0).transform(lambda s: s.rolling(14, min_periods=5).mean().shift(1)),
        "avg_volume": d["v"].groupby(level=0).transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1)),
        "nr7": (d["h"] - d["l"]).groupby(level=0).transform(lambda s: (s <= s.rolling(7, min_periods=7).min()).astype(float).shift(1)),
    })
    return ctx


def run(days: int | None = None, warmup: int = 6, reset: bool = True, progress=None, seed: int = 0) -> dict:
    cfg = config()
    root = DIR / "replay"
    if reset and root.exists():
        for f in root.glob("*"):
            f.unlink()
    bars = pd.read_parquet(BARS)
    ctx_all = daily_context(bars)
    bars = bars.assign(day=bars["ts"].dt.normalize())
    all_days = sorted(bars["day"].unique())[warmup:]
    if days:
        all_days = all_days[-days:]
    agent = Agent(root, cfg, seed)
    results = []
    for n, day in enumerate(all_days):
        db = bars[bars["day"] == day]
        nifty = db[db["ticker"] == "^NSEI"].set_index("ts")[["open", "high", "low", "close", "volume"]]
        per = {t: g.set_index("ts")[["open", "high", "low", "close", "volume"]] for t, g in db.groupby("ticker")
               if t != "^NSEI" and len(g) >= 20}
        clock = nifty.index if len(nifty) else None
        if clock is not None:                                         # align every stock to NIFTY's bar clock
            per = {t: b.reindex(clock).ffill().assign(volume=lambda x: x["volume"].fillna(0)) for t, b in per.items()
                   if b.index[0] <= clock[2]}
        try:
            cx = ctx_all.xs(day, level=1)
        except KeyError:
            continue
        ctx = {t: cx.loc[t].to_dict() for t in per if t in cx.index}
        res, trades, log = agent.trade_day(str(pd.Timestamp(day).date()), per, ctx, nifty)
        results.append(res.to_dict())
        if progress:
            progress(day, res, (n + 1) / len(all_days))
    return {"days": results, "summary": agent.summary(), "root": str(root)}


__all__ = ["download", "daily_context", "run", "np"]
