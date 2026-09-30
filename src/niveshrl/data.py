"""Download, clean, align and cache NSE market data.

Everything downstream consumes a single :class:`MarketData` object whose
arrays share one trading-day index (days on which the stocks traded).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    tickers: list[str]
    sectors: list[str]
    close: pd.DataFrame          # adjusted close, dates x tickers
    volume: pd.DataFrame         # shares traded, dates x tickers
    bench: pd.Series             # NIFTY 50 close
    vix: pd.Series               # India VIX (ffilled)
    fx: pd.Series                # USD/INR (ffilled)
    jumps: pd.DataFrame = field(default_factory=pd.DataFrame)  # flagged large moves

    @property
    def sector_names(self) -> list[str]:
        return sorted(set(self.sectors))

    @property
    def sector_ids(self) -> np.ndarray:
        names = self.sector_names
        return np.array([names.index(s) for s in self.sectors], dtype=np.int64)

    def split_mask(self, start: str, end: str) -> np.ndarray:
        return np.asarray((self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end)))


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, threads=True, group_by="column")
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")
    return df


def _cache_path(cfg: dict, name: str) -> Path:
    d = ROOT / cfg["data"]["cache_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.parquet"


def load_market_data(cfg: dict, refresh: bool = False) -> MarketData:
    uni = cfg["universe"]
    tickers = [s["ticker"] for s in uni["stocks"]]
    sectors = [s["sector"] for s in uni["stocks"]]
    start, end = cfg["data"]["start"], cfg["data"]["end"]

    stock_path = _cache_path(cfg, "stocks")
    ctx_path = _cache_path(cfg, "context")
    if refresh or not stock_path.exists() or not ctx_path.exists():
        raw = _download(tickers, start, end)
        stocks = pd.concat({"close": raw["Close"][tickers], "volume": raw["Volume"][tickers]}, axis=1)
        stocks.to_parquet(stock_path)
        ctx_tickers = [uni["benchmark"], uni["context"]["vix"], uni["context"]["fx"]]
        ctx_raw = _download(ctx_tickers, start, end)["Close"][ctx_tickers]
        ctx_raw.columns = ["bench", "vix", "fx"]
        ctx_raw.to_parquet(ctx_path)

    stocks = pd.read_parquet(stock_path)
    ctx = pd.read_parquet(ctx_path)
    return clean(stocks["close"], stocks["volume"], ctx, tickers, sectors, cfg)


def clean(close: pd.DataFrame, volume: pd.DataFrame, ctx: pd.DataFrame,
          tickers: list[str], sectors: list[str], cfg: dict) -> MarketData:
    """Align on stock trading days, fill short gaps, flag jumps and stale prices.

    The index comes from the stocks rather than from NIFTY, because yfinance's
    ^NSEI history only starts in Sep 2007. Benchmark-derived features are
    NaN (and so neutral after scaling) until NIFTY data exists.
    """
    max_ff = cfg["data"]["max_ffill_days"]
    close = close[tickers]
    idx = close.index[close.notna().mean(axis=1) >= 0.9]
    close = close.reindex(idx).ffill(limit=max_ff)
    volume = volume.reindex(idx)[tickers].fillna(0.0)
    # Drop leading days until every stock has a price, then any residual gaps.
    first_valid = close.apply(lambda s: s.first_valid_index()).max()
    close = close.loc[first_valid:]
    keep = close.notna().all(axis=1)
    close, volume = close[keep], volume.loc[close.index]
    bench = ctx["bench"].reindex(close.index).ffill(limit=max_ff)
    vix = ctx["vix"].reindex(close.index).ffill().bfill()
    fx = ctx["fx"].reindex(close.index).ffill().bfill()

    thr = cfg["data"]["jump_threshold"]
    # Repair one-day bad prints: a jump above the threshold that fully reverses
    # the next day (the two-day return is within 5%) is replaced by the prior close.
    r = close.pct_change()
    spike = (r.abs() > thr) & (((1 + r) * (1 + r.shift(-1)) - 1).abs() < 0.05)
    repaired = [(d.date(), t) for d, t in spike.stack()[lambda s: s].index]
    close = close.mask(spike).ffill()

    rets = close.pct_change()
    flagged = rets.abs().stack()
    flagged = flagged[flagged > thr]
    jumps = pd.DataFrame({"return": rets.stack()[flagged.index]})
    # Stale runs: the same price with zero volume for 10+ consecutive days.
    stale = (close.diff() == 0) & (volume == 0)
    runs = stale.apply(lambda s: s.astype(int).groupby((~s).cumsum()).cumsum().max())
    jumps.attrs["stale_runs"] = runs[runs >= 10].to_dict()
    jumps.attrs["repaired"] = repaired

    return MarketData(dates=close.index, tickers=tickers, sectors=sectors,
                      close=close, volume=volume, bench=bench, vix=vix, fx=fx, jumps=jumps)
