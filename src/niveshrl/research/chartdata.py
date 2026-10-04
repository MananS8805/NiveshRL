"""Chart data for the TradingView-style stock chart: bars at any timeframe and indicators on one OHLCV frame.

Timeframes: ``1m`` (Yahoo, last 7 days), ``5m`` / ``15m`` (last 60 days), ``1h`` (last 2 years), ``1D`` (the adjusted
daily panel) and ``1W`` (weekly bars resampled from the daily panel). Intraday bars are converted to IST, cached on
disk under ``data/charts`` and only re-downloaded when stale.

Indicators are pure functions of an OHLCV frame (columns Open/High/Low/Close/Volume) and never look ahead, except
``fibonacci`` which by definition is drawn from the swing high/low of the bars you are looking at.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import ROOT
from .data import Panel

CACHE = ROOT / "data" / "charts"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "1D", "1W"]
INTRADAY = {"1m": ("7d", 60), "5m": ("60d", 300), "15m": ("60d", 600), "1h": ("730d", 1800)}   # period, cache TTL (s)


def daily_bars(p: Panel, ticker: str) -> pd.DataFrame:
    df = pd.DataFrame({"Close": p.close[ticker]})
    for k, src in [("Open", p.open), ("High", p.high), ("Low", p.low)]:
        df[k] = src[ticker] if src is not None and ticker in src else np.nan
    df["Volume"] = p.volume[ticker] if ticker in p.volume else np.nan
    return df.dropna(subset=["Close"])[["Open", "High", "Low", "Close", "Volume"]]


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly bars (weeks ending Friday), labelled with the week's last trading day."""
    g = df.groupby(df.index.to_period("W-FRI"))
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum(min_count=1)})
    out.index = g.apply(lambda x: x.index[-1]).to_numpy()
    return out


def _clean_yahoo(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1) if raw.columns.nlevels > 1 else raw
    df = raw[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def intraday_bars(ticker: str, tf: str, max_age: float | None = None, download=None) -> pd.DataFrame:
    """Yahoo intraday bars in IST (tz-naive), via a small on-disk cache. Raises on network failure with no cache."""
    period, ttl = INTRADAY[tf]
    ttl = ttl if max_age is None else max_age
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker.replace('^', '_')}_{tf}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        return pd.read_parquet(path)
    try:
        if download is None:
            import yfinance as yf
            download = lambda: yf.download(ticker, period=period, interval=tf, auto_adjust=False,  # noqa: E731
                                           progress=False, threads=False)
        df = _clean_yahoo(download())
    except Exception:
        if path.exists():                              # stale beats nothing; the caller shows the bar times
            return pd.read_parquet(path)
        raise
    if len(df):
        df.to_parquet(path)
    return df


def bars(p: Panel, ticker: str, tf: str) -> pd.DataFrame:
    if tf == "1D":
        return daily_bars(p, ticker)
    if tf == "1W":
        return weekly(daily_bars(p, ticker))
    return intraday_bars(ticker, tf)


def is_intraday(index) -> bool:
    idx = pd.DatetimeIndex(index)
    return len(idx) > 1 and (idx[1:] - idx[:-1]).min() < pd.Timedelta(hours=20)


# --------------------------------------------------------------------------- indicators (one frame)
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP: cumulative (typical price × volume) ÷ cumulative volume, reset every trading day."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].fillna(0)
    day = pd.DatetimeIndex(df.index).normalize()
    cum_pv = (tp * v).groupby(day).cumsum()
    cum_v = v.groupby(day).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """(line, direction): the trailing Supertrend line and +1 (uptrend, line below price) / −1 (downtrend)."""
    a = atr(df, n).to_numpy()
    hl2 = ((df["High"] + df["Low"]) / 2).to_numpy()
    c = df["Close"].to_numpy()
    line, dirn = np.full(len(c), np.nan), np.full(len(c), np.nan)
    fu = fd = np.nan
    d = 1
    for t in range(len(c)):
        if not (np.isfinite(a[t]) and np.isfinite(c[t]) and np.isfinite(hl2[t])):
            continue
        nu, nd = hl2[t] - mult * a[t], hl2[t] + mult * a[t]
        if np.isfinite(fu) and t > 0 and np.isfinite(c[t - 1]):
            fu = max(nu, fu) if c[t - 1] > fu else nu
            fd = min(nd, fd) if c[t - 1] < fd else nd
        else:
            fu, fd = nu, nd
        if c[t] > fd:
            d = 1
        elif c[t] < fu:
            d = -1
        dirn[t] = d
        line[t] = fu if d == 1 else fd
    return pd.Series(line, index=df.index), pd.Series(dirn, index=df.index)


def pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Classic floor pivots and the CPR for each bar, from the *previous* period's high/low/close.

    The period is the previous trading day for intraday bars, the previous week for daily bars and the previous
    month for weekly bars. Columns: P, BC, TC (central pivot range), R1, S1, R2, S2.
    """
    idx = pd.DatetimeIndex(df.index)
    if is_intraday(idx):
        key = idx.normalize()
    else:
        step = (idx[1:] - idx[:-1]).median() if len(idx) > 1 else pd.Timedelta(days=1)
        key = idx.to_period("M" if step >= pd.Timedelta(days=5) else "W-FRI").to_timestamp()
    g = df.groupby(key)
    per = pd.DataFrame({"H": g["High"].max(), "L": g["Low"].min(), "C": g["Close"].last()}).shift()
    P = (per["H"] + per["L"] + per["C"]) / 3
    BC = (per["H"] + per["L"]) / 2
    out = pd.DataFrame({"P": P, "BC": BC, "TC": 2 * P - BC, "R1": 2 * P - per["L"], "S1": 2 * P - per["H"],
                        "R2": P + (per["H"] - per["L"]), "S2": P - (per["H"] - per["L"])})
    lo, hi = out[["BC", "TC"]].min(axis=1), out[["BC", "TC"]].max(axis=1)
    out["BC"], out["TC"] = lo, hi
    return out.reindex(key).set_axis(df.index)


FIB_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def fibonacci(df: pd.DataFrame) -> dict:
    """Retracement levels between the swing high and low of ``df`` (the bars on screen). If the high came after the
    low (an up-swing), levels are measured down from the high; otherwise up from the low."""
    if df.empty:
        return {}
    hi_i, lo_i = df["High"].idxmax(), df["Low"].idxmin()
    hi, lo = float(df["High"].max()), float(df["Low"].min())
    up = hi_i > lo_i
    rng = hi - lo
    levels = {f"{r:.1%}".replace(".0%", "%"): (hi - r * rng if up else lo + r * rng) for r in FIB_LEVELS}
    return {"levels": levels, "up": up, "high": hi, "low": lo, "high_at": hi_i, "low_at": lo_i}
