"""Technical indicators over the NIFTY 200 panel: the reference implementation.

Every indicator is causal (row t uses data up to t's close) and vectorised over
all stocks. The C++ core (``niveshrl_core``) must match these numbers; the
parity tests compare against this file. Conventions:

- Wilder smoothing (alpha = 1/n) for RSI, ATR and ADX, as on most charting tools.
- EMA with ``adjust=False`` (the recursive definition).
- 52-week = 252 trading days.

``indicator_frames`` returns full history (used by the next-day model);
``snapshot`` returns one row per stock for the latest day (used by the screener).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel


def _wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = _wilder(d.clip(lower=0), n)
    dn = _wilder(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(close.notna())


def macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def true_range(p: Panel) -> pd.DataFrame:
    prev = p.close.shift(1)
    return pd.concat([p.high - p.low, (p.high - prev).abs(), (p.low - prev).abs()], keys=range(3)).groupby(level=1).max() \
        .reindex(p.close.index)


def atr(p: Panel, n: int = 14) -> pd.DataFrame:
    return _wilder(true_range(p), n)


def adx(p: Panel, n: int = 14) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    up = p.high.diff()
    dn = -p.low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = _wilder(true_range(p), n)
    pdi = 100 * _wilder(plus_dm, n) / tr
    mdi = 100 * _wilder(minus_dm, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wilder(dx, n), pdi, mdi


def supertrend_dir(p: Panel, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """+1 when price is above the Supertrend line (uptrend), -1 below. Path-dependent, so a loop per stock.

    Uses the C++ core when installed (~870x faster, identical output; tests/test_core_parity.py).
    """
    try:
        import niveshrl_core
        hl2_ = ((p.high + p.low) / 2).to_numpy()
        out = niveshrl_core.supertrend(hl2_, atr(p, n).to_numpy(), p.close.to_numpy(), mult)
        return pd.DataFrame(out, index=p.close.index, columns=p.close.columns)
    except ImportError:
        pass
    a = atr(p, n).to_numpy()
    hl2 = ((p.high + p.low) / 2).to_numpy()
    c = p.close.to_numpy()
    up_b, dn_b = hl2 - mult * a, hl2 + mult * a
    T, N = c.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        fu = fd = np.nan
        d = 1
        for t in range(T):
            if not (np.isfinite(up_b[t, j]) and np.isfinite(c[t, j])):
                continue
            nu, nd = up_b[t, j], dn_b[t, j]
            if np.isfinite(fu) and np.isfinite(c[t - 1, j]):
                fu = max(nu, fu) if c[t - 1, j] > fu else nu
                fd = min(nd, fd) if c[t - 1, j] < fd else nd
            else:
                fu, fd = nu, nd
            if c[t, j] > fd:
                d = 1
            elif c[t, j] < fu:
                d = -1
            out[t, j] = d
    return pd.DataFrame(out, index=p.close.index, columns=p.close.columns)


def indicator_frames(p: Panel) -> dict[str, pd.DataFrame]:
    """Full-history indicator frames (dates x tickers)."""
    c, v = p.close, p.volume.reindex_like(p.close)
    f: dict[str, pd.DataFrame] = {}
    for n, k in [(1, "ret_1d"), (5, "ret_1w"), (21, "ret_1m"), (63, "ret_3m"), (126, "ret_6m"), (252, "ret_1y")]:
        f[k] = c / c.shift(n) - 1
    f["rsi14"] = rsi(c)
    f["macd"], f["macd_signal"], f["macd_hist"] = macd(c)
    for n in (20, 50, 200):
        f[f"sma{n}"] = c.rolling(n, min_periods=n).mean()
        f[f"ema{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
        f[f"vs_sma{n}"] = c / f[f"sma{n}"] - 1
    above = (f["sma50"] > f["sma200"]).astype(float).where(f["sma200"].notna())
    cross = above.diff()
    f["golden_cross_5d"] = (cross.rolling(5).max() > 0).astype(float).where(above.notna())
    f["death_cross_5d"] = (cross.rolling(5).min() < 0).astype(float).where(above.notna())
    hi252, lo252 = c.rolling(252, min_periods=120).max(), c.rolling(252, min_periods=120).min()
    f["from_52w_high"] = c / hi252 - 1
    f["from_52w_low"] = c / lo252 - 1
    f["new_52w_high"] = (c >= hi252).astype(float).where(hi252.notna())
    f["new_52w_low"] = (c <= lo252).astype(float).where(lo252.notna())
    f["vol_ratio"] = v / v.rolling(20, min_periods=10).mean().replace(0, np.nan)
    f["turnover_cr"] = (c * v).rolling(20, min_periods=10).mean() / 1e7
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    f["bb_width"] = 4 * sd / mid
    r = c.pct_change(fill_method=None)
    f["vol_60d"] = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    br = p.bench.reindex(c.index).pct_change(fill_method=None)
    f["beta_1y"] = r.rolling(252, min_periods=120).cov(br).div(br.rolling(252, min_periods=120).var(), axis=0)
    f["max_dd_1y"] = c / c.rolling(252, min_periods=60).max() - 1
    f["max_dd_1y"] = f["max_dd_1y"].rolling(252, min_periods=60).min()
    b = p.bench.reindex(c.index)
    for n, k in [(21, "rs_nifty_1m"), (63, "rs_nifty_3m")]:
        f[k] = (c / c.shift(n) - 1).sub(b / b.shift(n) - 1, axis=0)
    sec = p.sectors.reindex(c.columns)
    f["rs_sector_3m"] = f["ret_3m"] - f["ret_3m"].T.groupby(sec).transform("median").T
    f["donchian_breakout"] = (c > c.shift(1).rolling(20, min_periods=20).max()).astype(float).where(c.notna())
    if p.high is not None:
        f["atr14"] = atr(p)
        f["atr_pct"] = f["atr14"] / c
        f["adx14"], f["plus_di"], f["minus_di"] = adx(p)
        f["gap_pct"] = p.open / c.shift(1) - 1
        f["range_pct"] = (p.high - p.low) / c.shift(1)
        f["close_loc"] = (c - p.low) / (p.high - p.low).replace(0, np.nan)
        f["supertrend"] = supertrend_dir(p)
    return f


# Screener columns: key -> (label, group, format)
TECH_COLUMNS = {
    "ret_1d": ("1D %", "Returns", "pct"), "ret_1w": ("1W %", "Returns", "pct"), "ret_1m": ("1M %", "Returns", "pct"),
    "ret_3m": ("3M %", "Returns", "pct"), "ret_6m": ("6M %", "Returns", "pct"), "ret_1y": ("1Y %", "Returns", "pct"),
    "gap_pct": ("Gap %", "Returns", "pct"),
    "rsi14": ("RSI 14", "Momentum", "num"), "macd_hist": ("MACD hist", "Momentum", "num"),
    "adx14": ("ADX 14", "Trend", "num"), "supertrend": ("Supertrend (+1 up)", "Trend", "num"),
    "vs_sma20": ("vs SMA20 %", "Trend", "pct"), "vs_sma50": ("vs SMA50 %", "Trend", "pct"),
    "vs_sma200": ("vs SMA200 %", "Trend", "pct"), "golden_cross_5d": ("Golden cross (5d)", "Trend", "flag"),
    "death_cross_5d": ("Death cross (5d)", "Trend", "flag"), "donchian_breakout": ("20d breakout", "Trend", "flag"),
    "from_52w_high": ("From 52w high %", "Range", "pct"), "from_52w_low": ("From 52w low %", "Range", "pct"),
    "new_52w_high": ("New 52w high", "Range", "flag"), "new_52w_low": ("New 52w low", "Range", "flag"),
    "bb_pctb": ("Bollinger %B", "Volatility", "num"), "bb_width": ("Bollinger width", "Volatility", "pct"),
    "atr_pct": ("ATR %", "Volatility", "pct"), "vol_60d": ("Volatility 60d", "Volatility", "pct"),
    "beta_1y": ("Beta 1Y", "Risk", "num"), "max_dd_1y": ("Max DD 1Y %", "Risk", "pct"),
    "vol_ratio": ("Volume / 20d avg", "Volume", "num"), "turnover_cr": ("Avg turnover ₹Cr", "Volume", "num"),
    "rs_nifty_1m": ("RS vs NIFTY 1M", "Relative", "pct"), "rs_nifty_3m": ("RS vs NIFTY 3M", "Relative", "pct"),
    "rs_sector_3m": ("RS vs sector 3M", "Relative", "pct"),
}


def snapshot(frames: dict[str, pd.DataFrame], p: Panel) -> pd.DataFrame:
    """Latest available value of every screener indicator, one row per stock."""
    rows = {k: frames[k].ffill(limit=3).iloc[-1] for k in TECH_COLUMNS if k in frames}
    snap = pd.DataFrame(rows)
    snap.insert(0, "close", p.close.ffill(limit=3).iloc[-1])
    snap.insert(0, "sector", p.sectors.reindex(snap.index))
    snap.insert(0, "name", p.names.reindex(snap.index))
    return snap
