"""Market dashboard data: global markets and macro, breadth history and sector rotation (RRG).

- ``global_table``: world indices, crude, gold, USD/INR, the US 10-year yield and the dollar index from Yahoo (daily
  closes, cached on disk for 30 minutes), with 1D / 1W / 1M / YTD / 1Y changes and the 1-year correlation of *weekly*
  returns with NIFTY (weekly, because Asian, European and US sessions close at different times of the Indian day).
- ``breadth_history``: per day, the share of stocks above their 50- and 200-day averages, advancers minus decliners and
  its cumulative line, and new 52-week highs minus lows.
- ``rrg``: a relative rotation graph of NSE industries vs NIFTY on weekly data. RS = equal-weight industry index ÷
  NIFTY; RS-Ratio = 100 × RS ÷ its 10-week average (above 100 = outperforming its recent trend); RS-Momentum = 100 ×
  RS-Ratio ÷ RS-Ratio 4 weeks earlier (above 100 = relative strength improving). This is an open approximation of the
  JdK RRG idea (the commercial formula is proprietary), so levels will not match other vendors exactly.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import ROOT
from .data import Panel

CACHE = ROOT / "data" / "global.parquet"
GLOBAL = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "FTSE 100": "^FTSE", "DAX": "^GDAXI",
          "Nikkei 225": "^N225", "Hang Seng": "^HSI", "Shanghai": "000001.SS", "Brent crude": "BZ=F", "Gold": "GC=F",
          "Silver": "SI=F", "USD/INR": "INR=X", "Dollar index": "DX-Y.NYB", "US 10Y yield": "^TNX"}
GROUP = {"S&P 500": "US", "Nasdaq": "US", "Dow Jones": "US", "FTSE 100": "Europe", "DAX": "Europe", "Nikkei 225": "Asia",
         "Hang Seng": "Asia", "Shanghai": "Asia", "Brent crude": "Commodities", "Gold": "Commodities",
         "Silver": "Commodities", "USD/INR": "Currency & rates", "Dollar index": "Currency & rates",
         "US 10Y yield": "Currency & rates"}


def fetch_global(max_age: float = 1800, download=None) -> pd.DataFrame:
    """Daily closes (columns = names) for the last ~2 years. Falls back to a stale cache when offline."""
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < max_age:
        return pd.read_parquet(CACHE)
    try:
        if download is None:
            import yfinance as yf
            download = lambda: yf.download(list(GLOBAL.values()), period="2y", interval="1d",  # noqa: E731
                                           auto_adjust=False, progress=False, threads=True)
        raw = download()
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        inv = {v: k for k, v in GLOBAL.items()}
        df = close.rename(columns=inv)[[k for k in GLOBAL if k in close.rename(columns=inv).columns]]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None) if pd.DatetimeIndex(df.index).tz else pd.DatetimeIndex(df.index)
        df = df.dropna(how="all")
    except Exception:
        if CACHE.exists():
            return pd.read_parquet(CACHE)
        raise
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def _chg(s: pd.Series, days: int, diff: bool = False) -> float:
    """Change over ``days`` calendar days: percent, or a plain difference (``diff``, for yields in % points)."""
    s = s.dropna()
    if len(s) < 2:
        return float("nan")
    ref = s[s.index <= s.index[-1] - pd.Timedelta(days=days)] if days else s.iloc[:-1]
    if not len(ref):
        return float("nan")
    return float(s.iloc[-1] - ref.iloc[-1]) if diff else float(s.iloc[-1] / ref.iloc[-1] - 1)


def global_table(closes: pd.DataFrame, nifty: pd.Series) -> pd.DataFrame:
    rows = []
    nw = nifty.resample("W-FRI").last().pct_change()
    for name in closes.columns:
        s = closes[name].dropna()
        if len(s) < 2:
            continue
        ytd_ref = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
        w = s.resample("W-FRI").last().pct_change()
        j = w.index.intersection(nw.index)[-52:]
        corr = float(w[j].corr(nw[j])) if len(j) > 20 else float("nan")
        rate = name == "US 10Y yield"                     # a yield: changes in percentage points, not percent
        last = float(s.iloc[-1])
        ytd = (last - ytd_ref.iloc[-1] if rate else last / ytd_ref.iloc[-1] - 1) if len(ytd_ref) else float("nan")
        rows.append({"Market": name, "Group": GROUP.get(name, ""), "Last": last, "1D": _chg(s, 0, rate),
                     "1W": _chg(s, 7, rate), "1M": _chg(s, 30, rate), "YTD": float(ytd), "1Y": _chg(s, 365, rate),
                     "Corr. with NIFTY (weekly, 1y)": corr, "As of": s.index[-1], "unit": "pp" if rate else "%"})
    return pd.DataFrame(rows).set_index("Market")


def breadth_history(p: Panel, years: float = 3.0) -> pd.DataFrame:
    c = p.close
    ma50, ma200 = c.rolling(50, min_periods=40).mean(), c.rolling(200, min_periods=150).mean()
    above50 = (c > ma50).where(ma50.notna() & c.notna()).mean(axis=1)
    above200 = (c > ma200).where(ma200.notna() & c.notna()).mean(axis=1)
    r = c.pct_change(fill_method=None)
    adv, dec = (r > 0).sum(axis=1), (r < 0).sum(axis=1)
    hi52 = c.rolling(252, min_periods=200).max().shift()
    lo52 = c.rolling(252, min_periods=200).min().shift()
    nh, nl = (c > hi52).sum(axis=1), (c < lo52).sum(axis=1)
    out = pd.DataFrame({"% above 50-day": above50, "% above 200-day": above200, "Advancers": adv, "Decliners": dec,
                        "Adv − Dec": adv - dec, "New highs": nh, "New lows": nl, "Highs − lows": nh - nl,
                        "NIFTY": p.bench.reindex(c.index)})
    out = out[out.index >= out.index[-1] - pd.DateOffset(days=int(365 * years))]
    out["A/D line"] = out["Adv − Dec"].cumsum()
    return out


def sector_indices(p: Panel, min_members: int = 3) -> pd.DataFrame:
    r = p.close.pct_change(fill_method=None)
    out = {}
    for sec, members in p.sectors.groupby(p.sectors).groups.items():
        cols = [t for t in members if t in r.columns]
        if len(cols) >= min_members:
            out[sec] = (1 + r[cols].mean(axis=1).fillna(0)).cumprod()
    return pd.DataFrame(out)


def rrg(p: Panel, tail: int = 8, ratio_n: int = 10, mom_n: int = 4) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """(latest table, tails) — tails[sector] has the last ``tail`` weeks of RS-Ratio / RS-Momentum."""
    idx = sector_indices(p).resample("W-FRI").last()
    b = p.bench.resample("W-FRI").last().reindex(idx.index)
    rs = idx.div(b, axis=0)
    ratio = 100 * rs / rs.rolling(ratio_n).mean()
    mom = 100 * ratio / ratio.shift(mom_n)
    rows, tails = [], {}
    for sec in idx.columns:
        d = pd.DataFrame({"RS-Ratio": ratio[sec], "RS-Momentum": mom[sec]}).dropna()
        if len(d) < 2:
            continue
        tails[sec] = d.iloc[-tail:]
        x, y = d.iloc[-1]
        x4, y4 = d.iloc[-1 - mom_n] if len(d) > mom_n else (np.nan, np.nan)
        rows.append({"Sector": sec, "RS-Ratio": x, "RS-Momentum": y, "Quadrant": quadrant(x, y),
                     "4 weeks ago": quadrant(x4, y4), "Stocks": int((p.sectors == sec).sum()),
                     "13-week return vs NIFTY": float(rs[sec].iloc[-1] / rs[sec].iloc[-14] - 1) if len(rs) > 14 else np.nan})
    tab = pd.DataFrame(rows).set_index("Sector").sort_values(["RS-Ratio"], ascending=False) if rows else pd.DataFrame()
    return tab, tails


def quadrant(x: float, y: float) -> str:
    if not (np.isfinite(x) and np.isfinite(y)):
        return "–"
    if x >= 100:
        return "Leading" if y >= 100 else "Weakening"
    return "Improving" if y >= 100 else "Lagging"
