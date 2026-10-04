"""Per-stock deep-dive analytics: the views industry platforms show for a single stock.

All pure functions over the price Panel, the daily screener table and Yahoo data:
- ``report_card``: valuation, quality, growth, momentum and risk scores (0-100) as percentiles within the stock's NSE
  sector, each with plain-language reasons ("cheaper than 72% of IT peers on P/E").
- ``peers``: the sector peer table.
- ``relative_strength``: price relative to NIFTY and to an equal-weight index of its sector; 1-year beta and correlation.
- ``seasonality``: average return and hit rate by calendar month; ``drawdowns``: the deepest falls with recovery time.
- ``results_reaction``: how the stock moved around past quarterly results (Yahoo earnings dates), with EPS surprise.
- ``dividends``: dividend history and trailing yield.
- ``vol_cone``: price ranges the stock stays inside with 68% / 95% probability over 5-60 days, from forecast volatility
  (log-normal, zero drift: a statement about volatility, not direction).
- ``why_moving``: a one-paragraph explanation of today's move (vs NIFTY and sector, volume, gap, news, results, breakout).
"""
from __future__ import annotations

import math
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from .data import Panel

IST = timezone(timedelta(hours=5, minutes=30))

# category -> [(screener column, label, higher_is_better)]
CARD = {
    "Valuation": [("trailingPE", "P/E", False), ("priceToBook", "P/B", False), ("enterpriseToEbitda", "EV/EBITDA", False)],
    "Quality": [("returnOnEquity", "ROE", True), ("operatingMargins", "operating margin", True),
                ("debtToEquity", "debt/equity", False)],
    "Growth": [("revenueGrowth", "revenue growth", True), ("earningsGrowth", "earnings growth", True)],
    "Momentum": [("ret_3m", "3-month return", True), ("ret_6m", "6-month return", True), ("vs_sma200", "distance above the 200-day average", True)],
    "Risk": [("vol_60d", "volatility", False), ("beta_1y", "beta", False), ("max_dd_1y", "1-year max drawdown", True)],
}
VERDICT = {"Valuation": ("expensive", "fair", "cheap"), "Quality": ("weak", "average", "strong"),
           "Growth": ("slow", "average", "fast"), "Momentum": ("weak", "neutral", "strong"),
           "Risk": ("high risk", "average risk", "low risk")}


def report_card(table: pd.DataFrame, ticker: str) -> dict:
    """{category: {"score": 0-100 or nan, "verdict": str, "reasons": [str], "peers": n}} — percentiles within the sector."""
    if table is None or ticker not in table.index:
        return {}
    sec = table.loc[ticker, "sector"] if "sector" in table else None
    grp = table[table["sector"] == sec] if sec is not None else table
    if len(grp) < 4:                                  # too few sector peers: use the whole NIFTY 200
        grp, sec = table, "NIFTY 200"
    out = {}
    for cat, items in CARD.items():
        scores, reasons = [], []
        for col, label, hib in items:
            if col not in table:
                continue
            x = pd.to_numeric(grp[col], errors="coerce")
            if col in ("trailingPE", "priceToBook", "enterpriseToEbitda"):
                x = x.where(x > 0)                    # negative multiples (losses) are not 'cheap'
            v = pd.to_numeric(pd.Series([table.loc[ticker, col]]), errors="coerce").iloc[0]
            xs = x.dropna()
            if not np.isfinite(v) or len(xs) < 4 or (col in ("trailingPE", "priceToBook", "enterpriseToEbitda") and v <= 0):
                continue
            pct = float((xs < v).mean())
            sc = pct if hib else 1 - pct
            scores.append(sc)
            better = "higher" if hib else "lower"
            if sc >= 0.5:
                reasons.append(f"better {label} than {sc:.0%} of {sec} peers ({better} is better)")
            else:
                reasons.append(f"worse {label} than {1 - sc:.0%} of {sec} peers ({better} is better)")
        score = float(np.mean(scores) * 100) if scores else float("nan")
        lo, mid, hi = VERDICT[cat]
        verdict = "–" if not scores else hi if score >= 66 else lo if score < 34 else mid
        out[cat] = {"score": score, "verdict": verdict, "reasons": reasons, "peers": int(len(grp)), "group": sec}
    return out


def peers(table: pd.DataFrame, ticker: str, cols: list[str] | None = None, n: int = 15) -> pd.DataFrame:
    if table is None or ticker not in table.index:
        return pd.DataFrame()
    cols = cols or ["close", "ret_1m", "ret_1y", "marketCap", "trailingPE", "priceToBook", "returnOnEquity",
                    "revenueGrowth", "debtToEquity", "analyst_score", "prob_up"]
    sec = table.loc[ticker, "sector"]
    g = table[table["sector"] == sec]
    g = g.sort_values("marketCap", ascending=False) if "marketCap" in g else g
    return g[[c for c in cols if c in g]].head(n)


def relative_strength(p: Panel, ticker: str, years: float = 1.0) -> dict:
    c = p.close[ticker].dropna()
    start = c.index[-1] - pd.DateOffset(days=int(365 * years))
    c = c[c.index >= start]
    b = p.bench.reindex(c.index).ffill()
    sec = p.sectors.get(ticker)
    peers_ = [t for t in p.close.columns if p.sectors.get(t) == sec and t != ticker]
    sec_idx = (1 + p.close[peers_].reindex(c.index).pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod() if peers_ else None
    rs_n = (c / c.iloc[0]) / (b / b.iloc[0])
    rs_s = (c / c.iloc[0]) / (sec_idx / sec_idx.iloc[0]) if sec_idx is not None else None
    r, rb = c.pct_change().dropna(), b.pct_change().reindex(c.index).dropna()
    j = r.index.intersection(rb.index)
    beta = float(np.cov(r[j], rb[j])[0, 1] / np.var(rb[j], ddof=1)) if len(j) > 30 else float("nan")
    corr = float(np.corrcoef(r[j], rb[j])[0, 1]) if len(j) > 30 else float("nan")
    return {"rs_nifty": rs_n, "rs_sector": rs_s, "beta": beta, "corr": corr, "sector": sec, "n_peers": len(peers_),
            "vs_nifty": float(rs_n.iloc[-1] - 1), "vs_sector": float(rs_s.iloc[-1] - 1) if rs_s is not None else float("nan")}


def seasonality(p: Panel, ticker: str) -> pd.DataFrame:
    c = p.close[ticker].dropna()
    m = c.resample("ME").last().pct_change().dropna()
    if m.empty:
        return pd.DataFrame()
    g = m.groupby(m.index.month)
    b = p.bench.resample("ME").last().pct_change().reindex(m.index)
    gb = (m - b).groupby(m.index.month)
    import calendar
    return pd.DataFrame({"avg return": g.mean(), "median": g.median(), "up years": g.apply(lambda x: (x > 0).mean()),
                         "avg vs NIFTY": gb.mean(), "years": g.size()},
                        ).rename(index=lambda i: calendar.month_abbr[i])


def drawdowns(p: Panel, ticker: str, top: int = 5) -> pd.DataFrame:
    c = p.close[ticker].dropna()
    peak = c.cummax()
    dd = c / peak - 1
    out, i = [], 0
    vals, idx = dd.to_numpy(), dd.index
    while i < len(vals):
        if vals[i] < 0:
            j = i
            while j < len(vals) and vals[j] < 0:
                j += 1
            seg = dd.iloc[i:j]
            trough = seg.idxmin()
            out.append({"peak": idx[i - 1] if i > 0 else idx[i], "trough": trough, "depth": float(seg.min()),
                        "recovered": idx[j] if j < len(vals) else None,
                        "days to recover": int((idx[j] - trough).days) if j < len(vals) else None})
            i = j
        else:
            i += 1
    df = pd.DataFrame(out)
    return df.nsmallest(top, "depth") if len(df) else df


def reaction_day(ts: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The first session that could react to an announcement at ``ts``: the same day if announced before 15:30 IST
    on a trading day, otherwise the next trading day."""
    t = ts.tz_convert(IST) if ts.tzinfo else ts
    day = pd.Timestamp(t.date())
    after_close = (t.hour * 60 + t.minute) >= 15 * 60 + 30 if ts.tzinfo else False
    pos = trading_days.searchsorted(day + (pd.Timedelta(days=1) if after_close else pd.Timedelta(0)))
    return trading_days[pos] if pos < len(trading_days) else None


def results_reaction(p: Panel, ticker: str, earnings: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Moves around past results: reaction day vs the previous close, minus NIFTY, and the 20-day drift after."""
    if earnings is None or earnings.empty:
        return pd.DataFrame(), {}
    c = p.close[ticker].dropna()
    b = p.bench.reindex(c.index).ffill()
    rows = []
    for ts, r in earnings.iterrows():
        d = reaction_day(pd.Timestamp(ts), c.index)
        if d is None or d not in c.index:
            continue
        i = c.index.get_loc(d)
        if i < 1:
            continue
        move = c.iloc[i] / c.iloc[i - 1] - 1
        bm = b.iloc[i] / b.iloc[i - 1] - 1
        drift = (c.iloc[min(i + 20, len(c) - 1)] / c.iloc[i] - 1) - (b.iloc[min(i + 20, len(c) - 1)] / b.iloc[i] - 1) \
            if i + 20 < len(c) else float("nan")
        rows.append({"results": d, "EPS est.": r.get("EPS Estimate"), "EPS actual": r.get("Reported EPS"),
                     "surprise %": r.get("Surprise(%)"), "day move": move, "vs NIFTY": move - bm, "next 20d vs NIFTY": drift})
    if not rows:
        return pd.DataFrame(), {}
    df = pd.DataFrame(rows).sort_values("results", ascending=False)
    if df.empty:
        return df, {}
    summ = {"quarters": len(df), "avg abs move": float(df["day move"].abs().mean()),
            "up reactions": float((df["day move"] > 0).mean()),
            "avg 20d drift": float(df["next 20d vs NIFTY"].mean()),
            "beat estimate": float((df["surprise %"] > 0).mean()) if df["surprise %"].notna().any() else float("nan")}
    return df, summ


def dividends(divs: pd.Series | None, price: float, asof: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Dividend history, the total paid in the 12 months before ``asof`` (default today) and its yield on ``price``."""
    if divs is None or len(divs) == 0:
        return pd.DataFrame(), {}
    d = divs.copy()
    d.index = pd.to_datetime(d.index).tz_localize(None) if getattr(d.index, "tz", None) else pd.to_datetime(d.index)
    by_year = d.groupby(d.index.year).sum()
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize()
    last12 = d[(d.index > asof - pd.DateOffset(years=1)) & (d.index <= asof)].sum()
    return (pd.DataFrame({"dividend ₹": d}).sort_index(ascending=False).head(20),
            {"trailing 12m ₹": float(last12), "trailing yield": float(last12 / price) if price else float("nan"),
             "years paid": int((by_year > 0).sum()), "by year": by_year})


def vol_cone(price: float, sigma_annual: float, horizons=(5, 10, 20, 60)) -> pd.DataFrame:
    """Log-normal, zero-drift price ranges: 68% (±1σ) and 95% (±1.96σ) bands after h trading days."""
    rows = {}
    for h in horizons:
        s = sigma_annual * math.sqrt(h / 252)
        rows[f"{h} days"] = {"95% low": price * math.exp(-1.96 * s), "68% low": price * math.exp(-s),
                             "68% high": price * math.exp(s), "95% high": price * math.exp(1.96 * s), "1σ move": s}
    return pd.DataFrame(rows).T


def why_moving(ticker: str, table: pd.DataFrame | None, p: Panel, news: pd.DataFrame | None = None,
               live_change: float | None = None) -> str:
    c = p.close[ticker].dropna()
    r = live_change if live_change is not None else float(c.iloc[-1] / c.iloc[-2] - 1)
    b = p.bench.dropna()
    br = float(b.iloc[-1] / b.iloc[-2] - 1)
    sec = p.sectors.get(ticker)
    peers_ = [t for t in p.close.columns if p.sectors.get(t) == sec and t != ticker]
    sr = float((p.close[peers_].iloc[-1] / p.close[peers_].iloc[-2] - 1).mean()) if peers_ else float("nan")
    name = ticker.replace(".NS", "")
    parts = [f"{name} {'rose' if r >= 0 else 'fell'} {abs(r):.1%} while NIFTY moved {br:+.1%} and {sec} stocks averaged {sr:+.1%}."]
    if np.isfinite(sr) and abs(r - sr) < 0.005 and abs(sr) > 0.005:
        parts.append("The move is mostly the sector moving together, not stock-specific.")
    elif np.isfinite(sr) and abs(r - sr) >= 0.015:
        parts.append(f"That is {abs(r - sr):.1%} {'better' if r > sr else 'worse'} than its sector: something stock-specific.")
    row = table.loc[ticker] if table is not None and ticker in table.index else pd.Series(dtype=float)
    vr = row.get("vol_ratio")
    if vr is not None and np.isfinite(vr):
        parts.append(f"Volume was {vr:.1f}× its 20-day average" + (" (unusually heavy: the move had conviction)." if vr >= 1.5
                     else " (light: the move had little conviction)." if vr < 0.8 else "."))
    gap = row.get("gap_pct")
    if gap is not None and np.isfinite(gap) and abs(gap) >= 0.01:
        parts.append(f"It opened with a {gap:+.1%} gap, so overnight news or global cues drove much of it.")
    dte = row.get("days_to_earnings")
    if dte is not None and np.isfinite(dte) and -3 <= dte <= 7:
        parts.append(f"Quarterly results are {'due in ' + str(int(dte)) + ' days' if dte >= 0 else 'just out'}: moves around results are common.")
    if news is not None and len(news):
        n = news[news["ticker"] == ticker]
        if len(n):
            top = n.sort_values("published", ascending=False).iloc[0]
            tone = "positive" if n["score"].mean() > 0.2 else "negative" if n["score"].mean() < -0.2 else "mixed"
            parts.append(f"{len(n)} headline(s) in 48h, tone {tone}; latest: \"{str(top['title'])[:110]}\".")
    st = row.get("breakout_state")
    if isinstance(st, str) and st in ("breakout", "failed", "extended"):
        parts.append({"breakout": "It just broke above its 20-day high (the pivot).",
                      "failed": "It recently broke out but has fallen back under the pivot (a failed breakout).",
                      "extended": "It is far above its pivot (extended): chasing here carries more risk."}[st])
    return " ".join(parts)


def cone_sigma(ticker: str, vol_forecasts: pd.DataFrame | None, table: pd.DataFrame | None,
               asof: pd.Timestamp, max_age_days: int = 45) -> tuple[float, str]:
    """Annualised σ for the cone: the vol LSTM's forecast if it is recent, else 60-day realised volatility."""
    if vol_forecasts is not None and ticker in vol_forecasts.index.get_level_values(1):
        v = vol_forecasts.xs(ticker, level=1)
        d = v.index[-1]
        if (asof - pd.Timestamp(d)).days <= max_age_days and np.isfinite(v["lstm"].iloc[-1]):
            return float(v["lstm"].iloc[-1]), f"vol LSTM forecast as of {pd.Timestamp(d):%d %b %Y}"
    if table is not None and ticker in table.index and np.isfinite(table.loc[ticker, "vol_60d"]):
        return float(table.loc[ticker, "vol_60d"]), "60-day realised volatility"
    return float("nan"), "no volatility estimate"


def yahoo_events(ticker: str) -> dict:
    """Earnings dates (with EPS estimate/actual/surprise) and dividends from Yahoo; each part may be None if unavailable."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    out = {"earnings": None, "dividends": None, "errors": []}
    try:
        out["earnings"] = t.get_earnings_dates(limit=40)
    except Exception as e:  # noqa: BLE001 - Yahoo endpoints fail in many ways; report, never fill in
        out["errors"].append(f"earnings dates: {str(e)[:80]}")
    try:
        out["dividends"] = t.dividends
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"dividends: {str(e)[:80]}")
    return out
