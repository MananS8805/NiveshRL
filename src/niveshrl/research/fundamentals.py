"""Company basics and financial statements from Yahoo Finance (yfinance).

Used by the market monitor's stock details panel. Everything here is as
reported by Yahoo: figures are in INR as filed. Where Yahoo has no value the
field is left blank rather than estimated.

Unit quirks handled here: ``dividendYield`` already comes in percent (0.5 =
0.5%), and ``debtToEquity`` in percent (36.7 = 0.37x).
"""
from __future__ import annotations

import pandas as pd

PROFILE_KEYS = ["longName", "sector", "industry", "website", "fullTimeEmployees", "longBusinessSummary"]
STAT_KEYS = ["currentPrice", "previousClose", "dayHigh", "dayLow", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
             "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook", "trailingEps", "bookValue",
             "dividendYield", "beta", "profitMargins", "operatingMargins", "returnOnEquity", "returnOnAssets",
             "debtToEquity", "revenueGrowth", "earningsGrowth", "totalRevenue", "ebitda", "totalDebt", "totalCash",
             "volume", "averageVolume", "heldPercentInsiders", "heldPercentInstitutions", "recommendationKey",
             "targetMeanPrice", "targetLowPrice", "targetHighPrice", "numberOfAnalystOpinions"]
INCOME_ROWS = ["Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "Net Income", "Diluted EPS"]
BALANCE_ROWS = ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity", "Total Debt",
                "Cash And Cash Equivalents"]
CASHFLOW_ROWS = ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"]


def _rows(df: pd.DataFrame | None, rows: list[str], n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.reindex([r for r in rows if r in df.index]).iloc[:, :n]
    out.columns = [c.strftime("%b %Y") if hasattr(c, "strftime") else str(c) for c in out.columns]
    return out


def fetch(ticker: str) -> dict:
    """Profile, key statistics and the latest statements for one NSE ticker (e.g. 'RELIANCE.NS')."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    out = {"profile": {k: info.get(k) for k in PROFILE_KEYS}, "stats": {k: info.get(k) for k in STAT_KEYS}}
    for name, attr, rows, n in [("quarterly", "quarterly_income_stmt", INCOME_ROWS, 5),
                                ("annual", "income_stmt", INCOME_ROWS, 4),
                                ("balance", "balance_sheet", BALANCE_ROWS, 4),
                                ("cashflow", "cashflow", CASHFLOW_ROWS, 4)]:
        try:
            out[name] = _rows(getattr(tk, attr), rows, n)
        except Exception:
            out[name] = pd.DataFrame()
    return out


def crore(v) -> str:
    """Indian-style money formatting: Rs lakh crore / crore."""
    if v is None or v != v:
        return "–"
    cr = v / 1e7
    if abs(cr) >= 1e5:
        return f"₹{cr / 1e5:,.2f} L Cr"
    return f"₹{cr:,.0f} Cr"


# --------------------------------------------------------------------------- whole-universe snapshot
SNAPSHOT_KEYS = ["marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda",
                 "trailingPegRatio", "trailingEps", "earningsGrowth", "revenueGrowth", "returnOnEquity",
                 "returnOnAssets", "operatingMargins", "profitMargins", "debtToEquity", "totalCash", "totalDebt",
                 "dividendYield", "payoutRatio", "heldPercentInsiders", "heldPercentInstitutions",
                 "recommendationMean", "recommendationKey", "numberOfAnalystOpinions", "targetMeanPrice",
                 "currentPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]


def _one(ticker: str, retries: int = 3) -> dict:
    """info + analyst recommendation trend + next earnings date for one ticker (with retry/backoff)."""
    import time

    import yfinance as yf

    row = {"ticker": ticker}
    for attempt in range(retries):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            row.update({k: info.get(k) for k in SNAPSHOT_KEYS})
            try:
                rec = tk.recommendations
                if rec is not None and len(rec):
                    rec = rec.set_index("period")
                    for per, tag in [("0m", "now"), ("-3m", "3m")]:
                        if per in rec.index:
                            r = rec.loc[per]
                            tot = float(r.sum())
                            row[f"buy_share_{tag}"] = float(r.get("strongBuy", 0) + r.get("buy", 0)) / tot if tot else None
            except Exception:
                pass
            try:
                cal = tk.calendar or {}
                ed = cal.get("Earnings Date")
                if ed:
                    row["next_earnings"] = pd.Timestamp(ed[0] if isinstance(ed, (list, tuple)) else ed)
            except Exception:
                pass
            return row
        except Exception as e:                      # rate limit / network: back off and retry
            row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(2 * (attempt + 1))
    return row


def snapshot_all(tickers: list[str], workers: int = 8, progress=None) -> pd.DataFrame:
    """Fundamentals + analyst data for every ticker, fetched in parallel. ``progress(done, total)`` is optional."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, t) for t in tickers]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if progress:
                progress(i, len(futs))
    df = pd.DataFrame(rows).set_index("ticker").reindex(tickers)
    for k in ("dividendYield", "debtToEquity"):             # Yahoo gives these in percent
        if k in df:
            df[k] = pd.to_numeric(df[k], errors="coerce") / 100
    return df
