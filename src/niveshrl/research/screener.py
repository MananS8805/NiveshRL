"""Stock screener: one table of technical, fundamental, analyst, sentiment and model
columns for the NIFTY 200, plus filters and presets.

Shared by the Streamlit SCRN screen and the desktop app (the C++ core runs the
same filters on a column snapshot when it's built; ``apply`` is the reference).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .technicals import TECH_COLUMNS

FUND_COLUMNS = {
    "marketCap": ("Market cap ₹Cr", "Valuation", "cr"), "trailingPE": ("P/E (ttm)", "Valuation", "num"),
    "forwardPE": ("P/E (fwd)", "Valuation", "num"), "priceToBook": ("P/B", "Valuation", "num"),
    "enterpriseToEbitda": ("EV/EBITDA", "Valuation", "num"), "trailingEps": ("EPS (ttm) ₹", "Valuation", "num"),
    "dividendYield": ("Dividend yield", "Income", "pct"), "payoutRatio": ("Payout ratio", "Income", "pct"),
    "returnOnEquity": ("ROE", "Quality", "pct"), "returnOnAssets": ("ROA", "Quality", "pct"),
    "operatingMargins": ("Operating margin", "Quality", "pct"), "profitMargins": ("Net margin", "Quality", "pct"),
    "debtToEquity": ("Debt / equity (x)", "Balance sheet", "num"),
    "revenueGrowth": ("Revenue growth (yoy)", "Growth", "pct"), "earningsGrowth": ("Earnings growth (yoy)", "Growth", "pct"),
    "heldPercentInsiders": ("Promoters / insiders", "Ownership", "pct"),
    "heldPercentInstitutions": ("Institutions", "Ownership", "pct"),
    "analyst_score": ("Analyst score (0-100)", "Analysts", "num"), "target_upside": ("Target upside", "Analysts", "pct"),
    "numberOfAnalystOpinions": ("# analysts", "Analysts", "num"), "days_to_earnings": ("Days to results", "Events", "num"),
}
SIGNAL_COLUMNS = {
    "prob_up": ("Next-day P(up)", "Models", "pct"), "monthly_pct": ("Monthly ranker %ile", "Models", "pct"),
    "sentiment": ("News sentiment", "News", "num"), "sentiment_adj": ("Sentiment (adj.)", "News", "num"),
    "n_news": ("Headlines (48h)", "News", "num"), "buzz": ("News buzz", "News", "num"),
}
COLUMNS = {**TECH_COLUMNS, **FUND_COLUMNS, **SIGNAL_COLUMNS}

OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "=": np.equal}

# Each preset: (description, [(column, op, value), ...]); values in the column's own units
# (percent columns as fractions, e.g. 0.15 = 15%).
PRESETS = {
    "Momentum breakout": ("Near the 52-week high on heavy volume, strong but not overbought RSI",
                          [("from_52w_high", ">=", -0.05), ("vol_ratio", ">=", 1.5), ("rsi14", ">=", 55), ("rsi14", "<=", 72)]),
    "Oversold quality": ("Beaten down (RSI < 35) but profitable with modest debt",
                         [("rsi14", "<", 35), ("profitMargins", ">", 0.10), ("debtToEquity", "<", 1.0)]),
    "Value": ("Cheap on earnings and book", [("trailingPE", "<", 18), ("trailingPE", ">", 0), ("priceToBook", "<", 3)]),
    "High dividend": ("Dividend yield above 2.5% with sustainable payout",
                      [("dividendYield", ">=", 0.025), ("payoutRatio", "<", 0.9)]),
    "Earnings this week": ("Results due within 7 days", [("days_to_earnings", ">=", 0), ("days_to_earnings", "<=", 7)]),
    "Positive news buzz": ("Clearly positive FinBERT sentiment across several headlines",
                           [("sentiment_adj", ">=", 0.2), ("n_news", ">=", 3)]),
    "Strong trend": ("Above all moving averages with a strong ADX trend",
                     [("vs_sma50", ">", 0), ("vs_sma200", ">", 0), ("adx14", ">=", 25), ("supertrend", "=", 1)]),
    "Model favourites": ("Highest next-day probability and top-half monthly rank",
                         [("prob_up", ">=", 0.52), ("monthly_pct", ">=", 0.5)]),
}


def build_table(folder: Path, monthly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join every daily output into one screener table (index = ticker)."""
    tech = pd.read_parquet(folder / "technicals.parquet")
    t = tech.copy()
    day = pd.Timestamp(folder.name)
    f = folder / "fundamentals.parquet"
    if f.exists():
        fund = pd.read_parquet(f)
        ne = pd.to_datetime(fund.get("next_earnings"), errors="coerce")
        fund["days_to_earnings"] = (ne - day).dt.days
        if "marketCap" in fund:
            fund["marketCap"] = pd.to_numeric(fund["marketCap"], errors="coerce") / 1e7      # ₹ crore
        t = t.join(fund[[c for c in FUND_COLUMNS if c in fund]], how="left")
    s = folder / "sentiment.parquet"
    if s.exists():
        t = t.join(pd.read_parquet(s)[["sentiment", "sentiment_adj", "n_news", "buzz"]], how="left")
    n = folder / "nextday.parquet"
    if n.exists():
        t["prob_up"] = pd.read_parquet(n)["prob"].reindex(t.index)
    if monthly is not None and len(monthly):
        d = monthly.index.get_level_values(0).max()
        t["monthly_pct"] = monthly.xs(d, level=0)["score"].rank(pct=True).reindex(t.index)
    return t


def apply(table: pd.DataFrame, filters: list[tuple[str, str, float]]) -> pd.DataFrame:
    mask = pd.Series(True, index=table.index)
    for col, op, val in filters:
        if col not in table:
            continue
        x = pd.to_numeric(table[col], errors="coerce")
        mask &= pd.Series(OPS[op](x.to_numpy(), val), index=table.index) & x.notna()
    return table[mask]
