"""Stock details panel: basics, key statistics and financial statements for one stock.

Shown inline in the market monitor when a stock is clicked in the heatmap.
Fundamentals come from Yahoo Finance (cached for an hour). The price is live
when the stream has a tick for the stock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..research import fundamentals as fx
from . import store, theme


@st.cache_data(ttl=3600, show_spinner="Fetching company financials…")
def _fundamentals(ticker: str) -> dict:
    return fx.fetch(ticker)


def _pct(v, already_pct: bool = False) -> float | None:
    if v is None or v != v:
        return None
    return v / 100 if already_pct else v


def _money_table(df: pd.DataFrame) -> pd.DataFrame:
    """Statement rows in Rs crore (EPS stays in rupees)."""
    if df.empty:
        return df
    out = df.copy().astype(float)
    for r in out.index:
        if r != "Diluted EPS":
            out.loc[r] = out.loc[r] / 1e7
    out.index = [f"{r} (₹ Cr)" if r != "Diluted EPS" else "Diluted EPS (₹)" for r in out.index]
    return out


def render(ticker: str, live_tick: dict | None) -> None:
    p = store.panel()
    f = _fundamentals(ticker)
    s, prof = f["stats"], f["profile"]
    short = ticker.replace(".NS", "")
    hist = p.close[ticker].dropna()

    # ---- header: live price if streaming, else Yahoo's current/last price
    if live_tick:
        price, chg, chg_pct, src = live_tick["price"], live_tick.get("change"), live_tick.get("change_percent"), "LIVE"
    else:
        price = s.get("currentPrice") or float(hist.iloc[-1])
        prev = s.get("previousClose")
        chg = price - prev if prev else None
        chg_pct = chg / prev * 100 if prev else None
        src = "LAST"
    c1, c2 = st.columns([3, 2])
    c1.markdown(f"## {short} · {prof.get('longName') or p.names.get(ticker, '')}")
    c1.caption(f"{prof.get('sector') or ''} · {prof.get('industry') or ''} · NSE industry: {p.sectors.get(ticker, '')}")
    color = theme.GREEN if (chg or 0) >= 0 else theme.RED
    c2.markdown(
        f"<div style='text-align:right;font-family:monospace'><span style='font-size:1.9rem;color:#F2F2F2'>"
        f"₹{price:,.2f}</span> <span style='color:{color};font-size:1.05rem'>"
        f"{'' if chg is None else f'{chg:+,.2f}'} ({'' if chg_pct is None else f'{chg_pct:+.2f}%'})</span>"
        f"<br><span style='color:{theme.AMBER if src == 'LIVE' else theme.MUTED};font-size:.75rem'>{src}"
        f"{' · Yahoo stream, may be delayed' if src == 'LIVE' else ' · Yahoo quote'}</span></div>",
        unsafe_allow_html=True)

    lo, hi = s.get("fiftyTwoWeekLow"), s.get("fiftyTwoWeekHigh")
    pos52 = (price - lo) / (hi - lo) if lo and hi and hi > lo else None
    theme.kpis([
        ("Market cap", fx.crore(s.get("marketCap")), "text", False, f"EV {fx.crore(s.get('enterpriseValue'))}"),
        ("P/E (ttm)", s.get("trailingPE"), "x", False, f"fwd {s['forwardPE']:.1f}" if s.get("forwardPE") else ""),
        ("P/B", s.get("priceToBook"), "x", False, f"book ₹{s['bookValue']:,.0f}" if s.get("bookValue") else ""),
        ("EPS (ttm)", s.get("trailingEps"), "num", False, "₹ per share"),
        ("Dividend yield", _pct(s.get("dividendYield"), already_pct=True), "pct_plain", False),
        ("ROE", _pct(s.get("returnOnEquity")), "pct_plain", False),
        ("Debt / equity", _pct(s.get("debtToEquity"), already_pct=True), "x", False),
        ("Operating margin", _pct(s.get("operatingMargins")), "pct_plain", False),
        ("Net margin", _pct(s.get("profitMargins")), "pct_plain", False),
        ("Revenue growth", _pct(s.get("revenueGrowth")), "pct", True, "yoy, latest qtr"),
        ("Earnings growth", _pct(s.get("earningsGrowth")), "pct", True, "yoy, latest qtr"),
        ("52w position", pos52, "pct_plain", False, f"₹{lo:,.0f} – ₹{hi:,.0f}" if lo and hi else ""),
        ("Day range", f"₹{s['dayLow']:,.0f}–{s['dayHigh']:,.0f}" if s.get("dayLow") and s.get("dayHigh") else None,
         "text", False, "Yahoo quote"),
        ("Volume", (live_tick or {}).get("day_volume") and float(live_tick["day_volume"]) or s.get("volume"), "int", False,
         f"avg {s['averageVolume']:,.0f}" if s.get("averageVolume") else ""),
        ("Beta", s.get("beta"), "x", False, "Yahoo"),
    ])

    g1, g2 = st.columns(2)
    h1 = hist.iloc[-252:]
    fig = go.Figure(go.Scatter(x=h1.index, y=h1, line=dict(color=theme.AMBER, width=1.5), name="close"))
    if live_tick:
        fig.add_trace(go.Scatter(x=[pd.Timestamp.now().normalize()], y=[price], mode="markers", name="live",
                                 marker=dict(color=color, size=9, symbol="diamond")))
    for v, nm in [(hi, "52w high"), (lo, "52w low")]:
        if v:
            fig.add_hline(y=v, line=dict(color=theme.MUTED, dash="dot", width=1),
                          annotation_text=nm, annotation_font_color=theme.MUTED)
    fig.update_layout(height=300, title="1-year price · 52-week range", showlegend=False)
    g1.plotly_chart(fig, width="stretch")
    q = f["quarterly"]
    if not q.empty and "Total Revenue" in q.index:
        qq = q.T.iloc[::-1] / 1e7
        f2 = go.Figure([go.Bar(x=qq.index, y=qq["Total Revenue"], name="Revenue", marker_color=theme.BLUE)])
        if "Net Income" in qq:
            f2.add_trace(go.Bar(x=qq.index, y=qq["Net Income"], name="Net profit", marker_color=theme.GREEN))
        f2.update_layout(height=300, barmode="group", title="Quarterly revenue and net profit (₹ Cr)")
        g2.plotly_chart(f2, width="stretch")
    else:
        g2.info("Yahoo has no quarterly statements for this stock.")

    tabs = st.tabs(["Income statement", "Balance sheet", "Cash flow", "Ownership & analysts", "NiveshRL models", "About"])
    for tab, key in zip(tabs[:3], ["annual", "balance", "cashflow"]):
        with tab:
            t = _money_table(f[key])
            if t.empty:
                st.info("Not available from Yahoo for this stock.")
            else:
                st.dataframe(t.style.format("{:,.1f}"), width="stretch")
                st.caption("Annual figures as reported to Yahoo Finance; ₹ crore unless stated.")
    with tabs[3]:
        cols = st.columns(2)
        own = pd.DataFrame({"Holder": ["Promoters / insiders", "Institutions"],
                            "Share": [s.get("heldPercentInsiders"), s.get("heldPercentInstitutions")]})
        cols[0].dataframe(own.style.format({"Share": lambda v: "–" if v is None else f"{v:.1%}"}), hide_index=True,
                          width="stretch")
        n = s.get("numberOfAnalystOpinions")
        if n and s.get("targetMeanPrice"):
            up = s["targetMeanPrice"] / price - 1
            cols[1].markdown(
                f"**Analyst consensus ({n}):** {str(s.get('recommendationKey') or '').replace('_', ' ')}  \n"
                f"Mean target ₹{s['targetMeanPrice']:,.0f} ({up:+.1%} vs price) · range "
                f"₹{(s.get('targetLowPrice') or 0):,.0f} – ₹{(s.get('targetHighPrice') or 0):,.0f}")
            theme.note("Third-party analyst views as published on Yahoo Finance. Shown for context, not endorsed.")
        else:
            cols[1].info("No analyst coverage on Yahoo.")
    with tabs[4]:
        rows = []
        for m in ("ffnn", "lstm", "transformer"):
            pr = store.predictions(m)
            if pr is None:
                continue
            d = pr.index.get_level_values(0).max()
            sc = pr.xs(d, level=0)["score"]
            if ticker in sc.index:
                rows.append({"Model": m.upper(), "P(beat median next month)": sc[ticker],
                             "Rank percentile": sc.rank(pct=True)[ticker], "As of": d.date()})
        vf = store.vol_forecasts()
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"P(beat median next month)": "{:.1%}",
                                                          "Rank percentile": "{:.0%}"}), hide_index=True, width="stretch")
        if vf is not None and ticker in vf.index.get_level_values(1):
            v = vf.xs(ticker, level=1).iloc[-1]
            st.caption(f"Next-month volatility forecast (LSTM): {v['lstm']:.1%} · GARCH: {v['garch']:.1%}")
        theme.note("Walk-forward research output. Not a recommendation. See the RANK and EQ screens for history.")
    with tabs[5]:
        st.markdown(prof.get("longBusinessSummary") or "No description on Yahoo.")
        meta = [f"Employees: {prof['fullTimeEmployees']:,}" if prof.get("fullTimeEmployees") else "",
                f"[{prof['website']}]({prof['website']})" if prof.get("website") else ""]
        st.caption(" · ".join(m for m in meta if m))
