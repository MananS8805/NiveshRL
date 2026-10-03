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

from .. import watchlist as wl
from ..research import daily
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
    _desk_row(ticker, price)

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

    news_tab, tech_tab, *tabs = st.tabs(["News & sentiment", "Technicals", "Income statement", "Balance sheet",
                                         "Cash flow", "Ownership & analysts", "NiveshRL models", "About"])
    with news_tab:
        _news_tab(ticker)
    with tech_tab:
        _tech_tab(ticker)
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


# --------------------------------------------------------------------------- daily-desk additions
def _daily_frame(name: str) -> pd.DataFrame | None:
    d = daily.latest_dir()
    f = d / f"{name}.parquet" if d is not None else None
    return pd.read_parquet(f) if f is not None and f.exists() else None


def _desk_row(ticker: str, price: float) -> None:
    """Watchlist control + analyst score, next results, news sentiment and next-day P(up) from the daily run."""
    fund, sent, nxt = _daily_frame("fundamentals"), _daily_frame("sentiment"), _daily_frame("nextday")
    items = wl.load()
    c = st.columns([2, 1, 1, 1, 1])
    with c[0]:
        cur = items.get(ticker)
        if cur:
            extra = (f" · buy ≤ ₹{cur.target_buy:,.0f}" if cur.target_buy else "") + \
                    (f" · sell ≥ ₹{cur.target_sell:,.0f}" if cur.target_sell else "")
            st.markdown(f"**{wl.TIERS[cur.tier]}** on your watchlist{extra}")
            if st.button("Remove from watchlist", key=f"wl_rm_{ticker}"):
                wl.remove(ticker)
                st.rerun()
        else:
            a, b = st.columns(2)
            if a.button("★ Must have", key=f"wl_must_{ticker}", width="stretch"):
                wl.add(ticker, "must")
                st.rerun()
            if b.button("☆ Preferred", key=f"wl_pref_{ticker}", width="stretch"):
                wl.add(ticker, "preferred")
                st.rerun()
    an = fund.loc[ticker, "analyst_score"] if fund is not None and ticker in fund.index else None
    ne = None
    if fund is not None and ticker in fund.index and "next_earnings" in fund:
        ne = pd.to_datetime(fund.loc[ticker, "next_earnings"], errors="coerce")
    se = sent.loc[ticker] if sent is not None and ticker in sent.index else None
    pu = nxt.loc[ticker, "prob"] if nxt is not None and ticker in nxt.index else None
    c[1].metric("Analyst score", "–" if an is None or an != an else f"{an:.0f}/100")
    c[2].metric("Next results", "–" if ne is None or pd.isna(ne) else ne.strftime("%d %b"))
    c[3].metric("News sentiment", "–" if se is None else f"{se['sentiment']:+.2f}",
                None if se is None else f"{int(se['n_news'])} headlines")
    c[4].metric("Next-day P(up)", "–" if pu is None else f"{pu:.0%}")


def _news_tab(ticker: str) -> None:
    news = _daily_frame("news")
    if news is None or "score" not in news:
        st.info("No scored news yet. Run the daily refresh (sidebar) to fetch headlines and FinBERT sentiment.")
        return
    n = news[news["ticker"] == ticker].sort_values("published", ascending=False)
    if not len(n):
        st.info("No headlines about this stock in the last 48 hours.")
        return
    age_h = (pd.Timestamp.now(tz="Asia/Kolkata") - pd.to_datetime(n["published"])).dt.total_seconds() / 3600
    w = (0.5 ** (age_h / 12)).fillna(0.01)
    avg = float(np.average(n["score"], weights=w))
    g1, g2 = st.columns([1, 3])
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=avg, number=dict(valueformat="+.2f"),
                                   gauge=dict(axis=dict(range=[-1, 1]), bar=dict(color=theme.AMBER),
                                              steps=[dict(range=[-1, -0.2], color="#3A1414"),
                                                     dict(range=[-0.2, 0.2], color="#1A1F27"),
                                                     dict(range=[0.2, 1], color="#0E3320")])))
    gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=0), title="FinBERT sentiment (recency-weighted)")
    g1.plotly_chart(gauge, width="stretch")
    g1.caption(f"{len(n)} headlines · {int((n['score'] > 0.5).sum())} bullish · {int((n['score'] < -0.5).sum())} bearish")
    view = n[["published", "label", "score", "title", "source", "link"]].copy()
    view["published"] = pd.to_datetime(view["published"]).dt.strftime("%d %b %H:%M")
    g2.dataframe(view, hide_index=True, width="stretch", height=360,
                 column_config={"link": st.column_config.LinkColumn("Open", display_text="open"),
                                "score": st.column_config.ProgressColumn("Score", min_value=-1, max_value=1,
                                                                         format="%+.2f"),
                                "title": st.column_config.TextColumn("Headline", width="large")})
    theme.note("Scored by FinBERT (ProsusAI/finbert), a BERT model trained on financial text. "
               "Headlines from Google News, last 48 h.")


_SIGNALS = {
    "rsi14": lambda v: "overbought" if v > 70 else "oversold" if v < 30 else "neutral",
    "adx14": lambda v: "strong trend" if v >= 25 else "weak / no trend",
    "supertrend": lambda v: "uptrend" if v > 0 else "downtrend",
    "vs_sma200": lambda v: "above long-term average" if v > 0 else "below long-term average",
    "vol_ratio": lambda v: "unusual volume" if v >= 1.5 else "",
    "bb_pctb": lambda v: "at upper band" if v > 1 else "at lower band" if v < 0 else "",
    "macd_hist": lambda v: "bullish momentum" if v > 0 else "bearish momentum",
    "from_52w_high": lambda v: "near 52-week high" if v > -0.03 else "",
}


def _tech_tab(ticker: str) -> None:
    from ..research.technicals import TECH_COLUMNS
    tech = _daily_frame("technicals")
    if tech is None or ticker not in tech.index:
        st.info("Run the daily refresh to compute technical indicators.")
        return
    r = tech.loc[ticker]
    rows = []
    for k, (label, group, kind) in TECH_COLUMNS.items():
        v = r.get(k)
        missing = v is None or v != v
        if missing:
            val = "–"
        elif kind == "pct":
            val = f"{v:+.2%}"
        elif kind == "flag":
            val = "yes" if v == 1 else "no"
        else:
            val = f"{v:,.2f}"
        sig = "" if missing or k not in _SIGNALS else _SIGNALS[k](v)
        rows.append({"Group": group, "Indicator": label, "Value": val, "Signal": sig})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420)
