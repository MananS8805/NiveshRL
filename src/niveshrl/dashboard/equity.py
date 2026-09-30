"""EQ: single-stock drilldown."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ..metrics import drawdown
from . import store, theme
from .market import _regime_shapes


def page() -> None:
    p = store.panel()
    tickers = sorted(p.tickers)
    default = tickers.index("RELIANCE.NS") if "RELIANCE.NS" in tickers else 0
    c0, c1 = st.columns([1, 3])
    tk = c0.selectbox("Security", tickers, index=default,
                      format_func=lambda t: f"{t.replace('.NS', '')} · {str(p.names.get(t, ''))[:28]}")
    rng = c1.radio("Range", ["1Y", "3Y", "5Y", "MAX"], index=1, horizontal=True)
    s = p.close[tk].dropna()
    days = {"1Y": 252, "3Y": 756, "5Y": 1260, "MAX": len(s)}[rng]
    sv = s.iloc[-days:]
    st.markdown(f"# {tk.replace('.NS', '')} · {p.names.get(tk, '')}")
    r = s.pct_change().dropna()
    b = p.bench.reindex(r.index).pct_change()
    last252 = r.iloc[-252:]
    beta = np.cov(last252, b.reindex(last252.index).fillna(0))[0, 1] / b.reindex(last252.index).var()
    ch = lambda n: s.iloc[-1] / s.iloc[-1 - n] - 1 if len(s) > n else np.nan  # noqa: E731
    theme.kpis([
        ("Last", s.iloc[-1], "num", False, str(s.index[-1].date())),
        ("1M", ch(21), "pct", True), ("3M", ch(63), "pct", True), ("1Y", ch(252), "pct", True),
        ("Vol 1Y", last252.std() * np.sqrt(252), "pct_plain", False),
        ("Beta vs NIFTY", beta, "x", False),
        ("Drawdown", drawdown(s).iloc[-1], "pct", True, "from peak"),
        ("Sector", None, "num", False, str(p.sectors.get(tk, ""))[:22]),
    ])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=sv.index, y=sv, name="Close", line=dict(color=theme.AMBER, width=1.5)), 1, 1)
    for n, c in [(50, theme.BLUE), (200, "#C77DFF")]:
        fig.add_trace(go.Scatter(x=sv.index, y=s.rolling(n).mean().reindex(sv.index), name=f"{n}DMA",
                                 line=dict(color=c, width=1, dash="dot")), 1, 1)
    vol = p.volume[tk].reindex(sv.index)
    fig.add_trace(go.Bar(x=vol.index, y=vol, name="Volume", marker_color="#2A3442"), 2, 1)
    _regime_shapes(fig, store.regimes_weekly(), sv.index[0], row=1, col=1)
    fig.update_layout(height=440, title="Price · moving averages · regime shading · volume")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    f2 = go.Figure()
    for i, m in enumerate(["ffnn", "lstm", "transformer"]):
        pr = store.predictions(m)
        if pr is None:
            continue
        sc = pr["score"]
        rk = sc.groupby(level=0).rank(pct=True)
        try:
            srs = rk.xs(tk, level=1)
        except KeyError:
            continue
        srs = srs[srs.index >= sv.index[0]]
        f2.add_trace(go.Scatter(x=srs.index, y=srs * 100, name=m.upper(), mode="lines+markers",
                                line=dict(color=theme.SERIES[i], width=1.2), marker=dict(size=4)))
    f2.add_hline(y=50, line=dict(color=theme.MUTED, dash="dot"))
    f2.update_layout(height=300, title="Model rank percentile each month (100 = top of NIFTY 200)", yaxis=dict(range=[0, 100]))
    c1.plotly_chart(f2, width="stretch")
    vf = store.vol_forecasts()
    f3 = go.Figure()
    if vf is not None:
        try:
            v = vf.xs(tk, level=1)
            v = v[v.index >= sv.index[0]]
            f3.add_trace(go.Scatter(x=v.index, y=v["rv_next"] * 100, name="Realised next month",
                                    line=dict(color=theme.MUTED, width=1)))
            for m, c in [("lstm", theme.AMBER), ("garch", theme.BLUE)]:
                f3.add_trace(go.Scatter(x=v.index, y=v[m] * 100, name=f"{m.upper()} forecast", line=dict(color=c, width=1.3)))
        except KeyError:
            pass
    f3.update_layout(height=300, title="Next-month volatility: forecasts vs realised (%, annualised)")
    c2.plotly_chart(f3, width="stretch")

    sec = p.sectors.get(tk)
    peers = p.sectors[p.sectors == sec].index
    rows = []
    ff = store.predictions("ffnn")
    last_d = ff.index.get_level_values(0).max() if ff is not None else None
    for t in peers:
        px_ = p.close[t].dropna()
        if len(px_) < 253:
            continue
        rows.append({"Stock": t.replace(".NS", ""), "Last": px_.iloc[-1], "1M": px_.iloc[-1] / px_.iloc[-22] - 1,
                     "1Y": px_.iloc[-1] / px_.iloc[-253] - 1,
                     "Vol 1Y": px_.pct_change().iloc[-252:].std() * np.sqrt(252),
                     "FFNN %ile": (ff.xs(last_d, level=0)["score"].rank(pct=True).get(t, np.nan)
                                   if ff is not None else np.nan)})
    st.markdown(f"### Sector peers · {sec}")
    st.dataframe(pd.DataFrame(rows).sort_values("FFNN %ile", ascending=False).style.format(
        {"Last": "{:,.1f}", "1M": "{:+.1%}", "1Y": "{:+.1%}", "Vol 1Y": "{:.1%}", "FFNN %ile": "{:.0%}"}),
        hide_index=True, width="stretch")
