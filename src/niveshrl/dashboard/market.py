"""MKT: market monitor for the NIFTY 200 universe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ..research.regime import REGIME_COLORS
from . import store, theme

WINDOWS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def status_tape() -> None:
    p = store.panel()
    b = p.bench.dropna()
    v = p.vix.dropna()
    reg = store.regimes_weekly()
    ch = lambda s, n: s.iloc[-1] / s.iloc[-1 - n] - 1  # noqa: E731
    ytd = b.iloc[-1] / b[b.index.year < b.index[-1].year].iloc[-1] - 1
    ma200 = p.close.rolling(200, min_periods=150).mean().iloc[-1]
    breadth = float((p.close.iloc[-1] > ma200).mean())
    r = reg["regime"].iloc[-1] if reg is not None else "n/a"
    cls = lambda x: "pos" if x >= 0 else "neg"  # noqa: E731
    theme.tape([
        ("AS OF", str(b.index[-1].date()), "amb"),
        ("NIFTY", f"{b.iloc[-1]:,.1f}", ""),
        ("1D", f"{ch(b, 1):+.2%}", cls(ch(b, 1))),
        ("1M", f"{ch(b, 21):+.2%}", cls(ch(b, 21))),
        ("YTD", f"{ytd:+.2%}", cls(ytd)),
        ("INDIA VIX", f"{v.iloc[-1]:.2f}", "neg" if v.iloc[-1] > v.rolling(252).median().iloc[-1] else "pos"),
        ("BREADTH >200DMA", f"{breadth:.0%}", cls(breadth - 0.5)),
        ("REGIME", r, "pos" if r == "Bull" else "neg" if r == "Stress" else "amb"),
        ("UNIVERSE", f"{int(p.close.iloc[-1].notna().sum())} stocks", ""),
    ])


def _regime_shapes(fig, weekly: pd.DataFrame | None, start, row=None, col=None):
    if weekly is None:
        return
    w = weekly.loc[weekly.index >= start, "regime"]
    if w.empty:
        return
    seg_start, cur = w.index[0], w.iloc[0]
    for d, lab in list(w.items())[1:] + [(w.index[-1] + pd.Timedelta(days=7), None)]:
        if lab != cur:
            if cur != "Neutral":
                kw = dict(row=row, col=col) if row else {}
                fig.add_vrect(x0=seg_start, x1=d, fillcolor=REGIME_COLORS[cur], opacity=0.10, line_width=0,
                              layer="below", **kw)
            seg_start, cur = d, lab


def index_chart() -> None:
    p = store.panel()
    rng = st.radio("Range", ["6M", "1Y", "3Y", "5Y", "MAX"], index=1, horizontal=True, key="mkt_rng",
                   label_visibility="collapsed")
    days = {"6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "MAX": len(p.bench)}[rng]
    b = p.bench.dropna().iloc[-days:]
    v = p.vix.reindex(b.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=b.index, y=b, name="NIFTY 50", line=dict(color=theme.AMBER, width=1.6)), 1, 1)
    for n, c in [(50, theme.BLUE), (200, "#C77DFF")]:
        fig.add_trace(go.Scatter(x=b.index, y=p.bench.rolling(n).mean().reindex(b.index), name=f"{n}DMA",
                                 line=dict(color=c, width=1, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v, name="India VIX", line=dict(color=theme.RED, width=1.2),
                             fill="tozeroy", fillcolor="rgba(255,77,77,0.08)"), 2, 1)
    _regime_shapes(fig, store.regimes_weekly(), b.index[0], row=1, col=1)
    fig.update_layout(height=430, title="NIFTY 50 · India VIX · regime shading (green bull / red stress)")
    st.plotly_chart(fig, width="stretch")


def heatmap() -> None:
    p = store.panel()
    c1, c2 = st.columns([3, 1])
    win = c2.radio("Window", list(WINDOWS), index=2, key="hm_win")
    n = WINDOWS[win]
    last = p.close.ffill().iloc[-1]
    ret = last / p.close.ffill().iloc[-1 - n] - 1
    df = pd.DataFrame({"ticker": [t.replace(".NS", "") for t in ret.index], "sector": p.sectors.reindex(ret.index).values,
                       "ret": ret.values, "name": p.names.reindex(ret.index).values}).dropna(subset=["ret"])
    df["label"] = df["ticker"] + "<br>" + (df["ret"] * 100).round(1).astype(str) + "%"
    lim = float(np.nanpercentile(np.abs(df["ret"]), 95)) or 0.05
    fig = px.treemap(df, path=[px.Constant("NIFTY 200"), "sector", "label"], values=np.ones(len(df)),
                     color="ret", color_continuous_scale=[[0, "#8B0000"], [0.5, "#1A1F27"], [1, "#0B7A3E"]],
                     range_color=(-lim, lim), hover_data={"name": True, "ret": ":.2%"})
    fig.update_traces(marker=dict(line=dict(color="#07090C", width=1)), textfont=dict(family="monospace"),
                      root_color="#07090C")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0), coloraxis_showscale=False,
                      title=f"NIFTY 200 heatmap · {win} return")
    c1.plotly_chart(fig, width="stretch")
    with c2:
        sec = df.groupby("sector")["ret"].mean().sort_values()
        f2 = go.Figure(go.Bar(x=sec.values * 100, y=[s[:18] for s in sec.index], orientation="h",
                              marker_color=[theme.signed_color(v) for v in sec.values]))
        f2.update_layout(height=430, title=f"Sector avg {win} (%)", yaxis=dict(side="left"), showlegend=False)
        st.plotly_chart(f2, width="stretch")


def movers() -> None:
    p = store.panel()
    ret1 = p.close.ffill().iloc[-1] / p.close.ffill().iloc[-2] - 1
    ret1m = p.close.ffill().iloc[-1] / p.close.ffill().iloc[-22] - 1
    df = pd.DataFrame({"Stock": [t.replace(".NS", "") for t in ret1.index], "Sector": p.sectors.reindex(ret1.index).values,
                       "Last": p.close.ffill().iloc[-1].values, "1D": ret1.values, "1M": ret1m.values}).dropna()
    fmt = {"Last": "{:,.1f}", "1D": "{:+.2%}", "1M": "{:+.2%}"}
    sty = lambda d: d.style.format(fmt).map(lambda v: f"color: {theme.signed_color(v)}", subset=["1D", "1M"])  # noqa: E731
    c1, c2 = st.columns(2)
    c1.markdown("### Top gainers · 1D")
    c1.dataframe(sty(df.nlargest(10, "1D")), hide_index=True, width="stretch")
    c2.markdown("### Top losers · 1D")
    c2.dataframe(sty(df.nsmallest(10, "1D")), hide_index=True, width="stretch")


def breadth_chart() -> None:
    p = store.panel()
    ma = p.close.rolling(200, min_periods=150).mean()
    br = (p.close > ma).where(ma.notna()).mean(axis=1).iloc[-1260:]
    fig = go.Figure(go.Scatter(x=br.index, y=br * 100, line=dict(color=theme.BLUE, width=1.3), fill="tozeroy",
                               fillcolor="rgba(77,163,255,0.08)", name="% above 200DMA"))
    fig.add_hline(y=50, line=dict(color=theme.MUTED, dash="dot", width=1))
    fig.update_layout(height=260, title="Breadth: % of NIFTY 200 above 200-day average (5y)", showlegend=False)
    st.plotly_chart(fig, width="stretch")


def model_picks() -> None:
    sig = store.signals()
    models = [m for m in ("ffnn", "lstm", "transformer") if m in sig]
    if not models:
        theme.note("Train the rankers (<code>python scripts/train_rankers.py</code>) to see model picks here.")
        return
    p = store.panel()
    frames = []
    for m in models:
        pr = store.predictions(m)
        d = pr.index.get_level_values(0).max()
        s = pr.xs(d, level=0)["score"]
        frames.append(s.rank(pct=True).rename(m.upper()))
    tab = pd.concat(frames, axis=1)
    tab["CONSENSUS"] = tab.mean(axis=1)
    tab = tab.sort_values("CONSENSUS", ascending=False)
    tab.insert(0, "Sector", p.sectors.reindex(tab.index).str[:22])
    tab.index = [t.replace(".NS", "") for t in tab.index]
    st.markdown(f"### Model consensus · top 15 as of {d.date()}")
    theme.note("Percentile rank of each model's P(beat the median next month). 100% = most attractive. "
               "Research output, not a recommendation.")
    num = [c for c in tab.columns if c != "Sector"]
    st.dataframe(tab.head(15).style.format({c: "{:.0%}" for c in num}).background_gradient(
        subset=num, cmap="Greens", vmin=0.5, vmax=1.0), width="stretch")


def page() -> None:
    st.markdown("# Market monitor")
    status_tape()
    c1, c2 = st.columns([2, 1])
    with c1:
        index_chart()
    with c2:
        model_picks()
    heatmap()
    breadth_chart()
    movers()
