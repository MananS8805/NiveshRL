"""MKT: market monitor for the NIFTY 200 universe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ..research.regime import REGIME_COLORS
from . import live, stockcard, store, theme

WINDOWS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def live_table(f: live.LiveFeed) -> pd.DataFrame:
    """Per stock: last price (live tick if streaming, else last close), 1D change and whether it is live.

    1D uses the stream's own change vs the previous close. Longer windows compare
    the live price with the stored daily closes.
    """
    p = store.panel()
    close = p.close.ffill()
    snap = f.snapshot()
    last = close.iloc[-1].copy()
    chg1 = close.iloc[-1] / close.iloc[-2] - 1
    q = live.quote_snapshot(tuple(p.tickers))
    if q is not None:                      # today's quote + previous close for every stock
        common = last.index.intersection(q.index)
        last[common] = q.loc[common, "last"]
        chg1[common] = q.loc[common, "last"] / q.loc[common, "prev_close"] - 1
    is_live = pd.Series(False, index=last.index)
    for t in last.index:
        tk = snap.get(t)
        if tk:
            last[t] = tk["price"]
            if tk.get("change_percent") is not None:
                chg1[t] = tk["change_percent"] / 100
            is_live[t] = True
    return pd.DataFrame({"last": last, "chg1": chg1, "live": is_live})


def status_tape(f: live.LiveFeed) -> None:
    p = store.panel()
    b = p.bench.dropna()
    v = p.vix.dropna()
    reg = store.regimes_weekly()
    snap = f.snapshot()
    nifty, vix = snap.get("^NSEI"), snap.get("^INDIAVIX")
    b_last = nifty["price"] if nifty else b.iloc[-1]
    if nifty and nifty.get("change_percent") is not None:
        d1 = nifty["change_percent"] / 100
    else:
        d1 = b.iloc[-1] / b.iloc[-2] - 1
    v_last = vix["price"] if vix else v.iloc[-1]
    m1 = b_last / b.iloc[-22] - 1
    ytd = b_last / b[b.index.year < pd.Timestamp.now().year].iloc[-1] - 1
    ma200 = p.close.rolling(200, min_periods=150).mean().iloc[-1]
    lt = live_table(f)
    breadth = float((lt["last"] > ma200).where(ma200.notna()).mean())
    r = reg["regime"].iloc[-1] if reg is not None else "n/a"
    cls = lambda x: "pos" if x >= 0 else "neg"  # noqa: E731
    stat, stat_cls = live.status_text(f)
    theme.tape([
        ("FEED", stat, stat_cls),
        ("NIFTY", f"{b_last:,.1f}", ""),
        ("1D", f"{d1:+.2%}", cls(d1)),
        ("1M", f"{m1:+.2%}", cls(m1)),
        ("YTD", f"{ytd:+.2%}", cls(ytd)),
        ("INDIA VIX", f"{v_last:.2f}", "neg" if v_last > v.rolling(252).median().iloc[-1] else "pos"),
        ("BREADTH >200DMA", f"{breadth:.0%}", cls(breadth - 0.5)),
        ("ADV/DEC", f"{int((lt['chg1'] > 0).sum())}/{int((lt['chg1'] < 0).sum())}", ""),
        ("REGIME", r, "pos" if r == "Bull" else "neg" if r == "Stress" else "amb"),
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


def _selected_points(ev) -> list:
    if ev is None:
        return []
    sel = ev.get("selection") if isinstance(ev, dict) else getattr(ev, "selection", None)
    if sel is None:
        return []
    pts = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    return list(pts or [])


def _on_heatmap_select() -> None:
    """Remember the clicked stock (leaves only; a sector click just zooms)."""
    for pt in _selected_points(st.session_state.get("hm_chart")):
        for key in ("id", "customdata", "label"):
            tid = pt.get(key) if isinstance(pt, dict) else None
            if isinstance(tid, (list, tuple)):
                tid = tid[0] if tid else None
            if isinstance(tid, str) and tid.endswith(".NS"):
                st.session_state["mkt_sel"] = tid
                return


def heatmap(f: live.LiveFeed) -> None:
    p = store.panel()
    win = st.session_state.get("hm_win", "1D")
    n = WINDOWS[win]
    lt = live_table(f)
    close = p.close.ffill()
    ret = lt["chg1"] if n == 1 else lt["last"] / close.iloc[-1 - n] - 1
    df = pd.DataFrame({"ticker": ret.index, "ret": ret.values, "sector": p.sectors.reindex(ret.index).values,
                       "name": p.names.reindex(ret.index).values, "last": lt["last"].values}).dropna(subset=["ret"])
    secs = sorted(df["sector"].unique())
    sec_ret = df.groupby("sector")["ret"].mean()
    # Stable ids (the ticker for a stock, "sec:<name>" for a sector) keep a zoomed tile
    # zoomed across live refreshes; the changing return lives in the text only.
    ids = ["NIFTY 200"] + [f"sec:{x}" for x in secs] + list(df["ticker"])
    labels = ["NIFTY 200"] + secs + [t.replace(".NS", "") for t in df["ticker"]]
    parents = [""] + ["NIFTY 200"] * len(secs) + [f"sec:{x}" for x in df["sector"]]
    values = [0] + [0] * len(secs) + [1] * len(df)
    colors = [0.0] + [float(sec_ret[x]) for x in secs] + list(df["ret"])
    text = [""] + [f"{sec_ret[x]:+.2%}" for x in secs] + [f"{r:+.2%}<br>₹{l:,.1f}" for r, l in zip(df["ret"], df["last"])]
    hover = ["NIFTY 200"] + secs + [f"{nm}<br>{r:+.2%} · ₹{l:,.2f}" for nm, r, l in zip(df["name"], df["ret"], df["last"])]
    lim = float(np.nanpercentile(np.abs(df["ret"]), 95)) or 0.05
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="remainder",
        marker=dict(colors=colors, colorscale=[[0, "#8B0000"], [0.5, "#1A1F27"], [1, "#0B7A3E"]], cmin=-lim, cmax=lim,
                    line=dict(color="#07090C", width=1)),
        text=text, texttemplate="<b>%{label}</b><br>%{text}", hovertext=hover, hoverinfo="text",
        customdata=ids, root_color="#07090C", textfont=dict(family="monospace"),
        maxdepth=3,
    ))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=34, b=0), uirevision="heatmap",
                      title=f"NIFTY 200 heatmap · {win} · click a stock for its details")
    st.plotly_chart(fig, width="stretch", key="hm_chart", on_select=_on_heatmap_select, selection_mode="points")


def movers(f: live.LiveFeed) -> None:
    p = store.panel()
    lt = live_table(f)
    close = p.close.ffill()
    df = pd.DataFrame({"Stock": [t.replace(".NS", "") for t in lt.index], "Sector": p.sectors.reindex(lt.index).values,
                       "Last": lt["last"].values, "1D": lt["chg1"].values,
                       "1M": (lt["last"] / close.iloc[-22] - 1).values,
                       "Live": np.where(lt["live"], "●", "")}).dropna(subset=["1D"])
    fmt = {"Last": "{:,.2f}", "1D": "{:+.2%}", "1M": "{:+.2%}"}
    sty = lambda d: d.style.format(fmt).map(lambda v: f"color: {theme.signed_color(v)}", subset=["1D", "1M"])  # noqa: E731
    c1, c2 = st.columns(2)
    c1.markdown("### Top gainers · 1D")
    c1.dataframe(sty(df.nlargest(10, "1D")), hide_index=True, width="stretch")
    c2.markdown("### Top losers · 1D")
    c2.dataframe(sty(df.nsmallest(10, "1D")), hide_index=True, width="stretch")
    theme.note("● = live tick from the Yahoo stream this session; others show the last close.")


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


def _stock_picker() -> None:
    p = store.panel()
    opts = sorted(p.tickers)
    cur = st.session_state.get("mkt_sel")
    pick = st.selectbox("Open stock details", [None] + opts, index=(opts.index(cur) + 1) if cur in opts else 0,
                        format_func=lambda t: "Search a stock…" if t is None else f"{t.replace('.NS', '')} · {str(p.names.get(t, ''))[:30]}")
    if pick and pick != cur:
        st.session_state["mkt_sel"] = pick
        st.rerun(scope="fragment")


def page() -> None:
    st.markdown("# Market monitor")
    p = store.panel()
    f = live.feed(tuple(p.tickers))
    c0, c1 = st.columns([1, 3])
    auto = c0.toggle("Live updates", value=True, help="Refresh prices from the Yahoo stream every few seconds.")
    every = c1.select_slider("Refresh every", [3, 5, 10, 30, 60], value=5, format_func=lambda x: f"{x}s",
                             disabled=not auto, label_visibility="collapsed")

    @st.fragment(run_every=every if auto else None)
    def live_block() -> None:
        status_tape(f)
        sel = st.session_state.get("mkt_sel")
        if sel:
            # The clicked stock opens in the heatmap's own place, not in a pop-up.
            b1, b2 = st.columns([1, 5])
            if b1.button("← Back to heatmap", width="stretch"):
                st.session_state["mkt_sel"] = None
                st.rerun(scope="fragment")
            b2.caption("Details for the stock you clicked. Prices keep updating live.")
            with st.container(border=True):
                stockcard.render(sel, f.snapshot().get(sel))
        else:
            a1, a2 = st.columns([3, 1])
            with a2:
                st.radio("Window", list(WINDOWS), index=0, key="hm_win")
                _stock_picker()
            with a1:
                heatmap(f)
        movers(f)

    live_block()
    c1, c2 = st.columns([2, 1])
    with c1:
        index_chart()
    with c2:
        model_picks()
    breadth_chart()
