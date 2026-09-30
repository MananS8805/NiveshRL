"""LAB: build a strategy, backtest it with Indian costs, read the tearsheet, compare."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..metrics import drawdown
from ..research import backtest as bt
from ..research.regime import REGIME_COLORS, REGIMES
from . import store, theme

MAX_COMPARE = 6


def _builder() -> tuple[bt.StrategySpec, bool, bool]:
    sig = store.signals()
    p = store.panel()
    years = list(range(2012, p.close.index[-1].year + 1))
    with st.form("lab"):
        c1, c2, c3, c4 = st.columns(4)
        key = c1.selectbox("Signal", list(sig), format_func=lambda k: sig[k],
                           index=list(sig).index("ffnn") if "ffnn" in sig else 0)
        size_mode = c2.radio("Portfolio size", ["Top N", "Top decile"], horizontal=True)
        n = c2.slider("N stocks", 5, 50, 20, disabled=False)
        weighting = c3.selectbox("Weighting", ["equal", "inv_vol", "score"],
                                 format_func={"equal": "Equal", "inv_vol": "Inverse volatility",
                                              "score": "Score-weighted"}.get)
        reb = c3.selectbox("Rebalance every", [1, 2, 3, 6, 12], format_func=lambda m: f"{m} month(s)")
        maxw = c4.slider("Max weight per stock", 0.02, 0.25, 0.10, 0.01, format="%.2f")
        ls = c4.toggle("Long-short (research only)", value=False,
                       help="Also shorts the bottom bucket. Retail investors cannot short NSE delivery stocks overnight.")
        c5, c6, c7, c8 = st.columns(4)
        y0, y1 = c5.select_slider("Period", options=years, value=(years[0], years[-1]))
        cost = c6.slider("Cost multiplier", 0.0, 4.0, 1.0, 0.25, help="1.0 = real NSE delivery charges")
        capital = c6.number_input("Capital (₹)", 50_000, 100_000_000, 1_000_000, 50_000)
        vt = c7.selectbox("Volatility target", [None, 0.10, 0.12, 0.15, 0.20],
                          format_func=lambda v: "Off" if v is None else f"{v:.0%} annual")
        use_reg = c8.toggle("Regime filter", value=False,
                            help="Scale exposure by the walk-forward regime detector's label at each rebalance.")
        rx = {}
        if use_reg:
            cols = c8.columns(3)
            for i, r in enumerate(REGIMES):
                rx[r] = cols[i].number_input(r, 0.0, 1.0, {"Bull": 1.0, "Neutral": 0.8, "Stress": 0.3}[r], 0.1,
                                             key=f"rx_{r}")
        name = st.text_input("Strategy name", value="")
        b1, b2, b3 = st.columns([1, 1, 4])
        run = b1.form_submit_button("▶ RUN", type="primary")
        add = b2.form_submit_button("＋ ADD TO COMPARE")
    top = 0.1 if size_mode == "Top decile" else float(n)
    label = name or f"{sig[key].split(' (')[0]} · {'D10' if size_mode == 'Top decile' else f'top {n}'} · {weighting}" \
                    f"{' · LS' if ls else ''}{f' · VT{vt:.0%}' if vt else ''}{' · RF' if use_reg else ''}"
    spec = bt.StrategySpec(name=label, signal=key, top=top, weighting=weighting, long_short=ls,
                           rebalance_months=int(reb), max_weight=float(maxw), cost_scale=float(cost),
                           capital=float(capital), vol_target=vt, regime_exposure=rx or None,
                           start=f"{y0}-01-01", end=f"{y1}-12-31")
    return spec, run, add


def _bench(nav: pd.Series) -> pd.Series:
    b = store.panel().bench.reindex(nav.index).ffill()
    return b / b.iloc[0]


def _ew(spec: bt.StrategySpec) -> pd.Series:
    ew = bt.StrategySpec(name="Equal weight (universe)", signal="equal", top=1.0, max_weight=0.05,
                         cost_scale=spec.cost_scale, capital=spec.capital, start=spec.start, end=spec.end)
    return store.backtest(json.dumps(ew.to_dict()))["nav"]


def _kpis(m: dict, nav: pd.Series) -> None:
    b = _bench(nav)
    bcagr = (b.iloc[-1]) ** (365.25 / max((b.index[-1] - b.index[0]).days, 1)) - 1
    theme.kpis([
        ("CAGR", m["CAGR"], "pct", True, f"NIFTY {bcagr:+.1%}"),
        ("Sharpe", m["Sharpe"], "x", False, "rf 6.5%"),
        ("Sortino", m["Sortino"], "x", False),
        ("Volatility", m["Vol"], "pct_plain", False, "annualised"),
        ("Max drawdown", m["MaxDD"], "pct", True),
        ("Calmar", m["Calmar"], "x", False),
        ("Alpha vs NIFTY", m.get("Alpha"), "pct", True, f"beta {m.get('Beta', float('nan')):.2f}"),
        ("Info ratio", m.get("InfoRatio"), "x", False),
        ("Turnover / yr", m.get("Turnover/yr"), "x", False, "one-way"),
        ("Cost drag / yr", m.get("Cost drag %/yr"), "pct_plain", False, "STT+stamp+GST+DP+slip"),
        ("Hit rate (wk)", m.get("HitRate"), "pct_plain", False),
        ("Avg holdings", m.get("Avg holdings"), "num", False),
    ])


def _perf_tab(res: dict, spec: bt.StrategySpec) -> None:
    nav = res["nav"]
    ew = _ew(spec).reindex(nav.index).ffill()
    b = _bench(nav)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name=spec.name[:40], line=dict(color=theme.AMBER, width=2)))
    fig.add_trace(go.Scatter(x=ew.index, y=ew / ew.iloc[0], name="Equal weight (same universe)",
                             line=dict(color=theme.BLUE, width=1.2)))
    fig.add_trace(go.Scatter(x=b.index, y=b, name="NIFTY 50", line=dict(color=theme.MUTED, width=1.2, dash="dot")))
    fig.update_layout(height=380, yaxis_type="log", title="Growth of ₹1 after all costs (log scale)")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    dd = drawdown(nav)
    f2 = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=theme.RED, width=1),
                              fillcolor="rgba(255,77,77,0.15)", name="drawdown"))
    f2.add_trace(go.Scatter(x=b.index, y=drawdown(b) * 100, name="NIFTY", line=dict(color=theme.MUTED, width=1, dash="dot")))
    f2.update_layout(height=260, title="Drawdown (%)")
    c1.plotly_chart(f2, width="stretch")
    rs = bt.rolling_sharpe(nav)
    f3 = go.Figure(go.Scatter(x=rs.index, y=rs, line=dict(color=theme.GREEN, width=1.2), name="rolling 1y Sharpe"))
    f3.add_trace(go.Scatter(x=b.index, y=bt.rolling_sharpe(b), line=dict(color=theme.MUTED, width=1, dash="dot"),
                            name="NIFTY"))
    f3.add_hline(y=0, line=dict(color=theme.GRID))
    f3.update_layout(height=260, title="Rolling 1-year Sharpe")
    c2.plotly_chart(f3, width="stretch")


def _returns_tab(res: dict) -> None:
    nav = res["nav"]
    tab = bt.monthly_table(nav)
    z = tab.to_numpy() * 100
    fig = go.Figure(go.Heatmap(z=z, x=list(tab.columns), y=[str(y) for y in tab.index], zmid=0,
                               colorscale=[[0, "#8B0000"], [0.5, "#1A1F27"], [1, "#0B7A3E"]],
                               text=np.where(np.isnan(z), "", np.round(z, 1).astype(str)), texttemplate="%{text}",
                               hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>", showscale=False))
    fig.update_layout(height=34 * len(tab) + 80, title="Monthly returns (%) · last column = calendar year",
                      yaxis=dict(autorange="reversed", side="left"), hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    b = _bench(nav)
    yr = pd.DataFrame({"Strategy": nav.groupby(nav.index.year).last(), "NIFTY": b.groupby(b.index.year).last()})
    yr = yr / yr.shift(1).fillna(1.0) - 1
    f2 = go.Figure([go.Bar(x=yr.index, y=yr["Strategy"] * 100, name="Strategy", marker_color=theme.AMBER),
                    go.Bar(x=yr.index, y=yr["NIFTY"] * 100, name="NIFTY", marker_color=theme.MUTED)])
    f2.update_layout(height=280, barmode="group", title="Calendar-year returns (%)")
    st.plotly_chart(f2, width="stretch")


def _risk_tab(res: dict) -> None:
    nav = res["nav"]
    r = nav.pct_change().dropna()
    c1, c2 = st.columns(2)
    f1 = go.Figure(go.Histogram(x=r * 100, nbinsx=120, marker_color=theme.AMBER, opacity=0.85))
    var95 = np.percentile(r, 5)
    cvar = r[r <= var95].mean()
    f1.add_vline(x=var95 * 100, line=dict(color=theme.RED, dash="dot"))
    f1.update_layout(height=280, title=f"Daily returns · VaR95 {var95:.2%} · CVaR95 {cvar:.2%}", showlegend=False)
    c1.plotly_chart(f1, width="stretch")
    rv = r.rolling(63).std() * np.sqrt(252)
    f2 = go.Figure(go.Scatter(x=rv.index, y=rv * 100, line=dict(color=theme.BLUE, width=1.2)))
    f2.update_layout(height=280, title="Rolling 3-month volatility (%)", showlegend=False)
    c2.plotly_chart(f2, width="stretch")
    reg = store.regimes_daily()
    if reg is not None:
        d = pd.DataFrame({"r": r, "b": _bench(nav).pct_change(), "regime": reg.reindex(r.index)}).dropna()
        g = d.groupby("regime")
        tab = pd.DataFrame({"Days": g.size(), "Strategy ann. return": g["r"].mean() * 252,
                            "NIFTY ann. return": g["b"].mean() * 252,
                            "Strategy ann. vol": g["r"].std() * np.sqrt(252)}).reindex(REGIMES)
        st.markdown("### Performance by market regime (walk-forward labels)")
        st.dataframe(tab.style.format({"Days": "{:.0f}", "Strategy ann. return": "{:+.1%}",
                                       "NIFTY ann. return": "{:+.1%}", "Strategy ann. vol": "{:.1%}"}),
                     width="stretch")


def _trading_tab(res: dict) -> None:
    reb = res["rebalances"]
    if reb is None or not len(reb):
        st.info("No rebalances in range.")
        return
    c1, c2 = st.columns(2)
    f1 = go.Figure([go.Bar(x=reb.index, y=reb["turnover"] * 100, name="turnover %", marker_color=theme.AMBER)])
    f1.update_layout(height=260, title="One-way turnover per rebalance (%)")
    c1.plotly_chart(f1, width="stretch")
    f2 = go.Figure(go.Bar(x=reb.index, y=reb["cost"] / reb["value"] * 1e4, marker_color=theme.RED))
    f2.update_layout(height=260, title="Trading cost per rebalance (bps of portfolio)")
    c2.plotly_chart(f2, width="stretch")
    f3 = go.Figure([go.Scatter(x=reb.index, y=reb["invested"] * 100, name="% invested",
                               line=dict(color=theme.GREEN, shape="hv")),
                    go.Scatter(x=reb.index, y=reb["holdings"], name="holdings", yaxis="y2",
                               line=dict(color=theme.BLUE, shape="hv"))])
    f3.update_layout(height=260, title="Exposure (%) and number of holdings",
                     yaxis2=dict(overlaying="y", side="left", showgrid=False))
    st.plotly_chart(f3, width="stretch")


def _holdings_tab(res: dict) -> None:
    p = store.panel()
    w = res["last_weights"]
    if w is None or w.empty:
        st.info("No holdings.")
        return
    w = w.sort_values(ascending=False)
    tab = pd.DataFrame({"Stock": [t.replace(".NS", "") for t in w.index],
                        "Company": p.names.reindex(w.index).str[:32].values,
                        "Sector": p.sectors.reindex(w.index).values, "Weight": w.values})
    c1, c2 = st.columns([3, 2])
    c1.markdown(f"### Portfolio at last rebalance · {res['last_date'].date()}")
    c1.dataframe(tab.style.format({"Weight": "{:+.2%}"}).bar(subset=["Weight"], color=theme.AMBER),
                 hide_index=True, width="stretch", height=420)
    # Sector exposure over time.
    rows = {d: ws.groupby(p.sectors.reindex(ws.index).values).sum() for d, ws in res["weights"].items()}
    sec = pd.DataFrame(rows).T.fillna(0.0)
    top = sec.abs().mean().sort_values(ascending=False).index[:10]
    sec = sec[top]
    fig = go.Figure([go.Scatter(x=sec.index, y=sec[c] * 100, name=c[:20], stackgroup="one", line=dict(width=0.5))
                     for c in sec.columns])
    fig.update_layout(height=420, title="Sector exposure over time (top 10, %)", legend=dict(font=dict(size=8)))
    c2.plotly_chart(fig, width="stretch")


def _signal_tab(spec: bt.StrategySpec) -> None:
    dg = store.diagnostics(spec.signal, spec.start, spec.end) if spec.signal in store.signals() else None
    if dg is None:
        theme.note("Signal diagnostics are available for trained models (FFNN / LSTM / Transformer / logistic).")
        return
    dec, ic = dg["dec"], dg["ic"]
    c1, c2 = st.columns(2)
    means = dec.mean() * 100
    f1 = go.Figure(go.Bar(x=[f"D{i}" for i in means.index], y=means.values,
                          marker_color=[theme.signed_color(v - means.mean()) for v in means.values]))
    f1.update_layout(height=280, title="Avg next-month return by score decile (%, before costs)")
    c1.plotly_chart(f1, width="stretch")
    f2 = go.Figure([go.Bar(x=ic.index, y=ic, marker_color=[theme.signed_color(v) for v in ic], name="IC"),
                    go.Scatter(x=ic.index, y=ic.rolling(12).mean(), line=dict(color=theme.AMBER), name="12m avg")])
    f2.update_layout(height=280, title=f"Monthly rank IC · mean {ic.mean():.3f} · t {ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}")
    c2.plotly_chart(f2, width="stretch")
    spread = (dec.iloc[:, -1] - dec.iloc[:, 0]).fillna(0)
    f3 = go.Figure(go.Scatter(x=spread.index, y=(1 + spread).cumprod(), line=dict(color=theme.GREEN)))
    f3.update_layout(height=260, title="Cumulative top-minus-bottom decile (before costs, not investable as-is)")
    st.plotly_chart(f3, width="stretch")


def _compare() -> None:
    specs = st.session_state.get("compare", [])
    if not specs:
        theme.note("Use <b>＋ ADD TO COMPARE</b> to stack up to 6 strategies here.")
        return
    fig = go.Figure()
    rows = {}
    for i, sj in enumerate(specs):
        res = store.backtest(sj)
        nm = json.loads(sj)["name"]
        fig.add_trace(go.Scatter(x=res["nav"].index, y=res["nav"], name=nm[:40],
                                 line=dict(color=theme.SERIES[i % len(theme.SERIES)], width=1.6)))
        rows[nm] = res["metrics"]
    fig.update_layout(height=420, yaxis_type="log", title="Strategy comparison · growth of ₹1 after costs")
    st.plotly_chart(fig, width="stretch")
    cols = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "Alpha", "Beta", "Turnover/yr", "Cost drag %/yr"]
    tab = pd.DataFrame(rows).T[cols]
    st.dataframe(tab.style.format({"CAGR": "{:+.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                                   "MaxDD": "{:.1%}", "Calmar": "{:.2f}", "Alpha": "{:+.1%}", "Beta": "{:.2f}",
                                   "Turnover/yr": "{:.1f}", "Cost drag %/yr": "{:.2%}"})
                 .highlight_max(subset=["CAGR", "Sharpe", "Sortino", "Calmar"], color="#123B24")
                 .highlight_min(subset=["Cost drag %/yr"], color="#123B24"), width="stretch")
    if st.button("Clear comparison"):
        st.session_state["compare"] = []
        st.rerun()


def page() -> None:
    st.markdown("# Backtest lab")
    theme.warn("Universe = today's NIFTY 200 applied to the past (survivorship bias), which inflates every "
               "strategy's absolute returns. Judge strategies <b>against the equal-weight line on the same universe</b>, "
               "not against NIFTY. All model scores are walk-forward out-of-sample; costs are real NSE delivery charges.")
    spec, run, add = _builder()
    if add:
        lst = st.session_state.setdefault("compare", [])
        sj = json.dumps(spec.to_dict())
        if sj not in lst:
            lst.append(sj)
            del lst[:-MAX_COMPARE]
    if run or add or "lab_spec" not in st.session_state:
        st.session_state["lab_spec"] = json.dumps(spec.to_dict())
    sj = st.session_state["lab_spec"]
    spec = bt.StrategySpec(**json.loads(sj))
    try:
        res = store.backtest(sj)
    except ValueError as e:
        st.error(str(e))
        return
    st.markdown(f"## {spec.name}")
    _kpis(res["metrics"], res["nav"])
    csv = pd.DataFrame({"nav": res["nav"]}).to_csv().encode()
    st.download_button("⬇ NAV CSV", csv, file_name="niveshrl_backtest_nav.csv", mime="text/csv")
    tabs = st.tabs(["Performance", "Returns", "Risk", "Trading", "Holdings", "Signal", f"Compare ({len(st.session_state.get('compare', []))})"])
    with tabs[0]:
        _perf_tab(res, spec)
    with tabs[1]:
        _returns_tab(res)
    with tabs[2]:
        _risk_tab(res)
    with tabs[3]:
        _trading_tab(res)
    with tabs[4]:
        _holdings_tab(res)
    with tabs[5]:
        _signal_tab(spec)
    with tabs[6]:
        _compare()
