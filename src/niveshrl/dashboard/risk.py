"""RISK: volatility forecasting (LSTM vs GARCH) and market regimes (autoencoder)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..research.regime import FEATS, REGIME_COLORS, REGIMES, regime_stats
from ..research.volatility import VOL_MODELS, vol_metrics
from . import store, theme

VLABEL = {"hist": "Historical 21d", "ewma": "EWMA λ=0.94", "garch": "GARCH(1,1)", "lstm": "LSTM (deep learning)"}


def vol_section() -> None:
    st.markdown("## Volatility forecaster · next-month realised vol")
    vf = store.vol_forecasts()
    if vf is None:
        theme.note("Run <code>python scripts/train_volatility.py</code> first.")
        return
    d = vf.dropna(subset=["garch", "lstm"])
    tab = pd.DataFrame({VLABEL[m]: vol_metrics(d[m], d["rv_next"]) for m in VOL_MODELS}).T
    st.dataframe(tab.style.format({"RMSE log-vol": "{:.3f}", "MAE vol": "{:.2%}", "QLIKE": "{:.3f}", "Corr": "{:.3f}",
                                   "Bias (f/r)": "{:.2f}", "N": "{:,.0f}"})
                 .highlight_min(subset=["RMSE log-vol", "MAE vol", "QLIKE"], color="#123B24")
                 .highlight_max(subset=["Corr"], color="#123B24"), width="stretch")
    theme.note("Walk-forward, 194 stocks, month-end forecasts of the next 21 trading days. Lower RMSE/QLIKE and higher "
               "correlation are better; Bias = median forecast/realised (1.00 = unbiased). GARCH is refit yearly per "
               "stock; the LSTM is one pooled model retrained yearly on a rolling 8-year window.")
    dates = d.index.get_level_values(0)
    yr = {}
    for m in VOL_MODELS:
        e = (np.log(d[m]) - np.log(d["rv_next"])) ** 2
        yr[VLABEL[m]] = np.sqrt(e.groupby(dates.year).mean())
    yr = pd.DataFrame(yr)
    c1, c2 = st.columns(2)
    f1 = go.Figure([go.Scatter(x=yr.index, y=yr[c], name=c, mode="lines+markers") for c in yr.columns])
    f1.update_layout(height=300, title="RMSE of log-vol by year (lower = better)")
    c1.plotly_chart(f1, width="stretch")
    samp = d.sample(min(4000, len(d)), random_state=0)
    f2 = go.Figure([go.Scattergl(x=samp["lstm"] * 100, y=samp["rv_next"] * 100, mode="markers",
                                 marker=dict(size=3, color=theme.AMBER, opacity=0.35), name="LSTM"),
                    go.Scatter(x=[5, 120], y=[5, 120], line=dict(color=theme.MUTED, dash="dot"), name="perfect")])
    f2.update_layout(height=300, title="LSTM forecast vs realised (%, sample of 4k)", xaxis_type="log", yaxis_type="log",
                     hovermode="closest")
    c2.plotly_chart(f2, width="stretch")


def regime_section() -> None:
    st.markdown("## Market regimes · autoencoder + k-means (walk-forward)")
    w = store.regimes_weekly()
    if w is None:
        theme.note("Run <code>python scripts/train_regimes.py</code> first.")
        return
    p = store.panel()
    cur = w.iloc[-1]
    theme.kpis([("Current regime", None, "num", False, f"{cur['regime']} (week of {w.index[-1].date()})")] +
               [(f, cur[f], "x", False) for f in ["nifty_ret_20", "breadth", "nifty_dd"]])
    b = p.bench.resample("W-FRI").last().reindex(w.index)
    fig = go.Figure()
    for r in REGIMES:
        m = w["regime"] == r
        fig.add_trace(go.Scatter(x=w.index[m], y=b[m], mode="markers", name=r,
                                 marker=dict(color=REGIME_COLORS[r], size=5)))
    fig.add_trace(go.Scatter(x=w.index, y=b, line=dict(color="#39424E", width=1), name="NIFTY", showlegend=False))
    fig.update_layout(height=360, title="NIFTY 50, each week coloured by its out-of-sample regime label")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    f2 = go.Figure([go.Scattergl(x=w.loc[w.regime == r, "z1"], y=w.loc[w.regime == r, "z2"], mode="markers", name=r,
                                 marker=dict(color=REGIME_COLORS[r], size=5, opacity=0.7)) for r in REGIMES])
    f2.update_layout(height=320, title="Regime map: 2-D autoencoder embedding (refit yearly)", hovermode="closest")
    c1.plotly_chart(f2, width="stretch")
    stats = regime_stats(w, p)
    with c2:
        st.markdown("### Out-of-sample: what came next?")
        st.dataframe(stats.style.format({"weeks": "{:.0f}", "share": "{:.0%}", "next 4w NIFTY return": "{:+.2%}",
                                         "next-month NIFTY vol": "{:.1%}", "hit rate (4w > 0)": "{:.0%}"}),
                     width="stretch")
        theme.note("Regimes separate future <b>volatility</b> clearly; they do not reliably predict direction. "
                   "Stress weeks were often followed by rebounds (post-crash recoveries), which is why a regime filter "
                   "can reduce drawdowns and returns at the same time. Test it in the Lab.")
        med = w.groupby("regime")[FEATS].mean().reindex(REGIMES)
        st.dataframe(med.style.format("{:.3f}"), width="stretch")


def page() -> None:
    st.markdown("# Risk models")
    vol_section()
    regime_section()
