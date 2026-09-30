"""RANK: deep-learning stock rankers, their out-of-sample record, and today's ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..research.signals import MODEL_LABELS
from . import store, theme

MODELS = ["ffnn", "lstm", "transformer", "logreg", "momentum"]
LABEL = MODEL_LABELS | {"momentum": "Momentum 12-1 (baseline)"}


def _available() -> list[str]:
    return [m for m in MODELS if store.predictions(m) is not None]


def scoreboard() -> None:
    tab = store.results_csv("rankers_summary.csv")
    if tab is None:
        theme.note("Run <code>python scripts/train_rankers.py</code> to populate the scoreboard.")
        return
    tab = tab.drop(columns=[c for c in ["Train time (s)", "Months"] if c in tab.columns])
    tab.index = [LABEL.get(i, i) for i in tab.index]
    st.markdown("### Out-of-sample scoreboard (walk-forward, 2012 → today, before costs)")
    st.dataframe(tab.style.format({"AUC": "{:.3f}", "Accuracy": "{:.1%}", "IC mean": "{:.3f}", "IC t-stat": "{:.2f}",
                                   "IC hit rate": "{:.0%}", "Top decile / mo": "{:+.2%}",
                                   "Bottom decile / mo": "{:+.2%}", "Spread / mo": "{:+.2%}",
                                   "Spread t-stat": "{:.2f}"})
                 .highlight_max(subset=["IC mean", "IC t-stat", "Spread / mo"], color="#123B24"), width="stretch")
    theme.note("IC = Spearman rank correlation between the score and next month's return, per month. "
               "Spread = top-decile minus bottom-decile equal-weight return. A t-stat above ~2 means the "
               "average is unlikely to be luck. Same universe, same months, same features for every model.")


def history_charts(models: list[str]) -> None:
    c1, c2 = st.columns(2)
    f1, f2, f3 = go.Figure(), go.Figure(), go.Figure()
    for i, m in enumerate(models):
        dg = store.diagnostics(m)
        col = theme.SERIES[i % len(theme.SERIES)]
        y = dg["by_year"]
        f1.add_trace(go.Bar(x=y.index, y=y["IC"], name=m.upper(), marker_color=col))
        dec = dg["dec"]
        f2.add_trace(go.Scatter(x=[f"D{j}" for j in dec.columns], y=dec.mean() * 100, name=m.upper(),
                                mode="lines+markers", line=dict(color=col)))
        spread = (dec.iloc[:, -1] - dec.iloc[:, 0]).fillna(0)
        f3.add_trace(go.Scatter(x=spread.index, y=(1 + spread).cumprod(), name=m.upper(), line=dict(color=col)))
    f1.update_layout(height=300, barmode="group", title="Mean monthly IC by year")
    f2.update_layout(height=300, title="Avg next-month return by decile (%)")
    c1.plotly_chart(f1, width="stretch")
    c2.plotly_chart(f2, width="stretch")
    f3.update_layout(height=300, yaxis_type="log", title="Cumulative top-minus-bottom decile (log, before costs)")
    st.plotly_chart(f3, width="stretch")


def live_ranking(models: list[str]) -> None:
    p = store.panel()
    frames, probs = [], []
    for m in models:
        pr = store.predictions(m)
        d = pr.index.get_level_values(0).max()
        s = pr.xs(d, level=0)["score"]
        frames.append(s.rank(pct=True).rename(f"{m.upper()} %ile"))
        if m != "momentum":
            probs.append(s.rename(f"{m.upper()} P"))
    tab = pd.concat(frames + probs, axis=1)
    pct_cols = [c for c in tab.columns if c.endswith("%ile")]
    tab["Consensus"] = tab[pct_cols].mean(axis=1)
    tab["Agreement"] = 1 - tab[pct_cols].std(axis=1) * 2
    ret1m = p.close.ffill().iloc[-1] / p.close.ffill().iloc[-22] - 1
    tab.insert(0, "1M ret", ret1m.reindex(tab.index))
    tab.insert(0, "Sector", p.sectors.reindex(tab.index))
    tab = tab.sort_values("Consensus", ascending=False)
    tab.index = [t.replace(".NS", "") for t in tab.index]
    st.markdown(f"### Live ranking · features as of {d.date()} · predicting next month")
    c1, c2, c3 = st.columns([2, 1, 1])
    secs = c1.multiselect("Sector filter", sorted(tab["Sector"].dropna().unique()))
    show = c2.slider("Rows", 10, len(tab), 30)
    q = c3.text_input("Search ticker", "")
    view = tab
    if secs:
        view = view[view["Sector"].isin(secs)]
    if q:
        view = view[view.index.str.contains(q.upper())]
    num = [c for c in view.columns if c not in ("Sector", "1M ret")]
    st.dataframe(view.head(show).style.format({c: "{:.0%}" for c in num} | {"1M ret": "{:+.1%}"})
                 .background_gradient(subset=["Consensus"], cmap="RdYlGn", vmin=0, vmax=1), width="stretch", height=520)
    theme.note("P = model probability that the stock beats the cross-sectional median next month (0.5 = no view). "
               "Agreement near 100% = models concur. Research output, not a recommendation.")


def page() -> None:
    st.markdown("# Stock ranker · deep learning")
    theme.note("Feed-forward net with a 4-unit bottleneck (Takeuchi & Lee 2013, 33 inputs: 12 monthly + 20 daily "
               "cumulative returns + January dummy), an LSTM and a Transformer on the same task, vs logistic "
               "regression and plain 12-1 momentum. Target: beat the cross-sectional median next month. "
               "Rolling 8-year walk-forward: every score shown was produced by a model that never saw that month.")
    models = _available()
    if not models:
        st.warning("No ranker predictions yet: run `python scripts/train_rankers.py`.")
        return
    scoreboard()
    pick = st.multiselect("Models to chart", models, default=[m for m in ("ffnn", "lstm", "transformer", "momentum") if m in models])
    if pick:
        history_charts(pick)
    live_ranking([m for m in models if m != "logreg"])
