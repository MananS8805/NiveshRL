"""NiveshRL research terminal.

    streamlit run app.py

Screens (sidebar):
  TODAY daily briefing, market habits, top stocks to monitor tomorrow
  SCRN  technical + fundamental + sentiment + model screener with presets
  WATCH personal watchlist (must have / preferred) with alerts
  MKT   market monitor: NIFTY/VIX, regime, NIFTY 200 heatmap, breadth, movers, model consensus
  LAB   backtest lab: build a strategy from any model/factor signal, India costs, full tearsheet, compare
  RANK  deep-learning stock rankers (FFNN / LSTM / Transformer), out-of-sample record, live ranking
  EQ    single-stock drilldown
  RISK  volatility forecaster (LSTM vs GARCH) and market regimes (autoencoder)
  RL    reinforcement-learning allocator vs classical baselines
  PLAN  personalised investor plan from the RL allocator
  HELP  plain-language explainer

Heavy models are trained offline (scripts/train_*.py); the app loads their
walk-forward predictions and runs fast backtests on demand.

Educational tool only. It is not investment advice, and NiveshRL is not a
SEBI-registered Research Analyst or Investment Adviser.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="NiveshRL Terminal", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

from niveshrl.dashboard import desk, equity, lab, market, ranker, risk, rl, theme  # noqa: E402

theme.inject()

SCREENS = {
    "TODAY · Briefing + top picks": desk.today_page,
    "MKT  · Market monitor": market.page,
    "SCRN · Screener": desk.screener_page,
    "WATCH · Watchlist": desk.watch_page,
    "LAB  · Backtest lab": lab.page,
    "RANK · Stock ranker (DL)": ranker.page,
    "EQ   · Equity drilldown": equity.page,
    "RISK · Vol & regimes": risk.page,
    "RL   · RL allocator": rl.research_page,
    "PLAN · Investor plan": rl.investor_page,
    "HELP · How it works": lambda: rl.about_page(rl.runs_available()),
}

st.sidebar.markdown("# NiveshRL ▸ Terminal")
choice = st.sidebar.radio("Screen", list(SCREENS), label_visibility="collapsed")
st.sidebar.markdown("---")
desk.refresh_widget()
st.sidebar.markdown("---")
st.sidebar.caption(rl.DISCLAIMER)
SCREENS[choice]()
