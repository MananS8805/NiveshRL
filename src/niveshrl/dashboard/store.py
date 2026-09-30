"""Cached access to everything the dashboard reads. Heavy work (model training)
happens offline in scripts/; the dashboard only loads results and runs fast
backtests over them, which keeps it responsive."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..research import backtest as bt
from ..research.data import load_panel
from ..research.rankers import load_predictions
from ..research.regime import daily_regimes, load_regimes
from ..research.signals import available_signals, get_scores
from ..research.volatility import load_vol


@st.cache_resource(show_spinner="Loading NIFTY 200 panel…")
def panel():
    return load_panel()


@st.cache_data(show_spinner=False)
def scores(key: str) -> pd.DataFrame:
    return get_scores(panel(), key)


@st.cache_data(show_spinner=False, ttl=300)  # pick up newly trained models
def signals() -> dict[str, str]:
    return available_signals()


@st.cache_data(show_spinner=False, ttl=300)  # pick up newly trained models
def predictions(model: str) -> pd.DataFrame | None:
    return load_predictions(model)


@st.cache_data(show_spinner="Computing ranking diagnostics…")
def diagnostics(model: str, start: str | None = None, end: str | None = None) -> dict | None:
    """Decile returns, monthly IC and per-year table for a model's predictions (cached)."""
    from ..research.evaluate import by_year, decile_returns, monthly_ic
    pr = predictions(model)
    if pr is None:
        return None
    d = pr.index.get_level_values(0)
    if start:
        pr = pr[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    return {"dec": decile_returns(pr), "ic": monthly_ic(pr), "by_year": by_year(pr)}


@st.cache_data(show_spinner=False, ttl=300)  # pick up newly trained models
def regimes_weekly() -> pd.DataFrame | None:
    return load_regimes()


@st.cache_data(show_spinner=False)
def regimes_daily() -> pd.Series | None:
    w = regimes_weekly()
    return None if w is None else daily_regimes(w, panel().close.index)


@st.cache_data(show_spinner=False, ttl=300)  # pick up newly trained models
def vol_forecasts() -> pd.DataFrame | None:
    return load_vol()


@st.cache_data(show_spinner="Running backtest…", max_entries=64)
def backtest(spec_json: str) -> dict:
    """Backtest by JSON spec (hashable for the cache). Returns plain data for caching."""
    spec = bt.StrategySpec(**json.loads(spec_json))
    res = bt.run_backtest(panel(), scores(spec.signal), spec, regimes=regimes_daily())
    last = max(res.weights) if res.weights else None
    return {"nav": res.nav, "rebalances": res.rebalances, "metrics": res.metrics,
            "last_weights": res.weights.get(last) if last is not None else None, "last_date": last,
            "weights": res.weights}


def results_csv(name: str) -> pd.DataFrame | None:
    from ..config import ROOT
    path = ROOT / "report" / "results" / name
    return pd.read_csv(path, index_col=0) if path.exists() else None
