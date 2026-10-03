"""Framework-free data access for the desktop app (the Streamlit app's ``dashboard/store.py`` without Streamlit).

Everything is loaded lazily and cached in memory. Daily-pipeline outputs are
keyed on the run's ``ran_at`` stamp, so a new run (from the worker process)
is picked up on the next read without restarting. Caches are bounded: the
backtest cache keeps the last 32 specs, so memory stays flat over long sessions.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict

import pandas as pd

from ..config import ROOT
from ..research import daily

_lock = threading.RLock()
_cache: dict[str, tuple[float, object]] = {}
_bt_cache: OrderedDict[str, dict] = OrderedDict()
BT_CACHE_MAX = 32


def _cached(key: str, ttl: float | None, fn):
    with _lock:
        hit = _cache.get(key)
        if hit is not None and (ttl is None or time.time() - hit[0] < ttl):
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
    return val


def clear(prefix: str = "") -> None:
    with _lock:
        for k in [k for k in _cache if k.startswith(prefix)]:
            _cache.pop(k, None)
        if not prefix:
            _bt_cache.clear()


# --------------------------------------------------------------------------- research outputs
def panel():
    from ..research.data import load_panel
    return _cached("panel", None, load_panel)


def signals() -> dict[str, str]:
    from ..research.signals import available_signals
    return _cached("signals", 300, available_signals)


def scores(key: str) -> pd.DataFrame:
    from ..research.signals import get_scores
    return _cached(f"scores:{key}", None, lambda: get_scores(panel(), key))


def predictions(model: str) -> pd.DataFrame | None:
    from ..research.rankers import load_predictions
    return _cached(f"pred:{model}", 300, lambda: load_predictions(model))


def diagnostics(model: str, start: str | None = None, end: str | None = None) -> dict | None:
    from ..research.evaluate import by_year, decile_returns, monthly_ic

    def build():
        pr = predictions(model)
        if pr is None:
            return None
        if start:
            d = pr.index.get_level_values(0)
            pr2 = pr[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
        else:
            pr2 = pr
        return {"dec": decile_returns(pr2), "ic": monthly_ic(pr2), "by_year": by_year(pr2)}
    return _cached(f"diag:{model}:{start}:{end}", 600, build)


def regimes_weekly() -> pd.DataFrame | None:
    from ..research.regime import load_regimes
    return _cached("reg_w", 300, load_regimes)


def regimes_daily() -> pd.Series | None:
    from ..research.regime import daily_regimes

    def build():
        w = regimes_weekly()
        return None if w is None else daily_regimes(w, panel().close.index)
    return _cached("reg_d", 300, build)


def vol_forecasts() -> pd.DataFrame | None:
    from ..research.volatility import load_vol
    return _cached("vol", 300, load_vol)


def results_csv(name: str) -> pd.DataFrame | None:
    path = ROOT / "report" / "results" / name
    return pd.read_csv(path, index_col=0) if path.exists() else None


def backtest(spec) -> dict:
    """Run (or reuse) a backtest for a ``StrategySpec``; LRU-bounded cache."""
    from ..research import backtest as bt
    key = json.dumps(spec.to_dict(), sort_keys=True, default=str)
    with _lock:
        if key in _bt_cache:
            _bt_cache.move_to_end(key)
            return _bt_cache[key]
    res = bt.run_backtest(panel(), scores(spec.signal), spec, regimes=regimes_daily())
    last = max(res.weights) if res.weights else None
    out = {"nav": res.nav, "rebalances": res.rebalances, "metrics": res.metrics, "spec": spec,
           "last_weights": res.weights.get(last) if last is not None else None, "last_date": last}
    with _lock:
        _bt_cache[key] = out
        while len(_bt_cache) > BT_CACHE_MAX:
            _bt_cache.popitem(last=False)
    return out


# --------------------------------------------------------------------------- daily pipeline outputs
def daily_stamp() -> str:
    meta = daily.latest_meta()
    return meta["ran_at"] if meta else ""


def dload(name: str):
    stamp = daily_stamp()
    if not stamp:
        return None
    return _cached(f"daily:{stamp}:{name}", None, lambda: daily.load(name))


def screener_table() -> pd.DataFrame | None:
    from ..research import screener as scr
    stamp = daily_stamp()
    if not stamp or daily.latest_dir() is None:
        return None

    def build():
        monthly = predictions("transformer")
        if monthly is None:
            monthly = predictions("ffnn")
        return scr.build_table(daily.latest_dir(), monthly)
    return _cached(f"daily:{stamp}:screener", None, build)


def drop_stale_daily() -> None:
    """Forget daily outputs from older runs (called when a new run lands)."""
    stamp = daily_stamp()
    with _lock:
        for k in [k for k in _cache if k.startswith("daily:") and not k.startswith(f"daily:{stamp}:")]:
            _cache.pop(k, None)


def fundamentals(ticker: str) -> dict:
    from ..research import fundamentals as fx
    return _cached(f"fx:{ticker}", 3600, lambda: fx.fetch(ticker))


def prune_fundamentals(keep: int = 40) -> None:
    """Bound the per-stock fundamentals cache (each holds several statement frames)."""
    with _lock:
        fx = sorted((v[0], k) for k, v in _cache.items() if k.startswith("fx:"))
        for _, k in fx[:-keep]:
            _cache.pop(k, None)


def cache_size() -> int:
    with _lock:
        return len(_cache) + len(_bt_cache)
