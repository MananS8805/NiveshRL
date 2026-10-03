"""Phase D: chart patterns on textbook shapes, pivot states, range-model walk-forward leakage, bhavcopy caching."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from niveshrl.research.data import Panel


def _panel(close, high=None, low=None, volume=None, start="2023-01-02"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.DataFrame({"X.NS": close}, index=idx)
    h = pd.DataFrame({"X.NS": high if high is not None else close * 1.005}, index=idx)
    lo = pd.DataFrame({"X.NS": low if low is not None else close * 0.995}, index=idx)
    v = pd.DataFrame({"X.NS": volume if volume is not None else np.full(len(close), 1e6)}, index=idx)
    return Panel(close=c, volume=v, sectors=pd.Series({"X.NS": "S"}), names=pd.Series({"X.NS": "X"}),
                 bench=pd.Series(np.linspace(100, 120, len(close)), index=idx), vix=pd.Series(15.0, index=idx),
                 open=c, high=h, low=lo)


def test_vcp_flat_base_and_nr7_on_textbook_shapes():
    from niveshrl.research import patterns as PT
    up = np.linspace(50, 100, 220)                                   # long uptrend: above the 200-day average
    swings = []
    for amp in (0.12, 0.07, 0.03):                                    # three 15-day windows, each tighter
        t = np.linspace(0, 2 * np.pi, 15)
        swings.append(100 * (1 + amp / 2 * np.sin(t)))
    c = np.r_[up, *swings]
    vol = np.r_[np.full(220, 2e6), np.full(15, 1.5e6), np.full(15, 1.0e6), np.full(15, 0.6e6)]
    p = _panel(c, high=c * 1.002, low=c * 0.998, volume=vol)
    pats = PT.detect(p)
    assert pats["vcp"]["X.NS"].iloc[-1]
    flat = np.r_[np.linspace(50, 100, 220), 100 + 3 * np.sin(np.linspace(0, 6, 30))]
    pf = _panel(flat, high=flat * 1.003, low=flat * 0.997)
    assert PT.detect(pf)["flat_base"]["X.NS"].iloc[-1]
    rng = np.r_[np.full(10, 4.0), [1.0]]                              # the last day is the narrowest of 7
    c3 = np.full(11, 100.0)
    p3 = _panel(c3, high=c3 + rng / 2, low=c3 - rng / 2)
    assert PT.detect(p3)["nr7"]["X.NS"].iloc[-1]


def test_patterns_use_no_future_data():
    from niveshrl.research import patterns as PT
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    full = PT.detect(_panel(c))
    part = PT.detect(_panel(c[:300]))
    for k in PT.PATTERNS:
        pd.testing.assert_series_equal(full[k]["X.NS"].iloc[:300], part[k]["X.NS"], check_names=False)


def test_pivot_states():
    from niveshrl.research import patterns as PT
    c = np.r_[np.full(30, 100.0), 103.0]                             # pivot = 100.5 (20-day high) -> +2.5% breakout
    p = _panel(c, high=c + 0.5, low=c - 0.5)
    dist, state = PT.pivot_state(p)
    assert state["X.NS"].iloc[-1] == "breakout" and 0 < dist["X.NS"].iloc[-1] < 0.03
    c2 = np.r_[np.full(30, 100.0), 103.0, 97.0]                       # broke out, now > 2% back under: failed
    _, s2 = PT.pivot_state(_panel(c2, high=c2 + 0.5, low=c2 - 0.5))
    assert s2["X.NS"].iloc[-1] == "failed"


def test_range_model_walk_forward_does_not_see_future_labels():
    from niveshrl.research import range_model as RM
    rng = np.random.default_rng(1)
    n_days, n_st = 1600, 40
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    cols = [f"S{i}.NS" for i in range(n_st)]
    vol = rng.uniform(0.005, 0.03, n_st)
    r = rng.normal(0, 1, (n_days, n_st)) * vol
    c = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    hl = c * vol * rng.uniform(0.5, 1.5, (n_days, n_st))
    p = Panel(close=c, volume=c * 0 + 1e6, sectors=pd.Series("S", index=cols), names=pd.Series(cols, index=cols),
              bench=c.mean(axis=1), vix=pd.Series(15.0, index=idx), open=c, high=c + hl / 2, low=c - hl / 2)
    X, y, _ = RM.build(p)
    a = RM.walk_forward(X, y, first_year=2019, window=4)
    y2 = y.copy()
    late = y2.index.get_level_values(0).year >= 2019
    y2[late] = rng.permutation(y2[late].to_numpy())                    # scramble every label from the test years on
    b = RM.walk_forward(X, y2, first_year=2019, window=4)
    yr19 = a.index.get_level_values(0).year == 2019
    np.testing.assert_allclose(a[yr19].to_numpy(), b[yr19].to_numpy())  # 2019's predictions use only 2015-2018 labels


def test_bhavcopy_failures_are_never_cached(tmp_path, monkeypatch):
    import urllib.error
    from niveshrl.research import delivery as DV
    monkeypatch.setattr(DV, "CACHE", tmp_path)

    def boom(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(DV.urllib.request, "urlopen", boom)
    with pytest.raises(urllib.error.URLError):
        DV.fetch(pd.Timestamp("2026-10-01"))
    assert not list(tmp_path.glob("*.csv"))                          # nothing cached after a network failure

    def holiday(*a, **k):
        raise urllib.error.HTTPError("u", 404, "not found", None, None)
    monkeypatch.setattr(DV.urllib.request, "urlopen", holiday)
    assert DV.fetch(pd.Timestamp("2026-10-02")) is None and not list(tmp_path.glob("*.csv"))
