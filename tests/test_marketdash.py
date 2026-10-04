"""Market dashboard: global table changes (yields in points), weekly correlation, breadth counts, RRG quadrants on a
constructed leader/laggard, and the stale-cache fallback."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from niveshrl.research import marketdash as MD
from niveshrl.research.data import Panel


def test_quadrants():
    assert MD.quadrant(101, 101) == "Leading" and MD.quadrant(101, 99) == "Weakening"
    assert MD.quadrant(99, 99) == "Lagging" and MD.quadrant(99, 101) == "Improving"
    assert MD.quadrant(np.nan, 100) == "–"


def test_global_table_changes_and_yield_points():
    idx = pd.bdate_range("2025-01-01", "2026-03-31")
    n = len(idx)
    rng = np.random.default_rng(0)
    nifty = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    closes = pd.DataFrame({"S&P 500": nifty * 50, "US 10Y yield": np.linspace(4.0, 5.0, n)}, index=idx)
    t = MD.global_table(closes, nifty)
    assert t.loc["S&P 500", "Corr. with NIFTY (weekly, 1y)"] == pytest.approx(1.0)
    assert t.loc["S&P 500", "1D"] == pytest.approx(nifty.iloc[-1] / nifty.iloc[-2] - 1)
    ytd_ref = nifty[nifty.index < "2026-01-01"].iloc[-1]
    assert t.loc["S&P 500", "YTD"] == pytest.approx(nifty.iloc[-1] / ytd_ref - 1)
    y = closes["US 10Y yield"]
    assert t.loc["US 10Y yield", "unit"] == "pp" and t.loc["US 10Y yield", "1D"] == pytest.approx(y.iloc[-1] - y.iloc[-2])


def _panel(closes: dict, bench, sectors):
    idx = pd.bdate_range("2023-01-02", periods=len(bench))
    c = pd.DataFrame(closes, index=idx)
    return Panel(close=c, volume=c * 0 + 1, sectors=pd.Series(sectors), names=pd.Series({k: k for k in closes}),
                 bench=pd.Series(bench, index=idx), vix=pd.Series(15.0, index=idx))


def test_breadth_counts_and_rrg_leader():
    n = 400
    up = np.linspace(100, 200, n)
    down = np.linspace(200, 100, n)
    accel = np.r_[np.full(300, 100.0), 100 * np.exp(np.linspace(0, 1, 100) ** 2)]   # flat, then accelerating
    p = _panel({"U1": up, "U2": up * 1.01, "U3": up * 0.99, "D1": down, "D2": down * 1.01, "D3": down * 0.99,
                "A1": accel, "A2": accel * 1.01, "A3": accel * 0.99},
               np.full(n, 100.0), {"U1": "Up", "U2": "Up", "U3": "Up", "D1": "Down", "D2": "Down", "D3": "Down",
                                    "A1": "Accel", "A2": "Accel", "A3": "Accel"})
    b = MD.breadth_history(p, years=5)
    last = b.iloc[-1]
    assert last["% above 50-day"] == pytest.approx(6 / 9) and last["Advancers"] == 6 and last["Decliners"] == 3
    assert last["New highs"] == 6 and last["New lows"] == 3
    tab, tails = MD.rrg(p)
    assert tab.loc["Accel", "Quadrant"] == "Leading" and tab.loc["Down", "RS-Ratio"] < 100
    assert set(tails) == {"Up", "Down", "Accel"} and len(tails["Up"]) == 8


def test_global_fetch_falls_back_to_stale_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(MD, "CACHE", tmp_path / "g.parquet")
    idx = pd.date_range("2026-01-01", periods=3)
    raw = pd.concat({"Close": pd.DataFrame({"^GSPC": [1.0, 2.0, 3.0]}, index=idx)}, axis=1)
    df = MD.fetch_global(download=lambda: raw)
    assert list(df.columns) == ["S&P 500"]
    os.utime(MD.CACHE, (0, 0))

    def offline():
        raise ConnectionError("offline")
    assert len(MD.fetch_global(download=offline)) == 3
