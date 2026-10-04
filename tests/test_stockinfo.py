"""Stock deep-dive: report-card percentiles, relative strength/beta, seasonality, drawdowns, results reactions
(announcement timing), dividends and the volatility cone against its closed form."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from niveshrl.research import stockinfo as SI
from niveshrl.research.data import Panel


def _panel(closes: dict, bench, start="2020-01-01", sectors=None):
    idx = pd.bdate_range(start, periods=len(bench))
    c = pd.DataFrame(closes, index=idx)
    sec = pd.Series(sectors or {k: "S" for k in closes})
    return Panel(close=c, volume=c * 0 + 1e6, sectors=sec, names=pd.Series({k: k for k in closes}),
                 bench=pd.Series(bench, index=idx), vix=pd.Series(15.0, index=idx), open=c, high=c * 1.01, low=c * 0.99)


def test_report_card_percentiles_within_sector():
    t = pd.DataFrame({"sector": ["IT"] * 5 + ["Bank"] * 5,
                      "trailingPE": [10, 20, 30, 40, -5, 1, 2, 3, 4, 5],          # A cheapest of the profitable IT names
                      "returnOnEquity": [0.30, 0.2, 0.1, 0.05, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1]},
                     index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    rc = SI.report_card(t, "A")
    v, q = rc["Valuation"], rc["Quality"]
    assert v["group"] == "IT" and v["peers"] == 5
    assert v["score"] == pytest.approx(100.0)                 # lower P/E than all 3 other profitable peers
    assert q["score"] == pytest.approx(80.0) and q["verdict"] == "strong"   # beats 4 of 5 on ROE
    assert "better P/E than 100% of IT peers" in v["reasons"][0]
    assert math.isnan(SI.report_card(t, "E")["Valuation"]["score"])         # loss-maker: no valuation percentile
    assert rc["Growth"]["verdict"] == "–"                                    # no growth columns at all
    assert SI.report_card(t, "ZZZ") == {}


def test_relative_strength_and_beta():
    rng = np.random.default_rng(3)
    rb = rng.normal(0, 0.01, 400)
    bench = 100 * np.exp(np.cumsum(rb))
    x = 50 * np.exp(np.cumsum(1.5 * rb))                         # beta exactly 1.5, perfectly correlated
    y = 80 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    p = _panel({"X": x, "Y": y}, bench)
    rs = SI.relative_strength(p, "X")
    assert rs["beta"] == pytest.approx(1.5, rel=0.02) and rs["corr"] == pytest.approx(1.0, abs=1e-3)
    assert rs["rs_nifty"].iloc[0] == pytest.approx(1.0) and rs["n_peers"] == 1


def test_seasonality_and_drawdowns():
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    c = pd.Series(100.0, index=idx)
    c[idx.month == 3] = 90.0                                     # every March closes 10% lower, April recovers
    p = _panel({"X": c.to_numpy()}, np.full(len(idx), 100.0), start="2018-01-01")
    se = SI.seasonality(p, "X")
    assert se.loc["Mar", "avg return"] == pytest.approx(-0.10) and se.loc["Mar", "up years"] == 0
    assert se.loc["Apr", "avg return"] == pytest.approx(1 / 0.9 - 1)
    dd = SI.drawdowns(p, "X", top=2)
    assert len(dd) == 2 and np.allclose(dd["depth"], -0.10)
    assert dd["recovered"].notna().all() and (dd["days to recover"] > 0).all()


def test_results_reaction_timing_after_close_counts_next_day():
    idx = pd.bdate_range("2024-01-01", periods=60)
    c = np.full(60, 100.0)
    c[30:] = 110.0                                               # the jump happens on session 30 (2024-02-12)
    p = _panel({"X": c}, np.full(60, 100.0), start="2024-01-01")
    day29 = idx[29]                                              # announced on session 29 after the close (IST)
    ts = pd.Timestamp(f"{day29.date()} 16:30", tz="Asia/Kolkata").tz_convert("America/New_York")
    earn = pd.DataFrame({"EPS Estimate": [10.0], "Reported EPS": [11.0], "Surprise(%)": [10.0]}, index=[ts])
    df, s = SI.results_reaction(p, "X", earn)
    assert df["results"].iloc[0] == idx[30]
    assert df["day move"].iloc[0] == pytest.approx(0.10) and s["up reactions"] == 1.0 and s["beat estimate"] == 1.0
    before = pd.Timestamp(f"{idx[30].date()} 10:00", tz="Asia/Kolkata")
    assert SI.reaction_day(before, idx) == idx[30]               # during market hours: same day
    future = pd.DataFrame({"EPS Estimate": [1.0]}, index=[pd.Timestamp("2030-01-01", tz="UTC")])
    assert SI.results_reaction(p, "X", future)[0].empty           # an upcoming date is never a measured reaction


def test_dividends_trailing_yield():
    d = pd.Series([5.0, 6.0, 7.0], index=pd.to_datetime(["2023-06-01", "2024-06-01", "2025-01-10"]).tz_localize("Asia/Kolkata"))
    tab, s = SI.dividends(d, 100.0, asof=pd.Timestamp("2025-03-01"))
    assert s["trailing 12m ₹"] == pytest.approx(13.0) and s["trailing yield"] == pytest.approx(0.13)
    assert s["years paid"] == 3 and len(tab) == 3
    assert SI.dividends(pd.Series(dtype=float), 100.0)[1] == {} and SI.dividends(None, 1.0)[1] == {}


def test_vol_cone_matches_closed_form():
    cone = SI.vol_cone(1000.0, 0.252, horizons=(63,))
    s = 0.252 * math.sqrt(63 / 252)
    row = cone.loc["63 days"]
    assert row["68% high"] == pytest.approx(1000 * math.exp(s)) and row["95% low"] == pytest.approx(1000 * math.exp(-1.96 * s))
    assert row["68% low"] * row["68% high"] == pytest.approx(1000.0 ** 2)   # symmetric in log space


def test_cone_sigma_falls_back_when_forecast_is_stale():
    vf = pd.DataFrame({"lstm": [0.2]}, index=pd.MultiIndex.from_tuples([(pd.Timestamp("2026-01-30"), "X")]))
    tab = pd.DataFrame({"vol_60d": [0.33]}, index=["X"])
    assert SI.cone_sigma("X", vf, tab, pd.Timestamp("2026-02-20"))[0] == 0.2
    sig, src = SI.cone_sigma("X", vf, tab, pd.Timestamp("2026-10-01"))
    assert sig == 0.33 and "realised" in src
