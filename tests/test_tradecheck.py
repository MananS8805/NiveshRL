"""Trade understanding: scenario P&L arithmetic (costs, tax only on gains), similar-setup matching and relaxation,
context tags without look-ahead, and the Go / Wait / No-go rules."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from niveshrl.research import plans as P
from niveshrl.research import tradecheck as TC
from niveshrl.research.data import Panel

ZERO = {"brokerage_rate": 0.0, "brokerage_cap": 0, "stt_rate": 0.0, "exchange_rate": 0.0, "sebi_rate": 0.0,
        "stamp_rate": 0.0, "dp_charge": 0.0, "half_spread": 0.0, "gst_rate": 0.0}


def _plan(**kw):
    d = dict(entry=100.0, stop=90.0, risk_per_share=10.0, t1=115.0, t2=125.0, qty=30, capital=100_000.0, stop_pct=0.10,
             atr=5.0, rupee_risk=300.0, risk_pct=0.01, multiplier=1.0, position_value=3000.0)
    d.update(kw)
    return SimpleNamespace(**d)


def test_scenarios_without_costs_match_hand_arithmetic():
    sc = TC.scenarios(_plan(), stcg_rate=0.2, rates=ZERO)
    assert sc.loc["Stop hit", "Net ₹"] == pytest.approx(-300) and sc.loc["Stop hit", "Tax ₹"] == 0
    assert sc.loc["Stop hit", "Net R"] == pytest.approx(-1.0)
    # 10 shares at T1 (+15 each) and 20 at T2 (+25 each) = 650 gross, 20% tax
    assert sc.loc["T1, then T2", "Gross ₹"] == pytest.approx(650)
    assert sc.loc["T1, then T2", "Net ₹"] == pytest.approx(520) and sc.loc["T1, then T2", "Tax ₹"] == pytest.approx(130)
    assert sc.loc["T1, then stopped at entry", "Gross ₹"] == pytest.approx(150)
    assert sc.loc["Gap 3% through the stop", "Exit price"] == pytest.approx(87.3)
    assert sc.loc["Flat (sold at entry)", "Net ₹"] == 0


def test_scenarios_with_real_costs_flat_trade_loses_money():
    sc = TC.scenarios(_plan())
    flat = sc.loc["Flat (sold at entry)"]
    assert flat["Net ₹"] < 0 and flat["Costs ₹"] == pytest.approx(P.delivery_costs(100, 100, 30)["total"])
    assert (sc["Tax ₹"] >= 0).all() and (sc.loc[sc["Gross ₹"] <= 0, "Tax ₹"] == 0).all()


def test_similar_setups_relaxes_the_least_important_tag():
    rng = np.random.default_rng(0)
    n = 400
    h = pd.DataFrame({"trend": rng.choice(["above", "below"], n), "rsi_zone": rng.choice(["<40", "40-60"], n),
                      "near_high": rng.choice(["within 2%", "over 8% below"], n), "market": "above",
                      "r": rng.normal(0, 1, n), "days": 10, "reason": "stop"})
    h.loc[(h.trend == "above") & (h.rsi_zone == "<40"), "r"] += 5          # this context is clearly better
    now = {"trend": "above", "rsi_zone": "<40", "near_high": "within 2%", "market": "below"}   # no trade had market below
    s = TC.similar_setups(h, now, min_n=30)
    assert "market" in s.dropped and s.n >= 30 and set(s.used) <= {"trend", "rsi_zone", "near_high"}
    assert s.stats["avg_r"] > s.baseline["avg_r"] + 3
    assert TC.similar_setups(pd.DataFrame(), now).n == 0
    hist = TC.r_histogram(pd.Series([-1.1, -1.0, 0.2, 4.0]))
    assert hist.sum() == pytest.approx(1.0) and hist["> 3R"] == pytest.approx(0.25)


def test_context_tags_use_no_future_data():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    c = pd.DataFrame({"X.NS": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))}, index=idx)
    p = Panel(close=c, volume=c * 0 + 1, sectors=pd.Series({"X.NS": "S"}), names=pd.Series({"X.NS": "X"}),
              bench=c["X.NS"], vix=pd.Series(15.0, index=idx), high=c * 1.01)
    full = TC.context_tags(p)
    cut = Panel(close=c.iloc[:300], volume=c.iloc[:300], sectors=p.sectors, names=p.names, bench=c["X.NS"].iloc[:300],
                vix=p.vix, high=c.iloc[:300] * 1.01)
    part = TC.context_tags(cut)
    for k in ("trend", "rsi_zone", "near_high"):
        pd.testing.assert_series_equal(full[k]["X.NS"].iloc[:300], part[k]["X.NS"], check_names=False)
    assert TC.tags_now(p, "X.NS") == TC.tags_on(full, "X.NS", idx[-1])
    hist = pd.DataFrame({"ticker": ["X.NS", "X.NS"], "signal_date": [idx[250], idx[350]], "status": ["closed", "open"],
                         "r": [1.0, 2.0], "days": [5, 5], "reason": ["stop", "open"]})
    th = TC.tag_history(hist, full)
    assert len(th) == 1 and th["trend"].iloc[0] == full["trend"]["X.NS"].iloc[250]


def test_checklist_verdicts():
    ok_risk = SimpleNamespace(multiplier=1.0, state="Normal", reasons=[])
    row = pd.Series({"vs_sma200": 0.1, "turnover_cr": 500.0, "rsi14": 55.0, "prob_up": 0.6, "monthly_pct": 0.7})
    assert TC.checklist(_plan(stop_pct=0.05), ok_risk, row, None, None)[0] == "Go"
    bad = row.copy()
    bad["vs_sma200"], bad["rsi14"] = -0.1, 80.0
    assert TC.checklist(_plan(stop_pct=0.05), ok_risk, bad, None, None)[0] == "Wait"
    assert TC.checklist(_plan(qty=0), ok_risk, row, None, None)[0] == "No-go"
    v, checks = TC.checklist(_plan(stop_pct=0.05), ok_risk, row, None, 3)
    assert v == "Go" and any(c.item == "Results soon" and c.status == "caution" for c in checks)
