"""Trade plans, risk state, costs/R and portfolio storage, checked against hand-calculated numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from niveshrl.research import plans as P
from niveshrl.research.data import Panel


def _panel(close, low=None, high=None, bench=None):
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    c = pd.DataFrame({"X.NS": close}, index=idx)
    lo = pd.DataFrame({"X.NS": low if low is not None else close}, index=idx)
    hi = pd.DataFrame({"X.NS": high if high is not None else close}, index=idx)
    b = pd.Series(bench if bench is not None else np.linspace(100, 200, len(close)), index=idx)
    return Panel(close=c, volume=c * 0 + 1e6, sectors=pd.Series({"X.NS": "S"}), names=pd.Series({"X.NS": "X"}),
                 bench=b, vix=pd.Series(15.0, index=idx), open=c, high=hi, low=lo)


def test_stop_clamped_between_2_and_2_5_atr_and_targets_in_R():
    n = 60
    close = np.full(n, 100.0)
    p = _panel(close, low=close - 1, high=close + 1)          # true range 2 -> ATR ~ 2
    atr = pd.DataFrame({"X.NS": np.full(n, 2.0)}, index=p.close.index)
    pl = P.make_plan(p, "X.NS", capital=100_000, risk_pct=0.01, atr_df=atr)
    # 10-day low 99*0.998 = 98.80 -> distance 1.20 < 2*ATR=4 -> widened to 4
    assert pl.stop == pytest.approx(96.0) and pl.risk_per_share == pytest.approx(4.0)
    assert pl.t1 == pytest.approx(106.0) and pl.t2 == pytest.approx(110.0)
    # risk sizing gives 1,000 / 4 = 250 shares = Rs 25,000 > 20% of capital -> capped at 200 shares
    assert pl.qty == 200 and any("capped at 20%" in n for n in pl.notes)
    small = P.make_plan(p, "X.NS", capital=100_000, risk_pct=0.005, atr_df=atr)
    assert small.qty == 125                                   # 500 / 4, under the cap


def test_stop_capped_at_8_percent_and_position_cap():
    n = 60
    close = np.full(n, 100.0)
    p = _panel(close, low=close - 20, high=close + 20)
    atr = pd.DataFrame({"X.NS": np.full(n, 5.0)}, index=p.close.index)
    pl = P.make_plan(p, "X.NS", capital=100_000, risk_pct=0.01, atr_df=atr)
    # 2*ATR = 10 > 8% of 100 = 8 -> capped at 8
    assert pl.risk_per_share == pytest.approx(8.0) and pl.stop == pytest.approx(92.0)
    assert pl.qty == 125 and pl.position_value == pytest.approx(12_500)
    tiny = P.make_plan(p, "X.NS", capital=100_000, risk_pct=0.10, atr_df=atr)
    assert tiny.qty == 200                                    # 20% of capital / 100 caps 1,250 shares


def test_risk_state_halves_below_200dma_and_in_stress():
    n = 260
    up = _panel(np.linspace(50, 150, n), bench=np.linspace(100, 200, n))
    assert P.risk_state(up).multiplier == 1.0
    down = _panel(np.linspace(50, 150, n), bench=np.linspace(200, 100, n))
    rs = P.risk_state(down)
    assert rs.multiplier == 0.5 and rs.state == "Below 200-DMA"
    assert P.risk_state(up, regime="Stress").multiplier == 0.5


def test_delivery_costs_and_r_multiple_by_hand():
    rates = {"stt_rate": 0.001, "exchange_rate": 0.0000297, "sebi_rate": 0.000001, "stamp_rate": 0.00015,
             "gst_rate": 0.18, "brokerage_rate": 0.0, "brokerage_cap": 0.0, "dp_charge": 15.93, "half_spread": 0.0}
    c = P.delivery_costs(100.0, 110.0, 100, rates)
    turnover = 10_000 + 11_000
    expected = 0.001 * turnover + 0.0000297 * turnover + 0.000001 * turnover + 0.00015 * 10_000 + 15.93 \
        + 0.18 * (0.0000297 * turnover + 0.000001 * turnover)
    assert c["total"] == pytest.approx(expected)
    r = P.r_multiple(100.0, 95.0, 110.0, 100, rates=rates)
    assert r == pytest.approx((1_000 - expected) / 500)


def test_exit_line_only_ratchets_up():
    s = pd.Series([100.0, 110.0])
    a = pd.Series([5.0, 5.0])
    assert P.exit_line(s, a) == pytest.approx(95.0)
    assert P.exit_line(s, a, previous=99.0) == pytest.approx(99.0)
    assert P.exit_line(s, a, previous=90.0) == pytest.approx(95.0)


def test_kite_holdings_csv_and_journal(tmp_path, monkeypatch):
    from niveshrl import portfolio as pf
    monkeypatch.setattr(pf, "HOLDINGS", tmp_path / "h.json")
    monkeypatch.setattr(pf, "JOURNAL", tmp_path / "j.json")
    csv = ("Holdings statement\n\nInstrument,Qty.,Avg. cost,LTP,Cur. val,P&L\n"
           "TCS,10,3500.5,3600,36000,995\nINFY-BE,5,1,500.00,1600,8000,500\nTotal,,,,44000,\n")
    hs = pf.import_holdings(csv.replace("1,500.00", "1500.00"))
    assert set(hs) == {"TCS.NS", "INFY.NS"} and hs["TCS.NS"].avg_cost == 3500.5
    t = pf.Trade("TCS.NS", "2026-01-01", 100.0, 95.0, 100)
    pf.add_trade(t)
    pf.close_trade(t.id, 110.0, "2026-01-10")
    st = pf.journal_stats(pf.load_journal())
    assert st["closed"] == 1 and 1.5 < st["avg_r"] < 2.0          # +2R gross, less delivery costs
