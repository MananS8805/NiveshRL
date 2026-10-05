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


def test_forward_follow_stop_and_t1_rules():
    from niveshrl.research import forward as F
    n = 60
    base = np.full(n, 100.0)
    p = _panel(base, low=base - 1, high=base + 1)
    atr = pd.DataFrame({"X.NS": np.full(n, 2.0)}, index=p.close.index)
    d = p.close.index[39]                                   # plan: stop 4 below entry, T1 = +6
    # path 1: gaps below the stop (opens at 95, stop 96) -> filled at the open: -1.25R, minus ~0.11R delivery costs
    c = base.copy()
    c[40:] = [100, 99, 95, 94] + [94] * (n - 44)
    p1 = _panel(c, low=c - 1, high=c + 1)
    o1 = F.follow(p1, "X.NS", d, atr)
    assert o1.status == "closed" and o1.reason == "stop" and o1.exit == pytest.approx(95.0)
    assert -1.40 < o1.r < -1.25
    # path 2: reaches T1 (books a third); the 3xATR trail lifts the stop to 107 - 6 = 101; exits at 101:
    #         1/3 x 1.5R + 2/3 x (101-100)/4 = +0.667R, minus ~0.11R costs
    c2 = base.copy()
    c2[40:] = [100, 103, 107, 101, 99] + [99] * (n - 45)
    p2 = _panel(c2, low=c2 - 1, high=c2 + 1)
    o2 = F.follow(p2, "X.NS", d, atr)
    assert o2.status == "closed" and o2.reason == "trailing stop" and o2.exit == pytest.approx(101.0)
    assert 0.5 < o2.r < 0.6
    # the plan is computed as of the signal date: later bars cannot change the stop
    assert F.follow(p2, "X.NS", d, atr).entry == pytest.approx(o2.entry)


def test_edge_statistic():
    from niveshrl.research import forward as F
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"group": ["monitor list"] * 400 + ["random control"] * 400, "status": "closed",
                       "r": np.r_[rng.normal(0.3, 1, 400), rng.normal(0.0, 1, 400)]})
    e = F.edge(df)
    assert 0.15 < e["edge R"] < 0.45 and e["t-stat"] > 2


def test_plan_entry_uses_live_price_only_while_market_open(monkeypatch):
    """The desk / stock-page plan must not stay frozen at yesterday's close while the market trades."""
    from niveshrl import livefeed
    from niveshrl.desktop import data

    class Feed:
        def quote(self, t):
            return {"price": 2100.0, "ts": 1_791_250_000_000}           # milliseconds, as the stream sends them
    t = data.panel().tickers[0]
    monkeypatch.setattr(livefeed, "market_open", lambda *a, **k: True)
    e, lab = data.plan_entry(Feed(), t)
    assert e == 2100.0 and lab.startswith("live ")
    monkeypatch.setattr(livefeed, "market_open", lambda *a, **k: False)
    e2, lab2 = data.plan_entry(Feed(), t)
    assert e2 == float(data.panel().close[t].dropna().iloc[-1]) and lab2.startswith("last close")
    from niveshrl.research.plans import plans_for
    df = plans_for(data.panel(), [t], entries={t: 2100.0})
    assert df.loc[t, "entry"] == 2100.0
