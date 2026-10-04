"""Custom alert rules (each condition, once vs daily, missing data never fires) and paper trading (fills, limits,
bracket stop through a gap, costs, rejections, persistence)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from niveshrl import custom_alerts as CA
from niveshrl import paper as PP

ZERO = {"brokerage_rate": 0.0, "brokerage_cap": 0, "stt_rate": 0.0, "exchange_rate": 0.0, "sebi_rate": 0.0,
        "stamp_rate": 0.0, "dp_charge": 0.0, "half_spread": 0.0, "gst_rate": 0.0}


def _snap(price=100.0, closes=None, **kw):
    closes = closes if closes is not None else pd.Series(np.linspace(80, 99, 260))
    return {"price": price, "closes": closes, **kw}


def test_each_condition():
    assert CA.triggered(CA.Rule("X", "price", "below", 101), _snap())
    assert not CA.triggered(CA.Rule("X", "price", "above", 101), _snap())
    assert CA.triggered(CA.Rule("X", "day_pct", "below", -3), _snap(day_change=-0.035))
    assert CA.triggered(CA.Rule("X", "sma", "above", 50), _snap())               # rising series: above its average
    assert CA.triggered(CA.Rule("X", "rsi", "above", 70), _snap())               # steady rise: RSI 100
    assert CA.triggered(CA.Rule("X", "vol_ratio", "above", 2), _snap(volume=3e6, avg_volume=1e6))
    assert CA.triggered(CA.Rule("X", "high52", "above"), _snap(price=100.0))     # above every previous close (max 99)
    assert not CA.triggered(CA.Rule("X", "high52", "below"), _snap(price=100.0))
    # missing inputs never fire
    assert not CA.triggered(CA.Rule("X", "day_pct", "below", -3), _snap())
    assert not CA.triggered(CA.Rule("X", "vol_ratio", "above", 2), _snap())
    assert not CA.triggered(CA.Rule("X", "rsi", "above", 70), {"price": None})


def test_once_vs_daily_and_persistence(tmp_path):
    p = tmp_path / "alerts.json"
    CA.add(CA.Rule("X.NS", "price", "below", 101, repeat="once"), p)
    CA.add(CA.Rule("X.NS", "price", "below", 101, repeat="daily"), p)
    rules = CA.load(p)
    snap = lambda t: _snap()  # noqa: E731
    d1 = datetime(2026, 10, 5, 10, 0)
    fired, changed = CA.check(rules, snap, d1)
    assert len(fired) == 2 and changed and "X price ≤ ₹101.00" in fired[0][1]
    assert CA.check(rules, snap, d1.replace(hour=11))[0] == []                 # once: off; daily: already today
    assert len(CA.check(rules, snap, datetime(2026, 10, 6, 10))[0]) == 1        # daily fires again next day
    CA.save(rules, p)
    assert [r.active for r in CA.load(p)] == [False, True]
    with pytest.raises(ValueError):
        CA.add(CA.Rule("X.NS", "nonsense", "above", 1), p)
    CA.log([("X.NS", "hello")], tmp_path / "log.json")
    assert CA.load_log(tmp_path / "log.json")[0]["text"] == "hello"


def test_paper_market_buy_and_bracket_stop_through_gap():
    a = PP.Account(100_000, 100_000)
    PP.place(a, PP.Order("X.NS", "BUY", 10, stop=95.0, target=120.0))
    assert PP.process(a, {}, rates=ZERO) == []                                  # market closed: waits
    ev = PP.process(a, {"X.NS": 100.0}, rates=ZERO)
    assert "Filled BUY 10 X at ₹100.00" in ev[0] and a.cash == pytest.approx(99_000)
    pos = a.positions["X.NS"]
    assert pos.qty == 10 and pos.risk_per_share == pytest.approx(5.0)
    ev = PP.process(a, {"X.NS": 90.0}, rates=ZERO)                              # gapped through the 95 stop
    assert "Stop hit" in ev[0] and not a.positions
    c = a.closed[-1]
    assert c.exit == pytest.approx(90.0) and c.net == pytest.approx(-100) and c.r == pytest.approx(-2.0)
    assert PP.summary(a, {})["equity"] == pytest.approx(99_900)


def test_paper_limits_costs_and_rejections():
    a = PP.Account(10_000, 10_000)
    assert PP.place(a, PP.Order("X.NS", "SELL", 1)).status == "REJECTED"         # nothing to sell
    assert PP.place(a, PP.Order("X.NS", "BUY", 1, type="LIMIT")).status == "REJECTED"
    o = PP.place(a, PP.Order("X.NS", "BUY", 50, type="LIMIT", limit=100.0, target=110.0))
    PP.process(a, {"X.NS": 101.0})
    assert o.status == "OPEN"                                                   # above the limit: no fill
    PP.process(a, {"X.NS": 99.0})
    assert o.status == "FILLED" and o.fill_price == pytest.approx(99.0)         # gapped below: filled better
    big = PP.place(a, PP.Order("Y.NS", "BUY", 1000))
    PP.process(a, {"Y.NS": 100.0})
    assert big.status == "REJECTED" and "cash" in big.note
    PP.process(a, {"X.NS": 112.0})                                              # target
    c = a.closed[-1]
    assert c.reason == "target" and c.costs > 0 and c.net < (112.0 - 99.0) * 50


def test_paper_persistence(tmp_path):
    p = tmp_path / "paper.json"
    a = PP.reset(50_000, p)
    PP.place(a, PP.Order("X.NS", "BUY", 5))
    PP.process(a, {"X.NS": 200.0}, rates=ZERO)
    PP.save(a, p)
    b = PP.load(p)
    assert b.positions["X.NS"].qty == 5 and b.cash == pytest.approx(49_000) and b.orders[0].status == "FILLED"
