import numpy as np
import pandas as pd
import pytest

from niveshrl.research import swing_book as SB
from niveshrl.research.data import Panel


def _panel(n=120, k=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = [f"S{i}.NS" for i in range(k)]
    c = pd.DataFrame(100 * np.exp(np.cumsum(0.002 + 0.015 * rng.standard_normal((n, k)), axis=0)), index=idx, columns=cols)
    o = c.shift(1).fillna(c) * (1 + 0.003 * rng.standard_normal((n, k)))
    h = np.maximum(o, c) * (1 + 0.006 * rng.random((n, k)))
    l = np.minimum(o, c) * (1 - 0.006 * rng.random((n, k)))
    v = pd.DataFrame(1e6, index=idx, columns=cols)
    bench = pd.Series(np.linspace(100, 120, n), index=idx)
    return Panel(close=c, volume=v, sectors=pd.Series("X", index=cols), names=pd.Series(cols, index=cols), bench=bench,
                 vix=pd.Series(15.0, index=idx), open=o, high=h, low=l)


def test_engine_matches_the_record_rules_trade_by_trade():
    from niveshrl.research.forward import follow
    from niveshrl.research.technicals import atr
    p = _panel()
    a = atr(p)
    picks = {p.close.index[40]: list(p.close.columns)}
    b = SB.replay(p, picks, str(p.close.index[30].date()), capital=1e9, max_positions=100, max_position_pct=1e-3)
    for t in b.s["trades"]:
        oc = follow(p, t["ticker"], pd.Timestamp(t["signal_date"]), a, qty=t["qty"])
        if oc.status == "closed":
            assert t["exit_date"] == str(pd.Timestamp(oc.exit_date).date())
            assert t["r"] == pytest.approx(oc.r, abs=0.05)              # costs are charged per sale vs once


def test_cash_positions_and_costs_are_respected():
    p = _panel(seed=1)
    picks = {d: list(p.close.columns) for d in p.close.index[35:100:5]}
    b = SB.replay(p, picks, str(p.close.index[30].date()), capital=100_000, max_positions=3)
    eq = pd.DataFrame(b.s["equity"])
    assert (eq["cash"] >= -1e-6).all() and (eq["positions"] <= 3).all()
    for t in b.s["trades"]:
        assert t["qty"] > 0 and np.isfinite(t["net"])
    assert any(x["event"] == "skip" for x in b.s["log"])
    # every rupee is accounted for: final equity = capital + realized net of closed trades + open positions' P&L
    open_pl = sum((pos["last"] - pos["entry"]) * pos["qty"] + pos["realized"] for pos in b.s["positions"].values())
    assert b.equity() == pytest.approx(100_000 + sum(t["net"] for t in b.s["trades"]) + open_pl, rel=1e-6)


def test_stop_is_checked_before_target_on_the_same_bar():
    p = _panel(n=60, k=1)
    t = p.close.columns[0]
    d0, d1, d2 = p.close.index[40], p.close.index[41], p.close.index[42]
    b = SB.SwingBook(capital=100_000)
    from niveshrl.research.technicals import atr
    a = atr(p)
    b.run_day(p, d0, a, [t])
    b.run_day(p, d1, a)                                    # filled at d1's open
    pos = b.s["positions"][t]
    p.high.loc[d2, t] = pos["t1"] * 1.01                    # one wide bar touches both
    p.low.loc[d2, t] = pos["stop"] * 0.99
    b.run_day(p, d2, a)
    assert t not in b.s["positions"] and b.s["trades"][-1]["reason"] == "stop"


def test_daily_update_is_idempotent(tmp_path):
    p = _panel()
    d = p.close.index[50]
    r1 = SB.daily_update(p, d, list(p.close.columns[:3]), folder=tmp_path)
    r2 = SB.daily_update(p, d, list(p.close.columns[:3]), folder=tmp_path)
    assert r1["queued"] == list(p.close.columns[:3]) and "already up to date" in r2["note"]
    r3 = SB.daily_update(p, p.close.index[53], [], folder=tmp_path)
    assert r3["days_run"] == 3 and len(r3["open"]) >= 1
