"""Intraday paper agent: costs vs a hand-worked contract note, guardrails under an adversarial policy, no look-ahead,
square-off, daily loss stop, bandit learning, tax, and an end-to-end synthetic day."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from niveshrl.intraday.bandit import Bandit
from niveshrl.intraday.costs import fy_of, round_trip, speculative_tax
from niveshrl.intraday.setups import Signal, day_signals
from niveshrl.intraday.sim import SQUARE_OFF_MIN, Account, outcome


@pytest.fixture
def cfg():
    from niveshrl.config import load_yaml
    c = load_yaml("configs/intraday.yaml")             # the shipped defaults, never the user's saved settings
    c["pool"], c["leverage"] = 100_000.0, 1.0
    return c


def _bars(prices, start="2026-07-01 09:15", vol=50_000.0, spread=0.002):
    idx = pd.date_range(start, periods=len(prices), freq="5min")
    p = np.asarray(prices, float)
    o = np.r_[p[0], p[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, p) * (1 + spread), "low": np.minimum(o, p) * (1 - spread),
                         "close": p, "volume": np.full(len(p), vol)}, index=idx)


def test_costs_match_a_hand_worked_contract_note(cfg):
    c = cfg["costs"]
    # buy 100 @ 500, sell 100 @ 510: buy 50,000, sell 51,000
    tc = round_trip(500.0, 510.0, 100, "long", c | {"slippage": 0.0})
    brk = min(20, 0.0003 * 50_000) + min(20, 0.0003 * 51_000)          # 15 + 15.3
    exch = 0.0000297 * 101_000
    sebi = 0.000001 * 101_000
    expected = brk + 0.00025 * 51_000 + exch + sebi + 0.00003 * 50_000 + 0.18 * (brk + exch + sebi)
    assert tc.brokerage == pytest.approx(30.3)
    assert tc.total == pytest.approx(expected)
    big = round_trip(5000.0, 5000.0, 100, "long", c | {"slippage": 0.0})  # 5 lakh per side: brokerage capped at 20
    assert big.brokerage == pytest.approx(40.0)
    short = round_trip(510.0, 500.0, 100, "short", c | {"slippage": 0.0})  # short: sell first at 510, buy back at 500
    assert short.stt == pytest.approx(0.00025 * 51_000) and short.stamp == pytest.approx(0.00003 * 50_000)


def test_speculative_tax_and_financial_year(cfg):
    assert speculative_tax(-5_000, cfg["tax"]) == 0
    assert speculative_tax(10_000, cfg["tax"]) == pytest.approx(10_000 * 0.30 * 1.04)
    assert fy_of(pd.Timestamp("2026-03-31")) == "FY2025-26" and fy_of(pd.Timestamp("2026-04-01")) == "FY2026-27"


def test_signals_have_no_look_ahead(cfg):
    rng = np.random.default_rng(0)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 75)))
    b = _bars(p)
    ctx = {"prev_high": 101.0, "prev_low": 99.0, "prev_close": 100.0, "adr": 0.03, "atr": 2.0, "nr7": 1.0,
           "avg_volume": 75 * 50_000}
    full = day_signals("X.NS", b, ctx, cfg["setups"], cfg["guardrails"])
    for k in (10, 25, 40, 60):
        part = day_signals("X.NS", b.iloc[:k + 1], ctx, cfg["setups"], cfg["guardrails"])
        assert [(s.bar, s.setup, s.side, s.stop) for s in part] == \
               [(s.bar, s.setup, s.side, s.stop) for s in full if s.bar <= k]


def test_fill_is_next_bar_open_and_exit_rules(cfg):
    b = _bars([100, 100.2, 100.4, 101, 102, 103, 104, 105], spread=0.0)
    sig = Signal("X.NS", b.index[2], 2, "orb", "long", 100.4, 99.5)
    oc = outcome(sig, b, cfg)
    assert oc["fill"] == pytest.approx(b["open"].iloc[3])           # next bar's open, not the signal close
    assert oc["reason"] == "target" and oc["target"] == pytest.approx(oc["fill"] + 2 * oc["dist"])
    both = _bars([100, 100, 100, 100, 100], spread=0.0)
    both.iloc[4, both.columns.get_loc("high")] = 110
    both.iloc[4, both.columns.get_loc("low")] = 90                  # one bar touches stop and target
    oc2 = outcome(Signal("X.NS", both.index[2], 2, "orb", "long", 100, 99.0), both, cfg)
    assert oc2["reason"] == "stop"                                  # conservative: stop first


def test_square_off_by_1515(cfg):
    p = np.full(75, 100.0)
    b = _bars(p, spread=0.0005)
    sig = Signal("X.NS", b.index[60], 60, "vwap", "long", 100.0, 99.5)
    trades, _ = Account(cfg, 100_000).run_day([sig], {"X.NS": b}, lambda s: ("TAKE", 0.5, ""))
    assert trades and trades[0].reason == "square-off"
    t = pd.Timestamp(trades[0].exit_ts)
    assert t.hour * 60 + t.minute <= SQUARE_OFF_MIN


def test_guardrails_hold_under_an_adversarial_take_everything_policy(cfg):
    rng = np.random.default_rng(3)
    bars, sigs = {}, []
    for j in range(30):                                            # 30 stocks, a signal on every bar
        p = 100 * np.exp(np.cumsum(rng.normal(-0.002, 0.004, 75)))   # falling market: longs keep losing
        t = f"S{j}.NS"
        bars[t] = _bars(p)
        for i in range(3, 65):
            sigs.append(Signal(t, bars[t].index[i], i, "orb", "long", float(p[i]), float(p[i] * 0.995)))
    g = cfg["guardrails"]
    acct = Account(cfg, 100_000)
    trades, log = acct.run_day(sigs, bars, lambda s: ("TAKE", 0.9, "adversarial"))
    # never more than 5 open at once
    events = []
    for tr in trades:
        events += [(pd.Timestamp(tr.entry_ts), 1), (pd.Timestamp(tr.exit_ts), -1)]
    open_now = peak = 0
    for _, d in sorted(events, key=lambda x: (x[0], x[1])):
        open_now += d
        peak = max(peak, open_now)
    assert peak <= g["max_open_positions"]
    # risk per trade never above 1% of the pool, exposure never above pool x leverage
    for tr in trades:
        assert abs(tr.entry - tr.stop) * tr.qty <= g["risk_per_trade"] * 100_000 * 1.0001
        assert tr.entry * tr.qty <= 100_000 * cfg["leverage"] * 1.0001
    # the daily loss stop fires and no entry happens after it
    stop_bar = next((e["bar"] for e in log if e.get("event") == "daily loss stop"), None)
    if stop_bar is not None:
        late = [e for e in log if e.get("event") == "entry" and e.get("ts") and
                pd.Timestamp(e["ts"]) >= bars["S0.NS"].index[stop_bar]]
        assert not late


def test_bandit_learns_to_skip_a_losing_bucket_and_is_reproducible(cfg):
    rows = [{"setup": "ema_pullback", "side": "long", "minute": 60, "trend_aligned": 1, "ticker": f"T{i}", "ts": str(i),
             "net_r": -0.6 + 0.1 * np.sin(i), "win": 0.0} for i in range(60)]
    rows += [{"setup": "orb", "side": "long", "minute": 20, "trend_aligned": 1, "ticker": f"U{i}", "ts": str(i),
              "net_r": 0.4 + 0.1 * np.cos(i), "win": 1.0} for i in range(60)]
    a, b = Bandit(cfg, seed=7), Bandit(cfg, seed=7)
    for x in (a, b):
        changes = x.update(pd.DataFrame(rows), day="2026-07-01")
    assert any(c["to"] == "SKIP" for c in changes)
    f_bad = {"setup": "ema_pullback", "side": "long", "minute": 60, "trend_aligned": 1}
    f_good = {"setup": "orb", "side": "long", "minute": 20, "trend_aligned": 1}
    assert a.decide(f_bad)[0] == "SKIP" and a.decide(f_good)[0] == "TAKE"
    assert [a.decide(f_good) for _ in range(5)] == [b.decide(f_good) for _ in range(5)]


def test_agent_day_end_to_end_on_synthetic_bars(tmp_path, cfg):
    from niveshrl.intraday.agent import Agent
    rng = np.random.default_rng(5)
    bars, ctx = {}, {}
    for j in range(12):
        p = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.006, 75)))
        t = f"S{j}.NS"
        bars[t] = _bars(p, vol=200_000)
        ctx[t] = {"prev_high": 100.5, "prev_low": 99.0, "prev_close": 99.0, "adr": 0.02, "atr": 2.0, "nr7": 0.0,
                  "avg_volume": 75 * 100_000}
    ag = Agent(tmp_path, cfg)
    res, trades, log = ag.trade_day("2026-07-01", bars, ctx)
    assert res.pool_end == pytest.approx(res.pool_start + res.net)
    assert res.net == pytest.approx(sum(t.net for t in trades), abs=0.05)
    assert (tmp_path / "state.json").exists() and (tmp_path / "bandit.json").exists()
    again = Agent(tmp_path, cfg)                                    # state survives a restart
    assert again.state["pool"] == pytest.approx(res.pool_end)


def test_live_uses_only_finished_bars_and_skips_weekends(tmp_path, monkeypatch):
    from datetime import datetime
    from niveshrl.intraday import live
    b = _bars(np.full(10, 100.0), start="2026-07-01 09:15")
    at = datetime(2026, 7, 1, 9, 47, 30, tzinfo=live.IST)          # the 09:45 bar is still forming
    f = live.finished({"X.NS": b}, at)["X.NS"]
    assert f.index[-1] == pd.Timestamp("2026-07-01 09:40")
    monkeypatch.setattr(live, "DIR", tmp_path)
    monkeypatch.setattr(live, "now_ist", lambda: datetime(2026, 7, 4, 10, 0, tzinfo=live.IST))   # a Saturday
    monkeypatch.setattr(live, "warm_start", lambda: False)
    assert live.run_live()["phase"] == "weekend"


def test_warm_start_copies_learning_not_the_pool(tmp_path, monkeypatch):
    import json
    from niveshrl.intraday import live
    rep = tmp_path / "replay"
    rep.mkdir()
    (rep / "bandit.json").write_text(json.dumps({"stats": {"orb": [30, -3.0, 9.0]}, "log": []}))
    (rep / "state.json").write_text(json.dumps({"pool": 50_000}))
    monkeypatch.setattr(live, "DIR", tmp_path)
    assert live.warm_start() is True
    assert (tmp_path / "bandit.json").exists() and not (tmp_path / "state.json").exists()
    assert live.warm_start() is False                               # only once
