import numpy as np
import pandas as pd
import pytest

from niveshrl.intraday.copilot import config
from niveshrl.intraday.copilot import setups as CS
from niveshrl.intraday.copilot.engine import choose_day, passes
from niveshrl.intraday.copilot.tuner import tune
from niveshrl.research import robust as RB


def _bars(seed=0, n=75, drift=0.0004, start="2026-11-02 09:15"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min")
    c = 1000 * np.exp(np.cumsum(drift + 0.002 * rng.standard_normal(n)))
    o = np.r_[c[0] * 0.999, c[:-1]]
    h = np.maximum(o, c) * (1 + 0.001 * rng.random(n))
    l = np.minimum(o, c) * (1 - 0.001 * rng.random(n))
    v = rng.integers(5_000, 50_000, n).astype(float)
    v[5:12] *= 3
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


CTX = {"prev_close": 995.0, "prev_high": 1005.0, "prev_low": 985.0, "adr": 0.02, "atr": 15.0, "avg_volume": 1.5e6, "nr7": 0}


def test_setups_have_no_lookahead():
    cfg = config()
    full = _bars(1)
    nifty = _bars(2, drift=0.0002)
    sigs_full = CS.day_signals("X.NS", full, CTX, cfg["guardrails"], nifty)
    for k in range(5, 70, 7):
        trunc = CS.day_signals("X.NS", full.iloc[: k + 1], CTX, cfg["guardrails"], nifty.iloc[: k + 1])
        early = [(s.bar, s.setup, s.side, round(s.stop, 6)) for s in sigs_full if s.bar <= k]
        assert [(s.bar, s.setup, s.side, round(s.stop, 6)) for s in trunc] == early


def test_setup_stops_respect_bounds():
    cfg = config()
    g = cfg["guardrails"]
    for seed in range(6):
        for s in CS.day_signals("X.NS", _bars(seed, drift=0.001 * (seed % 3 - 1)), CTX, g, _bars(seed + 9)):
            d = abs(s.ref - s.stop) / s.ref
            assert g["min_stop_pct"] - 1e-9 <= d <= g["max_stop_pct"] + 1e-9
            assert (s.stop < s.ref) == (s.side == "long")


def _cards(days, setup="orb15", mean=-0.3, per_day=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(days):
        d = f"2026-10-{i + 1:02d}"
        for j in range(per_day):
            rows.append({"day": d, "setup": setup, "minute": 30 + j, "p": 0.5, "r": mean + 0.3 * rng.standard_normal()})
    return pd.DataFrame(rows)


def test_tuner_needs_many_days_before_benching():
    cfg = config()
    rules = {"setups": {"orb15": {"enabled": True, "min_p": 0.45}}, "last_entry": "14:30", "log": []}
    few = _cards(3, per_day=15)                                   # 45 bad cards, but only 3 days
    new, ch = tune(rules, few, few["day"].max(), cfg["tuning"])
    assert new["setups"]["orb15"]["enabled"] and not ch
    many = _cards(12, per_day=4)                                  # 48 bad cards over 12 days
    new, ch = tune(rules, many, many["day"].max(), cfg["tuning"])
    assert not new["setups"]["orb15"]["enabled"]
    assert ch and ch[0]["rule"] == "orb15.enabled" and ch[0]["n"] >= 30


def test_tuner_never_touches_guardrails():
    cfg = config()
    rules = {"setups": {"orb15": {"enabled": True, "min_p": 0.45}}, "last_entry": "14:30", "log": []}
    new, _ = tune(rules, _cards(15, mean=-0.5), "2026-10-15", cfg["tuning"])
    assert set(new) <= {"setups", "last_entry", "log"}
    assert set(new["setups"]["orb15"]) <= {"enabled", "min_p"}


def test_choose_day_caps_trades_and_stocks():
    g = config()["guardrails"]
    rules = {"setups": {s: {"enabled": True, "min_p": 0.0} for s in CS.SETUPS}, "last_entry": "14:30"}
    cards = pd.DataFrame([{"ticker": t, "setup": "orb15", "side": "long", "bar": b, "minute": 30 + 5 * b, "p": 0.6,
                           "r": -1.0, "exit_bar": b + 1} for b, t in enumerate(["A", "A", "B", "C", "D", "E"])])
    taken = choose_day(cards, rules, g)
    assert len(taken) <= g["max_trades_per_day"]
    assert cards.loc[taken, "ticker"].is_unique
    rules["setups"]["orb15"]["enabled"] = False
    assert choose_day(cards, rules, g) == []
    ok, why = passes(cards.iloc[0], rules)
    assert not ok and "benched" in why


def test_bootstrap_and_deflated_sharpe():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 500, 120)
    b = RB.bootstrap_mean(x, reps=1000)
    assert b["lo90"] < b["mean"] < b["hi90"]
    noise_sr = [RB.sharpe(rng.normal(0, 1, 120)) for _ in range(50)]
    one = RB.deflated_sharpe(x / 1e5, noise_sr[:2], n_trials=2)["dsr"]
    many = RB.deflated_sharpe(x / 1e5, noise_sr, n_trials=50)["dsr"]
    assert many <= one                                        # more variants tried -> harder to look skilful


def test_pbo_flags_pure_noise():
    vals = [RB.pbo(np.random.default_rng(s).normal(0, 1, (200, 12)))["pbo"] for s in range(6)]
    assert 0.3 < np.mean(vals) < 0.7                          # best-of-noise ranks below median OOS about half the time
    M = np.random.default_rng(1).normal(0, 1, (200, 12))
    M[:, 0] += 0.5                                            # one genuinely better variant
    assert RB.pbo(M)["pbo"] < 0.1


def test_verdict_labels():
    good = {"trades": 80, "holdout_net": 5, "boot": {"lo90": 1}, "dsr": 0.97, "pbo": 0.2, "random_p": 0.01, "net_2x": 3}
    assert RB.verdict(good)["label"] == "MONEY-READY"
    bad = dict(good, holdout_net=-1, net_2x=-1, random_p=0.4)
    assert RB.verdict(bad)["label"] == "NOT READY"


def test_engine_handles_a_day_without_setups(tmp_path, monkeypatch):
    import niveshrl.intraday.copilot as CP
    from niveshrl.intraday.copilot import engine as E
    monkeypatch.setattr(CP, "CDIR", tmp_path)
    monkeypatch.setattr(E, "MODEL", tmp_path / "model.pkl")
    cp = E.Copilot()
    bars = {"S0.NS": _bars(0).iloc[:3]}                       # 09:30: too early for any setup
    cards, trades, log, acct = cp.run(bars, {"S0.NS": CTX}, _bars(99).iloc[:3], {}, "2026-11-02")
    assert len(cards) == 0 and trades == [] and len(cp.top_cards(cards)) == 0


@pytest.mark.parametrize("close_at_end", [False, True])
def test_engine_run_freezes_decisions(tmp_path, monkeypatch, close_at_end):
    import niveshrl.intraday.copilot as CP
    from niveshrl.intraday.copilot import engine as E
    monkeypatch.setattr(CP, "CDIR", tmp_path)
    monkeypatch.setattr(E, "MODEL", tmp_path / "model.pkl")
    cp = E.Copilot()
    cp.rules = {"setups": {s: {"enabled": True, "min_p": 0.0} for s in CS.SETUPS}, "last_entry": "14:30", "log": []}
    bars = {f"S{i}.NS": _bars(i, drift=0.001 * (1 if i % 2 else -1)) for i in range(8)}
    ctx = {t: CTX for t in bars}
    frozen = {}
    cards, trades, log, acct = cp.run(bars, ctx, _bars(99), frozen, "2026-11-02", close_at_end=close_at_end)
    assert sum(v["take"] for v in frozen.values()) <= cp.g["max_trades_per_day"]
    before = dict(frozen)
    cp.run(bars, ctx, _bars(99), frozen, "2026-11-02", close_at_end=close_at_end)
    assert frozen == before                                   # a decision never changes after it is made
    if close_at_end:
        assert not acct.open_positions
