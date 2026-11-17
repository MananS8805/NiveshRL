import numpy as np
import pandas as pd
import pytest

from niveshrl.intraday.algo import research as R
from niveshrl.intraday.algo import tournament as T


def _day(n_stocks=12, n=75, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-02 09:15", periods=n, freq="5min")
    bars, ctx = {}, {}
    for s in range(n_stocks):
        c = 1000 * np.exp(np.cumsum(drift + 0.003 * rng.standard_normal(n)))
        o = np.r_[c[0], c[:-1]] * (1 + 0.0005 * rng.standard_normal(n))
        h, l = np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999
        v = rng.integers(5_000, 50_000, n).astype(float) * (1 + 4 * (rng.random(n) < 0.05))
        t = f"S{s}.NS"
        bars[t] = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
        ctx[t] = {"avg_volume": 20_000 * 75.0, "adr": 0.02}
    return bars, ctx


def _cfg(max_open=4, stop=0.02, rule=None):
    from niveshrl.intraday.copilot import config
    cfg = config()
    cfg["algo"] = cfg["algo"] | {"max_open": max_open, "daily_loss_stop": stop, "rule": rule}
    return cfg


@pytest.mark.parametrize("fam,p", R.grid()[::7])
def test_signals_have_no_lookahead(fam, p):
    bars, ctx = _day()
    tick = list(bars)
    full = R.matrices_from_dict(bars, tick)
    bv = np.full(len(tick), 20_000.0)
    adr = np.full(len(tick), 0.02)
    s_full = R.signals(full, fam, p | {"_bar_vol": bv, "_adr": adr})
    for cut in (20, 40):
        part = R.matrices_from_dict({t: b.iloc[:cut] for t, b in bars.items()}, tick)
        s_part = R.signals(part, fam, p | {"_bar_vol": bv, "_adr": adr})
        assert (s_part[:cut] == s_full[:cut]).all()


def _book(tmp_path, monkeypatch, **kw):
    from niveshrl.intraday.algo import book as B
    monkeypatch.setattr(B, "ADIR", tmp_path)
    monkeypatch.setattr(B, "TOURN", tmp_path / "tournament.parquet")
    bars, ctx = _day(seed=kw.pop("seed", 0))
    cfg = _cfg(**kw)
    u = pd.DataFrame({"close": 1000.0, "turnover_cr": np.arange(len(bars), 0, -1)}, index=list(bars))
    return B, B.AlgoBook(cfg, u, "2026-11-02"), bars, ctx, cfg


def test_book_freezes_decisions_and_respects_limits(tmp_path, monkeypatch):
    B, bk, bars, ctx, cfg = _book(tmp_path, monkeypatch, max_open=3, stop=1.0, rule="xs_mom look=1 every=3 k=3 hold=12")
    part = {t: b.iloc[:40] for t, b in bars.items()}
    c1, o1, _, _ = bk.run(part, ctx)
    early = {k: dict(v) for k, v in bk.frozen["entries"].items()}
    c2, o2, _, _ = bk.run(bars, ctx, close_at_end=True)
    for k, v in early.items():                                   # a decision once made never changes
        assert bk.frozen["entries"][k] == v
    # at no bar more than max_open positions are held
    allp = c2 + o2
    for b in range(R.OPEN_BAR, R.SQUARE):
        assert sum(1 for p in allp if p["entry_bar"] <= b < p["exit_bar"]) <= 3
    for t in c2:
        assert t["net"] == pytest.approx(t["gross"] - t["costs"])
        assert t["costs"] > 0 and t["entry"] * t["qty"] <= cfg["pool"] * cfg["algo"]["leverage"] / 3 + 1e-6
    assert any(not v["taken"] for v in bk.frozen["entries"].values())   # the cap skipped some signals


def test_loss_stop_halts_new_entries(tmp_path, monkeypatch):
    B, bk, bars, ctx, cfg = _book(tmp_path, monkeypatch, max_open=12, stop=0.0001, rule="xs_mom look=1 every=3 k=3 hold=3")
    closed, _, log, _ = bk.run(bars, ctx, close_at_end=True)
    assert any(x["event"] == "loss stop" for x in log)
    halt = next(x["bar"] for x in log if x["event"] == "loss stop")
    assert all(t["entry_bar"] <= halt for t in closed)


def test_settle_scores_every_rule_and_picks_tomorrows(tmp_path, monkeypatch):
    B, bk, bars, ctx, cfg = _book(tmp_path, monkeypatch)
    rec = bk.settle(bars, ctx, "2026-11-02T15:21:00")
    tab = pd.read_parquet(tmp_path / "tournament.parquet")
    assert set(tab["source"]) == {"live"} and tab["name"].nunique() > 10
    assert rec["review"]["learnings"][-1].startswith("Tomorrow's rule:")
    assert (tmp_path / "reviews" / "2026-11-02.json").exists()


def test_leader_uses_only_earlier_days():
    rows = []
    for d in range(30):
        day = f"2026-10-{d + 1:02d}"
        rows.append({"day": day, "name": "A", "trades": 60, "net": 10.0})
        rows.append({"day": day, "name": "B", "trades": 60, "net": 1e6 if d == 29 else -5.0})
        rows.append({"day": day, "name": "C", "trades": 5, "net": 999.0})        # too few trades: never eligible
    tab = pd.DataFrame(rows)
    assert T.leader(tab, "2026-10-30", 20, 50) == "A"          # B's huge day 30 is not visible yet
    assert T.leader(tab, "2026-10-31", 20, 50) == "B"
    assert T.leader(tab.iloc[:9], "2026-10-04", 20, 50) is None


def test_cost_per_trade_is_about_fourteen_to_seventeen_bps():
    from niveshrl.intraday.copilot import config
    c = config()["costs"]
    assert 13 < R.cost_bps(100_000, c) < 15.5 and 15.5 < R.cost_bps(41_667, c) < 18


def test_leader_skips_rules_the_book_cannot_carry():
    rows = [{"day": f"2026-10-{d + 1:02d}", "name": n, "trades": 70, "net": v}
            for d in range(20) for n, v in (("rvol_break m=2.0 hold=60", 100.0), ("xs_mom look=3 every=3 k=2 hold=12", 1.0))]
    tab = pd.DataFrame(rows)
    assert T.leader(tab, "2026-10-25", 20, 50) == "rvol_break m=2.0 hold=60"
    assert T.leader(tab, "2026-10-25", 20, 50, max_open=12) == "xs_mom look=3 every=3 k=2 hold=12"
