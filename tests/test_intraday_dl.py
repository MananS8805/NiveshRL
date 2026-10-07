"""Intraday deep learning: triple-barrier labels (order, costs, time barrier), causal features, TCN walk-forward without
leakage, the neural-linear bandit, conformal coverage, IQL exits on a toy path, mistake tags, skip breakdown, v2
decisions frozen in time order, the swing meta-labeler's walk-forward and the model registry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from niveshrl.config import load_yaml
from niveshrl.intraday.dl import dataset as DS


@pytest.fixture(scope="module")
def cfg():
    c = load_yaml("configs/intraday.yaml")
    c["pool"] = 100_000.0
    return c


def _day(close, start="2026-09-01 09:15", vol=1e5, spread=0.001):
    idx = pd.date_range(start, periods=len(close), freq="5min")
    c = np.asarray(close, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + spread), "low": np.minimum(o, c) * (1 - spread),
                         "close": c, "volume": vol}, index=idx)


def test_barrier_labels_target_stop_and_time(cfg):
    n = 40
    minutes = (pd.date_range("2026-09-01 09:15", periods=n, freq="5min").hour * 60
               + pd.date_range("2026-09-01 09:15", periods=n, freq="5min").minute).to_numpy()
    c = np.full(n, 100.0)
    up = c.copy()
    up[6:] = 103.0                                   # jumps far above a 2R target of a 1-rupee stop
    o = np.r_[up[0], up[:-1]]
    dist = np.full(n, 1.0)
    R, xb, why = DS.barrier_labels(o, np.maximum(o, up), np.minimum(o, up), up, minutes, dist, cfg, +1)
    assert why[3] == 1 and R[3] == pytest.approx(2.0, abs=0.15) and R[3] < 2.0      # target, minus costs
    Rs, _, ws = DS.barrier_labels(o, np.maximum(o, up), np.minimum(o, up), up, minutes, dist, cfg, -1)
    assert ws[3] == -1 and Rs[3] < -1.0                                                # short stopped (gap: worse than −1R)
    flat = np.full(n, 100.0)
    Rf, xf, wf = DS.barrier_labels(flat, flat * 1.0001, flat * 0.9999, flat, minutes, dist, cfg, +1)
    assert wf[0] == 0 and xf[0] == 1 + DS.H and -0.3 < Rf[0] < 0                     # time barrier: costs only


def test_features_are_causal(cfg):
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 70)))
    cx = {"prev_close": 100.0, "adr": 0.02, "atr": 2.0, "nr7": 0.0, "avg_volume": 75e5}
    a = DS.stock_day(_day(c), cx, None, cfg)
    c2 = c.copy()
    c2[50:] *= 1.05                                   # change the future only
    b = DS.stock_day(_day(c2), cx, None, cfg)
    np.testing.assert_allclose(a.seq[:50], b.seq[:50], rtol=1e-5)
    np.testing.assert_allclose(a.ctx[:49, :7], b.ctx[:49, :7], rtol=1e-5)
    w = DS.windows(a.seq, np.array([0, 30]))
    assert w.shape == (2, DS.W, len(DS.CHANNELS)) and (w[0, :-1] == 0).all()        # zero padding before the open
    assert np.isfinite(a.dist).any() and (a.ctx[:, 8][np.isfinite(a.dist)] <= cfg["guardrails"]["cost_r_max"] + 1e-6).all()


def _blocks(cfg, n_days=12, n_st=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_days):
        day = pd.Timestamp("2026-08-03") + pd.tseries.offsets.BDay(d)
        for s in range(n_st):
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 70)))
            b = _day(c, start=f"{day.date()} 09:15")
            b.attrs["ticker"] = f"S{s}"
            blk = DS.stock_day(b, {"prev_close": 100.0, "adr": 0.025, "atr": 2.5, "nr7": 0.0, "avg_volume": 75e5}, None, cfg)
            out.append(blk)
    return out


def test_tcn_walk_forward_does_not_leak(cfg):
    from niveshrl.intraday.dl import tcn as T
    bl = _blocks(cfg)
    days = sorted({b.day for b in bl})
    a = T.walk_forward(bl, first_test=6, refit_every=3, verbose=False, max_n=4000, epochs=1, val_days=2)
    for b in bl:                                      # scramble every label from day 9 on
        if b.day >= days[9]:
            b.rL[:] = np.random.default_rng(1).permutation(b.rL)
            b.rS[:] = np.random.default_rng(2).permutation(b.rS)
    c = T.walk_forward(bl, first_test=6, refit_every=3, verbose=False, max_n=4000, epochs=1, val_days=2)
    early = pd.to_datetime(a["ts"]).dt.normalize() < days[9]
    np.testing.assert_allclose(a.loc[early, "eL"], c.loc[early, "eL"], rtol=1e-5)
    assert {"pL", "pS", "eL", "eS", "rL", "rS"} <= set(a.columns)


def test_neural_bandit_learns_and_forgets():
    from niveshrl.intraday.dl.policies import NeuralLinearTS
    df = pd.DataFrame({"eL": np.linspace(-1, 1, 200), "eS": 0.0, "pL": 0.5, "pS": 0.5})
    X = NeuralLinearTS.features(df, "L")
    b = NeuralLinearTS(X.shape[1], half_life_days=None, seed=0)
    b.update(X, 0.8 * df["eL"].to_numpy())           # realised R rises with the model's prediction
    m = b.mean_r(X)
    assert m[-1] > 0.5 and m[0] < -0.5
    f = NeuralLinearTS(X.shape[1], half_life_days=1, seed=0)
    f.update(X, np.ones(200))
    start = abs(f.mean_r(X)).max()
    for _ in range(25):
        f.end_of_day()
    assert abs(f.mean_r(X)).max() < 0.05 * start and f.n < 1e-5           # old evidence has faded to the prior


def test_conformal_margin_covers():
    from niveshrl.intraday.dl.policies import ConformalGate
    rng = np.random.default_rng(0)
    g = ConformalGate(q=0.8, window_days=5)
    assert not g.allow(np.array([5.0])).any()                             # no history: abstain
    for _ in range(5):
        pred = rng.normal(0, 1, 500)
        g.update(pred, pred - rng.normal(0.3, 1, 500))                    # the model over-predicts by 0.3R
    err = rng.normal(0.3, 1, 5000)
    assert np.mean(err <= g.margin()) == pytest.approx(0.8, abs=0.03)


def test_iql_learns_to_exit_before_reversals():
    from niveshrl.intraday.dl.policies import IQLExit
    rows = []
    rng = np.random.default_rng(0)
    for t in range(600):                                                  # up to +1R at bar 3, then back to −1R
        path = [0.3, 0.7, 1.0, 0.4, -0.2, -1.0]
        for j, r in enumerate(path, 1):
            rows.append({"trade": t, "j": j, "r_now": r + rng.normal(0, 0.02), "held": j / 6, "vwap_dist": 0.0,
                         "ret1": (r - (path[j - 2] if j > 1 else 0)), "ret3": 0.0, "to_close": 0.5, "side": 1.0,
                         "done": j == len(path)})
    p = pd.DataFrame(rows)
    q = IQLExit(iters=6).fit(p)
    ev = q.evaluate(p)
    assert ev["policy avg R"] > ev["fixed avg R"] + 1.0                   # exits near the top instead of riding to −1R


def test_mistake_tags_and_skip_breakdown():
    from niveshrl.intraday import mistakes as M
    tags = M.classify_loss({"r": -1.1, "trend_aligned": -1, "minute": 300, "gap": -0.01, "cost_r": 0.3,
                            "entry_ts": "x", "exit_ts": "x", "reason": "stop"})
    assert {"against NIFTY's trend", "late in the day", "faded a gap", "costs ate it", "stopped in the first bar"} <= set(tags)
    assert M.classify_loss({"r": 0.5}) == []
    log = [{"ticker": "A", "ts": "t1", "event": "entry", "action": "TAKE", "why": "x"},
           {"ticker": "B", "ts": "t2", "event": "skip", "action": "SKIP", "why": "costs would eat 0.5R"},
           {"ticker": "C", "ts": "t3", "event": "skip", "action": "SKIP", "why": "ML: P(profit) 30% < 40%"}]
    sh = pd.DataFrame({"ticker": ["A", "B", "C"], "ts": ["t1", "t2", "t3"], "net_r": [1.0, -1.0, -0.5]})
    t = M.skip_breakdown(log, sh)
    assert t.loc["cost rule (stop too tight for costs)", "would-be total R"] == -1.0
    assert t["signals"].sum() == 3


def test_v2_decisions_are_frozen_and_causal(cfg, tmp_path):
    from niveshrl.intraday.engine_v2 import EngineV2
    eng = EngineV2(cfg, root=tmp_path)
    ts = pd.date_range("2026-09-01 10:00", periods=4, freq="5min")
    cand = pd.DataFrame({"ticker": ["A", "B", "A", "C"], "ts": ts, "bar": [9, 10, 11, 12], "side": ["L", "S", "L", "L"],
                         "e": [0.3, -0.1, 0.5, 0.2], "p": 0.5, "close": 100.0, "dist": 0.8})
    frozen = {}
    s1 = eng.decide(cand, frozen)
    assert [s.ticker for s in s1] == ["A", "A", "C"] and s1[0].stop == pytest.approx(99.2)   # threshold 0 at start
    cand2 = cand.assign(e=-1.0)                                           # later predictions cannot undo a decision
    assert [s.ticker for s in eng.decide(cand2, frozen)] == ["A", "A", "C"]


def test_swing_meta_walk_forward_uses_only_closed_trades():
    from niveshrl.research import swing_meta as SM
    rng = np.random.default_rng(0)
    n = 3000
    sig = pd.to_datetime("2015-01-05") + pd.to_timedelta(rng.integers(0, 365 * 4, n), "D")
    hist = pd.DataFrame({"signal_date": sig, "exit_date": sig + pd.to_timedelta(rng.integers(5, 60, n), "D"),
                         "r": rng.normal(0, 1, n), "group": "monitor list"})
    F = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=list("abcd"))
    a = SM.walk_forward(hist, F, first_year=2018)
    late = hist["exit_date"] >= "2018-01-01"
    h2 = hist.copy()
    h2.loc[late, "r"] = rng.normal(5, 1, int(late.sum()))                # outcomes not known by 1 Jan 2018
    b = SM.walk_forward(h2, F, first_year=2018)
    y18 = (hist["signal_date"].dt.year == 2018).to_numpy()
    np.testing.assert_allclose(a.loc[y18, "pred_lgbm"], b.loc[y18, "pred_lgbm"])


def test_model_registry_complete():
    from niveshrl import model_registry as MR
    keys = [c.key for c in MR.CARDS]
    assert len(keys) == len(set(keys)) and len(keys) >= 20
    for c in MR.CARDS:
        assert c.status in {"live", "shadow", "research", "rejected"} and c.why and c.job and c.architecture
        assert isinstance(MR.record(c), str)
    md = MR.markdown()
    assert all(c.name in md for c in MR.CARDS)
