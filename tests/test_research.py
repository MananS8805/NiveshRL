"""Research platform: features, walk-forward leakage, backtester, data hygiene, regimes."""
import numpy as np
import pandas as pd
import pytest

from niveshrl.research import backtest as bt
from niveshrl.research.data import Panel, adjust_corporate_actions
from niveshrl.research.features import FEATURES, build_rank_data
from niveshrl.research.rankers import TrainConfig, walk_forward


def synthetic_panel(n_stocks=40, years=7, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=252 * years)
    tick = [f"S{i:02d}.NS" for i in range(n_stocks)]
    mom = rng.normal(0, 0.0004, n_stocks)
    r = mom + 0.018 * rng.standard_normal((len(idx), n_stocks))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=tick)
    close.iloc[:400, -5:] = np.nan                         # five late listings
    vol = pd.DataFrame(1e6, index=idx, columns=tick)
    bench = close.mean(axis=1).ffill()
    return Panel(close=close, volume=vol, sectors=pd.Series(["A", "B"] * (n_stocks // 2), index=tick),
                 names=pd.Series(tick, index=tick), bench=bench, vix=pd.Series(15.0, index=idx))


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel()


@pytest.fixture(scope="module")
def rank(panel):
    return build_rank_data(panel)


def test_rank_features_shape_and_labels(rank):
    f = rank.frame
    assert len(FEATURES) == 33
    assert f[FEATURES].notna().all().all()
    lab = f["label"].dropna()
    assert 0.45 < lab.mean() < 0.55
    # z-scored cross-sectionally: each month's monthly features have ~0 mean
    assert f.groupby(level=0)["m12"].mean().abs().max() < 1e-6


def test_rank_features_no_lookahead(panel, rank):
    cut = panel.close.index[1200]
    short = Panel(close=panel.close.loc[:cut], volume=panel.volume.loc[:cut], sectors=panel.sectors,
                  names=panel.names, bench=panel.bench.loc[:cut], vix=panel.vix.loc[:cut])
    a = build_rank_data(short).frame
    common = a.index.get_level_values(0).unique()[:-2]     # last months differ only in ret_next/label
    b = rank.frame.loc[common, FEATURES]
    pd.testing.assert_frame_equal(a.loc[common, FEATURES], b)


def test_walk_forward_never_sees_future_labels(rank):
    """Scrambling labels/returns in the test year and later must not change that year's predictions."""
    cfg = TrainConfig(window_years=3)
    base = walk_forward(rank, "logreg", first_test_year=2015, cfg=cfg, verbose=False)
    f2 = rank.frame.copy()
    later = f2.index.get_level_values(0).year >= 2015
    rng = np.random.default_rng(1)
    f2.loc[later, "label"] = rng.integers(0, 2, later.sum()).astype(float)
    f2.loc[later, "ret_next"] = rng.normal(size=later.sum())
    scrambled = walk_forward(type(rank)(frame=f2, month_ends=rank.month_ends), "logreg", 2015, cfg, verbose=False)
    y15 = base.index.get_level_values(0).year == 2015
    np.testing.assert_allclose(base.loc[y15, "score"], scrambled.loc[y15, "score"])


def test_capped_weights():
    w = bt.capped_weights(pd.Series([5.0, 1, 1, 1, 1, 1]), 0.25)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.25 + 1e-9
    np.testing.assert_allclose(bt._cap_np(np.array([5.0, 1, 1, 1, 1, 1]), 0.25), w.to_numpy())


def test_zero_cost_equal_weight_matches_analytic(panel):
    scores = bt.factor_scores(panel, "equal")
    spec = bt.StrategySpec(signal="equal", top=1.0, max_weight=1.0, cost_scale=0.0, cash_rate=0.0,
                           min_trade=0.0, start="2012-01-01", end="2012-06-30")
    res = bt.run_backtest(panel, scores, spec)
    # Monthly-rebalanced equal weight: product over months of the average stock growth.
    reb = res.rebalances.index
    c = panel.close
    growth = 1.0
    ends = list(reb[1:]) + [res.nav.index[-1]]
    for d0, d1 in zip(reb, ends):
        elig = c.loc[d0].dropna().index
        growth *= (c.loc[d1, elig] / c.loc[d0, elig]).mean()
    assert res.nav.iloc[-1] == pytest.approx(growth, rel=1e-9)


def test_costs_charged_and_scale(panel):
    scores = bt.factor_scores(panel, "momentum")
    kw = dict(signal="momentum", top=10, start="2012-01-01", end="2013-12-31")
    cheap = bt.run_backtest(panel, scores, bt.StrategySpec(cost_scale=0.0, **kw))
    real = bt.run_backtest(panel, scores, bt.StrategySpec(cost_scale=1.0, **kw))
    assert cheap.rebalances["cost"].sum() == 0
    assert real.rebalances["cost"].sum() > 0
    assert real.nav.iloc[-1] < cheap.nav.iloc[-1]


def test_long_only_weights_and_holdings(panel):
    res = bt.run_backtest(panel, bt.factor_scores(panel, "momentum"),
                          bt.StrategySpec(signal="momentum", top=8, max_weight=0.2, start="2012-01-01", end="2013-12-31"))
    for w in res.weights.values():
        assert (w > 0).all() and w.sum() <= 1.0 + 1e-6 and len(w) <= 8


def test_corporate_action_detection():
    idx = pd.bdate_range("2020-01-01", periods=40)
    split = pd.Series(np.r_[np.full(20, 1000.0), np.full(20, 505.0)], index=idx)        # 1:2 split, +1% move
    crash = pd.Series(np.r_[np.full(20, 100.0), np.full(20, 56.0)], index=idx)           # genuine -44%
    close = pd.DataFrame({"SPLIT.NS": split, "CRASH.NS": crash})
    adj, ev = adjust_corporate_actions(close)
    assert [e[1] for e in ev] == ["SPLIT.NS"]
    assert adj["SPLIT.NS"].pct_change().abs().max() < 0.02
    pd.testing.assert_series_equal(adj["CRASH.NS"], close["CRASH.NS"])


def test_regimes_do_not_change_when_future_is_removed(panel):
    from niveshrl.research.regime import detect_regimes
    full = detect_regimes(panel, first_year=2013)
    cut = pd.Timestamp("2015-06-30")
    short = Panel(close=panel.close.loc[:cut], volume=panel.volume.loc[:cut], sectors=panel.sectors,
                  names=panel.names, bench=panel.bench.loc[:cut], vix=panel.vix.loc[:cut])
    part = detect_regimes(short, first_year=2013)
    common = part.index[part.index < pd.Timestamp("2015-01-01")]
    assert set(full["regime"]) <= {"Bull", "Neutral", "Stress"}
    pd.testing.assert_series_equal(full.loc[common, "regime"], part.loc[common, "regime"])
