import numpy as np
import pandas as pd
import pytest

from niveshrl.backtest import make_eval_env, run_policy
from niveshrl.baselines import EqualWeight
from niveshrl.constraints import is_feasible
from niveshrl.env import PortfolioEnv
from niveshrl.features import raw_features
from niveshrl.metrics import max_drawdown, sharpe
from niveshrl.profile import InvestorProfile


def test_features_have_no_lookahead(md):
    full_s, full_m = raw_features(md)
    cut = 600
    md_cut = type(md)(dates=md.dates[:cut], tickers=md.tickers, sectors=md.sectors,
                      close=md.close.iloc[:cut], volume=md.volume.iloc[:cut],
                      bench=md.bench.iloc[:cut], vix=md.vix.iloc[:cut], fx=md.fx.iloc[:cut])
    cut_s, cut_m = raw_features(md_cut)
    np.testing.assert_allclose(full_s[:cut], cut_s, equal_nan=True, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(full_m[:cut], cut_m, equal_nan=True, rtol=1e-9, atol=1e-12)


def test_random_policy_respects_invariants(md, fs, cfg):
    prof = InvestorProfile(min_cash=0.1)
    env = PortfolioEnv(md, fs, cfg, split="train", profile=prof, seed=1)
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
        w = env.trades[-1]["weights"]
        assert is_feasible(w, env.sector_ids, cfg["env"]["max_stock_weight"],
                           cfg["env"]["max_sector_weight"], prof.min_cash)
        assert abs(obs["weights"].sum() - 1) < 1e-5 and obs["weights"].min() >= -1e-9
        assert np.isfinite(r)


def test_zero_cost_equal_weight_hold_matches_analytic(md, fs, cfg):
    prof = InvestorProfile(min_cash=0.0)
    env = make_eval_env(md, fs, cfg, "train", profile=prof, cost_scale=0.0)

    first = {"done": False}

    def buy_once(e):
        if first["done"]:
            return None
        first["done"] = True
        return np.append(np.full(e.N, 1 / e.N), 0.0)

    res = run_policy(buy_once, env, "ew-hold")
    t0 = env.lo
    t1 = env.dates.get_loc(res.daily.index[-1])
    expected = np.mean(env.prices[t1] / env.prices[t0])
    assert res.nav.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_cost_charged_equals_cost_model(md, fs, cfg):
    env = make_eval_env(md, fs, cfg, "train", profile=InvestorProfile(min_cash=0.0))
    env.reset(seed=0)
    v0 = env.value()
    env.step(np.append(np.full(env.N, 1 / env.N), 0.0))
    tr = env.trades[-1]
    buy = np.full(env.N, 1 / env.N) * (v0 - tr["cost"])
    expected = env.costs.cost(buy, np.zeros(env.N), env.sigma[env.lo], env.adv[env.lo]).total
    assert tr["cost"] == pytest.approx(expected, rel=1e-6)


def test_hold_costs_nothing(md, fs, cfg):
    env = make_eval_env(md, fs, cfg, "train")
    res = run_policy(lambda e: None, env, "cash")
    assert res.trades["cost"].sum() == 0.0
    # All cash: NAV grows at the cash rate only.
    assert res.metrics["MaxDD"] == pytest.approx(0.0, abs=1e-12)


def test_equal_weight_baseline_runs(md, fs, cfg):
    res = run_policy(EqualWeight(), make_eval_env(md, fs, cfg, "val"), "ew")
    assert np.isfinite(res.metrics["Sharpe"])
    assert res.trades["cost"].sum() > 0


def test_sip_inflows_do_not_count_as_returns(md, fs, cfg):
    prof = InvestorProfile(min_cash=0.0, sip_monthly=10_000)
    res = run_policy(lambda e: None, make_eval_env(md, fs, cfg, "train", profile=prof), "cash-sip")
    assert res.daily["flows"].sum() > 0
    years = (res.nav.index[-1] - res.nav.index[0]).days / 365.25
    assert res.nav.iloc[-1] == pytest.approx((1 + cfg["env"]["cash_rate"]) ** years, rel=0.02)


def test_metrics_known_series():
    idx = pd.bdate_range("2020-01-01", periods=5)
    nav = pd.Series([1.0, 1.1, 0.99, 1.2, 0.9], index=idx)
    assert max_drawdown(nav) == pytest.approx(0.9 / 1.2 - 1)
    flat = pd.Series(np.full(300, 0.001))
    assert sharpe(flat + np.random.default_rng(0).normal(0, 1e-6, 300), rf=0.0) > 100


def test_trade_band_skips_tiny_trades(md, fs, cfg):
    env = make_eval_env(md, fs, cfg, "train", profile=InvestorProfile(min_cash=0.0))
    env.reset(seed=0)
    ew = np.append(np.full(env.N, 0.9 / env.N), 0.1)
    env.step(ew)
    w_before = env.current_weights()
    nudged = w_before.copy()
    nudged[0] += 0.002          # below the 0.5% band
    nudged[-1] -= 0.002
    env.step(nudged)
    assert env.trades[-1]["cost"] == 0.0
