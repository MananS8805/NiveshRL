import numpy as np
import pandas as pd
import pytest

from niveshrl.config import load_yaml
from niveshrl.constraints import is_feasible, project
from niveshrl.costs import IndiaCostModel
from niveshrl.tax import TaxLedger

RATES = load_yaml("configs/costs_india.yaml")


def test_buy_cost_matches_contract_note_hand_calc():
    c = IndiaCostModel(RATES).cost(np.array([100_000.0]), np.array([0.0]))
    # STT 100 + exch 2.97 + SEBI 0.10 + stamp 15 + GST 18% of (2.97 + 0.10)
    assert c.stt == pytest.approx(100.0)
    assert c.stamp == pytest.approx(15.0)
    assert c.gst == pytest.approx(0.18 * (2.97 + 0.10))
    assert c.dp == 0.0
    assert c.statutory == pytest.approx(100 + 2.97 + 0.10 + 15 + 0.5526, abs=1e-3)


def test_sell_cost_includes_dp_charge_per_scrip():
    c = IndiaCostModel(RATES).cost(np.zeros(3), np.array([50_000.0, 50_000.0, 0.0]))
    assert c.stamp == 0.0
    assert c.dp == pytest.approx(2 * RATES["dp_charge"])
    assert c.statutory == pytest.approx(100 + 2.97 + 0.10 + 0.5526 + 2 * 15.93, abs=1e-3)


def test_zero_scale_means_zero_cost():
    c = IndiaCostModel(RATES, scale=0.0).cost(np.array([1e5]), np.array([1e5]), np.array([0.02]), np.array([1e7]))
    assert c.total == 0.0


def test_stcg_and_ltcg_boundary():
    led = TaxLedger(1, RATES["tax"])
    led.buy(0, pd.Timestamp("2023-01-02"), 100, 100.0)
    # 150 days later: short-term, gain 50 * 50 = 2500 -> 20% = 500
    assert led.sell(0, pd.Timestamp("2023-06-01"), 50, 150.0) == pytest.approx(500.0)
    # Held 395 days: long-term gain 5000 < 1.25 lakh exemption -> no tax
    assert led.sell(0, pd.Timestamp("2024-02-01"), 50, 200.0) == pytest.approx(0.0)


def test_exactly_365_days_is_short_term():
    led = TaxLedger(1, RATES["tax"])
    led.buy(0, pd.Timestamp("2023-04-10"), 10, 100.0)
    assert led.sell(0, pd.Timestamp("2024-04-09"), 10, 200.0) == pytest.approx(0.2 * 1000)


def test_short_term_loss_offsets_gain_within_year():
    led = TaxLedger(2, RATES["tax"])
    d = pd.Timestamp("2023-05-01")
    led.buy(0, d, 10, 100.0)
    led.buy(1, d, 10, 100.0)
    assert led.sell(0, pd.Timestamp("2023-06-01"), 10, 200.0) == pytest.approx(200.0)
    # A 500 loss refunds 20% of it against the same FY's gains.
    assert led.sell(1, pd.Timestamp("2023-07-01"), 10, 50.0) == pytest.approx(-100.0)


@pytest.mark.parametrize("seed", range(50))
def test_projection_always_feasible(seed):
    rng = np.random.default_rng(seed)
    n = 29
    sectors = rng.integers(0, 9, n)
    sectors[:9] = np.arange(9)
    w = rng.dirichlet(np.full(n + 1, rng.uniform(0.05, 2)))
    min_cash = rng.uniform(0, 0.3)
    out = project(w, sectors, 0.10, 0.30, min_cash)
    assert is_feasible(out, sectors, 0.10, 0.30, min_cash)


def test_projection_keeps_feasible_input_unchanged():
    sectors = np.array([0, 0, 1, 1])
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
    np.testing.assert_allclose(project(w, sectors, 0.10, 0.30, 0.0), w)


def _reward_cfg(kind="utility"):
    from niveshrl.config import load_config
    c = load_config()["reward"]
    c["kind"] = kind
    return c


def test_utility_reward_ranks_risky_vs_safe_by_profile():
    """A cautious investor must prefer steady cash-like returns; a bold one the risky stream."""
    from niveshrl.rewards import RewardFn
    risky = [0.06, -0.04] * 10          # mean +1%/wk, high variance
    safe = [0.0015] * 20
    for ra, prefer_risky in [(0.0, True), (1.0, False)]:
        totals = []
        for seq in (risky, safe):
            f = RewardFn(_reward_cfg())
            totals.append(sum(f(r, 0.0, ra, 1.0) for r in seq))
        assert (totals[0] > totals[1]) == prefer_risky


def test_gamma_mapping_is_monotone():
    from niveshrl.rewards import gamma_for
    c = _reward_cfg()
    g = [gamma_for(x, c) for x in (0.0, 0.5, 1.0)]
    assert g[0] == c["gamma_min"] and g[-1] == c["gamma_max"] and g[0] < g[1] < g[2]
