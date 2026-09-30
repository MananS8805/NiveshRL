import numpy as np
import pandas as pd
import pytest

from niveshrl.planner import goal_projection, plan_orders
from niveshrl.profile import from_answers, suggest_max_stocks


@pytest.mark.parametrize("seed", range(20))
def test_orders_never_exceed_cash(seed):
    rng = np.random.default_rng(seed)
    n = 10
    prices = rng.uniform(50, 12_000, n)
    w = rng.dirichlet(np.ones(n + 1))
    holdings = rng.integers(0, 5, n)
    cash = float(rng.uniform(0, 50_000))
    orders, left = plan_orders(w, [f"S{i}" for i in range(n)], prices, holdings, cash, new_money=10_000)
    assert left >= -1e-6
    assert all(o.shares > 0 for o in orders)


def test_sip_buys_underweight_first():
    prices = np.array([100.0, 100.0])
    w = np.array([0.5, 0.5, 0.0])
    orders, _ = plan_orders(w, ["A", "B"], prices, holdings=np.array([100, 0]), cash=0, new_money=10_000)
    assert [o.ticker for o in orders if o.side == "BUY"] == ["B"]
    assert not any(o.side == "SELL" for o in orders)  # within drift band: no forced sell


def test_goal_projection_zero_returns_equals_contributions():
    r = pd.Series(np.zeros(500))
    out = goal_projection(r, initial=10_000, sip_monthly=1_000, years=1, target=21_000)
    assert out["final_median"] == pytest.approx(10_000 + 12 * 1_000)
    assert out["p_target"] == 1.0


def test_questionnaire_maps_to_profile():
    p = from_answers({"crash_reaction": "Hold and wait", "max_loss": "20%", "horizon": "1–3 years",
                      "experience": "None", "initial": 20_000, "sip": 2_000})
    assert p.risk_aversion >= 0.7          # short horizon forces caution
    assert p.min_cash == 0.15
    assert p.max_stocks == suggest_max_stocks(20_000) == 5


def test_expensive_share_does_not_block_cheaper_buys():
    # Regression: the largest gap (T) needs more than half a share, so it is
    # skipped; the cheaper stocks must still be bought.
    prices = np.array([4841.0, 1058.6, 166.4])
    w = np.array([0.04, 0.05, 0.05, 0.86])   # T gap Rs 2,000 < half a share (Rs 2,420)
    orders, left = plan_orders(w, ["T", "I", "W"], prices, cash=50_000)
    bought = {o.ticker: o.shares for o in orders}
    assert "T" not in bought and bought.get("I", 0) >= 2 and bought.get("W", 0) >= 14
    assert left >= 0.86 * 50_000 - 1
