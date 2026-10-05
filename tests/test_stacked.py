"""Rolling refits and the stacked next-day model: training windows end before validation and the test month, the
stack never uses labels from the month it predicts, patterns and notable changes, the drift monitor, the volatility
forecaster's live row, and bandit forgetting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from niveshrl.research import nextday as nd
from niveshrl.research import stacked as ST
from niveshrl.research.data import Panel


def _panel(n_days=1000, n_st=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i}.NS" for i in range(n_st)]
    r = rng.normal(0, 0.015, (n_days, n_st))
    c = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    hl = c * rng.uniform(0.01, 0.03, (n_days, n_st))
    return Panel(close=c, volume=c * 0 + 1e6 * rng.uniform(0.5, 1.5, (n_days, n_st)), sectors=pd.Series("S", index=cols),
                 names=pd.Series(cols, index=cols), bench=c.mean(axis=1), vix=pd.Series(15.0, index=idx),
                 open=c.shift().fillna(c), high=c + hl / 2, low=c - hl / 2)


def test_windows_never_overlap_the_test_month():
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    tr, va, te = ST._windows(dates, "2024-07-01", train_years=4, val_months=3)
    assert dates[tr].max() < dates[va].min() and dates[va].max() < pd.Timestamp("2024-07-01")
    assert dates[te].min() >= pd.Timestamp("2024-07-01") and dates[te].max() < pd.Timestamp("2024-08-01")
    assert dates[va].min() >= pd.Timestamp("2024-04-01") and dates[tr].min() >= pd.Timestamp("2020-07-01")


@pytest.fixture(scope="module")
def stacked_run():
    from niveshrl.research import range_model as RM
    p = _panel()
    dd = nd.build(p)
    X, y, _ = RM.build(p)
    base = ST.rolling(dd, "2022-07-01", "2023-11-30", models=("lgbm", "logreg"), range_xy=(X, y), verbose=False)
    return dd, base


def test_rolling_predictions_are_out_of_sample(stacked_run):
    dd, base = stacked_run
    refit = pd.to_datetime(base["refit"])
    d = base.index.get_level_values(0)
    assert (d >= refit).all() and (d < refit + pd.DateOffset(months=1)).all()     # each month by its own refit
    assert {"lgbm", "logreg", "range", "ensemble"} <= set(base.columns)
    assert base["lgbm"].between(0, 1).all()


def test_stack_never_sees_labels_of_the_month_it_predicts(stacked_run):
    _, base = stacked_run
    a = ST.stack(base, "2023-07-01", meta_months=6, verbose=False)
    scrambled = base.copy()
    late = scrambled.index.get_level_values(0) >= pd.Timestamp("2023-09-01")
    scrambled.loc[late, "y"] = np.random.default_rng(1).permutation(scrambled.loc[late, "y"].to_numpy())
    b = ST.stack(scrambled, "2023-07-01", meta_months=6, verbose=False)
    sel = (a.index.get_level_values(0) >= pd.Timestamp("2023-09-01")) & (a.index.get_level_values(0) < pd.Timestamp("2023-10-01"))
    np.testing.assert_allclose(a.loc[sel, "stacked"], b.loc[sel, "stacked"])        # September uses labels before Sept
    assert set(a["pattern"].dropna().str.split(" · ").str[0]) <= {"Up", "Down", "Flat"}


def test_patterns_and_notable_changes():
    idx = pd.Index(["A", "B", "C", "D"], name="ticker")
    today = pd.DataFrame({"pattern": ["Up · volatile", "Down · normal", "Flat · quiet", "Up · normal"],
                          "stacked": [0.60, 0.40, 0.50, 0.55]}, index=idx)
    prev = pd.DataFrame({"pattern": ["Down · volatile", "Down · normal", "Flat · volatile", "Flat · normal"],
                         "stacked": [0.40, 0.41, 0.50, 0.51]}, index=idx)
    ch = ST.pattern_changes(today, prev)
    assert list(ch.index) == ["A", "D", "C"]                          # B unchanged; sorted by |Δ P(up)|
    nb = ST.notable(ch)
    assert nb["A"] and nb["C"] and not nb["D"]                         # flip, size jump, nudge out of Flat
    df = pd.DataFrame({"stacked": [0.60, 0.50, 0.40, 0.50, 0.50], "range": [5.0, 4.0, 1.0, 3.0, 2.0]},
                      index=pd.MultiIndex.from_product([[pd.Timestamp("2026-01-01")], list("vwxyz")]))
    # range percentile ranks 1.0, 0.8, 0.2, 0.6, 0.4: top 30% volatile, bottom 30% quiet
    assert list(ST.add_patterns(df)["pattern"]) == ["Up · volatile", "Flat · volatile", "Down · quiet",
                                                    "Flat · normal", "Flat · normal"]


def test_model_health_rolling_ic():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2025-01-01", periods=80)
    idx = pd.MultiIndex.from_product([days, [f"S{i}" for i in range(30)]])
    ret = rng.normal(0, 0.01, len(idx))
    stk = pd.DataFrame({"ret_next": ret, "lgbm": ret + rng.normal(0, 0.01, len(idx)), "stacked": rng.normal(0, 1, len(idx))},
                       index=idx)
    h = ST.model_health(stk, window=20)
    assert h["lgbm"].iloc[-1] > 0.4 and abs(h["stacked"].iloc[-1]) < 0.15


def test_volatility_live_row_exists_without_future():
    from niveshrl.research import volatility as V
    p = _panel(n_days=400, n_st=25)
    frame, _ = V.build_vol_data(p)
    live, seq = V.build_vol_data(p, live=True)
    last = p.close.index[-1]
    assert last not in frame.index.get_level_values(0)                 # the old builder could never forecast now
    assert last in live.index.get_level_values(0) and live.xs(last, level=0)["rv_next"].isna().all()
    assert len(seq) == len(live)


def test_bandit_forgetting_fades_old_outcomes():
    from niveshrl.intraday.bandit import Bandit
    cfg = {"learning": {"min_trades_bucket": 5, "skip_t": -2.0, "half_below_r": 0.0, "ml_min_prob": 0.4}}
    f = {"setup": "orb", "side": "long", "minute": 10, "trend_aligned": 1}
    old = pd.DataFrame([f | {"net_r": -1.0, "ticker": "X", "ts": str(i)} for i in range(40)])
    new = pd.DataFrame([f | {"net_r": 1.0, "ticker": "X", "ts": str(i)} for i in range(10)])
    keep, fade = Bandit(cfg), Bandit({"learning": {**cfg["learning"], "half_life_days": 2}})
    for b in (keep, fade):
        b.update(old, day="d1")
        for k in range(6):
            b.update(new.iloc[:0] if k < 5 else new, day=f"d{k + 2}")    # five quiet days, then the market turns
    k = "orb|long"
    assert keep._summary(k)[1] < 0 < fade._summary(k)[1]              # remembering everything is still negative
    assert fade.stats[k][0] < 20                                       # effective count shrank
