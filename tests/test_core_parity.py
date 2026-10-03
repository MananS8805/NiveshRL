"""C++ core (niveshrl_core) vs the Python reference implementations: same numbers, faster."""
import time

import numpy as np
import pandas as pd
import pytest

core = pytest.importorskip("niveshrl_core")

from niveshrl.research import backtest as bt  # noqa: E402
from niveshrl.research import screener, technicals as T  # noqa: E402
from tests.test_desk import ohlc_panel  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    p = ohlc_panel(n_stocks=30, days=800)
    # Realistic NaN patterns: late listings (leading NaN) and a mid-series gap.
    for k in ("close", "open", "high", "low"):
        df = getattr(p, k)
        df.iloc[:150, -4:] = np.nan
        df.iloc[400:405, 3] = np.nan
    return p


def _eq(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN patterns differ"
    m = ~np.isnan(a)
    np.testing.assert_allclose(a[m], b[m], rtol=tol, atol=tol)


@pytest.mark.parametrize("alpha,minp", [(1 / 14, 14), (2 / 13, 0), (2 / 201, 200)])
def test_ewm(panel, alpha, minp):
    c = panel.close
    _eq(core.ewm(c.to_numpy(), alpha, minp), c.ewm(alpha=alpha, adjust=False, min_periods=minp).mean().to_numpy())


@pytest.mark.parametrize("n,minp", [(20, 20), (252, 120), (60, 40)])
def test_rolling(panel, n, minp):
    c = panel.close
    x = c.to_numpy()
    _eq(core.rolling_mean(x, n, minp), c.rolling(n, min_periods=minp).mean().to_numpy())
    _eq(core.rolling_max(x, n, minp), c.rolling(n, min_periods=minp).max().to_numpy())
    _eq(core.rolling_min(x, n, minp), c.rolling(n, min_periods=minp).min().to_numpy())
    _eq(core.rolling_std(x, n, minp, 0), c.rolling(n, min_periods=minp).std(ddof=0).to_numpy(), tol=1e-8)


def test_rsi_true_range_supertrend(panel):
    p = panel
    _eq(core.rsi(p.close.to_numpy(), 14), T.rsi(p.close).to_numpy())
    _eq(core.true_range(p.high.to_numpy(), p.low.to_numpy(), p.close.to_numpy()), T.true_range(p).to_numpy())
    atr = T.atr(p, 10)
    hl2 = (p.high + p.low) / 2
    _eq(core.supertrend(hl2.to_numpy(), atr.to_numpy(), p.close.to_numpy(), 3.0), T.supertrend_dir(p).to_numpy())


def test_screener_filters_match(panel):
    f = T.indicator_frames(panel)
    t = T.snapshot(f, panel)
    flt = [("rsi14", ">", 45), ("vol_ratio", ">=", 0.8), ("vs_sma50", "<", 0.05)]
    ref = screener.apply(t, flt)
    cols = [c for c, _, _ in flt]
    ops = {">": 0, ">=": 1, "<": 2, "<=": 3, "=": 4}
    X = t[cols].to_numpy(float)
    mask = core.apply_filters(X, [(i, ops[o], float(v)) for i, (_, o, v) in enumerate(flt)])
    assert list(t.index[mask]) == list(ref.index)


@pytest.mark.parametrize("kw", [dict(top=10), dict(top=0.2, weighting="inv_vol", vol_target=0.15),
                                dict(top=8, long_short=True, rebalance_months=3),
                                dict(top=6, weighting="score", cost_scale=2.0)])
def test_backtest_engines_identical(panel, kw):
    sc = bt.factor_scores(panel, "momentum")
    spec = bt.StrategySpec(signal="momentum", start="2016-01-01", **kw)
    a = bt.run_backtest(panel, sc, spec, engine="numpy")
    b = bt.run_backtest(panel, sc, spec, engine="cpp")
    np.testing.assert_allclose(a.nav.to_numpy(), b.nav.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(a.rebalances["cost"].to_numpy(), b.rebalances["cost"].to_numpy(), rtol=1e-9)


def test_tick_store_quotes_bars_and_bounded_memory():
    ts = core.TickStore(["A.NS", "B.NS"], bar_capacity=5)
    base = 1_790_000_000_000
    assert ts.update("A.NS", base, 100.0, 1.0, 1000)
    assert ts.update("A.NS", base + 10_000, 102.0, 2.0, 1500)          # same minute
    assert ts.update("A.NS", base + 70_000, 101.0, 1.5, 1700)          # next minute
    assert not ts.update("ZZZ.NS", base, 1.0, 0.0, 0)
    changed, ver = ts.changed_since(0)
    assert changed == [0] and ver == 3
    bars = ts.bars("A.NS")
    assert len(bars) == 2 and bars[0].high == 102.0 and bars[0].close == 102.0 and bars[0].volume == 500
    assert bars[1].open == 101.0 and bars[1].volume == 200
    for k in range(20):                                              # 20 more minutes into a 5-bar ring
        ts.update("B.NS", base + 60_000 * k, 50.0 + k, 0.0, 100 * k)
    assert len(ts.bars("B.NS")) == 5 and ts.bars("B.NS")[-1].close == 69.0
    assert ts.changed_since(ver)[0] == [1]


def test_cpp_is_faster_on_hot_paths(panel):
    """Benchmark: the C++ versions of the slowest Python indicators."""
    big = pd.concat([panel.close] * 6, axis=1)                         # 180 stocks x 800 days
    big.columns = range(big.shape[1])
    x = big.to_numpy()
    t0 = time.perf_counter(); big.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(); tp = time.perf_counter() - t0
    t0 = time.perf_counter(); core.ewm(x, 1 / 14, 14); tc = time.perf_counter() - t0
    p = panel
    t0 = time.perf_counter(); T.supertrend_dir(p); sp = time.perf_counter() - t0
    atr = T.atr(p, 10).to_numpy(); hl2 = ((p.high + p.low) / 2).to_numpy(); c = p.close.to_numpy()
    t0 = time.perf_counter(); core.supertrend(hl2, atr, c, 3.0); sc_ = time.perf_counter() - t0
    print(f"\newm: pandas {tp * 1e3:.1f} ms vs C++ {tc * 1e3:.1f} ms | supertrend: Python {sp * 1e3:.0f} ms vs C++ {sc_ * 1e3:.2f} ms")
    assert sc_ < sp / 10
