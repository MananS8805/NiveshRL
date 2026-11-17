import numpy as np
import pandas as pd

from niveshrl.research import volatility as V


def test_dm_test_detects_the_better_forecast():
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.date_range("2015-01-31", periods=120, freq="ME"), [f"S{i}" for i in range(20)]])
    err = rng.standard_normal(len(idx))
    good = pd.Series(err ** 2, index=idx)
    bad = pd.Series((err + 0.5 * rng.standard_normal(len(idx))) ** 2, index=idx)
    t = V.dm_test(good, bad)
    assert t["t"] < -2 and t["p"] < 0.05 and t["n_dates"] == 120
    same = V.dm_test(good, good + 0.0)
    assert same["t"] != same["t"] or abs(same["t"]) < 1e-6 or same["mean_diff"] == 0


def test_har_forecast_is_walk_forward():
    class P:                                       # minimal panel: returns() only
        def __init__(self, r):
            self.r = r

        def returns(self):
            return self.r
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2010-01-01", "2016-12-31")
    vol = np.exp(np.cumsum(0.02 * rng.standard_normal(len(days))))[:, None] * np.linspace(0.01, 0.03, 8)
    r = pd.DataFrame(vol * rng.standard_normal((len(days), 8)), index=days, columns=[f"S{i}" for i in range(8)])
    me = pd.DatetimeIndex(r.index.to_series().groupby(r.index.to_period("M")).max().to_numpy())
    rows = []
    for t in me[3:-1]:
        past, fut = r.loc[:t], r.loc[t:].iloc[1:22]
        for c in r:
            rows.append({"date": t, "ticker": c, "rv21": past[c].iloc[-21:].std() * V.ANN,
                         "rv63": past[c].iloc[-63:].std() * V.ANN, "rv_next": fut[c].std() * V.ANN})
    frame = pd.DataFrame(rows).set_index(["date", "ticker"])
    f = V.har_forecast(P(r), frame, first_year=2014, window_years=4)
    dates = f.index.get_level_values(0)
    assert f[dates.year < 2014].isna().all() and f[dates.year >= 2014].notna().mean() > 0.9
    # changing a future label must not change any earlier forecast (no look-ahead)
    frame2 = frame.copy()
    frame2.loc[frame2.index.get_level_values(0).year == 2016, "rv_next"] *= 3
    f2 = V.har_forecast(P(r), frame2, first_year=2014, window_years=4)
    early = dates.year < 2016
    assert np.allclose(f[early].dropna(), f2[early].dropna())
