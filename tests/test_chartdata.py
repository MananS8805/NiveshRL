"""Chart data: intraday cache + IST conversion, weekly bars, VWAP session reset, pivots from the previous period,
Supertrend side, Fibonacci direction, and the price chart drawing every indicator at every kind of timeframe."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from niveshrl.research import chartdata as C


def _intraday(days=3, bars=75, start="2026-09-28 09:15"):
    idx = []
    for d in pd.bdate_range(start[:10], periods=days):
        idx += list(pd.date_range(f"{d.date()} 09:15", periods=bars, freq="5min"))
    n = len(idx)
    c = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.2, n))
    return pd.DataFrame({"Open": c, "High": c + 0.3, "Low": c - 0.3, "Close": c, "Volume": 1000.0}, index=pd.DatetimeIndex(idx))


def test_vwap_resets_each_session():
    df = _intraday()
    v = C.vwap(df)
    first_bars = df.index.normalize().to_series().diff().ne(pd.Timedelta(0)).to_numpy()
    tp = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy()
    np.testing.assert_allclose(v.to_numpy()[first_bars], tp[first_bars])    # each day starts at its first bar's price


def test_pivots_come_from_previous_period():
    df = _intraday(days=2)
    pv = C.pivots(df)
    d1 = df[df.index.normalize() == df.index.normalize()[0]]
    H, L, Cl = d1["High"].max(), d1["Low"].min(), d1["Close"].iloc[-1]
    P = (H + L + Cl) / 3
    day2 = pv[df.index.normalize() == df.index.normalize()[-1]]
    assert day2["P"].nunique() == 1 and day2["P"].iloc[0] == pytest.approx(P)
    assert day2["R1"].iloc[0] == pytest.approx(2 * P - L) and day2["S1"].iloc[0] == pytest.approx(2 * P - H)
    assert (day2["TC"] >= day2["BC"]).all()
    assert pv["P"].iloc[:75].isna().all()                                   # day 1 has no previous day


def test_weekly_bars_and_intraday_detection():
    idx = pd.bdate_range("2026-09-07", periods=10)
    df = pd.DataFrame({"Open": np.arange(10.0), "High": np.arange(10.0) + 1, "Low": np.arange(10.0) - 1,
                       "Close": np.arange(10.0) + 0.5, "Volume": 1.0}, index=idx)
    w = C.weekly(df)
    assert len(w) == 2 and w["Open"].iloc[0] == 0 and w["Close"].iloc[0] == 4.5 and w["Volume"].iloc[1] == 5
    assert w.index[0] == pd.Timestamp("2026-09-11")
    assert not C.is_intraday(df.index) and C.is_intraday(_intraday().index)


def test_supertrend_line_sits_on_the_trend_side_and_fib_direction():
    c = np.r_[np.linspace(100, 150, 80), np.linspace(150, 100, 80)]
    df = pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1.0},
                      index=pd.bdate_range("2025-01-01", periods=160))
    line, d = C.supertrend(df)
    up = d == 1
    assert (line[up] < df["Close"][up]).all() and (line[d == -1] > df["Close"][d == -1]).all()
    assert d.iloc[60] == 1 and d.iloc[-1] == -1
    fb = C.fibonacci(df.iloc[:80])                                          # up-swing: levels measured down from the high
    assert fb["up"] and fb["levels"]["0%"] == pytest.approx(151) and fb["levels"]["100%"] == pytest.approx(99)
    assert fb["levels"]["61.8%"] == pytest.approx(151 - 0.618 * 52)


def test_intraday_cache_converts_to_ist_and_survives_outages(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CACHE", tmp_path)
    utc = pd.date_range("2026-10-01 03:45", periods=3, freq="5min", tz="UTC")
    raw = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Adj Close": 1.5, "Volume": 10}, index=utc)
    df = C.intraday_bars("X.NS", "5m", download=lambda: raw)
    assert df.index[0] == pd.Timestamp("2026-10-01 09:15") and df.index.tz is None
    path = tmp_path / "X.NS_5m.parquet"
    os.utime(path, (0, 0))                                                  # make the cache stale

    def offline():
        raise ConnectionError("offline")
    assert len(C.intraday_bars("X.NS", "5m", download=offline)) == 3     # stale cache beats nothing
    with pytest.raises(ConnectionError):
        C.intraday_bars("Y.NS", "5m", download=offline)                   # and no cache means a visible error


def test_price_chart_draws_every_indicator(qtbot):
    from niveshrl.desktop.pricechart import INDICATORS, PriceChart
    from niveshrl.research.plans import Plan  # noqa: F401  (plan objects only need entry/stop/t1/t2)

    class _P:
        entry, stop, t1, t2 = 100.0, 95.0, 107.5, 112.5
    ch = PriceChart(controls=True)
    qtbot.addWidget(ch)
    ch.ind = {k for k, _ in INDICATORS if k}
    intr = _intraday()
    daily = pd.DataFrame({k: intr[k].to_numpy()[:200] for k in intr}, index=pd.bdate_range("2025-01-01", periods=200))
    for log in (False, True):
        ch.log = log
        ch.cmp_on = {"NIFTY 50"}
        ch.plot(intr, "intraday", {"NIFTY 50": intr["Close"] * 200}, _P())
        assert len(ch.plots) == 4                                           # price + volume + RSI + MACD
        assert ch.buttons["1D"].isVisibleTo(ch) and not ch.buttons["1Y"].isVisibleTo(ch)
        ch.set_range("1D")
        ch.plot(daily, "daily", None, _P())
        assert not ch.buttons["1D"].isVisibleTo(ch)
    ch.ind = {"sma"}
    ch.plot(daily, "daily")
    assert len(ch.plots) == 1
