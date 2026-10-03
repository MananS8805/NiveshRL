"""Daily desk: indicators, analyst score, news filter, FinBERT, next-day leakage, monitor list, watchlist."""
import numpy as np
import pandas as pd
import pytest

from niveshrl import watchlist as wl
from niveshrl.research import analyst, monitor, nextday as nd, news, technicals as T
from niveshrl.research.data import Panel


def ohlc_panel(n_stocks=30, days=700, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=days)
    tick = [f"S{i:02d}.NS" for i in range(n_stocks)]
    r = 0.0003 + 0.015 * rng.standard_normal((days, n_stocks))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=tick)
    spread = np.abs(rng.normal(0, 0.008, (days, n_stocks)))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = close.shift(1).fillna(close) * (1 + rng.normal(0, 0.003, (days, n_stocks)))
    high, low = np.maximum(high, open_), np.minimum(low, open_)
    vol = pd.DataFrame(rng.integers(1e5, 1e6, (days, n_stocks)).astype(float), index=idx, columns=tick)
    return Panel(close=close, volume=vol, sectors=pd.Series(["A", "B", "C"] * (n_stocks // 3), index=tick),
                 names=pd.Series(tick, index=tick), bench=close.mean(axis=1), vix=pd.Series(15.0, index=idx),
                 open=open_, high=high, low=low)


@pytest.fixture(scope="module")
def panel():
    return ohlc_panel()


# ---------------------------------------------------------------- indicators
def test_rsi_extremes_and_midpoint():
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = pd.DataFrame({"x": np.arange(60, dtype=float) + 100}, index=idx)
    assert T.rsi(up)["x"].iloc[-1] == pytest.approx(100.0)
    zig = pd.DataFrame({"x": 100 + np.tile([1.0, -1.0], 30).cumsum()}, index=idx)
    assert 40 < T.rsi(zig)["x"].iloc[-1] < 60


def test_macd_of_constant_is_zero():
    c = pd.DataFrame({"x": np.full(100, 50.0)}, index=pd.bdate_range("2020-01-01", periods=100))
    line, sig, hist = T.macd(c)
    assert abs(line["x"].iloc[-1]) < 1e-12 and abs(hist["x"].iloc[-1]) < 1e-12


def test_atr_of_constant_range():
    idx = pd.bdate_range("2020-01-01", periods=80)
    c = pd.DataFrame({"x": np.full(80, 100.0)}, index=idx)
    p = Panel(close=c, volume=c * 0 + 1, sectors=pd.Series({"x": "A"}), names=pd.Series({"x": "x"}),
              bench=c["x"], vix=c["x"], open=c, high=c + 2, low=c - 2)
    assert T.atr(p)["x"].iloc[-1] == pytest.approx(4.0)


def test_indicator_snapshot_columns(panel):
    f = T.indicator_frames(panel)
    snap = T.snapshot(f, panel)
    assert set(T.TECH_COLUMNS) <= set(snap.columns)
    assert snap["rsi14"].between(0, 100).all()
    assert set(snap["supertrend"].dropna().unique()) <= {-1.0, 1.0}


# ---------------------------------------------------------------- analyst score
def test_analyst_score_formula():
    df = pd.DataFrame({"recommendationMean": [1.0, 5.0], "targetMeanPrice": [140.0, 80.0],
                       "numberOfAnalystOpinions": [25, 25], "buy_share_now": [0.8, 0.2],
                       "buy_share_3m": [0.6, 0.4]}, index=["A", "B"])
    s = analyst.score(df, pd.Series({"A": 100.0, "B": 100.0}))
    # A: consensus 100, upside +40% -> 100, revision +20 pts -> 100, coverage 100 => 100
    assert s["A"] == pytest.approx(100.0)
    # B: consensus 0, upside -20% -> 0, revision -20 pts -> 0, coverage 100 => 10
    assert s["B"] == pytest.approx(10.0)


def test_analyst_score_nan_without_coverage():
    df = pd.DataFrame({"recommendationMean": [None], "numberOfAnalystOpinions": [0]}, index=["X"])
    assert np.isnan(analyst.score(df)["X"])


# ---------------------------------------------------------------- news
def test_news_relevance_and_dedupe(monkeypatch):
    items = [
        {"title": "Titan shares rise 3% after strong festive demand", "source": "ET", "link": "a", "published": None},
        {"title": "Titan shares rise 3% after strong festive demand", "source": "Mint", "link": "b", "published": None},
        {"title": "Titanic exhibition opens in Mumbai", "source": "X", "link": "c", "published": None},
        {"title": "Titan Company stock price, news, quote and history", "source": "Yahoo", "link": "d", "published": None},
        {"title": "Gold prices hit record", "source": "Y", "link": "e", "published": None},
    ]
    monkeypatch.setattr(news, "fetch_rss", lambda q, days=2, timeout=20: items)
    df = news.stock_news("TITAN.NS", "Titan Company Limited")
    assert df["title"].tolist() == ["Titan shares rise 3% after strong festive demand"]


def test_short_name():
    assert news.short_name("ICICI Bank Limited") == "ICICI Bank"
    assert news.short_name("Titan Company Limited") == "Titan"


# ---------------------------------------------------------------- FinBERT (downloads ~440 MB on first run)
def test_finbert_reads_financial_language():
    pytest.importorskip("transformers")
    from niveshrl.research import sentiment
    r = sentiment.score_texts(["Company beats estimates as net profit jumps 25%",
                               "Shares crash 20% after fraud allegations",
                               "Board meeting scheduled for Thursday"])
    assert r["label"].tolist() == ["positive", "negative", "neutral"]
    assert r["score"].iloc[0] > 0.5 and r["score"].iloc[1] < -0.5


# ---------------------------------------------------------------- next-day model
def test_nextday_features_no_lookahead(panel):
    full = nd.build(panel)
    cut = 500
    short = Panel(close=panel.close.iloc[:cut], volume=panel.volume.iloc[:cut], sectors=panel.sectors,
                  names=panel.names, bench=panel.bench.iloc[:cut], vix=panel.vix.iloc[:cut],
                  open=panel.open.iloc[:cut], high=panel.high.iloc[:cut], low=panel.low.iloc[:cut])
    part = nd.build(short)
    np.testing.assert_allclose(full.X[:cut - 1], part.X[:cut - 1], equal_nan=True, rtol=1e-6, atol=1e-6)


def test_nextday_walk_forward_ignores_future_labels(panel):
    dd = nd.build(panel)
    base = nd.walk_forward(dd, first_test_year=2017, window=2, models=("logreg",), verbose=False, train_subsample=1.0)
    y2 = dd.y.copy()
    later = dd.dates.year >= 2017
    y2[later] = np.random.default_rng(1).integers(0, 2, y2[later].shape).astype(np.float32)
    dd2 = nd.DayData(dates=dd.dates, tickers=dd.tickers, X=dd.X, y=y2, ret_next=dd.ret_next, valid=dd.valid)
    scr = nd.walk_forward(dd2, first_test_year=2017, window=2, models=("logreg",), verbose=False, train_subsample=1.0)
    y17 = base.index.get_level_values(0).year == 2017
    np.testing.assert_allclose(base.loc[y17, "logreg"].to_numpy(), scr.loc[y17, "logreg"].to_numpy())


# ---------------------------------------------------------------- monitor list
def test_monitor_ranking_order():
    idx = ["GOOD.NS", "BAD.NS", "MID.NS"]
    tech = pd.DataFrame({"vol_ratio": [3.0, 1.0, 1.0], "rsi14": [60.0, 25.0, 50.0], "donchian_breakout": [1, 0, 0],
                         "new_52w_high": [1, 0, 0], "golden_cross_5d": [0, 0, 0], "supertrend": [1, -1, 1],
                         "from_52w_high": [0.0, -0.4, -0.1]}, index=idx)
    prob = pd.Series([0.58, 0.42, 0.50], index=idx)
    sent = pd.DataFrame({"sentiment_adj": [0.5, -0.5, 0.0], "n_news": [5, 5, 0]}, index=idx)
    ml = monitor.monitor_list(tech, prob, sent, n=1)
    top = ml[ml["list"] == "watch for strength"].index[0]
    bottom = ml[ml["list"] == "watch for weakness"].index[0]
    assert top == "GOOD.NS" and bottom == "BAD.NS"
    assert "20-day breakout" in ml.loc["GOOD.NS", "reasons"].iloc[0] if isinstance(ml.loc["GOOD.NS", "reasons"], pd.Series) \
        else "20-day breakout" in ml.loc["GOOD.NS", "reasons"]


# ---------------------------------------------------------------- watchlist
def test_watchlist_roundtrip_and_alerts(tmp_path):
    path = tmp_path / "wl.json"
    wl.add("TCS.NS", "must", "core IT", target_buy=2000, path=path)
    wl.add("DLF.NS", "preferred", path=path)
    items = wl.load(path)
    assert set(items) == {"TCS.NS", "DLF.NS"} and items["TCS.NS"].tier == "must"
    a = wl.alerts(items["TCS.NS"], price=1990, day_change=0.035, days_to_earnings=5)
    assert any("buy target" in x for x in a) and any("+3.5%" in x for x in a) and any("results in 5" in x for x in a)
    wl.remove("DLF.NS", path=path)
    assert set(wl.load(path)) == {"TCS.NS"}
    with pytest.raises(ValueError):
        wl.add("X.NS", "favourite", path=path)
