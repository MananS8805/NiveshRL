"""Point-in-time membership: no look-ahead, renames by ISIN, reviewed aliases, genuine crashes kept, luck test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from niveshrl.research import constituents as cs
from niveshrl.research import data as D


def _snap(rows):
    return pd.DataFrame(rows, columns=["Company Name", "Industry", "Symbol", "Series", "ISIN Code"])


@pytest.fixture
def snapdir(tmp_path, monkeypatch):
    d = tmp_path / "constituents"
    d.mkdir()
    _snap([["A Ltd", "X", "AAA", "EQ", "IN1"], ["B Ltd", "X", "BBB", "EQ", "IN2"]]).to_csv(d / "nifty200_20150101.csv", index=False)
    # BBB leaves; AAA renamed to AAX (same ISIN); CCC joins
    _snap([["A Ltd", "X", "AAX", "EQ", "IN1"], ["C Ltd", "Y", "CCC", "EQ", "IN3"]]).to_csv(d / "nifty200_20180101.csv", index=False)
    monkeypatch.setattr(cs, "DIR", d)
    monkeypatch.setattr(cs, "CURRENT", tmp_path / "missing.csv")
    monkeypatch.setattr(cs, "ALIASES", tmp_path / "no_aliases.csv")
    return d


def test_membership_is_as_of_with_no_look_ahead(snapdir):
    h = cs.history()
    idx = pd.to_datetime(["2014-06-01", "2015-01-01", "2017-12-31", "2018-01-01", "2020-01-01"])
    m = cs.member_mask(idx, ["AAX.NS", "BBB.NS", "CCC.NS"], h)
    assert not m.loc["2014-06-01"].any()                       # before the first snapshot: unknown
    assert m.loc["2015-01-01", "BBB.NS"] and m.loc["2017-12-31", "BBB.NS"]
    assert not m.loc["2017-12-31", "CCC.NS"]                   # joins only from its snapshot date, never earlier
    assert m.loc["2018-01-01", "CCC.NS"] and not m.loc["2018-01-01", "BBB.NS"]
    assert m["AAX.NS"].loc["2015-01-01":].all()                 # AAA and AAX are one company (ISIN)


def test_backfill_only_affects_dates_before_first_snapshot(snapdir):
    h = cs.history()
    idx = pd.to_datetime(["2010-01-01", "2016-01-01"])
    a = cs.member_mask(idx, ["AAX.NS", "BBB.NS", "CCC.NS"], h, backfill=True)
    b = cs.member_mask(idx, ["AAX.NS", "BBB.NS", "CCC.NS"], h)
    assert a.loc["2010-01-01", "BBB.NS"] and not b.loc["2010-01-01"].any()
    assert a.loc["2016-01-01"].equals(b.loc["2016-01-01"])


def test_renames_and_reviewed_aliases(snapdir, tmp_path, monkeypatch):
    al = tmp_path / "aliases.csv"
    al.write_text("symbol,yahoo,note\nBBB,BBN.NS,renamed\n")
    monkeypatch.setattr(cs, "ALIASES", al)
    tm = cs.ticker_map(cs.history())
    assert tm["AAA"] == "AAX.NS" and tm["BBB"] == "BBN.NS" and tm["CCC"] == "CCC.NS"


def test_genuine_crash_is_not_treated_as_a_bonus(tmp_path, monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=10)
    px = pd.Series([100.0] * 5 + [66.5] * 5, index=idx)        # -33.5%: looks like a 3:2 bonus
    close = pd.DataFrame({"ZZZ.NS": px})
    monkeypatch.setattr(D, "ACTIONS_CSV", tmp_path / "none.csv")
    monkeypatch.setattr(D, "GENUINE_CSV", tmp_path / "none2.csv")
    adj, ev = D.adjust_corporate_actions(close)
    assert ev and adj["ZZZ.NS"].iloc[0] == pytest.approx(66.5)  # unreviewed: treated as a ratio
    g = tmp_path / "genuine.csv"
    g.write_text(f"date,ticker,note\n{idx[5].date()},ZZZ.NS,crash\n")
    monkeypatch.setattr(D, "GENUINE_CSV", g)
    adj, ev = D.adjust_corporate_actions(close)
    assert not ev and adj["ZZZ.NS"].iloc[0] == 100.0           # reviewed crash: the loss stays


def test_luck_test_is_reproducible_and_bounded():
    from niveshrl.research import backtest as bt
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=600)
    cols = [f"S{i}.NS" for i in range(30)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (600, 30)), 0)), index=idx, columns=cols)
    p = D.Panel(close=close, volume=close * 0 + 1e6, sectors=pd.Series("X", index=cols), names=pd.Series(cols, index=cols),
                bench=close.mean(axis=1), vix=pd.Series(15.0, index=idx))
    me = p.month_ends()
    scores = close.pct_change(60).reindex(me)
    spec = bt.StrategySpec(signal="momentum", top=5, start="2018-04-01")
    a = bt.luck_test(p, scores, spec, n_paths=8, seed=3)
    b = bt.luck_test(p, scores, spec, n_paths=8, seed=3)
    pd.testing.assert_frame_equal(a["paths"], b["paths"])
    assert 0 <= a["percentile"]["CAGR"] <= 1 and len(a["paths"]) == 8
