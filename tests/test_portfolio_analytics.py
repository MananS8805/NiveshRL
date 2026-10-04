"""Portfolio analytics: Kite tradebook parsing, FIFO lots, the 12-month rule, FY tax with set-offs and the LTCG
exemption, harvesting ideas, XIRR against known answers and portfolio risk numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from niveshrl.research import portfolio_analytics as PA
from niveshrl.research.data import Panel

KITE = """symbol,isin,trade_date,exchange,segment,series,trade_type,auction,quantity,price,trade_id,order_id,order_execution_time
INFY,INE009A01021,2024-04-10,NSE,EQ,EQ,buy,false,10,1400.00,1,1,2024-04-10T10:00:00
INFY,INE009A01021,2024-09-01,NSE,EQ,EQ,buy,false,10,1800.00,2,2,2024-09-01T10:00:00
INFY,INE009A01021,2025-06-01,NSE,EQ,EQ,sell,false,15,1600.00,3,3,2025-06-01T10:00:00
TCS-EQ,INE467B01029,2025-05-02,NSE,EQ,EQ,buy,false,5,3000.00,4,4,2025-05-02T10:00:00
"""


def test_parse_and_fifo():
    t = PA.parse_tradebook_csv(KITE)
    assert list(t["ticker"].unique()) == ["INFY.NS", "TCS.NS"] and t["side"].tolist().count("sell") == 1
    f = PA.fifo(t)
    r = f.realised
    # 15 sold: first the 10 from Apr 2024 (held > 12 months: LT), then 5 of the Sep 2024 lot (ST)
    assert r["qty"].tolist() == [10, 5] and r["term"].tolist() == ["LT", "ST"]
    assert r["gain"].tolist() == [pytest.approx(2000.0), pytest.approx(-1000.0)]
    ol = f.open_lots.set_index("ticker")
    assert ol.loc["INFY.NS", "qty"] == 5 and ol.loc["TCS.NS", "qty"] == 5 and not f.unmatched


def test_twelve_month_rule_and_fy():
    b = pd.Timestamp("2024-03-15")
    assert not PA.long_term(b, pd.Timestamp("2025-03-15")) and PA.long_term(b, pd.Timestamp("2025-03-16"))
    assert PA.fy_of("2026-03-31") == 2025 and PA.fy_of("2026-04-01") == 2026 and PA.fy_label(2025) == "FY 2025-26"


def _real(rows):
    return pd.DataFrame(rows, columns=["ticker", "buy_date", "sell_date", "qty", "buy_price", "sell_price", "days",
                                       "term", "gain"])


def test_tax_summary_set_off_and_exemption():
    d = pd.Timestamp("2025-08-01")
    # ST: +100k gain, −30k loss; LT: +200k gain, −20k loss
    r = _real([["A", d, d, 1, 0, 0, 10, "ST", 100_000], ["B", d, d, 1, 0, 0, 10, "ST", -30_000],
               ["C", d, d, 1, 0, 0, 400, "LT", 200_000], ["D", d, d, 1, 0, 0, 400, "LT", -20_000]])
    s = PA.tax_summary(r, 2025)
    assert s["net_st"] == 70_000 and s["net_lt"] == 180_000
    assert s["taxable_lt"] == pytest.approx(180_000 - 125_000) and s["exempt_left"] == 0
    assert s["tax"] == pytest.approx((70_000 * 0.20 + 55_000 * 0.125) * 1.04)
    # a net ST loss reduces LT gains; a net LT loss never reduces ST gains
    s2 = PA.tax_summary(_real([["A", d, d, 1, 0, 0, 10, "ST", -50_000], ["C", d, d, 1, 0, 0, 400, "LT", 200_000]]), 2025)
    assert s2["net_lt"] == 150_000 and s2["taxable_st"] == 0 and s2["taxable_lt"] == 25_000
    s3 = PA.tax_summary(_real([["A", d, d, 1, 0, 0, 10, "ST", 50_000], ["C", d, d, 1, 0, 0, 400, "LT", -80_000]]), 2025)
    assert s3["taxable_st"] == 50_000 and s3["carry_forward_loss"] == 80_000
    assert PA.tax_summary(r, 2024)["n_sales"] == 0


def test_harvest_ideas():
    lots = pd.DataFrame({"ticker": ["L.NS", "G.NS", "N.NS"],
                         "buy_date": pd.to_datetime(["2026-01-01", "2024-01-01", "2025-09-01"]),
                         "qty": [10, 10, 10], "price": [100.0, 100.0, 100.0]})
    u = PA.unrealised(lots, {"L.NS": 80.0, "G.NS": 150.0, "N.NS": 130.0}, today="2026-08-01")
    assert u.set_index("ticker").loc["G.NS", "term"] == "LT" and u.set_index("ticker").loc["N.NS", "days_to_lt"] == 32
    s = {"taxable_st": 1_000.0, "taxable_lt": 0.0, "exempt_left": 125_000.0}
    h = PA.harvest(u, s)
    assert h.loc["L.NS", "Est. tax effect ₹"] == pytest.approx(200 * 0.20 * 1.04)      # 200 loss vs 1,000 ST gain
    assert "allowance" in h.loc["G.NS", "Idea"] and "32 days" in h.loc["N.NS", "Idea"]


def test_xirr_known_answers():
    assert PA.xirr([("2025-01-01", -1000), ("2026-01-01", 1100)]) == pytest.approx(0.10, abs=1e-4)
    assert PA.xirr([("2025-01-01", -1000), ("2025-07-02", 0), ("2027-01-01", 1210)]) == pytest.approx(0.10, abs=1e-3)
    assert np.isnan(PA.xirr([("2025-01-01", -1000)]))


def test_risk_numbers():
    idx = pd.bdate_range("2025-01-01", periods=300)
    rng = np.random.default_rng(0)
    rb = rng.normal(0, 0.01, 300)
    b = pd.Series(100 * np.exp(np.cumsum(rb)), index=idx)
    c = pd.DataFrame({"X.NS": 50 * np.exp(np.cumsum(2 * rb)), "Y.NS": 50 * np.exp(np.cumsum(2 * rb))}, index=idx)
    p = Panel(close=c, volume=c * 0 + 1, sectors=pd.Series({"X.NS": "A", "Y.NS": "B"}),
              names=pd.Series({"X.NS": "X", "Y.NS": "Y"}), bench=b, vix=pd.Series(15.0, index=idx))
    last = c.iloc[-1]
    rk = PA.risk(p, {"X.NS": 100 / last["X.NS"], "Y.NS": 300 / last["Y.NS"]})
    assert rk["value"] == pytest.approx(400) and rk["top_weight"] == pytest.approx(0.75)
    assert rk["beta"] == pytest.approx(2.0, rel=0.05) and rk["effective_n"] == pytest.approx(1 / (0.75 ** 2 + 0.25 ** 2))
    assert rk["sectors"]["B"] == pytest.approx(0.75) and rk["var95"] > 0
    assert rk["risk_contrib"].sum() == pytest.approx(1.0)


def test_monthly_pnl():
    r = _real([["A", None, pd.Timestamp("2025-01-10"), 1, 0, 0, 1, "ST", 100.0],
               ["A", None, pd.Timestamp("2025-01-20"), 1, 0, 0, 1, "ST", -40.0],
               ["A", None, pd.Timestamp("2025-03-01"), 1, 0, 0, 1, "ST", 10.0]])
    m = PA.monthly_pnl(r)
    assert m.loc[2025, "Jan"] == 60 and m.loc[2025, "Full year"] == 70 and np.isnan(m.loc[2025, "Feb"])
