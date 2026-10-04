"""Optional NSE data, offline: parsing of each endpoint's JSON, max pain and chain statistics by hand, the FII/DII
history merge, the 403 retry and loud failures (never silent fill-ins)."""
from __future__ import annotations

import pandas as pd
import pytest

from niveshrl.research import nse as N


class _Resp:
    def __init__(self, status, payload=None, text=None):
        self.status_code, self._p, self.text = status, payload, text or ""

    def json(self):
        if self._p is None:
            raise ValueError("not json")
        return self._p


class _Session:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        for k, v in self.routes.items():
            if k in url:
                return v.pop(0) if isinstance(v, list) else v
        return _Resp(200, {})


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(N, "MIN_GAP", 0.0)


def test_fii_dii_parse_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "DIR", tmp_path)
    s = _Session({"fiidii": _Resp(200, [
        {"buyValue": "25,420.04", "category": "DII", "date": "01-Oct-2026", "netValue": "10041.84", "sellValue": "15378.2"},
        {"buyValue": "12260.26", "category": "FII/FPI", "date": "01-Oct-2026", "netValue": "-9484.22", "sellValue": "21744.48"}])})
    f = N.fii_dii(N.Client(s))
    assert set(f["category"]) == {"DII", "FII"} and f.loc[f.category == "DII", "buy"].iloc[0] == pytest.approx(25420.04)
    h = N.fii_dii_history(f)
    later = f.assign(date=pd.Timestamp("2026-10-03"))
    h2 = N.fii_dii_history(later)
    assert len(h) == 2 and len(h2) == 4 and len(N.fii_dii_history(later)) == 4      # no duplicates on re-fetch


def test_shareholding_and_deals_parse():
    s = _Session({"share-holdings": _Resp(200, [
        {"date": "30-JUN-2026", "pr_and_prgrp": "71.77", "public_val": "28.23", "employeeTrusts": "0", "broadcastDate": "x"},
        {"date": "31-MAR-2026", "pr_and_prgrp": "72.27", "public_val": "27.73", "employeeTrusts": "0", "broadcastDate": "y"}]),
        "largedeal": _Resp(200, {"as_on_date": "01-Oct-2026", "BULK_DEALS_DATA": [
            {"date": "01-Oct-2026", "symbol": "ABC", "name": "Abc Ltd", "clientName": "FUND", "buySell": "BUY",
             "qty": "1000000", "watp": "250"}], "BLOCK_DEALS_DATA": []})})
    c = N.Client(s)
    sh = N.shareholding("TCS", c)
    assert list(sh.index) == sorted(sh.index) and sh["Promoter change (pp)"].iloc[-1] == pytest.approx(-0.5)
    d = N.large_deals(c)
    assert d["bulk"]["Value ₹ Cr"].iloc[0] == pytest.approx(25.0) and d["block"].empty


def test_max_pain_and_stats_by_hand():
    df = pd.DataFrame({"Strike": [90.0, 100.0, 110.0], "Call OI": [0.0, 100.0, 500.0], "Put OI": [400.0, 100.0, 0.0],
                       "Call ΔOI": [0.0, 10.0, 50.0], "Put ΔOI": [20.0, 10.0, 0.0],
                       "Call IV": [0.0, 30.0, 28.0], "Put IV": [35.0, 32.0, 0.0]})
    # buyers' payoff if expiry = 90: puts struck at 100 pay 10 × 100 = 1,000; at 100: nothing pays (the in-the-money
    # 90 call and 110 put have no OI) → 0; at 110: calls struck at 100 pay 10 × 100 = 1,000. Max pain = 100.
    assert N.max_pain(df) == 100.0
    st = N.chain_stats(df, 101.0)
    assert st["pcr"] == pytest.approx(500 / 600) and st["call_wall"] == 110 and st["put_wall"] == 90
    assert st["atm_strike"] == 100 and st["atm_iv"] == pytest.approx(0.31) and st["pcr_change"] == pytest.approx(30 / 60)


def test_option_chain_parse_and_index_type():
    payload = {"records": {"timestamp": "t", "underlyingValue": 22400, "expiryDates": ["06-Oct-2026"], "data": [
        {"strikePrice": 22400, "CE": {"openInterest": 10, "changeinOpenInterest": 2, "impliedVolatility": 12,
                                      "lastPrice": 100, "totalTradedVolume": 5},
         "PE": {"openInterest": 8, "changeinOpenInterest": -1, "impliedVolatility": 13, "lastPrice": 90,
                "totalTradedVolume": 4}}]}}
    s = _Session({"contract-info": _Resp(200, {"expiryDates": ["06-Oct-2026"]}), "option-chain-v3": _Resp(200, payload)})
    df, meta = N.option_chain("NIFTY", c=N.Client(s))
    assert "type=Indices" in [u for u in s.calls if "v3" in u][0]
    assert df.loc[0, "Call OI"] == 10 and df.loc[0, "Put ΔOI"] == -1 and meta["underlying"] == 22400


def test_failures_are_loud_and_403_retries():
    s = _Session({"fiidii": [_Resp(403), _Resp(200, [{"category": "DII", "date": "01-Oct-2026", "buyValue": 1,
                                                       "sellValue": 1, "netValue": 0}])]})
    assert len(N.fii_dii(N.Client(s))) == 1                            # retried with fresh cookies
    with pytest.raises(N.NSEUnavailable):
        N.fii_dii(N.Client(_Session({"fiidii": _Resp(503, text="<html>")})))
    with pytest.raises(N.NSEUnavailable):
        N.fii_dii(N.Client(_Session({"fiidii": _Resp(200, None, "<html>")})))      # a web page, not JSON
    with pytest.raises(N.NSEUnavailable):
        N.expiries("XYZ", N.Client(_Session({"contract-info": _Resp(200, {})})))
