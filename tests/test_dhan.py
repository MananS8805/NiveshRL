import pandas as pd
import pytest

from niveshrl import dhan as D


class _Resp:
    def __init__(self, js, code=200):
        self._js, self.status_code, self.content = js, code, b"x"

    def json(self):
        return self._js


class _Sess:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return self.routes[url.rsplit("/v2", 1)[1]](json)

    def get(self, url, timeout=None):
        return self.routes[url.rsplit("/v2", 1)[1]](None)


def _client(routes):
    c = D.Dhan("cid", "tok", session=_Sess(routes))
    c._ids = pd.Series({"RELIANCE": 2885, "TCS": 11536})
    c.quote_rate = c.data_rate = D._Limiter(1000)
    return c


def test_ohlc_maps_ids_and_parses():
    c = _client({"/marketfeed/ohlc": lambda b: _Resp({"status": "success", "data": {"NSE_EQ": {
        "2885": {"last_price": 1500.5, "ohlc": {"open": 1490, "high": 1510, "low": 1485, "close": 1495}}}}})})
    df = c.ohlc(["RELIANCE.NS", "TCS.NS", "UNKNOWN.NS"])
    assert c.s.calls[0][1] == {"NSE_EQ": [2885, 11536]}
    assert df.loc["RELIANCE.NS", "price"] == 1500.5 and "TCS.NS" not in df.index


def test_intraday_candles_are_ist_and_columnar():
    ts = [1763350500, 1763350800]                       # 2025-11-17 09:05/09:10 IST in epoch seconds
    c = _client({"/charts/intraday": lambda b: _Resp({"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, 2.5],
                                                      "volume": [10, 20], "timestamp": ts})})
    from datetime import datetime
    f = c.intraday("RELIANCE.NS", datetime(2025, 11, 17, 9, 15), datetime(2025, 11, 17, 15, 30))
    body = c.s.calls[0][1]
    assert body["securityId"] == "2885" and body["exchangeSegment"] == "NSE_EQ" and body["interval"] == "5"
    assert list(f.columns) == ["open", "high", "low", "close", "volume"] and f.index[0].hour == 9 and f.index.tz is None


def test_errors_raise_and_status_never_does():
    c = _client({"/marketfeed/ohlc": lambda b: _Resp({"status": "failure", "errorCode": "DH-902"}, 400),
                 "/profile": lambda b: _Resp({"dataPlan": "Deactive"})})
    with pytest.raises(D.DhanError):
        c.ohlc(["RELIANCE.NS"])
    st = c.status()
    assert st["ok"] is False and "Data API" in st["why"]


def test_module_has_no_order_endpoints():
    import inspect
    src = inspect.getsource(D).lower()
    for bad in ("/orders", "/super/orders", "/forever", "placeorder", "place_order"):
        assert bad not in src
