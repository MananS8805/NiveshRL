"""DhanHQ v2 market data, read only. This module has no order, position or funds code and never will: NiveshRL is paper
only. Facts below are from https://dhanhq.co/docs/v2/ (read 2026-11-17).

Credentials (you supply them; NiveshRL never asks for your password or PIN):
- ``DHAN_CLIENT_ID`` and ``DHAN_ACCESS_TOKEN`` environment variables, or
- ``configs/dhan.local.yaml`` (git-ignored) with ``client_id:`` and ``access_token:``.
An access token from web.dhan.co lasts 24 hours. Market data needs Dhan's separate Data API plan (₹499 on their pricing
page); without it the API answers error 806 / DH-902 and NiveshRL falls back to Yahoo.

Endpoints used (base https://api.dhan.co/v2):
- ``POST /marketfeed/ohlc``: last price + day OHLC, up to 1000 instruments a request, 1 request a second.
- ``POST /charts/intraday``: 1/5/15/25/60-minute candles, up to 90 days a request, 5 years back; 5 data requests a
  second, 100,000 a day.
- ``POST /charts/historical``: daily candles since listing.
- ``GET /profile``: token check (``dataPlan``, ``dataValidity``).
Security IDs come from the public scrip master CSV (column ``SEM_SMST_SECURITY_ID``).
"""
from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from .config import ROOT

BASE = "https://api.dhan.co/v2"
SCRIP_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
CACHE = ROOT / "data" / "dhan"
IST = timezone(timedelta(hours=5, minutes=30))


class DhanError(RuntimeError):
    pass


def credentials() -> tuple[str, str] | None:
    cid, tok = os.environ.get("DHAN_CLIENT_ID"), os.environ.get("DHAN_ACCESS_TOKEN")
    if cid and tok:
        return cid.strip(), tok.strip()
    f = ROOT / "configs" / "dhan.local.yaml"
    if f.exists():
        import yaml
        d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if d.get("client_id") and d.get("access_token"):
            return str(d["client_id"]).strip(), str(d["access_token"]).strip()
    return None


class _Limiter:
    def __init__(self, per_second: float):
        self.gap, self.next, self.lock = 1.0 / per_second, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next:
                time.sleep(self.next - now)
            self.next = max(now, self.next) + self.gap


def security_ids(refresh: bool = False, max_age_days: int = 7) -> pd.Series:
    """NSE equity symbol (e.g. 'RELIANCE') -> Dhan security ID, from the public scrip master (cached weekly)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "nse_eq.csv"
    fresh = f.exists() and (time.time() - f.stat().st_mtime) < max_age_days * 86400
    if refresh or not fresh:
        df = pd.read_csv(SCRIP_URL, usecols=["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID",
                                             "SEM_INSTRUMENT_NAME", "SEM_TRADING_SYMBOL", "SEM_SERIES"], low_memory=False)
        df = df[(df["SEM_EXM_EXCH_ID"] == "NSE") & (df["SEM_SEGMENT"] == "E") & (df["SEM_SERIES"] == "EQ")]
        df[["SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"]].to_csv(f, index=False)
    m = pd.read_csv(f)
    return pd.Series(m["SEM_SMST_SECURITY_ID"].astype(int).to_numpy(), index=m["SEM_TRADING_SYMBOL"].astype(str))


def _sym(ticker: str) -> str:
    return ticker.replace(".NS", "")


class Dhan:
    """Read-only client. ``Dhan.available()`` is None when unconfigured; check ``status()`` before relying on it."""

    def __init__(self, client_id: str, access_token: str, session: requests.Session | None = None):
        self.s = session or requests.Session()
        self.s.headers.update({"access-token": access_token, "client-id": client_id, "Content-Type": "application/json",
                               "Accept": "application/json"})
        self.quote_rate, self.data_rate = _Limiter(1.0), _Limiter(4.0)      # docs: 1/s quotes, 5/s data
        self._ids: pd.Series | None = None

    @classmethod
    def available(cls) -> "Dhan | None":
        c = credentials()
        return cls(*c) if c else None

    @property
    def ids(self) -> pd.Series:
        if self._ids is None:
            self._ids = security_ids()
        return self._ids

    def _post(self, path: str, body: dict, limiter: _Limiter) -> dict:
        limiter.wait()
        r = self.s.post(f"{BASE}{path}", json=body, timeout=20)
        try:
            js = r.json()
        except ValueError:
            js = {}
        if r.status_code != 200 or (isinstance(js, dict) and js.get("status") == "failure"):
            code = js.get("errorCode") or js.get("data", {}).get("errorCode") if isinstance(js, dict) else None
            raise DhanError(f"{path}: HTTP {r.status_code} {code or ''} {str(js)[:200]}")
        return js

    def status(self) -> dict:
        """Token and data-plan check. Never raises: returns {'ok': False, 'why': ...} on any problem."""
        try:
            r = self.s.get(f"{BASE}/profile", timeout=15)
            js = r.json() if r.content else {}
            if r.status_code != 200:
                return {"ok": False, "why": f"HTTP {r.status_code}: {str(js)[:160]}"}
            plan = str(js.get("dataPlan", "")).lower() == "active"
            return {"ok": plan, "why": "" if plan else "the Data API plan is not active on this account",
                    "data_validity": js.get("dataValidity"), "token_validity": js.get("tokenValidity")}
        except Exception as e:                               # noqa: BLE001
            return {"ok": False, "why": str(e)[:200]}

    def ohlc(self, tickers: list[str]) -> pd.DataFrame:
        """Last price and today's open/high/low/close (previous close before the open) for NSE stocks."""
        ids = self.ids
        known = [(t, int(ids[_sym(t)])) for t in tickers if _sym(t) in ids.index]
        rows = []
        for i in range(0, len(known), 1000):
            chunk = known[i:i + 1000]
            js = self._post("/marketfeed/ohlc", {"NSE_EQ": [sid for _, sid in chunk]}, self.quote_rate)
            data = (js.get("data") or {}).get("NSE_EQ", {})
            for t, sid in chunk:
                d = data.get(str(sid))
                if d:
                    o = d.get("ohlc") or {}
                    rows.append({"ticker": t, "price": d.get("last_price"), "open": o.get("open"), "high": o.get("high"),
                                 "low": o.get("low"), "close": o.get("close")})
        return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

    def intraday(self, ticker: str, start: datetime, end: datetime, interval: int = 5) -> pd.DataFrame:
        """Minute candles (IST index; open, high, low, close, volume). ``interval`` in {1, 5, 15, 25, 60}."""
        sid = self.ids.get(_sym(ticker))
        if sid is None:
            return pd.DataFrame()
        body = {"securityId": str(int(sid)), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY",
                "interval": str(interval), "oi": False,
                "fromDate": start.strftime("%Y-%m-%d %H:%M:%S"), "toDate": end.strftime("%Y-%m-%d %H:%M:%S")}
        return _frame(self._post("/charts/intraday", body, self.data_rate))

    def daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        sid = self.ids.get(_sym(ticker))
        if sid is None:
            return pd.DataFrame()
        body = {"securityId": str(int(sid)), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "expiryCode": 0,
                "oi": False, "fromDate": start.isoformat(), "toDate": (end + timedelta(days=1)).isoformat()}
        f = _frame(self._post("/charts/historical", body, self.data_rate))
        if len(f):
            f.index = f.index.normalize().tz_localize(None)
        return f

    def today_bars(self, tickers: list[str], interval: int = 5) -> dict[str, pd.DataFrame]:
        """Today's 5-minute bars for many stocks (one request each, about 4 a second): the live loop's Dhan path."""
        now = datetime.now(IST)
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        out = {}
        for t in tickers:
            try:
                f = self.intraday(t, start, now + timedelta(minutes=1), interval)
            except DhanError:
                continue
            if len(f):
                out[t] = f[f.index.date == now.date()]
        return out


def _frame(js: dict) -> pd.DataFrame:
    """Dhan's columnar candles -> DataFrame indexed by IST timestamp (naive, like the Yahoo bars)."""
    ts = js.get("timestamp") or []
    if not ts:
        return pd.DataFrame()
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
    return pd.DataFrame({k: js.get(k) for k in ("open", "high", "low", "close", "volume")}, index=idx).astype(float)
