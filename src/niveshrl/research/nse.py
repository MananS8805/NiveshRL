"""Optional data from NSE's public website APIs (off by default; Options → Optional NSE data).

These endpoints are undocumented and change without notice (NSE moved the option chain to ``option-chain-v3`` in 2025,
and its pledge endpoint currently returns nothing), they need a browser-like session with cookies, and NSE blocks
clients that call too often. So every call here is slow-paced (≥ 0.7 s apart), cached, and fails loudly: a missing
answer is shown as "unavailable", never filled in or estimated.

- ``fii_dii``: the latest day's FII/FPI and DII cash-market buy/sell/net (₹ crore). NSE only serves the latest day, so
  ``fii_dii_history`` keeps every day this app has seen in ``data/nse/fii_dii.csv`` (history starts when you turn it on).
- ``large_deals``: the latest day's bulk and block deals.
- ``shareholding``: promoter & promoter group vs public holding % by quarter (about 5 years).
- ``option_chain``: calls and puts per strike for an expiry (stocks and NIFTY/BANKNIFTY), with ``chain_stats``: put-call
  ratio, max pain, the strikes with the most call/put open interest and ATM implied volatility.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pandas as pd

from ..config import ROOT

BASE = "https://www.nseindia.com"
DIR = ROOT / "data" / "nse"
INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/128.0 Safari/537.36",
           "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://www.nseindia.com/option-chain"}
MIN_GAP = 0.7


class NSEUnavailable(RuntimeError):
    pass


class Client:
    """One cookie session, paced calls, one retry with fresh cookies on 401/403."""

    def __init__(self, session=None):
        self._s = session
        self._lock = threading.Lock()
        self._last = 0.0
        self._warm = 0.0

    def _session(self):
        if self._s is None:
            import requests
            self._s = requests.Session()
            self._s.headers.update(HEADERS)
        return self._s

    def _pace(self) -> None:
        wait = MIN_GAP - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _warmup(self, force: bool = False) -> None:
        if force or time.time() - self._warm > 600:
            self._pace()
            self._session().get(BASE + "/option-chain", timeout=15)
            self._warm = time.time()

    def get(self, path: str):
        with self._lock:
            for attempt in (0, 1):
                self._warmup(force=attempt == 1)
                self._pace()
                r = self._session().get(BASE + path, timeout=20)
                if r.status_code in (401, 403) and attempt == 0:
                    continue
                if r.status_code != 200:
                    raise NSEUnavailable(f"NSE answered HTTP {r.status_code} for {path.split('?')[0]}")
                try:
                    return r.json()
                except (ValueError, json.JSONDecodeError):
                    raise NSEUnavailable(f"NSE sent a non-JSON page for {path.split('?')[0]}") from None
        raise NSEUnavailable("NSE refused the session")


_client: Client | None = None


def client() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


def _num(v) -> float:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- flows and deals
def fii_dii(c: Client | None = None) -> pd.DataFrame:
    j = (c or client()).get("/api/fiidiiTradeReact")
    if not isinstance(j, list) or not j:
        raise NSEUnavailable("NSE returned no FII/DII data")
    rows = [{"date": pd.to_datetime(x.get("date"), format="%d-%b-%Y", errors="coerce"),
             "category": "FII" if "FII" in str(x.get("category", "")) else str(x.get("category", "")),
             "buy": _num(x.get("buyValue")), "sell": _num(x.get("sellValue")), "net": _num(x.get("netValue"))} for x in j]
    return pd.DataFrame(rows).dropna(subset=["date"])


def fii_dii_history(latest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every FII/DII day seen so far (latest merged in and saved)."""
    path = DIR / "fii_dii.csv"
    old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=["date", "category", "buy",
                                                                                              "sell", "net"])
    if latest is not None and len(latest):
        old = pd.concat([old, latest]).drop_duplicates(["date", "category"], keep="last").sort_values("date")
        DIR.mkdir(parents=True, exist_ok=True)
        old.to_csv(path, index=False)
    return old.reset_index(drop=True)


def large_deals(c: Client | None = None) -> dict[str, pd.DataFrame]:
    j = (c or client()).get("/api/snapshot-capital-market-largedeal")
    out = {"as_of": j.get("as_on_date")}
    for key, name in (("BULK_DEALS_DATA", "bulk"), ("BLOCK_DEALS_DATA", "block")):
        rows = j.get(key) or []
        out[name] = pd.DataFrame([{"Date": x.get("date"), "Symbol": x.get("symbol"), "Name": x.get("name"),
                                   "Client": x.get("clientName"), "Side": x.get("buySell"),
                                   "Quantity": _num(x.get("qty")), "Price ₹": _num(x.get("watp")),
                                   "Value ₹ Cr": _num(x.get("qty")) * _num(x.get("watp")) / 1e7} for x in rows])
    return out


# --------------------------------------------------------------------------- ownership
def shareholding(symbol: str, c: Client | None = None) -> pd.DataFrame:
    j = (c or client()).get(f"/api/corporate-share-holdings-master?index=equities&symbol={symbol}")
    if not isinstance(j, list) or not j:
        raise NSEUnavailable(f"NSE returned no shareholding data for {symbol}")
    rows = [{"Quarter": pd.to_datetime(x.get("date"), format="%d-%b-%Y", errors="coerce"),
             "Promoter %": _num(x.get("pr_and_prgrp")), "Public %": _num(x.get("public_val")),
             "Employee trusts %": _num(x.get("employeeTrusts")), "Filed": x.get("broadcastDate")} for x in j]
    df = pd.DataFrame(rows).dropna(subset=["Quarter"]).drop_duplicates("Quarter", keep="first")
    df = df.sort_values("Quarter")
    df["Promoter change (pp)"] = df["Promoter %"].diff()
    return df.set_index("Quarter")


# --------------------------------------------------------------------------- options
def expiries(symbol: str, c: Client | None = None) -> list[str]:
    j = (c or client()).get(f"/api/option-chain-contract-info?symbol={symbol}")
    ex = j.get("expiryDates") or []
    if not ex:
        raise NSEUnavailable(f"{symbol} has no options on NSE (not an F&O stock?)")
    return ex


def option_chain(symbol: str, expiry: str | None = None, c: Client | None = None) -> tuple[pd.DataFrame, dict]:
    c = c or client()
    expiry = expiry or expiries(symbol, c)[0]
    kind = "Indices" if symbol in INDICES else "Equity"
    j = c.get(f"/api/option-chain-v3?type={kind}&symbol={symbol}&expiry={expiry}")
    rec = j.get("records") or {}
    data = rec.get("data") or []
    if not data:
        raise NSEUnavailable(f"NSE returned an empty option chain for {symbol} {expiry}")
    rows = []
    for d in data:
        ce, pe = d.get("CE") or {}, d.get("PE") or {}
        k = _num(d.get("strikePrice", ce.get("strikePrice", pe.get("strikePrice"))))
        rows.append({"Call OI": _num(ce.get("openInterest")), "Call ΔOI": _num(ce.get("changeinOpenInterest")),
                     "Call volume": _num(ce.get("totalTradedVolume")), "Call IV": _num(ce.get("impliedVolatility")),
                     "Call LTP": _num(ce.get("lastPrice")), "Strike": k,
                     "Put LTP": _num(pe.get("lastPrice")), "Put IV": _num(pe.get("impliedVolatility")),
                     "Put volume": _num(pe.get("totalTradedVolume")), "Put ΔOI": _num(pe.get("changeinOpenInterest")),
                     "Put OI": _num(pe.get("openInterest"))})
    df = pd.DataFrame(rows).dropna(subset=["Strike"]).sort_values("Strike").reset_index(drop=True)
    meta = {"symbol": symbol, "expiry": expiry, "underlying": _num(rec.get("underlyingValue")),
            "timestamp": rec.get("timestamp"), "expiries": rec.get("expiryDates") or [expiry]}
    return df, meta


def max_pain(df: pd.DataFrame) -> float:
    """The expiry price at which option buyers' total payoff (and so option writers' total payout) is smallest."""
    k = df["Strike"].to_numpy(dtype=float)
    co, po = np.nan_to_num(df["Call OI"].to_numpy(dtype=float)), np.nan_to_num(df["Put OI"].to_numpy(dtype=float))
    pay = [float((co * np.maximum(s - k, 0)).sum() + (po * np.maximum(k - s, 0)).sum()) for s in k]
    return float(k[int(np.argmin(pay))]) if len(k) else float("nan")


def chain_stats(df: pd.DataFrame, underlying: float) -> dict:
    co, po = df["Call OI"].fillna(0), df["Put OI"].fillna(0)
    atm_i = int((df["Strike"] - underlying).abs().idxmin()) if len(df) and np.isfinite(underlying) else None
    ivs = [v for v in (df.loc[atm_i, "Call IV"], df.loc[atm_i, "Put IV"]) if np.isfinite(v) and v > 0] if atm_i is not None else []
    return {"pcr": float(po.sum() / co.sum()) if co.sum() > 0 else float("nan"),
            "pcr_change": float(df["Put ΔOI"].fillna(0).sum() / df["Call ΔOI"].fillna(0).sum())
            if df["Call ΔOI"].fillna(0).sum() > 0 else float("nan"),
            "max_pain": max_pain(df), "call_wall": float(df.loc[co.idxmax(), "Strike"]) if co.sum() > 0 else float("nan"),
            "put_wall": float(df.loc[po.idxmax(), "Strike"]) if po.sum() > 0 else float("nan"),
            "atm_strike": float(df.loc[atm_i, "Strike"]) if atm_i is not None else float("nan"),
            "atm_iv": float(np.mean(ivs)) / 100 if ivs else float("nan"),
            "call_oi": float(co.sum()), "put_oi": float(po.sum())}
