"""Real-time prices from Yahoo Finance's streaming WebSocket.

One background thread per server process subscribes to every NIFTY 200 stock
plus NIFTY 50 and India VIX, and keeps the latest tick per symbol. Screens read
the snapshot; nothing blocks the UI.

Caveats, shown in the app:
- Yahoo's stream is not exchange-grade. Quotes can lag NSE and some symbols
  tick rarely.
- Outside market hours (NSE 09:15-15:30 IST, Mon-Fri) no ticks arrive. The
  app then falls back to the last close.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import streamlit as st

IST = timezone(timedelta(hours=5, minutes=30))
INDEX_SYMBOLS = ["^NSEI", "^INDIAVIX"]


@dataclass
class LiveFeed:
    symbols: list[str]
    ticks: dict[str, dict] = field(default_factory=dict)
    last_msg_at: float = 0.0
    connected: bool = False
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _on_message(self, msg: dict) -> None:
        sid = msg.get("id")
        if not sid or msg.get("price") is None:
            return
        with self.lock:
            self.ticks[sid] = msg
            self.last_msg_at = time.time()

    def _run(self) -> None:
        import yfinance as yf

        backoff = 2
        while True:                                   # reconnect forever; the stream drops now and then
            try:
                ws = yf.WebSocket(verbose=False)
                ws.subscribe(self.symbols)
                self.connected, self.error, backoff = True, None, 2
                ws.listen(self._on_message)           # blocks until the socket closes
            except Exception as e:                    # network error, server close
                self.error = f"{type(e).__name__}: {str(e)[:120]}"
            self.connected = False
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def start(self) -> "LiveFeed":
        threading.Thread(target=self._run, name="niveshrl-live", daemon=True).start()
        return self

    def snapshot(self) -> dict[str, dict]:
        with self.lock:
            return dict(self.ticks)

    def price(self, symbol: str) -> float | None:
        t = self.ticks.get(symbol)
        return float(t["price"]) if t else None


@st.cache_resource(show_spinner=False)
def feed(symbols: tuple[str, ...]) -> LiveFeed:
    """Started once per server process and shared by every browser session."""
    return LiveFeed(list(symbols) + INDEX_SYMBOLS).start()


@st.cache_data(ttl=120, show_spinner=False)
def quote_snapshot(symbols: tuple[str, ...]):
    """Today's latest price and the previous close for every symbol, from one batch request.

    This fills in stocks that haven't ticked on the stream yet, so nothing falls
    back to the (possibly days-old) stored panel. Refreshed every 2 minutes.
    Returns a DataFrame indexed by symbol with columns last, prev_close, or None.
    """
    import pandas as pd
    import yfinance as yf

    try:
        px = yf.download(list(symbols), period="5d", interval="1d", auto_adjust=False, progress=False,
                         threads=True)["Close"]
    except Exception:
        return None
    px = px.dropna(how="all")
    if len(px) < 2:
        return None
    rows = {}
    for s in px.columns:
        c = px[s].dropna()
        if len(c) >= 2:
            rows[s] = (float(c.iloc[-1]), float(c.iloc[-2]))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["last", "prev_close"])


def market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= t <= 15 * 60 + 30


def status_text(f: LiveFeed) -> tuple[str, str]:
    """(label, css class) for the status tape."""
    snap = f.snapshot()
    n = sum(1 for s in snap if s not in INDEX_SYMBOLS)
    if f.last_msg_at:
        ago = time.time() - f.last_msg_at
        when = datetime.fromtimestamp(f.last_msg_at, IST).strftime("%H:%M:%S")
        if ago < 90:
            return f"LIVE · last tick {when} IST · {n} stocks streaming", "pos"
        return f"STALE · last tick {when} IST", "amb"
    if not market_open():
        return "MARKET CLOSED · showing last close", "amb"
    return ("CONNECTING…" if f.connected or not f.error else f"OFFLINE · {f.error}"), "amb"
