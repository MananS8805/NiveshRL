"""Framework-free live price feed (Yahoo Finance streaming WebSocket).

Used by the desktop app (and wrapped by ``dashboard/live.py`` for Streamlit).
Ticks go into the C++ ``TickStore`` when the core is installed (latest quotes,
change tracking, 1-minute bars, fixed memory); otherwise into a plain dict.
One daemon thread per feed; it reconnects with exponential backoff forever.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
INDEX_SYMBOLS = ["^NSEI", "^INDIAVIX"]


class Feed:
    def __init__(self, symbols: list[str], bar_capacity: int = 400):
        self.symbols = list(symbols) + [s for s in INDEX_SYMBOLS if s not in symbols]
        try:
            import niveshrl_core
            self.store = niveshrl_core.TickStore(self.symbols, bar_capacity)
        except ImportError:
            self.store = None
        self._dict: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.last_msg_at = 0.0
        self.connected = False
        self.error: str | None = None
        self.n_msgs = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ws = None

    # ------------------------------------------------------------------ stream
    def _on_message(self, msg: dict) -> None:
        sid, px = msg.get("id"), msg.get("price")
        if not sid or px is None:
            return
        ts = int(msg.get("time") or time.time() * 1000)
        chg = float(msg.get("change_percent") or 0.0)
        vol = float(msg.get("day_volume") or 0.0)
        if self.store is not None:
            self.store.update(sid, ts, float(px), chg, vol)
        with self._lock:
            self._dict[sid] = {"price": float(px), "change_percent": chg, "day_volume": vol, "ts": ts}
            self.last_msg_at = time.time()
            self.n_msgs += 1

    def _run(self) -> None:
        import yfinance as yf

        backoff = 2
        while not self._stop.is_set():
            try:
                self._ws = yf.WebSocket(verbose=False)
                self._ws.subscribe(self.symbols)
                self.connected, self.error, backoff = True, None, 2
                self._ws.listen(self._on_message)
            except Exception as e:
                self.error = f"{type(e).__name__}: {str(e)[:120]}"
            self.connected = False
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 60)

    def start(self) -> "Feed":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="niveshrl-feed", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ reads
    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._dict)

    def quote(self, symbol: str) -> dict | None:
        with self._lock:
            return self._dict.get(symbol)

    def bars(self, symbol: str) -> list:
        return self.store.bars(symbol) if self.store is not None else []

    def status(self) -> tuple[str, str]:
        """(text, level) where level is 'live', 'stale', 'closed' or 'offline'."""
        if self.last_msg_at:
            ago = time.time() - self.last_msg_at
            when = datetime.fromtimestamp(self.last_msg_at, IST).strftime("%H:%M:%S")
            n = sum(1 for s in self.snapshot() if s not in INDEX_SYMBOLS)
            if ago < 90:
                return f"LIVE {when} IST · {n} streaming", "live"
            return f"STALE since {when} IST", "stale"
        if not market_open():
            return "MARKET CLOSED · last close", "closed"
        return ("CONNECTING…" if self.connected or not self.error else f"OFFLINE · {self.error}"), "offline"


def market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= t <= 15 * 60 + 30
