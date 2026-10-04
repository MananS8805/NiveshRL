"""Your own alert rules: "TCS price below 2,000", "RELIANCE RSI(14) above 70", "INFY day move below −3%",
"HDFCBANK price above its 50-day average", "SBIN volume above 2× its 20-day average", "new 52-week high".

Stored locally as JSON (``data/alerts.json``, kept out of git). Each rule fires once (then switches itself off) or once
per day, and every firing is logged (``data/alerts_log.json``, last 500). Evaluation is a pure function of a snapshot
of the stock (price, day change, the daily closes and today's volume), so it is easy to test and never fills in
missing data: a rule whose input is unavailable simply does not fire.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .portfolio import _read, _write

PATH = ROOT / "data" / "alerts.json"
LOG = ROOT / "data" / "alerts_log.json"

# kind -> (label, unit shown next to the value, needs a parameter n)
KINDS = {
    "price": ("Price", "₹"),
    "day_pct": ("Day move %", "%"),
    "sma": ("Price vs SMA (value = days)", "days"),
    "rsi": ("RSI (14)", ""),
    "vol_ratio": ("Volume ÷ 20-day average", "×"),
    "high52": ("52-week high / low (above = new high, below = new low)", ""),
}
OPS = {"above": "≥", "below": "≤"}


@dataclass
class Rule:
    ticker: str
    kind: str
    op: str                       # above | below
    value: float = 0.0
    repeat: str = "once"          # once | daily
    note: str = ""
    active: bool = True
    last_fired: str | None = None
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def describe(self) -> str:
        t = self.ticker.replace(".NS", "")
        if self.kind == "price":
            return f"{t} price {OPS[self.op]} ₹{self.value:,.2f}"
        if self.kind == "day_pct":
            return f"{t} day move {OPS[self.op]} {self.value:+.1f}%"
        if self.kind == "sma":
            return f"{t} price {self.op} its {int(self.value)}-day average"
        if self.kind == "rsi":
            return f"{t} RSI(14) {OPS[self.op]} {self.value:g}"
        if self.kind == "vol_ratio":
            return f"{t} volume {OPS[self.op]} {self.value:g}× its 20-day average"
        if self.kind == "high52":
            return f"{t} makes a new 52-week {'high' if self.op == 'above' else 'low'}"
        return f"{t} {self.kind} {self.op} {self.value}"


def load(path: Path | None = None) -> list[Rule]:
    path = path or PATH
    return [Rule(**d) for d in _read(path).get("rules", [])]


def save(rules: list[Rule], path: Path | None = None) -> None:
    path = path or PATH
    _write(path, {"rules": [asdict(r) for r in rules]})


def add(rule: Rule, path: Path | None = None) -> list[Rule]:
    path = path or PATH
    if rule.kind not in KINDS or rule.op not in OPS:
        raise ValueError(f"unknown alert {rule.kind} {rule.op}")
    rules = load(path) + [rule]
    save(rules, path)
    return rules


def remove(rule_id: str, path: Path | None = None) -> list[Rule]:
    path = path or PATH
    rules = [r for r in load(path) if r.id != rule_id]
    save(rules, path)
    return rules


def _rsi(c: np.ndarray, n: int = 14) -> float:
    s = pd.Series(c)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    return float(100.0 if dn == 0 else 100 - 100 / (1 + up / dn))


def measure(rule: Rule, snap: dict) -> tuple[float | None, float | None]:
    """(observed value, threshold) for a rule, or (None, None) when the input is missing.

    snap: price, day_change (fraction), closes (pd.Series of previous daily closes, not including today),
    volume (today's volume so far), avg_volume (20-day average)."""
    price = snap.get("price")
    closes = snap.get("closes")
    if rule.kind == "price":
        return price, rule.value
    if rule.kind == "day_pct":
        dc = snap.get("day_change")
        return (None, None) if dc is None else (dc * 100, rule.value)
    if price is None or closes is None or not len(closes):
        return None, None
    c = np.r_[closes.dropna().to_numpy(dtype=float), price]          # today's live price as the latest close
    if rule.kind == "sma":
        n = int(rule.value)
        return (None, None) if len(c) < n or n < 2 else (price, float(c[-n:].mean()))
    if rule.kind == "rsi":
        return (None, None) if len(c) < 30 else (_rsi(c), rule.value)
    if rule.kind == "vol_ratio":
        v, av = snap.get("volume"), snap.get("avg_volume")
        return (None, None) if not v or not av else (v / av, rule.value)
    if rule.kind == "high52":
        prev = c[-253:-1]
        if len(prev) < 200:
            return None, None
        return price, float(prev.max() if rule.op == "above" else prev.min())
    return None, None


def triggered(rule: Rule, snap: dict) -> bool:
    obs, thr = measure(rule, snap)
    if obs is None or thr is None or (isinstance(obs, float) and math.isnan(obs)):
        return False
    return obs >= thr if rule.op == "above" else obs <= thr


def check(rules: list[Rule], snapshot, now: datetime | None = None) -> tuple[list[tuple[Rule, str]], bool]:
    """Evaluate active rules. ``snapshot(ticker) -> dict``. Returns the firings and whether any rule changed state
    (so the caller saves). 'once' rules switch off after firing; 'daily' rules fire at most once per calendar day."""
    now = now or datetime.now()
    today = now.date().isoformat()
    fired, changed = [], False
    cache: dict[str, dict] = {}
    for r in rules:
        if not r.active or (r.repeat == "daily" and r.last_fired and r.last_fired[:10] == today):
            continue
        snap = cache.setdefault(r.ticker, snapshot(r.ticker) or {})
        if triggered(r, snap):
            obs, thr = measure(r, snap)
            if r.kind == "rsi":
                msg = f"{r.describe()} (now {obs:.0f})"
            elif r.kind in ("day_pct",):
                msg = f"{r.describe()} (now {obs:+.1f}%)"
            elif r.kind == "vol_ratio":
                msg = f"{r.describe()} (now {obs:.1f}×)"
            elif r.kind == "sma":
                msg = f"{r.describe()} (₹{obs:,.2f} vs ₹{thr:,.2f})"
            else:
                msg = f"{r.describe()} (₹{obs:,.2f})"
            fired.append((r, msg + (f" · {r.note}" if r.note else "")))
            r.last_fired = now.isoformat(timespec="seconds")
            if r.repeat == "once":
                r.active = False
            changed = True
    return fired, changed


def log(entries: list[tuple[str, str]], path: Path | None = None, keep: int = 500) -> None:
    path = path or LOG
    if not entries:
        return
    old = _read(path).get("log", [])
    now = datetime.now().isoformat(timespec="seconds")
    _write(path, {"log": ([{"at": now, "ticker": t, "text": m} for t, m in entries] + old)[:keep]})


def load_log(path: Path | None = None) -> list[dict]:
    path = path or LOG
    return _read(path).get("log", [])
