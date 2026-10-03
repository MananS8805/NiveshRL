"""Personal watchlist / wish list: tiers, notes, target prices and alert rules.

Stored locally as JSON (``data/watchlist.json``, single user, kept out of git).
Shared by the Streamlit app and the desktop app.

Tiers: "must" (★ must have) and "preferred" (☆ preferred).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import ROOT

PATH = ROOT / "data" / "watchlist.json"
TIERS = {"must": "★ Must have", "preferred": "☆ Preferred"}


@dataclass
class Item:
    ticker: str
    tier: str = "preferred"
    note: str = ""
    target_buy: float | None = None      # alert when price falls to or below
    target_sell: float | None = None     # alert when price rises to or above
    added: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def load(path: Path = PATH) -> dict[str, Item]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {d["ticker"]: Item(**d) for d in raw.get("items", [])}


def save(items: dict[str, Item], path: Path = PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")                     # write-then-rename: never a half-written file
    tmp.write_text(json.dumps({"items": [asdict(i) for i in items.values()]}, indent=2), encoding="utf-8")
    tmp.replace(path)


def add(ticker: str, tier: str = "preferred", note: str = "", target_buy: float | None = None,
        target_sell: float | None = None, path: Path = PATH) -> dict[str, Item]:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {list(TIERS)}")
    items = load(path)
    old = items.get(ticker)
    items[ticker] = Item(ticker, tier, note or (old.note if old else ""), target_buy, target_sell,
                         added=old.added if old else datetime.now().isoformat(timespec="seconds"))
    save(items, path)
    return items


def remove(ticker: str, path: Path = PATH) -> dict[str, Item]:
    items = load(path)
    items.pop(ticker, None)
    save(items, path)
    return items


def alerts(item: Item, price: float | None, day_change: float | None, sentiment: float | None = None,
           prev_sentiment: float | None = None, days_to_earnings: int | None = None,
           in_top_n: bool | None = None, was_in_top_n: bool | None = None) -> list[str]:
    """Alert badges for one watchlist stock. Every input is optional; missing data raises no alert."""
    out = []
    if price is not None:
        if item.target_buy is not None and price <= item.target_buy:
            out.append(f"at/below buy target ₹{item.target_buy:,.2f}")
        if item.target_sell is not None and price >= item.target_sell:
            out.append(f"at/above sell target ₹{item.target_sell:,.2f}")
    if day_change is not None and abs(day_change) >= 0.03:
        out.append(f"moved {day_change:+.1%} today")
    if sentiment is not None and prev_sentiment is not None and (sentiment > 0.2) != (prev_sentiment > 0.2) \
            and abs(sentiment - prev_sentiment) > 0.3:
        out.append("news sentiment flipped " + ("positive" if sentiment > prev_sentiment else "negative"))
    if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
        out.append(f"results in {days_to_earnings} day{'s' if days_to_earnings != 1 else ''}")
    if in_top_n is not None and was_in_top_n is not None and in_top_n != was_in_top_n:
        out.append("entered tomorrow's top list" if in_top_n else "dropped out of tomorrow's top list")
    return out
