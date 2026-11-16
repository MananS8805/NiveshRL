"""Broker extension point. Every screen that places or tracks orders talks to a ``Broker``, never to a broker SDK.

Today the only working implementation is ``PaperBroker`` (the existing paper account in ``paper.py``). Real brokers
are registered as named slots (Zerodha Kite Connect, Upstox, Fyers, Angel One SmartAPI) so the wiring point exists,
but they are deliberately inert: they report themselves as not configured and **refuse every order**. NiveshRL is
paper-only; a live adapter must be written, reviewed and switched on as a separate, explicit change.

Notes for whoever writes a live adapter later (SEBI retail algo framework, in force from 1 April 2026; verify the
current circular before building): orders must come from a broker-issued API key and a static IP whitelisted by the
broker, the session needs a fresh 2FA login every trading day, and staying under 10 orders per second keeps a client
a regular API user rather than a registered algo provider.

Config: ``configs/broker.yaml`` (``provider: paper``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from . import paper
from .config import load_yaml


class BrokerNotConfigured(RuntimeError):
    """The selected broker has no working adapter yet."""


class LiveTradingDisabled(RuntimeError):
    """Raised by every live broker slot: NiveshRL never sends real orders."""


@runtime_checkable
class Broker(Protocol):
    name: str
    is_live: bool

    def status(self) -> dict: ...
    def place(self, order: paper.Order) -> paper.Order: ...
    def cancel(self, order_id: str) -> None: ...
    def orders(self) -> list[paper.Order]: ...
    def positions(self) -> dict[str, paper.Position]: ...
    def sync(self, quotes: dict[str, float]) -> list[str]: ...


class PaperBroker:
    """The paper account behind the Broker interface (fills against live quotes, costs included)."""

    name, is_live = "paper", False

    def __init__(self, path: Path | None = None, start_cash: float = 100_000.0):
        self.path = path
        self.acct = paper.load(path, start_cash)

    def status(self) -> dict:
        return {"provider": self.name, "live": False, "connected": True, "detail": "Paper account: nothing is sent to a broker."}

    def place(self, order: paper.Order) -> paper.Order:
        o = paper.place(self.acct, order)
        paper.save(self.acct, self.path)
        return o

    def cancel(self, order_id: str) -> None:
        paper.cancel(self.acct, order_id)
        paper.save(self.acct, self.path)

    def orders(self) -> list[paper.Order]:
        return list(self.acct.orders)

    def positions(self) -> dict[str, paper.Position]:
        return dict(self.acct.positions)

    def sync(self, quotes: dict[str, float]) -> list[str]:
        events = paper.process(self.acct, quotes)
        paper.save(self.acct, self.path)
        return events


@dataclass
class LiveBrokerSlot:
    """Placeholder for a real broker. Status only; every trading call raises."""

    name: str
    display: str
    docs: str
    is_live: bool = True

    def status(self) -> dict:
        return {"provider": self.name, "live": True, "connected": False,
                "detail": f"{self.display} integration is an extension slot only (not built). See {self.docs}."}

    def place(self, order: paper.Order) -> paper.Order:
        raise LiveTradingDisabled(f"{self.display}: NiveshRL is paper-only and never sends real orders.")

    def cancel(self, order_id: str) -> None:
        raise LiveTradingDisabled(f"{self.display}: NiveshRL is paper-only.")

    def orders(self) -> list[paper.Order]:
        raise BrokerNotConfigured(f"{self.display} adapter not built yet.")

    def positions(self) -> dict[str, paper.Position]:
        raise BrokerNotConfigured(f"{self.display} adapter not built yet.")

    def sync(self, quotes: dict[str, float]) -> list[str]:
        raise BrokerNotConfigured(f"{self.display} adapter not built yet.")


SLOTS = {
    "zerodha": LiveBrokerSlot("zerodha", "Zerodha Kite Connect", "kite.trade/docs/connect"),
    "upstox": LiveBrokerSlot("upstox", "Upstox API", "upstox.com/developer/api-documentation"),
    "fyers": LiveBrokerSlot("fyers", "Fyers API", "myapi.fyers.in/docsv3"),
    "angelone": LiveBrokerSlot("angelone", "Angel One SmartAPI", "smartapi.angelbroking.com/docs"),
}


def get_broker(cfg: dict | None = None) -> Broker:
    """The configured broker; unknown providers fall back to paper so nothing ever routes somewhere unexpected."""
    try:
        cfg = cfg if cfg is not None else load_yaml("configs/broker.yaml")
    except (OSError, ValueError):
        cfg = {}
    provider = str(cfg.get("provider", "paper")).lower()
    return SLOTS.get(provider) or PaperBroker()
