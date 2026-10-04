"""Manual paper trading: place pretend delivery orders and watch them fill against live prices. Nothing is ever sent
to a broker.

- Orders: BUY or SELL, MARKET or LIMIT, with an optional protective stop and target on buys (a bracket). A market order
  fills at the next live price plus half the spread; a limit buy fills when the live price trades at or below the limit
  (at the limit, or better if it gaps). Orders placed while the market is closed wait for the next live price.
- Positions: average cost per stock. A bracket's stop sells everything when the live price is at or below it (at the
  live price, so a gap through the stop fills worse); its target sells everything at or above it.
- Costs: the same delivery cost model as the trade plans (``configs/costs_india.yaml``), charged on each round trip;
  realised P&L is after costs, and R uses the stop at entry.

Stored as JSON in ``data/paper.json`` (atomic writes, previous version kept as .bak).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import ROOT, load_yaml
from .portfolio import _read, _write
from .research.plans import delivery_costs

PATH = ROOT / "data" / "paper.json"


@dataclass
class Order:
    ticker: str
    side: str                       # BUY | SELL
    qty: int
    type: str = "MARKET"            # MARKET | LIMIT
    limit: float | None = None
    stop: float | None = None       # buys only: protective stop for the position
    target: float | None = None     # buys only
    status: str = "OPEN"            # OPEN | FILLED | CANCELLED | REJECTED
    note: str = ""
    fill_price: float | None = None
    filled_at: str | None = None
    placed_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))


@dataclass
class Position:
    ticker: str
    qty: int
    avg: float
    stop: float | None = None
    target: float | None = None
    opened_at: str = ""
    risk_per_share: float | None = None


@dataclass
class Closed:
    ticker: str
    qty: int
    entry: float
    exit: float
    opened_at: str
    closed_at: str
    reason: str
    costs: float
    net: float
    r: float | None


@dataclass
class Account:
    start_cash: float = 100_000.0
    cash: float = 100_000.0
    orders: list = field(default_factory=list)
    positions: dict = field(default_factory=dict)
    closed: list = field(default_factory=list)


def load(path: Path | None = None, start_cash: float = 100_000.0) -> Account:
    path = path or PATH
    raw = _read(path)
    if not raw:
        return Account(start_cash, start_cash)
    return Account(raw.get("start_cash", start_cash), raw.get("cash", start_cash),
                   [Order(**o) for o in raw.get("orders", [])],
                   {k: Position(**v) for k, v in raw.get("positions", {}).items()},
                   [Closed(**c) for c in raw.get("closed", [])])


def save(acct: Account, path: Path | None = None) -> None:
    path = path or PATH
    _write(path, {"start_cash": acct.start_cash, "cash": acct.cash, "orders": [asdict(o) for o in acct.orders[-500:]],
                  "positions": {k: asdict(v) for k, v in acct.positions.items()},
                  "closed": [asdict(c) for c in acct.closed[-1000:]]})


def reset(start_cash: float, path: Path | None = None) -> Account:
    path = path or PATH
    acct = Account(start_cash, start_cash)
    save(acct, path)
    return acct


def place(acct: Account, order: Order) -> Order:
    """Validate and queue an order (it fills in ``process``). Invalid orders are kept as REJECTED with the reason."""
    why = None
    if order.qty <= 0:
        why = "quantity must be positive"
    elif order.side not in ("BUY", "SELL") or order.type not in ("MARKET", "LIMIT"):
        why = "unknown side or type"
    elif order.type == "LIMIT" and not (order.limit and order.limit > 0):
        why = "a limit order needs a limit price"
    elif order.side == "SELL" and acct.positions.get(order.ticker, Position(order.ticker, 0, 0)).qty < order.qty:
        why = "you can only sell shares you hold (no short selling in delivery)"
    elif order.side == "BUY" and order.stop is not None and order.limit and order.stop >= order.limit:
        why = "the stop must be below the buy price"
    if why:
        order.status, order.note = "REJECTED", why
    acct.orders.append(order)
    return order


def cancel(acct: Account, order_id: str) -> None:
    for o in acct.orders:
        if o.id == order_id and o.status == "OPEN":
            o.status = "CANCELLED"


def _close(acct: Account, pos: Position, qty: int, px: float, when: str, reason: str, rates) -> Closed:
    costs = delivery_costs(pos.avg, px, qty, rates)["total"]
    net = (px - pos.avg) * qty - costs
    r = net / (pos.risk_per_share * qty) if pos.risk_per_share else None
    acct.cash += px * qty
    c = Closed(pos.ticker, qty, pos.avg, px, pos.opened_at, when, reason, costs, net, r)
    acct.closed.append(c)
    pos.qty -= qty
    if pos.qty <= 0:
        acct.positions.pop(pos.ticker, None)
    return c


def process(acct: Account, quotes: dict[str, float], now: datetime | None = None,
            rates: dict | None = None) -> list[str]:
    """Fill open orders and trigger bracket stops/targets against live prices. Returns human-readable events."""
    now = now or datetime.now()
    when = now.isoformat(timespec="seconds")
    rates = rates or load_yaml("configs/costs_india.yaml")
    half = rates.get("half_spread", 0.0)
    events = []
    for o in acct.orders:
        if o.status != "OPEN" or o.ticker not in quotes:
            continue
        px = float(quotes[o.ticker])
        if o.side == "BUY":
            if o.type == "LIMIT" and px > o.limit:
                continue
            fill = min(px, o.limit) if o.type == "LIMIT" else px * (1 + half)
            if fill * o.qty > acct.cash:
                o.status, o.note = "REJECTED", f"not enough paper cash (needs ₹{fill * o.qty:,.0f})"
                events.append(f"Rejected BUY {o.qty} {o.ticker}: {o.note}")
                continue
            acct.cash -= fill * o.qty
            pos = acct.positions.get(o.ticker)
            if pos is None:
                pos = acct.positions[o.ticker] = Position(o.ticker, 0, 0.0, opened_at=when)
            pos.avg = (pos.avg * pos.qty + fill * o.qty) / (pos.qty + o.qty)
            pos.qty += o.qty
            if o.stop is not None:
                pos.stop = o.stop
                pos.risk_per_share = max(pos.avg - o.stop, 0.0) or None
            if o.target is not None:
                pos.target = o.target
        else:
            if o.type == "LIMIT" and px < o.limit:
                continue
            pos = acct.positions.get(o.ticker)
            if pos is None or pos.qty < o.qty:
                o.status, o.note = "REJECTED", "position no longer large enough"
                continue
            fill = max(px, o.limit) if o.type == "LIMIT" else px * (1 - half)
            _close(acct, pos, o.qty, fill, when, "manual sell", rates)
        o.status, o.fill_price, o.filled_at = "FILLED", fill, when
        events.append(f"Filled {o.side} {o.qty} {o.ticker.replace('.NS', '')} at ₹{fill:,.2f}")
    for t, pos in list(acct.positions.items()):
        if t not in quotes:
            continue
        px = float(quotes[t])
        if pos.stop is not None and px <= pos.stop:
            c = _close(acct, pos, pos.qty, px * (1 - half), when, "stop", rates)
            events.append(f"Stop hit: sold {c.qty} {t.replace('.NS', '')} at ₹{c.exit:,.2f} (net ₹{c.net:+,.0f})")
        elif pos.target is not None and px >= pos.target:
            c = _close(acct, pos, pos.qty, max(pos.target, px * (1 - half)), when, "target", rates)
            events.append(f"Target hit: sold {c.qty} {t.replace('.NS', '')} at ₹{c.exit:,.2f} (net ₹{c.net:+,.0f})")
    return events


def summary(acct: Account, quotes: dict[str, float]) -> dict:
    mv = sum(p.qty * quotes.get(t, p.avg) for t, p in acct.positions.items())
    cost = sum(p.qty * p.avg for p in acct.positions.values())
    realised = sum(c.net for c in acct.closed)
    rs = [c.r for c in acct.closed if c.r is not None]
    return {"equity": acct.cash + mv, "cash": acct.cash, "invested": cost, "unrealised": mv - cost,
            "realised": realised, "return": (acct.cash + mv) / acct.start_cash - 1 if acct.start_cash else 0.0,
            "trades": len(acct.closed), "win_rate": (sum(c.net > 0 for c in acct.closed) / len(acct.closed))
            if acct.closed else float("nan"), "avg_r": sum(rs) / len(rs) if rs else float("nan")}
