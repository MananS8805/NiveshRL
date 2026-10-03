"""Your holdings (imported from a Kite holdings CSV) and your trade journal. Local JSON only.

- data/holdings.json: ticker, qty, avg cost, plus each holding's exit line (close − 3 × ATR, only raised).
- data/journal.json: trades you actually took (entry, stop, exit, qty); R after delivery costs is computed
  exactly like research/plans.r_multiple, so your real results sit next to the backtests.
Writes are atomic (write-then-rename) and keep the previous version as *.bak.
"""
from __future__ import annotations

import io
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ROOT

HOLDINGS = ROOT / "data" / "holdings.json"
JOURNAL = ROOT / "data" / "journal.json"


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        bad = path.with_suffix(f".corrupt-{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy2(path, bad)                    # never silently discard an unreadable file
        return {}


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(".bak"))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- holdings
@dataclass
class Holding:
    ticker: str
    qty: float
    avg_cost: float
    exit_line: float | None = None
    imported: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def parse_holdings_csv(text: str) -> list[Holding]:
    """Kite (Zerodha) holdings export, or any table with an instrument/symbol, a quantity and an average price.
    Title lines above the header are skipped; NSE symbols get '.NS'."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if any(k in ln.lower() for k in ("instrument", "symbol", "ticker"))), 0)
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    cols = {c: c.strip().lower() for c in df.columns}
    pick = lambda *keys: next((c for c, low in cols.items() if any(k in low for k in keys)), None)  # noqa: E731
    sym, qty, avg = pick("instrument", "symbol", "ticker"), pick("qty", "quantity"), pick("avg", "average", "buy price")
    if not (sym and qty and avg):
        raise ValueError("need columns for the instrument/symbol, quantity and average cost")
    out = []
    for _, r in df.iterrows():
        s = str(r[sym]).strip().upper()
        if not s or s == "NAN" or s.startswith("TOTAL"):
            continue
        s = s.split("-")[0] if s.endswith(("-BE", "-EQ", "-BZ")) else s
        try:
            q, a = float(str(r[qty]).replace(",", "")), float(str(r[avg]).replace(",", ""))
        except ValueError:
            continue
        if q > 0:
            out.append(Holding(ticker=s if s.endswith(".NS") else s + ".NS", qty=q, avg_cost=a))
    return out


def load_holdings() -> dict[str, Holding]:
    return {d["ticker"]: Holding(**d) for d in _read(HOLDINGS).get("holdings", [])}


def save_holdings(items: dict[str, Holding]) -> None:
    _write(HOLDINGS, {"holdings": [asdict(h) for h in items.values()]})


def import_holdings(text: str) -> dict[str, Holding]:
    """Replace holdings with the CSV's, keeping each existing exit line (it only ratchets up)."""
    old = load_holdings()
    new = {}
    for h in parse_holdings_csv(text):
        if h.ticker in old:
            h.exit_line = old[h.ticker].exit_line
        new[h.ticker] = h
    save_holdings(new)
    return new


# --------------------------------------------------------------------------- journal
@dataclass
class Trade:
    ticker: str
    entry_date: str
    entry: float
    stop: float
    qty: int
    side: str = "long"
    exit_date: str | None = None
    exit: float | None = None
    note: str = ""
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))

    @property
    def closed(self) -> bool:
        return self.exit is not None

    def r(self) -> float | None:
        from .research.plans import r_multiple
        return r_multiple(self.entry, self.stop, self.exit, self.qty, self.side) if self.closed else None


def load_journal() -> list[Trade]:
    return [Trade(**d) for d in _read(JOURNAL).get("trades", [])]


def save_journal(trades: list[Trade]) -> None:
    _write(JOURNAL, {"trades": [asdict(t) for t in trades]})


def add_trade(t: Trade) -> list[Trade]:
    trades = load_journal() + [t]
    save_journal(trades)
    return trades


def close_trade(trade_id: str, exit_price: float, exit_date: str) -> list[Trade]:
    trades = load_journal()
    for t in trades:
        if t.id == trade_id:
            t.exit, t.exit_date = float(exit_price), exit_date
    save_journal(trades)
    return trades


def delete_trade(trade_id: str) -> list[Trade]:
    trades = [t for t in load_journal() if t.id != trade_id]
    save_journal(trades)
    return trades


def journal_stats(trades: list[Trade]) -> dict:
    rs = [t.r() for t in trades if t.closed]
    rs = [x for x in rs if x is not None and x == x]
    if not rs:
        return {"closed": 0}
    wins = [x for x in rs if x > 0]
    losses = [-x for x in rs if x <= 0]
    return {"closed": len(rs), "win_rate": len(wins) / len(rs), "avg_r": sum(rs) / len(rs),
            "median_r": float(pd.Series(rs).median()), "total_r": sum(rs),
            "profit_factor": (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else float("inf")}
