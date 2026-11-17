"""Swing book: the swing plan traded automatically as a paper portfolio (delivery, long only). Paper only.

The same rules as the measured swing record (research/forward.py ``follow``), plus what an R-per-trade record leaves
out: real cash, position sizing, overlapping positions and costs on every sale.

- **Picks**: the day's "watch for strength" list (in score order). A stock picked within the last 30 days is skipped.
- **Entry**: next trading day's open. Distance to the stop comes from the plan on the signal day (stop just under the
  10-day low, 2-2.5× ATR, at most 8%); stop = entry − distance, T1 = entry + 1.5 × distance.
- **Size**: equity × risk% × risk-state multiplier ÷ distance, at most 20% of equity and only from cash (delivery: no
  leverage), at most ``max_positions`` open. A pick that does not fit is logged as skipped with the reason.
- **Exits** (checked on each day's high/low, stop before target as the record does): stop → sell all at the stop (or
  the open if it gapped below); T1 → sell a third and move the stop to entry; then trail 3 × ATR under the highest
  close; after 60 trading days sell at the close.
- **Costs**: delivery charges (configs/costs_india.yaml) on every buy and sale.

``SwingBook.run_day`` processes one trading day from daily bars only, so the backtest and the live book (a daily
pipeline step after the close) run the same code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from . import plans as P

BOOK_DIR = ROOT / "data" / "swing_book"
FOLLOW_DAYS = 60
DEDUP_DAYS = 30


def _costs(entry: float, px: float, qty: int, rates: dict) -> float:
    return float(P.delivery_costs(entry, px, qty, rates)["total"]) if qty > 0 else 0.0


class SwingBook:
    def __init__(self, capital: float = 100_000.0, risk_pct: float = 0.01, max_positions: int = 10,
                 max_position_pct: float = P.MAX_POSITION_PCT, state: dict | None = None):
        self.s = state or {"capital": capital, "cash": capital, "risk_pct": risk_pct, "max_positions": max_positions,
                           "max_position_pct": max_position_pct,
                           "positions": {}, "pending": [], "trades": [], "equity": [], "recent": {}, "log": [],
                           "last_day": None}
        self.rates = P.load_yaml("configs/costs_india.yaml")

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, folder: Path = BOOK_DIR) -> "SwingBook":
        f = folder / "state.json"
        return cls(state=json.loads(f.read_text(encoding="utf-8"))) if f.exists() else cls()

    def save(self, folder: Path = BOOK_DIR) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "state.json.tmp"
        tmp.write_text(json.dumps(self.s, default=str), encoding="utf-8")
        tmp.replace(folder / "state.json")

    # ------------------------------------------------------------------ helpers
    def equity(self, p: P.Panel | None = None, day=None) -> float:
        val = 0.0
        for t, pos in self.s["positions"].items():
            px = pos["last"]
            if p is not None and day is not None and t in p.close:
                c = p.close[t].get(day, np.nan)
                px = float(c) if np.isfinite(c) else px
            val += pos["qty"] * px
        return self.s["cash"] + val

    def _log(self, day, event: str, ticker: str = "", detail: str = "") -> None:
        self.s["log"].append({"day": str(pd.Timestamp(day).date()), "event": event, "ticker": ticker, "detail": detail})

    def _sell(self, day, t: str, qty: int, px: float, reason: str) -> None:
        pos = self.s["positions"][t]
        sell_cost = _costs(pos["entry"], px, qty, self.rates) - pos["buy_cost_per_share"] * qty   # buy side paid at entry
        self.s["cash"] += px * qty - sell_cost
        pos["qty"] -= qty
        pos["realized"] += (px - pos["entry"]) * qty - sell_cost
        self._log(day, "sell", t, f"{qty} @ ₹{px:,.2f} ({reason})")
        if pos["qty"] <= 0:
            r = pos["realized"] / (pos["dist"] * pos["qty0"])
            self.s["trades"].append({"ticker": t, "signal_date": pos["signal_date"], "entry_date": pos["entry_date"],
                                     "exit_date": str(pd.Timestamp(day).date()), "entry": pos["entry"], "exit": px,
                                     "qty": pos["qty0"], "net": round(pos["realized"], 2), "r": round(r, 3),
                                     "days": pos["days"], "reason": reason})
            del self.s["positions"][t]

    # ------------------------------------------------------------------ one trading day
    def run_day(self, p: P.Panel, day: pd.Timestamp, atr_df: pd.DataFrame, picks: list[str] | None = None,
                multiplier: float = 1.0) -> None:
        """Fill yesterday's picks at today's open, manage every open position on today's bar, mark to the close, then
        queue today's ``picks`` for tomorrow's open."""
        day = pd.Timestamp(day)
        o = p.open if p.open is not None else p.close
        h = p.high if p.high is not None else p.close
        lo = p.low if p.low is not None else p.close
        # 1) entries at today's open
        pend, self.s["pending"] = self.s["pending"], []
        for q in pend:
            t = q["ticker"]
            op = float(o[t].get(day, np.nan)) if t in o else np.nan
            if not np.isfinite(op) or op <= 0:
                self._log(day, "skip", t, "no open price")
                continue
            if t in self.s["positions"]:
                continue
            if len(self.s["positions"]) >= self.s["max_positions"]:
                self._log(day, "skip", t, f"{self.s['max_positions']} positions already open")
                continue
            dist = q["dist"]
            eq = q["equity"]
            qty = int(eq * self.s["risk_pct"] * q["mult"] // dist)
            qty = min(qty, int(eq * self.s.get("max_position_pct", P.MAX_POSITION_PCT) // op))
            want = qty
            if qty > 0 and qty * op * 1.002 > self.s["cash"]:                # delivery: only from cash
                qty = int(self.s["cash"] / (op * 1.002))
                if qty < want / 2:                     # a scrap of a position: fixed charges would swamp it
                    self._log(day, "skip", t, "not enough cash")
                    continue
            if qty <= 0:
                self._log(day, "skip", t, "not enough cash" if self.s["cash"] < op * 2 else "size rounds to zero")
                continue
            bc = _costs(op, op, qty, self.rates) / 2
            self.s["cash"] -= qty * op + bc
            self.s["positions"][t] = {"qty": qty, "qty0": qty, "entry": op, "dist": dist, "stop": op - dist,
                                      "t1": op + P.T1_R * dist, "frac_done": False, "hi_close": op,
                                      "entry_date": str(day.date()), "signal_date": q["signal_date"], "days": 0,
                                      "realized": -bc, "buy_cost_per_share": bc / qty, "last": op}
            self._log(day, "buy", t, f"{qty} @ ₹{op:,.2f}, stop ₹{op - dist:,.2f}, T1 ₹{op + P.T1_R * dist:,.2f}")
        # 2) manage open positions on today's bar (stop first, then T1, then trail; same order as the record)
        for t in list(self.s["positions"]):
            pos = self.s["positions"][t]
            ok, hk, lk, ck = (float(x[t].get(day, np.nan)) for x in (o, h, lo, p.close))
            if not np.isfinite(ck):
                continue
            pos["days"] += 1
            first = pos["entry_date"] == str(day.date())
            if lk <= pos["stop"]:
                px = pos["stop"] if first else min(ok, pos["stop"])
                why = "stop" if not pos["frac_done"] else ("breakeven stop" if pos["stop"] <= pos["entry"] + 1e-9 else "trailing stop")
                self._sell(day, t, pos["qty"], px, why)
                continue
            if not pos["frac_done"] and hk >= pos["t1"]:
                k = pos["qty"] // 3
                if k > 0:
                    self._sell(day, t, k, pos["t1"], "T1: a third booked")
                pos["frac_done"] = True
                pos["stop"] = max(pos["stop"], pos["entry"])
            pos["hi_close"] = max(pos["hi_close"], ck)
            a = float(atr_df[t].get(day, np.nan)) if t in atr_df else np.nan
            if pos["frac_done"] and np.isfinite(a):
                pos["stop"] = max(pos["stop"], pos["hi_close"] - P.TRAIL_ATR * a)
            pos["last"] = ck
            if pos["days"] >= FOLLOW_DAYS:
                self._sell(day, t, pos["qty"], ck, "60-day cap")
        eq = self.equity(p, day)
        self.s["equity"].append({"day": str(day.date()), "equity": round(eq, 2), "cash": round(self.s["cash"], 2),
                                 "positions": len(self.s["positions"])})
        # 3) queue today's picks for tomorrow's open
        for t in picks or []:
            last = self.s["recent"].get(t)
            if t in self.s["positions"] or (last and (day - pd.Timestamp(last)).days < DEDUP_DAYS):
                continue
            plan = P.make_plan(p, t, atr_df=atr_df, asof=day)
            if plan is None or plan.risk_per_share <= 0:
                continue
            self.s["recent"][t] = str(day.date())
            self.s["pending"].append({"ticker": t, "signal_date": str(day.date()), "dist": float(plan.risk_per_share),
                                      "equity": eq, "mult": multiplier})
        self.s["last_day"] = str(day.date())

    # ------------------------------------------------------------------ reporting
    def summary(self, bench: pd.Series | None = None) -> dict:
        eq = pd.DataFrame(self.s["equity"])
        if eq.empty:
            return {}
        e = eq.set_index(pd.to_datetime(eq["day"]))["equity"]
        yrs = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
        r = e.pct_change().dropna()
        out = {"start": str(e.index[0].date()), "end": str(e.index[-1].date()), "equity": float(e.iloc[-1]),
               "cagr": float((e.iloc[-1] / self.s["capital"]) ** (1 / yrs) - 1),
               "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
               "max_dd": float((e / e.cummax() - 1).min()), "trades": len(self.s["trades"]),
               "avg_r": float(np.mean([t["r"] for t in self.s["trades"]])) if self.s["trades"] else np.nan,
               "exposure": float((eq["equity"] - eq["cash"]).div(eq["equity"]).mean())}
        if bench is not None:
            b = bench.reindex(e.index).ffill().dropna()
            if len(b) > 1:
                out["bench_cagr"] = float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1)
                out["bench_max_dd"] = float((b / b.cummax() - 1).min())
                br = b.pct_change().dropna()
                out["bench_sharpe"] = float(br.mean() / br.std() * np.sqrt(252))
        return out


def replay(p: P.Panel, picks_by_day: dict, start: str, end: str | None = None, **kw) -> SwingBook:
    """Run the book day by day over history. ``picks_by_day``: signal date -> ordered tickers."""
    from .technicals import atr
    a = atr(p)
    book = SwingBook(**kw)
    days = p.close.loc[start:end].index
    reg = None
    for d in days:
        mult = 1.0
        if d in picks_by_day:
            if reg is None:
                reg = _multipliers(p, days)
            mult = reg.get(d, 1.0)
        book.run_day(p, d, a, picks_by_day.get(d), mult)
    return book


def _multipliers(p: P.Panel, days) -> dict:
    """Risk-state size multiplier per day from prices only (half size below NIFTY's 200-day average or when fewer than
    half of stocks are above their 50-day average). The regime input is left out: it has no point-in-time history here."""
    c = p.close
    b = p.bench.reindex(c.index).ffill()
    below = b < b.rolling(200).mean()
    breadth = (c > c.rolling(50).mean()).sum(axis=1) / c.notna().sum(axis=1).clip(lower=1)
    m = pd.Series(1.0, index=c.index).where(~(below | (breadth < 0.5)), 0.5)
    return m.reindex(days).to_dict()


def daily_update(p: P.Panel, day: pd.Timestamp, picks: list[str], folder: Path = BOOK_DIR) -> dict:
    """The live paper book, called by the daily pipeline after the close: process every trading day since the last
    run (fills at the open, exits on the bar), then queue today's picks for tomorrow's open. Running twice on the same
    day changes nothing."""
    from .technicals import atr
    book = SwingBook.load(folder)
    day = pd.Timestamp(day)
    last = pd.Timestamp(book.s["last_day"]) if book.s.get("last_day") else None
    if last is not None and last >= day:
        return {"note": f"already up to date ({last.date()})", **book.summary(p.bench)}
    days = [d for d in p.close.index if (last is None and d == day) or (last is not None and last < d <= day)]
    a = atr(p)
    mult = _multipliers(p, days)
    for d in days:
        book.run_day(p, d, a, picks if d == day else None, mult.get(d, 1.0))
    book.save(folder)
    out = book.summary(p.bench) | {"queued": [q["ticker"] for q in book.s["pending"]],
                                   "open": sorted(book.s["positions"]), "days_run": len(days)}
    (folder / "summary.json").write_text(json.dumps(out, default=str, indent=1), encoding="utf-8")
    return out
