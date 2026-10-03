"""The paper account: fills, stops, targets, guardrails, costs, and shadow outcomes for every signal.

Bar loop for one day (5-minute bars, all stocks share the same clock):
  at bar k: (1) square off everything if bar k starts at/after 15:15;
            (2) open positions for signals that finished on bar k−1, at bar k's open (+ slippage), if the
                guardrails allow (≤ 5 open, daily loss stop not hit, risk ≤ 1% of the day's starting pool ×
                action size, total exposure ≤ pool × leverage, quantity ≤ 2% of the signal bar's volume);
            (3) exit on bar k's range: stop first if one bar touches both stop and target (conservative);
                gaps fill at the bar's open when it is already beyond the level.
Every trade records gross P&L, each charge, net P&L and R. ``shadow`` gives every signal (taken or not) the
outcome it would have had at full size, which is what the learner uses, so wrong SKIPs are learned too.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .costs import round_trip
from .setups import Signal

SQUARE_OFF_MIN = 15 * 60 + 15


@dataclass
class Trade:
    ticker: str
    setup: str
    side: str
    action: str               # TAKE | HALF
    signal_ts: str
    entry_ts: str
    entry: float
    stop: float
    target: float
    qty: int
    exit_ts: str = ""
    exit: float = float("nan")
    reason: str = ""          # target | stop | square-off
    gross: float = 0.0
    costs: float = 0.0
    net: float = 0.0
    r: float = float("nan")   # net R
    prob: float = float("nan")
    why: str = ""
    cost_detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _exit_scan(side, o, h, l, start, stop, target, times):
    """First exit from bar ``start`` on: (index, price, reason). Stop first if a bar touches both."""
    for k in range(start, len(o)):
        tmin = times[k]
        if tmin >= SQUARE_OFF_MIN:
            return k, o[k], "square-off"
        if side == "long":
            if l[k] <= stop:
                return k, min(o[k], stop), "stop"
            if h[k] >= target:
                return k, max(o[k], target) if k > start else target, "target"
        else:
            if h[k] >= stop:
                return k, max(o[k], stop), "stop"
            if l[k] <= target:
                return k, min(o[k], target) if k > start else target, "target"
    k = len(o) - 1
    return k, float(o[k]), "square-off"                 # data ended before 15:15


def outcome(sig: Signal, bars: pd.DataFrame, cfg: dict, qty: int | None = None, slip: float | None = None) -> dict | None:
    """Fill at the next bar's open and follow to target / stop / square-off. Returns gross/costs/net/R or None."""
    g = cfg["guardrails"]
    i = sig.bar + 1
    if i >= len(bars):
        return None
    o, h, l = (bars[k].to_numpy(float) for k in ("open", "high", "low"))
    times = (bars.index.hour * 60 + bars.index.minute).to_numpy()
    if times[i] >= SQUARE_OFF_MIN:
        return None
    s = cfg["costs"]["slippage"] if slip is None else slip
    fill = o[i]
    dist = (fill - sig.stop) if sig.side == "long" else (sig.stop - fill)
    if not math.isfinite(dist) or dist <= 0:
        return None                                   # gapped through the stop before entry: no trade
    target = fill + g["target_r"] * dist if sig.side == "long" else fill - g["target_r"] * dist
    k, px, why = _exit_scan(sig.side, o, h, l, i, sig.stop, target, times)
    if qty is None:
        qty = max(1, int(cfg["pool"] * g["risk_per_trade"] // dist))
    gross = (px - fill) * qty if sig.side == "long" else (fill - px) * qty
    tc = round_trip(fill, px, qty, sig.side, cfg["costs"] | {"slippage": s})
    net = gross - tc.total
    return {"entry_bar": i, "exit_bar": k, "fill": float(fill), "exit": float(px), "reason": why, "stop": float(sig.stop),
            "target": float(target), "dist": float(dist), "qty": int(qty), "gross": float(gross), "costs": float(tc.total),
            "net": float(net), "r": float(net / (dist * qty)), "cost_detail": tc.to_dict()}


class Account:
    """One trading day of the paper account. ``decide(sig) -> (action, prob, why)`` with action TAKE/HALF/SKIP."""

    def __init__(self, cfg: dict, pool: float):
        self.cfg = cfg
        self.g = cfg["guardrails"]
        self.pool0 = pool
        self.lev = float(cfg.get("leverage", 1.0))

    def run_day(self, signals: list[Signal], bars: dict[str, pd.DataFrame], decide,
                close_at_end: bool = True) -> tuple[list[Trade], list[dict]]:
        g, cfg = self.g, self.cfg
        by_bar: dict[int, list[Signal]] = {}
        for s in sorted(signals, key=lambda x: (x.bar, x.ticker)):
            by_bar.setdefault(s.bar, []).append(s)
        clock = next(iter(bars.values())).index if bars else pd.DatetimeIndex([])
        nb = max((len(b) for b in bars.values()), default=0)
        open_pos: list[dict] = []
        trades: list[Trade] = []
        log: list[dict] = []
        realized = 0.0
        stopped_for_day = False
        for k in range(nb):
            tmin = clock[k].hour * 60 + clock[k].minute if k < len(clock) else 0
            # (1) square-off
            if tmin >= SQUARE_OFF_MIN:
                for pos in list(open_pos):
                    realized += self._close(pos, k, bars[pos["t"].ticker], "square-off", trades)
                    open_pos.remove(pos)
            # (2) entries for signals finished on bar k-1
            mtm = realized + sum(self._mtm(pos, bars[pos["t"].ticker], k - 1) for pos in open_pos)
            if mtm <= -g["daily_loss_stop"] * self.pool0 and not stopped_for_day:
                stopped_for_day = True
                log.append({"bar": k, "event": "daily loss stop", "detail": f"day P&L ₹{mtm:,.0f}: no new trades today"})
            for sig in by_bar.get(k - 1, []):
                action, prob, why = decide(sig)
                entry = {"ticker": sig.ticker, "setup": sig.setup, "side": sig.side, "ts": str(sig.ts), "action": action,
                         "prob": prob, "why": why}
                if action == "SKIP":
                    log.append(entry | {"event": "skip"})
                    continue
                reason = self._blocked(sig, k, open_pos, stopped_for_day, tmin, bars)
                if reason:
                    log.append(entry | {"event": "blocked", "detail": reason})
                    continue
                b = bars[sig.ticker]
                fill = float(b["open"].iloc[k]) * (1 + cfg["costs"]["slippage"] * (1 if sig.side == "long" else -1))
                dist = (fill - sig.stop) if sig.side == "long" else (sig.stop - fill)
                if dist <= 0:
                    log.append(entry | {"event": "blocked", "detail": "opened beyond the stop"})
                    continue
                size = 0.5 if action == "HALF" else 1.0
                qty = int(self.pool0 * g["risk_per_trade"] * size // dist)
                exposure = sum(p["t"].entry * p["t"].qty for p in open_pos)
                qty = min(qty, int(max(self.pool0 * self.lev - exposure, 0) // fill))
                qty = min(qty, int(g["max_volume_share"] * float(b["volume"].iloc[sig.bar])))
                if qty <= 0:
                    log.append(entry | {"event": "blocked", "detail": "no size left (exposure / volume cap)"})
                    continue
                target = fill + g["target_r"] * dist if sig.side == "long" else fill - g["target_r"] * dist
                t = Trade(sig.ticker, sig.setup, sig.side, action, str(sig.ts), str(b.index[k]), round(fill, 2),
                          round(sig.stop, 2), round(target, 2), qty, prob=prob, why=why)
                open_pos.append({"t": t, "k0": k, "dist": dist})
                log.append(entry | {"event": "entry", "detail": f"{qty} @ ₹{fill:,.2f}, stop ₹{sig.stop:,.2f}, target ₹{target:,.2f}"})
            # (3) exits on bar k
            for pos in list(open_pos):
                b = bars[pos["t"].ticker]
                if k >= len(b):
                    continue
                t = pos["t"]
                hi, lo, op = float(b["high"].iloc[k]), float(b["low"].iloc[k]), float(b["open"].iloc[k])
                if t.side == "long":
                    if lo <= t.stop:
                        realized += self._close(pos, k, b, "stop", trades, px=min(op, t.stop) if k > pos["k0"] else t.stop)
                        open_pos.remove(pos)
                    elif hi >= t.target:
                        realized += self._close(pos, k, b, "target", trades, px=max(op, t.target) if k > pos["k0"] else t.target)
                        open_pos.remove(pos)
                else:
                    if hi >= t.stop:
                        realized += self._close(pos, k, b, "stop", trades, px=max(op, t.stop) if k > pos["k0"] else t.stop)
                        open_pos.remove(pos)
                    elif lo <= t.target:
                        realized += self._close(pos, k, b, "target", trades, px=min(op, t.target) if k > pos["k0"] else t.target)
                        open_pos.remove(pos)
        self.open_positions = []
        for pos in list(open_pos):                     # data ended before 15:15
            b = bars[pos["t"].ticker]
            if close_at_end:
                realized += self._close(pos, len(b) - 1, b, "square-off", trades, px=float(b["close"].iloc[-1]))
            else:                                      # live, mid-day: still open, marked to the last close
                t = pos["t"]
                last = float(b["close"].iloc[-1])
                self.open_positions.append(t.to_dict() | {"last": last, "unrealized": round(
                    (last - t.entry) * t.qty if t.side == "long" else (t.entry - last) * t.qty, 2)})
        return trades, log

    def _blocked(self, sig, k, open_pos, stopped, tmin, bars) -> str:
        g = self.g
        if stopped:
            return "daily loss stop hit"
        if tmin >= SQUARE_OFF_MIN or tmin > 14 * 60 + 45 + 5:
            return "too late in the day"
        if len(open_pos) >= g["max_open_positions"]:
            return f"already {g['max_open_positions']} open positions"
        if any(p["t"].ticker == sig.ticker for p in open_pos):
            return "already holding this stock"
        if k >= len(bars[sig.ticker]):
            return "no next bar"
        return ""

    def _mtm(self, pos, b, k) -> float:
        if k < 0 or k >= len(b):
            return 0.0
        t, px = pos["t"], float(b["close"].iloc[k])
        return (px - t.entry) * t.qty if t.side == "long" else (t.entry - px) * t.qty

    def _close(self, pos, k, b, reason, trades, px: float | None = None) -> float:
        t = pos["t"]
        k = min(k, len(b) - 1)
        px = float(b["open"].iloc[k]) if px is None else float(px)
        px *= (1 - self.cfg["costs"]["slippage"] * (1 if t.side == "long" else -1))
        gross = (px - t.entry) * t.qty if t.side == "long" else (t.entry - px) * t.qty
        tc = round_trip(t.entry, px, t.qty, t.side, self.cfg["costs"] | {"slippage": 0.0})
        t.exit_ts, t.exit, t.reason = str(b.index[k]), round(px, 2), reason
        t.gross, t.costs, t.net = round(gross, 2), round(tc.total, 2), round(gross - tc.total, 2)
        t.r = (gross - tc.total) / (pos["dist"] * t.qty)
        t.cost_detail = {k2: round(v, 2) for k2, v in tc.to_dict().items()}
        trades.append(t)
        return t.net


def shadow(signals: list[Signal], bars: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Full-size outcome for every signal (taken or not): the learner's training data."""
    rows = []
    for s in signals:
        oc = outcome(s, bars[s.ticker], cfg)
        if oc is None:
            continue
        rows.append(s.features | {"ticker": s.ticker, "ts": str(s.ts), "net_r": oc["r"], "win": float(oc["r"] > 0),
                                   "reason": oc["reason"]})
    return pd.DataFrame(rows)


__all__ = ["Account", "Trade", "outcome", "shadow", "np"]
