"""Trade plans: where to put the stop, where to take profit, and how many shares to buy.

For a long swing trade bought at ``entry``:
- **Stop**: just under the lowest low of the last 10 days, kept between 2× and 2.5× ATR(14) below entry
  (closer is widened, wider is tightened) and never more than 8% below entry.
- **R** = entry − stop: the rupee risk per share. **T1** = entry + 1.5R, **T2** = entry + 2.5R.
  Suggested management: book about a third at T1 and move the stop to entry, then trail the rest
  3 × ATR under the highest close.
- **Size** = floor(capital × risk% × risk-state multiplier ÷ R), and never more than 20% of capital.
- **Risk state**: half risk when NIFTY is below its 200-day average, when breadth is narrow (fewer than
  half of stocks above their 50-day average) or in a Stress regime. It changes size only, never which
  stocks qualify.

"Watch for weakness" stocks get the mirror image (stop above the 10-day high) labelled as an exit /
avoid level, because Indian delivery accounts cannot short overnight.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import load_yaml
from .data import Panel

STOP_LOOKBACK = 10
ATR_MIN, ATR_MAX, MAX_STOP_PCT = 2.0, 2.5, 0.08
T1_R, T2_R, TRAIL_ATR = 1.5, 2.5, 3.0
MAX_POSITION_PCT = 0.20


@dataclass
class RiskState:
    state: str              # Normal / Narrow / Below 200-DMA / Stress
    multiplier: float
    reasons: list

    def to_dict(self) -> dict:
        return asdict(self)


def risk_state(p: Panel, regime: str | None = None, asof: pd.Timestamp | None = None) -> RiskState:
    c = p.close if asof is None else p.close.loc[:asof]
    b = p.bench.reindex(c.index).ffill()
    reasons = []
    below200 = bool(len(b.dropna()) >= 200 and b.iloc[-1] < b.rolling(200).mean().iloc[-1])
    if below200:
        reasons.append(f"NIFTY {b.iloc[-1]:,.0f} is below its 200-day average {b.rolling(200).mean().iloc[-1]:,.0f}")
    ma50 = c.rolling(50, min_periods=40).mean().iloc[-1]
    breadth50 = float((c.iloc[-1] > ma50).where(ma50.notna()).mean())
    narrow = breadth50 < 0.5
    if narrow:
        reasons.append(f"only {breadth50:.0%} of stocks are above their 50-day average (narrow market)")
    stress = regime == "Stress"
    if stress:
        reasons.append("the regime detector reads Stress")
    if not reasons:
        return RiskState("Normal", 1.0, [f"NIFTY above its 200-day average and {breadth50:.0%} of stocks above "
                                         "their 50-day average"])
    state = "Stress" if stress else "Below 200-DMA" if below200 else "Narrow"
    return RiskState(state, 0.5, reasons)


@dataclass
class Plan:
    ticker: str
    side: str               # long | exit-level (for weak stocks)
    entry: float
    stop: float
    risk_per_share: float   # R in rupees
    stop_pct: float
    t1: float
    t2: float
    trail_atr: float        # rupees: 3 x ATR, the trailing distance after T1
    atr: float
    qty: int
    capital: float
    risk_pct: float
    multiplier: float
    rupee_risk: float
    position_value: float
    notes: list

    def to_dict(self) -> dict:
        return asdict(self)


def make_plan(p: Panel, ticker: str, capital: float = 100_000.0, risk_pct: float = 0.01,
              multiplier: float = 1.0, entry: float | None = None, side: str = "long",
              atr_df: pd.DataFrame | None = None, asof: pd.Timestamp | None = None) -> Plan | None:
    """Plan for one stock from its daily bars up to ``asof`` (default: the latest bar)."""
    from .technicals import atr as atr_fn
    c = p.close[ticker] if asof is None else p.close[ticker].loc[:asof]
    c = c.dropna()
    if len(c) < 30:
        return None
    lo = (p.low[ticker] if p.low is not None else p.close[ticker]).reindex(c.index)
    hi = (p.high[ticker] if p.high is not None else p.close[ticker]).reindex(c.index)
    a_series = (atr_df[ticker] if atr_df is not None else atr_fn(p)[ticker]).reindex(c.index)
    a = float(a_series.iloc[-1])
    if not math.isfinite(a) or a <= 0:
        return None
    e = float(entry if entry is not None else c.iloc[-1])
    notes = []
    if side == "long":
        raw = float(lo.iloc[-STOP_LOOKBACK:].min()) * 0.998
        dist = e - raw
        if dist < ATR_MIN * a:
            dist = ATR_MIN * a
            notes.append(f"stop widened to {ATR_MIN:g}× ATR (the 10-day low was too close)")
        elif dist > ATR_MAX * a:
            dist = ATR_MAX * a
            notes.append(f"stop tightened to {ATR_MAX:g}× ATR (the 10-day low was far below)")
        if dist > MAX_STOP_PCT * e:
            dist = MAX_STOP_PCT * e
            notes.append("stop capped at 8% below entry")
        stop = e - dist
        t1, t2 = e + T1_R * dist, e + T2_R * dist
    else:
        raw = float(hi.iloc[-STOP_LOOKBACK:].max()) * 1.002
        dist = min(max(raw - e, ATR_MIN * a), ATR_MAX * a, MAX_STOP_PCT * e)
        stop = e + dist
        t1, t2 = e - T1_R * dist, e - T2_R * dist
        notes.append("weak stock: for holders this is the level to exit; NSE delivery accounts cannot short overnight")
    budget = capital * risk_pct * multiplier
    qty = int(budget // dist) if dist > 0 else 0
    cap_qty = int(capital * MAX_POSITION_PCT // e)
    if qty > cap_qty:
        qty = cap_qty
        notes.append(f"size capped at {MAX_POSITION_PCT:.0%} of capital")
    if multiplier < 1:
        notes.append(f"risk halved by the market risk state (×{multiplier:g})")
    return Plan(ticker=ticker, side=side, entry=round(e, 2), stop=round(stop, 2), risk_per_share=round(dist, 2),
                stop_pct=dist / e, t1=round(t1, 2), t2=round(t2, 2), trail_atr=round(TRAIL_ATR * a, 2), atr=round(a, 2),
                qty=qty, capital=capital, risk_pct=risk_pct, multiplier=multiplier, rupee_risk=round(qty * dist, 2),
                position_value=round(qty * e, 2), notes=notes)


def plans_for(p: Panel, tickers: list[str], capital: float = 100_000.0, risk_pct: float = 0.01,
              multiplier: float = 1.0, side: str = "long") -> pd.DataFrame:
    from .technicals import atr as atr_fn
    a = atr_fn(p)
    rows = []
    for t in tickers:
        if t in p.close.columns:
            pl = make_plan(p, t, capital, risk_pct, multiplier, side=side, atr_df=a)
            if pl is not None:
                rows.append(pl.to_dict())
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- costs and R
def delivery_costs(entry: float, exit_: float, qty: int, rates: dict | None = None) -> dict:
    """Round-trip delivery charges in rupees (buy at entry, sell at exit), from configs/costs_india.yaml."""
    r = rates or load_yaml("configs/costs_india.yaml")
    buy, sell = entry * qty, exit_ * qty
    turnover = buy + sell
    brokerage = turnover * r.get("brokerage_rate", 0.0)
    if r.get("brokerage_cap", 0) > 0:
        brokerage = min(brokerage, 2 * r["brokerage_cap"])
    out = {"brokerage": brokerage, "stt": r["stt_rate"] * turnover, "exchange": r["exchange_rate"] * turnover,
           "sebi": r["sebi_rate"] * turnover, "stamp": r["stamp_rate"] * buy, "dp": r["dp_charge"] if qty else 0.0,
           "slippage": r["half_spread"] * turnover}
    out["gst"] = r["gst_rate"] * (out["brokerage"] + out["exchange"] + out["sebi"])
    out["total"] = sum(out.values())
    return out


def r_multiple(entry: float, stop: float, exit_: float, qty: int, side: str = "long",
               rates: dict | None = None) -> float:
    """(exit − entry) ÷ (entry − stop) per share, after round-trip costs."""
    risk = abs(entry - stop)
    if risk <= 0 or qty <= 0:
        return float("nan")
    gross = (exit_ - entry) * qty if side == "long" else (entry - exit_) * qty
    net = gross - delivery_costs(entry, exit_, qty, rates)["total"]
    return net / (risk * qty)


def exit_line(close: pd.Series, atr_s: pd.Series, previous: float | None = None, k: float = 3.0) -> float:
    """Holding exit line = close − k × ATR, only ever raised (a ratchet)."""
    v = float(close.dropna().iloc[-1] - k * atr_s.dropna().iloc[-1])
    return max(v, previous) if previous is not None and np.isfinite(previous) else v
