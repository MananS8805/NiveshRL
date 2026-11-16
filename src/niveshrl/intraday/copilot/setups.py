"""Copilot setups on finished 5-minute bars of liquid large caps. No look-ahead: a signal at bar i uses bars 0..i
(and NIFTY bars 0..i); the fill is the next bar's open (sim.outcome / sim.Account). Stops sit at a *structural* level
(not a tight bar extreme), floored at 0.3% of price, so round-trip costs stay a small part of 1R; signals with stops
wider than 1.5% are dropped.

- **orb15** — first close beyond the 15-minute opening range (09:15-09:30), 09:35-11:00, on >= 1.2x average bar
  volume, on the right side of VWAP. Stop: the opening range's midpoint.
- **vwap_hold** — trend continuation from 10:00: the stock spent >= 6 of the last 8 bars on one side of VWAP, this bar
  tests VWAP (within 0.1%) and closes back on the trend side, the stock is stronger (weaker) than NIFTY since the
  open, and NIFTY's 9/20 EMA trend agrees. Stop: just beyond the lower of the last two bars' lows (or VWAP).
- **rs_breakout** — relative strength from 10:30: the stock is >= 1% ahead of (behind) NIFTY since the open and
  closes at a new high (low) of the day on >= 1.2x volume, NIFTY trend agrees. Stop: the last 6 bars' low (high).
- **trend_pullback** — trend day: by 10:30-13:30 the day's range already uses >= 60% of the stock's average daily
  range and the 9 EMA is above (below) the 20 EMA; the bar dips to the 20 EMA and closes back beyond it. Stop: the
  pullback's extreme.
Each setup fires at most once per stock, side and day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..setups import Signal, _ema

MIN0 = 9 * 60 + 15
SETUPS = ("orb15", "vwap_hold", "rs_breakout", "trend_pullback")


def _m(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def nifty_series(nifty: pd.DataFrame | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(return since the open, 9/20 EMA trend ±1) per bar, aligned to the first n bars."""
    if nifty is None or not len(nifty):
        return np.zeros(n), np.zeros(n)
    c = nifty["close"].to_numpy(float)
    o0 = float(nifty["open"].iloc[0])
    ret = c / o0 - 1
    tr = np.where(_ema(c, 9) >= _ema(c, 20), 1.0, -1.0)
    pad = max(0, n - len(c))
    return np.r_[ret, np.full(pad, ret[-1] if len(ret) else 0)][:n], np.r_[tr, np.full(pad, tr[-1] if len(tr) else 0)][:n]


def day_signals(ticker: str, bars: pd.DataFrame, ctx: dict, g: dict, nifty: pd.DataFrame | None = None,
                upto: int | None = None, setups=SETUPS) -> list[Signal]:
    if len(bars) < 4:
        return []
    o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    t = bars.index
    end_min = (t.hour * 60 + t.minute).to_numpy() + 5                       # bar END, minutes of day
    n = len(c) if upto is None else min(len(c), upto + 1)
    tp = (h + l + c) / 3
    cumv = np.cumsum(v)
    vwap = np.where(cumv > 0, np.cumsum(tp * v) / np.maximum(cumv, 1e-9), c)
    e9, e20 = _ema(c, 9), _ema(c, 20)
    avg_v = cumv / np.arange(1, len(v) + 1)
    or_hi, or_lo = h[:3].max(), l[:3].min()
    n_ret, n_tr = nifty_series(nifty, len(c))
    pc = ctx.get("prev_close", np.nan)
    adr = ctx.get("adr", np.nan)
    gap = o[0] / pc - 1 if pc and pc == pc else 0.0
    last_new = _m(g.get("no_new_after", "14:30"))
    out: list[Signal] = []
    used: set = set()

    def add(i, setup, side, stop):
        if (setup, side) in used:
            return
        ref = c[i]
        dist = (ref - stop) if side == "long" else (stop - ref)
        if not np.isfinite(dist) or dist <= 0:
            return
        if dist < g["min_stop_pct"] * ref:
            dist = g["min_stop_pct"] * ref
            stop = ref - dist if side == "long" else ref + dist
        if dist > g["max_stop_pct"] * ref:
            return
        used.add((setup, side))
        sg = 1 if side == "long" else -1
        rng = h[:i + 1].max() - l[:i + 1].min()
        mins_open = end_min[i] - MIN0
        pace = cumv[i] / (ctx["avg_volume"] * mins_open / 375) if ctx.get("avg_volume") and mins_open > 0 else np.nan
        feats = {"setup": setup, "side": side, "minute": int(mins_open), "gap": float(gap) * sg,
                 "rvol_bar": float(v[i] / max(avg_v[i], 1e-9)), "vol_pace": float(pace),
                 "vwap_dist": float((ref / vwap[i] - 1) * sg), "stop_pct": float(dist / ref),
                 "rs": float((ref / o[0] - 1 - n_ret[i]) * sg), "nifty_trend": float(n_tr[i] * sg),
                 "range_used": float(rng / (adr * pc)) if adr and pc and adr == adr else np.nan,
                 "pos_in_range": float(((ref - l[:i + 1].min()) / rng if rng > 0 else 0.5) if sg > 0
                                       else ((h[:i + 1].max() - ref) / rng if rng > 0 else 0.5)),
                 "ema_spread": float((e9[i] / e20[i] - 1) * sg), "adr": float(adr) if adr == adr else np.nan}
        out.append(Signal(ticker, t[i] + pd.Timedelta(minutes=5), i, setup, side, float(ref), float(stop), feats))

    for i in range(3, n):
        em = end_min[i]
        if em > last_new:
            break
        rs = (c[i] / o[0] - 1) - n_ret[i]
        if "orb15" in setups and em <= 11 * 60 and v[i] >= 1.2 * avg_v[i]:
            mid = (or_hi + or_lo) / 2
            if c[i] > or_hi and c[3:i].max(initial=-np.inf) <= or_hi and c[i] > vwap[i]:
                add(i, "orb15", "long", mid)
            elif c[i] < or_lo and c[3:i].min(initial=np.inf) >= or_lo and c[i] < vwap[i]:
                add(i, "orb15", "short", mid)
        if "vwap_hold" in setups and em >= 10 * 60 and i >= 8:
            above = (c[i - 8:i] > vwap[i - 8:i]).sum() >= 6
            below = (c[i - 8:i] < vwap[i - 8:i]).sum() >= 6
            if above and l[i] <= vwap[i] * 1.001 and c[i] > vwap[i] and rs > 0 and n_tr[i] > 0:
                add(i, "vwap_hold", "long", min(l[i - 1:i + 1].min(), vwap[i]) * 0.999)
            elif below and h[i] >= vwap[i] * 0.999 and c[i] < vwap[i] and rs < 0 and n_tr[i] < 0:
                add(i, "vwap_hold", "short", max(h[i - 1:i + 1].max(), vwap[i]) * 1.001)
        if "rs_breakout" in setups and em >= 10 * 60 + 30 and i >= 6 and v[i] >= 1.2 * avg_v[i]:
            if rs >= 0.01 and n_tr[i] > 0 and c[i] >= h[:i].max():
                add(i, "rs_breakout", "long", l[i - 5:i + 1].min())
            elif rs <= -0.01 and n_tr[i] < 0 and c[i] <= l[:i].min():
                add(i, "rs_breakout", "short", h[i - 5:i + 1].max())
        if "trend_pullback" in setups and 10 * 60 + 30 <= em <= 13 * 60 + 30 and i >= 20 and adr == adr and pc == pc:
            used_adr = (h[:i + 1].max() - l[:i + 1].min()) / (adr * pc)
            if used_adr >= 0.6:
                if (e9[i - 3:i] > e20[i - 3:i]).all() and l[i] <= e20[i] and c[i] > e20[i]:
                    add(i, "trend_pullback", "long", l[i - 2:i + 1].min() * 0.999)
                elif (e9[i - 3:i] < e20[i - 3:i]).all() and h[i] >= e20[i] and c[i] < e20[i]:
                    add(i, "trend_pullback", "short", h[i - 2:i + 1].max() * 1.001)
    return out


def all_signals(bars: dict[str, pd.DataFrame], ctx: dict[str, dict], g: dict, nifty: pd.DataFrame | None,
                upto: int | None = None, setups=SETUPS) -> list[Signal]:
    out = []
    for t, b in bars.items():
        if t in ctx:
            out += day_signals(t, b, ctx[t], g, nifty, upto, setups)
    return sorted(out, key=lambda s: (s.bar, s.ticker))
