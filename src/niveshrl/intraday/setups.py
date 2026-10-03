"""Rule-based intraday setups on finished 5-minute bars (no look-ahead: every signal at bar *i* uses bars 0..i
only; the fill is the next bar's open, decided elsewhere).

Setups (long and short):
- **orb** — opening-range breakout: first 5-minute close beyond the 15-minute opening range, before 11:00, on a
  bar with ≥ 1.5× the day's average bar volume, on the right side of VWAP. Stop: the signal bar's far end.
- **vwap** — VWAP reclaim / rejection: three closes on one side of VWAP, then a close back across it (from 09:45).
  Stop: the extreme of the last three bars.
- **ema_pullback** — 9 EMA above (below) the 20 EMA for 3 bars; the bar dips (rallies) to the 20 EMA and closes back
  beyond it, with the trend (from 09:45). Stop: the bar's far end.
- **pdh_pdl** — first close above the previous day's high (below its low) between 09:30 and 13:00 on ≥ 1.2× volume.
- **gap_and_go** — gap ≥ 1%; the first bar closes in the gap's direction; a later close beyond the first bar's
  high (low) before 11:30. Stop: the signal bar's far end (the first bar of a gapping stock is usually wider than
  the 1% stop limit, which made this setup inert when the stop sat at the first bar's end).
- **nr7** — yesterday was the narrowest range of 7 days; first close beyond yesterday's high (low) before 13:00.

Stops are widened to ≥ 0.15% of price and a signal is dropped if its stop is wider than 1%; target = 2R.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

MIN_OF_DAY0 = 9 * 60 + 15


@dataclass
class Signal:
    ticker: str
    ts: pd.Timestamp          # end of the finished signal bar
    bar: int                  # index of the signal bar within the day
    setup: str
    side: str                 # long | short
    ref: float                # signal bar close (reference; the fill is the next bar's open)
    stop: float
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = str(self.ts)
        return d


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def day_signals(ticker: str, bars: pd.DataFrame, ctx: dict, setups: list[str], g: dict,
                upto: int | None = None, nifty_trend: np.ndarray | None = None) -> list[Signal]:
    """All signals of one stock for one day. ``bars``: that day's 5-minute bars (open/high/low/close/volume, indexed by
    bar *start* time). ``upto``: only evaluate bars ≤ upto (live mode: the latest finished bar). ``ctx``: prev_high,
    prev_low, prev_close, adr, atr, nr7, avg_volume. ``nifty_trend``: +1/−1 per bar (NIFTY 9/20 EMA)."""
    if len(bars) < 3:
        return []
    o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    t = bars.index
    mins = (t.hour * 60 + t.minute).to_numpy() - MIN_OF_DAY0 + 5           # minutes since open at bar END
    n = len(c) if upto is None else min(len(c), upto + 1)
    tp = (h + l + c) / 3
    cumv = np.cumsum(v)
    vwap = np.where(cumv > 0, np.cumsum(tp * v) / np.maximum(cumv, 1e-9), c)
    e9, e20 = _ema(c, 9), _ema(c, 20)
    or_hi, or_lo = h[:3].max(), l[:3].min()
    avg_bar_v = np.cumsum(v) / np.arange(1, len(v) + 1)
    pc, ph, pl = ctx.get("prev_close", np.nan), ctx.get("prev_high", np.nan), ctx.get("prev_low", np.nan)
    gap = o[0] / pc - 1 if pc and pc == pc else 0.0
    adr = ctx.get("adr", np.nan)
    day_avg_v = ctx.get("avg_volume", np.nan)
    out: list[Signal] = []
    used: dict[tuple, int] = {}

    def add(i, setup, side, stop):
        key = (setup, side)
        if used.get(key, 0) >= (2 if setup in ("vwap", "ema_pullback") else 1):
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
        used[key] = used.get(key, 0) + 1
        rng_so_far = h[:i + 1].max() - l[:i + 1].min()
        vol_pace = (cumv[i] / (day_avg_v * mins[i] / 375)) if day_avg_v and day_avg_v == day_avg_v and mins[i] > 0 else np.nan
        sgn = 1 if side == "long" else -1
        feats = {"setup": setup, "side": side, "minute": int(mins[i]), "gap": float(gap) * sgn,
                 "rvol_bar": float(v[i] / max(avg_bar_v[i], 1e-9)), "vol_pace": float(vol_pace),
                 "vwap_dist": float((ref / vwap[i] - 1) * sgn), "stop_pct": float(dist / ref),
                 "atr_pct": float(ctx.get("atr", np.nan) / ref), "adr": float(adr),
                 "range_used": float(rng_so_far / (adr * pc)) if adr and pc and adr == adr else np.nan,
                 "ema_spread": float((e9[i] / e20[i] - 1) * sgn),
                 "trend_aligned": float(nifty_trend[i] * sgn) if nifty_trend is not None and i < len(nifty_trend) else 0.0,
                 "first_signal": float(sum(used.values()) == 1)}
        out.append(Signal(ticker, t[i] + pd.Timedelta(minutes=5), i, setup, side, float(ref), float(stop), feats))

    for i in range(3, n):
        m = mins[i]
        if m > (14 * 60 + 45 - MIN_OF_DAY0):                               # no new signals after 14:45
            break
        if "orb" in setups and m <= 11 * 60 - MIN_OF_DAY0 and v[i] >= 1.5 * avg_bar_v[i]:
            if c[i] > or_hi and c[i - 1] <= or_hi and c[i] > vwap[i]:
                add(i, "orb", "long", l[i])
            elif c[i] < or_lo and c[i - 1] >= or_lo and c[i] < vwap[i]:
                add(i, "orb", "short", h[i])
        if "vwap" in setups and m >= 30 and i >= 4:
            below = (c[i - 3:i] < vwap[i - 3:i]).all()
            above = (c[i - 3:i] > vwap[i - 3:i]).all()
            if below and c[i] > vwap[i]:
                add(i, "vwap", "long", l[i - 3:i + 1].min())
            elif above and c[i] < vwap[i]:
                add(i, "vwap", "short", h[i - 3:i + 1].max())
        if "ema_pullback" in setups and m >= 30 and i >= 22:
            up = (e9[i - 3:i] > e20[i - 3:i]).all()
            dn = (e9[i - 3:i] < e20[i - 3:i]).all()
            if up and l[i] <= e20[i] and c[i] > e20[i]:
                add(i, "ema_pullback", "long", l[i])
            elif dn and h[i] >= e20[i] and c[i] < e20[i]:
                add(i, "ema_pullback", "short", h[i])
        if "pdh_pdl" in setups and 15 <= m <= 13 * 60 - MIN_OF_DAY0 and v[i] >= 1.2 * avg_bar_v[i]:
            if ph == ph and c[i] > ph and c[:i].max() <= ph:
                add(i, "pdh_pdl", "long", l[i])
            elif pl == pl and c[i] < pl and c[:i].min() >= pl:
                add(i, "pdh_pdl", "short", h[i])
        if "gap_and_go" in setups and abs(gap) >= 0.01 and m <= 11 * 60 + 30 - MIN_OF_DAY0:
            if gap > 0 and c[0] > o[0] and c[i] > h[0] and c[:i].max() <= h[0]:
                add(i, "gap_and_go", "long", l[i])
            elif gap < 0 and c[0] < o[0] and c[i] < l[0] and c[:i].min() >= l[0]:
                add(i, "gap_and_go", "short", h[i])
        if "nr7" in setups and ctx.get("nr7") == 1 and m <= 13 * 60 - MIN_OF_DAY0:
            if ph == ph and c[i] > ph and c[:i].max() <= ph:
                add(i, "nr7", "long", l[i])
            elif pl == pl and c[i] < pl and c[:i].min() >= pl:
                add(i, "nr7", "short", h[i])
    return out


def nifty_trend(nifty_bars: pd.DataFrame) -> np.ndarray:
    """+1 when NIFTY's 5-minute 9 EMA is above its 20 EMA, −1 below (per bar)."""
    c = nifty_bars["close"].to_numpy(float)
    if len(c) == 0:
        return np.array([])
    return np.where(_ema(c, 9) >= _ema(c, 20), 1.0, -1.0)
