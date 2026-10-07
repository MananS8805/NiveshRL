"""Training data for the intraday deep-learning models: every liquid stock, every 5-minute bar, long and short.

**Bars.** Yahoo serves only the last ~60 days of 5-minute bars, so every day's bars are also archived
(``data/intraday/archive/YYYY-MM-DD.parquet``) and ``load_bars`` merges the archive with the latest download: the
history grows by a day every trading day.

**Triple-barrier labels** (López de Prado). For a signal at the close of bar i the trade fills at bar i+1's open.
Stop distance = max(1.5 × the stock's 5-minute ATR, the distance at which round-trip costs are 0.2R); bars whose
stop would be wider than 1% are not labelled. Target = 2R. Time barrier: exit at the open 12 bars (60 min) after
entry, or at the 15:15 square-off. The label is the trade's **R after real costs** (brokerage, STT, exchange, SEBI,
stamp, GST, slippage) for a ₹1 lakh pool risking 1%, for a long and for a short. Stop first when one bar touches both.

**Inputs (causal: only bars ≤ i).** A window of the last 24 bars × 10 channels (return, range, body, close location,
VWAP distance, relative volume, return since open, NIFTY's return, distance from the day's high / low), with return-like
channels divided by the stock's average daily range so stocks are comparable; plus a context vector (time of day,
gap, average daily range, daily ATR, share of the usual range already used, NR7, NIFTY's return since open, stop size,
costs in R).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import DIR

ARCHIVE = DIR / "archive"
W = 24                         # window length (bars)
H = 12                         # time barrier (bars after entry)
CHANNELS = ["ret", "range", "body", "loc", "vwap_dist", "rvol", "since_open", "nifty_ret", "from_high", "from_low"]
CONTEXT = ["minute", "gap", "adr", "atr_pct", "range_used", "nr7", "nifty_since_open", "stop_pct", "cost_r", "valid_frac"]
OPEN_MIN, SQ_MIN, LAST_ENTRY_MIN = 9 * 60 + 15, 15 * 60 + 15, 14 * 60 + 45


# --------------------------------------------------------------------------- bars
def archive_day(bars_long: pd.DataFrame) -> int:
    """Save each day of a long bars table (ticker, ts, open, high, low, close, volume) to the archive."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    n = 0
    for day, g in bars_long.groupby(bars_long["ts"].dt.normalize()):
        p = ARCHIVE / f"{pd.Timestamp(day).date()}.parquet"
        if p.exists():
            old = pd.read_parquet(p)
            g = pd.concat([old, g]).drop_duplicates(["ticker", "ts"], keep="last")
        g.to_parquet(p)
        n += 1
    return n


def load_bars(latest: pd.DataFrame | None = None) -> pd.DataFrame:
    """The archive plus the latest download, de-duplicated."""
    from ..replay import BARS
    parts = [pd.read_parquet(f) for f in sorted(ARCHIVE.glob("*.parquet"))] if ARCHIVE.exists() else []
    if latest is None and BARS.exists():
        latest = pd.read_parquet(BARS)
    if latest is not None:
        parts.append(latest)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "ts"], keep="last")
    return df.sort_values(["ticker", "ts"]).reset_index(drop=True)


# --------------------------------------------------------------------------- labels
def cost_fraction(c: dict) -> float:
    """Proportional round-trip charges as a fraction of price (brokerage at its 0.03% rate, the small-order case)."""
    return (c["stt_sell"] + c["stamp_buy"] + 2 * (c["exchange"] + c["sebi"]) * (1 + c["gst"]) + 2 * c["slippage"]
            + 2 * c["brokerage_pct"] * (1 + c["gst"]))


def _costs(fill: np.ndarray, exit_: np.ndarray, qty: np.ndarray, side: int, c: dict) -> np.ndarray:
    """Vectorised round trip, same formula as costs.round_trip."""
    buy, sell = (fill, exit_) if side > 0 else (exit_, fill)
    bv, sv = buy * qty, sell * qty
    brk = np.minimum(c["brokerage_flat"], c["brokerage_pct"] * bv) + np.minimum(c["brokerage_flat"], c["brokerage_pct"] * sv)
    exch = c["exchange"] * (bv + sv)
    sebi = c["sebi"] * (bv + sv)
    return brk + c["stt_sell"] * sv + exch + sebi + c["stamp_buy"] * bv + c["gst"] * (brk + exch + sebi) + c["slippage"] * (bv + sv)


def barrier_labels(o, h, l, c, minutes, dist, cfg: dict, side: int, horizon: int = H) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R after costs, exit bar and exit reason code (1 target, -1 stop, 0 time/square-off) for an entry at the next
    bar's open after every bar i. NaN where no trade is possible."""
    n = len(c)
    g, cc = cfg["guardrails"], cfg["costs"]
    R = np.full(n, np.nan)
    xb = np.full(n, -1)
    why = np.zeros(n, dtype=np.int8)
    ent = np.arange(1, n + 1)
    ok = (ent < n) & np.isfinite(dist) & (dist > 0)
    ok[ok] &= minutes[ent[ok]] < SQ_MIN
    idx = np.flatnonzero(ok)
    if not len(idx):
        return R, xb, why
    fill = o[ent[idx]]
    d = dist[idx]
    stop = fill - side * d
    tgt = fill + side * g["target_r"] * d
    px = np.full(len(idx), np.nan)
    done = np.zeros(len(idx), bool)
    for j in range(horizon + 1):
        k = ent[idx] + j
        live = ~done & (k < n)
        if not live.any():
            break
        kk = np.minimum(k, n - 1)
        sq = live & ((minutes[kk] >= SQ_MIN) | (j == horizon))          # time barrier / square-off: exit at open
        px[sq], xb[idx[sq]], why[idx[sq]] = o[kk[sq]], kk[sq], 0
        done |= sq
        live &= ~sq
        if side > 0:
            hit_s = live & (l[kk] <= stop)
            hit_t = live & ~hit_s & (h[kk] >= tgt)
            px[hit_s] = np.where(j > 0, np.minimum(o[kk[hit_s]], stop[hit_s]), stop[hit_s])
            px[hit_t] = np.where(j > 0, np.maximum(o[kk[hit_t]], tgt[hit_t]), tgt[hit_t])
        else:
            hit_s = live & (h[kk] >= stop)
            hit_t = live & ~hit_s & (l[kk] <= tgt)
            px[hit_s] = np.where(j > 0, np.maximum(o[kk[hit_s]], stop[hit_s]), stop[hit_s])
            px[hit_t] = np.where(j > 0, np.minimum(o[kk[hit_t]], tgt[hit_t]), tgt[hit_t])
        xb[idx[hit_s | hit_t]] = kk[hit_s | hit_t]
        why[idx[hit_s]], why[idx[hit_t]] = -1, 1
        done |= hit_s | hit_t
    last = ~done                                                         # data ended: exit at the last close
    px[last], xb[idx[last]] = c[n - 1], n - 1
    qty = np.maximum(1.0, np.floor(cfg["pool"] * g["risk_per_trade"] / d))
    gross = side * (px - fill) * qty
    R[idx] = (gross - _costs(fill, px, qty, side, cc)) / (d * qty)
    return R, xb, why


# --------------------------------------------------------------------------- features
def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a, out, m = 2 / (n + 1), np.empty_like(x), x[0]
    for i, v in enumerate(x):
        m = a * v + (1 - a) * m if i else v
        out[i] = m
    return out


@dataclass
class DayBlock:
    """One stock-day: features (bars × channels), context (bars × context), labels and bookkeeping."""
    ticker: str
    day: pd.Timestamp
    ts: pd.DatetimeIndex
    seq: np.ndarray          # (n, C) float32
    ctx: np.ndarray          # (n, K) float32
    rL: np.ndarray           # R after costs, long entry after bar i
    rS: np.ndarray
    dist: np.ndarray         # stop distance in rupees
    close: np.ndarray


def stock_day(b: pd.DataFrame, cx: dict, nifty: pd.DataFrame | None, cfg: dict, atr_mult: float = 1.5) -> DayBlock | None:
    """Features and labels for one stock-day. ``b``: that day's bars (index = bar start). ``cx``: prev_close, adr,
    atr, nr7, avg_volume (all known before the open)."""
    if len(b) < 20 or not cx.get("prev_close") or not cx.get("adr") or cx.get("adr") != cx.get("adr"):
        return None
    o, h, l, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    ts = b.index
    minutes = (ts.hour * 60 + ts.minute).to_numpy()
    pc, adr = float(cx["prev_close"]), float(cx["adr"])
    prev = np.r_[pc, c[:-1]]
    tp = (h + l + c) / 3
    cv = np.cumsum(np.nan_to_num(v))
    vwap = np.where(cv > 0, np.cumsum(tp * np.nan_to_num(v)) / np.maximum(cv, 1e-9), c)
    per_bar_v = max(float(cx.get("avg_volume") or 0) / 75, 1.0)
    hi_so_far, lo_so_far = np.maximum.accumulate(h), np.minimum.accumulate(l)
    nr = np.zeros(n)
    if nifty is not None and len(nifty):
        nc = nifty["close"].reindex(ts).ffill().to_numpy(float)
        nr = np.nan_to_num(np.r_[0.0, np.diff(np.log(nc))]) if n > 1 else nr
        n_since = nc / nc[0] - 1 if np.isfinite(nc[0]) else np.zeros(n)
    else:
        n_since = np.zeros(n)
    s = adr / np.sqrt(75)                                    # a typical 5-minute move, as a fraction of price
    seq = np.column_stack([np.log(c / prev) / s, (h - l) / c / s, (c - o) / c / s,
                           np.where(h > l, (c - l) / np.maximum(h - l, 1e-9), 0.5) - 0.5,
                           (c / vwap - 1) / s, np.log1p(np.nan_to_num(v) / per_bar_v),
                           (c / o[0] - 1) / s, nr / s, (c / hi_so_far - 1) / s, (c / lo_so_far - 1) / s]).astype(np.float32)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr5 = _ema(tr, 14)
    frac = cost_fraction(cfg["costs"])
    dist = np.maximum(atr_mult * atr5, frac * c / cfg["guardrails"]["cost_r_max"])
    dist = np.where(dist <= cfg["guardrails"]["max_stop_pct"] * c, dist, np.nan)
    rL, _, _ = barrier_labels(o, h, l, c, minutes, dist, cfg, +1)
    rS, _, _ = barrier_labels(o, h, l, c, minutes, dist, cfg, -1)
    late = minutes > LAST_ENTRY_MIN
    rL[late], rS[late] = np.nan, np.nan
    rng_used = (hi_so_far - lo_so_far) / (adr * pc)
    ctx = np.column_stack([(minutes - OPEN_MIN) / 375, np.full(n, o[0] / pc - 1) / adr, np.full(n, adr),
                           np.full(n, float(cx.get("atr") or np.nan) / pc), rng_used, np.full(n, float(cx.get("nr7") or 0)),
                           n_since / adr, dist / c, frac * c / dist, np.minimum(np.arange(1, n + 1) / W, 1.0)]
                          ).astype(np.float32)
    return DayBlock(str(b.attrs.get("ticker", "")), ts[0].normalize(), ts, seq, np.nan_to_num(ctx), rL, rS, dist, c)


def windows(seq: np.ndarray, i: np.ndarray) -> np.ndarray:
    """(len(i), W, C) windows ending at bars i, zero-padded before the open."""
    n, C = seq.shape
    out = np.zeros((len(i), W, C), np.float32)
    for j in range(W):
        k = i - (W - 1 - j)
        ok = k >= 0
        out[ok, j] = seq[k[ok]]
    return out


def build(bars_long: pd.DataFrame, cfg: dict, tickers: set | None = None, progress=None) -> list[DayBlock]:
    """DayBlocks for every stock-day (optionally only ``tickers``), with context from earlier days only."""
    from ..replay import daily_context
    ctx_all = daily_context(bars_long)
    bl = bars_long.assign(day=bars_long["ts"].dt.normalize())
    nifty_all = bl[bl["ticker"] == "^NSEI"]
    blocks = []
    days = sorted(bl["day"].unique())
    for di, day in enumerate(days):
        db = bl[bl["day"] == day]
        nf = nifty_all[nifty_all["day"] == day].set_index("ts")[["open", "high", "low", "close", "volume"]]
        try:
            cx = ctx_all.xs(day, level=1)
        except KeyError:
            continue
        for t, g in db.groupby("ticker"):
            if t == "^NSEI" or (tickers is not None and t not in tickers) or t not in cx.index:
                continue
            b = g.set_index("ts")[["open", "high", "low", "close", "volume"]]
            b.attrs["ticker"] = t
            blk = stock_day(b, cx.loc[t].to_dict(), nf, cfg)
            if blk is not None:
                blocks.append(blk)
        if progress:
            progress((di + 1) / len(days))
    return blocks
