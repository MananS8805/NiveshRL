"""Which high-turnover (≈50 trades/day) intraday rules survive NSE costs? Measured on the stored 5-minute history.

Every family is evaluated point-in-time: the day's universe comes from earlier days only (top ``top_liquid`` by trailing
turnover, optionally narrowed to the ``in_play`` stocks with the widest 20-day average range, which predicts tomorrow's
range almost as well as the range model: IC 0.47 vs 0.51), a signal on bar ``i`` (known at its close) is filled at bar
``i+1``'s open, and every trade pays the real MIS round-trip charges plus slippage at the notional it would carry.

Families (each a small parameter grid):
- ``xs_rev`` / ``xs_mom``: every ``every`` bars rank the universe by the last ``look`` bars' return; trade the ``k``
  worst/best (reversal) or best/worst (momentum) long/short, hold ``hold`` bars.
- ``vwap_fade``: price stretched ``z`` intraday sigmas from VWAP is faded; exit after ``hold`` bars.
- ``rvol_break``: a new high (low) of the day on ``m``× normal bar volume is bought (sold); exit after ``hold`` bars.

Parameters are chosen on the first half of the days only; the second half is the holdout that is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..costs import round_trip

OPEN_BAR = 3          # first decision on the 09:30 bar close (bars 09:15, 09:20, 09:25 form the opening range)
LAST_ENTRY = 63       # 14:30 bar (09:15 + 63*5min); exits no later than the 15:15 bar
SQUARE = 72           # 15:15 bar index


def day_matrices(db: pd.DataFrame, tickers) -> dict | None:
    """Wide (bar x ticker) open/high/low/close/volume for one day on the NIFTY clock."""
    n = db[db["ticker"] == "^NSEI"].set_index("ts")
    clock = n.index
    if len(clock) < SQUARE + 1:
        return None
    sub = db[db["ticker"].isin(tickers)]
    out = {}
    for f in ("open", "high", "low", "close", "volume"):
        w = sub.pivot_table(index="ts", columns="ticker", values=f, aggfunc="last").reindex(clock)
        out[f] = w
    out["close"] = out["close"].ffill()
    for f in ("open", "high", "low"):
        out[f] = out[f].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0)
    ok = out["close"].iloc[2].notna()
    return {f: w.loc[:, ok].to_numpy(float) for f, w in out.items()} | {"tickers": list(out["close"].columns[ok])}


def cost_bps(notional: float, cost_cfg: dict, px: float = 1000.0) -> float:
    """Round-trip charges + slippage, in basis points of notional, for a flat trade of this size."""
    qty = max(int(notional // px), 1)
    c = round_trip(px, px, qty, "long", cost_cfg)
    return c.total / (px * qty) * 1e4


def _trades_from_mask(m: dict, sig: np.ndarray, hold: int) -> list[tuple[int, int, int, float]]:
    """sig[bar, j] in {-1, 0, +1}; one position per stock at a time. Returns (bar, j, side, gross return)."""
    o, c = m["open"], m["close"]
    nb, ns = sig.shape
    busy = np.full(ns, -1)
    out = []
    for i in range(OPEN_BAR - 1, min(LAST_ENTRY, nb - 2) + 1):
        for j in np.nonzero(sig[i])[0]:
            if busy[j] >= i:
                continue
            e = i + 1
            x = min(e + hold, SQUARE)
            entry, exit_ = o[e, j], (o[x, j] if x < nb else c[-1, j])
            if not (entry > 0 and exit_ > 0):
                continue
            side = int(sig[i, j])
            out.append((i, j, side, side * (exit_ / entry - 1.0)))
            busy[j] = x
    return out


def _ret(c: np.ndarray, look: int) -> np.ndarray:
    r = np.full_like(c, np.nan)
    r[look:] = c[look:] / c[:-look] - 1
    return r


def signals(m: dict, fam: str, p: dict) -> np.ndarray:
    c, h, l, v = m["close"], m["high"], m["low"], m["volume"]
    nb, ns = c.shape
    sig = np.zeros((nb, ns), dtype=np.int8)
    if fam in ("xs_rev", "xs_mom"):
        r = _ret(c, p["look"])
        for i in range(OPEN_BAR - 1, min(LAST_ENTRY, nb - 1) + 1, p["every"]):
            row = r[i]
            ok = np.isfinite(row)
            if ok.sum() < 2 * p["k"] + 2:
                continue
            idx = np.nonzero(ok)[0]
            order = idx[np.argsort(row[ok])]
            lo, hi = order[: p["k"]], order[-p["k"]:]
            s = 1 if fam == "xs_rev" else -1
            sig[i, lo] = s
            sig[i, hi] = -s
    elif fam == "vwap_fade":
        tp = (h + l + c) / 3
        cv = np.cumsum(v, axis=0)
        vwap = np.cumsum(tp * v, axis=0) / np.where(cv > 0, cv, np.nan)
        # stretch from VWAP in units of a quarter of the stock's 20-day average range (known before the open)
        z = (c / vwap - 1) / (p["_adr"] / 4)
        sig[(z > p["z"])] = -1
        sig[(z < -p["z"])] = 1
    elif fam == "rvol_break":
        hh = np.maximum.accumulate(h, axis=0)
        ll = np.minimum.accumulate(l, axis=0)
        avgv = p["_bar_vol"]                           # normal 5-minute volume from earlier days (avg daily / 75)
        hot = v > p["m"] * avgv
        up = np.zeros_like(hot)
        dn = np.zeros_like(hot)
        up[1:] = (c[1:] > hh[:-1]) & hot[1:]
        dn[1:] = (c[1:] < ll[:-1]) & hot[1:]
        sig[up] = 1
        sig[dn] = -1
    sig[:OPEN_BAR - 1] = 0
    sig[LAST_ENTRY + 1:] = 0
    return sig


def matrices_from_dict(bars: dict[str, pd.DataFrame], tickers: list[str]) -> dict:
    """Live: the same wide arrays from per-ticker frames already on the NIFTY clock (a partial day is fine)."""
    tickers = [t for t in tickers if t in bars and len(bars[t])]
    out = {f: np.column_stack([bars[t][f].to_numpy(float) for t in tickers]) if tickers else np.zeros((0, 0))
           for f in ("open", "high", "low", "close", "volume")}
    return out | {"tickers": tickers}


def day_trades(m: dict, bar_vol: np.ndarray, adr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Every grid rule's trades on one day: (rule index, bar, stock, side, gross return)."""
    rows = []
    for gi, (fam, p) in enumerate(grid()):
        sig = signals(m, fam, p | {"_bar_vol": bar_vol, "_adr": adr})
        rows += [(gi, i, j, side, r) for i, j, side, r in _trades_from_mask(m, sig, p["hold"])]
    return rows


def rule_names() -> list[str]:
    return [f"{fam} " + " ".join(f"{a}={b}" for a, b in p.items()) for fam, p in grid()]


def rule_by_name(name: str) -> tuple[str, dict]:
    return grid()[rule_names().index(name)]


def grid() -> list[tuple[str, dict]]:
    g = []
    for look in (1, 3, 6):
        for every in (3, 6):
            for k in (2, 3):
                for hold in (3, 6, 12):
                    g.append(("xs_rev", {"look": look, "every": every, "k": k, "hold": hold}))
                    g.append(("xs_mom", {"look": look, "every": every, "k": k, "hold": hold}))
    for z in (1.5, 2.0, 2.5):
        for hold in (3, 6, 12):
            g.append(("vwap_fade", {"z": z, "hold": hold}))
    for mm in (2.0, 3.0, 5.0):
        for hold in (3, 6, 12, 60):
            g.append(("rvol_break", {"m": mm, "hold": hold}))
    return g


def run(bars: pd.DataFrame, days: list, ctx_all: pd.DataFrame, top_liquid: int = 150, in_play: int | None = 60,
        min_price: float = 100.0, warmup: int = 6, progress=None) -> pd.DataFrame:
    """One row per (rule, trade): day, rule, side, gross return. Universe and bar-volume norms are point-in-time."""
    rows = []
    G = grid()
    for k, day in enumerate(days[warmup:]):
        try:
            cx = ctx_all.xs(day, level=1)
        except KeyError:
            continue
        cx = cx[(cx["prev_close"] >= min_price) & cx["turn"].notna()]
        cx = cx.drop(index="^NSEI", errors="ignore").sort_values("turn", ascending=False).head(top_liquid)
        if in_play:
            cx = cx.sort_values("adr", ascending=False).head(in_play)
        db = bars[bars["day"] == day]
        m = day_matrices(db, set(cx.index) | {"^NSEI"})
        if m is None:
            continue
        keep = [t != "^NSEI" for t in m["tickers"]]
        m = {f: (w[:, keep] if isinstance(w, np.ndarray) else [t for t, kk in zip(w, keep) if kk]) for f, w in m.items()}
        bar_vol = cx["avg_volume"].reindex(m["tickers"]).to_numpy(float) / 75.0
        adr = cx["adr"].reindex(m["tickers"]).to_numpy(float)
        dstr = str(pd.Timestamp(day).date())
        rows += [(dstr, gi, side, r, i) for gi, i, j, side, r in day_trades(m, bar_vol, adr)]
        if progress:
            progress(dstr, (k + 1) / len(days[warmup:]))
    df = pd.DataFrame(rows, columns=["day", "rule", "side", "gross", "bar"])
    names = dict(enumerate(rule_names()))
    df["name"] = df["rule"].map(names)
    df["family"] = df["rule"].map({gi: fam for gi, (fam, _) in enumerate(G)})
    return df


def summarize(tr: pd.DataFrame, cost: float, split_day: str) -> pd.DataFrame:
    """Per rule and half: trades/day, gross and net bps per trade, daily net t-stat (per-day clustering)."""
    tr = tr.assign(net=tr["gross"] * 1e4 - cost, half=np.where(tr["day"] < split_day, "design", "holdout"))
    out = []
    for (name, half), g in tr.groupby(["name", "half"]):
        d = g.groupby("day")["net"].agg(["sum", "count"])
        n = len(d)
        t = d["sum"].mean() / (d["sum"].std(ddof=1) / np.sqrt(n)) if n > 2 and d["sum"].std() > 0 else np.nan
        out.append({"rule": name, "family": g["family"].iloc[0], "half": half, "days": n,
                    "trades/day": len(g) / n, "gross bps": g["gross"].mean() * 1e4, "net bps": g["net"].mean(),
                    "win": float((g["net"] > 0).mean()), "daily t": t})
    return pd.DataFrame(out)
