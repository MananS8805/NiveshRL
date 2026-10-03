"""Forward truth: did the app's suggestions work out?

1. **Forward tracker** — every saved 'watch for strength' pick (data/daily/<date>/monitor.parquet) and an equal-size
   random sample of that day's other eligible stocks, followed with the same swing rules on real daily prices:
   buy at the next open; stop and T1 from research/plans.py (plan computed with data up to the signal date);
   at T1 book a third and move the stop to entry; trail the rest 3 × ATR under the highest close; 60-day safety cap.
   R is after delivery costs. A stock counts once per 30 days per group. Each pick carries a hash of the rules so a
   rule change is visible. No verdict before 100 closed trades per group.
2. **Historical replay of the monitor list** — the same list rebuilt for every past day from the walk-forward next-day
   probabilities and technicals (no news: there is no news history), same exits, plus the random control. Gives
   evidence now instead of after months of forward data. Runs on point-in-time members when available.
3. **Sentiment forward test** — FinBERT sentiment buckets vs the next 1 and 5 days' excess return.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ROOT
from . import plans as P
from .data import Panel

FOLLOW_DAYS = 60
DEDUP_DAYS = 30


def rules_hash() -> str:
    rules = {"stop": [P.STOP_LOOKBACK, P.ATR_MIN, P.ATR_MAX, P.MAX_STOP_PCT], "t": [P.T1_R, P.T2_R, P.TRAIL_ATR],
             "follow": FOLLOW_DAYS, "monitor": __import__("niveshrl.research.monitor", fromlist=["WEIGHTS"]).WEIGHTS}
    return hashlib.sha1(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:10]


@dataclass
class Outcome:
    status: str          # closed | open | no-entry
    entry_date: object
    entry: float
    exit_date: object
    exit: float
    r: float             # after costs (closed) or marked to last close (open)
    days: int
    reason: str


def follow(p: Panel, ticker: str, signal_date: pd.Timestamp, atr_df: pd.DataFrame, qty: int = 100) -> Outcome:
    """Swing rules from the day after ``signal_date``. Uses the plan as of the signal date (no look-ahead)."""
    c = p.close[ticker]
    idx = c.index
    pos = idx.searchsorted(signal_date, side="right")
    if pos >= len(idx) or not np.isfinite(c.iloc[pos]):
        return Outcome("no-entry", None, np.nan, None, np.nan, np.nan, 0, "no next day yet")
    plan = P.make_plan(p, ticker, entry=None, atr_df=atr_df, asof=signal_date)
    if plan is None:
        return Outcome("no-entry", None, np.nan, None, np.nan, np.nan, 0, "no plan")
    o = (p.open[ticker] if p.open is not None else c)
    h = (p.high[ticker] if p.high is not None else c)
    lo = (p.low[ticker] if p.low is not None else c)
    entry = float(o.iloc[pos]) if np.isfinite(o.iloc[pos]) else float(c.iloc[pos])
    dist = plan.risk_per_share
    stop = entry - dist
    t1 = entry + P.T1_R * dist
    booked = 0.0                                   # R already realised on the first third
    frac = 1.0
    hi_close = entry
    a = atr_df[ticker]
    end = min(len(idx), pos + FOLLOW_DAYS)
    for k in range(pos, end):
        ok, hk, lk, ck = o.iloc[k], h.iloc[k], lo.iloc[k], c.iloc[k]
        if not np.isfinite(ck):
            continue
        if lk <= stop:
            px = min(ok, stop) if k > pos else stop
            r = booked + frac * (px - entry) / dist
            cost = P.delivery_costs(entry, px, qty)["total"] / (dist * qty)
            return Outcome("closed", idx[pos], entry, idx[k], float(px), r - cost, k - pos + 1,
                           "trailing stop" if frac < 1 and stop > entry else
                           "breakeven stop" if frac < 1 and stop >= entry else "stop")
        if frac == 1.0 and hk >= t1:
            booked += (1 / 3) * P.T1_R
            frac = 2 / 3
            stop = max(stop, entry)
        hi_close = max(hi_close, ck)
        if frac < 1.0 and np.isfinite(a.iloc[k]):
            stop = max(stop, hi_close - P.TRAIL_ATR * float(a.iloc[k]))
    k = end - 1
    last = float(c.iloc[k])
    r = booked + frac * (last - entry) / dist
    status = "closed" if end - pos >= FOLLOW_DAYS else "open"
    cost = P.delivery_costs(entry, last, qty)["total"] / (dist * qty)
    return Outcome(status, idx[pos], entry, idx[k] if status == "closed" else None, last, r - cost, k - pos + 1,
                   "60-day cap" if status == "closed" else "open (marked to last close)")


def _summary(df: pd.DataFrame) -> dict:
    closed = df[df["status"] == "closed"]
    op = df[df["status"] == "open"]
    r = closed["r"]
    return {"picks": len(df), "closed": len(closed), "open": len(op),
            "win rate": float((r > 0).mean()) if len(r) else np.nan, "avg R": float(r.mean()) if len(r) else np.nan,
            "median R": float(r.median()) if len(r) else np.nan,
            "profit factor": float(r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else np.nan,
            "open avg R (marked)": float(op["r"].mean()) if len(op) else np.nan}


def _dedup(rows: list[dict]) -> list[dict]:
    last: dict[tuple, pd.Timestamp] = {}
    out = []
    for r in sorted(rows, key=lambda x: x["signal_date"]):
        key = (r["group"], r["ticker"])
        d = pd.Timestamp(r["signal_date"])
        if key in last and (d - last[key]).days < DEDUP_DAYS:
            continue
        last[key] = d
        out.append(r)
    return out


def track_saved_picks(p: Panel, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward tracker over every saved daily run. Returns (picks with outcomes, summary by group)."""
    from .technicals import atr
    a = atr(p)
    rng = np.random.default_rng(seed)
    rows = []
    h = rules_hash()
    for folder in sorted((ROOT / "data" / "daily").glob("20*")):
        f = folder / "monitor.parquet"
        if not f.exists():
            continue
        d = pd.Timestamp(folder.name)
        mon = pd.read_parquet(f)
        picks = [t for t in mon[mon["list"] == "watch for strength"].index if t in p.close.columns]
        elig = [t for t in p.close.columns if t not in picks and np.isfinite(p.close[t].asof(d) if d >= p.close.index[0] else np.nan)]
        if p.member is not None:
            elig = [t for t in elig if bool(p.member[t].asof(d))]
        ctrl = list(rng.choice(elig, size=min(len(picks), len(elig)), replace=False)) if elig else []
        for grp, ts in (("monitor list", picks), ("random control", ctrl)):
            for t in ts:
                rows.append({"group": grp, "ticker": t, "signal_date": d, "rules": h})
    rows = _dedup(rows)
    out = []
    for r in rows:
        oc = follow(p, r["ticker"], pd.Timestamp(r["signal_date"]), a)
        out.append(r | oc.__dict__)
    df = pd.DataFrame(out)
    summ = pd.DataFrame({g: _summary(df[df["group"] == g]) for g in df["group"].unique()}).T if len(df) else pd.DataFrame()
    return df, summ


def historical_monitor(p: Panel, pred: pd.DataFrame, n: int = 10, step: int = 5, seed: int = 0,
                       start: str = "2015-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the 'watch for strength' list on past days from walk-forward next-day probabilities and technicals
    (news left out: no history), every ``step`` trading days, and follow it with the swing rules; plus a random control
    of the same size from that day's eligible stocks."""
    from .monitor import components, score
    from .technicals import atr, indicator_frames
    a = atr(p)
    fr = indicator_frames(p)
    keys = ["donchian_breakout", "new_52w_high", "golden_cross_5d", "supertrend", "rsi14", "vol_ratio", "from_52w_high"]
    prob = pred["prob"] if "prob" in pred else pred.get("ensemble", pred.iloc[:, 0])
    days = sorted(set(prob.index.get_level_values(0)))
    days = [d for d in days if d >= pd.Timestamp(start)][::step]
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        pr = prob.xs(d, level=0)
        tech = pd.DataFrame({k: fr[k].loc[d] for k in keys if k in fr}).reindex(pr.index)
        if tech.empty:
            continue
        comp = components(tech, pr, None)
        sc = score(comp).dropna().sort_values(ascending=False)
        picks = list(sc.index[:n])
        others = [t for t in sc.index if t not in picks]
        ctrl = list(rng.choice(others, size=min(n, len(others)), replace=False)) if others else []
        for grp, ts in (("monitor list", picks), ("random control", ctrl)):
            for t in ts:
                rows.append({"group": grp, "ticker": t, "signal_date": d})
    rows = _dedup(rows)
    out = [r | follow(p, r["ticker"], r["signal_date"], a).__dict__ for r in rows]
    df = pd.DataFrame(out)
    summ = pd.DataFrame({g: _summary(df[df["group"] == g]) for g in df["group"].unique()}).T if len(df) else pd.DataFrame()
    return df, summ


def sentiment_test(p: Panel) -> pd.DataFrame:
    """FinBERT sentiment buckets on each saved day vs the next 1 and 5 trading days' return minus the average stock
    (not the median: returns are skewed, so 'beat the median' alone flatters every bucket)."""
    rows = []
    c = p.close
    for folder in sorted((ROOT / "data" / "daily").glob("20*")):
        f = folder / "sentiment.parquet"
        if not f.exists():
            continue
        d = pd.Timestamp(folder.name)
        s = pd.read_parquet(f)
        if "sentiment_adj" not in s or d not in c.index:
            continue
        i = c.index.get_loc(d)
        for h in (1, 5):
            if i + h >= len(c.index):
                continue
            ret = c.iloc[i + h] / c.iloc[i] - 1
            ex = ret - ret.mean()
            for t, v in s["sentiment_adj"].items():
                if t in ex.index and np.isfinite(ex[t]) and np.isfinite(v):
                    rows.append({"date": d, "h": h, "bucket": "negative" if v < -0.2 else "positive" if v > 0.2 else "neutral",
                                 "excess": float(ex[t])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["h", "bucket"])["excess"]
    return pd.DataFrame({"n": g.size(), "avg excess": g.mean(), "hit (beat median)": g.apply(lambda x: (x > 0).mean()),
                         "days": df.groupby(["h", "bucket"])["date"].nunique()})


def edge(df: pd.DataFrame) -> dict:
    """Monitor list minus random control, closed trades only: difference in average R and its t-statistic."""
    c = df[df["status"] == "closed"]
    a = c[c["group"] == "monitor list"]["r"]
    b = c[c["group"] == "random control"]["r"]
    if len(a) < 2 or len(b) < 2:
        return {"edge R": np.nan, "t-stat": np.nan, "n": (len(a), len(b))}
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"edge R": float(diff), "t-stat": float(diff / se) if se > 0 else np.nan, "n": (len(a), len(b))}
