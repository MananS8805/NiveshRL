"""The rule tournament behind the 50-trade book: every rule in the grid is scored every day after costs (live days
in shadow, replay days from the stored history), and the book trades the trailing leader among rules that make at
least ``min_trades`` trades a day. Choosing the leader only from *earlier* days is itself a policy, so it is
benchmarked walk-forward exactly like that (``leader_policy``)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import research as R


def notional(cfg: dict) -> float:
    a = cfg["algo"]
    return float(cfg["pool"]) * float(a["leverage"]) / int(a["max_open"])


def daily_table(trades: pd.DataFrame, cost_bps: float, notional_rs: float) -> pd.DataFrame:
    """(day, rule) -> trades, gross bps, net ₹ (every trade at the book's notional, after costs)."""
    t = trades.assign(net_bps=trades["gross"] * 1e4 - cost_bps)
    g = t.groupby(["day", "name"]).agg(trades=("net_bps", "size"), gross_bps=("gross", lambda x: x.mean() * 1e4),
                                       net_bps=("net_bps", "mean"), net_sum=("net_bps", "sum"))
    g["net"] = g["net_sum"] / 1e4 * notional_rs
    return g.drop(columns="net_sum").reset_index()


def capacity(name: str, max_open: int) -> float:
    """Trades a day the book can carry for a rule: ``max_open`` slots, each busy for the rule's holding period."""
    hold = int(name.rsplit("hold=", 1)[1].split()[0])
    return max_open * (R.LAST_ENTRY - R.OPEN_BAR + 2) / hold


def leader(table: pd.DataFrame, before: str, window: int, min_trades: float, max_open: int | None = None) -> str | None:
    """Best trailing net ₹/day over the last ``window`` days before ``before``, among rules averaging ≥ min_trades/day
    that the book can actually carry that often (``max_open`` slots × holding period). Live rows replace replay rows."""
    if "source" in table:
        table = table.sort_values("source", key=lambda s: s.eq("live")).drop_duplicates(["day", "name"], keep="last")
    days = sorted(d for d in table["day"].unique() if d < before)[-window:]
    if len(days) < max(5, window // 2):
        return None
    w = table[table["day"].isin(days)]
    full = pd.MultiIndex.from_product([days, w["name"].unique()], names=["day", "name"])
    w = w.set_index(["day", "name"]).reindex(full).fillna({"trades": 0, "net": 0}).reset_index()
    s = w.groupby("name").agg(tpd=("trades", "mean"), net=("net", "mean"))
    s = s[s["tpd"] >= min_trades]
    if max_open:
        s = s[[capacity(n, max_open) >= min_trades for n in s.index]]
    return None if s.empty else str(s["net"].idxmax())


def leader_policy(table: pd.DataFrame, window: int, min_trades: float, max_open: int | None = None) -> pd.DataFrame:
    """Walk-forward: each day trade the leader chosen from earlier days only. One row per day traded."""
    out = []
    for d in sorted(table["day"].unique()):
        name = leader(table, d, window, min_trades, max_open)
        if name is None:
            continue
        row = table[(table["day"] == d) & (table["name"] == name)]
        out.append({"day": d, "rule": name, "trades": int(row["trades"].sum()), "net": float(row["net"].sum())})
    return pd.DataFrame(out)


def replay_table(cfg: dict, progress=None) -> pd.DataFrame:
    from ..copilot.backtest import load_days
    from ..replay import daily_context
    bars, days = load_days()
    ctx = daily_context(bars.drop(columns="day"))
    ctx["turn"] = ctx["avg_volume"] * ctx["prev_close"]
    a = cfg["algo"]
    tr = R.run(bars, days, ctx, top_liquid=a["top_liquid"], in_play=a.get("in_play"), progress=progress)
    n = notional(cfg)
    return daily_table(tr, R.cost_bps(n, cfg["costs"]), n)


def cost_needed(cfg: dict) -> float:
    return R.cost_bps(notional(cfg), cfg["costs"])


__all__ = ["daily_table", "leader", "leader_policy", "replay_table", "notional", "cost_needed", "np"]
