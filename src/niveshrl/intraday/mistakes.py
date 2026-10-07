"""Why the agent did or did not trade, and what kind of mistake each loss was.

- ``skip_breakdown``: groups a day's decisions by reason (taken, cost rule, ML scorer, learned bucket, guardrail) and,
  once the day's shadow outcomes exist, what each group *would* have made at full size. A group of skips with a
  negative average R means skipping saved money; a positive one means the agent missed profit.
- ``classify_loss``: tags a losing trade (or shadow outcome) with every mistake that fits, so the screen can count
  which mistakes keep happening and which the learner has stopped making.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REASONS = ["taken", "cost rule (stop too tight for costs)", "ML scorer (low profit chance)",
           "learned: this context loses", "learned: uncertain → half size", "guardrail (limits, time, exposure)",
           "model (v2)"]


def reason_of(row: dict) -> str:
    ev, why, det = str(row.get("event", "")), str(row.get("why", "")), str(row.get("detail", ""))
    act = str(row.get("action", ""))
    if ev == "entry":
        return "taken"
    if ev == "blocked":
        return "guardrail (limits, time, exposure)"
    if why.startswith("costs would eat"):
        return "cost rule (stop too tight for costs)"
    if why.startswith("ML:"):
        return "ML scorer (low profit chance)"
    if why.startswith("v2:") or why.startswith("model:"):
        return "model (v2)"
    if act == "HALF":
        return "learned: uncertain → half size"
    if ev == "skip":
        return "learned: this context loses"
    return det or "other"


def skip_breakdown(log: list[dict] | pd.DataFrame, shadow_day: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per reason: signals, share, and (when known) the average / total R they would have made."""
    lg = pd.DataFrame(log)
    if lg.empty or "event" not in lg:
        return pd.DataFrame()
    lg = lg[lg["event"].isin(["entry", "skip", "blocked"])].copy()
    if lg.empty:
        return pd.DataFrame()
    lg["reason"] = [reason_of(r) for r in lg.to_dict("records")]
    out = lg.groupby("reason").size().rename("signals").to_frame()
    out["share"] = out["signals"] / out["signals"].sum()
    if shadow_day is not None and len(shadow_day) and {"ticker", "ts", "net_r"} <= set(shadow_day.columns):
        sh = shadow_day[["ticker", "ts", "net_r"]].drop_duplicates(["ticker", "ts"])
        j = lg.merge(sh, on=["ticker", "ts"], how="left")
        g = j.groupby("reason")["net_r"]
        out["would-be avg R"] = g.mean()
        out["would-be total R"] = g.sum(min_count=1)
    order = [r for r in REASONS if r in out.index] + [r for r in out.index if r not in REASONS]
    return out.reindex(order)


# --------------------------------------------------------------------------- loss taxonomy
MISTAKES = {
    "stopped in the first bar": "the stop was hit on the entry bar: entered into noise or too close a stop",
    "against NIFTY's trend": "traded opposite to the market's 5-minute trend",
    "costs ate it": "the trade was profitable before costs, or costs were over 20% of the risk",
    "late in the day": "entered after 13:45, little time to work",
    "faded a gap": "traded against the day's opening gap",
    "chased an extended move": "entered after the stock had already used most of its usual daily range",
}


def classify_loss(t: dict) -> list[str]:
    """All mistake tags that fit one losing trade. ``t`` needs r and, where available, the signal's features
    (trend_aligned, cost_r, minute, gap, range_used), entry/exit bars or timestamps, gross and costs."""
    tags = []
    if t.get("r", 0) is None or not (t.get("r", 0) < 0):
        return tags
    eb, xb = t.get("entry_bar"), t.get("exit_bar")
    if eb is not None and xb is not None and xb == eb and t.get("reason") == "stop":
        tags.append("stopped in the first bar")
    elif t.get("entry_ts") and t.get("exit_ts") and t.get("entry_ts") == t.get("exit_ts") and t.get("reason") == "stop":
        tags.append("stopped in the first bar")
    if (t.get("trend_aligned") or 0) < 0:
        tags.append("against NIFTY's trend")
    gross, costs = t.get("gross"), t.get("costs")
    if (gross is not None and costs is not None and gross > 0) or (t.get("cost_r") or 0) > 0.2:
        tags.append("costs ate it")
    if (t.get("minute") or 0) > (13 * 60 + 45 - (9 * 60 + 15)):
        tags.append("late in the day")
    if (t.get("gap") or 0) < -0.005:                       # features store gap × side: negative = against the gap
        tags.append("faded a gap")
    if (t.get("range_used") or 0) > 0.9:
        tags.append("chased an extended move")
    return tags


def mistake_table(rows: pd.DataFrame) -> pd.DataFrame:
    """For each mistake: how common it is among losses vs among wins (a mistake that is just as common in wins is not
    what loses money), its average R, and its share of losses in the earlier vs the recent half of the history (is
    the learner making it less often?)."""
    if rows is None or rows.empty or ("net_r" not in rows and "r" not in rows):
        return pd.DataFrame()
    df = rows.copy()
    if "r" not in df:
        df["r"] = df["net_r"]
    df = df.dropna(subset=["r"])
    probe = df.assign(r=-1.0)                                # tag every row as if it lost, to compare with winners
    tags = [classify_loss(r) for r in probe.to_dict("records")]
    lose = (df["r"] < 0).to_numpy()
    days = df["day"].astype(str).to_numpy() if "day" in df else np.array(["all"] * len(df))
    uniq = sorted(set(days))
    cut = uniq[len(uniq) // 2] if len(uniq) > 1 else None
    out = []
    for m, meaning in MISTAKES.items():
        hit = np.array([m in tg for tg in tags])
        if not hit.any():
            continue
        in_l, in_w = hit[lose].mean() if lose.any() else np.nan, hit[~lose].mean() if (~lose).any() else np.nan
        early = hit[lose & (days < cut)].mean() if cut and (lose & (days < cut)).any() else np.nan
        late = hit[lose & (days >= cut)].mean() if cut and (lose & (days >= cut)).any() else np.nan
        out.append({"mistake": m, "losses with it": int((hit & lose).sum()), "in losses": float(in_l),
                    "in wins": float(in_w), "lift": float(in_l / in_w) if in_w else np.nan,
                    "avg R with it": float(df.loc[hit, "r"].mean()), "avg R without": float(df.loc[~hit, "r"].mean())
                    if (~hit).any() else np.nan, "earlier half": float(early), "recent half": float(late),
                    "meaning": meaning})
    return pd.DataFrame(out).set_index("mistake").sort_values("lift", ascending=False)
