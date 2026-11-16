"""Evidence-based rule tuning. After each day the copilot looks at the last ``window_days`` trading days of cards
(every signal, taken or not, with its full-size R after costs) and may change only these rules:

- **bench / restore a setup**: bench when >= ``min_cards_disable`` outcomes average below 0 with t <= ``disable_t``;
  restore when >= ``min_cards_enable`` outcomes average > 0.05R with t >= ``enable_t``.
- **probability bar per setup** (``min_p``): the grid value with the best mean R among cards at or above it (>=
  ``min_cards_threshold`` cards); it moves one grid step per day, and only for a gain >= ``min_gain_r``.
- **last entry time**: the latest cutoff whose later cards are not significantly negative.

All t-statistics are on per-day mean R with >= 8 distinct days (cards on one day are not independent: an earlier
card-level version benched setups after 3 days of evidence and is kept in the benchmark as rejected).
Risk per trade, max trades, daily loss stop and square-off are guardrails and are never tuned. Every change is logged
with its evidence (n, mean R, t). The function is pure, so the backtest replays exactly what the live copilot learns.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd


MIN_DAYS = 8
CLUSTER = True


def _stats(r: pd.Series, day: pd.Series | None = None) -> tuple[int, float, float]:
    """(cards, mean R per card, t). Cards from the same day move together (one market, one session), so by default the
    t-statistic is computed on per-day mean R and is NaN with fewer than ``MIN_DAYS`` distinct days."""
    ok = r.notna()
    r = r[ok]
    n = len(r)
    if n < 2:
        return n, float(r.mean()) if n else float("nan"), float("nan")
    if CLUSTER and day is not None:
        dm = r.groupby(day[ok]).mean()
        if len(dm) < MIN_DAYS:
            return n, float(r.mean()), float("nan")
        sd = dm.std(ddof=1)
        return n, float(r.mean()), float(dm.mean() / (sd / np.sqrt(len(dm)))) if sd > 0 else float("nan")
    sd = r.std(ddof=1)
    return n, float(r.mean()), float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")


def _minute(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return (int(h) * 60 + int(m)) - (9 * 60 + 15)


def tune(rules: dict, cards: pd.DataFrame, day: str, tcfg: dict, rcol: str = "r") -> tuple[dict, list[dict]]:
    global CLUSTER
    CLUSTER = bool(tcfg.get("cluster_by_day", True))
    """``cards``: columns day, setup, minute (since 09:15 at bar end), p, and ``rcol`` (full-size R after costs).
    Only cards from days before or on ``day`` within the window are used. Returns (new rules, changes)."""
    new = copy.deepcopy(rules)
    new.setdefault("log", [])
    changes: list[dict] = []
    if cards is None or not len(cards):
        return new, changes
    days = sorted(d for d in cards["day"].unique() if d <= day)[-int(tcfg["window_days"]):]
    w = cards[cards["day"].isin(days)].dropna(subset=[rcol])

    def log(rule, frm, to, n, mean, t, why):
        changes.append({"day": day, "rule": rule, "from": frm, "to": to, "n": int(n), "mean_r": round(mean, 3),
                        "t": None if t != t else round(t, 2), "why": why})

    for s, sr in new["setups"].items():
        c = w[w["setup"] == s]
        n, m, t = _stats(c[rcol], c["day"])
        if sr.get("enabled", True):
            if n >= tcfg["min_cards_disable"] and m < 0 and t == t and t <= tcfg["disable_t"]:
                sr["enabled"] = False
                log(f"{s}.enabled", True, False, n, m, t, f"last {len(days)} days: {n} cards averaged {m:+.2f}R (t {t:.1f})")
        else:
            if n >= tcfg["min_cards_enable"] and m > 0.05 and t == t and t >= tcfg["enable_t"]:
                sr["enabled"] = True
                log(f"{s}.enabled", False, True, n, m, t, f"recovered: {n} cards averaged {m:+.2f}R (t {t:.1f})")
        if "p" in c and c["p"].notna().sum() >= tcfg["min_cards_threshold"]:
            grid = list(tcfg["p_grid"])
            cur = sr.get("min_p", grid[1])
            i0 = min(range(len(grid)), key=lambda k: abs(grid[k] - cur))
            need_t = float(tcfg.get("threshold_t", 0.0))
            moved = False
            for nxt in ([grid[i0 + 1]] if i0 + 1 < len(grid) else []) + ([grid[i0 - 1]] if i0 > 0 else []):
                lo, hi = min(cur, nxt), max(cur, nxt)
                bm = (c["p"] >= lo) & (c["p"] < hi)                 # the cards this one step adds or removes
                n2, m2, t2 = _stats(c.loc[bm, rcol], c.loc[bm, "day"])
                if n2 < tcfg["min_cards_threshold"] or t2 != t2:
                    continue
                raise_bar = nxt > cur
                # raising the bar drops the band: only if the band clearly loses; lowering adds it: only if it clearly wins
                if (raise_bar and m2 < -tcfg["min_gain_r"] and t2 <= -need_t) or                    (not raise_bar and m2 > tcfg["min_gain_r"] and t2 >= need_t):
                    sr["min_p"] = nxt
                    log(f"{s}.min_p", cur, nxt, n2, m2, t2,
                        f"the {n2} cards with P between {lo:.2f} and {hi:.2f} averaged {m2:+.2f}R (t {t2:.1f}): "
                        + ("raised the bar to drop them" if raise_bar else "lowered the bar to include them"))
                    moved = True
                    break
            if not moved:
                pass
    cur_last = new.get("last_entry", "14:30")
    best_last = tcfg["late_cutoffs"][-1]
    for cut in tcfg["late_cutoffs"]:                       # earliest cutoff after which cards are clearly negative
        late = w[w["minute"] > _minute(cut)]
        n, m, t = _stats(late[rcol], late["day"])
        if n >= tcfg["min_cards_disable"] and m < 0 and t == t and t <= tcfg["disable_t"]:
            best_last = cut
            break
    if best_last != cur_last:
        late = w[w["minute"] > _minute(min(best_last, cur_last))]
        n, m, t = _stats(late[rcol], late["day"])
        log("last_entry", cur_last, best_last, n, m, t,
            "entries after this time lost significantly" if best_last < cur_last else "late entries no longer significantly negative")
        new["last_entry"] = best_last
    new["log"] = (new["log"] + changes)[-500:]
    return new, changes
