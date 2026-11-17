"""Replay the copilot over the stored 5-minute history and produce the evidence for the money benchmarks.

Point-in-time throughout: each day's universe is the ``top_liquid`` stocks by *trailing* average turnover (from earlier
days only), context (previous high/low/close, ADR, ATR, average volume) comes from earlier days, the model is fit
walk-forward on earlier days' cards (refit every ``refit_every_days``), and the adaptive policy (the one that runs
live) replays the tuner day by day on the cards known at each close.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import sim
from ..replay import daily_context
from . import CDIR, config
from .engine import cards_frame, choose_day, outcome_cols
from .model import CardModel
from .setups import all_signals
from .tuner import tune


def load_days(path=None) -> tuple[pd.DataFrame, list]:
    bars = pd.read_parquet(path or CDIR / "hist5m.parquet")
    bars["ts"] = pd.to_datetime(bars["ts"])
    bars = bars.assign(day=bars["ts"].dt.normalize())
    n = bars[bars["ticker"] == "^NSEI"]
    full = n.groupby("day")["ts"].agg(["count", "max"])
    ok = full[(full["count"] >= 70) & (full["max"].dt.hour * 60 + full["max"].dt.minute >= 15 * 60 + 15)].index
    return bars[bars["day"].isin(ok)], sorted(ok)


def day_bars(db: pd.DataFrame, tickers) -> tuple[dict, pd.DataFrame]:
    nifty = db[db["ticker"] == "^NSEI"].set_index("ts")[["open", "high", "low", "close", "volume"]]
    per = {t: g.set_index("ts")[["open", "high", "low", "close", "volume"]] for t, g in db.groupby("ticker")
           if t in tickers and len(g) >= 20}
    clock = nifty.index
    per = {t: b.reindex(clock).ffill().assign(volume=lambda x: x["volume"].fillna(0)) for t, b in per.items()
           if len(clock) > 2 and b.index[0] <= clock[2]}
    return per, nifty


def build_cards(warmup: int = 6, progress=None) -> pd.DataFrame:
    cfg = config()
    bars, days = load_days()
    ctx_all = daily_context(bars.drop(columns="day"))
    ctx_all["turn"] = ctx_all["avg_volume"] * ctx_all["prev_close"]
    out = []
    for k, day in enumerate(days[warmup:]):
        try:
            cx = ctx_all.xs(day, level=1)
        except KeyError:
            continue
        cx = cx[(cx["prev_close"] >= cfg["universe"]["min_price"]) & cx["turn"].notna()]
        tick = set(cx.sort_values("turn", ascending=False).index[: cfg["universe"]["top_liquid"]]) - {"^NSEI"}
        per, nifty = day_bars(bars[bars["day"] == day], tick)
        ctx = {t: cx.loc[t].to_dict() for t in per}
        dstr = str(pd.Timestamp(day).date())
        sigs = all_signals(per, ctx, cfg["guardrails"], nifty)
        c = cards_frame(sigs, cfg, None, dstr)
        if len(c):
            out.append(outcome_cols(c, per, cfg))
        if progress:
            progress(dstr, (k + 1) / len(days[warmup:]))
    cards = pd.concat(out, ignore_index=True)
    cards["source"] = "replay"
    return cards


def walk_forward_p(cards: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    days = sorted(cards["day"].unique())
    cards = cards.copy()
    cards["p"] = np.nan                                     # gated: only from a model with held-out skill (live policy)
    cards["p_raw"] = np.nan                                 # ungated model output (the fixed "model P>=x" research variants)
    skills = []
    m, last_fit = None, -10 ** 9
    for i, d in enumerate(days):
        if i < cfg["model"]["min_train_days"]:
            continue
        if i - last_fit >= cfg["model"]["refit_every_days"]:
            m = CardModel.fit(cards[cards["day"] < d])
            last_fit = i
            skills.append({"day": d, **m.skill, "skilled": m.skilled})
        sel = cards["day"] == d
        if m is not None and m.ready:
            cards.loc[sel, "p_raw"] = m.prob(cards[sel])
            cards.loc[sel, "p"] = m.gated_prob(cards[sel])
    cards.attrs["skills"] = skills
    return cards


def _rules(enabled=None, min_p=0.0, last="14:30") -> dict:
    from .setups import SETUPS
    return {"setups": {s: {"enabled": enabled is None or s in enabled, "min_p": min_p} for s in SETUPS},
            "last_entry": last, "log": []}


LIVE = "adaptive (live policy) · 2R"


def variants() -> dict[str, dict]:
    from .setups import SETUPS
    v = {"rules only · 2R": {"rules": _rules(), "exit": "2r"}, "rules only · hold to close": {"rules": _rules(), "exit": "eod"}}
    for th in (0.45, 0.50, 0.55):
        v[f"model P≥{th:.2f} · 2R"] = {"rules": _rules(min_p=th), "exit": "2r"}
        v[f"model P≥{th:.2f} · hold to close"] = {"rules": _rules(min_p=th), "exit": "eod"}
    for s in SETUPS:
        v[f"{s} only · 2R"] = {"rules": _rules(enabled=[s]), "exit": "2r"}
    v["adaptive, loose tuner: no t-test (rejected) · 2R"] = {"adaptive": True, "exit": "2r",
                                                            "tuning": {"min_cards_threshold": 15, "threshold_t": 0.0,
                                                                       "cluster_by_day": False}}
    v["adaptive, card-level t (rejected) · 2R"] = {"adaptive": True, "exit": "2r", "tuning": {"cluster_by_day": False}}
    v["adaptive (live policy) · 2R"] = {"adaptive": True, "exit": "2r"}
    return v


def simulate(cards: pd.DataFrame, cfg: dict, warm_days: int):
    """Daily net ₹ per variant (1x and 2x slippage) over the evaluation days, the adaptive policy's taken cards, and
    its final rules."""
    g = cfg["guardrails"]
    unit = cfg["pool"] * g["risk_per_trade"]
    days = sorted(cards["day"].unique())
    ev_days = days[warm_days:]
    V = variants()
    net1 = pd.DataFrame(index=ev_days, columns=list(V), dtype=float)
    net2 = net1.copy()
    trades = pd.DataFrame(index=ev_days, columns=list(V), dtype=float)
    start = {"setups": {s: dict(v) for s, v in cfg["rules"]["setups"].items()}, "last_entry": cfg["rules"]["last_entry"], "log": []}
    state = {k: json.loads(json.dumps(start)) for k, v in V.items() if v.get("adaptive")}
    taken_rows = []
    for d in days:
        day_cards = cards[cards["day"] == d]
        raw = day_cards.assign(p=day_cards["p_raw"]) if "p_raw" in day_cards else day_cards
        for name, v in V.items():
            cd = day_cards if name == LIVE else raw             # only the live policy obeys the skill gate
            if v.get("adaptive"):
                rl = state[name]
            else:
                rl = v["rules"]
            rcol = f"r_{v['exit']}_1x"
            idx = choose_day(cd, rl, g, rcol=rcol)
            if d in net1.index:
                net1.at[d, name] = float(cd.loc[idx, rcol].fillna(0).sum() * unit)
                net2.at[d, name] = float(cd.loc[idx, f"r_{v['exit']}_2x"].fillna(0).sum() * unit)
                trades.at[d, name] = len(idx)
                if name == LIVE:
                    taken_rows.append(cd.loc[idx].assign(rcol=rcol))
        known = cards[cards["day"] <= d]
        known_raw = known.assign(p=known["p_raw"]) if "p_raw" in known else known
        for name in state:                                                 # each adaptive variant learns after the close
            state[name], _ = tune(state[name], known if name == LIVE else known_raw, d,
                                  cfg["tuning"] | V[name].get("tuning", {}))
    json_rules = state[LIVE]
    taken = pd.concat(taken_rows, ignore_index=True) if taken_rows else pd.DataFrame()
    return net1, net2, json_rules, taken, trades


def matched_random(taken: pd.DataFrame, cfg: dict, reps: int = 200, seed: int = 0, exit_: str = "2r") -> np.ndarray:
    """Mean R of ``reps`` random portfolios: for each taken card, a random liquid stock on the same day, same signal bar,
    same side and same stop width (percent), traded with the same exits and costs."""
    if not len(taken):
        return np.array([])
    rng = np.random.default_rng(seed)
    bars, _ = load_days()
    ctx_all = daily_context(bars.drop(columns="day"))
    ctx_all["turn"] = ctx_all["avg_volume"] * ctx_all["prev_close"]
    c2 = cfg | {"guardrails": cfg["guardrails"] | {"target_r": cfg["guardrails"]["target_r"] if exit_ == "2r" else 1000.0}}
    draws = np.full((reps, len(taken)), np.nan)
    for d, grp in taken.groupby("day"):
        day = pd.Timestamp(d)
        cx = ctx_all.xs(day, level=1)
        cx = cx[(cx["prev_close"] >= cfg["universe"]["min_price"]) & cx["turn"].notna()]
        tick = list(set(cx.sort_values("turn", ascending=False).index[: cfg["universe"]["top_liquid"]]) - {"^NSEI"})
        per, _ = day_bars(bars[bars["day"] == day], set(tick))
        names = list(per)
        for j, row in grp.iterrows():
            pos = taken.index.get_loc(j)
            for b in range(reps):
                t = names[rng.integers(len(names))]
                bb = per[t]
                ref = float(bb["close"].iloc[int(row["bar"])])
                sp = float(row["stop_pct"])
                stop = ref * (1 - sp) if row["side"] == "long" else ref * (1 + sp)
                oc = sim.outcome(sim.Signal(t, pd.Timestamp(row["ts"]), int(row["bar"]), "random", row["side"], ref, stop), bb, c2)
                draws[b, pos] = oc["r"] if oc else np.nan
    return np.nanmean(draws, axis=1)
