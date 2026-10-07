"""Which intraday decision policy works best after costs? Uses the walk-forward (out-of-sample) predictions from
scripts/intraday_dl_eval.py and replays each policy day by day on the in-play stocks, with the agent's limits
(one position per stock per day, at most ``max_trades`` a day, 1% risk per full-size trade). Every trade's result is
its real triple-barrier R after costs. Writes report/results/intraday_v2.md and intraday_v2_decision.json.

    python scripts/compare_intraday_v2.py [--model tcn|lgbm]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from niveshrl.config import ROOT
from niveshrl.intraday import DIR, config
from niveshrl.intraday.dl import dataset as DS
from niveshrl.intraday.dl.policies import ConformalGate, IQLExit, NeuralLinearTS, exit_paths

OUT = ROOT / "report" / "results"
DLDIR = DIR / "dl"


def candidates(pred: pd.DataFrame, in_play: pd.DataFrame) -> pd.DataFrame:
    d = pred.copy()
    d["day"] = pd.to_datetime(d["ts"]).dt.normalize()
    d = d.merge(in_play.assign(day=pd.to_datetime(in_play["day"]))[["ticker", "day", "join_bar"]], on=["ticker", "day"])
    d = d[d["bar"] >= d["join_bar"]]                       # causal: only bars after the stock came into play
    d["side"] = np.where(d["eL"] >= d["eS"], "L", "S")
    d["e"] = np.where(d["side"] == "L", d["eL"], d["eS"])
    d["r"] = np.where(d["side"] == "L", d["rL"], d["rS"])
    if "pL" not in d:
        d["pL"], d["pS"] = np.nan, np.nan
    return d.dropna(subset=["e", "r"])


def chrono(day_df: pd.DataFrame, allow, max_trades: int, size=None) -> pd.DataFrame:
    """Walk the day's candidates in time order (no peeking at later bars): take one when ``allow(row)`` is true, at
    most one position per stock and ``max_trades`` a day. ``size(row)`` gives 1 / 0.5 (default 1)."""
    taken, seen = [], set()
    for _, r in day_df.sort_values("ts").iterrows():
        if len(taken) >= max_trades:
            break
        if r["ticker"] in seen or not allow(r):
            continue
        sz = 1.0 if size is None else size(r)
        if sz <= 0:
            continue
        seen.add(r["ticker"])
        taken.append(r.to_dict() | {"size": sz})
    return pd.DataFrame(taken)


def summarise(trades: pd.DataFrame, days: list, pool: float, risk: float) -> dict:
    if trades.empty:
        return {"trades": 0, "trades/day": 0.0, "avg R": np.nan, "win rate": np.nan, "total R": 0.0, "net ₹": 0.0,
                "daily t": np.nan, "max drawdown R": 0.0}
    w = trades["size"] * trades["r"]
    by_day = w.groupby(trades["day"]).sum().reindex(days, fill_value=0.0)
    eq = by_day.cumsum()
    return {"trades": int(len(trades)), "trades/day": float(len(trades) / len(days)),
            "avg R": float((w / trades["size"]).mean()), "win rate": float((trades["r"] > 0).mean()),
            "total R": float(w.sum()), "net ₹": float(w.sum() * pool * risk),
            "daily t": float(by_day.mean() / by_day.std() * np.sqrt(len(by_day))) if by_day.std() > 0 else np.nan,
            "max drawdown R": float((eq.cummax() - eq).max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn")
    ap.add_argument("--max-trades", type=int, default=5)
    a = ap.parse_args()
    cfg = config()
    pool, risk = float(cfg["pool"]), cfg["guardrails"]["risk_per_trade"]
    pred = pd.read_parquet(DLDIR / f"oos_{a.model}.parquet")
    ip = pd.read_parquet(DLDIR / "in_play.parquet")
    cand = candidates(pred, ip)
    days = sorted(cand["day"].unique())
    rng = np.random.default_rng(0)
    res, picks_by = {}, {}

    def run(name, chooser):
        out = []
        for d in days:
            dd = cand[cand["day"] == d]
            t = chooser(d, dd)
            if t is not None and len(t):
                out.append(t)
        tr = pd.concat(out) if out else pd.DataFrame(columns=list(cand.columns) + ["size"])
        if "size" not in tr or tr["size"].isna().all():
            tr["size"] = 1.0
        res[name] = summarise(tr, days, pool, risk)
        picks_by[name] = tr
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[name].items()}), flush=True)

    mt = a.max_trades
    # causal thresholds: from the previous 5 days' candidates only
    def thr_for(d, q_trades):
        prev = [x for x in days if x < d][-5:]
        if not prev:
            return np.inf
        e = cand.loc[cand["day"].isin(prev), "e"]
        per_day = len(e) / len(prev)
        q = 1 - min(1.0, q_trades / max(per_day, 1))
        return float(np.quantile(e, q))
    run("random in-play bars (control)", lambda d, dd: chrono(
        dd, lambda r: rng.random() < mt / max(len(dd), 1) * 1.5, mt))
    run(f"model: top-{mt} threshold (causal)", lambda d, dd: chrono(dd, lambda r, t=thr_for(d, mt): r["e"] >= t, mt))
    run("model: only predicted R > 0", lambda d, dd: chrono(dd, lambda r: r["e"] > 0, mt))

    # neural-linear Thompson bandit on the top-20-threshold pool; learns every candidate's outcome after the close
    bandit = NeuralLinearTS(6)
    def bandit_choose(d, dd):
        t20 = thr_for(d, 20)
        pool_ = dd[dd["e"] >= t20]
        def size(r):
            X = NeuralLinearTS.features(pd.DataFrame([r]), r["side"])
            a_ = bandit.decide(X)[0]
            return 1.0 if a_ == "TAKE" else 0.5 if a_ == "HALF" else 0.0
        chosen = chrono(pool_, lambda r: True, mt, size)
        if len(pool_):
            X = np.vstack([NeuralLinearTS.features(pool_.iloc[[i]], s_)[0] for i, s_ in enumerate(pool_["side"])])
            bandit.update(X, pool_["r"].to_numpy(float))
        bandit.end_of_day()
        return chosen
    run("model + neural bandit", bandit_choose)

    gate = ConformalGate(q=0.8, window_days=10)
    def conformal_choose(d, dd):
        t20 = thr_for(d, 20)
        pool_ = dd[dd["e"] >= t20]
        m = gate.margin()
        chosen = chrono(pool_, lambda r: r["e"] - m > 0, mt)
        gate.update(pool_["e"].to_numpy(float), pool_["r"].to_numpy(float))
        return chosen
    run("model + conformal abstention", conformal_choose)

    # offline-RL exits on the 'top N' entries: train on paths from earlier days, apply on later days
    print("building exit paths", flush=True)
    bars = DS.load_bars()
    keep_days = set(pd.to_datetime(days))
    bars = bars[bars["ts"].dt.normalize().isin(keep_days | {d - pd.Timedelta(days=k) for d in keep_days for k in range(1, 30)})]
    tick = set(cand["ticker"])
    blocks = [b for b in DS.build(bars[bars["ticker"].isin(tick | {"^NSEI"})], cfg) if b.day in keep_days]
    top_all = picks_by[f"model: top-{mt} threshold (causal)"].copy()
    top_all["ts"] = pd.to_datetime(top_all["ts"])
    paths = exit_paths(blocks, top_all[["ticker", "ts", "side", "day"]].reset_index(drop=True), cfg)
    tday = top_all.reset_index(drop=True)["day"]
    paths["day"] = tday.reindex(paths["trade"]).to_numpy()
    half = days[len(days) // 2]
    iql = IQLExit().fit(paths[paths["day"] < half])
    ev = iql.evaluate(paths[paths["day"] >= half])
    res["model top + learned exits (IQL), 2nd half"] = {"trades": ev["trades"], "avg R": ev["policy avg R"],
                                                       "fixed exits avg R (same trades)": ev["fixed avg R"],
                                                       "diff vs fixed": ev["diff"], "diff t": ev["t"],
                                                       "avg bars held": ev["avg bars held (policy)"]}
    print("IQL", ev, flush=True)

    t = pd.DataFrame(res).T
    t.index.name = "Policy"
    v1 = {"days": 52, "trades": 151, "avg R": -0.17, "net ₹": -13578}
    eligible = [k for k in res if "control" not in k and "IQL" not in k and res[k].get("trades", 0) >= 20]
    best = max(eligible, key=lambda k: res[k].get("total R", -1e9)) if eligible else "none (no policy made 20 trades)"
    b = res.get(best, {})
    switch = bool(b and b["trades"] >= 20 and b["avg R"] > 0 and (b.get("daily t") or 0) > 1.5)
    cols = [str(c) for c in t.columns]
    lines = ["| Policy | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for i, r in t.iterrows():
        lines.append("| " + i + " | " + " | ".join("" if v != v else f"{v:,.3f}" if isinstance(v, float) else str(v)
                                                   for v in r) + " |")
    md = (f"# Intraday agent v2: decision policies (out of sample, {len(days)} days, {a.model.upper()} predictions)\n\n"
          + "\n".join(lines) +
          f"\n\nFor reference, the current agent (v1, rule setups + tabular bandit + LightGBM scorer) lost "
          f"₹{-v1['net ₹']:,} over its 52-day replay: {v1['trades']} trades at {v1['avg R']:+.2f}R.\n\n"
          f"Best policy by total R: **{best}**. Switch rule: at least 20 trades, average R > 0 after costs and daily "
          f"t > 1.5. Decision: **{'switch the live agent to v2' if switch else 'keep v1 live; run v2 as a shadow paper account'}**."
          f"\n\nSimplifications: one position per stock per day, no overlap limit across stocks, the triple-barrier exit "
          f"(2R target, ATR stop, 60 minutes) for every trade.\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intraday_v2.md").write_text(md, encoding="utf-8")
    t.to_csv(OUT / "intraday_v2.csv")
    (OUT / "intraday_v2_decision.json").write_text(json.dumps({"best": best, "switch": switch, "model": a.model,
                                                               "best_stats": {k: (None if v != v else v) for k, v in b.items()},
                                                               "iql": ev}, indent=2, default=float))
    print(md)


if __name__ == "__main__":
    main()
