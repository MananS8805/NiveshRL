"""Money-readiness benchmark for the intraday copilot (and, for comparison, the v1 agent's replay).

1. Replays the copilot over data/intraday/copilot/hist5m.parquet (point-in-time universe, walk-forward model).
2. Simulates every policy variant tried (rules / model thresholds / exits / single setups / the adaptive live policy).
3. Applies the reliability tests in niveshrl.research.robust: untouched holdout, stationary bootstrap, deflated Sharpe
   over all variants tried, probability of backtest overfitting (CSCV), matched random entries, doubled costs.
4. Seeds the live copilot: cards (source=replay) for the tuner, the walk-forward model refit on all days, and the
   adaptive policy's final rules.

    python scripts/copilot_benchmark.py [--reps 200]
Writes report/results/copilot_benchmark.md and .json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niveshrl.intraday import DIR  # noqa: E402
from niveshrl.intraday.copilot import CDIR, config, save_json  # noqa: E402
from niveshrl.intraday.copilot import backtest as B  # noqa: E402
from niveshrl.intraday.copilot.engine import CARDS, MODEL  # noqa: E402
from niveshrl.intraday.copilot.model import CardModel  # noqa: E402
from niveshrl.research import robust as RB  # noqa: E402

OUT = ROOT / "report" / "results"


def stats_for(name, net1, net2, trades, cards_taken, trial_sr, M, holdout_days, random_means=None):
    x = net1[name].astype(float)
    pool = config()["pool"]
    boot = RB.bootstrap_mean(x.to_numpy())
    dsr = RB.deflated_sharpe((x / pool).to_numpy(), trial_sr, n_trials=len(trial_sr))
    rp = RB.matched_random_p(cards_taken, random_means) if random_means is not None else float("nan")
    st = {"trades": int(trades[name].sum()), "net": float(x.sum()), "holdout_net": float(x.loc[holdout_days].sum()),
          "boot": boot, "dsr": dsr["dsr"], "pbo": M["pbo"], "random_p": rp, "net_2x": float(net2[name].sum()),
          "sharpe_daily": dsr["sr"], "sr0": dsr["sr0"], "win_days": float((x > 0).mean()),
          "max_dd": float((x.cumsum() - x.cumsum().cummax()).min())}
    st["verdict"] = RB.verdict(st)
    return st


def v1_check() -> dict | None:
    p = DIR / "replay" / "state.json"
    if not p.exists():
        return None
    days = pd.DataFrame(json.loads(p.read_text(encoding="utf-8")).get("days", []))
    if not len(days):
        return None
    x = days["net"].astype(float).to_numpy()
    boot = RB.bootstrap_mean(x)
    return {"days": len(x), "trades": int(days["trades"].sum()), "net": float(x.sum()), "boot": boot}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--cached", action="store_true", help="reuse the replay cards from the last run")
    a = ap.parse_args()
    cfg = config()
    t0 = time.time()
    cache = CDIR / "bench_cards.parquet"
    if a.cached and cache.exists():
        cards = B.walk_forward_p(pd.read_parquet(cache).drop(columns=["p", "p_raw"], errors="ignore"), cfg)
    else:
        print("building cards…", flush=True)
        cards = B.build_cards(progress=lambda d, f: print(f"  {d} {f:.0%}", flush=True) if int(f * 100) % 10 == 0 else None)
        cards = B.walk_forward_p(cards, cfg)
        CDIR.mkdir(parents=True, exist_ok=True)
        cards.to_parquet(cache)
    days = sorted(cards["day"].unique())
    warm = cfg["model"]["min_train_days"]
    net1, net2, final_rules, taken, trades = B.simulate(cards, cfg, warm)
    ev = list(net1.index)
    holdout = ev[len(ev) // 2:]
    design = ev[: len(ev) // 2]
    trial_sr = [RB.sharpe(net1[c].astype(float).to_numpy()) for c in net1]
    M = RB.pbo(net1.astype(float).fillna(0).to_numpy())
    print(f"variants {len(net1.columns)}, eval days {len(ev)}, PBO {M['pbo']:.2f}", flush=True)
    live = B.LIVE
    best_design = net1.loc[design].sum().idxmax()
    rand_live = B.matched_random(taken[taken["day"].isin(ev)], cfg, reps=a.reps) if len(taken) else np.array([])
    live_mean_r = float(taken.loc[taken["day"].isin(ev), "r_2r_1x"].mean()) if len(taken) else float("nan")
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "days_total": len(days), "eval_days": len(ev),
           "design_days": [design[0], design[-1]] if design else None, "holdout_days": [holdout[0], holdout[-1]] if holdout else None,
           "cards": int(len(cards)), "variants": list(net1.columns), "pbo": M,
           "live_policy": stats_for(live, net1, net2, trades, live_mean_r, trial_sr, M, holdout, rand_live),
           "best_in_design": best_design,
           "best_in_design_stats": stats_for(best_design, net1, net2, trades, float("nan"), trial_sr, M, holdout),
           "table": [], "random_mean_r": {"policy": live_mean_r, "random_median": float(np.nanmedian(rand_live)) if len(rand_live) else None,
                                          "random_p95": float(np.nanquantile(rand_live, 0.95)) if len(rand_live) else None},
           "per_setup": cards[cards["day"].isin(ev)].groupby(["setup", "side"])["r_2r_1x"].agg(["count", "mean"]).round(3).reset_index().to_dict("records"),
           "final_rules": {k: v for k, v in final_rules.items() if k != "log"}, "rule_changes": final_rules.get("log", []),
           "model_skill": cards.attrs.get("skills", []),
           "v1_agent": v1_check()}
    for c in net1.columns:
        x = net1[c].astype(float)
        res["table"].append({"variant": c, "trades": int(trades[c].sum()), "net": round(float(x.sum()), 0),
                             "holdout_net": round(float(x.loc[holdout].sum()), 0), "net_2x_costs": round(float(net2[c].sum()), 0),
                             "win_days": round(float((x > 0).mean()), 3), "sharpe_daily": round(RB.sharpe(x.to_numpy()), 3)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "copilot_benchmark.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (OUT / "copilot_benchmark.md").write_text(markdown(res), encoding="utf-8")
    pd.DataFrame(res["table"]).set_index("variant").to_csv(OUT / "copilot_benchmark.csv")
    # seed the live copilot with what the replay learned
    CDIR.mkdir(parents=True, exist_ok=True)
    cards.to_parquet(CARDS)
    m = CardModel.fit(cards)
    if m.ready:
        m.save(MODEL)
    save_json("rules.json", final_rules)
    print(markdown(res))
    print(f"done in {time.time() - t0:.0f}s")


def _f(x, pct=False):
    if x is None or x != x:
        return "n/a"
    return f"{x:.0%}" if pct else f"{x:,.2f}"


def markdown(r: dict) -> str:
    L = ["# Intraday copilot: money-readiness benchmark", "",
         f"Generated {r['generated']}. {r['days_total']} trading days of 5-minute bars; the first days train the model "
         f"(walk-forward), leaving **{r['eval_days']} evaluation days** (design {r['design_days'][0]} to {r['design_days'][1]}, "
         f"untouched holdout {r['holdout_days'][0]} to {r['holdout_days'][1]}). {r['cards']:,} cards (every signal, taken or not). "
         "Universe: the 150 most-traded stocks by trailing turnover (point in time). ₹1,00,000 pool, 1% risk per trade, "
         "≤ 3 trades a day, NSE intraday costs + 0.03% slippage per side.", "",
         "## The live policy (adaptive rules, 2R exits)", ""]
    s = r["live_policy"]
    L += ["| Test | Result | Pass |", "| --- | --- | --- |"]
    ck = s["verdict"]["checks"]
    rows = [("Trades in evaluation", f"{s['trades']}", ck["enough_trades"]),
            ("Net ₹ on the untouched holdout", f"₹{s['holdout_net']:+,.0f}", ck["holdout_positive"]),
            ("Mean daily P&L, 90% bootstrap CI", f"₹{s['boot']['mean']:+,.0f} [{s['boot']['lo90']:+,.0f}, {s['boot']['hi90']:+,.0f}]", ck["bootstrap_lo_positive"]),
            ("Deflated Sharpe (vs best of all variants by luck)", _f(s["dsr"]), ck["deflated_sharpe"]),
            ("Probability of backtest overfitting", _f(r["pbo"]["pbo"]), ck["low_overfitting"]),
            ("Matched random entries: p-value", _f(s["random_p"]), ck["beats_matched_random"]),
            ("Net ₹ with slippage doubled", f"₹{s['net_2x']:+,.0f}", ck["survives_2x_costs"])]
    L += [f"| {a} | {b} | {'✓' if c else '✗'} |" for a, b, c in rows]
    L += ["", f"**Verdict: {s['verdict']['label']}** ({s['verdict']['passed']}/{s['verdict']['total']} checks). "
          f"Whole evaluation: net ₹{s['net']:+,.0f}, {s['win_days']:.0%} winning days, max drawdown ₹{s['max_dd']:,.0f}.", "",
          f"Matched random control: policy {_f(r['random_mean_r']['policy'])}R per trade vs random median "
          f"{_f(r['random_mean_r']['random_median'])}R (95th percentile {_f(r['random_mean_r']['random_p95'])}R).", "",
          "## Every variant tried (all count toward the deflated Sharpe and PBO)", "",
          "| Variant | Trades | Net ₹ | Holdout ₹ | Net ₹, 2× slippage | Winning days | Daily Sharpe |", "| --- | --- | --- | --- | --- | --- | --- |"]
    L += [f"| {t['variant']} | {t['trades']} | {t['net']:+,.0f} | {t['holdout_net']:+,.0f} | {t['net_2x_costs']:+,.0f} | {t['win_days']:.0%} | {t['sharpe_daily']:.3f} |"
          for t in r["table"]]
    b = r["best_in_design_stats"]
    L += ["", f"Best variant chosen on the design half: **{r['best_in_design']}** → holdout ₹{b['holdout_net']:+,.0f}, "
          f"verdict {b['verdict']['label']} ({b['verdict']['passed']}/{b['verdict']['total']}).", "",
          "## Every setup, all cards in the evaluation (full size, 2R exits)", "", "| Setup | Side | Cards | Mean R |", "| --- | --- | --- | --- |"]
    L += [f"| {p['setup']} | {p['side']} | {p['count']} | {p['mean']:+.3f} |" for p in r["per_setup"]]
    sk = r.get("model_skill", [])
    if sk:
        n_sk = sum(1 for s in sk if s.get("skilled"))
        aucs = [s["auc"] for s in sk if s.get("auc") is not None]
        bss = [s["brier_skill"] for s in sk if s.get("brier_skill") is not None]
        L += ["", "## Does the P(win) model have skill?", "",
              f"At each of {len(sk)} walk-forward refits the model was scored on held-out days before it could filter trades "
              f"(needs AUC ≥ 0.55 and a positive Brier skill score vs the base rate). It qualified at **{n_sk} of {len(sk)}** "
              f"refits; held-out AUC ranged {min(aucs):.2f}–{max(aucs):.2f}"
              + (f", Brier skill {min(bss):+.3f} to {max(bss):+.3f}." if bss else ".")
              + " When it does not qualify, the live policy runs on the rules alone."]
    L += ["", "## What the adaptive policy learned over the replay", ""]
    L += [f"- {c['day']}: {c['rule']} {c['from']} → {c['to']} ({c['why']})" for c in r["rule_changes"][-15:]] or ["- no rule changes"]
    v1 = r.get("v1_agent")
    if v1:
        L += ["", "## For comparison: the v1 agent's replay", "",
              f"{v1['days']} days, {v1['trades']} trades, net ₹{v1['net']:+,.0f}; mean daily ₹{v1['boot']['mean']:+,.0f} "
              f"[{v1['boot']['lo90']:+,.0f}, {v1['boot']['hi90']:+,.0f}] (90% bootstrap CI)."]
    L += ["", "## How to read this", "",
          "- A strategy is called MONEY-READY only when all seven checks pass. Anything less stays paper-only; the forward "
          "scorecard in the app keeps applying the same tests to live paper results as they accumulate.",
          "- Limits: ~4 months of Yahoo 5-minute bars, fills modelled at the next bar's open plus slippage (not real order-book "
          "fills), no volume cap in the backtest (liquid large caps only), and one market regime."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
