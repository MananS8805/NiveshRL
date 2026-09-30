"""Evaluate trained agents against every baseline on one split.

    python scripts/evaluate.py --split val --agent "NiveshRL=runs/ppo_cnn_dsr_reb5_s0"
    python scripts/evaluate.py --split test --agent ... --cost-sweep --frontier   # final, run ONCE

Writes the metrics table, bootstrap confidence intervals and figures to report/.
"""
import argparse
from pathlib import Path

import pandas as pd

from niveshrl import plots
from niveshrl.agents import SB3Policy, TorchPolicy, load_run
from niveshrl.backtest import benchmark_result, make_eval_env, run_policy
from niveshrl.baselines import ALL_BASELINES, EqualWeight, Markowitz, MinVariance
from niveshrl.metrics import block_bootstrap_sharpe_diff, daily_returns
from niveshrl.pipeline import load_all
from niveshrl.profile import PRESETS, InvestorProfile

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test", "forward"])
ap.add_argument("--agent", action="append", default=[],
                help="NAME=RUN_DIR (custom PPO) or NAME=sb3:RUN_DIR (Stable-Baselines3)")
ap.add_argument("--profile", default="aggressive", choices=list(PRESETS))
ap.add_argument("--cost-sweep", action="store_true")
ap.add_argument("--frontier", action="store_true")
ap.add_argument("--checkpoint", default="best")
args = ap.parse_args()
if args.split == "test":
    print("NOTE: TEST split. Final evaluation only; do not tune anything on these numbers.")

cfg, md, fs = load_all()
fig_dir = Path("report/figures") / args.split
res_dir = Path("report/results")
fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)


def agent_factory(spec: str):
    """Returns (name, action_mode, policy_factory)."""
    name, path = spec.split("=", 1)
    if path.startswith("sb3:"):
        from stable_baselines3 import PPO, SAC
        run = Path(path[4:])
        algo = SAC if "sac" in run.name else PPO
        model = algo.load(run / args.checkpoint, device="cpu")
        return name, "logits", lambda: SB3Policy(model)
    run = Path(path)
    model = load_run(run, md, fs, cfg, args.checkpoint)
    return name, "weights", lambda: TorchPolicy(model)


agents = [agent_factory(s) for s in args.agent]
profile = PRESETS[args.profile]


def run_all(cost_scale=1.0, baselines=ALL_BASELINES):
    out = {}
    for cls in baselines:
        pol = cls()
        out[pol.name] = run_policy(pol, make_eval_env(md, fs, cfg, args.split, cost_scale=cost_scale), pol.name)
    for name, mode, fac in agents:
        env = make_eval_env(md, fs, cfg, args.split, profile=profile, action_mode=mode, cost_scale=cost_scale)
        out[name] = run_policy(fac(), env, name)
    return out


results = run_all()
first = next(iter(results.values()))
bench = benchmark_result(md, first.daily.index, cfg["env"]["cash_rate"])
results[bench.name] = bench

table = pd.DataFrame({n: r.metrics for n, r in results.items()}).T
table.to_csv(res_dir / f"metrics_{args.split}.csv")
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print(table.to_string())

navs = {n: r.nav for n, r in results.items()}
plots.equity_curves(navs, fig_dir / "equity_curves.png")
plots.drawdowns(navs, fig_dir / "drawdowns.png")

if agents:
    ci_rows = []
    for name, _, _ in agents:
        ra = daily_returns(results[name].nav)
        for other in [bench.name, "Equal weight", "Markowitz max-Sharpe"]:
            ci = block_bootstrap_sharpe_diff(ra, daily_returns(results[other].nav), rf=cfg["env"]["cash_rate"])
            ci_rows.append({"agent": name, "vs": other, **ci})
        plots.sector_allocation(results[name].trades, md.tickers, md.sectors, md.vix,
                                fig_dir / f"sector_allocation_{name}.png")
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(res_dir / f"sharpe_ci_{args.split}.csv", index=False)
    print("\nSharpe difference, 95% block-bootstrap CI:\n" + ci.to_string(index=False))

if args.cost_sweep:
    rows = {}
    for scale in [0.0, 0.5, 1.0, 2.0, 4.0]:
        r = run_all(scale, baselines=[EqualWeight, Markowitz, MinVariance])
        rows[scale] = {n: x.metrics["Sharpe"] for n, x in r.items()}
        print(f"cost x{scale}: " + ", ".join(f"{n} {v:.2f}" for n, v in rows[scale].items()))
    sweep = pd.DataFrame(rows).T
    sweep.to_csv(res_dir / f"cost_sweep_{args.split}.csv")
    plots.cost_sensitivity(sweep, fig_dir / "cost_sensitivity.png")

if args.frontier and agents:
    name, mode, fac = agents[0]
    pts = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        prof = InvestorProfile(risk_aversion=lam, dd_tol=0.35 - 0.25 * lam, horizon_years=10,
                               min_cash=0.0)
        r = run_policy(fac(), make_eval_env(md, fs, cfg, args.split, profile=prof, action_mode=mode), name)
        pts.append({"risk_aversion": lam, **r.metrics})
    pts = pd.DataFrame(pts)
    pts.to_csv(res_dir / f"frontier_{args.split}.csv", index=False)
    plots.frontier(pts, fig_dir / "frontier.png")
    mono = pts["Vol"].is_monotonic_decreasing
    print("\nfrontier:\n" + pts[["risk_aversion", "CAGR", "Vol", "MaxDD", "Sharpe"]].to_string(index=False))
    print(f"volatility monotone decreasing in lambda: {mono}")

    prof_rows = {}
    for pname, prof in PRESETS.items():
        r = run_policy(fac(), make_eval_env(md, fs, cfg, args.split, profile=prof, action_mode=mode), pname)
        prof_rows[pname] = r.metrics
    pd.DataFrame(prof_rows).T.to_csv(res_dir / f"profiles_{args.split}.csv")
    print("\npreset profiles:\n" + pd.DataFrame(prof_rows).T[["CAGR", "Vol", "Sharpe", "MaxDD"]].to_string())

print(f"\nfigures -> {fig_dir}   tables -> {res_dir}")
