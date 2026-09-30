"""Backtest every classical baseline on a split and print the metrics table.

    python scripts/run_baselines.py --split val [--cost-scale 1.0]
"""
import argparse
from pathlib import Path

import pandas as pd

from niveshrl.backtest import benchmark_result, make_eval_env, run_policy
from niveshrl.baselines import ALL_BASELINES
from niveshrl.pipeline import load_all

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["train", "val", "test", "forward"])
ap.add_argument("--cost-scale", type=float, default=1.0)
ap.add_argument("--out", default="report/results")
args = ap.parse_args()
if args.split == "test":
    print("WARNING: evaluating on the TEST split. Do this once, for the final report.")

cfg, md, fs = load_all()
results = []
for cls in ALL_BASELINES:
    pol = cls()
    env = make_eval_env(md, fs, cfg, args.split, cost_scale=args.cost_scale)
    results.append(run_policy(pol, env, pol.name))
    print(f"done: {pol.name}", flush=True)
results.append(benchmark_result(md, results[0].daily.index, cfg["env"]["cash_rate"]))

table = pd.DataFrame({r.name: r.metrics for r in results}).T
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print(table.to_string())
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / f"baselines_{args.split}_cost{args.cost_scale}.csv")
pd.DataFrame({r.name: r.nav for r in results}).to_csv(out / f"baselines_{args.split}_nav.csv")
