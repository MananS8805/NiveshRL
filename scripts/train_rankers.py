"""Walk-forward train every stock ranker and save out-of-sample predictions.

    python scripts/train_rankers.py                     # all models
    python scripts/train_rankers.py --models ffnn lstm  # a subset
    python scripts/train_rankers.py --window 8 --first-year 2012

Writes data/predictions/<model>.parquet (every score is out-of-sample) and
report/results/rankers_summary.csv, then prints the diagnostics table.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from niveshrl.research import evaluate
from niveshrl.research.data import load_panel
from niveshrl.research.features import build_rank_data
from niveshrl.research.rankers import ALL_MODELS, TrainConfig, save_predictions, walk_forward

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
ap.add_argument("--window", type=int, default=8, help="rolling training window, years")
ap.add_argument("--first-year", type=int, default=2012)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rd = build_rank_data(load_panel())
cfg = TrainConfig(window_years=args.window, seed=args.seed)
rows = {}
for name in args.models:
    t = time.time()
    pred = walk_forward(rd, name, args.first_year, cfg)
    save_predictions(name, pred)
    rows[name] = evaluate.summary(pred) | {"Train time (s)": round(time.time() - t)}
    print(f"{name}: done in {time.time() - t:.0f}s", flush=True)

table = pd.DataFrame(rows).T
out = Path("report/results")
out.mkdir(parents=True, exist_ok=True)
summary_path = out / "rankers_summary.csv"
if summary_path.exists():  # keep rows for models not retrained this time
    old = pd.read_csv(summary_path, index_col=0)
    table = pd.concat([old.drop(index=table.index, errors="ignore"), table])
table.to_csv(summary_path)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(table.to_string())
