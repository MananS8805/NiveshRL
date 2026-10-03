"""Walk-forward next-day models (LightGBM, deep sequence model, logistic, reversal, ensemble).

    python scripts/train_nextday.py                  # 2015 -> today, all models (~1 h on CPU)
    python scripts/train_nextday.py --first-year 2024 --models lgbm logreg

Writes data/predictions/nextday.parquet (every score out-of-sample) and
report/results/nextday_summary.csv.
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

from niveshrl.research import nextday as nd
from niveshrl.research.data import load_panel
from niveshrl.research.rankers import PRED_DIR

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--first-year", type=int, default=2015)
ap.add_argument("--models", nargs="+", default=["lgbm", "seq", "logreg"])
args = ap.parse_args()

t = time.time()
dd = nd.build(load_panel())
print(f"features {dd.X.shape} in {time.time() - t:.0f}s", flush=True)
pred = nd.walk_forward(dd, args.first_year, models=tuple(args.models))
PRED_DIR.mkdir(parents=True, exist_ok=True)
pred.to_parquet(PRED_DIR / "nextday.parquet")
cols = [m for m in ["ensemble", "lgbm", "seq", "logreg", "reversal"] if m in pred]
table = pd.DataFrame({m: nd.evaluate(pred, m) for m in cols}).T
Path("report/results").mkdir(parents=True, exist_ok=True)
table.to_csv("report/results/nextday_summary.csv")
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(f"\nout-of-sample {pred.index.get_level_values(0).min().date()} -> {pred.index.get_level_values(0).max().date()} "
      f"({time.time() - t:.0f}s total)")
print(table.to_string())
