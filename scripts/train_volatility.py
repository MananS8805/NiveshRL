"""Walk-forward next-month volatility forecasts: LSTM vs GARCH(1,1), EWMA, historical.

    python scripts/train_volatility.py [--first-year 2012]

Writes data/predictions/vol_forecasts.parquet and report/results/vol_summary.csv.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from niveshrl.research.data import load_panel
from niveshrl.research import volatility as V

ap = argparse.ArgumentParser()
ap.add_argument("--first-year", type=int, default=2012)
args = ap.parse_args()

p = load_panel()
t = time.time()
frame, seq = V.build_vol_data(p)
print(f"vol dataset {frame.shape}, seq {seq.shape} in {time.time() - t:.0f}s", flush=True)
frame["hist"] = frame["rv21"]
frame["ewma"] = V.ewma_forecast(p, frame).to_numpy()
t = time.time()
frame["garch"] = V.garch_forecast(p, frame, args.first_year)
print(f"garch done in {time.time() - t:.0f}s", flush=True)
t = time.time()
frame["lstm"] = V.lstm_forecast(frame, seq, args.first_year)
print(f"lstm done in {time.time() - t:.0f}s", flush=True)

oos = frame[frame.index.get_level_values(0).year >= args.first_year].dropna(subset=["garch", "lstm"])
V.save_vol(frame[frame.index.get_level_values(0).year >= args.first_year])
table = pd.DataFrame({m: V.vol_metrics(oos[m], oos["rv_next"]) for m in V.VOL_MODELS}).T
Path("report/results").mkdir(parents=True, exist_ok=True)
table.to_csv("report/results/vol_summary.csv")
print(f"\nout-of-sample {oos.index.get_level_values(0).min().date()} -> {oos.index.get_level_values(0).max().date()}, rows where every model has a forecast:")
print(table.round(4).to_string())
