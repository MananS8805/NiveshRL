"""Roadmap step 5: does the volatility LSTM beat the HAR baseline (not just GARCH)?

Builds the same month-end dataset as train_volatility.py, fits a walk-forward pooled HAR (from 2012, 8-year window,
yearly refit), scores it on exactly the rows where the stored LSTM/EWMA/GARCH forecasts exist, and runs
Diebold-Mariano tests on squared log-vol errors (per-date means, Newey-West).

    python scripts/vol_har.py
Writes report/results/vol_har.md and adds a "har" row to report/results/vol_summary.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niveshrl.research import volatility as V  # noqa: E402
from niveshrl.research.data import load_panel  # noqa: E402

p = load_panel()
frame, _ = V.build_vol_data(p)
frame["har"] = V.har_forecast(p, frame, first_year=2012)
stored = V.load_vol().drop(columns="har", errors="ignore")
d = stored.join(frame[["har"]], how="left")
d = d.dropna(subset=["rv_next", "hist", "ewma", "garch", "lstm", "har"])
d = d[(d["rv_next"] > 0)]
models = ["hist", "ewma", "garch", "har", "lstm"]
table = pd.DataFrame({m: V.vol_metrics(d[m], d["rv_next"]) for m in models}).T
loss = {m: (np.log(d[m]) - np.log(d["rv_next"])) ** 2 for m in models}
tests = {f"lstm vs {m}": V.dm_test(loss["lstm"], loss[m]) for m in ("har", "ewma", "garch")}
tests["har vs garch"] = V.dm_test(loss["har"], loss["garch"])

stored["har"] = frame["har"].reindex(stored.index)
V.save_vol(stored)
summ = ROOT / "report/results/vol_summary.csv"
old = pd.read_csv(summ, index_col=0)
old.loc["har"] = V.vol_metrics(d["har"], d["rv_next"])
old.to_csv(summ)

dates = d.index.get_level_values(0)
L = ["# Volatility: LSTM vs the HAR baseline", "",
     f"Out of sample {dates.min().date()} to {dates.max().date()}, {len(d):,} stock-months where every model has a forecast. "
     "HAR = pooled log-HAR on 5-, 21- and 63-day realised volatility, walk-forward (8-year window, refit yearly, one-month "
     "purge). Target: next 21 trading days' realised volatility.", "",
     "| Model | RMSE log-vol | QLIKE | Corr | Bias (f/r) |", "| --- | --- | --- | --- | --- |"]
L += [f"| {m} | {r['RMSE log-vol']:.4f} | {r['QLIKE']:.4f} | {r['Corr']:.3f} | {r['Bias (f/r)']:.3f} |" for m, r in table.iterrows()]
L += ["", "Diebold-Mariano on squared log errors (per-month cross-sectional means, Newey-West, 1 lag); negative t = first model better:", "",
      "| Comparison | mean loss diff | t | p | months |", "| --- | --- | --- | --- | --- |"]
L += [f"| {k} | {v['mean_diff']:+.4f} | {v['t']:+.2f} | {v['p']:.3f} | {v['n_dates']} |" for k, v in tests.items()]
lv = tests["lstm vs har"]
verdict = ("The LSTM beats HAR (p < 0.05): keep it." if lv["t"] < 0 and lv["p"] < 0.05 else
           "HAR beats the LSTM (p < 0.05): HAR should replace it." if lv["t"] > 0 and lv["p"] < 0.05 else
           "No significant difference between the LSTM and HAR: prefer HAR (simpler, no training, no PyTorch).")
L += ["", f"**Gate (roadmap step 5): keep the LSTM only if it beats HAR at p < 0.05.** {verdict}", ""]
(ROOT / "report/results/vol_har.md").write_text("\n".join(L), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(L))
