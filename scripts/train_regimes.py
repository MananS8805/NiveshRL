"""Walk-forward market regimes (autoencoder + k-means). Writes data/predictions/regimes.parquet.

    python scripts/train_regimes.py
"""
import pandas as pd

from niveshrl.research.data import load_panel
from niveshrl.research.regime import detect_regimes, regime_stats, save_regimes

p = load_panel()
w = detect_regimes(p)
save_regimes(w)
stats = regime_stats(w, p)
stats.to_csv("report/results/regime_stats.csv")
print(f"{len(w)} weeks labelled, {w.index[0].date()} -> {w.index[-1].date()}")
print("current regime:", w["regime"].iloc[-1], "as of", w.index[-1].date())
print(stats.round(3).to_string())
by_year = w.groupby([w.index.year, "regime"]).size().unstack(fill_value=0)
print(by_year.to_string())
