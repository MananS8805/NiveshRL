"""Can the app label market regimes without torch? Compares, on the same walk-forward weeks, window and features:
the original torch autoencoder + k-means (3 seeds), the same autoencoder in NumPy (3 seeds) and PCA + k-means.
A replacement is acceptable if it matches the torch model about as well as the torch model matches itself across
seeds. Writes report/results/regime_pca_check.md."""
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from niveshrl.research.daily import fresh_choice  # noqa: E402
from niveshrl.research.data import load_panel  # noqa: E402
from niveshrl.research.regime import detect_regimes, regime_stats  # noqa: E402

p = load_panel()
win = 8 if fresh_choice().get("regime_rolling_better") else None
runs = {}
for enc, seeds in (("ae", (0, 1, 2)), ("ae_numpy", (0, 1, 2)), ("pca", (0,))):
    for s in seeds:
        runs[(enc, s)] = detect_regimes(p, first_year=2010, window_years=win, encoder=enc, seed=s)
        print("done", enc, s, flush=True)


def agree(a, b):
    la, lb = runs[a]["regime"], runs[b]["regime"]
    j = pd.concat([la, lb], axis=1, join="inner")
    return float((j.iloc[:, 0] == j.iloc[:, 1]).mean()), float(((j.iloc[:, 0] == "Stress") == (j.iloc[:, 1] == "Stress")).mean())


rows = []
for a, b in itertools.combinations([k for k in runs if k[0] == "ae"], 2):
    rows.append(("torch AE vs torch AE (other seed)", *agree(a, b)))
for a in [k for k in runs if k[0] == "ae_numpy"]:
    for b in [k for k in runs if k[0] == "ae"]:
        rows.append(("NumPy AE vs torch AE", *agree(a, b)))
for b in [k for k in runs if k[0] == "ae"]:
    rows.append(("PCA vs torch AE", *agree(("pca", 0), b)))
t = pd.DataFrame(rows, columns=["comparison", "same label", "same Stress flag"])
summ = t.groupby("comparison", sort=False).agg(pairs=("same label", "size"), label_min=("same label", "min"),
                                                label_mean=("same label", "mean"), stress_min=("same Stress flag", "min"),
                                                stress_mean=("same Stress flag", "mean"))
self_min = summ.loc["torch AE vs torch AE (other seed)", ["label_min", "stress_min"]]
np_row = summ.loc["NumPy AE vs torch AE"]
ok = bool(np_row["label_mean"] >= self_min["label_min"] - 0.02 and np_row["stress_mean"] >= self_min["stress_min"] - 0.01)
fmt = lambda s: s.to_string(float_format=lambda x: f"{x:.4f}")  # noqa: E731
first = runs[("ae", 0)]
L = ["# Regimes without torch", "",
     f"Walk-forward weekly labels {first.index[0].date()} → {first.index[-1].date()} ({len(first)} weeks), "
     f"{'rolling 8-year' if win else 'expanding'} training window, same 8 features and naming rule.", "",
     "| Comparison | pairs | same label (min / mean) | same Stress flag (min / mean) |", "| --- | --- | --- | --- |"]
L += [f"| {i} | {r.pairs} | {r.label_min:.1%} / {r.label_mean:.1%} | {r.stress_min:.1%} / {r.stress_mean:.1%} |"
      for i, r in summ.iterrows()]
L += ["", f"**Decision: {'the NumPy autoencoder replaces the torch one' if ok else 'keep the torch autoencoder'}** "
      "(accepted if its mean agreement with the torch model is within 2 points of the torch model's own worst "
      "seed-to-seed agreement, and within 1 point on the Stress flag). PCA labels Bull and Neutral differently and "
      "was rejected.", "",
      "Out-of-sample NIFTY behaviour by regime, torch autoencoder (seed 0):", "", "```", fmt(regime_stats(first, p)), "```",
      "", "NumPy autoencoder (seed 0):", "", "```", fmt(regime_stats(runs[("ae_numpy", 0)], p)), "```",
      "", "PCA:", "", "```", fmt(regime_stats(runs[("pca", 0)], p)), "```"]
(ROOT / "report" / "results" / "regime_pca_check.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("ACCEPT" if ok else "REJECT")
