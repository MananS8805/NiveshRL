"""Measure, out of sample, whether refitting the next-day models monthly (rolling window) beats the yearly refit the
pipeline used, and whether the 4-model stack beats its best base model. Writes report/results/refit_comparison.{csv,md}
and seeds the rolling base predictions, the stacked predictions and the latest month's model bundle.

    python scripts/compare_refits.py                  # lgbm + logreg + range (+ seq with --seq)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from niveshrl.config import ROOT
from niveshrl.research import nextday as nd
from niveshrl.research import range_model as RM
from niveshrl.research import stacked as ST
from niveshrl.research.data import load_panel

OUT = ROOT / "report" / "results"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.to_numpy()]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", action="store_true", help="include the sequence net (slow on CPU)")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--base-start", default="2024-01-01", help="rolling base predictions start (meta needs 12 months)")
    a = ap.parse_args()
    t0 = time.time()
    p = load_panel()
    dd = nd.build(p)
    X, y, _ = RM.build(p)
    models = ("lgbm", "seq", "logreg") if a.seq else ("lgbm", "logreg")
    test_year = pd.Timestamp(a.test_start).year

    print("yearly refit (what the pipeline did)", flush=True)
    annual = nd.walk_forward(dd, first_test_year=test_year, models=("lgbm", "logreg"), verbose=True)   # as the pipeline ran
    print("monthly rolling refit", flush=True)
    base = ST.rolling(dd, a.base_start, models=models, range_xy=(X, y), save_dir=ST.MODEL_DIR)
    print("stacking", flush=True)
    stk = ST.stack(base, a.test_start)

    lo = pd.Timestamp(a.test_start)
    keep = lambda df: df[df.index.get_level_values(0) >= lo]  # noqa: E731
    annual, base_t, stk_t = keep(annual), keep(base), keep(stk)
    rows = {}
    for name, df, col in [("yearly lgbm (old pipeline)", annual, "lgbm"), ("yearly logreg", annual, "logreg"),
                          ("monthly lgbm", base_t, "lgbm"), ("monthly logreg", base_t, "logreg"),
                          ("monthly seq", base_t, "seq"), ("monthly ensemble", base_t, "ensemble"),
                          ("stacked (4 models)", stk_t, "stacked")]:
        if col in df and df[col].notna().any():
            rows[name] = nd.evaluate(df.dropna(subset=[col]), col)
    table = pd.DataFrame(rows).T
    table.index.name = "Model"
    j = annual[["lgbm"]].rename(columns={"lgbm": "yearly"}).join(
        base_t[["lgbm", "ret_next"]].rename(columns={"lgbm": "monthly"}), how="inner").join(
        stk_t[["stacked"]], how="left")
    tests = {"monthly lgbm vs yearly lgbm": ST.paired_ic(j, "monthly", "yearly"),
             "stacked vs monthly lgbm": ST.paired_ic(j, "stacked", "monthly"),
             "stacked vs yearly lgbm": ST.paired_ic(j, "stacked", "yearly")}
    tt = pd.DataFrame(tests).T
    tt.index.name = "Comparison (daily rank IC, same stock-days)"

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "refit_comparison.csv")
    tt.to_csv(OUT / "refit_tests.csv")
    period = f"{lo:%d %b %Y} → {j.index.get_level_values(0).max():%d %b %Y}"
    md = (f"# Next-day models: yearly vs monthly refit, and the 4-model stack\n\nOut of sample, {period}, today's "
          f"NIFTY 200 members (as the daily pipeline uses). Yearly = refit each January on years Y-5..Y-2, calibrated on "
          f"Y-1. Monthly = refit each month on the last 4 years up to 3 months before, calibrated on those 3 months. "
          f"Stacked = logistic meta model over the base models' ranks, trained on their previous 12 months of "
          f"out-of-sample predictions.\n\n## Skill\n\n{md_table(table)}\n\n## Paired tests\n\n{md_table(tt)}\n\n"
          f"A positive diff with t > 2 means the first model ranked tomorrow's returns better on the same days. "
          f"Run time {time.time() - t0:,.0f}s.\n")
    (OUT / "refit_comparison.md").write_text(md, encoding="utf-8")
    ST.BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(ST.BASE_PATH)
    stk.to_parquet(ST.STACK_PATH)
    (OUT / "refit_decision.json").write_text(json.dumps({
        "monthly_beats_yearly": bool(tests["monthly lgbm vs yearly lgbm"]["diff"] > 0),
        "monthly_vs_yearly_t": tests["monthly lgbm vs yearly lgbm"]["t"],
        "stack_beats_monthly": bool(tests["stacked vs monthly lgbm"]["diff"] > 0),
        "stack_vs_monthly_t": tests["stacked vs monthly lgbm"]["t"], "models": list(models), "period": period},
        indent=2))
    print(md)


if __name__ == "__main__":
    sys.exit(main())
