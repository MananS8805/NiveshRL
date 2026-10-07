"""Does meta-labeling improve the swing picks? Walk-forward 2017-2026 on point-in-time members.
Writes report/results/swing_meta.md, swing_meta_decision.json and the out-of-sample predictions.

    python scripts/compare_swing_meta.py
"""
from __future__ import annotations

import json
import time

import pandas as pd

from niveshrl.config import ROOT
from niveshrl.research import swing_meta as SM
from niveshrl.research.data import load_panel

OUT = ROOT / "report" / "results"


def main() -> None:
    t0 = time.time()
    hist = pd.read_parquet(SM.HIST).reset_index(drop=True)
    hist = hist[hist["status"] == "closed"].reset_index(drop=True)
    p = load_panel(universe="pit")
    F = SM.features(p, hist)
    print(f"features {F.shape} in {time.time() - t0:.0f}s", flush=True)
    pred = SM.walk_forward(hist, F)
    ev = SM.evaluate(hist, pred)
    rows = {k: v for k, v in ev.items() if isinstance(v, dict) and "trades" in v}
    t = pd.DataFrame(rows).T
    t.index.name = "Swing trades (test years 2017-2026)"
    k = ev["kept − all"]
    switch = bool(k["diff R"] > 0 and k["t"] > 2)
    cols = [str(c) for c in t.columns]
    lines = ["| Group | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for i, r in t.iterrows():
        lines.append("| " + i + " | " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    md = ("# Track record v2: meta-labeling the swing picks\n\nPoint-in-time trades under the plan rules, R after "
          "costs. Each year's predictions come from LightGBM + a small neural network trained only on trades that had "
          "closed before that year.\n\n" + "\n".join(lines) +
          f"\n\nKeeping the better half of each day's picks vs all picks: {k['diff R']:+.3f}R per trade "
          f"(t {k['t']:.2f}, {k['days']} signal days). Rank correlation of predicted and realised R: "
          f"{ev['corr(pred, R)']:.3f}.\n\nDecision: **{'use the meta-label to rank the daily list' if switch else 'keep the list as it is (no reliable improvement)'}**. "
          f"Run time {time.time() - t0:,.0f}s.\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "swing_meta.md").write_text(md, encoding="utf-8")
    t.to_csv(OUT / "swing_meta.csv")
    hist.join(pred).to_parquet(OUT / "swing_meta_oos.parquet")
    (OUT / "swing_meta_decision.json").write_text(json.dumps({"switch": switch, **k, "corr": ev["corr(pred, R)"]}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
