"""Does forgetting help the intraday agent? Replays the same ~60 days of 5-minute bars with the bandit and ML scorer
remembering everything (the original) vs fading old outcomes with a half-life, each in its own temporary folder (the
app's own replay is untouched). Writes report/results/forgetting_comparison.md.

    python scripts/compare_forgetting.py [--half-lives 10 20]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from niveshrl.config import ROOT
from niveshrl.intraday import replay

OUT = ROOT / "report" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-lives", type=float, nargs="*", default=[10, 20])
    a = ap.parse_args()
    rows = {}
    for hl in [None, *a.half_lives]:
        name = "remember everything" if hl is None else f"half-life {hl:g} days"
        root = Path(tempfile.mkdtemp())
        out = replay.run(root=root, learning={"half_life_days": hl})
        d = pd.DataFrame(out["days"])
        tr = pd.read_csv(root / "trades.csv") if (root / "trades.csv").exists() else pd.DataFrame()
        half = len(d) // 2
        late = tr[tr["day"].astype(str) >= str(d["day"].iloc[half])] if len(tr) else tr
        rows[name] = {"days": len(d), "pool end ₹": float(d["pool_end"].iloc[-1]), "net ₹": float(d["net"].sum()),
                      "costs ₹": float(d["costs"].sum()), "trades": int(len(tr)),
                      "avg R": float(tr["r"].mean()) if len(tr) else float("nan"),
                      "win rate": float((tr["net"] > 0).mean()) if len(tr) else float("nan"),
                      "second-half avg R": float(late["r"].mean()) if len(late) else float("nan"),
                      "second-half trades": int(len(late)),
                      "random control net ₹": float(d["control_net"].sum())}
        print(name, json.dumps(rows[name]), flush=True)
    t = pd.DataFrame(rows).T
    t.index.name = "Learning"
    best = t["net ₹"].idxmax()
    cols = [str(c) for c in t.columns]
    lines = ["| Learning | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for i, r in t.iterrows():
        lines.append("| " + i + " | " + " | ".join(f"{v:,.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    note = ("Same bars, same rules and guardrails; only how much the bandit and the ML scorer weight old outcomes "
            "differs. ~60 days is a small sample: a difference of a few thousand rupees is within luck. The second-half "
            "columns matter most, because forgetting can only help after there is something to forget.")
    md = "# Intraday agent: forgetting vs remembering everything\n\n" +"\n".join(lines) + f"\n\n{note}\n\nBest net: {best}.\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "forgetting_comparison.md").write_text(md, encoding="utf-8")
    t.to_csv(OUT / "forgetting_comparison.csv")
    print(md)


if __name__ == "__main__":
    main()
