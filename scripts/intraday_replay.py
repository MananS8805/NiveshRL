"""Replay the intraday paper agent over Yahoo's last ~60 days of 5-minute bars (whole liquid NSE universe).

    python scripts/intraday_replay.py               # download bars (if missing) and replay
    python scripts/intraday_replay.py --download    # force a fresh download first

Prints each day (trades, gross, costs, net, the random control) and the totals, and writes
report/results/intraday_replay.csv. History only, paper only.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from niveshrl.intraday import replay, universe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--download", action="store_true")
ap.add_argument("--days", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

if args.download or not replay.BARS.exists():
    u = universe.load()
    print(f"downloading 5-minute bars for {len(u)} stocks + NIFTY…", flush=True)
    replay.download(u.index.tolist(), progress=lambda f: print(f"  {f:.0%}", end="", flush=True))
    print()


def show(day, r, frac):
    print(f"{pd.Timestamp(day).date()}  signals {r.signals:4d}  trades {r.trades:3d}  gross ₹{r.gross:+9,.0f}  "
          f"costs ₹{r.costs:7,.0f}  net ₹{r.net:+9,.0f}  pool ₹{r.pool_end:,.0f}  control ₹{r.control_net:+8,.0f}"
          f"{'  ★ +10%' if r.hit_target else ''}{'  learned ' + str(len(r.learned)) if r.learned else ''}", flush=True)


out = replay.run(days=args.days, seed=args.seed, progress=show)
s = out["summary"]
print("\n=== replay summary ===")
for k, v in s.items():
    print(f"  {k:16} {v:,.2f}" if isinstance(v, float) else f"  {k:16} {v}")
pd.DataFrame(out["days"]).drop(columns=["learned"]).to_csv(ROOT / "report" / "results" / "intraday_replay.csv", index=False)
