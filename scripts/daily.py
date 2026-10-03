"""Run the daily pipeline (prices, technicals, fundamentals, news, FinBERT sentiment,
next-day model, monitor list, briefing) and write data/daily/<date>/.

    python scripts/daily.py                    # everything (~3-6 min; first run also downloads FinBERT)
    python scripts/daily.py --steps technicals monitor briefing
    python scripts/daily.py --with-seq         # also retrain the deep sequence model for tomorrow's scores (slower)
"""
import argparse
import warnings

from niveshrl.research.daily import STEPS, run

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--steps", nargs="+", choices=STEPS, default=None)
ap.add_argument("--with-seq", action="store_true")
args = ap.parse_args()

meta = run(args.steps, progress=lambda s, f: print(f"[{f:4.0%}] {s}", flush=True), with_seq=args.with_seq)
print(f"\ntrading day {meta['trading_day']} -> data/daily/{meta['folder']}/")
failed = 0
for name, st in meta["steps"].items():
    mark = "ok  " if st["ok"] else "FAIL"
    failed += not st["ok"]
    print(f"  {mark} {name:12s} {st['seconds']:6.1f}s  {st.get('note') or st.get('error', '')}")
raise SystemExit(1 if failed else 0)
