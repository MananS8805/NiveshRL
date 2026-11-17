"""Backtest the swing book (research/swing_book.py) on the point-in-time pick history (report/results/
monitor_history_pit.parquet: the 'watch for strength' list rebuilt every 5 trading days from walk-forward
probabilities) and on the random control of the same history, against NIFTY 50 buy-and-hold. Same engine as the live
book. Writes report/results/swing_book_benchmark.md/.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from niveshrl.research import robust as RB  # noqa: E402
from niveshrl.research import swing_book as SB  # noqa: E402
from niveshrl.research.data import load_panel  # noqa: E402

OUT = ROOT / "report" / "results"
t0 = time.time()
p = load_panel(universe=sys.argv[1] if len(sys.argv) > 1 else "pit")
hist = pd.read_parquet(OUT / "monitor_history_pit.parquet")
start = "2017-01-01"
res, books = {}, {}
for grp in ("monitor list", "random control"):
    h = hist[(hist["group"] == grp) & (hist["signal_date"] >= start)]
    picks = {pd.Timestamp(d): list(g["ticker"]) for d, g in h.groupby("signal_date", sort=True)}
    b = SB.replay(p, picks, start)
    books[grp] = b
    res[grp] = b.summary(p.bench)
    print(grp, f"{time.time() - t0:.0f}s", json.dumps(res[grp], default=float), flush=True)
eq = {g: pd.DataFrame(b.s["equity"]).set_index("day")["equity"] for g, b in books.items()}
m = pd.DataFrame(eq)
m.index = pd.to_datetime(m.index)
mon = m.resample("ME").last().pct_change().dropna()
bm = p.bench.reindex(m.index).ffill().resample("ME").last().pct_change().reindex(mon.index)
ex_b = (mon["monitor list"] - bm).dropna()
ex_r = (mon["monitor list"] - mon["random control"]).dropna()
boot_b, boot_r = RB.bootstrap_mean(ex_b.to_numpy(), mean_block=3), RB.bootstrap_mean(ex_r.to_numpy(), mean_block=3)
half = ex_b.index[len(ex_b) // 2]
st = {"book": res["monitor list"], "random_book": res["random control"], "monthly_excess_vs_nifty": boot_b,
      "monthly_excess_vs_random": boot_r, "holdout_from": str(half.date()),
      "holdout_excess_vs_nifty": float(ex_b[ex_b.index >= half].mean()),
      "holdout_excess_vs_random": float(ex_r[ex_r.index >= half].mean()),
      "skips": pd.Series([x["detail"] for x in books["monitor list"].s["log"] if x["event"] == "skip"]).value_counts().head(5).to_dict()}
json.dump(st, open(OUT / "swing_book_benchmark.json", "w"), indent=1, default=float)
b, r = res["monitor list"], res["random control"]
f = lambda x: f"{x:+.1%}"
L = ["# Swing book: the swing plan as a paper portfolio", "",
     f"₹1,00,000, 1% risk per trade (halved below NIFTY's 200-day average or on narrow breadth), at most 20% of equity "
     f"per stock and 10 positions, delivery only (no leverage), real delivery costs on every buy and sale. Picks: the "
     f"point-in-time 'watch for strength' list every 5 trading days, {b['start']} → {b['end']}.", "",
     "| | CAGR | Sharpe | Max drawdown | Trades | Avg R | Avg invested |", "| --- | --- | --- | --- | --- | --- | --- |",
     f"| Swing book | {f(b['cagr'])} | {b['sharpe']:.2f} | {b['max_dd']:.1%} | {b['trades']:,} | {b['avg_r']:+.2f} | {b['exposure']:.0%} |",
     f"| Same rules on random picks | {f(r['cagr'])} | {r['sharpe']:.2f} | {r['max_dd']:.1%} | {r['trades']:,} | {r['avg_r']:+.2f} | {r['exposure']:.0%} |",
     f"| NIFTY 50 buy and hold | {f(b['bench_cagr'])} | {b['bench_sharpe']:.2f} | {b['bench_max_dd']:.1%} | | | 100% |", "",
     f"Monthly excess return vs NIFTY: {boot_b['mean']:+.2%} (90% bootstrap {boot_b['lo90']:+.2%} to {boot_b['hi90']:+.2%}); "
     f"second half (from {half.date()}): {st['holdout_excess_vs_nifty']:+.2%}/month.",
     f"Monthly excess vs the random-pick book: {boot_r['mean']:+.2%} (90% bootstrap {boot_r['lo90']:+.2%} to {boot_r['hi90']:+.2%}); "
     f"second half: {st['holdout_excess_vs_random']:+.2%}/month.", "",
     "Most common reasons a pick was not bought: " + "; ".join(f"{k} ({v})" for k, v in st["skips"].items()) + "."]
(OUT / "swing_book_benchmark.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
