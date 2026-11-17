"""Measure high-turnover intraday rule families net of real NSE MIS costs (see niveshrl.intraday.algo.research)."""
import sys
import time

import pandas as pd

from niveshrl.intraday.algo import research as R
from niveshrl.intraday.copilot import config
from niveshrl.intraday.copilot.backtest import load_days
from niveshrl.intraday.replay import daily_context

sys.stdout.reconfigure(encoding="utf-8")
cfg = config()
bars, days = load_days()
ctx = daily_context(bars.drop(columns="day"))
ctx["turn"] = ctx["avg_volume"] * ctx["prev_close"]
t0 = time.time()
out = {}
for label, ip in (("in-play 60", 60), ("liquid 150", None)):
    tr = R.run(bars, days, ctx, in_play=ip)
    tr.to_parquet(f"data/intraday/algo_research_{ip or 150}.parquet")
    out[label] = tr
    print(label, len(tr), "trades", f"{time.time() - t0:.0f}s", flush=True)
ev = sorted(out["in-play 60"]["day"].unique())
split = ev[len(ev) // 2]
for notional in (100_000, 33_000):
    c = R.cost_bps(notional, cfg["costs"])
    print(f"\ncost at ₹{notional:,} notional: {c:.1f} bps round trip; split {split}")
    for label, tr in out.items():
        s = R.summarize(tr, c, split)
        d = s[s["half"] == "design"].set_index("rule")
        h = s[s["half"] == "holdout"].set_index("rule")
        j = d.join(h, lsuffix=" D", rsuffix=" H")
        j = j[j["trades/day D"] >= 20].sort_values("net bps D", ascending=False)
        print(f"-- {label}: best 8 rules by design-half net (≥20 trades/day)")
        print(j[["trades/day D", "gross bps D", "net bps D", "daily t D", "gross bps H", "net bps H", "daily t H"]].head(8).round(2).to_string())
        fam = s.groupby(["family", "half"])[["gross bps"]].mean().unstack().round(2)
        print(fam.to_string())
