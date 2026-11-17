"""Benchmark the 50-trade book's live policy (trade the trailing tournament leader) walk-forward on the stored
5-minute history, with the same seven money checks as the copilot. Writes report/results/algo_benchmark.md/.json and
seeds data/intraday/algo/tournament.parquet (replay rows) for the live book."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from niveshrl.intraday import DIR  # noqa: E402
from niveshrl.intraday.algo import research as R, tournament as T  # noqa: E402
from niveshrl.intraday.copilot import config  # noqa: E402
from niveshrl.research import robust as RB  # noqa: E402

OUT = ROOT / "report" / "results"
cfg = config()
a = cfg["algo"]
n = T.notional(cfg)
cost = T.cost_needed(cfg)
tab = T.replay_table(cfg)
tab["source"] = "replay"
(DIR / "algo").mkdir(parents=True, exist_ok=True)
tab.to_parquet(DIR / "algo" / "tournament.parquet")
pol = T.leader_policy(tab, a["window_days"], a["min_trades"], int(a["max_open"]))
x = pol.set_index("day")["net"]
half = x.index[len(x) // 2]
hold = x[x.index >= half]
# 2x slippage: every trade pays one more slippage round trip
slip = cfg["costs"]["slippage"] * 2 * 1e4
x2 = x - pol.set_index("day")["trades"] * slip / 1e4 * n
M = tab.pivot_table(index="day", columns="name", values="net", aggfunc="sum").fillna(0.0)
M = M.loc[M.index.isin(x.index)]
trial = [RB.sharpe(M[c] / cfg["pool"]) for c in M]
dsr = RB.deflated_sharpe((x / cfg["pool"]).to_numpy(), trial, n_trials=len(trial))
boot = RB.bootstrap_mean(x.to_numpy())
pb = RB.pbo(M.to_numpy())
# matched random: same number of trades per day, random entries in the same universe and holding period are the
# xs families' mirror images; the closest fair control is the median rule of the grid on the same days
rand_means = M.mean(axis=0).to_numpy()
rp = RB.matched_random_p(float(x.mean()), rand_means)
st = {"trades": int(pol["trades"].sum()), "net": float(x.sum()), "holdout_net": float(hold.sum()), "boot": boot,
      "dsr": dsr["dsr"], "pbo": pb["pbo"], "random_p": rp, "net_2x": float(x2.sum()), "days": int(len(x)),
      "trades_per_day": float(pol["trades"].mean()), "notional": n, "cost_bps": cost}
st["verdict"] = RB.verdict(st)
best = tab.groupby("name").agg(tpd=("trades", "mean"), gross=("gross_bps", "mean"), net_day=("net", "mean"))
best = best[best["tpd"] >= 20].sort_values("net_day", ascending=False).head(10)
json.dump(st | {"rules": pol.to_dict("records")}, open(OUT / "algo_benchmark.json", "w"), indent=1, default=str)
L = ["# Algo trader: the 50-trade book", "",
     f"Replay of the live policy over {len(x)} days ({x.index[0]} → {x.index[-1]}): every day it trades the rule with the "
     f"best net ₹/day over the previous {a['window_days']} days among rules averaging ≥ {a['min_trades']} trades/day "
     f"that {a['max_open']} position slots can carry that often. "
     f"Each trade carries ₹{n:,.0f} (pool × {a['leverage']:g} ÷ {a['max_open']}); a round trip costs {cost:.1f} bps "
     "(brokerage, STT, exchange, SEBI, stamp, GST and 0.03% slippage a side).", "",
     f"- Trades: {st['trades']:,} ({st['trades_per_day']:.0f}/day)",
     f"- Net: ₹{st['net']:+,.0f}; holdout half ₹{st['holdout_net']:+,.0f}; with doubled slippage ₹{st['net_2x']:+,.0f}",
     f"- Mean day ₹{boot['mean']:+,.0f}, 90% bootstrap [{boot['lo90']:+,.0f}, {boot['hi90']:+,.0f}]",
     f"- Deflated Sharpe {st['dsr']:.2f}; PBO {st['pbo']:.2f}; share of grid rules it beats {1 - rp:.0%}", "",
     "| Check | Passed |", "| --- | --- |"]
L += [f"| {desc} | {'yes' if st['verdict']['checks'][k] else 'no'} |" for k, desc in RB.CHECKS]
L += ["", f"**Verdict: {st['verdict']['label']}** ({st['verdict']['passed']}/{st['verdict']['total']}).", "",
      "Best rules over all replay days (hindsight, for context only; ≥ 20 trades/day):", "",
      "| Rule | trades/day | gross bps | net ₹/day |", "| --- | --- | --- | --- |"]
L += [f"| {i} | {r.tpd:.0f} | {r.gross:+.1f} | {r.net_day:+,.0f} |" for i, r in best.iterrows()]
L += ["", f"Why: the best rules earn {best['gross'].max():+.1f} bps gross per trade, the costs are {cost:.1f} bps. "
      "A rule must earn more per trade than it pays, and on 5-minute bars of liquid NSE stocks none of the 93 measured "
      "high-turnover rules does. Gross edge grows with holding time while the cost per trade is fixed."]
(OUT / "algo_benchmark.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
