"""Survivorship report: every result on today's NIFTY 200 (biased) vs the index as it was (point-in-time).

    python scripts/survivorship_report.py [--paths 200]

Needs data/constituents/ (scripts/fetch_constituents.py) and, for the model rows, the point-in-time
predictions from `python scripts/train_rankers.py --universe pit` (and train_nextday.py --universe pit).
Both universes are measured over the same months (from the first snapshot), with the same rules and costs:
- ranking skill (IC, spread) of each monthly model;
- a top-20, equal-weight, monthly strategy per signal, after costs;
- a luck test: the same rules with random picks (5/50/95% bands, the strategy's percentile);
- NIFTY buy-and-hold over the same days.
Writes report/results/survivorship.csv and report/results/survivorship.md.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from niveshrl.research import backtest as bt, constituents as cs, evaluate  # noqa: E402
from niveshrl.research.data import load_panel  # noqa: E402
from niveshrl.research.rankers import load_predictions  # noqa: E402
from niveshrl.research.signals import FACTORS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--paths", type=int, default=200)
ap.add_argument("--top", type=float, default=20)
args = ap.parse_args()

start = (cs.first_date() + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
panels = {"today's members (biased)": load_panel(), "point-in-time members": load_panel(universe="pit")}
SIGNALS = ["momentum", "equal", "ffnn", "lstm", "transformer", "logreg"]
rows = []
for uni, p in panels.items():
    pit = p.member is not None
    for sig in SIGNALS:
        if sig in FACTORS:
            scores = bt.factor_scores(p, sig)
            pred = None
        else:
            pred = load_predictions(sig + ("_pit" if pit else ""))
            if pred is None:
                print(f"  {uni}: no predictions for {sig}, skipped")
                continue
            pred = pred[pred.index.get_level_values(0) >= pd.Timestamp(start)]
            scores = pred["score"].unstack("ticker")
        row = {"universe": uni, "signal": sig}
        if pred is not None and len(pred):
            s = evaluate.summary(pred)
            row |= {"IC mean": s["IC mean"], "IC t-stat": s["IC t-stat"], "Spread / mo": s["Spread / mo"]}
        spec = bt.StrategySpec(name=sig, signal=sig, top=1.0 if sig == "equal" else args.top,
                               max_weight=0.05 if sig == "equal" else 0.10, start=start)
        lt = bt.luck_test(p, scores, spec, n_paths=0 if sig == "equal" else args.paths, seed=0)
        st = lt["strategy"]
        row |= {"CAGR": st["CAGR"], "Sharpe": st["Sharpe"], "MaxDD": st["MaxDD"], "Return/DD": st["Return/DD"]}
        if sig != "equal":
            b = lt["bands"]
            row |= {"Random CAGR p5": b.loc[0.05, "CAGR"], "Random CAGR p50": b.loc[0.5, "CAGR"],
                    "Random CAGR p95": b.loc[0.95, "CAGR"], "Beats % of random (CAGR)": lt["percentile"]["CAGR"],
                    "Beats % of random (Return/DD)": lt["percentile"]["Return/DD"]}
        row |= {"NIFTY CAGR": lt["buy_hold"]["CAGR"], "NIFTY Return/DD": lt["buy_hold"]["Return/DD"],
                "Period": f"{lt['period'][0]:%Y-%m} → {lt['period'][1]:%Y-%m}"}
        rows.append(row)
        print(f"{uni:28} {sig:12} CAGR {st['CAGR']:+.1%}  Sharpe {st['Sharpe']:.2f}  MaxDD {st['MaxDD']:.1%}", flush=True)

tab = pd.DataFrame(rows)
out = ROOT / "report" / "results"
out.mkdir(parents=True, exist_ok=True)
tab.to_csv(out / "survivorship.csv", index=False)

# coverage: share of member-days that have prices
p = panels["point-in-time members"]
h = cs.history()
allt = list(cs.all_tickers(h).index)
M = cs.member_mask(p.close.index, allt, h)
M = M[M.index >= pd.Timestamp(start)]
have = pd.DataFrame(False, index=M.index, columns=allt)
cols = [t for t in allt if t in p.close.columns]
have[cols] = p.close.reindex(M.index)[cols].notna().values
coverage = float((M & have).values.sum() / M.values.sum())
snaps = ", ".join(d.strftime("%Y-%m-%d") for d, _ in cs.snapshot_files())

fmt = {"IC mean": "{:.3f}", "IC t-stat": "{:.2f}", "Spread / mo": "{:+.2%}", "CAGR": "{:+.1%}", "Sharpe": "{:.2f}",
       "MaxDD": "{:.1%}", "Return/DD": "{:.2f}", "Random CAGR p5": "{:+.1%}", "Random CAGR p50": "{:+.1%}",
       "Random CAGR p95": "{:+.1%}", "Beats % of random (CAGR)": "{:.0%}", "Beats % of random (Return/DD)": "{:.0%}",
       "NIFTY CAGR": "{:+.1%}", "NIFTY Return/DD": "{:.2f}"}
show = tab.copy()
for c, f in fmt.items():
    if c in show:
        show[c] = show[c].map(lambda v, f=f: "–" if pd.isna(v) else f.format(v))
md = ["# Survivorship report\n",
      f"Period {start} → {p.close.index[-1]:%Y-%m-%d}; top {args.top:g} stocks, equal weight, monthly, after NSE costs; "
      f"luck test = {args.paths} random-pick paths with the same rules (seed 0).\n",
      f"Point-in-time membership from {len(cs.snapshot_files())} archived NSE snapshots ({snaps}); a stock counts "
      f"from the snapshot it appears in (stale between snapshots, never anticipated). Price coverage of member-days: "
      f"**{coverage:.1%}**; the missing ones are mostly delisted or merged companies, which still flatters results.\n",
      show.to_markdown(index=False)]
(out / "survivorship.md").write_text("\n".join(md), encoding="utf-8")
print(f"\ncoverage {coverage:.1%}; wrote {out / 'survivorship.csv'} and survivorship.md")
