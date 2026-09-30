"""Does the policy actually change its allocation with the investor's risk aversion?

    python scripts/frontier_check.py runs/verify_utility_s0 runs/ppo_cnn_dsr_reb5_s0

For each run, backtest on validation with risk aversion in {0, .25, .5, .75, 1}
(no cash floor, so any cash comes from the policy itself). Prints average
cash weight, volatility, CAGR and max drawdown per level.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from niveshrl.agents import TorchPolicy, load_run
from niveshrl.backtest import make_eval_env, run_policy
from niveshrl.pipeline import load_all
from niveshrl.profile import InvestorProfile

cfg, md, fs = load_all()
for run in map(Path, sys.argv[1:]):
    a = json.loads((run / "args.json").read_text())
    model = load_run(run, md, fs, cfg)
    rows = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        prof = InvestorProfile(risk_aversion=lam, dd_tol=0.35 - 0.25 * lam, horizon_years=10, min_cash=0.0)
        res = run_policy(TorchPolicy(model), make_eval_env(md, fs, cfg, "val", profile=prof), str(lam))
        m = res.metrics
        rows.append({"risk_aversion": lam, "avg_cash": res.trades["CASH"].mean(), "Vol": m["Vol"],
                     "CAGR": m["CAGR"], "MaxDD": m["MaxDD"], "Sharpe": m["Sharpe"]})
    df = pd.DataFrame(rows)
    print(f"\n{run.name} (reward: {a.get('cfg_reward', {}).get('kind', '?')})")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"vol spread {df['Vol'].iloc[0] - df['Vol'].iloc[-1]:+.3f}, "
          f"cash spread {df['avg_cash'].iloc[-1] - df['avg_cash'].iloc[0]:+.3f}, "
          f"vol monotone decreasing: {df['Vol'].is_monotonic_decreasing}")
