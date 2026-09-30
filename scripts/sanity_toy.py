"""Toy sanity check for the custom PPO: can it find the one stock with positive drift?

Six synthetic stocks with identical volatility; only S2 has a strong drift.
A working agent should push S2 to its cap and pay little in turnover.

    python scripts/sanity_toy.py --steps 40000
"""
import argparse
import copy

import numpy as np
import pandas as pd

from niveshrl.agents import TorchPolicy, build_model
from niveshrl.algos.ppo import PPOConfig, PPOTrainer
from niveshrl.backtest import make_eval_env, run_policy
from niveshrl.config import load_config
from niveshrl.data import MarketData
from niveshrl.env import PortfolioEnv
from niveshrl.features import build_features
from niveshrl.profile import InvestorProfile

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40_000)
args = ap.parse_args()

rng = np.random.default_rng(0)
n_days = 2500
dates = pd.bdate_range("2010-01-01", periods=n_days)
tickers = [f"S{i}.NS" for i in range(6)]
drift = np.array([0.0, 0.0, 0.0015, 0.0, 0.0, 0.0])
rets = drift + 0.012 * rng.standard_normal((n_days, 6))
close = pd.DataFrame(100 * np.exp(np.cumsum(rets, 0)), index=dates, columns=tickers)
md = MarketData(dates=dates, tickers=tickers, sectors=["A", "B", "C", "D", "E", "F"], close=close,
                volume=pd.DataFrame(1e6, index=dates, columns=tickers), bench=close.mean(1),
                vix=pd.Series(15.0, index=dates), fx=pd.Series(75.0, index=dates))

cfg = copy.deepcopy(load_config())
cfg["splits"] = {"train": ["2010-01-01", "2016-12-31"], "val": ["2017-01-01", "2019-12-31"]}
cfg["env"]["max_stock_weight"] = 0.6
cfg["env"]["max_sector_weight"] = 1.0
fs = build_features(md, md.split_mask(*cfg["splits"]["train"]))
prof = InvestorProfile(risk_aversion=0.0, dd_tol=0.5, min_cash=0.0)

model = build_model(md, fs, cfg)
env_fns = [lambda i=i: PortfolioEnv(md, fs, cfg, "train", profile=prof, action_mode="weights", seed=i)
           for i in range(8)]
PPOTrainer(model, env_fns, PPOConfig(total_steps=args.steps, eval_every=10**9), out_dir="runs/sanity_toy").train()

res = run_policy(TorchPolicy(model), make_eval_env(md, fs, cfg, "val", profile=prof), "toy")
w = res.trades[tickers + ["CASH"]].mean()
print("\nmean allocation on held-out period:\n" + w.round(3).to_string())
print(f"turnover/yr {res.metrics['Turnover/yr']:.2f}")
ok = w.idxmax() == "S2.NS" and w["S2.NS"] > 0.45
print("SANITY PASS" if ok else "SANITY FAIL")
