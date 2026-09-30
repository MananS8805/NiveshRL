"""Deterministic rollout of any policy over a full split."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .data import MarketData
from .env import PortfolioEnv
from .features import FeatureSet
from .metrics import summarize
from .profile import PRESETS, InvestorProfile


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame     # nav, value, flows per trading day
    trades: pd.DataFrame    # per rebalance: cost, tax, turnover, weights
    metrics: dict

    @property
    def nav(self) -> pd.Series:
        return self.daily["nav"]


def make_eval_env(md: MarketData, fs: FeatureSet, cfg: dict, split: str,
                  profile: InvestorProfile | None = None, action_mode: str = "weights",
                  **kw) -> PortfolioEnv:
    profile = profile or PRESETS["aggressive"]
    return PortfolioEnv(md, fs, cfg, split=split, profile=profile, action_mode=action_mode,
                        full_episode=True, random_init_weights=False, **kw)


def run_policy(policy: Callable, env: PortfolioEnv, name: str) -> BacktestResult:
    """``policy(env) -> action`` (or None to hold). RL policies read ``env.observation()``."""
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(policy(env))
        done = term or trunc
    daily, trades = env.daily_frame(), env.trades_frame()
    bench = benchmark_nav(env.md, daily.index)
    return BacktestResult(name, daily, trades, summarize(daily["nav"], bench, trades,
                                                         rf=env.ecfg["cash_rate"]))


def benchmark_nav(md: MarketData, index: pd.DatetimeIndex) -> pd.Series:
    b = md.bench.reindex(index).ffill()
    return b / b.iloc[0]


def benchmark_result(md: MarketData, index: pd.DatetimeIndex, cash_rate: float) -> BacktestResult:
    """NIFTY 50 index buy-and-hold (no costs; an ETF would add ~0.05%/yr expense)."""
    nav = benchmark_nav(md, index)
    daily = pd.DataFrame({"nav": nav, "value": nav, "flows": 0.0})
    return BacktestResult("NIFTY 50 (buy & hold)", daily, pd.DataFrame(),
                          summarize(nav, nav, None, rf=cash_rate))
