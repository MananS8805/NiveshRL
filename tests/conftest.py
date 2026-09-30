import copy

import numpy as np
import pandas as pd
import pytest

from niveshrl.config import load_config
from niveshrl.data import MarketData
from niveshrl.features import build_features


def synthetic_md(n_days: int = 900, seed: int = 0) -> MarketData:
    """Six GBM stocks in three sectors, on business days from 2016."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    tickers = [f"S{i}.NS" for i in range(6)]
    sectors = ["A", "A", "B", "B", "C", "C"]
    drift = np.array([0.0006, 0.0002, 0.0004, -0.0001, 0.0003, 0.0005])
    rets = drift + 0.015 * rng.standard_normal((n_days, 6))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    volume = pd.DataFrame(rng.integers(1e5, 1e6, (n_days, 6)).astype(float), index=dates, columns=tickers)
    bench = close.mean(axis=1) * 100
    vix = pd.Series(15 + rng.standard_normal(n_days).cumsum() * 0.1, index=dates).clip(8)
    fx = pd.Series(75 + rng.standard_normal(n_days).cumsum() * 0.05, index=dates)
    return MarketData(dates=dates, tickers=tickers, sectors=sectors, close=close,
                      volume=volume, bench=bench, vix=vix, fx=fx)


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["splits"] = {"train": ["2016-01-01", "2018-06-30"], "val": ["2018-07-01", "2019-06-30"],
                   "test": ["2018-07-01", "2019-06-30"]}
    c["features"]["lookback"] = 4
    c["env"]["max_stock_weight"] = 0.5
    c["env"]["max_sector_weight"] = 1.0
    c["env"]["episode_steps"] = 10
    return c


@pytest.fixture
def md():
    return synthetic_md()


@pytest.fixture
def fs(md, cfg):
    return build_features(md, md.split_mask(*cfg["splits"]["train"]))
