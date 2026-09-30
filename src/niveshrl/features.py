"""Causal feature engineering.

Every feature at row t is computed only from data up to and including the
close of day t (pandas rolling windows are trailing). The agent acts at the
close of t and the resulting return is realised from t to the next rebalance,
so there is no lookahead. ``tests/test_features.py`` checks this by
truncating the input and asserting the past rows are unchanged.

The scaler is fitted on the training split only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData

STOCK_FEATURES = [
    "ret_1", "ret_5", "ret_20", "vol_20", "vol_60", "mom_60", "mom_120",
    "rsi_14", "dd_252", "beta_60", "volume_z", "mom_rank", "vol_rank", "sector_rel_20",
]
MARKET_FEATURES = ["nifty_ret_20", "nifty_vol_20", "vix_level", "vix_chg_5", "fx_ret_20"]
WARMUP = 252  # rows needed before every feature is defined


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank each day, centred to [-0.5, 0.5]."""
    return df.rank(axis=1, pct=True) - 0.5


def stock_feature_frames(md: MarketData) -> dict[str, pd.DataFrame]:
    c, v = md.close, md.volume
    logp = np.log(c)
    r1 = logp.diff()
    bench_r = np.log(md.bench).diff()

    f: dict[str, pd.DataFrame] = {}
    f["ret_1"] = r1
    f["ret_5"] = logp.diff(5)
    f["ret_20"] = logp.diff(20)
    f["vol_20"] = r1.rolling(20).std() * np.sqrt(252)
    f["vol_60"] = r1.rolling(60).std() * np.sqrt(252)
    f["mom_60"] = logp.diff(60)
    f["mom_120"] = logp.diff(120)
    f["rsi_14"] = (_rsi(c) - 50) / 50
    f["dd_252"] = c / c.rolling(252, min_periods=1).max() - 1
    cov = r1.rolling(60).cov(bench_r)
    f["beta_60"] = cov.div(bench_r.rolling(60).var(), axis=0)
    lv = np.log1p(v)
    f["volume_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std().replace(0, np.nan)
    f["mom_rank"] = _cs_rank(f["mom_120"])
    f["vol_rank"] = _cs_rank(f["vol_60"])
    sec = pd.Series(md.sectors, index=md.tickers)
    sec_mean = f["ret_20"].T.groupby(sec).transform("mean").T
    f["sector_rel_20"] = f["ret_20"] - sec_mean
    return f


def market_feature_frame(md: MarketData) -> pd.DataFrame:
    lb = np.log(md.bench)
    br = lb.diff()
    return pd.DataFrame({
        "nifty_ret_20": lb.diff(20),
        "nifty_vol_20": br.rolling(20).std() * np.sqrt(252),
        "vix_level": md.vix / 100.0,
        "vix_chg_5": np.log(md.vix).diff(5),
        "fx_ret_20": np.log(md.fx).diff(20),
    }, index=md.dates)


@dataclass
class FeatureSet:
    stock: np.ndarray    # (T, N, F) scaled
    market: np.ndarray   # (T, M) scaled
    valid_from: int      # first row index where all raw features are defined

    @property
    def n_stock_features(self) -> int:
        return self.stock.shape[2]

    @property
    def n_market_features(self) -> int:
        return self.market.shape[1]


def raw_features(md: MarketData) -> tuple[np.ndarray, np.ndarray]:
    frames = stock_feature_frames(md)
    stock = np.stack([frames[k].to_numpy(dtype=np.float64) for k in STOCK_FEATURES], axis=-1)
    market = market_feature_frame(md)[MARKET_FEATURES].to_numpy(dtype=np.float64)
    return stock, market


def build_features(md: MarketData, train_mask: np.ndarray, clip: float = 5.0) -> FeatureSet:
    stock, market = raw_features(md)
    valid_from = WARMUP
    fit = train_mask.copy()
    fit[:valid_from] = False
    # Per-feature statistics pooled over stocks and training days only.
    s_mu = np.nanmean(stock[fit], axis=(0, 1))
    s_sd = np.nanstd(stock[fit], axis=(0, 1)) + 1e-8
    m_mu = np.nanmean(market[fit], axis=0)
    m_sd = np.nanstd(market[fit], axis=0) + 1e-8
    stock = np.clip(np.nan_to_num((stock - s_mu) / s_sd), -clip, clip).astype(np.float32)
    market = np.clip(np.nan_to_num((market - m_mu) / m_sd), -clip, clip).astype(np.float32)
    return FeatureSet(stock=stock, market=market, valid_from=valid_from)
