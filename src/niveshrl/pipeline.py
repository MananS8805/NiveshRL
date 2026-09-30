"""One-call loading of config, market data and features."""
from __future__ import annotations

from .config import load_config
from .data import MarketData, load_market_data
from .features import FeatureSet, build_features


def load_all(config: str = "configs/default.yaml", refresh: bool = False) -> tuple[dict, MarketData, FeatureSet]:
    cfg = load_config(config)
    md = load_market_data(cfg, refresh=refresh)
    fs = build_features(md, md.split_mask(*cfg["splits"]["train"]), clip=cfg["features"]["clip"])
    return cfg, md, fs
