"""Load YAML configs relative to the project root."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    cfg = load_yaml(path)
    cfg["universe"] = load_yaml(cfg["data"]["universe"])
    cfg["costs"] = load_yaml("configs/costs_india.yaml")
    return cfg
