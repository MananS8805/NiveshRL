"""Load YAML configs relative to the project root."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


def _root() -> Path:
    """Project root holding configs/, data/, report/ and runs/.

    * ``NIVESHRL_HOME`` overrides everything.
    * In the packaged desktop app (PyInstaller) it is a writable per-user folder,
      ``%LOCALAPPDATA%/NiveshRL``, seeded from the install folder on first run
      (see ``desktop/app.py``), because Program Files is read-only.
    * Otherwise it is the repository checkout.
    """
    if os.environ.get("NIVESHRL_HOME"):
        return Path(os.environ["NIVESHRL_HOME"])
    if getattr(sys, "frozen", False):
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NiveshRL"
    return Path(__file__).resolve().parents[2]


ROOT = _root()


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
