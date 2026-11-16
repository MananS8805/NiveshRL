"""Intraday copilot: a pre-market brief, a live scanner of four structural setups on the most liquid NSE stocks,
trade cards with a calibrated probability, a paper account, an end-of-day review (strengths, weaknesses, learnings)
and a bounded rule tuner. Paper only: nothing in this package places orders.

Modules: setups (signals), model (LightGBM P(win after costs)), engine (live + settle), brief (morning brief and
day review), tuner (evidence-based rule changes), backtest (history replay for the benchmarks).
State lives in data/intraday/copilot/.
"""
from __future__ import annotations

import json

from ...config import load_yaml
from .. import DIR, config as intraday_config

CDIR = DIR / "copilot"


def config() -> dict:
    """Intraday costs/tax/pool from configs/intraday.yaml, copilot guardrails/rules from configs/copilot.yaml."""
    base = intraday_config()
    cp = load_yaml("configs/copilot.yaml")
    cfg = {**base, **{k: v for k, v in cp.items() if k != "guardrails"}}
    cfg["guardrails"] = {**base["guardrails"], **cp["guardrails"]}
    cfg["leverage"] = 1.0                                    # copilot sizes on cash only
    return cfg


def rules(cfg: dict | None = None) -> dict:
    """Current tuned rules (data/intraday/copilot/rules.json), else the config's starting rules."""
    cfg = cfg or config()
    p = CDIR / "rules.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"setups": json.loads(json.dumps(cfg["rules"]["setups"])), "last_entry": cfg["rules"]["last_entry"], "log": []}


def save_json(name: str, obj) -> None:
    CDIR.mkdir(parents=True, exist_ok=True)
    p = CDIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def load_json(name: str, default=None):
    p = CDIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    except (ValueError, OSError):
        return default


__all__ = ["CDIR", "config", "rules", "save_json", "load_json"]
