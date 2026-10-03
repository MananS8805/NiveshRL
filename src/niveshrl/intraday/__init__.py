"""Intraday paper-trading agent: a virtual pool that trades NSE stocks intraday on its own, learns from its
results, and pays realistic costs and tax. Paper only: nothing in this package places orders.

Modules: universe (liquid NSE list), costs (charges + speculative tax), setups (rule-based signals on
intraday bars), scorer (ML probability a signal reaches its target), bandit (learns TAKE / HALF / SKIP
from net results), sim (the account: fills, stops, guardrails), replay (history), live (market hours),
store (state on disk).
"""
from pathlib import Path

from ..config import ROOT, load_yaml

DIR = ROOT / "data" / "intraday"


def config() -> dict:
    cfg = load_yaml("configs/intraday.yaml")
    over = DIR / "settings.json"                     # pool / leverage / tax slab set from the app
    if over.exists():
        import json
        try:
            o = json.loads(over.read_text(encoding="utf-8"))
            for k in ("pool", "leverage"):
                if k in o:
                    cfg[k] = o[k]
            if "slab_rate" in o:
                cfg["tax"]["slab_rate"] = o["slab_rate"]
        except (ValueError, OSError):
            pass
    cfg["leverage"] = max(1.0, min(float(cfg["leverage"]), float(cfg["max_leverage"])))
    return cfg


def save_settings(**kw) -> None:
    import json
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / "settings.json"
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            cur = {}
    cur.update(kw)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    tmp.replace(p)


__all__ = ["DIR", "config", "save_settings", "Path"]
