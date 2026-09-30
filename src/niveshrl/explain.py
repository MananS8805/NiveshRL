"""Plain-language explanations of rebalancing decisions.

For each stock whose weight changes a lot, compute gradient x input
attribution of that stock's allocation with respect to the latest feature
row: its own features plus the market features. The top drivers are then
mapped to fixed sentence templates.

Everything is deterministic and uses no LLM. The numbers in an explanation
are exactly the model's.
"""
from __future__ import annotations

import numpy as np
import torch

from .algos.ppo import obs_to_torch
from .features import MARKET_FEATURES, STOCK_FEATURES

PHRASES = {
    "ret_1": ("fell sharply yesterday", "jumped yesterday"),
    "ret_5": ("had a weak week", "had a strong week"),
    "ret_20": ("had a weak month", "had a strong month"),
    "vol_20": ("its recent volatility is low", "its recent volatility spiked"),
    "vol_60": ("its volatility is low", "its volatility is high"),
    "mom_60": ("its 3-month trend is negative", "its 3-month trend is positive"),
    "mom_120": ("its 6-month trend is negative", "its 6-month trend is positive"),
    "rsi_14": ("it looks oversold (RSI)", "it looks overbought (RSI)"),
    "dd_252": ("it is far below its 52-week high", "it is near its 52-week high"),
    "beta_60": ("it moves less than the market", "it moves more than the market"),
    "volume_z": ("trading volume is unusually low", "trading volume is unusually high"),
    "mom_rank": ("its momentum ranks low among peers", "its momentum ranks high among peers"),
    "vol_rank": ("it is among the calmer stocks", "it is among the most volatile stocks"),
    "sector_rel_20": ("it lagged its sector this month", "it beat its sector this month"),
    "nifty_ret_20": ("NIFTY fell over the past month", "NIFTY rose over the past month"),
    "nifty_vol_20": ("market volatility is low", "market volatility is elevated"),
    "vix_level": ("India VIX is low", "India VIX is high"),
    "vix_chg_5": ("India VIX eased this week", "India VIX jumped this week"),
    "fx_ret_20": ("the rupee strengthened", "the rupee weakened"),
}


def attributions(model, obs: dict, stock_idx: int) -> dict[str, float]:
    """Gradient x input of the mean weight of ``stock_idx`` w.r.t. the latest feature row."""
    t = obs_to_torch([obs])
    t["stock"].requires_grad_(True)
    t["market"].requires_grad_(True)
    model.mean_weights(t)[0, stock_idx].backward()
    gs = (t["stock"].grad * t["stock"])[0, -1, stock_idx].detach().numpy()
    gm = (t["market"].grad * t["market"])[0, -1].detach().numpy()
    out = {k: float(v) for k, v in zip(STOCK_FEATURES, gs)}
    out.update({k: float(v) for k, v in zip(MARKET_FEATURES, gm)})
    return out


def explain_change(model, obs: dict, tickers: list[str], sectors: list[str], w_old: np.ndarray,
                   w_new: np.ndarray, threshold: float = 0.02, top_k: int = 2) -> list[str]:
    """One sentence per stock whose weight moved by more than ``threshold``."""
    model.eval()
    lines = []
    order = np.argsort(-np.abs(w_new[:-1] - w_old[:-1]))
    for i in order:
        delta = w_new[i] - w_old[i]
        if abs(delta) < threshold:
            break
        attr = attributions(model, obs, int(i))
        # Keep drivers pushing in the same direction as the change.
        sign = np.sign(delta)
        drivers = sorted(((k, v) for k, v in attr.items() if np.sign(v) == sign), key=lambda kv: -abs(kv[1]))
        reasons = []
        for k, _ in drivers[:top_k]:
            is_mkt = k in MARKET_FEATURES
            x = obs["market"][-1][MARKET_FEATURES.index(k)] if is_mkt else obs["stock"][-1, i, STOCK_FEATURES.index(k)]
            reasons.append(PHRASES[k][int(x > 0)])
        verb = "Raised" if delta > 0 else "Cut"
        name = tickers[i].replace(".NS", "")
        why = " and ".join(reasons) if reasons else "the model's combined signals"
        lines.append(f"{verb} {name} ({sectors[i]}) {100 * w_old[i]:.1f}% → {100 * w_new[i]:.1f}%: {why}.")
    dc = w_new[-1] - w_old[-1]
    if abs(dc) >= threshold:
        lines.append(f"{'Moved into' if dc > 0 else 'Deployed'} cash: {100 * w_old[-1]:.1f}% → {100 * w_new[-1]:.1f}%.")
    return lines
