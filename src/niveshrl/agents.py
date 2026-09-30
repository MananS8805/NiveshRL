"""Model construction, policy wrappers and the validation routine shared by the training scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .algos.ppo import obs_to_torch
from .backtest import make_eval_env, run_policy
from .features import FeatureSet
from .metrics import daily_returns
from .rewards import gamma_for
from .data import MarketData
from .models.actor_critic import ActorCritic
from .models.encoder import PortfolioEncoder
from .profile import PRESETS, PROFILE_DIM


def build_model(md: MarketData, fs: FeatureSet, cfg: dict, temporal: str = "cnn",
                attention: bool = True, d: int = 64, split: bool = True) -> ActorCritic:
    enc = PortfolioEncoder(fs.n_stock_features, fs.n_market_features, PROFILE_DIM, md.sector_ids,
                           lookback=cfg["features"]["lookback"], d=d, temporal=temporal,
                           attention=attention)
    return ActorCritic(enc, split=split)


class TorchPolicy:
    """Deterministic policy for backtests: the Dirichlet mean allocation."""

    def __init__(self, model: ActorCritic):
        self.model = model.eval()

    def __call__(self, env) -> np.ndarray:
        a, _, _ = self.model.act(obs_to_torch([env.observation()]), deterministic=True)
        return a[0].numpy()


class SB3Policy:
    def __init__(self, model):
        self.model = model

    def __call__(self, env) -> np.ndarray:
        a, _ = self.model.predict(env.observation(), deterministic=True)
        return a


def validate(policy_factory, md, fs, cfg, split: str = "val", action_mode: str = "weights",
             profiles=("conservative", "moderate", "aggressive")) -> dict:
    """Backtest each preset profile and score each by its OWN mean-variance utility.

    The checkpoint score is the mean over profiles of annualised
    mean - (gamma_p / 2) * variance. Averaging Sharpe instead would never
    reward a policy for treating a cautious investor differently from a
    bold one.
    """
    out, utils = {}, []
    for name in profiles:
        env = make_eval_env(md, fs, cfg, split, profile=PRESETS[name], action_mode=action_mode)
        res = run_policy(policy_factory(), env, name)
        out[f"Sharpe_{name}"] = res.metrics["Sharpe"]
        out[f"MaxDD_{name}"] = res.metrics["MaxDD"]
        out[f"Turnover_{name}"] = res.metrics.get("Turnover/yr", 0.0)
        out[f"Vol_{name}"] = res.metrics["Vol"]
        r = daily_returns(res.nav)
        g = gamma_for(PRESETS[name].risk_aversion, cfg["reward"])
        out[f"Utility_{name}"] = float(r.mean() * 252 - 0.5 * g * r.var() * 252)
        utils.append(out[f"Utility_{name}"])
    out["Sharpe"] = float(np.mean([out[f"Sharpe_{p}"] for p in profiles]))
    out["Utility"] = float(np.mean(utils))
    out["score"] = out["Utility"]
    return out


def load_model(path: str, md, fs, cfg, **kw) -> ActorCritic:
    m = build_model(md, fs, cfg, **kw)
    res = m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    # Checkpoints saved before the separate equity exploration schedule lack that
    # buffer; it only shapes sampling noise, never the deterministic allocation.
    allowed = {"equity_concentration"}
    if set(res.missing_keys) - allowed or res.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing {res.missing_keys}, unexpected {res.unexpected_keys}")
    return m.eval()


def run_kwargs(run_dir) -> dict:
    """Model-construction kwargs recorded in a run's args.json.

    Runs from before the equity/stock split have no ``split_policy`` key and
    load as the legacy single-Dirichlet policy.
    """
    a = json.loads((Path(run_dir) / "args.json").read_text())
    return {"temporal": a["temporal"], "attention": not a["no_attention"], "split": a.get("split_policy", False)}


def load_run(run_dir, md, fs, cfg, checkpoint: str = "best") -> ActorCritic:
    return load_model(Path(run_dir) / f"{checkpoint}.pt", md, fs, cfg, **run_kwargs(run_dir))
