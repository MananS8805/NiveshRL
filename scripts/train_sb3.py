"""Stable-Baselines3 baselines: PPO / SAC with an MLP policy, or PPO with our encoder.

    python scripts/train_sb3.py --algo ppo --steps 500000
    python scripts/train_sb3.py --algo sac --steps 200000
    python scripts/train_sb3.py --algo ppo --custom-encoder
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from niveshrl.agents import SB3Policy, validate
from niveshrl.env import PortfolioEnv
from niveshrl.models.encoder import PortfolioEncoder
from niveshrl.pipeline import load_all
from niveshrl.profile import PROFILE_DIM

ap = argparse.ArgumentParser()
ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
ap.add_argument("--steps", type=int, default=500_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-envs", type=int, default=8)
ap.add_argument("--custom-encoder", action="store_true")
ap.add_argument("--eval-every", type=int, default=20_000)
ap.add_argument("--tag", default=None, help="run dir name under runs/ (default derived from args)")
args = ap.parse_args()

cfg, md, fs = load_all()
tag = args.tag or f"sb3_{args.algo}{'_enc' if args.custom_encoder else '_mlp'}_s{args.seed}"
out = Path("runs") / tag
out.mkdir(parents=True, exist_ok=True)


class EncoderExtractor(BaseFeaturesExtractor):
    """Wraps PortfolioEncoder so SB3's own heads sit on top of it."""

    def __init__(self, space, d: int = 64):
        super().__init__(space, features_dim=2 * d)
        self.enc = PortfolioEncoder(fs.n_stock_features, fs.n_market_features, PROFILE_DIM,
                                    md.sector_ids, lookback=cfg["features"]["lookback"], d=d)

    def forward(self, obs):
        ctx, tok = self.enc(obs)
        return torch.cat([ctx, tok.mean(1)], -1)


class ValCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best, self.next = -np.inf, args.eval_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next:
            self.next += args.eval_every
            ev = validate(lambda: SB3Policy(self.model), md, fs, cfg, action_mode="logits")
            for k, v in ev.items():
                self.logger.record(f"val/{k}", v)
            print(f"[{tag}] step {self.num_timesteps}  val_sharpe {ev['Sharpe']:.3f}", flush=True)
            log = out / "val_scores.csv"
            pd.DataFrame([{"step": self.num_timesteps, **ev}]).to_csv(log, mode="a", header=not log.exists(), index=False)
            if ev["score"] > self.best:
                self.best = ev["score"]
                self.model.save(out / "best")
        return True


n_envs = 1 if args.algo == "sac" else args.n_envs
venv = DummyVecEnv([lambda i=i: PortfolioEnv(md, fs, cfg, split="train", action_mode="logits",
                                             seed=args.seed * 100 + i) for i in range(n_envs)])
policy_kwargs = {}
if args.custom_encoder:
    policy_kwargs = {"features_extractor_class": EncoderExtractor, "net_arch": dict(pi=[64], vf=[64])}
else:
    policy_kwargs = {"net_arch": [256, 256], "activation_fn": nn.Tanh} if args.algo == "ppo" else {"net_arch": [256, 256]}

if args.algo == "ppo":
    model = PPO("MultiInputPolicy", venv, n_steps=128, batch_size=256, n_epochs=8, learning_rate=3e-4,
                ent_coef=0.005, gae_lambda=0.95, gamma=0.99, seed=args.seed, policy_kwargs=policy_kwargs,
                tensorboard_log=str(out / "tb"), verbose=0)
else:
    model = SAC("MultiInputPolicy", venv, learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                learning_starts=5_000, seed=args.seed, policy_kwargs=policy_kwargs,
                tensorboard_log=str(out / "tb"), verbose=0)
cb = ValCallback()
model.learn(total_timesteps=args.steps, callback=cb)
model.save(out / "last")
print(f"best validation score {cb.best:.3f}; saved to {out}")
