"""Train the from-scratch PPO/A2C with the permutation-equivariant Transformer policy.

    python scripts/train_custom.py --steps 500000 --seed 0
    python scripts/train_custom.py --algo a2c --temporal lstm --no-attention   # ablations
"""
import argparse
import json
from pathlib import Path

import torch

from niveshrl.agents import TorchPolicy, build_model, validate
from niveshrl.algos.ppo import PPOConfig, PPOTrainer
from niveshrl.env import PortfolioEnv
from niveshrl.pipeline import load_all

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=500_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c"])
ap.add_argument("--temporal", default="cnn", choices=["cnn", "lstm", "mlp"])
ap.add_argument("--no-attention", action="store_true")
ap.add_argument("--reward", default=None, choices=["utility", "dsr", "sortino", "penalized"])
ap.add_argument("--rebalance", type=int, default=None, help="trading days between rebalances")
ap.add_argument("--tax-aware", action="store_true")
ap.add_argument("--n-envs", type=int, default=8)
ap.add_argument("--eval-every", type=int, default=20_000)
ap.add_argument("--out", default=None)
args = ap.parse_args()

torch.set_num_threads(max(1, torch.get_num_threads()))
cfg, md, fs = load_all()
if args.reward:
    cfg["reward"]["kind"] = args.reward
if args.rebalance:
    cfg["env"]["rebalance_every"] = args.rebalance
cfg["env"]["tax_aware"] = args.tax_aware

tag = args.out or (f"{args.algo}_{args.temporal}{'' if not args.no_attention else '_noattn'}"
                   f"_{cfg['reward']['kind']}_reb{cfg['env']['rebalance_every']}"
                   f"{'_tax' if args.tax_aware else ''}_s{args.seed}")
out = Path("runs") / tag
out.mkdir(parents=True, exist_ok=True)
(out / "args.json").write_text(json.dumps(vars(args) | {"split_policy": True, "cfg_env": cfg["env"],
                                                       "cfg_reward": cfg["reward"]}, indent=2))

model = build_model(md, fs, cfg, temporal=args.temporal, attention=not args.no_attention)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
env_fns = [lambda i=i: PortfolioEnv(md, fs, cfg, split="train", action_mode="weights",
                                    seed=args.seed * 100 + i) for i in range(args.n_envs)]
pcfg = PPOConfig(algo=args.algo, n_envs=args.n_envs, total_steps=args.steps,
                 eval_every=args.eval_every, seed=args.seed)
trainer = PPOTrainer(model, env_fns, pcfg,
                     evaluate=lambda m: validate(lambda: TorchPolicy(m), md, fs, cfg), out_dir=out)
trainer.train()
print(f"best validation score {trainer.best_score:.3f}; checkpoints in {out}")
