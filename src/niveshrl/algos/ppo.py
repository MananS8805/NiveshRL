"""From-scratch PPO (and A2C) for dict-observation, simplex-action environments.

- GAE(lambda) advantages, bootstrapped on truncation.
- Clipped surrogate objective, clipped value loss, entropy bonus with linear
  decay, gradient-norm clipping, linear learning-rate decay, and advantage
  normalisation.
- ``algo="a2c"`` switches off ratio clipping and uses a single full-batch
  epoch per rollout, so A2C vs PPO is a one-flag ablation.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..models.actor_critic import ActorCritic


@dataclass
class PPOConfig:
    algo: str = "ppo"
    n_envs: int = 8
    n_steps: int = 64              # per env per rollout
    epochs: int = 8
    minibatch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0         # exploration comes from the concentration schedule
    ent_coef_final: float = 0.0
    lr: float = 3e-4
    conc_start: float = 10.0       # Dirichlet concentration schedule (log-linear)
    conc_end: float = 300.0
    equity_conc_start: float = 4.0     # wider exploration of the stocks-vs-cash share
    equity_conc_end: float = 60.0
    equity_lr_mult: float = 5.0        # the equity head learns faster than the shared network
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    total_steps: int = 500_000
    eval_every: int = 20_000       # env steps between validation backtests
    seed: int = 0
    extra: dict = field(default_factory=dict)


def obs_to_torch(obs_list: list[dict], device="cpu") -> dict:
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32, device=device)
            for k in obs_list[0]}


class SimpleVecEnv:
    """Steps a list of envs and resets finished ones right away. A finished
    env's final observation is kept, so truncated episodes can be bootstrapped."""

    def __init__(self, envs):
        self.envs = envs

    def reset(self, seed: int) -> list[dict]:
        return [e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)]

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, final_obs, infos = [], [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, te, tr, info = e.step(a)
            final_obs.append(o)
            if te or tr:
                o, _ = e.reset()
            obs.append(o)
            rew.append(r)
            term.append(te)
            trunc.append(tr)
            infos.append(info)
        return obs, np.array(rew, np.float32), np.array(term), np.array(trunc), final_obs, infos


class PPOTrainer:
    def __init__(self, model: ActorCritic, env_fns: list[Callable], cfg: PPOConfig,
                 evaluate: Callable[[ActorCritic], dict] | None = None, out_dir: str | Path = "runs/ppo"):
        self.m, self.cfg = model, cfg
        self.venv = SimpleVecEnv([f() for f in env_fns])
        self.evaluate = evaluate
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        eq = list(model.equity_head.parameters()) if getattr(model, "split", False) else []
        eq_ids = {id(q) for q in eq}
        groups = [{"params": [q for q in model.parameters() if id(q) not in eq_ids], "mult": 1.0}]
        if eq:
            groups.append({"params": eq, "mult": cfg.equity_lr_mult})
        self.opt = torch.optim.Adam([{"params": g["params"], "lr": cfg.lr * g["mult"]} for g in groups], eps=1e-5)
        self.lr_mult = [g["mult"] for g in groups]
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.out / "tb"))
        except Exception:  # tensorboard optional
            self.tb = None
        self.log_path = self.out / "log.csv"
        self.rows: list[dict] = []
        self.best_score = -np.inf

    def _log(self, step: int, row: dict) -> None:
        # Validation rows add columns, so rewrite the whole file with the union
        # of columns rather than appending under a stale header.
        self.rows.append({"step": step, **row})
        pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
        if self.tb:
            for k, v in row.items():
                self.tb.add_scalar(k, v, step)

    def _rollout(self, obs):
        c, T, E = self.cfg, self.cfg.n_steps, self.cfg.n_envs
        buf = {"obs": [], "act": [], "logp": [], "val": [], "rew": np.zeros((T, E), np.float32),
               "done": np.zeros((T, E), np.float32), "boot": np.zeros((T, E), np.float32)}
        ep_stats = {"turnover": [], "cost": [], "ret": []}
        self.m.eval()
        for t in range(T):
            ob = obs_to_torch(obs)
            a, logp, v = self.m.act(ob)
            nobs, r, term, trunc, final_obs, infos = self.venv.step(a.numpy())
            buf["obs"].append(obs)
            buf["act"].append(a)
            buf["logp"].append(logp)
            buf["val"].append(v)
            buf["rew"][t] = r
            buf["done"][t] = term | trunc
            if trunc.any():  # bootstrap the value of the truncated state
                idx = np.flatnonzero(trunc)
                _, _, vf = self.m.act(obs_to_torch([final_obs[i] for i in idx]))
                buf["boot"][t, idx] = c.gamma * vf.numpy()
            for inf in infos:
                ep_stats["turnover"].append(inf["turnover"])
                ep_stats["cost"].append(inf["cost"] / max(inf["value"], 1.0))
                ep_stats["ret"].append(inf["return"])
            obs = nobs
        _, _, last_v = self.m.act(obs_to_torch(obs))
        return obs, buf, last_v.numpy(), ep_stats

    def _gae(self, buf, last_v):
        c = self.cfg
        vals = torch.stack(buf["val"]).numpy()          # (T, E)
        T = len(vals)
        adv = np.zeros_like(vals)
        gae = np.zeros(vals.shape[1], np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else vals[t + 1]
            nonterm = 1.0 - buf["done"][t]
            # Terminal/truncated: next value is 0 (or the bootstrap for truncation).
            delta = buf["rew"][t] + buf["boot"][t] + c.gamma * nv * nonterm - vals[t]
            gae = delta + c.gamma * c.gae_lambda * nonterm * gae
            adv[t] = gae
        return adv, adv + vals

    def train(self) -> ActorCritic:
        c = self.cfg
        obs = self.venv.reset(c.seed)
        per_rollout = c.n_steps * c.n_envs
        n_updates = max(1, c.total_steps // per_rollout)
        step, next_eval, t0 = 0, c.eval_every, time.time()
        for u in range(n_updates):
            frac = 1.0 - u / n_updates
            for g, mult in zip(self.opt.param_groups, self.lr_mult):
                g["lr"] = c.lr * mult * frac
            ent_coef = c.ent_coef_final + (c.ent_coef - c.ent_coef_final) * frac
            conc = c.conc_start * (c.conc_end / c.conc_start) ** (1.0 - frac)
            self.m.concentration.fill_(conc)
            if hasattr(self.m, "equity_concentration"):
                self.m.equity_concentration.fill_(
                    c.equity_conc_start * (c.equity_conc_end / c.equity_conc_start) ** (1.0 - frac))

            obs, buf, last_v, ep_stats = self._rollout(obs)
            adv, ret = self._gae(buf, last_v)
            step += per_rollout

            flat_obs = [o for row in buf["obs"] for o in row]
            b_obs = obs_to_torch(flat_obs)
            b_act = torch.cat(buf["act"])
            b_logp = torch.cat(buf["logp"])
            b_val = torch.cat(buf["val"])
            b_adv = torch.as_tensor(adv.reshape(-1))
            b_ret = torch.as_tensor(ret.reshape(-1))

            self.m.train()
            N = len(b_act)
            epochs, mb = (1, N) if c.algo == "a2c" else (c.epochs, c.minibatch)
            stats = {"pg_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": []}
            stop = False
            for _ in range(epochs):
                perm = torch.randperm(N)
                for s in range(0, N, mb):
                    i = perm[s:s + mb]
                    ob = {k: v[i] for k, v in b_obs.items()}
                    d, v = self.m.dist(ob)
                    logp = d.log_prob(b_act[i])
                    ratio = torch.exp(logp - b_logp[i])
                    a = b_adv[i]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    if c.algo == "a2c":
                        pg = -(logp * a).mean()
                        v_loss = 0.5 * ((v - b_ret[i]) ** 2).mean()
                    else:
                        pg = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                        v_clip = b_val[i] + (v - b_val[i]).clamp(-c.clip, c.clip)
                        v_loss = 0.5 * torch.max((v - b_ret[i]) ** 2, (v_clip - b_ret[i]) ** 2).mean()
                    ent = d.entropy().mean()
                    loss = pg + c.vf_coef * v_loss - ent_coef * ent
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.m.parameters(), c.max_grad_norm)
                    self.opt.step()
                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    stats["pg_loss"].append(pg.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["entropy"].append(ent.item())
                    stats["kl"].append(kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > c.clip).float().mean().item())
                    if c.algo == "ppo" and c.target_kl and kl > 1.5 * c.target_kl:
                        stop = True
                        break
                if stop:
                    break

            row = {k: float(np.mean(v)) for k, v in stats.items()}
            row.update({"reward": float(buf["rew"].mean()),
                        "turnover": float(np.mean(ep_stats["turnover"])),
                        "cost_frac": float(np.mean(ep_stats["cost"])),
                        "step_return": float(np.mean(ep_stats["ret"])),
                        "lr": c.lr * frac, "concentration": conc, "fps": step / (time.time() - t0)})
            if self.evaluate and step >= next_eval:
                next_eval += c.eval_every
                ev = self.evaluate(self.m)
                row.update({f"val_{k}": v for k, v in ev.items()})
                score = ev.get("score", -np.inf)
                if score > self.best_score:
                    self.best_score = score
                    torch.save(self.m.state_dict(), self.out / "best.pt")
            self._log(step, row)
            print(f"[{c.algo}] step {step:>8d}  rew {row['reward']:+.3f}  ent {row['entropy']:+.2f}  "
                  f"kl {row['kl']:.4f}  turn {row['turnover']:.3f}"
                  + (f"  val_sharpe {row['val_Sharpe']:.3f}  val_utility {row.get('val_Utility', float('nan')):.4f}"
                     if "val_Sharpe" in row else ""),
                  flush=True)
        torch.save(self.m.state_dict(), self.out / "last.pt")
        return self.m


_PROFILES = ("conservative", "moderate", "aggressive")
# Validation keys in the order agents.validate() emits them, for logs written
# before log.csv was fixed (older runs had fewer keys per profile).
_LEGACY_VAL_KEYS = {
    11: [f"{k}_{p}" for p in _PROFILES for k in ("Sharpe", "MaxDD", "Turnover")] + ["Sharpe", "score"],
    18: [f"{k}_{p}" for p in _PROFILES for k in ("Sharpe", "MaxDD", "Turnover", "Vol", "Utility")]
        + ["Sharpe", "Utility", "score"],
}


def read_log(path) -> pd.DataFrame:
    """Read a training log.csv, including legacy files whose validation rows
    carry more fields than the header (they were appended under a stale header)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, out = rows[0], []
    for r in rows[1:]:
        extra = r[len(header):]
        rec = dict(zip(header, r[:len(header)]))
        if extra:
            names = _LEGACY_VAL_KEYS.get(len(extra))
            if names is None:
                raise ValueError(f"{path}: cannot map {len(extra)} extra fields")
            rec.update({f"val_{k}": v for k, v in zip(names, extra)})
        out.append(rec)
    return pd.DataFrame(out).apply(pd.to_numeric, errors="coerce")
