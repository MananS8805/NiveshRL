"""Portfolio actor + value critic on top of :class:`PortfolioEncoder`.

Two policy heads, chosen with ``split``:

- ``split=True`` (default for new runs): **how much in stocks x which stocks.**
  A Beta distribution picks the equity share ``e``. It reads the context
  token *and* the raw investor profile directly. A Dirichlet over the N
  stocks picks the mix. The action is ``[e * w_stocks, 1 - e]``. Risk
  appetite then becomes a clean one-dimensional decision.

- ``split=False`` (legacy runs): a single Dirichlet over N stocks + cash. In
  that design cash was just 1 of 30 softmax scores starting from uniform. A
  cautious investor needed that single score to climb far above the other
  29, and per-step returns were too noisy for it. Trained policies stayed
  at about 4% cash for every profile (validation vol 21.4% vs 21.1%, bold vs
  cautious).

Both heads score every stock with the same weights, so the model stays
permutation-equivariant. A concentration ``c`` sets how tightly samples
cluster around the mean. It follows a schedule set by the trainer (wide
early, tight late) rather than being learned: when it was learnable, the
policy collapsed its own exploration noise, because noise means turnover and
turnover costs money (see scripts/sanity_toy.py). The deterministic action
is the distribution mean.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.distributions import Beta, Dirichlet

from ..profile import PROFILE_DIM
from .encoder import PortfolioEncoder

EPS = 1e-6


class SplitAllocation:
    """Joint distribution of (equity share ~ Beta) x (stock mix ~ Dirichlet), over N+1 weights."""

    def __init__(self, beta: Beta, mix: Dirichlet):
        self.beta, self.mix = beta, mix

    @staticmethod
    def _split(a: torch.Tensor):
        e = (1.0 - a[..., -1]).clamp(EPS, 1 - EPS)
        w = a[..., :-1].clamp_min(EPS)
        return e, w / w.sum(-1, keepdim=True)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        e, w = self._split(a)
        return self.beta.log_prob(e) + self.mix.log_prob(w)

    def entropy(self) -> torch.Tensor:
        return self.beta.entropy() + self.mix.entropy()

    def _join(self, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.cat([e[..., None] * w, (1 - e)[..., None]], -1)

    @property
    def mean(self) -> torch.Tensor:
        return self._join(self.beta.mean, self.mix.mean)

    def sample(self) -> torch.Tensor:
        e = self.beta.sample().clamp(EPS, 1 - EPS)
        return self._join(e, self.mix.sample())


class ActorCritic(nn.Module):
    def __init__(self, encoder: PortfolioEncoder, concentration: float = 10.0, alpha_floor: float = 0.05,
                 split: bool = True):
        super().__init__()
        d = encoder.d
        self.encoder = encoder
        self.split = split
        self.stock_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        if split:
            self.equity_head = nn.Sequential(nn.Linear(d + PROFILE_DIM, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.equity_head[-1].weight)
            nn.init.zeros_(self.equity_head[-1].bias)   # start at 50% stocks / 50% cash
        else:
            self.cash_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.cash_head[-1].weight)
            nn.init.zeros_(self.cash_head[-1].bias)
        self.critic = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d), nn.GELU(),
                                    nn.Linear(d, 1))
        self.register_buffer("concentration", torch.tensor(float(concentration)))
        # Separate, wider exploration for the one-number equity decision (set by the trainer).
        self.register_buffer("equity_concentration", torch.tensor(float(concentration)))
        self.alpha_floor = alpha_floor
        # Start from a uniform stock mix.
        nn.init.zeros_(self.stock_head[-1].weight)
        nn.init.zeros_(self.stock_head[-1].bias)

    def dist(self, obs: dict):
        ctx, tok = self.encoder(obs)
        value = self.critic(torch.cat([ctx, tok.mean(1)], -1)).squeeze(-1)
        stock_scores = self.stock_head(tok).squeeze(-1)                       # (B, N)
        c = self.concentration
        if not self.split:
            scores = torch.cat([stock_scores, self.cash_head(ctx)], -1)
            return Dirichlet(Fn.softmax(scores, -1) * c + self.alpha_floor), value
        mix = Dirichlet(Fn.softmax(stock_scores, -1) * c + self.alpha_floor)
        m = torch.sigmoid(self.equity_head(torch.cat([ctx, obs["profile"]], -1)).squeeze(-1))
        m = m.clamp(1e-3, 1 - 1e-3)
        ce = self.equity_concentration
        beta = Beta(m * ce + self.alpha_floor, (1 - m) * ce + self.alpha_floor)
        return SplitAllocation(beta, mix), value

    def mean_weights(self, obs: dict) -> torch.Tensor:
        """Deterministic allocation (differentiable; used for explanations)."""
        return self.dist(obs)[0].mean

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False):
        d, v = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        a = a.clamp_min(1e-8)
        a = a / a.sum(-1, keepdim=True)
        return a, d.log_prob(a), v
