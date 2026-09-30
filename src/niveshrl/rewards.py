"""Per-step reward functions.

- ``DifferentialSharpe`` (Moody & Saffell, 1998) gives a dense per-step
  reward whose running sum approximates the change in the episode's Sharpe
  ratio. It is the default: a raw rolling-window Sharpe is noisy and
  non-Markov when used as a per-step reward.
- ``utility`` (the default) is the mean-variance utility r - (gamma/2) r^2,
  with gamma taken from the investor's risk aversion. It is the default
  because a Sharpe-style reward cannot express a risk preference: mixing
  the same stocks with more or less cash leaves the Sharpe roughly
  unchanged. A DSR-trained policy barely reacted to the profile (validation
  vol 21.4% -> 21.1% from the boldest to the most cautious investor).
  Under utility, the optimal stock share is Merton's mu / (gamma sigma^2),
  so each gamma has a genuinely different best allocation.
- Risk-profile terms (the downside penalty and the penalty for drawdown
  beyond the investor's tolerance) are added on top. Their strength comes
  from the investor profile ``p``, so one conditioned policy can serve
  every profile.
"""
from __future__ import annotations

import numpy as np


class DifferentialSharpe:
    def __init__(self, eta: float = 0.04, init_var: float = 4e-4):
        self.eta = eta
        self.init_var = init_var
        self.reset()

    def reset(self) -> None:
        self.A = 0.0            # EMA of returns
        self.B = self.init_var  # EMA of squared returns

    def step(self, r: float) -> float:
        dA, dB = r - self.A, r * r - self.B
        var = self.B - self.A ** 2
        dsr = 0.0 if var <= 1e-12 else (self.B * dA - 0.5 * self.A * dB) / var ** 1.5
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(np.clip(dsr, -5.0, 5.0))


class RollingSortino:
    """Change in rolling Sortino ratio over a window (an ablation reward)."""

    def __init__(self, window: int = 12):
        self.window = window
        self.reset()

    def reset(self) -> None:
        self.rets: list[float] = []
        self.prev = 0.0

    def _value(self) -> float:
        r = np.array(self.rets[-self.window:])
        if len(r) < 3:
            return 0.0
        dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) + 1e-4
        return float(r.mean() / dd)

    def step(self, r: float) -> float:
        self.rets.append(r)
        v = self._value()
        out, self.prev = v - self.prev, v
        return float(np.clip(out, -5.0, 5.0))


def gamma_for(risk_aversion: float, cfg: dict) -> float:
    """Map risk aversion in [0, 1] to a relative risk-aversion coefficient, log-linearly."""
    lo, hi = cfg["gamma_min"], cfg["gamma_max"]
    return float(lo * (hi / lo) ** risk_aversion)


class RewardFn:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        kind = cfg["kind"]
        self.base = (DifferentialSharpe(cfg["dsr_eta"]) if kind == "dsr"
                     else RollingSortino() if kind == "sortino" else None)

    def reset(self) -> None:
        if self.base is not None:
            self.base.reset()

    def __call__(self, r: float, drawdown: float, risk_aversion: float, dd_tol: float,
                 glide: float = 1.0) -> float:
        """r: net step return; drawdown: current drawdown (<= 0); glide >= 1 near goal."""
        c = self.cfg
        sref = c["sigma_ref"]
        if c["kind"] == "utility":
            base = (r - 0.5 * gamma_for(risk_aversion, c) * r * r) / sref
        elif self.base is not None:
            base = self.base.step(r)
        else:
            base = float(np.log1p(r)) / sref
        downside = risk_aversion * c["downside_k"] * (min(r, 0.0) / sref) ** 2
        # dd_mu is per unit of excess drawdown: e.g. 5 * 0.10 excess = 0.5 per step,
        # the same order as a typical DSR step.
        dd_pen = glide * c["dd_mu"] * max(0.0, -drawdown - dd_tol)
        return base - downside - dd_pen
