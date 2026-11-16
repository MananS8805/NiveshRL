"""Real-money reliability tests for a trading strategy's daily P&L. A positive backtest is not evidence of an edge on
its own: these tests ask whether it would survive selection bias, luck, worse fills and a fair comparison.

- ``bootstrap_mean``: stationary block bootstrap over days (Politis & Romano 1994) -> CI of mean daily P&L and the
  probability it is above zero. Blocks keep day-to-day dependence that an i.i.d. bootstrap would ignore.
- ``deflated_sharpe``: probability the Sharpe ratio is above what the best of N tried variants would show by luck,
  corrected for skew, fat tails and sample length (Bailey & López de Prado 2014).
- ``pbo``: probability of backtest overfitting via combinatorially symmetric cross-validation (Bailey, Borwein,
  López de Prado & Zhu 2017): how often the variant that is best in-sample ranks below median out of sample.
- ``matched_random_p``: the share of random-entry portfolios (same days, same bar, same side, same stop width,
  random liquid stock) that do at least as well: a control matched on everything except the stock choice.
- ``verdict``: the money-readiness checklist applied to all of the above plus cost stress and sample size.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import norm


def bootstrap_mean(x, reps: int = 5000, mean_block: float = 5.0, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return {"n": n, "mean": float(np.mean(x)) if n else float("nan"), "lo90": float("nan"), "hi90": float("nan"),
                "p_pos": float("nan")}
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    means = np.empty(reps)
    for b in range(reps):
        idx = np.empty(n, int)
        i = rng.integers(n)
        for k in range(n):
            idx[k] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        means[b] = x[idx].mean()
    return {"n": n, "mean": float(x.mean()), "lo90": float(np.quantile(means, 0.05)), "hi90": float(np.quantile(means, 0.95)),
            "p_pos": float((means > 0).mean())}


def sharpe(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


def deflated_sharpe(x, trial_sharpes, n_trials: int | None = None) -> dict:
    """``x``: daily returns of the chosen strategy; ``trial_sharpes``: per-period Sharpe of every variant tried."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = len(x)
    sr = sharpe(x)
    ts = np.asarray([s for s in trial_sharpes if s == s], float)
    N = int(n_trials or max(len(ts), 1))
    if T < 10 or sr != sr:
        return {"sr": sr, "sr0": float("nan"), "dsr": float("nan"), "n_trials": N, "T": T}
    v = ts.var(ddof=1) if len(ts) > 1 else 0.0
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e))) if N > 1 else 0.0
    z = (x - x.mean()) / x.std(ddof=1)
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    den = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(den, 1e-9))))
    return {"sr": sr, "sr0": float(sr0), "dsr": dsr, "n_trials": N, "T": T, "skew": skew, "kurt": kurt}


def pbo(M, blocks: int = 10) -> dict:
    """``M``: days x variants matrix of daily P&L. Returns PBO and the median out-of-sample rank of the IS winner."""
    M = np.asarray(M, float)
    T, N = M.shape
    if N < 2 or T < blocks * 2:
        return {"pbo": float("nan"), "combos": 0}
    parts = np.array_split(np.arange(T), blocks)
    logits, ranks = [], []
    for comb in itertools.combinations(range(blocks), blocks // 2):
        is_idx = np.concatenate([parts[i] for i in comb])
        oos_idx = np.concatenate([parts[i] for i in range(blocks) if i not in comb])
        is_perf = np.array([sharpe(M[is_idx, j]) for j in range(N)])
        oos_perf = np.array([sharpe(M[oos_idx, j]) for j in range(N)])
        is_perf = np.nan_to_num(is_perf, nan=-np.inf)
        best = int(np.argmax(is_perf))
        oos = np.nan_to_num(oos_perf, nan=-np.inf)
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1)
        w = (rank + 1) / (N + 1)
        ranks.append(w)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "combos": len(logits), "median_oos_rank": float(np.median(ranks))}


def matched_random_p(policy_mean: float, random_means) -> float:
    r = np.asarray(random_means, float)
    r = r[np.isfinite(r)]
    return float((r >= policy_mean).mean()) if len(r) else float("nan")


CHECKS = [
    ("enough_trades", "At least 50 trades in the test period"),
    ("holdout_positive", "Positive net ₹ on the untouched holdout days"),
    ("bootstrap_lo_positive", "90% bootstrap lower bound of mean daily P&L > 0"),
    ("deflated_sharpe", "Deflated Sharpe ≥ 0.95 (beats the best-of-N-by-luck Sharpe)"),
    ("low_overfitting", "Probability of backtest overfitting < 0.5"),
    ("beats_matched_random", "Beats matched random entries (p < 0.05)"),
    ("survives_2x_costs", "Still positive with slippage doubled"),
]


def verdict(stats: dict) -> dict:
    """``stats`` keys: trades, holdout_net, boot (bootstrap dict), dsr, pbo, random_p, net_2x."""
    res = {
        "enough_trades": stats.get("trades", 0) >= 50,
        "holdout_positive": stats.get("holdout_net", float("nan")) > 0,
        "bootstrap_lo_positive": stats.get("boot", {}).get("lo90", float("nan")) > 0,
        "deflated_sharpe": stats.get("dsr", float("nan")) >= 0.95,
        "low_overfitting": stats.get("pbo", float("nan")) < 0.5,
        "beats_matched_random": stats.get("random_p", float("nan")) < 0.05,
        "survives_2x_costs": stats.get("net_2x", float("nan")) > 0,
    }
    passed = sum(res.values())
    label = "MONEY-READY" if passed == len(res) else "PROMISING (not proven)" if passed >= 5 else "NOT READY"
    return {"checks": res, "passed": passed, "total": len(res), "label": label}
