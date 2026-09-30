"""Which stock share does the reward itself prefer, for each investor profile?

Scores fixed stock/cash mixes (equal-weight stocks + cash, weekly) under the
configured reward over rolling one-year episodes of the training period. A
working risk-conditioned reward should give a graded ladder: bold -> mostly
stocks, cautious -> mostly cash. The magnitudes should also be similar
across profiles, otherwise one kind of investor dominates the policy
gradient.

    python scripts/reward_audit.py
"""
import numpy as np

from niveshrl.pipeline import load_all
from niveshrl.rewards import RewardFn

cfg, md, fs = load_all()
a, b = cfg["splits"]["train"]
ew = md.close.loc[a:b].iloc[::5].pct_change().dropna().mean(axis=1).to_numpy()
cash = (1 + cfg["env"]["cash_rate"]) ** (5 / 252) - 1
mixes = [1.0, 0.75, 0.5, 0.25, 0.0]


def episode_total(rets, ra, tol):
    f = RewardFn(cfg["reward"])
    f.reset()
    nav = peak = 1.0
    s = 0.0
    for r in rets:
        nav *= 1 + r
        peak = max(peak, nav)
        s += f(r, nav / peak - 1, ra, tol)
    return s


r = cfg["reward"]
print(f"reward: {r['kind']}, gamma {r['gamma_min']}->{r['gamma_max']}, downside_k {r['downside_k']}, dd_mu {r['dd_mu']}")
for ra, tol in [(0.0, 0.35), (0.25, 0.28), (0.5, 0.20), (0.75, 0.15), (1.0, 0.10)]:
    vals = [np.mean([episode_total(m * ew[s:s + 52] + (1 - m) * cash, ra, tol)
                     for s in range(0, len(ew) - 52, 13)]) for m in mixes]
    print(f"risk aversion {ra:.2f}: best {mixes[int(np.argmax(vals))]:.0%} stocks | "
          + "  ".join(f"{int(m * 100)}%:{v:+.1f}" for m, v in zip(mixes, vals)))
