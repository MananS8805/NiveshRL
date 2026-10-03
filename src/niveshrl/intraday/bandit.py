"""Learning from mistakes: a Thompson-sampling bandit that decides TAKE / HALF / SKIP per signal context.

Every signal's *shadow* outcome (net R after costs, as if taken at full size) is added to its context buckets,
whether or not the agent took it, so a wrong SKIP is learned just like a losing TAKE. Buckets go from specific
to general: (setup, side, time of day, with/against NIFTY's trend) → (setup, side) → (setup). The most specific
bucket with ≥ ``min_trades_bucket`` outcomes decides:
- **SKIP** when its average R is clearly negative (t-statistic < ``skip_t``);
- **HALF** when a draw from the posterior of its average R is below ``half_below_r``;
- **TAKE** otherwise, and while there is not yet enough history (exploration, within the fixed guardrails).
On days the pool ends ≥ +10% (the user's target) each trade taken that day earns a bonus in its buckets.
The bandit only chooses among signals and sizes 0 / ½ / 1; it can never raise risk above the guardrails.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def tod(minute: float) -> str:
    return "open" if minute <= 30 else "morning" if minute <= 120 else "midday" if minute <= 270 else "late"


def keys(f: dict) -> list[tuple]:
    trend = "with" if f.get("trend_aligned", 0) > 0 else "against"
    return [(f["setup"], f["side"], tod(f["minute"]), trend), (f["setup"], f["side"]), (f["setup"],)]


class Bandit:
    def __init__(self, cfg: dict, state: dict | None = None, seed: int = 0):
        self.lc = cfg["learning"]
        self.stats: dict[str, list[float]] = (state or {}).get("stats", {})       # key -> [n, sum, sumsq]
        self.log: list[dict] = (state or {}).get("log", [])
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ decisions
    def _summary(self, k: str):
        n, s, s2 = self.stats.get(k, [0, 0.0, 0.0])
        if n < 2:
            return n, float("nan"), float("nan"), float("nan")
        mean = s / n
        var = max(s2 / n - mean * mean, 1e-6) * n / (n - 1)
        se = math.sqrt(var / n)
        return n, mean, se, mean / se

    def decide(self, f: dict, prob: float | None = None, scorer_ready: bool = False) -> tuple[str, str]:
        if scorer_ready and prob is not None and prob == prob and prob < self.lc["ml_min_prob"]:
            return "SKIP", f"ML: P(profit) {prob:.0%} < {self.lc['ml_min_prob']:.0%}"
        for k in keys(f):
            ks = "|".join(k)
            n, mean, se, t = self._summary(ks)
            if n < self.lc["min_trades_bucket"]:
                continue
            if t < self.lc["skip_t"]:
                return "SKIP", f"{ks}: avg {mean:+.2f}R over {n} signals (t {t:.1f})"
            draw = self.rng.normal(mean, se)
            if draw < self.lc["half_below_r"]:
                return "HALF", f"{ks}: avg {mean:+.2f}R over {n}, uncertain"
            return "TAKE", f"{ks}: avg {mean:+.2f}R over {n}"
        return "TAKE", "exploring: not enough history for this context yet"

    def policy(self, k: str) -> str:
        """Deterministic state of a bucket (for the learning log): SKIP / HALF / TAKE / learning."""
        n, mean, se, t = self._summary(k)
        if n < self.lc["min_trades_bucket"]:
            return "learning"
        if t < self.lc["skip_t"]:
            return "SKIP"
        return "HALF" if mean < self.lc["half_below_r"] else "TAKE"

    # ------------------------------------------------------------------ learning
    def update(self, shadow: pd.DataFrame, taken_keys: set | None = None, bonus: float = 0.0, day: str = "") -> list[dict]:
        """Add a day's shadow outcomes; returns the learning-log entries for buckets whose policy changed."""
        if shadow is None or shadow.empty:
            return []
        before = {k: self.policy(k) for k in self.stats}
        for _, r in shadow.iterrows():
            f = r.to_dict()
            reward = float(r["net_r"])
            if bonus and taken_keys and (f.get("ticker"), f.get("ts")) in taken_keys:
                reward += bonus
            for k in keys(f):
                ks = "|".join(k)
                st = self.stats.setdefault(ks, [0, 0.0, 0.0])
                st[0] += 1
                st[1] += reward
                st[2] += reward * reward
        changes = []
        for k in self.stats:
            new, old = self.policy(k), before.get(k, "learning")
            if new != old:
                n, mean, _, t = self._summary(k)
                changes.append({"day": day, "bucket": k, "from": old, "to": new, "n": n, "avg_r": round(mean, 3),
                                "t": round(t, 2), "when": datetime.now().isoformat(timespec="seconds"),
                                "why": f"{n} outcomes, average {mean:+.2f}R (t {t:.1f})"})
        self.log.extend(changes)
        self.log = self.log[-500:]
        return changes

    def table(self) -> pd.DataFrame:
        rows = []
        for k in self.stats:
            n, mean, se, t = self._summary(k)
            rows.append({"bucket": k, "signals": n, "avg R": mean, "t-stat": t, "policy": self.policy(k)})
        return pd.DataFrame(rows).sort_values(["signals"], ascending=False) if rows else pd.DataFrame()

    # ------------------------------------------------------------------ persistence
    def state(self) -> dict:
        return {"stats": self.stats, "log": self.log}

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state()), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, cfg: dict, path: Path, seed: int = 0) -> "Bandit":
        st = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return cls(cfg, st, seed)
