"""Investor profile: questionnaire answers -> profile vector ``p``.

``p`` goes into the policy's observation, so a single trained model gives
a different allocation for each investor without retraining.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

HORIZON_MAX = 15.0
PROFILE_DIM = 4


@dataclass
class InvestorProfile:
    risk_aversion: float = 0.5   # 0 = aggressive, 1 = very conservative
    dd_tol: float = 0.20         # largest drawdown the investor says they can tolerate
    horizon_years: float = 10.0  # years until the goal
    min_cash: float = 0.05       # minimum cash buffer
    initial: float = 500_000.0   # Rs
    sip_monthly: float = 0.0     # Rs per month
    max_stocks: int | None = None
    rebalance: str = "weekly"

    def vector(self, horizon_left: float | None = None) -> np.ndarray:
        h = self.horizon_years if horizon_left is None else horizon_left
        return np.array([self.risk_aversion, self.dd_tol, min(h, HORIZON_MAX) / HORIZON_MAX,
                         self.min_cash], dtype=np.float32)

    def glide(self, horizon_left: float) -> float:
        """Penalty multiplier that rises over the last 3 years before the goal."""
        return 1.0 + 2.0 * max(0.0, 1.0 - horizon_left / 3.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def sample(rng: np.random.Generator, ranges: dict) -> "InvestorProfile":
        u = lambda k: float(rng.uniform(*ranges[k]))  # noqa: E731
        return InvestorProfile(risk_aversion=u("risk_aversion"), dd_tol=u("dd_tol"),
                               horizon_years=u("horizon_years"), min_cash=u("min_cash"))


PRESETS = {
    "conservative": InvestorProfile(risk_aversion=0.9, dd_tol=0.10, horizon_years=3, min_cash=0.15),
    "moderate": InvestorProfile(risk_aversion=0.5, dd_tol=0.20, horizon_years=7, min_cash=0.05),
    "aggressive": InvestorProfile(risk_aversion=0.1, dd_tol=0.35, horizon_years=15, min_cash=0.0),
}

# The six onboarding questions. Each option maps to profile fields.
QUESTIONNAIRE = [
    {"key": "crash_reaction",
     "q": "Your Rs 10,000 investment falls to Rs 7,000 in a month. You would…",
     "options": {"Sell everything": 1.0, "Sell some": 0.7, "Hold and wait": 0.4, "Buy more": 0.1}},
    {"key": "max_loss",
     "q": "The biggest temporary fall you could live with is…",
     "options": {"10%": 0.10, "20%": 0.20, "30%": 0.30, "40%+": 0.35}},
    {"key": "horizon",
     "q": "When will you need this money?",
     "options": {"1–3 years": 2.0, "3–5 years": 4.0, "5–10 years": 7.5, "10+ years": 15.0}},
    {"key": "experience",
     "q": "How much investing experience do you have?",
     "options": {"None": 0.15, "Mutual funds / SIP": 0.05, "Direct stocks": 0.0}},
    {"key": "initial", "q": "Starting amount (Rs)", "numeric": True},
    {"key": "sip", "q": "Monthly SIP amount (Rs)", "numeric": True},
]


def from_answers(ans: dict) -> InvestorProfile:
    """Map questionnaire answers (option labels or numbers) to a profile."""
    lookup = {q["key"]: q for q in QUESTIONNAIRE}
    val = lambda k: lookup[k]["options"][ans[k]]  # noqa: E731
    ra = val("crash_reaction")
    horizon = val("horizon")
    # Short horizons are nudged towards caution regardless of stated appetite.
    if horizon <= 2:
        ra = max(ra, 0.7)
    initial = float(ans.get("initial", 100_000))
    return InvestorProfile(
        risk_aversion=ra, dd_tol=val("max_loss"), horizon_years=horizon,
        min_cash=val("experience"), initial=initial, sip_monthly=float(ans.get("sip", 0.0)),
        max_stocks=suggest_max_stocks(initial),
    )


def suggest_max_stocks(capital: float) -> int:
    """Cap the number of holdings for small accounts.

    Whole shares and the flat Rs ~16 DP charge per stock sold make many tiny
    positions uneconomic.
    """
    if capital < 25_000:
        return 5
    if capital < 100_000:
        return 8
    if capital < 500_000:
        return 12
    return 20
