"""P(a copilot card ends positive after costs): LightGBM on what is known at the signal bar, isotonic-calibrated on
the most recent fifth of the training days. Small (200 trees, 15 leaves), trains in seconds on CPU, no PyTorch.
Fit only on earlier days (walk-forward); with too little history it returns NaN and the rules decide alone."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .setups import SETUPS

NUM = ["minute", "gap", "rvol_bar", "vol_pace", "vwap_dist", "stop_pct", "rs", "nifty_trend", "range_used",
       "pos_in_range", "ema_spread", "adr", "cost_r"]
MIN_ROWS = 200
SKILL_AUC = 0.55


def _skill(p: np.ndarray, y: np.ndarray, day: np.ndarray, base: float) -> dict:
    """Held-out skill of raw scores ``p``: AUC on all held-out cards; Brier skill of isotonic-calibrated probabilities
    fitted on the first half of the held-out days and scored on the second half, against predicting ``base``."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    out = {"n": int(len(y)), "base_rate": round(base, 4)}
    if len(np.unique(y)) < 2:
        return out
    out["auc"] = round(float(roc_auc_score(y, p)), 4)
    ud = np.unique(day)
    first = np.isin(day, ud[: max(1, len(ud) // 2)])
    if first.sum() >= 15 and (~first).sum() >= 15 and len(np.unique(y[first])) == 2:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[first], y[first])
        b_model = brier_score_loss(y[~first], iso.predict(p[~first]))
        b_base = brier_score_loss(y[~first], np.full((~first).sum(), base))
        out["brier_skill"] = round(float(1 - b_model / b_base), 4) if b_base > 0 else None
    return out


def matrix(df: pd.DataFrame) -> np.ndarray:
    X = df.reindex(columns=NUM).astype(float).to_numpy()
    oh = np.stack([(df["setup"] == s).to_numpy(float) for s in SETUPS], 1)
    side = (df["side"] == "long").to_numpy(float)[:, None]
    return np.concatenate([X, oh, side], 1)


class CardModel:
    def __init__(self, model=None, cal=None, n: int = 0, days: list | None = None, skill: dict | None = None):
        self.model, self.cal, self.n, self.days = model, cal, n, days or []
        self.skill = skill or {}

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def skilled(self) -> bool:
        """Allowed to filter trades only when it beat the base rate on held-out days: AUC >= 0.55 and a positive Brier
        skill score (calibration fitted on the first half of the held-out days, scored on the second half)."""
        s = self.skill
        return bool(self.ready and s.get("auc", 0) >= SKILL_AUC and s.get("brier_skill", -1) > 0)

    @classmethod
    def fit(cls, hist: pd.DataFrame, seed: int = 0) -> "CardModel":
        """``hist``: one row per card with features, ``day`` and ``r`` (full-size R after costs)."""
        hist = hist.dropna(subset=["r"])
        if len(hist) < MIN_ROWS:
            return cls(None, None, len(hist))
        y = (hist["r"] > 0).astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            return cls(None, None, len(hist))
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        days = sorted(pd.to_datetime(hist["day"]).unique())
        cut = days[int(len(days) * 0.8)] if len(days) >= 5 else days[-1]
        d = pd.to_datetime(hist["day"])
        tr, ca = (d < cut).to_numpy(), (d >= cut).to_numpy()
        if tr.sum() < MIN_ROWS // 2 or ca.sum() < 30 or len(np.unique(y[tr])) < 2:
            tr, ca = np.ones(len(y), bool), None
        kw = dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=25, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1)
        m = lgb.LGBMClassifier(**kw).fit(matrix(hist[tr]), y[tr])
        cal, skill = None, {}
        if ca is not None:
            pc = m.predict_proba(matrix(hist[ca]))[:, 1]
            yc = y[ca]
            skill = _skill(pc, yc, d[ca].to_numpy(), float(y[tr].mean()))
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pc, yc)
            m = lgb.LGBMClassifier(**kw).fit(matrix(hist), y)          # final trees on everything
        return cls(m, cal, len(hist), [str(pd.Timestamp(x).date()) for x in days], skill)

    def prob(self, df: pd.DataFrame) -> np.ndarray:
        if not self.ready or not len(df):
            return np.full(len(df), np.nan)
        p = self.model.predict_proba(matrix(df))[:, 1]
        return self.cal.predict(p) if self.cal is not None else p

    def gated_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilities only from a model with measured skill; otherwise NaN, so the rules decide alone."""
        return self.prob(df) if self.skilled else np.full(len(df), np.nan)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "cal": self.cal, "n": self.n, "days": self.days, "skill": self.skill}, f)

    @classmethod
    def load(cls, path: Path) -> "CardModel":
        if not path.exists():
            return cls()
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            return cls(d["model"], d["cal"], d["n"], d.get("days"), d.get("skill"))
        except Exception:                                              # noqa: BLE001 - a bad file means "no model"
            return cls()
