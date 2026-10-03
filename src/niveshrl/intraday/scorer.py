"""Pattern-recognition ML: LightGBM estimating P(a signal ends in profit after costs) from what is known at the
signal bar (setup, side, time, gap, relative volume, VWAP distance, stop width, ATR, range used, EMA spread,
NIFTY trend, first signal of the day). Trained only on earlier days' shadow outcomes (walk-forward); used only
once it has ≥ 300 outcomes, otherwise the bandit decides alone."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

NUM = ["minute", "gap", "rvol_bar", "vol_pace", "vwap_dist", "stop_pct", "atr_pct", "adr", "range_used", "ema_spread",
       "trend_aligned", "first_signal"]
SETUPS = ["orb", "vwap", "ema_pullback", "pdh_pdl", "gap_and_go", "nr7"]
MIN_ROWS = 300


def matrix(df: pd.DataFrame) -> np.ndarray:
    X = df.reindex(columns=NUM).astype(float).to_numpy()
    oh = np.stack([(df["setup"] == s).to_numpy(float) for s in SETUPS], 1)
    side = (df["side"] == "long").to_numpy(float)[:, None]
    return np.concatenate([X, oh, side], 1)


class Scorer:
    def __init__(self, model=None, n: int = 0, auc: float | None = None):
        self.model, self.n, self.auc = model, n, auc

    @property
    def ready(self) -> bool:
        return self.model is not None

    def fit(self, hist: pd.DataFrame, seed: int = 0) -> "Scorer":
        hist = hist.dropna(subset=["win"])
        if len(hist) < MIN_ROWS or hist["win"].nunique() < 2:
            return Scorer(None, len(hist))
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               random_state=seed, verbose=-1)
        m.fit(matrix(hist), hist["win"].to_numpy())
        return Scorer(m, len(hist))

    def prob(self, feats: dict) -> float:
        if not self.ready:
            return float("nan")
        return float(self.model.predict_proba(matrix(pd.DataFrame([feats])))[0, 1])

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "n": self.n, "auc": self.auc}, f)

    @classmethod
    def load(cls, path: Path) -> "Scorer":
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["model"], d["n"], d.get("auc"))
