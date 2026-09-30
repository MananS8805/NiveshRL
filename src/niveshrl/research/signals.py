"""One place to get any strategy signal as a (month-end x ticker) score table."""
from __future__ import annotations

import pandas as pd

from .backtest import factor_scores
from .data import Panel
from .rankers import ALL_MODELS, load_predictions

FACTORS = ["momentum", "reversal", "lowvol", "equal"]
MODEL_LABELS = {
    "ffnn": "FFNN ranker (Takeuchi & Lee)",
    "lstm": "LSTM ranker",
    "transformer": "Transformer ranker",
    "logreg": "Logistic regression",
    "momentum_model": "Momentum 12-1 (ranker baseline)",
}
FACTOR_LABELS = {
    "momentum": "Momentum 12-1 (factor)",
    "reversal": "Short-term reversal",
    "lowvol": "Low volatility",
    "equal": "Equal weight (all eligible)",
}


def available_signals() -> dict[str, str]:
    """signal key -> display label, for trained models present on disk plus factors."""
    out = {m: MODEL_LABELS[m] for m in ALL_MODELS if m != "momentum" and load_predictions(m) is not None}
    return out | FACTOR_LABELS


def get_scores(p: Panel, key: str) -> pd.DataFrame:
    if key in FACTORS:
        return factor_scores(p, key)
    pred = load_predictions(key)
    if pred is None:
        raise FileNotFoundError(f"no predictions for {key}; run scripts/train_rankers.py")
    return pred["score"].unstack("ticker")
