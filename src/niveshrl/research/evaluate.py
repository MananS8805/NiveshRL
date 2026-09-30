"""Ranking diagnostics: are the scores any good, before trading costs?

- AUC and accuracy on the binary above-median label.
- Information coefficient (IC): Spearman correlation between score and next
  month's return, computed per month. Reported as mean, t-stat and hit rate.
- Decile portfolios: the equal-weight next-month return of each score
  decile, and the top-minus-bottom spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_ic(pred: pd.DataFrame) -> pd.Series:
    def ic(g):
        g = g.dropna(subset=["ret_next"])
        return g["score"].rank().corr(g["ret_next"].rank()) if len(g) > 10 else np.nan
    return pred.groupby(level=0).apply(ic).dropna()


def decile_returns(pred: pd.DataFrame, q: int = 10) -> pd.DataFrame:
    """dates x decile (1 = lowest score ... q = highest), equal-weight next-month return."""
    def per_month(g):
        g = g.dropna(subset=["ret_next"])
        if len(g) < q * 2:
            return pd.Series(np.nan, index=range(1, q + 1))
        b = pd.qcut(g["score"].rank(method="first"), q, labels=range(1, q + 1))
        return g.groupby(b, observed=True)["ret_next"].mean()
    return pred.groupby(level=0).apply(per_month)


def summary(pred: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    p = pred.dropna(subset=["label"])
    ic = monthly_ic(pred)
    dec = decile_returns(pred)
    spread = (dec.iloc[:, -1] - dec.iloc[:, 0]).dropna()
    return {
        "AUC": float(roc_auc_score(p["label"], p["score"])) if p["label"].nunique() > 1 else np.nan,
        "Accuracy": float(((p["score"] > 0.5) == (p["label"] > 0.5)).mean()),
        "IC mean": float(ic.mean()),
        "IC t-stat": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 2 else np.nan,
        "IC hit rate": float((ic > 0).mean()),
        "Top decile / mo": float(dec.iloc[:, -1].mean()),
        "Bottom decile / mo": float(dec.iloc[:, 0].mean()),
        "Spread / mo": float(spread.mean()),
        "Spread t-stat": float(spread.mean() / spread.std() * np.sqrt(len(spread))) if len(spread) > 2 else np.nan,
        "Months": int(len(ic)),
    }


def by_year(pred: pd.DataFrame) -> pd.DataFrame:
    ic = monthly_ic(pred)
    dec = decile_returns(pred)
    spread = dec.iloc[:, -1] - dec.iloc[:, 0]
    return pd.DataFrame({"IC": ic.groupby(ic.index.year).mean(),
                         "Spread/mo": spread.groupby(spread.index.year).mean()})
