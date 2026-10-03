"""Financial-news sentiment with FinBERT (ProsusAI/finbert).

FinBERT is BERT further trained on financial text (Reuters TRC2 + the
Financial PhraseBank). It labels a sentence positive / neutral / negative. It
reads finance correctly where word lists fail: "cuts losses", "beats
estimates" and "margin pressure" all come out the right way round.

Per headline, score = P(positive) - P(negative), in [-1, 1].
Per stock:
- ``sentiment``: the recency-weighted mean score (half-life 12 h);
- ``n_news``: the headline count;
- ``bull`` / ``bear``: counts of clearly positive / negative headlines (|score| > 0.5);
- ``sentiment_adj``: sentiment x n / (n + 3), so one or two headlines can't dominate a
  ranking. Used by the monitor list and the screener;
- ``buzz``: today's count vs the stock's own history, filled in by the daily pipeline.

The model (~440 MB) downloads once on first use into the Hugging Face cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL = "ProsusAI/finbert"
_PIPE = None


def _pipeline():
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
        _PIPE = (tok, mdl, torch)
    return _PIPE


def score_texts(texts: list[str], batch: int = 32) -> pd.DataFrame:
    """P(pos), P(neu), P(neg), label and score for each text."""
    if not texts:
        return pd.DataFrame(columns=["p_pos", "p_neu", "p_neg", "label", "score"])
    tok, mdl, torch = _pipeline()
    labels = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
            probs.append(torch.softmax(mdl(**enc).logits, -1).numpy())
    P = pd.DataFrame(np.vstack(probs), columns=labels)
    out = pd.DataFrame({"p_pos": P["positive"], "p_neu": P["neutral"], "p_neg": P["negative"]})
    out["label"] = out[["p_pos", "p_neu", "p_neg"]].idxmax(axis=1).str[2:].map({"pos": "positive", "neu": "neutral",
                                                                                 "neg": "negative"})
    out["score"] = out["p_pos"] - out["p_neg"]
    return out


def score_news(news: pd.DataFrame) -> pd.DataFrame:
    if news is None or not len(news):
        return news
    s = score_texts(news["title"].tolist())
    return pd.concat([news.reset_index(drop=True), s], axis=1)


def aggregate(scored: pd.DataFrame, now: pd.Timestamp, half_life_h: float = 12.0) -> pd.DataFrame:
    """Per-stock sentiment summary from scored headlines."""
    if scored is None or not len(scored):
        return pd.DataFrame(columns=["sentiment", "n_news", "bull", "bear", "latest"])
    d = scored.copy()
    age_h = ((now - pd.to_datetime(d["published"])).dt.total_seconds() / 3600).clip(lower=0).fillna(48)
    d["w"] = 0.5 ** (age_h / half_life_h)
    g = d.groupby("ticker")
    out = pd.DataFrame({
        "sentiment": g.apply(lambda x: float(np.average(x["score"], weights=x["w"])) if x["w"].sum() > 0 else np.nan,
                             include_groups=False),
        "n_news": g.size(),
        "bull": g["score"].apply(lambda s: int((s > 0.5).sum())),
        "bear": g["score"].apply(lambda s: int((s < -0.5).sum())),
        "latest": g["published"].max(),
    })
    out["sentiment_adj"] = out["sentiment"] * out["n_news"] / (out["n_news"] + 3)
    return out
