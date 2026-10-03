"""Tomorrow's monitor list: which stocks deserve attention at the next open.

    score = 0.60 x model + 0.15 x news + 0.15 x activity + 0.10 x setup

Each component is scaled to roughly [-1, 1]:
- model: calibrated next-day P(beat the median) from the ensemble, (P - 0.5) / 0.1, clipped;
- news: confidence-adjusted FinBERT sentiment (``sentiment_adj``);
- activity: volume vs its 20-day average, (ratio - 1) / 2, clipped (unusual interest);
- setup: the share of five bullish technical setups present (20-day breakout,
  new 52-week high, golden cross, Supertrend up, RSI 55-70), mapped to [-1, 1].

This is a list to *watch*, not to buy blindly. The next-day model's measured
out-of-sample accuracy is shown beside it in the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"model": 0.60, "news": 0.15, "activity": 0.15, "setup": 0.10}


def components(tech: pd.DataFrame, prob: pd.Series | None, sent: pd.DataFrame | None) -> pd.DataFrame:
    idx = tech.index
    p = prob.reindex(idx) if prob is not None else pd.Series(np.nan, index=idx)
    s = sent["sentiment_adj"].reindex(idx) if sent is not None and "sentiment_adj" in sent else pd.Series(np.nan, index=idx)
    setups = pd.DataFrame({
        "breakout": tech.get("donchian_breakout") == 1,
        "high52": tech.get("new_52w_high") == 1,
        "golden": tech.get("golden_cross_5d") == 1,
        "trend_up": tech.get("supertrend") == 1,
        "rsi_zone": tech["rsi14"].between(55, 70),
    }, index=idx)
    return pd.DataFrame({
        "model": ((p - 0.5) / 0.1).clip(-1, 1),
        "news": s.clip(-1, 1),
        "activity": ((tech["vol_ratio"] - 1) / 2).clip(-1, 1),
        "setup": setups.mean(axis=1) * 2 - 1,
        "prob": p,
    }, index=idx).join(setups)


def score(comp: pd.DataFrame) -> pd.Series:
    w = pd.Series(WEIGHTS)
    vals = comp[list(WEIGHTS)]
    has = vals.notna()
    return (vals.fillna(0) * w).sum(axis=1) / (has * w).sum(axis=1).replace(0, np.nan)


def reasons(row: pd.Series, tech_row: pd.Series, sent_row: pd.Series | None) -> str:
    r = []
    if pd.notna(row.get("prob")):
        r.append(f"P(up) {row['prob']:.0%}")
    if pd.notna(tech_row.get("vol_ratio")) and tech_row["vol_ratio"] >= 1.5:
        r.append(f"volume {tech_row['vol_ratio']:.1f}× normal")
    if row.get("breakout"):
        r.append("20-day breakout")
    if row.get("high52"):
        r.append("new 52-week high")
    elif pd.notna(tech_row.get("from_52w_high")) and tech_row["from_52w_high"] > -0.03:
        r.append("near 52-week high")
    if row.get("golden"):
        r.append("golden cross")
    if pd.notna(tech_row.get("rsi14")) and tech_row["rsi14"] < 30:
        r.append(f"oversold (RSI {tech_row['rsi14']:.0f})")
    if sent_row is not None and pd.notna(sent_row.get("sentiment_adj")) and abs(sent_row["sentiment_adj"]) >= 0.2:
        r.append(f"{'positive' if sent_row['sentiment_adj'] > 0 else 'negative'} news ({int(sent_row['n_news'])} headlines)")
    return ", ".join(r)


def monitor_list(tech: pd.DataFrame, prob: pd.Series | None, sent: pd.DataFrame | None, n: int = 10) -> pd.DataFrame:
    comp = components(tech, prob, sent)
    comp["score"] = score(comp)
    comp = comp.dropna(subset=["score"]).sort_values("score", ascending=False)
    comp["direction"] = np.where(comp["score"] >= 0, "up", "down")
    comp["reasons"] = [reasons(comp.loc[t], tech.loc[t], sent.loc[t] if sent is not None and t in sent.index else None)
                       for t in comp.index]
    top = comp.head(n).assign(list="watch for strength")
    weak = comp.tail(n).iloc[::-1].assign(list="watch for weakness")
    return pd.concat([top, weak])
