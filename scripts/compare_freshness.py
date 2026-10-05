"""Measure, out of sample, whether fresher training windows help the volatility forecaster and the regime detector.

- Volatility LSTM: refit yearly (train ≤ Dec Y-2, validate Y-1) vs refit monthly (validate on the 12 months before).
  Scored on next-month realised volatility: RMSE of log vol, QLIKE, correlation.
- Regimes: expanding window (all history since 2005) vs a rolling 8-year window. A regime label is useful if it
  separates what comes next, so it is scored by the next 4 weeks' NIFTY realised volatility and return per label,
  and by how often the label flips.

    python scripts/compare_freshness.py
Writes report/results/freshness_comparison.md and freshness_decision.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from niveshrl.config import ROOT
from niveshrl.research import regime as RG
from niveshrl.research import volatility as V
from niveshrl.research.data import load_panel

OUT = ROOT / "report" / "results"


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for i, r in df.iterrows():
        rows.append("| " + " | ".join([str(i)] + [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r]) + " |")
    return "\n".join(rows)


def regime_score(w: pd.DataFrame, p, start: str) -> dict:
    b = p.bench
    lr = np.log(b).diff()
    fut_vol = lr[::-1].rolling(20).std()[::-1].shift(-1) * np.sqrt(252)       # next 20 trading days
    fut_ret = (b.shift(-20) / b - 1)
    w = w[w.index >= start]
    fv = fut_vol.reindex(w.index, method="ffill")
    fr = fut_ret.reindex(w.index, method="ffill")
    d = pd.DataFrame({"regime": w["regime"], "fv": fv, "fr": fr}).dropna()
    g = d.groupby("regime")
    stress_vs_rest = g["fv"].mean().get("Stress", np.nan) / d.loc[d.regime != "Stress", "fv"].mean()
    out = {"weeks": int(len(d)), "flips per year": float((w["regime"] != w["regime"].shift()).sum() / (len(w) / 52)),
           "future vol, Stress ÷ others": float(stress_vs_rest),
           "future 4w return, Bull − Stress": float(g["fr"].mean().get("Bull", np.nan) - g["fr"].mean().get("Stress", np.nan))}
    for k, v in g["fv"].mean().items():
        out[f"future vol, {k}"] = float(v)
    for k, v in g.size().items():
        out[f"weeks {k}"] = int(v)
    return out


def main() -> None:
    t0 = time.time()
    p = load_panel()
    lo = "2025-01-01"
    print("volatility data", flush=True)
    frame, seq = V.build_vol_data(p)
    dates = frame.index.get_level_values(0)
    print("vol LSTM yearly", flush=True)
    yearly = V.lstm_forecast(frame, seq, 2025, refit="yearly", verbose=False)
    print("vol LSTM monthly", flush=True)
    monthly = V.lstm_forecast(frame, seq, 2025, refit="monthly", verbose=False)
    sel = dates >= lo
    rv = frame["rv_next"][sel]
    both = pd.concat([yearly[sel].rename("y"), monthly[sel].rename("m"), rv.rename("r")], axis=1).dropna()
    vt = pd.DataFrame({"yearly refit": V.vol_metrics(both["y"], both["r"]),
                       "monthly refit": V.vol_metrics(both["m"], both["r"]),
                       "hist (last 21 days)": V.vol_metrics(frame["rv21"][sel].reindex(both.index), both["r"])}).T
    vt.index.name = "Volatility forecaster"
    # paired: squared log error per stock-month, monthly − yearly
    e = (np.log(both["m"]) - np.log(both["r"])) ** 2 - (np.log(both["y"]) - np.log(both["r"])) ** 2
    em = e.groupby(level=0).mean()
    vol_t = float(em.mean() / em.std() * np.sqrt(len(em))) if em.std() > 0 else np.nan

    print("regimes expanding / rolling", flush=True)
    rexp = RG.detect_regimes(p, first_year=2015)
    rroll = RG.detect_regimes(p, first_year=2015, window_years=8)
    rt = pd.DataFrame({"expanding (since 2005)": regime_score(rexp, p, "2015-01-01"),
                       "rolling 8 years": regime_score(rroll, p, "2015-01-01")}).T
    rt.index.name = "Regime detector, 2015-2026"
    agree = float((rexp["regime"] == rroll["regime"].reindex(rexp.index)).mean())

    vol_better = bool(em.mean() < 0)
    sep_e, sep_r = rt.iloc[0]["future vol, Stress ÷ others"], rt.iloc[1]["future vol, Stress ÷ others"]
    reg_better = bool(sep_r > sep_e)
    text = (f"# Fresher training windows: volatility forecaster and regime detector\n\n"
            f"## Volatility LSTM, out of sample from {lo} ({len(both):,} stock-months)\n\n{md(vt)}\n\n"
            f"Paired: mean change in squared log error, monthly − yearly = {em.mean():+.4f} per month "
            f"(t {vol_t:.2f}, {len(em)} months; negative = monthly is better).\n\n"
            f"## Regimes\n\n{md(rt)}\n\nThe two agree on {agree:.0%} of weeks. A better detector shows a larger "
            f"future-volatility ratio for Stress weeks (it warns of turbulence) without flipping much more often.\n\n"
            f"Decision: volatility → {'monthly' if vol_better else 'yearly'} refit; regimes → "
            f"{'rolling 8 years' if reg_better else 'expanding'}. Run time {time.time() - t0:,.0f}s.\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "freshness_comparison.md").write_text(text, encoding="utf-8")
    (OUT / "freshness_decision.json").write_text(json.dumps({
        "vol_monthly_better": vol_better, "vol_t": vol_t, "regime_rolling_better": reg_better,
        "stress_sep_expanding": sep_e, "stress_sep_rolling": sep_r, "regime_agreement": agree}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
