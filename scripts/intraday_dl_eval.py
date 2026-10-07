"""Walk-forward evaluation of the intraday deep-learning meta-labeler (TCN) against a LightGBM baseline on the same
inputs, over every archived day of 5-minute bars. Saves out-of-sample predictions (data/intraday/dl/oos.parquet)
for the bandit / conformal / exit-policy experiments and writes report/results/intraday_dl.md.

    python scripts/intraday_dl_eval.py [--refit-every 5] [--first-test 15]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from niveshrl.config import ROOT
from niveshrl.intraday import DIR, config
from niveshrl.intraday.dl import dataset as DS
from niveshrl.intraday.dl import tcn as T

OUT = ROOT / "report" / "results"
DLDIR = DIR / "dl"


def flat_features(blocks, bi, ii):
    S, C, rl, rs = T.tensors(blocks, bi, ii)
    return np.concatenate([C, S[:, -1], S[:, -2], S[:, -3], S[:, -6:].mean(1), S[:, -12:].mean(1)], 1), rl, rs


def lgbm_walk_forward(blocks, first_test, refit_every, seed=0):
    import lightgbm as lgb
    days = sorted({b.day for b in blocks})
    out = []
    for s in range(first_test, len(days), refit_every):
        tr = [b for b in blocks if b.day < days[s]]
        te_days = set(days[s:s + refit_every])
        te = [b for b in blocks if b.day in te_days]
        bi, ii = T.samples(tr, max_n=400_000, seed=seed)
        X, rl, rs = flat_features(tr, bi, ii)
        tbi, tii = T.samples(te, need_label=False)
        Xt, trl, trs = flat_features(te, tbi, tii)
        res = pd.DataFrame({"ticker": [te[k].ticker for k in tbi], "ts": [te[k].ts[i] for k, i in zip(tbi, tii)],
                            "rL": trl, "rS": trs})
        for side, y in (("L", rl), ("S", rs)):
            ok = np.isfinite(y)
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
            m.fit(X[ok], np.clip(y[ok], *T.R_CLIP))
            res["e" + side] = m.predict(Xt)
        out.append(res)
        print(f"  [lgbm] test {days[s]:%Y-%m-%d}", flush=True)
    return pd.concat(out, ignore_index=True)


def score(pred: pd.DataFrame, in_play: set | None = None, k: int = 5) -> dict:
    """Skill that matters for trading: daily rank IC, and the realised R of the top-k picks per day (best side)."""
    d = pred.copy()
    d["day"] = pd.to_datetime(d["ts"]).dt.normalize()
    if in_play is not None:
        d = d[[(t, dd) in in_play for t, dd in zip(d["ticker"], d["day"])]]
    long_ = d[["day", "ticker", "ts", "eL", "rL"]].rename(columns={"eL": "e", "rL": "r"})
    short = d[["day", "ticker", "ts", "eS", "rS"]].rename(columns={"eS": "e", "rS": "r"})
    both = pd.concat([long_, short]).dropna(subset=["e", "r"])
    ic = both.groupby("day").apply(lambda x: x["e"].rank().corr(x["r"].rank()) if len(x) > 50 else np.nan).dropna()
    top = both.sort_values("e", ascending=False).groupby("day").head(k)
    by_day = top.groupby("day")["r"].mean()
    pos = both[both["e"] > 0]
    return {"days": int(ic.size), "IC mean": float(ic.mean()), "IC t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
            f"top{k}/day avg R": float(top["r"].mean()), f"top{k}/day win": float((top["r"] > 0).mean()),
            f"top{k}/day t": float(by_day.mean() / by_day.std() * np.sqrt(len(by_day))) if len(by_day) > 2 else np.nan,
            "E>0 trades/day": float(len(pos) / max(ic.size, 1)), "E>0 avg R": float(pos["r"].mean()) if len(pos) else np.nan,
            "all bars avg R": float(both["r"].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-every", type=int, default=5)
    ap.add_argument("--first-test", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()
    cfg = config()
    bars = DS.load_bars()
    DS.archive_day(bars)                                     # seed the archive with what Yahoo still serves
    print(f"bars {len(bars):,} rows, {bars['ts'].dt.normalize().nunique()} days", flush=True)
    blocks = DS.build(bars, cfg)
    days = sorted({b.day for b in blocks})
    print(f"{len(blocks):,} stock-days over {len(days)} days in {time.time() - t0:.0f}s", flush=True)
    # which stock-days were 'in play' (the agent's own universe each day)
    from niveshrl.intraday.agent import in_play_matrix
    from niveshrl.intraday.replay import daily_context
    ctx_all = daily_context(bars)
    bl = bars.assign(day=bars["ts"].dt.normalize())
    in_play = set()
    for day in days:
        db = bl[bl["day"] == day]
        per = {t: g.set_index("ts")[["open", "high", "low", "close", "volume"]] for t, g in db.groupby("ticker") if t != "^NSEI"}
        try:
            cx = ctx_all.xs(day, level=1)
        except KeyError:
            continue
        ctx = {t: cx.loc[t].to_dict() for t in per if t in cx.index}
        tick, S = in_play_matrix(per, ctx, cfg["universe"])
        sc = np.nanmax(np.where(np.isfinite(S), S, -np.inf), axis=1) if S.size else np.array([])
        top = np.argsort(-sc)[:cfg["universe"]["in_play_top"]]
        in_play |= {(tick[i], day) for i in top if np.isfinite(sc[i])}
    print("in-play stock-days:", len(in_play), flush=True)

    print("TCN walk-forward", flush=True)
    tcn = T.walk_forward(blocks, a.first_test, a.refit_every)
    print("LightGBM walk-forward", flush=True)
    gbm = lgbm_walk_forward(blocks, a.first_test, a.refit_every)
    rows = {"TCN (deep learning), all stocks": score(tcn), "LightGBM, all stocks": score(gbm),
            "TCN, in-play stocks": score(tcn, in_play), "LightGBM, in-play stocks": score(gbm, in_play)}
    t = pd.DataFrame(rows).T
    t.index.name = "Model"
    DLDIR.mkdir(parents=True, exist_ok=True)
    tcn.to_parquet(DLDIR / "oos_tcn.parquet")
    gbm.to_parquet(DLDIR / "oos_lgbm.parquet")
    pd.DataFrame(sorted(in_play), columns=["ticker", "day"]).to_parquet(DLDIR / "in_play.parquet")
    period = f"{days[a.first_test]:%d %b} → {days[-1]:%d %b %Y}"
    cols = [str(c) for c in t.columns]
    lines = ["| Model | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for i, r in t.iterrows():
        lines.append("| " + i + " | " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    md = (f"# Intraday meta-labeler: TCN vs LightGBM (walk-forward, {period})\n\nEvery liquid stock's every 5-minute bar, "
          f"long and short, triple-barrier labels (ATR stop ≥ the 0.2R-cost distance, 2R target, 60-minute limit), "
          f"R after real costs. Each block of {a.refit_every} days is predicted by a model trained on all earlier days "
          f"only.\n\n" + "\n".join(lines) + "\n\nIC = daily rank correlation between predicted and realised R. "
          f"Top5/day = the 5 bars with the highest predicted R each day, ranked with the whole day in view: a test of ranking skill, not a tradable policy (scripts/compare_intraday_v2.py tests causal policies). "
          f"A useful model has IC t > 2 and a positive top-5 average R after costs.\n\nRun time {time.time() - t0:,.0f}s.\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intraday_dl.md").write_text(md, encoding="utf-8")
    t.to_csv(OUT / "intraday_dl.csv")
    (OUT / "intraday_dl.json").write_text(json.dumps({k: {kk: (None if vv != vv else vv) for kk, vv in v.items()}
                                                      for k, v in rows.items()}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
