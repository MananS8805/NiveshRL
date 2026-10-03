"""The daily pipeline: everything the TODAY / SCRN / WATCH views need, refreshed after the close.

Steps (each isolated: one failing step is recorded and the rest still run):
  prices -> technicals -> fundamentals (+ analyst score) -> news -> sentiment
  -> nextday (model probabilities for tomorrow) -> monitor (top-N lists)
  -> briefing (+ market habits)

Output: ``data/daily/<last trading date>/`` (parquet + json), plus
``data/daily/latest.json`` (folder, run time, and status/duration of each step).

Called by ``scripts/daily.py``, the Streamlit refresh button and the desktop
app's worker process. ``progress(step, fraction)`` is an optional callback.
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT

DAILY = ROOT / "data" / "daily"
STEPS = ["prices", "technicals", "fundamentals", "news", "sentiment", "nextday", "monitor", "briefing"]


def latest_dir() -> Path | None:
    meta = DAILY / "latest.json"
    if not meta.exists():
        return None
    d = DAILY / json.loads(meta.read_text())["folder"]
    return d if d.exists() else None


def latest_meta() -> dict | None:
    meta = DAILY / "latest.json"
    return json.loads(meta.read_text()) if meta.exists() else None


def load(name: str, folder: Path | None = None):
    """Read one output of the latest run: '<name>.parquet' or '<name>.json'. None if missing."""
    d = folder or latest_dir()
    if d is None:
        return None
    pq, js = d / f"{name}.parquet", d / f"{name}.json"
    if pq.exists():
        return pd.read_parquet(pq)
    if js.exists():
        return json.loads(js.read_text())
    return None


def _buzz(sent: pd.DataFrame, today: Path) -> pd.Series:
    """Headline count vs the stock's average over previous daily runs (z-like ratio)."""
    hist = []
    for d in sorted(p for p in DAILY.glob("20*") if p.is_dir() and p != today)[-20:]:
        f = d / "sentiment.parquet"
        if f.exists():
            hist.append(pd.read_parquet(f)["n_news"])
    if not hist:
        return pd.Series(np.nan, index=sent.index)
    base = pd.concat(hist, axis=1).mean(axis=1).reindex(sent.index)
    return (sent["n_news"] - base) / (base + 1)


def run(steps: list[str] | None = None, progress=None, with_seq: bool = False) -> dict:
    from .data import load_panel
    from .technicals import indicator_frames, snapshot

    steps = steps or STEPS
    status: dict[str, dict] = {}
    ctx: dict = {}

    def step(name: str, fn) -> None:
        if name not in steps:
            return
        if progress:
            progress(name, STEPS.index(name) / len(STEPS))
        t = time.time()
        try:
            note = fn() or ""
            status[name] = {"ok": True, "seconds": round(time.time() - t, 1), "note": note}
        except Exception as e:
            status[name] = {"ok": False, "seconds": round(time.time() - t, 1),
                            "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}

    def prices():
        ctx["panel"] = load_panel(refresh=True)
        return f"{ctx['panel'].close.shape[1]} stocks to {ctx['panel'].close.index[-1].date()}"

    step("prices", prices)
    if "panel" not in ctx:                       # refresh failed or skipped: use the cache
        ctx["panel"] = load_panel()
    p = ctx["panel"]
    day = p.close.index[-1]
    out = DAILY / str(day.date())
    out.mkdir(parents=True, exist_ok=True)

    def technicals():
        ctx["frames"] = indicator_frames(p)
        ctx["tech"] = snapshot(ctx["frames"], p)
        ctx["tech"].to_parquet(out / "technicals.parquet")
        return f"{len(ctx['tech'])} stocks, {ctx['tech'].shape[1]} columns"

    def fundamentals():
        from . import analyst
        from .fundamentals import snapshot_all
        f = snapshot_all(p.tickers, progress=(lambda i, n: progress("fundamentals", (2 + i / n) / len(STEPS)))
                         if progress else None)
        f["analyst_score"] = analyst.score(f, p.close.ffill().iloc[-1])
        f["target_upside"] = analyst.components(f, p.close.ffill().iloc[-1])["target_upside"]
        f.to_parquet(out / "fundamentals.parquet")
        ctx["fund"] = f
        return f"{int(f['marketCap'].notna().sum())} with data, {int(f['analyst_score'].notna().sum())} with analyst cover"

    def news():
        from .news import all_news, market_news
        n = all_news(p.tickers, p.names.to_dict())
        n.to_parquet(out / "news.parquet")
        ctx["news"] = n
        try:
            market_news().to_parquet(out / "market_news.parquet")
        except Exception:
            pass
        return f"{len(n)} headlines for {n['ticker'].nunique() if len(n) else 0} stocks"

    def sentiment():
        from . import sentiment as S
        n = ctx.get("news")
        if n is None:
            n = pd.read_parquet(out / "news.parquet")
        scored = ["p_pos", "p_neu", "p_neg", "label", "score"]
        n = n.drop(columns=[c for c in scored if c in n])     # re-runs re-score instead of duplicating columns
        sc = S.score_news(n)
        sc.to_parquet(out / "news.parquet")
        agg = S.aggregate(sc, pd.Timestamp.now(tz="Asia/Kolkata"))
        agg["buzz"] = _buzz(agg, out)
        agg.to_parquet(out / "sentiment.parquet")
        ctx["sent"] = agg
        mk = out / "market_news.parquet"
        if mk.exists():
            m = pd.read_parquet(mk)
            m = m.drop(columns=[c for c in scored if c in m])
            if len(m):
                pd.concat([m.reset_index(drop=True), S.score_texts(m["title"].tolist())], axis=1).to_parquet(mk)
        return f"{len(sc)} headlines scored, market mean {agg['sentiment'].mean():+.2f}"

    def nextday():
        from . import nextday as nd
        dd = nd.build(p, ctx.get("frames"))
        models = ("lgbm", "seq", "logreg") if with_seq else ("lgbm", "logreg")
        pred = nd.walk_forward(dd, first_test_year=day.year, models=models, verbose=False)
        last = pred.xs(day, level=0)
        last = last.assign(prob=last["ensemble"] if "ensemble" in last else last["lgbm"])
        last.to_parquet(out / "nextday.parquet")
        ctx["prob"] = last["prob"]
        ytd = pred.dropna(subset=["y"])
        meta = {m: nd.evaluate(ytd, m) for m in ("ensemble", "lgbm", "logreg") if m in ytd}
        (out / "nextday_ytd.json").write_text(json.dumps(meta, indent=2))
        return f"P(up) for {len(last)} stocks; YTD AUC {meta.get('ensemble', meta.get('lgbm', {})).get('AUC', float('nan')):.3f}"

    def monitor():
        from .monitor import monitor_list
        tech = ctx.get("tech") if "tech" in ctx else pd.read_parquet(out / "technicals.parquet")
        prob = ctx.get("prob")
        if prob is None and (out / "nextday.parquet").exists():
            prob = pd.read_parquet(out / "nextday.parquet")["prob"]
        sent = ctx.get("sent")
        if sent is None and (out / "sentiment.parquet").exists():
            sent = pd.read_parquet(out / "sentiment.parquet")
        ml = monitor_list(tech, prob, sent, n=10)
        ml.to_parquet(out / "monitor.parquet")
        return f"top {int((ml['list'] == 'watch for strength').sum())} / bottom {int((ml['list'] == 'watch for weakness').sum())}"

    def briefing():
        from .regime import daily_regimes, load_regimes
        from .summary import briefing as brief, habits
        tech = ctx.get("tech") if "tech" in ctx else pd.read_parquet(out / "technicals.parquet")
        fund = ctx.get("fund")
        if fund is None and (out / "fundamentals.parquet").exists():
            fund = pd.read_parquet(out / "fundamentals.parquet")
        sent = ctx.get("sent")
        if sent is None and (out / "sentiment.parquet").exists():
            sent = pd.read_parquet(out / "sentiment.parquet")
        w = load_regimes()
        reg = daily_regimes(w, p.close.index) if w is not None else None
        b = brief(p, tech, fund, sent, regime=reg.iloc[-1] if reg is not None else None)
        (out / "briefing.json").write_text(json.dumps(b, indent=2, default=str))
        intraday = None
        try:
            import yfinance as yf
            intraday = yf.download("^NSEI", period="60d", interval="5m", progress=False)
            if isinstance(intraday.columns, pd.MultiIndex):
                intraday.columns = intraday.columns.get_level_values(0)
            intraday.index = intraday.index.tz_convert("Asia/Kolkata")
        except Exception:
            intraday = None
        h = habits(p, reg, intraday)
        (out / "habits.json").write_text(json.dumps(h, indent=2, default=str))
        return f"{len(b['narrative'])} briefing lines"

    for name, fn in [("technicals", technicals), ("fundamentals", fundamentals), ("news", news),
                     ("sentiment", sentiment), ("nextday", nextday), ("monitor", monitor), ("briefing", briefing)]:
        step(name, fn)
    # A partial run (e.g. --steps briefing) updates the day's status instead of replacing it.
    prev = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    merged = {**prev.get("steps", {}), **status}
    meta = {"folder": out.name, "trading_day": str(day.date()), "ran_at": datetime.now().isoformat(timespec="seconds"),
            "steps": {k: merged[k] for k in STEPS if k in merged}}
    (out / "status.json").write_text(json.dumps(meta, indent=2))
    (DAILY / "latest.json").write_text(json.dumps(meta, indent=2))
    if progress:
        progress("done", 1.0)
    return meta
