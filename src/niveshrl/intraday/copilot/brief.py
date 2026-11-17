"""Morning brief (before 09:15, updated after the opening range) and the end-of-day review. Plain, deterministic
text from the day's numbers: no language model, so nothing is invented. Saved as JSON under
data/intraday/copilot/briefs/ and reviews/ for the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load_json, rules as load_rules, save_json


def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:+.1f}%"


def daily_history(tickers: list[str], days: int = 30) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    raw = yf.download(list(tickers) + ["^NSEI", "^INDIAVIX"], period=f"{days}d", interval="1d", auto_adjust=False,
                      progress=False, threads=True, group_by="ticker")
    out = {}
    for t in list(tickers) + ["^NSEI", "^INDIAVIX"]:
        try:
            d = raw[t].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(d):
            out[t] = d.rename(columns=str.lower)
    return out


def watchlist(daily: dict[str, pd.DataFrame], k: int = 15) -> pd.DataFrame:
    rows = []
    for t, d in daily.items():
        if t.startswith("^") or len(d) < 12:
            continue
        h, l, c, v = d["high"], d["low"], d["close"], d["volume"]
        rng = (h - l)
        last = d.iloc[-1]
        adr = float((rng / c).iloc[-21:-1].mean())
        nr7 = bool(rng.iloc[-1] <= rng.iloc[-7:].min())
        inside = bool(h.iloc[-1] <= h.iloc[-2] and l.iloc[-1] >= l.iloc[-2])
        close_loc = float((last["close"] - last["low"]) / (last["high"] - last["low"])) if last["high"] > last["low"] else 0.5
        rvol = float(v.iloc[-1] / v.iloc[-21:-1].mean()) if v.iloc[-21:-1].mean() > 0 else np.nan
        chg = float(c.iloc[-1] / c.iloc[-2] - 1)
        why, score = [], 0.0
        compression = 0.0
        if nr7 or inside:
            avg_rng = float(rng.iloc[-21:-1].mean())
            compression = max(0.0, 1 - float(rng.iloc[-1]) / avg_rng) if avg_rng > 0 else 0.0
            why.append(("NR7" if nr7 else "inside day") + f" (range {compression:.0%} below its 20-day average)")
            score += 0.5 + compression                       # tighter coils rank higher, not alphabetical ties
        if close_loc >= 0.9 and rvol >= 1.3:
            why.append(f"closed at the high on {rvol:.1f}x volume")
            score += 1.5
        elif close_loc <= 0.1 and rvol >= 1.3:
            why.append(f"closed at the low on {rvol:.1f}x volume")
            score += 1.5
        if abs(chg) >= 0.03:
            why.append(f"moved {chg * 100:+.1f}% yesterday")
            score += 1.0 + min(abs(chg) * 10, 1.0)
        if not why:
            continue
        rows.append({"ticker": t, "why": "; ".join(why), "score": score, "close": float(last["close"]), "chg": chg,
                     "pdh": float(last["high"]), "pdl": float(last["low"]), "adr_pct": adr, "rvol": rvol,
                     "bias": "long" if close_loc >= 0.9 else "short" if close_loc <= 0.1 else "either",
                     "only_compression": len(why) == 1 and compression > 0})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df = df.sort_values("score", ascending=False)
    # keep the list varied: at most a third of it may be names whose only reason is a narrow range
    comp = df[df["only_compression"]].head(max(1, k // 3))
    df = pd.concat([df[~df["only_compression"]], comp]).sort_values("score", ascending=False)
    return df.head(k).drop(columns="only_compression").reset_index(drop=True)


def morning_brief(day: str, tickers: list[str], cfg: dict) -> dict:
    daily = daily_history(tickers)
    n = daily.get("^NSEI")
    vix = daily.get("^INDIAVIX")
    mkt = {}
    if n is not None and len(n) >= 21:
        c = n["close"]
        mkt = {"nifty": float(c.iloc[-1]), "d1": float(c.iloc[-1] / c.iloc[-2] - 1), "d5": float(c.iloc[-1] / c.iloc[-6] - 1),
               "vs_20d": float(c.iloc[-1] / c.iloc[-20:].mean() - 1)}
    if vix is not None and len(vix):
        mkt["vix"] = float(vix["close"].iloc[-1])
    wl = watchlist(daily)
    r = load_rules(cfg)
    st = load_json("state.json", {})
    last = (st.get("days") or [{}])[-1]
    g = cfg["guardrails"]
    risk = cfg["pool"] * g["risk_per_trade"]
    tone = "trend-friendly" if mkt.get("vs_20d", 0) > 0.01 else "weak / below its 20-day average" if mkt.get("vs_20d", 0) < -0.01 else "range-bound"
    lines = [f"NIFTY {mkt.get('nifty', float('nan')):,.0f} ({_pct(mkt.get('d1'))} yesterday, {_pct(mkt.get('d5'))} over 5 days): "
             f"{tone}." + (f" India VIX {mkt['vix']:.1f}." if "vix" in mkt else "")]
    if last:
        lines.append(f"Yesterday ({last.get('day')}): {last.get('trades', 0)} trade(s), net ₹{last.get('net', 0):+,.0f}.")
    on = [f"{s} (P ≥ {v['min_p']:.0%})" for s, v in r["setups"].items() if v.get("enabled", True)]
    off = [s for s, v in r["setups"].items() if not v.get("enabled", True)]
    from .engine import MODEL
    from .model import CardModel
    m = CardModel.load(MODEL)
    sk = m.skill
    lines.append("P(win) model: " + (f"has held-out skill (AUC {sk.get('auc', 0):.2f}); it filters trades." if m.skilled else
                 f"no measured skill on held-out days (AUC {sk.get('auc', float('nan')):.2f}, needs 0.55 and a positive "
                 "Brier skill), so the rules decide alone." if m.ready else "not trained yet; the rules decide alone."))
    lines.append("Active setups: " + ", ".join(on) + (f". Benched by the learner: {', '.join(off)}." if off else "."))
    lines.append(f"Plan: at most {g['max_trades_per_day']} trades, ₹{risk:,.0f} risk each, stop for the day at "
                 f"−₹{cfg['pool'] * g['daily_loss_stop']:,.0f}; no new entries after {r.get('last_entry')}; square-off {g['square_off']}.")
    brief = {"day": day, "phase": "pre-market", "market": mkt, "lines": lines,
             "watchlist": wl.to_dict("records") if len(wl) else [], "rules": {k: v for k, v in r.items() if k != "log"}}
    save_json(f"briefs/{day}.json", brief)
    return brief


def opening_update(day: str, bars: dict[str, pd.DataFrame], ctx: dict[str, dict]) -> dict | None:
    """After the 15-minute opening range: gaps, opening-range width and relative volume for the watchlist."""
    b = load_json(f"briefs/{day}.json")
    if not b or b.get("opening"):
        return b
    rows = []
    for w in b.get("watchlist", []):
        t = w["ticker"]
        d = bars.get(t)
        if d is None or len(d) < 3:
            continue
        pc = ctx.get(t, {}).get("prev_close", w.get("close"))
        or_hi, or_lo = float(d["high"].iloc[:3].max()), float(d["low"].iloc[:3].min())
        rows.append({"ticker": t, "gap": float(d["open"].iloc[0] / pc - 1) if pc else None,
                     "or_hi": or_hi, "or_lo": or_lo, "or_width": (or_hi - or_lo) / or_lo,
                     "last": float(d["close"].iloc[-1])})
    b["opening"] = rows
    b["phase"] = "opening range set"
    save_json(f"briefs/{day}.json", b)
    return b


def _grp(cards: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return cards.groupby(by)["r"].agg(["count", "mean", "sum"]).reset_index()


def day_review(day: str, rec: dict, cards: pd.DataFrame, trades, changes: list[dict], nifty, state: dict,
               refit: str = "") -> dict:
    """Strengths, weaknesses and learnings of the day, from the trades and every card's outcome."""
    tr = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    nifty_move = (float(nifty["close"].iloc[-1] / nifty["open"].iloc[0] - 1) if nifty is not None and len(nifty) else None)
    summary = [f"{len(tr)} trade(s), net ₹{rec['net']:+,.0f} after ₹{rec['costs']:,.0f} costs"
               + (f", average {tr['r'].mean():+.2f}R" if len(tr) else "") + f"; pool ₹{rec['pool_end']:,.0f}.",
               f"{rec['cards']} setups appeared" + (f"; taking every one of them would have averaged {cards['r'].mean():+.2f}R."
                                                     if len(cards) and cards['r'].notna().any() else ".")]
    if nifty_move is not None:
        summary.append(f"NIFTY moved {nifty_move * 100:+.2f}% open to close.")
    strengths, weaknesses = [], []
    if len(cards) and cards["r"].notna().any():
        g = _grp(cards.dropna(subset=["r"]), ["setup", "side"])
        for r in g[(g["count"] >= 3) & (g["mean"] > 0.2)].sort_values("mean", ascending=False).head(3).itertuples():
            strengths.append(f"{r.setup} {r.side}s worked today: {r.count} setups averaged {r.mean:+.2f}R.")
        for r in g[(g["count"] >= 3) & (g["mean"] < -0.3)].sort_values("mean").head(3).itertuples():
            weaknesses.append(f"{r.setup} {r.side}s failed today: {r.count} setups averaged {r.mean:+.2f}R.")
        sk = cards[~cards["take"].astype(bool)] if "take" in cards else cards.iloc[0:0]
        if len(sk) >= 5:
            m = sk["r"].mean()
            (strengths if m < 0 else weaknesses).append(
                f"The {len(sk)} skipped setups would have averaged {m:+.2f}R: "
                + ("skipping them was right." if m < 0 else "the filters cost money today."))
        missed = sk[(sk["reason"] == "target")] if "reason" in sk else sk.iloc[0:0]
        if len(missed):
            top = missed.sort_values("r", ascending=False).head(2)
            weaknesses.append("Missed winners: " + "; ".join(f"{r.ticker.replace('.NS', '')} {r.setup} {r.side} "
                                                               f"(+{r.r:.1f}R, skipped: {str(r.why) or 'n/a'})"
                                                               for r in top.itertuples()) + ".")
    if len(tr):
        tr["k_entry"] = pd.to_datetime(tr["entry_ts"])
        tr["k_exit"] = pd.to_datetime(tr["exit_ts"])
        fast = tr[(tr["reason"] == "stop") & ((tr["k_exit"] - tr["k_entry"]) <= pd.Timedelta(minutes=10))]
        if len(fast):
            weaknesses.append(f"{len(fast)} trade(s) stopped within 10 minutes: entries came late in the move.")
        wins = tr[tr["net"] > 0]
        if len(wins):
            strengths.append(f"{len(wins)} of {len(tr)} trades made money (best: {wins.sort_values('net').iloc[-1]['ticker'].replace('.NS', '')} "
                             f"₹{wins['net'].max():+,.0f}).")
        cost_share = tr["costs"].sum() / max(tr["gross"].abs().sum(), 1)
        if cost_share > 0.3:
            weaknesses.append(f"Costs were {cost_share:.0%} of the gross move: too little room per trade.")
    learnings = [f"{c['rule']}: {c['from']} → {c['to']}. {c['why']}." for c in changes]
    if not changes:
        learnings.append("No rule changed today: the evidence window does not yet meet the bar for any change "
                         "(changes need 20-30 outcomes over at least 8 days and a clear per-day t-statistic).")
    if refit:
        learnings.append(refit.capitalize() + ".")
    fl = rec.get("forward_log")
    if fl:
        learnings.append(f"Forward log: {fl['entries']} hash-chained entries, chain {'intact' if fl['ok'] else 'BROKEN'} "
                         f"(head {fl['head'][:12]}). Every decision was recorded before its outcome.")
    days = state.get("days", [])
    fwd = pd.DataFrame(days)
    if len(fwd):
        learnings.append(f"Forward record: {int(fwd['trades'].sum())} trades over {len(fwd)} days, net ₹{fwd['net'].sum():+,.0f}.")
    review = {"day": day, "summary": summary, "strengths": strengths or ["Nothing stood out as working today."],
              "weaknesses": weaknesses or ["No clear weakness today."], "learnings": learnings,
              "changes": changes, "record": rec}
    return review
