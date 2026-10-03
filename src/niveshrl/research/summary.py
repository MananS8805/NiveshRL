"""Daily market briefing and "market habits": numbers first, then template sentences.

Nothing here is an opinion or an LLM output. Every sentence is filled from
a computed number, so the briefing can't say anything the data doesn't.

``briefing``: what happened today (index, sectors, breadth, extremes, volume,
VIX, regime, sentiment, upcoming results).
``habits``: recurring statistical behaviour of the market, from daily NIFTY
data since 2010 and, when available, 60 days of 5-minute NIFTY bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel


def briefing(p: Panel, tech: pd.DataFrame, fund: pd.DataFrame | None = None, sent: pd.DataFrame | None = None,
             regime: str | None = None) -> dict:
    c = p.close.ffill()
    b = p.bench.dropna()
    v = p.vix.dropna()
    day = c.index[-1]
    r1 = tech["ret_1d"]
    sec = tech.groupby("sector")["ret_1d"].mean().sort_values()
    sec_w = tech.groupby("sector")["ret_1w"].mean()
    rot = (sec.rank(ascending=False) - sec_w.rank(ascending=False)).sort_values()   # negative = rising in rank
    out = {
        "date": str(day.date()),
        "nifty": float(b.iloc[-1]), "nifty_1d": float(b.iloc[-1] / b.iloc[-2] - 1),
        "nifty_1w": float(b.iloc[-1] / b.iloc[-6] - 1), "nifty_ytd": float(b.iloc[-1] / b[b.index.year < day.year].iloc[-1] - 1),
        "vix": float(v.iloc[-1]), "vix_1d": float(v.iloc[-1] / v.iloc[-2] - 1),
        "advancers": int((r1 > 0).sum()), "decliners": int((r1 < 0).sum()),
        "breadth_200": float((tech["vs_sma200"] > 0).mean()),
        "new_highs": tech.index[tech["new_52w_high"] == 1].tolist(),
        "new_lows": tech.index[tech["new_52w_low"] == 1].tolist(),
        "best_sectors": sec.tail(3)[::-1].round(4).to_dict(), "worst_sectors": sec.head(3).round(4).to_dict(),
        "rotation_up": rot.head(3).index.tolist(), "rotation_down": rot.tail(3).index.tolist(),
        "gainers": r1.nlargest(5).round(4).to_dict(), "losers": r1.nsmallest(5).round(4).to_dict(),
        "volume_spikes": tech["vol_ratio"].nlargest(5).round(2).to_dict(),
        "gap_ups": tech["gap_pct"].nlargest(3).round(4).to_dict() if "gap_pct" in tech else {},
        "gap_downs": tech["gap_pct"].nsmallest(3).round(4).to_dict() if "gap_pct" in tech else {},
        "regime": regime,
    }
    if sent is not None and len(sent):
        out["market_sentiment"] = float(sent["sentiment"].mean())
        out["most_mentioned"] = sent["n_news"].nlargest(5).to_dict()
    if fund is not None and "next_earnings" in fund:
        ne = pd.to_datetime(fund["next_earnings"], errors="coerce")
        soon = ne[(ne >= day) & (ne <= day + pd.Timedelta(days=7))].sort_values()
        out["earnings_week"] = {t: str(d.date()) for t, d in soon.items()}
    out["narrative"] = narrative(out)
    return out


def _pct(x: float) -> str:
    return f"{x:+.2%}"


def narrative(s: dict) -> list[str]:
    """Deterministic sentences from the briefing numbers."""
    lines = []
    tone = "rose" if s["nifty_1d"] > 0.002 else "fell" if s["nifty_1d"] < -0.002 else "was flat"
    lines.append(f"NIFTY {tone} {_pct(s['nifty_1d'])} to {s['nifty']:,.0f} ({_pct(s['nifty_1w'])} this week, "
                 f"{_pct(s['nifty_ytd'])} this year).")
    br = s["advancers"] / max(s["advancers"] + s["decliners"], 1)
    width = "broad-based buying" if br > 0.65 else "broad-based selling" if br < 0.35 else "mixed"
    lines.append(f"Breadth showed {width}: {s['advancers']} stocks up, {s['decliners']} down; "
                 f"{s['breadth_200']:.0%} of the NIFTY 200 trade above their 200-day average"
                 f"{' (a weak tape)' if s['breadth_200'] < 0.4 else ' (a healthy tape)' if s['breadth_200'] > 0.6 else ''}.")
    best, worst = list(s["best_sectors"].items()), list(s["worst_sectors"].items())
    if best and worst:
        lines.append(f"{best[0][0]} led ({_pct(best[0][1])}) while {worst[0][0]} lagged ({_pct(worst[0][1])}).")
    vix_word = "jumped to" if s["vix_1d"] > 0.05 else "eased to" if s["vix_1d"] < -0.05 else "held steady at"
    lines.append(f"India VIX {vix_word} {s['vix']:.1f} ({_pct(s['vix_1d'])})"
                 f"{'; the regime model reads ' + s['regime'] if s.get('regime') else ''}.")
    if s["new_highs"] or s["new_lows"]:
        lines.append(f"{len(s['new_highs'])} stocks hit 52-week highs and {len(s['new_lows'])} hit 52-week lows.")
    if "market_sentiment" in s:
        m = s["market_sentiment"]
        lines.append(f"News sentiment across stocks averaged {m:+.2f} "
                     f"({'positive' if m > 0.1 else 'negative' if m < -0.1 else 'neutral'}).")
    if s.get("earnings_week"):
        names = ", ".join(t.replace(".NS", "") for t in list(s["earnings_week"])[:6])
        lines.append(f"Results due in the next 7 days: {names}{'…' if len(s['earnings_week']) > 6 else ''}.")
    return lines


def habits(p: Panel, regimes: pd.Series | None = None, intraday: pd.DataFrame | None = None) -> dict:
    """Recurring market behaviour, measured on NIFTY since 2010."""
    b = p.bench.dropna()
    b = b[b.index >= "2010-01-01"]
    r = b.pct_change().dropna()
    g = r.groupby(r.index.dayofweek)
    dow = pd.DataFrame({"mean": g.mean(), "hit": g.apply(lambda x: (x > 0).mean()), "n": g.size()})
    dow = dow[dow["n"] >= 50].drop(columns="n")       # drop rare special sessions (Budget Saturdays, Muhurat)
    dow.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in dow.index]
    nxt = r.shift(-1)
    big_up, big_dn = r > 0.02, r < -0.02
    out = {
        "since": str(b.index[0].date()),
        "day_of_week": dow.round(4).to_dict("index"),
        "after_big_up": {"days": int(big_up.sum()), "next_day_mean": float(nxt[big_up].mean()),
                         "next_day_up": float((nxt[big_up] > 0).mean())},
        "after_big_down": {"days": int(big_dn.sum()), "next_day_mean": float(nxt[big_dn].mean()),
                           "next_day_up": float((nxt[big_dn] > 0).mean())},
        "up_day_share": float((r > 0).mean()),
        "streak_up_3_next": float((nxt[(r > 0) & (r.shift(1) > 0) & (r.shift(2) > 0)] > 0).mean()),
        "streak_dn_3_next": float((nxt[(r < 0) & (r.shift(1) < 0) & (r.shift(2) < 0)] > 0).mean()),
    }
    if regimes is not None:
        reg = regimes.reindex(r.index)
        out["by_regime"] = {k: {"ann_return": float(g.mean() * 252), "ann_vol": float(g.std() * np.sqrt(252))}
                            for k, g in r.groupby(reg) if len(g) > 20}
    if intraday is not None and len(intraday) > 100:
        out["intraday"] = intraday_habits(intraday)
    return out


def intraday_habits(bars: pd.DataFrame) -> dict:
    """From 5-minute NIFTY bars (columns Open/High/Low/Close, tz-aware IST index)."""
    df = bars.copy()
    df["d"] = df.index.date
    days = []
    for d, g in df.groupby("d"):
        if len(g) < 40:
            continue
        o, cl = g["Open"].iloc[0], g["Close"].iloc[-1]
        first_hr = g.iloc[:12]                       # 09:15-10:15
        fh_dir = np.sign(first_hr["Close"].iloc[-1] - o)
        hi_t, lo_t = g["High"].idxmax(), g["Low"].idxmin()
        days.append({"fh_agree": fh_dir == np.sign(cl - o), "range": (g["High"].max() - g["Low"].min()) / o,
                     "hi_hour": hi_t.hour + hi_t.minute / 60, "lo_hour": lo_t.hour + lo_t.minute / 60})
    d = pd.DataFrame(days)
    if d.empty:
        return {}
    bucket = lambda h: "09:15-10:15" if h < 10.25 else "10:15-13:00" if h < 13 else "13:00-15:30"  # noqa: E731
    return {"days": int(len(d)), "first_hour_sets_direction": float(d["fh_agree"].mean()),
            "avg_range": float(d["range"].mean()),
            "high_formed": d["hi_hour"].map(bucket).value_counts(normalize=True).round(3).to_dict(),
            "low_formed": d["lo_hour"].map(bucket).value_counts(normalize=True).round(3).to_dict()}
