"""Understanding a trade before taking it: a pre-entry checklist, what happened to similar past setups, and the
rupee outcome of each scenario after costs and tax.

- ``context_tags``: for every stock and day, the state a trade would start in: market risk (NIFTY above/below its
  200-day average), the stock's trend (above/below its own 200-day average), RSI(14) zone and distance from the
  20-day high. Computed from data up to that day only.
- ``similar_setups``: closed trades from the point-in-time historical replay of the swing rules
  (``report/results/monitor_history_pit.parquet``, both the model picks and the random control: every trade followed
  the same plan rules), filtered to those that started in the same context as the stock today. If fewer than
  ``min_n`` match, the least important tag is dropped, and the result says which tags were used.
- ``scenarios``: ₹ and R for stop, gap through the stop, T1 then breakeven, T1 then T2, and flat — after delivery
  costs and short-term capital gains tax.
- ``checklist``: Go / Wait / No-go from plain rules, each with its reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import plans as P
from ..config import ROOT
from .data import Panel

HISTORY = ROOT / "report" / "results" / "monitor_history_pit.parquet"
TAGGED = ROOT / "report" / "results" / "monitor_history_tagged.parquet"

STCG_RATE = 0.20 * 1.04            # listed equity held < 12 months (STT paid): 20% + 4% cess, from 23 Jul 2024
TAG_ORDER = ["trend", "rsi_zone", "near_high", "market"]   # dropped from the end when too few matches
TAG_LABELS = {"market": "NIFTY vs its 200-day average", "trend": "stock vs its 200-day average",
              "rsi_zone": "RSI(14) zone", "near_high": "distance from the 20-day high"}


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def context_tags(p: Panel) -> dict[str, pd.DataFrame]:
    c = p.close
    hi = p.high if p.high is not None else c
    sma200 = c.rolling(200, min_periods=150).mean()
    trend = pd.DataFrame(np.where(c > sma200, "above", "below"), index=c.index, columns=c.columns).where(sma200.notna())
    r = _rsi(c)
    zone = pd.DataFrame(np.select([r < 40, r < 60, r < 70], ["<40", "40-60", "60-70"], ">70"),
                        index=c.index, columns=c.columns).where(r.notna())
    dist = c / hi.rolling(20).max() - 1
    near = pd.DataFrame(np.select([dist >= -0.02, dist >= -0.08], ["within 2%", "2-8% below"], "over 8% below"),
                        index=c.index, columns=c.columns).where(dist.notna())
    b = p.bench
    mkt = pd.Series(np.where(b > b.rolling(200).mean(), "above", "below"), index=b.index).where(b.rolling(200).mean().notna())
    return {"trend": trend, "rsi_zone": zone, "near_high": near, "market": mkt}


def tags_now(p: Panel, ticker: str) -> dict:
    """The context tags for one stock on the panel's last day (the same rules as ``context_tags``, one column)."""
    sub = Panel(close=p.close[[ticker]].iloc[-260:], volume=p.volume[[ticker]].iloc[-260:], sectors=p.sectors,
                names=p.names, bench=p.bench.iloc[-260:], vix=p.vix,
                high=p.high[[ticker]].iloc[-260:] if p.high is not None else None)
    return tags_on(context_tags(sub), ticker, sub.close.index[-1])


def tags_on(tags: dict, ticker: str, date) -> dict:
    out = {}
    for k, v in tags.items():
        s = v if isinstance(v, pd.Series) else v[ticker] if ticker in v else None
        if s is None:
            continue
        s = s.loc[:date].dropna()
        if len(s):
            out[k] = s.iloc[-1]
    return out


def tag_history(hist: pd.DataFrame, tags: dict) -> pd.DataFrame:
    """Adds the context tags on each trade's signal date (vectorised lookups)."""
    h = hist[hist["status"] == "closed"].copy()
    if h.empty:
        return h
    for k, v in tags.items():
        if isinstance(v, pd.Series):
            h[k] = v.reindex(pd.DatetimeIndex(h["signal_date"])).to_numpy()
        else:
            st = v.stack()
            h[k] = st.reindex(pd.MultiIndex.from_arrays([pd.DatetimeIndex(h["signal_date"]), h["ticker"]])).to_numpy()
    return h


def build_tagged(p: Panel | None = None, hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tag the historical replay with the point-in-time panel (covers stocks no longer in the index) and save it."""
    if p is None:
        from .data import load_panel
        p = load_panel(universe="pit")
    hist = pd.read_parquet(HISTORY) if hist is None else hist
    out = tag_history(hist, context_tags(p))
    out.to_parquet(TAGGED)
    return out


@dataclass
class Similar:
    n: int
    used: list
    dropped: list
    trades: pd.DataFrame
    stats: dict
    baseline: dict


def _stats(r: pd.Series, days: pd.Series | None = None, reasons: pd.Series | None = None) -> dict:
    r = r.dropna()
    if not len(r):
        return {"n": 0}
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan
    return {"n": int(len(r)), "win_rate": float((r > 0).mean()), "avg_r": float(r.mean()), "median_r": float(r.median()),
            "p10": float(r.quantile(0.10)), "p90": float(r.quantile(0.90)), "t": float(r.mean() / se) if se and se > 0 else np.nan,
            "avg_days": float(days.mean()) if days is not None else np.nan}


def similar_setups(tagged: pd.DataFrame, now: dict, min_n: int = 30) -> Similar:
    if tagged is None or tagged.empty:
        return Similar(0, [], [], pd.DataFrame(), {"n": 0}, {"n": 0})
    base = _stats(tagged["r"], tagged["days"], tagged["reason"])
    keys = [k for k in TAG_ORDER if k in now]
    dropped = []
    while True:
        m = np.ones(len(tagged), dtype=bool)
        for k in keys:
            m &= (tagged[k] == now[k]).to_numpy()
        sel = tagged[m]
        if len(sel) >= min_n or not keys:
            break
        dropped.append(keys.pop())
    return Similar(len(sel), keys, dropped, sel, _stats(sel["r"], sel["days"], sel["reason"]), base)


def r_histogram(r: pd.Series, edges=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)) -> pd.Series:
    """Share of trades in each R bucket (open-ended at both ends)."""
    r = r.dropna()
    bins = [-np.inf, *edges, np.inf]
    labels = [f"< {edges[0]:g}R"] + [f"{a:g} to {b:g}R" for a, b in zip(edges[:-1], edges[1:])] + [f"> {edges[-1]:g}R"]
    return pd.cut(r, bins=bins, labels=labels, right=False).value_counts(normalize=True).reindex(labels).fillna(0)


# --------------------------------------------------------------------------- scenarios
def scenarios(plan, stcg_rate: float = STCG_RATE, gap_pct: float = 0.03, rates: dict | None = None) -> pd.DataFrame:
    """₹ outcome of each way the swing plan can end, for ``plan.qty`` shares, after costs and tax.

    Tax is short-term capital gains on a net gain (losses carry no tax here, and may offset other gains)."""
    q = plan.qty
    e, R = plan.entry, plan.risk_per_share
    gap_px = plan.stop * (1 - gap_pct)
    t1, t2 = plan.t1, plan.t2
    a, b = q // 3, q - q // 3                           # a third booked at T1, the rest after
    legs = {
        "Stop hit": [(q, plan.stop)],
        f"Gap {gap_pct:.0%} through the stop": [(q, gap_px)],
        "Flat (sold at entry)": [(q, e)],
        "T1, then stopped at entry": [(a, t1), (b, e)],
        "T1, then T2": [(a, t1), (b, t2)],
        "All at T2": [(q, t2)],
    }
    rows = []
    for name, ls in legs.items():
        gross = sum(n * (px - e) for n, px in ls)
        cost = sum(P.delivery_costs(e, px, n, rates)["total"] for n, px in ls if n > 0)
        pre_tax = gross - cost
        tax = max(pre_tax, 0.0) * stcg_rate
        net = pre_tax - tax
        rows.append({"Scenario": name, "Exit price": sum(n * px for n, px in ls) / q if q else np.nan,
                     "Gross ₹": gross, "Costs ₹": cost, "Tax ₹": tax, "Net ₹": net,
                     "Net R": net / (R * q) if q and R > 0 else np.nan,
                     "% of capital": net / plan.capital if plan.capital else np.nan})
    return pd.DataFrame(rows).set_index("Scenario")


# --------------------------------------------------------------------------- checklist
@dataclass
class Check:
    item: str
    status: str      # ok | caution | stop
    detail: str


def checklist(plan, risk, row: pd.Series | None, similar: Similar | None, results_in_days: float | None) -> tuple[str, list]:
    out = []
    if plan is None or plan.qty <= 0:
        out.append(Check("Position size", "stop", "Your capital and risk % cannot buy even one share with this stop."))
    else:
        out.append(Check("Position size", "ok", f"{plan.qty:,} shares, ₹{plan.rupee_risk:,.0f} at risk "
                                                f"({plan.risk_pct * plan.multiplier:.2%} of capital)."))
    out.append(Check("Market risk state", "ok" if risk.multiplier >= 1 else "caution",
                     f"{risk.state}: " + ("; ".join(risk.reasons) if risk.reasons else "no warnings")
                     + ("" if risk.multiplier >= 1 else " (size already halved)")))
    row = row if row is not None else pd.Series(dtype=float)
    v = row.get("vs_sma200")
    if v is not None and np.isfinite(v):
        out.append(Check("Long-term trend", "ok" if v > 0 else "caution",
                         f"{'above' if v > 0 else 'below'} its 200-day average ({v:+.1%})"
                         + ("" if v > 0 else ": buying against the trend")))
    if results_in_days is not None and np.isfinite(results_in_days) and 0 <= results_in_days <= 7:
        out.append(Check("Results soon", "caution", f"quarterly results in {int(results_in_days)} day(s): the price can "
                                                    "gap past the stop on results day."))
    if plan is not None:
        out.append(Check("Stop width", "caution" if plan.stop_pct > 0.06 else "ok",
                         f"{plan.stop_pct:.1%} below entry ({plan.risk_per_share / plan.atr:.1f}× ATR)"
                         + (": wide, so the position is small" if plan.stop_pct > 0.06 else "")))
    to = row.get("turnover_cr")
    if to is not None and np.isfinite(to):
        heavy = plan is not None and plan.position_value > 0.01 * to * 1e7
        out.append(Check("Liquidity", "caution" if to < 5 or heavy else "ok",
                         f"₹{to:,.0f} Cr traded a day" + (": thin, expect slippage" if to < 5 else
                                                          ": your order is over 1% of a day's value" if heavy else "")))
    rsi, st = row.get("rsi14"), row.get("breakout_state")
    stretched = (rsi is not None and np.isfinite(rsi) and rsi > 75) or st == "extended"
    if rsi is not None and np.isfinite(rsi):
        out.append(Check("Chasing?", "caution" if stretched else "ok",
                         f"RSI {rsi:.0f}" + (f", {st} from its pivot" if isinstance(st, str) else "")
                         + (": stretched, pullbacks are common" if stretched else "")))
    pu, rk = row.get("prob_up"), row.get("monthly_pct")
    if pu is not None and np.isfinite(pu) and rk is not None and np.isfinite(rk):
        weak = pu < 0.5 and rk < 0.5
        out.append(Check("Model views", "caution" if weak else "ok",
                         f"next-day P(up) {pu:.0%}, monthly ranker percentile {rk:.0%}"
                         + (": both below average" if weak else "")))
    if similar is not None and similar.stats.get("n", 0) >= 30:
        s = similar.stats
        out.append(Check("Similar past setups", "caution" if s["avg_r"] < 0 else "ok",
                         f"{s['n']:,} trades, average {s['avg_r']:+.2f}R, {s['win_rate']:.0%} winners"))
    n_stop = sum(c.status == "stop" for c in out)
    n_caut = sum(c.status == "caution" for c in out)
    verdict = "No-go" if n_stop else "Wait" if n_caut >= 2 else "Go"
    return verdict, out
