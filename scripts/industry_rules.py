"""Well-known published trading rules, tested on NiveshRL's own NSE data after realistic costs.

Intraday (the archived 5-minute bars, in-play stocks, the intraday cost model: brokerage, STT, exchange, SEBI, stamp,
GST and 0.03% slippage per side; 1% of a ₹1 lakh pool risked per trade):
  1. Opening-range breakout on stocks in play (Zarattini, Barbon & Aziz 2024): direction of the first 5-minute bar,
     enter at the second bar's open, stop 10% of the daily ATR away, exit at the 15:15 square-off.
  2. VWAP trend (Zarattini & Aziz 2023) on NIFTY: long above the session VWAP, short below, re-checked every bar.
  3. Intraday momentum (Gao, Han, Li & Zhou 2018) on NIFTY: the first half-hour's sign sets the last half-hour's trade.
Daily (point-in-time NIFTY 200 members 2014-2026, 0.15% per side for delivery costs + slippage):
  4. Connors RSI(2): buy when close > 200-day average and 2-day RSI < 10; sell when close > 5-day average.
  5. Turtle / Donchian breakout: buy a close above the 55-day high, sell a close below the 20-day low.
  6. Time-series momentum on NIFTY (Hurst, Ooi & Pedersen): hold NIFTY when its 12-month return is positive, else cash.
  7. Short-term reversal: each week buy the 10 biggest 5-day losers, hold a week.
Writes report/results/industry_rules.md / .csv.

    python scripts/industry_rules.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from niveshrl.config import ROOT, load_yaml
from niveshrl.intraday.dl.dataset import _costs, load_bars
from niveshrl.research.data import load_panel

OUT = ROOT / "report" / "results"
SIDE_COST = 0.0015


def stats_trades(r: pd.Series, days: int, label: str) -> dict:
    """Per-trade R after costs → summary (R per trade, total, ₹ at 1% risk of ₹1 lakh)."""
    r = r.dropna()
    return {"rule": label, "kind": "intraday", "trades": int(len(r)), "avg R / trade": float(r.mean()) if len(r) else np.nan,
            "win rate": float((r > 0).mean()) if len(r) else np.nan, "total R": float(r.sum()),
            "net ₹ (₹1L pool)": float(r.sum() * 1000), "t-stat": float(r.mean() / r.std() * np.sqrt(len(r))) if len(r) > 2 else np.nan,
            "days": days}


def r_after_costs(entry, exit_, dist, side, cc, pool=100_000.0, risk=0.01, max_lev=4.0):
    qty = np.maximum(1.0, np.minimum(np.floor(pool * risk / dist), np.floor(max_lev * pool / np.asarray(entry, float))))
    gross = side * (exit_ - entry) * qty
    return (gross - _costs(np.atleast_1d(entry), np.atleast_1d(exit_), np.atleast_1d(qty), side, cc)) / (dist * qty)


def _notional(entry, exit_, side, cc, notional=100_000.0) -> float:
    """Net ₹ of a ₹1 lakh position (index level standing in for NIFTYBEES / futures), expressed per ₹1,000 like R."""
    qty = notional / entry
    net = side * (exit_ - entry) * qty - _costs(np.array([entry]), np.array([exit_]), np.array([qty]), side, cc)[0]
    return float(net / 1000)


def intraday_rules(cfg) -> list[dict]:
    cc = cfg["costs"]
    bars = load_bars()
    bars = bars.assign(day=bars["ts"].dt.normalize(), m=bars["ts"].dt.hour * 60 + bars["ts"].dt.minute)
    days = sorted(bars["day"].unique())
    from niveshrl.intraday.dl.dataset import causal_joins
    ip = causal_joins(bars.drop(columns=["day", "m"]), cfg, top=10_000)
    ip = ip[np.isfinite(ip["bar0_score"])].sort_values("bar0_score", ascending=False).groupby("day").head(20)
    ip["day"] = pd.to_datetime(ip["day"])
    from niveshrl.intraday.replay import daily_context
    ctx = daily_context(bars.drop(columns=["day", "m"]))
    out = []
    # 1. ORB on stocks in play
    rs = []
    keys = set(zip(ip["ticker"], ip["day"]))
    for (t, d), g in bars[bars["ticker"] != "^NSEI"].groupby(["ticker", "day"]):
        if (t, d) not in keys or len(g) < 10:
            continue
        try:
            atr = float(ctx.loc[(t, d), "atr"])
        except KeyError:
            continue
        o, c = g["open"].to_numpy(), g["close"].to_numpy()
        h, l, m = g["high"].to_numpy(), g["low"].to_numpy(), g["m"].to_numpy()
        if not np.isfinite(atr) or c[0] == o[0]:
            continue
        side = 1 if c[0] > o[0] else -1
        entry, dist = o[1], 0.10 * atr
        stop = entry - side * dist
        px = None
        for k in range(1, len(c)):
            if m[k] >= 15 * 60 + 15:
                px = o[k]
                break
            if (side > 0 and l[k] <= stop) or (side < 0 and h[k] >= stop):
                px = stop if k == 1 else (min(o[k], stop) if side > 0 else max(o[k], stop))
                break
        px = c[-1] if px is None else px
        px *= (1 - cc["slippage"] * side)
        e_fill = entry * (1 + cc["slippage"] * side)
        qty = max(1.0, min(np.floor(1000 / dist), np.floor(4 * 100_000 / e_fill)))     # 1% risk, at most 4x leverage
        net = side * (px - e_fill) * qty - _costs(np.array([e_fill]), np.array([px]), np.array([qty]), side,
                                                  cc | {"slippage": 0})[0]
        rs.append(((d, t), float(net / 1000)))                                          # in R of ₹1,000 (1% of ₹1 lakh)
    s = pd.Series([v for _, v in rs], index=pd.MultiIndex.from_tuples([k for k, _ in rs])) if rs else pd.Series(dtype=float)
    out.append(stats_trades(s, len(days), "Opening-range breakout, stocks in play (Zarattini 2024)"))
    # 2 + 3 on NIFTY: index level used as a stand-in for NIFTY futures / NIFTYBEES (costs charged as an equity trade)
    nf = bars[bars["ticker"] == "^NSEI"]
    vw_r, mom_r = [], []
    for d, g in nf.groupby("day"):
        if len(g) < 60:
            continue
        o, c, h, l, m = (g[k].to_numpy(float) for k in ("open", "close", "high", "low", "m"))
        tp = (h + l + c) / 3
        vwap = np.cumsum(tp) / np.arange(1, len(tp) + 1)          # index has no volume: equal-weight typical price
        atr = None
        pos, entry = 0, 0.0
        for k in range(1, len(c)):
            if m[k] >= 15 * 60 + 15:
                break
            want = 1 if c[k - 1] > vwap[k - 1] else -1
            if want != pos:
                if pos:
                    vw_r.append(((d, k), _notional(entry, o[k], pos, cc)))
                pos, entry = want, o[k]
        if pos:
            vw_r.append(((d, len(c)), _notional(entry, o[min(k, len(o) - 1)], pos, cc)))
        first = c[np.searchsorted(m, 9 * 60 + 45) - 1] / o[0] - 1
        i0 = np.searchsorted(m, 14 * 60 + 55)
        if i0 < len(c) - 1 and first != 0:
            side = 1 if first > 0 else -1
            mom_r.append((d, _notional(o[i0], c[-1], side, cc)))
    s = pd.Series([v for _, v in vw_r], index=pd.MultiIndex.from_tuples([k for k, _ in vw_r])) if vw_r else pd.Series(dtype=float)
    out.append(stats_trades(s, len(days), "VWAP trend on NIFTY (Zarattini & Aziz 2023)"))
    s = pd.Series([v for _, v in mom_r], index=[k for k, _ in mom_r])
    out.append(stats_trades(s, len(days), "First-half-hour momentum on NIFTY (Gao et al. 2018)"))
    return out


# --------------------------------------------------------------------------- daily rules
def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def run_positions(enter: pd.DataFrame, exit_: pd.DataFrame, ret: pd.DataFrame, member: pd.DataFrame,
                  max_pos: int = 10, rank: pd.DataFrame | None = None):
    """Daily book: enter at the next close after a signal, exit after an exit signal; equal slots of 1/max_pos."""
    held: dict = {}
    eq, trades = [], []
    E, X, R = enter.to_numpy(bool), exit_.to_numpy(bool), ret.fillna(0).to_numpy()
    M = member.to_numpy(bool) if member is not None else np.ones_like(E)
    cols = list(enter.columns)
    for i in range(len(enter) - 1):
        day_r = sum(R[i + 1, j] for j in held) / max_pos
        cost = 0.0
        for j in [j for j in held if X[i, j]]:
            cost += SIDE_COST / max_pos
            trades.append(np.prod([1 + x for x in held.pop(j)]) - 1 - 2 * SIDE_COST)
        cand = [j for j in np.flatnonzero(E[i] & M[i]) if j not in held]
        if rank is not None and cand:
            cand = sorted(cand, key=lambda j: rank.iat[i, j])
        for j in cand[:max_pos - len(held)]:
            held[j] = []
            cost += SIDE_COST / max_pos
        for j in held:
            held[j].append(R[i + 1, j])
        eq.append(day_r - cost)
    s = pd.Series(eq, index=enter.index[1:])
    return s, np.array(trades), cols


def stats_daily(r: pd.Series, trades, label: str) -> dict:
    nav = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"rule": label, "kind": "daily", "trades": int(len(trades)), "avg trade net": float(np.mean(trades)) if len(trades) else np.nan,
            "win rate": float(np.mean(np.asarray(trades) > 0)) if len(trades) else np.nan, "CAGR": float(cagr),
            "Sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan, "max drawdown": float(dd),
            "days": int(len(r))}


def daily_rules() -> list[dict]:
    p = load_panel(universe="pit")
    c = p.close.loc["2013-06-01":]
    lo_s = (p.low if p.low is not None else p.close).loc[c.index]
    ret = c.pct_change(fill_method=None)
    mem = p.member.reindex(c.index).fillna(False) if p.member is not None else None
    start = "2014-01-01"
    out = []
    sma200, sma5, rsi2 = c.rolling(200, min_periods=150).mean(), c.rolling(5).mean(), _rsi(c, 2)
    r, t, _ = run_positions((c > sma200) & (rsi2 < 10), c > sma5, ret, mem, 10, rsi2)
    out.append(stats_daily(r.loc[start:], t, "Connors RSI(2) mean reversion"))
    hi55, lo20 = c.rolling(55).max().shift(1), lo_s.rolling(20).min().shift(1)
    r, t, _ = run_positions(c > hi55, c < lo20, ret, mem, 10, -(c / hi55))
    out.append(stats_daily(r.loc[start:], t, "Turtle / Donchian 55-20 breakout"))
    b = p.bench.reindex(c.index)
    sig = (b / b.shift(252) - 1 > 0).shift(1).fillna(False)
    br = b.pct_change().fillna(0)
    switches = sig.astype(int).diff().abs().fillna(0)
    rr = br * sig - switches * SIDE_COST
    out.append(stats_daily(rr.loc[start:], [], "NIFTY 12-month trend (time-series momentum)") | {"trades": int(switches.loc[start:].sum())})
    wk = c.index.to_series().dt.to_period("W").ne(c.index.to_series().dt.to_period("W").shift(-1))
    r5 = c / c.shift(5) - 1
    enter = pd.DataFrame(False, index=c.index, columns=c.columns)
    rank5 = r5.where(mem if mem is not None else True).rank(axis=1)
    enter[wk.to_numpy()] = (rank5 <= 10)[wk.to_numpy()]
    exit_ = pd.DataFrame(wk.to_numpy()[:, None].repeat(c.shape[1], 1), index=c.index, columns=c.columns)
    r, t, _ = run_positions(enter, exit_, ret, mem, 10, r5)
    out.append(stats_daily(r.loc[start:], t, "Weekly short-term reversal (10 biggest 5-day losers)"))
    ew = ret.where(mem if mem is not None else True).mean(axis=1).fillna(0).loc[start:]
    out.append(stats_daily(ew, [], "Benchmark: equal weight of members (no costs)"))
    out.append(stats_daily(br.loc[start:], [], "Benchmark: NIFTY 50 buy and hold"))
    return out


def main() -> None:
    cfg = load_yaml("configs/intraday.yaml")
    cfg["pool"] = 100_000.0
    rows = intraday_rules(cfg) + daily_rules()
    t = pd.DataFrame(rows).set_index("rule")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "industry_rules.csv")
    def tbl(df):
        df = df.dropna(axis=1, how="all")
        cols = [str(x) for x in df.columns if x != "kind"]
        lines = ["| Rule | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 1)]
        for i, r in df.iterrows():
            lines.append("| " + i + " | " + " | ".join(f"{r[x]:,.3f}" if isinstance(r[x], float) else str(r[x]) for x in cols) + " |")
        return "\n".join(lines)
    md = ("# Published trading rules on NiveshRL's NSE data, after costs\n\n## Intraday (5-minute bars, about 58 days)\n\n"
          + tbl(t[t["kind"] == "intraday"]) + "\n\n## Daily (point-in-time NIFTY 200 members, 2014-2026)\n\n"
          + tbl(t[t["kind"] == "daily"]) + "\n\nFor comparison, 12-1 cross-sectional momentum (top 20, monthly) earned 19.3% "
          "a year after costs on the same point-in-time data (report/results/survivorship.md).\n")
    (OUT / "industry_rules.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
