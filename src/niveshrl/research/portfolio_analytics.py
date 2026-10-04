"""Portfolio analytics and Indian capital-gains tax for your real holdings.

- ``parse_tradebook_csv``: a Kite (Zerodha Console) tradebook export — symbol, trade_date, trade_type (buy/sell),
  quantity, price — or any table with those columns.
- ``fifo``: matches sells to the oldest buys (first in, first out, as the Income Tax rules require for shares held in
  demat) and returns the realised matches and the lots still open.
- ``tax_summary``: a financial year's (April-March) short- and long-term gains with the set-off rules (short-term losses
  offset either kind of gain, long-term losses only long-term gains), the ₹1.25 lakh yearly LTCG exemption and the
  rates for listed shares sold from 23 July 2024: STCG 20%, LTCG 12.5% (plus 4% cess). Older rates (15% / 10%, ₹1 lakh)
  are not modelled: the summary says so for sales before that date.
- ``harvest``: open lots with unrealised losses that could offset this year's gains, and long-term gains that fit in the
  unused exemption.
- ``xirr``: the annualised return of dated cash flows (buys negative, sells and today's value positive).
- ``risk``: value, sector allocation, concentration, 1-year beta and volatility, and 1-day 95% value at risk of the
  current holdings, from the price panel.
- ``monthly_pnl``: realised P&L by calendar month.

Simplified estimates for understanding, not tax advice: no surcharge, no grandfathering of pre-2018 gains, no
intraday/F&O income, no corporate-action adjustment of lots.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .data import Panel

STCG_RATE, LTCG_RATE, CESS = 0.20, 0.125, 0.04
LTCG_EXEMPTION = 125_000.0
NEW_RATES_FROM = pd.Timestamp("2024-07-23")


# --------------------------------------------------------------------------- tradebook
def parse_tradebook_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "symbol" in ln.lower() or "instrument" in ln.lower()), 0)
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    low = {c: c.strip().lower() for c in df.columns}
    pick = lambda *keys: next((c for c, v in low.items() if any(k == v or k in v for k in keys)), None)  # noqa: E731
    sym, dt = pick("symbol", "instrument", "tradingsymbol"), pick("trade_date", "trade date", "date")
    side, qty, px = pick("trade_type", "trade type", "type", "side"), pick("quantity", "qty"), pick("price")
    if not all((sym, dt, side, qty, px)):
        raise ValueError("need symbol, trade date, trade type (buy/sell), quantity and price columns")
    out = pd.DataFrame({
        "ticker": df[sym].astype(str).str.strip().str.upper().str.replace(r"-(EQ|BE|BZ)$", "", regex=True),
        "date": pd.to_datetime(df[dt], errors="coerce", dayfirst=False),
        "side": df[side].astype(str).str.strip().str.lower().map(lambda s: "buy" if s.startswith("b") else
                                                                  "sell" if s.startswith("s") else None),
        "qty": pd.to_numeric(df[qty].astype(str).str.replace(",", ""), errors="coerce"),
        "price": pd.to_numeric(df[px].astype(str).str.replace(",", ""), errors="coerce"),
    }).dropna()
    out = out[(out["qty"] > 0) & (out["price"] > 0)]
    out["ticker"] = [t if t.endswith(".NS") else t + ".NS" for t in out["ticker"]]
    return out.sort_values(["date", "side"]).reset_index(drop=True)     # buys before sells on the same day


@dataclass
class Fifo:
    realised: pd.DataFrame     # ticker, buy_date, sell_date, qty, buy_price, sell_price, days, term, gain
    open_lots: pd.DataFrame    # ticker, buy_date, qty, price
    unmatched: list            # sells with no earlier buy in the file (bought before the tradebook starts)


def fifo(trades: pd.DataFrame) -> Fifo:
    lots: dict[str, list[list]] = {}
    real, unmatched = [], []
    for r in trades.itertuples():
        if r.side == "buy":
            lots.setdefault(r.ticker, []).append([r.date, float(r.qty), float(r.price)])
            continue
        q = float(r.qty)
        book = lots.get(r.ticker, [])
        while q > 1e-9 and book:
            bd, bq, bp = book[0]
            m = min(q, bq)
            days = (r.date - bd).days
            real.append({"ticker": r.ticker, "buy_date": bd, "sell_date": r.date, "qty": m, "buy_price": bp,
                         "sell_price": float(r.price), "days": days, "term": "LT" if long_term(bd, r.date) else "ST",
                         "gain": m * (float(r.price) - bp)})
            q -= m
            book[0][1] -= m
            if book[0][1] <= 1e-9:
                book.pop(0)
        if q > 1e-9:
            unmatched.append((r.ticker, r.date, q))
    open_lots = pd.DataFrame([{"ticker": t, "buy_date": d, "qty": q, "price": p}
                              for t, b in lots.items() for d, q, p in b if q > 1e-9],
                             columns=["ticker", "buy_date", "qty", "price"])
    return Fifo(pd.DataFrame(real, columns=["ticker", "buy_date", "sell_date", "qty", "buy_price", "sell_price", "days",
                                            "term", "gain"]), open_lots, unmatched)


def long_term(buy: pd.Timestamp, sell: pd.Timestamp) -> bool:
    """Listed shares are long-term when held for more than 12 months."""
    b = pd.Timestamp(buy)
    anniversary = b + pd.DateOffset(years=1)
    return pd.Timestamp(sell) > anniversary


def fy_of(d) -> int:
    """Financial year by its starting calendar year: 15 Mar 2026 -> 2025 (FY 2025-26)."""
    d = pd.Timestamp(d)
    return d.year if d.month >= 4 else d.year - 1


def fy_label(fy: int) -> str:
    return f"FY {fy}-{str(fy + 1)[-2:]}"


def tax_summary(realised: pd.DataFrame, fy: int) -> dict:
    r = realised[[fy_of(d) == fy for d in realised["sell_date"]]] if len(realised) else realised
    st = r.loc[r["term"] == "ST", "gain"] if len(r) else pd.Series(dtype=float)
    lt = r.loc[r["term"] == "LT", "gain"] if len(r) else pd.Series(dtype=float)
    st_gain, st_loss = float(st[st > 0].sum()), float(-st[st < 0].sum())
    lt_gain, lt_loss = float(lt[lt > 0].sum()), float(-lt[lt < 0].sum())
    net_st = st_gain - st_loss
    net_lt = lt_gain - lt_loss
    # set-off: a net short-term loss can reduce long-term gains; a net long-term loss cannot touch short-term gains
    if net_st < 0 and net_lt > 0:
        used = min(-net_st, net_lt)
        net_lt -= used
        net_st += used
    taxable_st = max(net_st, 0.0)
    exempt_used = min(max(net_lt, 0.0), LTCG_EXEMPTION)
    taxable_lt = max(net_lt - LTCG_EXEMPTION, 0.0)
    tax = (taxable_st * STCG_RATE + taxable_lt * LTCG_RATE) * (1 + CESS)
    carry = max(-net_st, 0.0) + max(-net_lt, 0.0)
    old = bool(len(r) and (pd.to_datetime(r["sell_date"]) < NEW_RATES_FROM).any())
    return {"fy": fy, "st_gain": st_gain, "st_loss": st_loss, "lt_gain": lt_gain, "lt_loss": lt_loss,
            "net_st": net_st, "net_lt": net_lt, "taxable_st": taxable_st, "taxable_lt": taxable_lt,
            "exempt_used": exempt_used, "exempt_left": LTCG_EXEMPTION - exempt_used, "tax": tax,
            "carry_forward_loss": carry, "old_rates_apply": old, "n_sales": int(len(r))}


def unrealised(open_lots: pd.DataFrame, prices: dict[str, float], today=None) -> pd.DataFrame:
    today = pd.Timestamp(today or date.today())
    if open_lots.empty:
        return pd.DataFrame(columns=["ticker", "buy_date", "qty", "price", "last", "gain", "term", "days_to_lt"])
    d = open_lots.copy()
    d["last"] = [prices.get(t, np.nan) for t in d["ticker"]]
    d["gain"] = d["qty"] * (d["last"] - d["price"])
    d["term"] = ["LT" if long_term(b, today) else "ST" for b in d["buy_date"]]
    d["days_to_lt"] = [max(((pd.Timestamp(b) + pd.DateOffset(years=1)) - today).days + 1, 0) for b in d["buy_date"]]
    return d


def harvest(unreal: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """Suggestions to look at (not advice): losses that offset this year's taxable gains, long-term gains that fit
    in the unused ₹1.25 lakh exemption, and short-term winners close to turning long-term."""
    rows = []
    st_left, lt_left, ex_left = summary["taxable_st"], summary["taxable_lt"], summary["exempt_left"]
    for r in unreal.dropna(subset=["last"]).sort_values("gain").itertuples():
        if r.gain < 0:
            # a short-term loss offsets short-term gains first, then long-term; a long-term loss only long-term
            save, loss = 0.0, -r.gain
            if r.term == "ST" and st_left > 0:
                u = min(loss, st_left)
                save, st_left, loss = save + u * STCG_RATE, st_left - u, loss - u
            if loss > 0 and lt_left > 0:
                u = min(loss, lt_left)
                save, lt_left = save + u * LTCG_RATE, lt_left - u
            if save > 0:
                rows.append({"ticker": r.ticker, "Idea": f"book the {r.term} loss to offset this year's taxable gains",
                             "Unrealised ₹": r.gain, "Est. tax effect ₹": save * (1 + CESS), "Bought": r.buy_date})
        elif r.gain > 0 and r.term == "LT" and ex_left > 0:
            use = min(r.gain, ex_left)
            ex_left -= use
            rows.append({"ticker": r.ticker, "Idea": "sell and rebuy to use the tax-free LTCG allowance "
                                                     "(raises your cost price)", "Unrealised ₹": r.gain,
                         "Est. tax effect ₹": use * LTCG_RATE * (1 + CESS), "Bought": r.buy_date})
        elif r.gain > 0 and r.term == "ST" and 0 < r.days_to_lt <= 45:
            diff = r.gain * (STCG_RATE - LTCG_RATE) * (1 + CESS)
            rows.append({"ticker": r.ticker, "Idea": f"turns long-term in {r.days_to_lt} days: selling now costs more tax",
                         "Unrealised ₹": r.gain, "Est. tax effect ₹": diff, "Bought": r.buy_date})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- returns
def xirr(flows: list[tuple], lo: float = -0.99, hi: float = 10.0) -> float:
    """Annual rate r with Σ cf / (1+r)^(days/365) = 0, by bisection (robust; needs a sign change)."""
    if not flows:
        return float("nan")
    t0 = min(pd.Timestamp(d) for d, _ in flows)
    ts = np.array([(pd.Timestamp(d) - t0).days / 365.0 for d, _ in flows])
    cf = np.array([float(v) for _, v in flows])
    if not ((cf > 0).any() and (cf < 0).any()):
        return float("nan")
    f = lambda r: float(np.sum(cf / (1 + r) ** ts))  # noqa: E731
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < 1e-7:
            break
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return mid


def portfolio_flows(trades: pd.DataFrame, value_now: float, today=None) -> list[tuple]:
    flows = [(r.date, -r.qty * r.price if r.side == "buy" else r.qty * r.price) for r in trades.itertuples()]
    flows.append((pd.Timestamp(today or date.today()), value_now))
    return flows


def monthly_pnl(realised: pd.DataFrame) -> pd.DataFrame:
    """Realised gains by year (rows) and month (columns), plus a full-year total."""
    if realised.empty:
        return pd.DataFrame()
    d = realised.assign(y=pd.to_datetime(realised["sell_date"]).dt.year, m=pd.to_datetime(realised["sell_date"]).dt.month)
    t = d.pivot_table(index="y", columns="m", values="gain", aggfunc="sum").reindex(columns=range(1, 13))
    import calendar
    t.columns = [calendar.month_abbr[m] for m in t.columns]
    t["Full year"] = t.sum(axis=1, min_count=1)
    t.index.name = "Year"
    return t


# --------------------------------------------------------------------------- risk
def risk(p: Panel, qty: dict[str, float], prices: dict[str, float] | None = None, days: int = 252) -> dict:
    prices = prices or {}
    tick = [t for t in qty if t in p.close.columns]
    if not tick:
        return {}
    last = {t: prices.get(t) or float(p.close[t].dropna().iloc[-1]) for t in tick}
    val = pd.Series({t: qty[t] * last[t] for t in tick})
    total = float(val.sum())
    w = val / total
    r = p.close[tick].pct_change(fill_method=None).iloc[-days:].fillna(0)
    port = r @ w
    b = p.bench.pct_change().reindex(r.index).fillna(0)
    beta = float(np.cov(port, b)[0, 1] / np.var(b, ddof=1)) if b.var() > 0 else float("nan")
    vol = float(port.std() * np.sqrt(252))
    var_hist = float(-np.percentile(port, 5) * total)
    var_param = float(1.645 * port.std() * total)
    sec = w.groupby(p.sectors.reindex(w.index).fillna("Other")).sum().sort_values(ascending=False)
    # each stock's share of portfolio variance (Euler allocation)
    cov = r.cov() * 252
    mrc = cov.values @ w.values
    contrib = pd.Series(w.values * mrc / (w.values @ mrc), index=w.index) if (w.values @ mrc) > 0 else w * np.nan
    worst = port.nsmallest(1)
    return {"value": total, "weights": w.sort_values(ascending=False), "beta": beta, "vol": vol, "var95": var_hist,
            "var95_param": var_param, "sectors": sec, "top_weight": float(w.max()),
            "hhi": float((w ** 2).sum()), "effective_n": float(1 / (w ** 2).sum()), "risk_contrib": contrib,
            "worst_day": (worst.index[0], float(worst.iloc[0])) if len(worst) else None,
            "daily": port, "n_days": int(len(port))}
