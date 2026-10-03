"""Cross-sectional strategy backtester with India delivery costs.

A strategy is a monthly **signal** (a score per stock per month-end) plus
**portfolio rules**: how many stocks, how to weight them, how often to
rebalance, and optional overlays (volatility targeting, regime filter).

Mechanics:
- Rebalance at the close of a month-end (every ``rebalance_months``). Only
  stocks with a score *and* a price that day are eligible.
- Between rebalances, holdings drift with daily returns. A missing daily
  price (suspension) counts as a 0% return for that day. Cash earns
  ``cash_rate``.
- Trades pay the same ``IndiaCostModel`` as the RL simulator, in rupees:
  STT, stamp duty, exchange, SEBI and GST charges, the flat DP charge per
  stock sold, and half-spread plus square-root impact from each stock's
  trailing volatility and traded value.
- Long-short mode (short the bottom bucket) is included for research only.
  Retail investors can't short NSE delivery stocks overnight. Shorts are
  charged the same costs, and cash earns the cash rate on the full capital.

No overlay looks ahead. Volatility targeting uses the target portfolio's
trailing 63-day realised volatility at the rebalance date, and the regime
filter uses the regime label known at that date.
"""
from __future__ import annotations

import calendar
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import load_yaml
from ..costs import IndiaCostModel
from ..metrics import summarize
from .data import Panel


@dataclass
class StrategySpec:
    name: str = "Strategy"
    signal: str = "ffnn"                 # model name, or factor: momentum / reversal / lowvol / equal
    top: float = 0.1                     # <= 1: fraction of eligible stocks; > 1: number of stocks
    weighting: str = "equal"             # equal | score | inv_vol
    long_short: bool = False
    rebalance_months: int = 1
    max_weight: float = 0.10
    cost_scale: float = 1.0
    capital: float = 1_000_000.0
    cash_rate: float = 0.065
    vol_target: float | None = None      # annualised, e.g. 0.15
    regime_exposure: dict | None = None  # e.g. {"Bull": 1.0, "Neutral": 0.7, "Stress": 0.3}
    min_trade: float = 0.001             # skip per-stock trades smaller than this share of the portfolio
    start: str = "2012-01-01"
    end: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    spec: StrategySpec
    nav: pd.Series                      # daily, starts at 1
    rebalances: pd.DataFrame            # per rebalance: turnover, cost, holdings, invested
    weights: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.nav.pct_change().dropna()


def capped_weights(raw: pd.Series, cap: float) -> pd.Series:
    """Normalise positive weights to sum to 1, with no weight above ``cap``
    (the excess is redistributed pro rata)."""
    w = raw.clip(lower=0)
    w = w / w.sum() if w.sum() > 0 else w
    cap = max(cap, 1.0 / max(len(w), 1))
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        room = ~over & (w > 0)
        if not room.any():
            break
        w[room] += excess * w[room] / w[room].sum()
    return w


def _cap_np(raw: np.ndarray, cap: float) -> np.ndarray:
    """NumPy twin of :func:`capped_weights` for the backtest's inner loop."""
    w = np.clip(np.asarray(raw, dtype=float), 0, None)
    w = w / w.sum() if w.sum() > 0 else w
    cap = max(cap, 1.0 / max(len(w), 1))
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        room = ~over & (w > 0)
        if not room.any():
            break
        w[room] += excess * w[room] / w[room].sum()
    return w


def factor_scores(p: Panel, name: str) -> pd.DataFrame:
    """Classic factor signals at month-ends (higher = more attractive)."""
    me = p.month_ends()
    c = p.close
    if name == "momentum":        # 12-1 month momentum
        s = c.shift(21) / c.shift(252) - 1
    elif name == "reversal":      # short-term reversal: last month's losers
        s = -(c / c.shift(21) - 1)
    elif name == "lowvol":        # low volatility
        s = -p.returns().rolling(63, min_periods=40).std()
    elif name == "equal":         # every eligible stock
        s = c.notna().astype(float).where(c.notna())
    else:
        raise ValueError(name)
    return s.reindex(me)


_ARRAYS: dict[int, tuple] = {}


def _market_arrays(p: Panel) -> tuple:
    """Daily returns and the rolling stats the engine needs, computed once per panel."""
    key = id(p)
    if key not in _ARRAYS:
        r = p.returns()
        _ARRAYS[key] = (r.fillna(0.0), r.rolling(63, min_periods=40).std(), r.rolling(20, min_periods=5).std(),
                        (p.close * p.volume).rolling(20, min_periods=5).mean())
    return _ARRAYS[key]


def _core():
    try:
        import niveshrl_core
        return niveshrl_core
    except ImportError:
        return None


def run_backtest(p: Panel, scores: pd.DataFrame, spec: StrategySpec,
                 regimes: pd.Series | None = None, bench: pd.Series | None = None,
                 engine: str = "auto") -> BacktestResult:
    """``scores``: month-end dates x tickers (NaN = not eligible).

    ``engine``: "cpp" (niveshrl_core), "numpy", or "auto" (C++ when the extension is
    installed). Both produce the same numbers; tests/test_core_parity.py checks it.
    """
    close = p.close
    rets, vol63, sigma20, adv = _market_arrays(p)
    costs = IndiaCostModel(load_yaml("configs/costs_india.yaml"), scale=spec.cost_scale)
    cash_d = (1 + spec.cash_rate) ** (1 / 252) - 1

    start = pd.Timestamp(spec.start)
    end = pd.Timestamp(spec.end) if spec.end else close.index[-1]
    reb_dates = [d for d in scores.index if start <= d <= end and d in close.index]
    reb_dates = reb_dates[::max(1, spec.rebalance_months)]
    if not reb_dates:
        raise ValueError("no rebalance dates in range")

    days = close.index[(close.index >= reb_dates[0]) & (close.index <= end)]
    cols = close.columns
    R_all = rets.to_numpy()
    day0 = close.index.get_loc(days[0])
    R = R_all[day0:day0 + len(days)]
    reb_pos = days.get_indexer(reb_dates)
    # Everything the rebalance step needs, as arrays aligned to the rebalance dates.
    S = scores.reindex(index=reb_dates, columns=cols).to_numpy()
    if p.member is not None:                          # point-in-time universe: only index members on that date
        S = np.where(p.member.reindex(index=reb_dates, columns=cols).fillna(False).to_numpy(), S, np.nan)
    OK = close.loc[reb_dates].notna().to_numpy() & np.isfinite(S)
    VOL = vol63.loc[reb_dates].to_numpy()
    SIG = np.nan_to_num(sigma20.loc[reb_dates].to_numpy(), nan=0.02)
    ADV = np.nan_to_num(adv.loc[reb_dates].to_numpy(), nan=1e9)
    REG = regimes.reindex(reb_dates, method="ffill").to_numpy() if (spec.regime_exposure and regimes is not None) else None

    core = _core() if engine in ("auto", "cpp") else None
    if engine == "cpp" and core is None:
        raise ImportError("niveshrl_core is not installed: pip install ./cpp")
    if core is not None:
        rates = load_yaml("configs/costs_india.yaml")
        cr = core.CostRates(stt=rates["stt_rate"], exchange=rates["exchange_rate"], sebi=rates["sebi_rate"],
                            stamp=rates["stamp_rate"], gst=rates["gst_rate"], brokerage_rate=rates["brokerage_rate"],
                            brokerage_cap=rates.get("brokerage_cap", 0.0), dp=rates["dp_charge"],
                            half_spread=rates["half_spread"], impact_k=rates["impact_k"], scale=spec.cost_scale)
        regm = np.array([spec.regime_exposure.get(x, 1.0) for x in REG], dtype=float) if REG is not None \
            else np.ones(len(reb_pos))
        wmap = {"equal": 0, "score": 1, "inv_vol": 2}
        o = core.run_backtest(np.ascontiguousarray(R, dtype=float), np.ascontiguousarray(R_all, dtype=float), int(day0),
                              np.ascontiguousarray(reb_pos, dtype=np.int64),
                              np.ascontiguousarray(np.nan_to_num(S, nan=-np.inf)),
                              np.ascontiguousarray(OK, dtype=np.uint8), np.ascontiguousarray(VOL),
                              np.ascontiguousarray(SIG), np.ascontiguousarray(ADV), regm, float(spec.top), wmap.get(spec.weighting, 0), bool(spec.long_short),
                              float(spec.max_weight), float(spec.capital), float(cash_d),
                              float(spec.vol_target or 0.0), float(spec.min_trade), cr)
        nav_vals = o["nav"]
        rows, weights = [], {}
        for k, j in enumerate(reb_pos):
            if not o["traded"][k]:
                continue
            d = days[j]
            wk = o["weights"][k]
            nz = wk != 0
            weights[d] = pd.Series(wk[nz], index=cols[nz])
            rows.append({"date": d, "turnover": float(o["turnover"][k]), "cost": float(o["cost"][k]),
                         "holdings": int(o["holdings"][k]), "invested": float(o["invested"][k]),
                         "value": float(o["value"][k])})
    else:
        hold = np.zeros(len(cols))                        # rupee value per stock (negative = short)
        cash = V = spec.capital
        nav_vals = np.empty(len(days))
        rows, weights = [], {}

        # Trades happen at rebalance closes; in between, holdings drift with daily
        # returns, vectorised over each holding period.
        bounds = list(reb_pos) + [len(days) - 1]
        for k, j in enumerate(reb_pos):
            d = days[j]
            V = cash + hold.sum()
            elig = np.flatnonzero(OK[k])
            if len(elig) >= 5:
                n = int(round(spec.top * len(elig))) if spec.top <= 1 else int(spec.top)
                n = max(1, min(n, len(elig)))
                order = elig[np.argsort(-S[k, elig], kind="stable")]
                longs = order[:n]
                if spec.weighting == "score":
                    raw = np.clip(S[k, longs] - S[k, elig].min(), 1e-9, None)
                elif spec.weighting == "inv_vol":
                    v = VOL[k, longs]
                    v = np.where(np.isfinite(v) & (v > 0), v, np.nanmedian(VOL[k, elig]))
                    raw = 1.0 / v
                else:
                    raw = np.ones(n)
                w = np.zeros(len(cols))
                w[longs] = _cap_np(raw, spec.max_weight)
                if spec.long_short:
                    w[order[-n:]] -= _cap_np(np.ones(n), spec.max_weight)
                invested = 1.0
                if REG is not None:
                    invested *= spec.regime_exposure.get(REG[k], 1.0)
                if spec.vol_target:
                    g = day0 + j
                    pv = float((R_all[max(0, g - 62):g + 1] @ w).std() * np.sqrt(252))
                    if pv > 0:
                        invested *= min(1.0, spec.vol_target / pv)
                # Size trades on the value left *after* costs, so a fully invested
                # book doesn't end with negative cash (hidden leverage).
                V_eff = V
                for _ in range(3):
                    delta = w * invested * V_eff - hold
                    delta[np.abs(delta) < spec.min_trade * V] = 0.0      # dust filter, as in the RL env
                    buy, sell = np.clip(delta, 0, None), np.clip(-delta, 0, None)
                    c = costs.cost(buy, sell, SIG[k], ADV[k]).total
                    V_eff = V - c
                # Positions left untouched by the dust filter can leave cash a hair
                # negative; trim buys pro rata so the book never borrows.
                short_cash = (hold + delta).sum() + c - V
                if not spec.long_short and short_cash > 0 and buy.sum() > 0:
                    delta = np.where(delta > 0, delta * max(0.0, 1 - short_cash / buy.sum()), delta)
                    buy = np.clip(delta, 0, None)
                    c = costs.cost(buy, sell, SIG[k], ADV[k]).total
                turnover = float(np.abs(delta).sum() / V / (2 if not spec.long_short else 4))
                hold = hold + delta
                cash = V - hold.sum() - c
                nz = hold != 0
                weights[d] = pd.Series(hold[nz] / (cash + hold.sum()), index=cols[nz])
                rows.append({"date": d, "turnover": turnover, "cost": c, "holdings": int(nz.sum()),
                             "invested": invested, "value": V})
            nav_vals[j] = cash + hold.sum()
            j1 = bounds[k + 1]
            if j1 > j:
                growth = np.cumprod(1 + R[j + 1:j1 + 1], axis=0)
                H = hold * growth
                cgrow = cash * (1 + cash_d) ** np.arange(1, j1 - j + 1)
                nav_vals[j + 1:j1 + 1] = cgrow + H.sum(1)
                hold, cash = H[-1], cgrow[-1]

    nav = pd.Series(nav_vals, index=days) / spec.capital
    reb = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    b = (bench if bench is not None else p.bench).reindex(nav.index).ffill()
    m = summarize(nav, b / b.iloc[0], reb if len(reb) else None, rf=spec.cash_rate)
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    # Rupee totals compound with the account; the drag in %/yr is comparable across strategies.
    m["Cost drag %/yr"] = float((reb["cost"] / reb["value"]).sum() / years) if len(reb) else 0.0
    m["Avg holdings"] = float(reb["holdings"].mean()) if len(reb) else 0.0
    m["Avg invested"] = float(reb["invested"].mean()) if len(reb) else 0.0
    return BacktestResult(spec=spec, nav=nav, rebalances=reb, weights=weights, metrics=m)


def benchmark_nav(p: Panel, start: str, end: str | None = None) -> pd.Series:
    b = p.bench.loc[start:end].dropna()
    return b / b.iloc[0]


def monthly_table(nav: pd.Series) -> pd.DataFrame:
    """Year x month returns (the classic tearsheet heatmap), plus a full-year column."""
    me = nav.resample("ME").last()
    m = me.pct_change()
    m.iloc[0] = me.iloc[0] / nav.iloc[0] - 1
    tab = pd.DataFrame({"year": m.index.year, "month": m.index.month, "r": m.values})         .pivot(index="year", columns="month", values="r")
    tab.columns = [calendar.month_abbr[c] for c in tab.columns]
    ye = nav.groupby(nav.index.year).last()
    prev = ye.shift(1)
    prev.iloc[0] = nav.iloc[0]
    tab["Year"] = (ye / prev - 1).reindex(tab.index)
    return tab


def rolling_sharpe(nav: pd.Series, window: int = 252, rf: float = 0.065) -> pd.Series:
    r = nav.pct_change()
    ex = r - ((1 + rf) ** (1 / 252) - 1)
    return ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(252)


def _nav_stats(nav: pd.Series) -> dict:
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    cagr = float(nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    dd = float((nav / nav.cummax() - 1).min())
    r = nav.pct_change().dropna()
    ex = r - ((1.065) ** (1 / 252) - 1)
    sharpe = float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else float("nan")
    return {"CAGR": cagr, "MaxDD": dd, "Sharpe": sharpe, "Return/DD": cagr / abs(dd) if dd < 0 else float("nan")}


def luck_test(p: Panel, scores: pd.DataFrame, spec: StrategySpec, n_paths: int = 200, seed: int = 0,
              regimes: pd.Series | None = None) -> dict:
    """Is the strategy's selection better than luck?

    Re-runs the *same* rules (portfolio size, weighting, caps, rebalance dates, costs, overlays) ``n_paths``
    times, but at every rebalance picks stocks at random from the same eligible set (stocks the signal
    scored that date). Returns the random paths' metrics, their 5/50/95% bands, the strategy's percentile
    among them, and NIFTY buy-and-hold over the same days. Reproducible with ``seed``.
    """
    real = run_backtest(p, scores, spec, regimes=regimes)
    rng = np.random.default_rng(seed)
    eligible = scores.notna()
    rspec = StrategySpec(**{**spec.to_dict(), "weighting": "equal" if spec.weighting == "score" else spec.weighting,
                            "name": "random"})
    rows = []
    for _ in range(n_paths):
        rnd = pd.DataFrame(rng.random(scores.shape), index=scores.index, columns=scores.columns).where(eligible)
        rows.append(_nav_stats(run_backtest(p, rnd, rspec, regimes=regimes).nav))
    paths = pd.DataFrame(rows)
    strat = _nav_stats(real.nav)
    bench = benchmark_nav(p, str(real.nav.index[0].date()), str(real.nav.index[-1].date()))
    bands = paths.quantile([0.05, 0.5, 0.95])
    pct = {k: float((paths[k] < strat[k]).mean()) if k != "MaxDD" else float((paths[k] < strat[k]).mean())
           for k in strat}
    return {"strategy": strat, "paths": paths, "bands": bands, "percentile": pct,
            "buy_hold": _nav_stats(bench), "nav": real.nav, "n_paths": n_paths, "seed": seed,
            "period": (real.nav.index[0], real.nav.index[-1])}
