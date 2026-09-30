"""Turn a target allocation into an actionable plan for a small retail account.

- Whole shares only (NSE delivery has no fractional shares).
- New SIP cash goes to the most underweight stocks first, so small accounts
  rebalance by buying rather than selling. This avoids STT on sells, the DP
  charge and capital-gains tax.
- A sell is suggested only when a holding drifts more than ``drift_band``
  above target.
- Goal projection uses a block bootstrap of the strategy's historical daily
  returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Order:
    ticker: str
    side: str        # BUY / SELL
    shares: int
    price: float

    @property
    def value(self) -> float:
        return self.shares * self.price


def plan_orders(target_w: np.ndarray, tickers: list[str], prices: np.ndarray,
                holdings: np.ndarray | None = None, cash: float = 0.0, new_money: float = 0.0,
                drift_band: float = 0.05) -> tuple[list[Order], float]:
    """Whole-share orders that move towards ``target_w`` (over stocks + cash).

    Returns (orders, cash_left). Buys never spend more than the cash available.
    """
    n = len(tickers)
    holdings = np.zeros(n, dtype=np.int64) if holdings is None else np.asarray(holdings, dtype=np.int64)
    cash = float(cash) + float(new_money)
    value = float(holdings @ prices) + cash
    target_val = target_w[:n] * value
    cur_val = holdings * prices
    orders: list[Order] = []

    # 1. Sells only for positions that drifted well above target.
    for i in range(n):
        over = cur_val[i] - target_val[i]
        if cur_val[i] > 0 and over > drift_band * value:
            q = int(over // prices[i])
            if q > 0:
                orders.append(Order(tickers[i], "SELL", q, float(prices[i])))
                cash += q * prices[i]
                cur_val[i] -= q * prices[i]

    # 2. Buys: greedily fill the largest rupee shortfall one share at a time.
    shortfall = target_val - cur_val
    buys = np.zeros(n, dtype=np.int64)
    budget = cash - target_w[n] * value  # keep the target cash buffer
    while True:
        gap = shortfall - buys * prices
        # Only stocks we can afford AND where one share reduces the gap (buying it
        # doesn't overshoot by more than it fills). Checking only the single
        # largest gap would stop at an expensive share (e.g. TITAN at Rs 4,841 for a
        # Rs 2,320 gap) even when cheaper underweight stocks are still buyable.
        ok = (prices <= budget) & (gap > prices * 0.5)
        if not ok.any():
            break
        i = int(np.argmax(np.where(ok, gap, -np.inf)))
        buys[i] += 1
        budget -= prices[i]
    for i in np.flatnonzero(buys):
        orders.append(Order(tickers[i], "BUY", int(buys[i]), float(prices[i])))
        cash -= buys[i] * prices[i]
    return orders, cash


def orders_frame(orders: list[Order]) -> pd.DataFrame:
    return pd.DataFrame([{"Stock": o.ticker.replace(".NS", ""), "Action": o.side, "Shares": o.shares,
                          "Price (Rs)": round(o.price, 2), "Amount (Rs)": round(o.value, 0)}
                         for o in orders])


def goal_projection(daily_returns: pd.Series, initial: float, sip_monthly: float, years: float,
                    target: float | None = None, n_paths: int = 2000, block: int = 20,
                    seed: int = 0) -> dict:
    """Block-bootstrap wealth paths. Returns percentile bands per month and P(target)."""
    r = daily_returns.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    days = int(round(years * 252))
    n_blocks = int(np.ceil(days / block))
    starts = rng.integers(0, len(r) - block, (n_paths, n_blocks))
    paths_r = r[(starts[..., None] + np.arange(block)).reshape(n_paths, -1)[:, :days]]
    wealth = np.full(n_paths, float(initial))
    months = [0]
    bands = [np.full(n_paths, float(initial))]
    for d in range(days):
        wealth *= 1 + paths_r[:, d]
        if (d + 1) % 21 == 0:
            wealth += sip_monthly
            months.append((d + 1) // 21)
            bands.append(wealth.copy())
    W = np.stack(bands, 1)
    invested = initial + sip_monthly * np.array(months)
    out = {
        "months": np.array(months),
        "p10": np.percentile(W, 10, 0), "p50": np.percentile(W, 50, 0), "p90": np.percentile(W, 90, 0),
        "invested": invested,
        "final_median": float(np.median(wealth)),
    }
    if target:
        out["p_target"] = float((wealth >= target).mean())
    return out
