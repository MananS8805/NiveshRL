"""India equity-delivery transaction cost model, in rupees.

Statutory charges follow an NSE contract note (STT, exchange transaction
charge, SEBI fee, stamp duty, GST, DP charge). Market-impact slippage is a
half-spread plus a square-root impact term. Rates live in
``configs/costs_india.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostBreakdown:
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp: float = 0.0
    gst: float = 0.0
    brokerage: float = 0.0
    dp: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return (self.stt + self.exchange + self.sebi + self.stamp + self.gst
                + self.brokerage + self.dp + self.slippage)

    @property
    def statutory(self) -> float:
        return self.total - self.slippage


class IndiaCostModel:
    def __init__(self, rates: dict, scale: float = 1.0):
        """``scale`` multiplies every proportional charge (for cost-sensitivity sweeps)."""
        self.r = rates
        self.scale = scale

    def cost(self, buy_value: np.ndarray, sell_value: np.ndarray,
             sigma_daily: np.ndarray | None = None, adv_value: np.ndarray | None = None) -> CostBreakdown:
        """Charges for one rebalance.

        buy_value / sell_value: rupee value traded per stock (non-negative).
        sigma_daily: daily return volatility per stock (for impact).
        adv_value: average daily traded value in rupees per stock (for impact).
        """
        r, k = self.r, self.scale
        buy = np.asarray(buy_value, dtype=np.float64)
        sell = np.asarray(sell_value, dtype=np.float64)
        turnover = buy + sell
        tot = float(turnover.sum())

        br = turnover * r["brokerage_rate"] * k
        if r.get("brokerage_cap", 0) > 0:
            br = np.minimum(br, r["brokerage_cap"])
        c = CostBreakdown()
        c.brokerage = float(br.sum())
        c.stt = r["stt_rate"] * tot * k
        c.exchange = r["exchange_rate"] * tot * k
        c.sebi = r["sebi_rate"] * tot * k
        c.stamp = r["stamp_rate"] * float(buy.sum()) * k
        c.gst = r["gst_rate"] * (c.brokerage + c.exchange + c.sebi)
        c.dp = r["dp_charge"] * int((sell > 1e-6).sum()) * (1.0 if k > 0 else 0.0)

        slip = r["half_spread"] * turnover
        if sigma_daily is not None and adv_value is not None:
            adv = np.maximum(np.asarray(adv_value, dtype=np.float64), 1.0)
            slip = slip + r["impact_k"] * np.asarray(sigma_daily) * np.sqrt(turnover / adv) * turnover
        c.slippage = float(slip.sum()) * k
        return c
