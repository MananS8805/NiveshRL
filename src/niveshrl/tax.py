"""Capital-gains tax on listed equity (India), FIFO lots.

- STCG: gains on shares held <= 12 months, taxed at ``stcg_rate``.
- LTCG: gains on shares held > 12 months, taxed at ``ltcg_rate`` above an
  annual exemption of ``ltcg_exemption`` rupees.
- Set-off: a short-term loss offsets short- and long-term gains; a long-term
  loss offsets only long-term gains. (Carry-forward across years is ignored.)

Tax is charged incrementally at the time of each sale, as the change in
the financial year's total liability. The yearly total therefore comes out
exact, and the agent feels the cost at the moment it decides to sell.
Financial years run April to March.
"""
from __future__ import annotations

from collections import deque

import pandas as pd

HOLD_DAYS_LT = 365


def financial_year(d: pd.Timestamp) -> int:
    return d.year if d.month >= 4 else d.year - 1


class TaxLedger:
    def __init__(self, n_stocks: int, rates: dict):
        self.rates = rates
        self.lots: list[deque] = [deque() for _ in range(n_stocks)]  # (date, shares, cost/share)
        self.fy: int | None = None
        self.st_net = 0.0
        self.lt_net = 0.0
        self.paid_this_fy = 0.0
        self.total_paid = 0.0

    def buy(self, i: int, date: pd.Timestamp, shares: float, price: float) -> None:
        if shares > 1e-12:
            self.lots[i].append([date, shares, price])

    def _liability(self) -> float:
        st, lt = self.st_net, self.lt_net
        if st < 0:              # ST loss offsets LT gains
            lt, st = lt + st, 0.0
        lt_taxable = max(0.0, lt - self.rates["ltcg_exemption"])
        return st * self.rates["stcg_rate"] + lt_taxable * self.rates["ltcg_rate"]

    def sell(self, i: int, date: pd.Timestamp, shares: float, price: float) -> float:
        """Record a FIFO sale. Returns the incremental tax due now (Rs, >= 0 or a refund < 0 within FY)."""
        fy = financial_year(date)
        if fy != self.fy:
            self.fy, self.st_net, self.lt_net, self.paid_this_fy = fy, 0.0, 0.0, 0.0
        remaining = shares
        q = self.lots[i]
        while remaining > 1e-12 and q:
            lot = q[0]
            take = min(remaining, lot[1])
            gain = take * (price - lot[2])
            if (date - lot[0]).days > HOLD_DAYS_LT:
                self.lt_net += gain
            else:
                self.st_net += gain
            lot[1] -= take
            remaining -= take
            if lot[1] <= 1e-12:
                q.popleft()
        due = self._liability() - self.paid_this_fy
        self.paid_this_fy += due
        self.total_paid += due
        return due
