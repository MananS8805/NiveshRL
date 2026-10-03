"""NSE intraday (MIS) charges per trade and speculative-income tax. Every rate is in configs/intraday.yaml.

A round trip = two executed orders (buy + sell). For a long trade the buy is the entry; for a short the
sell is the entry. Charges:
- brokerage: min(flat ₹20, 0.03% of the order value) per order;
- STT 0.025% on the sell side only; stamp duty 0.003% on the buy side only;
- NSE transaction charge 0.00297% and SEBI ₹10/crore on both sides; GST 18% on brokerage + exchange + SEBI;
- slippage 0.03% per side (in addition to filling at the next bar's open).
Tax: intraday profit is speculative business income, taxed at your slab rate plus cess on the financial
year's *net* speculative profit (losses offset gains within the year).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TradeCosts:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst + self.slippage

    def to_dict(self) -> dict:
        return asdict(self) | {"total": self.total}


def round_trip(entry: float, exit_: float, qty: int, side: str, c: dict) -> TradeCosts:
    buy_px, sell_px = (entry, exit_) if side == "long" else (exit_, entry)
    buy, sell = buy_px * qty, sell_px * qty
    brk = min(c["brokerage_flat"], c["brokerage_pct"] * buy) + min(c["brokerage_flat"], c["brokerage_pct"] * sell)
    exch = c["exchange"] * (buy + sell)
    sebi = c["sebi"] * (buy + sell)
    return TradeCosts(brokerage=brk, stt=c["stt_sell"] * sell, exchange=exch, sebi=sebi, stamp=c["stamp_buy"] * buy,
                      gst=c["gst"] * (brk + exch + sebi), slippage=c["slippage"] * (buy + sell))


def speculative_tax(fy_net_profit: float, tax_cfg: dict) -> float:
    """Tax on the financial year's net speculative profit so far (zero if the year is net negative)."""
    return max(fy_net_profit, 0.0) * tax_cfg["slab_rate"] * (1 + tax_cfg["cess"])


def fy_of(day) -> str:
    """Indian financial year label (April–March) for a date."""
    y = day.year if day.month >= 4 else day.year - 1
    return f"FY{y}-{(y + 1) % 100:02d}"
