"""Views over the optional NSE data (Options → Optional NSE data): a stock's option chain, shareholding trend and
today's large deals; the market's FII/DII flows, NIFTY/BANKNIFTY option positioning and bulk/block deals.

Everything loads in the background only when the tab is opened, and shows a plain "unavailable" message when NSE
does not answer. Nothing is estimated or filled in."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

from ...research import nse as N
from .. import data, theme
from ..widgets import ExplainButton, FrameTable, KpiRow, h2, muted, run_async
from . import scrolling, vbox

OFF = ("Optional NSE data is turned off. Turn it on in Options → Optional NSE data. It uses NSE's public website "
       "APIs, which are unofficial, change without notice and can block frequent use.")


def _chain_view(df: pd.DataFrame, underlying: float, width: int = 12) -> pd.DataFrame:
    if df.empty or not np.isfinite(underlying):
        return df
    i = int((df["Strike"] - underlying).abs().idxmin())
    v = df.iloc[max(0, i - width): i + width + 1].copy()
    v.index = [("▶ " if j == i else "") + f"{k:,.0f}" for j, k in zip(v.index, v["Strike"])]
    return v.drop(columns=["Strike"])


def _chain_kpis(st: dict, underlying: float) -> list:
    return [
        ("Put-call ratio (OI)", f"{st['pcr']:.2f}", None, f"today's change {st['pcr_change']:.2f}"
         if st["pcr_change"] == st["pcr_change"] else ""),
        ("Max pain", f"₹{st['max_pain']:,.0f}", None, f"{st['max_pain'] / underlying - 1:+.1%} vs spot"),
        ("Highest call OI", f"₹{st['call_wall']:,.0f}", theme.RED, "often read as resistance"),
        ("Highest put OI", f"₹{st['put_wall']:,.0f}", theme.GREEN, "often read as support"),
        ("ATM implied vol", "–" if st["atm_iv"] != st["atm_iv"] else f"{st['atm_iv']:.1%}", None,
         f"strike ₹{st['atm_strike']:,.0f}"),
    ]


CHAIN_FMT = {"Call OI": "{:,.0f}", "Call ΔOI": "{:+,.0f}", "Call volume": "{:,.0f}", "Call IV": "{:.1f}",
             "Call LTP": "{:,.2f}", "Put LTP": "{:,.2f}", "Put IV": "{:.1f}", "Put volume": "{:,.0f}",
             "Put ΔOI": "{:+,.0f}", "Put OI": "{:,.0f}"}


def _chain_table() -> FrameTable:
    t = FrameTable(fmt=CHAIN_FMT, signed={"Call ΔOI", "Put ΔOI"})
    t.model_.term_overrides = {c: "option_chain" for c in CHAIN_FMT}
    t.model_.term_overrides.update({"Call IV": "implied_vol", "Put IV": "implied_vol", "Call OI": "open_interest",
                                    "Put OI": "open_interest", "Call ΔOI": "open_interest", "Put ΔOI": "open_interest"})
    return t


class StockNSETab(QWidget):
    """Stock page → F&O & ownership."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = vbox(self, 0)
        inner = QWidget()
        outer.addWidget(scrolling(inner))
        lay = vbox(inner)
        self.status = muted(OFF)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addWidget(h2("Option chain", "option_chain"))
        self.expiry = QComboBox()
        self.expiry.setMinimumWidth(130)
        self.expiry.activated.connect(lambda _: self._load_chain(self.expiry.currentText()))
        row.addWidget(self.expiry)
        self.reload = QPushButton("Reload")
        self.reload.clicked.connect(lambda: (data.clear(f"nse:oc:{self.sym}"), self.load()))
        row.addWidget(self.reload)
        row.addStretch(1)
        row.addWidget(ExplainButton("option_chain"))
        lay.addLayout(row)
        self.kpi = KpiRow(cols=5)
        lay.addWidget(self.kpi)
        self.chain = _chain_table()
        self.chain.setMinimumHeight(420)
        lay.addWidget(self.chain)
        self.chain_note = muted("")
        lay.addWidget(self.chain_note)
        lay.addWidget(h2("Shareholding by quarter (NSE filings)", "shareholding"))
        self.sh = FrameTable(fmt={"Promoter %": "{:.2f}", "Public %": "{:.2f}", "Employee trusts %": "{:.2f}",
                                  "Promoter change (pp)": "{:+.2f}"}, signed={"Promoter change (pp)"})
        self.sh.model_.term_overrides = {c: "shareholding" for c in ("Promoter %", "Public %", "Employee trusts %",
                                                                      "Promoter change (pp)", "Filed")}
        self.sh.setMinimumHeight(220)
        lay.addWidget(self.sh)
        lay.addWidget(h2("Bulk and block deals in this stock (latest day)", "large_deals"))
        self.deals = FrameTable(fmt={"Quantity": "{:,.0f}", "Price ₹": "{:,.2f}", "Value ₹ Cr": "{:,.2f}"})
        self.deals.model_.term_overrides = {c: "large_deals" for c in ("Date", "Client", "Side", "Quantity", "Price ₹",
                                                                        "Value ₹ Cr", "Type")}
        self.deals.setMinimumHeight(120)
        lay.addWidget(self.deals)
        self.sym = None
        self.ticker = None
        self._loaded_for = None

    def set_stock(self, ticker: str) -> None:
        self.ticker, self.sym = ticker, ticker.replace(".NS", "")
        self._loaded_for = None
        for t in (self.chain, self.sh, self.deals):
            t.set_frame(pd.DataFrame())
        self.kpi.set_items([])
        self.expiry.clear()

    def load(self) -> None:
        """Called when the tab is shown (never on every stock open)."""
        if not data.nse_enabled():
            self.status.setText(OFF)
            return
        if self.sym is None or self._loaded_for == self.sym:
            return
        self._loaded_for = self.sym
        self.status.setText(f"Loading {self.sym} from NSE…")
        self._load_chain(None)
        sym = self.sym
        run_async(data.nse_shareholding, lambda d, s=sym: s == self.sym and self._got_sh(d), sym,
                  on_error=lambda e, s=sym: s == self.sym and self.sh.set_frame(
                      pd.DataFrame({"": [f"Shareholding unavailable: {e.splitlines()[-1][:120]}"]})))
        run_async(data.nse_deals, lambda d, s=sym: s == self.sym and self._got_deals(d),
                  on_error=lambda e: self.deals.set_frame(pd.DataFrame({"": ["Deals unavailable from NSE."]})))

    def _load_chain(self, expiry: str | None) -> None:
        sym = self.sym
        run_async(data.nse_chain, lambda r, s=sym: s == self.sym and self._got_chain(*r), sym, expiry,
                  on_error=lambda e, s=sym: s == self.sym and self._chain_error(e))

    def _chain_error(self, e: str) -> None:
        msg = e.splitlines()[-1][:160]
        self.status.setText(f"Option chain unavailable: {msg.split(': ', 1)[-1]}")
        self.chain.set_frame(pd.DataFrame({"": ["No option chain (not an F&O stock, or NSE did not answer)."]}))
        self.kpi.set_items([])

    def _got_chain(self, df: pd.DataFrame, meta: dict) -> None:
        if self.expiry.count() == 0:
            self.expiry.addItems(meta["expiries"])
        self.expiry.setCurrentText(meta["expiry"])
        st = N.chain_stats(df, meta["underlying"])
        self.kpi.ticker = self.ticker
        self.kpi.set_items(_chain_kpis(st, meta["underlying"]))
        self.chain.set_frame(_chain_view(df, meta["underlying"]))
        self.status.setText(f"NSE · {meta['timestamp']} · spot ₹{meta['underlying']:,.2f} · expiry {meta['expiry']}")
        self.chain_note.setText("Strikes around the money (▶ = nearest to spot). OI = open contracts; ΔOI = change today. "
                                "Positioning, not a forecast: option writers can be wrong and OI can be hedges.")

    def _got_sh(self, d: pd.DataFrame) -> None:
        v = d.copy()
        v.index = v.index.strftime("%b %Y")
        self.sh.set_frame(v.iloc[::-1])

    def _got_deals(self, d: dict) -> None:
        frames = []
        for kind in ("bulk", "block"):
            f = d.get(kind)
            if f is not None and len(f):
                f = f[f["Symbol"] == self.sym]
                if len(f):
                    frames.append(f.assign(Type=kind).drop(columns=["Symbol", "Name"]))
        self.deals.set_frame(pd.concat(frames).reset_index(drop=True) if frames else
                             pd.DataFrame({"": [f"No bulk or block deals in {self.sym} on {d.get('as_of')}."]}))


class MarketNSETab(QWidget):
    """Market → Flows & F&O."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = vbox(self, 0)
        inner = QWidget()
        outer.addWidget(scrolling(inner))
        lay = self.lay = vbox(inner)
        self.status = muted(OFF)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addWidget(h2("FII / DII cash-market flows (₹ crore)", "fii_dii"))
        row.addStretch(1)
        row.addWidget(ExplainButton("fii_dii"))
        lay.addLayout(row)
        self.f_kpi = KpiRow(cols=4)
        lay.addWidget(self.f_kpi)
        self.flows = FrameTable(fmt={"FII net": "{:+,.0f}", "DII net": "{:+,.0f}", "FII buy": "{:,.0f}",
                                     "FII sell": "{:,.0f}", "DII buy": "{:,.0f}", "DII sell": "{:,.0f}"},
                                signed={"FII net", "DII net"})
        self.flows.model_.term_overrides = {c: "fii_dii" for c in ("FII net", "DII net", "FII buy", "FII sell",
                                                                    "DII buy", "DII sell")}
        self.flows.setMinimumHeight(160)
        lay.addWidget(self.flows)
        lay.addWidget(h2("Index options positioning (nearest expiry)", "option_chain"))
        self.n_kpi = KpiRow(cols=5)
        self.b_kpi = KpiRow(cols=5)
        lay.addWidget(muted("NIFTY"))
        lay.addWidget(self.n_kpi)
        lay.addWidget(muted("BANKNIFTY"))
        lay.addWidget(self.b_kpi)
        lay.addWidget(h2("Bulk and block deals (latest day)", "large_deals"))
        self.deals = FrameTable(fmt={"Quantity": "{:,.0f}", "Price ₹": "{:,.2f}", "Value ₹ Cr": "{:,.2f}"})
        self.deals.model_.term_overrides = {c: "large_deals" for c in ("Date", "Symbol", "Name", "Client", "Side",
                                                                        "Quantity", "Price ₹", "Value ₹ Cr", "Type")}
        self.deals.setMinimumHeight(380)
        lay.addWidget(self.deals)
        self._loaded = False

    def load(self, force: bool = False) -> None:
        if not data.nse_enabled():
            self.status.setText(OFF)
            return
        if self._loaded and not force:
            return
        self._loaded = True
        self.status.setText("Loading from NSE…")
        run_async(data.nse_fii_dii, self._got_flows,
                  on_error=lambda e: self.status.setText(f"FII/DII unavailable: {e.splitlines()[-1][:140]}"))
        for sym, kpi in (("NIFTY", self.n_kpi), ("BANKNIFTY", self.b_kpi)):
            run_async(data.nse_chain, lambda r, k=kpi: k.set_items(_chain_kpis(N.chain_stats(r[0], r[1]["underlying"]),
                                                                              r[1]["underlying"])),
                      sym, None, on_error=lambda e, k=kpi: k.set_items([("Option chain", "unavailable", theme.MUTED)]))
        run_async(data.nse_deals, self._got_deals,
                  on_error=lambda e: self.deals.set_frame(pd.DataFrame({"": ["Deals unavailable from NSE."]})))

    def _got_flows(self, h: pd.DataFrame) -> None:
        if h.empty:
            return
        w = h.pivot_table(index="date", columns="category", values=["buy", "sell", "net"], aggfunc="last")
        w.columns = [f"{c} {v}" for v, c in w.columns]
        w = w.sort_index(ascending=False)
        last = w.iloc[0]
        d = w.index[0]
        self.f_kpi.set_items([
            ("FII net", f"₹{last.get('FII net', np.nan):+,.0f} Cr", theme.signed(last.get("FII net", 0)), f"{d:%d %b %Y}"),
            ("DII net", f"₹{last.get('DII net', np.nan):+,.0f} Cr", theme.signed(last.get("DII net", 0)), f"{d:%d %b %Y}"),
            ("FII net, days seen", f"₹{w['FII net'].sum():+,.0f} Cr" if "FII net" in w else "–", None,
             f"{len(w)} day(s) since you turned NSE data on"),
            ("DII net, days seen", f"₹{w['DII net'].sum():+,.0f} Cr" if "DII net" in w else "–", None, ""),
        ])
        view = w.copy()
        view.index = view.index.strftime("%d %b %Y")
        cols = [c for c in ("FII net", "DII net", "FII buy", "FII sell", "DII buy", "DII sell") if c in view]
        self.flows.set_frame(view[cols])
        self.status.setText("NSE public APIs. FII/DII history builds up one day at a time from when the switch is on "
                            "(NSE only serves the latest day).")

    def _got_deals(self, d: dict) -> None:
        frames = [d[k].assign(Type=k) for k in ("block", "bulk") if d.get(k) is not None and len(d[k])]
        df = pd.concat(frames).sort_values("Value ₹ Cr", ascending=False).reset_index(drop=True) if frames else pd.DataFrame()
        self.deals.set_frame(df if len(df) else pd.DataFrame({"": [f"No deals on {d.get('as_of')}."]}))
        self.deals.verticalHeader().setVisible(False)
