"""My desk → Portfolio & tax: risk and allocation of your holdings, XIRR, a realised P&L calendar and an estimate of
this financial year's capital-gains tax with harvesting ideas, from your Kite holdings and tradebook exports."""
from __future__ import annotations

from datetime import date

import pandas as pd
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QPushButton, QSplitter, QWidget
from PySide6.QtCore import Qt

from ... import portfolio as pf
from ...research import portfolio_analytics as PA
from .. import data, theme
from ..widgets import ExplainButton, FrameTable, KpiRow, h2, muted
from . import scrolling, vbox


class PortfolioTab(QWidget):
    def __init__(self, ctx, open_stock, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        outer = vbox(self, 0)
        inner = QWidget()
        outer.addWidget(scrolling(inner))
        lay = vbox(inner)
        row = QHBoxLayout()
        b_imp = QPushButton("Import Kite tradebook CSV…")
        b_imp.clicked.connect(self._import)
        row.addWidget(b_imp)
        self.msg = muted("Holdings come from the Holdings tab (Kite holdings CSV); XIRR and tax need your tradebook "
                         "(Kite Console → Reports → Tradebook → Equity, one file per year; imports are merged).")
        row.addWidget(self.msg, 1)
        lay.addLayout(row)
        hr = QHBoxLayout()
        hr.addWidget(h2("Risk and allocation", "portfolio_risk"))
        hr.addStretch(1)
        hr.addWidget(ExplainButton("portfolio_risk"))
        lay.addLayout(hr)
        self.risk_kpi = KpiRow(cols=6)
        lay.addWidget(self.risk_kpi)
        split = QSplitter(Qt.Horizontal)
        self.sectors = FrameTable(fmt={"Weight": "{:.1%}", "Value ₹": "{:,.0f}"})
        self.sectors.model_.term_overrides = {"Weight": "portfolio_risk", "Value ₹": "portfolio_risk"}
        self.weights = FrameTable(fmt={"Weight": "{:.1%}", "Share of risk": "{:.1%}", "Value ₹": "{:,.0f}",
                                       "Beta": "{:.2f}"})
        self.weights.model_.term_overrides = {"Weight": "portfolio_risk", "Share of risk": "risk_contrib",
                                              "Value ₹": "portfolio_risk"}
        self.weights.row_clicked.connect(lambda t: open_stock(str(t)))
        split.addWidget(self.sectors)
        split.addWidget(self.weights)
        split.setSizes([400, 700])
        split.setMinimumHeight(260)
        lay.addWidget(split)
        tr = QHBoxLayout()
        tr.addWidget(h2("Returns and capital-gains tax", "cg_tax"))
        self.fy = QComboBox()
        self.fy.currentIndexChanged.connect(lambda _: self._tax())
        tr.addWidget(self.fy)
        tr.addStretch(1)
        tr.addWidget(ExplainButton("cg_tax"))
        lay.addLayout(tr)
        self.tax_kpi = KpiRow(cols=6)
        lay.addWidget(self.tax_kpi)
        self.tax_note = muted("")
        lay.addWidget(self.tax_note)
        lay.addWidget(h2("Tax ideas for this year (not advice)", "tax_harvest"))
        self.ideas = FrameTable(fmt={"Unrealised ₹": "{:+,.0f}", "Est. tax effect ₹": "{:,.0f}"}, signed={"Unrealised ₹"})
        self.ideas.model_.term_overrides = {"Idea": "tax_harvest", "Unrealised ₹": "tax_harvest",
                                            "Est. tax effect ₹": "tax_harvest", "Bought": "tax_harvest"}
        self.ideas.setMinimumHeight(150)
        lay.addWidget(self.ideas)
        lay.addWidget(h2("Realised P&L by month (₹, FIFO, before costs)", "pnl_calendar"))
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Full year"]
        self.cal = FrameTable(fmt={m: "{:+,.0f}" for m in months}, signed=set(months))
        self.cal.model_.term_overrides = {m: "pnl_calendar" for m in months}
        self.cal.setMinimumHeight(140)
        lay.addWidget(self.cal)
        lay.addWidget(h2("Realised sales matched to buys (FIFO)", "cg_tax"))
        self.real = FrameTable(fmt={"Qty": "{:,.0f}", "Buy ₹": "{:,.2f}", "Sell ₹": "{:,.2f}", "Gain ₹": "{:+,.0f}",
                                    "Days held": "{:,.0f}"}, signed={"Gain ₹"})
        self.real.model_.term_overrides = {c: "cg_tax" for c in ("Bought", "Sold", "Buy ₹", "Sell ₹", "Gain ₹",
                                                                  "Days held", "Term", "FY")}
        self.real.setMinimumHeight(240)
        lay.addWidget(self.real)
        self._fifo = None

    # ------------------------------------------------------------------ data
    def _prices(self, tickers) -> dict[str, float]:
        p = data.panel()
        snap = self.ctx.feed.snapshot() if self.ctx.feed else {}
        out = {}
        for t in tickers:
            if t in snap and snap[t].get("price"):
                out[t] = snap[t]["price"]
            elif t in p.close:
                s = p.close[t].dropna()
                if len(s):
                    out[t] = float(s.iloc[-1])
        return out

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Kite tradebook CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                df = pf.import_tradebook(f.read())
            self.msg.setText(f"Tradebook now holds {len(df):,} trades from {df['date'].min():%d %b %Y} to "
                             f"{df['date'].max():%d %b %Y}.")
        except Exception as e:  # noqa: BLE001 - show the reason, keep the old data
            self.msg.setText(f"Could not read that file: {e}")
        self.refresh()

    def refresh(self) -> None:
        trades = pf.load_tradebook()
        self._fifo = PA.fifo(trades) if len(trades) else None
        hs = pf.load_holdings()
        qty = {t: h.qty for t, h in hs.items()}
        src = "Kite holdings"
        if not qty and self._fifo is not None and len(self._fifo.open_lots):
            qty = self._fifo.open_lots.groupby("ticker")["qty"].sum().to_dict()
            src = "open lots in your tradebook"
        self._risk(qty, src)
        years = sorted({PA.fy_of(d) for d in self._fifo.realised["sell_date"]}, reverse=True) \
            if self._fifo is not None and len(self._fifo.realised) else []
        cur = PA.fy_of(date.today())
        years = sorted(set(years) | {cur}, reverse=True)
        keep = self.fy.currentData()
        self.fy.blockSignals(True)
        self.fy.clear()
        for y in years:
            self.fy.addItem(PA.fy_label(y), y)
        self.fy.setCurrentIndex(max(0, years.index(keep)) if keep in years else 0)
        self.fy.blockSignals(False)
        self._tax()

    def _risk(self, qty: dict, src: str) -> None:
        p = data.panel()
        if not qty:
            self.risk_kpi.set_items([])
            self.sectors.set_frame(pd.DataFrame({"": ["No holdings: import your Kite holdings (Holdings tab) or "
                                                      "tradebook."]}))
            self.weights.set_frame(pd.DataFrame())
            return
        prices = self._prices(qty)
        rk = PA.risk(p, qty, prices)
        if not rk:
            self.risk_kpi.set_items([])
            return
        missing = [t for t in qty if t not in p.close.columns]
        wd = rk["worst_day"]
        self.risk_kpi.set_items([
            ("Portfolio value", f"₹{rk['value']:,.0f}", None, f"from {src}" + (f"; {len(missing)} not in NIFTY 200"
                                                                               if missing else "")),
            ("Portfolio beta", f"{rk['beta']:.2f}", None, "vs NIFTY, 1 year daily"),
            ("Portfolio volatility", f"{rk['vol']:.1%}", None, "annualised, 1 year"),
            ("1-day VaR (95%)", f"₹{rk['var95']:,.0f}", theme.RED, f"{rk['var95'] / rk['value']:.1%} of value · "
                                                                   "historical"),
            ("Largest holding", f"{rk['top_weight']:.0%}", theme.AMBER if rk["top_weight"] > 0.25 else None,
             rk["weights"].index[0].replace(".NS", "")),
            ("Effective holdings", f"{rk['effective_n']:.1f}", None,
             f"of {len(rk['weights'])} · worst day {wd[1]:+.1%}" if wd else ""),
        ])
        sec = pd.DataFrame({"Weight": rk["sectors"], "Value ₹": rk["sectors"] * rk["value"]}).rename_axis("Sector")
        self.sectors.set_frame(sec)
        r = p.close[list(rk["weights"].index)].pct_change(fill_method=None).iloc[-252:]
        b = p.bench.pct_change().reindex(r.index)
        betas = {t: float(r[t].cov(b) / b.var()) if b.var() > 0 else float("nan") for t in r.columns}
        self.weights.set_frame(pd.DataFrame({"Weight": rk["weights"], "Share of risk": rk["risk_contrib"],
                                             "Value ₹": rk["weights"] * rk["value"],
                                             "Beta": pd.Series(betas)}).sort_values("Weight", ascending=False))

    def _tax(self) -> None:
        f = self._fifo
        fy = self.fy.currentData()
        if f is None or fy is None:
            self.tax_kpi.set_items([])
            self.tax_note.setText("Import your Kite tradebook to see XIRR, realised P&L and the tax estimate.")
            for t in (self.ideas, self.cal, self.real):
                t.set_frame(pd.DataFrame())
            return
        s = PA.tax_summary(f.realised, fy)
        trades = pf.load_tradebook()
        prices = self._prices(set(f.open_lots["ticker"])) if len(f.open_lots) else {}
        value_now = float(sum(q * prices.get(t, p) for t, q, p in
                              f.open_lots[["ticker", "qty", "price"]].itertuples(index=False))) if len(f.open_lots) else 0.0
        x = PA.xirr(PA.portfolio_flows(trades, value_now))
        self.tax_kpi.set_items([
            ("XIRR", "–" if x != x else f"{x:+.1%}", theme.signed(x) if x == x else None,
             "all trades + today's value, before costs"),
            ("Realised gains", f"₹{s['net_st'] + s['net_lt']:+,.0f}" if s["n_sales"] else "₹0",
             theme.signed(s["st_gain"] - s["st_loss"] + s["lt_gain"] - s["lt_loss"]), f"{PA.fy_label(fy)}, {s['n_sales']} sales"),
            ("Taxable STCG", f"₹{s['taxable_st']:,.0f}", None, f"gains ₹{s['st_gain']:,.0f} − losses ₹{s['st_loss']:,.0f}"),
            ("Taxable LTCG", f"₹{s['taxable_lt']:,.0f}", None, f"after ₹{s['exempt_used']:,.0f} exempt"),
            ("Estimated tax", f"₹{s['tax']:,.0f}", theme.AMBER if s["tax"] > 0 else None, "20% / 12.5% + 4% cess"),
            ("LTCG allowance left", f"₹{s['exempt_left']:,.0f}", None, "of ₹1,25,000 this year"),
        ])
        notes = []
        if f.unmatched:
            notes.append(f"{len(f.unmatched)} sale(s) have no earlier buy in the tradebook (bought before it starts): "
                         "they are left out of gains and XIRR; import older years to include them.")
        if s["old_rates_apply"]:
            notes.append("Some sales are before 23 Jul 2024, when rates were 15% / 10% with a ₹1 lakh exemption; "
                         "this estimate uses the new rates for all of them.")
        if s["carry_forward_loss"] > 0:
            notes.append(f"₹{s['carry_forward_loss']:,.0f} of losses can be carried forward (up to 8 years, if you "
                         "file your return on time).")
        notes.append("Estimate only: FIFO, before brokerage/STT, no surcharge or grandfathering. Not tax advice.")
        self.tax_note.setText(" ".join(notes))
        un = PA.unrealised(f.open_lots, prices)
        h = PA.harvest(un, s) if fy == PA.fy_of(date.today()) else pd.DataFrame()
        if len(h):
            h = h.copy()
            h["Bought"] = pd.to_datetime(h["Bought"]).dt.strftime("%d %b %Y")
        self.ideas.set_frame(h if len(h) else pd.DataFrame({"": ["Nothing to suggest for this year."]}))
        self.cal.set_frame(PA.monthly_pnl(f.realised))
        r = f.realised
        if len(r):
            v = pd.DataFrame({"Stock": r["ticker"].str.replace(".NS", "", regex=False),
                              "Bought": r["buy_date"].dt.strftime("%d %b %Y"), "Sold": r["sell_date"].dt.strftime("%d %b %Y"),
                              "Qty": r["qty"], "Buy ₹": r["buy_price"], "Sell ₹": r["sell_price"], "Gain ₹": r["gain"],
                              "Days held": r["days"], "Term": r["term"],
                              "FY": [PA.fy_label(PA.fy_of(d)) for d in r["sell_date"]]}).iloc[::-1]
            self.real.set_frame(v.reset_index(drop=True))
            self.real.verticalHeader().setVisible(False)
        else:
            self.real.set_frame(pd.DataFrame({"": ["No sales in the tradebook."]}))
