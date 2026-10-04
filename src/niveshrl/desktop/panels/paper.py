"""My desk → Paper trading: a manual order ticket (market/limit, optional bracket stop and target, prefilled from the
swing plan), open orders, positions marked to live prices and closed paper trades. Paper only: no order ever leaves
this computer."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (QComboBox, QCompleter, QDoubleSpinBox, QHBoxLayout, QLineEdit, QPushButton, QSpinBox,
                               QWidget)

from ... import paper as PP
from .. import data, theme
from ..widgets import FrameTable, KpiRow, h2, muted
from . import vbox


class PaperTab(QWidget):
    def __init__(self, ctx, open_stock, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        lay = vbox(self)
        lay.addWidget(muted("Practise the plan without money: orders fill against the live stream (market orders at the "
                            "next live price plus half the spread; limits when the price trades through). Stops and "
                            "targets close the whole position; delivery costs are charged like the trade plans. Orders "
                            "placed while the market is closed wait for the next live price. Nothing is sent to a broker."))
        self.kpi = KpiRow(cols=6)
        lay.addWidget(self.kpi)
        lay.addWidget(h2("Order ticket", "paper_trading"))
        row = QHBoxLayout()
        self.f_ticker = QLineEdit()
        self.f_ticker.setPlaceholderText("Symbol")
        self.f_ticker.setMaximumWidth(130)
        self.f_side = QComboBox()
        self.f_side.addItems(["BUY", "SELL"])
        self.f_type = QComboBox()
        self.f_type.addItems(["MARKET", "LIMIT"])
        self.f_qty = QSpinBox()
        self.f_qty.setRange(1, 1_000_000)
        self.f_qty.setPrefix("qty ")
        self.f_limit = QDoubleSpinBox()
        self.f_limit.setRange(0, 10_000_000)
        self.f_limit.setPrefix("limit ₹")
        self.f_stop = QDoubleSpinBox()
        self.f_stop.setRange(0, 10_000_000)
        self.f_stop.setPrefix("stop ₹")
        self.f_target = QDoubleSpinBox()
        self.f_target.setRange(0, 10_000_000)
        self.f_target.setPrefix("target ₹")
        self.b_plan = QPushButton("Fill from trade plan")
        self.b_plan.clicked.connect(self._from_plan)
        self.b_place = QPushButton("Place paper order")
        self.b_place.setStyleSheet(f"QPushButton{{color:{theme.AMBER};border-color:{theme.AMBER};}}")
        self.b_place.clicked.connect(self._place)
        self.f_type.currentTextChanged.connect(lambda t: self.f_limit.setEnabled(t == "LIMIT"))
        self.f_limit.setEnabled(False)
        for w in (self.f_ticker, self.f_side, self.f_type, self.f_qty, self.f_limit, self.f_stop, self.f_target,
                  self.b_plan, self.b_place):
            row.addWidget(w)
        lay.addLayout(row)
        self.msg = muted("Stop and target are optional (0 = none) and apply to buys.")
        lay.addWidget(self.msg)
        lay.addWidget(h2("Open orders", "paper_trading"))
        self.orders = FrameTable(fmt={"Qty": "{:,.0f}", "Limit": "₹{:,.2f}", "Stop": "₹{:,.2f}", "Target": "₹{:,.2f}"})
        self.orders.model_.term_overrides = {c: "paper_trading" for c in ("Side", "Type", "Qty", "Limit", "Stop", "Target",
                                                                          "Placed", "Status")}
        self.orders.setMaximumHeight(150)
        lay.addWidget(self.orders)
        brow = QHBoxLayout()
        self.b_cancel = QPushButton("Cancel selected order")
        self.b_cancel.clicked.connect(self._cancel)
        self.b_close = QPushButton("Sell selected position at market")
        self.b_close.clicked.connect(self._close_pos)
        self.b_reset = QPushButton("Reset paper account to desk capital")
        self.b_reset.clicked.connect(self._reset)
        for w in (self.b_cancel, self.b_close):
            brow.addWidget(w)
        brow.addStretch(1)
        brow.addWidget(self.b_reset)
        lay.addLayout(brow)
        lay.addWidget(h2("Positions", "paper_trading"))
        self.pos = FrameTable(fmt={"Qty": "{:,.0f}", "Avg": "₹{:,.2f}", "Price": "₹{:,.2f}", "Stop": "₹{:,.2f}",
                                   "Target": "₹{:,.2f}", "P&L ₹": "{:+,.0f}", "P&L %": "{:+.2%}", "Open R": "{:+.2f}"},
                              signed={"P&L ₹", "P&L %", "Open R"})
        self.pos.model_.term_overrides = {"Avg": "paper_trading", "Stop": "paper_trading", "Target": "paper_trading",
                                          "P&L ₹": "paper_trading", "P&L %": "paper_trading", "Open R": "r_multiple"}
        self.pos.row_clicked.connect(lambda t: open_stock(str(t)))
        self.pos.setMaximumHeight(170)
        lay.addWidget(self.pos)
        lay.addWidget(h2("Closed paper trades", "paper_trading"))
        self.closed = FrameTable(fmt={"Qty": "{:,.0f}", "Entry": "₹{:,.2f}", "Exit": "₹{:,.2f}", "Costs ₹": "{:,.0f}",
                                      "Net ₹": "{:+,.0f}", "R": "{:+.2f}"}, signed={"Net ₹", "R"})
        self.closed.model_.term_overrides = {"Entry": "paper_trading", "Exit": "paper_trading", "Opened": "paper_trading",
                                             "Closed": "paper_trading", "Reason": "paper_trading",
                                             "Costs ₹": "scenario_pnl", "Net ₹": "scenario_pnl", "R": "r_multiple"}
        lay.addWidget(self.closed, 1)

    # ------------------------------------------------------------------ helpers
    def _acct(self) -> PP.Account:
        from .desk import desk_settings
        return PP.load(start_cash=desk_settings()["capital"])

    def _quotes(self) -> dict[str, float]:
        snap = self.ctx.feed.snapshot() if self.ctx.feed else {}
        return {k: v["price"] for k, v in snap.items() if v.get("price")}

    def _marks(self) -> dict[str, float]:
        """Live prices where streaming, else the last close (for display only; fills need live prices)."""
        p = data.panel()
        acct = self._acct()
        q = self._quotes()
        out = {}
        for t in acct.positions:
            if t in q:
                out[t] = q[t]
            elif t in p.close:
                out[t] = float(p.close[t].dropna().iloc[-1])
        return out

    def _ticker(self) -> str | None:
        t = self.f_ticker.text().strip().upper()
        if not t:
            return None
        t = t if t.endswith(".NS") else t + ".NS"
        return t if t in data.panel().close.columns else None

    def prefill(self, ticker: str) -> None:
        self.f_ticker.setText(ticker.replace(".NS", ""))
        self._from_plan()

    # ------------------------------------------------------------------ actions
    def _from_plan(self) -> None:
        from ...research.plans import make_plan
        from .desk import current_risk_state, desk_settings
        t = self._ticker()
        if t is None:
            self.msg.setText("Type a NIFTY 200 symbol first.")
            return
        st, rs = desk_settings(), current_risk_state()
        pl = make_plan(data.panel(), t, st["capital"], st["risk_pct"], rs.multiplier)
        if pl is None or pl.qty <= 0:
            self.msg.setText("No plan for this stock (not enough history, or capital too small for one share).")
            return
        self.f_side.setCurrentText("BUY")
        self.f_type.setCurrentText("MARKET")
        self.f_qty.setValue(int(pl.qty))
        self.f_stop.setValue(round(pl.stop, 2))
        self.f_target.setValue(round(pl.t2, 2))
        self.msg.setText(f"From the swing plan: {pl.qty} shares, stop ₹{pl.stop:,.2f}, target T2 ₹{pl.t2:,.2f} "
                         f"(₹{pl.rupee_risk:,.0f} at risk). Edit anything before placing.")

    def _place(self) -> None:
        t = self._ticker()
        if t is None:
            self.msg.setText("Unknown symbol: paper trading works for the NIFTY 200 stocks the app streams.")
            return
        acct = self._acct()
        o = PP.Order(t, self.f_side.currentText(), int(self.f_qty.value()), self.f_type.currentText(),
                     limit=self.f_limit.value() or None if self.f_type.currentText() == "LIMIT" else None,
                     stop=self.f_stop.value() or None if self.f_side.currentText() == "BUY" else None,
                     target=self.f_target.value() or None if self.f_side.currentText() == "BUY" else None)
        PP.place(acct, o)
        events = PP.process(acct, self._quotes())        # fills at once when the stock is streaming
        PP.save(acct)
        if o.status == "REJECTED":
            self.msg.setText(f"Rejected: {o.note}.")
        elif o.status == "FILLED":
            self.msg.setText(events[0] if events else "Filled.")
        else:
            self.msg.setText(f"Placed {o.side} {o.qty} {t.replace('.NS', '')}: waiting for a live price "
                             + ("at or below the limit." if o.type == "LIMIT" and o.side == "BUY" else
                                "at or above the limit." if o.type == "LIMIT" else "(market closed or not streaming)."))
        self.refresh()

    def _selected(self, table: FrameTable) -> str | None:
        idx = table.currentIndex()
        if not idx.isValid():
            return None
        return table.model_.labels[table.proxy.mapToSource(idx).row()]

    def _cancel(self) -> None:
        oid = self._selected(self.orders)
        if oid:
            acct = self._acct()
            PP.cancel(acct, oid)
            PP.save(acct)
            self.refresh()

    def _close_pos(self) -> None:
        t = self._selected(self.pos)
        if not t:
            return
        acct = self._acct()
        pos = acct.positions.get(t)
        if pos is None:
            return
        PP.place(acct, PP.Order(t, "SELL", pos.qty, note="closed from the positions table"))
        PP.process(acct, self._quotes())
        PP.save(acct)
        self.msg.setText(f"Sell {pos.qty} {t.replace('.NS', '')} at market "
                         + ("filled." if t not in acct.positions else "queued until a live price arrives."))
        self.refresh()

    def _reset(self) -> None:
        from .desk import desk_settings
        PP.reset(desk_settings()["capital"])
        self.msg.setText("Paper account reset to your desk capital.")
        self.refresh()

    # ------------------------------------------------------------------ view
    def refresh(self) -> None:
        if self.f_ticker.completer() is None:
            self.f_ticker.setCompleter(QCompleter(sorted(t.replace(".NS", "") for t in data.panel().tickers), self))
        acct = self._acct()
        marks = self._marks()
        s = PP.summary(acct, marks)
        self.kpi.set_items([
            ("Paper equity", f"₹{s['equity']:,.0f}", theme.signed(s["return"]), f"{s['return']:+.2%} since start"),
            ("Paper cash", f"₹{s['cash']:,.0f}", None, f"start ₹{acct.start_cash:,.0f}"),
            ("Unrealised P&L", f"₹{s['unrealised']:+,.0f}", theme.signed(s["unrealised"]), "before exit costs"),
            ("Realised P&L", f"₹{s['realised']:+,.0f}", theme.signed(s["realised"]), "after costs"),
            ("Closed paper trades", str(s["trades"]), None,
             "" if s["trades"] == 0 else f"{s['win_rate']:.0%} winners"),
            ("Average R", "–" if s["avg_r"] != s["avg_r"] else f"{s['avg_r']:+.2f}R", None, "trades with a stop"),
        ])
        op = [o for o in acct.orders if o.status == "OPEN"]
        self.orders.set_frame(pd.DataFrame([{"id": o.id, "Stock": o.ticker.replace(".NS", ""), "Side": o.side,
                                             "Type": o.type, "Qty": o.qty, "Limit": o.limit, "Stop": o.stop,
                                             "Target": o.target, "Placed": o.placed_at.replace("T", " ")} for o in op]
                                           ).set_index("id") if op else pd.DataFrame({"": ["No open orders."]}))
        self.orders.verticalHeader().setVisible(False)
        rows = []
        for t, p in acct.positions.items():
            px = marks.get(t, p.avg)
            rows.append({"ticker": t, "Qty": p.qty, "Avg": p.avg, "Price": px, "Stop": p.stop, "Target": p.target,
                         "P&L ₹": (px - p.avg) * p.qty, "P&L %": px / p.avg - 1,
                         "Open R": (px - p.avg) / p.risk_per_share if p.risk_per_share else None,
                         "Opened": p.opened_at.replace("T", " ")})
        self.pos.set_frame(pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame({"": ["No paper positions."]}))
        cl = list(reversed(acct.closed[-200:]))
        self.closed.set_frame(pd.DataFrame([{"Stock": c.ticker.replace(".NS", ""), "Qty": c.qty, "Entry": c.entry,
                                             "Exit": c.exit, "Reason": c.reason, "Costs ₹": c.costs, "Net ₹": c.net,
                                             "R": c.r, "Opened": c.opened_at.replace("T", " "),
                                             "Closed": c.closed_at.replace("T", " ")} for c in cl])
                              if cl else pd.DataFrame({"": ["No closed paper trades yet."]}))
        self.closed.verticalHeader().setVisible(False)

    def process(self) -> list[str]:
        """Called by the main window every 15 s: fill orders and trigger brackets against live prices."""
        acct = self._acct()
        if not any(o.status == "OPEN" for o in acct.orders) and not acct.positions:
            return []
        events = PP.process(acct, self._quotes())
        if events:
            PP.save(acct)
            self.refresh()
        return events
