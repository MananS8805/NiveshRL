"""DESK: capital planner, your holdings (Kite CSV) with exit lines, and your trade journal."""
from __future__ import annotations

from datetime import date

import pandas as pd
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QSpinBox,
                               QTabWidget, QWidget)

from ... import portfolio as pf
from ... import watchlist as wl
from ...research import plans as P
from .. import data, theme
from ..widgets import FrameTable, KpiRow, h2, muted
from . import Panel, vbox

_S = lambda: QSettings("NiveshRL", "NiveshRL")  # noqa: E731


def desk_settings() -> dict:
    s = _S()
    return {"capital": s.value("capital", 100_000.0, type=float), "risk_pct": s.value("risk_pct", 1.0, type=float) / 100,
            "max_positions": s.value("max_positions", 5, type=int)}


def current_risk_state():
    reg = data.regimes_weekly()
    r = reg["regime"].iloc[-1] if reg is not None else None
    return P.risk_state(data.panel(), r)


class DeskPanel(Panel):
    title = "My desk"
    code = "DESK"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        top = QHBoxLayout()
        st = desk_settings()
        self.capital = QDoubleSpinBox()
        self.capital.setRange(10_000, 100_000_000)
        self.capital.setSingleStep(10_000)
        self.capital.setDecimals(0)
        self.capital.setPrefix("Capital ₹ ")
        self.capital.setValue(st["capital"])
        self.risk = QDoubleSpinBox()
        self.risk.setRange(0.1, 3.0)
        self.risk.setSingleStep(0.25)
        self.risk.setSuffix(" % risk per trade")
        self.risk.setValue(st["risk_pct"] * 100)
        self.maxpos = QSpinBox()
        self.maxpos.setRange(1, 20)
        self.maxpos.setPrefix("max positions ")
        self.maxpos.setValue(st["max_positions"])
        for w in (self.capital, self.risk, self.maxpos):
            top.addWidget(w)
            (w.valueChanged).connect(self._save_settings)
        top.addStretch(1)
        lay.addLayout(top)
        self.state = KpiRow(cols=4)
        lay.addWidget(self.state)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        # planner
        pw = QWidget()
        pl = vbox(pw)
        row = QHBoxLayout()
        self.source = QComboBox()
        for k, v in [("monitor", "Tomorrow's 'watch for strength' list"), ("watch", "My watchlist (★ first)"),
                     ("both", "Both")]:
            self.source.addItem(v, k)
        self.source.currentIndexChanged.connect(self.refresh)
        row.addWidget(h2("Capital planner"))
        row.addWidget(self.source)
        row.addStretch(1)
        pl.addLayout(row)
        self.totals = KpiRow(cols=5)
        pl.addWidget(self.totals)
        self.orders = FrameTable(fmt={"Entry": "₹{:,.2f}", "Stop": "₹{:,.2f}", "Stop %": "{:.1%}", "T1": "₹{:,.2f}",
                                      "T2": "₹{:,.2f}", "Qty": "{:,.0f}", "Amount": "₹{:,.0f}", "₹ risk": "₹{:,.0f}",
                                      "₹ at T1": "₹{:,.0f}", "₹ at T2": "₹{:,.0f}"})
        self.orders.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        pl.addWidget(self.orders, 1)
        pl.addWidget(muted("Entry = last close (you'd buy at the next open). Stop under the 10-day low, kept between 2× and "
                           "2.5× ATR and at most 8% away; T1 = 1.5R, T2 = 2.5R; size = capital × risk% × risk-state "
                           "multiplier ÷ R, max 20% of capital per stock, at most 2 per sector. Suggested: book a third "
                           "at T1 and move the stop to entry. A plan, not advice: nothing is ordered."))
        self.tabs.addTab(pw, "Capital planner")

        # holdings
        hw = QWidget()
        hl = vbox(hw)
        hrow = QHBoxLayout()
        imp = QPushButton("⬆ Import Kite holdings CSV")
        imp.clicked.connect(self._import)
        hrow.addWidget(imp)
        self.hmsg = muted("Kite → Portfolio → Holdings → Download. Stored only on this PC (data/holdings.json).")
        hrow.addWidget(self.hmsg, 1)
        hl.addLayout(hrow)
        self.hsum = KpiRow(cols=4)
        hl.addWidget(self.hsum)
        self.holdings = FrameTable(fmt={"Qty": "{:,.0f}", "Avg cost": "₹{:,.2f}", "Last": "₹{:,.2f}", "P&L": "₹{:+,.0f}",
                                        "P&L %": "{:+.1%}", "Exit line": "₹{:,.2f}", "To exit line": "{:+.1%}"},
                                   signed={"P&L", "P&L %"})
        self.holdings.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        hl.addWidget(self.holdings, 1)
        hl.addWidget(muted("Exit line = close − 3 × ATR, only ever raised. EXIT when the price is at or below it; REVIEW "
                           "when the stock is on tomorrow's 'watch for weakness' list; NO DATA for stocks outside the "
                           "NIFTY 200 panel."))
        self.tabs.addTab(hw, "Holdings")

        # journal
        jw = QWidget()
        jl = vbox(jw)
        form = QHBoxLayout()
        self.j_sym = QLineEdit()
        self.j_sym.setPlaceholderText("Symbol, e.g. TCS")
        self.j_date = QLineEdit(date.today().isoformat())
        self.j_entry, self.j_stop, self.j_exit = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for sp, lab in [(self.j_entry, "Entry ₹ "), (self.j_stop, "Stop ₹ "), (self.j_exit, "Exit ₹ ")]:
            sp.setRange(0, 1_000_000)
            sp.setDecimals(2)
            sp.setPrefix(lab)
        self.j_qty = QSpinBox()
        self.j_qty.setRange(1, 10_000_000)
        self.j_qty.setPrefix("Qty ")
        self.j_note = QLineEdit()
        self.j_note.setPlaceholderText("Note")
        add = QPushButton("＋ Log trade")
        add.setObjectName("primary")
        add.clicked.connect(self._add_trade)
        for w in (self.j_sym, self.j_date, self.j_entry, self.j_stop, self.j_qty, self.j_note, add):
            form.addWidget(w)
        jl.addLayout(form)
        crow = QHBoxLayout()
        crow.addWidget(self.j_exit)
        close = QPushButton("Close selected trade at exit price")
        close.clicked.connect(self._close_trade)
        dele = QPushButton("Delete selected")
        dele.clicked.connect(self._delete_trade)
        crow.addWidget(close)
        crow.addWidget(dele)
        crow.addStretch(1)
        jl.addLayout(crow)
        self.jstats = KpiRow(cols=5)
        jl.addWidget(self.jstats)
        self.journal = FrameTable(fmt={"Entry": "₹{:,.2f}", "Stop": "₹{:,.2f}", "Exit": "₹{:,.2f}", "Qty": "{:,.0f}",
                                       "R (after costs)": "{:+.2f}"}, signed={"R (after costs)"})
        self.journal.model_.term_overrides = {"R (after costs)": "r_multiple"}
        self.journal.row_clicked.connect(self._pick_trade)
        jl.addWidget(self.journal, 1)
        jl.addWidget(muted("R = (exit − entry) ÷ (entry − stop) after delivery costs, exactly like the backtests. Your journal "
                           "never feeds any model or screen."))
        self.tabs.addTab(jw, "Journal")
        self._sel_trade = None

    def _save_settings(self) -> None:
        s = _S()
        s.setValue("capital", self.capital.value())
        s.setValue("risk_pct", self.risk.value())
        s.setValue("max_positions", self.maxpos.value())
        self.refresh()
        self.ctx.watchlist_changed.emit()                 # Today/Stock plans resize with the new capital

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        rs = current_risk_state()
        st = desk_settings()
        self.state.set_items([
            ("Risk state", rs.state, theme.GREEN if rs.multiplier == 1 else theme.AMBER, "; ".join(rs.reasons)[:90]),
            ("Risk multiplier", f"×{rs.multiplier:g}", None, "applied to position size only"),
            ("Risk per trade", f"₹{st['capital'] * st['risk_pct'] * rs.multiplier:,.0f}", None,
             f"{st['risk_pct']:.2%} of ₹{st['capital']:,.0f}"),
            ("Max positions", str(st["max_positions"]), None, "2 per sector at most"),
        ])
        self._planner(rs, st)
        self._holdings()
        self._journal()

    def _candidates(self) -> list[str]:
        src = self.source.currentData()
        out = []
        if src in ("monitor", "both"):
            mon = data.dload("monitor")
            if mon is not None:
                out += list(mon[mon["list"] == "watch for strength"].index)
        if src in ("watch", "both"):
            items = wl.load()
            out += [t for t, it in sorted(items.items(), key=lambda kv: kv[1].tier != "must")]
        return list(dict.fromkeys(out))

    @staticmethod
    def _verdict(p, t, st, rs) -> str:
        """The stock page's pre-entry checklist verdict (Go / Wait / No-go) with its cautions, for the planner."""
        from ...research import tradecheck as TC
        try:
            pl = P.make_plan(p, t, st["capital"], st["risk_pct"], rs.multiplier)
            hist = data.tagged_history()
            sim = TC.similar_setups(hist, TC.tags_now(p, t)) if hist is not None else None
            table = data.screener_table()
            row = table.loc[t] if table is not None and t in table.index else None
            verdict, checks = TC.checklist(pl, rs, row, sim, row.get("days_to_earnings") if row is not None else None)
        except (KeyError, IndexError, ValueError):
            return "–"
        flags = [c.item.lower() for c in checks if c.status != "ok"]
        return verdict + (f" ({', '.join(flags)})" if flags else "")

    def _planner(self, rs, st) -> None:
        p = data.panel()
        cands = [t for t in self._candidates() if t in p.close.columns]
        if not cands:
            self.orders.set_frame(pd.DataFrame({"": ["No candidates: run the daily refresh (F5) or add stocks to your "
                                                     "watchlist."]}))
            self.totals.set_items([])
            return
        plans = P.plans_for(p, cands, st["capital"], st["risk_pct"], rs.multiplier)
        rows, per_sector, used = [], {}, 0.0
        for t, r in plans.iterrows():
            sec = p.sectors.get(t, "")
            if per_sector.get(sec, 0) >= 2 or len(rows) >= st["max_positions"] or r["qty"] <= 0:
                continue
            if used + r["position_value"] > st["capital"]:
                continue
            per_sector[sec] = per_sector.get(sec, 0) + 1
            used += r["position_value"]
            rows.append({"ticker": t, "Sector": sec, "Entry": r["entry"], "Stop": r["stop"], "Stop %": r["stop_pct"],
                         "T1": r["t1"], "T2": r["t2"], "Qty": r["qty"], "Amount": r["position_value"],
                         "₹ risk": r["rupee_risk"], "₹ at T1": r["qty"] * (r["t1"] - r["entry"]),
                         "₹ at T2": r["qty"] * (r["t2"] - r["entry"]),
                         "Checklist": self._verdict(p, t, st, rs), "Notes": "; ".join(r["notes"])})
        df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame({"": ["Nothing fits the limits today."]})
        self.orders.set_frame(df)
        if rows:
            tot = pd.DataFrame(rows)
            self.totals.set_items([
                ("Positions", str(len(rows)), None), ("Invested", f"₹{tot['Amount'].sum():,.0f}", None,
                                                       f"{tot['Amount'].sum() / st['capital']:.0%} of capital"),
                ("Total risk", f"₹{tot['₹ risk'].sum():,.0f}", theme.RED, "if every stop is hit"),
                ("At T1", f"₹{tot['₹ at T1'].sum():+,.0f}", theme.GREEN), ("At T2", f"₹{tot['₹ at T2'].sum():+,.0f}", theme.GREEN)])

    # ------------------------------------------------------------------ holdings
    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Kite holdings CSV", "", "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                hs = pf.import_holdings(f.read())
            self.hmsg.setText(f"Imported {len(hs)} holdings.")
        except Exception as e:
            self.hmsg.setText(f"Could not read that file: {e}")
        self._holdings()

    def _holdings(self) -> None:
        hs = pf.load_holdings()
        if not hs:
            self.holdings.set_frame(pd.DataFrame({"": ["No holdings yet. Import your Kite holdings CSV."]}))
            self.hsum.set_items([])
            return
        from ...research.technicals import atr
        p = data.panel()
        a = data._cached("atr14", 600, lambda: atr(p))
        snap = self.ctx.feed.snapshot() if self.ctx.feed else {}
        mon = data.dload("monitor")
        weak = set(mon[mon["list"] == "watch for weakness"].index) if mon is not None else set()
        rows, changed = [], False
        for t, h in hs.items():
            if t not in p.close.columns:
                rows.append({"ticker": t, "Qty": h.qty, "Avg cost": h.avg_cost, "Call": "NO DATA"})
                continue
            last = snap.get(t, {}).get("price") or float(p.close[t].dropna().iloc[-1])
            line = P.exit_line(p.close[t], a[t], h.exit_line)
            if line != h.exit_line:
                h.exit_line, changed = line, True
            call = "EXIT" if last <= line else "REVIEW" if t in weak else "HOLD"
            rows.append({"ticker": t, "Qty": h.qty, "Avg cost": h.avg_cost, "Last": last,
                         "P&L": (last - h.avg_cost) * h.qty, "P&L %": last / h.avg_cost - 1, "Exit line": line,
                         "To exit line": line / last - 1, "Call": call})
        if changed:
            pf.save_holdings(hs)
        df = pd.DataFrame(rows).set_index("ticker")
        self.holdings.set_frame(df)
        v = df.dropna(subset=["Last"]) if "Last" in df else df.iloc[0:0]
        cost = float((v["Avg cost"] * v["Qty"]).sum()) if len(v) else 0.0
        pnl = float(v["P&L"].sum()) if len(v) else 0.0
        self.hsum.set_items([("Holdings", str(len(df)), None), ("Invested", f"₹{cost:,.0f}", None),
                             ("Unrealised P&L", f"₹{pnl:+,.0f}", theme.signed(pnl), f"{pnl / cost:+.1%}" if cost else ""),
                             ("EXIT calls", str(int((df.get("Call") == "EXIT").sum())), theme.RED)])

    # ------------------------------------------------------------------ journal
    def _add_trade(self) -> None:
        s = self.j_sym.text().strip().upper()
        if not s or self.j_entry.value() <= 0 or self.j_stop.value() <= 0:
            return
        pf.add_trade(pf.Trade(ticker=s if s.endswith(".NS") else s + ".NS", entry_date=self.j_date.text().strip(),
                              entry=self.j_entry.value(), stop=self.j_stop.value(), qty=self.j_qty.value(),
                              note=self.j_note.text().strip()))
        self.j_sym.clear()
        self.j_note.clear()
        self._journal()

    def _pick_trade(self, label) -> None:
        self._sel_trade = label

    def _close_trade(self) -> None:
        if self._sel_trade and self.j_exit.value() > 0:
            pf.close_trade(self._sel_trade, self.j_exit.value(), date.today().isoformat())
            self._journal()

    def _delete_trade(self) -> None:
        if self._sel_trade:
            pf.delete_trade(self._sel_trade)
            self._sel_trade = None
            self._journal()

    def _journal(self) -> None:
        tr = pf.load_journal()
        if not tr:
            self.journal.set_frame(pd.DataFrame({"": ["No trades logged yet."]}))
            self.jstats.set_items([])
            return
        df = pd.DataFrame([{"id": t.id, "Stock": t.ticker.replace(".NS", ""), "Entry date": t.entry_date, "Entry": t.entry,
                            "Stop": t.stop, "Qty": t.qty, "Exit date": t.exit_date or "open", "Exit": t.exit,
                            "R (after costs)": t.r(), "Note": t.note} for t in tr]).set_index("id")
        df.index.name = None
        self.journal.set_frame(df)
        s = pf.journal_stats(tr)
        if s["closed"]:
            self.jstats.set_items([("Closed trades", str(s["closed"]), None), ("Win rate", f"{s['win_rate']:.0%}", None),
                                   ("Average R", f"{s['avg_r']:+.2f}", theme.signed(s["avg_r"])),
                                   ("Median R", f"{s['median_r']:+.2f}", theme.signed(s["median_r"])),
                                   ("Total R", f"{s['total_r']:+.1f}", theme.signed(s["total_r"]))])
        else:
            self.jstats.set_items([("Closed trades", "0", None, "close a trade to see R")])
