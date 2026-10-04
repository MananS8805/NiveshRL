"""ALRT: your own alert rules (price, day move, moving average, RSI, volume, 52-week high/low) plus the watchlist
alerts, with a log of everything that fired. Rules are checked every 15 seconds by the main window and also raise a
Windows notification."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QCompleter, QDoubleSpinBox, QHBoxLayout, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton)

from ... import custom_alerts as CA
from ...livefeed import IST
from .. import data
from ..widgets import FrameTable, h2, muted
from . import Panel, vbox


def stock_snapshot(feed, ticker: str) -> dict:
    """What an alert rule sees: live price and day move when streaming (else the last close), previous daily closes
    (today's bar excluded so the live price stands in for it), today's volume and the prior 20-day average."""
    p = data.panel()
    if ticker not in p.close:
        return {}
    closes = p.close[ticker].dropna()
    vols = p.volume[ticker].dropna() if ticker in p.volume else pd.Series(dtype=float)
    tick = feed.quote(ticker) if feed else None
    today = datetime.now(IST).date()
    if tick:
        if len(closes) and closes.index[-1].date() == today:
            closes, vols = closes.iloc[:-1], vols.iloc[:-1]
        return {"price": tick["price"], "day_change": (tick.get("change_percent") or 0) / 100, "closes": closes,
                "volume": tick.get("day_volume") or None, "avg_volume": float(vols.iloc[-20:].mean()) if len(vols) else None,
                "live": True}
    if len(closes) < 2:
        return {}
    return {"price": float(closes.iloc[-1]), "day_change": float(closes.iloc[-1] / closes.iloc[-2] - 1),
            "closes": closes.iloc[:-1], "volume": float(vols.iloc[-1]) if len(vols) else None,
            "avg_volume": float(vols.iloc[-21:-1].mean()) if len(vols) > 21 else None, "live": False}


class AlertsPanel(Panel):
    title = "Alerts"
    code = "ALRT"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        lay.addWidget(h2("New alert", "custom_alerts"))
        row = QHBoxLayout()
        self.f_ticker = QLineEdit()
        self.f_ticker.setPlaceholderText("Symbol, e.g. TCS")
        self.f_ticker.setMaximumWidth(160)
        self.f_kind = QComboBox()
        for k, (label, _) in CA.KINDS.items():
            self.f_kind.addItem(label, k)
        self.f_op = QComboBox()
        self.f_op.addItem("is at or above", "above")
        self.f_op.addItem("is at or below", "below")
        self.f_value = QDoubleSpinBox()
        self.f_value.setRange(-1_000_000, 10_000_000)
        self.f_value.setDecimals(2)
        self.f_value.setMaximumWidth(140)
        self.f_repeat = QComboBox()
        self.f_repeat.addItem("once", "once")
        self.f_repeat.addItem("once a day", "daily")
        self.f_note = QLineEdit()
        self.f_note.setPlaceholderText("note (optional)")
        self.b_add = QPushButton("Add alert")
        self.b_add.clicked.connect(self._add)
        self.f_kind.currentIndexChanged.connect(self._kind_changed)
        for w in (self.f_ticker, self.f_kind, self.f_op, self.f_value, self.f_repeat, self.f_note, self.b_add):
            row.addWidget(w)
        lay.addLayout(row)
        self.msg = muted("Rules are checked every 15 seconds against the live stream (the last close when the market "
                         "is closed) and raise a Windows notification. Price, day move, SMA, RSI, volume and 52-week "
                         "rules; 'once' switches itself off after firing.")
        lay.addWidget(self.msg)
        lay.addWidget(h2("Your rules", "custom_alerts"))
        self.rules = FrameTable()
        self.rules.model_.term_overrides = {c: "custom_alerts" for c in ("Rule", "Repeat", "Status", "Last fired", "Now")}
        self.rules.setMaximumHeight(240)
        lay.addWidget(self.rules)
        rrow = QHBoxLayout()
        self.b_del = QPushButton("Delete selected rule")
        self.b_del.clicked.connect(self._delete)
        self.b_toggle = QPushButton("Switch selected on/off")
        self.b_toggle.clicked.connect(self._toggle)
        rrow.addWidget(self.b_del)
        rrow.addWidget(self.b_toggle)
        rrow.addStretch(1)
        lay.addLayout(rrow)
        lay.addWidget(h2("Fired (newest first)", "alerts"))
        lay.addWidget(muted("Your rules and the watchlist alerts (targets, ±3% days, sentiment flips, results within 7 "
                            "days, tomorrow's top list). Double-click one to open the stock."))
        self.list = QListWidget()
        self.empty = QListWidgetItem("No alerts yet today.")
        self.list.addItem(self.empty)
        lay.addWidget(self.list, 1)
        self.list.itemDoubleClicked.connect(lambda it: it.data(Qt.UserRole) and self.stock_selected.emit(it.data(Qt.UserRole)))
        for e in reversed(CA.load_log()[:100]):           # yesterday's firings survive a restart
            self._insert(e["ticker"], e["text"], e["at"][11:19] if len(e["at"]) > 18 else e["at"])
        self._kind_changed()

    def prefill(self, ticker: str, price: float | None = None) -> None:
        self.f_ticker.setText(ticker.replace(".NS", ""))
        self.f_kind.setCurrentIndex(0)
        if price:
            self.f_value.setValue(round(price, 2))

    def _kind_changed(self) -> None:
        k = self.f_kind.currentData()
        self.f_value.setEnabled(k != "high52")
        defaults = {"day_pct": 3.0, "sma": 50, "rsi": 70, "vol_ratio": 2.0}
        if k in defaults:
            self.f_value.setValue(defaults[k])
        self.f_value.setSuffix({"price": "", "day_pct": " %", "sma": " days", "vol_ratio": " ×"}.get(k, ""))

    def ensure_loaded(self) -> None:
        super().ensure_loaded()
        if self.f_ticker.completer() is None:
            try:
                names = sorted(t.replace(".NS", "") for t in data.panel().tickers)
                self.f_ticker.setCompleter(QCompleter(names, self))
            except Exception:  # noqa: BLE001 - the panel loads later on a cold start
                pass

    def refresh(self) -> None:
        self._show_rules()

    def _ticker(self) -> str | None:
        t = self.f_ticker.text().strip().upper()
        if not t:
            return None
        t = t if t.endswith(".NS") else t + ".NS"
        return t if t in data.panel().close.columns else None

    def _add(self) -> None:
        t = self._ticker()
        if t is None:
            self.msg.setText("Unknown symbol: alerts work for the NIFTY 200 stocks the app streams.")
            return
        r = CA.Rule(t, self.f_kind.currentData(), self.f_op.currentData(), float(self.f_value.value()),
                    self.f_repeat.currentData(), self.f_note.text().strip())
        CA.add(r)
        self.msg.setText(f"Added: {r.describe()} ({r.repeat}).")
        self.f_note.clear()
        self._show_rules()

    def _selected_id(self) -> str | None:
        idx = self.rules.currentIndex()
        if not idx.isValid():
            return None
        src = self.rules.proxy.mapToSource(idx)
        return self.rules.model_.labels[src.row()]

    def _delete(self) -> None:
        rid = self._selected_id()
        if rid:
            CA.remove(rid)
            self._show_rules()

    def _toggle(self) -> None:
        rid = self._selected_id()
        if not rid:
            return
        rules = CA.load()
        for r in rules:
            if r.id == rid:
                r.active = not r.active
        CA.save(rules)
        self._show_rules()

    def _show_rules(self) -> None:
        rules = CA.load()
        if not rules:
            self.rules.set_frame(pd.DataFrame({"": ["No rules yet: add one above, or use 'Set alert' on a stock page."]}))
            return
        rows = []
        for r in rules:
            obs, thr = CA.measure(r, stock_snapshot(self.ctx.feed, r.ticker))
            now = "–" if obs is None else (f"{obs:,.2f}" if r.kind not in ("rsi", "day_pct", "vol_ratio") else
                                           f"{obs:.0f}" if r.kind == "rsi" else f"{obs:+.1f}%" if r.kind == "day_pct"
                                           else f"{obs:.1f}×")
            rows.append({"id": r.id, "Rule": r.describe() + (f" · {r.note}" if r.note else ""), "Now": now,
                         "Repeat": r.repeat, "Status": "on" if r.active else "off (fired)" if r.last_fired else "off",
                         "Last fired": (r.last_fired or "").replace("T", " ")})
        df = pd.DataFrame(rows).set_index("id")
        self.rules.set_frame(df)
        self.rules.verticalHeader().setVisible(False)

    def _insert(self, ticker: str, text: str, when: str) -> None:
        if self.empty is not None:
            self.list.takeItem(self.list.row(self.empty))
            self.empty = None
        it = QListWidgetItem(f"{when}  {ticker.replace('.NS', ''):<12} {text}")
        it.setData(Qt.UserRole, ticker if ticker.endswith(".NS") else None)
        self.list.insertItem(0, it)
        while self.list.count() > 500:                    # bounded
            self.list.takeItem(self.list.count() - 1)

    def add(self, ticker: str, text: str) -> None:
        self._insert(ticker, text, f"{datetime.now(IST):%H:%M:%S}")
