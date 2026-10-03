"""WATCH: personal watchlist with ★ must have / ☆ preferred tiers, live prices and alert badges."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCompleter, QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QPushButton)

from ... import watchlist as wl
from ...research import daily
from .. import data
from ..widgets import FrameTable, h2, muted
from . import Panel, vbox


def watch_rows(feed) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One row per watchlist stock plus its alerts. Shared with the alert notifier."""
    items = wl.load()
    if not items:
        return pd.DataFrame(), {}
    p = data.panel()
    snap = feed.snapshot() if feed else {}
    t = data.screener_table()
    prev = None
    folders = sorted(x for x in daily.DAILY.glob("20*") if x.is_dir())
    if len(folders) >= 2 and (folders[-2] / "sentiment.parquet").exists():
        prev = data._cached(f"prevsent:{folders[-2].name}", None,
                            lambda: pd.read_parquet(folders[-2] / "sentiment.parquet"))
    mon = data.dload("monitor")
    top_now = set(mon[mon["list"] == "watch for strength"].index) if mon is not None else set()
    rows, alerts = [], {}
    for tkr, it in sorted(items.items(), key=lambda kv: (kv[1].tier != "must", kv[0])):
        tick = snap.get(tkr)
        hist = p.close[tkr].dropna() if tkr in p.close else pd.Series(dtype=float)
        price = tick["price"] if tick else (float(hist.iloc[-1]) if len(hist) else None)
        chg = (tick["change_percent"] / 100) if tick and tick.get("change_percent") is not None else \
            (float(t.loc[tkr, "ret_1d"]) if t is not None and tkr in t.index else None)
        r = t.loc[tkr] if t is not None and tkr in t.index else pd.Series(dtype=float)
        dte = r.get("days_to_earnings")
        a = wl.alerts(it, price, chg, r.get("sentiment"),
                      prev["sentiment"].get(tkr) if prev is not None else None,
                      int(dte) if pd.notna(dte) else None, tkr in top_now, None)
        alerts[tkr] = a
        rows.append({"Tier": wl.TIERS[it.tier][0], "Stock": tkr.replace(".NS", ""), "Price": price, "1D": chg,
                     "Live": "●" if tick else "", "P(up) 1D": r.get("prob_up"), "Sentiment": r.get("sentiment_adj"),
                     "Analyst": r.get("analyst_score"), "Results in": dte,
                     "Buy ≤": it.target_buy, "Sell ≥": it.target_sell, "Alerts": " · ".join(a), "Note": it.note,
                     "_t": tkr})
    df = pd.DataFrame(rows).set_index("_t")
    df.index.name = None
    return df, alerts


class WatchlistPanel(Panel):
    title = "Watchlist"
    code = "WATCH"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        lay.addWidget(h2("Add a stock"))
        row = QHBoxLayout()
        self.pick = QComboBox()
        self.pick.setEditable(True)
        self.pick.setInsertPolicy(QComboBox.NoInsert)
        self.pick.completer().setFilterMode(Qt.MatchContains)
        self.pick.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.tier = QComboBox()
        for k, v in wl.TIERS.items():
            self.tier.addItem(v, k)
        self.buy = QDoubleSpinBox()
        self.sell = QDoubleSpinBox()
        for sp, label in [(self.buy, "Buy ≤ ₹ "), (self.sell, "Sell ≥ ₹ ")]:
            sp.setRange(0, 1_000_000)
            sp.setDecimals(1)
            sp.setPrefix(label)
            sp.setSpecialValueText(label + "(none)")
        self.note = QLineEdit()
        self.note.setPlaceholderText("Note (optional)")
        add = QPushButton("＋ Add / update")
        add.setObjectName("primary")
        add.clicked.connect(self._add)
        for w, s in [(self.pick, 3), (self.tier, 1), (self.buy, 1), (self.sell, 1), (self.note, 2), (add, 0)]:
            row.addWidget(w, s)
        lay.addLayout(row)
        fmt = {"Price": "₹{:,.2f}", "1D": "{:+.2%}", "P(up) 1D": "{:.0%}", "Sentiment": "{:+.2f}", "Analyst": "{:.0f}",
               "Results in": "{:.0f}d", "Buy ≤": "₹{:,.0f}", "Sell ≥": "₹{:,.0f}"}
        self.table = FrameTable(fmt=fmt, signed={"1D", "Sentiment"})
        self.table.row_clicked.connect(self._clicked)
        lay.addWidget(self.table, 1)
        brow = QHBoxLayout()
        self.rm = QPushButton("Remove selected")
        self.rm.clicked.connect(self._remove)
        self.hint = muted("Alerts: price at your buy/sell target · ±3% day move · news sentiment flip · results "
                          "within 7 days · entered/left tomorrow's top list. Desktop notifications fire once per alert per day.")
        brow.addWidget(self.rm)
        brow.addWidget(self.hint, 1)
        lay.addLayout(brow)
        self.sel: str | None = None
        self._n_rows = -1
        ctx.watchlist_changed.connect(self.refresh)

    def refresh(self) -> None:
        if self.pick.count() == 0:
            p = data.panel()
            for t in sorted(p.tickers):
                self.pick.addItem(f"{t.replace('.NS', '')} · {str(p.names.get(t, ''))[:30]}", t)
        df, _ = watch_rows(self.ctx.feed)
        if df.empty:
            self.table.set_frame(pd.DataFrame({"": ["Your watchlist is empty. Add stocks above, or from the stock "
                                                    "panel / screener / monitor list."]}))
        else:
            self.table.set_frame(df)
        self._n_rows = len(df)

    def on_tick(self) -> None:
        if self._n_rows > 0:
            df, _ = watch_rows(self.ctx.feed)
            self.table.set_frame(df, live=True)

    def _add(self) -> None:
        t = self.pick.currentData()
        if not t:
            txt = self.pick.currentText().split(" ")[0].upper()
            t = txt if txt.endswith(".NS") else txt + ".NS"
            if t not in data.panel().tickers:
                self.hint.setText(f"Unknown symbol {txt}.")
                return
        wl.add(t, self.tier.currentData(), self.note.text(), self.buy.value() or None, self.sell.value() or None)
        self.note.clear()
        self.ctx.watchlist_changed.emit()

    def _clicked(self, tk) -> None:
        if isinstance(tk, str) and tk.endswith(".NS"):
            self.sel = tk
            self.rm.setText(f"Remove {tk.replace('.NS', '')}")
            self.stock_selected.emit(tk)

    def _remove(self) -> None:
        if self.sel:
            wl.remove(self.sel)
            self.sel = None
            self.rm.setText("Remove selected")
            self.ctx.watchlist_changed.emit()
