"""SCRN: technical + fundamental + sentiment + model screener over the NIFTY 200.

Filters run in the C++ core (``niveshrl_core.apply_filters``) on a float column
matrix built once per daily run, so re-filtering is instant; without the core
it falls back to ``research.screener.apply`` (the reference implementation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QMenu, QPushButton, QToolButton,
                               QVBoxLayout, QWidget)

from ...research import screener as scr
from .. import data
from ..widgets import FrameTable, h2, muted
from . import Panel, vbox

OPC = {">": 0, ">=": 1, "<": 2, "<=": 3, "=": 4}
DEFAULT_COLS = ["close", "ret_1d", "ret_1m", "rsi14", "vs_sma200", "from_52w_high", "vol_ratio", "trailingPE",
                "returnOnEquity", "analyst_score", "sentiment_adj", "prob_up"]


class ColumnStore:
    """Numeric snapshot of the screener table for the C++ filter engine."""

    def __init__(self, table: pd.DataFrame):
        self.table = table
        self.cols = [c for c in table.columns if c in scr.COLUMNS]
        self.pos = {c: i for i, c in enumerate(self.cols)}
        self.X = np.ascontiguousarray(table[self.cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64))
        try:
            import niveshrl_core
            self.core = niveshrl_core
        except ImportError:
            self.core = None

    def apply(self, filters: list[tuple[str, str, float]]) -> pd.DataFrame:
        flt = [f for f in filters if f[0] in self.pos]
        if self.core is None:
            return scr.apply(self.table, flt)
        mask = self.core.apply_filters(self.X, [(self.pos[c], OPC[o], float(v)) for c, o, v in flt])
        return self.table[np.asarray(mask, dtype=bool)]


class FilterRow(QWidget):
    def __init__(self, labels: dict, on_change, on_remove):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.col = QComboBox()
        for k, v in labels.items():
            self.col.addItem(v, k)
        self.op = QComboBox()
        self.op.addItems(list(scr.OPS))
        self.val = QDoubleSpinBox()
        self.val.setRange(-1e9, 1e9)
        self.val.setDecimals(2)
        rm = QPushButton("✕")
        rm.setFixedWidth(30)
        rm.clicked.connect(lambda: on_remove(self))
        for w in (self.col, self.op, self.val):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged).connect(on_change)
        self.col.currentIndexChanged.connect(self._suffix)
        lay.addWidget(self.col, 3)
        lay.addWidget(self.op, 1)
        lay.addWidget(self.val, 2)
        lay.addWidget(rm)
        self._suffix()

    def _suffix(self):
        self.val.setSuffix(" %" if scr.COLUMNS.get(self.col.currentData(), ("", "", ""))[2] == "pct" else "")

    def get(self) -> tuple[str, str, float]:
        c = self.col.currentData()
        v = self.val.value()
        return c, self.op.currentText(), v / 100 if scr.COLUMNS[c][2] == "pct" else v


def _fmt(cols) -> dict:
    out = {}
    for c in cols:
        label, _, kind = scr.COLUMNS.get(c, (c, "", "num"))
        out[label] = {"pct": "{:+.1%}", "cr": "{:,.0f}", "flag": "{:.0f}"}.get(kind, "{:,.2f}")
    return out


class ScreenerPanel(Panel):
    title = "Screener"
    code = "SCRN"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        top = QHBoxLayout()
        top.addWidget(h2("Preset"))
        self.preset = QComboBox()
        self.preset.currentIndexChanged.connect(self._run)
        top.addWidget(self.preset, 2)
        self.desc = muted("")
        top.addWidget(self.desc, 3)
        lay.addLayout(top)
        frow = QHBoxLayout()
        add = QPushButton("＋ Filter")
        add.clicked.connect(lambda: self._add_filter())
        self.cols_btn = QToolButton()
        self.cols_btn.setText("Columns ▾")
        self.cols_btn.setPopupMode(QToolButton.InstantPopup)
        self.cols_menu = QMenu(self)
        self.cols_btn.setMenu(self.cols_menu)
        csv = QPushButton("⬇ CSV")
        csv.clicked.connect(self._csv)
        frow.addWidget(add)
        frow.addWidget(self.cols_btn)
        frow.addWidget(csv)
        self.summary = muted("")
        frow.addWidget(self.summary, 1)
        lay.addLayout(frow)
        self.filters_box = QVBoxLayout()
        lay.addLayout(self.filters_box)
        self.rows: list[FilterRow] = []
        self.table = FrameTable()
        self.table.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        lay.addWidget(self.table, 1)
        self.store: ColumnStore | None = None
        self.res = pd.DataFrame()
        self.show_cols: list[str] = []

    def refresh(self) -> None:
        t = data.screener_table()
        if t is None:
            self.table.set_frame(pd.DataFrame({"": ["No daily data yet. Press F5 to run the daily pipeline."]}))
            return
        self.store = ColumnStore(t)
        self.labels = {k: f"{v[1]} · {v[0]}" for k, v in scr.COLUMNS.items() if k in t}
        if not self.show_cols:
            self.show_cols = [c for c in DEFAULT_COLS if c in t]
        self.cols_menu.clear()
        for c in [c for c in t.columns if c in scr.COLUMNS or c == "close"]:
            a = self.cols_menu.addAction(scr.COLUMNS.get(c, (c,))[0])
            a.setCheckable(True)
            a.setChecked(c in self.show_cols)
            a.toggled.connect(lambda on, c=c: self._toggle_col(c, on))
        names = list(scr.PRESETS)
        counts = {n: len(self.store.apply(scr.PRESETS[n][1])) for n in names}
        first = next((i for i, n in enumerate(names) if counts[n]), 0)   # weak tapes empty momentum screens
        self.preset.blockSignals(True)
        self.preset.clear()
        self.preset.addItem("(custom)", None)
        for n in names:
            self.preset.addItem(f"{n} ({counts[n]})", n)
        self.preset.setCurrentIndex(first + 1)
        self.preset.blockSignals(False)
        self._run()

    def _toggle_col(self, c: str, on: bool) -> None:
        if on and c not in self.show_cols:
            self.show_cols.append(c)
        elif not on and c in self.show_cols:
            self.show_cols.remove(c)
        self._run()

    def _add_filter(self) -> None:
        if self.store is None:
            return
        r = FilterRow(self.labels, self._run, self._remove_filter)
        self.rows.append(r)
        self.filters_box.addWidget(r)
        self._run()

    def _remove_filter(self, r: FilterRow) -> None:
        self.rows.remove(r)
        r.setParent(None)
        r.deleteLater()
        self._run()

    def _run(self) -> None:
        if self.store is None:
            return
        name = self.preset.currentData()
        base = list(scr.PRESETS[name][1]) if name else []
        self.desc.setText(scr.PRESETS[name][0] if name else "Build your own filters with ＋ Filter.")
        flt = base + [r.get() for r in self.rows]
        res = self.store.apply(flt)
        self.res = res
        t = self.store.table
        cond = " AND ".join(f"{scr.COLUMNS[c][0]} {o} {v:.0%}" if scr.COLUMNS[c][2] == "pct" else
                            f"{scr.COLUMNS[c][0]} {o} {v:g}" for c, o, v in flt if c in scr.COLUMNS) or "no filters"
        self.summary.setText(f"{len(res)} of {len(t)} stocks match · {cond}")
        if not len(res):
            self.table.set_frame(pd.DataFrame({"": ["No stocks match these filters today. Loosen a condition or pick another preset."]}))
            return
        cols = [c for c in self.show_cols if c in res]
        view = res[["sector"] + cols].copy()
        view.columns = ["Sector"] + [scr.COLUMNS.get(c, (c,))[0] for c in cols]
        self.table.model_.fmt = _fmt(cols)
        self.table.model_.signed = {scr.COLUMNS[c][0] for c in cols if c in scr.COLUMNS and scr.COLUMNS[c][2] == "pct"}
        self.table.set_frame(view)

    def _csv(self) -> None:
        if self.res is None or self.res.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save screener results", "niveshrl_screener.csv", "CSV (*.csv)")
        if path:
            self.res.to_csv(path)
