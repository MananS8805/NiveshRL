"""Reusable desktop widgets: DataFrame table, KPI tiles, treemap heatmap, charts, background tasks."""
from __future__ import annotations

import math
import traceback

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import (QAbstractTableModel, QEvent, QModelIndex, QObject, QRectF, QRunnable, QSortFilterProxyModel, Qt,
                            QThreadPool, Signal)
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (QFrame, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableView, QToolTip,
                               QVBoxLayout, QWidget)

from . import theme

pg.setConfigOptions(antialias=True, background=theme.BG, foreground=theme.MUTED)


# --------------------------------------------------------------------------- background work
class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    """Run ``fn`` on the Qt thread pool; results come back on the UI thread via signals."""

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw
        self.signals = _Signals()

    def run(self) -> None:
        try:
            try:
                out = self.fn(*self.args, **self.kw)
            except Exception:
                self.signals.failed.emit(traceback.format_exc()[-1200:])
            else:
                self.signals.done.emit(out)
        except RuntimeError:              # receiver/app already gone (e.g. during shutdown)
            pass
        finally:
            _live.discard(self)


_live: set = set()                        # keep tasks (and their signal objects) alive until they finish


def run_async(fn, on_done, *args, on_error=None, **kw) -> Task:
    t = Task(fn, *args, **kw)
    t.setAutoDelete(False)
    t.signals.done.connect(on_done)
    if on_error:
        t.signals.failed.connect(on_error)
    _live.add(t)
    QThreadPool.globalInstance().start(t)
    return t


# --------------------------------------------------------------------------- DataFrame table
class FrameModel(QAbstractTableModel):
    """Read-only DataFrame model. ``fmt``: column -> python format string; ``signed``: columns coloured +/-.

    Values are copied once into plain Python lists: Qt asks for cells thousands of
    times per repaint/sort, and pandas scalar access (``iat``) is far too slow for that.
    """

    def __init__(self, df: pd.DataFrame | None = None, fmt: dict | None = None, signed: set | None = None):
        super().__init__()
        self.fmt = fmt or {}
        self.signed = signed or set()
        self._load(df if df is not None else pd.DataFrame())

    def _load(self, df: pd.DataFrame) -> None:
        self.df = df
        self.cols = [str(c) for c in df.columns]
        self.labels = list(df.index)
        self.vals = df.astype(object).to_numpy().tolist() if df.shape[1] else [[] for _ in range(len(df))]

    def set_frame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._load(df)
        self.endResetModel()

    def update_cells(self, df: pd.DataFrame) -> None:
        """Same rows/columns as before (live refresh): repaint without resetting selection or scroll."""
        if df.shape != self.df.shape or [str(c) for c in df.columns] != self.cols:
            self.set_frame(df)
            return
        relabel = list(df.index) != self.labels
        self._load(df)
        self.dataChanged.emit(self.index(0, 0), self.index(len(df) - 1, df.shape[1] - 1))
        if relabel:                                   # e.g. a different top-12: no model reset needed
            self.headerDataChanged.emit(Qt.Vertical, 0, len(df) - 1)

    def rowCount(self, parent=QModelIndex()):
        return len(self.vals)

    def columnCount(self, parent=QModelIndex()):
        return len(self.cols)

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        v = self.vals[idx.row()][idx.column()]
        if role == Qt.UserRole:                           # raw value, for sorting
            return v
        num = isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
        if role == Qt.DisplayRole:
            if v is None or (num and v != v) or v is pd.NaT:
                return "–"
            f = self.fmt.get(self.cols[idx.column()])
            try:
                return f.format(v) if f else (f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v))
            except (ValueError, TypeError):
                return str(v)
        if role == Qt.ForegroundRole and num and v == v and self.cols[idx.column()] in self.signed:
            return _BRUSH[v >= 0]
        if role == Qt.TextAlignmentRole and num:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.cols[section]
        return str(self.labels[section]).removesuffix(".NS")


_BRUSH = {True: QBrush(QColor(theme.GREEN)), False: QBrush(QColor(theme.RED))}


class _SortProxy(QSortFilterProxyModel):
    def lessThan(self, a, b):
        x, y = a.data(Qt.UserRole), b.data(Qt.UserRole)
        try:
            xn = x is None or (isinstance(x, float) and math.isnan(x))
            yn = y is None or (isinstance(y, float) and math.isnan(y))
            if xn or yn:
                return bool(yn and not xn)
            return bool(x < y)
        except TypeError:
            return str(x) < str(y)


class FrameTable(QTableView):
    """Sortable DataFrame table; ``row_clicked`` emits the row's index label (e.g. a ticker)."""
    row_clicked = Signal(object)

    def __init__(self, fmt=None, signed=None, parent=None):
        super().__init__(parent)
        self.model_ = FrameModel(fmt=fmt, signed=signed)
        self.proxy = _SortProxy()
        self.proxy.setSourceModel(self.model_)
        self.setModel(self.proxy)
        self.setSortingEnabled(True)
        # Enabling sorting sorts by column 0 descending by default, which silently reorders ranked lists
        # (rank 194 first, orders shuffled). Start unsorted: show rows in the order given until a header is clicked.
        self.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.proxy.sort(-1)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setDefaultSectionSize(22)
        self.clicked.connect(self._clicked)

    def set_frame(self, df: pd.DataFrame, live: bool = False) -> None:
        if live:
            self.model_.update_cells(df)
        else:
            self.model_.set_frame(df)
            if self.horizontalHeader().sortIndicatorSection() >= df.shape[1]:
                self.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)   # new frame, fewer columns
                self.proxy.sort(-1)
            self.resizeColumnsToContents()
            for i in range(df.shape[1]):                     # very long text columns shouldn't push others off-screen
                if self.columnWidth(i) > 420:
                    self.setColumnWidth(i, 420)

    def _clicked(self, idx):
        src = self.proxy.mapToSource(idx)
        if src.isValid():
            self.row_clicked.emit(self.model_.labels[src.row()])


# --------------------------------------------------------------------------- KPI tiles
class KpiRow(QWidget):
    """A grid of small labelled values; ``set_items([(label, text, color, sub), ...])``."""

    def __init__(self, cols: int = 6, parent=None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.cols = cols
        self.cells: list[tuple[QLabel, QLabel, QLabel]] = []

    def set_items(self, items: list[tuple]) -> None:
        while len(self.cells) < len(items):
            f = QFrame()
            f.setObjectName("kpi")
            v = QVBoxLayout(f)
            v.setContentsMargins(8, 5, 8, 5)
            v.setSpacing(1)
            lab, val, sub = QLabel(), QLabel(), QLabel()
            lab.setStyleSheet(f"color:{theme.MUTED};font-size:10px;")
            val.setStyleSheet("font-size:15px;")
            sub.setStyleSheet(f"color:{theme.MUTED};font-size:10px;")
            for w in (lab, val, sub):
                v.addWidget(w)
            i = len(self.cells)
            self.grid.addWidget(f, i // self.cols, i % self.cols)
            self.cells.append((lab, val, sub))
        for (lab, val, sub), it in zip(self.cells, items):
            label, text, color, subtext = (list(it) + [None, ""])[:4]
            for w, t in ((lab, str(label).upper()), (val, text), (sub, subtext or "")):
                if w.text() != t:                        # setText/setStyleSheet trigger relayout: skip no-ops
                    w.setText(t)
            css = f"font-size:15px;color:{color or theme.TEXT};"
            if val.property("css") != css:
                val.setProperty("css", css)
                val.setStyleSheet(css)


# --------------------------------------------------------------------------- treemap heatmap
def _squarify(values: list[float], x: float, y: float, w: float, h: float) -> list[tuple]:
    """Squarified treemap layout (Bruls et al.). Returns rects aligned with ``values`` (sorted desc by caller)."""
    rects: list[tuple] = []
    vals = list(values)
    total = sum(vals)
    if total <= 0:
        return [(x, y, 0, 0)] * len(vals)
    scale = w * h / total
    vals = [v * scale for v in vals]

    def worst(row, side):
        s = sum(row)
        return max(max(side * side * r / (s * s), (s * s) / (side * side * r)) for r in row)

    i = 0
    while i < len(vals):
        side = min(w, h)
        row = [vals[i]]
        j = i + 1
        while j < len(vals) and worst(row + [vals[j]], side) <= worst(row, side):
            row.append(vals[j])
            j += 1
        s = sum(row)
        if w >= h:                                     # lay the row out vertically on the left
            cw = s / h
            cy = y
            for r in row:
                rects.append((x, cy, cw, r / cw))
                cy += r / cw
            x, w = x + cw, w - cw
        else:
            ch = s / w
            cx = x
            for r in row:
                rects.append((cx, y, r / ch, ch))
                cx += r / ch
            y, h = y + ch, h - ch
        i = j
    return rects


def heat_color(r: float, lim: float) -> QColor:
    if r != r:
        return QColor("#1A1F27")
    t = max(-1.0, min(1.0, r / lim))
    base = np.array([26, 31, 39])
    tgt = np.array([11, 122, 62]) if t > 0 else np.array([139, 0, 0])
    c = base + (tgt - base) * abs(t)
    return QColor(int(c[0]), int(c[1]), int(c[2]))


class Treemap(QGraphicsView):
    """Sector-grouped heatmap: tiles sized equally within sectors, coloured by return. Click emits the ticker."""
    clicked_ticker = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setStyleSheet(f"background:{theme.BG};border:1px solid {theme.GRID};")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.df = pd.DataFrame()
        self.items: dict[str, tuple] = {}
        self.setMouseTracking(True)

    def set_data(self, df: pd.DataFrame) -> None:
        """df: index ticker; columns sector, ret, last, name."""
        layout_changed = set(df.index) != set(self.df.index)
        self.df = df
        if layout_changed or not self.items:
            self._layout()
        else:
            self._recolor()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout()

    def _layout(self) -> None:
        sc = self.scene()
        sc.clear()
        self.items = {}
        self._shown = {}
        if self.df.empty:
            return
        W, H = max(self.viewport().width() - 2, 50), max(self.viewport().height() - 2, 50)
        sc.setSceneRect(0, 0, W, H)
        groups = self.df.groupby("sector").size().sort_values(ascending=False)
        srects = _squarify(groups.tolist(), 0, 0, W, H)
        lim = float(np.nanpercentile(np.abs(self.df["ret"]), 95)) or 0.02
        font = QFont(theme.MONO.split(",")[0], 8)
        hfont = QFont(theme.MONO.split(",")[0], 8, QFont.Bold)
        for (sec, n), (sx, sy, sw, sh) in zip(groups.items(), srects):
            sub = self.df[self.df["sector"] == sec].sort_values("ret", ascending=False)
            hdr = 14 if sh > 40 else 0
            label = QGraphicsSimpleTextItem(str(sec)[:max(4, int(sw / 7))])
            label.setFont(hfont)
            label.setBrush(QColor(theme.AMBER))
            label.setPos(sx + 3, sy + 1)
            sc.addItem(label)
            rects = _squarify([1.0] * len(sub), sx + 1, sy + hdr, sw - 2, sh - hdr - 1)
            for (tk, row), (x, y, w, h) in zip(sub.iterrows(), rects):
                rect = QGraphicsRectItem(QRectF(x, y, w, h))
                rect.setPen(QPen(QColor(theme.BG), 1))
                rect.setBrush(QBrush(heat_color(row["ret"], lim)))
                rect.setData(0, tk)
                sc.addItem(rect)
                txt = None
                if w > 34 and h > 18:
                    txt = QGraphicsSimpleTextItem(f"{tk.replace('.NS', '')}\n{row['ret']:+.1%}" if h > 28
                                                  else tk.replace(".NS", ""))
                    txt.setFont(font)
                    txt.setBrush(QColor(theme.TEXT))
                    txt.setPos(x + 2, y + 1)
                    txt.setData(0, tk)                    # clicking/hovering the label acts on the tile
                    sc.addItem(txt)
                self.items[tk] = (rect, txt)
        self._lim = lim

    def _recolor(self) -> None:
        """Repaint only tiles whose rounded return/price changed (cheap enough for every live tick)."""
        lim = getattr(self, "_lim", 0.02)
        shown = self.__dict__.setdefault("_shown", {})
        names = self.df["name"] if "name" in self.df else self.df.index.to_series()
        for tk, r, last, name in zip(self.df.index, self.df["ret"].to_numpy(), self.df["last"].to_numpy(), names):
            item = self.items.get(tk)
            if item is None:
                continue
            key = round(float(r), 3) if r == r else None     # the tile shows 0.1% steps
            if shown.get(tk) == key:
                continue
            shown[tk] = key
            rect, txt = item
            rect.setBrush(QBrush(heat_color(r, lim)))
            if txt is not None and "\n" in txt.text():
                txt.setText(f"{tk.replace('.NS', '')}\n{r:+.1%}")

    def viewportEvent(self, e):
        """Tooltips are built on hover from the latest data (setting 194 tooltips per tick is slow)."""
        if e.type() == QEvent.ToolTip:
            it = self.itemAt(e.pos())
            tk = it.data(0) if it is not None else None
            if tk is not None and tk in self.df.index:
                row = self.df.loc[tk]
                QToolTip.showText(e.globalPos(), f"{row.get('name', tk)}\n{row['ret']:+.2%}  ₹{row['last']:,.2f}", self)
            else:
                QToolTip.hideText()
            return True
        return super().viewportEvent(e)

    def mousePressEvent(self, e):
        it = self.itemAt(e.position().toPoint())
        while it is not None and it.data(0) is None:
            it = None
        if it is not None:
            self.clicked_ticker.emit(it.data(0))
        super().mousePressEvent(e)


# --------------------------------------------------------------------------- charts
class _DateAxis(pg.AxisItem):
    def __init__(self, dates, **kw):
        super().__init__(orientation="bottom", **kw)
        self.dates = dates

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            i = int(round(v))
            out.append(self.dates[i].strftime("%d %b %y") if 0 <= i < len(self.dates) else "")
        return out


class CandleItem(pg.GraphicsObject):
    def __init__(self, o, h, l, c):
        super().__init__()
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        for i in range(len(c)):
            if not np.isfinite([o[i], h[i], l[i], c[i]]).all():
                continue
            col = QColor(theme.GREEN if c[i] >= o[i] else theme.RED)
            p.setPen(pg.mkPen(col))
            p.drawLine(pg.QtCore.QPointF(i, l[i]), pg.QtCore.QPointF(i, h[i]))
            p.setBrush(pg.mkBrush(col))
            p.drawRect(QRectF(i - 0.35, o[i], 0.7, c[i] - o[i]))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())


RANGES = [("1M", pd.DateOffset(months=1)), ("3M", pd.DateOffset(months=3)), ("6M", pd.DateOffset(months=6)),
          ("1Y", pd.DateOffset(years=1)), ("3Y", pd.DateOffset(years=3)), ("5Y", pd.DateOffset(years=5)),
          ("All", None)]
HINT = "wheel: zoom time · drag: pan · double-click: reset"


class ChartFrame(QWidget):
    """A chart with a range bar (1M … All), a value readout and consistent, simple navigation:
    the mouse wheel zooms the time axis only, dragging pans along time, the y-axis always fits
    what is visible, and a double-click returns to the default window."""

    def __init__(self, parent=None, default: str = "1Y"):
        super().__init__(parent)
        self.default = default
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.title = QLabel("")
        self.title.setStyleSheet(f"color:{theme.AMBER};font-weight:600;")
        bar.addWidget(self.title)
        bar.addStretch(1)
        self.buttons: dict[str, QPushButton] = {}
        for name, _ in RANGES:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setFixedWidth(44)
            b.setStyleSheet("QPushButton{padding:2px 4px;} QPushButton:checked{color:%s;border-color:%s;}"
                            % (theme.AMBER, theme.AMBER))
            b.clicked.connect(lambda _=False, n=name: self.set_range(n))
            self.buttons[name] = b
            bar.addWidget(b)
        lay.addLayout(bar)
        self.readout = QLabel(HINT)
        self.readout.setStyleSheet(f"color:{theme.MUTED};font-size:11px;")
        lay.addWidget(self.readout)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.BG)
        lay.addWidget(self.glw, 1)
        self.dates: list = []
        self.plots: list = []
        self._handlers: list = []

    # -- plumbing shared by every chart ------------------------------------------------
    def _reset_scene(self) -> None:
        sc = self.glw.scene()
        for sig, fn in self._handlers:                   # drop handlers that hold the previous data
            try:
                getattr(sc, sig).disconnect(fn)
            except (RuntimeError, TypeError):
                pass
        self._handlers = []
        self.glw.clear()
        self.plots = []

    def _connect(self, sig: str, fn) -> None:
        getattr(self.glw.scene(), sig).connect(fn)
        self._handlers.append((sig, fn))

    def _setup_nav(self, plots: list) -> None:
        n = len(self.dates)
        self.plots = plots
        for p in plots:
            vb = p.getViewBox()
            vb.setMouseEnabled(x=True, y=False)          # zoom/pan along time only
            vb.setLimits(xMin=-2, xMax=n + 2, minXRange=min(10, max(n - 1, 1)))
            p.setAutoVisible(y=True)                     # y fits what is on screen
            p.enableAutoRange(axis="y")
            p.getAxis("left").setWidth(60)
            p.setMenuEnabled(False)
        for p in plots[1:]:
            p.setXLink(plots[0])

        def clicked(ev):
            if ev.double():
                self.set_range(self.default)
        self._connect("sigMouseClicked", clicked)
        self.set_range(self.default)

    def set_range(self, name: str) -> None:
        for k, b in self.buttons.items():
            b.setChecked(k == name)
        if not self.plots or not self.dates:
            return
        n = len(self.dates)
        off = dict(RANGES).get(name)
        lo = 0
        if off is not None:
            start = pd.Timestamp(self.dates[-1]) - off
            lo = int(np.searchsorted(pd.DatetimeIndex(self.dates).values, np.datetime64(start)))
        self.plots[0].setXRange(max(0, lo) - 0.5, n - 0.5, padding=0.01)
        for p in self.plots:
            if p.getViewBox().state["autoVisibleOnly"][1]:
                p.enableAutoRange(axis="y")


class PriceChart(ChartFrame):
    """Candles + SMA50/200 + volume + RSI(14), one shared time axis."""

    def plot(self, df: pd.DataFrame, title: str = "") -> None:
        """df: Open/High/Low/Close/Volume indexed by date."""
        self._reset_scene()
        df = df.dropna(subset=["Close"])
        self.title.setText(title)
        if df.empty:
            return
        self.dates = list(df.index)
        dates = self.dates
        x = np.arange(len(df))
        g = self.glw
        p1 = g.addPlot(row=0, col=0)
        p1.hideAxis("bottom")
        p1.showGrid(x=True, y=True, alpha=0.15)
        if df[["Open", "High", "Low"]].notna().all().all():
            p1.addItem(CandleItem(df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()))
        else:
            p1.plot(x, df["Close"].to_numpy(), pen=pg.mkPen(theme.AMBER, width=1.5))
        for n, col in [(50, theme.BLUE), (200, "#C77DFF")]:
            p1.plot(x, df["Close"].rolling(n).mean().to_numpy(), pen=pg.mkPen(col, width=1, style=Qt.DashLine))
        p2 = g.addPlot(row=1, col=0)
        p2.hideAxis("bottom")
        p2.setMaximumHeight(80)
        p2.getAxis("left").setStyle(showValues=False)        # volume: bars only, the readout shows the number
        vol = df["Volume"].fillna(0).to_numpy() if "Volume" in df else np.zeros(len(df))
        p2.addItem(pg.BarGraphItem(x=x, height=vol, width=0.7, brush=pg.mkBrush("#2A3442")))
        p3 = g.addPlot(row=2, col=0, axisItems={"bottom": _DateAxis(dates)})
        p3.setMaximumHeight(90)
        d = df["Close"].diff()
        up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        p3.plot(x, rsi.to_numpy(), pen=pg.mkPen(theme.AMBER, width=1))
        for lvl in (30, 70):
            p3.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen(theme.MUTED, style=Qt.DotLine)))
        lines = [pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.MUTED, style=Qt.DotLine)) for _ in range(3)]
        for p, ln in zip((p1, p2, p3), lines):
            p.addItem(ln, ignoreBounds=True)
        o, h, l, c = (df[k].to_numpy() if k in df else np.full(len(df), np.nan) for k in ("Open", "High", "Low", "Close"))
        sma50 = df["Close"].rolling(50).mean().to_numpy()
        rsi_v = rsi.to_numpy()

        def moved(pos):
            for p in (p1, p2, p3):
                if p.sceneBoundingRect().contains(pos):
                    i = int(round(p.vb.mapSceneToView(pos).x()))
                    if 0 <= i < len(dates):
                        for ln in lines:
                            ln.setPos(i)
                        chg = c[i] / c[i - 1] - 1 if i > 0 else np.nan
                        self.readout.setText(
                            f"<span style='color:{theme.TEXT}'>{dates[i]:%a %d %b %Y}</span> &nbsp; O {o[i]:,.2f} &nbsp; "
                            f"H {h[i]:,.2f} &nbsp; L {l[i]:,.2f} &nbsp; C <b>{c[i]:,.2f}</b> "
                            f"<span style='color:{theme.signed(chg)}'>{chg:+.2%}</span> &nbsp; Vol {vol[i]:,.0f} &nbsp; "
                            f"SMA50 {sma50[i]:,.1f} &nbsp; RSI {rsi_v[i]:.0f} &nbsp;&nbsp; "
                            f"<span style='color:{theme.MUTED}'>{HINT}</span>")
                    return
        self._connect("sigMouseMoved", moved)
        lo_all = np.where(np.isfinite(l), l, c)
        hi_all = np.where(np.isfinite(h), h, c)

        def fit_y(*_):
            (x0, x1), _ = p1.vb.viewRange()
            a, b = max(0, int(np.floor(x0))), min(len(c), int(np.ceil(x1)) + 1)
            if b - a < 2:
                return
            lo, hi = np.nanmin(lo_all[a:b]), np.nanmax(hi_all[a:b])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = (hi - lo) * 0.06
                p1.setYRange(lo - pad, hi + pad, padding=0)
            vmax = np.nanmax(vol[a:b]) if b > a else 0
            p2.setYRange(0, max(vmax, 1) * 1.05, padding=0)
        self._setup_nav([p1, p2, p3])
        for p in (p1, p2, p3):
            p.disableAutoRange(axis="y")
            p.setAutoVisible(y=False)
        p3.setYRange(0, 100, padding=0)
        p1.sigXRangeChanged.connect(fit_y)
        fit_y()


class LineChart(ChartFrame):
    """Several date-indexed series on one time axis (aligned to the first series' dates)."""

    def __init__(self, series: dict, title: str = "", logy: bool = False, default: str = "All", parent=None):
        super().__init__(parent, default=default)
        self._reset_scene()
        self.title.setText(title)
        first = next(iter(series.values()))
        self.dates = list(first.index)
        dates = self.dates
        p = self.glw.addPlot(row=0, col=0, axisItems={"bottom": _DateAxis(dates)})
        p.showGrid(x=True, y=True, alpha=0.15)
        p.addLegend(offset=(10, 10))
        if logy:
            p.setLogMode(y=True)
        vals = {}
        for i, (name, s) in enumerate(series.items()):
            v = s.reindex(first.index).to_numpy(dtype=float)
            vals[name] = (v, theme.SERIES[i % len(theme.SERIES)])
            p.plot(np.arange(len(v)), v, pen=pg.mkPen(theme.SERIES[i % len(theme.SERIES)], width=1.6), name=name)
        ln = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.MUTED, style=Qt.DotLine))
        p.addItem(ln, ignoreBounds=True)

        def moved(pos):
            if p.sceneBoundingRect().contains(pos):
                i = int(round(p.vb.mapSceneToView(pos).x()))
                if 0 <= i < len(dates):
                    ln.setPos(i)
                    parts = [f"<span style='color:{col}'>{name[:28]}</span> {v[i]:,.3f}" for name, (v, col) in vals.items()]
                    self.readout.setText(f"<span style='color:{theme.TEXT}'>{pd.Timestamp(dates[i]):%d %b %Y}</span>"
                                         f" &nbsp; " + " &nbsp; ".join(parts))
        self._connect("sigMouseMoved", moved)
        self._setup_nav([p])


def line_chart(series: dict[str, pd.Series], title: str = "", logy: bool = False) -> LineChart:
    return LineChart(series, title, logy)


def h1(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("h1")
    return lab


def h2(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("h2")
    return lab


def muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("muted")
    lab.setWordWrap(True)
    return lab
