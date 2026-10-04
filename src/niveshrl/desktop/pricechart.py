"""TradingView-style price chart: timeframes (1m … 1W), an indicator picker (SMA, EMA, VWAP, Bollinger, Supertrend,
pivots/CPR, Fibonacci; volume, RSI and MACD panes), a compare overlay rebased to the visible window, log scale and
the trade plan drawn as levels. Same navigation as every chart: wheel zooms time, drag pans, y fits the view."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QMenu, QPushButton, QToolButton

from ..research import chartdata as C
from . import theme
from .widgets import HINT, RANGES, CandleItem, ChartFrame, run_async

INDICATORS = [("sma", "SMA 50 / 200"), ("ema", "EMA 9 / 21"), ("vwap", "VWAP (intraday)"), ("bb", "Bollinger bands (20, 2)"),
              ("st", "Supertrend (10, 3)"), ("pivots", "Pivots + CPR"), ("fib", "Fibonacci (visible swing)"),
              (None, None), ("volume", "Volume pane"), ("rsi", "RSI (14) pane"), ("macd", "MACD (12, 26, 9) pane")]
DEFAULT_IND = {"sma", "vwap", "volume", "rsi"}
CHART_RANGES = [("1D", pd.DateOffset(days=1)), ("5D", pd.DateOffset(days=7))] + RANGES
DEFAULT_RANGE = {"1m": "1D", "5m": "5D", "15m": "1M", "1h": "3M", "1D": "1Y", "1W": "5Y"}
PURPLE, TEAL, PINK = "#C77DFF", "#5EEAD4", "#FF6B9A"


class _TimeAxis(pg.AxisItem):
    def __init__(self, dates, intraday: bool, **kw):
        super().__init__(orientation="bottom", **kw)
        self.dates, self.fmt = dates, "%d %b %H:%M" if intraday else "%d %b %y"

    def tickStrings(self, values, scale, spacing):
        n = len(self.dates)
        return [self.dates[int(round(v))].strftime(self.fmt) if 0 <= int(round(v)) < n else "" for v in values]


class _PriceAxis(pg.AxisItem):
    """Left axis that shows prices when the pane is plotted in log10 space."""
    log = False

    def tickStrings(self, values, scale, spacing):
        if not self.log:
            return super().tickStrings(values, scale, spacing)
        return [f"{10 ** v:,.0f}" if 10 ** v >= 100 else f"{10 ** v:,.2f}" for v in values]


class PriceChart(ChartFrame):
    """Candles + overlays + optional panes on one shared time axis.

    ``plot(df, title, compare=None, plan=None)`` draws a frame directly. ``set_source(fn)`` (with ``controls=True``)
    lets the chart fetch its own bars: ``fn(tf)`` returns ``{"df", "title", "compare": {name: Series}, "plan"}`` and
    runs in the background for intraday timeframes.
    """

    def __init__(self, parent=None, controls: bool = False):
        super().__init__(parent)
        self.explain_key = "candle"
        self.ind = set(DEFAULT_IND)
        self.log = False
        self.show_plan = True
        self.cmp_on: set[str] = set()
        self.tf = "1D"
        self._source = None
        self._args: tuple = (pd.DataFrame(), "", None, None)
        self.range_map = dict(CHART_RANGES)
        for name, off in CHART_RANGES[:2]:               # 1D / 5D buttons for intraday timeframes
            b = QPushButton(name)
            b.setCheckable(True)
            b.setFixedWidth(44)
            b.setStyleSheet(next(iter(self.buttons.values())).styleSheet())
            b.clicked.connect(lambda _=False, n=name: self.set_range(n))
            self.buttons = {name: b, **self.buttons}
            bar = self.layout().itemAt(0).layout()
            bar.insertWidget(bar.indexOf(self.buttons["1M"]), b)
        if controls:
            self._build_controls()

    # ------------------------------------------------------------------ controls
    def _build_controls(self) -> None:
        try:
            saved = QSettings("NiveshRL", "NiveshRL").value("chart/indicators")
            if saved:
                self.ind = set(str(saved).split(",")) & {k for k, _ in INDICATORS if k}
            self.log = QSettings("NiveshRL", "NiveshRL").value("chart/log", False, type=bool)
        except Exception:  # noqa: BLE001 - settings are a convenience
            pass
        row = QHBoxLayout()
        row.setSpacing(4)
        self.tf_buttons: dict[str, QPushButton] = {}
        for tf in C.TIMEFRAMES:
            b = QPushButton(tf)
            b.setCheckable(True)
            b.setFixedWidth(40)
            b.setStyleSheet("QPushButton{padding:2px 4px;} QPushButton:checked{color:%s;border-color:%s;}"
                            % (theme.AMBER, theme.AMBER))
            b.clicked.connect(lambda _=False, t=tf: self.set_timeframe(t))
            self.tf_buttons[tf] = b
            row.addWidget(b)
        row.addSpacing(12)
        self.ind_btn = QToolButton()
        self.ind_btn.setText("Indicators ▾")
        self.ind_btn.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self.ind_btn)
        self.ind_actions = {}
        for key, label in INDICATORS:
            if key is None:
                m.addSeparator()
                continue
            a = m.addAction(label)
            a.setCheckable(True)
            a.setChecked(key in self.ind)
            a.toggled.connect(lambda on, k=key: self._toggle(k, on))
            self.ind_actions[key] = a
        self.ind_btn.setMenu(m)
        row.addWidget(self.ind_btn)
        self.cmp_btn = QToolButton()
        self.cmp_btn.setText("Compare ▾")
        self.cmp_btn.setPopupMode(QToolButton.InstantPopup)
        self.cmp_menu = QMenu(self.cmp_btn)
        self.cmp_btn.setMenu(self.cmp_menu)
        row.addWidget(self.cmp_btn)
        self.log_box = QCheckBox("Log scale")
        self.log_box.setChecked(self.log)
        self.log_box.toggled.connect(self._set_log)
        row.addWidget(self.log_box)
        self.plan_box = QCheckBox("Trade plan")
        self.plan_box.setChecked(True)
        self.plan_box.toggled.connect(self._set_plan)
        row.addWidget(self.plan_box)
        row.addStretch(1)
        self.status = QPushButton("")
        self.status.setFlat(True)
        self.status.setStyleSheet(f"color:{theme.MUTED};border:none;")
        row.addWidget(self.status)
        self.layout().insertLayout(1, row)

    def _save(self) -> None:
        try:
            s = QSettings("NiveshRL", "NiveshRL")
            s.setValue("chart/indicators", ",".join(sorted(self.ind)))
            s.setValue("chart/log", self.log)
        except Exception:  # noqa: BLE001
            pass

    def _toggle(self, key: str, on: bool) -> None:
        (self.ind.add if on else self.ind.discard)(key)
        self._save()
        self._replot()

    def _set_log(self, on: bool) -> None:
        self.log = on
        self._save()
        self._replot()

    def _set_plan(self, on: bool) -> None:
        self.show_plan = on
        self._replot()

    def _set_compare(self, name: str, on: bool) -> None:
        (self.cmp_on.add if on else self.cmp_on.discard)(name)
        self._replot()

    def set_source(self, fn, tf: str | None = None) -> None:
        self._source = fn
        self.set_timeframe(tf or self.tf)

    def set_timeframe(self, tf: str) -> None:
        self.tf = tf
        for k, b in getattr(self, "tf_buttons", {}).items():
            b.setChecked(k == tf)
        if self._source is None:
            return
        token = object()
        self._token = token
        if tf in ("1D", "1W"):
            self._got(token, tf, self._source(tf))
            return
        if hasattr(self, "status"):
            self.status.setText(f"loading {tf} bars from Yahoo…")
        run_async(self._source, lambda r, t=token, f=tf: self._got(t, f, r), tf,
                  on_error=lambda e, t=token: t is self._token and hasattr(self, "status")
                  and self.status.setText(f"{tf} bars unavailable: {e.splitlines()[-1][:80]}"))

    def _got(self, token, tf: str, r: dict) -> None:
        if token is not self._token:
            return                                        # the user picked another timeframe meanwhile
        df = r.get("df")
        if hasattr(self, "status"):
            self.status.setText("" if df is None or df.empty else
                                f"{len(df):,} bars · last {pd.Timestamp(df.index[-1]):%d %b %H:%M}" if tf not in ("1D", "1W")
                                else f"{len(df):,} bars")
        if hasattr(self, "cmp_menu"):
            self.cmp_menu.clear()
            for name in (r.get("compare") or {}):
                a = self.cmp_menu.addAction(name)
                a.setCheckable(True)
                a.setChecked(name in self.cmp_on)
                a.toggled.connect(lambda on, n=name: self._set_compare(n, on))
        self.default = DEFAULT_RANGE.get(tf, "1Y")
        self.plot(df if df is not None else pd.DataFrame(), r.get("title", ""), r.get("compare"), r.get("plan"))

    def _replot(self) -> None:
        df, title, cmp, plan = self._args
        keep = self._visible()
        self.plot(df, title, cmp, plan)
        if keep is not None and self.plots:
            self.plots[0].setXRange(*keep, padding=0)

    def _visible(self):
        return tuple(self.plots[0].vb.viewRange()[0]) if self.plots else None

    # ------------------------------------------------------------------ drawing
    def plot(self, df: pd.DataFrame, title: str = "", compare: dict | None = None, plan=None) -> None:
        """df: Open/High/Low/Close/Volume indexed by date or timestamp."""
        self._args = (df, title, compare, plan)
        self._reset_scene()
        df = df.dropna(subset=["Close"]) if len(df) else df
        self.title.setText(title)
        if df is None or df.empty:
            self.dates = []
            return
        intraday = C.is_intraday(df.index)
        self.dates = list(pd.DatetimeIndex(df.index))
        dates, n = self.dates, len(df)
        first, last = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
        for k, b in self.buttons.items():              # only ranges that make sense for this data
            off = self.range_map.get(k)
            b.setVisible(off is None or ((intraday or k not in ("1D", "5D"))
                                         and first <= last - off + pd.Timedelta(days=3)))
        x = np.arange(n)
        f = np.log10 if self.log else (lambda v: v)
        o, h, l, c = (df[k].to_numpy(dtype=float) if k in df else np.full(n, np.nan) for k in ("Open", "High", "Low", "Close"))
        vol = df["Volume"].fillna(0).to_numpy(dtype=float) if "Volume" in df else np.zeros(n)
        g = self.glw
        panes = [k for k in ("volume", "rsi", "macd") if k in self.ind]
        axis = _PriceAxis(orientation="left")
        axis.log = self.log
        p1 = g.addPlot(row=0, col=0, axisItems={"left": axis} if panes else
                       {"left": axis, "bottom": _TimeAxis(dates, intraday)})
        if panes:
            p1.hideAxis("bottom")
        p1.showGrid(x=True, y=True, alpha=0.15)
        p1.addLegend(offset=(10, 6), labelTextColor=theme.MUTED)
        has_ohlc = np.isfinite(o).any() and np.isfinite(h).any() and np.isfinite(l).any()
        if has_ohlc:
            p1.addItem(CandleItem(f(o), f(h), f(l), f(c)))
        else:
            p1.plot(x, f(c), pen=pg.mkPen(theme.AMBER, width=1.5))
        close = df["Close"].astype(float)
        series_for_fit = []
        readout_extra = {}

        def line(vals, col, name=None, width=1.0, style=Qt.SolidLine):
            v = np.asarray(vals, dtype=float)
            p1.plot(x, f(v), pen=pg.mkPen(col, width=width, style=style), name=name, connect="finite")
            series_for_fit.append(v)
            return v

        if "sma" in self.ind:
            readout_extra["SMA50"] = line(C.sma(close, 50), theme.BLUE, "SMA 50", style=Qt.DashLine)
            line(C.sma(close, 200), PURPLE, "SMA 200", style=Qt.DashLine)
        if "ema" in self.ind:
            line(C.ema(close, 9), TEAL, "EMA 9")
            line(C.ema(close, 21), PINK, "EMA 21")
        if "vwap" in self.ind and intraday and has_ohlc:
            readout_extra["VWAP"] = line(C.vwap(df), theme.AMBER, "VWAP", width=1.4)
        if "bb" in self.ind:
            lo_b, mid_b, up_b = C.bollinger(close)
            line(up_b, theme.MUTED, "Bollinger", style=Qt.DotLine)
            line(lo_b, theme.MUTED, style=Qt.DotLine)
            line(mid_b, "#5A6577")
        if "st" in self.ind and has_ohlc:
            st, d = C.supertrend(df)
            sv, dv = st.to_numpy(), d.to_numpy()
            line(np.where(dv > 0, sv, np.nan), theme.GREEN, "Supertrend", width=1.4)
            line(np.where(dv < 0, sv, np.nan), theme.RED, width=1.4)
        if "pivots" in self.ind and has_ohlc and n > 2:
            pv = C.pivots(df)
            for k, col, sty in [("P", theme.TEXT, Qt.SolidLine), ("TC", theme.BLUE, Qt.DotLine),
                                ("BC", theme.BLUE, Qt.DotLine), ("R1", theme.RED, Qt.DashLine), ("S1", theme.GREEN, Qt.DashLine),
                                ("R2", theme.RED, Qt.DotLine), ("S2", theme.GREEN, Qt.DotLine)]:
                line(pv[k], col, "Pivots / CPR" if k == "P" else None, style=sty)
        cmp_items = {}
        for i, (name, s) in enumerate((compare or {}).items()):
            if name not in self.cmp_on:
                continue
            v = pd.Series(s).reindex(df.index).ffill().to_numpy(dtype=float)
            col = theme.SERIES[(i + 3) % len(theme.SERIES)]
            cmp_items[name] = (v, p1.plot(x, f(v), pen=pg.mkPen(col, width=1.4), name=f"{name} (rebased)",
                                          connect="finite"))
        plan_levels = []
        if plan is not None and self.show_plan:
            for val, lab, col in [(plan.entry, "Entry", theme.AMBER), (plan.stop, "Stop", theme.RED),
                                  (plan.t1, "T1 1.5R", theme.GREEN), (plan.t2, "T2 2.5R", theme.GREEN)]:
                ln = pg.InfiniteLine(f(val), angle=0, movable=False,
                                     pen=pg.mkPen(col, width=1, style=Qt.DashLine),
                                     label=f"{lab} ₹{val:,.2f}", labelOpts={"position": 0.85, "color": col})
                p1.addItem(ln, ignoreBounds=True)
                plan_levels.append(val)
        fib_lines = []
        if "fib" in self.ind:
            for r in C.FIB_LEVELS:
                ln = pg.InfiniteLine(0, angle=0, movable=False, pen=pg.mkPen(theme.MUTED, width=1, style=Qt.DotLine),
                                     label="", labelOpts={"position": 0.97, "color": theme.MUTED})
                p1.addItem(ln, ignoreBounds=True)
                fib_lines.append(ln)
        # panes
        plots = [p1]
        rsi_v = C.rsi(close).to_numpy()
        for j, key in enumerate(panes):
            last = j == len(panes) - 1
            pp = g.addPlot(row=j + 1, col=0, axisItems={"bottom": _TimeAxis(dates, intraday)} if last else None)
            if not last:
                pp.hideAxis("bottom")
            pp.showGrid(x=True, y=False, alpha=0.1)
            pp.getAxis("left").setWidth(60)
            if key == "volume":
                pp.setMaximumHeight(80)
                pp.getAxis("left").setStyle(showValues=False)
                up = c >= np.where(np.isfinite(o), o, c)
                pp.addItem(pg.BarGraphItem(x=x[up], height=vol[up], width=0.7, brush=pg.mkBrush("#1F4D3A")))
                pp.addItem(pg.BarGraphItem(x=x[~up], height=vol[~up], width=0.7, brush=pg.mkBrush("#5A2626")))
            elif key == "rsi":
                pp.setMaximumHeight(90)
                pp.plot(x, rsi_v, pen=pg.mkPen(theme.AMBER, width=1))
                for lvl in (30, 70):
                    pp.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen(theme.MUTED, style=Qt.DotLine)))
                pp.setYRange(0, 100, padding=0)
            elif key == "macd":
                pp.setMaximumHeight(100)
                ml, sg, hist = (s.to_numpy() for s in C.macd(close))
                hp = np.nan_to_num(hist)
                pp.addItem(pg.BarGraphItem(x=x, height=hp, width=0.7,
                                           brushes=[pg.mkBrush("#1F4D3A" if v >= 0 else "#5A2626") for v in hp]))
                pp.plot(x, ml, pen=pg.mkPen(theme.BLUE, width=1))
                pp.plot(x, sg, pen=pg.mkPen(theme.AMBER, width=1))
                pp.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(theme.MUTED, style=Qt.DotLine)))
                pp._macd = (ml, sg, hp)
            pp._key = key
            plots.append(pp)
        vlines = [pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.MUTED, style=Qt.DotLine)) for _ in plots]
        for pp, ln in zip(plots, vlines):
            pp.addItem(ln, ignoreBounds=True)
        tfmt = "%a %d %b %H:%M" if intraday else "%a %d %b %Y"

        def moved(pos):
            for pp in plots:
                if pp.sceneBoundingRect().contains(pos):
                    i = int(round(pp.vb.mapSceneToView(pos).x()))
                    if 0 <= i < n:
                        for ln in vlines:
                            ln.setPos(i)
                        chg = c[i] / c[i - 1] - 1 if i > 0 else np.nan
                        extra = " &nbsp; ".join(f"{k} {v[i]:,.1f}" for k, v in readout_extra.items() if np.isfinite(v[i]))
                        self.readout.setText(
                            f"<span style='color:{theme.TEXT}'>{dates[i]:{tfmt}}</span> &nbsp; O {o[i]:,.2f} &nbsp; "
                            f"H {h[i]:,.2f} &nbsp; L {l[i]:,.2f} &nbsp; C <b>{c[i]:,.2f}</b> "
                            f"<span style='color:{theme.signed(chg)}'>{chg:+.2%}</span> &nbsp; Vol {vol[i]:,.0f} &nbsp; "
                            f"{extra} &nbsp; RSI {rsi_v[i]:.0f} &nbsp;&nbsp; <span style='color:{theme.MUTED}'>{HINT}</span>")
                    return
        self._connect("sigMouseMoved", moved)
        lo_all = np.where(np.isfinite(l), l, c)
        hi_all = np.where(np.isfinite(h), h, c)

        def fit_y(*_):
            (x0, x1), _ = p1.vb.viewRange()
            a, b = max(0, int(np.floor(x0))), min(n, int(np.ceil(x1)) + 1)
            if b - a < 2:
                return
            lo, hi = np.nanmin(lo_all[a:b]), np.nanmax(hi_all[a:b])
            for name, (v, item) in cmp_items.items():   # rebase the comparison to the first visible bar
                w = v[a:b]
                k = np.flatnonzero(np.isfinite(w) & np.isfinite(c[a:b]))
                if len(k):
                    r = v * (c[a + k[0]] / v[a + k[0]])
                    item.setData(x, f(r), connect="finite")
                    lo, hi = min(lo, np.nanmin(r[a:b])), max(hi, np.nanmax(r[a:b]))
            near = (hi - lo) * 0.6                       # plan levels widen the view only when they are close
            for v in plan_levels:
                if lo - near <= v <= hi + near:
                    lo, hi = min(lo, v), max(hi, v)
            if fib_lines:
                fb = C.fibonacci(df.iloc[a:b])
                for ln, (lab, val) in zip(fib_lines, fb["levels"].items()):
                    ln.setValue(f(val))
                    ln.label.setFormat(f"{lab}  ₹{val:,.1f}")
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                lo_t, hi_t = (f(lo), f(hi))
                pad = (hi_t - lo_t) * 0.06
                p1.setYRange(lo_t - pad, hi_t + pad, padding=0)
            for pp in plots[1:]:
                if pp._key == "volume":
                    vmax = np.nanmax(vol[a:b]) if b > a else 0
                    pp.setYRange(0, max(vmax, 1) * 1.05, padding=0)
                elif pp._key == "macd":
                    ml, sg, hp = pp._macd
                    seg = np.r_[ml[a:b], sg[a:b], hp[a:b]]
                    seg = seg[np.isfinite(seg)]
                    if len(seg):
                        m = max(abs(seg.min()), abs(seg.max()), 1e-9) * 1.1
                        pp.setYRange(-m, m, padding=0)
        self._setup_nav(plots)
        for pp in plots:
            pp.disableAutoRange(axis="y")
            pp.setAutoVisible(y=False)
        for pp in plots[1:]:
            if pp._key == "rsi":
                pp.setYRange(0, 100, padding=0)
        p1.sigXRangeChanged.connect(fit_y)
        fit_y()
