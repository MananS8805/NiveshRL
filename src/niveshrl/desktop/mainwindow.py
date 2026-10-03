"""Main window: command bar, live ticker strip, a sidebar that shows one screen at a time, status bar,
alerts and watchdogs.

Threading model (nothing slow runs on the UI thread):
  * UI thread        painting + cheap overlays of live ticks on precomputed frames
  * feed thread      Yahoo WebSocket -> C++ TickStore + dict (``livefeed.Feed``)
  * Qt thread pool   file loads, Yahoo fundamentals, backtests, RL plan (``widgets.run_async``)
  * worker process   the daily pipeline: FinBERT, LightGBM/DL, briefing (``worker.PipelineRunner``)
"""
from __future__ import annotations

import gc
import logging
import os
import time
from datetime import datetime

import psutil
from PySide6.QtCore import QByteArray, QObject, QSettings, QSize, QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QPainter, QTextDocument
from PySide6.QtWidgets import (QCompleter, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSizePolicy, QStackedWidget, QToolBar,
                               QWidget)

from .. import watchlist as wl
from ..livefeed import IST, Feed, market_open
from . import data, theme
from .panels import Panel, vbox
from .explain import ExplainPanel
from .panels.agent import AgentPanel
from .panels.desk import DeskPanel
from .panels.glossary import GlossaryPanel
from .panels.lab import LabPanel
from .panels.market import MarketBase, MarketPanel, ticker_strip_text
from .panels.research import PlanPanel, RankersPanel, RiskPanel
from .panels.screener import ScreenerPanel
from .panels.stock import StockPanel
from .panels.today import TodayPanel
from .panels.watchlist import WatchlistPanel, watch_rows
from . import widgets
from .widgets import run_async
from .worker import PipelineRunner, Scheduler

log = logging.getLogger("niveshrl.ui")
APPDATA = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "NiveshRL")
STALE_RESTART_S = 300          # restart the feed if no tick for 5 min while the market is open


class AppContext(QObject):
    watchlist_changed = Signal()
    daily_updated = Signal()
    alert = Signal(str, str)                  # (title, message)

    def __init__(self):
        super().__init__()
        self.feed: Feed | None = None
        self.runner = PipelineRunner(self)
        self.scheduler = Scheduler(self.runner, parent=self)
        from .agent_worker import AgentRunner
        self.agent = AgentRunner(self)
        self.feed_restarts = 0


class TickerStrip(QWidget):
    """Continuously scrolling tape of index and watchlist quotes (rendered rich text, 30 fps)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(400)
        self.doc = QTextDocument()
        self.doc.setDefaultStyleSheet(f"body {{ color:{theme.TEXT}; font-family:{theme.MONO}; font-size:12px; }}")
        self.offset = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._step)
        self.timer.start()

    def set_html(self, html: str) -> None:
        if html != getattr(self, "_html", None):
            self._html = html
            self.doc.setHtml(f"<body><nobr>{html} &nbsp;│&nbsp; </nobr></body>")
            self.doc.setTextWidth(-1)

    def _step(self) -> None:
        if not self.isVisible():
            return
        self.offset += 1.0
        w = self.doc.idealWidth()
        if w > 0 and self.offset > w:
            self.offset -= w
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        w = self.doc.idealWidth()
        if w <= 0:
            return
        x = -self.offset
        while x < self.width():
            p.save()
            p.translate(x, 1)
            self.doc.drawContents(p)
            p.restore()
            x += w


class AlertsPanel(Panel):
    title = "Alerts"
    code = "ALRT"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        lay.addWidget(QLabel("Watchlist alerts appear here (newest first) and as Windows notifications: price at your "
                             "buy/sell target, a ±3% day move, a news-sentiment flip, results within 7 days, or a stock "
                             "entering/leaving tomorrow's top list. Each alert fires once per day. Double-click one to "
                             "open the stock."))
        lay.itemAt(0).widget().setWordWrap(True)
        lay.itemAt(0).widget().setObjectName("muted")
        self.list = QListWidget()
        self.empty = QListWidgetItem("No alerts yet today. Add stocks to your Watchlist (with optional buy/sell targets) "
                                     "to get alerts.")
        self.list.addItem(self.empty)
        lay.addWidget(self.list)
        self.list.itemDoubleClicked.connect(lambda it: it.data(Qt.UserRole) and self.stock_selected.emit(it.data(Qt.UserRole)))

    def add(self, ticker: str, text: str) -> None:
        if self.empty is not None:
            self.list.takeItem(self.list.row(self.empty))
            self.empty = None
        it = QListWidgetItem(f"{datetime.now(IST):%H:%M:%S}  {ticker.replace('.NS', ''):<12} {text}")
        it.setData(Qt.UserRole, ticker if ticker.endswith(".NS") else None)
        self.list.insertItem(0, it)
        while self.list.count() > 500:                    # bounded
            self.list.takeItem(self.list.count() - 1)


PANELS = [MarketPanel, TodayPanel, ScreenerPanel, WatchlistPanel, LabPanel, RankersPanel, RiskPanel, PlanPanel,
          AlertsPanel, GlossaryPanel, DeskPanel, AgentPanel]
# Sidebar order: (code, label). Ctrl+1 … Ctrl+0 jump to these in order.
NAV = [("MKT", "Market"), ("TODAY", "Today"), ("SCRN", "Screener"), ("WATCH", "Watchlist"), ("DESK", "My desk"),
       ("ALGO", "Intraday agent"),
       ("DES", "Stock"),
       ("RANK", "Rankers"), ("LAB", "Backtest lab"), ("RISK", "Risk"), ("PLAN", "RL & plan"), ("ALRT", "Alerts"),
       ("GLOS", "Glossary")]


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext | None = None, start_feed: bool = True):
        super().__init__()
        self.ctx = ctx or AppContext()
        self.start_feed = start_feed
        self.setWindowTitle("NiveshRL · trading desk")
        self.resize(1600, 960)
        self.settings = QSettings(os.path.join(APPDATA, "layout.ini"), QSettings.IniFormat)
        self.panels: dict[str, Panel] = {}
        self.history: list[str] = []                 # pages visited, for Back (Alt+Left)
        self.base: MarketBase | None = None
        self._ticks_prev = (time.time(), 0)
        self._alerted: set[tuple[str, str, str]] = set()
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self._build_toolbar()
        self.strip = TickerStrip()
        tb2 = QToolBar("Ticker")
        tb2.setMovable(False)
        tb2.addWidget(self.strip)
        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, tb2)
        self._build_pages()
        self._build_status()
        self._wire()
        self.loading = QLabel("Loading NIFTY 200 panel…")
        self.statusBar().addWidget(self.loading)
        run_async(data.panel, self._panel_ready, on_error=lambda e: self._fatal(e))

    # ------------------------------------------------------------------ construction
    def _build_toolbar(self) -> None:
        tb = QToolBar("Command")
        tb.setMovable(False)
        tb.setObjectName("command")
        logo = QLabel(f"<b style='color:{theme.AMBER}'>NIVESH</b><b>RL</b> ")
        tb.addWidget(logo)
        self.back_btn = QPushButton("← Back")
        self.back_btn.setToolTip("Previous screen (Alt+Left)")
        self.back_btn.clicked.connect(self.go_back)
        self.back_btn.setEnabled(False)
        tb.addWidget(self.back_btn)
        self.cmd = QLineEdit()
        self.cmd.setPlaceholderText("Command: RELIANCE <Enter> · MKT · TODAY · SCRN · WATCH · LAB · RANK · RISK · PLAN · ALRT · HELP   (Ctrl+K)")
        self.cmd.setMinimumWidth(520)
        self.cmd.returnPressed.connect(self._command)
        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.cmd.setCompleter(self.completer)
        tb.addWidget(self.cmd)
        self.refresh_btn = QPushButton("⟳ Refresh today's data (F5)")
        self.refresh_btn.setObjectName("primary")
        self.refresh_btn.clicked.connect(self.run_pipeline)
        tb.addWidget(self.refresh_btn)
        self.prog = QProgressBar()
        self.prog.setMaximumWidth(260)
        self.prog.setVisible(False)
        tb.addWidget(self.prog)
        self.addToolBar(Qt.TopToolBarArea, tb)
        keys = [("Ctrl+K", lambda: (self.cmd.setFocus(), self.cmd.selectAll())), ("F5", self.run_pipeline),
                ("Alt+Left", self.go_back), ("F1", lambda: self.show_panel("GLOS")),
                ("Escape", lambda: self.explain_panel.hide_panel())]
        keys += [(f"Ctrl+{(i + 1) % 10}", lambda c=c: self.show_panel(c)) for i, (c, _) in enumerate(NAV[:10])]
        for key, fn in keys:
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(fn)
            self.addAction(a)
        self.view_menu = self.menuBar().addMenu("&Go")
        opts = self.menuBar().addMenu("&Options")
        self.auto_act = QAction("Run daily pipeline automatically at 16:00 (Mon–Fri)", self, checkable=True)
        self.auto_act.setChecked(self.ctx.scheduler.enabled)
        self.auto_act.toggled.connect(self.ctx.scheduler.set_enabled)
        opts.addAction(self.auto_act)
        self.tray_act = QAction("Keep running in the tray when the window is closed", self, checkable=True)
        self.tray_act.setChecked(QSettings("NiveshRL", "NiveshRL").value("close_to_tray", True, type=bool))
        self.tray_act.toggled.connect(lambda on: QSettings("NiveshRL", "NiveshRL").setValue("close_to_tray", on))
        opts.addAction(self.tray_act)
        helpm = self.menuBar().addMenu("&Help")
        about = QAction("About NiveshRL", self)
        about.triggered.connect(self._about)
        helpm.addAction(about)

    def _build_pages(self) -> None:
        """Sidebar on the left; the selected screen fills the rest of the window (one screen at a time)."""
        self.stock = StockPanel(self.ctx)
        for cls in PANELS:
            pnl = cls(self.ctx)
            self.panels[pnl.code] = pnl
        self.panels[self.stock.code] = self.stock
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(170)
        self.nav.setIconSize(QSize(0, 0))
        self.pages = QStackedWidget()
        self.page_index: dict[str, int] = {}
        for i, (code, label) in enumerate(NAV):
            it = QListWidgetItem(f"{label}")
            it.setData(Qt.UserRole, code)
            it.setToolTip(f"{label}  ·  {code}" + (f"  ·  Ctrl+{(i + 1) % 10}" if i < 10 else "  ·  F1"))
            self.nav.addItem(it)
            self.page_index[code] = self.pages.addWidget(self.panels[code])
            act = QAction(f"{label}\t{code}", self)
            act.triggered.connect(lambda _=False, c=code: self.show_panel(c))
            self.view_menu.addAction(act)
            self.panels[code].stock_selected.connect(self.open_stock)
        self.nav.currentRowChanged.connect(lambda r: r >= 0 and self.show_panel(NAV[r][0]))
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.nav)
        lay.addWidget(self.pages, 1)
        self.explain_panel = ExplainPanel()
        lay.addWidget(self.explain_panel)
        self.setCentralWidget(central)
        widgets.EXPLAIN = self.explain
        geo = self.settings.value("geometry")
        if isinstance(geo, QByteArray):
            self.restoreGeometry(geo)
        start = self.settings.value("page", "MKT")
        self.current = None
        self.show_panel(start if start in self.panels and start != "DES" else "MKT", record=False)

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.s_feed, self.s_ticks, self.s_pipe, self.s_data, self.s_sys, self.s_clock = (QLabel() for _ in range(6))
        for w in (self.s_feed, self.s_ticks, self.s_pipe, self.s_data):
            sb.addWidget(w)
        sb.addPermanentWidget(self.s_sys)
        sb.addPermanentWidget(self.s_clock)

    def _wire(self) -> None:
        r = self.ctx.runner
        r.progress.connect(self._pipe_progress)
        r.finished.connect(self._pipe_done)
        r.failed.connect(self._pipe_failed)
        self.ctx.watchlist_changed.connect(self._watchlist_changed)
        self.t_tick = QTimer(self)
        self.t_tick.setInterval(1000)
        self.t_tick.timeout.connect(self._tick)
        self.t_status = QTimer(self)
        self.t_status.setInterval(2000)
        self.t_status.timeout.connect(self._status)
        self.t_status.start()
        self.t_alerts = QTimer(self)
        self.t_alerts.setInterval(60_000)
        self.t_alerts.timeout.connect(self._check_alerts)
        self.t_house = QTimer(self)
        self.t_house.setInterval(10 * 60_000)
        self.t_house.timeout.connect(self._housekeeping)
        self.t_house.start()
        self.t_watch = QTimer(self)
        self.t_watch.setInterval(30_000)
        self.t_watch.timeout.connect(self._watchdog)
        self.t_watch.start()

    # ------------------------------------------------------------------ startup
    def _panel_ready(self, p) -> None:
        self.loading.setText("")
        self.statusBar().removeWidget(self.loading)
        self.base = MarketBase()
        names = [f"{t.replace('.NS', '')} · {str(p.names.get(t, ''))[:30]}" for t in sorted(p.tickers)]
        codes = [f"{c} · {pp.title}" for c, pp in self.panels.items()] + ["HELP · keyboard and commands"]
        self.completer.setModel(QStringListModel(codes + names, self.completer))
        if self.start_feed:
            self._start_feed()
        self.panels[self.current].ensure_loaded()
        self._refresh_strip()
        self.t_tick.start()
        self.t_alerts.start()
        QTimer.singleShot(5000, self._check_alerts)
        log.info("panel ready: %d tickers, last date %s", len(p.tickers), p.close.index[-1].date())

    def _start_feed(self) -> None:
        if self.ctx.feed is not None:
            self.ctx.feed.stop()
        self.ctx.feed = Feed(data.panel().tickers).start()
        self._ticks_prev = (time.time(), 0)
        log.info("feed started (%s)", "C++ TickStore" if self.ctx.feed.store is not None else "dict store")

    def _fatal(self, msg: str) -> None:
        log.error("startup failed: %s", msg)
        QMessageBox.critical(self, "NiveshRL", "Could not load the price panel.\n\nRun demo.bat (or "
                             "python scripts/build_panel.py) first.\n\n" + msg[-600:])

    # ------------------------------------------------------------------ navigation
    def open_stock(self, ticker: str) -> None:
        if not ticker or ticker not in data.panel().tickers:
            return
        self.stock._loaded = True
        self.stock.show_stock(ticker)
        self.show_panel("DES")

    def show_panel(self, code: str, record: bool = True) -> None:
        """Show one screen; everything else is hidden (and stops refreshing)."""
        if code not in self.page_index:
            return
        if record and self.current and self.current != code:
            self.history.append(self.current)
            self.history = self.history[-50:]
        self.current = code
        self.pages.setCurrentIndex(self.page_index[code])
        row = [c for c, _ in NAV].index(code)
        if self.nav.currentRow() != row:
            self.nav.blockSignals(True)
            self.nav.setCurrentRow(row)
            self.nav.blockSignals(False)
        self.back_btn.setEnabled(bool(self.history))
        if self.base is not None:
            self.panels[code].ensure_loaded()
            self.panels[code].on_tick()

    def explain(self, label, value=None, ticker=None) -> bool:
        """Open the Explain panel for a label or glossary key; False if the glossary has nothing on it."""
        e = widgets.term_for(label)
        if e is None:
            self.statusBar().showMessage(f"No glossary entry for '{label}'.", 3000)
            return False
        self.explain_panel.show_entry(e, value, ticker)
        return True

    def go_back(self) -> None:
        if not self.history:
            return
        self.show_panel(self.history.pop(), record=False)
        self.back_btn.setEnabled(bool(self.history))

    def _command(self) -> None:
        txt = self.cmd.text().strip()
        self.cmd.clear()
        if not txt:
            return
        head = txt.split("·")[0].strip().split()[0].upper()
        if head in self.panels:
            self.show_panel(head)
            return
        if head == "HELP":
            self._about()
            return
        t = head if head.endswith(".NS") else head + ".NS"
        if self.base is not None and t in data.panel().tickers:
            self.open_stock(t)
            return
        p = data.panel()
        hits = [k for k, v in p.names.items() if txt.lower() in str(v).lower()]
        if hits:
            self.open_stock(hits[0])
        else:
            self.statusBar().showMessage(f"Unknown command or symbol: {txt}", 4000)

    # ------------------------------------------------------------------ timers
    def _visible_panels(self):
        if self.current and self.isVisible():
            yield self.panels[self.current]

    def _tick(self) -> None:
        t0 = time.perf_counter()
        for p in self._visible_panels():
            if p._loaded:
                try:
                    p.on_tick()
                except Exception:
                    log.exception("on_tick failed in %s", p.code)
        self._strip_n = getattr(self, "_strip_n", 0) + 1
        if self._strip_n % 2 == 0:
            self._refresh_strip()
        self._ui_ms = (time.perf_counter() - t0) * 1000

    def _refresh_strip(self) -> None:
        try:
            mt = wl.PATH.stat().st_mtime
        except OSError:
            mt = 0
        if getattr(self, "_wl_mtime", None) != mt:      # re-read the watchlist file only when it changes
            self._wl_mtime, self._wl_syms = mt, sorted(wl.load())[:30]
        syms = self._wl_syms or (data.panel().tickers[:20] if self.base else [])
        self.strip.set_html(ticker_strip_text(self.ctx.feed, self.base, syms))

    def _status(self) -> None:
        f = self.ctx.feed
        if f is not None:
            text, lvl = f.status()
            col = {"live": theme.GREEN, "stale": theme.AMBER, "closed": theme.MUTED, "offline": theme.RED}[lvl]
            self.s_feed.setText(f"<span style='color:{col}'>● {text}</span>")
            now = time.time()
            t_prev, n_prev = self._ticks_prev
            rate = (f.n_msgs - n_prev) / max(now - t_prev, 1e-6)
            self._ticks_prev = (now, f.n_msgs)
            self.s_ticks.setText(f"{rate:.1f} ticks/s · {f.n_msgs:,} total")
        r = self.ctx.runner
        if r.running:
            self.s_pipe.setText(f"pipeline: {r.state[0]} {r.state[1]:.0%}")
        else:
            nxt = "auto 16:00 on" if self.ctx.scheduler.enabled else "auto off"
            self.s_pipe.setText(f"pipeline: idle · {nxt}")
        m = data.daily.latest_meta()
        self.s_data.setText(f"data {m['trading_day']} (ran {m['ran_at'][5:16].replace('T', ' ')})" if m else "no daily data")
        mem = self.proc.memory_info().rss / 2 ** 20
        cpu = self.proc.cpu_percent(None) / max(psutil.cpu_count() or 1, 1)
        ui = getattr(self, "_ui_ms", 0.0)
        self.s_sys.setText(f"CPU {cpu:.0f}% · RAM {mem:,.0f} MB · UI {ui:.0f} ms")
        self.s_clock.setText(datetime.now(IST).strftime("%a %d %b %H:%M:%S IST"))

    def _watchdog(self) -> None:
        f = self.ctx.feed
        if f is None or not self.start_feed:
            return
        restart = None
        if not f.alive:
            restart = "feed thread died"
        elif market_open() and f.last_msg_at and time.time() - f.last_msg_at > STALE_RESTART_S:
            restart = f"no ticks for {time.time() - f.last_msg_at:.0f}s during market hours"
        if restart:
            self.ctx.feed_restarts += 1
            log.warning("watchdog: restarting feed (%s), restart #%d", restart, self.ctx.feed_restarts)
            self._start_feed()

    def _housekeeping(self) -> None:
        data.prune_fundamentals()
        data.drop_stale_daily()
        gc.collect()
        log.info("housekeeping: rss=%.0f MB cache=%d entries feed_msgs=%s restarts=%d",
                 self.proc.memory_info().rss / 2 ** 20, data.cache_size(),
                 self.ctx.feed.n_msgs if self.ctx.feed else 0, self.ctx.feed_restarts)

    def _check_alerts(self) -> None:
        if self.base is None:
            return
        try:
            _, alerts = watch_rows(self.ctx.feed)
        except Exception:
            log.exception("alert check failed")
            return
        today = datetime.now(IST).date().isoformat()
        for tk, items in alerts.items():
            for a in items:
                key = (today, tk, a.split(" ")[0] + a.split(" ")[-1])
                if key in self._alerted:
                    continue
                self._alerted.add(key)
                self.panels["ALRT"].add(tk, a)
                self.ctx.alert.emit(f"NiveshRL · {tk.replace('.NS', '')}", a)
                log.info("alert %s: %s", tk, a)
        self._alerted = {k for k in self._alerted if k[0] == today}           # bounded: today only

    # ------------------------------------------------------------------ pipeline
    def run_pipeline(self) -> None:
        if self.ctx.runner.running:
            self.statusBar().showMessage("The daily pipeline is already running.", 3000)
            return
        self.ctx.runner.start()

    def _pipe_progress(self, step: str, frac: float) -> None:
        self.prog.setVisible(True)
        self.prog.setValue(int(frac * 100))
        self.prog.setFormat(f"{step} %p%")
        self.refresh_btn.setEnabled(False)

    def _pipe_done(self, meta: dict) -> None:
        self.prog.setVisible(False)
        self.refresh_btn.setEnabled(True)
        data.clear("daily:")

        def reload_panel():                       # prices step updated the parquet: reload off the UI thread
            data.clear("panel")
            data.panel()
            return MarketBase()
        run_async(reload_panel, self._panel_reloaded)
        bad = [k for k, v in meta.get("steps", {}).items() if not v.get("ok")]
        msg = f"Daily data refreshed for {meta.get('trading_day')}" + (f" · failed: {', '.join(bad)}" if bad else "")
        self.statusBar().showMessage(msg, 10_000)
        self.ctx.alert.emit("NiveshRL", msg)

    def _panel_reloaded(self, base) -> None:
        self.base = base
        for p in self.panels.values():
            p.on_daily_update()
        self.ctx.daily_updated.emit()

    def _pipe_failed(self, msg: str) -> None:
        self.prog.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.statusBar().showMessage("Daily pipeline failed: " + msg.strip().splitlines()[-1][:160], 15_000)

    def _watchlist_changed(self) -> None:
        self.panels["WATCH"].refresh()
        if self.panels["TODAY"]._loaded:
            self.panels["TODAY"].refresh()
        self._refresh_strip()

    # ------------------------------------------------------------------ misc
    def _about(self) -> None:
        QMessageBox.information(self, "NiveshRL", (
            "NiveshRL trading desk\n\n"
            "Commands (Ctrl+K): type a symbol (RELIANCE) or a screen code: MKT, TODAY, SCRN, WATCH, LAB, RANK, RISK, "
            "PLAN, ALRT, DES, GLOS.\nPick a screen in the left menu (or Ctrl+1 … Ctrl+0; F1 = Glossary); Alt+Left goes back.\n"
            "Explanations: click any label marked ⓘ, a KPI tile or a chart's 'What am I looking at?' button; right-click "
            "a table header or value. Esc closes the Explain panel.\n"
            "Charts: mouse wheel zooms time, drag pans, the range buttons (1M … All) jump, double-click resets.\n"
            "F5 runs the daily pipeline (prices, fundamentals, news + FinBERT, next-day model, briefing).\n\n"
            "Educational project, not investment advice. Not registered with SEBI."))

    def save_layout(self) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        if self.current:
            self.settings.setValue("page", self.current)
        self.settings.sync()

    def shutdown(self) -> None:
        self.save_layout()
        for t in (self.t_tick, self.t_status, self.t_alerts, self.t_house, self.t_watch, self.strip.timer):
            t.stop()
        if self.ctx.feed is not None:
            self.ctx.feed.stop()
        if self.ctx.runner.running:
            self.ctx.runner.stop()
        if self.ctx.agent.running:                        # let the paper agent save and exit cleanly
            self.ctx.agent.stop()

    def closeEvent(self, e):
        self.save_layout()
        if getattr(self, "tray", None) is not None and self.tray_act.isChecked() and not getattr(self, "_quitting", False):
            e.ignore()
            self.hide()
            self.tray.notify_hidden()
            return
        self.shutdown()
        e.accept()
