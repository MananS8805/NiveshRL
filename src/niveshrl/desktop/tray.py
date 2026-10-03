"""System tray icon: keeps the feed, alerts and the 16:00 pipeline alive while the window is closed."""
from __future__ import annotations

import sys

from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from . import theme

RUN_KEY = r"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run"


def make_icon(size: int = 256) -> QIcon:
    """The app icon, drawn in code (amber N on the terminal background) so no binary asset is needed."""
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, size):
        pm = QPixmap(s, s)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(theme.PANEL))
        p.setPen(QColor(theme.AMBER))
        r = max(1, s // 16)
        p.drawRoundedRect(QRectF(r / 2, r / 2, s - r, s - r), s * 0.18, s * 0.18)
        f = QFont("Segoe UI", int(s * 0.55), QFont.Bold)
        p.setFont(f)
        p.setPen(QColor(theme.AMBER))
        p.drawText(pm.rect(), Qt.AlignCenter, "N")
        p.setPen(QColor(theme.GREEN))
        p.setBrush(QColor(theme.GREEN))
        d = max(2, s // 7)
        p.drawEllipse(QRectF(s - d - r * 2, r * 2, d, d))
        p.end()
        icon.addPixmap(pm)
    return icon


def autostart_enabled() -> bool:
    return bool(QSettings(RUN_KEY, QSettings.NativeFormat).value("NiveshRL", ""))


def set_autostart(on: bool) -> None:
    """Start with Windows (minimised to tray). Per-user Run key; toggled only from the tray menu."""
    s = QSettings(RUN_KEY, QSettings.NativeFormat)
    if on:
        if getattr(sys, "frozen", False):
            cmd = f'"{sys.executable}" --tray'
        else:
            exe = sys.executable.replace("python.exe", "pythonw.exe")
            cmd = f'"{exe}" -m niveshrl.desktop --tray'
        s.setValue("NiveshRL", cmd)
    else:
        s.remove("NiveshRL")


class Tray(QSystemTrayIcon):
    def __init__(self, window, app: QApplication):
        super().__init__(make_icon(), app)
        self.window = window
        self.app = app
        self._hidden_told = False
        self.setToolTip("NiveshRL trading desk")
        m = QMenu()
        for text, fn in [("Open NiveshRL", self.show_window), ("Today's briefing", lambda: self._panel("TODAY")),
                         ("Watchlist", lambda: self._panel("WATCH")), (None, None),
                         ("⟳ Refresh today's data", window.run_pipeline), (None, None)]:
            if text is None:
                m.addSeparator()
                continue
            a = QAction(text, m)
            a.triggered.connect(fn)
            m.addAction(a)
        self.auto = QAction("Start with Windows (in tray)", m, checkable=True)
        self.auto.setChecked(autostart_enabled())
        self.auto.toggled.connect(set_autostart)
        m.addAction(self.auto)
        m.addSeparator()
        q = QAction("Quit", m)
        q.triggered.connect(self.quit)
        m.addAction(q)
        self.menu = m
        self.setContextMenu(m)
        self.activated.connect(lambda r: r in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick) and self.show_window())
        self.messageClicked.connect(lambda: self._panel("ALRT"))
        window.ctx.alert.connect(self.notify)

    def notify(self, title: str, msg: str) -> None:
        if self.isVisible():
            self.showMessage(title, msg, make_icon(), 8000)

    def notify_hidden(self) -> None:
        if not self._hidden_told:
            self._hidden_told = True
            self.showMessage("NiveshRL is still running",
                             "Live prices, watchlist alerts and the 16:00 pipeline keep running here. "
                             "Right-click the tray icon to quit.", make_icon(), 6000)

    def show_window(self) -> None:
        w = self.window
        w.show()
        w.setWindowState((w.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive)
        w.raise_()
        w.activateWindow()

    def _panel(self, code: str) -> None:
        self.show_window()
        self.window.show_panel(code)

    def quit(self) -> None:
        self.window._quitting = True
        self.window.shutdown()
        self.hide()
        self.app.quit()
