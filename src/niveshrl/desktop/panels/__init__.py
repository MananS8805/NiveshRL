"""Dockable panels of the desktop app. Each panel is a QWidget with ``refresh()``
(re-read data) and, where it lists stocks, a ``stock_selected(str)`` signal."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget


class Panel(QWidget):
    stock_selected = Signal(str)
    title = "Panel"
    code = "PANEL"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx                       # AppContext: feed, runner, alerts
        self._loaded = False

    def ensure_loaded(self) -> None:
        """Build expensive content the first time the panel becomes visible."""
        if not self._loaded:
            self._loaded = True
            self.refresh()

    def refresh(self) -> None:               # re-read data from disk / caches
        pass

    def on_tick(self) -> None:               # called by the live timer while visible
        pass

    def on_daily_update(self) -> None:       # a new daily pipeline run landed
        if self._loaded:
            self.refresh()


def scrolling(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.setFrameShape(QScrollArea.NoFrame)
    return sa


def vbox(w: QWidget, margins: int = 8, spacing: int = 8) -> QVBoxLayout:
    lay = QVBoxLayout(w)
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    return lay
