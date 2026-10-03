"""Dark terminal theme for the desktop app: same palette as the Streamlit dashboard."""
from __future__ import annotations

AMBER = "#FF9F1C"
GREEN = "#26D07C"
RED = "#FF4D4D"
BLUE = "#4DA3FF"
MUTED = "#8A94A6"
GRID = "#1C2430"
BG = "#07090C"
PANEL = "#10151C"
TEXT = "#E8E6E3"
SERIES = [AMBER, BLUE, GREEN, "#C77DFF", "#FF6B9A", "#5EEAD4", "#FACC15", "#94A3B8"]
MONO = "Cascadia Mono, Consolas, IBM Plex Mono, monospace"

QSS = f"""
* {{ font-family: {MONO}; font-size: 12px; color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {PANEL}; color: {AMBER}; padding: 4px 8px; border-bottom: 1px solid {GRID};
                      text-transform: uppercase; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {GRID}; top: -1px; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 5px 12px; border: 1px solid {GRID}; border-bottom: none; }}
QTabBar::tab:selected {{ color: {AMBER}; background: {BG}; border-bottom: 2px solid {AMBER}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {PANEL}; border: 1px solid {GRID}; padding: 4px 6px;
    selection-background-color: {AMBER}; selection-color: {BG}; }}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {AMBER}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: #2A2010; }}
QPushButton {{ background: {PANEL}; border: 1px solid {GRID}; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {AMBER}; color: {AMBER}; }}
QPushButton#primary {{ border-color: {AMBER}; color: {AMBER}; font-weight: 600; }}
QTableView {{ background: {BG}; alternate-background-color: #0B0F14; gridline-color: {GRID}; border: 1px solid {GRID};
              selection-background-color: #2A2010; selection-color: {TEXT}; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; border-right: 1px solid {GRID};
                        border-bottom: 1px solid {GRID}; padding: 4px 6px; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: {BG}; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {GRID}; min-height: 20px; min-width: 20px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {GRID}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; }}
QLabel#h1 {{ color: {AMBER}; font-size: 16px; font-weight: 600; }}
QLabel#h2 {{ color: {AMBER}; font-size: 13px; font-weight: 600; text-transform: uppercase; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#tape {{ background: {PANEL}; border: 1px solid {GRID}; border-left: 3px solid {AMBER}; padding: 5px 10px; }}
QFrame#kpi {{ background: {PANEL}; border: 1px solid {GRID}; }}
QTextBrowser {{ background: {BG}; border: 1px solid {GRID}; }}
QProgressBar {{ background: {PANEL}; border: 1px solid {GRID}; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {AMBER}; }}
QMenu {{ background: {PANEL}; border: 1px solid {GRID}; }}
QMenu::item:selected {{ background: #2A2010; color: {AMBER}; }}
QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {GRID}; }}
QMenuBar::item {{ background: transparent; color: {TEXT}; padding: 4px 10px; }}
QMenuBar::item:selected {{ background: #2A2010; color: {AMBER}; }}
QToolBar {{ background: {BG}; border: none; spacing: 6px; padding: 2px 4px; }}
QListWidget#nav {{ background: {PANEL}; border: none; border-right: 1px solid {GRID}; padding-top: 8px; outline: 0; }}
QListWidget#nav::item {{ color: {MUTED}; padding: 11px 16px; border-left: 3px solid transparent; font-size: 13px; }}
QListWidget#nav::item:hover {{ color: {TEXT}; background: #161C25; }}
QListWidget#nav::item:selected {{ color: {AMBER}; background: #1A1610; border-left: 3px solid {AMBER}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {AMBER}; }}
"""


def signed(v: float | None) -> str:
    if v is None or v != v:
        return MUTED
    return GREEN if v >= 0 else RED
