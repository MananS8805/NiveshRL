"""The desktop app, its daily pipeline steps and the live market loop must run without torch or transformers: the
packaged app excludes them (packaging/niveshrl.spec). Research code may still use torch; the app may not."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import sys
for m in ("torch", "transformers", "stable_baselines3", "gymnasium"):
    sys.modules[m] = None                                   # any import of these now raises ImportError
import importlib
mods = ["niveshrl.research.daily", "niveshrl.research.news", "niveshrl.research.nextday", "niveshrl.research.stacked",
        "niveshrl.research.range_model", "niveshrl.research.volatility", "niveshrl.research.regime",
        "niveshrl.research.monitor", "niveshrl.research.swing_book", "niveshrl.research.delivery",
        "niveshrl.research.fundamentals", "niveshrl.research.technicals", "niveshrl.research.summary",
        "niveshrl.intraday.live", "niveshrl.intraday.copilot.engine", "niveshrl.intraday.algo.book", "niveshrl.dhan",
        "niveshrl.desktop.worker", "niveshrl.desktop.agent_worker"]
for m in mods:
    importlib.import_module(m)
from PySide6.QtWidgets import QApplication
app = QApplication([])
from niveshrl.desktop.mainwindow import NAV, MainWindow
w = MainWindow(start_feed=False)
for code, _ in NAV:
    if code in w.panels:
        w.show_panel(code)
        app.processEvents()
print("OK", len(mods), "modules,", len(NAV), "screens")
"""


def test_app_runs_without_torch_or_transformers():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-2000:], r.stderr[-4000:])
