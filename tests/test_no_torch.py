"""The desktop app, its daily pipeline steps and the live market loop must run without torch or transformers: the
packaged app excludes them (packaging/niveshrl.spec). Research code may still use torch; the app may not."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import sys
import importlib.abc


class _Block(importlib.abc.MetaPathFinder):                 # behaves as if these packages were not installed
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "transformers", "stable_baselines3", "gymnasium"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return None


sys.meta_path.insert(0, _Block())
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


def test_regime_vote_runs_without_torch():
    script = r"""
import sys
import importlib.abc


class _Block(importlib.abc.MetaPathFinder):                 # behaves as if these packages were not installed
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "transformers", "stable_baselines3", "gymnasium"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return None


sys.meta_path.insert(0, _Block())
import numpy as np, pandas as pd
from niveshrl.research.data import Panel
from niveshrl.research.regime import detect_regimes_vote
rng = np.random.default_rng(0)
idx = pd.bdate_range("2012-01-02", "2016-12-30")
c = pd.DataFrame(100 * np.exp(np.cumsum(0.01 * rng.standard_normal((len(idx), 30)), axis=0)), index=idx,
                 columns=[f"S{i}" for i in range(30)])
p = Panel(close=c, volume=c * 0 + 1e6, sectors=pd.Series("X", index=c.columns), names=pd.Series(c.columns, index=c.columns),
          bench=c.mean(axis=1), vix=pd.Series(15 + 5 * np.abs(np.sin(np.arange(len(idx)) / 50)), index=idx))
w = detect_regimes_vote(p, first_year=2016, seeds=(0, 1, 2))
assert set(w["regime"]) <= {"Bull", "Neutral", "Stress"} and len(w) > 40
print("OK", w["regime"].value_counts().to_dict())
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-1000:], r.stderr[-3000:])
