"""Desktop app smoke tests (pytest-qt, offscreen): every panel builds and refreshes on real data,
stock selection works, the screener's C++ filter path matches the reference, and the
scheduler/worker plumbing behaves. Skipped when PySide6 / pytest-qt or the price panel are missing."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["APPDATA"] = tempfile.mkdtemp(prefix="niveshrl_test_appdata_")   # keep layouts/logs out of the real profile

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from niveshrl.research.data import PANEL_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(not PANEL_PATH.exists(), reason="needs data/nifty200_panel.parquet")


@pytest.fixture(scope="module")
def window(qapp):
    from PySide6.QtCore import QSettings
    QSettings.setDefaultFormat(QSettings.IniFormat)
    from niveshrl.desktop import theme
    from niveshrl.desktop.mainwindow import MainWindow
    qapp.setStyleSheet(theme.QSS)
    w = MainWindow(start_feed=False)
    w.ctx.scheduler.set_enabled(False)
    w.show()
    yield w
    w._quitting = True
    w.shutdown()
    w.close()


def _wait_ready(qtbot, w):
    qtbot.waitUntil(lambda: w.base is not None, timeout=60_000)


def test_every_panel_loads(qtbot, window):
    _wait_ready(qtbot, window)
    for code, p in window.panels.items():
        if code == "DES":
            continue
        window.show_panel(code)
        p.ensure_loaded()
        assert p._loaded, code
        p.on_tick()


def test_market_heatmap_has_tiles(qtbot, window):
    _wait_ready(qtbot, window)
    mk = window.panels["MKT"]
    mk.ensure_loaded()
    mk.resize(1000, 700)
    mk.on_tick(force_layout=True)
    assert len(mk.tree.items) > 150
    assert mk.up.model_.rowCount() > 0


def test_open_stock_and_command_bar(qtbot, window):
    _wait_ready(qtbot, window)
    window.cmd.setText("RELIANCE")
    window._command()
    assert window.stock.ticker == "RELIANCE.NS"
    assert "RELIANCE" in window.stock.name.text()
    window.cmd.setText("SCRN")
    window._command()
    assert window.panels["SCRN"]._loaded


def test_screener_cpp_matches_reference(qtbot, window):
    from niveshrl.desktop import data
    from niveshrl.desktop.panels.screener import ColumnStore
    from niveshrl.research import screener as scr
    t = data.screener_table()
    if t is None:
        pytest.skip("no daily pipeline output")
    cs = ColumnStore(t)
    for name, (_, flt) in scr.PRESETS.items():
        assert list(cs.apply(flt).index) == list(scr.apply(t, flt).index), name


def test_treemap_layout_fills_area():
    from niveshrl.desktop.widgets import _squarify
    vals = sorted(np.random.default_rng(0).uniform(1, 10, 40), reverse=True)
    rects = _squarify(vals, 0, 0, 800, 500)
    area = sum(w * h for _, _, w, h in rects)
    assert abs(area - 800 * 500) < 1e-6
    for (x, y, w, h), v in zip(rects, vals):
        assert w * h == pytest.approx(v / sum(vals) * 800 * 500)
        assert -1e-9 <= x and x + w <= 800 + 1e-6 and -1e-9 <= y and y + h <= 500 + 1e-6


def test_frame_model_sorts_numbers_with_nans_last(qtbot):
    import pandas as pd
    from PySide6.QtCore import Qt
    from niveshrl.desktop.widgets import FrameTable
    t = FrameTable()
    qtbot.addWidget(t)
    t.set_frame(pd.DataFrame({"x": [3.0, np.nan, 1.0, 2.0]}, index=list("abcd")))
    t.sortByColumn(0, Qt.AscendingOrder)
    got = [t.proxy.index(i, 0).data(Qt.UserRole) for i in range(4)]
    assert got[:3] == [1.0, 2.0, 3.0] and np.isnan(got[3])


def test_scheduler_runs_once_per_weekday_after_four(qtbot, monkeypatch):
    from PySide6.QtCore import QSettings
    from niveshrl.desktop.worker import IST, PipelineRunner, Scheduler
    QSettings.setDefaultFormat(QSettings.IniFormat)
    r = PipelineRunner()
    started = []
    monkeypatch.setattr(r, "start", lambda *a, **k: started.append(1) or True)
    s = Scheduler(r)
    s.timer.stop()
    s.set_enabled(True)
    s.settings.setValue("last_auto_run", "")
    assert not s.check(datetime(2026, 10, 5, 15, 59, tzinfo=IST))      # Monday, before 16:00
    assert s.check(datetime(2026, 10, 5, 16, 1, tzinfo=IST))
    assert not s.check(datetime(2026, 10, 5, 17, 0, tzinfo=IST))       # already ran today
    assert not s.check(datetime(2026, 10, 10, 16, 30, tzinfo=IST))     # Saturday
    s.set_enabled(False)
    assert not s.check(datetime(2026, 10, 6, 16, 30, tzinfo=IST))
    assert started == [1]


def _worker_child(q, steps, with_seq):        # module-level: picklable for the spawn context
    q.put(("progress", "x", 0.5))
    q.put(("done", {"steps": {"x": {"ok": True}}, "trading_day": "2026-10-01"}))


def test_pipeline_runner_reports_progress_and_done(qtbot, monkeypatch):
    from niveshrl.desktop import worker
    monkeypatch.setattr(worker, "_pipeline_main", _worker_child)
    r = worker.PipelineRunner()
    got = []
    r.progress.connect(lambda s, f: got.append((s, f)))
    with qtbot.waitSignal(r.finished, timeout=60_000) as blocker:
        assert r.start()
    assert blocker.args[0]["trading_day"] == "2026-10-01"
    assert ("x", 0.5) in got
    assert not r.running


def test_frame_table_keeps_given_order_until_header_clicked(qtbot):
    """Regression: enabling sorting used to sort by column 0 descending, so ranked lists showed rank 194 first."""
    import pandas as pd
    from PySide6.QtCore import Qt
    from niveshrl.desktop.widgets import FrameTable
    t = FrameTable()
    qtbot.addWidget(t)
    t.set_frame(pd.DataFrame({"Rank": [1, 2, 3], "x": ["c", "a", "b"]}, index=["A", "B", "C"]))
    shown = [t.proxy.index(i, 0).data(Qt.UserRole) for i in range(3)]
    assert shown == [1, 2, 3]


def test_one_screen_at_a_time_and_back(qtbot, window):
    _wait_ready(qtbot, window)
    window.show_panel("MKT")
    window.show_panel("RANK")
    assert window.pages.currentWidget() is window.panels["RANK"]
    assert [p.code for p in window._visible_panels()] == ["RANK"]
    window.go_back()
    assert window.current == "MKT"


def test_rankers_live_ranking_starts_at_rank_one(qtbot, window):
    _wait_ready(qtbot, window)
    window.show_panel("RANK")
    rk = window.panels["RANK"]
    if rk.table.empty:
        pytest.skip("no ranker predictions")
    assert rk.live.proxy.index(0, 0).data() == "1"
    rk.search.setText("bank")
    assert all("bank" in (str(lbl) + rk.table.loc[lbl, "Company"]).lower() for lbl in rk.live.model_.labels)
    rk.search.setText("")


def test_price_chart_y_axis_fits_visible_window(qtbot):
    import pandas as pd
    from niveshrl.desktop.widgets import PriceChart
    c = PriceChart()
    qtbot.addWidget(c)
    idx = pd.bdate_range("2020-01-01", periods=1500)
    close = pd.Series(np.r_[np.full(1250, 4000.0), np.full(250, 2000.0)], index=idx)
    df = pd.DataFrame({"Open": close, "High": close + 10, "Low": close - 10, "Close": close, "Volume": 1.0})
    c.plot(df)
    c.set_range("6M")
    y0, y1 = c.plots[0].vb.viewRange()[1]
    assert y1 < 2500, "y-axis should fit the visible last 6 months (~2000), not the whole history (4000)"
