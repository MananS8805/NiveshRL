"""Model worker: runs the daily pipeline (prices, FinBERT, LightGBM/DL, briefing) in a separate process.

A fresh process per run keeps the UI responsive (no GIL contention) and returns
all model memory (FinBERT is ~1 GB in RAM) to the OS when the run ends, which
matters for an app left open all day. Progress comes back over a queue that the
UI polls with a QTimer; a watchdog notices a worker that died without reporting.

``Scheduler`` triggers the run automatically at 16:00 IST on weekdays, once per day.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import traceback
from datetime import datetime

from PySide6.QtCore import QObject, QSettings, QTimer, Signal

from ..livefeed import IST

log = logging.getLogger("niveshrl.worker")


def _pipeline_main(q, steps, with_seq) -> None:          # runs in the child process
    try:
        from niveshrl.research import daily
        meta = daily.run(steps=steps, with_seq=with_seq, progress=lambda s, f: q.put(("progress", s, float(f))))
        q.put(("done", meta))
    except BaseException:
        q.put(("error", traceback.format_exc()[-2000:]))


class PipelineRunner(QObject):
    progress = Signal(str, float)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = mp.get_context("spawn")
        self.proc: mp.Process | None = None
        self.q = None
        self.started_at: datetime | None = None
        self.state = ("idle", 0.0)
        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._poll)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, steps: list[str] | None = None, with_seq: bool = False) -> bool:
        if self.running:
            return False
        self.q = self.ctx.Queue()
        self.proc = self.ctx.Process(target=_pipeline_main, args=(self.q, steps, with_seq),
                                     name="niveshrl-pipeline", daemon=True)
        self.proc.start()
        self.started_at = datetime.now()
        self.state = ("starting", 0.0)
        self.progress.emit("starting", 0.0)
        self._timer.start()
        log.info("pipeline started pid=%s steps=%s", self.proc.pid, steps or "all")
        return True

    def stop(self) -> None:
        if self.running:
            self.proc.terminate()
            self.proc.join(5)
            log.warning("pipeline terminated by user")
        self._finish()

    def _finish(self) -> None:
        self._timer.stop()
        self.proc, self.q = None, None
        self.state = ("idle", 0.0)

    def _poll(self) -> None:
        got_end = False
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "progress":
                    self.state = (msg[1], msg[2])
                    self.progress.emit(msg[1], msg[2])
                elif msg[0] == "done":
                    got_end = True
                    log.info("pipeline done: %s", {k: v.get("ok") for k, v in msg[1].get("steps", {}).items()})
                    self.proc.join(10)
                    self._finish()
                    self.finished.emit(msg[1])
                    return
                elif msg[0] == "error":
                    got_end = True
                    log.error("pipeline error: %s", msg[1])
                    self.proc.join(10)
                    self._finish()
                    self.failed.emit(msg[1])
                    return
        except queue.Empty:
            pass
        except (EOFError, OSError):
            pass
        if not got_end and self.proc is not None and not self.proc.is_alive():      # watchdog
            code = self.proc.exitcode
            log.error("pipeline worker died (exit code %s) without reporting", code)
            self._finish()
            self.failed.emit(f"Worker process exited unexpectedly (code {code}).")


class Scheduler(QObject):
    """Starts the pipeline at ``at`` (HH:MM IST) on weekdays, once per calendar day."""
    due = Signal()

    def __init__(self, runner: PipelineRunner, at: str = "16:00", parent=None):
        super().__init__(parent)
        self.runner = runner
        self.at = at
        self.settings = QSettings("NiveshRL", "NiveshRL")
        self.timer = QTimer(self)
        self.timer.setInterval(30_000)
        self.timer.timeout.connect(self.check)
        self.timer.start()

    @property
    def enabled(self) -> bool:
        return self.settings.value("auto_pipeline", True, type=bool)

    def set_enabled(self, on: bool) -> None:
        self.settings.setValue("auto_pipeline", bool(on))

    def check(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(IST)
        if not self.enabled or now.weekday() >= 5:
            return False
        hh, mm = (int(x) for x in self.at.split(":"))
        if (now.hour, now.minute) < (hh, mm):
            return False
        today = now.date().isoformat()
        if self.settings.value("last_auto_run", "") == today or self.runner.running:
            return False
        self.settings.setValue("last_auto_run", today)
        log.info("scheduled pipeline run at %s", now.isoformat(timespec="seconds"))
        self.runner.start()
        self.due.emit()
        return True
