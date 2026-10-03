"""Runs the intraday paper agent (live market hours, or a replay over history) in its own process, so model work
never touches the UI thread. Auto-starts at 09:05 IST on weekdays when enabled; a watchdog restarts a live run that
died during market hours (at most 3 times a day). Stopping writes a stop file the agent checks every few seconds."""
from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from datetime import datetime

from PySide6.QtCore import QObject, QSettings, QTimer, Signal

from ..livefeed import IST

log = logging.getLogger("niveshrl.agent")


def _agent_main(mode: str, stop_path: str, q) -> None:     # child process
    try:
        from pathlib import Path
        from niveshrl.desktop.worker import _child_logging
        plog = _child_logging()
        plog.info("intraday agent %s started", mode)
        if mode == "replay":
            from niveshrl.intraday import replay, universe
            if not replay.BARS.exists():
                u = universe.load()
                replay.download(u.index.tolist(), progress=lambda f: q.put(("progress", f"downloading {f:.0%}")))
            out = replay.run(progress=lambda day, r, f: q.put(("progress", f"replay {f:.0%} · {str(day)[:10]} · pool ₹{r.pool_end:,.0f}")))
            q.put(("done", {"mode": "replay", "summary": out["summary"]}))
        else:
            from niveshrl.intraday.live import run_live
            out = run_live(stop_flag=lambda: Path(stop_path).exists())
            q.put(("done", {"mode": "live", **out}))
    except BaseException:
        q.put(("error", traceback.format_exc()[-2000:]))


class AgentRunner(QObject):
    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        from ..intraday import DIR
        self.dir = DIR
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.q = None
        self.mode = ""
        self.restarts_today = (None, 0)
        self.settings = QSettings("NiveshRL", "NiveshRL")
        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._check)
        self._clock = QTimer(self)
        self._clock.setInterval(30_000)
        self._clock.timeout.connect(self._schedule)
        self._clock.start()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    @property
    def auto(self) -> bool:
        return self.settings.value("agent_auto", False, type=bool)

    def set_auto(self, on: bool) -> None:
        self.settings.setValue("agent_auto", bool(on))

    def start(self, mode: str = "live") -> bool:
        if self.running:
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        stop = self.dir / "STOP"
        if stop.exists():
            stop.unlink()
        self.q = self.ctx.Queue()
        self.proc = self.ctx.Process(target=_agent_main, args=(mode, str(stop), self.q), name=f"niveshrl-agent-{mode}",
                                     daemon=True)
        self.proc.start()
        self.mode = mode
        self._poll.start()
        log.info("agent %s started pid=%s", mode, self.proc.pid)
        self.progress.emit(f"{mode} starting…")
        return True

    def stop(self) -> None:
        (self.dir / "STOP").write_text("stop", encoding="utf-8")
        log.info("agent stop requested")

    def _check(self) -> None:
        import queue
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "progress":
                    self.progress.emit(msg[1])
                elif msg[0] == "done":
                    self._end()
                    self.finished.emit(msg[1])
                    return
                elif msg[0] == "error":
                    self._end()
                    log.error("agent error: %s", msg[1])
                    self.failed.emit(msg[1])
                    return
        except (queue.Empty, EOFError, OSError):
            pass
        if self.proc is not None and not self.proc.is_alive():
            code, mode = self.proc.exitcode, self.mode
            self._end()
            log.error("agent process exited unexpectedly (code %s)", code)
            self.failed.emit(f"Agent process exited unexpectedly (code {code}).")
            self._maybe_restart(mode)

    def _end(self) -> None:
        self._poll.stop()
        self.proc, self.q = None, None

    def _maybe_restart(self, mode: str) -> None:
        n = datetime.now(IST)
        day, k = self.restarts_today
        k = k if day == n.date() else 0
        if mode == "live" and n.weekday() < 5 and (9 * 60 + 5) <= n.hour * 60 + n.minute < 15 * 60 + 20 and k < 3:
            self.restarts_today = (n.date(), k + 1)
            log.warning("watchdog: restarting the live agent (#%d today)", k + 1)
            self.start("live")

    def _schedule(self, now: datetime | None = None) -> bool:
        n = now or datetime.now(IST)
        if not self.auto or self.running or n.weekday() >= 5:
            return False
        hm = n.hour * 60 + n.minute
        if not (9 * 60 + 5 <= hm < 15 * 60 + 15):
            return False
        today = n.date().isoformat()
        if self.settings.value("agent_last_auto", "") == today:
            return False
        self.settings.setValue("agent_last_auto", today)
        return self.start("live")
