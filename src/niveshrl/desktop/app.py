"""Entry point: ``python -m niveshrl.desktop [--tray] [--no-feed]`` or NiveshRL.exe.

Sets up rotating logs in %APPDATA%\\NiveshRL\\logs, a global exception hook (an
error in one panel is logged, never kills the app), single-instance handling
(a second launch just brings the running window forward), the tray icon and
the main window.
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import traceback


def _setup_logging() -> str:
    from .mainwindow import APPDATA
    d = os.path.join(APPDATA, "logs")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "niveshrl.log")
    h = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(h)
    if sys.stderr is not None:
        root.addHandler(logging.StreamHandler())
    return path


def seed_user_data() -> str | None:
    """Packaged app: copy the bundled research data (``<install>/seed``) to the writable
    ``config.ROOT`` on first run, without overwriting anything the user already has."""
    import shutil
    from pathlib import Path

    from ..config import ROOT
    if not getattr(sys, "frozen", False):
        return None
    seed = Path(sys.executable).parent / "seed"
    if not seed.exists():
        return None
    copied = 0
    for src in seed.rglob("*"):
        if src.is_file():
            dst = ROOT / src.relative_to(seed)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
    return f"seeded {copied} files into {ROOT}" if copied else None


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    ap = argparse.ArgumentParser(prog="NiveshRL")
    ap.add_argument("--tray", action="store_true", help="start minimised to the tray")
    ap.add_argument("--no-feed", action="store_true", help="don't connect the live price stream")
    ap.add_argument("--selftest-pipeline", metavar="STEPS",
                    help="run these daily-pipeline steps (comma-separated) in the worker process, log, and exit")
    args, _ = ap.parse_known_args(argv)

    from PySide6.QtCore import Qt
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    log_path = _setup_logging()
    log = logging.getLogger("niveshrl")

    def hook(exc_type, exc, tb):
        log.error("unhandled exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
    sys.excepthook = hook
    try:
        msg = seed_user_data()
        if msg:
            log.info(msg)
    except OSError:
        log.exception("could not seed user data")

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("NiveshRL")
    app.setOrganizationName("NiveshRL")
    app.setQuitOnLastWindowClosed(False)

    # single instance: a second launch asks the first to show itself
    sock = QLocalSocket()
    sock.connectToServer("NiveshRL-desk")
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.flush()
        sock.waitForBytesWritten(300)
        return 0
    server = QLocalServer()
    QLocalServer.removeServer("NiveshRL-desk")
    server.listen("NiveshRL-desk")

    from . import theme
    from .mainwindow import MainWindow
    from .tray import Tray, make_icon

    app.setStyleSheet(theme.QSS)
    app.setWindowIcon(make_icon())
    win = MainWindow(start_feed=not args.no_feed)
    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = Tray(win, app)
        tray.show()
    win.tray = tray

    def on_conn():
        c = server.nextPendingConnection()
        c.readyRead.connect(lambda: tray.show_window() if tray else (win.show(), win.raise_()))
    server.newConnection.connect(on_conn)

    if not (args.tray and tray):
        win.show()
    else:
        win.hide()

    if args.selftest_pipeline:                       # packaging check: does the frozen worker process work?
        r = win.ctx.runner

        def done(meta):
            log.info("SELFTEST OK %s", {k: v.get("ok") for k, v in meta.get("steps", {}).items()})
            win._quitting = True
            win.shutdown()
            app.exit(0 if all(v.get("ok") for v in meta.get("steps", {}).values()) else 1)

        def failed(msg):
            log.error("SELFTEST FAILED %s", msg)
            win._quitting = True
            win.shutdown()
            app.exit(2)
        r.finished.connect(done)
        r.failed.connect(failed)
        r.start(steps=args.selftest_pipeline.split(","))

    app.aboutToQuit.connect(lambda: getattr(win, "_quitting", False) or win.shutdown())
    if tray is None:
        app.setQuitOnLastWindowClosed(True)
    log.info("NiveshRL desktop started (pid %s), log %s", os.getpid(), log_path)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
