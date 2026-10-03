"""Long-run soak test for the desktop app.

Runs the real MainWindow (offscreen by default) for ``--minutes`` with either the
live Yahoo feed (``--live``, use during NSE hours) or a synthetic feed that pushes
random-walk ticks for all 194 stocks at ``--rate`` ticks/s through the same code
path (``Feed._on_message`` -> C++ TickStore). Every few seconds it switches panels
and opens a random stock, so every panel's refresh/tick path is exercised.

Samples process RSS and the UI tick-handler time every 30 s and writes
``runs/soak_<timestamp>.csv`` plus a summary. Pass criteria (from the plan):
memory growth after warm-up within ±50 MB, UI tick handler p99 < 50 ms.

    python scripts/soak_desktop.py --minutes 360 --live        # market-hours soak
    python scripts/soak_desktop.py --minutes 60                 # synthetic, any time
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SyntheticFeed:
    """``livefeed.Feed`` with the WebSocket replaced by a random-walk tick generator."""

    def __new__(cls, symbols, rate: float):
        from niveshrl.livefeed import Feed

        class _F(Feed):
            def _run(self):
                from niveshrl.desktop import data
                last = data.panel().close.ffill().iloc[-1]
                px = {s: float(last.get(s, 100.0)) for s in self.symbols}
                px.setdefault("^NSEI", 22_000.0)
                px.setdefault("^INDIAVIX", 14.0)
                base = dict(px)
                vol = {s: 0.0 for s in self.symbols}
                self.connected = True
                while not self._stop.is_set():
                    t0 = time.time()
                    for _ in range(max(1, int(rate / 10))):
                        s = random.choice(self.symbols)
                        px[s] *= 1 + random.gauss(0, 0.0008)
                        vol[s] += random.randint(1, 500)
                        self._on_message({"id": s, "price": px[s], "time": int(time.time() * 1000),
                                          "change_percent": (px[s] / base[s] - 1) * 100, "day_volume": vol[s]})
                    self._stop.wait(max(0.0, 0.1 - (time.time() - t0)))
        return _F(symbols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--rate", type=float, default=200, help="synthetic ticks per second")
    ap.add_argument("--live", action="store_true", help="use the real Yahoo stream")
    ap.add_argument("--visible", action="store_true", help="show the window instead of offscreen")
    args = ap.parse_args()
    if not args.visible:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

    import psutil
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    from niveshrl.desktop import data, theme
    from niveshrl.desktop.mainwindow import MainWindow

    app.setStyleSheet(theme.QSS)
    if not args.live:                                   # no network calls in synthetic mode
        stub = {"stats": {}, "profile": {}, "annual": None, "balance": None, "cashflow": None, "quarterly": None}
        data.fundamentals = lambda t: stub
    w = MainWindow(start_feed=args.live)
    w.ctx.scheduler.set_enabled(False)
    w.resize(1600, 960)
    w.show()
    proc = psutil.Process()
    out = ROOT / "runs" / f"soak_{datetime.now():%Y%m%d_%H%M}.csv"
    out.parent.mkdir(exist_ok=True)
    rows: list[tuple] = []
    ui_ms: list[float] = []
    t_start = time.time()
    codes = ["MKT", "TODAY", "SCRN", "WATCH", "RANK", "RISK", "ALRT"]
    state = {"i": 0}

    orig_tick = w._tick

    def timed_tick():
        t0 = time.perf_counter()
        orig_tick()
        ui_ms.append((time.perf_counter() - t0) * 1000)
    w.t_tick.timeout.disconnect()
    w.t_tick.timeout.connect(timed_tick)

    def started():
        if w.base is None:
            QTimer.singleShot(500, started)
            return
        if not args.live:
            w.ctx.feed = SyntheticFeed(data.panel().tickers, args.rate).start()
        print(f"soak started: {'live' if args.live else f'synthetic {args.rate:.0f} ticks/s'}, "
              f"{args.minutes:.0f} min -> {out}", flush=True)

    def churn():
        if w.base is None:
            return
        w.show_panel(codes[state["i"] % len(codes)])
        if state["i"] % 3 == 0:
            w.open_stock(random.choice(data.panel().tickers))
        state["i"] += 1

    def sample():
        el = (time.time() - t_start) / 60
        rss = proc.memory_info().rss / 2 ** 20
        recent = ui_ms[-30:] or [0.0]
        f = w.ctx.feed
        rows.append((round(el, 2), round(rss, 1), round(max(recent), 2), round(sum(recent) / len(recent), 2),
                     f.n_msgs if f else 0, threading.active_count(), data.cache_size()))
        print(f"{el:7.1f} min  rss {rss:7.1f} MB  ui max {max(recent):6.1f} ms  ticks {rows[-1][4]:,}  "
              f"threads {rows[-1][5]}  cache {rows[-1][6]}", flush=True)
        out.write_text("minutes,rss_mb,ui_max_ms,ui_mean_ms,ticks,threads,cache_entries\n"
                       + "\n".join(",".join(map(str, r)) for r in rows))
        if el >= args.minutes:
            finish()

    def finish():
        import numpy as np
        warm = [r for r in rows if r[0] >= min(10, args.minutes / 4)]
        rss = np.array([r[1] for r in warm]) if warm else np.array([rows[-1][1]])
        p99 = float(np.percentile(ui_ms, 99)) if ui_ms else float("nan")
        growth = float(rss[-1] - rss[0])
        ok = abs(growth) <= 50 and p99 < 50
        summary = (f"SOAK {'PASS' if ok else 'FAIL'}: {args.minutes:.0f} min, {rows[-1][4]:,} ticks, "
                   f"RSS {rss[0]:.0f} -> {rss[-1]:.0f} MB after warm-up (growth {growth:+.0f} MB, max {rss.max():.0f}), "
                   f"UI tick p50 {np.percentile(ui_ms, 50):.1f} ms / p99 {p99:.1f} ms / max {max(ui_ms):.1f} ms")
        print(summary, flush=True)
        out.with_suffix(".txt").write_text(summary + "\n")
        w._quitting = True
        w.shutdown()
        app.exit(0 if ok else 1)

    QTimer.singleShot(500, started)
    t1 = QTimer()
    t1.timeout.connect(churn)
    t1.start(4000)
    t2 = QTimer()
    t2.timeout.connect(sample)
    t2.start(30_000)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
