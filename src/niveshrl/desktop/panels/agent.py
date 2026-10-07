"""ALGO: the intraday paper-trading agent: status, controls, today's trades and decisions, history vs the random
control, what it has learned, and the replay over history. Paper only: nothing here places an order."""
from __future__ import annotations

import json

import pandas as pd
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QMessageBox, QPushButton, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from ... import intraday as ID
from ...intraday.costs import speculative_tax
from .. import theme
from ..widgets import FrameTable, KpiRow, h2, line_chart, muted
from . import Panel, vbox


AGENT_TERMS = {c: "agent_trade" for c in ("ticker", "setup", "side", "action", "entry", "stop", "target", "qty", "last",
                                           "exit", "reason", "ts", "event", "detail", "why", "time in", "time out")} | {
    "gross": "intraday_costs", "costs": "intraday_costs", "net": "intraday_costs", "unrealized": "intraday_costs",
    "r": "r_multiple", "prob": "agent_ml", "day": "agent_day", "pool_start": "agent_day", "pool_end": "agent_day",
    "tax_accrued": "speculative_tax", "trades": "agent_day", "signals": "agent_day", "skipped": "agent_day",
    "control_net": "random_control", "hit_target": "agent_day", "bucket": "agent_bandit", "avg R": "agent_bandit",
    "t-stat": "agent_bandit", "policy": "agent_bandit", "from": "agent_bandit", "to": "agent_bandit", "n": "agent_bandit",
    "avg_r": "agent_bandit", "t": "agent_bandit", "when": "agent_bandit"}


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (ValueError, OSError):
        return None


def _with_times(df: pd.DataFrame) -> pd.DataFrame:
    """'time in' = the bar the fill happened on (the open just after the signal bar ended); 'time out' = the exit bar."""
    if df.empty:
        return df
    for src, dst in (("entry_ts", "time in"), ("exit_ts", "time out")):
        if src in df:
            ts = pd.to_datetime(df[src], errors="coerce")
            df[dst] = ts.dt.strftime("%H:%M").where(ts.notna(), "")
    return df


class AgentPanel(Panel):
    title = "Intraday agent"
    code = "ALGO"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        self.runner = ctx.agent
        lay = vbox(self)
        lay.addWidget(muted("A virtual pool that trades liquid NSE stocks intraday on its own, during market hours, with "
                            "realistic costs and tax, and learns from every result. Paper only: no order is ever placed. "
                            "Its first 52-day replay lost money (see Replay): treat it as an experiment, not a strategy."))
        bar = QHBoxLayout()
        self.go = QPushButton("▶ Start agent")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self._toggle)
        bar.addWidget(self.go)
        self.auto = QCheckBox("Auto-start 09:05 on weekdays")
        self.auto.setChecked(self.runner.auto)
        self.auto.toggled.connect(self.runner.set_auto)
        bar.addWidget(self.auto)
        cfg = ID.config()
        self.lev = QComboBox()
        for x in (1.0, 2.0, 3.0, 4.0, 5.0):
            self.lev.addItem(f"Leverage {x:g}×" + (" (none)" if x == 1 else ""), x)
        self.lev.setCurrentIndex(max(0, int(cfg["leverage"]) - 1))
        self.lev.currentIndexChanged.connect(lambda: ID.save_settings(leverage=self.lev.currentData()))
        bar.addWidget(self.lev)
        self.slab = QComboBox()
        for x in (0.0, 0.05, 0.10, 0.20, 0.30):
            self.slab.addItem(f"Tax slab {x:.0%}", x)
        self.slab.setCurrentIndex([0.0, 0.05, 0.10, 0.20, 0.30].index(cfg["tax"]["slab_rate"])
                                  if cfg["tax"]["slab_rate"] in (0.0, 0.05, 0.10, 0.20, 0.30) else 4)
        self.slab.currentIndexChanged.connect(lambda: (ID.save_settings(slab_rate=self.slab.currentData()), self.refresh()))
        bar.addWidget(self.slab)
        self.pool = QDoubleSpinBox()
        self.pool.setRange(10_000, 10_000_000)
        self.pool.setDecimals(0)
        self.pool.setSingleStep(10_000)
        self.pool.setPrefix("Pool ₹ ")
        self.pool.setValue(float(cfg["pool"]))
        bar.addWidget(self.pool)
        reset = QPushButton("Reset account to this pool")
        reset.clicked.connect(self._reset)
        bar.addWidget(reset)
        self.replay_btn = QPushButton("⟲ Replay last 60 days")
        self.replay_btn.clicked.connect(lambda: self.runner.start("replay"))
        bar.addWidget(self.replay_btn)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.status = muted("")
        lay.addWidget(self.status)
        self.kpis = KpiRow(cols=6)
        lay.addWidget(self.kpis)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        money = {"entry": "₹{:,.2f}", "stop": "₹{:,.2f}", "target": "₹{:,.2f}", "exit": "₹{:,.2f}", "last": "₹{:,.2f}",
                 "gross": "₹{:+,.0f}", "costs": "₹{:,.0f}", "net": "₹{:+,.0f}", "unrealized": "₹{:+,.0f}", "r": "{:+.2f}",
                 "prob": "{:.0%}", "qty": "{:,.0f}"}
        tw = QWidget()
        tl = QVBoxLayout(tw)
        tl.addWidget(h2("Why it traded or skipped (latest day)", "agent_why"))
        self.why = FrameTable(fmt={"signals": "{:,.0f}", "share": "{:.0%}", "would-be avg R": "{:+.2f}",
                                   "would-be total R": "{:+.1f}"}, signed={"would-be avg R", "would-be total R"})
        self.why.model_.term_overrides = {c: "agent_why" for c in ("signals", "share", "would-be avg R",
                                                                    "would-be total R")}
        self.why.setMaximumHeight(170)
        tl.addWidget(self.why)
        self.why_note = muted("")
        tl.addWidget(self.why_note)
        tl.addWidget(h2("Open positions"))
        self.positions = FrameTable(fmt=money, signed={"unrealized"})
        self.positions.model_.term_overrides = AGENT_TERMS
        tl.addWidget(self.positions, 1)
        tl.addWidget(h2("Closed today"))
        self.trades = FrameTable(fmt=money, signed={"net", "gross", "r"})
        self.trades.model_.term_overrides = AGENT_TERMS
        tl.addWidget(self.trades, 1)
        tl.addWidget(h2("Every decision today (taken, half, skipped, blocked) with its reason"))
        self.decisions = FrameTable(fmt={"prob": "{:.0%}"})
        self.decisions.model_.term_overrides = AGENT_TERMS
        tl.addWidget(self.decisions, 1)
        self.tabs.addTab(tw, "Today")
        hw = QWidget()
        self.hl = QVBoxLayout(hw)
        self.days = FrameTable(fmt={"pool_start": "₹{:,.0f}", "pool_end": "₹{:,.0f}", "gross": "₹{:+,.0f}", "costs": "₹{:,.0f}",
                                    "net": "₹{:+,.0f}", "tax_accrued": "₹{:,.0f}", "control_net": "₹{:+,.0f}"},
                               signed={"gross", "net", "control_net"})
        self.days.model_.term_overrides = AGENT_TERMS
        self.hl.addWidget(self.days, 1)
        self.tabs.addTab(hw, "History")
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.addWidget(muted("Buckets of similar signals and what the agent now does with them. Every signal's outcome is "
                           "followed even when skipped, so a wrong SKIP is learned too. Policies change only on "
                           "statistically clear evidence; risk limits never change."))
        self.buckets = FrameTable(fmt={"avg R": "{:+.2f}", "t-stat": "{:+.1f}", "signals": "{:,.0f}"}, signed={"avg R"})
        self.buckets.model_.term_overrides = AGENT_TERMS
        ll.addWidget(self.buckets, 2)
        ll.addWidget(h2("Mistakes: what losing signals have in common (vs winning ones)", "agent_mistakes"))
        self.mistakes = FrameTable(fmt={"losses with it": "{:,.0f}", "in losses": "{:.0%}", "in wins": "{:.0%}",
                                        "lift": "{:.2f}", "avg R with it": "{:+.2f}", "avg R without": "{:+.2f}",
                                        "earlier half": "{:.0%}", "recent half": "{:.0%}"},
                                   signed={"avg R with it", "avg R without"})
        self.mistakes.model_.term_overrides = {c: "agent_mistakes" for c in (
            "losses with it", "in losses", "in wins", "lift", "avg R with it", "avg R without", "earlier half",
            "recent half", "meaning")}
        self.mistakes.setMaximumHeight(200)
        ll.addWidget(self.mistakes)
        ll.addWidget(h2("Learning log (what changed and why)"))
        self.learnlog = FrameTable()
        self.learnlog.model_.term_overrides = AGENT_TERMS
        ll.addWidget(self.learnlog, 1)
        self.tabs.addTab(lw, "Learning")
        vw = QWidget()
        vl = QVBoxLayout(vw)
        vl.addWidget(muted("v2 trades its own ₹1 lakh paper pool beside v1 every market day: a temporal convolutional "
                           "network scores every in-play stock's every 5-minute bar (long and short, expected R after "
                           "costs), entries are taken in time order above a threshold calibrated on recent days, with "
                           "ATR stops wide enough that costs stay under 0.2R, a 2R target and a 60-minute time exit. "
                           "It is 'shadow' until it beats v1 on measured evidence (Performance → Overview)."))
        self.v2_kpis = KpiRow(cols=6)
        vl.addWidget(self.v2_kpis)
        vl.addWidget(h2("v2 open positions and today's trades", "agent_engines"))
        self.v2_pos = FrameTable(fmt=money, signed={"unrealized", "net", "r"})
        self.v2_pos.model_.term_overrides = AGENT_TERMS
        vl.addWidget(self.v2_pos, 1)
        vl.addWidget(h2("v1 vs v2, day by day (₹ net after costs)", "agent_engines"))
        self.v2_days = FrameTable(fmt={"v1 net": "₹{:+,.0f}", "v2 net": "₹{:+,.0f}", "v1 trades": "{:,.0f}",
                                       "v2 trades": "{:,.0f}", "v2 candidates": "{:,.0f}",
                                       "v2 candidates avg R": "{:+.2f}"},
                                  signed={"v1 net", "v2 net", "v2 candidates avg R"})
        self.v2_days.model_.term_overrides = {c: "agent_engines" for c in ("v1 net", "v2 net", "v1 trades", "v2 trades",
                                                                            "v2 candidates", "v2 candidates avg R")}
        vl.addWidget(self.v2_days, 1)
        self.tabs.addTab(vw, "Deep-learning engine (v2)")
        rw = QWidget()
        rl = QVBoxLayout(rw)
        self.replay_kpis = KpiRow(cols=6)
        rl.addWidget(self.replay_kpis)
        self.replay_box = QVBoxLayout()
        rl.addLayout(self.replay_box, 1)
        self.replay_note = QTextBrowser()
        self.replay_note.setMaximumHeight(120)
        rl.addWidget(self.replay_note)
        self.tabs.addTab(rw, "Replay (history)")
        self.runner.progress.connect(lambda m: self.status.setText(f"Agent: {m}"))
        self.runner.finished.connect(lambda d: (self.status.setText(f"Agent finished: {d.get('mode')} {d.get('phase', '')}"),
                                                self.refresh()))
        self.runner.failed.connect(lambda m: self.status.setText("Agent error: " + m.strip().splitlines()[-1][:200]))
        self.timer = QTimer(self)
        self.timer.setInterval(10_000)
        self.timer.timeout.connect(lambda: self.isVisible() and self.refresh())
        self.timer.start()

    def _toggle(self) -> None:
        if self.runner.running:
            self.runner.stop()
            self.status.setText("Stopping the agent…")
        else:
            self.runner.start("live")

    def _reset(self) -> None:
        if self.runner.running:
            QMessageBox.information(self, "NiveshRL", "Stop the agent first.")
            return
        if QMessageBox.question(self, "NiveshRL", f"Reset the paper account to ₹{self.pool.value():,.0f}? Its trade history, "
                                "learning and model are archived (renamed), not deleted.") != QMessageBox.Yes:
            return
        import time
        arch = ID.DIR / f"archive_{time.strftime('%Y%m%d_%H%M%S')}"
        arch.mkdir(parents=True, exist_ok=True)
        for name in ("state.json", "trades.csv", "bandit.json", "scorer.pkl", "shadow.parquet", "live.json"):
            p = ID.DIR / name
            if p.exists():
                p.rename(arch / name)
        ID.save_settings(pool=self.pool.value())
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.go.setText("■ Stop agent" if self.runner.running else "▶ Start agent")
        self.replay_btn.setEnabled(not self.runner.running)
        cfg = ID.config()
        st = _json(ID.DIR / "state.json") or {"pool": cfg["pool"], "start_pool": cfg["pool"], "control_pool": cfg["pool"],
                                               "fy_net": {}, "days": []}
        live = _json(ID.DIR / "live.json") or {}
        tax = sum(speculative_tax(v, cfg["tax"]) for v in st.get("fy_net", {}).values())
        start = float(st.get("start_pool", cfg["pool"]))
        pool = float(st.get("pool", start))
        days = pd.DataFrame(st.get("days", []))
        tr_path = ID.DIR / "trades.csv"
        tr = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame()
        n_closed = len(tr)
        verdict = "" if n_closed >= cfg["learning"]["verdict_after_trades"] else \
            f"no verdict before {cfg['learning']['verdict_after_trades']} trades"
        self.kpis.set_items([
            ("Agent", (live.get("phase") or "idle").upper(), theme.GREEN if live.get("phase") == "trading" else theme.MUTED,
             (live.get("message") or "")[:60]),
            ("Pool", f"₹{pool:,.0f}", theme.signed(pool - start), f"started ₹{start:,.0f} · {pool / start - 1:+.1%}"),
            ("Today P&L", f"₹{live.get('day_pnl', 0):+,.0f}" if live.get("phase") == "trading" else "–",
             theme.signed(live.get("day_pnl", 0)), "incl. open positions"),
            ("After tax", f"₹{pool - tax:,.0f}", None, f"tax due this FY ₹{tax:,.0f} at {cfg['tax']['slab_rate']:.0%}+cess"),
            ("Closed trades", f"{n_closed}", None,
             (f"win {float((tr['net'] > 0).mean()):.0%} · avg {tr['r'].mean():+.2f}R" if n_closed else "") + (f" · {verdict}" if verdict else "")),
            ("vs random control", f"₹{float(st.get('control_pool', start)):,.0f}", None, "same signals, coin-flip entries"),
        ])
        pos = _with_times(pd.DataFrame(live.get("positions", [])))
        self.positions.set_frame(pos.reindex(
            columns=["ticker", "setup", "side", "action", "time in", "entry", "stop", "target", "qty", "last", "unrealized",
                     "prob", "why"]) if len(pos) else pd.DataFrame({"": ["No open positions."]}))
        t = _with_times(pd.DataFrame(live.get("trades", [])))
        self.trades.set_frame(t.reindex(columns=["ticker", "setup", "side", "action", "time in", "entry", "time out", "exit",
                                                 "reason", "qty", "gross", "costs", "net", "r", "why"])
                              if len(t) else pd.DataFrame({"": ["No closed trades today."]}))
        lg = pd.DataFrame(live.get("log", []))
        self.decisions.set_frame(lg.reindex(columns=["ts", "ticker", "setup", "side", "event", "action", "prob", "why", "detail"])
                                 if len(lg) else pd.DataFrame({"": ["No decisions yet today."]}))
        self._why(live, days)
        self._v2(days)
        self.days.set_frame(days.drop(columns=["learned"], errors="ignore") if len(days) else
                            pd.DataFrame({"": ["No trading days yet. Start the agent during market hours (09:15–15:30)."]}))
        while self.hl.count() > 1:
            w = self.hl.takeAt(0).widget()
            if w:
                w.deleteLater()
        if len(days) >= 2:
            eq = pd.DataFrame({"Agent": days["pool_end"].to_numpy(),
                               "Random control": start + days["control_net"].cumsum().to_numpy()},
                              index=pd.to_datetime(days["day"]))
            self.hl.insertWidget(0, line_chart({c: eq[c] for c in eq}, "Paper pool: agent vs random control (₹)"), 1)
        b = _json(ID.DIR / "bandit.json")
        if b:
            from ...intraday.bandit import Bandit
            bd = Bandit(cfg, b)
            self.buckets.set_frame(bd.table().set_index("bucket") if bd.stats else pd.DataFrame())
            self.learnlog.set_frame(pd.DataFrame(bd.log[::-1]) if bd.log else pd.DataFrame({"": ["Nothing learned yet."]}))
        else:
            self.buckets.set_frame(pd.DataFrame({"": ["Nothing learned yet."]}))
            self.learnlog.set_frame(pd.DataFrame())
        self._replay()

    def _v2(self, v1_days: pd.DataFrame) -> None:
        root = ID.DIR / "v2"
        st, lv = _json(root / "state.json") or {}, _json(root / "live.json") or {}
        start = float(st.get("start_pool", ID.config()["pool"]))
        pool = float(st.get("pool", start))
        d2 = pd.DataFrame(st.get("days", []))
        model = lv.get("model") or {}
        self.v2_kpis.set_items([
            ("v2 status", (lv.get("phase") or "not started").upper(), theme.AMBER, (lv.get("message") or "")[:60]),
            ("v2 pool", f"₹{pool:,.0f}", theme.signed(pool - start), f"{pool / start - 1:+.1%} since start"),
            ("v2 today", f"₹{lv.get('day_pnl', 0):+,.0f}" if lv.get("phase") == "trading" else "–",
             theme.signed(lv.get("day_pnl", 0)), f"{lv.get('candidates', 0)} candidates scored"),
            ("v2 days", str(len(d2)), None, f"{int(d2['trades'].sum()) if len(d2) else 0} trades"),
            ("Model", f"{model.get('n_train', 0):,} samples" if model else "not trained", None,
             f"refit {model.get('when', '–')}"),
            ("Policy", (lv.get("policy") or "top-5 threshold")[:26], None, "chosen by measurement"),
        ])
        pos = _with_times(pd.DataFrame(lv.get("positions", []) + lv.get("trades", [])))
        self.v2_pos.set_frame(pos.reindex(columns=["ticker", "side", "time in", "entry", "stop", "target", "qty", "time out",
                                                   "exit", "reason", "net", "r", "unrealized", "why"])
                              if len(pos) else pd.DataFrame({"": ["No v2 positions today."]}))
        if len(d2):
            v1 = v1_days.set_index("day")[["net", "trades"]].rename(columns={"net": "v1 net", "trades": "v1 trades"}) \
                if len(v1_days) else pd.DataFrame()
            t = d2.set_index("day")[["net", "trades", "candidates", "candidate_avg_r"]].rename(
                columns={"net": "v2 net", "trades": "v2 trades", "candidates": "v2 candidates",
                         "candidate_avg_r": "v2 candidates avg R"})
            self.v2_days.set_frame(t.join(v1, how="left").sort_index(ascending=False))
        else:
            self.v2_days.set_frame(pd.DataFrame({"": ["No v2 days yet: it starts with the next live session."]}))

    def _why(self, live: dict, days: pd.DataFrame) -> None:
        """Skip reasons for today (live) or the latest settled day, with what the skipped signals would have made."""
        from ...intraday.mistakes import mistake_table, skip_breakdown
        log, day = live.get("log") or [], live.get("day")
        if not log and len(days):
            day = str(days["day"].iloc[-1])
            log = _json(ID.DIR / f"log_{day}.json") or []
        sh_all = pd.read_parquet(ID.DIR / "shadow.parquet") if (ID.DIR / "shadow.parquet").exists() else pd.DataFrame()
        sh = sh_all[sh_all["day"].astype(str) == str(day)] if len(sh_all) and "day" in sh_all else None
        t = skip_breakdown(log, sh)
        self.why.set_frame(t if len(t) else pd.DataFrame({"": ["No decisions yet."]}))
        if len(t) and "would-be total R" in t:
            skipped = t.drop(index="taken", errors="ignore")
            tot = skipped["would-be total R"].sum()
            self.why_note.setText(f"{day}: the {int(skipped['signals'].sum())} skipped signals would together have made "
                                  f"{tot:+.1f}R at full size: skipping {'saved' if tot < 0 else 'cost'} about "
                                  f"₹{abs(tot) * float(ID.config()['pool']) * ID.config()['guardrails']['risk_per_trade']:,.0f}. "
                                  "Few trades on a day like this means the learner is protecting the pool; more trades need "
                                  "better signals, not looser filters.")
        else:
            self.why_note.setText("What skipped signals would have made is known after the close (shadow outcomes).")
        m = mistake_table(sh_all) if len(sh_all) else pd.DataFrame()
        self.mistakes.set_frame(m if len(m) else pd.DataFrame({"": ["Not enough outcomes yet."]}))

    def _replay(self) -> None:
        root = ID.DIR / "replay"
        st = _json(root / "state.json")
        while self.replay_box.count():
            w = self.replay_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        if not st or not st.get("days"):
            self.replay_kpis.set_items([("Replay", "not run", None, "press ⟲ Replay last 60 days")])
            return
        d = pd.DataFrame(st["days"])
        start = float(st["start_pool"])
        tr = pd.read_csv(root / "trades.csv") if (root / "trades.csv").exists() else pd.DataFrame()
        self.replay_kpis.set_items([
            ("Days", str(len(d)), None, f"{d['day'].iloc[0]} → {d['day'].iloc[-1]}"),
            ("Pool", f"₹{float(st['pool']):,.0f}", theme.signed(float(st["pool"]) - start), f"{float(st['pool']) / start - 1:+.1%}"),
            ("Gross", f"₹{d['gross'].sum():+,.0f}", theme.signed(d["gross"].sum()), "before costs"),
            ("Costs", f"₹{d['costs'].sum():,.0f}", theme.RED, "brokerage, STT, GST, stamp, slippage"),
            ("Random control", f"₹{float(st['control_pool']):,.0f}", None, "same signals, coin-flip entries"),
            ("Trades", str(len(tr)), None, f"avg {tr['r'].mean():+.2f}R · win {float((tr['net'] > 0).mean()):.0%}" if len(tr) else ""),
        ])
        eq = pd.DataFrame({"Agent": d["pool_end"].to_numpy(), "Random control": start + d["control_net"].cumsum().to_numpy()},
                          index=pd.to_datetime(d["day"]))
        self.replay_box.addWidget(line_chart({c: eq[c] for c in eq}, "Replay: paper pool, agent vs random control (₹)"))
        self.replay_note.setHtml(
            f"<p>Yahoo's last ~60 days of 5-minute bars for the liquid universe, one day at a time; each day learns only from "
            f"earlier days. Fills at the next bar's open plus slippage; stop assumed first when one bar touches both. "
            f"<b style='color:{theme.AMBER}'>History is short (≈2 months) and in-sample for the cost rule</b>; the honest "
            f"test is live paper trading over months. No day reached the +10% target.</p>")
