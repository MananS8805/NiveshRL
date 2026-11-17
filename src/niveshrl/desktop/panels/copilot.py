"""COPILOT (shown as "Algo trader"): two paper books in one screen. The 50-trade book trades the tournament leader
among high-turnover rules (``intraday.algo``); the copilot takes at most 3 structural setups a day. The copilot's day: Morning brief and watchlist, the best setups right now (always
three, labelled when below the bar), the paper account, the end-of-day review (strengths, weaknesses, learnings) and the
money scorecard (backtest benchmark + forward record under the same tests). It runs inside the intraday agent's
market-hours process, so the agent's Start button / 09:05 auto-start runs it too. Paper only."""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from ...config import ROOT
from ...intraday.copilot import CDIR, load_json
from .. import theme
from ..widgets import FrameTable, KpiRow, h2, muted
from . import Panel, vbox

IST = timezone(timedelta(hours=5, minutes=30))
BENCH = ROOT / "report" / "results" / "copilot_benchmark.json"
ALGO_BENCH = ROOT / "report" / "results" / "algo_benchmark.json"
MONEY = {"ref": "₹{:,.2f}", "stop": "₹{:,.2f}", "entry": "₹{:,.2f}", "target": "₹{:,.2f}", "exit": "₹{:,.2f}",
         "last": "₹{:,.2f}", "net": "₹{:+,.0f}", "gross": "₹{:+,.0f}", "costs": "₹{:,.0f}", "unrealized": "₹{:+,.0f}",
         "r": "{:+.2f}", "P(win)": "{:.0%}", "cost R": "{:.2f}", "stop %": "{:.2%}", "close": "₹{:,.2f}",
         "PDH": "₹{:,.2f}", "PDL": "₹{:,.2f}", "change": "{:+.1%}", "ADR": "{:.1%}", "rvol": "{:.1f}×",
         "gap": "{:+.2%}", "OR high": "₹{:,.2f}", "OR low": "₹{:,.2f}", "OR width": "{:.2%}"}


def _set(table, df: pd.DataFrame, key: str) -> None:
    """Fill a table and route every column's ⓘ to one glossary entry."""
    table.model_.term_overrides = {c: key for c in df.columns}
    table.set_frame(df)


def _today() -> str:
    return datetime.now(IST).date().isoformat()


def _ul(items) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(str(x))}</li>" for x in items) + "</ul>"


class CopilotPanel(Panel):
    title = "Algo trader"
    code = "COPILOT"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        self.runner = ctx.agent
        self.settings = QSettings("NiveshRL", "NiveshRL")
        lay = vbox(self)
        lay.addWidget(muted("Two paper books on the 150 most liquid NSE stocks. The 50-trade book runs the best high-turnover "
                            "rule from a daily tournament of 93 rules scored after real costs. The copilot takes at most 3 "
                            "structural setups a day and tunes its rules from the evidence. Both review their day after the "
                            "close. Paper only: no order is ever placed. Neither is proven to make money: check each "
                            "scorecard before trusting it."))
        bar = QHBoxLayout()
        self.go = QPushButton("▶ Start for today")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self._toggle)
        bar.addWidget(self.go)
        self.auto = QCheckBox("Run automatically every weekday (09:05 brief → trading → 15:20 review)")
        self.auto.setChecked(self.runner.auto)
        self.auto.toggled.connect(self.runner.set_auto)
        bar.addWidget(self.auto)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.status = muted("")
        lay.addWidget(self.status)
        self.kpis = KpiRow(cols=7)
        lay.addWidget(self.kpis)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        # Today
        tw = QWidget()
        tl = QVBoxLayout(tw)
        self.brief = QTextBrowser()
        self.brief.setMaximumHeight(150)
        tl.addWidget(h2("Morning brief", "copilot_brief"))
        tl.addWidget(self.brief)
        tl.addWidget(h2("Best setups right now (always three; ✗ = below the bar, for awareness only)", "copilot_card"))
        self.top = FrameTable(fmt=MONEY, signed={"r"})
        self.top.setMaximumHeight(140)
        tl.addWidget(self.top)
        tl.addWidget(h2("Paper trades today (open and closed)", "copilot_card"))
        self.trades = FrameTable(fmt=MONEY, signed={"net", "gross", "r", "unrealized"})
        tl.addWidget(self.trades, 1)
        tl.addWidget(h2("Watchlist and opening levels", "copilot_brief"))
        self.watch = FrameTable(fmt=MONEY, signed={"change", "gap"})
        tl.addWidget(self.watch, 1)
        self.tabs.addTab(tw, "Copilot today")
        # 50-trade book
        bw = QWidget()
        bl = QVBoxLayout(bw)
        self.book_head = QTextBrowser()
        self.book_head.setMaximumHeight(190)
        bl.addWidget(self.book_head)
        bl.addWidget(h2("Book trades today (open and closed)", "algo_book"))
        self.book_trades = FrameTable(fmt=MONEY, signed={"net", "gross", "unrealized"})
        bl.addWidget(self.book_trades, 1)
        bl.addWidget(h2("Rule tournament: last 20 days, after costs (best first)", "algo_book"))
        self.book_tour = FrameTable(fmt={"net ₹/day": "₹{:+,.0f}", "gross bps": "{:+.1f}", "net bps": "{:+.1f}",
                                         "trades/day": "{:.0f}"}, signed={"net ₹/day", "gross bps", "net bps"})
        bl.addWidget(self.book_tour, 1)
        self.tabs.insertTab(0, bw, "50-trade book")
        self.tabs.setCurrentIndex(0)
        # All cards
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.addWidget(muted("Every setup today, taken or not, with the reason. After the close each one gets its outcome: "
                           "this is what the copilot learns from, so skipped setups teach it as much as trades do."))
        self.cards = FrameTable(fmt=MONEY, signed={"r"})
        cl.addWidget(self.cards, 1)
        self.tabs.addTab(cw, "All setups")
        # Review
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rb = QHBoxLayout()
        rb.addWidget(muted("Day:"))
        self.day = QComboBox()
        self.day.currentIndexChanged.connect(self._show_review)
        rb.addWidget(self.day)
        rb.addStretch(1)
        rl.addLayout(rb)
        self.review = QTextBrowser()
        rl.addWidget(self.review, 1)
        self.tabs.addTab(rw, "Day review")
        # Scorecard
        sw = QWidget()
        sl = QVBoxLayout(sw)
        self.score = QTextBrowser()
        sl.addWidget(self.score, 1)
        self.tabs.addTab(sw, "Scorecard")
        # Rules
        uw = QWidget()
        ul = QVBoxLayout(uw)
        ul.addWidget(muted("The learner may only change these rules, one bounded step at a time and only on evidence (n, mean R, "
                           "t-statistic). Risk per trade, 3 trades a day, the 2% daily loss stop and square-off are fixed."))
        self.rules = FrameTable(fmt={"min P(win)": "{:.0%}"})
        self.rules.setMaximumHeight(170)
        ul.addWidget(self.rules)
        ul.addWidget(h2("Every rule change and its evidence", "copilot_rules"))
        self.rlog = FrameTable(fmt={"mean R": "{:+.3f}", "t": "{:+.2f}"}, signed={"mean R", "t"})
        ul.addWidget(self.rlog, 1)
        self.tabs.addTab(uw, "Rules & learning")
        self.runner.progress.connect(lambda m: self.status.setText(f"Running: {m}"))
        self.timer = QTimer(self)
        self.timer.setInterval(15_000)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ------------------------------------------------------------------
    def _toggle(self) -> None:
        if self.runner.running:
            self.runner.stop()
            self.status.setText("Stopping…")
        else:
            self.runner.start("live")
            self.status.setText("Starting: the morning brief comes first, trading decisions from 09:20.")

    def _tick(self) -> None:
        """Notify once per day when the brief and the review land, and refresh when visible."""
        d = _today()
        for kind, path, title in (("brief", CDIR / "briefs" / f"{d}.json", "Copilot: morning brief ready"),
                                  ("review", CDIR / "reviews" / f"{d}.json", "Copilot: day review ready")):
            key = f"copilot_notified_{kind}"
            if path.exists() and self.settings.value(key, "") != d:
                self.settings.setValue(key, d)
                obj = load_json(str(path.relative_to(CDIR)).replace("\\", "/")) or {}
                msg = " ".join((obj.get("lines") or obj.get("summary") or [])[:2])
                self.ctx.alert.emit(title, msg[:240] or "Open NiveshRL → Algo trader.")
        if self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        self.go.setText("■ Stop" if self.runner.running else "▶ Start for today")
        live = load_json("live.json", {}) or {}
        st = load_json("state.json", {}) or {}
        days = pd.DataFrame(st.get("days", []))
        d = _today()
        pool = float(st.get("pool", 100_000))
        start = float(st.get("start_pool", 100_000))
        bench = json.loads(BENCH.read_text(encoding="utf-8")) if BENCH.exists() else {}
        verdict = bench.get("live_policy", {}).get("verdict", {})
        fwd_tr = int(days["trades"].sum()) if len(days) else 0
        self.kpis.set_items([
            ("Copilot pool", f"₹{pool:,.0f}", theme.GREEN if pool >= start else theme.RED, f"started ₹{start:,.0f}"),
            ("Copilot today", f"₹{live.get('day_pnl', 0):+,.0f}" if live.get("day") == d else "—", None, live.get("phase", "idle")),
            ("Copilot setups today", str(len(live.get("cards", []))) if live.get("day") == d else "—", None, "taken or not"),
            ("Copilot forward trades", str(fwd_tr), None, f"{len(days)} day(s) live"),
            ("Copilot backtest verdict", verdict.get("label", "not run"), theme.GREEN if verdict.get("label") == "MONEY-READY" else theme.AMBER,
             f"{verdict.get('passed', 0)}/{verdict.get('total', 7)} checks"),
            ("Copilot run mode", "AUTO" if self.runner.auto else "manual", None, "running" if self.runner.running else "not running"),
            ("Copilot model skill", self._skill_text(), theme.GREEN if (live.get("model") or {}).get("skilled") else theme.AMBER,
             "held-out AUC; needs 0.55 to filter trades"),
        ])
        if live.get("message"):
            self.status.setText(f"{live.get('phase', '')}: {live.get('message')} ({str(live.get('at', ''))[11:16]})")
        self._book(d)
        self._brief(d)
        self._today_tables(live, d)
        self._reviews()
        self._scorecard(bench, days)
        self._rules()

    def _book(self, d: str) -> None:
        from ...intraday.algo.book import TOURN, load_json as book_json
        lv = book_json("live.json", {}) or {}
        st = book_json("state.json", {}) or {}
        days = pd.DataFrame(st.get("days", []))
        bench = json.loads(ALGO_BENCH.read_text(encoding="utf-8")) if ALGO_BENCH.exists() else {}
        g, red, amb = theme.GREEN, theme.RED, theme.AMBER
        parts = []
        if lv.get("day") == d:
            res = lv.get("result") or {}
            pnl = lv.get("day_pnl", res.get("net", 0)) or 0
            rule = lv.get("rule") or res.get("rule", "")
            why = f" ({html.escape(lv['why'])})" if lv.get("why") else ""
            parts.append(f"<h3>Today: <span style='color:{g if pnl >= 0 else red}'>₹{pnl:+,.0f}</span> · "
                         f"{html.escape(lv.get('phase', ''))}</h3><p>Rule: <b>{html.escape(str(rule))}</b>{why}. "
                         f"{html.escape(lv.get('message', ''))}</p>")
            rv = lv.get("review")
            if rv:
                parts.append("<h4>Day review</h4>" + _ul(rv.get("summary", []) + rv.get("learnings", [])))
        else:
            parts.append(f"<p style='color:{theme.MUTED}'>Not running today yet: it starts with the copilot at 09:05 "
                         "(auto) or when you press Start.</p>")
        if bench:
            v = bench.get("verdict", {})
            b = bench.get("boot", {})
            parts.append(f"<p style='color:{red if v.get('label') == 'NOT READY' else amb}'><b>Backtest verdict: "
                         f"{html.escape(v.get('label', '?'))}</b> ({v.get('passed')}/{v.get('total')} checks) · "
                         f"{bench.get('trades_per_day', 0):.0f} trades/day · mean day ₹{b.get('mean', 0):+,.0f} "
                         f"(90% CI ₹{b.get('lo90', 0):+,.0f} to ₹{b.get('hi90', 0):+,.0f}) · each trade costs "
                         f"{bench.get('cost_bps', 0):.1f} bps. A rule must earn more than that per trade; none of the 93 "
                         "measured rules does yet. Report: report/results/algo_benchmark.md</p>")
        if len(days):
            parts.append(f"<p>Forward paper record: {len(days)} day(s), {int(days['trades'].sum())} trades, "
                         f"net ₹{days['net'].sum():+,.0f}; pool ₹{st.get('pool', 0):,.0f}.</p>")
        self.book_head.setHtml("".join(parts))
        rows = ([dict(p, status="open") for p in lv.get("positions", [])] + [dict(t, status="closed") for t in lv.get("trades", [])]
                if lv.get("day") == d else [])
        tr = pd.DataFrame(rows)
        if len(tr):
            tr["ticker"] = tr["ticker"].str.replace(".NS", "", regex=False)
            tr = tr[[c for c in ["status", "ticker", "side", "entry", "exit", "last", "qty", "gross", "costs", "net",
                                 "unrealized"] if c in tr]]
        _set(self.book_trades, tr, "algo_book")
        try:
            tab = pd.read_parquet(TOURN)
            tab = tab.sort_values("source", key=lambda x: x.eq("live")).drop_duplicates(["day", "name"], keep="last")
            last = sorted(tab["day"].unique())[-20:]
            t = tab[tab["day"].isin(last)].groupby("name").agg(**{"trades/day": ("trades", "mean"),
                                                                    "gross bps": ("gross_bps", "mean"),
                                                                    "net bps": ("net_bps", "mean"),
                                                                    "net ₹/day": ("net", "mean")})
            t = t.sort_values("net ₹/day", ascending=False).reset_index().rename(columns={"name": "rule"})
            _set(self.book_tour, t, "algo_book")
        except Exception:                                   # noqa: BLE001
            _set(self.book_tour, pd.DataFrame(), "algo_book")

    def _skill_text(self) -> str:
        try:
            from ...intraday.copilot.engine import MODEL
            from ...intraday.copilot.model import CardModel
            m = CardModel.load(MODEL)
        except Exception:                                   # noqa: BLE001
            return "—"
        if not m.ready:
            return "not trained"
        auc = m.skill.get("auc")
        return ("skilled" if m.skilled else "no skill") + (f" · AUC {auc:.2f}" if auc is not None else "")

    def _brief(self, d: str) -> None:
        b = load_json(f"briefs/{d}.json")
        if not b:
            self.brief.setHtml(f"<p style='color:{theme.MUTED}'>No brief for today yet: it is written at 09:05 on weekdays "
                               "when the copilot runs (or press Start).</p>")
            _set(self.watch, pd.DataFrame(), "copilot_brief")
            return
        self.brief.setHtml(_ul(b.get("lines", [])))
        wl = pd.DataFrame(b.get("watchlist", []))
        if len(wl):
            wl = wl.rename(columns={"chg": "change", "pdh": "PDH", "pdl": "PDL", "adr_pct": "ADR"})
            wl["ticker"] = wl["ticker"].str.replace(".NS", "", regex=False)
            op = pd.DataFrame(b.get("opening", []))
            if len(op):
                op["ticker"] = op["ticker"].str.replace(".NS", "", regex=False)
                wl = wl.merge(op.rename(columns={"or_hi": "OR high", "or_lo": "OR low", "or_width": "OR width"})
                              [["ticker", "gap", "OR high", "OR low", "OR width"]], on="ticker", how="left")
            cols = [c for c in ["ticker", "why", "bias", "close", "change", "PDH", "PDL", "ADR", "rvol", "gap", "OR high",
                                "OR low", "OR width"] if c in wl]
            _set(self.watch, wl[cols], "copilot_brief")

    def _today_tables(self, live: dict, d: str) -> None:
        if live.get("day") != d:
            for t in (self.top, self.trades, self.cards):
                _set(t, pd.DataFrame(), "copilot_card")
            return
        top = pd.DataFrame(live.get("top", []))
        if len(top):
            top["take"] = top["take"].map(lambda x: "✓ taken" if x else "✗ below bar")
            top["ticker"] = top["ticker"].str.replace(".NS", "", regex=False)
            top = top.rename(columns={"p": "P(win)", "cost_r": "cost R", "stop_pct": "stop %"})
            _set(self.top, top[[c for c in ["take", "ticker", "setup", "side", "ts", "ref", "stop", "P(win)", "cost R",
                                                "stop %", "why"] if c in top]], "copilot_card")
        else:
            _set(self.top, pd.DataFrame(), "copilot_card")
        rows = [dict(p, status="open") for p in live.get("positions", [])] + [dict(t, status="closed") for t in live.get("trades", [])]
        tr = pd.DataFrame(rows)
        if len(tr):
            tr["ticker"] = tr["ticker"].str.replace(".NS", "", regex=False)
            tr = tr.rename(columns={"prob": "P(win)"})
            _set(self.trades, tr[[c for c in ["status", "ticker", "setup", "side", "entry_ts", "entry", "stop", "target",
                                                  "qty", "last", "unrealized", "exit_ts", "exit", "reason", "net", "r", "P(win)"]
                                      if c in tr]], "copilot_card")
        else:
            _set(self.trades, pd.DataFrame(), "copilot_card")
        cards = pd.DataFrame(live.get("cards", []))
        if len(cards):
            cards["ticker"] = cards["ticker"].str.replace(".NS", "", regex=False)
            cards["take"] = cards["take"].map(lambda x: "✓" if x else "")
            cards = cards.rename(columns={"p": "P(win)", "cost_r": "cost R", "stop_pct": "stop %"})
            _set(self.cards, cards[[c for c in ["ts", "take", "ticker", "setup", "side", "ref", "stop", "stop %", "P(win)",
                                                    "cost R", "r", "why"] if c in cards]], "copilot_card")
        else:
            _set(self.cards, pd.DataFrame(), "copilot_card")

    def _reviews(self) -> None:
        files = sorted((CDIR / "reviews").glob("*.json"), reverse=True) if (CDIR / "reviews").exists() else []
        names = [f.stem for f in files]
        if names != [self.day.itemText(i) for i in range(self.day.count())]:
            cur = self.day.currentText()
            self.day.blockSignals(True)
            self.day.clear()
            self.day.addItems(names)
            if cur in names:
                self.day.setCurrentText(cur)
            self.day.blockSignals(False)
            self._show_review()
        elif not names:
            self.review.setHtml(f"<p style='color:{theme.MUTED}'>No day reviewed yet. After 15:20 on a day the copilot ran, "
                                "it writes what happened, what worked, what failed and what it changed.</p>")

    def _show_review(self) -> None:
        day = self.day.currentText()
        r = load_json(f"reviews/{day}.json") if day else None
        if not r:
            return
        g, red, amb = theme.GREEN, theme.RED, theme.AMBER
        self.review.setHtml(
            f"<h3>{html.escape(day)}</h3>" + _ul(r.get("summary", []))
            + f"<h4 style='color:{g}'>Strengths</h4>" + _ul(r.get("strengths", []))
            + f"<h4 style='color:{red}'>Weaknesses</h4>" + _ul(r.get("weaknesses", []))
            + f"<h4 style='color:{amb}'>Learnings → rules for tomorrow</h4>" + _ul(r.get("learnings", [])))

    def _scorecard(self, bench: dict, days: pd.DataFrame) -> None:
        from ...research import robust as RB
        parts = []
        lp = bench.get("live_policy")
        if lp:
            names = dict(RB.CHECKS)
            rows = "".join(f"<tr><td>{'✓' if ok else '✗'}</td><td>{html.escape(names[k])}</td></tr>"
                           for k, ok in lp["verdict"]["checks"].items())
            b = lp["boot"]
            parts.append(f"<h3>Backtest ({bench.get('eval_days')} evaluation days, {lp['trades']} trades): "
                         f"{html.escape(lp['verdict']['label'])}</h3><table cellpadding=3>{rows}</table>"
                         f"<p>Net ₹{lp['net']:+,.0f} · holdout ₹{lp['holdout_net']:+,.0f} · mean day ₹{b['mean']:+,.0f} "
                         f"(90% CI ₹{b['lo90']:+,.0f} to ₹{b['hi90']:+,.0f}) · deflated Sharpe {lp['dsr']:.2f} · "
                         f"PBO {bench['pbo']['pbo']:.2f} · matched-random p {lp['random_p']:.2f} · "
                         f"2× slippage ₹{lp['net_2x']:+,.0f}</p>"
                         f"<p style='color:{theme.MUTED}'>Full report: report/results/copilot_benchmark.md (generated {bench.get('generated')}).</p>")
        else:
            parts.append(f"<p style='color:{theme.MUTED}'>No backtest benchmark yet (scripts/copilot_benchmark.py).</p>")
        need = 50
        if len(days):
            x = days["net"].astype(float).to_numpy()
            n_tr = int(days["trades"].sum())
            bt = RB.bootstrap_mean(x) if len(x) >= 5 else None
            parts.append(f"<h3>Forward paper record: {len(days)} day(s), {n_tr} trade(s), net ₹{x.sum():+,.0f}</h3>")
            if bt:
                parts.append(f"<p>Mean day ₹{bt['mean']:+,.0f} (90% CI ₹{bt['lo90']:+,.0f} to ₹{bt['hi90']:+,.0f}); "
                             f"P(mean > 0) {bt['p_pos']:.0%}.</p>")
            parts.append(f"<p>{'Enough trades for a verdict.' if n_tr >= need else f'No money verdict before {need} forward trades ({n_tr} so far).'} "
                         "The forward record is the only evidence that matters for real money: it can't be fitted after the fact.</p>")
        else:
            parts.append("<h3>Forward paper record: none yet</h3><p>Starts on the first weekday the copilot runs.</p>")
        try:
            from ...intraday.copilot.forward_log import verify
            v = verify()
            if v["entries"]:
                parts.append(f"<h4>Forward log: {'chain intact' if v['ok'] else 'CHAIN BROKEN at entry ' + str(v['broken_at'])}</h4>"
                             f"<p>{v['decisions']} decisions and {v['outcomes']} outcomes, each hash-chained to the one before "
                             f"(head <code>{v['head'][:16]}…</code>). Decisions are written when made, outcomes after the close, "
                             "so the record cannot be edited after the fact.</p>")
        except Exception:                                   # noqa: BLE001
            pass
        try:
            from ...intraday.copilot.engine import forward_hypotheses
            for h in forward_hypotheses():
                dd = pd.DataFrame(h["days"])
                rec = (f"{len(dd)} day(s), {int(dd['trades'].sum())} trade(s), net ₹{dd['net'].sum():+,.0f}"
                       if len(dd) else "no live days yet")
                parts.append(f"<h4>Forward hypothesis: {html.escape(h['name'])}</h4><p>{html.escape(h['why'])}. "
                             f"Scored only on live days after it was registered: {rec}. Not traded.</p>")
        except Exception:                                   # noqa: BLE001
            pass
        self.score.setHtml("".join(parts))

    def _rules(self) -> None:
        from ...intraday.copilot import rules as cur_rules
        r = cur_rules()
        _set(self.rules, pd.DataFrame([{"setup": s, "active": "yes" if v.get("enabled", True) else "benched",
                                            "min P(win)": v.get("min_p")} for s, v in r["setups"].items()]
                                          + [{"setup": "last entry", "active": r.get("last_entry"), "min P(win)": None}]),
             "copilot_rules")
        lg = pd.DataFrame(r.get("log", []))
        if len(lg):
            lg = lg.rename(columns={"mean_r": "mean R"}).iloc[::-1]
            _set(self.rlog, lg[[c for c in ["day", "rule", "from", "to", "n", "mean R", "t", "why"] if c in lg]], "copilot_rules")
        else:
            _set(self.rlog, pd.DataFrame(), "copilot_rules")
