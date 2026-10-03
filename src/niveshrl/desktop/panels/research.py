"""RANK (deep-learning rankers), RISK (volatility + regimes) and RL / PLAN (allocator + investor plan)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from ...config import ROOT
from ...research.regime import FEATS, REGIME_COLORS, REGIMES
from ...research.signals import MODEL_LABELS
from .. import data, theme
from ..widgets import ExplainButton, FrameTable, KpiRow, h2, line_chart, muted, run_async
from . import Panel, vbox

RANK_MODELS = ["ffnn", "lstm", "transformer", "logreg", "momentum"]
LABEL = MODEL_LABELS | {"momentum": "Momentum 12-1 (baseline)"}
VLABEL = {"hist": "Historical 21d", "ewma": "EWMA λ=0.94", "garch": "GARCH(1,1)", "lstm": "LSTM (deep learning)"}
DISCLAIMER = ("Educational project, not investment advice. NiveshRL is not registered with SEBI as a Research Analyst "
              "or Investment Adviser. Backtests use historical data, include survivorship bias, and past performance "
              "does not predict future returns.")


def _tab(*widgets, stretch_last: bool = True) -> QWidget:
    w = QWidget()
    lay = vbox(w)
    for i, x in enumerate(widgets):
        lay.addWidget(x, 1 if (stretch_last and i == len(widgets) - 1) else 0)
    return w


def _clear(box) -> None:
    while box.count():
        old = box.takeAt(0).widget()
        if old:
            old.deleteLater()


class RankersPanel(Panel):
    """Monthly deep-learning stock rankers. Tabs: today's ranking (searchable), the out-of-sample
    scoreboard, history charts and the next-day model's walk-forward record."""
    title = "Rankers"
    code = "RANK"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        self.head = KpiRow(cols=5)
        lay.addWidget(self.head)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        # --- tab 1: live ranking
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search symbol or company…")
        self.search.textChanged.connect(self._filter)
        self.sector = QComboBox()
        self.sector.currentIndexChanged.connect(self._filter)
        self.show_n = QComboBox()
        for label, n in [("All 194", None), ("Top 20", 20), ("Top 50", 50), ("Bottom 20", -20)]:
            self.show_n.addItem(label, n)
        self.show_n.currentIndexChanged.connect(self._filter)
        bar.addWidget(self.search, 3)
        bar.addWidget(self.sector, 2)
        bar.addWidget(self.show_n, 1)
        self.count = muted("")
        bar.addWidget(self.count, 2)
        top = QWidget()
        top.setLayout(bar)
        self.live = FrameTable()
        self.live.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        self.tabs.addTab(_tab(top, muted(
            "Ranked by Consensus = the average percentile of every model (100% = the models' favourite for next month). "
            "Model columns are each model's percentile; Agreement near 100% means the models concur. "
            "Click a row to open the stock. Research output, not a recommendation."), self.live), "Live ranking")

        # --- tab 2: scoreboard
        self.board = FrameTable(fmt={"AUC": "{:.3f}", "Accuracy": "{:.1%}", "IC mean": "{:.3f}", "IC t-stat": "{:.2f}",
                                     "IC hit rate": "{:.0%}", "Top decile / mo": "{:+.2%}", "Bottom decile / mo": "{:+.2%}",
                                     "Spread / mo": "{:+.2%}", "Spread t-stat": "{:.2f}"},
                                signed={"Top decile / mo", "Bottom decile / mo", "Spread / mo"})
        self.tabs.addTab(_tab(h2("Out-of-sample scoreboard · walk-forward, 2012 → today, before costs"), self.board, muted(
            "IC = rank correlation between a model's score and next month's return, averaged over months. "
            "Spread = top-decile minus bottom-decile monthly return. A t-stat above ~2 means the result is unlikely to be luck. "
            "Same universe, months and features for every model."), stretch_last=False), "Scoreboard")

        # --- tab 3: history charts
        self.hist_box = QVBoxLayout()
        hw = QWidget()
        hw.setLayout(self.hist_box)
        self.tabs.addTab(hw, "History charts")

        # --- tab 4: next-day model
        self.next_box = FrameTable(fmt={"AUC": "{:.3f}", "Accuracy": "{:.1%}", "IC mean": "{:.3f}", "IC t-stat": "{:.1f}",
                                        "Top10 hit rate": "{:.1%}", "Top10 excess / day": "{:+.2%}",
                                        "Top10 net of costs / day": "{:+.2%}", "Days": "{:,.0f}"},
                                   signed={"Top10 excess / day", "Top10 net of costs / day"})
        self.tabs.addTab(_tab(h2("Next-day model · walk-forward 2015 → today"), self.next_box, muted(
            "Predicts whether each stock beats tomorrow's cross-sectional median. The signal is real and consistent, but "
            "small: the top 10 picks lose money after ~0.25% round-trip costs, so the app uses it as a watch list (TODAY), "
            "never as a trading system."), stretch_last=False), "Next-day model")
        self.table = pd.DataFrame()

    def refresh(self) -> None:
        tab = data.results_csv("rankers_summary.csv")
        if tab is not None:
            tab = tab.drop(columns=[c for c in ["Train time (s)", "Months"] if c in tab.columns])
            tab.index = [LABEL.get(i, i) for i in tab.index]
            self.board.set_frame(tab)
        nd = data.results_csv("nextday_summary.csv")
        if nd is not None:
            nd.index = [{"ensemble": "Ensemble (LightGBM + SeqNet)", "lgbm": "LightGBM", "seq": "SeqNet (CNN + Transformer)",
                         "logreg": "Logistic regression", "reversal": "Short-term reversal"}.get(i, i) for i in nd.index]
            self.next_box.set_frame(nd)
        models = [m for m in RANK_MODELS if data.predictions(m) is not None]
        if not models:
            self.live.set_frame(pd.DataFrame({"": ["No ranker predictions yet. Run python scripts/train_rankers.py."]}))
            return
        p = data.panel()
        frames, asof = [], None
        for m in models:
            pr = data.predictions(m)
            d = pr.index.get_level_values(0).max()
            asof = d if asof is None else max(asof, d)
            s_ = pr.xs(d, level=0)["score"]
            frames.append(s_.rank(pct=True).rename(m.upper()))
        t = pd.concat(frames, axis=1)
        pct = list(t.columns)
        t["Consensus"] = t[pct].mean(axis=1)
        t["Agreement"] = 1 - t[pct].std(axis=1) * 2
        close = p.close.ffill()
        t.insert(0, "1M ret", (close.iloc[-1] / close.iloc[-22] - 1).reindex(t.index))
        t.insert(0, "Sector", p.sectors.reindex(t.index).fillna("").astype(object))
        t.insert(0, "Company", p.names.reindex(t.index).fillna("").astype(object).str[:28])
        t = t.sort_values("Consensus", ascending=False)
        t.insert(0, "Rank", np.arange(1, len(t) + 1))
        self.table = t
        self.live.model_.fmt = {c: "{:.0%}" for c in pct + ["Consensus", "Agreement"]} | {"1M ret": "{:+.1%}", "Rank": "{:.0f}"}
        self.live.model_.signed = {"1M ret"}
        self.sector.blockSignals(True)
        self.sector.clear()
        self.sector.addItem("All sectors", None)
        for sec in sorted(x for x in t["Sector"].unique() if x):
            self.sector.addItem(sec, sec)
        self.sector.blockSignals(False)
        self._filter()
        best = t.iloc[0]
        self.head.set_items([
            ("Predictions as of", f"{asof:%d %b %Y}", theme.AMBER, "month-end features"),
            ("Predicting", "next month", None, "beat the median?"),
            ("Models", str(len(models)), None, ", ".join(m.upper() for m in models)),
            ("Top consensus", t.index[0].replace(".NS", ""), theme.GREEN, f"{best['Consensus']:.0%} avg percentile"),
            ("Bottom consensus", t.index[-1].replace(".NS", ""), theme.RED, f"{t.iloc[-1]['Consensus']:.0%} avg percentile"),
        ])
        self._history(models)

    def _filter(self) -> None:
        t = self.table
        if t.empty:
            return
        q = self.search.text().strip().lower()
        if q:
            t = t[t.index.str.lower().str.contains(q, regex=False) | t["Company"].str.lower().str.contains(q, regex=False)]
        sec = self.sector.currentData()
        if sec:
            t = t[t["Sector"] == sec]
        n = self.show_n.currentData()
        if n:
            t = t.head(n) if n > 0 else t.tail(-n)
        self.count.setText(f"{len(t)} of {len(self.table)} stocks")
        self.live.set_frame(t)

    def _history(self, models: list) -> None:
        _clear(self.hist_box)
        row = QHBoxLayout()
        ic_plot = pg.PlotWidget(title="Mean monthly IC by year (higher = better ranking)")
        dec_plot = pg.PlotWidget(title="Average next-month return by score decile (%), D1 = worst, D10 = best")
        ic_plot.addLegend(offset=(60, 4), colCount=len(models))
        dec_plot.addLegend(offset=(60, 4), colCount=len(models))
        for w in (ic_plot, dec_plot):
            w.showGrid(x=True, y=True, alpha=0.15)
            w.setMouseEnabled(x=False, y=False)            # small summary charts: nothing to navigate
            w.setMenuEnabled(False)
        width = 0.8 / len(models)
        spreads = {}
        for i, m in enumerate(models):
            dg = data.diagnostics(m)
            col = theme.SERIES[i % len(theme.SERIES)]
            y = dg["by_year"]
            ic_plot.addItem(pg.BarGraphItem(x=np.asarray(y.index, float) + (i - (len(models) - 1) / 2) * width,
                                            height=y["IC"].to_numpy(), width=width, brush=col, name=m.upper()))
            dec = dg["dec"]
            dec_plot.plot(np.arange(1, dec.shape[1] + 1), dec.mean().to_numpy() * 100, pen=pg.mkPen(col, width=2),
                          symbol="o", symbolSize=6, symbolBrush=col, name=m.upper())
            sp = (dec.iloc[:, -1] - dec.iloc[:, 0]).fillna(0)
            spreads[m.upper()] = (1 + sp).cumprod()
        dec_plot.getAxis("bottom").setTicks([[(i, f"D{i}") for i in range(1, 11)]])
        for w in (ic_plot, dec_plot):                    # headroom so the one-row legend clears the data
            (y0, y1) = w.getViewBox().childrenBounds()[1] or (0, 1)
            w.setYRange(y0, y1 + (y1 - y0) * 0.22, padding=0.02)
        for plot, key in ((ic_plot, "chart_ic_year"), (dec_plot, "chart_decile")):
            col = QVBoxLayout()
            col.addWidget(ExplainButton(key), 0, Qt.AlignLeft)
            col.addWidget(plot, 1)
            row.addLayout(col)
        top = QWidget()
        top.setLayout(row)
        top.setMinimumHeight(260)
        self.hist_box.addWidget(top, 1)
        first = next(iter(spreads.values()))
        al = {k: v.reindex(first.index) for k, v in spreads.items()}
        self.hist_box.addWidget(line_chart(al, "Cumulative top-minus-bottom decile (log, before costs)", logy=True), 1)


class RiskPanel(Panel):
    title = "Risk"
    code = "RISK"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self.vol = FrameTable(fmt={"RMSE log-vol": "{:.3f}", "MAE vol": "{:.2%}", "QLIKE": "{:.3f}", "Corr": "{:.3f}",
                                   "Bias (f/r)": "{:.2f}", "N": "{:,.0f}"})
        self.vol.setMaximumHeight(170)
        self.vol_year = QVBoxLayout()
        yw = QWidget()
        yw.setLayout(self.vol_year)
        self.tabs.addTab(_tab(h2("Volatility forecaster · next-month realised volatility"), self.vol, muted(
            "Walk-forward, 194 stocks, month-end forecasts of the next 21 trading days. Lower RMSE/QLIKE and higher "
            "correlation are better; Bias = median forecast ÷ realised (1.00 = unbiased)."), yw), "Volatility")
        self.kpis = KpiRow(cols=4)
        self.reg_plot_box = QVBoxLayout()
        rw = QWidget()
        rw.setLayout(self.reg_plot_box)
        tables = QWidget()
        row = QHBoxLayout(tables)
        row.setContentsMargins(0, 0, 0, 0)
        self.stats = FrameTable(fmt={"weeks": "{:.0f}", "share": "{:.0%}", "next 4w NIFTY return": "{:+.2%}",
                                     "next-month NIFTY vol": "{:.1%}", "hit rate (4w > 0)": "{:.0%}"})
        self.feats = FrameTable(fmt={f: "{:.3f}" for f in FEATS})
        for t in (self.stats, self.feats):
            row.addWidget(t)
        tables.setMaximumHeight(170)
        regw = QWidget()
        rl = vbox(regw)
        rl.addWidget(self.kpis)
        rl.addWidget(rw, 1)
        rl.addWidget(tables)
        rl.addWidget(muted("Regimes separate future volatility clearly; they do not reliably predict direction. "
                           "Test a regime filter in the Backtest lab."))
        self.tabs.addTab(regw, "Market regimes")

    def refresh(self) -> None:
        from ...research.regime import regime_stats
        from ...research.volatility import VOL_MODELS, vol_metrics
        vf = data.vol_forecasts()
        _clear(self.vol_year)
        if vf is not None:
            d = vf.dropna(subset=["garch", "lstm"])
            self.vol.set_frame(pd.DataFrame({VLABEL[m]: vol_metrics(d[m], d["rv_next"]) for m in VOL_MODELS}).T)
            dates = d.index.get_level_values(0)
            yr = pd.DataFrame({VLABEL[m]: np.sqrt(((np.log(d[m]) - np.log(d["rv_next"])) ** 2).groupby(dates.year).mean())
                               for m in VOL_MODELS})
            pw = pg.PlotWidget(title="Forecast error by year: RMSE of log-volatility (lower = better)")
            pw.addLegend(offset=(10, 10))
            for i, c in enumerate(yr.columns):
                pw.plot(np.asarray(yr.index, float), yr[c].to_numpy(), pen=pg.mkPen(theme.SERIES[i], width=2), symbol="o",
                        symbolSize=6, symbolBrush=theme.SERIES[i], name=c)
            pw.showGrid(x=True, y=True, alpha=0.15)
            pw.setMouseEnabled(x=False, y=False)
            pw.setMenuEnabled(False)
            self.vol_year.addWidget(ExplainButton("chart_vol_year"), 0, Qt.AlignLeft)
            self.vol_year.addWidget(pw)
        else:
            self.vol.set_frame(pd.DataFrame({"": ["No volatility forecasts. Run python scripts/train_volatility.py."]}))
        w = data.regimes_weekly()
        _clear(self.reg_plot_box)
        if w is None:
            return
        p = data.panel()
        cur = w.iloc[-1]
        self.kpis.set_items([("Current regime", cur["regime"], REGIME_COLORS.get(cur["regime"]), f"week ending {w.index[-1]:%d %b %Y}")]
                            + [(f, f"{cur[f]:.3f}", None) for f in ["nifty_ret_20", "breadth", "nifty_dd"]])
        b = p.bench.resample("W-FRI").last().reindex(w.index)
        chart = line_chart({"NIFTY 50 (weekly)": b}, "NIFTY 50, each week coloured by its out-of-sample regime")
        chart.default = "5Y"
        plot = chart.plots[0]
        x = np.arange(len(w))
        for r in REGIMES:
            m = (w["regime"] == r).to_numpy()
            plot.plot(x[m], b.to_numpy()[m], pen=None, symbol="o", symbolSize=5, symbolBrush=REGIME_COLORS[r],
                      symbolPen=None, name=r)
        chart.set_range("5Y")
        self.reg_plot_box.addWidget(chart)
        self.stats.set_frame(regime_stats(w, p))
        self.feats.set_frame(w.groupby("regime")[FEATS].mean().reindex(REGIMES))


class PlanPanel(Panel):
    """RL allocator research results plus the personalised investor plan (runs the RL policy in a thread)."""
    title = "RL allocator and plan"
    code = "PLAN"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        # research tab
        res = QWidget()
        rl = QVBoxLayout(res)
        rl.addWidget(h2("RL allocator · 29 NIFTY 50 stocks, weekly · validation 2019–20"))
        self.metrics = FrameTable(fmt={"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                                       "MaxDD": "{:.1%}", "Turnover/yr": "{:.2f}", "Costs(Rs)": "₹{:,.0f}"})
        self.metrics.setMinimumHeight(160)
        rl.addWidget(self.metrics)
        self.nav_box = QVBoxLayout()
        rl.addLayout(self.nav_box, 1)
        self.tabs.addTab(res, "Research")
        # investor plan tab
        plan = QWidget()
        pl = QVBoxLayout(plan)
        pl.addWidget(muted(DISCLAIMER))
        from ...profile import QUESTIONNAIRE
        form = QFormLayout()
        self.answers = {}
        for q in QUESTIONNAIRE:
            if q.get("numeric"):
                w = QSpinBox()
                w.setRange(0, 100_000_000)
                w.setSingleStep(1000)
                w.setValue(50_000 if q["key"] == "initial" else 5_000)
                w.setPrefix("₹ ")
            else:
                w = QComboBox()
                w.addItems(list(q["options"]))
                w.setCurrentIndex(min(1, len(q["options"]) - 1))
            self.answers[q["key"]] = w
            form.addRow(q["q"], w)
        pl.addLayout(form)
        self.go = QPushButton("Build my plan")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self._plan)
        pl.addWidget(self.go)
        self.plan_out = QTextBrowser()
        self.orders = FrameTable()
        row = QHBoxLayout()
        row.addWidget(self.plan_out, 1)
        row.addWidget(self.orders, 1)
        pl.addLayout(row, 1)
        self.tabs.addTab(plan, "Investor plan")

    def refresh(self) -> None:
        m = data.results_csv("metrics_val.csv")
        if m is not None:
            self.metrics.set_frame(m)
        nav = data.results_csv("baselines_val_nav.csv")
        if nav is not None:
            nav.index = pd.to_datetime(nav.index)
            while self.nav_box.count():
                w = self.nav_box.takeAt(0).widget()
                if w:
                    w.deleteLater()
            self.nav_box.addWidget(line_chart({c: nav[c] for c in nav.columns}, "Growth of ₹1 after costs (validation)",
                                              logy=True))

    @staticmethod
    def _run_name() -> str | None:
        runs = sorted(str(p.parent) for p in (ROOT / "runs").glob("*/best.pt") if (p.parent / "args.json").exists())
        if not runs:
            return None
        return next((r for r in runs if Path(r).name == "verify_split_s0"), runs[0])

    def _plan(self) -> None:
        run = self._run_name()
        if run is None:
            self.plan_out.setHtml("No trained RL model found. Train one first: <code>python scripts/train_custom.py</code>.")
            return
        ans = {k: (w.value() if isinstance(w, QSpinBox) else w.currentText()) for k, w in self.answers.items()}
        self.go.setEnabled(False)
        self.plan_out.setHtml("Building your plan (loading the RL policy)…")
        run_async(_build_plan, self._show_plan, run, ans, on_error=self._err)

    def _err(self, msg: str) -> None:
        self.go.setEnabled(True)
        self.plan_out.setHtml(f"<pre>{msg[-800:]}</pre>")

    def _show_plan(self, out: dict) -> None:
        self.go.setEnabled(True)
        prof = out["profile"]
        w = out["weights"]
        rows = "".join(f"<tr><td>{k}</td><td align=right>{v:.1%}</td></tr>" for k, v in w.items())
        why = "".join(f"<li>{x}</li>" for x in out["why"])
        self.plan_out.setHtml(
            f"<p style='color:{theme.MUTED}'>Profile → risk aversion {prof.risk_aversion:.1f}, drawdown tolerance "
            f"{prof.dd_tol:.0%}, horizon {prof.horizon_years:g} yrs, cash buffer {prof.min_cash:.0%}, up to "
            f"{prof.max_stocks} stocks</p><h3 style='color:{theme.AMBER}'>Plan as of {out['date']}</h3>"
            f"<table cellpadding=3>{rows}</table><p>Cash left after whole-share orders: ₹{out['cash_left']:,.0f}</p>"
            f"<h3 style='color:{theme.AMBER}'>Why these stocks?</h3><ul>{why}</ul>")
        self.orders.set_frame(out["orders"])


_RL_CACHE: dict = {}


def _build_plan(run: str, ans: dict) -> dict:
    from ...agents import TorchPolicy, load_run
    from ...backtest import make_eval_env
    from ...explain import explain_change
    from ...pipeline import load_all
    from ...planner import orders_frame, plan_orders
    from ...profile import from_answers
    if "data" not in _RL_CACHE:
        _RL_CACHE["data"] = load_all()
    cfg, md, fs = _RL_CACHE["data"]
    if _RL_CACHE.get("run") != run:
        _RL_CACHE["model"], _RL_CACHE["run"] = load_run(run, md, fs, cfg), run
    model = _RL_CACHE["model"]
    prof = from_answers(ans)
    env = make_eval_env(md, fs, cfg, "forward", profile=prof)
    env.reset(seed=0)
    env.t = env.hi
    obs = env.observation()
    w_old = env.current_weights()
    w_new = env.feasible(TorchPolicy(model)(env))
    orders, cash_left = plan_orders(w_new, md.tickers, env.prices[env.t], cash=prof.initial)
    ws = pd.Series(w_new, index=[t.replace(".NS", "") for t in md.tickers] + ["Cash"])
    ws = ws[ws > 0.005].sort_values(ascending=False)
    return {"profile": prof, "weights": ws.to_dict(), "orders": orders_frame(orders), "cash_left": cash_left,
            "date": str(env.dates[env.t].date()),
            "why": explain_change(model, obs, md.tickers, md.sectors, w_old, w_new, threshold=0.03)[:8]}
