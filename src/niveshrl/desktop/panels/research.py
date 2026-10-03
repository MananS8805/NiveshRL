"""RANK (deep-learning rankers), RISK (volatility + regimes) and RL / PLAN (allocator + investor plan)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QPushButton, QSpinBox, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from ...config import ROOT
from ...research.regime import FEATS, REGIME_COLORS, REGIMES
from ...research.signals import MODEL_LABELS
from .. import data, theme
from ..widgets import FrameTable, KpiRow, _DateAxis, h2, line_chart, muted, run_async
from . import Panel, scrolling, vbox

RANK_MODELS = ["ffnn", "lstm", "transformer", "logreg", "momentum"]
LABEL = MODEL_LABELS | {"momentum": "Momentum 12-1 (baseline)"}
VLABEL = {"hist": "Historical 21d", "ewma": "EWMA λ=0.94", "garch": "GARCH(1,1)", "lstm": "LSTM (deep learning)"}
DISCLAIMER = ("Educational project, not investment advice. NiveshRL is not registered with SEBI as a Research Analyst "
              "or Investment Adviser. Backtests use historical data, include survivorship bias, and past performance "
              "does not predict future returns.")


class RankersPanel(Panel):
    title = "Rankers"
    code = "RANK"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        outer = vbox(self, 0)
        inner = QWidget()
        outer.addWidget(scrolling(inner))
        self.lay = vbox(inner)
        self.lay.addWidget(h2("Out-of-sample scoreboard (walk-forward, 2012 → today, before costs)"))
        self.board = FrameTable(fmt={"AUC": "{:.3f}", "Accuracy": "{:.1%}", "IC mean": "{:.3f}", "IC t-stat": "{:.2f}",
                                     "IC hit rate": "{:.0%}", "Top decile / mo": "{:+.2%}", "Bottom decile / mo": "{:+.2%}",
                                     "Spread / mo": "{:+.2%}", "Spread t-stat": "{:.2f}"})
        self.board.setMinimumHeight(170)
        self.lay.addWidget(self.board)
        self.lay.addWidget(muted("IC = Spearman rank correlation between the score and next month's return. Spread = "
                                 "top-decile minus bottom-decile return. A t-stat above ~2 means the average is unlikely to be luck."))
        self.next_box = FrameTable(fmt={c: "{:.3f}" for c in ["AUC", "IC", "IC t"]} | {"Accuracy": "{:.1%}"})
        self.charts = QHBoxLayout()
        self.lay.addLayout(self.charts)
        self.lay.addWidget(h2("Next-day model (walk-forward by year)"))
        self.next_box.setMinimumHeight(160)
        self.lay.addWidget(self.next_box)
        self.lay.addWidget(h2("Live ranking · predicting next month"))
        self.live = FrameTable()
        self.live.setMinimumHeight(420)
        self.live.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        self.lay.addWidget(self.live)
        self.lay.addWidget(muted("P = probability the stock beats the cross-sectional median next month (0.5 = no view). "
                                 "Research output, not a recommendation."))

    def refresh(self) -> None:
        tab = data.results_csv("rankers_summary.csv")
        if tab is not None:
            tab = tab.drop(columns=[c for c in ["Train time (s)", "Months"] if c in tab.columns])
            tab.index = [LABEL.get(i, i) for i in tab.index]
            self.board.set_frame(tab)
        models = [m for m in RANK_MODELS if data.predictions(m) is not None]
        while self.charts.count():
            w = self.charts.takeAt(0).widget()
            if w:
                w.deleteLater()
        if models:
            ic_plot = pg.PlotWidget(title="Mean monthly IC by year")
            dec_plot = pg.PlotWidget(title="Avg next-month return by decile (%)")
            ic_plot.addLegend()
            dec_plot.addLegend()
            width = 0.8 / len(models)
            for i, m in enumerate(models):
                dg = data.diagnostics(m)
                col = theme.SERIES[i % len(theme.SERIES)]
                y = dg["by_year"]
                ic_plot.addItem(pg.BarGraphItem(x=np.asarray(y.index, float) + (i - len(models) / 2) * width, height=y["IC"].to_numpy(),
                                                width=width, brush=col, name=m.upper()))
                dec = dg["dec"]
                dec_plot.plot(np.arange(1, dec.shape[1] + 1), dec.mean().to_numpy() * 100, pen=pg.mkPen(col, width=2),
                              symbol="o", symbolSize=5, symbolBrush=col, name=m.upper())
            for w in (ic_plot, dec_plot):
                w.setMinimumHeight(280)
                w.showGrid(x=True, y=True, alpha=0.15)
                self.charts.addWidget(w)
            p = data.panel()
            frames, probs = [], []
            for m in models:
                pr = data.predictions(m)
                d = pr.index.get_level_values(0).max()
                s = pr.xs(d, level=0)["score"]
                frames.append(s.rank(pct=True).rename(f"{m.upper()} %ile"))
                if m != "momentum":
                    probs.append(s.rename(f"{m.upper()} P"))
            t = pd.concat(frames + probs, axis=1)
            pct = [c for c in t.columns if c.endswith("%ile")]
            t["Consensus"] = t[pct].mean(axis=1)
            t["Agreement"] = 1 - t[pct].std(axis=1) * 2
            ret1m = p.close.ffill().iloc[-1] / p.close.ffill().iloc[-22] - 1
            t.insert(0, "1M ret", ret1m.reindex(t.index))
            t.insert(0, "Sector", p.sectors.reindex(t.index))
            t = t.sort_values("Consensus", ascending=False)
            self.live.model_.fmt = {c: "{:.0%}" for c in t.columns if c not in ("Sector", "1M ret")} | {"1M ret": "{:+.1%}"}
            self.live.model_.signed = {"1M ret"}
            self.live.set_frame(t)
        nd = data.results_csv("nextday_summary.csv")
        if nd is not None:
            self.next_box.set_frame(nd)
        else:
            self.next_box.set_frame(pd.DataFrame({"": ["Run python scripts/train_nextday.py to produce the walk-forward table."]}))


class RiskPanel(Panel):
    title = "Risk"
    code = "RISK"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        outer = vbox(self, 0)
        inner = QWidget()
        outer.addWidget(scrolling(inner))
        lay = vbox(inner)
        lay.addWidget(h2("Volatility forecaster · next-month realised vol"))
        self.vol = FrameTable(fmt={"RMSE log-vol": "{:.3f}", "MAE vol": "{:.2%}", "QLIKE": "{:.3f}", "Corr": "{:.3f}",
                                   "Bias (f/r)": "{:.2f}", "N": "{:,.0f}"})
        self.vol.setMinimumHeight(150)
        lay.addWidget(self.vol)
        lay.addWidget(muted("Walk-forward, 194 stocks, month-end forecasts of the next 21 trading days. Lower RMSE/QLIKE "
                            "and higher correlation are better; Bias = median forecast/realised (1.00 = unbiased)."))
        self.vol_year = QVBoxLayout()
        lay.addLayout(self.vol_year)
        lay.addWidget(h2("Market regimes · autoencoder + k-means (walk-forward)"))
        self.kpis = KpiRow(cols=4)
        lay.addWidget(self.kpis)
        self.reg_plot_box = QVBoxLayout()
        lay.addLayout(self.reg_plot_box)
        row = QHBoxLayout()
        self.stats = FrameTable(fmt={"weeks": "{:.0f}", "share": "{:.0%}", "next 4w NIFTY return": "{:+.2%}",
                                     "next-month NIFTY vol": "{:.1%}", "hit rate (4w > 0)": "{:.0%}"})
        self.feats = FrameTable(fmt={f: "{:.3f}" for f in FEATS})
        for t in (self.stats, self.feats):
            t.setMinimumHeight(140)
            row.addWidget(t)
        lay.addLayout(row)
        lay.addWidget(muted("Regimes separate future volatility clearly; they do not reliably predict direction. "
                            "Test a regime filter in the LAB."))

    def refresh(self) -> None:
        from ...research.regime import regime_stats
        from ...research.volatility import VOL_MODELS, vol_metrics
        vf = data.vol_forecasts()
        if vf is not None:
            d = vf.dropna(subset=["garch", "lstm"])
            self.vol.set_frame(pd.DataFrame({VLABEL[m]: vol_metrics(d[m], d["rv_next"]) for m in VOL_MODELS}).T)
            dates = d.index.get_level_values(0)
            yr = {VLABEL[m]: np.sqrt(((np.log(d[m]) - np.log(d["rv_next"])) ** 2).groupby(dates.year).mean())
                  for m in VOL_MODELS}
            self._replace(self.vol_year, self._year_plot(pd.DataFrame(yr), "RMSE of log-vol by year (lower = better)"))
        w = data.regimes_weekly()
        if w is None:
            return
        p = data.panel()
        cur = w.iloc[-1]
        self.kpis.set_items([("Current regime", cur["regime"], REGIME_COLORS.get(cur["regime"]), f"week of {w.index[-1].date()}")]
                            + [(f, f"{cur[f]:.3f}", None) for f in ["nifty_ret_20", "breadth", "nifty_dd"]])
        b = p.bench.resample("W-FRI").last().reindex(w.index)
        pw = pg.PlotWidget(axisItems={"bottom": _DateAxis(list(w.index))},
                           title="NIFTY 50, each week coloured by its out-of-sample regime label")
        x = np.arange(len(w))
        pw.plot(x, b.to_numpy(), pen=pg.mkPen("#39424E", width=1))
        for r in REGIMES:
            m = (w["regime"] == r).to_numpy()
            pw.plot(x[m], b.to_numpy()[m], pen=None, symbol="o", symbolSize=4, symbolBrush=REGIME_COLORS[r],
                    symbolPen=None, name=r)
        pw.addLegend()
        pw.setMinimumHeight(340)
        self._replace(self.reg_plot_box, pw)
        self.stats.set_frame(regime_stats(w, p))
        self.feats.set_frame(w.groupby("regime")[FEATS].mean().reindex(REGIMES))

    @staticmethod
    def _replace(box, widget) -> None:
        while box.count():
            old = box.takeAt(0).widget()
            if old:
                old.deleteLater()
        box.addWidget(widget)

    @staticmethod
    def _year_plot(df: pd.DataFrame, title: str) -> pg.PlotWidget:
        pw = pg.PlotWidget(title=title)
        pw.addLegend()
        for i, c in enumerate(df.columns):
            pw.plot(np.asarray(df.index, float), df[c].to_numpy(), pen=pg.mkPen(theme.SERIES[i], width=2), symbol="o",
                    symbolSize=5, symbolBrush=theme.SERIES[i], name=c)
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.setMinimumHeight(260)
        return pw


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
