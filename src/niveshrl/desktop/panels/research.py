"""RISK: volatility forecasts and market regimes. (The Rankers and RL & plan screens were removed on 2026-11-17:
no measured trading edge; their research code stays in the repo.)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QTabWidget, QVBoxLayout, QWidget

from ...research.regime import FEATS, REGIME_COLORS, REGIMES
from .. import data, theme
from ..widgets import ExplainButton, FrameTable, KpiRow, h2, line_chart, muted
from . import Panel, vbox

VLABEL = {"hist": "Historical 21d", "ewma": "EWMA λ=0.94", "garch": "GARCH(1,1)", "lstm": "LSTM (deep learning)"}


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
