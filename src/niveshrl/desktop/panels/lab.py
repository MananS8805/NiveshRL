"""LAB: backtest lab. Build a strategy from any signal, run it with real NSE costs (C++ engine
when installed), and inspect performance, drawdowns, monthly returns, trading and holdings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout,
                               QLineEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ...metrics import drawdown
from ...research import backtest as bt
from ...research.regime import REGIMES
from .. import data, theme
from ..widgets import FrameTable, KpiRow, line_chart, muted, run_async
from . import Panel, vbox


class LabPanel(Panel):
    title = "Backtest lab"
    code = "LAB"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        form = QGridLayout()
        self.signal = QComboBox()
        self.size = QComboBox()
        self.size.addItems(["Top N", "Top decile"])
        self.n = QSpinBox()
        self.n.setRange(5, 50)
        self.n.setValue(20)
        self.weighting = QComboBox()
        for k, v in {"equal": "Equal", "inv_vol": "Inverse volatility", "score": "Score-weighted"}.items():
            self.weighting.addItem(v, k)
        self.reb = QComboBox()
        for m in (1, 2, 3, 6, 12):
            self.reb.addItem(f"every {m} month(s)", m)
        self.maxw = QDoubleSpinBox()
        self.maxw.setRange(0.02, 0.25)
        self.maxw.setSingleStep(0.01)
        self.maxw.setValue(0.10)
        self.cost = QDoubleSpinBox()
        self.cost.setRange(0, 4)
        self.cost.setSingleStep(0.25)
        self.cost.setValue(1.0)
        self.capital = QSpinBox()
        self.capital.setRange(50_000, 100_000_000)
        self.capital.setSingleStep(50_000)
        self.capital.setValue(1_000_000)
        self.capital.setPrefix("₹ ")
        self.vt = QComboBox()
        for v in (None, 0.10, 0.12, 0.15, 0.20):
            self.vt.addItem("Off" if v is None else f"{v:.0%} annual", v)
        self.y0, self.y1 = QSpinBox(), QSpinBox()
        self.ls = QCheckBox("Long-short (research only)")
        self.reg = QCheckBox("Regime filter")
        self.rx = {}
        rxw = QHBoxLayout()
        for r, v in {"Bull": 1.0, "Neutral": 0.8, "Stress": 0.3}.items():
            sp = QDoubleSpinBox()
            sp.setRange(0, 1)
            sp.setSingleStep(0.1)
            sp.setValue(v)
            sp.setPrefix(f"{r} ")
            self.rx[r] = sp
            rxw.addWidget(sp)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Strategy name (optional)")
        items = [("Signal", self.signal), ("Portfolio size", self.size), ("N stocks", self.n),
                 ("Weighting", self.weighting), ("Rebalance", self.reb), ("Max weight", self.maxw),
                 ("Cost multiplier", self.cost), ("Capital", self.capital), ("Vol target", self.vt),
                 ("From year", self.y0), ("To year", self.y1)]
        for i, (label, w) in enumerate(items):
            f = QFormLayout()
            f.addRow(muted(label), w)
            form.addLayout(f, i // 4, i % 4)
        lay.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(self.ls)
        row.addWidget(self.reg)
        row.addLayout(rxw)
        row.addWidget(self.name, 1)
        self.run_btn = QPushButton("▶ RUN")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        row.addWidget(self.run_btn)
        lay.addLayout(row)
        self.status = muted("")
        lay.addWidget(self.status)
        self.kpis = KpiRow(cols=6)
        lay.addWidget(self.kpis)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

    def refresh(self) -> None:
        sig = data.signals()
        cur = self.signal.currentData()
        self.signal.clear()
        for k, v in sig.items():
            self.signal.addItem(v, k)
        i = self.signal.findData(cur or "ffnn")
        self.signal.setCurrentIndex(max(i, 0))
        last = data.panel().close.index[-1].year
        for sp, v in [(self.y0, 2012), (self.y1, last)]:
            sp.setRange(2012, last)
            sp.setValue(v)

    def spec(self) -> bt.StrategySpec:
        decile = self.size.currentText() == "Top decile"
        vt = self.vt.currentData()
        rx = {r: sp.value() for r, sp in self.rx.items()} if self.reg.isChecked() else None
        sig_label = self.signal.currentText().split(" (")[0]
        label = self.name.text() or f"{sig_label} · {'D10' if decile else f'top {self.n.value()}'} · {self.weighting.currentData()}" \
            f"{' · LS' if self.ls.isChecked() else ''}{f' · VT{vt:.0%}' if vt else ''}{' · RF' if rx else ''}"
        return bt.StrategySpec(name=label, signal=self.signal.currentData(), top=0.1 if decile else float(self.n.value()),
                               weighting=self.weighting.currentData(), long_short=self.ls.isChecked(),
                               rebalance_months=int(self.reb.currentData()), max_weight=float(self.maxw.value()),
                               cost_scale=float(self.cost.value()), capital=float(self.capital.value()), vol_target=vt,
                               regime_exposure=rx, start=f"{self.y0.value()}-01-01", end=f"{self.y1.value()}-12-31")

    def _run(self) -> None:
        spec = self.spec()
        self.run_btn.setEnabled(False)
        self.status.setText(f"Running {spec.name}…")

        def work():
            res = data.backtest(spec)
            ew = data.backtest(bt.StrategySpec(name="Equal weight (universe)", signal="equal", top=1.0, max_weight=0.05,
                                               cost_scale=spec.cost_scale, capital=spec.capital, start=spec.start,
                                               end=spec.end))
            return res, ew
        run_async(work, self._show, on_error=self._err)

    def _err(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText("Backtest failed: " + msg.strip().splitlines()[-1][:200])

    def _show(self, out) -> None:
        res, ew = out
        self.run_btn.setEnabled(True)
        spec = res["spec"]
        nav = res["nav"]
        m = res["metrics"]
        b = data.panel().bench.reindex(nav.index).ffill()
        b = b / b.iloc[0]
        bcagr = b.iloc[-1] ** (365.25 / max((b.index[-1] - b.index[0]).days, 1)) - 1
        engine = "C++ engine" if bt._core() is not None else "NumPy engine"
        self.status.setText(f"{spec.name} · {nav.index[0]:%b %Y} → {nav.index[-1]:%b %Y} · {engine}")
        g = lambda k: m.get(k, float("nan"))  # noqa: E731
        self.kpis.set_items([
            ("CAGR", f"{g('CAGR'):+.1%}", theme.signed(g("CAGR")), f"NIFTY {bcagr:+.1%}"),
            ("Sharpe", f"{g('Sharpe'):.2f}", None, "rf 6.5%"), ("Sortino", f"{g('Sortino'):.2f}", None),
            ("Volatility", f"{g('Vol'):.1%}", None, "annualised"),
            ("Max drawdown", f"{g('MaxDD'):.1%}", theme.RED), ("Calmar", f"{g('Calmar'):.2f}", None),
            ("Alpha vs NIFTY", f"{g('Alpha'):+.1%}", theme.signed(g("Alpha")), f"beta {g('Beta'):.2f}"),
            ("Info ratio", f"{g('InfoRatio'):.2f}", None), ("Turnover / yr", f"{g('Turnover/yr'):.2f}", None, "one-way"),
            ("Cost drag / yr", f"{g('Cost drag %/yr'):.2%}", None, "STT+stamp+GST+DP+slip"),
            ("Hit rate (wk)", f"{g('HitRate'):.0%}", None), ("Avg holdings", f"{g('Avg holdings'):.1f}", None),
        ])
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()
        e = ew["nav"].reindex(nav.index).ffill()
        perf = QWidget()
        pl = QVBoxLayout(perf)
        pl.addWidget(line_chart({spec.name[:40]: nav, "Equal weight (same universe)": e / e.iloc[0], "NIFTY 50": b},
                                "Growth of ₹1 after all costs (log)", logy=True), 3)
        hl = QHBoxLayout()
        hl.addWidget(line_chart({"Strategy drawdown %": drawdown(nav) * 100, "NIFTY drawdown %": drawdown(b) * 100},
                                "Drawdown (%)"))
        hl.addWidget(line_chart({"Rolling 1y Sharpe": bt.rolling_sharpe(nav), "NIFTY": bt.rolling_sharpe(b)},
                                "Rolling 1-year Sharpe"))
        pl.addLayout(hl, 2)
        self.tabs.addTab(perf, "Performance")
        mt = bt.monthly_table(nav)
        mtab = FrameTable(fmt={c: "{:+.1%}" for c in mt.columns}, signed=set(mt.columns))
        mtab.set_frame(mt)
        self.tabs.addTab(mtab, "Monthly returns")
        reb = res["rebalances"]
        if reb is not None and len(reb):
            rv = reb.copy()
            tr = FrameTable(fmt={"turnover": "{:.1%}", "cost": "₹{:,.0f}", "value": "₹{:,.0f}", "invested": "{:.0%}"})
            rv.index = [d.strftime("%Y-%m-%d") for d in rv.index]
            tr.set_frame(rv)
            self.tabs.addTab(tr, "Trading")
        w = res["last_weights"]
        if w is not None and not w.empty:
            p = data.panel()
            w = w.sort_values(ascending=False)
            ht = FrameTable(fmt={"Weight": "{:+.2%}"}, signed={"Weight"})
            ht.set_frame(pd.DataFrame({"Company": p.names.reindex(w.index).str[:32].values,
                                       "Sector": p.sectors.reindex(w.index).values, "Weight": w.values}, index=w.index))
            ht.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
            self.tabs.addTab(ht, f"Holdings ({res['last_date']:%d %b %Y})")
        r = nav.pct_change().dropna()
        reg = data.regimes_daily()
        if reg is not None:
            d = pd.DataFrame({"r": r, "b": b.pct_change(), "regime": reg.reindex(r.index)}).dropna()
            gg = d.groupby("regime")
            rt = pd.DataFrame({"Days": gg.size(), "Strategy ann. return": gg["r"].mean() * 252,
                               "NIFTY ann. return": gg["b"].mean() * 252,
                               "Strategy ann. vol": gg["r"].std() * np.sqrt(252)}).reindex(REGIMES)
            var95 = np.percentile(r, 5)
            rt.loc["All days"] = [len(r), r.mean() * 252, b.pct_change().mean() * 252, r.std() * np.sqrt(252)]
            rtab = FrameTable(fmt={"Days": "{:.0f}", "Strategy ann. return": "{:+.1%}", "NIFTY ann. return": "{:+.1%}",
                                   "Strategy ann. vol": "{:.1%}"}, signed={"Strategy ann. return", "NIFTY ann. return"})
            rtab.set_frame(rt)
            rw = QWidget()
            rl = QVBoxLayout(rw)
            rl.addWidget(muted(f"Daily VaR95 {var95:.2%} · CVaR95 {r[r <= var95].mean():.2%}. "
                               "Performance by market regime (walk-forward labels):"))
            rl.addWidget(rtab)
            self.tabs.addTab(rw, "Risk")
