"""MODELS: every model in the app, what job it does, why that architecture, how it is trained, where its output shows,
and its measured out-of-sample record (read live from report/results/), with live / shadow / research / rejected
status. Click a row for the full card."""
from __future__ import annotations

import html

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QTextBrowser

from ... import model_registry as MR
from .. import theme
from ..widgets import ExplainButton, FrameTable, KpiRow, h2, muted
from . import Panel, vbox

STATUS_COL = {"live": theme.GREEN, "shadow": theme.AMBER, "research": theme.BLUE, "rejected": theme.RED}

FLOW = [("Daily prices, volumes, VIX (Yahoo), NSE bhavcopy",
         ["Rankers: FFNN · LSTM · Transformer", "Next-day: LightGBM · sequence net · logistic", "Range model",
          "Volatility LSTM (vs GARCH)", "Regimes: autoencoder + k-means"],
         "Rankers, Today, Screener, Stock pages, Risk, Backtest lab"),
        ("Base-model predictions", ["Stacked meta-model (4 models)", "Swing meta-labeler (rejected)"],
         "Today → Pattern changes, Track record"),
        ("Headlines (Google News)", ["FinBERT sentiment"], "News tabs, Screener, monitor list"),
        ("5-minute bars of ~930 liquid stocks", ["Rule setups → tabular bandit + LightGBM scorer (v1, live)",
                                                 "TCN meta-labeler → neural bandit · conformal gate · IQL exits (v2, shadow)"],
         "Intraday agent"),
        ("Weekly features + investor profile", ["RL allocator (PPO, Transformer encoder)"], "RL & plan")]


class ModelsPanel(Panel):
    title = "Models"
    code = "MODELS"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        lay.addWidget(muted("Every model the app uses, why it is built the way it is, and how it measured out of sample. "
                            "Status: live = drives what you see; shadow = runs and is measured, not yet trusted; research = "
                            "kept for comparison; rejected = measured and found not to help. Click a model for its card."))
        self.kpi = KpiRow(cols=5)
        lay.addWidget(self.kpi)
        split = QSplitter(Qt.Vertical)
        self.table = FrameTable()
        self.table.model_.term_overrides = {c: "model_card" for c in ("Family", "Status", "Job", "Measured record", "Used in")}
        self.table.clicked.connect(self._pick)
        split.addWidget(self.table)
        self.card = QTextBrowser()
        split.addWidget(self.card)
        split.setSizes([420, 380])
        lay.addWidget(split, 1)
        row = h2("How they connect (data → models → screens)", "model_card")
        lay.addWidget(row)
        self.flow = QTextBrowser()
        self.flow.setMaximumHeight(210)
        lay.addWidget(self.flow)
        lay.addWidget(ExplainButton("model_card"))

    def refresh(self) -> None:
        t = MR.table()
        self.table.set_frame(t)
        counts = t["Status"].value_counts()
        fam = t["Family"].value_counts()
        self.kpi.set_items([
            ("Models", str(len(t)), None, f"{int(fam.get('Deep learning', 0))} deep learning"),
            ("Live", str(int(counts.get("live", 0))), theme.GREEN, "drive what you see"),
            ("Shadow", str(int(counts.get("shadow", 0))), theme.AMBER, "measured, not yet trusted"),
            ("Research", str(int(counts.get("research", 0))), theme.BLUE, "kept as benchmarks"),
            ("Rejected", str(int(counts.get("rejected", 0))), theme.RED, "measured: did not help"),
        ])
        self._show(MR.CARDS[0])
        rows = "".join(
            f"<tr><td style='padding:4px 10px;color:{theme.MUTED}'>{html.escape(src)}</td>"
            f"<td style='padding:4px 10px'>→ {'<br>→ '.join(html.escape(m) for m in ms)}</td>"
            f"<td style='padding:4px 10px;color:{theme.AMBER}'>→ {html.escape(scr)}</td></tr>" for src, ms, scr in FLOW)
        self.flow.setHtml(f"<table>{rows}</table>")

    def _pick(self, idx) -> None:
        src = self.table.proxy.mapToSource(idx)
        name = self.table.model_.labels[src.row()]
        card = next((c for c in MR.CARDS if c.name == name), None)
        if card:
            self._show(card)

    def _show(self, c: MR.ModelCard) -> None:
        col = STATUS_COL.get(c.status, theme.MUTED)
        rows = [("Job", c.job), ("Architecture", c.architecture), ("Why this architecture", c.why), ("Inputs", c.inputs),
                ("Training", c.training), ("Where it shows", c.used_in), ("Measured (out of sample)", MR.record(c)),
                ("Limits", c.limits)]
        body = "".join(f"<tr><td style='color:{theme.MUTED};padding:3px 10px;vertical-align:top;white-space:nowrap'>{k}</td>"
                       f"<td style='padding:3px 10px'>{html.escape(v)}</td></tr>" for k, v in rows)
        self.card.setHtml(f"<h3 style='color:{theme.AMBER}'>{html.escape(c.name)} "
                          f"<span style='color:{col};font-size:12px'>[{c.status}]</span> "
                          f"<span style='color:{theme.MUTED};font-size:12px'>{c.family}</span></h3><table>{body}</table>")


def table_frame() -> pd.DataFrame:                     # for tests / docs
    return MR.table()
