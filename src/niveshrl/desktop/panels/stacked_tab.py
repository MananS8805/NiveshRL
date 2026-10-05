"""Today → Pattern changes: the 4-model stacked next-day predictor, run at market close.

Shows tonight's notable pattern changes (direction flipped Up↔Down, or size jumped quiet↔volatile), every stock's
stacked P(up) and pattern with each base model's vote, the meta model's weights, the measured out-of-sample record and
a rolling 60-day skill chart per model (the drift monitor)."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QHBoxLayout, QWidget

from ...config import ROOT
from .. import data, theme
from ..widgets import ExplainButton, FrameTable, KpiRow, h2, line_chart, muted
from . import scrolling, vbox

VOTE_FMT = {"P(up) now": "{:.1%}", "P(up) before": "{:.1%}", "Δ P(up)": "{:+.1%}", "LightGBM": "{:.0%}",
            "Sequence net": "{:.0%}", "Logistic": "{:.0%}", "Range model": "{:.0%}"}


def _votes(df: pd.DataFrame) -> pd.DataFrame:
    """Each base model's cross-sectional rank today (0% = most bearish/quietest, 100% = most bullish/most volatile)."""
    names = {"lgbm": "LightGBM", "seq": "Sequence net", "logreg": "Logistic", "range": "Range model"}
    return df[[c for c in names if c in df]].rank(pct=True).rename(columns=names)


def _record() -> str:
    """The measured out-of-sample record (scripts/compare_refits.py), shown next to the model, good or bad."""
    f = ROOT / "report" / "results" / "refit_comparison.csv"
    try:
        t = pd.read_csv(f, index_col=0)
        s, l = t.loc["stacked (4 models)"], t.loc["yearly lgbm (old pipeline)"]
        return (f" Measured out of sample ({int(s['Days'])} days, 2025-26): stacked daily IC {s['IC mean']:.3f} vs "
                f"{l['IC mean']:.3f} for LightGBM alone, so the stack adds no measured edge; its top-10 picks "
                f"{s['Top10 net of costs / day']:+.2%} a day after costs.")
    except (OSError, KeyError, ValueError):
        return ""


class StackedTab(QWidget):
    def __init__(self, open_stock, parent=None):
        super().__init__(parent)
        outer = vbox(self, 0)
        inner = QWidget()
        outer.addWidget(scrolling(inner))
        lay = vbox(inner)
        row = QHBoxLayout()
        row.addWidget(h2("Stacked next-day model: pattern changes since the last close", "stacked_model"))
        row.addStretch(1)
        row.addWidget(ExplainButton("stacked_model"))
        lay.addLayout(row)
        self.kpi = KpiRow(cols=6)
        lay.addWidget(self.kpi)
        self.note = muted("")
        lay.addWidget(self.note)
        lay.addWidget(h2("Notable changes tonight", "pattern_change"))
        self.changes = FrameTable(fmt=VOTE_FMT, signed={"Δ P(up)"})
        self.changes.model_.term_overrides = {c: "pattern_change" for c in ("Pattern now", "Pattern before", "P(up) now",
                                                                             "P(up) before", "Δ P(up)", "Change")}
        self.changes.model_.term_overrides.update({c: "stacked_model" for c in ("LightGBM", "Sequence net", "Logistic",
                                                                                  "Range model")})
        self.changes.row_clicked.connect(lambda t: open_stock(str(t)))
        self.changes.setMinimumHeight(260)
        lay.addWidget(self.changes)
        lay.addWidget(h2("All stocks: stacked P(up), pattern and each model's vote", "stacked_model"))
        self.all = FrameTable(fmt=VOTE_FMT)
        self.all.model_.term_overrides = dict(self.changes.model_.term_overrides)
        self.all.row_clicked.connect(lambda t: open_stock(str(t)))
        self.all.setMinimumHeight(360)
        lay.addWidget(self.all)
        lay.addWidget(h2("Model health: rolling 60-day skill (rank IC with next-day returns)", "model_health"))
        self.health = None
        self.health_holder = QWidget()
        self.health_lay = vbox(self.health_holder, 0)
        self.health_holder.setMinimumHeight(300)
        lay.addWidget(self.health_holder)
        self.record = muted("")
        lay.addWidget(self.record)

    def refresh(self) -> None:
        st, ch = data.dload("stacked"), data.dload("pattern_changes")
        meta = data.dload("stacked_meta") or {}
        if st is None or st.empty:
            self.kpi.set_items([])
            self.note.setText("No stacked output yet: it is produced by the daily pipeline at market close (F5 runs it "
                              "now). It needs the rolling base-model history from scripts/compare_refits.py.")
            for t in (self.changes, self.all):
                t.set_frame(pd.DataFrame())
            return
        votes = _votes(st)
        n_ch = len(ch) if ch is not None else 0
        n_not = int(ch["notable"].sum()) if ch is not None and len(ch) else 0
        pats = st["pattern"].value_counts()
        w = meta.get("weights", {})
        top = max(w, key=lambda k: abs(w[k])) if w else "–"
        self.kpi.set_items([
            ("Stocks scored", str(len(st)), None, "NIFTY 200"),
            ("Pattern changes", str(n_ch), None, "since the previous close"),
            ("Notable changes", str(n_not), theme.AMBER if n_not else None, "Up↔Down or quiet↔volatile"),
            ("Up patterns", str(int(sum(v for k, v in pats.items() if k.startswith("Up")))), theme.GREEN, "P(up) ≥ 53%"),
            ("Down patterns", str(int(sum(v for k, v in pats.items() if k.startswith("Down")))), theme.RED, "P(up) ≤ 47%"),
            ("Heaviest meta weight", top.replace("r_", "").replace("dir_x_range", "direction × range"), None,
             f"{w.get(top, 0):+.2f}" if w else ""),
        ])
        self.note.setText("Four base models (LightGBM, a sequence network over the last 20 days, logistic regression, "
                          "and the range model) are refit every month (the sequence net every quarter). A logistic meta "
                          "model, trained on their previous 12 months of out-of-sample predictions, combines them into "
                          "P(beats tomorrow's median). Pattern = direction (Up ≥ 53%, Down ≤ 47%) × expected range "
                          "(volatile = top 30%, quiet = bottom 30%). A forecast, not advice." + _record())
        if ch is not None and len(ch):
            c = ch.copy()
            c = c.sort_values(["notable", "Δ P(up)"], key=lambda s: s.abs() if s.name == "Δ P(up)" else s,
                              ascending=False)
            view = pd.DataFrame({"Change": ["notable" if x else "" for x in c["notable"]],
                                 "Pattern before": c["pattern_prev"], "Pattern now": c["pattern"],
                                 "P(up) before": c["stacked_prev"], "P(up) now": c["stacked"], "Δ P(up)": c["Δ P(up)"]},
                                index=c.index).join(votes)
            self.changes.set_frame(view)
        else:
            self.changes.set_frame(pd.DataFrame({"": ["No pattern changes since the previous close."]}))
        allv = pd.DataFrame({"Pattern now": st["pattern"], "P(up) now": st["stacked"]}).join(votes)
        self.all.set_frame(allv.sort_values("P(up) now", ascending=False))
        self._health()

    def _health(self) -> None:
        from ...research import stacked as ST
        if not ST.STACK_PATH.exists():
            return
        h = data._cached("stack_health:" + data.daily_stamp(), None,
                         lambda: ST.model_health(pd.read_parquet(ST.STACK_PATH)).dropna(how="all"))
        if self.health is not None:
            self.health_lay.removeWidget(self.health)
            self.health.deleteLater()
        names = {"lgbm": "LightGBM", "seq": "Sequence net", "logreg": "Logistic", "stacked": "Stacked"}
        self.health = line_chart({names[c]: h[c] for c in h.columns},
                                 "Rolling 60-day rank IC with next-day returns (higher = better; 0 = no skill)")
        self.health.explain_key = "model_health"
        self.health_lay.addWidget(self.health)
        last = h.iloc[-1] if len(h) else pd.Series(dtype=float)
        rec = ROOT / "report" / "results" / "refit_comparison.md"
        drift = [names[c] for c in h.columns if len(h[c].dropna()) > 120 and
                 h[c].iloc[-1] < h[c].dropna().quantile(0.1)]
        self.record.setText(
            "Latest 60-day IC: " + ", ".join(f"{names[c]} {last[c]:+.3f}" for c in h.columns if c in last and last[c] == last[c])
            + (f". ⚠ Below its own 10th percentile (possible drift): {', '.join(drift)}." if drift else ".")
            + (f" Measured record: {rec.name} in report/results." if rec.exists() else ""))
