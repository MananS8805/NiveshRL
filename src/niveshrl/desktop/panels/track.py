"""TRACK: did the app's suggestions work out? Forward tracker, historical replay of the monitor list, sentiment test."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTabWidget, QVBoxLayout, QWidget

from ...config import ROOT
from ...research import forward as F
from .. import data, theme
from ..widgets import FrameTable, KpiRow, line_chart, muted, run_async
from . import Panel, vbox

HIST = ROOT / "report" / "results" / "monitor_history_pit.parquet"
FMT = {"win rate": "{:.0%}", "avg R": "{:+.3f}", "median R": "{:+.2f}", "profit factor": "{:.2f}",
       "open avg R (marked)": "{:+.2f}", "picks": "{:,.0f}", "closed": "{:,.0f}", "open": "{:,.0f}"}
TERMS = {k: "forward_tracker" for k in ("picks", "closed", "open", "win rate", "avg R", "median R", "profit factor",
                                         "open avg R (marked)", "group", "status", "signal_date", "entry_date",
                                         "exit_date", "reason", "days", "rules")} | {"r": "r_multiple"}


class TrackPanel(Panel):
    title = "Track record"
    code = "TRACK"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        lay.addWidget(muted("Every 'watch for strength' pick is followed with the same swing rules as the trade plans (buy at "
                            "the next open; stop and T1 from the plan; a third booked at T1 with the stop moved to entry; the "
                            "rest trailed 3×ATR; 60-day cap; after delivery costs), next to an equal-size random sample of "
                            "that day's other stocks. Nothing is judged before 100 closed trades per group."))
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        fw = QWidget()
        fl = QVBoxLayout(fw)
        self.f_kpis = KpiRow(cols=4)
        fl.addWidget(self.f_kpis)
        self.f_sum = FrameTable(fmt=FMT, signed={"avg R"})
        self.f_sum.model_.term_overrides = TERMS
        self.f_sum.setMaximumHeight(110)
        fl.addWidget(self.f_sum)
        self.f_picks = FrameTable(fmt={"entry": "₹{:,.2f}", "exit": "₹{:,.2f}", "r": "{:+.2f}"}, signed={"r"})
        self.f_picks.model_.term_overrides = TERMS
        self.f_picks.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        fl.addWidget(self.f_picks, 1)
        self.tabs.addTab(fw, "Forward (live picks)")
        hw = QWidget()
        hl = QVBoxLayout(hw)
        row = QHBoxLayout()
        self.h_kpis = KpiRow(cols=4)
        row.addWidget(self.h_kpis, 1)
        self.recompute = QPushButton("⟳ Recompute (~2 min)")
        self.recompute.clicked.connect(self._recompute)
        row.addWidget(self.recompute)
        hl.addLayout(row)
        self.h_sum = FrameTable(fmt=FMT, signed={"avg R"})
        self.h_sum.model_.term_overrides = TERMS
        self.h_sum.setMaximumHeight(110)
        hl.addWidget(self.h_sum)
        self.h_box = QHBoxLayout()
        hl.addLayout(self.h_box, 2)
        self.h_year = FrameTable(fmt={"monitor list": "{:+.3f}", "random control": "{:+.3f}", "difference": "{:+.3f}"},
                                 signed={"monitor list", "random control", "difference"})
        self.h_year.model_.term_overrides = TERMS | {"monitor list": "forward_tracker", "random control": "random_control",
                                                      "difference": "forward_tracker"}
        hl.addWidget(self.h_year, 1)
        hl.addWidget(muted("Rebuilt for every 5th trading day 2015 → today on point-in-time NIFTY 200 members, from the "
                           "walk-forward next-day model and technicals; news is left out (no news history)."))
        self.tabs.addTab(hw, "Historical replay")
        sw = QWidget()
        sl = QVBoxLayout(sw)
        self.s_tab = FrameTable(fmt={"avg excess": "{:+.2%}", "hit (beat median)": "{:.0%}", "n": "{:,.0f}", "days": "{:,.0f}"},
                                signed={"avg excess"})
        self.s_tab.model_.term_overrides = {c: "sentiment_test" for c in ("avg excess", "hit (beat median)", "n", "days")}
        sl.addWidget(self.s_tab, 1)
        sl.addWidget(muted("FinBERT sentiment (adjusted) on each saved day, grouped negative / neutral / positive, against "
                           "the next 1 and 5 trading days' return minus the median stock. A few days prove nothing: this "
                           "table becomes meaningful after months of daily runs."))
        self.tabs.addTab(sw, "Sentiment test")

    def refresh(self) -> None:
        run_async(lambda: F.track_saved_picks(data.panel()), self._show_forward,
                  on_error=lambda m: self.f_kpis.set_items([("Forward tracker", "error", theme.RED, m.splitlines()[-1][:80])]))
        self._show_history()
        run_async(lambda: F.sentiment_test(data.panel()), self._show_sent)

    def _show_forward(self, out) -> None:
        df, s = out
        closed = int((df["status"] == "closed").sum()) if len(df) else 0
        days = df["signal_date"].nunique() if len(df) else 0
        self.f_kpis.set_items([("Saved days", str(days), None, "daily pipeline runs with a monitor list"),
                               ("Picks followed", str(len(df)), None, "monitor list + random control"),
                               ("Closed", str(closed), None, "no verdict before 100 per group"),
                               ("Rules", F.rules_hash(), None, "hash of the plan + monitor rules")])
        self.f_sum.set_frame(s if len(s) else pd.DataFrame({"": ["No saved monitor lists yet."]}))
        if len(df):
            v = df[["ticker", "group", "signal_date", "status", "entry_date", "entry", "exit_date", "exit", "r", "days",
                    "reason"]].sort_values("signal_date", ascending=False).set_index("ticker")
            v.index.name = None
            self.f_picks.set_frame(v)

    def _show_history(self) -> None:
        if not HIST.exists():
            self.h_kpis.set_items([("Historical replay", "not computed", None, "press Recompute")])
            return
        df = pd.read_parquet(HIST)
        s = pd.DataFrame({g: F._summary(df[df["group"] == g]) for g in df["group"].unique()}).T
        self.h_sum.set_frame(s)
        e = F.edge(df)
        verdict = "statistically clear" if e["t-stat"] >= 2 else "not clearly different from random"
        self.h_kpis.set_items([("Edge vs random", f"{e['edge R']:+.3f}R", theme.signed(e["edge R"]), "per closed trade, after costs"),
                               ("t-statistic", f"{e['t-stat']:.2f}", theme.GREEN if e["t-stat"] >= 2 else theme.AMBER, verdict),
                               ("Trades", f"{e['n'][0]:,} vs {e['n'][1]:,}", None, "monitor list vs control"),
                               ("Period", f"{pd.to_datetime(df['signal_date']).min():%Y} → {pd.to_datetime(df['signal_date']).max():%Y}",
                                None, "every 5th trading day")])
        c = df[df["status"] == "closed"].assign(year=lambda x: pd.to_datetime(x["signal_date"]).dt.year)
        by = c.groupby(["year", "group"])["r"].mean().unstack()
        by["difference"] = by["monitor list"] - by["random control"]
        self.h_year.set_frame(by)
        while self.h_box.count():
            w = self.h_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        cum = {g: c[c["group"] == g].sort_values("signal_date").set_index(pd.to_datetime(
            c[c["group"] == g].sort_values("signal_date")["signal_date"]))["r"].cumsum() for g in ("monitor list", "random control")}
        first = cum["monitor list"]
        al = {k: v[~v.index.duplicated(keep="last")].reindex(first.index[~first.index.duplicated(keep="last")], method="ffill")
              for k, v in cum.items()}
        self.h_box.addWidget(line_chart(al, "Cumulative R after costs: monitor list vs random control"))

    def _show_sent(self, t) -> None:
        self.s_tab.set_frame(t if t is not None and len(t) else pd.DataFrame({"": ["No saved sentiment days yet."]}))

    def _recompute(self) -> None:
        self.recompute.setEnabled(False)

        def work():
            from ...research.data import load_panel
            p = load_panel(universe="pit")
            pred = pd.read_parquet(ROOT / "data" / "predictions" / "nextday_pit.parquet")
            df, _ = F.historical_monitor(p, pred)
            df.to_parquet(HIST)
            return True
        run_async(work, lambda _: (self.recompute.setEnabled(True), self._show_history()),
                  on_error=lambda m: self.recompute.setEnabled(True))
