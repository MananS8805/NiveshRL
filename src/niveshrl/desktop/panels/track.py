"""TRACK → Performance: did the app's suggestions work out? An overview of the swing picks and the intraday agent side
by side (live vs shadow engines, every measured experiment), then the forward tracker, the historical replay of the
monitor list and the sentiment test."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTabWidget, QVBoxLayout, QWidget

from ...config import ROOT
from ...research import forward as F
from .. import data, theme
from ..widgets import FrameTable, KpiRow, h2, line_chart, muted, run_async
from . import Panel, vbox

HIST = ROOT / "report" / "results" / "monitor_history_pit.parquet"
FMT = {"win rate": "{:.0%}", "avg R": "{:+.3f}", "median R": "{:+.2f}", "profit factor": "{:.2f}",
       "open avg R (marked)": "{:+.2f}", "picks": "{:,.0f}", "closed": "{:,.0f}", "open": "{:,.0f}"}
TERMS = {k: "forward_tracker" for k in ("picks", "closed", "open", "win rate", "avg R", "median R", "profit factor",
                                         "open avg R (marked)", "group", "status", "signal_date", "entry_date",
                                         "exit_date", "reason", "days", "rules")} | {"r": "r_multiple"}


def _json(name: str) -> dict:
    import json
    try:
        return json.loads((ROOT / "report" / "results" / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class TrackPanel(Panel):
    title = "Performance"
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
        ow = QWidget()
        ol = QVBoxLayout(ow)
        ol.addWidget(h2("Swing picks (days to weeks)", "forward_tracker"))
        self.o_swing = KpiRow(cols=5)
        ol.addWidget(self.o_swing)
        ol.addWidget(h2("Intraday agent (minutes to hours)", "agent_engines"))
        self.o_intra = KpiRow(cols=5)
        ol.addWidget(self.o_intra)
        ol.addWidget(h2("Every improvement tried, measured out of sample", "agent_engines"))
        self.o_exp = FrameTable()
        self.o_exp.model_.term_overrides = {c: "agent_engines" for c in ("Area", "Result", "Decision")}
        ol.addWidget(self.o_exp, 1)
        ol.addWidget(muted("Swing and intraday are different strategies on different time scales: the swing list is a "
                           "next-day/technical screen held for days with wide stops; the intraday agent opens and closes "
                           "within the day, where costs are a much larger share of each trade."))
        self.tabs.addTab(ow, "Overview")
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

    def _overview(self) -> None:
        import json
        from ... import intraday as ID
        hist = pd.read_parquet(HIST) if HIST.exists() else pd.DataFrame()
        if len(hist):
            e = F.edge(hist)
            mon = hist[(hist["group"] == "monitor list") & (hist["status"] == "closed")]
            rnd = hist[(hist["group"] == "random control") & (hist["status"] == "closed")]
            self.o_swing.set_items([
                ("Swing picks, 2015-26", f"{mon['r'].mean():+.3f}R", theme.signed(mon["r"].mean()), f"{len(mon):,} trades after costs"),
                ("Random picks", f"{rnd['r'].mean():+.3f}R", None, f"{len(rnd):,} trades, same rules"),
                ("Edge", f"{e['edge R']:+.3f}R", theme.signed(e["edge R"]), f"t {e['t-stat']:.1f}"),
                ("Win rate", f"{(mon['r'] > 0).mean():.0%}", None, "the median trade still loses"),
                ("Meta-label filter", "rejected" if not _json("swing_meta_decision.json").get("switch") else "in use",
                 None, "measured: no improvement"),
            ])
        st = {}
        try:
            st = json.loads((ID.DIR / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        start = float(st.get("start_pool", ID.config()["pool"]))
        pool = float(st.get("pool", start))
        dec = _json("intraday_v2_decision.json")
        bst = dec.get("best_stats", {})
        self.o_intra.set_items([
            ("Live agent (v1)", f"₹{pool:,.0f}", theme.signed(pool - start), f"{pool / start - 1:+.1%} · {len(st.get('days', []))} days"),
            ("v1 replay (52 days)", "−₹10,847", theme.RED, "-0.12R a trade, 154 trades"),
            ("Best v2 policy", dec.get("best", "not measured")[:28], None,
             f"{bst.get('avg R', float('nan')):+.3f}R · {bst.get('trades', 0)} trades" if bst else ""),
            ("v2 status", "live" if dec.get("switch") else "shadow", theme.GREEN if dec.get("switch") else theme.AMBER,
             "switch rule: avg R > 0, t > 1.5"),
            ("DL meta-labeler IC", self._dl_ic(), None, "TCN, in-play stocks, walk-forward"),
        ])
        rows = []
        for area, f, key in (("Next-day: monthly vs yearly refit", "refit_decision.json", "monthly_beats_yearly"),
                             ("Next-day: 4-model stack vs LightGBM", "refit_decision.json", "stack_beats_monthly"),
                             ("Volatility: monthly refit", "freshness_decision.json", "vol_monthly_better"),
                             ("Regimes: rolling 8-year window", "freshness_decision.json", "regime_rolling_better"),
                             ("Swing picks: meta-labeling", "swing_meta_decision.json", "switch"),
                             ("Intraday: v2 deep-learning engine", "intraday_v2_decision.json", "switch")):
            d = _json(f)
            if not d:
                rows.append({"Area": area, "Result": "not measured yet", "Decision": "–"})
                continue
            ok = bool(d.get(key))
            rows.append({"Area": area, "Result": "better" if ok else "not better", "Decision": "adopted" if ok else "kept the old one"})
        rows.append({"Area": "Intraday: forgetting (20-day half-life)", "Result": "slightly better (weak)", "Decision": "adopted"})
        f = ROOT / "report" / "results" / "industry_rules.csv"
        if f.exists():
            ir = pd.read_csv(f, index_col=0)
            intr = ir[ir["kind"] == "intraday"]
            pos = [i.split(" (")[0] for i, r in intr.iterrows() if r["avg R / trade"] > 0 and r["t-stat"] > 2]
            rows.append({"Area": "Published intraday rules (ORB, VWAP trend, 30-min momentum)",
                         "Result": "none profitable after NSE costs" if not pos else "profitable: " + ", ".join(pos),
                         "Decision": "not adopted" if not pos else "review"})
            if "Turtle / Donchian 55-20 breakout" in ir.index:
                t = ir.loc["Turtle / Donchian 55-20 breakout"]
                rows.append({"Area": "Published daily rules (Turtle, RSI(2), reversal, 12-m trend)",
                             "Result": f"best: Turtle {t['CAGR']:.1%}/yr vs 12-1 momentum 19.3%",
                             "Decision": "momentum kept as the reference"})
        self.o_exp.set_frame(pd.DataFrame(rows).set_index("Area"))

    @staticmethod
    def _dl_ic() -> str:
        f = ROOT / "report" / "results" / "intraday_dl.csv"
        try:
            t = pd.read_csv(f, index_col=0)
            return f"{t.loc['TCN, in-play stocks', 'IC mean']:+.3f}"
        except (OSError, KeyError, ValueError):
            return "–"

    def refresh(self) -> None:
        self._overview()
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
            from ...research.tradecheck import build_tagged
            build_tagged(p, df)                     # 'similar setups' on the stock page read this
            return True
        run_async(work, lambda _: (self.recompute.setEnabled(True), self._show_history()),
                  on_error=lambda m: self.recompute.setEnabled(True))
