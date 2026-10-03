"""TODAY: daily briefing, top stocks to monitor tomorrow, and measured market habits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QSpinBox, QTabWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from ... import watchlist as wl
from .. import data, theme
from ..widgets import FrameTable, KpiRow, h2, muted
from . import Panel, vbox


class TodayPanel(Panel):
    title = "Today"
    code = "TODAY"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        self.kpis = KpiRow(cols=8)
        lay.addWidget(self.kpis)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        # --- tab 1: briefing
        brief = QWidget()
        grid = QGridLayout(brief)
        left = QVBoxLayout()
        left.addWidget(h2("What happened"))
        self.story = QTextBrowser()
        self.story.setMaximumHeight(210)
        left.addWidget(self.story)
        left.addWidget(h2("Your watchlist today"))
        self.mine = FrameTable(fmt={"1D": "{:+.2%}", "P(up)": "{:.0%}", "Sentiment": "{:+.2f}", "Analyst": "{:.0f}"},
                               signed={"1D", "Sentiment"})
        self.mine.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        left.addWidget(self.mine, 1)
        grid.addLayout(left, 0, 0)
        right = QVBoxLayout()
        right.addWidget(h2("Best and worst sectors (1D %)"))
        self.sectors = pg.PlotWidget()
        self.sectors.setMouseEnabled(x=False, y=False)
        self.sectors.setMenuEnabled(False)
        right.addWidget(self.sectors, 2)
        right.addWidget(h2("Results this week"))
        self.earn = FrameTable()
        self.earn.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        right.addWidget(self.earn, 1)
        grid.addLayout(right, 0, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        self.stamp = muted("")
        grid.addWidget(self.stamp, 1, 0, 1, 2)
        self.tabs.addTab(brief, "Briefing")

        # --- tab 2: top stocks to monitor tomorrow
        watch = QWidget()
        wl_ = vbox(watch)
        hrow = QHBoxLayout()
        hrow.addWidget(h2("Top stocks to monitor tomorrow"))
        hrow.addStretch(1)
        hrow.addWidget(muted("how many"))
        self.n = QSpinBox()
        self.n.setRange(5, 10)
        self.n.setValue(10)
        self.n.valueChanged.connect(self.refresh)
        hrow.addWidget(self.n)
        wl_.addLayout(hrow)
        self.warn = QTextBrowser()
        self.warn.setMaximumHeight(62)
        wl_.addWidget(self.warn)
        mon = QHBoxLayout()
        self.lists = {}
        for name in ("watch for strength", "watch for weakness"):
            box = QVBoxLayout()
            box.addWidget(h2(name.capitalize()))
            t = FrameTable(fmt={"Score": "{:+.2f}", "P(up)": "{:.0%}"}, signed={"Score"})
            t.row_clicked.connect(lambda tk: self.stock_selected.emit(str(tk)))
            box.addWidget(t, 1)
            self.lists[name] = t
            mon.addLayout(box)
        wl_.addLayout(mon, 1)
        wl_.addWidget(muted("Click a stock to open its chart, news and technicals."))
        self.tabs.addTab(watch, "Watch tomorrow")

        # --- tab 3: market habits
        habw = QWidget()
        hl = vbox(habw)
        hl.addWidget(h2("Market habits (NIFTY, measured from history, not opinions)"))
        hab = QHBoxLayout()
        self.dow = FrameTable(fmt={"mean": "{:+.3%}", "hit": "{:.0%}", "n": "{:.0f}"}, signed={"mean"})
        self.dow.model_.term_overrides = {"mean": "market_habits", "hit": "market_habits", "n": "market_habits"}
        self.habits = QTextBrowser()
        self.intraday = QTextBrowser()
        for w in (self.dow, self.habits, self.intraday):
            hab.addWidget(w)
        hl.addLayout(hab, 1)
        hl.addWidget(muted("Day of week: mean NIFTY return and share of up days (hit). Use as context, not as a signal."))
        self.tabs.addTab(habw, "Market habits")

    def refresh(self) -> None:
        b, mon, hab = data.dload("briefing"), data.dload("monitor"), data.dload("habits")
        if b is None:
            self.story.setHtml("<p>No daily data yet. Press <b>F5</b> (or the tray menu → Refresh) to run the daily "
                               "pipeline, or run <code>python scripts/daily.py</code>.</p>")
            return
        ms = b.get("market_sentiment", float("nan"))
        self.kpis.set_items([
            ("Trading day", b["date"], theme.AMBER),
            ("NIFTY", f"{b['nifty']:,.1f}", None),
            ("1D", f"{b['nifty_1d']:+.2%}", theme.signed(b["nifty_1d"])),
            ("YTD", f"{b['nifty_ytd']:+.2%}", theme.signed(b["nifty_ytd"])),
            ("VIX", f"{b['vix']:.2f}", theme.signed(-b["vix_1d"]), f"{b['vix_1d']:+.1%} today"),
            ("Adv / Dec", f"{b['advancers']} / {b['decliners']}", None),
            ("Regime", b.get("regime") or "n/a", theme.AMBER),
            ("News sentiment", f"{ms:+.2f}", theme.signed(ms), "market average, FinBERT"),
        ])
        self.story.setHtml("<ul>" + "".join(f"<li style='margin-bottom:4px'>{x}</li>" for x in b["narrative"]) + "</ul>")
        # watchlist rows
        mine = wl.load()
        t = data.screener_table()
        if mine and t is not None:
            cols = {"ret_1d": "1D", "prob_up": "P(up)", "sentiment_adj": "Sentiment", "analyst_score": "Analyst"}
            rows = t.reindex(list(mine))[[c for c in cols if c in t]].rename(columns=cols)
            rows.insert(0, "Tier", [wl.TIERS[mine[x].tier][0] for x in rows.index])
            self.mine.set_frame(rows)
        else:
            self.mine.set_frame(pd.DataFrame({"": ["Add stocks to your watchlist (WATCH) to see them here."]}))
        # sectors bar chart
        sec = pd.Series({**b["worst_sectors"], **b["best_sectors"]}).sort_values()
        pw = self.sectors
        pw.clear()
        y = np.arange(len(sec))
        pw.addItem(pg.BarGraphItem(x0=0, y=y, height=0.7, width=sec.to_numpy() * 100,
                                   brushes=[pg.mkBrush(theme.signed(v)) for v in sec.to_numpy()]))
        pw.getAxis("left").setTicks([[(i, s[:22]) for i, s in enumerate(sec.index)]])
        pw.showGrid(x=True, alpha=0.15)
        ew = b.get("earnings_week") or {}
        self.earn.set_frame(pd.DataFrame({"Results date": pd.Series(ew)}) if ew
                            else pd.DataFrame({"": ["No results due this week among the NIFTY 200."]}))
        # monitor lists
        ytd = data.dload("nextday_ytd") or {}
        e = ytd.get("ensemble") or ytd.get("lgbm") or {}
        self.warn.setHtml(
            f"<span style='color:{theme.AMBER}'>For <b>watching</b>, not blind buying.</span> The next-day model's "
            f"out-of-sample record this year: AUC {e.get('AUC', float('nan')):.3f}, top-10 hit rate "
            f"{e.get('Top10 hit rate', float('nan')):.0%}, accuracy {e.get('Accuracy', float('nan')):.1%} "
            "(51-54% is a good result for next-day direction). Daily turnover at Indian costs (~0.25% round trip) "
            "usually eats the edge.")
        if mon is not None:
            for name, tab in self.lists.items():
                df = mon[mon["list"] == name].head(self.n.value())
                view = pd.DataFrame({"Score": df["score"], "P(up)": df["prob"], "Why": df["reasons"]})
                tab.set_frame(view)
        # habits
        if hab:
            dow = pd.DataFrame(hab["day_of_week"]).T
            self.dow.set_frame(dow)
            bu, bd = hab["after_big_up"], hab["after_big_down"]
            self.habits.setHtml(
                f"<b style='color:{theme.AMBER}'>After big days (since {hab['since'][:4]})</b><ul>"
                f"<li>After a +2% day ({bu['days']}×): next day up <b>{bu['next_day_up']:.0%}</b>, avg {bu['next_day_mean']:+.2%}</li>"
                f"<li>After a −2% day ({bd['days']}×): next day up <b>{bd['next_day_up']:.0%}</b>, avg {bd['next_day_mean']:+.2%}</li>"
                f"<li>After 3 up days: next day up <b>{hab['streak_up_3_next']:.0%}</b></li>"
                f"<li>After 3 down days: next day up <b>{hab['streak_dn_3_next']:.0%}</b></li></ul>")
            i = hab.get("intraday")
            if i:
                self.intraday.setHtml(
                    f"<b style='color:{theme.AMBER}'>Intraday (last {i['days']} sessions)</b><ul>"
                    f"<li>First hour sets the day's direction <b>{i['first_hour_sets_direction']:.0%}</b> of the time</li>"
                    f"<li>Average day range <b>{i['avg_range']:.2%}</b></li>"
                    f"<li>High formed: {', '.join(f'{k} {v:.0%}' for k, v in i['high_formed'].items())}</li>"
                    f"<li>Low formed: {', '.join(f'{k} {v:.0%}' for k, v in i['low_formed'].items())}</li></ul>")
        m = data.daily.latest_meta()
        if m:
            bad = [k for k, v in m["steps"].items() if not v.get("ok")]
            self.stamp.setText(f"Data as of trading day {m['trading_day']} · pipeline ran {m['ran_at'].replace('T', ' ')}"
                               + (f" · ⚠ failed: {', '.join(bad)}" if bad else ""))
