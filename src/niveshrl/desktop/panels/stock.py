"""Stock panel: live price, watchlist controls, overview (report card + why it is moving), candlestick chart, key
stats, peers, seasonality & results reactions, risk (relative strength, volatility cone, drawdowns), news + FinBERT,
technicals, financial statements, ownership/analysts, model outputs and company profile."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ... import watchlist as wl
from ...research import fundamentals as fx
from ...research import stockinfo as SI
from ...research import tradecheck as TC
from ...research.screener import COLUMNS as SCR_COLS
from ...research.technicals import TECH_COLUMNS
from .. import data, theme
from ..widgets import ExplainButton, FrameTable, KpiRow, PriceChart, h1, h2, line_chart, muted, run_async
from . import Panel, scrolling, vbox

SIGNALS = {
    "rsi14": lambda v: "overbought" if v > 70 else "oversold" if v < 30 else "neutral",
    "adx14": lambda v: "strong trend" if v >= 25 else "weak / no trend",
    "supertrend": lambda v: "uptrend" if v > 0 else "downtrend",
    "vs_sma200": lambda v: "above long-term average" if v > 0 else "below long-term average",
    "vol_ratio": lambda v: "unusual volume" if v >= 1.5 else "",
    "bb_pctb": lambda v: "at upper band" if v > 1 else "at lower band" if v < 0 else "",
    "macd_hist": lambda v: "bullish momentum" if v > 0 else "bearish momentum",
    "from_52w_high": lambda v: "near 52-week high" if v > -0.03 else "",
}


def _num(v, f="{:,.2f}", empty="–"):
    try:
        return empty if v is None or v != v else f.format(v)
    except (TypeError, ValueError):
        return empty


def money_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().astype(float)
    for r in out.index:
        if r != "Diluted EPS":
            out.loc[r] = out.loc[r] / 1e7
    out.index = [f"{r} (₹ Cr)" if r != "Diluted EPS" else "Diluted EPS (₹)" for r in out.index]
    out.columns = [str(c)[:10] for c in out.columns]
    return out


class StockPanel(Panel):
    title = "Stock"
    code = "DES"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        self.ticker: str | None = None
        self.fund: dict | None = None
        outer = vbox(self, 0)
        inner = QWidget()
        outer.addWidget(scrolling(inner))
        lay = vbox(inner)
        top = QHBoxLayout()
        left = QVBoxLayout()
        self.name = h1("Select a stock: type its symbol in the command bar or click one anywhere")
        self.sub = muted("")
        left.addWidget(self.name)
        left.addWidget(self.sub)
        top.addLayout(left, 3)
        self.price = QLabel("")
        self.price.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.price.setTextFormat(Qt.RichText)
        top.addWidget(self.price, 2)
        lay.addLayout(top)
        wrow = QHBoxLayout()
        self.wl_label = muted("")
        self.b_must = QPushButton("★ Must have")
        self.b_pref = QPushButton("☆ Preferred")
        self.b_rm = QPushButton("Remove from watchlist")
        self.b_must.clicked.connect(lambda: self._wl("must"))
        self.b_pref.clicked.connect(lambda: self._wl("preferred"))
        self.b_rm.clicked.connect(lambda: self._wl(None))
        self.b_alert = QPushButton("🔔 Set alert")
        self.b_alert.clicked.connect(lambda: self.ticker and self.ctx.goto.emit("ALRT", self.ticker))
        self.b_paper = QPushButton("Paper trade this plan")
        self.b_paper.clicked.connect(lambda: self.ticker and self.ctx.goto.emit("PAPER", self.ticker))
        for w in (self.wl_label, self.b_must, self.b_pref, self.b_rm, self.b_alert, self.b_paper):
            wrow.addWidget(w)
        wrow.addStretch(1)
        lay.addLayout(wrow)
        self.desk = KpiRow(cols=4)
        lay.addWidget(self.desk)
        self.stats = KpiRow(cols=4)
        self.tabs = QTabWidget()
        self._build_overview()
        self.chart = PriceChart(controls=True)
        self.tabs.addTab(self.chart, "Chart")
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.addWidget(self.stats)
        sl.addStretch(1)
        self.tabs.addTab(sw, "Key stats")
        self._build_plan_tab()
        self._build_peers()
        self._build_events()
        self._build_risk()
        self.news = FrameTable(fmt={"Score": "{:+.2f}"}, signed={"Score"})
        self.news.model_.term_overrides = {"Score": "finbert"}
        self.news.doubleClicked.connect(self._open_news)
        nw = QWidget()
        nl = QVBoxLayout(nw)
        self.news_head = KpiRow(cols=4)
        nl.addWidget(self.news_head)
        nl.addWidget(self.news, 1)
        nl.addWidget(muted("Scored by FinBERT (ProsusAI/finbert). Headlines from Google News, last 48 h. "
                           "Double-click a headline to open it in your browser."))
        self.tabs.addTab(nw, "News && sentiment")
        self.tech = FrameTable()
        self.tabs.addTab(self.tech, "Technicals")
        self.stmts = {}
        for key, label in [("annual", "Income statement"), ("balance", "Balance sheet"), ("cashflow", "Cash flow")]:
            t = FrameTable(fmt={})
            t.explainable = False                      # raw statement line items
            self.stmts[key] = t
            self.tabs.addTab(t, label)
        self.owner = QTextBrowser()
        self.tabs.addTab(self.owner, "Ownership && analysts")
        self.models = QTextBrowser()
        self.tabs.addTab(self.models, "NiveshRL models")
        from .nse_views import StockNSETab
        self.nse = StockNSETab()
        self.tabs.addTab(self.nse, "F&&O && shareholding")
        self.tabs.currentChanged.connect(lambda i: self.tabs.widget(i) is self.nse and self.nse.load())
        self.about = QTextBrowser()
        self.about.setOpenExternalLinks(True)
        self.tabs.addTab(self.about, "About")
        self.tabs.setMinimumHeight(600)
        lay.addWidget(self.tabs, 1)
        self._set_wl_buttons()

    # ------------------------------------------------------------------ deep-dive tabs
    def _build_overview(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(h2("Report card", "report_card"))
        row.addStretch(1)
        row.addWidget(ExplainButton("report_card"))
        lay.addLayout(row)
        self.card = KpiRow(cols=5)
        lay.addWidget(self.card)
        self.card_view = QTextBrowser()
        lay.addWidget(self.card_view, 2)
        lay.addWidget(h2("Why is it moving?", "why_moving"))
        self.why = QTextBrowser()
        self.why.setMaximumHeight(140)
        lay.addWidget(self.why)
        self.tabs.addTab(w, "Overview")

    def _build_peers(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.peer_note = muted("")
        self.peers = FrameTable()
        self.peers.row_clicked.connect(lambda t: self.stock_selected.emit(str(t)))
        lay.addWidget(self.peer_note)
        lay.addWidget(self.peers, 1)
        lay.addWidget(muted("Same NSE industry, largest first. Click a row to open that stock; right-click a value "
                            "to explain it."))
        self.tabs.addTab(w, "Peers")

    def _build_events(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(h2("Seasonality: average return by calendar month", "seasonality"))
        self.season = FrameTable(fmt={"avg return": "{:+.2%}", "median": "{:+.2%}", "up years": "{:.0%}",
                                      "avg vs NIFTY": "{:+.2%}", "years": "{:.0f}"},
                                 signed={"avg return", "median", "avg vs NIFTY"})
        self.season.model_.term_overrides = {c: "seasonality" for c in ("avg return", "median", "up years",
                                                                         "avg vs NIFTY", "years")}
        self.season.setMinimumHeight(300)
        lay.addWidget(self.season)
        lay.addWidget(h2("How it reacted to quarterly results", "results_reaction"))
        self.ev_kpi = KpiRow(cols=5)
        lay.addWidget(self.ev_kpi)
        self.react = FrameTable(fmt={"EPS est.": "{:,.2f}", "EPS actual": "{:,.2f}", "surprise %": "{:+.1f}",
                                     "day move": "{:+.2%}", "vs NIFTY": "{:+.2%}", "next 20d vs NIFTY": "{:+.2%}"},
                                signed={"surprise %", "day move", "vs NIFTY", "next 20d vs NIFTY"})
        self.react.model_.term_overrides = {"results": "results_reaction", "EPS est.": "eps_surprise",
                                            "EPS actual": "eps_surprise", "surprise %": "eps_surprise",
                                            "day move": "results_reaction", "vs NIFTY": "results_reaction",
                                            "next 20d vs NIFTY": "post_results_drift"}
        self.react.setMinimumHeight(260)
        lay.addWidget(self.react)
        lay.addWidget(h2("Dividends", "dividend_history"))
        self.div_kpi = KpiRow(cols=3)
        lay.addWidget(self.div_kpi)
        self.divs = FrameTable(fmt={"dividend ₹": "{:,.2f}"})
        self.divs.model_.term_overrides = {"Ex-date": "dividend_history", "dividend ₹": "dividend_history"}
        self.divs.setMinimumHeight(200)
        lay.addWidget(self.divs)
        self.ev_note = muted("")
        lay.addWidget(self.ev_note)
        self.tabs.addTab(scrolling(w), "Seasonality && events")

    def _build_risk(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.rs_kpi = KpiRow(cols=4)
        lay.addWidget(self.rs_kpi)
        self.rs_box = QVBoxLayout()
        lay.addLayout(self.rs_box)
        self.rs_chart = None
        lay.addWidget(h2("Volatility cone: where the price may be (not where it will go)", "vol_cone"))
        self.cone = FrameTable(fmt={"95% low": "₹{:,.1f}", "68% low": "₹{:,.1f}", "68% high": "₹{:,.1f}",
                                    "95% high": "₹{:,.1f}", "1σ move": "±{:.1%}"})
        self.cone.model_.term_overrides = {c: "vol_cone" for c in ("95% low", "68% low", "68% high", "95% high",
                                                                    "1σ move")}
        self.cone.setMinimumHeight(150)
        lay.addWidget(self.cone)
        self.cone_note = muted("")
        lay.addWidget(self.cone_note)
        lay.addWidget(h2("Deepest falls in the price history", "drawdown_history"))
        self.dds = FrameTable(fmt={"depth": "{:.1%}", "days to recover": "{:,.0f}"})
        self.dds.model_.term_overrides = {c: "drawdown_history" for c in ("peak", "trough", "depth", "recovered",
                                                                           "days to recover")}
        self.dds.setMinimumHeight(180)
        lay.addWidget(self.dds)
        self.tabs.addTab(scrolling(w), "Risk")

    def _deep_dive(self) -> None:
        t, p, table = self.ticker, data.panel(), data.screener_table()
        # overview: report card + why it is moving
        card = SI.report_card(table, t)
        self.card.ticker = t
        self.card.set_items([(f"{k} score", "–" if v["score"] != v["score"] else f"{v['score']:.0f}/100",
                              None if v["score"] != v["score"] else theme.GREEN if v["score"] >= 66 else
                              theme.RED if v["score"] < 34 else None, v["verdict"]) for k, v in card.items()])
        if card:
            grp = next(iter(card.values()))
            html = "".join(f"<p><b style='color:{theme.AMBER}'>{k}: {v['verdict']}</b><br>"
                           + ("<br>".join("· " + r for r in v["reasons"]) or "· not enough data") + "</p>"
                           for k, v in card.items())
            html += (f"<p style='color:{theme.MUTED}'>Each score is the stock's percentile among the {grp['peers']} "
                     f"{grp['group']} stocks in the NIFTY 200 on each measure (100 = best in the group), averaged. "
                     "Relative to its peers, not absolute; a description, not a forecast or advice.</p>")
        else:
            html = "<p>Run the daily refresh (F5) to build the screener table this needs.</p>"
        self.card_view.setHtml(html)
        tk = self._live()
        live = tk["change_percent"] / 100 if tk and tk.get("change_percent") is not None else None
        try:
            self.why.setHtml(f"<p>{SI.why_moving(t, table, p, data.dload('news'), live)}</p>"
                             f"<p style='color:{theme.MUTED}'>Built from today's move, its sector, volume, gap, the "
                             "results calendar, headlines and the pivot state. Coincidences, not proven causes.</p>")
        except (KeyError, IndexError, ValueError) as e:
            self.why.setHtml(f"<p>Not enough data to explain today's move ({e}).</p>")
        # peers (index = tickers, so a click opens the stock)
        pe = SI.peers(table, t)
        self.peer_note.setText(f"{len(pe)} largest stocks in {p.sectors.get(t, '')} (NIFTY 200); "
                               f"{t.replace('.NS', '')} is marked ◀" if len(pe) else
                               "Run the daily refresh to build the peer table.")
        if len(pe):
            pe = pe.rename(columns={c: SCR_COLS[c][0] for c in pe.columns if c in SCR_COLS})
            pe.insert(0, "", ["◀" if i == t else "" for i in pe.index])
            self.peers.model_.fmt = {lab: ("{:,.0f}" if "score" in lab.lower() or "cap" in lab.lower() else "{:,.2f}")
                                     if kind != "pct" else "{:+.1%}" if grp == "Returns" or "growth" in lab.lower()
                                     else "{:.1%}" for lab, grp, kind in SCR_COLS.values() if lab in pe.columns}
            self.peers.model_.signed = {lab for lab, grp, _ in SCR_COLS.values()
                                        if lab in pe.columns and (grp == "Returns" or "growth" in lab.lower())}
        self.peers.ticker_hint = t
        self.peers.set_frame(pe if len(pe) else pd.DataFrame())
        # seasonality
        se = SI.seasonality(p, t)
        self.season.set_frame(se.rename_axis("Month").reset_index().set_index("Month") if len(se) else pd.DataFrame())
        # risk: relative strength, cone, drawdowns
        rs = SI.relative_strength(p, t)
        self.rs_kpi.ticker = t
        self.rs_kpi.set_items([
            ("vs NIFTY (1 year)", f"{rs['vs_nifty']:+.1%}", theme.signed(rs["vs_nifty"]), "relative performance"),
            ("vs sector (1 year)", "–" if rs["vs_sector"] != rs["vs_sector"] else f"{rs['vs_sector']:+.1%}",
             theme.signed(rs["vs_sector"]), f"vs {rs['n_peers']} peers, equal weight"),
            ("Beta", "–" if rs["beta"] != rs["beta"] else f"{rs['beta']:.2f}", None, "1 year vs NIFTY, daily"),
            ("Correlation with NIFTY", "–" if rs["corr"] != rs["corr"] else f"{rs['corr']:.2f}", None, "1 year, daily"),
        ])
        if self.rs_chart is not None:
            self.rs_box.removeWidget(self.rs_chart)
            self.rs_chart.deleteLater()
        series = {"vs NIFTY": rs["rs_nifty"]}
        if rs["rs_sector"] is not None:
            series["vs sector"] = rs["rs_sector"]
        self.rs_chart = line_chart(series, "Relative strength: stock ÷ benchmark, rebased to 1 a year ago "
                                           "(rising = beating it)")
        self.rs_chart.explain_key = "rs_line"
        self.rs_chart.setMinimumHeight(280)
        self.rs_box.addWidget(self.rs_chart)
        price = float(p.close[t].dropna().iloc[-1])
        sig, src = SI.cone_sigma(t, data.vol_forecasts(), table, p.close.index[-1])
        if sig == sig:
            self.cone.set_frame(SI.vol_cone(price, sig).rename_axis("Horizon"))
            self.cone_note.setText(f"From ₹{price:,.2f} using σ = {sig:.1%} a year ({src}). Log-normal, zero drift: "
                                   "if volatility stays as estimated, about 68% of outcomes land inside the inner band "
                                   "and 95% inside the outer one. Results days and crashes break out of it more often.")
        else:
            self.cone.set_frame(pd.DataFrame())
            self.cone_note.setText("No volatility estimate for this stock yet.")
        dd = SI.drawdowns(p, t)
        if len(dd):
            dd = dd.copy()
            for c in ("peak", "trough", "recovered"):
                dd[c] = [d.strftime("%d %b %Y") if isinstance(d, pd.Timestamp) and not pd.isna(d) else "not yet"
                         for d in dd[c]]
            dd = dd.reset_index(drop=True)
            dd.index = [f"#{i + 1}" for i in range(len(dd))]
        self.dds.set_frame(dd)
        # results and dividends arrive from Yahoo in _got_events
        for t_ in (self.react, self.divs):
            t_.set_frame(pd.DataFrame())
        self.ev_kpi.set_items([])
        self.div_kpi.set_items([])

    def _got_events(self, ticker: str, ev: dict) -> None:
        if ticker != self.ticker:
            return                                    # user moved on while this was loading
        data.prune_fundamentals()
        p = data.panel()
        df, summ = SI.results_reaction(p, ticker, ev.get("earnings"))
        self.ev_kpi.ticker = ticker
        if summ:
            self.ev_kpi.set_items([
                ("Results quarters", str(summ["quarters"]), None, "with price data"),
                ("Avg results-day move", f"±{summ['avg abs move']:.1%}", None, "absolute, vs previous close"),
                ("Up reactions", f"{summ['up reactions']:.0%}", None, "results days that closed up"),
                ("Avg 20-day drift", "–" if summ["avg 20d drift"] != summ["avg 20d drift"]
                 else f"{summ['avg 20d drift']:+.1%}", theme.signed(summ["avg 20d drift"]), "after results, vs NIFTY"),
                ("Beat estimate", "–" if summ["beat estimate"] != summ["beat estimate"] else f"{summ['beat estimate']:.0%}",
                 None, "EPS above consensus"),
            ])
            v = df.copy()
            v.index = v.pop("results").dt.strftime("%d %b %Y")
            v.index.name = "results"
            self.react.set_frame(v)
        price = float(p.close[ticker].dropna().iloc[-1])
        dv, ds = SI.dividends(ev.get("dividends"), price)
        if ds:
            self.div_kpi.ticker = ticker
            self.div_kpi.set_items([
                ("Trailing 12m dividend", f"₹{ds['trailing 12m ₹']:,.2f}", None, "per share"),
                ("Trailing yield", f"{ds['trailing yield']:.2%}", None, "÷ last close"),
                ("Years paid", str(ds["years paid"]), None, "calendar years with a dividend"),
            ])
            dv.index = dv.index.strftime("%d %b %Y")
            self.divs.set_frame(dv)
        errs = ev.get("errors") or []
        self.ev_note.setText(("Yahoo: " + "; ".join(errs) + ". " if errs else "")
                             + ("" if summ else "No past results dates from Yahoo for this stock. ")
                             + ("" if ds else "No dividends recorded on Yahoo. ")
                             + "Results day = the first session that could react (the same day if announced before "
                               "15:30 IST). Dividends per share as Yahoo reports them.")

    # ------------------------------------------------------------------ public
    def show_stock(self, ticker: str) -> None:
        self.ticker = ticker
        for k in (self.desk, self.stats, self.news_head):    # tiles describe this stock (sector context)
            k.ticker = ticker
        self.tech.ticker_hint = ticker
        self.fund = None
        p = data.panel()
        self.name.setText(f"{ticker.replace('.NS', '')} · {p.names.get(ticker, '')}")
        self.sub.setText(f"NSE industry: {p.sectors.get(ticker, '')} · loading Yahoo financials…")
        self._set_wl_buttons()
        self._price_header()
        self._desk_row()
        self._chart()
        self._news()
        self._technicals()
        self._models()
        self._plan()
        self._deep_dive()
        self.nse.set_stock(ticker)
        if self.tabs.currentWidget() is self.nse:
            self.nse.load()
        for t in self.stmts.values():
            t.set_frame(pd.DataFrame())
        self.stats.set_items([])
        tk = ticker
        run_async(data.fundamentals, lambda f, tk=tk: self._got_fund(tk, f), tk,
                  on_error=lambda e, tk=tk: self.sub.setText(f"Yahoo financials unavailable: {e.splitlines()[-1][:120]}"))
        self.ev_note.setText("Loading results dates and dividends from Yahoo…")
        run_async(data.stock_events, lambda ev, tk=tk: self._got_events(tk, ev), tk,
                  on_error=lambda e, tk=tk: tk == self.ticker and self.ev_note.setText(
                      f"Yahoo results/dividend data unavailable: {e.splitlines()[-1][:120]}"))

    def refresh(self) -> None:
        if self.ticker:
            self.show_stock(self.ticker)

    def nse_changed(self) -> None:
        if self.ticker:
            self.nse.set_stock(self.ticker)
            if self.tabs.currentWidget() is self.nse:
                self.nse.load()

    def on_tick(self) -> None:
        if self.ticker:
            self._price_header()

    # ------------------------------------------------------------------ pieces
    def _live(self):
        return self.ctx.feed.quote(self.ticker) if self.ctx.feed else None

    def _price_header(self) -> None:
        p = data.panel()
        hist = p.close[self.ticker].dropna()
        tk = self._live()
        s = (self.fund or {}).get("stats", {})
        if tk:
            price, chg_pct, src = tk["price"], tk.get("change_percent"), "LIVE · Yahoo stream, may be delayed"
            chg = price * chg_pct / (100 + chg_pct) if chg_pct is not None else None
        else:
            price = s.get("currentPrice") or float(hist.iloc[-1])
            prev = s.get("previousClose") or float(hist.iloc[-2])
            chg = price - prev
            chg_pct = chg / prev * 100
            src = "LAST · " + ("Yahoo quote" if s.get("currentPrice") else f"close {hist.index[-1]:%d %b %Y}")
        col = theme.signed(chg)
        self.price.setText(
            f"<span style='font-size:24px'>₹{price:,.2f}</span> <span style='color:{col};font-size:14px'>"
            f"{_num(chg, '{:+,.2f}', '')} ({_num(chg_pct, '{:+.2f}%', '')})</span><br>"
            f"<span style='color:{theme.AMBER if tk else theme.MUTED};font-size:10px'>{src}</span>")
        self._last_price = price

    def _wl(self, tier: str | None) -> None:
        if not self.ticker:
            return
        if tier is None:
            wl.remove(self.ticker)
        else:
            wl.add(self.ticker, tier)
        self._set_wl_buttons()
        self.ctx.watchlist_changed.emit()

    def _set_wl_buttons(self) -> None:
        cur = wl.load().get(self.ticker) if self.ticker else None
        on = self.ticker is not None
        self.b_must.setVisible(on and cur is None)
        self.b_pref.setVisible(on and cur is None)
        self.b_rm.setVisible(on and cur is not None)
        if cur:
            extra = (f" · buy ≤ ₹{cur.target_buy:,.0f}" if cur.target_buy else "") + \
                    (f" · sell ≥ ₹{cur.target_sell:,.0f}" if cur.target_sell else "")
            self.wl_label.setText(f"{wl.TIERS[cur.tier]} on your watchlist{extra}")
        else:
            self.wl_label.setText("Not on your watchlist" if on else "")

    def _desk_row(self) -> None:
        t = self.ticker
        fund, sent, nxt = data.dload("fundamentals"), data.dload("sentiment"), data.dload("nextday")
        an = fund.loc[t, "analyst_score"] if fund is not None and t in fund.index and "analyst_score" in fund else None
        ne = pd.to_datetime(fund.loc[t, "next_earnings"], errors="coerce") \
            if fund is not None and t in fund.index and "next_earnings" in fund else None
        se = sent.loc[t] if sent is not None and t in sent.index else None
        pu = nxt.loc[t, "prob"] if nxt is not None and t in nxt.index else None
        self.desk.set_items([
            ("Analyst score", "–" if an is None or an != an else f"{an:.0f}/100", None, "consensus, upside, revisions"),
            ("Next results", "–" if ne is None or pd.isna(ne) else ne.strftime("%d %b %Y"), None, "Yahoo calendar"),
            ("News sentiment", "–" if se is None else f"{se['sentiment']:+.2f}",
             None if se is None else theme.signed(se["sentiment"]),
             "" if se is None else f"{int(se['n_news'])} headlines · FinBERT"),
            ("Next-day P(up)", "–" if pu is None or pu != pu else f"{pu:.0%}", None, "calibrated model, not advice"),
        ])

    def _got_fund(self, ticker: str, f: dict) -> None:
        if ticker != self.ticker:
            return                                    # user moved on while this was loading
        self.fund = f
        data.prune_fundamentals()
        s, prof = f["stats"], f["profile"]
        p = data.panel()
        self.sub.setText(f"{prof.get('sector') or ''} · {prof.get('industry') or ''} · NSE industry: {p.sectors.get(ticker, '')}")
        self._price_header()
        price = self._last_price
        lo, hi = s.get("fiftyTwoWeekLow"), s.get("fiftyTwoWeekHigh")
        pos52 = (price - lo) / (hi - lo) if lo and hi and hi > lo else None
        pct = lambda v, k=1: None if v is None or v != v else v / k  # noqa: E731
        self.stats.set_items([
            ("Market cap", fx.crore(s.get("marketCap")), None, f"EV {fx.crore(s.get('enterpriseValue'))}"),
            ("P/E (ttm)", _num(s.get("trailingPE"), "{:.1f}x"), None, f"fwd {_num(s.get('forwardPE'), '{:.1f}')}"),
            ("P/B", _num(s.get("priceToBook"), "{:.2f}x"), None, f"book ₹{_num(s.get('bookValue'), '{:,.0f}')}"),
            ("EPS (ttm)", _num(s.get("trailingEps"), "₹{:,.2f}"), None),
            ("Dividend yield", _num(pct(s.get("dividendYield"), 100), "{:.2%}"), None),
            ("ROE", _num(s.get("returnOnEquity"), "{:.1%}"), None),
            ("Debt / equity", _num(pct(s.get("debtToEquity"), 100), "{:.2f}x"), None),
            ("Net margin", _num(s.get("profitMargins"), "{:.1%}"), None),
            ("Revenue growth", _num(s.get("revenueGrowth"), "{:+.1%}"), theme.signed(s.get("revenueGrowth")), "yoy, latest qtr"),
            ("Earnings growth", _num(s.get("earningsGrowth"), "{:+.1%}"), theme.signed(s.get("earningsGrowth")), "yoy, latest qtr"),
            ("52w position", _num(pos52, "{:.0%}"), None, f"₹{_num(lo, '{:,.0f}')} – ₹{_num(hi, '{:,.0f}')}"),
            ("Beta", _num(s.get("beta"), "{:.2f}"), None, "Yahoo"),
            ("Volume", _num(s.get("volume"), "{:,.0f}"), None, f"avg {_num(s.get('averageVolume'), '{:,.0f}')}"),
            ("Operating margin", _num(s.get("operatingMargins"), "{:.1%}"), None),
            ("Promoters", _num(s.get("heldPercentInsiders"), "{:.1%}"), None, "insiders"),
            ("Institutions", _num(s.get("heldPercentInstitutions"), "{:.1%}"), None),
        ])
        for key, t in self.stmts.items():
            m = money_table(f.get(key))
            t.model_.fmt = {c: "{:,.1f}" for c in m.columns}
            t.set_frame(m)
        n = s.get("numberOfAnalystOpinions")
        html = (f"<h3 style='color:{theme.AMBER}'>Ownership</h3>"
                f"<p>Promoters / insiders: {_num(s.get('heldPercentInsiders'), '{:.1%}')}<br>"
                f"Institutions: {_num(s.get('heldPercentInstitutions'), '{:.1%}')}</p>")
        if n and s.get("targetMeanPrice"):
            up = s["targetMeanPrice"] / price - 1
            html += (f"<h3 style='color:{theme.AMBER}'>Analyst consensus ({n})</h3>"
                     f"<p>{str(s.get('recommendationKey') or '').replace('_', ' ')} · mean target "
                     f"₹{s['targetMeanPrice']:,.0f} ({up:+.1%} vs price) · range ₹{(s.get('targetLowPrice') or 0):,.0f}"
                     f" – ₹{(s.get('targetHighPrice') or 0):,.0f}</p>")
        else:
            html += "<p>No analyst coverage on Yahoo.</p>"
        html += f"<p style='color:{theme.MUTED}'>Third-party analyst views as published on Yahoo Finance. Shown for context, not endorsed.</p>"
        self.owner.setHtml(html)
        meta = [f"Employees: {prof['fullTimeEmployees']:,}" if prof.get("fullTimeEmployees") else "",
                f"<a style='color:{theme.BLUE}' href='{prof['website']}'>{prof['website']}</a>" if prof.get("website") else ""]
        self.about.setHtml(f"<p>{prof.get('longBusinessSummary') or 'No description on Yahoo.'}</p>"
                           f"<p style='color:{theme.MUTED}'>{' · '.join(m for m in meta if m)}</p>")

    def _chart(self) -> None:
        t = self.ticker
        self.chart.set_source(lambda tf, t=t: chart_source(t, tf))

    def _news(self) -> None:
        news = data.dload("news")
        if news is None or "score" not in news:
            self.news.set_frame(pd.DataFrame({"": ["No scored news yet. Run the daily refresh (F5)."]}))
            self.news_head.set_items([])
            return
        n = news[news["ticker"] == self.ticker].sort_values("published", ascending=False)
        if not len(n):
            self.news.set_frame(pd.DataFrame({"": ["No headlines about this stock in the last 48 hours."]}))
            self.news_head.set_items([])
            return
        age_h = (pd.Timestamp.now(tz="Asia/Kolkata") - pd.to_datetime(n["published"])).dt.total_seconds() / 3600
        w = (0.5 ** (age_h / 12)).fillna(0.01)
        avg = float(np.average(n["score"], weights=w))
        self.news_head.set_items([
            ("FinBERT sentiment", f"{avg:+.2f}", theme.signed(avg), "recency-weighted, −1…+1"),
            ("Headlines", str(len(n)), None, "last 48 h"),
            ("Bullish", str(int((n["score"] > 0.5).sum())), theme.GREEN),
            ("Bearish", str(int((n["score"] < -0.5).sum())), theme.RED),
        ])
        view = pd.DataFrame({"When": pd.to_datetime(n["published"]).dt.strftime("%d %b %H:%M").to_numpy(),
                             "Label": n["label"].to_numpy(), "Score": n["score"].to_numpy(),
                             "Headline": n["title"].to_numpy(), "Source": n["source"].to_numpy()})
        self._links = list(n["link"])
        self.news.set_frame(view)

    def _open_news(self, idx) -> None:
        src = self.news.proxy.mapToSource(idx)
        links = getattr(self, "_links", [])
        if src.isValid() and src.row() < len(links) and str(links[src.row()]).startswith("http"):
            QDesktopServices.openUrl(QUrl(links[src.row()]))

    def _technicals(self) -> None:
        tech = data.dload("technicals")
        if tech is None or self.ticker not in tech.index:
            self.tech.set_frame(pd.DataFrame({"": ["Run the daily refresh to compute technical indicators."]}))
            return
        r = tech.loc[self.ticker]
        rows = []
        for k, (label, group, kind) in TECH_COLUMNS.items():
            v = r.get(k)
            missing = v is None or v != v
            val = "–" if missing else f"{v:+.2%}" if kind == "pct" else ("yes" if v == 1 else "no") if kind == "flag" else f"{v:,.2f}"
            rows.append({"Group": group, "Indicator": label, "Value": val,
                         "Signal": "" if missing or k not in SIGNALS else SIGNALS[k](v)})
        self.tech.set_frame(pd.DataFrame(rows))

    def _build_plan_tab(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(h2("Pre-entry checklist", "checklist"))
        row.addStretch(1)
        row.addWidget(ExplainButton("checklist"))
        lay.addLayout(row)
        self.check_view = QTextBrowser()
        self.check_view.setMinimumHeight(250)
        lay.addWidget(self.check_view)
        self.plan_view = QTextBrowser()
        self.plan_view.setMinimumHeight(330)
        lay.addWidget(self.plan_view)
        lay.addWidget(h2("Scenarios: what each ending is worth after costs and tax", "scenario_pnl"))
        self.scen = FrameTable(fmt={"Exit price": "₹{:,.2f}", "Gross ₹": "{:+,.0f}", "Costs ₹": "{:,.0f}",
                                    "Tax ₹": "{:,.0f}", "Net ₹": "{:+,.0f}", "Net R": "{:+.2f}",
                                    "% of capital": "{:+.2%}"}, signed={"Gross ₹", "Net ₹", "Net R", "% of capital"})
        self.scen.model_.term_overrides = {c: "scenario_pnl" for c in ("Exit price", "Gross ₹", "Costs ₹", "Net ₹",
                                                                        "Net R", "% of capital")}
        self.scen.model_.term_overrides["Tax ₹"] = "stcg"
        self.scen.setMinimumHeight(210)
        lay.addWidget(self.scen)
        lay.addWidget(h2("What happened to similar past setups", "similar_setups"))
        self.sim_note = muted("")
        lay.addWidget(self.sim_note)
        self.sim_kpi = KpiRow(cols=5)
        lay.addWidget(self.sim_kpi)
        self.sim_hist = FrameTable(fmt={"Similar setups": "{:.0%}", "All trades": "{:.0%}"})
        self.sim_hist.model_.term_overrides = {c: "similar_setups" for c in ("Similar setups", "All trades", "")}
        self.sim_hist.setMinimumHeight(300)
        lay.addWidget(self.sim_hist)
        self.tabs.addTab(scrolling(w), "Trade plan")

    def _plan(self) -> None:
        from ...research.plans import make_plan
        from .desk import current_risk_state, desk_settings
        st, rs = desk_settings(), current_risk_state()
        p = data.panel()
        pl = make_plan(p, self.ticker, st["capital"], st["risk_pct"], rs.multiplier)
        if pl is None:
            self.plan_view.setHtml("<p>Not enough price history for a plan.</p>")
            self.check_view.setHtml("")
            self.scen.set_frame(pd.DataFrame())
            return
        A, M = theme.AMBER, theme.MUTED
        rows = [("Entry (last close)", f"₹{pl.entry:,.2f}", "you'd buy at the next open"),
                ("Stop", f"₹{pl.stop:,.2f}", f"{pl.stop_pct:.1%} below entry · {pl.risk_per_share / pl.atr:.1f}× ATR"),
                ("R (risk per share)", f"₹{pl.risk_per_share:,.2f}", f"ATR(14) ₹{pl.atr:,.2f}"),
                ("T1 = 1.5R", f"₹{pl.t1:,.2f}", "book about a third, move the stop to entry"),
                ("T2 = 2.5R", f"₹{pl.t2:,.2f}", f"then trail the rest ₹{pl.trail_atr:,.2f} (3× ATR) under the highest close"),
                ("Quantity", f"{pl.qty:,}", f"₹{pl.position_value:,.0f} invested"),
                ("Rupee risk", f"₹{pl.rupee_risk:,.0f}", f"{pl.risk_pct:.2%} of ₹{pl.capital:,.0f} × {pl.multiplier:g}"),
                ("At T1 / T2", f"₹{pl.qty * (pl.t1 - pl.entry):+,.0f} / ₹{pl.qty * (pl.t2 - pl.entry):+,.0f}", "before costs")]
        tr = "".join(f"<tr><td style='color:{M};padding:3px 10px'>{a}</td><td style='padding:3px 10px'><b>{b}</b></td>"
                     f"<td style='color:{M};padding:3px 10px'>{c}</td></tr>" for a, b, c in rows)
        notes = "".join(f"<li>{n}</li>" for n in pl.notes)
        self.plan_view.setHtml(
            f"<h3 style='color:{A}'>Swing plan for {self.ticker.replace('.NS', '')}</h3><table>{tr}</table>"
            f"<p><b style='color:{A}'>Market risk state:</b> {rs.state} (×{rs.multiplier:g}): {'; '.join(rs.reasons)}</p>"
            + (f"<ul>{notes}</ul>" if notes else "")
            + f"<p style='color:{M}'>Capital and risk % are set in My desk. Rules, not a forecast or advice; nothing is "
              "ordered. Gaps can fill beyond the stop. The plan is also drawn on the Chart tab.</p>")
        # scenarios
        self.scen.set_frame(TC.scenarios(pl))
        # similar setups
        hist = data.tagged_history()
        now = TC.tags_now(p, self.ticker)
        sim = TC.similar_setups(hist, now) if hist is not None else None
        self.sim_kpi.ticker = self.ticker
        if sim is not None and sim.n:
            s, b = sim.stats, sim.baseline
            desc = ", ".join(f"{TC.TAG_LABELS[k]}: {now[k]}" for k in sim.used)
            self.sim_note.setText(
                f"Trades that started like {self.ticker.replace('.NS', '')} today ({desc})"
                + (f"; ignoring {', '.join(TC.TAG_LABELS[k] for k in sim.dropped)} to get at least 30" if sim.dropped else "")
                + f". From the point-in-time replay of these exact plan rules, 2015-2026, after costs: {s['n']:,} of "
                  f"{b['n']:,} trades in any NIFTY 200 stock, not this stock's own history.")
            self.sim_kpi.set_items([
                ("Similar trades", f"{s['n']:,}", None, f"of {b['n']:,}"),
                ("Win rate", f"{s['win_rate']:.0%}", None, f"all trades {b['win_rate']:.0%}"),
                ("Average R", f"{s['avg_r']:+.2f}R", theme.signed(s["avg_r"]), f"all trades {b['avg_r']:+.2f}R · t {s['t']:.1f}"),
                ("Median R", f"{s['median_r']:+.2f}R", theme.signed(s["median_r"]), "the typical trade"),
                ("Average days held", f"{s['avg_days']:.0f}", None, "max 60"),
            ])
            hs, hb = TC.r_histogram(sim.trades["r"]), TC.r_histogram(hist["r"])
            self.sim_hist.set_frame(pd.DataFrame({"Similar setups": hs, "All trades": hb,
                                                  "": ["█" * int(round(v * 60)) for v in hs]}).rename_axis("Outcome"))
        else:
            self.sim_note.setText("No replay history yet: open Track record and press Recompute, or run "
                                  "scripts/tag_setups.py.")
            self.sim_kpi.set_items([])
            self.sim_hist.set_frame(pd.DataFrame())
        # checklist
        table = data.screener_table()
        row = table.loc[self.ticker] if table is not None and self.ticker in table.index else None
        dte = row.get("days_to_earnings") if row is not None else None
        verdict, checks = TC.checklist(pl, rs, row, sim, dte)
        col = {"Go": theme.GREEN, "Wait": theme.AMBER, "No-go": theme.RED}[verdict]
        icon = {"ok": ("✓", theme.GREEN), "caution": ("!", theme.AMBER), "stop": ("✖", theme.RED)}
        items = "".join(f"<tr><td style='color:{icon[c.status][1]};padding:2px 8px'><b>{icon[c.status][0]}</b></td>"
                        f"<td style='padding:2px 8px'><b>{c.item}</b></td><td style='color:{M};padding:2px 8px'>{c.detail}"
                        f"</td></tr>" for c in checks)
        self.check_view.setHtml(
            f"<p style='font-size:16px'>Verdict: <b style='color:{col}'>{verdict}</b> "
            f"<span style='color:{M};font-size:11px'>Go = no more than one caution · Wait = two or more cautions · "
            f"No-go = a blocking problem</span></p><table>{items}</table>"
            f"<p style='color:{M}'>A checklist of the rules this app uses, to slow down an impulsive entry. Not advice; "
            "the cautions are reasons to look closer, not predictions.</p>")

    def _models(self) -> None:
        t = self.ticker
        rows = []
        for m in ("ffnn", "lstm", "transformer"):
            pr = data.predictions(m)
            if pr is None:
                continue
            d = pr.index.get_level_values(0).max()
            sc = pr.xs(d, level=0)["score"]
            if t in sc.index:
                rows.append(f"<tr><td>{m.upper()}</td><td>{sc[t]:.1%}</td><td>{sc.rank(pct=True)[t]:.0%}</td>"
                            f"<td>{d.date()}</td></tr>")
        nxt = data.dload("nextday")
        html = ""
        if nxt is not None and t in nxt.index:
            html += f"<p><b>Next-day P(up):</b> {nxt.loc[t, 'prob']:.1%} (LightGBM, calibrated walk-forward)</p>"
        if rows:
            html += ("<table cellpadding=4><tr style='color:#8A94A6'><td>Model</td><td>P(beat median next month)</td>"
                     "<td>Rank %ile</td><td>As of</td></tr>" + "".join(rows) + "</table>")
        vf = data.vol_forecasts()
        if vf is not None and t in vf.index.get_level_values(1):
            v = vf.xs(t, level=1).iloc[-1]
            html += f"<p>Next-month volatility forecast: LSTM {v['lstm']:.1%} · GARCH {v['garch']:.1%}</p>"
        html += f"<p style='color:{theme.MUTED}'>Walk-forward research output. Not a recommendation.</p>"
        self.models.setHtml(html)


TF_NAMES = {"1m": "1-minute", "5m": "5-minute", "15m": "15-minute", "1h": "hourly", "1D": "daily (adjusted)",
            "1W": "weekly (adjusted)"}


def chart_source(ticker: str, tf: str) -> dict:
    """Bars, comparison series and the swing plan for the stock chart (runs off the UI thread for intraday)."""
    from ...research import chartdata as C
    from ...research.plans import make_plan
    from .desk import current_risk_state, desk_settings
    p = data.panel()
    df = C.bars(p, ticker, tf)
    compare = {}
    if tf in ("1D", "1W"):
        compare["NIFTY 50"] = p.bench
        sec = p.sectors.get(ticker)
        peers = [x for x in p.close.columns if p.sectors.get(x) == sec and x != ticker]
        if peers:
            r = p.close[peers].pct_change(fill_method=None).mean(axis=1).fillna(0)
            compare[f"{sec} (equal weight)"] = (1 + r).cumprod()
        if tf == "1W":
            compare = {k: v.reindex(df.index, method="ffill") for k, v in compare.items()}
    else:
        try:
            compare["NIFTY 50"] = C.intraday_bars("^NSEI", tf)["Close"]
        except Exception:  # noqa: BLE001 - the comparison is optional; the chart still draws
            pass
    st, rs = desk_settings(), current_risk_state()
    plan = make_plan(p, ticker, st["capital"], st["risk_pct"], rs.multiplier)
    return {"df": df, "compare": compare, "plan": plan,
            "title": f"{ticker.replace('.NS', '')} · {TF_NAMES.get(tf, tf)}" + (" · IST, Yahoo (may be delayed)"
                                                                               if tf not in ("1D", "1W") else "")}
