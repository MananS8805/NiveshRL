"""MKT: live market monitor (status tape, sector heatmap, movers), global markets & macro, breadth history and
sector rotation (RRG) for the NIFTY 200."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSplitter, QTabWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from .. import data, theme
from ...livefeed import INDEX_SYMBOLS
from ...research import marketdash as MD
from ..widgets import ExplainButton, FrameTable, KpiRow, Treemap, h2, line_chart, muted, run_async
from . import Panel, vbox

WINDOWS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


class MarketBase:
    """Precomputed daily reference values; live ticks are overlaid cheaply on each refresh."""

    def __init__(self):
        p = data.panel()
        self.p = p
        close = p.close.ffill()
        self.last_close = close.iloc[-1]
        self.prev_close = close.iloc[-2]
        self.ref = {k: close.iloc[-1 - n] if k != "1D" else close.iloc[-2] for k, n in WINDOWS.items()}
        self.ma200 = p.close.rolling(200, min_periods=150).mean().iloc[-1]
        # plain object-dtype index/labels: Arrow-backed strings are slow to iterate on every tick
        self.idx = pd.Index(list(self.last_close.index), dtype=object)
        self.sectors = pd.Series(p.sectors.reindex(p.tickers).fillna("Other").astype(object).to_numpy(), index=self.idx)
        self.names = pd.Series(p.names.reindex(p.tickers).fillna("").astype(object).to_numpy(), index=self.idx)
        b, v = p.bench.dropna(), p.vix.dropna()
        self.bench, self.vix = b, v
        self.vix_med = float(v.rolling(252).median().iloc[-1])
        self.ytd_base = float(b[b.index.year < b.index[-1].year].iloc[-1]) if (b.index.year < b.index[-1].year).any() else float(b.iloc[0])

    def table(self, snap: dict) -> pd.DataFrame:
        """Vectorised overlay of live ticks (plain numpy writes; pandas per-cell setitem is ~100x slower)."""
        if not hasattr(self, "_pos"):
            self._pos = {t: i for i, t in enumerate(self.last_close.index)}
            self._last0 = self.last_close.to_numpy(dtype=float)
            self._chg0 = (self.last_close / self.prev_close - 1).to_numpy(dtype=float)
        last, chg1 = self._last0.copy(), self._chg0.copy()
        live = np.zeros(len(last), dtype=bool)
        pos = self._pos
        for t, tk in snap.items():
            i = pos.get(t)
            if i is not None:
                last[i] = tk["price"]
                cp = tk.get("change_percent")
                if cp is not None:
                    chg1[i] = cp / 100
                live[i] = True
        return pd.DataFrame({"last": last, "chg1": chg1, "live": live}, index=self.idx)


class MarketPanel(Panel):
    title = "Market monitor"
    code = "MKT"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        outer = vbox(self, 0)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        live = QWidget()
        lay = vbox(live)
        self.tabs.addTab(live, "Live")
        self.kpis = KpiRow(cols=9)
        lay.addWidget(self.kpis)
        bar = QHBoxLayout()
        bar.addWidget(h2("Heatmap"))
        self.window = QComboBox()
        self.window.addItems(list(WINDOWS))
        self.window.currentTextChanged.connect(lambda _: self.on_tick(force_layout=True))
        bar.addWidget(QLabel("window"))
        bar.addWidget(self.window)
        bar.addStretch(1)
        self.note = muted("")
        bar.addWidget(self.note)
        lay.addLayout(bar)
        split = QSplitter(Qt.Horizontal)
        self.tree = Treemap()
        self.tree.clicked_ticker.connect(self.stock_selected)
        split.addWidget(self.tree)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(h2("Top gainers"))
        self.up = FrameTable(fmt={"Last": "₹{:,.2f}", "Chg": "{:+.2%}"}, signed={"Chg"})
        self.dn = FrameTable(fmt={"Last": "₹{:,.2f}", "Chg": "{:+.2%}"}, signed={"Chg"})
        for t in (self.up, self.dn):
            t.row_clicked.connect(lambda tk: self.stock_selected.emit(str(tk)))
            t.setSortingEnabled(False)                 # already ranked; skip re-sorting on every live update
            t.proxy.setDynamicSortFilter(False)
        sl.addWidget(self.up)
        sl.addWidget(h2("Top losers"))
        sl.addWidget(self.dn)
        split.addWidget(side)
        split.setSizes([1000, 260])
        lay.addWidget(split, 1)
        self.base: MarketBase | None = None
        self._sig = None
        self._build_global()
        self._build_breadth()
        self._build_rrg()
        self._done: set[int] = set()
        self.tabs.currentChanged.connect(self._tab_changed)

    # ------------------------------------------------------------------ extra tabs (built on first view)
    def _build_global(self) -> None:
        w = QWidget()
        lay = vbox(w)
        row = QHBoxLayout()
        row.addWidget(h2("Global markets & macro", "global_markets"))
        row.addStretch(1)
        row.addWidget(ExplainButton("global_markets"))
        lay.addLayout(row)
        self.g_kpi = KpiRow(cols=7)
        lay.addWidget(self.g_kpi)
        pct = "{:+.2%}"
        self.g_tab = FrameTable(fmt={"Last": "{:,.2f}", "1D": pct, "1W": pct, "1M": pct, "YTD": pct, "1Y": pct,
                                     "Corr. with NIFTY (weekly, 1y)": "{:+.2f}"}, signed={"1D", "1W", "1M", "YTD", "1Y"})
        self.g_tab.model_.term_overrides = {"Group": "global_markets", "Last": "global_markets", "As of": "global_markets",
                                            "Corr. with NIFTY (weekly, 1y)": "global_corr"}
        lay.addWidget(self.g_tab, 1)
        self.g_note = muted("")
        lay.addWidget(self.g_note)
        self.tabs.addTab(w, "Global && macro")

    def _build_breadth(self) -> None:
        w = QWidget()
        self.b_lay = vbox(w)
        self.b_kpi = KpiRow(cols=6)
        self.b_lay.addWidget(self.b_kpi)
        self.b_lay.addWidget(muted("Breadth = how many stocks take part in a move. Today's NIFTY 200 members over the "
                                   "last 3 years: stocks that left the index are missing, so older readings are a little "
                                   "flattering."))
        self.b_charts: list = []
        self.tabs.addTab(w, "Breadth")

    def _build_rrg(self) -> None:
        w = QWidget()
        lay = vbox(w)
        row = QHBoxLayout()
        row.addWidget(h2("Sector rotation (relative rotation graph)", "rrg"))
        row.addStretch(1)
        row.addWidget(ExplainButton("rrg"))
        lay.addLayout(row)
        split = QSplitter(Qt.Horizontal)
        self.rrg_plot = pg.PlotWidget()
        self.rrg_plot.setBackground(theme.BG)
        self.rrg_plot.setMenuEnabled(False)
        self.rrg_plot.showGrid(x=True, y=True, alpha=0.1)
        self.rrg_plot.setLabel("bottom", "RS-Ratio (relative trend vs NIFTY) →")
        self.rrg_plot.setLabel("left", "RS-Momentum (is it improving?) →")
        split.addWidget(self.rrg_plot)
        self.rrg_tab = FrameTable(fmt={"RS-Ratio": "{:.2f}", "RS-Momentum": "{:.2f}", "Stocks": "{:.0f}",
                                       "13-week return vs NIFTY": "{:+.1%}"}, signed={"13-week return vs NIFTY"})
        self.rrg_tab.model_.term_overrides = {c: "rrg" for c in ("RS-Ratio", "RS-Momentum", "Quadrant", "4 weeks ago",
                                                                  "Stocks", "13-week return vs NIFTY")}
        self.rrg_tab.explainable = True
        split.addWidget(self.rrg_tab)
        split.setSizes([800, 520])
        lay.addWidget(split, 1)
        lay.addWidget(muted("Weekly. Tails show the last 5 weeks, the big dot is this week. Sectors usually rotate clockwise: "
                            "Improving → Leading → Weakening → Lagging. Equal-weight industry indices of today's NIFTY 200 "
                            "members with at least 3 stocks; an open approximation of the JdK RRG, so levels differ from "
                            "commercial charts."))
        self.tabs.addTab(w, "Sector rotation")

    def _tab_changed(self, i: int) -> None:
        if i in self._done or i == 0:
            return
        self._done.add(i)
        {1: self._load_global, 2: self._load_breadth, 3: self._load_rrg}.get(i, lambda: None)()

    def _load_global(self) -> None:
        self.g_note.setText("Loading global markets from Yahoo…")
        run_async(MD.fetch_global, self._got_global,
                  on_error=lambda e: (self._done.discard(1), self.g_note.setText(
                      f"Global data unavailable: {e.splitlines()[-1][:120]}")))

    def _got_global(self, closes) -> None:
        t = MD.global_table(closes, data.panel().bench)
        if t.empty:
            self.g_note.setText("No global data.")
            return
        view = t.drop(columns=["unit"]).copy()
        for c in ("1D", "1W", "1M", "YTD", "1Y"):            # the yield moves in percentage points, not percent
            view[c] = view[c].astype(object)
            for name in t.index[t["unit"] == "pp"]:
                v = t.loc[name, c]
                view.loc[name, c] = "–" if v != v else f"{v * 100:+.0f} bp"
        view["As of"] = pd.to_datetime(view["As of"]).dt.strftime("%d %b")
        self.g_tab.set_frame(view)
        pick = ["S&P 500", "Nasdaq", "Nikkei 225", "Hang Seng", "Brent crude", "Gold", "USD/INR"]
        self.g_kpi.set_items([(n, f"{t.loc[n, 'Last']:,.2f}", theme.signed(t.loc[n, "1D"]), f"{t.loc[n, '1D']:+.2%} on the day")
                              for n in pick if n in t.index])
        self.g_note.setText("Yahoo daily closes, refreshed every 30 minutes; each market's last close (see 'As of'). "
                            "Correlation uses weekly returns over the last year because the sessions close at different "
                            "times. 10-year yield changes in basis points (1 bp = 0.01%).")

    def _load_breadth(self) -> None:
        b = MD.breadth_history(data.panel())
        last = b.iloc[-1]
        self.b_kpi.set_items([
            ("% above 50-day", f"{last['% above 50-day']:.0%}", theme.signed(last["% above 50-day"] - 0.5), "of NIFTY 200"),
            ("% above 200-day", f"{last['% above 200-day']:.0%}", theme.signed(last["% above 200-day"] - 0.5)),
            ("Adv / Dec", f"{int(last['Advancers'])} / {int(last['Decliners'])}", None, f"{b.index[-1]:%d %b}"),
            ("New highs − lows", f"{int(last['Highs − lows']):+d}", theme.signed(last["Highs − lows"]),
             f"{int(last['New highs'])} highs, {int(last['New lows'])} lows"),
            ("A/D line, 1 month", f"{int(b['A/D line'].iloc[-1] - b['A/D line'].iloc[-22]):+d}",
             theme.signed(b["A/D line"].iloc[-1] - b["A/D line"].iloc[-22]), "net advancers"),
            ("Breadth 1 month ago", f"{b['% above 50-day'].iloc[-22]:.0%}", None, "% above 50-day"),
        ])
        for c in self.b_charts:
            self.b_lay.removeWidget(c)
            c.deleteLater()
        charts = [
            (line_chart({"% above 50-day": b["% above 50-day"] * 100, "% above 200-day": b["% above 200-day"] * 100},
                        "Share of stocks above their 50- and 200-day averages (%)"), "chart_breadth"),
            (line_chart({"A/D line": b["A/D line"]}, "Advance-decline line (cumulative advancers − decliners)"),
             "ad_line"),
            (line_chart({"Highs − lows": b["Highs − lows"].rolling(5).mean(), "NIFTY % change":
                         (b["NIFTY"] / b["NIFTY"].iloc[0] - 1) * 100},
                        "New 52-week highs minus lows (5-day average) and NIFTY % change"), "highs_lows"),
        ]
        self.b_charts = []
        for ch, key in charts:
            ch.explain_key = key
            ch.setMinimumHeight(230)
            ch.set_range("1Y")
            self.b_lay.addWidget(ch, 1)
            self.b_charts.append(ch)

    def _load_rrg(self) -> None:
        tab, tails = MD.rrg(data.panel(), tail=5)
        self.rrg_tab.set_frame(tab)
        pl = self.rrg_plot
        pl.clear()
        if tab.empty:
            return
        xs = np.r_[[t["RS-Ratio"].to_numpy() for t in tails.values()]].ravel() if tails else np.array([100])
        ys = np.r_[[t["RS-Momentum"].to_numpy() for t in tails.values()]].ravel() if tails else np.array([100])
        dx = max(abs(np.nanmax(xs) - 100), abs(np.nanmin(xs) - 100), 1) * 1.15
        dy = max(abs(np.nanmax(ys) - 100), abs(np.nanmin(ys) - 100), 1) * 1.15
        for (x0, y0, col, name) in [(100, 100, "#12301F", "Leading"), (100, 100 - dy, "#33290F", "Weakening"),
                                    (100 - dx, 100 - dy, "#3A1414", "Lagging"), (100 - dx, 100, "#122436", "Improving")]:
            r = pg.QtWidgets.QGraphicsRectItem(x0, y0, dx, dy)
            r.setBrush(pg.mkBrush(col))
            r.setPen(pg.mkPen(None))
            r.setZValue(-10)
            pl.addItem(r)
            lab = pg.TextItem(name, color=theme.MUTED, anchor=(0, 0) if y0 >= 100 else (0, 1))
            lab.setPos(x0 + dx * 0.02 if x0 >= 100 else x0 + dx * 0.02, y0 + dy * 0.98 if y0 >= 100 else y0 + dy * 0.02)
            pl.addItem(lab)
        quad_col = {"Leading": theme.GREEN, "Weakening": theme.AMBER, "Lagging": theme.RED, "Improving": theme.BLUE}
        for sec, t in tails.items():
            q = tab.loc[sec, "Quadrant"] if sec in tab.index else "–"
            col = quad_col.get(q, theme.MUTED)
            pl.plot(t["RS-Ratio"].to_numpy(), t["RS-Momentum"].to_numpy(), pen=pg.mkPen(col, width=0.9),
                    symbol="o", symbolSize=4, symbolBrush=col, symbolPen=None)
            pl.plot([t["RS-Ratio"].iloc[-1]], [t["RS-Momentum"].iloc[-1]], pen=None, symbol="o", symbolSize=10,
                    symbolBrush=col, symbolPen=pg.mkPen(theme.TEXT))
            lab = pg.TextItem(sec[:22], color=col, anchor=(0, 1))
            lab.setPos(t["RS-Ratio"].iloc[-1], t["RS-Momentum"].iloc[-1])
            pl.addItem(lab)
        pl.addLine(x=100, pen=pg.mkPen(theme.MUTED, style=Qt.DashLine))
        pl.addLine(y=100, pen=pg.mkPen(theme.MUTED, style=Qt.DashLine))
        pl.setXRange(100 - dx, 100 + dx, padding=0)
        pl.setYRange(100 - dy, 100 + dy, padding=0)

    def refresh(self) -> None:
        self.base = MarketBase()
        self.on_tick(force_layout=True)

    def on_tick(self, force_layout: bool = False) -> None:
        if self.base is None:
            return
        b = self.base
        snap = self.ctx.feed.snapshot() if self.ctx.feed else {}
        sig = (len(snap), max((v["ts"] for v in snap.values()), default=0), self.window.currentText())
        if sig == self._sig and not force_layout:
            return                                          # nothing new since last paint
        self._sig = sig
        # Spread the work: heatmap on one tick, movers + tape on the next, so no single
        # UI-thread slice is long (each half still refreshes every 2 s).
        self._phase = not getattr(self, "_phase", False)
        do_map, do_rest = force_layout or self._phase, force_layout or not self._phase
        lt = b.table(snap)
        if do_map:
            self._heatmap(b, lt, force_layout)
        if do_rest:
            self._rest(b, lt, snap, force_layout)

    def _heatmap(self, b, lt, force_layout) -> None:
        win = self.window.currentText()
        ret = lt["chg1"] if win == "1D" else lt["last"] / b.ref[win].to_numpy() - 1
        hm = pd.DataFrame({"sector": b.sectors.to_numpy(), "ret": np.asarray(ret, dtype=float),
                           "last": lt["last"].to_numpy(), "name": b.names.to_numpy()}, index=b.idx)
        hm = hm[np.isfinite(hm["last"].to_numpy())]
        if force_layout:
            self.tree.df = pd.DataFrame()
        self.tree.set_data(hm)

    def _rest(self, b, lt, snap, force_layout) -> None:
        chg, last = lt["chg1"].to_numpy(), lt["last"].to_numpy()
        ok = np.flatnonzero(np.isfinite(chg))
        order = ok[np.argsort(chg[ok], kind="stable")]
        for table, idx in ((self.up, order[::-1][:12]), (self.dn, order[:12])):
            table.set_frame(pd.DataFrame({"Last": last[idx], "Chg": chg[idx]}, index=b.idx[idx]),
                            live=not force_layout)
        n_live = int(lt["live"].to_numpy().sum())
        self.note.setText(f"{n_live} streaming · others at last close {b.p.close.index[-1]:%d %b %Y}")
        # status tape
        nifty, vix = snap.get("^NSEI"), snap.get("^INDIAVIX")
        b_last = nifty["price"] if nifty else float(b.bench.iloc[-1])
        d1 = nifty["change_percent"] / 100 if nifty and nifty.get("change_percent") is not None \
            else float(b.bench.iloc[-1] / b.bench.iloc[-2] - 1)
        v_last = vix["price"] if vix else float(b.vix.iloc[-1])
        m1 = b_last / float(b.bench.iloc[-22]) - 1
        ytd = b_last / b.ytd_base - 1
        ma = b.ma200.to_numpy()
        breadth = float(np.nanmean(np.where(np.isnan(ma), np.nan, lt["last"].to_numpy() > ma)))
        reg = data.regimes_weekly()
        r = reg["regime"].iloc[-1] if reg is not None else "n/a"
        stat, lvl = self.ctx.feed.status() if self.ctx.feed else ("feed off", "offline")
        col = {"live": theme.GREEN, "stale": theme.AMBER, "closed": theme.MUTED, "offline": theme.RED}[lvl]
        self.kpis.set_items([
            ("Feed", stat.split(" · ")[0], col, " · ".join(stat.split(" · ")[1:])),
            ("NIFTY 50", f"{b_last:,.1f}", None),
            ("1D", f"{d1:+.2%}", theme.signed(d1)),
            ("1M", f"{m1:+.2%}", theme.signed(m1)),
            ("YTD", f"{ytd:+.2%}", theme.signed(ytd)),
            ("India VIX", f"{v_last:.2f}", theme.RED if v_last > b.vix_med else theme.GREEN, f"1y median {b.vix_med:.1f}"),
            ("Breadth >200DMA", f"{breadth:.0%}", theme.signed(breadth - 0.5)),
            ("Adv / Dec", f"{int((chg > 0).sum())} / {int((chg < 0).sum())}", None),
            ("Regime", r, theme.GREEN if r == "Bull" else theme.RED if r == "Stress" else theme.AMBER),
        ])


def ticker_strip_text(feed, base: MarketBase | None, symbols: list[str]) -> str:
    """HTML for the scrolling ticker strip: indices first, then the given symbols."""
    snap = feed.snapshot() if feed else {}
    parts = []
    for s in INDEX_SYMBOLS + symbols:
        tk = snap.get(s)
        if tk:
            px, ch = tk["price"], (tk.get("change_percent") or 0) / 100
        elif base is not None and s in base.last_close.index:
            px, ch = float(base.last_close[s]), float(base.last_close[s] / base.prev_close[s] - 1)
        elif base is not None and s == "^NSEI":
            px, ch = float(base.bench.iloc[-1]), float(base.bench.iloc[-1] / base.bench.iloc[-2] - 1)
        elif base is not None and s == "^INDIAVIX":
            px, ch = float(base.vix.iloc[-1]), float(base.vix.iloc[-1] / base.vix.iloc[-2] - 1)
        else:
            continue
        if not np.isfinite(px):
            continue
        name = {"^NSEI": "NIFTY", "^INDIAVIX": "VIX"}.get(s, s.replace(".NS", ""))
        parts.append(f"<span style='color:{theme.AMBER}'>{name}</span> {px:,.2f} "
                     f"<span style='color:{theme.signed(ch)}'>{ch:+.2%}</span>")
    return " &nbsp;│&nbsp; ".join(parts)
