"""MKT: live market monitor: status tape, sector heatmap and movers for the NIFTY 200."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from .. import data, theme
from ...livefeed import INDEX_SYMBOLS
from ..widgets import FrameTable, KpiRow, Treemap, h2, muted
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
        lay = vbox(self)
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
        sl.addWidget(self.up)
        sl.addWidget(h2("Top losers"))
        sl.addWidget(self.dn)
        split.addWidget(side)
        split.setSizes([1000, 260])
        lay.addWidget(split, 1)
        self.base: MarketBase | None = None
        self._sig = None

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
        mv = pd.DataFrame({"Last": lt["last"], "Chg": lt["chg1"]}).dropna(subset=["Chg"])
        self.up.set_frame(mv.nlargest(12, "Chg"), live=not force_layout)
        self.dn.set_frame(mv.nsmallest(12, "Chg"), live=not force_layout)
        n_live = int(lt["live"].sum())
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
            ("Adv / Dec", f"{int((lt['chg1'] > 0).sum())} / {int((lt['chg1'] < 0).sum())}", None),
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
