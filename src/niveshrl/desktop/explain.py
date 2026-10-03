"""The Explain panel: renders a glossary entry (what it is, how it's computed here, value ranges with the current
value highlighted, sector context, the app's own measured record, caveats and related terms)."""
from __future__ import annotations

import html
import math

import pandas as pd
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout

from .. import glossary as gl
from . import data, theme

# Glossary keys whose value is best judged against the stock's sector (screener column used for the median).
SECTOR_COLS = {"pe": "trailingPE", "pb": "priceToBook", "ev_ebitda": "enterpriseToEbitda", "roe": "returnOnEquity",
               "dividend_yield": "dividendYield", "debt_equity": "debtToEquity", "net_margin": "profitMargins",
               "operating_margin": "operatingMargins", "revenue_growth": "revenueGrowth", "rsi": "rsi14",
               "vol_60d": "vol_60d", "beta": "beta_1y", "analyst_score": "analyst_score"}
PCT_COLS = {"Accuracy", "IC hit rate", "Top decile / mo", "Bottom decile / mo", "Spread / mo", "Top10 hit rate",
            "Top10 excess / day", "Top10 net of costs / day", "MAE vol", "CAGR", "Vol", "MaxDD", "HitRate", "Alpha"}


def _f(col: str, v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if math.isnan(x):
        return "–"
    if col in PCT_COLS:
        signed = any(k in col for k in ("decile", "excess", "net", "Spread")) or col == "Alpha"
        return f"{x:+.2%}" if signed else f"{x:.1%}"
    if col in ("Months", "Days", "N"):
        return f"{x:,.0f}"
    if col.startswith(("Costs", "Tax")):
        return f"₹{x:,.0f}"
    return f"{x:.3f}"


def render_html(e: gl.Entry, value=None, ticker: str | None = None) -> str:
    A, M, T = theme.AMBER, theme.MUTED, theme.TEXT
    h = [f"<h2 style='color:{A};margin:0'>{html.escape(e.title)}</h2>",
         f"<div style='color:{M};font-size:11px;margin-bottom:8px'>{html.escape(e.category)}</div>"]
    v = gl.parse_value(value) if value is not None and not isinstance(value, float) else value
    hit = gl.band(e, v) if v is not None else None
    if v is not None and not (isinstance(v, float) and math.isnan(v)):
        who = f"{ticker.replace('.NS', '')}: " if ticker else ""
        shown = value if isinstance(value, str) else gl.fmt(e, v)
        verdict = f" → <b style='color:{A}'>{html.escape(hit[1][1])}</b>" if hit else ""
        h.append(f"<p style='font-size:15px;color:{T}'>{html.escape(who)}<b>{html.escape(str(shown))}</b>{verdict}</p>")
    h.append(f"<p><b style='color:{A}'>What it is.</b> {html.escape(e.what)}</p>")
    if e.how:
        h.append(f"<p><b style='color:{A}'>How this app computes it.</b> {html.escape(e.how)}</p>")
    if e.reading:
        h.append(f"<p><b style='color:{A}'>How to read it.</b> {html.escape(e.reading)}</p>")
    if e.bands:
        rows = []
        for i, (rng, label, meaning) in enumerate(gl.band_ranges(e)):
            on = hit is not None and hit[0] == i
            style = f"background:#2A2010;color:{A};font-weight:600" if on else ""
            mark = " ◀ here" if on else ""
            rows.append(f"<tr style='{style}'><td style='padding:3px 8px;white-space:nowrap'>{html.escape(rng)}</td>"
                        f"<td style='padding:3px 8px'><b>{html.escape(label)}</b>{mark}</td>"
                        f"<td style='padding:3px 8px;color:{T if on else M}'>{html.escape(meaning)}</td></tr>")
        h.append(f"<p><b style='color:{A}'>What different values usually mean</b></p>"
                 f"<table cellspacing=0 style='border:1px solid {theme.GRID}'>{''.join(rows)}</table>"
                 f"<div style='color:{M};font-size:10px'>Common conventions and this project's measurements, "
                 f"not rules or advice.</div>")
    ctx = _sector_context(e, ticker)
    if ctx:
        h.append(ctx)
    if e.use:
        h.append(f"<p><b style='color:{A}'>How it helps.</b> {html.escape(e.use)}</p>")
    rec = _measured(e)
    if rec:
        h.append(rec)
    if e.caveat:
        h.append(f"<p><b style='color:{theme.RED}'>Watch out.</b> {html.escape(e.caveat)}</p>")
    rel = [gl.G[k] for k in e.related if k in gl.G]
    if rel:
        links = " · ".join(f"<a style='color:{theme.BLUE}' href='term:{r.key}'>{html.escape(r.title)}</a>" for r in rel)
        h.append(f"<p><b style='color:{A}'>Related.</b> {links}</p>")
    return "".join(h)


def _sector_context(e: gl.Entry, ticker: str | None) -> str:
    col = SECTOR_COLS.get(e.key)
    if not col or not ticker:
        return ""
    t = data.screener_table()
    if t is None or ticker not in t.index or col not in t:
        return ""
    sec = t.loc[ticker, "sector"] if "sector" in t else None
    x = pd.to_numeric(t[col], errors="coerce")
    if e.key in ("pe",):
        x = x.where(x > 0)
    peers = x[t["sector"] == sec].dropna() if sec is not None else x.dropna()
    if len(peers) < 3:
        return ""
    med, allmed, me = peers.median(), x.median(), x.get(ticker)
    pct = (peers < me).mean() if me == me else float("nan")
    f = (lambda z: gl.fmt(e, z))
    me_s = f(me) if me == me else "–"
    return (f"<p><b style='color:{theme.AMBER}'>In context.</b> {html.escape(ticker.replace('.NS', ''))}: {me_s} · "
            f"median of {len(peers)} {html.escape(str(sec))} stocks: {f(med)} · NIFTY 200 median: {f(allmed)}"
            + (f" · higher than {pct:.0%} of its sector peers" if pct == pct else "") + "</p>")


def _measured(e: gl.Entry) -> str:
    if not e.results:
        return ""
    csv, row = e.results
    df = data.results_csv(csv)
    if df is None or row not in df.index:
        return ""
    r = df.loc[row].drop(labels=[c for c in ["Train time (s)", "TaxLiability(Rs)"] if c in df.columns])
    cells = "".join(f"<tr><td style='padding:2px 8px;color:{theme.MUTED}'>{html.escape(str(c))}</td>"
                    f"<td style='padding:2px 8px'>{_f(str(c), v)}</td></tr>" for c, v in r.items())
    src = {"rankers_summary.csv": "monthly walk-forward 2012 → today, before costs",
           "nextday_summary.csv": "daily walk-forward 2015 → today",
           "vol_summary.csv": "walk-forward 2012 → today, next-month volatility",
           "metrics_val.csv": "RL validation period 2019–20, after costs"}.get(csv, "")
    return (f"<p><b style='color:{theme.AMBER}'>Measured in this app</b> <span style='color:{theme.MUTED}'>"
            f"({html.escape(src)}; {html.escape(csv)})</span></p><table>{cells}</table>")


class ExplainPanel(QFrame):
    """Right-hand side panel. Stays open while you click other terms; ✕ hides it."""
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("explain")
        self.setFixedWidth(430)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        t = QLabel("EXPLAIN")
        t.setObjectName("h2")
        top.addWidget(t)
        top.addStretch(1)
        self.back = QPushButton("←")
        self.back.setFixedWidth(34)
        self.back.setToolTip("Previous explanation")
        self.back.clicked.connect(self._back)
        top.addWidget(self.back)
        x = QPushButton("✕")
        x.setFixedWidth(34)
        x.setToolTip("Close (Esc)")
        x.clicked.connect(self.hide_panel)
        top.addWidget(x)
        lay.addLayout(top)
        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._link)
        lay.addWidget(self.view, 1)
        hint = QLabel("Click any ⓘ label, tile or chart button to explain it. Right-click a table header or value.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        lay.addWidget(hint)
        self.history: list[tuple] = []
        self.hide()

    def show_entry(self, e: gl.Entry, value=None, ticker: str | None = None, record: bool = True) -> None:
        if record:
            self.history.append((e.key, value, ticker))
            self.history = self.history[-30:]
        self.view.setHtml(render_html(e, value, ticker))
        self.back.setEnabled(len(self.history) > 1)
        self.show()

    def _back(self) -> None:
        if len(self.history) > 1:
            self.history.pop()
            k, v, t = self.history[-1]
            self.show_entry(gl.G[k], v, t, record=False)

    def _link(self, url: QUrl) -> None:
        s = url.toString()
        if s.startswith("term:") and s[5:] in gl.G:
            self.show_entry(gl.G[s[5:]])

    def hide_panel(self) -> None:
        self.hide()
        self.closed.emit()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.hide_panel()
        else:
            super().keyPressEvent(ev)
