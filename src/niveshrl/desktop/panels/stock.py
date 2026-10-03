"""Stock panel: live price, watchlist controls, key stats, candlestick chart, news + FinBERT,
technicals, financial statements, ownership/analysts, model outputs and company profile."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ... import watchlist as wl
from ...research import fundamentals as fx
from ...research.technicals import TECH_COLUMNS
from .. import data, theme
from ..widgets import FrameTable, KpiRow, PriceChart, h1, muted, run_async
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
        for w in (self.wl_label, self.b_must, self.b_pref, self.b_rm):
            wrow.addWidget(w)
        wrow.addStretch(1)
        lay.addLayout(wrow)
        self.desk = KpiRow(cols=4)
        lay.addWidget(self.desk)
        self.stats = KpiRow(cols=4)
        self.tabs = QTabWidget()
        self.chart = PriceChart()
        self.tabs.addTab(self.chart, "Chart")
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.addWidget(self.stats)
        sl.addStretch(1)
        self.tabs.addTab(sw, "Key stats")
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
        self.about = QTextBrowser()
        self.about.setOpenExternalLinks(True)
        self.tabs.addTab(self.about, "About")
        self.tabs.setMinimumHeight(600)
        lay.addWidget(self.tabs, 1)
        self._set_wl_buttons()

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
        for t in self.stmts.values():
            t.set_frame(pd.DataFrame())
        self.stats.set_items([])
        tk = ticker
        run_async(data.fundamentals, lambda f, tk=tk: self._got_fund(tk, f), tk,
                  on_error=lambda e, tk=tk: self.sub.setText(f"Yahoo financials unavailable: {e.splitlines()[-1][:120]}"))

    def refresh(self) -> None:
        if self.ticker:
            self.show_stock(self.ticker)

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
        p = data.panel()
        t = self.ticker
        df = pd.DataFrame({"Close": p.close[t]})
        for k, src in [("Open", p.open), ("High", p.high), ("Low", p.low)]:
            df[k] = src[t] if src is not None and t in src else np.nan
        df["Volume"] = p.volume[t] if t in p.volume else np.nan
        self.chart.plot(df.iloc[-1500:], title=f"{t.replace('.NS', '')} · daily (adjusted) · SMA50 / SMA200 · volume · RSI(14)")

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
