"""Daily trading desk screens: TODAY (briefing + monitor list), SCRN (screener), WATCH (watchlist).

All read the daily pipeline's output (``research/daily.py``); the sidebar's refresh
button reruns it in a background thread.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import watchlist as wl
from ..research import daily, screener as scr
from . import live, stockcard, store, theme

_RUN = {"thread": None, "progress": ("", 0.0), "result": None}


@st.cache_data(ttl=60, show_spinner=False)
def _load(name: str, ran_at: str):
    return daily.load(name)


def dload(name: str):
    meta = daily.latest_meta()
    return None if meta is None else _load(name, meta["ran_at"])


@st.cache_data(ttl=60, show_spinner=False)
def _table(ran_at: str) -> pd.DataFrame:
    monthly = store.predictions("transformer")
    if monthly is None:
        monthly = store.predictions("ffnn")
    return scr.build_table(daily.latest_dir(), monthly)


def table() -> pd.DataFrame | None:
    meta = daily.latest_meta()
    return None if meta is None or daily.latest_dir() is None else _table(meta["ran_at"])


# --------------------------------------------------------------------------- refresh (sidebar)
def refresh_widget() -> None:
    meta = daily.latest_meta()
    st.sidebar.markdown("**Daily data**")
    if meta:
        bad = [k for k, v in meta["steps"].items() if not v.get("ok")]
        st.sidebar.caption(f"Trading day {meta['trading_day']} · ran {meta['ran_at'].replace('T', ' ')}"
                           + (f" · ⚠ {', '.join(bad)} failed" if bad else ""))
    else:
        st.sidebar.caption("No daily data yet.")
    th = _RUN["thread"]
    if th is not None and th.is_alive():
        step, frac = _RUN["progress"]
        st.sidebar.progress(min(frac, 1.0), text=f"Refreshing: {step}…")
    elif st.sidebar.button("⟳ Refresh today's data", width="stretch",
                           help="Prices, technicals, fundamentals, news, FinBERT sentiment, next-day model, briefing (~4-6 min)."):
        def work():
            _RUN["result"] = daily.run(progress=lambda s, f: _RUN.__setitem__("progress", (s, f)))
            st.cache_data.clear()
        _RUN["thread"] = threading.Thread(target=work, daemon=True, name="niveshrl-daily")
        _RUN["thread"].start()
        st.rerun()


def _no_data() -> None:
    st.info("No daily data yet. Click **⟳ Refresh today's data** in the sidebar, or run `python scripts/daily.py`.")


def _fmt_cols(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        kind = scr.COLUMNS.get(c, (c, "", "num"))[2]
        out[c] = {"pct": "{:+.1%}", "cr": "{:,.0f}", "flag": "{:.0f}"}.get(kind, "{:,.2f}")
    return out


# --------------------------------------------------------------------------- TODAY
def today_page() -> None:
    st.markdown("# Today · market briefing")
    b, mon, hab = dload("briefing"), dload("monitor"), dload("habits")
    if b is None:
        _no_data()
        return
    theme.tape([
        ("TRADING DAY", b["date"], "amb"), ("NIFTY", f"{b['nifty']:,.1f}", ""),
        ("1D", f"{b['nifty_1d']:+.2%}", "pos" if b["nifty_1d"] >= 0 else "neg"),
        ("YTD", f"{b['nifty_ytd']:+.2%}", "pos" if b["nifty_ytd"] >= 0 else "neg"),
        ("VIX", f"{b['vix']:.2f} ({b['vix_1d']:+.1%})", "neg" if b["vix_1d"] > 0 else "pos"),
        ("ADV/DEC", f"{b['advancers']}/{b['decliners']}", ""), ("REGIME", b.get("regime") or "n/a", "amb"),
        ("SENTIMENT", f"{b.get('market_sentiment', float('nan')):+.2f}", "pos" if b.get("market_sentiment", 0) > 0 else "neg"),
    ])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### What happened")
        for line in b["narrative"]:
            st.markdown(f"- {line}")
        mine = wl.load()
        if mine:
            t = table()
            st.markdown("### Your watchlist today")
            if t is not None:
                rows = t.reindex(list(mine))[["ret_1d", "prob_up", "sentiment_adj", "analyst_score"]] \
                    if "prob_up" in t else t.reindex(list(mine))[["ret_1d"]]
                rows.index = [f"{wl.TIERS[mine[x].tier][0]} {x.replace('.NS', '')}" for x in rows.index]
                st.dataframe(rows.style.format({"ret_1d": "{:+.2%}", "prob_up": "{:.0%}", "sentiment_adj": "{:+.2f}",
                                                "analyst_score": "{:.0f}"}, na_rep="–"), width="stretch")
    with c2:
        st.markdown("### Sectors (1D)")
        best, worst = b["best_sectors"], b["worst_sectors"]
        sec = pd.Series({**worst, **best}).sort_values()
        fig = go.Figure(go.Bar(x=sec.values * 100, y=[s[:22] for s in sec.index], orientation="h",
                               marker_color=[theme.signed_color(v) for v in sec.values]))
        fig.update_layout(height=260, yaxis=dict(side="left"), showlegend=False, title="Best and worst sectors (%)")
        st.plotly_chart(fig, width="stretch")
        if b.get("earnings_week"):
            st.markdown("### Results this week")
            st.dataframe(pd.Series(b["earnings_week"], name="date").rename(lambda x: x.replace(".NS", "")),
                         width="stretch")

    st.markdown("## Top stocks to monitor tomorrow")
    if mon is None:
        st.info("Monitor list not computed yet.")
    else:
        ytd = dload("nextday_ytd") or {}
        e = ytd.get("ensemble") or ytd.get("lgbm") or {}
        theme.warn(f"For <b>watching</b>, not blind buying. The next-day model's out-of-sample record this year: "
                   f"AUC {e.get('AUC', float('nan')):.3f}, top-10 hit rate {e.get('Top10 hit rate', float('nan')):.0%}, "
                   f"accuracy {e.get('Accuracy', float('nan')):.1%} (51-54% is a good result for next-day direction). "
                   "Daily turnover at Indian costs (~0.25% round trip) usually eats the edge.")
        n = st.slider("How many", 5, 10, 10, key="mon_n")
        up = mon[mon["list"] == "watch for strength"].head(n)
        dn = mon[mon["list"] == "watch for weakness"].head(n)
        cols = ["score", "prob", "reasons"]
        cu, cd = st.columns(2)
        for col, df, title in [(cu, up, "Watch for strength"), (cd, dn, "Watch for weakness")]:
            with col:
                st.markdown(f"### {title}")
                view = df[cols].rename(index=lambda x: x.replace(".NS", ""))
                ev = st.dataframe(view.style.format({"score": "{:+.2f}", "prob": "{:.0%}"}, na_rep="–"),
                                  width="stretch", on_select="rerun", selection_mode="single-row", key=f"mon_{title}")
                if ev.selection.rows:
                    st.session_state["desk_sel"] = df.index[ev.selection.rows[0]]
    if hab:
        st.markdown("## Market habits (NIFTY, measured)")
        h1, h2, h3 = st.columns(3)
        dow = pd.DataFrame(hab["day_of_week"]).T
        h1.markdown("**Day of week since " + hab["since"][:4] + "**")
        h1.dataframe(dow.style.format({"mean": "{:+.3%}", "hit": "{:.0%}"}), width="stretch")
        h2.markdown("**After big days**")
        h2.markdown(f"- After a +2% day ({hab['after_big_up']['days']}×): next day up "
                    f"**{hab['after_big_up']['next_day_up']:.0%}**, avg {hab['after_big_up']['next_day_mean']:+.2%}\n"
                    f"- After a −2% day ({hab['after_big_down']['days']}×): next day up "
                    f"**{hab['after_big_down']['next_day_up']:.0%}**, avg {hab['after_big_down']['next_day_mean']:+.2%}\n"
                    f"- After 3 up days: next day up **{hab['streak_up_3_next']:.0%}**; "
                    f"after 3 down days: **{hab['streak_dn_3_next']:.0%}**")
        if hab.get("intraday"):
            i = hab["intraday"]
            h3.markdown(f"**Intraday (last {i['days']} sessions)**")
            h3.markdown(f"- First hour sets the day's direction **{i['first_hour_sets_direction']:.0%}** of the time\n"
                        f"- Average day range **{i['avg_range']:.2%}**\n"
                        f"- High formed: " + ", ".join(f"{k} {v:.0%}" for k, v in i["high_formed"].items()) + "\n"
                        f"- Low formed: " + ", ".join(f"{k} {v:.0%}" for k, v in i["low_formed"].items()))
    _details_if_selected()


def _details_if_selected() -> None:
    sel = st.session_state.get("desk_sel")
    if sel:
        p = store.panel()
        f = live.feed(tuple(p.tickers))
        with st.container(border=True):
            if st.button("✕ Close details", key="desk_close"):
                st.session_state["desk_sel"] = None
                st.rerun()
            stockcard.render(sel, f.snapshot().get(sel))


# --------------------------------------------------------------------------- SCRN
def screener_page() -> None:
    st.markdown("# Screener")
    t = table()
    if t is None:
        _no_data()
        return
    names = list(scr.PRESETS)
    counts = {n: len(scr.apply(t, scr.PRESETS[n][1])) for n in names}
    # Open on the first preset that actually has matches today (a weak tape empties momentum screens).
    first = next((i for i, n in enumerate(names) if counts[n]), 0)
    c1, c2 = st.columns([1, 2])
    preset = c1.selectbox("Preset", ["(custom)"] + names, index=first + 1,
                          format_func=lambda n: n if n == "(custom)" else f"{n} ({counts[n]})")
    if preset != "(custom)":
        c2.caption(scr.PRESETS[preset][0])
    base = [] if preset == "(custom)" else list(scr.PRESETS[preset][1])
    labels = {k: f"{v[1]} · {v[0]}" for k, v in scr.COLUMNS.items() if k in t}
    with st.expander("Filters", expanded=preset == "(custom)"):
        n_extra = st.number_input("Extra filters", 0, 6, 0)
        extra = []
        for i in range(int(n_extra)):
            a, b, c = st.columns([3, 1, 2])
            col = a.selectbox("Column", list(labels), format_func=labels.get, key=f"f_col_{i}")
            op = b.selectbox("Op", list(scr.OPS), key=f"f_op_{i}")
            kind = scr.COLUMNS[col][2]
            val = c.number_input("Value" + (" (%)" if kind == "pct" else ""), value=0.0, key=f"f_val_{i}")
            extra.append((col, op, val / 100 if kind == "pct" else val))
    flt = base + extra
    res = scr.apply(t, flt)
    st.caption(f"{len(res)} of {len(t)} stocks match · " + (" AND ".join(
        f"{scr.COLUMNS[c][0]} {o} {v:.0%}" if scr.COLUMNS[c][2] == "pct" else f"{scr.COLUMNS[c][0]} {o} {v:g}"
        for c, o, v in flt) or "no filters"))
    default_cols = ["close", "ret_1d", "ret_1m", "rsi14", "vs_sma200", "from_52w_high", "vol_ratio", "trailingPE",
                    "returnOnEquity", "analyst_score", "sentiment_adj", "prob_up"]
    show = st.multiselect("Columns", [c for c in t.columns if c in scr.COLUMNS or c == "close"],
                          default=[c for c in default_cols if c in t], format_func=lambda c: scr.COLUMNS.get(c, (c,))[0])
    if not len(res):
        st.info("No stocks match these filters today. Loosen a condition or pick another preset.")
        return
    view = res[["sector"] + show].rename(index=lambda x: x.replace(".NS", ""))
    view.columns = ["Sector"] + [scr.COLUMNS.get(c, (c,))[0] for c in show]
    fmt = {scr.COLUMNS.get(c, (c,))[0]: v for c, v in _fmt_cols(res[show]).items()}
    ev = st.dataframe(view.style.format(fmt, na_rep="–"), width="stretch", height=480, on_select="rerun",
                      selection_mode="single-row", key="scr_table")
    st.download_button("⬇ CSV", res.to_csv().encode(), "niveshrl_screener.csv", "text/csv")
    if ev.selection.rows:
        st.session_state["desk_sel"] = res.index[ev.selection.rows[0]]
    _details_if_selected()


# --------------------------------------------------------------------------- WATCH
def watch_page() -> None:
    st.markdown("# Watchlist")
    p = store.panel()
    items = wl.load()
    with st.form("wl_add", clear_on_submit=True):
        a, b, c, d, e = st.columns([3, 2, 2, 2, 1])
        tk = a.selectbox("Stock", sorted(p.tickers), format_func=lambda t: f"{t.replace('.NS', '')} · {str(p.names.get(t, ''))[:26]}")
        tier = b.radio("Tier", list(wl.TIERS), format_func=wl.TIERS.get, horizontal=True)
        tb = c.number_input("Buy target ₹ (optional)", min_value=0.0, value=0.0)
        ts = d.number_input("Sell target ₹ (optional)", min_value=0.0, value=0.0)
        note = st.text_input("Note", "")
        if e.form_submit_button("＋ Add"):
            wl.add(tk, tier, note, tb or None, ts or None)
            st.rerun()
    if not items:
        st.info("Your watchlist is empty. Add stocks above, or from the screener / stock details.")
        return
    f = live.feed(tuple(p.tickers))
    snap = f.snapshot()
    t = table()
    prev = None
    folders = sorted(x for x in daily.DAILY.glob("20*") if x.is_dir())
    if len(folders) >= 2 and (folders[-2] / "sentiment.parquet").exists():
        prev = pd.read_parquet(folders[-2] / "sentiment.parquet")
    mon = dload("monitor")
    top_now = set(mon[mon["list"] == "watch for strength"].index) if mon is not None else set()
    rows = []
    for tkr, it in sorted(items.items(), key=lambda kv: (kv[1].tier != "must", kv[0])):
        tick = snap.get(tkr)
        price = tick["price"] if tick else float(p.close[tkr].dropna().iloc[-1])
        chg = (tick["change_percent"] / 100) if tick and tick.get("change_percent") is not None else \
            (t.loc[tkr, "ret_1d"] if t is not None and tkr in t.index else None)
        r = t.loc[tkr] if t is not None and tkr in t.index else pd.Series(dtype=float)
        alerts = wl.alerts(it, price, chg, r.get("sentiment"),
                           prev["sentiment"].get(tkr) if prev is not None else None,
                           int(r["days_to_earnings"]) if pd.notna(r.get("days_to_earnings")) else None,
                           tkr in top_now, None)
        rows.append({"Tier": wl.TIERS[it.tier][0], "Stock": tkr.replace(".NS", ""), "Price": price, "1D": chg,
                     "Live": "●" if tick else "", "P(up) 1D": r.get("prob_up"), "Sentiment": r.get("sentiment_adj"),
                     "Analyst": r.get("analyst_score"), "Results in": r.get("days_to_earnings"),
                     "Buy ≤": it.target_buy, "Sell ≥": it.target_sell, "Alerts": " · ".join(alerts), "Note": it.note,
                     "_t": tkr})
    df = pd.DataFrame(rows)
    ev = st.dataframe(df.drop(columns="_t").style.format(
        {"Price": "₹{:,.2f}", "1D": "{:+.2%}", "P(up) 1D": "{:.0%}", "Sentiment": "{:+.2f}", "Analyst": "{:.0f}",
         "Results in": "{:.0f}d", "Buy ≤": "₹{:,.0f}", "Sell ≥": "₹{:,.0f}"}, na_rep="–")
        .map(lambda v: f"color: {theme.AMBER}" if isinstance(v, str) and v else "", subset=["Alerts"]),
        width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", key="wl_table")
    c1, c2 = st.columns([1, 5])
    if ev.selection.rows:
        tkr = df.iloc[ev.selection.rows[0]]["_t"]
        st.session_state["desk_sel"] = tkr
        if c1.button(f"Remove {tkr.replace('.NS', '')}"):
            wl.remove(tkr)
            st.session_state["desk_sel"] = None
            st.rerun()
    _details_if_selected()
