"""Terminal-style look for the dashboard: palette, CSS, Plotly template, KPI tiles."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

AMBER = "#FF9F1C"
GREEN = "#26D07C"
RED = "#FF4D4D"
BLUE = "#4DA3FF"
MUTED = "#8A94A6"
GRID = "#1C2430"
BG = "#07090C"
PANEL = "#10151C"
SERIES = [AMBER, BLUE, GREEN, "#C77DFF", "#FF6B9A", "#5EEAD4", "#FACC15", "#94A3B8", "#F97316", "#A3E635"]

CSS = f"""
<style>
.block-container {{padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%;}}
h1, h2, h3, h4 {{font-family: 'IBM Plex Mono', monospace !important; letter-spacing: .02em;}}
h1 {{color: {AMBER} !important; font-size: 1.35rem !important; text-transform: uppercase; margin-bottom: .2rem;}}
h2, h3 {{color: {AMBER} !important; font-size: 1.0rem !important; text-transform: uppercase;
        border-bottom: 1px solid {GRID}; padding-bottom: .25rem; margin-top: .6rem;}}
[data-testid="stSidebar"] {{background: #0B0F14; border-right: 1px solid {GRID};}}
[data-testid="stSidebar"] h1 {{font-size: 1.1rem !important;}}
.tape {{display: flex; gap: 1.6rem; flex-wrap: wrap; background: {PANEL}; border: 1px solid {GRID};
        border-left: 3px solid {AMBER}; padding: .45rem .8rem; font-family: monospace; font-size: .85rem;
        margin-bottom: .6rem;}}
.tape b {{color: {MUTED}; font-weight: 400; margin-right: .35rem;}}
.kpis {{display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: .45rem; margin: .3rem 0 .6rem;}}
.kpi {{background: {PANEL}; border: 1px solid {GRID}; padding: .45rem .6rem; font-family: monospace;}}
.kpi .l {{color: {MUTED}; font-size: .68rem; text-transform: uppercase; letter-spacing: .06em;}}
.kpi .v {{font-size: 1.15rem; color: #F2F2F2; margin-top: .1rem;}}
.kpi .s {{color: {MUTED}; font-size: .7rem;}}
.pos {{color: {GREEN} !important;}} .neg {{color: {RED} !important;}} .amb {{color: {AMBER} !important;}}
.note {{color: {MUTED}; font-size: .78rem; border-left: 2px solid {GRID}; padding-left: .6rem; margin: .3rem 0 .6rem;}}
.warn {{background: #1F1405; border: 1px solid #5A3A0A; color: #FFD8A0; padding: .5rem .7rem; font-size: .8rem;
        margin: .3rem 0 .7rem;}}
div[data-testid="stTabs"] button {{font-family: monospace; text-transform: uppercase; font-size: .8rem;}}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="IBM Plex Mono, monospace", color="#D8D8D8", size=11),
        colorway=SERIES, margin=dict(l=10, r=10, t=64, b=10), hovermode="x unified",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, showspikes=True, spikecolor=MUTED,
                   spikethickness=1, spikedash="dot", spikemode="across"),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, side="right"),
        # Title in the top margin, legend on its own row just above the plot area.
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        title=dict(font=dict(color=AMBER, size=12), x=0, xanchor="left", y=0.99, yanchor="top"),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=AMBER, font=dict(family="monospace")),
    )
    pio.templates["terminal"] = template
    pio.templates.default = "terminal"


def fmt(v, kind: str = "pct", digits: int = 1) -> str:
    if v is None or v != v:
        return "–"
    if kind == "pct":
        return f"{v * 100:+.{digits}f}%"
    if kind == "pct_plain":
        return f"{v * 100:.{digits}f}%"
    if kind == "x":
        return f"{v:.2f}"
    if kind == "inr":
        return f"₹{v:,.0f}"
    return f"{v:,.{digits}f}"


def _cls(v, signed: bool) -> str:
    if not signed or v is None or v != v:
        return ""
    return "pos" if v > 0 else "neg" if v < 0 else ""


def kpis(items: list[tuple]) -> None:
    """items: (label, value, kind, signed[, subtitle])."""
    cells = []
    for it in items:
        label, value, kind, signed = it[:4]
        sub = it[4] if len(it) > 4 else ""
        cells.append(f'<div class="kpi"><div class="l">{label}</div>'
                     f'<div class="v {_cls(value, signed)}">{fmt(value, kind)}</div>'
                     f'<div class="s">{sub}</div></div>')
    st.markdown(f'<div class="kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def tape(items: list[tuple[str, str, str]]) -> None:
    """Top status strip: (label, value, css class)."""
    parts = [f'<span><b>{k}</b><span class="{c}">{v}</span></span>' for k, v, c in items]
    st.markdown(f'<div class="tape">{"".join(parts)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="warn">{text}</div>', unsafe_allow_html=True)


def signed_color(v: float) -> str:
    return GREEN if v >= 0 else RED
