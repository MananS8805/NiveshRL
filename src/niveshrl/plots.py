"""Report figures (matplotlib, static PNGs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import drawdown  # noqa: E402

# Fixed palette so a strategy keeps its colour in every figure.
PALETTE = ["#1E8A72", "#C1592E", "#3B6FB6", "#8E5CA8", "#D4A017", "#5B8C3A", "#B0457B", "#6B7B8C",
           "#2A9D8F", "#E76F51"]
BENCH_COLOR = "#16241F"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="600")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _colors(names):
    out, k = {}, 0
    for n in names:
        if n.startswith("NIFTY"):
            out[n] = BENCH_COLOR
        else:
            out[n] = PALETTE[k % len(PALETTE)]
            k += 1
    return out


def equity_curves(navs: dict[str, pd.Series], path: Path, title="Growth of Rs 1 (after costs)"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    col = _colors(navs)
    for name, nav in navs.items():
        rl = not name.startswith(("NIFTY", "Equal", "Momentum", "Risk", "Markowitz", "Minimum", "HRP"))
        ax.plot(nav.index, nav.values, label=name, color=col[name], lw=2.4 if rl else 1.3,
                ls="--" if name.startswith("NIFTY") else "-")
    ax.set_yscale("log")
    _style(ax, title)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def drawdowns(navs: dict[str, pd.Series], path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    col = _colors(navs)
    for name, nav in navs.items():
        ax.plot(nav.index, 100 * drawdown(nav), label=name, color=col[name], lw=1.3)
    ax.set_ylabel("drawdown (%)")
    _style(ax, "Drawdowns")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sector_allocation(trades: pd.DataFrame, tickers: list[str], sectors: list[str], vix: pd.Series,
                      path: Path, title="Agent's sector allocation vs India VIX"):
    w = trades[tickers + ["CASH"]]
    sec = w[tickers].T.groupby(pd.Series(sectors, index=tickers)).sum().T
    sec["Cash"] = w["CASH"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(sec.index, (100 * sec).T.values, labels=sec.columns,
                 colors=(PALETTE * 2)[: sec.shape[1]], alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_ylabel("weight (%)")
    ax2 = ax.twinx()
    v = vix.reindex(sec.index, method="ffill")
    ax2.plot(v.index, v.values, color=BENCH_COLOR, lw=1.2, label="India VIX")
    ax2.set_ylabel("India VIX")
    _style(ax, title)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.07, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_sensitivity(table: pd.DataFrame, path: Path):
    """table: index = cost scale, columns = strategy, values = Sharpe."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    col = _colors(table.columns)
    for name in table.columns:
        ax.plot(table.index, table[name], marker="o", label=name, color=col[name])
    ax.set_xlabel("cost multiplier (1.0 = actual NSE delivery charges)")
    ax.set_ylabel("Sharpe")
    _style(ax, "Sharpe vs transaction-cost level")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def frontier(points: pd.DataFrame, path: Path):
    """points: columns risk_aversion, Vol, CAGR, MaxDD (one row per lambda)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(100 * points["Vol"], 100 * points["CAGR"], c=points["risk_aversion"], cmap="viridis", s=60)
    ax.plot(100 * points["Vol"], 100 * points["CAGR"], color="#8B9691", lw=1, zorder=0)
    fig.colorbar(sc, label="risk aversion λ")
    ax.set_xlabel("annualised volatility (%)")
    ax.set_ylabel("CAGR (%)")
    _style(ax, "Risk-return frontier from ONE conditioned policy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_with_errors(df: pd.DataFrame, path: Path, title: str, ylabel="validation Sharpe"):
    """df: index = variant, columns mean, std."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df.index, df["mean"], yerr=df["std"], color=PALETTE[: len(df)], capsize=4)
    ax.set_ylabel(ylabel)
    _style(ax, title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(logs: dict[str, pd.DataFrame], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, log) in enumerate(logs.items()):
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(log["step"], log["reward"].rolling(10, min_periods=1).mean(), color=c, label=name)
        if "val_Sharpe" in log:
            v = log.dropna(subset=["val_Sharpe"])
            axes[1].plot(v["step"], v["val_Sharpe"], color=c, marker=".", label=name)
    _style(axes[0], "Training reward (smoothed)")
    _style(axes[1], "Validation Sharpe")
    axes[1].legend(frameon=False, fontsize=8)
    for a in axes:
        a.set_xlabel("env steps")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

