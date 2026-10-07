"""Every model in NiveshRL, why it exists and how well it measured: the source for the Models screen and the README's
"How the models work" section.

Each entry says what job the model does, its architecture, *why that architecture fits the job*, its inputs, how it is
trained (window, refit schedule), where its output appears in the app, its limits, and its status (live / shadow /
research only / rejected). ``record(key)`` reads the measured out-of-sample result from report/results/ every time, so
the numbers shown are never typed in by hand and never go stale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import ROOT

RES = ROOT / "report" / "results"


@dataclass
class ModelCard:
    key: str
    name: str
    family: str                 # Deep learning | Machine learning | Statistical | Reinforcement learning | NLP | Rules
    job: str
    architecture: str
    why: str
    inputs: str
    training: str
    used_in: str
    limits: str
    status: str                 # live | shadow | research | rejected
    record_src: tuple | None = None      # (csv file, row label, [metric columns])
    feeds: list = field(default_factory=list)


def _r(f, row, cols):
    return (f, row, cols)


CARDS: list[ModelCard] = [
    ModelCard("ffnn", "FFNN ranker (Takeuchi & Lee)", "Deep learning",
              "Rank all NIFTY 200 stocks by the chance of beating the median over the next month.",
              "Feed-forward net 33 → 40 → 4 → 50 → 2 with a 4-unit bottleneck, dropout, early stopping.",
              "Monthly cross-sectional ranking from 33 return features is a tabular problem with weak, nonlinear "
              "signal; a small bottlenecked MLP captures interactions without the data a sequence model needs. It is "
              "the classic deep-learning momentum paper's design, so it doubles as a reproducible baseline.",
              "12 monthly + 20 daily cumulative returns, January flag (cross-sectionally ranked).",
              "Walk-forward, rolling 8-year window, refit yearly (by hand).", "Rankers, Backtest lab, stock page models tab",
              "IC is real but monthly turnover costs eat it: after costs it does not beat equal weight on point-in-time data.",
              "live", _r("rankers_summary_pit.csv", "ffnn", ["IC mean", "IC t-stat", "Spread / mo"]), ["Backtest lab"]),
    ModelCard("lstm", "LSTM ranker", "Deep learning",
              "Same monthly ranking job, from the sequence of the last 60 daily returns.",
              "1-layer LSTM (hidden 32) over 60 days of returns + static features → probability.",
              "Tests whether the *order* of recent returns (not just their sums) carries information; LSTMs are the "
              "standard recurrent baseline for that question.",
              "60 daily returns per stock + the FFNN's static features.", "Walk-forward, rolling 8 years, yearly refit.",
              "Rankers", "No better than the FFNN after costs; slower to train.", "live",
              _r("rankers_summary_pit.csv", "lstm", ["IC mean", "IC t-stat", "Spread / mo"])),
    ModelCard("transformer", "Transformer ranker", "Deep learning",
              "Same monthly ranking job with attention across time.",
              "2-layer Transformer encoder (d = 32, 4 heads) over 60 daily return tokens.",
              "Attention can weight a few important days (earnings gaps, crashes) more than an LSTM's fading memory; "
              "included to measure whether that matters here.",
              "60 daily returns + static features.", "Walk-forward, rolling 8 years, yearly refit.", "Rankers",
              "Similar IC to the FFNN at ~15× the training time.", "live",
              _r("rankers_summary_pit.csv", "transformer", ["IC mean", "IC t-stat", "Spread / mo"])),
    ModelCard("nextday_lgbm", "Next-day model (LightGBM)", "Machine learning",
              "P(each stock beats tomorrow's median return): what to watch tomorrow.",
              "Gradient-boosted trees (600 trees, early stopping), isotonic calibration.",
              "~40 engineered daily features with interactions and outliers: boosted trees are the strongest, fastest "
              "learner for tabular data of this size and are robust to feature scaling.",
              "Returns, gaps, ranges, RSI, volume, VIX, breadth, day of week (cross-sectionally ranked).",
              "Walk-forward, train Y-5..Y-2, calibrate Y-1, refit yearly (monthly refit measured worse: kept yearly).",
              "Today (watch tomorrow, P(up)), screener, stock page", "Top-10 picks lose after costs; use for watching, not trading.",
              "live", _r("nextday_summary_pit.csv", "lgbm", ["AUC", "IC mean", "IC t-stat", "Top10 net of costs / day"])),
    ModelCard("nextday_seq", "Next-day sequence net", "Deep learning",
              "Same next-day job from the shape of the last 20 days.",
              "Window convolutions → 2-layer Transformer encoder → MLP with the tabular features, isotonic calibration.",
              "Tests whether multi-day *patterns* (not just today's numbers) add to the tabular model; convolution "
              "finds local shapes, attention relates them.",
              "20 days × 8 daily features + the tabular row.", "Walk-forward yearly; in the stack refit quarterly (slow on CPU).",
              "Stacked model", "Weaker than LightGBM alone (IC 0.050 vs 0.058); ~30 min per fit on this CPU.", "research",
              _r("nextday_summary_pit.csv", "seq", ["AUC", "IC mean", "IC t-stat"])),
    ModelCard("stacked", "Stacked next-day model (4 models)", "Machine learning",
              "One P(up) per stock each evening from LightGBM, the sequence net, logistic regression and the range model, "
              "plus pattern changes (Up/Down × quiet/volatile).",
              "Logistic meta-model over the base models' daily ranks, trained on their previous 12 months of out-of-sample predictions.",
              "Stacking lets models that see different things vote with learned weights; training the meta-model only "
              "on out-of-sample predictions prevents it trusting a base model's in-sample overconfidence.",
              "The four base models' predictions.", "Base models monthly (sequence net quarterly); meta model monthly.",
              "Today → Pattern changes", "Measured: no edge over LightGBM alone (IC 0.035 vs 0.042).", "live",
              _r("refit_comparison.csv", "stacked (4 models)", ["AUC", "IC mean", "IC t-stat"])),
    ModelCard("range", "Next-day range model", "Machine learning",
              "How big tomorrow's high-low range will be (who will move, not which way).",
              "LightGBM regression on 13 range/volatility features.",
              "Volatility clusters and is far more predictable than direction; trees handle the skewed, interacting "
              "range features (NR7, gaps, ATR) well.",
              "Recent ranges, ATR, gap and move size, volume ratio, NR7/NR4, 60-day vol, VIX.",
              "Rolling 5 years up to yesterday, refit daily.", "Today → Who will move, stacked model",
              "Predicts size only.", "live", _r("range_summary.csv", "model", ["IC mean", "IC t-stat", "Top 20 next-day range"])),
    ModelCard("vol_lstm", "Volatility forecaster (LSTM)", "Deep learning",
              "Next month's volatility per stock (risk sizing, the volatility cone).",
              "LSTM (hidden 32) over 60 days of r and |r| + log realised vols, VIX, NIFTY vol.",
              "Volatility has long memory and asymmetric reactions; an LSTM learns that from data across all stocks "
              "at once, where GARCH fits each stock alone with a fixed form.",
              "60 daily returns and absolute returns, 21/63-day realised vol, VIX, NIFTY 20-day vol.",
              "Rolling 8 years, refit monthly (measured slightly better than yearly), forecast daily.",
              "Stock page Risk tab (cone), Risk screen", "Underreacts to sudden shocks.", "live",
              _r("vol_summary.csv", "lstm", ["RMSE log-vol", "QLIKE", "Corr"])),
    ModelCard("garch", "GARCH(1,1)", "Statistical",
              "The textbook volatility benchmark the LSTM must beat.",
              "Constant-mean GARCH(1,1) per stock (arch package).",
              "The industry-standard volatility model: if the LSTM cannot beat it, the LSTM is not needed.",
              "Daily returns of one stock.", "4-year fits, yearly.", "Risk screen comparison",
              "Worse than the LSTM and even EWMA here.", "research", _r("vol_summary.csv", "garch", ["RMSE log-vol", "QLIKE", "Corr"])),
    ModelCard("regimes", "Market regimes (autoencoder + k-means)", "Deep learning",
              "Label each week Bull / Neutral / Stress (risk state, briefing).",
              "Autoencoder compresses 8 market features to 2 numbers; k-means finds 3 clusters, named by their volatility and return.",
              "Unsupervised: there are no labels for 'regime'. The autoencoder removes redundancy between correlated "
              "market measures so clusters are not dominated by one of them.",
              "NIFTY returns and volatility, VIX, breadth, dispersion, drawdown (each relative to its 1-year median).",
              "Expanding window since 2005, yearly (rolling 8 years measured worse); relabelled weekly by the pipeline.",
              "Risk state, Today, briefing", "Labels describe volatility, not direction.", "live",
              _r("regime_stats.csv", "Stress", ["weeks", "next-month NIFTY vol"])),
    ModelCard("finbert", "News sentiment (FinBERT)", "NLP",
              "Tone of the last 48 h of headlines per stock.",
              "FinBERT (BERT fine-tuned on financial text, ProsusAI/finbert).",
              "A pretrained finance language model understands 'misses estimates' or 'downgrade' without training data "
              "of our own; general sentiment models do not.",
              "Google News headlines.", "Pretrained, not retrained.", "News tabs, screener, monitor list",
              "Whether tone predicts returns can only be measured forward (no headline history).", "live"),
    ModelCard("rl", "RL allocator (PPO)", "Reinforcement learning",
              "How a given investor should split money across 29 NIFTY 50 stocks and cash, weekly.",
              "PPO actor-critic; permutation-equivariant Transformer encoder; Beta (equity share) × Dirichlet (stock mix) policy.",
              "Allocation is a sequential decision with costs and taxes: RL optimises the investor's utility directly "
              "instead of a one-step forecast; the equivariant encoder scores every stock with the same weights.",
              "Causal price features, investor profile.", "Trained 2008-18, validated after.", "RL & plan",
              "Does not beat simple baselines (equal weight, minimum variance) after costs on validation.", "research",
              _r("metrics_val.csv", "NiveshRL", ["CAGR", "Sharpe", "MaxDD"])),
    ModelCard("bandit", "Intraday bandit (tabular)", "Reinforcement learning",
              "TAKE / HALF / SKIP each rule signal by its context's track record.",
              "Context buckets (setup × side × time × trend) with t-tests and Thompson draws; 20-day forgetting.",
              "Few outcomes per context: simple, transparent statistics that only act on clear evidence.",
              "Every signal's shadow outcome after costs.", "Daily, after the close.", "Intraday agent → Learning",
              "Can only filter; it cannot create a good signal from losing setups.", "live",
              _r("forgetting_comparison.csv", "half-life 20 days", ["trades", "avg R", "net ₹"])),
    ModelCard("scorer", "Intraday ML scorer (LightGBM)", "Machine learning",
              "P(a rule signal ends in profit after costs).", "LightGBM classifier on 12 signal features, recency-weighted.",
              "Fast, works from a few hundred outcomes.", "Signal-bar features (time, gap, volume, VWAP distance, stop size…).",
              "Retrained daily on all shadow outcomes (20-day half-life).", "Intraday agent", "Same limit as the bandit.", "live",
              _r("forgetting_comparison.csv", "half-life 20 days", ["trades", "avg R", "net ₹"])),
    ModelCard("tcn", "Intraday meta-labeler (TCN)", "Deep learning",
              "P(profit) and expected R after costs for a long and a short at every 5-minute bar of every liquid stock.",
              "Temporal convolutional network: 4 dilated causal residual blocks (1, 2, 4, 8 bars) + context MLP; "
              "multi-task heads; isotonic calibration (~25k parameters).",
              "Intraday decisions depend on the shape of the last two hours at several time scales; dilated causal "
              "convolutions see them all, never look ahead, and train in minutes on a CPU. Triple-barrier labels for "
              "every bar give millions of examples instead of a few thousand rule signals.",
              "24 bars × 10 channels (returns, range, VWAP distance, relative volume, NIFTY…) + 10 context numbers.",
              "Walk-forward by day, refit weekly on all earlier days.", "Intraday agent (v2)",
              "~60 days of 5-minute history (growing daily); no order book.", "shadow",
              _r("intraday_dl.csv", "TCN, in-play stocks", ["IC mean", "IC t", "top5/day avg R"])),
    ModelCard("tcn_lgbm", "Intraday baseline (LightGBM on the same inputs)", "Machine learning",
              "The benchmark the TCN must beat on the same labels.", "LightGBM regression on the context + summarised bars.",
              "If a tree model on summaries does as well, the deep network is not earning its complexity.",
              "Same as the TCN, flattened.", "Same walk-forward.", "Comparison only", "—", "research",
              _r("intraday_dl.csv", "LightGBM, in-play stocks", ["IC mean", "IC t", "top5/day avg R"])),
    ModelCard("neural_bandit", "Neural-linear Thompson bandit", "Reinforcement learning",
              "Decide TAKE / HALF / SKIP among the meta-labeler's candidates and learn from every outcome.",
              "Bayesian linear regression on the network's outputs with Thompson sampling and a 20-day half-life.",
              "Keeps a posterior instead of point estimates, so it explores when unsure and generalises across similar "
              "situations; the standard practical deep contextual bandit.",
              "TCN predictions per candidate.", "Updated after every close.", "Intraday agent (v2)",
              "Only as good as the features it sits on.", "shadow",
              _r("intraday_v2.csv", "model + neural bandit", ["trades", "avg R", "total R"])),
    ModelCard("conformal", "Conformal abstention", "Statistical",
              "Trade only when the predicted R stays positive after the model's typical over-optimism.",
              "Split conformal prediction on a rolling 10-day window of out-of-sample errors (80% quantile).",
              "Distribution-free and calibrated by construction: a principled 'not sure enough, skip'.",
              "Predicted and realised R of recent candidates.", "Rolling daily.", "Intraday agent (v2)",
              "With noisy intraday R the margin is large, so it trades rarely.", "shadow",
              _r("intraday_v2.csv", "model + conformal abstention", ["trades", "avg R", "total R"])),
    ModelCard("iql", "Learned exits (implicit Q-learning)", "Reinforcement learning",
              "When to close an open intraday trade instead of a fixed 2R target / stop.",
              "Offline RL: fitted Q-iteration with expectile value regression (IQL) over logged price paths; actions hold / exit.",
              "Learns from history alone without risky live exploration; expectile regression keeps it conservative "
              "about situations the data rarely shows.",
              "Per bar of an open trade: R so far, bars held, VWAP distance, recent returns, time to close.",
              "Trained on earlier days' paths, evaluated on later ones.", "Intraday agent (v2)",
              "Short history; exits only, the hard stop stays.", "shadow",
              _r("intraday_v2.csv", "model top + learned exits (IQL), 2nd half", ["trades", "avg R", "diff vs fixed"])),
    ModelCard("swing_meta", "Swing pick meta-labeler (LightGBM + MLP)", "Deep learning",
              "Predict each swing pick's R to keep only the best half of the daily list.",
              "LightGBM + 2×64 MLP averaged, on 40 point-in-time features incl. every research model's view.",
              "Meta-labeling: a second model decides which of the primary model's picks to act on.",
              "Stock and market state at the signal date, model ranks, context tags.", "Walk-forward by year (2017-2026).",
              "Track record (comparison)", "Measured: no improvement (rank correlation −0.03): the list is kept as is.",
              "rejected", _r("swing_meta.csv", "best 50% by meta-label", ["trades", "avg R", "t (per signal day)"])),
    ModelCard("orb", "Opening-range breakout (Zarattini, Barbon & Aziz 2024)", "Rules",
              "Published 'profitable day-trading strategy': trade the direction of the first 5-minute bar in stocks in play.",
              "Rule: first 5-minute bar up → long at the second bar's open (down → short), stop 10% of daily ATR, exit at the close.",
              "The best-known recent academic day-trading result; tested here because it is the industry's reference ORB.",
              "5-minute bars; stocks already in play at the first bar (top 20 a day).", "No training (fixed rules).",
              "Comparison only", "Paper: +0.13R/trade gross on US stocks without spread/slippage; independent replication "
              "found net ≈ 0. Here, after NSE costs: slightly negative.", "rejected",
              _r("industry_rules.csv", "Opening-range breakout, stocks in play (Zarattini 2024)", ["trades", "avg R / trade", "t-stat"])),
    ModelCard("vwap_rule", "VWAP trend (Zarattini & Aziz 2023)", "Rules",
              "Long above the session VWAP, short below, re-checked every bar.",
              "Rule on NIFTY 5-minute bars (index level as a stand-in for NIFTYBEES / futures).",
              "Marketed as the 'holy grail' of day trading on QQQ; tested for NSE.", "NIFTY 5-minute bars.", "No training.",
              "Comparison only", "Many round trips a day: costs dominate.", "rejected",
              _r("industry_rules.csv", "VWAP trend on NIFTY (Zarattini & Aziz 2023)", ["trades", "avg R / trade", "net ₹ (₹1L pool)"])),
    ModelCard("turtle", "Turtle / Donchian 55-20 breakout", "Rules",
              "Classic trend following: buy a close above the 55-day high, sell a close below the 20-day low.",
              "Rule on daily closes, 10 equal slots, point-in-time NIFTY 200 members, 0.15% per side costs.",
              "The most famous trend-following rule set (Turtle traders); trend following has a century of positive "
              "evidence across asset classes (Hurst, Ooi & Pedersen).", "Daily prices.", "No training.",
              "Comparison only", "Positive and beats NIFTY, but below equal weight and below 12-1 momentum (19.3%/yr).",
              "research", _r("industry_rules.csv", "Turtle / Donchian 55-20 breakout", ["CAGR", "Sharpe", "max drawdown"])),
    ModelCard("rsi2", "Connors RSI(2) mean reversion", "Rules",
              "Buy short dips in uptrends: close > 200-day average and 2-day RSI < 10; sell when close > 5-day average.",
              "Rule on daily closes, 10 slots, point-in-time members, 0.15% per side.",
              "One of the most popular published short-term rules.", "Daily prices.", "No training.", "Comparison only",
              "High win rate but tiny average gains: costs and crash drawdowns erase them.", "rejected",
              _r("industry_rules.csv", "Connors RSI(2) mean reversion", ["CAGR", "Sharpe", "max drawdown"])),
]


def record(card: ModelCard) -> str:
    """'metric value · metric value …' from the result file, or a plain note when there is none yet."""
    if card.record_src is None:
        return "no out-of-sample record (see limits)"
    f, row, cols = card.record_src
    p = RES / f
    if not p.exists():
        return f"not measured yet ({f} missing)"
    try:
        t = pd.read_csv(p, index_col=0)
        r = t.loc[row]
    except (KeyError, ValueError, OSError):
        return f"not measured yet ({row} not in {f})"
    parts = []
    for c in cols:
        v = r.get(c)
        if v is None or v != v:
            continue
        v = float(v)
        signed = any(k in c for k in ("R", "IC", "Spread", "diff", "net", "CAGR", "MaxDD", "Corr"))
        if abs(v) >= 100:
            parts.append(f"{c} {v:,.0f}")
        elif "t" == c.split()[0] or "t-stat" in c or c.endswith(" t"):
            parts.append(f"{c} {v:.2f}")
        else:
            parts.append(f"{c} {v:+.3f}" if signed else f"{c} {v:.3f}")
    return " · ".join(parts) or "—"


def table() -> pd.DataFrame:
    return pd.DataFrame([{"Model": c.name, "Family": c.family, "Status": c.status, "Job": c.job,
                          "Measured record": record(c), "Used in": c.used_in} for c in CARDS]).set_index("Model")


def markdown() -> str:
    """The README section."""
    out = ["## How the models work (and whether each one earns its place)", "",
           "Generated from `src/niveshrl/model_registry.py` by `python scripts/models_doc.py`; the records are read "
           "from `report/results/`, all out of sample.", "",
           "| Model | Family | Status | Job | Measured record |", "| --- | --- | --- | --- | --- |"]
    for c in CARDS:
        out.append(f"| {c.name} | {c.family} | {c.status} | {c.job} | {record(c)} |")
    out.append("")
    for c in CARDS:
        out += [f"### {c.name}", f"- **Job:** {c.job}", f"- **Architecture:** {c.architecture}",
                f"- **Why this architecture:** {c.why}", f"- **Inputs:** {c.inputs}", f"- **Training:** {c.training}",
                f"- **Where it shows:** {c.used_in}", f"- **Limits:** {c.limits}", f"- **Measured:** {record(c)}", ""]
    return "\n".join(out)
