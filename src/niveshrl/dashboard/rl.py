"""RL allocator screens: strategy comparison (research) and the personalised investor plan,
plus the plain-language explainer. Moved from the original app.py; behaviour unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import about
from ..agents import TorchPolicy, load_run
from ..backtest import benchmark_nav, make_eval_env, run_policy
from ..baselines import EqualWeight, Markowitz, MinVariance
from ..explain import explain_change
from ..metrics import daily_returns, drawdown
from ..pipeline import load_all
from ..planner import goal_projection, orders_frame, plan_orders
from ..profile import PRESETS, QUESTIONNAIRE, from_answers
from .theme import AMBER as ACCENT, MUTED as INK, RED as FLAG

# Returns used for the goal projection. This stays on "val" until the final,
# one-time test evaluation is done; switching it earlier would leak test
# performance into development.
PROJECTION_SPLIT = "val"
DISCLAIMER = ("**Educational project, not investment advice.** NiveshRL is not registered with SEBI as a "
              "Research Analyst or Investment Adviser. Backtests use historical data, include survivorship "
              "bias (today's NIFTY constituents), and past performance does not predict future returns.")


@st.cache_resource
def data():
    return load_all()


@st.cache_resource
def model_for(run: str):
    cfg, md, fs = data()
    return load_run(run, md, fs, cfg)


@st.cache_data(show_spinner="Running backtest…")
def backtest(run: str, split: str, strategy: str, cost_scale: float, profile_key: str | None = None,
             profile_json: str | None = None):
    cfg, md, fs = data()
    from niveshrl.profile import InvestorProfile
    prof = InvestorProfile(**json.loads(profile_json)) if profile_json else (PRESETS[profile_key] if profile_key else None)
    if strategy == "agent":
        env = make_eval_env(md, fs, cfg, split, profile=prof, action_mode="weights", cost_scale=cost_scale)
        res = run_policy(TorchPolicy(model_for(run)), env, "NiveshRL")
    else:
        cls = {"Equal weight": EqualWeight, "Markowitz": Markowitz, "Min variance": MinVariance}[strategy]
        res = run_policy(cls(), make_eval_env(md, fs, cfg, split, cost_scale=cost_scale), strategy)
    return res.daily, res.trades, res.metrics


def runs_available() -> list[str]:
    return sorted(str(p.parent) for p in Path("runs").glob("*/best.pt") if (p.parent / "args.json").exists())


METRICS_VAL = Path("report/results/metrics_val.csv")


def about_page(runs: list[str]) -> None:
    """Plain-language explainer. Needs no data and no trained model."""
    st.header("How NiveshRL works, in plain words")
    st.info(DISCLAIMER)
    st.markdown(about.INTRO)
    st.graphviz_chart(about.PIPELINE_DOT, width="stretch")

    st.subheader("1. The problem")
    st.markdown(about.PROBLEM)
    st.subheader("2. What NiveshRL does")
    st.markdown(about.WHAT)
    st.subheader("3. How it learned: like practising a game")
    st.markdown(about.HOW)
    st.subheader("4. Why trading costs matter")
    st.markdown(about.COSTS)
    st.subheader("5. Made for you")
    st.markdown(about.PERSONAL)
    st.subheader("6. What you'll see in this app")
    c1, c2 = st.columns(2)
    c1.markdown(about.APP_GUIDE_INVESTOR)
    c2.markdown(about.APP_GUIDE_RESEARCH)
    if not runs:
        st.warning("No trained model is available yet, so Investor and Research mode can't run. "
                   "Train one first: `python scripts/train_custom.py`.")

    st.subheader("7. How has it done so far?")
    # Numbers are only ever read from the evaluation output, never written into the app.
    try:
        res = pd.read_csv(METRICS_VAL, index_col=0) if METRICS_VAL.exists() else None
    except Exception:
        res = None
    cols = [c for c in ["CAGR", "Sharpe", "MaxDD"] if res is not None and c in res.columns]
    if res is not None and cols:
        st.markdown("Results on the **2019–2020 validation years** (which include the COVID crash), after "
                    "all trading costs. The AI never practised on these years. *CAGR* is the average yearly "
                    "growth, *Sharpe* is growth per unit of wobble (higher is smoother), *MaxDD* is the worst "
                    "fall from a peak.")
        st.dataframe(res[cols].style.format({"CAGR": "{:.1%}", "Sharpe": "{:.2f}", "MaxDD": "{:.1%}"}, na_rep="–"),
                     width="stretch")
        st.caption(f"Read from `{METRICS_VAL.as_posix()}`. Past results, not a forecast.")
    else:
        st.markdown("The evaluation results are still being produced. Once they are ready they will "
                    "appear here, read directly from the evaluation output.")

    st.subheader("8. Honest limits")
    st.markdown(about.LIMITS)
    with st.expander("Glossary: the jargon in one line each"):
        for term, meaning in about.GLOSSARY.items():
            st.markdown(f"- **{term}**: {meaning}")




def _pick_run() -> str | None:
    runs = runs_available()
    if not runs:
        st.error("No trained RL model found. Train one first: `python scripts/train_custom.py`.")
        return None
    # Prefer the newest design (split policy, utility reward) when it exists.
    default = next((i for i, r in enumerate(runs) if Path(r).name == "verify_split_s0"), 0)
    return st.selectbox("RL checkpoint", runs, index=default)


def research_page() -> None:
    st.markdown("# RL allocator · 29 NIFTY 50 stocks, weekly")
    run = _pick_run()
    if run is None:
        return
    cfg, md, fs = data()
    c1, c2 = st.columns(2)
    split = c1.selectbox("Period", ["val", "test"], format_func=lambda s: {"val": "Validation 2019–20", "test": "Test 2021–25 (final report only)"}[s])
    cost = c2.slider("Transaction-cost multiplier (1.0 = real NSE delivery charges)", 0.0, 4.0, 1.0, 0.5)
    rows, fig = {}, go.Figure()
    for strat, color in [("agent", ACCENT), ("Equal weight", "#3B6FB6"), ("Markowitz", "#8E5CA8"), ("Min variance", "#D4A017")]:
        daily, trades, m = backtest(run, split, strat, cost, "aggressive")
        name = "NiveshRL" if strat == "agent" else strat
        rows[name] = m
        fig.add_trace(go.Scatter(x=daily.index, y=daily["nav"], name=name, line=dict(color=color, width=3 if strat == "agent" else 1.5)))
        if strat == "agent":
            agent_trades, agent_daily = trades, daily
    b = benchmark_nav(md, agent_daily.index)
    fig.add_trace(go.Scatter(x=b.index, y=b.values, name="NIFTY 50", line=dict(color=INK, dash="dash")))
    fig.update_layout(yaxis_type="log", height=420, margin=dict(l=10, r=10, t=30, b=10), title="Growth of ₹1 after costs")
    st.plotly_chart(fig, width="stretch")
    tbl = pd.DataFrame(rows).T[["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Turnover/yr", "Costs(Rs)"]]
    st.dataframe(tbl.style.format({"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                                   "MaxDD": "{:.1%}", "Turnover/yr": "{:.2f}", "Costs(Rs)": "₹{:,.0f}"}))
    st.subheader("How the agent allocated across sectors")
    w = agent_trades[md.tickers + ["CASH"]]
    sec = w[md.tickers].T.groupby(pd.Series(md.sectors, index=md.tickers)).sum().T
    sec["Cash"] = w["CASH"]
    area = go.Figure([go.Scatter(x=sec.index, y=100 * sec[c], name=c, stackgroup="one") for c in sec.columns])
    area.update_layout(height=380, yaxis_title="weight (%)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(area, width="stretch")



def investor_page() -> None:
    run = _pick_run()
    if run is None:
        return
    cfg, md, fs = data()
    st.markdown("# Investor plan · personalised")
    st.info(DISCLAIMER)
    with st.form("q"):
        ans = {}
        cols = st.columns(2)
        for i, q in enumerate(QUESTIONNAIRE):
            col = cols[i % 2]
            if q.get("numeric"):
                ans[q["key"]] = col.number_input(q["q"], min_value=0, value=50_000 if q["key"] == "initial" else 5_000, step=1_000)
            else:
                ans[q["key"]] = col.radio(q["q"], list(q["options"]), index=min(1, len(q["options"]) - 1))
        target = st.number_input("Goal amount (₹), optional", min_value=0, value=0, step=50_000)
        go_btn = st.form_submit_button("Build my plan")
    if not go_btn and "profile" not in st.session_state:
        return
    if go_btn:
        st.session_state.profile = from_answers(ans)
        st.session_state.target = target
    prof = st.session_state.profile
    st.caption(f"Profile → risk aversion {prof.risk_aversion:.1f}, drawdown tolerance {prof.dd_tol:.0%}, "
               f"horizon {prof.horizon_years:g} yrs, cash buffer {prof.min_cash:.0%}, up to {prof.max_stocks} stocks")

    model = model_for(run)
    env = make_eval_env(md, fs, cfg, "forward", profile=prof)
    env.reset(seed=0)
    env.t = env.hi  # latest available date; a new investor starts from cash
    obs = env.observation()
    w_old = env.current_weights()
    w_new = env.feasible(TorchPolicy(model)(env))
    prices = env.prices[env.t]
    orders, cash_left = plan_orders(w_new, md.tickers, prices, cash=prof.initial)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(f"Plan as of {env.dates[env.t].date()}")
        pie_w = pd.Series(w_new, index=[t.replace(".NS", "") for t in md.tickers] + ["Cash"])
        pie_w = pie_w[pie_w > 0.005]
        st.plotly_chart(go.Figure(go.Pie(labels=pie_w.index, values=pie_w.values, hole=0.5)).update_layout(
            height=360, margin=dict(l=0, r=0, t=0, b=0)), width="stretch")
    with c2:
        st.subheader("Orders (whole shares)")
        st.dataframe(orders_frame(orders), hide_index=True, width="stretch")
        st.caption(f"Cash left: ₹{cash_left:,.0f}. Each month's SIP goes to the most underweight stocks first, "
                   "so the plan rebalances without selling. That avoids STT on sales, DP charges and capital-gains tax.")

    st.subheader("Why these stocks?")
    for line in explain_change(model, obs, md.tickers, md.sectors, w_old, w_new, threshold=0.03)[:8]:
        st.markdown(f"- {line}")

    st.subheader("Where could this take you?")
    daily, _, _ = backtest(run, PROJECTION_SPLIT, "agent", 1.0, profile_json=json.dumps(
        {k: v for k, v in prof.to_dict().items() if k != "rebalance"} | {"sip_monthly": 0.0}))
    proj = goal_projection(daily_returns(daily["nav"]), prof.initial, prof.sip_monthly, prof.horizon_years,
                           target=st.session_state.target or None)
    x = proj["months"] / 12
    fan = go.Figure([
        go.Scatter(x=x, y=proj["p90"], line=dict(width=0), showlegend=False),
        go.Scatter(x=x, y=proj["p10"], fill="tonexty", fillcolor="rgba(30,138,114,0.18)", line=dict(width=0), name="10–90% range"),
        go.Scatter(x=x, y=proj["p50"], line=dict(color=ACCENT, width=3), name="median"),
        go.Scatter(x=x, y=proj["invested"], line=dict(color=INK, dash="dot"), name="amount invested"),
    ])
    fan.update_layout(height=380, xaxis_title="years", yaxis_title="₹", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fan, width="stretch")
    msg = f"Median outcome after {prof.horizon_years:g} years: **₹{proj['final_median']:,.0f}** (invested ₹{proj['invested'][-1]:,.0f})."
    if "p_target" in proj:
        msg += f" Chance of reaching ₹{st.session_state.target:,.0f}: **{proj['p_target']:.0%}**."
    period = {"val": "2019–20", "test": "2021–25"}[PROJECTION_SPLIT]
    st.markdown(msg + f" Resampled from the model's {period} backtest returns; real outcomes can be worse.")

    st.subheader("Stress test: the 2020 COVID crash")
    daily_v, _, _ = backtest(run, "val", "agent", 1.0, profile_json=json.dumps(
        {k: v for k, v in prof.to_dict().items() if k != "rebalance"} | {"sip_monthly": 0.0}))
    nav = daily_v["nav"].loc["2020-01-01":"2020-12-31"]
    nav = nav / nav.iloc[0]
    b = benchmark_nav(md, nav.index)
    crash = go.Figure([go.Scatter(x=nav.index, y=100 * drawdown(nav), name="your plan", line=dict(color=ACCENT)),
                       go.Scatter(x=b.index, y=100 * drawdown(b), name="NIFTY 50", line=dict(color=FLAG))])
    crash.update_layout(height=320, yaxis_title="fall from peak (%)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(crash, width="stretch")
    st.markdown(f"Worst fall in 2020: your plan **{drawdown(nav).min():.0%}** vs NIFTY **{drawdown(b).min():.0%}**.")
