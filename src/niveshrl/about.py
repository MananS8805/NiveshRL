"""Plain-language copy for the app's "How it works" page.

Kept out of app.py so the dashboard code stays readable. Everything here is
static text. The page reads performance numbers only at runtime, from
report/results/metrics_val.csv, and never hardcodes a result for the agent.
"""
from __future__ import annotations

# 29 stocks x Rs 15.93 DP charge per sale x 52 weeks = Rs 24,022 a year.
N_STOCKS, DP_CHARGE, WEEKS = 29, 15.93, 52
DP_YEARLY = N_STOCKS * DP_CHARGE * WEEKS

INTRO = """
Think of NiveshRL as a **practice-trained assistant for splitting your money across big Indian companies**.
This page explains, without the jargon, what it does, how it learned, and where it can go wrong.
"""

PROBLEM = """
Deciding **how much money to put in which stocks, and when to adjust**, is hard. New investors often put
everything into one or two stocks they have heard of, or panic and sell everything when the market crashes,
which locks in the loss just before the recovery.
"""

WHAT = """
NiveshRL is an AI trained on about **17 years of real NSE price history (2008 onwards)**. It practised on the
older years (2008–2018) and was then checked on later years it never saw during practice. From that practice it
learned how to split money across **29 large NIFTY 50 companies plus cash**, and when to *rebalance* (nudge the
split back into shape as prices move).
"""

HOW = """
It learned the way you learn a video game: **by trial and error**.

1. It looks at recent prices and picks a split, e.g. "8% Infosys, 6% HDFC Bank, 10% cash…".
2. The simulator moves forward one week using what the market *actually* did.
3. The AI gets a **score**. The score is not just "did you make money". It rewards **steady** growth and
   takes points off for **big falls**, because a portfolio that drops 40% is painful even if it recovers later.
4. Repeat, over hundreds of thousands of practice weeks, until it gets consistently better scores.

Nobody told it rules like "buy banks when…". It worked out its own habits from the score.
"""

COSTS = f"""
Every trade in India costs money: **STT** (securities transaction tax), **stamp duty**, exchange fees,
**GST** on the broker's charges, and a flat **DP charge of about ₹16 each time you sell a stock** from your
demat account. The practice simulator charges all of these, in rupees, the way a real contract note would.

**Why that matters, with numbers:** if you sold a little of all {N_STOCKS} stocks every week, the DP charge alone is
{N_STOCKS} × ₹{DP_CHARGE:.2f} × {WEEKS} weeks ≈ **₹{DP_YEARLY:,.0f} a year**, almost 5% of a ₹5 lakh account,
gone before any tax or other charges. Because the AI pays these costs in every practice run, it learns
**not to over-trade**.
"""

PERSONAL = """
The **six questions** in Investor mode turn into a simple profile: how comfortable you are with ups and
downs, when you need the money, and how much you can add each month through a **SIP** (systematic investment
plan, a fixed monthly top-up).

The **same AI** reads that profile as an input, so it gives a *cautious* plan to a cautious investor and a
*bolder* plan to someone with a long horizon, without being retrained for each person.

On top of that there are **hard safety limits the AI can never break**, however confident it is:
- at most **10% in any one stock**,
- at most **30% in any one sector** (e.g. banking),
- always a **cash cushion**, bigger for cautious profiles.

If the AI suggests something outside these limits, the app mathematically pulls it back inside before you see it.
"""

APP_GUIDE_INVESTOR = """
**Investor mode**
- A plan in **whole shares** (no fractions), with the cash left over.
- **Why each stock?** Short reasons for the biggest picks.
- **Goal chart**: a fan of possible outcomes for your money over your timeline.
- **2020 crash replay**: how a plan like yours fell and recovered during COVID, next to the NIFTY 50.
"""

APP_GUIDE_RESEARCH = """
**Research mode**
- The AI side by side with simple strategies, such as **equal weight** (the same amount in every stock)
  and the **NIFTY 50 index** itself.
- A slider to make trading costs cheaper or dearer and see who copes best.
- How the AI spread money across sectors over time.
"""

LIMITS = """
- This is a **student research project**, not a product.
- All results come from **past data**. Markets change, and past returns do not predict future ones.
- **Survivorship bias**, in plain words: we only tested on companies that are big *today*. Companies that
  shrank or collapsed over the years are missing, which makes the past look rosier than it really was
  for someone investing at the time.
- There are **no guarantees**. The plan can lose money.
- NiveshRL is **not SEBI-registered** and this is **not investment advice**. Talk to a registered adviser
  before investing real money.
"""

GLOSSARY = {
    "Sharpe ratio": "Return earned per unit of up-and-down wobble. Higher means smoother growth for the same return.",
    "Drawdown": "How far the portfolio has fallen from its previous peak, e.g. −30% means it is 30% below its best value so far.",
    "Rebalancing": "Buying or selling a little to bring your split back to the plan after prices have moved it.",
    "SIP": "Systematic Investment Plan: investing a fixed amount every month instead of all at once.",
    "NAV": "Net Asset Value: what the whole portfolio is worth right now, holdings plus cash.",
    "CAGR": "Compound annual growth rate: the steady yearly growth that would get you from start to end value.",
}

# Plain-words pipeline diagram (Graphviz DOT). Colours match app.py's ACCENT/FLAG/INK.
PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
        color="#1E8A72", fillcolor="#DCEEE8", fontcolor="#16241F", margin="0.18,0.10"];
  edge [color="#16241F", arrowsize=0.8];
  data  [label="Real NSE prices\\n2008 onwards"];
  sim   [label="Practice simulator\\nwith real Indian costs"];
  learn [label="AI learns by\\ntrial and error"];
  you   [label="Your 6 answers\\n(risk, goal, SIP)", color="#C1592E", fillcolor="#F4E1D6"];
  safe  [label="Safety limits\\n10% stock · 30% sector · cash"];
  plan  [label="Your plan\\nin whole shares", fillcolor="#1E8A72", fontcolor="white"];
  ai    [label="Trained AI\\nsuggests a split"];
  data -> sim -> learn -> ai -> safe -> plan;
  you -> ai;
}
"""
