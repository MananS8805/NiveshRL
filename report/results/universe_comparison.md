# Intraday universe: top 150 vs all liquid NSE stocks

Same 93 high-turnover rules (intraday/algo/research.py), same 82 evaluation days (Jul-Nov 2026, 5-minute bars),
same costs (16.6 bps a round trip at ₹41,667 a trade, 0.03% slippage a side). "All liquid" = every NSE EQ stock with
20-day average turnover ≥ ₹10 crore and price ≥ ₹50 (871 with data). Design half picks the rule; holdout reports it.

| Universe | Trades measured | Rules net-positive (design / holdout) | Best design-half rule | its holdout gross / net bps |
| --- | --- | --- | --- | --- |
| Top 150 by turnover | 483,175 | 0 / 0 of 92 | xs_mom look=6 every=6 k=2 hold=6 | −0.3 / −16.9 |
| All liquid (871) | 1,155,944 | 0 / 0 of 93 | xs_rev look=6 every=3 k=2 hold=12 | +7.2 / −9.4 |

On the wide universe, short-term reversal of the most extreme movers earns 7-12 bps gross in both halves (0-5 bps
on the top 150): the closest thing to an intraday edge measured so far, but below the cost of a round trip, and the
extra stocks are thinner than the 0.03% slippage assumption allows. **Decision: the intraday books keep the top 150.**
Revisit if costs fall (e.g. lower brokerage) or if longer Dhan minute history confirms the reversal effect.
