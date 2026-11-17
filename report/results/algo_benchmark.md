# Algo trader: the 50-trade book

Replay of the live policy over 71 days (2026-08-04 → 2026-11-16): every day it trades the rule with the best net ₹/day over the previous 20 days among rules averaging ≥ 50 trades/day that 12 position slots can carry that often. Each trade carries ₹41,667 (pool × 5 ÷ 12); a round trip costs 16.6 bps (brokerage, STT, exchange, SEBI, stamp, GST and 0.03% slippage a side).

- Trades: 3,857 (54/day)
- Net: ₹-241,843; holdout half ₹-144,630; with doubled slippage ₹-338,268
- Mean day ₹-3,406, 90% bootstrap [-4,051, -2,794]
- Deflated Sharpe 0.00; PBO 0.00; share of grid rules it beats 67%

| Check | Passed |
| --- | --- |
| At least 50 trades in the test period | yes |
| Positive net ₹ on the untouched holdout days | no |
| 90% bootstrap lower bound of mean daily P&L > 0 | no |
| Deflated Sharpe ≥ 0.95 (beats the best-of-N-by-luck Sharpe) | no |
| Probability of backtest overfitting < 0.5 | yes |
| Beats matched random entries (p < 0.05) | no |
| Still positive with slippage doubled | no |

**Verdict: NOT READY** (2/7).

Best rules over all replay days (hindsight, for context only; ≥ 20 trades/day):

| Rule | trades/day | gross bps | net ₹/day |
| --- | --- | --- | --- |
| rvol_break m=5.0 hold=60 | 24 | +1.3 | -1,325 |
| xs_mom look=6 every=6 k=2 hold=12 | 33 | +4.9 | -1,602 |
| vwap_fade z=2.5 hold=12 | 21 | +0.6 | -1,650 |
| xs_mom look=3 every=6 k=2 hold=12 | 34 | +4.8 | -1,716 |
| rvol_break m=5.0 hold=6 | 31 | +2.0 | -1,774 |
| rvol_break m=5.0 hold=12 | 29 | -0.5 | -1,845 |
| vwap_fade z=2.5 hold=6 | 28 | +3.0 | -1,932 |
| xs_mom look=6 every=6 k=2 hold=6 | 36 | +3.3 | -1,980 |
| xs_mom look=3 every=6 k=2 hold=6 | 36 | +3.1 | -2,053 |
| rvol_break m=5.0 hold=3 | 33 | -0.4 | -2,269 |

Why: the best rules earn +4.9 bps gross per trade, the costs are 16.6 bps. A rule must earn more per trade than it pays, and on 5-minute bars of liquid NSE stocks none of the 93 measured high-turnover rules does. Gross edge grows with holding time while the cost per trade is fixed.