# Intraday copilot: money-readiness benchmark

Generated 2026-11-16 23:41. 81 trading days of 5-minute bars; the first days train the model (walk-forward), leaving **66 evaluation days** (design 2026-08-11 to 2026-09-25, untouched holdout 2026-09-28 to 2026-11-16). 2,290 cards (every signal, taken or not). Universe: the 150 most-traded stocks by trailing turnover (point in time). ₹1,00,000 pool, 1% risk per trade, ≤ 3 trades a day, NSE intraday costs + 0.03% slippage per side.

## The live policy (adaptive rules, 2R exits)

| Test | Result | Pass |
| --- | --- | --- |
| Trades in evaluation | 115 | ✓ |
| Net ₹ on the untouched holdout | ₹-6,993 | ✗ |
| Mean daily P&L, 90% bootstrap CI | ₹-221 [-456, +10] | ✗ |
| Deflated Sharpe (vs best of all variants by luck) | 0.01 | ✗ |
| Probability of backtest overfitting | 0.65 | ✗ |
| Matched random entries: p-value | 0.77 | ✗ |
| Net ₹ with slippage doubled | ₹-20,704 | ✗ |

**Verdict: NOT READY** (1/7 checks). Whole evaluation: net ₹-14,563, 33% winning days, max drawdown ₹-19,846.

Matched random control: policy -0.13R per trade vs random median -0.07R (95th percentile 0.04R).

## Every variant tried (all count toward the deflated Sharpe and PBO)

| Variant | Trades | Net ₹ | Holdout ₹ | Net ₹, 2× slippage | Winning days | Daily Sharpe |
| --- | --- | --- | --- | --- | --- | --- |
| rules only · 2R | 198 | -3,484 | +1,598 | -13,846 | 44% | -0.028 |
| rules only · hold to close | 198 | +14,631 | +7,150 | +4,264 | 46% | 0.082 |
| model P≥0.45 · 2R | 125 | -1,307 | -1,565 | -7,751 | 41% | -0.012 |
| model P≥0.45 · hold to close | 125 | +5,660 | -4,067 | -789 | 35% | 0.039 |
| model P≥0.50 · 2R | 104 | +181 | -5,593 | -5,296 | 33% | 0.002 |
| model P≥0.50 · hold to close | 104 | +8,156 | -4,985 | +2,677 | 30% | 0.058 |
| model P≥0.55 · 2R | 53 | -1,398 | -6,632 | -4,151 | 20% | -0.021 |
| model P≥0.55 · hold to close | 53 | -5,332 | -8,227 | -8,086 | 17% | -0.083 |
| orb15 only · 2R | 198 | -3,484 | +1,598 | -13,846 | 44% | -0.028 |
| vwap_hold only · 2R | 102 | -6,170 | -6,194 | -11,693 | 36% | -0.071 |
| rs_breakout only · 2R | 196 | -6,737 | +732 | -16,908 | 47% | -0.051 |
| trend_pullback only · 2R | 104 | -8,720 | -4,345 | -14,447 | 33% | -0.098 |
| adaptive, loose tuner: no t-test (rejected) · 2R | 96 | -15,798 | -12,551 | -20,857 | 26% | -0.183 |
| adaptive, card-level t (rejected) · 2R | 95 | -13,824 | -10,577 | -18,812 | 29% | -0.158 |
| adaptive (live policy) · 2R | 115 | -14,563 | -6,993 | -20,704 | 33% | -0.141 |

Best variant chosen on the design half: **model P≥0.50 · hold to close** → holdout ₹-4,985, verdict NOT READY (2/7).

## Every setup, all cards in the evaluation (full size, 2R exits)

| Setup | Side | Cards | Mean R |
| --- | --- | --- | --- |
| orb15 | long | 534 | -0.085 |
| orb15 | short | 403 | -0.021 |
| rs_breakout | long | 419 | -0.016 |
| rs_breakout | short | 308 | -0.004 |
| trend_pullback | long | 56 | +0.034 |
| trend_pullback | short | 56 | -0.081 |
| vwap_hold | long | 69 | -0.046 |
| vwap_hold | short | 46 | -0.011 |

## What the adaptive policy learned over the replay

- 2026-08-04: orb15.enabled True → False (last 11 days: 171 cards averaged -0.16R (t -2.1))
- 2026-09-11: rs_breakout.min_p 0.45 → 0.4 (the 41 cards with P between 0.40 and 0.45 averaged +0.24R (t 1.5): lowered the bar to include them)
- 2026-10-22: rs_breakout.min_p 0.4 → 0.45 (the 59 cards with P between 0.40 and 0.45 averaged -0.18R (t -2.0): raised the bar to drop them)

## For comparison: the v1 agent's replay

52 days, 151 trades, net ₹-13,578; mean daily ₹-261 [-450, -81] (90% bootstrap CI).

## How to read this

- A strategy is called MONEY-READY only when all seven checks pass. Anything less stays paper-only; the forward scorecard in the app keeps applying the same tests to live paper results as they accumulate.
- Limits: ~4 months of Yahoo 5-minute bars, fills modelled at the next bar's open plus slippage (not real order-book fills), no volume cap in the backtest (liquid large caps only), and one market regime.
