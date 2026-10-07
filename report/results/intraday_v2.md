# Intraday agent v2: decision policies (out of sample, 38 days, TCN predictions)

| Policy | trades | trades/day | avg R | win rate | total R | net ₹ | daily t | max drawdown R | fixed exits avg R (same trades) | diff vs fixed | diff t | avg bars held |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random in-play bars (control) | 182.000 | 4.789 | -0.197 | 0.379 | -35.814 | -35,814.350 | -3.667 | 37.881 |  |  |  |  |
| model: top-5 threshold (causal) | 70.000 | 1.842 | 0.016 | 0.386 | 1.142 | 1,142.308 | 0.194 | 3.747 |  |  |  |  |
| model: only predicted R > 0 | 44.000 | 1.158 | -0.045 | 0.295 | -1.974 | -1,973.878 | -0.473 | 5.257 |  |  |  |  |
| model + neural bandit | 53.000 | 1.395 | -0.081 | 0.396 | -2.578 | -2,577.725 | -0.753 | 4.032 |  |  |  |  |
| model + conformal abstention | 2.000 | 0.053 | -0.013 | 0.500 | -0.026 | -26.010 | -0.121 | 0.163 |  |  |  |  |
| model top + learned exits (IQL), 2nd half | 26.000 |  | -0.087 |  |  |  |  |  | 0.021 | -0.107 | -1.527 | 10.000 |

For reference, the current agent (v1, rule setups + tabular bandit + LightGBM scorer) lost ₹13,578 over its 52-day replay: 151 trades at -0.17R.

Best policy by total R: **model: top-5 threshold (causal)**. Switch rule: at least 20 trades, average R > 0 after costs and daily t > 1.5. Decision: **keep v1 live; run v2 as a shadow paper account**.

Simplifications: one position per stock per day, no overlap limit across stocks, the triple-barrier exit (2R target, ATR stop, 60 minutes) for every trade.
