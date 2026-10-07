# Track record v2: meta-labeling the swing picks

Point-in-time trades under the plan rules, R after costs. Each year's predictions come from LightGBM + a small neural network trained only on trades that had closed before that year.

| Group | trades | avg R | win | t (per signal day) |
| --- | --- | --- | --- | --- |
| all picks | 3451.0000 | 0.1629 | 0.4662 | 4.1478 |
| best 50% by meta-label | 1597.0000 | 0.1542 | 0.4709 | 3.4487 |
| dropped picks | 1854.0000 | 0.1705 | 0.4622 | 3.5652 |
| random control | 3995.0000 | 0.0834 | 0.4378 | 2.0089 |

Keeping the better half of each day's picks vs all picks: -0.005R per trade (t -0.18, 482 signal days). Rank correlation of predicted and realised R: -0.034.

Decision: **keep the list as it is (no reliable improvement)**. Run time 41s.
