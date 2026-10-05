# Intraday agent: forgetting vs remembering everything

| Learning | days | pool end ₹ | net ₹ | costs ₹ | trades | avg R | win rate | second-half avg R | second-half trades | random control net ₹ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| remember everything | 52.00 | 86,421.67 | -13,578.33 | 6,724.26 | 151.00 | -0.17 | 0.38 | -0.31 | 63.00 | -30,546.22 |
| half-life 10 days | 52.00 | 89,794.27 | -10,205.73 | 6,495.05 | 148.00 | -0.19 | 0.39 | -0.37 | 62.00 | -30,546.22 |
| half-life 20 days | 52.00 | 89,153.07 | -10,846.93 | 6,983.43 | 154.00 | -0.12 | 0.40 | -0.28 | 71.00 | -30,546.22 |

Same bars, same rules and guardrails; only how much the bandit and the ML scorer weight old outcomes differs. ~60 days is a small sample: a difference of a few thousand rupees is within luck. The second-half columns matter most, because forgetting can only help after there is something to forget.

Best net: half-life 10 days.
