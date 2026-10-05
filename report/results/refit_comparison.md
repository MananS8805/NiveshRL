# Next-day models: yearly vs monthly refit, and the 4-model stack

Out of sample, 01 Jan 2025 → 01 Oct 2026, today's NIFTY 200 members (as the daily pipeline uses). Yearly = refit each January on years Y-5..Y-2, calibrated on Y-1. Monthly = refit each month on the last 4 years up to 3 months before, calibrated on those 3 months. Stacked = logistic meta model over the base models' ranks, trained on their previous 12 months of out-of-sample predictions.

## Skill

| Model | AUC | Accuracy | IC mean | IC t-stat | Top10 hit rate | Top10 excess / day | Top10 net of costs / day | Days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yearly lgbm (old pipeline) | 0.5248 | 0.5210 | 0.0424 | 7.3004 | 0.5313 | 0.0006 | -0.0017 | 433.0000 |
| yearly logreg | 0.5179 | 0.5132 | 0.0317 | 5.0455 | 0.5397 | 0.0007 | -0.0016 | 433.0000 |
| monthly lgbm | 0.5204 | 0.5157 | 0.0369 | 7.0265 | 0.5338 | 0.0011 | -0.0012 | 433.0000 |
| monthly logreg | 0.5203 | 0.5154 | 0.0353 | 5.9669 | 0.5406 | 0.0008 | -0.0015 | 433.0000 |
| monthly seq | 0.5110 | 0.5114 | 0.0192 | 2.7144 | 0.5164 | 0.0002 | -0.0020 | 433.0000 |
| monthly ensemble | 0.5194 | 0.5166 | 0.0357 | 5.8360 | 0.5377 | 0.0008 | -0.0015 | 433.0000 |
| stacked (4 models) | 0.5202 | 0.5157 | 0.0349 | 5.7784 | 0.5390 | 0.0003 | -0.0020 | 433.0000 |

## Paired tests

| Comparison (daily rank IC, same stock-days) | IC a | IC b | diff | t | days | a better on % of days |
| --- | --- | --- | --- | --- | --- | --- |
| monthly lgbm vs yearly lgbm | 0.0369 | 0.0424 | -0.0055 | -1.3380 | 433.0000 | 0.4434 |
| stacked vs monthly lgbm | 0.0349 | 0.0368 | -0.0019 | -0.5648 | 433.0000 | 0.5150 |
| stacked vs yearly lgbm | 0.0349 | 0.0424 | -0.0075 | -1.7262 | 433.0000 | 0.4688 |

A positive diff with t > 2 means the first model ranked tomorrow's returns better on the same days. Run time 2,854s.
