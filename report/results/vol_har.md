# Volatility: LSTM vs the HAR baseline

Out of sample 2012-01-31 to 2026-08-31, 28,461 stock-months where every model has a forecast. HAR = pooled log-HAR on 5-, 21- and 63-day realised volatility, walk-forward (8-year window, refit yearly, one-month purge). Target: next 21 trading days' realised volatility.

| Model | RMSE log-vol | QLIKE | Corr | Bias (f/r) |
| --- | --- | --- | --- | --- |
| hist | 0.4009 | 10.9143 | 0.509 | 1.013 |
| ewma | 0.3681 | 0.3028 | 0.558 | 1.073 |
| garch | 0.3998 | 0.3461 | 0.469 | 1.181 |
| har | 0.3365 | 0.2837 | 0.556 | 1.034 |
| lstm | 0.3372 | 0.2775 | 0.553 | 1.035 |

Diebold-Mariano on squared log errors (per-month cross-sectional means, Newey-West, 1 lag); negative t = first model better:

| Comparison | mean loss diff | t | p | months |
| --- | --- | --- | --- | --- |
| lstm vs har | +0.0005 | +0.33 | 0.739 | 176 |
| lstm vs ewma | -0.0219 | -10.40 | 0.000 | 176 |
| lstm vs garch | -0.0450 | -8.81 | 0.000 | 176 |
| har vs garch | -0.0455 | -8.15 | 0.000 | 176 |

**Gate (roadmap step 5): keep the LSTM only if it beats HAR at p < 0.05.** No significant difference between the LSTM and HAR: prefer HAR (simpler, no training, no PyTorch).
