# Fresher training windows: volatility forecaster and regime detector

## Volatility LSTM, out of sample from 2025-01-01 (3,869 stock-months)

| Volatility forecaster | RMSE log-vol | MAE vol | QLIKE | Corr | Bias (f/r) | N |
| --- | --- | --- | --- | --- | --- | --- |
| yearly refit | 0.3173 | 0.0704 | 0.2079 | 0.5583 | 1.0983 | 3869.0000 |
| monthly refit | 0.3149 | 0.0698 | 0.2107 | 0.5560 | 1.0788 | 3869.0000 |
| hist (last 21 days) | 0.3622 | 0.0810 | 0.3118 | 0.5376 | 1.0407 | 3869.0000 |

Paired: mean change in squared log error, monthly − yearly = -0.0015 per month (t -0.99, 20 months; negative = monthly is better).

## Regimes

| Regime detector, 2015-2026 | weeks | flips per year | future vol, Stress ÷ others | future 4w return, Bull − Stress | future vol, Bull | future vol, Neutral | future vol, Stress | weeks Bull | weeks Neutral | weeks Stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expanding (since 2005) | 609.0000 | 4.9967 | 1.4015 | -0.0186 | 0.1169 | 0.1512 | 0.1947 | 207.0000 | 372.0000 | 30.0000 |
| rolling 8 years | 609.0000 | 6.3518 | 1.3076 | -0.0021 | 0.1164 | 0.1428 | 0.1742 | 176.0000 | 307.0000 | 126.0000 |

The two agree on 69% of weeks. A better detector shows a larger future-volatility ratio for Stress weeks (it warns of turbulence) without flipping much more often.

Decision: volatility → monthly refit; regimes → expanding. Run time 987s.
