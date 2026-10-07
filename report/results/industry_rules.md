# Published trading rules on NiveshRL's NSE data, after costs

## Intraday (5-minute bars, about 58 days)

| Rule | trades | avg R / trade | win rate | total R | net ₹ (₹1L pool) | t-stat | days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Opening-range breakout, stocks in play (Zarattini 2024) | 1057 | -0.057 | 0.146 | -60.751 | -60,751.300 | -0.488 | 58 |
| VWAP trend on NIFTY (Zarattini & Aziz 2023) | 472 | -0.141 | 0.089 | -66.466 | -66,465.760 | -22.959 | 58 |
| First-half-hour momentum on NIFTY (Gao et al. 2018) | 58 | -0.203 | 0.138 | -11.748 | -11,747.610 | -7.779 | 58 |

## Daily (point-in-time NIFTY 200 members, 2014-2026)

| Rule | trades | win rate | days | avg trade net | CAGR | Sharpe | max drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Connors RSI(2) mean reversion | 6092 | 0.638 | 3151 | 0.001 | -0.005 | 0.064 | -0.589 |
| Turtle / Donchian 55-20 breakout | 608 | 0.413 | 3151 | 0.035 | 0.128 | 0.752 | -0.347 |
| NIFTY 12-month trend (time-series momentum) | 80 | nan | 3151 | nan | 0.036 | 0.353 | -0.255 |
| Weekly short-term reversal (10 biggest 5-day losers) | 6930 | 0.492 | 3151 | 0.001 | -0.028 | 0.020 | -0.746 |
| Benchmark: equal weight of members (no costs) | 0 | nan | 3151 | nan | 0.165 | 0.954 | -0.462 |
| Benchmark: NIFTY 50 buy and hold | 0 | nan | 3151 | nan | 0.107 | 0.719 | -0.384 |

For comparison, 12-1 cross-sectional momentum (top 20, monthly) earned 19.3% a year after costs on the same point-in-time data (report/results/survivorship.md).
