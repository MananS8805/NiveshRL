# Model logic review (6 Oct 2026)

Question: *"most models use monolithic data from the beginning and do not follow the trend — is my logic correct,
where does it fail?"* Answered by reading how every model picks its training data, then **measuring** whether a
fresher window predicts better out of sample before changing anything. All numbers below are out of sample.

## What each model actually did

| Model | Training data | Refit | Problem found | Measured fix | Decision |
| --- | --- | --- | --- | --- | --- |
| Next-day (Today, P(up)) | rolling 5 years: train Y-5..Y-2, calibrate Y-1 | daily, but on the **same** data all year | 2026 predictions came from a model that saw nothing after Dec 2024 (9-21 months stale) | monthly refit on the last 4 years: IC 0.037 vs 0.042 yearly (t −1.34, 433 days) | **kept yearly**: fresher was not better |
| Stacked (new) | base models monthly, meta on prior 12 months of their OOS predictions | monthly (sequence net quarterly) | new | IC 0.035 vs 0.037 for LightGBM alone (t −0.56) | shipped as asked, labelled "no measured edge" |
| FFNN / LSTM / Transformer rankers | rolling 8 years, early-stopping on the last 12 months | **only when `train_rankers.py` is run by hand** | Oct 2026 scores from models trained on data to 2024 | not measured yet (see recommendations) | unchanged |
| Volatility LSTM / GARCH | rolling 8 / 4 years | by hand | **bug**: a row needed the next 21 days of returns, so the newest forecast was always ≥ 1 month old; stopped at 31 Aug 2026 | monthly vs yearly RMSE(log vol) 0.315 vs 0.317 (t −0.99) | **fixed + monthly**, now in the daily pipeline |
| Market regimes | **expanding** since 2005 | by hand | the one truly "monolithic" model | rolling 8y: Stress-week future vol ratio 1.31 vs 1.40, flips 6.4 vs 5.0 per year | **kept expanding** (it warns better); now relabelled weekly by the pipeline |
| Intraday bandit | every outcome since day 1, equal weight | daily | never forgets: adapts ever more slowly | half-life 20 days: avg R −0.12 vs −0.17, net −₹10.8k vs −₹13.6k (52 days) | **enabled (20 days)**, weak evidence |
| Intraday ML scorer | all shadow history, equal weight | daily | same | same replay (weights by age) | enabled with the bandit |
| Range model | rolling 5 years up to yesterday | daily | none | — | fine |
| Momentum, screener, trade plans | current data | daily | none | — | fine |

**So your intuition was half right.** Most models were not trained on all history; they used rolling windows. The
real faults were (1) refitting only once a year (or by hand), so the newest data was ignored for up to 21 months,
(2) the volatility forecaster could not forecast the current month at all, and (3) the intraday learners never
forget. But when tested, a fresher window did **not** reliably predict better for the next-day model or the regime
detector: daily stock returns are mostly noise, and more data often beats newer data. That is why every switch was
measured first and two were rejected.

## Other logic issues found in this round

- **Desk planner / stock trade plan used the last close all day** (the "static entry"). Now the live price during
  market hours, labelled `live HH:MM`.
- **Agent signal and entry times look identical**: correct, not look-ahead (the signal is stamped at the end of its
  bar = the next bar's open, where the fill happens). The tables now show *time in* / *time out*.
- **Model skill drifts**: the new drift monitor shows every next-day model's 60-day IC fell to about zero in
  Feb-Jun 2026 and recovered after. A model can be "right" on average and useless for months.
- **Top-10 next-day picks lose after costs** in every variant (−0.12% to −0.20% a day). The model ranks stocks
  slightly better than chance; trading its top 10 daily does not pay. Use it to choose what to *look at*.

## Recommendations (in order of value)

1. **Refit the rankers monthly inside the pipeline**, measured the same way (`scripts/compare_refits.py` pattern). They
   are the models most likely to benefit, because they are monthly-horizon and 8-year windows move slowly.
2. **Act on the drift monitor**: when a model's 60-day IC falls below its own 10th percentile, show its outputs greyed
   out and drop it from the stack until it recovers.
3. **Turn next-day probabilities into decisions only with costs**: rank by expected return net of the cost of a round
   trip, and only show stocks where the expected edge exceeds it (today almost none do).
4. **Recency in "similar past setups"**: show the last 12 months beside the 2015-2026 average, so a setup that stopped
   working is visible.
5. **Regime-conditional models**: train/evaluate the next-day and intraday models separately by regime (or add regime
   as a feature) — the drift chart suggests skill depends on the market state.
6. **Probability calibration report**: for each model, a reliability chart (predicted 55% → happened 55%?) in Track
   record; the isotonic calibration assumes the validation period resembles the test period.
7. **Live intraday evidence**: the agent needs 100+ live paper trades before any verdict; the 52-day replay is too
   short for tuning (the forgetting result could be luck).
8. **Point-in-time everywhere**: the daily pipeline scores today's NIFTY 200 members; research uses point-in-time
   membership. Keep comparisons on the same universe (done here) and prefer PIT for any new backtest.
9. **One "model card" per model in the glossary**: training window, refit schedule, last refit date, measured record,
   drift status — so staleness is always visible in the app.
10. **Automate the measurement**: run the three `compare_*.py` scripts monthly (e.g. on the first weekend) and let the
    pipeline follow their decision files, so switching stays evidence-based as markets change.

Reproduce: `python scripts/compare_refits.py --seq`, `python scripts/compare_freshness.py`,
`python scripts/compare_forgetting.py`. Reports: `report/results/refit_comparison.md`,
`freshness_comparison.md`, `forgetting_comparison.md`.
