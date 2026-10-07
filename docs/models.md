## How the models work (and whether each one earns its place)

Generated from `src/niveshrl/model_registry.py` by `python scripts/models_doc.py`; the records are read from `report/results/`, all out of sample.

| Model | Family | Status | Job | Measured record |
| --- | --- | --- | --- | --- |
| FFNN ranker (Takeuchi & Lee) | Deep learning | live | Rank all NIFTY 200 stocks by the chance of beating the median over the next month. | IC mean +0.043 · IC t-stat 3.78 · Spread / mo +0.007 |
| LSTM ranker | Deep learning | live | Same monthly ranking job, from the sequence of the last 60 daily returns. | IC mean +0.036 · IC t-stat 3.57 · Spread / mo +0.006 |
| Transformer ranker | Deep learning | live | Same monthly ranking job with attention across time. | IC mean +0.043 · IC t-stat 3.74 · Spread / mo +0.006 |
| Next-day model (LightGBM) | Machine learning | live | P(each stock beats tomorrow's median return): what to watch tomorrow. | AUC 0.530 · IC mean +0.058 · IC t-stat 24.78 · Top10 net of costs / day -0.001 |
| Next-day sequence net | Deep learning | research | Same next-day job from the shape of the last 20 days. | AUC 0.527 · IC mean +0.050 · IC t-stat 20.47 |
| Stacked next-day model (4 models) | Machine learning | live | One P(up) per stock each evening from LightGBM, the sequence net, logistic regression and the range model, plus pattern changes (Up/Down × quiet/volatile). | AUC 0.520 · IC mean +0.035 · IC t-stat 5.78 |
| Next-day range model | Machine learning | live | How big tomorrow's high-low range will be (who will move, not which way). | IC mean +0.508 · IC t-stat 284 · Top 20 next-day range 0.048 |
| Volatility forecaster (LSTM) | Deep learning | live | Next month's volatility per stock (risk sizing, the volatility cone). | RMSE log-vol +0.337 · QLIKE 0.277 · Corr +0.553 |
| GARCH(1,1) | Statistical | research | The textbook volatility benchmark the LSTM must beat. | RMSE log-vol +0.400 · QLIKE 0.346 · Corr +0.469 |
| Market regimes (autoencoder + k-means) | Deep learning | live | Label each week Bull / Neutral / Stress (risk state, briefing). | weeks 49.000 · next-month NIFTY vol 0.206 |
| News sentiment (FinBERT) | NLP | live | Tone of the last 48 h of headlines per stock. | no out-of-sample record (see limits) |
| RL allocator (PPO) | Reinforcement learning | research | How a given investor should split money across 29 NIFTY 50 stocks and cash, weekly. | CAGR +0.188 · Sharpe 0.638 · MaxDD -0.331 |
| Intraday bandit (tabular) | Reinforcement learning | live | TAKE / HALF / SKIP each rule signal by its context's track record. | trades 154 · avg R -0.123 · net ₹ -10,847 |
| Intraday ML scorer (LightGBM) | Machine learning | live | P(a rule signal ends in profit after costs). | trades 154 · avg R -0.123 · net ₹ -10,847 |
| Intraday meta-labeler (TCN) | Deep learning | shadow | P(profit) and expected R after costs for a long and a short at every 5-minute bar of every liquid stock. | IC mean +0.032 · IC t 1.65 · top5/day avg R -0.090 |
| Intraday baseline (LightGBM on the same inputs) | Machine learning | research | The benchmark the TCN must beat on the same labels. | IC mean +0.013 · IC t 0.65 · top5/day avg R -0.071 |
| Neural-linear Thompson bandit | Reinforcement learning | shadow | Decide TAKE / HALF / SKIP among the meta-labeler's candidates and learn from every outcome. | trades 53.000 · avg R -0.081 · total R -2.578 |
| Conformal abstention | Statistical | shadow | Trade only when the predicted R stays positive after the model's typical over-optimism. | trades 2.000 · avg R -0.013 · total R -0.026 |
| Learned exits (implicit Q-learning) | Reinforcement learning | shadow | When to close an open intraday trade instead of a fixed 2R target / stop. | trades 26.000 · avg R -0.087 · diff vs fixed -0.107 |
| Swing pick meta-labeler (LightGBM + MLP) | Deep learning | rejected | Predict each swing pick's R to keep only the best half of the daily list. | trades 1,597 · avg R +0.154 · t (per signal day) 3.45 |
| Opening-range breakout (Zarattini, Barbon & Aziz 2024) | Rules | rejected | Published 'profitable day-trading strategy': trade the direction of the first 5-minute bar in stocks in play. | trades 1,057 · avg R / trade -0.057 · t-stat -0.49 |
| VWAP trend (Zarattini & Aziz 2023) | Rules | rejected | Long above the session VWAP, short below, re-checked every bar. | trades 472 · avg R / trade -0.141 · net ₹ (₹1L pool) -66,466 |
| Turtle / Donchian 55-20 breakout | Rules | research | Classic trend following: buy a close above the 55-day high, sell a close below the 20-day low. | CAGR +0.128 · Sharpe 0.752 · max drawdown -0.347 |
| Connors RSI(2) mean reversion | Rules | rejected | Buy short dips in uptrends: close > 200-day average and 2-day RSI < 10; sell when close > 5-day average. | CAGR -0.005 · Sharpe 0.064 · max drawdown -0.589 |

### FFNN ranker (Takeuchi & Lee)
- **Job:** Rank all NIFTY 200 stocks by the chance of beating the median over the next month.
- **Architecture:** Feed-forward net 33 → 40 → 4 → 50 → 2 with a 4-unit bottleneck, dropout, early stopping.
- **Why this architecture:** Monthly cross-sectional ranking from 33 return features is a tabular problem with weak, nonlinear signal; a small bottlenecked MLP captures interactions without the data a sequence model needs. It is the classic deep-learning momentum paper's design, so it doubles as a reproducible baseline.
- **Inputs:** 12 monthly + 20 daily cumulative returns, January flag (cross-sectionally ranked).
- **Training:** Walk-forward, rolling 8-year window, refit yearly (by hand).
- **Where it shows:** Rankers, Backtest lab, stock page models tab
- **Limits:** IC is real but monthly turnover costs eat it: after costs it does not beat equal weight on point-in-time data.
- **Measured:** IC mean +0.043 · IC t-stat 3.78 · Spread / mo +0.007

### LSTM ranker
- **Job:** Same monthly ranking job, from the sequence of the last 60 daily returns.
- **Architecture:** 1-layer LSTM (hidden 32) over 60 days of returns + static features → probability.
- **Why this architecture:** Tests whether the *order* of recent returns (not just their sums) carries information; LSTMs are the standard recurrent baseline for that question.
- **Inputs:** 60 daily returns per stock + the FFNN's static features.
- **Training:** Walk-forward, rolling 8 years, yearly refit.
- **Where it shows:** Rankers
- **Limits:** No better than the FFNN after costs; slower to train.
- **Measured:** IC mean +0.036 · IC t-stat 3.57 · Spread / mo +0.006

### Transformer ranker
- **Job:** Same monthly ranking job with attention across time.
- **Architecture:** 2-layer Transformer encoder (d = 32, 4 heads) over 60 daily return tokens.
- **Why this architecture:** Attention can weight a few important days (earnings gaps, crashes) more than an LSTM's fading memory; included to measure whether that matters here.
- **Inputs:** 60 daily returns + static features.
- **Training:** Walk-forward, rolling 8 years, yearly refit.
- **Where it shows:** Rankers
- **Limits:** Similar IC to the FFNN at ~15× the training time.
- **Measured:** IC mean +0.043 · IC t-stat 3.74 · Spread / mo +0.006

### Next-day model (LightGBM)
- **Job:** P(each stock beats tomorrow's median return): what to watch tomorrow.
- **Architecture:** Gradient-boosted trees (600 trees, early stopping), isotonic calibration.
- **Why this architecture:** ~40 engineered daily features with interactions and outliers: boosted trees are the strongest, fastest learner for tabular data of this size and are robust to feature scaling.
- **Inputs:** Returns, gaps, ranges, RSI, volume, VIX, breadth, day of week (cross-sectionally ranked).
- **Training:** Walk-forward, train Y-5..Y-2, calibrate Y-1, refit yearly (monthly refit measured worse: kept yearly).
- **Where it shows:** Today (watch tomorrow, P(up)), screener, stock page
- **Limits:** Top-10 picks lose after costs; use for watching, not trading.
- **Measured:** AUC 0.530 · IC mean +0.058 · IC t-stat 24.78 · Top10 net of costs / day -0.001

### Next-day sequence net
- **Job:** Same next-day job from the shape of the last 20 days.
- **Architecture:** Window convolutions → 2-layer Transformer encoder → MLP with the tabular features, isotonic calibration.
- **Why this architecture:** Tests whether multi-day *patterns* (not just today's numbers) add to the tabular model; convolution finds local shapes, attention relates them.
- **Inputs:** 20 days × 8 daily features + the tabular row.
- **Training:** Walk-forward yearly; in the stack refit quarterly (slow on CPU).
- **Where it shows:** Stacked model
- **Limits:** Weaker than LightGBM alone (IC 0.050 vs 0.058); ~30 min per fit on this CPU.
- **Measured:** AUC 0.527 · IC mean +0.050 · IC t-stat 20.47

### Stacked next-day model (4 models)
- **Job:** One P(up) per stock each evening from LightGBM, the sequence net, logistic regression and the range model, plus pattern changes (Up/Down × quiet/volatile).
- **Architecture:** Logistic meta-model over the base models' daily ranks, trained on their previous 12 months of out-of-sample predictions.
- **Why this architecture:** Stacking lets models that see different things vote with learned weights; training the meta-model only on out-of-sample predictions prevents it trusting a base model's in-sample overconfidence.
- **Inputs:** The four base models' predictions.
- **Training:** Base models monthly (sequence net quarterly); meta model monthly.
- **Where it shows:** Today → Pattern changes
- **Limits:** Measured: no edge over LightGBM alone (IC 0.035 vs 0.042).
- **Measured:** AUC 0.520 · IC mean +0.035 · IC t-stat 5.78

### Next-day range model
- **Job:** How big tomorrow's high-low range will be (who will move, not which way).
- **Architecture:** LightGBM regression on 13 range/volatility features.
- **Why this architecture:** Volatility clusters and is far more predictable than direction; trees handle the skewed, interacting range features (NR7, gaps, ATR) well.
- **Inputs:** Recent ranges, ATR, gap and move size, volume ratio, NR7/NR4, 60-day vol, VIX.
- **Training:** Rolling 5 years up to yesterday, refit daily.
- **Where it shows:** Today → Who will move, stacked model
- **Limits:** Predicts size only.
- **Measured:** IC mean +0.508 · IC t-stat 284 · Top 20 next-day range 0.048

### Volatility forecaster (LSTM)
- **Job:** Next month's volatility per stock (risk sizing, the volatility cone).
- **Architecture:** LSTM (hidden 32) over 60 days of r and |r| + log realised vols, VIX, NIFTY vol.
- **Why this architecture:** Volatility has long memory and asymmetric reactions; an LSTM learns that from data across all stocks at once, where GARCH fits each stock alone with a fixed form.
- **Inputs:** 60 daily returns and absolute returns, 21/63-day realised vol, VIX, NIFTY 20-day vol.
- **Training:** Rolling 8 years, refit monthly (measured slightly better than yearly), forecast daily.
- **Where it shows:** Stock page Risk tab (cone), Risk screen
- **Limits:** Underreacts to sudden shocks.
- **Measured:** RMSE log-vol +0.337 · QLIKE 0.277 · Corr +0.553

### GARCH(1,1)
- **Job:** The textbook volatility benchmark the LSTM must beat.
- **Architecture:** Constant-mean GARCH(1,1) per stock (arch package).
- **Why this architecture:** The industry-standard volatility model: if the LSTM cannot beat it, the LSTM is not needed.
- **Inputs:** Daily returns of one stock.
- **Training:** 4-year fits, yearly.
- **Where it shows:** Risk screen comparison
- **Limits:** Worse than the LSTM and even EWMA here.
- **Measured:** RMSE log-vol +0.400 · QLIKE 0.346 · Corr +0.469

### Market regimes (autoencoder + k-means)
- **Job:** Label each week Bull / Neutral / Stress (risk state, briefing).
- **Architecture:** Autoencoder compresses 8 market features to 2 numbers; k-means finds 3 clusters, named by their volatility and return.
- **Why this architecture:** Unsupervised: there are no labels for 'regime'. The autoencoder removes redundancy between correlated market measures so clusters are not dominated by one of them.
- **Inputs:** NIFTY returns and volatility, VIX, breadth, dispersion, drawdown (each relative to its 1-year median).
- **Training:** Expanding window since 2005, yearly (rolling 8 years measured worse); relabelled weekly by the pipeline.
- **Where it shows:** Risk state, Today, briefing
- **Limits:** Labels describe volatility, not direction.
- **Measured:** weeks 49.000 · next-month NIFTY vol 0.206

### News sentiment (FinBERT)
- **Job:** Tone of the last 48 h of headlines per stock.
- **Architecture:** FinBERT (BERT fine-tuned on financial text, ProsusAI/finbert).
- **Why this architecture:** A pretrained finance language model understands 'misses estimates' or 'downgrade' without training data of our own; general sentiment models do not.
- **Inputs:** Google News headlines.
- **Training:** Pretrained, not retrained.
- **Where it shows:** News tabs, screener, monitor list
- **Limits:** Whether tone predicts returns can only be measured forward (no headline history).
- **Measured:** no out-of-sample record (see limits)

### RL allocator (PPO)
- **Job:** How a given investor should split money across 29 NIFTY 50 stocks and cash, weekly.
- **Architecture:** PPO actor-critic; permutation-equivariant Transformer encoder; Beta (equity share) × Dirichlet (stock mix) policy.
- **Why this architecture:** Allocation is a sequential decision with costs and taxes: RL optimises the investor's utility directly instead of a one-step forecast; the equivariant encoder scores every stock with the same weights.
- **Inputs:** Causal price features, investor profile.
- **Training:** Trained 2008-18, validated after.
- **Where it shows:** RL & plan
- **Limits:** Does not beat simple baselines (equal weight, minimum variance) after costs on validation.
- **Measured:** CAGR +0.188 · Sharpe 0.638 · MaxDD -0.331

### Intraday bandit (tabular)
- **Job:** TAKE / HALF / SKIP each rule signal by its context's track record.
- **Architecture:** Context buckets (setup × side × time × trend) with t-tests and Thompson draws; 20-day forgetting.
- **Why this architecture:** Few outcomes per context: simple, transparent statistics that only act on clear evidence.
- **Inputs:** Every signal's shadow outcome after costs.
- **Training:** Daily, after the close.
- **Where it shows:** Intraday agent → Learning
- **Limits:** Can only filter; it cannot create a good signal from losing setups.
- **Measured:** trades 154 · avg R -0.123 · net ₹ -10,847

### Intraday ML scorer (LightGBM)
- **Job:** P(a rule signal ends in profit after costs).
- **Architecture:** LightGBM classifier on 12 signal features, recency-weighted.
- **Why this architecture:** Fast, works from a few hundred outcomes.
- **Inputs:** Signal-bar features (time, gap, volume, VWAP distance, stop size…).
- **Training:** Retrained daily on all shadow outcomes (20-day half-life).
- **Where it shows:** Intraday agent
- **Limits:** Same limit as the bandit.
- **Measured:** trades 154 · avg R -0.123 · net ₹ -10,847

### Intraday meta-labeler (TCN)
- **Job:** P(profit) and expected R after costs for a long and a short at every 5-minute bar of every liquid stock.
- **Architecture:** Temporal convolutional network: 4 dilated causal residual blocks (1, 2, 4, 8 bars) + context MLP; multi-task heads; isotonic calibration (~25k parameters).
- **Why this architecture:** Intraday decisions depend on the shape of the last two hours at several time scales; dilated causal convolutions see them all, never look ahead, and train in minutes on a CPU. Triple-barrier labels for every bar give millions of examples instead of a few thousand rule signals.
- **Inputs:** 24 bars × 10 channels (returns, range, VWAP distance, relative volume, NIFTY…) + 10 context numbers.
- **Training:** Walk-forward by day, refit weekly on all earlier days.
- **Where it shows:** Intraday agent (v2)
- **Limits:** ~60 days of 5-minute history (growing daily); no order book.
- **Measured:** IC mean +0.032 · IC t 1.65 · top5/day avg R -0.090

### Intraday baseline (LightGBM on the same inputs)
- **Job:** The benchmark the TCN must beat on the same labels.
- **Architecture:** LightGBM regression on the context + summarised bars.
- **Why this architecture:** If a tree model on summaries does as well, the deep network is not earning its complexity.
- **Inputs:** Same as the TCN, flattened.
- **Training:** Same walk-forward.
- **Where it shows:** Comparison only
- **Limits:** —
- **Measured:** IC mean +0.013 · IC t 0.65 · top5/day avg R -0.071

### Neural-linear Thompson bandit
- **Job:** Decide TAKE / HALF / SKIP among the meta-labeler's candidates and learn from every outcome.
- **Architecture:** Bayesian linear regression on the network's outputs with Thompson sampling and a 20-day half-life.
- **Why this architecture:** Keeps a posterior instead of point estimates, so it explores when unsure and generalises across similar situations; the standard practical deep contextual bandit.
- **Inputs:** TCN predictions per candidate.
- **Training:** Updated after every close.
- **Where it shows:** Intraday agent (v2)
- **Limits:** Only as good as the features it sits on.
- **Measured:** trades 53.000 · avg R -0.081 · total R -2.578

### Conformal abstention
- **Job:** Trade only when the predicted R stays positive after the model's typical over-optimism.
- **Architecture:** Split conformal prediction on a rolling 10-day window of out-of-sample errors (80% quantile).
- **Why this architecture:** Distribution-free and calibrated by construction: a principled 'not sure enough, skip'.
- **Inputs:** Predicted and realised R of recent candidates.
- **Training:** Rolling daily.
- **Where it shows:** Intraday agent (v2)
- **Limits:** With noisy intraday R the margin is large, so it trades rarely.
- **Measured:** trades 2.000 · avg R -0.013 · total R -0.026

### Learned exits (implicit Q-learning)
- **Job:** When to close an open intraday trade instead of a fixed 2R target / stop.
- **Architecture:** Offline RL: fitted Q-iteration with expectile value regression (IQL) over logged price paths; actions hold / exit.
- **Why this architecture:** Learns from history alone without risky live exploration; expectile regression keeps it conservative about situations the data rarely shows.
- **Inputs:** Per bar of an open trade: R so far, bars held, VWAP distance, recent returns, time to close.
- **Training:** Trained on earlier days' paths, evaluated on later ones.
- **Where it shows:** Intraday agent (v2)
- **Limits:** Short history; exits only, the hard stop stays.
- **Measured:** trades 26.000 · avg R -0.087 · diff vs fixed -0.107

### Swing pick meta-labeler (LightGBM + MLP)
- **Job:** Predict each swing pick's R to keep only the best half of the daily list.
- **Architecture:** LightGBM + 2×64 MLP averaged, on 40 point-in-time features incl. every research model's view.
- **Why this architecture:** Meta-labeling: a second model decides which of the primary model's picks to act on.
- **Inputs:** Stock and market state at the signal date, model ranks, context tags.
- **Training:** Walk-forward by year (2017-2026).
- **Where it shows:** Track record (comparison)
- **Limits:** Measured: no improvement (rank correlation −0.03): the list is kept as is.
- **Measured:** trades 1,597 · avg R +0.154 · t (per signal day) 3.45

### Opening-range breakout (Zarattini, Barbon & Aziz 2024)
- **Job:** Published 'profitable day-trading strategy': trade the direction of the first 5-minute bar in stocks in play.
- **Architecture:** Rule: first 5-minute bar up → long at the second bar's open (down → short), stop 10% of daily ATR, exit at the close.
- **Why this architecture:** The best-known recent academic day-trading result; tested here because it is the industry's reference ORB.
- **Inputs:** 5-minute bars; stocks already in play at the first bar (top 20 a day).
- **Training:** No training (fixed rules).
- **Where it shows:** Comparison only
- **Limits:** Paper: +0.13R/trade gross on US stocks without spread/slippage; independent replication found net ≈ 0. Here, after NSE costs: slightly negative.
- **Measured:** trades 1,057 · avg R / trade -0.057 · t-stat -0.49

### VWAP trend (Zarattini & Aziz 2023)
- **Job:** Long above the session VWAP, short below, re-checked every bar.
- **Architecture:** Rule on NIFTY 5-minute bars (index level as a stand-in for NIFTYBEES / futures).
- **Why this architecture:** Marketed as the 'holy grail' of day trading on QQQ; tested for NSE.
- **Inputs:** NIFTY 5-minute bars.
- **Training:** No training.
- **Where it shows:** Comparison only
- **Limits:** Many round trips a day: costs dominate.
- **Measured:** trades 472 · avg R / trade -0.141 · net ₹ (₹1L pool) -66,466

### Turtle / Donchian 55-20 breakout
- **Job:** Classic trend following: buy a close above the 55-day high, sell a close below the 20-day low.
- **Architecture:** Rule on daily closes, 10 equal slots, point-in-time NIFTY 200 members, 0.15% per side costs.
- **Why this architecture:** The most famous trend-following rule set (Turtle traders); trend following has a century of positive evidence across asset classes (Hurst, Ooi & Pedersen).
- **Inputs:** Daily prices.
- **Training:** No training.
- **Where it shows:** Comparison only
- **Limits:** Positive and beats NIFTY, but below equal weight and below 12-1 momentum (19.3%/yr).
- **Measured:** CAGR +0.128 · Sharpe 0.752 · max drawdown -0.347

### Connors RSI(2) mean reversion
- **Job:** Buy short dips in uptrends: close > 200-day average and 2-day RSI < 10; sell when close > 5-day average.
- **Architecture:** Rule on daily closes, 10 slots, point-in-time members, 0.15% per side.
- **Why this architecture:** One of the most popular published short-term rules.
- **Inputs:** Daily prices.
- **Training:** No training.
- **Where it shows:** Comparison only
- **Limits:** High win rate but tiny average gains: costs and crash drawdowns erase them.
- **Measured:** CAGR -0.005 · Sharpe 0.064 · max drawdown -0.589

