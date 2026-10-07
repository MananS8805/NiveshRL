# Intraday meta-labeler: TCN vs LightGBM (walk-forward, 10 Aug → 01 Oct 2026)

Every liquid stock's every 5-minute bar, long and short, triple-barrier labels (ATR stop ≥ the 0.2R-cost distance, 2R target, 60-minute limit), R after real costs. Each block of 5 days is predicted by a model trained on all earlier days only.

| Model | days | IC mean | IC t | top5/day avg R | top5/day win | top5/day t | E>0 trades/day | E>0 avg R | all bars avg R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TCN (deep learning), all stocks | 38.0000 | 0.0311 | 1.5903 | -0.1100 | 0.3474 | -1.6130 | 208.5263 | -0.2895 | -0.1544 |
| LightGBM, all stocks | 38.0000 | 0.0100 | 0.5241 | -0.0336 | 0.4737 | -0.2932 | 2144.6579 | -0.2817 | -0.1544 |
| TCN, in-play stocks | 38.0000 | 0.0315 | 1.6515 | -0.0899 | 0.4053 | -1.0484 | 16.3421 | -0.1084 | -0.1512 |
| LightGBM, in-play stocks | 38.0000 | 0.0134 | 0.6516 | -0.0705 | 0.4684 | -0.6595 | 134.0000 | -0.2156 | -0.1512 |

IC = daily rank correlation between predicted and realised R. Top5/day = the 5 bars with the highest predicted R each day, ranked with the whole day in view: a test of ranking skill, not a tradable policy (scripts/compare_intraday_v2.py tests causal policies). A useful model has IC t > 2 and a positive top-5 average R after costs.

Run time 2,791s.
