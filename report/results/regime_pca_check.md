# Regimes without torch

Walk-forward weekly labels 2011-01-07 → 2026-10-02 (822 weeks), expanding training window, same 8 features and naming rule.

| Comparison | pairs | same label (min / mean) | same Stress flag (min / mean) |
| --- | --- | --- | --- |
| torch AE vs torch AE (other seed) | 3.0 | 85.4% / 86.8% | 96.8% / 97.0% |
| NumPy AE vs torch AE | 9.0 | 81.6% / 84.6% | 95.0% / 96.2% |
| PCA vs torch AE | 3.0 | 55.6% / 56.2% | 90.1% / 90.4% |

**Decision: the NumPy autoencoder replaces the torch one** (accepted if its mean agreement with the torch model is within 2 points of the torch model's own worst seed-to-seed agreement, and within 1 point on the Stress flag). PCA labels Bull and Neutral differently and was rejected.

Out-of-sample NIFTY behaviour by regime, torch autoencoder (seed 0):

```
         weeks  share  next 4w NIFTY return  next-month NIFTY vol  hit rate (4w > 0)
regime                                                                              
Bull       291 0.3540                0.0047                0.1259             0.5704
Neutral    480 0.5839                0.0061                0.1534             0.6083
Stress      51 0.0620                0.0254                0.2073             0.6863
```

NumPy autoencoder (seed 0):

```
         weeks  share  next 4w NIFTY return  next-month NIFTY vol  hit rate (4w > 0)
regime                                                                              
Bull       340 0.4136                0.0044                0.1314             0.5529
Neutral    455 0.5535                0.0080                0.1539             0.6396
Stress      27 0.0328                0.0174                0.2250             0.5185
```

PCA:

```
         weeks  share  next 4w NIFTY return  next-month NIFTY vol  hit rate (4w > 0)
regime                                                                              
Bull       378 0.4599                0.0059                0.1413             0.5635
Neutral    370 0.4501                0.0049                0.1337             0.6108
Stress      74 0.0900                0.0211                0.2415             0.7297
```
## Follow-up: seed variance and the 5-seed vote

Stress weeks over five seeds: torch autoencoder 37-58 (51, 57, 37, 48, 58); NumPy autoencoder 27-76 (27, 76, 35, 59,
61). The NumPy network varies more from seed to seed, so a single seed is not used. The app takes the **majority
label over 5 NumPy seeds** (ties go to Neutral):

| Comparison | same label | same Stress flag |
| --- | --- | --- |
| 5-seed vote (seeds 0-4) vs 5-seed vote (seeds 5-9) | 91.6% | 97.9% |
| 5-seed vote vs torch autoencoder, seeds 0 / 1 / 2 | 90.1% / 89.4% / 89.7% | 98.4% / 98.4% / 97.9% |
| torch autoencoder vs itself (other seed), for reference | 85.4-88.3% | 96.8-97.1% |

The vote flags 50 Stress weeks (torch seed 0: 51). Out of sample by regime (vote, seeds 0-4):

```
         weeks   share  next 4w NIFTY return  next-month NIFTY vol  hit rate (4w > 0)
Bull       293  0.3564                0.0051                0.1224             0.5666
Neutral    479  0.5827                0.0055                0.1530             0.6096
Stress      50  0.0608                0.0300                0.2337             0.7000
```

**Decision: the daily pipeline labels regimes with the 5-seed NumPy vote (no torch).**
