# Multi-split conformal calibration (mean ± std over 20 random splits)

Pool: 200 AirfRANS test cases, split 100 cal / 100 test per seed; alpha=0.1 (target coverage 0.90). q refit per split per arm; sigma = frozen 5-member ensemble std for all arms. std is sample std (ddof=1); ddof=0 values in the JSON.

Gate: split seed 0 / ch p / backbone reproduced the published q=2.352087, coverage=0.915496 (|dq|=0.0e+00, |dcov|=0.0e+00).

| channel | arm | q | coverage | ECE |
|---|---|---|---|---|
| u | backbone | 2.325 ± 0.069 | 0.895 ± 0.013 | 0.052 ± 0.013 |
| u | corrected_seed0 | 2.227 ± 0.070 | 0.895 ± 0.014 | 0.051 ± 0.014 |
| u | corrected_seed1 | 2.376 ± 0.078 | 0.895 ± 0.013 | 0.055 ± 0.014 |
| u | corrected_seed2 | 2.427 ± 0.076 | 0.895 ± 0.013 | 0.054 ± 0.014 |
| v | backbone | 2.551 ± 0.053 | 0.902 ± 0.009 | 0.056 ± 0.010 |
| v | corrected_seed0 | 2.311 ± 0.051 | 0.902 ± 0.010 | 0.052 ± 0.011 |
| v | corrected_seed1 | 2.279 ± 0.053 | 0.902 ± 0.010 | 0.055 ± 0.011 |
| v | corrected_seed2 | 2.304 ± 0.051 | 0.902 ± 0.010 | 0.054 ± 0.011 |
| p | backbone | 2.261 ± 0.048 | 0.899 ± 0.009 | 0.053 ± 0.009 |
| p | corrected_seed0 | 2.029 ± 0.043 | 0.899 ± 0.009 | 0.053 ± 0.009 |
| p | corrected_seed1 | 2.024 ± 0.037 | 0.897 ± 0.008 | 0.052 ± 0.009 |
| p | corrected_seed2 | 2.068 ± 0.045 | 0.898 ± 0.009 | 0.054 ± 0.010 |
