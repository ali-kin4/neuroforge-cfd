# NullBench — harmonised cross-benchmark view

Protocol: **10-fold out-of-sample OLS, seed 0, R2 (sklearn convention), constant reference area where both conventions are published** (`docs/paper/review/null_mechanism.md; scripts/null_mechanism.py (pre-registration commit ffbec98)`)

Cross-benchmark-comparable by construction: identical protocol, identical metric, applied to all five. NOT the per-benchmark worked examples in the sibling leaderboard files, which use each benchmark's own split/metric and are not comparable to each other or to this table -- see benchmarks.py and harmonised.py.

| benchmark | n | metadata_null_r2 | 95% CI | flexible ceiling (sourced) | linearity share (sourced) |
|---|---|---|---|---|---|
| AirfRANS | 200 | 0.6833 | [0.6197, 0.7372] | 0.9463 | 0.7221 |
| AhmedML | 499 | 0.6842 | [0.6377, 0.7273] | 0.7748 | 0.8831 |
| WindsorML | 355 | 0.1045 | [-0.1427, 0.2668] | 0.2516 | 0.4153 |
| DrivAerML | 484 | 0.9598 | [0.9523, 0.9659] | 0.9883 | 0.9711 |
| DrivAerNet++ | 4165 | 0.8248 | [0.8151, 0.8342] | 0.8811 | 0.9361 |

Ranking by metadata_null_r2 (highest to lowest): DrivAerML (0.9598), DrivAerNet++ (0.8248), AhmedML (0.6842), AirfRANS (0.6833), WindsorML (0.1045)

> published entries were scored on each benchmark's OWN protocol (see benchmarks.py); comparing them against a harmonised-protocol null would repeat the exact mixing error this artifact exists to flag, so none is attempted here

Flexible-ceiling values are READ from `results/review/null_mechanism.json`, not recomputed by this harness -- see `harmonised.py`'s module docstring.
