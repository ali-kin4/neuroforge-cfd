## DrivAerNet++

8121 car designs; category tokens (rear/underbody/wheel/mirror family, one-hot, levels fit on TRAIN only) + the 23 published design parameters where they exist (zero-filled for the ~half of designs -- the DrivAerNet-v1 fastbacks -- whose 50-parameter table is unpublished). This is a LOWER bound on what published metadata supports: it is strictly weaker than a null with every design's parameters.

Source verification: `docs/paper/review/null_travels.md#22-drivaernet--the-null-beats-what-the-field-cites-not-what-the-fields-best-is`

### target: `cd` (metric: r2, protocol: `official_split`, n_fit=5819, n_score=1154)

> The dataset paper's own NeurIPS checklist answers "[No]" to error bars, so these rows carry no seed spread.

Null out-of-sample: **0.7365** [0.7050, 0.7643] (95% case-level bootstrap, n_boot=10000)

| model | published | published std | null | null CI95 | covariate-null fraction | flags | verdict | source |
|---|---|---|---|---|---|---|---|---|
| PointNet | 0.6430 | -- | 0.7365 | [0.7050, 0.7643] | 1.1454 | null_exceeds_published | BELOW THE NULL | NeurIPS 2024 D&B Table 4 |
| GCNN | 0.5960 | -- | 0.7365 | [0.7050, 0.7643] | 1.2357 | null_exceeds_published | BELOW THE NULL | NeurIPS 2024 D&B Table 4 |
| RegDGCNN | 0.6410 | -- | 0.7365 | [0.7050, 0.7643] | 1.1489 | null_exceeds_published | BELOW THE NULL | NeurIPS 2024 D&B Table 4 |
| TripNet | 0.9570 | -- | 0.7365 | [0.7050, 0.7643] | 0.7696 |  | clears | arXiv 2503.17400 Table 5 (current SOTA) |
| PointNet2D+BiLSTM | 0.9528 | -- | 0.7365 | [0.7050, 0.7643] | 0.7730 |  | clears | arXiv 2601.02112 Table 1 (preprint) |

**Headline** (target `cd`, metric r2): null 0.7365 [0.7050, 0.7643] vs best verified published `TripNet` = 0.9570 -> covariate-null fraction 0.7696. 
