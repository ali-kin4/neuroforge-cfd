## AirfRANS

2-D RANS airfoils, official `full` task. Parameters: inlet speed U, angle of attack alpha (+ alpha^2), and the NACA digits parsed from the case file name -- exactly the geometry a surrogate is given as input.

Source verification: `docs/paper/review/published_baselines_verified.md`

### target: `cl` (metric: spearman, protocol: `official_split`, n_fit=800, n_score=200)

metadata_null_spearman: **0.9821** [0.9737, 0.9866] (95% case-level bootstrap, n_boot=10000); metadata_null_mse: 0.012

| model | published | published std | metadata null | null CI95 | published-relative ratio | flags | verdict | source |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.9130 | 0.0180 | 0.9821 | [0.9737, 0.9866] | 1.0757 | null_exceeds_published | BELOW THE NULL | NeurIPS 2022 D&B Table 3 |
| GraphSAGE | 0.9650 | 0.0110 | 0.9821 | [0.9737, 0.9866] | 1.0178 | null_exceeds_published | BELOW THE NULL | NeurIPS 2022 D&B Table 3 |
| PointNet | 0.9380 | 0.0230 | 0.9821 | [0.9737, 0.9866] | 1.0471 | null_exceeds_published | BELOW THE NULL | NeurIPS 2022 D&B Table 3 |
| Graph U-Net | 0.9670 | 0.0190 | 0.9821 | [0.9737, 0.9866] | 1.0157 | null_exceeds_published | BELOW THE NULL | NeurIPS 2022 D&B Table 3 |
| Transolver | 0.9978 | -- | 0.9821 | [0.9737, 0.9866] | 0.9843 |  | clears | Wu et al. ICML 2024, arXiv 2402.02366 |

### target: `cd` (metric: spearman, protocol: `official_split`, n_fit=800, n_score=200)

> Every published rho_D is negative; three of four 95% intervals span zero. The published-relative ratio is undefined for all four rows -- see stats.py's floor convention -- because the published metric never exceeds its own floor.

metadata_null_spearman: **0.9318** [0.8981, 0.9531] (95% case-level bootstrap, n_boot=10000); metadata_null_mse: 0.000

| model | published | published std | metadata null | null CI95 | published-relative ratio | flags | verdict | source |
|---|---|---|---|---|---|---|---|---|
| MLP | -0.1170 | 0.2560 | 0.9318 | [0.8981, 0.9531] | -- | published_at_or_below_floor | BELOW THE NULL | NeurIPS 2022 D&B Table 3 |
| GraphSAGE | -0.3030 | 0.1240 | 0.9318 | [0.8981, 0.9531] | -- | published_at_or_below_floor | BELOW THE NULL | NeurIPS 2022 D&B Table 3 |
| PointNet | -0.0220 | 0.0970 | 0.9318 | [0.8981, 0.9531] | -- | published_at_or_below_floor | BELOW THE NULL | NeurIPS 2022 D&B Table 3 |
| Graph U-Net | -0.1380 | 0.2580 | 0.9318 | [0.8981, 0.9531] | -- | published_at_or_below_floor | BELOW THE NULL | NeurIPS 2022 D&B Table 3 |

**Headline** (target `cd`, metric spearman): metadata null 0.9318 [0.8981, 0.9531] vs best verified published `None` = -- -> published-relative ratio --. every verified published entry for this target is at or below the metric's own floor (0.0) -- no ratio is defined for any of them; see stats.py's floor convention. The published entries themselves not clearing the floor is the more important fact here.
