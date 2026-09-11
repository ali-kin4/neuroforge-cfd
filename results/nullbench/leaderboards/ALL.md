# NullBench — corrected leaderboards (per-benchmark, own protocol)

Machine-readable form: the sibling `.json` files in this directory. Regenerate with `neuroforge nullbench leaderboard --all`. See `docs/NULLBENCH.md` for the statistic's definition, and `harmonised.md` in this directory for the SEPARATE cross-benchmark view -- these per-benchmark tables are not comparable to each other; see benchmarks.py's module docstring.

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

## DrivAerML

Automotive aerodynamics, DrivAer variants. 16 published morph parameters (arXiv 2408.11969 geo_parameters_all.csv), scored on PhysicsNeMo-CFD's own drag-force label and 436/48 split -- an exact head-to-head, same target, same split, as the published models.

Source verification: `docs/paper/review/null_travels.md#21-drivaerml--the-clean-kill`

### target: `drag_force_N` (metric: r2, protocol: `official_split`, n_fit=436, n_score=48)

> PhysicsNeMo's validation set is built by sorting on drag and taking the top/bottom deciles, which RAISES its variance relative to a random 10% and inflates R2 for every model scored on it equally -- including the null. The head-to-head is unaffected; the absolute R2 is not comparable to a random-split R2. Published values are given to 2 decimals -- see harmonised.py for the split-free anchor (10-fold OOS, R2 0.9598) that IS comparable across benchmarks, and stats.py's rounding-sensitivity discussion for why this benchmark is the one where 2-decimal precision was checked explicitly.

metadata_null_r2: **0.9731** [0.9581, 0.9826] (95% case-level bootstrap, n_boot=10000); metadata_null_mse: 129.985

| model | published | published std | metadata null | null CI95 | published-relative ratio | flags | verdict | source |
|---|---|---|---|---|---|---|---|---|
| X-MeshGraphNet | 0.9200 | -- | 0.9731 | [0.9581, 0.9826] | 1.0577 | null_exceeds_published | BELOW THE NULL | arXiv 2507.10747 Table 6 (surface mesh) |
| FIGConvNet | 0.9700 | -- | 0.9731 | [0.9581, 0.9826] | 1.0032 | null_exceeds_published | straddles | arXiv 2507.10747 Table 6 (surface mesh) |
| DoMINO | 0.9800 | -- | 0.9731 | [0.9581, 0.9826] | 0.9929 |  | straddles | arXiv 2507.10747 Table 6 (surface mesh) |

### target: `drag_force_N_rank` (metric: spearman, protocol: `official_split`, n_fit=436, n_score=48)

> same split and label as drag_force_N, scored in Spearman (Table 4).

metadata_null_spearman: **0.9836** [0.9564, 0.9909] (95% case-level bootstrap, n_boot=10000); metadata_null_mse: 129.985

| model | published | published std | metadata null | null CI95 | published-relative ratio | flags | verdict | source |
|---|---|---|---|---|---|---|---|---|
| X-MeshGraphNet | 0.9600 | -- | 0.9836 | [0.9564, 0.9909] | 1.0246 | null_exceeds_published | straddles | arXiv 2507.10747 Table 4 |
| FIGConvNet | 0.9900 | -- | 0.9836 | [0.9564, 0.9909] | 0.9935 |  | straddles | arXiv 2507.10747 Table 4 |
| DoMINO | 0.9900 | -- | 0.9836 | [0.9564, 0.9909] | 0.9935 |  | straddles | arXiv 2507.10747 Table 4 |

**Headline** (target `drag_force_N`, metric r2): metadata null 0.9731 [0.9581, 0.9826] vs best verified published `DoMINO` = 0.9800 -> published-relative ratio 0.9929. 

## DrivAerNet++

8121 car designs; category tokens (rear/underbody/wheel/mirror family, one-hot, levels fit on TRAIN only) + the 23 published design parameters where they exist (zero-filled for the ~half of designs -- the DrivAerNet-v1 fastbacks -- whose 50-parameter table is unpublished). This is a LOWER bound on what published metadata supports: it is strictly weaker than a null with every design's parameters.

Source verification: `docs/paper/review/null_travels.md#22-drivaernet--the-null-beats-what-the-field-cites-not-what-the-fields-best-is`

### target: `cd` (metric: r2, protocol: `official_split`, n_fit=5819, n_score=1154)

> The dataset paper's own NeurIPS checklist answers "[No]" to error bars, so these rows carry no seed spread. This target zero-fills the 595 test designs with no published parameters at all -- see harmonised.py for the parametric-pool-only anchor (0.8248) that excludes them.

metadata_null_r2: **0.7365** [0.7050, 0.7643] (95% case-level bootstrap, n_boot=10000); metadata_null_mse: 0.000

| model | published | published std | metadata null | null CI95 | published-relative ratio | flags | verdict | source |
|---|---|---|---|---|---|---|---|---|
| PointNet | 0.6430 | -- | 0.7365 | [0.7050, 0.7643] | 1.1454 | null_exceeds_published | BELOW THE NULL | NeurIPS 2024 D&B Table 4 |
| GCNN | 0.5960 | -- | 0.7365 | [0.7050, 0.7643] | 1.2357 | null_exceeds_published | BELOW THE NULL | NeurIPS 2024 D&B Table 4 |
| RegDGCNN | 0.6410 | -- | 0.7365 | [0.7050, 0.7643] | 1.1489 | null_exceeds_published | BELOW THE NULL | NeurIPS 2024 D&B Table 4 |
| TripNet | 0.9570 | -- | 0.7365 | [0.7050, 0.7643] | 0.7696 |  | clears | arXiv 2503.17400 Table 5 (current SOTA) |
| PointNet2D+BiLSTM | 0.9528 | -- | 0.7365 | [0.7050, 0.7643] | 0.7730 |  | clears | arXiv 2601.02112 Table 1 (preprint) |

**Headline** (target `cd`, metric r2): metadata null 0.7365 [0.7050, 0.7643] vs best verified published `TripNet` = 0.9570 -> published-relative ratio 0.7696. 

## AhmedML

500 Ahmed-body variants, 8 published shape parameters (geo_parameters_all.csv), force_mom_all.csv (CONSTANT reference area 0.112 m^2, so no area term enters the label). No published split -> 10-fold out-of-sample.

Source verification: `docs/paper/review/null_travels.md#23-ahmedml--a-bar-nobody-has-cleared`

### target: `cd` (metric: r2, protocol: `kfold_oos_k10`, n_fit=499, n_score=499)

> No published ML drag baseline found: searched the dataset paper (arXiv 2407.20801, which reports no ML results), NeuralCFD/GP-UPT (arXiv 2502.09692, whose drag result is on DrivAerML not AhmedML), PhysicsNeMo-CFD (DrivAerML only), and FIGConvNet (its "Ahmed body" is a different dataset). Not adjudicable: a benchmark cannot be said to be beaten by a model whose number was never published. R2 0.684 [0.638, 0.727] (const-area) is reported here as the bar any future AhmedML drag surrogate must clear.

metadata_null_r2: **0.6842** [0.6377, 0.7273] (95% case-level bootstrap, n_boot=10000); metadata_null_mse: 0.001

_No published entries verified for this target._

### target: `cl` (metric: r2, protocol: `kfold_oos_k10`, n_fit=499, n_score=499)

> no published ML lift baseline found.

metadata_null_r2: **0.4515** [0.4039, 0.4934] (95% case-level bootstrap, n_boot=10000); metadata_null_mse: 0.025

_No published entries verified for this target._

**Headline** (target `cd`, metric r2): metadata null 0.6842 [0.6377, 0.7273] vs best verified published `None` = -- -> published-relative ratio --. no verified published point-estimate baseline for this target

## WindsorML

355 Windsor-body variants, 6 published shape parameters PLUS a derived frontal_area column (7 CSV columns total; ordinary covariates here because force_mom_all.csv uses a CONSTANT reference area). The WindsorML paper (arXiv 2407.19320) itself describes SEVEN design parameters; the published geo_parameters_all.csv is missing `ratio_length_front_rear`, and the frontal_area column it ships instead is 97.1% explained by `clearance` alone -- see docs/paper/review/null_mechanism.md Sec 4, which argues this metadata gap, not the body's physics, most likely explains why this is the one benchmark where the null decisively fails (Sec 4 Finding 4: a single withheld parameter is empirically worth up to R2 0.71 elsewhere). No published split id lists here -> 10-fold out-of-sample.

Source verification: `docs/paper/review/null_travels.md#24-windsorml--the-null-does-not-travel-and-i-am-reporting-that-plainly`

### target: `cd` (metric: r2, protocol: `kfold_oos_k10`, n_fit=355, n_score=355)

> The one published number here is a BOUND, not a point estimate -- see `windsor_implied_r2_floor`. This is the benchmark where the null decisively fails: R2 0.104 [-0.143, 0.267], while the published bound implies the MeshGraphNet attains R2 >= 0.79.

metadata_null_r2: **0.1045** [-0.1427, 0.2668] (95% case-level bootstrap, n_boot=10000); metadata_null_mse: 0.001

| model | published | published std | metadata null | null CI95 | published-relative ratio | flags | verdict | source |
|---|---|---|---|---|---|---|---|---|
| MeshGraphNet (direct KPI head) | 0.7916* | -- | 0.1045 | [-0.1427, 0.2668] | -- | bound | clears (published bound beats the null outright) | WindsorML arXiv 2407.19320 SI D.2 (MSE bound, converted to implied R2) |

**Headline** (target `cd`, metric r2): metadata null 0.1045 [-0.1427, 0.2668] vs best verified published `None` = -- -> published-relative ratio --. no verified published point-estimate baseline for this target (a published BOUND exists -- MeshGraphNet (direct KPI head): clears (published bound beats the null outright) -- see the target's own comparisons; a bound never yields a published-relative ratio)
