## WindsorML

355 Windsor-body variants, 6 published shape parameters PLUS a derived frontal_area column (7 CSV columns total; ordinary covariates here because force_mom_all.csv uses a CONSTANT reference area). The WindsorML paper (arXiv 2407.19320) itself describes SEVEN design parameters; the published geo_parameters_all.csv is missing `ratio_length_front_rear`, and the frontal_area column it ships instead is 97.1% explained by `clearance` alone -- see docs/paper/review/null_mechanism.md Sec 4, which argues this metadata gap, not the body's physics, most likely explains why this is the one benchmark where the null decisively fails (Sec 4 Finding 4: a single withheld parameter is empirically worth up to R2 0.71 elsewhere). No published split id lists here -> 10-fold out-of-sample.

Source verification: `docs/paper/review/null_travels.md#24-windsorml--the-null-does-not-travel-and-i-am-reporting-that-plainly`

### target: `cd` (metric: r2, protocol: `kfold_oos_k10`, n_fit=355, n_score=355)

> The one published number here is a BOUND, not a point estimate -- see `windsor_implied_r2_floor`. This is the benchmark where the null decisively fails: R2 0.104 [-0.143, 0.267], while the published bound implies the MeshGraphNet attains R2 >= 0.79.

Null out-of-sample: **0.1045** [-0.1427, 0.2668] (95% case-level bootstrap, n_boot=10000)

| model | published | published std | null | null CI95 | covariate-null fraction | flags | verdict | source |
|---|---|---|---|---|---|---|---|---|
| MeshGraphNet (direct KPI head) | 0.7916* | -- | 0.1045 | [-0.1427, 0.2668] | -- | bound | clears (published bound beats the null outright) | WindsorML arXiv 2407.19320 SI D.2 (MSE bound, converted to implied R2) |

**Headline** (target `cd`, metric r2): null 0.1045 [-0.1427, 0.2668] vs best verified published `None` = -- -> covariate-null fraction --. no verified published point-estimate baseline for this target (a published BOUND exists -- MeshGraphNet (direct KPI head): clears (published bound beats the null outright) -- see the target's own comparisons; a bound never yields a covariate-null fraction)
