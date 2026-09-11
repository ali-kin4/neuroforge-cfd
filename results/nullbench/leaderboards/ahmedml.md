## AhmedML

500 Ahmed-body variants, 8 published shape parameters (geo_parameters_all.csv), force_mom_all.csv (CONSTANT reference area 0.112 m^2, so no area term enters the label). No published split -> 10-fold out-of-sample.

Source verification: `docs/paper/review/null_travels.md#23-ahmedml--a-bar-nobody-has-cleared`

### target: `cd` (metric: r2, protocol: `kfold_oos_k10`, n_fit=499, n_score=499)

> No published ML drag baseline found: searched the dataset paper (arXiv 2407.20801, which reports no ML results), NeuralCFD/GP-UPT (arXiv 2502.09692, whose drag result is on DrivAerML not AhmedML), PhysicsNeMo-CFD (DrivAerML only), and FIGConvNet (its "Ahmed body" is a different dataset). Not adjudicable: a benchmark cannot be said to be beaten by a model whose number was never published. R2 0.684 [0.638, 0.727] (const-area) is reported here as the bar any future AhmedML drag surrogate must clear.

Null out-of-sample: **0.6842** [0.6377, 0.7273] (95% case-level bootstrap, n_boot=10000)

_No published entries verified for this target._

### target: `cl` (metric: r2, protocol: `kfold_oos_k10`, n_fit=499, n_score=499)

> no published ML lift baseline found.

Null out-of-sample: **0.4515** [0.4039, 0.4934] (95% case-level bootstrap, n_boot=10000)

_No published entries verified for this target._

**Headline** (target `cd`, metric r2): null 0.6842 [0.6377, 0.7273] vs best verified published `None` = -- -> covariate-null fraction --. no verified published point-estimate baseline for this target
