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
