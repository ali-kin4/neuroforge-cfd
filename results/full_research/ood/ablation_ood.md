### OOD ablation — airfrans, 3 seeds (mean ± std)

**OUT-OF-DISTRIBUTION evaluation.** Each arm is trained on a task's *train* split and evaluated on that task's *test* split, whose airfrans-protocol parameter range (e.g. Reynolds number / angle of attack) is **disjoint** from the train range. These metrics therefore measure generalization to unseen flow regimes, not in-distribution fit.

#### OOD task: `reynolds` (train range -> held-out test range)

| arm | mse_u | mse_v | mse_p | surface_mse_p | rho_cl | rho_cd | residual_error_spearman |
|---|---|---|---|---|---|---|---|
| backbone | 16.218±0.746 | 1.187±0.149 | 7220.675±612.167 | 964641.291±61419.425 | 0.950±0.008 | 0.893±0.009 | 0.748±0.077 |
| backbone (no physics loss) | 12.551±2.536 | 1.337±0.056 | 6758.053±476.974 | 1904445.994±130745.218 | 0.940±0.015 | 0.904±0.010 | 0.717±0.041 |
| backbone + local corrector | 14.186±3.671 | 1.294±0.039 | 7508.284±137.664 | 985257.747±37682.153 | 0.950±0.007 | 0.905±0.006 | 0.722±0.043 |
| backbone + DEQ corrector | 5.300±1.544 | 1.208±0.255 | 6469.453±438.027 | 652280.989±45389.842 | 0.943±0.008 | 0.915±0.009 | 0.742±0.041 |

#### OOD task: `aoa` (train range -> held-out test range)

| arm | mse_u | mse_v | mse_p | surface_mse_p | rho_cl | rho_cd | residual_error_spearman |
|---|---|---|---|---|---|---|---|
| backbone | 4.312±0.077 | 1.310±0.039 | 6042.767±242.470 | 2000591.804±55092.051 | 0.963±0.003 | 0.926±0.002 | 0.314±0.064 |
| backbone (no physics loss) | 2.990±0.206 | 1.258±0.078 | 4733.591±142.073 | 3313418.253±322368.201 | 0.974±0.004 | 0.943±0.003 | 0.329±0.062 |
| backbone + local corrector | 4.055±0.148 | 1.314±0.022 | 5558.268±136.852 | 1963823.278±77835.229 | 0.960±0.005 | 0.928±0.005 | 0.336±0.028 |
| backbone + DEQ corrector | 3.538±0.681 | 1.407±0.089 | 4620.586±170.247 | 1042710.930±78297.565 | 0.966±0.006 | 0.905±0.013 | 0.746±0.027 |

_Lower MSE is better; rho closer to 1 is better. The correction loop reduces the in-distribution -> OOD gap iff `backbone + corrector` beats `backbone` on these (OOD) metrics by a wider margin than it does in-distribution. Compare against the in-distribution `run_ablation` table on the same arms to read off the gap._
