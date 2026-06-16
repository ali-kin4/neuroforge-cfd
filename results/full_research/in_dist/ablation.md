### Ablation — airfrans/full, 3 seeds (mean ± std)

| arm | mse_u | mse_v | mse_p | surface_mse_p | rho_cl | rho_cd | residual_error_spearman |
|---|---|---|---|---|---|---|---|
| backbone | 3.479±0.105 | 0.385±0.012 | 2444.843±120.719 | 548822.833±27946.051 | 0.987±0.000 | 0.895±0.013 | 0.397±0.003 |
| backbone (no physics loss) | 1.995±0.042 | 0.323±0.008 | 1963.533±55.458 | 1123649.009±104821.626 | 0.992±0.000 | 0.945±0.008 | 0.605±0.016 |
| backbone + local corrector | 3.482±0.058 | 0.398±0.007 | 2469.430±71.035 | 541533.592±20452.859 | 0.985±0.002 | 0.888±0.012 | 0.389±0.031 |
| backbone + DEQ corrector | 3.457±0.811 | 0.880±0.064 | 3832.687±331.278 | 361680.912±12272.795 | 0.986±0.002 | 0.923±0.014 | 0.827±0.002 |

_Lower MSE is better; rho closer to 1 is better; `residual_error_spearman` > 0 means a low residual tracks low error (the trust signal is valid). The corrector helps iff `backbone + corrector` beats `backbone` on MSE / rho._
