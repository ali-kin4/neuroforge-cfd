### Table 2 — baselines on AirfRANS/full, 3 seeds (mean +/- std)

_Train: n_train=800, epochs=80, n_points/step=16384, optimizer=adamw-onecycle. Native point-cloud input; scored by the SHARED `evaluate_cases` after identical rasterisation to the GT grid. `residual_error_spearman` is n/a for a plain baseline (NeuroForge-checker-only)._

| model | n_params | mse_u | mse_v | mse_p | mse_nut | surface_mse_p | rho_cl | rho_cd | cl_rel_err_mean | cd_rel_err_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| transolver | 7,350,420 | 0.1202+/-0.0047 | 0.08798+/-0.013 | 628.5+/-29 | 5.733e-09+/-5.9e-10 | 9110+/-5e+02 | 0.9992+/-0.00017 | 0.9963+/-0.0012 | 0.05811+/-0.0057 | 0.0899+/-0.028 |
