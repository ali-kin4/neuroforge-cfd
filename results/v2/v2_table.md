### NeuroForge v2 — Transolver backbone on AirfRANS/full, 3 seed(s) (mean +/- std)

_Backbone: Transolver width=256 L=10 (7,350,420 params), trained 80 epochs on the native point cloud. Corrector: deq, 20 epochs on the rasterised Transolver outputs' residuals. Scored by the SHARED `evaluate_cases` (same rasterisation as the GT grid). (a) = backbone alone; (b) = backbone + certified loop. The LOCAL loop guarantees non-increasing physics-residual norm (not GT MSE); the DEQ loop applies its fixed-point delta directly (no acceptance test)._

| variant | mse_u | mse_v | mse_p | mse_nut | mse_speed | surface_mse_p | rho_cl | rho_cd | cl_rel_err_mean | cd_rel_err_mean | residual_error_spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) backbone alone | 0.1267+/-0.009 | 0.09985+/-0.0035 | 629.6+/-42 | 6.147e-09+/-3.9e-10 | 0.1263+/-0.0098 | 1.079e+04+/-9.1e+02 | 0.9992+/-0.00018 | 0.9954+/-0.00068 | 0.05623+/-0.0023 | 0.07479+/-0.015 | 0.6253+/-0.019 |
| (b) backbone + deq loop | 0.1157+/-0.0033 | 0.07924+/-0.0053 | 471.3+/-15 | 6.124e-09+/-3.9e-10 | 0.1146+/-0.0039 | 1.081e+04+/-1e+03 | 0.9991+/-0.00018 | 0.996+/-0.00094 | 0.05162+/-0.0041 | 0.06668+/-0.0048 | 0.6354+/-0.021 |
