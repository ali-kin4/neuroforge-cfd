# Per-cell trust localization (deployed ensemble mean, n=200)

Date: 2026-08-03 14:48:54  |  runtime 202s (CPU, cached predictions)

Can the per-cell physics-residual map localize WHERE the prediction is wrong? Published control (OLD FNO, n=15): per-cell Spearman 0.22 +/- 0.06.

| Variant | Scale | per-case Spearman (mean +/- std) | pooled Pearson | delta vs V0 | % cases improved | patch AUROC (top-decile err) |
|---|---|---|---|---|---|---|
| V0_raw | cell | 0.166 +/- 0.155 | 0.079 | -- | -- | -- |
| V1_smooth_s1 | cell | 0.216 +/- 0.176 | 0.120 | +0.050 | 96% | -- |
| V1_smooth_s2 | cell | 0.251 +/- 0.193 | 0.150 | +0.085 | 96% | -- |
| V1_smooth_s4 | cell | 0.301 +/- 0.220 | 0.200 | +0.135 | 93% | -- |
| V2_patch_k4 | patch | 0.220 +/- 0.178 | 0.123 | +0.054 | 95% | 0.717 +/- 0.116 |
| V2_patch_k8 | patch | 0.269 +/- 0.199 | 0.165 | +0.103 | 96% | 0.710 +/- 0.131 |
| V2_patch_k16 | patch | 0.323 +/- 0.232 | 0.204 | +0.157 | 93% | 0.684 +/- 0.150 |
| V3_dynp_norm | cell | 0.175 +/- 0.154 | 0.013 | +0.008 | 68% | -- |
| V4_rank | cell | 0.166 +/- 0.155 | 0.166 | +0.000 | 40% | -- |

Sanity gate: V0 per-case Spearman = 0.166, plausible band (0.1, 0.4) -> PASS. V4 rank-identity max deviation from V0: 3.33e-16.

VERDICT: post-processing meaningfully improves localization -- V1_smooth_s4 (cell scale): 0.301 vs V0 0.166 (+0.135, 93% of cases); V2_patch_k8 (patch scale): 0.269 vs V0 0.166 (+0.103, 96% of cases); V2_patch_k16 (patch scale): 0.323 vs V0 0.166 (+0.157, 93% of cases).
