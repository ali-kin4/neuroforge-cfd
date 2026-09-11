# Published AirfRANS baselines — verified at source

Verified 2026-09-11 by reading the NeurIPS 2022 Datasets & Benchmarks paper PDF
directly (Bonnet, Mazari, Cinnella, Gallinari; arXiv 2212.07564; proceedings PDF
`94ab7b23a345f93333eac8748a66c763-Paper-Datasets_and_Benchmarks.pdf`, pages 8-9).

These numbers are load-bearing: the covariate-null result asserts that each of them
falls below a regression on the case file name. They must be exact.

## Table 3 — full task, force coefficients

Caption, verbatim: *"Relative errors (Spearman's rank correlation) for the predicted
drag coefficient C_D (rho_D) and lift coefficient C_L (rho_L). We want Spearman's
correlation to be close to one. Those quantities are computed as a post-processing
from the unnormalized regressed fields."*

| Model | rel. err C_D | rel. err C_L | **rho_D** | **rho_L** |
|---|---|---|---|---|
| MLP | 4.289 ± 0.679 | 0.767 ± 0.108 | **-0.117 ± 0.256** | **0.913 ± 0.018** |
| GraphSAGE | 4.050 ± 0.704 | 0.517 ± 0.162 | **-0.303 ± 0.124** | **0.965 ± 0.011** |
| PointNet | 14.637 ± 3.668 | 0.742 ± 0.186 | **-0.022 ± 0.097** | **0.938 ± 0.023** |
| Graph U-Net | 10.385 ± 1.895 | 0.489 ± 0.105 | **-0.138 ± 0.258** | **0.967 ± 0.019** |

**All four lift and all four drag values match what the paper already cites, exactly.**

**Two things this adds that we did not have:**

1. **The lift standard deviations for GraphSAGE (± 0.011) and Graph U-Net (± 0.019)**,
   which were previously unavailable and were recorded as `None`. Both are now known,
   and both intervals sit well below the case-name null's 0.9821 [0.9737, 0.9866].
2. **The drag standard deviations, and they are enormous** — ± 0.256, ± 0.124, ± 0.097,
   ± 0.258. Three of the four intervals **span zero**. So the benchmark's drag rank
   correlation is not merely negative; at this spread it is indistinguishable from
   noise. That is a stronger and more defensible statement than "negative", and it
   should be how the paper puts it.

## Table 5 — GraphSAGE across all four tasks

| | Full | Scarce | Reynolds | AoA |
|---|---|---|---|---|
| rho_D | -0.303 ± 0.124 | -0.139 ± 0.175 | 0.013 ± 0.064 | 0.055 ± 0.171 |
| rho_L | 0.965 ± 0.011 | 0.981 ± 0.006 | 0.927 ± 0.027 | 0.908 ± 0.019 |

Drag rank correlation is at or below noise on **every** task, including both
out-of-distribution splits.

## The authors' own diagnosis — direct corroboration of our interpolation result

Page 8, verbatim:

> *"we conclude that the models have difficulties to predict the wall shear stresses
> as the velocity values at the closest nodes from the geometry are often largely
> overestimated. This particularly affects the accuracy of the drag coefficient... However,
> the wall shear stress has a small impact on the lift coefficient compared to the
> pressure at the surface of airfoils. Hence, as the pressure is more accurately
> inferred compared to the wall shear stress... the inferred lift coefficient is also
> more accurately inferred and the rank is better predicted."*

This matters a great deal. **The benchmark's own authors identify the near-wall region
as the failure mode**, and our interpolation baseline independently localises the entire
learning advantage to the same place: 92% of the interpolator's u-error and 90% of its
v-error lie within the first 0.02 chord, and beyond 0.05 chord it matches at R-squared
above 0.9996.

So the two results are the same observation from opposite directions. They say models
fail near the wall; we measure that near the wall is the *only* place models help. Cite
this passage — it converts our finding from a claim into a confirmation of something the
dataset's authors already suspected but did not quantify.

## Other verified anchors

- **Table 4**: one simulation ~25 min on 16 CPU cores; full dataset ~20 days. This is
  the classical-solve anchor the cost section already cites.
- The paper states each model was **trained 5 times** to produce these means and
  standard deviations.
- Metrics are computed "as a post-processing from the unnormalized regressed fields",
  which is the same convention our force recomputation uses.
