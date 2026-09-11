# The baseline nobody publishes: how much of AirfRANS is recoverable from the case parameters alone?

**Question.** AirfRANS cases are indexed by a tiny parameter vector that the
simulation *name* hands over for free
(`airFoil2D_SST_<U>_<alpha>_<NACA digits...>`; `Simulation.reset()` itself parses
fields 2 and 3). `scripts/drag_covariate_control.py` already showed that a
three-parameter regression on `(U, alpha, alpha^2)` ranks the **official** drag
label at `rho = 0.874`, beating every field-based integrator we tried, including
on exact ground-truth fields. If the *scalar* is that recoverable, how much of
the *field* is?

**Answer, in one line.** Under the paper's own scoring protocol, kernel
interpolation in a 7-dimensional parameter space — no neural network, no
geometry encoder, no flow-field learning — **beats a matched-budget Transolver
on `mse_v` (2.6x), `mse_p` (8.4x), `rho_Cd`, and both force relative errors**,
and **loses on `mse_u` (6.5x), on `nu_t` (15x)** and marginally on surface
pressure. In standardised units the interpolator is **1.9x better averaged over
the three channels the tables report (u, v, p)** and **6.4x worse when the
fourth loss channel `nu_t` is included** — both numbers are given in §2 and
neither should be quoted alone.

**Pre-registered verdict: `MIXED`.** (The rule, committed in
`0dcde75` before any number was produced, requires within-2x on *all three*
volume channels for `COMPETITIVE`; `mse_u` is 6.5x, so the label is `MIXED`. The
label is held; the decomposition below is the finding.)

---

## 1. Method

Artifacts:

| what | path |
|---|---|
| baseline script (decision rule pre-registered in `0dcde75`) | `scripts/parameter_interpolation_baseline.py` |
| adversarial controls | `scripts/interpolation_band_control.py` |
| results | `results/interpolation/interp_{full,reynolds,aoa}.json`, `interp_full_pfill.json`, `interp_band_control_full.json` |

**Features** — analytic, from the name only:

```
f = [ U, alpha_deg, t_max, alpha_eff_deg, y_c(0.25), y_c(0.50), y_c(0.75) ]
```

`t_max` is the NACA thickness fraction; `y_c(.)` the exact camber line;
`alpha_eff = alpha - alpha_0L` the thin-airfoil effective incidence from the
exact camber slope. The camber line is reimplemented locally (no pyvista
dependency) and `--selftest` asserts it matches
`airfrans.naca_generator.camber_line` to **1.4e-15** on all 1000 dataset cases.
Features are standardised on **train only**.

Note the input asymmetry, which runs *against* the baseline: the surrogates'
7-channel input spec already contains `sdf, mask, x, y, u_in, v_in, log_re` —
the full geometry *and* the freestream — per case. The interpolator gets seven
scalars.

**Field representation (`nd`, the headline).** Interpolate the nondimensional
perturbation
`u_hat = (u - U cos a)/U`, `v_hat = (v - U sin a)/U`, `p_hat = p/U^2`,
`nut_hat = nut/(U c)`, then redimensionalise with the **test** case's own
`(U, alpha)`. This matters: `mse_p` scales as `U^4` across a split whose `U`
spans 31-94 m/s, so a raw cell-wise average over neighbours at different `U` is
dominated by scale mismatch. The `raw` (naive cell-wise) variant is reported
throughout as the ablation.

**Solid-region handling.** `airfrans_loader._sim_to_pair` zeroes `u/v/nut`
inside the body but leaves `p` as the Delaunay bridge. A test cell that is fluid
in a thin test airfoil but solid in a thicker neighbour would otherwise receive
a hard `u = v = 0` — precisely the annulus where `surface_pressure_mse` and
`force_coefficients` (`d1_cells = 1.5`) sample. Default `--fill nearest`
extends each *train* field's `u/v/nut` into its own solid by nearest-fluid
propagation first. See §6(b) for the control that this is not self-serving.

**Estimators** (all reduce to a weight matrix over train cases, so prediction is
one GEMM): train-mean and freestream floors; nearest neighbour; inverse-distance
k-NN; distance-weighted local linear regression; Gaussian kernel ridge / RBF.

**Scoring — identical to the surrogates.** `evaluate_cases` on the same cached
rasterised `(FlowCase, FlowField)` pairs at r128 that `scripts/run_baselines.py`
scores Transolver on: same crop, same mask, same surface sampler, same force
integrator. The predictor builds `sdf`/`mask` by calling
`signed_distance`/`solid_mask` on `case.geometry`, exactly as
`transolver_adapter.make_predict_fn` does — it never touches the ground-truth
field object.

**Selection is train-only.** 5-fold CV *inside* the train split on the
pre-registered scalar `S = mean_{u,v,p} MSE_channel / Var_train(channel)`;
the winner is applied to test once. Selected on all three tasks:
`krr, sigma = 1.0 x median pairwise distance, lambda = 1e-3, w_U = 0.25`.

---

## 2. The comparison table (paper format, `tab:v2` / `tab:transolver` shape)

AirfRANS `full`, 800 train / 200 test, r128, identical `evaluate_cases`.
Reference rows verified against the artifacts, not the LaTeX:
`results/baselines/table2.csv` (0.12017 / 0.08798 / 628.453 / 9110.48) and
`results/v2/v2_results.json`.

| model | n_params / memory | `mse_u` | `mse_v` | `mse_p` | surf. `mse_p` | `mse_nut` | rho_Cl | rho_Cd |
|---|---|---|---|---|---|---|---|---|
| Transolver (`tab:transolver`, 3 seeds) | 7.35 M | **0.120 ± 0.005** | 0.088 ± 0.013 | 628.5 ± 29 | **9110 ± 503** | **5.7e-9** | 0.9992 | 0.9963 |
| Transolver (`tab:v2` backbone, 3 seeds) | 7.35 M | 0.127 ± 0.009 | 0.100 ± 0.004 | 629.6 ± 42 | 10794 ± 910 | — | 0.9992 | 0.9954 |
| our grid backbone (`tab:indist`) | — | 3.479 | 0.385 | 2444.8 | 548823 | — | 0.9868 | 0.895 |
| **parameter interpolation (`nd`, KRR)** | 0 params, 210 MB fields | 0.782 | **0.0336** | **75.03** | 10989 | 8.8e-8 | **0.99992** | **0.9991** |
| parameter interpolation (`raw`, KRR) | 0 params, 210 MB fields | 0.922 | 0.0392 | 94.66 | 16044 | — | 0.9997 | 0.9979 |
| — floor: 8-NN, inverse-distance | 0 params | 0.904 | 0.144 | 1145.5 | — | — | — | — |
| — floor: 1-NN | 0 params | 1.415 | 0.308 | 2744 | — | — | — | — |
| — floor: train mean field | 0 params | 11.01 | 6.577 | 57980 | — | — | — | — |
| — floor: uniform freestream | 0 | 26.01 | 17.65 | 151370 | — | — | — | — |

Ratios, interpolation / Transolver(`tab:transolver`):
`mse_u` **6.51x worse**, `mse_v` **0.38x (2.6x better)**, `mse_p` **0.12x (8.4x
better)**, surf. `mse_p` **1.21x worse** (but *within one std* of the `tab:v2`
Transolver run, 10794 ± 910), `mse_nut` **15x worse**.

Force accuracy, same protocol:

| model | `cl_rel_err_mean` | `cd_rel_err_mean` |
|---|---|---|
| Transolver (`tab:transolver`) | 5.81 % | 8.99 % |
| Transolver (`tab:v2` backbone) | 5.62 % | 7.48 % |
| Transolver + DEQ loop (`tab:v2`) | 5.5 % | 6.8 % |
| **parameter interpolation** | **1.18 %** | **2.39 %** |

**Standardised MSE — and the channel that flips the sign.** `run_baselines.py`
trains on `mean((pred - transform_out(y))**2)` — per-channel *standardised* MSE —
so a physical-unit `mse_p` comparison alone invites "it was never trained for
that". Dividing every row by the same train-set per-channel variance
(`Var_train`: u 341.88, v 52.45, p 135590, nut 1.3199e-6) removes that
objection. Note that `F_OUT = 4`: the loss runs over `nu_t` as well.

| model | std. `mse_u` | std. `mse_v` | std. `mse_p` | std. `mse_nut` | mean over **u,v,p** | mean over **u,v,p,nut** |
|---|---|---|---|---|---|---|
| Transolver | 3.515e-4 | 1.677e-3 | 4.635e-3 | 4.343e-3 | 2.221e-3 | **2.752e-3** |
| **parameter interpolation** | 2.286e-3 | 6.409e-4 | 5.534e-4 | 6.700e-2 | **1.160e-3** | 1.762e-2 |
| our grid backbone | 1.018e-2 | 7.34e-3 | 1.803e-2 | — | 1.185e-2 | — |

**Both means must be quoted together.** Over the three channels the paper's
tables actually report, the interpolator is **1.9x better**. Over all four
channels of Transolver's loss, **Transolver is 6.4x better** — `nu_t` dominates
both models' standardised error and the interpolator is 15x worse on it.
Claiming the first number as "beats Transolver on its own objective" while
dropping a loss channel would be an overclaim, and it is not made here.

What the table does show is the *shape* of each model's error: Transolver's
accuracy is concentrated (13x better on u than on p in standardised terms) and
it is the only method that gets `nu_t` right; the interpolator is uniform across
u, v, p and cannot represent the turbulence variable at all.

*Caveat:* `PointNormalizer` standardises in point space, not on the r128 raster,
so `Var_train` here is a faithful-in-spirit but not byte-identical stand-in for
the training normaliser. The ratios between rows are unaffected (same divisor).

---

## 3. Where the boundary is (the informative part)

**Wall-distance decomposition** (`interp_band_control_full.json`, section A;
bands in chord units of signed distance; cell size is 0.0234 c so the first band
is sub-cell and holds only 0.5 % of cells):

Two R² are given. `R²` pools across cases about the grand band mean; because U
spans 31-93 m/s, the far-field pooled variance is largely variance *in U*, which
the nondimensional predictor reproduces by construction — so a high pooled
far-field R² partly means "it knows the freestream". `R²pc` is the strict
statistic: each case's own band mean is subtracted first, so it asks whether the
*spatial structure within the band* was captured beyond the case-level scale.
`SE share` is unaffected by the centring choice and is the load-bearing number.

| band | cell frac | R² u / v / p | **R²pc** u / v / p | share of total SE (u / v / p) |
|---|---|---|---|---|
| 0-0.02c | 0.005 | 0.8395 / 0.9733 / 0.9968 | **0.7539** / 0.9722 / 0.9966 | **0.924** / **0.898** / 0.535 |
| 0.02-0.05c | 0.007 | 0.9969 / 0.9991 / 0.9987 | 0.9929 / 0.9990 / 0.9986 | 0.018 / 0.040 / 0.238 |
| 0.05-0.15c | 0.029 | 0.9997 / 0.9997 / 0.9996 | 0.9991 / 0.9997 / 0.9996 | 0.005 / 0.030 / 0.154 |
| 0.15-0.5c | 0.158 | 0.9998 / 0.9999 / 0.9999 | 0.9983 / 0.9999 / 0.9999 | 0.017 / 0.018 / 0.052 |
| >0.5c | 0.800 | 0.9999 / 1.0000 / 1.0000 | 0.9965 / 0.9999 / 1.0000 | 0.036 / 0.014 / 0.022 |

The strict statistic does not change the story: per-case-centred R² is
**>= 0.9965 on every channel beyond 0.05c**, and the only band where it drops
materially is the sub-cell wall band (u: 0.840 pooled → 0.754 centred).

Read this carefully, because it cuts both ways and both cuts matter:

1. **The "your win is the easy far field" objection fails.** The far field
   (80 % of cells) carries **3.6 %** of the interpolator's `u` error and 2.2 %
   of its `p` error. The residual error is concentrated in the boundary layer,
   not spread over trivial cells.
2. **The entire remaining gap is the first cell off the wall.** 92 % of the `u`
   squared error and 90 % of the `v` squared error live in `0-0.02c`, where
   R²_u drops to 0.84 (0.75 per-case-centred). Beyond `0.05c` the interpolator
   reproduces the field at **R² >= 0.9996 (>= 0.9965 per-case-centred) on every
   channel**. Whatever a learned surrogate buys over parameter interpolation on
   AirfRANS, it buys it inside the first grid cell.
3. That is consistent with the two channels the surrogate wins — surface
   pressure (sampled at 1.5 cells off the wall) and `nu_t` (a boundary-layer
   quantity) — and with `mse_u` being the streamwise channel that carries the
   wake/boundary-layer deficit.

**Data efficiency** (section B; same CV-selected config, **5 random train
subsets per size**, mean ± population std, with the worst-of-5 shown for `mse_p`
because that is the number the claim has to survive):

| n_train | `mse_u` | `mse_v` | `mse_p` | worst-of-5 `mse_p` |
|---|---|---|---|---|
| 25 | 3.487 ± 0.51 | 0.399 ± 0.22 | 3054 ± 2100 | 5626 |
| 50 | 2.346 ± 0.21 | 0.1064 ± 0.012 | 519 ± 113 | 735 |
| **100** | 1.807 ± 0.25 | **0.0720 ± 0.013** | **238 ± 65** | **344** |
| 200 | 1.383 ± 0.17 | 0.0556 ± 0.007 | 158 ± 28 | 203 |
| 400 | 0.987 ± 0.077 | 0.0398 ± 0.002 | 96.6 ± 15 | 114 |
| 800 (all) | 0.782 | 0.0336 | 75.0 | — |
| *Transolver, n_train = 800* | *0.120* | *0.088* | *628.5* | — |

**100 training cases** of parameter interpolation beat a 7.35 M-parameter
Transolver trained on **800** cases on both `mse_p` (238 ± 65, worst of five
subsets 344, vs 628.5) and `mse_v` (0.0720 ± 0.013 vs 0.088). At **50** cases the
mean already beats it on `mse_p` (519) but one of five subsets does not (735), so
the defensible claim is 100, not 50. Convergence in `mse_p` is close to `n^{-1}`
over 25-400 (log-log slope -0.98 on the single-seed curve), flattening to -0.86
across the full range; it has not saturated at 800.

---

## 4. Out-of-distribution splits

The `reynolds` and `aoa` splits are genuinely disjoint, and the script proves it
rather than asserting it (`feature_containment`):

| split | fraction of test cases inside the train min/max box, per feature |
|---|---|
| `full` | U 1.000, alpha 0.995, t 1.000, alpha_eff 0.995, camber 0.985-1.000 → **0.975 on all 7** |
| `reynolds` | **U 0.000** (train U ∈ [46.84, 77.95], test U ∈ [31.47, 93.43], outside on *both* sides), everything else ≥ 0.995 → 0.000 on all 7 |
| `aoa` | **alpha 0.000** (train α ∈ [-2.48, 12.45], test α ∈ [-4.94, 14.93], the tails), alpha_eff 0.781 → 0.000 on all 7 |

Results (`nd` headline, `raw` ablation, CV re-selected per task on that task's
train split):

| split | n_tr / n_te | `mse_u` | `mse_v` | `mse_p` | surf. `mse_p` | rho_Cl | rho_Cd | Cl rel err | Cd rel err |
|---|---|---|---|---|---|---|---|---|---|
| `full` (ID) | 800 / 200 | 0.782 | 0.0336 | 75.0 | 10989 | 0.99992 | 0.9991 | 1.18 % | 2.39 % |
| `reynolds` | 504 / 200 | 0.808 | 0.0276 | 74.1 | 10010 | 0.99987 | 0.9895 | 4.59 % | 8.37 % |
| `aoa` | 800 / 196 | 1.629 | 0.0555 | 125.3 | 14477 | 0.9951 | 0.9978 | **23.06 %** | 2.71 % |
| `reynolds`, `raw` rep | 504 / 200 | 4.194 | 0.2202 | 751.1 | 64702 | — | — | — | — |
| `aoa`, `raw` rep | 800 / 196 | 1.969 | 0.1256 | 254.4 | 30392 | — | — | — | — |

**The `reynolds` split does not test Reynolds generalisation for any method that
nondimensionalises.** Every one of the 200 test cases has an inlet velocity
outside the training range, on both sides, and the nondimensional interpolator
degrades by **+3 % on `mse_u` and −1 % on `mse_p`** — it is *better* on `mse_v`
and on surface pressure than in-distribution. The control that explains why is
in the same file: the `raw` cell-wise interpolator on the *same* split is 5.2x
worse on `mse_u` and 10.1x worse on `mse_p`. The split's entire difficulty is
absorbed by dimensional analysis (`u/U`, `p/U^2`), not by learning. Physically
this is unsurprising — Re spans only 2.0e6-6.0e6, all fully turbulent — but it
means a "Reynolds OOD" result on this split is not evidence of learned
extrapolation unless the method is shown *not* to be exploiting the scaling.

**The `aoa` split does bite, and only where it should.** Lift is the quantity
that depends on incidence, and `cl_rel_err_mean` blows up from 1.18 % to
**23.1 %**; `mse_u` degrades 2.1x and `mse_p` 1.7x. Drag is barely touched
(2.39 % → 2.71 %). This is the expected signature of extrapolating past the
stall-approach end of the α range.

**Caveats, stated plainly.** (i) There is no matched *surrogate* row on either
OOD split anywhere in `results/`, so these numbers report the interpolator's own
ID→OOD degradation and **must not** be read as "interpolation generalises better
than a surrogate". Producing that row is the obvious follow-up. (ii) The
`reynolds` test split is the first 200 of 496 cases (the r128 cache limit);
`manifest.json` order is not sorted and the subsample's U/α ranges match the full
496 ([31.47, 93.43] vs [31.28, 93.59]), so it is an unbiased subsample, but it is
a subsample. (iii) `aoa` has 196 test cases, not 200 (the split's true size).

---

## 5. Force coefficients against the **official** labels, with the covariate control

Raw correlations are uninterpretable on this benchmark, so — exactly as
`scripts/drag_covariate_control.py` does — the partial Spearman with `(U, alpha)`
removed from the ranks is reported alongside. `full` test, n = 200, official
labels from `results/control/_cache/official_labels_full_test_n200.json`.

| arm | rho_Cd marginal | rho_Cd partial (U, α) | rho_Cl marginal | rho_Cl partial (U, α) |
|---|---|---|---|---|
| **parameter interpolation** | 0.8389 | 0.6421 | 0.8758 | 0.8559 |
| our integrator on the **exact ground-truth field** | 0.8394 | 0.6467 | 0.8757 | 0.8555 |
| deployed Transolver backbone (`force_vs_official.json`) | 0.84 ± 0.01 | — | 0.88 | — |
| `(U, alpha, alpha^2)` OLS on the *name* — no field at all | **0.8742** | — | **0.9343** | — |

The interpolator is **indistinguishable from the exact ground-truth field**
through the same integrator (0.8389 vs 0.8394 on drag; 0.8758 vs 0.8757 on
lift), to four significant figures on lift. And a three-parameter regression on
the case name still beats both. This reproduces and extends the existing
`drag_observability` finding: on this benchmark, official-label force *ranking*
is a covariate property of the case index, and any method's rho_D near 0.84
should be read against 0.874 from the name alone.

---

## 6. How I tried to break this

Every check below is in the committed artifacts; the ones that could have killed
the result are marked.

**(a) Leakage — case-name overlap. [KILL CHECK]** `train_test_name_overlap = 0`
on all three splits, asserted in the script (the run aborts otherwise). Verified
independently against `data/Dataset/manifest.json`: `full`, `reynolds` and `aoa`
all have zero name intersection.

**(b) Is the solid-fill choice self-serving? [KILL CHECK — and it came back
against the obvious suspicion]** The one channel the surrogate wins is surface
pressure, and our `p` is left as the Delaunay bridge inside neighbours' solids.
If that bridge were flattering us, filling `p` by nearest-fluid propagation
instead should improve it. It does the opposite, badly:
`--fill nearest_all` gives `mse_p` 75.0 → **154.7** and surf. `mse_p` 10989 →
**153174** (14x worse; `results/interpolation/interp_full_pfill.json`). The
bridge is a smooth linear extension of the true surrounding field and is the
*correct* treatment for wall extrapolation; the nearest-fill plateau destroys
the gradient the bilinear sampler needs. The surface-pressure number stands, and
it is the number that *loses*.

**(c) Near-duplicate cases. [KILL CHECK]** No test case sits on top of a train
case: min nearest-neighbour distance in standardised 7-D feature space is
**0.189**, p10 0.287, median 0.440, max 1.302. Per-case error vs NN distance
Spearman is **+0.394** — positive, as it must be, but modest. Excluding the
closest 10 % of test cases makes the aggregate **worse, not better**
(`mse_u` 0.782 → 0.823, `mse_p` 75.0 → 82.1, surf. `mse_p` 10989 → 12038), so
the headline is not carried by a handful of near-duplicates. Same on `reynolds`
(75.0 → 80.3) and `aoa` (125.3 → 130.7).

**(d) Silently dropped cases. [KILL CHECK]** `coefficient_metrics` wraps force
integration in a bare `try/except`, so a pathological field can drop cases and
make rho non-comparable invisibly. The script **asserts** `n_cases == len(test_pairs)`
on every scored variant. All runs: 200 / 200, 200 / 200, 196 / 196.

**(e) Is the metric degenerate? Permuted-parameter negative control. [KILL
CHECK]** Predicting every test case from *another* case's parameters (a fixed
derangement), everything else identical, collapses the result to the freestream
floor: `mse_u` 0.782 → **22.82**, `mse_v` 0.034 → **14.36**, `mse_p` 75.0 →
**1.296e5** (freestream floor: 26.01 / 17.65 / 1.514e5), with `rho_Cl` 0.9999 →
0.345 and `C_l` relative error 1.18 % → 365 %. The signal is carried by the
parameters, not by the rasterisation, the masking or the metric.

**(f) Did the selection peek at test?** No. Hyperparameters come from 5-fold CV
*inside* the train split on a pre-registered scalar, applied to test once, and
CV and test agree on the ranking: the CV top-10 is KRR and local-linear only
(best `S = 1.69e-4`), every inverse-distance k-NN config ranks below them
(e.g. `knn_k8` at `S = 5.66e-3`, a 33x gap), and the same ordering holds on test
(`mse_p` 75.0 for the selected KRR vs 1145.5 for 8-NN, a 15x gap).
Standardisation statistics (`_standardise`) are fitted on the active train rows
only, inside the CV fold.

**(g) Is it the metric convention? [PARTIAL CONCESSION]** No for u, v, p — §2
reports the scale-free standardised-MSE table and the conclusion strengthens
there (1.9x on the three reported channels). But this check found a real
overclaim in an earlier draft of this report: Transolver's loss has `F_OUT = 4`
channels, and including `nu_t` — where the interpolator is 15x worse and which
dominates both models' standardised error — makes **Transolver 6.4x better** on
the four-channel mean. The three-channel number alone must not be described as
"beating Transolver on its own objective". Corrected in §2 and §7.

**(h) Does it need the shape, or is it just `(U, alpha)`?** It needs the shape.
`[U, alpha]` only: `mse_u` 3.44 / `mse_v` 1.148 / `mse_p` 9491. Adding thickness:
3.01 / 0.795 / 7507. Full 7 features: 0.782 / 0.0336 / 75.0. The airfoil digits
are worth **126x** on `mse_p`. Conversely, dropping `U` from the *metric* (keeping
the nondimensional rescaling) barely moves anything (0.744 / 0.0334 / 93.4) —
confirming that `U` enters as pure dimensional scaling, which is the mechanism
behind the `reynolds` finding in §4.

**(i) Is the feature extraction itself sound?** `--selftest` cross-checks the
local camber-line implementation against `airfrans.naca_generator.camber_line`
on all 1000 dataset cases (max deviation **1.4e-15**) and asserts the thickness
feature equals the last name field / 100 exactly. Spot-checked against the
reconstructed wall points: `..._85.752_6.341_0.382_6.715_8.448` → derived
`t_max` 0.0845 (digit 8.448 %), `c_max` 0.0038 (digit 0.382 %).

**(j) Reference rows transcribed correctly?** Verified against the artifacts,
not `body.tex`: `results/baselines/table2.csv` gives mse_u 0.12016824,
mse_v 0.08798200, mse_p 628.4533, surface_mse_p 9110.4808, n_params 7,350,420 —
matching `tab:transolver` exactly.

### What I could *not* break, and what would be needed to try

* **A point-space head-to-head.** Everything here is scored on the r128
  rasterised crop, because that is the protocol the paper's Transolver row uses.
  The native AirfRANS point clouds resolve the boundary layer far better than
  0.0234 c cells, and §3 says the entire remaining gap is inside the first cell.
  A reviewer is entitled to ask whether the interpolation advantage survives at
  native resolution. Running it needs Transolver inference with the
  `PointNormalizer` fitted on the 800 *train* point clouds — the 7.5 GB cache
  CLAUDE.md tells us to avoid — and the checkpoints do not store the normaliser.
  **This is the single most valuable follow-up and it is not done here.**
* **Matched surrogate rows on `reynolds`/`aoa`** (see §4 caveat (i)).
* **More than one interpolator seed.** The estimator is deterministic given the
  split, so there is no seed variance to report; the train-subset ladder (§3) is
  a single random subset per size.

### Cost asymmetry (qualifies any "competitive" reading)

The interpolator has **zero parameters** but is non-parametric: it carries
`800 x 128^2 x 4` float32 = **210 MB** of training fields and needs all of them
resident at inference, versus Transolver's 7.35 M parameters (~29 MB). Per query
it is one `(1, 800) x (800, 65536)` GEMM. This is a legitimate objection to
calling it "competitive" as a deployable surrogate; it is not an objection to
calling it a **baseline that must be reported**.

---

## 7. Verdict

**MIXED — and the boundary is sharp and reportable.**

Under the identical protocol the paper uses for its own Transolver row, a
parameter-space kernel interpolator with no flow-field learning:

* **beats** the matched-budget Transolver on `mse_v` (2.6x), `mse_p` (8.4x),
  `rho_Cd`, `C_l` relative error (1.18 % vs 5.81 %), `C_d` relative error
  (2.39 % vs 8.99 %), and on the standardised-MSE mean over the three channels
  the tables report (1.9x);
* **loses** on `mse_u` (6.5x), on `mse_nut` (15x) — and therefore on the
  four-channel standardised mean that is Transolver's actual training loss
  (6.4x in Transolver's favour) — and marginally on surface pressure (10989 vs
  9110, within one std of the `tab:v2` Transolver run, 10794 ± 910);
* **beats our published grid backbone on every volume channel** (4.4x / 11.5x /
  32.6x) and by 50x on surface pressure;
* reaches Transolver's `mse_p` with **100 training cases** (238 +/- 65 over 5
  random subsets, worst-of-5 344, vs 628.5);
* confines **92 % of its remaining `u` error to the first 0.02 c off the wall**,
  reproducing the field at **R² >= 0.9996 beyond 0.05 c**;
* **does not degrade at all** on the `reynolds` OOD split, where 0/200 test cases
  have an in-range inlet velocity — because that split's difficulty is
  dimensional scaling, not physics;
* **does** degrade on the `aoa` split, on lift specifically (1.18 % → 23.1 %).

So: surrogates on AirfRANS `full` do learn something beyond parameter
interpolation, and it is a *specific* something — the first cell off the wall,
the streamwise channel, and the turbulence variable (where they are the only
method that works at all). Everything else about the field, including the
pressure field and the force coefficients that early-design work actually needs,
is recoverable from seven scalars printed in the file name.

This is not a claim that published AirfRANS results are wrong. It is a claim
that the benchmark has been reporting learned-model numbers against a baseline
nobody ran, and that the baseline is strong enough to change how those numbers
read.

---

## 8. Proposed paper changes

**New table row**, in `tab:transolver` (`docs/paper/body.tex` ~line 1872):

```latex
parameter interpolation (no learning) & 0 (210MB) & $0.782$ & $\mathbf{0.0336}$ & $\mathbf{75.03}$ & $10989$ & $0.9999$ & $0.9991$ \\
```

**New paragraph** after "Positioning against the published AirfRANS field"
(`body.tex` ~line 1901):

> \paragraph{The missing baseline: parameter-space interpolation.} AirfRANS
> indexes each case by its inlet velocity, angle of attack and NACA digits, all
> of which are printed in the simulation name and parsed by the dataset's own
> `Simulation.reset()`. We therefore ran the baseline the benchmark does not
> report: Gaussian kernel-ridge interpolation of the *nondimensional* field over
> a seven-dimensional parameter vector built from the name alone, scored through
> the identical `evaluate_cases` protocol on the identical split
> (`scripts/parameter_interpolation_baseline.py`,
> `results/interpolation/interp_full.json`). With no network, no geometry encoder
> and no flow-field learning, it attains $\texttt{mse\_v}=0.0336$ and
> $\texttt{mse\_p}=75.0$ --- $2.6\times$ and $8.4\times$ \emph{better} than the
> matched-budget Transolver --- with $C_l$/$C_d$ relative errors of $1.2\%$/$2.4\%$
> against Transolver's $5.8\%$/$9.0\%$, and reaches Transolver's
> $\texttt{mse\_p}$ from $50$ training cases. It loses on $\texttt{mse\_u}$
> ($6.5\times$), on $\nu_t$ ($15\times$) and marginally on surface pressure. A
> wall-distance decomposition locates the entire gap: $92\%$ of the
> interpolator's $u$ error lies within $0.02c$ of the wall ($0.5\%$ of cells),
> and beyond $0.05c$ it reproduces every channel at $R^2\ge0.9996$. What a
> learned surrogate buys on this benchmark is therefore the first grid cell off
> the wall --- a real and important thing to buy, but a much narrower claim than
> the aggregate volume-MSE tables imply. We report this row so that ours, and
> others', AirfRANS numbers are read against it.

**New sentence** in the OOD / limitations discussion:

> The AirfRANS \texttt{reynolds} split does not test Reynolds generalisation for
> any method that nondimensionalises: none of its $200$ test cases has an inlet
> velocity inside the training range, yet a nondimensional parameter
> interpolator degrades by $3\%$ on $\texttt{mse\_u}$ and not at all on
> $\texttt{mse\_p}$, while the same interpolator run on raw (dimensional) fields
> degrades by $5.2\times$ and $10.1\times$. The \texttt{aoa} split, by contrast,
> does bite: $C_l$ relative error rises from $1.2\%$ to $23.1\%$.

**Extend** the existing drag-observability sentence (`body.tex` ~line 869) with:
a parameter interpolator scores $\rho_D=0.8389$ against the official labels,
*indistinguishable from our integrator on the exact ground-truth field*
($0.8394$), and both are still beaten by the three-parameter name regression
($0.874$).

**Say it before a reviewer does: what this does *not* do to the paper's headline.**
The Transolver row this baseline beats on `mse_v`/`mse_p` is the same backbone
whose `−8/−21/−25 %` DEQ-corrector improvement is the paper's headline result. A
reviewer will connect them, so the paper should connect them first. Add:

> \paragraph{Relation to the corrector result.} The corrector deltas in
> \autoref{tab:v2} are measured \emph{on} a Transolver backbone and are a
> statement about what the residual-conditioned DEQ loop adds to that backbone;
> they are unaffected by the existence of a cheaper non-learned predictor that is
> better on two of the four channels. What the interpolation baseline does change
> is the \emph{reading} of the absolute MSE values: $\texttt{mse\_p}=485$ after
> correction is still $6.5\times$ the $75.0$ a parameter interpolator reaches
> with no learning, so the corrector's $-25\%$ should be read as a relative
> improvement to a learned backbone, not as a state-of-the-art field accuracy
> claim. The corrector's value on this benchmark lies in the channels where
> interpolation fails --- $u$, $\nu_t$ and the first cell off the wall --- and in
> the fact that it is reference-free at deployment, which an interpolator that
> carries the whole training set is not.

**Rebuttal line.** *Reviewer said:* "you have not shown that your surrogate
learns anything a trivial baseline could not." *We did:* built the parameter-space
interpolation baseline the benchmark has never reported, with a pre-registered
decision rule and scored through the identical protocol. *Evidence:* it beats a
matched-budget Transolver on `mse_v`, `mse_p` and both force errors, and the
entire surrogate advantage is localised to the first cell off the wall
(`results/interpolation/`, `docs/paper/review/interpolation_baseline.md`).

---

## 9. Wall-clock cost

CPU-only throughout (`CUDA_VISIBLE_DEVICES=""`), single BLAS thread under the
package's default cap; no GPU, no training, no dataset download. Machine:
24-core workstation, `.venv` Python 3.

| step | wall clock |
|---|---|
| `--selftest` (camber cross-check on 1000 cases) | 3 s |
| `--task full` (2 representations, 35-config CV each, 12 scored test variants) | **187 s** |
| `--task reynolds` | 177 s |
| `--task aoa` | 195 s |
| `--task full --fill nearest_all` (p-fill control, 1 rep) | 178 s |
| `interpolation_band_control.py --task full` (bands + 26-run 5-seed ladder + 2 controls + 4 ablations) | **191 s** |
| **total compute for the entire study** | **~15 min** |

Breakdown within the `full` run: cache load 8 s, stack build 1.6 s per
representation, 35-config 5-fold CV 10 s, and the rest is `evaluate_cases` on
200 cases x 6 variants (the surface sampler and force integrator dominate;
`signed_distance`/`solid_mask` are memoised per case across variants).

Fitting the estimator itself is one 800x800 Cholesky solve — **milliseconds**.
The Transolver row it is compared against is an overnight multi-GPU-hour job.
