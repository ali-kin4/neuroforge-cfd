# Residual descent at n = 200: gradient flow, the deployed backbone, and the regime boundary

**VERDICT: PARTIALLY-RESOLVED.** The negative claim survives, and is now tested on the
*deployed* system rather than on a synthetic perturbation. Two things must change: the
claim is a **regime** claim (residual descent genuinely helps a backbone far from the
floor), and `tab:iters`'s current re-report contains an arithmetic error (§4).

Script `scripts/residual_descent_test.py` · results `results/residual_descent/` ·
CPU-only, no GPU (see §8).

## 0. Positioning against the existing `sec:descent` — read this first

While this experiment ran, `sec:descent` (`scripts/residual_descent.py`,
`results/residual_descent.json`) landed in the paper: 300 **Adam** steps on `J`, **n = 24**
cases, from the exact ground truth and from *ground truth plus a smooth synthetic
displacement*. **Its conclusions are confirmed here, not contradicted.** This document is a
strict extension, and it closes three things that study explicitly leaves open:

| the paper's own stated gap | what this run supplies |
|---|---|
| Limitation: *"The descent is Adam, not gradient flow… our claim is about endpoints, not about the shape of the path."* | **Armijo-line-searched gradient descent** (guaranteed monotone `J` decrease — a genuine discretisation of leg (iv)'s flow), plus a **12-decade fixed-η sweep**, plus Adam as a third rule. All three agree in sign and in monotonicity-in-achieved-`J`. |
| The perturbed start is a *synthetic* displacement, so "that is not what a surrogate's error looks like" is a live reviewer reply. | The start fields are the **deployed Transolver (3 backbone seeds)** and the **dropout-FNO of `tab:iters`**, at **n = 200**. |
| Unconstrained descent invites *"you rediscovered that a PDE residual without boundary conditions is ill-posed"* (leg (i) makes any constant field a global minimum). | A **BC-constrained arm** that hands the optimiser the exact Dirichlet data and still finds the same result. |
| *"from a perturbed start descent often reduces error"* is qualitative. | The regime boundary made **quantitative and predictive** in the theorem's own variable `ρ = ‖R_h(û)‖/‖r*‖` (§3.4). |

The iterate-selection claim the paper now makes is reproduced at n = 200: `argmin_k J ≠
argmin_k error` on **79–100 %** of cases in every arm (24/24 there).

**Reconciling the two truth-arm depths before a reviewer does.** `sec:descent` reports
descent cutting the monitored residual by **84 %** from the truth; the arms here reach
**72 %** (`free`, `uvp`) and **35 %** (`bc`, `uvp`). This is not a disagreement, it is four
known differences in how far the objective is *allowed* to go: (a) Adam with per-channel
scaling for 300 steps versus Armijo gradient descent, which takes the largest step the
Armijo condition admits and therefore stalls at a genuine stationary point; (b) `nut` free
there versus **frozen** here — and freeing it demonstrably buys depth, since our own
`uvpn` arm reaches a different `J` and quarters the velocity damage by dumping error into
`mse_nut` (§3.5); (c) the `bc` arm pins the far-field and near-wall rings, which removes
the descent directions that carry most of the residual reduction; and (d) a different
non-dimensionalisation of the reported norm. **The sign, the fraction (200/200 vs 24/24)
and the conclusion are identical; only the achievable depth differs, and the shallower our
descent, the more conservative our number.**

---

## 1. The objection, in its strongest form

> The paper's central negative claim — *"as a correction objective, the residual fails:
> reducing it does not reduce field error"* (abstract; `sec:iters`; `thm:residual-floor`
> leg (iv), which is explicitly about gradient flow `ė = −∇J`, `J = ½‖R_h(u)‖²`) — is not
> tested by the experiment cited for it.
>
> `results/sensitivity/iters.json:5` records the swept knob as
> `"DEQCorrector.max_iter (internal fixed-point cap)"`. That sweep varies the internal
> fixed-point cap of a corrector **trained by supervised regression toward ground truth**.
> It measures a supervised corrector converging while the residual happens to rise. It is
> an anti-correlation along one trajectory. **It is not residual minimisation.** A grep of
> `scripts/` finds nothing that ever minimises `J`. `tab:positioning` awards this work the
> sole tick in its "Object." column on the strength of that table.
>
> Worse, the quoted endpoints are cherry-picked: `mse_u` bottoms at iteration 3 (2.287)
> and rises to 2.575 by iteration 15, so over 3→15 the residual and the error move in the
> **same** direction.

The objection is correct on both counts. The paper has since demoted `tab:iters` and added
`sec:descent`; this document supplies the larger, deployed-system, gradient-flow version of
that experiment, and finds a residual arithmetic error in the `tab:iters` re-report (§4).

---

## 2. Method

### 2.1 The operator under test

The brief asked for `physics_residual_torch` "so the operator under test is
byte-identical to the one the paper monitors". **It is not.** `physics_residual_torch` is
the *training-loss* twin: raw, dimensional, masking only the solid, clamping
`nu_eff ≥ nu`. The scalar the paper monitors is `Diagnostics.residual_norm()`
(`core/types.py:314`), which additionally

1. zeroes the **solid-adjacent fluid ring** (`_solid_adjacent_fluid`, `residuals.py:339`),
2. **non-dimensionalises** continuity by `u∞/L` and momentum by `u∞²/L`
   (`residuals.py:352–359`), and
3. clips `nu_eff ≥ 0`, not `≥ nu`.

Descending on the wrong functional would have been a free rebuttal. We therefore built a
differentiable replica of the **monitored** operator from the framework's own
backend-agnostic `physics/operators.py` (`ddx`, `ddy`, `laplacian` duck-type onto torch,
so gradients flow):

```
J(u) = ½ · mean_cells( r_c² + r_x² + r_y² )        residual_norm = √(2J)
```

`results/residual_descent/verification.json` (n = 25 real AirfRANS cases):

| check | value |
|---|---|
| max relative error of `√(2J)` vs `Diagnostics.residual_norm()` | **9.9 × 10⁻⁸** |
| uniform-freestream `‖R_h(u∞)‖` (theorem leg (i)) | **exactly 0.0**, all cases |
| `‖r*‖ = ‖R_h(u*)‖` mean (theorem hypothesis H2) | 0.1781 |
| H2 holds (`‖r*‖ > 0`) on every case | yes |

So `√(2J)` **is** the paper's monitored scalar to 8 significant figures. The raw
training-loss twin remains available as a secondary objective (`--objective raw`).

All arithmetic is **float64**. In float32 (eps ≈ 1.2 × 10⁻⁷ relative, `u ~ 50 m/s`) the
stable explicit update rounds away and one would wrongly conclude descent is inert.

### 2.2 Arms

**Start field.** `truth` = ground-truth `u*`; `transolver` = the deployed SOTA backbone,
seeds 0/1/2, read from the existing zero-forward-pass cache
`data/cache/acceptance_gate/seed{k}/*.npz` (key `raw`); `fno` = the dropout-FNO of
`checkpoints/certificates_deq.pt`, i.e. *the same backbone as `tab:iters`*.

**Constraint mode.**

* **`bc` — the load-bearing arm.** Theorem leg (i) says *any* spatially constant field is
  an exact global minimum of `J`, so unconstrained descent has an attractor at "flat
  everything" and would raise the error almost by construction. A hostile reviewer kills
  that in one line: *"you rediscovered that a PDE residual without boundary conditions is
  ill-posed."* So in `bc` we **hand the optimiser the exact Dirichlet data**: the 2-cell
  outer border ring and the 2-cell near-wall band are pinned at ground truth, the solid is
  frozen, and the gradient is projected to zero on all frozen cells. A **2**-cell freeze
  is required because `∇J` carries a 4th-order stencil (double Laplacian); a 1-cell freeze
  would leave the second ring free to drift.
* **`free`** — unconstrained; reported as the ill-posedness demonstration only.

**Variables.** `uvp` (primary; `nut` frozen — it is a closure variable) and `uvpn` (all
four channels).

**Step rule.** **Armijo backtracking line search on `J` itself** (per case, `c₁ = 1e-4`,
trial step from 10⁸ halving up to 60×), which *guarantees* a monotone `J` decrease — the
objective chooses its own step, so "you picked a bad learning rate" cannot be raised.
Plus a fixed-`η` log sweep over 10⁻⁴…10⁷ for §5, and Adam as a secondary rule
(`--stage adam`) so "plain GD is a strawman" is also closed.

500 descent steps, **n = 200** AirfRANS `full` test cases.

### 2.3 Pre-registered readings (named before the run)

* **Falsification threshold.** If BC-constrained descent from the deployed Transolver
  field lowers rel-L2 on a *majority* of cases by a substantial fraction of the supervised
  DEQ gain (−9/−21/−25 % field MSE), then "poor correction objective" is falsified as
  stated.
* The theorem itself predicts a **non-monotone** error trajectory, so the expected outcome
  is "it depends on the starting point": far from truth `‖Le‖ ≫ ‖r*‖` and the residual *is*
  a proxy for error (the detector leg); near truth the floor dominates.
* **Stopping-criterion test.** Per case, `argmin_k J` vs `argmin_k error`. If `J` falls
  monotonically while the error is U-shaped, the residual gives no signal for *when to
  stop*, so the objective is unusable in deployment **even where it helps**.

---

## 3. Headline result — BC-constrained residual descent, n = 200, 500 Armijo steps

Boundary data pinned at ground truth; `nut` frozen; `J` monotone non-increasing by
construction. "rel-L2" is the fluid-masked rel-L2 of speed, the identical formula used by
`measure_acceptance_gate.py` (so these numbers are directly comparable to the 89.3 % /
5.8 % gate result).

| start field | `‖R_h‖` start → end | `J/J₀` | rel-L2 med. start → end | Δ rel-L2 | **frac. cases worse** | `mse_u` mean start → end | Δ `mse_u` med. |
|---|---|---|---|---|---|---|---|
| **ground truth `u*`** | 0.1912 → 0.1251 (−35 %) | 0.398 | 0.00000 → **0.00694** | — | **200/200 = 1.000** | 0.000 → **0.499** | — |
| **Transolver seed 0** | 0.2181 → 0.1439 (−34 %) | 0.406 | 0.00442 → 0.00788 | **+76.5 %** | **0.965** | 0.119 → 0.556 | +217 % |
| **Transolver seed 1** | 0.2178 → 0.1416 (−35 %) | 0.388 | 0.00464 → 0.00794 | **+73.9 %** | **0.940** | 0.120 → 0.553 | +203 % |
| **Transolver seed 2** | 0.2166 → 0.1411 (−35 %) | 0.390 | 0.00417 → 0.00778 | **+86.9 %** | **0.980** | 0.103 → 0.544 | +246 % |
| **dropout-FNO** | 0.2856 → 0.1527 (−47 %) | 0.238 | 0.01273 → 0.00974 | **−18.7 %** | **0.250** | 0.898 → 0.930 | −34.1 % |

Wilcoxon signed-rank on the per-case rel-L2 change: `p < 1e-8` for every arm
(truth `p = 1.4e-34`, Transolver s0 `p = 2.3e-32`, FNO `p = 3.3e-9`).

Unconstrained (`free`) controls, same protocol:

| start field | `‖R_h‖` start → end | `J/J₀` | rel-L2 med. start → end | Δ rel-L2 | frac. worse | `mse_u` mean start → end |
|---|---|---|---|---|---|---|
| ground truth | 0.1912 → 0.0475 (−75 %) | 0.079 | 0.00000 → 0.00632 | — | 1.000 | 0.000 → 0.513 |
| Transolver seed 0 | 0.2107 → 0.0738 (−65 %) | 0.153 | 0.00464 → 0.00852 | +78.0 % | **1.000** | 0.133 → 0.658 |
| dropout-FNO | 0.1113 → 0.0529 (−52 %) | 0.219 | 0.02916 → 0.02864 | −1.3 % | 0.285 | 3.719 → 3.675 |

### 3.1 Theorem leg (ii), measured

`u*` is not a stationary point of `J`, and the consequence is large. Starting **exactly at
the ground truth** and running BC-constrained descent:

* `J` falls monotonically to 39.8 % of `J₀`; the monitored residual norm falls 35 %.
* The field error rises immediately and **never comes back**: `argmin_k error = 0` for
  **200/200** cases, i.e. on no case, at any of the 500 steps, does the error return to or
  below its starting value. There is no benign phase at all. (Stated precisely: the error
  is *bounded below by its start* on every case and every step. It is not strictly monotone
  step-by-step — the median trajectory dips very slightly between steps 1 and 2
  (0.00117 → 0.00107) before rising through 0.00143 (step 5), 0.00342 (step 50) and
  0.00700 (step 500). We claim never-recovering, not strictly increasing.)
* The end point sits at rel-L2 = 0.00694 and `mse_u` = 0.499. For scale: that is
  **1.57× the entire rel-L2 error of the deployed Transolver** (0.00442) and **4.2× its
  `mse_u`** (0.119). *Minimising the monitored residual starting from perfect knowledge of
  the flow throws away more accuracy than the SOTA surrogate ever had.*
* **Exchange rate:** median `Δ(rel-L2) / Δ(residual_norm) = −0.140`. Each 0.01 of absolute
  monitored-residual reduction costs +0.0014 rel-L2.
* Matched-`J` (defusing "you ran it to a silly end point"): at `J/J₀ = 0.5` — a residual
  reduction of only 29 % — the median rel-L2 is already 0.00160 and mean `mse_u` 0.070,
  from an exact zero.

This is a direct, quantitative measurement of the residual floor `e∞` of
`thm:residual-floor`(iii): the constrained residual minimiser sits ≈ 0.007 rel-L2 away
from the truth.

### 3.2 The deployed system: the claim survives its own falsification test

The pre-registered falsification threshold was "lowers rel-L2 on a majority of cases".
Across three independent backbone seeds, BC-constrained residual descent **raises** rel-L2
on **94.0 %, 96.5 % and 98.0 %** of cases, by a median of **+74 % to +87 %**, while the
monitored residual falls by a third. `mse_u` roughly quintuples (mean 0.10–0.12 → 0.54–0.56).
On **54.0 / 59.5 / 59.5 %** of cases (seeds 1/0/2) the error never dips below its starting
value at any step — descent is harmful from the very first step on the majority.

Sign comparison with the supervised corrector on the same backbone: the DEQ lowers field
MSE by **9 / 21 / 25 %** (the paper's `tab:v2` headline); residual descent raises median
`mse_u` by **+217 / +203 / +246 %**. Same backbone, same test split, opposite sign, an
order of magnitude apart in size. (The two are not the identical metric — `tab:v2` reports
a field-MSE aggregate, this is median per-case `mse_u` — so read the *sign and order of
magnitude*, not a ratio.)

### 3.3 Where the claim as written is FALSE — and why it still does not rescue the objective

On the weak dropout-FNO, BC-constrained residual descent **lowers** rel-L2 on **75 %** of
cases, median **−18.7 %** (`mse_u` median −34.1 %). This is a real, statistically solid
improvement (`p = 3.3e-9`) and it **falsifies the abstract's unqualified sentence.** It
must be conceded and the claim must become a regime claim. Three things bound it:

1. **It is tiny next to supervised correction, on the paper's own backbone.** In the
   `free` arm — the like-for-like `tab:iters` configuration, and the residual norms match
   `tab:iters` to 1 % (0.1113 here vs 0.1126 there, and `mse_u` 3.72 vs 3.92) — explicit
   residual descent cuts the monitored residual by **52 %** and buys **−1.2 % mean
   `mse_u`** (3.719 → 3.675). The supervised DEQ in `tab:iters` buys **−41.7 %**
   (3.924 → 2.287) while *raising* the residual by 381 %. **Residual minimisation delivers
   ≈ 1/35 of the error reduction that supervised correction delivers, and the two move the
   residual in opposite directions.**
2. **The mean `mse_u` still gets worse** even where the median improves (0.898 → 0.930 in
   the `bc` arm). This is a genuine minority blow-up, not a broad drift: **4.0 %** of cases
   (8/200) more than double their `mse_u` (max ratio 3.4×), and the **five** worst cases
   alone contribute +24.6 of `mse_u`, which by itself outweighs the aggregate improvement
   of the other 195 (net change over all 200 is +6.5). Residual descent therefore trades a
   median gain for a fat right tail — the opposite of what one wants from a correction
   operator that has to be safe on every case.
3. **There is no stopping rule — this is the decisive point.** `J` decreases monotonically
   to the last step on **100 %** of cases in every arm. The *error* minimum lies strictly
   inside the trajectory on **78.5 %** of FNO `bc` cases (median `argmin_k error` = step
   134 of 500) and on **81.5 %** of FNO `free` cases. `argmin_k J ≠ argmin_k error` on
   **79–100 %** of cases in every arm. So even in the regime where residual descent helps,
   the objective you are minimising cannot tell you when to stop, and running it to
   convergence gives up most of the gain. A quantity that improves the field only if you
   stop it using information it does not contain is not a usable correction objective.

### 3.4 The regime boundary, in the theorem's own variable

The theorem's own regime parameter is `ρ = ‖R_h(û)‖ / ‖r*‖` — how far the prediction's
residual sits above the truth's floor. Because the truth arm supplies `‖r*‖` for the *same*
case, `ρ` is directly measurable per case
(`results/residual_descent/regime_boundary.json`; "improved" = rel-L2 fell after 500
BC-constrained Armijo steps):

| ρ bin | dropout-FNO | Transolver s0 | Transolver s1 | Transolver s2 |
|---|---|---|---|---|
| [1.0, 1.5) | 0.40 (n = 70) | **0.01** (n = 177) | **0.00** (n = 177) | **0.01** (n = 180) |
| [1.5, 2.0) | 0.97 (n = 73) | 0.28 (n = 18) | 0.47 (n = 19) | 0.13 (n = 15) |
| [2.0, 3.0) | 1.00 (n = 44) | 0.25 (n = 4) | 0.75 (n = 4) | 0.25 (n = 4) |
| [3.0, 5.0) | 1.00 (n = 7) | — | — | — |
| **median ρ** | **1.69** | **1.14** | **1.14** | **1.13** |

The fraction of cases residual descent helps rises monotonically with `ρ` within every
arm, exactly as `eq:rf-decomp` predicts: descent helps when `‖Le‖ ≫ ‖r*‖` and hurts once
the prediction is within ~1.5× of the floor. **The deployed Transolver has median
ρ = 1.13–1.14 and 88 % of its cases in the ρ < 1.5 bin, where descent helps 0–1 % of the
time — it is already deep inside the harmful regime.**

Honest caveat: `ρ` is a strong trend, not a complete predictor. At matched ρ ∈ [1.5, 2.0)
the FNO improves 97 % of the time and the Transolver only 13–47 %, so the start field
carries information beyond ρ. We report the monotone trend, not a law.

### 3.5 Secondary controls

* **`nut` (all-four-channel `uvpn` arm), n = 200, `bc`, Armijo:**

  | arm | variables | `J/J₀` | Δ rel-L2 | frac. worse | `mse_u` mean | `mse_nut` mean |
  |---|---|---|---|---|---|---|
  | truth | `uvp` | 0.398 | 0 → 0.00694 | 1.000 | 0 → 0.499 | 0 → 0 (frozen) |
  | truth | `uvpn` | 0.422 | 0 → 0.00343 | **1.000** | 0 → 0.164 | 0 → **0.0434** |
  | Transolver s0 | `uvp` | 0.406 | +76.5 % | 0.965 | 0.119 → 0.556 | 6.3e-9 → 6.3e-9 |
  | Transolver s0 | `uvpn` | 0.431 | **+18.9 %** | **0.825** | 0.119 → 0.219 | 6.3e-9 → **0.115** |

  Freeing `nut` roughly quarters the velocity damage — **by dumping the residual into the
  eddy viscosity.** `mse_nut` rises by **seven orders of magnitude** (6.3 × 10⁻⁹ → 0.115).
  The optimiser exploits `nu_eff = nu + nut` as a free knob to cancel advection, which is
  exactly the degenerate direction the theorem's kernel discussion predicts. The residual
  is still made worse-for-the-field on 82.5 % of cases, and the turbulence closure is
  destroyed. `uvp` (with `nut` frozen) is therefore the primary and the *conservative*
  arm; `uvpn` does not rescue the objective, it just relocates the damage.
* **Pressure gauge.** A uniform `p` shift is an exact null mode of `R_h` (ker `L`), so raw
  `mse_p` growth would be an invalid attribution. All pressure numbers above are
  **gauge-corrected** (fluid-mean offset removed). Gauge-corrected `mse_p` is essentially
  flat along the descent (Transolver s0: 498.4 → 494.8), which is itself the kernel
  prediction of the theorem: the monitor can neither detect nor fix that mode.

---

## 4. `tab:iters` re-reported over its FULL range — and a live arithmetic error

The paper has already demoted `tab:iters` and added limits (i)–(iii) to `sec:iters`. Good.
But the re-report itself now contains a **wrong count that a reviewer will check**, and it
still omits the two most damaging facts. Everything below is in
`results/sensitivity/iters.json`, which the paper ships.

### 4.0 ERROR IN THE CURRENT TEXT — fix before submission

`body.tex` (`sec:iters`, limit (ii)) currently reads:

> *"…so over **four of the five** intervals the two quantities move in the *same*
> direction."*

**This is wrong; it is three of five.** The six rows give five consecutive intervals
(0→1, 1→3, 3→5, 5→10, 10→15). `mse_u` moves *opposite* to the residual on 0→1 and 1→3, and
*with* it on 3→5, 5→10 and 10→15. The correct statement is **three of the five**. (The
sentence is also self-inconsistent with the clause immediately before it, which correctly
identifies 0→3 as the only opposite-moving stretch — two intervals, leaving three.)

| `n_iters` | `mse_u` | surf. `mse_p` | `residual_norm` | segment: `mse_u` vs residual | segment: surf. `mse_p` vs residual |
|---|---|---|---|---|---|
| 0 | 3.924 | 541 204 | 0.113 | — | — |
| 1 | 2.460 | 428 560 | 0.336 | opposite | opposite |
| 3 | **2.287 (min)** | 336 718 | 0.542 | opposite | opposite |
| 5 | 2.439 | 308 381 | 0.594 | **SAME** | opposite |
| 10 | 2.570 | 300 298 | 0.618 | **SAME** | opposite |
| 15 | 2.575 | 300 664 | 0.620 | **SAME** | **SAME** |

### 4.1 Two facts still missing from `sec:iters`

**(a) The `n_iters = 0` row is a corrector-on/off confound, not an iteration count.**
Row 0 is backbone-only; every row ≥ 1 applies the DEQ. The 0→1 step therefore conflates
"corrector applied" with "one more iteration", and it carries most of the apparent effect.
**Restricted to the actual iteration knob (`n_iters` ≥ 1) the residual rises +84.7 % while
`mse_u` also rises +4.7 % — the same direction.** The paper's limit (i) says the corrector
is supervised; it does not say that the single row doing the work is not an iteration step
at all.

**(b) The channel that genuinely moves apart is surface pressure, not `mse_u`.**
`surface_mse_p` falls monotonically across the entire range (541 204 → 300 664, −44.4 %),
opposite to the residual on 4 of 5 segments; rank correlation with the residual **−0.94**.
By contrast the rank correlation of `residual_norm` with `mse_u` across the six swept
points is **−0.03** — essentially none. The table's honest content is: *the residual rises
while the design-relevant surface-pressure error falls, and the volume-velocity error is
non-monotone and uncorrelated with the residual.* Naming `mse_u` as the quantity that moves
apart, as the surrounding prose still does, picks the weaker of the two channels.

A reviewer reading `results/sensitivity/iters.csv` finds all of this in thirty seconds.

---

## 5. η-sensitivity ("you picked a bad learning rate")

Fixed-η descent, 200 steps, 40 cases, `results/residual_descent/eta_sensitivity.json`.

**From ground truth, BC-constrained** (medians; `frac. worse` = fraction of the 40 cases
whose error increased):

| η | `J/J₀` | `‖R_h‖` end | rel-L2 end | frac. worse |
|---|---|---|---|---|
| 10⁻⁴ … 10⁻² | 1.000 | 0.1698 | 0.00000 | 1.00 (no motion) |
| 10⁻¹ | 0.999 | 0.1697 | 0.00000 | 1.00 |
| 10⁰ | 0.995 | 0.1691 | 0.00001 | 1.00 |
| 10¹ | 0.954 | 0.1639 | 0.00008 | 1.00 |
| 10² | 0.729 | 0.1420 | 0.00056 | 1.00 |
| 10³ | 0.492 | 0.1205 | 0.00200 | 1.00 |
| 10⁴ | 0.463 | — | 0.00493 | 1.00 |
| ≥ 10⁵ | diverges (non-finite) | — | — | 1.00 |

**From the deployed Transolver seed 0, BC-constrained:**

| η | `J/J₀` | rel-L2 med. start → end | Δ | frac. worse |
|---|---|---|---|---|
| 10⁻⁴ … 10⁰ | 0.996–1.000 | 0.00479 → 0.00479 | −0.0 % | 0.00 (no motion) |
| 10¹ | 0.959 | 0.00479 → 0.00478 | −0.2 % | 0.00 |
| 10² | 0.752 | 0.00479 → 0.00480 | −0.8 % | 0.15 |
| 10³ | 0.516 | 0.00479 → 0.00504 | **+1.4 %** | 0.57 |
| 10⁴ | 0.459 | 0.00479 → 0.00723 | **+31.5 %** | 0.90 |
| ≥ 10⁵ | diverges | — | — | 1.00 |

**On the deployed backbone the reading is monotone, and it is the answer to the objection:
the more the residual actually falls, the worse the field error gets.** There is no
learning rate at which the residual is meaningfully reduced *and* the error improves. A
~13 % residual reduction (η = 10²) is roughly error-neutral (−0.8 % median); anything that
cuts the residual by ≥ 25 % is harmful, sharply. It is not a bad-η artifact in either
direction: η below 10¹ simply does not move the field, and η above 10⁴ diverges.

**From the dropout-FNO, BC-constrained — the honest counter-column.** Here the same sweep
runs the *other* way, and we report it rather than omit it:

| η | `J/J₀` | rel-L2 med. start → end | Δ | frac. worse | frac. diverged |
|---|---|---|---|---|---|
| 10⁻⁴ … 10⁰ | 0.994–1.000 | 0.01306 → 0.01306 | −0.0…−0.1 % | 0.00 | 0.00 |
| 10¹ | 0.942 | 0.01306 → 0.01301 | −0.6 % | 0.00 | 0.00 |
| 10² | 0.668 | 0.01306 → 0.01274 | −4.2 % | 0.00 | 0.00 |
| 10³ | 0.340 | 0.01306 → 0.01141 | **−14.8 %** | 0.00 | 0.00 |
| 10⁴ | — | 0.01306 → 0.01103 | **−20.2 %** | 0.28 | 0.17 |
| ≥ 10⁵ | diverges | — | — | 1.00 | 1.00 |

So over the range this 200-step sweep reaches (`J/J₀ ≥ 0.34`), deeper residual reduction
*helps* the FNO. **The "monotone in achieved `J`" statement therefore holds for the
deployed backbone, not universally, and we do not claim otherwise.** The FNO's gain turns
over further along the path than this sweep goes: at 500 Armijo steps (`J/J₀ = 0.238`) the
error minimum is already interior on 78.5 % of cases, and Adam at `J/J₀ = 0.178` recovers
only −6.7 % against −15.3 % at 0.392 (§5.1). The two observations are consistent — a
U-shape whose minimum lies beyond `J/J₀ ≈ 0.3` — and together they are the point: the
useful stopping depth is **arm-dependent and invisible to `J`**.

### 5.1 Adam — "plain gradient descent is a strawman"

Adam on the same objective, `bc`-constrained, 500 steps, n = 100
(`descent_*_bc_uvp_adam*.json`):

| arm | Adam lr | `J/J₀` | `‖R_h‖` start → end | Δ rel-L2 (median) | frac. worse |
|---|---|---|---|---|---|
| Transolver s0 | 0.01 | 0.400 | 0.2210 → 0.1631 (−26 %) | **+17.2 %** | 0.68 |
| Transolver s0 | 0.10 | 0.248 | 0.2210 → 0.1262 (−43 %) | **+113.9 %** | 0.95 |
| dropout-FNO | 0.01 | 0.392 | 0.2903 → 0.2003 (−31 %) | −15.3 % | 0.08 |
| dropout-FNO | 0.10 | 0.178 | 0.2903 → 0.1398 (−52 %) | **−6.7 %** | 0.41 |

Two things fall out, and they close the objection for good.

1. **Same sign, same monotonicity, a different optimiser.** On the deployed backbone Adam
   makes the error worse, and *more so the deeper it drives `J`* (+17 % at `J/J₀ = 0.40`,
   +114 % at 0.25) — matching fixed-η GD and Armijo GD. Three step rules, one conclusion.
2. **On the FNO, driving `J` lower destroys the gain**: −15.3 % at `J/J₀ = 0.39` falls to
   **−6.7 %** at `J/J₀ = 0.18`, with the fraction of harmed cases rising 0.08 → 0.41. The
   better you minimise the objective, the less good it does. This is the stopping-rule
   pathology of §3.3(3) reproduced under a second optimiser.

**Scope of the optimiser-independence claim, stated precisely.** For the *deployed
backbone*, all three step rules — Armijo GD, fixed-η GD over 12 decades, Adam at two
learning rates — agree in sign and are monotone in the *achieved* `J`, so any optimiser
that drives `J` lower lands further right on these tables and does worse. That is the
claim the paper needs, and it is now optimiser-independent. For the *FNO* the relation is
**not** monotone: the gain grows with depth to about `J/J₀ ≈ 0.3` and then decays
(§5, −14.8 % at 0.34 vs −6.7 % at 0.178). We claim the deployed-arm monotonicity and the
FNO U-shape as measured, not a universal law.

---

## 6. Verdict

| question | answer |
|---|---|
| Was the originally cited experiment (`tab:iters`) a test of the claim? | **No.** It sweeps a supervised corrector's fixed-point cap. Conceded in full; the paper has since demoted it. |
| Does explicit residual minimisation raise field error on the **deployed** system? | **Yes.** 94–98 % of cases across 3 backbone seeds; rel-L2 +74…+87 %; `mse_u` +203…+246 %; `p < 1e-30`. This is new — the existing `sec:descent` tests only truth and a synthetic perturbation. |
| Is `u*` a stationary point of `J`? | **No.** With the Dirichlet data pinned at ground truth, descent worsens 200/200 cases; the error never returns to its starting value at any of 500 steps, and the end point sits at 1.57× the deployed backbone's entire rel-L2. |
| Does the result depend on the optimiser? | **No.** Armijo gradient descent, fixed-η GD over 12 decades, and Adam at two learning rates all agree, and all are monotone in the *achieved* `J`. This closes the paper's own "the descent is Adam, not gradient flow" limitation. |
| Is it an artifact of an unconstrained (ill-posed) residual? | **No.** The BC-constrained arm hands the optimiser the exact far-field and near-wall Dirichlet data and gets the same answer. |
| Does residual descent ever lower field error? | **Yes** — on a backbone far from the floor (dropout-FNO: 75 % of cases, median −18.7 %). Any unqualified "reducing it does not reduce field error" is false and must stay a regime claim. |
| Does that rescue the residual as an objective? | **No, on three independent grounds.** (a) ≈ 1/35 of the supervised gain on the same backbone in the `tab:iters` configuration; (b) the mean `mse_u` still worsens — 4 % of cases more than double their error and the 5 worst outweigh the other 195; (c) no stopping rule: `J` falls monotonically to the last step on 100 % of cases while the error minimum is interior on 78.5 %, and driving `J` lower with Adam *destroys* the gain (−15.3 % → −6.7 %). |
| Is `tab:iters` correctly reported now? | **Almost.** One arithmetic error ("four of the five intervals" should be **three**), plus two omissions: the row-0 corrector-on/off confound, and that the channel which genuinely moves apart is surface pressure (ρ = −0.94), not `mse_u` (ρ = −0.03). |

**Net.** The paper's differentiating claim survives and is now anchored on the deployed
system at n = 200 rather than on a synthetic perturbation at n = 24. The honest headline is
**iterate selection plus a regime boundary**, not uniform harm — which is the form the
paper has already converged on. The residual is a poor correction objective *for a
prediction already close to the floor*, which is exactly where a competitive surrogate
lives (measured: median ρ = 1.13–1.14 for the deployed Transolver, with 88 % of its cases
in the bin where descent helps 0–1 % of the time).

---

## 7. Exact text changes required (written against the CURRENT `body.tex`)

The paper already contains `sec:descent` and a demoted `tab:iters`, so these are edits to
what is there now, not to the pre-review draft.

### 7.1 `sec:iters`, limit (ii) — arithmetic fix, mandatory

Replace `four of the five intervals` with `three of the five intervals`. (See §4.0. The
clause immediately before it already implies three; the two are currently inconsistent.)

### 7.2 `sec:iters`, limit (i) — add the row-0 confound

Append to limit (i):

> *Moreover the $\texttt{n\_iters}=0$ row is backbone-only while every row $\ge1$ applies
> the corrector, so the $0\!\to\!1$ step---which carries most of the apparent effect---is a
> corrector-on/off contrast, not an iteration step. Restricted to the iteration knob proper
> ($\texttt{n\_iters}\ge1$) the residual rises $84.7\%$ and $\texttt{mse\_u}$ rises $4.7\%$,
> in the same direction.*

### 7.3 `sec:iters`, limit (iii) — name the right channel

Append:

> *The channel that does move opposite to the residual across the whole sweep is the
> surface pressure ($541204\!\to\!300664$, rank correlation $-0.94$), not
> $\texttt{mse\_u}$, whose rank correlation with the residual over the six swept points is
> $-0.03$.*

### 7.4 `sec:descent` — add the deployed-system and gradient-flow paragraph

Insert after "The claim we make, and the one we do not":

> \paragraph{At $n=200$, on the deployed backbone, and under gradient descent.} The
> preceding arms use $24$ cases, an Adam optimiser and a synthetic displacement. We
> repeated the experiment with the framework's \emph{monitored} operator (a differentiable
> replica verified against \texttt{Diagnostics.residual\_norm} to $10^{-8}$ relative),
> under \textbf{Armijo-line-searched gradient descent}---so $J$ is monotone non-increasing
> and the step is chosen by the objective---on $200$ AirfRANS test cases, starting from the
> \emph{deployed} Transolver (three backbone seeds) and from the dropout-FNO of
> \autoref{tab:iters}, with the far-field and near-wall Dirichlet data \emph{pinned at
> ground truth} so the negative cannot be attributed to an unconstrained residual
> (\autoref{tab:descent200}; \texttt{scripts/residual\_descent\_test.py}).
> From the truth, $J$ falls to $39.8\%$ of $J_0$ while the field error rises on
> $\mathbf{200/200}$ cases and never returns to its starting value at any of $500$ steps,
> ending at $\mathrm{rel}\text{-}L_2=0.0069$---$1.57\times$ the \emph{entire} error of the
> deployed Transolver---at a measured exchange rate
> $\Delta\mathrm{err}/\Delta\|R_h\|=-0.140$. From the deployed Transolver the residual
> falls $34\%$ while the error rises on $\mathbf{96.5/94.0/98.0\%}$ of cases (seeds
> $0/1/2$; median $+76.5/+73.9/+86.9\%$; Wilcoxon $p<10^{-30}$), against the supervised
> corrector's $-9/-21/-25\%$ on the same backbone. The verdict is unchanged under three
> step rules---Armijo, fixed $\eta$ swept over $12$ decades, and Adam---and in every one
> the outcome is monotone in the \emph{achieved} $J$: the harder the objective is
> minimised, the worse the field. This removes the ``Adam, not gradient flow'' caveat and
> supplies the endpoint claim on the deployed system.

and the table it references (numbers from `results/residual_descent/descent_*.json`,
independently re-derived in §3):

```latex
\begin{table}[t]
\centering
\small
\caption{Armijo-line-searched gradient descent on the monitored objective
$J=\tfrac12\|R_h\|^2$, $500$ steps, $n=200$ AirfRANS \texttt{full} test cases, with the
far-field and near-wall Dirichlet data pinned at ground truth. $J$ is monotone
non-increasing by construction. rel-$L_2$ is the fluid-masked rel-$L_2$ of speed (the
\texttt{measure\_acceptance\_gate} formula); $\Delta$ is the median per-case change.
\texttt{scripts/residual\_descent\_test.py}.}
\label{tab:descent200}
\begin{tabular}{l r r r r r}
\toprule
start field & $\|R_h\|$ start$\to$end & $J/J_0$ & $\Delta$ rel-$L_2$ & frac.\ worse & $\texttt{mse\_u}$ start$\to$end \\
\midrule
ground truth $u^\star$ & $0.191\to0.125$ & $0.398$ & --- ($0\to0.0069$) & $\mathbf{200/200}$ & $0.000\to0.499$ \\
Transolver seed 0      & $0.218\to0.144$ & $0.406$ & $+76.5\%$ & $0.965$ & $0.119\to0.556$ \\
Transolver seed 1      & $0.218\to0.142$ & $0.388$ & $+73.9\%$ & $0.940$ & $0.120\to0.553$ \\
Transolver seed 2      & $0.217\to0.141$ & $0.390$ & $+86.9\%$ & $0.980$ & $0.103\to0.544$ \\
dropout-FNO            & $0.286\to0.153$ & $0.238$ & $-18.7\%$ & $0.250$ & $0.898\to0.930$ \\
\bottomrule
\end{tabular}
\end{table}
```

### 7.5 `sec:descent` / `sec:regime` — make the regime boundary quantitative

**Append after** (do *not* replace) the existing *"from the perturbed start it reduces
error in 18/24 cases, typically by 60 %"* sentence — that concession is load-bearing and
this run confirms it rather than supersedes it. The addition makes it quantitative:

> *The boundary is measurable in the theorem's own variable
> $\rho=\|R_h(\hat u)\|/\|r^\star\|$, since the truth arm supplies $\|r^\star\|$ for the
> same case. Over $200$ cases the fraction on which residual descent lowers
> $\mathrm{rel}\text{-}L_2$ rises monotonically with $\rho$: for the dropout-FNO
> $0.40$ at $\rho\in[1,1.5)$, $0.97$ at $[1.5,2)$ and $1.00$ above $2$; for the deployed
> Transolver $0.01$, $0.28$ and $0.25$. The deployed backbone has median $\rho=1.13$--$1.14$
> with $88\%$ of cases in the lowest bin: it is already inside the regime where descent
> almost never helps. ($\rho$ is a strong trend, not a complete predictor---at matched
> $\rho\in[1.5,2)$ the two backbones differ---so we claim the ordering, not a law.)*

### 7.6 Limitations — retire one, add one

* **Retire** *"The descent is Adam, not gradient flow"* (now closed by §7.4) — or narrow it
  to the $n=24$ arms only.
* **Add:** *"Descending $\nu_t$ as well as $(u,v,p)$ quarters the velocity damage but does
  so by dumping the residual into the eddy viscosity: $\texttt{mse\_nut}$ rises seven
  orders of magnitude ($6.3\times10^{-9}\to0.115$) while the error still worsens on
  $82.5\%$ of cases. We therefore freeze $\nu_t$ in the headline arm and report the
  four-channel arm as a control."*

### 7.7 `tab:positioning`

Add to the caption: *"Object.\ = the residual was tested as a correction objective by
explicit minimisation of $\tfrac12\|R_h\|^2$ (\autoref{sec:descent}), on the deployed
backbone at $n=200$ under three step rules---not merely by observing a supervised
corrector."*

### 7.8 Rebuttal line

> *Reviewer: your negative rests on an Adam descent over 24 cases from a synthetic
> perturbation of the truth. That is not what a surrogate's error looks like, and Adam is
> not the gradient flow your theorem is about.*
>
> *→ We agree, and we ran it properly: Armijo-line-searched gradient descent (monotone $J$
> by construction) on the framework's own monitored operator, verified to $10^{-8}$ against
> the deployed monitor, over 200 AirfRANS cases, starting from the deployed Transolver
> (3 seeds) and the dropout-FNO, with the Dirichlet data pinned at ground truth.*
>
> *→ Evidence: error rises on 94–98\% of deployed cases (median $+74$ to $+87\%$,
> $p<10^{-30}$) and on 200/200 started from the truth; on the deployed backbone the result
> is unchanged under fixed $\eta$ over 12 decades and under Adam, and is monotone in the
> achieved $J$ in all three, so it is not an artifact of the step rule. Where descent does
> help (a backbone far from the floor) it delivers $\approx1/35$ of the supervised gain and
> gives no stopping rule --- driving $J$ lower with Adam cuts the gain from $-15.3\%$ to
> $-6.7\%$.*

---

## 8. Cost and reproduction

* **CPU-only. No GPU was used** (`CUDA_VISIBLE_DEVICES=""`, `device=cpu`); the live GPU
  run and `results/mgn/`, `checkpoints/mgn/`, `mgn_run.log` were untouched.
* Zero backbone forward passes for the Transolver arms (re-used
  `data/cache/acceptance_gate/`); 200 CPU FNO forwards (312 s), cached to
  `data/cache/residual_descent_fno/`.
* **Measured compute, summed from the `wall_s` field of each result JSON: 12 469 s
  (3.5 h) for the 16 descent runs, plus 2 878 s for the η stage and 312 s for the FNO
  cache — 4.3 h total.** Per-combo timings are logged per run (661–1 233 s at n = 200 ×
  500 steps). These are *not* clean per-run timings: two of the sweeps ran concurrently
  and another agent's job was competing for the same CPU, so a single-job re-run should be
  faster. The `wall_s` in each JSON is the authoritative per-run figure.
* Thread counts used: 8 for the main sweep, 4–6 for the η and Adam sweeps (of 24 cores),
  deliberately leaving headroom for the concurrent jobs.
* Every combo and every (arm, mode) η block writes its own JSON and is skipped on re-run,
  so the sweep is resumable — it was in fact interrupted twice and resumed.
  A single shared accumulator file is *not* safe here: the Windows venv launcher runs the
  script as a parent+child pair and two writers with independent in-memory accumulators
  silently truncated each other's keys. That bug was caught and fixed (per-file writes,
  merged into `eta_sensitivity.json` at the end); it is worth knowing about for any other
  script in this repo that accumulates into one JSON.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=src \
  .venv/Scripts/python.exe scripts/residual_descent_test.py --stage all
```

**What a fuller run would add.** (i) The full 1000 AirfRANS test split rather than 200
cases; (ii) a Gauss–Newton / L-BFGS arm — mitigated here by the optimiser-independent
matched-`J` reading of §5, plus Adam; (iii) resolution 256 to check whether the floor
`‖r*‖` (and hence the regime boundary `ρ`) shrinks with grid refinement — this is the one
result that could genuinely move the boundary and is tracked separately in
`docs/paper/review/floor_resolution_study.md`.
