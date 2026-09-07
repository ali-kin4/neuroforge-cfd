# Residual descent: running the experiment the paper never ran

**Status: PARTIALLY-RESOLVED — the claim survives for the deployed system but is
over-stated as written, and `tab:iters` must be re-reported.**

Script `scripts/residual_descent_test.py` · results `results/residual_descent/` ·
CPU-only, no GPU, ~3.4 h wall-clock (see §8).

---

## 1. The objection, in its strongest form

> The paper's central negative claim — *"as a correction objective, the residual fails:
> reducing it does not reduce field error"* (abstract; `sec:iters`; `thm:residual-floor`
> leg (iv), which is explicitly about gradient flow `edot = -grad J`, `J = ½‖R_h(u)‖²`) —
> is not tested by the experiment cited for it.
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

The objection is correct on both counts. This document runs the missing experiment and
re-reports `tab:iters` in full.

---

## 2. Method

### 2.1 The operator under test

The brief asked for `physics_residual_torch`, "so the operator under test is
byte-identical to the one the paper monitors." **It is not.** `physics_residual_torch`
is the *training-loss* twin: raw, dimensional, masking only the solid, and clamping
`nu_eff >= nu`. The scalar the paper monitors is `Diagnostics.residual_norm()`
(`core/types.py:314`), which additionally

1. zeroes the **solid-adjacent fluid ring** (`_solid_adjacent_fluid`, `residuals.py:339`),
2. **non-dimensionalises** continuity by `u_inf/L` and momentum by `u_inf²/L`
   (`residuals.py:352-359`), and
3. clips `nu_eff >= 0`, not `>= nu`.

Descending on the wrong functional would have been a free rebuttal. We therefore built a
differentiable replica of the **monitored** operator out of the framework's own
backend-agnostic `physics/operators.py` (`ddx`, `ddy`, `laplacian` duck-type onto torch,
so gradients flow), and verified it:

```
J(u) = ½ · mean_cells( r_c² + r_x² + r_y² )        residual_norm = sqrt(2J)
```

`results/residual_descent/verification.json` (n = 25 real AirfRANS cases):

| check | value |
|---|---|
| max relative error of `sqrt(2J)` vs `Diagnostics.residual_norm()` | **9.9 × 10⁻⁸** |
| uniform-freestream field `‖R_h(u_inf)‖` (theorem leg (i)) | **0.0 exactly**, all cases |
| `‖r*‖ = ‖R_h(u*)‖` mean (theorem hypothesis H2) | 0.1781 |
| H2 holds (`‖r*‖ > 0`) on every case | yes |

The raw training-loss twin is available as a secondary objective (`--objective raw`) so
the result can be shown not to hinge on the choice.

All arithmetic is **float64**. In float32 (eps ≈ 1.2 × 10⁻⁷ relative, with `u ~ 50`) the
stable explicit update rounds away entirely and one would wrongly conclude that descent is
inert.

### 2.2 Arms

Start field:

* **`truth`** — the ground-truth field `u*`. Sharpest test of leg (ii): if `u*` is not a
  stationary point of `J`, descent from truth must move away from truth.
* **`transolver`** — the deployed SOTA backbone, seeds 0/1/2, read from the existing
  zero-forward-pass cache `data/cache/acceptance_gate/seed{k}/*.npz` (key `raw`).
* **`fno`** — the dropout-FNO of `checkpoints/certificates_deq.pt`, i.e. the same
  backbone as `tab:iters`, so the numbers are comparable.

Constraint mode:

* **`bc` — the load-bearing arm.** Theorem leg (i) says *any spatially constant field* is
  an exact global minimum of `J`, so unconstrained descent has an attractor at "flat
  everything" and would raise the error essentially by construction. A hostile reviewer
  kills that in one line: *"you rediscovered that a PDE residual with no boundary
  conditions is ill-posed."* So in the `bc` arm we **hand the optimiser the exact
  Dirichlet data**: the 2-cell outer border ring and the 2-cell near-wall band are pinned
  at ground truth, the solid is frozen, and the gradient is projected to zero on all
  frozen cells. A **2**-cell freeze is required because `grad J` carries a 4th-order
  stencil (double Laplacian); a 1-cell freeze would leave the second ring free to drift.
* **`free`** — unconstrained. Reported as the ill-posedness demonstration, not as the
  headline.

Variables: `uvp` (primary; `nut` frozen, it is a closure variable) and `uvpn` (all four).

Step rule: **Armijo backtracking line search on `J` itself** (per case, `c1 = 1e-4`,
trial step from 10⁸ halving up to 60 times), which *guarantees* a monotone `J` decrease —
the objective chooses its own step, so "you picked a bad learning rate" cannot be raised.
Plus a fixed-`eta` log sweep over 10⁻⁴ … 10⁷ for the sensitivity table (§5).

500 descent steps, n = 200 AirfRANS `full` test cases.

### 2.3 Pre-registered readings (named before the run)

* **Falsification threshold.** If BC-constrained descent from the deployed Transolver
  field lowers rel-L2 on a majority of cases by a substantial fraction of what the
  supervised DEQ achieves (−9/−21/−25 % field MSE), then "poor correction objective" is
  falsified as stated.
* The theorem itself predicts a **non-monotone** error trajectory, so the expected outcome
  is "it depends on the starting point": far from truth `‖Le‖ ≫ ‖r*‖` and the residual
  *is* a proxy for error (that is the detector leg); near truth the floor dominates.
* **Stopping-criterion test.** Per case, `argmin_k J` vs `argmin_k error`. If `J` falls
  monotonically while the error is U-shaped, the residual supplies no signal for *when to
  stop*, so the objective is unusable in deployment **even in the regime where it helps**.

---

## 3. Results

<!--RESULTS_TABLE-->

---

## 8. Cost and reproduction

<!--COST-->
