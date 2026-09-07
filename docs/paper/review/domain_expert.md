# Domain-expert review: *NeuroForge — Self-Auditing Neural CFD with a Physics-Residual Trust Layer*

**Reviewer stance:** CFD/RANS + neural operators. I read `docs/paper/body.tex` (1718 lines),
`docs/paper/sections/residual_floor_theorem.tex`, `docs/paper/abstract.tex`, `docs/paper/refs.bib`,
the physics layer (`src/neuroforge/physics/{residuals,operators}.py`), `src/neuroforge/core/types.py`,
`scripts/{probe_residual_floor,run_v2}.py`, and the result artefacts named below. Verdicts are
grounded in specific lines and JSON values, not impressions.

**Score: 4 / 10 — Reject as written; major revision with new experiments.**
The paper is honest, unusually well-controlled for a solo project, and the trust-signal half is
probably real. But the *single claim that differentiates it from prior art is not tested by the
evidence offered for it*, and the theorem that is meant to carry the theoretical weight describes
structure already known in the FEM a-posteriori literature, with a non-vacuity that follows from
implementation choices. Both desk rejections were, on the merits, **defensible**.

---

## 0. What the paper claims, as I understand it

A one-shot neural surrogate for 2-D steady incompressible RANS (AirfRANS, 128² Cartesian raster)
computes the discrete steady-RANS residual `R_h` of its own prediction and uses it in three roles:

- **trust signal** — per-case field-mean residual ranks which predictions are worst (ρ ≈ 0.61 on
  Transolver, 0.40–0.83 on a dropout-FNO, 0.85 on MeshGraphNet), feeding a split-conformal band;
- **correction objective** — claimed to *fail*;
- **acceptance gate** — monotone-residual backtracking test, claimed to *work*.

A supervised DEQ corrector (trained toward ground truth, residual as an input channel) cuts volume
MSE −8 / −21 / −25 % on Transolver; a W1 ablation shows the residual *input* contributes nothing.
A "residual-floor theorem" (`sec:residual_floor`) is offered as the mechanism unifying "good
detector, bad fixer".

The thesis rests on the **two-way dissociation** (signal yes / objective no) plus the theorem.
That is exactly where I have to press.

---

## 1. The residual-floor theorem: correct, non-vacuous, but known structure — and its
##    non-vacuity is self-inflicted

### 1a. Is the mathematics right?

Mostly yes, with one genuine direction error and one coherence problem.

- **Leg (i)** (`residual_floor_theorem.tex:51–53, 71–73`). `R_h(u_∞) = 0` exactly for a spatially
  constant field, so `min J = 0` while `J(u*) > 0`. **Correct**, and I verified it is correct *in
  the code*, not just on paper: `continuity_residual` and `momentum_residual`
  (`src/neuroforge/physics/residuals.py:57–110`) are pure finite differences of `u,v,p`, all of
  which vanish on a constant field, and `_diff_axis` (`operators.py:63–115`) returns exact zeros
  for constant input at interior *and* one-sided edges. `results/certificates/residual_floor_realdata.json`
  confirms `norm_uniform = 0.0` in 200/200 cases.
  But leg (i) is a restatement of a textbook fact: **a PDE residual with the boundary conditions
  omitted does not determine the solution.** Every uniform flow is an exact solution of interior
  steady Navier–Stokes. This is why PINN/PINO practice hard-enforces or heavily weights BCs.
- **Leg (ii)** `∇J(u*) = Lᵀr*`. **Correct**, trivial.
- **Leg (iii)** — **direction error.** The section is titled "*Quantified floor*" and the text
  calls `‖e_∞‖ ≤ ‖r*‖/σ_min` an "irreducible field error". An **upper** bound cannot establish a
  floor. To claim the error is irreducibly large you need a **lower** bound. The honest statement
  is the two-sided bracket
  `‖P_range(L) r*‖ / σ_max ≤ ‖L⁺r*‖ ≤ ‖r*‖ / σ_min`.
  Likewise "which grows as σ_min → 0 (ill-conditioning)" describes the *bound*, not the quantity;
  `‖L⁺r*‖` need not grow at all. Fix the statement or drop the word "floor" from (iii).
- **Legs (i) and (iii) are about different objects.** (i) locates the *global* minimum of `J` at
  `u_∞`, arbitrarily far from `u*`. (iii) locates the *local, linearised* minimiser at
  `u* − L⁺r*`, arbitrarily close. They are bundled as one mechanism and they are not one
  mechanism. A referee will call this out.
- **Leg (iv)** `d/dt ½‖e‖²|₀ = −‖Le‖² − (Le)·r*`, positive on the open cone
  `(Le)·r* < −‖Le‖²`, feasible for `Le = −α r*/‖r*‖`, `0 < α < ‖r*‖`. I re-derived it;
  **correct**. It is also the elementary observation that least-squares descent on `‖r* + Le‖`
  drives `e → −L⁺r* ≠ 0`.

### 1b. Is the kernel claim right? (It is *understated* — to the authors' credit and cost.)

The paper says `ker L` contains a constant pressure offset and "the residual-blind part of ν_t"
(`residual_floor_theorem.tex:99–107`). That is incomplete. Pressure enters `R_h` **only** through
`ddx(p)`/`ddy(p)`, which are **2nd-order central** differences (`operators.py:108`). The Nyquist
checkerboard `p_ij = (−1)^{i+j}` has *identically zero* central first difference in both
directions. So

```
ker L ⊇ { constant p } ⊕ { checkerboard p } ⊕ { residual-blind ν_t }
```

exact in the interior, perturbed only at the two edge rows/columns where one-sided stencils apply.
This is the classical **odd–even / checkerboard pressure decoupling** of a colocated grid — the
mode Rhie–Chow interpolation exists to suppress. Two consequences, both worth stating in the paper:

1. **Credit.** It makes the "modes the monitor can neither detect nor fix" point *much* stronger
   than the gauge argument: the monitor is blind to an O(1)-energy pressure mode, not merely to a
   gauge constant. If the surrogate ever emits checkerboard pressure noise, the audit is silent.
2. **Indictment.** A monitor that cannot see checkerboard pressure is a monitor built on the wrong
   stencil for a colocated grid, not an inevitable property of steady-RANS residuals.

### 1c. Is it standard consistency-error knowledge from the FEM a-posteriori literature?

**Substantially yes — and I will name the prior art, because the paper does not.**

- **Least-squares FEM.** LSFEM *is* "minimise `‖R_h‖` over a discrete space", and its central
  theorem is exactly the paper's detector leg: the least-squares residual functional is a built-in
  a-posteriori error estimator **if and only if the functional is norm-equivalent to the error
  norm**, with equivalence constants set by operator coercivity/continuity — the paper's `κ(L)`.
  Counterexamples where norm equivalence fails and the functional therefore *cannot* be used as an
  error estimator are published. See Bochev & Gunzburger, *Least-Squares Finite Element Methods*
  (Applied Mathematical Sciences 166, Springer, 2009), and the norm-equivalence-failure literature
  it spawned. The paper's `σ_min ‖e‖ ≤ ‖Le‖ ≤ σ_max ‖e‖` is the discrete finite-dimensional
  version of that statement.
- **Data oscillation.** Morin, Nochetto & Siebert, *Data Oscillation and Convergence of Adaptive
  FEM*, SIAM J. Numer. Anal. 38 (2000) 466–488, introduce exactly a *floor*: "intrinsic
  information missed by the averaging process", with a proof that the oscillation term **cannot be
  avoided** for the estimator to be practical. That is "a residual estimator has a nonzero floor
  and a set of modes it structurally cannot see", published 26 years ago.
- The paper *does* cite `beckerrannacher2001dwr` and `hillebrecht2025posteriori`, and correctly
  observes that DWR-style estimators need an adjoint weight. It then claims novelty in (a) the
  quantified operator-specific floor, (b) the detector/fixer duality from one decomposition, and
  (c) the neural-CFD demonstration. My assessment, stated precisely so as not to overclaim against
  them: **(a)** is a modest extension of the classical forward/backward-error bound the paper
  itself attributes to `trefethenbau1997` — the pseudoinverse form `L⁺r*` under a non-injective
  `L` *is* a step past the textbook statement, so I do not call the object standard; I call its
  *presentation* wrong (an upper bound sold as a floor). **(b)** the two ingredients are known
  separately — the detector half is LSFEM norm-equivalence, the floor-plus-invisible-modes half is
  data oscillation — but packaging both verdicts as consequences of a single decomposition is the
  authors' own framing and I credit it as theirs. **(c)** is genuine, and is an empirical
  contribution rather than a theoretical one.

**Verdict on item 1:** legs (i), (ii), (iv) are elementary and correct; leg (iii) is correct as
algebra but has a direction error in its presentation, and its non-injective form is a modest
extension of the classical bound rather than a new object. The *structure* the theorem describes —
a residual estimator with a nonzero floor at the reference solution plus a set of modes it cannot
see — is LSFEM / data-oscillation structure, and **neither literature is cited**. So the theorem is
**correct**, **non-vacuous** (H2 is measured: `norm_truth_mean = 0.192`, `median = 0.133`, 200/200
cases), and **not new mathematics**. Its non-obvious content is the *measured magnitude* of the
floor on AirfRANS. That is worth reporting — as a measurement, in one paragraph, not as a theorem
section carrying the paper's theoretical claim.

### 1d. The theorem's own account of the detector leg is contradicted by the paper's own data

This is the sharpest internal inconsistency I found and it is not flagged anywhere in the paper.

The "good detector" argument (`residual_floor_theorem.tex:86–92`) requires the **far-from-truth
regime**, `‖Le‖ ≫ ‖r*‖`, so that `‖R_h(û)‖ ≈ ‖Le‖` is a two-sided proxy for `‖e‖`. But
`results/certificates/residual_floor_realdata.json` reports

```
norm_truth_mean (= ‖r*‖) = 0.1920      norm_pred_mean (= ‖R_h(û)‖) = 0.1136
frac_pred_lt_truth = 0.80              (160/200 cases)
```

The deployed prediction's monitored residual is **smaller than the floor** in 80 % of cases.
Since `R_h(û) = r* + Le`, that means `‖Le‖ ≲ ‖r*‖` with partial cancellation: the model sits
**squarely in the floor-dominated regime**, which is precisely the regime the theorem says
*decouples* residual from error. The paper notices the 160/200 number
(`residual_floor_theorem.tex:113–117`) but reads it only as evidence for the *objective* failure.
It destroys the theorem's explanation of the *detector* success.

So either the theorem's regime story is wrong, or the measured ρ ≈ 0.6 comes from something other
than `Le`. That "something other" has an obvious candidate — see §2d.

---

## 2. Physics correctness

### 2a. What is right

Genuinely correct and worth crediting:

- Kinematic pressure, no spurious `1/ρ`. `momentum_residual` (`residuals.py:108–109`) is
  `u u_x + v u_y + p_x − ν_eff ∇²u`. Dimensionally consistent for `p = p/ρ`.
- Residuals **are** computed on physical, denormalised fields; the torch twin denormalises before
  calling (`run_v2.py:251–252`). The paper's claim on this point is true.
- `ν_eff = ν + ν_t` is pointwise and clipped ≥ 0 (`residuals.py:48–54`); the torch version clamps
  to `≥ ν` so eddy viscosity can never reduce diffusion (`residuals.py:507–515`). Correct.
- Non-dimensionalisation by *each term's own* scale — continuity by `U/L`, momentum by `U²/L`
  (`residuals.py:355–360`) — is right, and the code comment records that an earlier version
  summed terms of different units. Good catch by the authors.
- Zeroing residuals inside the solid, and zeroing the solid-adjacent wall ring where the central
  stencil straddles the `u=v=0` discontinuity (`residuals.py:326–348`), is defensible practice and
  is disclosed in `sec:limitations` (`body.tex:1572–1578`). I accept the wall-ring exclusion.

### 2b. What is wrong: the variable-viscosity diffusion term (major)

`ν_eff ∇²u` is **not** the RANS diffusion term when `ν_t` varies in space. For incompressible flow,

```
∇·[ν_eff(∇u + ∇uᵀ)] = ν_eff ∇²u + (∇ν_eff·∇)u + (∇u)ᵀ∇ν_eff
```

and the last term reduces via continuity to `∂_j ν_eff ∂_i u_j`. The implementation drops **two**
`∇ν_t`-weighted terms; the paper concedes only one ("`∇ν_t·∇u`", `body.tex:468`,
`residual_floor_theorem.tex:151`). This matters because the paper itself establishes that
`ν_t ≈ 5.2 ν` and supplies ≈ 84 % of `ν_eff` (`body.tex:468`), and `ν_t` goes from ~0 at the wall
to large in the wake — i.e. `∇ν_t` is *not* small anywhere the flow is interesting.

**Consequence for the theorem.** Hypothesis (H2) — the floor is nonzero — is satisfied in part by a
modelling omission the authors chose. A referee will say: implement the conservative
variable-viscosity form (`ddx(ν_eff · ddx(u))` etc., ~10 lines in `residuals.py`, no new
machinery) and report how much of the 0.192 floor survives. **That ablation is missing and it is
the first thing I would ask for.** Without it, "the floor is genuine" is not established; only
"the floor of *this* operator is genuine" is.

### 2c. Is a RANS residual with the *predicted* `ν_t` a sound audit? Partly — and the paper
###     under-states the circularity

The audit consumes the closure emitted by the model it audits. With `ν_t` supplying 84 % of
`ν_eff`, the diffusive part of the momentum residual is *84 % the model's own guess*. The paper
concedes this in `sec:limitations` (`body.tex:1629–1635`) and reports `R² = 0.996` (grid backbone)
vs `0.67–0.70` (MeshGraphNet). Two things follow that the paper does not draw:

1. **The audit is not independent, and its degree of dependence is backbone-specific.** On
   MeshGraphNet, a third of the `ν_t` variance is unexplained, so the "same verifier across three
   backbones" claim is weaker than stated: the *functional form* is the same, but the *operator*
   differs backbone to backbone because `ν_eff` differs. A genuinely backbone-agnostic control
   exists and is cheap: recompute the residual for **all** backbones with `ν_t` taken from the
   *ground-truth* field (or with `ν_t ≡ 0`, a laminar monitor), and re-report ρ. If ρ survives,
   the trust claim is much stronger and the circularity objection is answered. If it collapses,
   the audit is reading its own closure error.
2. **A convergent-error failure mode is unaddressed.** A model that mispredicts `u` *and* `ν_t`
   in a compensating direction lowers its own audited residual. This is not hypothetical — it is
   exactly what "prediction below the truth floor in 160/200 cases" looks like.

### 2d. The label-rasterisation confound (major, and the decisive missing control)

`norm_truth_continuity_mean = 0.1226` vs `norm_truth_momentum_mean = 0.1448` (ratio 1.18).
**The continuity residual contains no `ν_t`, no closure, no no-slip term, and no pressure.** It is
`u_x + v_y` on the rasterised label. Its being 12 % of `U/L` in RMS means roughly *half* of the
measured floor is **pure interpolation error from rasterising an unstructured OpenFOAM solution
onto a 128² Cartesian grid**. (The AirfRANS reference solutions satisfy continuity to solver
tolerance on their own mesh; `rasterize_point_cloud(..., method="linear")` does not preserve
divergence-freedom.)

That is a serious problem for the *trust signal*, not just the theorem, because it admits an
untested confound: **cases whose geometry rasterises badly (thin sections, high camber, high AoA)
are both high-`‖r*‖` and hard to predict.** A per-case correlation of ρ ≈ 0.6 between residual and
error may be substantially a *geometry-difficulty proxy* rather than an error detector — which
would explain §1d (why the detector works even though the model is in the floor-dominated regime).

**The discriminating control costs one join on data already committed.** Compute the partial
Spearman correlation of per-case residual with per-case field error, **controlling for that case's
own `norm_truth`** (the per-case floor, present in the `per_case` array of
`results/certificates/residual_floor_realdata.json`, matched by case name to the per-case metrics).
If ρ(residual, error | ‖r*‖) collapses, the headline claim is confounded. If it survives, the
trust claim is *stronger than the paper currently argues* and should be reported that way. This is
the single most decisive number available and it requires no new training.

### 2e. Two residual operators, silently different (major for interpretation, minor for numbers)

The numpy monitor uses the **compact 3-point** Laplacian (`operators.py:186`,
`(f[i+1] − 2f[i] + f[i−1])/h²`). The differentiable twin used to build the corrector's residual
*input channels* uses a **repeated first difference** (`residuals.py:524–525`):

```python
lap_u = _ddx_t(_ddx_t(u, dx), dx) + _ddy_t(_ddy_t(u, dy), dy)
```

which is `(f[i+2] − 2f[i] + f[i−2]) / (4h²)` — a **wide, odd-even-decoupled** stencil with a
different symbol, different null space, and different noise characteristics. So the residual the
corrector is conditioned on (`run_v2.py:249–255`) is **not** the residual the trust layer scores
and not the one the gate tests.

Severity assessment: I checked whether this contaminates the headline numbers. It does not —
`run_v2.py:192` trains the Transolver with plain `torch.mean((pred - yb)**2)`, no physics term. So
this is not a numerical-validity problem for Table `tab:v2`. **But it is a live alternative
explanation for the W1 null result**: the corrector may have found the residual input useless
because it was fed a noisier, odd-even-decoupled operator, not because residual conditioning is
useless per se. Re-run W1 with the compact stencil before concluding "the residual is not the
source of the gain". Also note `body.tex:649–652` describes the training loss as a `CompositeLoss`
with a physics-residual term — **that description does not match the headline runs.**

### 2f. Two scalars both called `residual_norm` (minor)

`Diagnostics.residual_norm()` (`core/types.py:314–317`) averages `r²` over the **entire (ny,nx)
array** — no fluid mask — including the zeroed solid interior and the zeroed wall ring. The
certificate script uses a **fluid-masked** RMS (`scripts/probe_residual_floor.py:50–54`), which
still includes the zeroed wall ring in the denominator. Neither is wrong per se, but they are
different functions and the paper treats them as one quantity. The unmasked version introduces a
geometry-dependent deflation `√(N_fluid/N_total)` into the headline trust scalar. The airfoil
solid fraction is small (~0.3 % of a 128² crop), so the numerical effect is small — but state
which scalar produced which number.

---

## 3. Does the 128² grid at Re ≈ 10⁶ invalidate the residual? The honest answer

**Not fatal to the trust claim as scoped, but fatal to any wall-quantity claim — and it is the
reason the negative result is not generalisable.**

The honest expert position, in three parts:

1. **The reviewer's objection is right about what the operator is.** With `h ≈ 0.05 c` and
   `δ ≲ 0.02 c` at Re ~ 10⁶, the first off-wall cell centre sits at ~0.025 c — outside the boundary
   layer. The monitored `R_h` is the residual of a *smoothed, boundary-layer-free caricature* of
   the flow, not of the RANS solution. Wall shear does not exist on this grid in any meaningful
   sense.
2. **The paper's scoping is adequate for the *ranking* claims and it states the caveat honestly.**
   `sec:limitations` (`body.tex:1572–1578`) says exactly the right thing: the grid cannot resolve
   the BL, wall quantities are approximate, absolute MSE is not comparable to body-fitted solvers,
   and the wall ring is excluded so the signal is structurally blind near the wall. Rank
   correlations of a *bulk* consistency monitor against *bulk* field error are a legitimate
   measurement on this grid. I do not think the grid alone should sink the trust-signal claim.
3. **It is fatal to the generality of the negative.** The paper's negative ("the residual is a bad
   correction objective") is presented as a property of steady-RANS residual monitors and used to
   demarcate a regime boundary against `huang2025physicscorrect` and `learned2023residualcorrection`
   (`body.tex:141–145, 228–235`). But the floor that creates the boundary is composed of
   (a) rasterisation error in the labels (§2d), (b) dropped `∇ν_t` terms (§2b), (c) omitted
   no-slip (§4), (d) sub-cell BL. Items (a)–(c) are **implementation choices**, and (d) is a
   resolution choice. On a body-fitted grid with a conservative variable-viscosity operator and
   proper wall treatment, none of this analysis is known to hold. The paper must say so and stop
   presenting the boundary as intrinsic.

---

## 4. The λ-sweep "closes the objection in full" — it does not (major)

`residual_floor_theorem.tex:119–146` anticipates the obvious objection (the residual omits
no-slip, so of course uniform flow wins) and claims to close it "in closed form rather than on a
grid of sampled values", for **every** λ ≥ 0. Two defects break the argument.

**(1) `r_bc` does not measure no-slip violation.** In `bc_violation` (`residuals.py:195–201`) the
no-slip term is `prox × speed` with `prox = exp(−|sdf| / 3h)`, and wall-layer cells are boosted to
the full `speed`. But `3h ≈ 0.15 c` on this grid — so the penalty extends *far* outside any
boundary layer, into the suction-side accelerated region **where the true flow's speed exceeds
U∞**. `r_bc` therefore penalises *speed near the body*, a quantity the correct viscous solution
maximises. This is the mechanism behind the paper's own headline number,
`truth_bc2 = 0.00731 > uniform_bc2 = 0.00532`
(`results/control/bc_weight_sweep.json:11–12`): the truth looks *more* no-slip-violating than
uniform flow.

**(2) `diagnose` then deletes the only cells that measure no-slip properly.**
`residuals.py:348`: `bc[wall_ring] = 0.0`. The wall ring is exactly where `bc_violation` had set
the penalty to full `speed`. So the strongest and most physically meaningful part of the no-slip
signal is zeroed before the sweep ever sees it.

**Why this breaks the claim, not just the numbers.** `scripts/bc_weight_sweep.py` is a *post-hoc
rescaling of two already-logged scalars* (`bc_weight_sweep.json:6`: "no forward passes"). λ can
only rescale a fixed, malformed `r_bc²`. It cannot discover that a differently *formed* boundary
term flips the sign — and a correctly formed one (surface-interpolated velocity, or simply the
wall-ring cells that were deleted) assigns uniform flow the **maximum possible** no-slip violation,
`|U| = U∞` at the wall. So the paper's "there is no crossover weight" and "we can close it in full
rather than concede it" **overclaim**: closed-form over λ, yes; closed-form over the space of
no-slip formulations, no. Also note n = 10 (`results/control/bc_inclusive_sweep.json:4`) on a
dropout-FNO, not the headline Transolver.

A residual **partial credit** survives, and the paper should claim only this: at 128² even a
correctly formed no-slip term may not cleanly separate truth from uniform flow, because the first
off-wall cell already carries near-edge velocity. That converts leg (A) from a structural property
of steady-RANS residuals into an artefact of *formulation plus resolution*. Say both; the second
does not erase the first.

---

## 5. The residual-as-objective claim is not tested by the evidence offered (FATAL)

This is the decisive finding of the review.

`results/sensitivity/iters.json:5` records:

```json
"swept_knob": "DEQCorrector.max_iter (internal fixed-point cap)"
```

The DEQ corrector is trained by **supervised regression toward ground truth**
(`body.tex:586–590`). So `tab:iters` measures: *as a supervised corrector converges toward its own
fixed point, the monitored residual rises while field error falls.* That is anti-correlation along
**one** trajectory produced by an objective that is **not** the residual. Nothing in this paper
ever minimises `J = ½‖R_h‖²`. I grepped `scripts/` for any residual-descent experiment; the only
match was `make_cylinder_ood_figure.py`, which is a visualisation script.

The consequences are structural, not cosmetic:

- Contribution 3 is titled **"Residual-as-objective fails"** (`body.tex:128`).
- The **abstract** states "reducing it does not reduce field error" (`abstract.tex:17`) — but the
  residual is never reduced anywhere in the paper. It rises monotonically, 0.113 → 0.620.
- Theorem leg (iv) is explicitly about gradient flow `ė = −∇J`. **Never run.**
- `tab:positioning` (`body.tex:339–354`) awards this work the **only •** in the "Object." column
  and uses it to differentiate from *every* neighbouring paper.

**So the single claimed differentiator is the untested one.** The paper's own limitations bullet
concedes the sweep is "single-checkpoint, unseeded" (`body.tex:1639–1641`) — it never concedes the
sweep does not test the role at all.

**Compounding: the endpoints are cherry-picked.** In `iters.json` `mse_u` runs
3.924 → 2.460 → **2.287** → 2.439 → 2.570 → 2.575, while `residual_norm` runs
0.113 → 0.336 → 0.542 → 0.594 → 0.618 → 0.620. The text (`body.tex:1102–1104`) reports "residual
rises 0.11 → 0.62 while error falls 3.92 → 2.29 by iter 3" — comparing a 0→3 error change against
a 0→15 residual change. **Over iterations 3 → 15 the residual and the error both rise**, i.e. they
are *positively* correlated over most of the sweep, the opposite of the headline. The same pattern
is in `bc_inclusive_sweep.json`: `mse_u` 3.17 → **1.89** (iter 1) → 2.32 (iter 15), yet the theorem
section reports it as a monotone fall "3.17 → 2.32" (`residual_floor_theorem.tex:126, 142`).

**The fix is cheap and I want it before this claim is made again.** Freeze a trained prediction and
run N steps of gradient descent on `J(u) = ½‖R_h(u)‖²` **with respect to the field** (not the
weights) — a few dozen lines using the existing differentiable `physics_residual_torch`. Plot field
error against residual along that path. That either establishes the negative properly (and would be
a genuinely useful result) or falsifies it. Until then, the claim must be restated as what was
actually measured: *"along the supervised corrector's own convergence path, the monitored residual
and the field error are not co-monotone."* That is a much smaller claim, and it removes the • from
the "Object." column.

---

## 6. Forces: ρ_D = 0.84 is a measurement ceiling, not a model result; ρ_L = 0.998 is the easy
##    number twice over

From `results/control/force_vs_official_multischeme.json`:

| scheme | ρ_D (GT fields) | ρ_D (pred) | median C_D rel. err | ρ_L (GT) | ρ_L (pred) |
|---|---|---|---|---|---|
| legacy near-field | 0.839 | 0.84 | **3.32 (= 332 %)** | 0.876 | 0.88 |
| wall-extrapolated | 0.800 | 0.76–0.82 | 2.20 (220 %) | 0.882 | 0.89–0.91 |
| control volume | **0.611** | **0.31 / 0.51 / …** | 0.235 (GT) / ≈2.7 (pred) | 0.99999 | 0.998 |

**The honest reading:** *no scheme delivers both ranking and magnitude.* The scheme that ranks drag
(legacy, ρ_D = 0.84) has a **332 % median magnitude error on ground-truth fields**. The scheme that
gets magnitude closest (CV, 23.5 % on GT) ranks drag at ρ_D = 0.61 on GT and **0.31–0.51 on
predictions**. The paper's own framing — "the model saturates the measurement ceiling rather than
the prediction being the limit" (`body.tex:776–778`) — is factually supported and I credit it. But
it means **ρ_D = 0.84 is a statement about the integrator, not about NeuroForge**, and it should
not appear in the contributions list (`body.tex:112–115`) as if it were a model result.

**Structurally, drag is unavailable on this grid.** At Re ~ 10⁶ on an airfoil, skin friction is a
large fraction of total drag. The wall ring is zeroed and the BL is sub-cell (§3), so there is no
wall shear to integrate. Pressure drag alone, on a 128² raster, cannot recover C_D. The 332 % /
270 % errors are the expected consequence, not a bug.

**Does the paper lean on the easy lift number? Yes — and doubly.** (i) AoA is an **input channel**
(`body.tex:453`), and lift across an AoA sweep of roughly −5° to +15° is essentially determined by
AoA; ranking it is close to ranking the input. (ii) The CV integrator recovers lift from an
**outer-ring momentum balance**, i.e. circulation via Kutta–Joukowski — a far-field integral that
is robust at *any* resolution because it never touches the body. Drag from the same balance
requires the **wake momentum deficit**, which a linearly-interpolated, numerically diffused 128²
wake cannot supply — hence 270 %. So "ρ_L = 0.998, engineering-grade lift from the surrogate"
(`body.tex:786–792`) is true but nearly content-free for an aerodynamicist, while the quantity that
matters is the one that fails. To its credit the paper *does* say "we make no claim of accurate
absolute drag" (`body.tex:795–798`). The problem is the composite impression created by putting
−9/−21/−25 %, ρ_L = 0.998 and ρ_D = 0.84 in the same sentence of the contributions list.

---

## 7. Baseline fairness and the "three backbones" claim (major)

`results/mgn/mgn_results.json` shows MeshGraphNet at `mse_u` ≈ **10.97–16.93** against Transolver's
**0.133** — roughly **100× worse**, at matched parameter count (7.35 M both). Three problems:

1. **MGN's ρ = 0.851 is close to free.** The paper concedes it "partly reflects that model's wide
   error spread, which is easy to rank" (`body.tex:88–91`). That concession is the whole story: a
   model whose per-case error spans two orders of magnitude will show high rank correlation with
   *almost any* monotone quantity. This is not independent evidence of backbone robustness.
2. **It is not a MeshGraphNet.** It is trained at `"resolution": 128` on the rasterised grid
   (`mgn_results.json:14`) with `k = 8` neighbours — i.e. a message-passing net on a *grid graph*,
   not Pfaff et al.'s native-mesh MeshGraphNet. The paper should not present it as the
   message-passing baseline of record. (This is a hit on *that baseline*, not on the pipeline's
   geometry handling: the headline Transolver genuinely runs whole-cloud inference on the native
   point cloud — see the credit below.)
3. **"Three architecturally distinct backbones" is really 1 strong + 1 weak + 1 broken.** With
   `mse_u` at 0.13 / ~3 / ~13, the three points are not sampling architecture — they are sampling
   *model quality*. A backbone-robustness claim needs backbones of comparable competence.

I did not find evidence that MGN's weakness is a deliberate handicap; the harness looks
symmetric (identical rasterisation via `transolver_adapter`, identical `evaluate_cases`). My
reading is that it is a genuinely under-trained/ill-suited grid-graph model, which is a fair
outcome but a poor foundation for a robustness claim.

Separately, and to the authors' credit: the Transolver adapter runs **whole-cloud inference on the
native point cloud** and rasterises with the *byte-identical* path used for the labels
(`transolver_adapter.py:6–15, 96–97`). That is the right way to score a point-cloud model against
a rasterised target, and it is fair.

---

## 8. Novelty against the specific prior art

| Work | What it actually does | What NeuroForge adds beyond it |
|---|---|---|
| **`gopakumar2025pre`** (Gopakumar et al., *Calibrated Physics-Informed UQ*, arXiv 2502.04406) | Uses the **PDE residual itself as the conformal nonconformity score**, giving data-free coverage — in *residual* space. | Calibrates in **solution** space on the **deployed, corrected** field; adds the observation that residual-space smallness cannot certify solution accuracy when the monitor has a floor. **Real but incremental.** The paper honestly says it does "not claim to originate" residual-as-data-free-signal (`body.tex:266`). |
| **`roy2025anchor`** (ANCHOR) | Residual-based error estimator that **triggers a classical solver** on neural-operator time marching. | Steady-state instead of time-marching; conformal calibration instead of a threshold; per-case rather than per-step. **Setting change, not mechanism change.** NeuroForge's own fallback is a **stub** (`body.tex:602–610`), so it does not even instantiate ANCHOR's key component. |
| **`huang2025physicscorrect`** (Huang & Perdikaris, AAAI'26) | **Training-free** correction: solves a linearised inverse problem on the PDE residual each rollout step; up to 100× error reduction on transient benchmarks. | NeuroForge claims the complementary *negative* in the floored steady-RANS regime. **This is the intended differentiator and it is the untested one (§5).** |
| **`learned2023residualcorrection`** (Jha, CMAME 419:116595) | Residual-based **error-corrector operator** for neural-operator surrogates of nonlinear *variational* BVPs. | Same as above: the claimed boundary is asserted, not demonstrated by a residual-descent experiment. |
| **`mukherjee2026certification`** | Proves vanishing residuals ⇒ solution convergence only under extra compactness assumptions. | NeuroForge is the "practical complement" where the discrete residual provably does not vanish. **Fair positioning**, but the theorem establishing it is §1's known structure. |
| **`hillebrecht2025posteriori`** (IEEE TNNLS 36(1):1583) | Rigorous a-posteriori bound: true PINN error ≤ residual × operator stability constant. | NeuroForge notes its monitor lacks the vanishing-residual precondition, hence ranks rather than bounds. **Correct and well-argued** — this is one of the paper's better passages. |
| **`jia2026multigranularity`** | Multi-granularity conformal (case-level quantile regression + residual-normalised point-level bands) for neural-operator automotive surrogates on DrivAerML. | Nonconformity score is the **physics residual** rather than data-driven; and calibration on the corrected field. **Genuinely close prior art**, concurrent, and the paper handles it fairly. |
| **`garg2025dfuq`** (JCP 534:114012), **`ma2024uqno`** (TMLR), **`yu2026conformalpinn`** (JCP 561:114979) | Distribution-free / conformal bands for operators and PINNs; all calibrate the **raw** surrogate. | Calibrates the **deployed corrected** field, and demonstrates the raw-`q` certificate does not transfer through the corrector (`body.tex:1168–1216`). **This is a small but real and correctly-executed contribution** and, in my view, the most defensible novel claim in the paper. |
| **`wu2024transolver`** (ICML'24) | Physics-slice attention, linear-cost operator. | Used as a backbone, not extended. No architectural novelty is claimed, correctly. |
| **LSFEM / data oscillation** (Bochev–Gunzburger 2009; Morin–Nochetto–Siebert, SINUM 38:466, 2000) | **Not cited.** Residual functional as built-in estimator under norm equivalence; oscillation floor that is provably unavoidable. | The theorem's structural content is theirs. **Must be cited.** |

### Is it a new primitive, a new combination, or known?

**A new combination, on the weak end.** Every component is prior art: residual-as-error-signal
(`gopakumar2025pre`, `roy2025anchor`), split-conformal for operators (`ma2024uqno`, `garg2025dfuq`),
DEQ + JFB (`bai2019deq`, `fung2022jfb`), monotone-residual acceptance (any line-search/globalisation
in classical CFD), residual floor / undetectable modes (LSFEM, data oscillation). The assembly is
competent and the controls are unusually honest. The *only* claimed new primitive is the
dissociation's negative half, and that is the untested claim.

### Were the two desk rejections defensible on the merits?

**Yes.** I say this without pleasure. An editor at CMAME or JCP triaging on originality would ask
"what is the new thing?", find the answer in `tab:positioning` to be the sole • in the "Object."
column, and — if they looked, as a good editor does — find that the supporting artefact sweeps a
DEQ fixed-point cap rather than minimising a residual. The trust-signal half is explicitly conceded
as not-originated-here. The theorem restates LSFEM/data-oscillation structure without citing either
literature, in a journal (JCP) whose readership *is* that literature. That is a defensible desk
reject on originality, twice.

It is *not* a verdict that the work is bad. The engineering is careful, the negative results are
reported rather than buried, and the reproducibility infrastructure is better than most accepted
papers. The problem is that the paper is currently *marketed* on its weakest claim.

---

## 9. Which claims survive expert scrutiny

**Survive.**
- Residual is computed on physical denormalised fields with `ν_eff = ν + ν_t`, kinematic pressure.
  (Verified in code.)
- Positive per-case residual↔error rank correlation on Transolver, ρ = 0.611 ± 0.054 over 5 seeds
  — **conditional on the partial-correlation control of §2d.**
- Split-conformal coverage at target, stable over 20 calibration re-draws (0.895–0.902, std ≤ 0.014).
  This is properly done.
- The **corrected-field recalibration** finding: the raw-backbone `q` does not transfer through the
  corrector, with a clean within-run frozen-`q` contrast. Genuinely useful and, as far as I can
  tell, not in `ma2024uqno`/`garg2025dfuq`/`yu2026conformalpinn`.
- The DEQ corrector's −8 / −21 / −25 % on Transolver, sign-consistent on 5 seeds, with the honest
  disclosure that surface-pressure MSE is flat-to-worse (+0.6 %).
- The W1 null (residual input contributes nothing) — **subject to the stencil-mismatch caveat of §2e.**
- Banach contraction of the DEQ operator in `δ` (measured 0.78 < κ = 0.9), and the careful
  insistence that this is *not* a statement about the field's PDE residual.
- The resolution caveat as written in `sec:limitations`. It is honest and correctly scoped.

**Do not survive as stated.**
- "Residual-as-objective fails" (§5). Untested by the offered evidence; endpoints cherry-picked.
- "No weighting rescues the residual / no crossover λ" (§4). Rests on a malformed `r_bc` plus
  deletion of the wall ring, and on a post-hoc rescaling that cannot explore boundary-term *form*.
- "Quantified floor" leg (iii) (§1a). Upper bound presented as a lower bound.
- The theorem's explanation of *why* the detector works (§1d). Contradicted by
  `frac_pred_lt_truth = 0.80` in the authors' own certificate.
- "Backbone-robust across three architecturally distinct backbones" (§7). One strong, one weak,
  one ~100× worse grid-graph model whose high ρ the paper itself attributes to error spread.
- "Drag ranking recovered at ρ_D = 0.84" as a contribution (§6). It is the integrator's ceiling.
- "The monitored operator is a genuine RANS residual" (`body.tex:468`) — it is a RANS residual with
  two `∇ν_t` terms dropped and the closure supplied by the audited model itself (§2b, §2c).
- Implementation §: "CompositeLoss sums a masked data MSE, a differentiable physics-residual term,
  and a no-slip BC term" (`body.tex:649–652`) does not describe the headline runs
  (`run_v2.py:192`).

---

## 10. Questions I would require answers to

1. Where in the repository does *any* experiment minimise `J = ½‖R_h(u)‖²` with respect to the
   field? If nowhere, on what basis does `tab:positioning` claim the "Object." column?
2. What is the partial Spearman ρ(residual, error | per-case ‖r*‖)? (Data already committed.)
3. What fraction of the 0.192 floor survives when `ν_eff ∇²u` is replaced by the conservative
   `∇·(ν_eff ∇u)` form including both `∇ν_t` terms?
4. What is ρ(residual, error) when `ν_t` is taken from ground truth (or set to 0) for all three
   backbones — i.e. with a *backbone-independent* operator?
5. Does the λ-sweep conclusion survive a no-slip term evaluated on the **wall-ring cells that
   `diagnose` currently zeroes**, or on surface-interpolated velocity, rather than
   `exp(−|sdf|/0.15c) × speed`?
6. Does the W1 null survive when the corrector is fed the **compact-stencil** residual that the
   trust layer actually scores?
7. Given `frac_pred_lt_truth = 0.80`, the deployed model is in the floor-dominated regime the
   theorem says decouples residual from error. How do you reconcile that with ρ = 0.61?
8. Why is `Diagnostics.residual_norm()` unmasked while the certificate uses a fluid-masked RMS?
   Which produced `residual_error_spearman`?
9. What is the continuity residual of the AirfRANS labels **on their native mesh**? (If ~0, the
   0.123 measured on the raster is entirely your interpolation, and should be named as such.)
10. Is the MeshGraphNet a native-mesh model or a `k = 8` grid-graph at resolution 128? If the
    latter, on what grounds is it the message-passing baseline of record?

---

## 11. What this paper would need to be unignorable

Not more experiments in general — **four specific ones**, all cheap, plus one reframing.

1. **Run the residual-descent experiment** (§5). Field-space gradient descent on `J`, error vs
   residual along the path, on the Transolver. This is the missing evidence for the claim the whole
   novelty case rests on. If it confirms the negative, the paper has a real result; if it
   falsifies it, the paper has an *even more interesting* result (the residual is a usable
   objective and the corrector was just never trained that way).
2. **The partial-correlation control** (§2d). Prove the trust signal is not a rasterisation-
   difficulty proxy. One join on committed data.
3. **The conservative-viscosity floor ablation** (§2b) and the **ground-truth-`ν_t` monitor** (§2c).
   Together these convert "the floor exists" from "of our operator" to "of a competently
   discretised steady-RANS monitor" — or honestly report that it shrinks, which is also publishable.
4. **A correctly formed no-slip term** (§4), or an explicit withdrawal of "closed form for all λ".
5. **Reframe the contribution.** Lead with what is defensible and, as far as I can find,
   genuinely not in the literature: *a physics-residual trust signal that is calibrated on the
   **deployed, corrected** field, with the demonstration that the raw-backbone conformal
   certificate does not transfer through a corrector, plus the measured selective-prediction /
   risk-coverage numbers.* That is a solid, correct, modest paper — and it is what the evidence
   actually supports. Then report the residual floor as a *measurement* (with LSFEM and data-
   oscillation citations), and report the objective negative *as what was measured*: non-co-monotone
   behaviour along a supervised corrector's convergence path, on a floored under-resolved monitor.

A modest, correct paper is publishable. A paper marketed on its one untested claim is not.

---

## Recommendation

**Reject as written (4/10); resubmit after major revision.** The biggest technical risk to the
claims is §5 — the headline differentiator is unsupported by the artefact cited for it — compounded
by §2d, which admits a confound that could account for a substantial share of the headline trust
correlation. Fix those two and the paper crosses from "desk-rejectable on originality" to
"reviewable, modest, and correct".

**Credit where due:** the disclosure discipline in this manuscript is well above average. Reporting
the flat surface-pressure channel, the W1 null that undercuts the authors' own deployed design, the
retraction of the earlier "gate is vacuous" argument, and the `sec:limitations` list are the marks
of someone doing science rather than marketing. The problem is not integrity; it is that the claim
structure was built around the one thing that was never measured.

---

### Sources consulted for prior-art grounding
- [Bochev & Gunzburger, *Least-Squares Finite Element Methods*, Applied Mathematical Sciences 166, Springer, 2009](https://www.semanticscholar.org/paper/Least-Squares-Finite-Element-Methods-Bochev-Gunzburger/375cb6153d898bf1287fde06740c62a88d51691b)
- [Morin, Nochetto & Siebert, *Data Oscillation and Convergence of Adaptive FEM*, SIAM J. Numer. Anal. 38 (2000) 466–488](https://epubs.siam.org/doi/10.1137/S0036142999360044)
- [Scaling-robust built-in a posteriori error estimation for discontinuous LSFEM, IMA J. Numer. Anal.](https://doi.org/10.1093/imanum/drae105)
- [AirfRANS (Bonnet et al., NeurIPS 2022 D&B)](https://arxiv.org/html/arXiv:2212.07564)
- [AirfRANS reference implementation](https://github.com/Extrality/AirfRANS)
