# Audit: the residual-floor theorem (`sec:residual_floor`)

Auditor: theory/formal-guarantees. Date 2026-09-07.
Sources read: `docs/paper/sections/residual_floor_theorem.tex`, `docs/paper/body.tex`,
`src/neuroforge/physics/residuals.py`, `src/neuroforge/physics/operators.py`,
`src/neuroforge/core/types.py` (`Diagnostics.residual_norm`),
`src/neuroforge/solver/correction_loop.py`,
`src/neuroforge/data/airfrans_loader.py`, `src/neuroforge/data/pointcloud.py`,
`scripts/probe_residual_floor.py`,
`results/certificates/residual_floor_realdata.json`,
`results/control/bc_weight_sweep.json`.

**Execution constraint.** No shell was available in this session, so every number below is
either (a) copied from committed artifacts, (b) derived analytically from the source, or
(c) an explicitly-labelled order-of-magnitude estimate. Estimates are flagged `[EST]` and
each has a matching entry in §9 specifying the measurement that would settle it. A ready
measurement script for the two highest-value items is staged at
`C:\Users\Ali\AppData\Local\Temp\claude\D--Codes-Github-neuroforge-cfd\8c9dda4b-38f5-410c-b3a1-637308e0872e\scratchpad\audit_a.py`.

---

## 0. Verdict in one paragraph

The theorem is **not wrong in a way that invalidates the paper's thesis, but it is
mis-stated in its load-bearing leg, mis-scoped in its hypotheses, and it argues for the
thesis with the weakest instrument available.** Leg (iii) — the "quantified floor" that the
paper advertises as contribution (a) — is an **upper** bound where a floor requires a
**lower** bound, and it is numerically vacuous even in its own direction. Leg (i) is true
but elementary. Leg (iv) describes a dynamics the system never runs and is offered in
support of a table that shows the opposite sign. Meanwhile the two genuinely sharp,
genuinely novel, genuinely provable facts that are sitting in the code are relegated to
parentheticals: (1) the monitored operator is an **inconsistent** discretisation of the
governing equations, so its floor has a **nonzero continuum limit in closed form** and does
not vanish under grid refinement — this, not "discretisation error exists", is the answer to
the reviewer's killer objection; and (2) the monitor is **3 equations per cell constraining 4
unknowns per cell**, so at least a quarter of the state space is in the kernel by counting
alone, and the missing direction is *exactly* `nu_t`, whose Jacobian block is **diagonal with
closed-form entries**. Restated around (1) and (2), the result is defensible, tight, and
original. Restated as it currently is, a competent CFD reviewer kills it on leg (iii) alone.

---

## 1. The theorem as currently written, restated formally

### 1.1 The monitored operator, as implemented

Let `Omega_h` be the `128 x 128` uniform Cartesian grid on the crop
`[-1,2] x [-1.5,1.5]` (chord `c = 1`), so `h = dx = dy = 3/128 ~ 0.0234 c`
(`airfrans_loader.py:64,184`). Let `M ⊂ Omega_h` be the fluid cells (`mask > 0.5`) and

```
W  := { fluid cells with a 4-neighbour in the solid }          (the "wall ring")
A  := M \ W                                                     (the ACTIVE set)
```

(`residuals.py:140-154`, `residuals.py:344-348`). The state is
`u = (u, v, p, nu_t) ∈ R^{4|Omega_h|}`. The monitored operator is
`R_h : R^{4|Omega_h|} → R^{3|A|}`,

```
R_h(u) = ( r_c / (U/L),  r_x / (U^2/L),  r_y / (U^2/L) ) |_A ,

r_c = D_x u + D_y v
r_x = u D_x u + v D_y u + D_x p - (nu + nu_t) Lap_h u
r_y = u D_x v + v D_y v + D_y p - (nu + nu_t) Lap_h v
```

with `D_x, D_y` second-order central in the interior and **first-order one-sided on the two
edges of each axis** (`operators.py:106-111`), `Lap_h` the **compact 3-point** second
difference per axis with one-sided 3-point stencils at the edges
(`operators.py:186-189`), `nu_eff = clip(nu + nu_t, 0, inf)` pointwise
(`residuals.py:48-54`), and the per-equation non-dimensionalisation of
`residuals.py:355-360`. The scalar monitored norm used by the gate is
`Diagnostics.residual_norm` = RMS over the **whole grid** of
`sqrt(r_c^2+r_x^2+r_y^2)` (`types.py:314-317`); the probe script uses RMS over the **fluid**
(`probe_residual_floor.py:50-54`). The two differ by `sqrt(|M|/|Omega_h|) ~ 0.99` here, so
nothing turns on it — but the paper should say which one it means. **No BC term enters
`residual_norm`.**

`u*` denotes the *rasterised* AirfRANS ground truth: an OpenFOAM `kOmegaSST` finite-volume
solution on an unstructured mesh, **linearly interpolated** onto `Omega_h`
(`airfrans_loader.py:194`, `method="linear"`), with `u=v=nu_t=0` forced inside the solid.

### 1.2 The theorem as stated in the paper

> **Hypotheses.** (H1) `R_h` is the monitored operator above. (H2) `r* := R_h(u*) != 0`.
> Write `L := DR_h(u*)`, `sigma_min` its smallest nonzero singular value.
>
> **(i)** `R_h(u_inf) = 0` exactly, so `min J = 0` while `J(u*) = ||r*||^2/2 > 0`; `u*` is not a minimiser of `J = ||R_h||^2/2`.
> **(ii)** `grad J(u*) = L^T r*`; `u*` is stationary iff `L^T r* = 0`.
> **(iii)** The min-norm minimiser of the linearised objective is `e_inf = -L^+ r*`, with `||e_inf|| = ||L^+ r*|| <= ||r*|| / sigma_min`.
> **(iv)** Under residual-gradient flow, `d/dt (1/2)||e||^2 |_0 = -||Le||^2 - (Le)·r*`, positive on an open cone.

---

## 2. Correctness, leg by leg

| Leg | Verdict | Why |
|---|---|---|
| (i) | **True, and exactly proved.** | Every stencil in `operators.py`, including the one-sided edge stencils, annihilates a constant. `nu_eff` constant, `Lap_h` of a constant is 0. So `R_h(u_inf) = 0` identically. |
| (ii) | **True.** Chain rule; nothing to object to. |
| (iii) | **WRONG DIRECTION, and vacuous even in its own direction.** See §2.1. |
| (iv) | **Algebra correct; feasibility condition mis-stated; irrelevant to the deployed system.** See §2.2. |

### 2.1 Leg (iii) is the defect that must be fixed

Three distinct problems, in increasing severity.

**(a) The inequality points the wrong way.** A *floor* is a claim that the minimiser is
displaced from the truth by **at least** something. `||L^+ r*|| <= ||r*||/sigma_min` says the
displacement is **at most** something — it is an upper bound, and an upper bound on
displacement is evidence *for* the residual, not against it. As written, leg (iii) proves
"minimising the residual does not take you very far from the truth". That is the opposite of
the paper's thesis. The correct statement is one line:

```
sigma_max ||e_inf|| >= ||L e_inf|| = ||P_{range L} r*||
  =>  ||e_inf|| >= ||P_{range L} r*|| / sigma_max .
```

This is the honest quantified floor, and it is non-vacuous exactly when
`P_{range L} r* != 0` — which the paper already argues (the floor is not a pure gauge mode).

**(b) `sigma_min` is not a usable constant here.** `L` is roughly `3|A| x 4|Omega_h| ~
47000 x 65500` for a discretised steady advection-diffusion Jacobian with `nu ~ 1.6e-5` and
`h ~ 0.023`. Its smallest *nonzero* singular value is not separable from zero in floating
point and, on any physical grounds, is `O(nu/h^2)`-suppressed in the near-null directions.
`||r*||/sigma_min` is therefore an astronomically large number: the bound is true and empty.
Worse, `sigma_min` as defined is not even well-posed numerically — "smallest nonzero" of a
matrix with a continuum of near-zero singular values is a definition, not a quantity.

**(c) It identifies the wrong point.** The minimiser set of the linearised objective is the
affine space `e_inf + ker L`. Any descent method started at `u_hat` converges to the
**projection of `u_hat` onto that affine set**, not to the min-norm point. So
"irreducible field error `= ||L^+ r*||`" is wrong about *which* point the loop reaches,
independently of the bound direction. The min-norm selection is an arbitrary tie-break with
no dynamical meaning.

**Net:** contribution (a) of the "Relation to prior work" paragraph — "the *quantified*
operator-specific floor `||e_inf|| = ||L^+ r*|| <= ||r*||/sigma_min`" — is the paper's
advertised novelty and it is the part that does not survive review. Replace it.

### 2.2 Leg (iv)

The algebra `d/dt (1/2)||e||^2 = -||Le||^2 - (Le)·r*` is correct. Two corrections:

- **Feasibility.** The construction takes `Le = -alpha r*/||r*||`, which requires
  `r* ∈ range L`. With only "a nonzero component in `range L`" the correct choice is
  `Le = -alpha w`, `w = P_{range L} r* / ||P_{range L} r*||`, giving
  `(Le)·r* = -alpha ||P_{range L} r*||` and the condition `alpha < ||P_{range L} r*||`.
  The paper writes `alpha < ||r*||`, which is the wrong constant (too large).
- **Relevance.** *The deployed system never runs residual-gradient flow.* The DEQ corrector
  is supervised toward truth (`body.tex`: "the deployed corrector supervises toward truth
  rather than minimising the residual"), and the acceptance gate uses the residual as a
  **one-bit veto with backtracking** on a step generated elsewhere
  (`correction_loop.py:222-235`). Leg (iv) is a statement about a method the paper does not
  use, and the measured gate data *contradicts its practical import*: 99.8% acceptance and
  89.3% of accepted steps reduce true error. The paper must either drop (iv) or state
  explicitly that it characterises a hypothetical residual-descent corrector, and reconcile
  it with the gate result (see §6, overreach O4).

---

## 3. Hidden assumptions, and where they are false for the actual system

**A1. "`u*` is the discrete ground-truth field."** It is not a discrete solution of anything.
It is a *linear interpolant* of an unstructured finite-volume solution onto a uniform
Cartesian grid. The most direct evidence is in the paper's own artifact:
`norm_truth_continuity_mean = 0.1226`, i.e. `|div_h u*| ~ 0.12 U/L` in RMS. The continuum
truth is exactly divergence-free; continuity involves **no closure, no boundary condition and
no viscosity**. So 0.1226 is *pure representation error of the rasterisation and the grid* —
and it is 85% as large as the momentum floor (0.1448). This must be stated. As written, the
paper attributes the floor to "omits the no-slip closure *and* under-resolves the boundary
layer *and* drops `grad(nu_t)·grad(u)`" — none of which touch continuity.

**A2. The boundary layer is sub-cell — provably, not approximately.** At
`Re_c ~ 3e6`, `delta/c ~ 0.37 Re^{-1/5} ~ 0.019`, and `h = 0.0234 c`. So
`delta ~ 0.8 h`: **the entire turbulent boundary layer fits inside one grid cell.** This is
a computable fact and it is the single most persuasive sentence available for (H2). It is
currently only gestured at ("the `128^2` grid under-resolves the boundary layer"). Say the
number.

**A3. The wall ring is zeroed — including in the BC term.** `residuals.py:344-348` zeroes
`cont, r_x, r_y` **and `bc`** on `W`. Consequence: `bc_violation`'s deliberate emphasis of
the wall layer (`residuals.py:200`, `noslip[wall_layer] = max(noslip, speed)`) is applied to
exactly the cells that `diagnose` then sets to zero. **The wall-layer boost is dead code as
far as the monitor is concerned.** This is a code-level fact, not an inference, and it is
load-bearing for the "Robustness to the boundary term" paragraph (§6, overreach O3).

**A4. `nu_t` is predicted, unsupervised by the monitor, and enters only multiplicatively.**
`R_h` is **exactly affine in `nu_t`** (it appears only as `-(nu+nu_t) Lap_h u`). This is
stronger than the paper's "the residual-blind part of `nu_t`" and yields an exact, not
first-order, blindness result (§7, Lemma 3).

**A5. The monitored operator is not the RANS operator.** See §5 — this is the big one.

**A6. Two different operators are called "the monitored residual".** `PhysicsChecker` uses
the **compact** Laplacian (`operators.py:186`); `physics_residual_torch` — the training /
DEQ-loss operator — builds it as `ddx(ddx(.))`, i.e. the **wide** stencil
`(f_{i+2} - 2 f_i + f_{i-2}) / 4h^2` (`residuals.py:524-525`), which annihilates every
period-2 mode. The training-loss operator therefore has a **strictly larger** kernel than the
monitored operator. The committed artifact
`results/certificates/residual_floor_realdata.json` states
`"monitored_residual": "... matching physics_residual_torch"` — that string is **false**;
the probe calls `PhysicsChecker.diagnose`. Route to engineering: either fix the metadata
string or make the two Laplacians agree.

---

## 4. Non-vacuity and tightness

**Non-vacuous: yes, and the evidence for it is stronger than the theorem that uses it.**
The decisive numbers already in `residual_floor_realdata.json` are not the ones the paper
leads with:

- `||r*||` mean 0.192, median 0.133 (200/200 cases) — the floor.
- `||R_h(u_hat)||` mean 0.114 — the residual of a *demonstrably wrong* prediction.
- **`||R_h(u_hat)|| < ||r*||` on 160/200 cases (80%).**

That last line is the whole thesis in one number and it is gauge-free, linearisation-free
and assumption-free: *the monitor already ranks a wrong field above the truth, on four cases
in five.* The floor is not "real but negligible"; it exceeds the operating-point residual by
~1.7x in the mean. The paper buries this in the "Empirical confirmation" paragraph as a
consistency remark ("consistent with the loop's drive away from `u*`"). It should be the
headline.

**Tight: no, and in two places not even the right kind of statement.**

- The floor is quantified *against nothing*. `||r*|| = 0.192` is reported; there is no
  reference scale, no lower bound in terms of `h` or `Re`, and no decomposition into the
  causes the caveat lists. §7 gives the reference scale that makes it tight (the
  omitted-flux norm) and §9 gives the measurement.
- The kernel is an **existence statement with two examples**, not a characterisation. "A
  constant pressure shift, and the residual-blind part of `nu_t`" gives one dimension plus a
  hand-wave. The actual situation is far better and is provable: §7 Lemmas 1-3 give a
  dimension count (`>= 25%` of the state space), an exact diagonal Jacobian block with
  closed-form singular values, and an exact nonlinear null direction. The paper is leaving
  its strongest material on the table.
- A caution the paper is right about and should keep: **do not claim checkerboard pressure
  modes.** The one-sided edge stencils (`operators.py:110-111`) return `2*eps/h` on a
  period-2 field, so period-2 pressure is neither an exact nor a small-singular-value
  direction of this operator. The current text correctly does not claim it; keep it that way.

---

## 5. Is it more than "discretisation error exists"? — the decisive question

**As currently stated: no, and the reviewer wins.** Legs (i)-(iv) are, modulo the errors in
§2, a correct rehearsal of "the discrete operator applied to a non-solution leaves a
residual, and least-squares on an inconsistent linear system has a displaced minimiser".
A reviewer will write: *"Theorem 1 is the statement that `R_h(u*_h) = O(h^p) != 0`. This is
truncation error. Refine the grid."*

**But there is a sharper true statement available, and it defeats that objection outright.**
The monitored momentum operator is **not a consistent discretisation of the equations the
data solves.** For incompressible RANS with a Boussinesq eddy-viscosity closure,

```
u_j d_j u_i = -d_i P + d_j [ (nu + nu_t)(d_j u_i + d_i u_j) ]
            = -d_i P + nu_eff Lap u_i + (d_j nu_t)(d_j u_i + d_i u_j)
```

(using `d_j u_j = 0`; `P` is OpenFOAM's modified pressure, so the `2k/3` term is already
absorbed and does not appear — `simpleFoam`'s `divDevReff` carries the full symmetric
stress). The monitored residual omits the last term. Therefore, **for the exact continuum
solution `u*`**,

```
R(u*) = ( 0 , T_1 , T_2 ),      T_i := (d_j nu_t)(d_j u_i + d_i u_j) = 2 (S nabla nu_t)_i
```

with `S` the strain-rate tensor. This is a *consistency* error, not a *truncation* error:
it is `O(1)` in `h`, it survives `h -> 0`, and no grid refinement removes it.

This converts (H2) from an *empirically verified hypothesis* into a *proved consequence*,
which is exactly what makes the theorem non-vacuous and reviewer-proof. It also gives the
floor a closed form and hence a scale to be tight against.

**[EST] Magnitude sanity check.** In the wake/BL at `Re ~ 3e6`: `nu_t/nu ~ 1e3-1e4` so
`nu_t ~ 1e-2 m^2/s`; `|grad nu_t| ~ nu_t / delta_wake ~ 1e-2/0.05 = 0.2 m/s`;
`|S| ~ U/delta ~ 60/0.05 = 1200 s^-1`. Then `|T| ~ 2 |S| |grad nu_t| ~ 480 m/s^2`, and
non-dimensionalised by `U^2/L ~ 3600`, `|T| ~ 0.13`. The measured momentum floor is
`0.1448`. **These agree to within the precision of the estimate**, which is strong
circumstantial support that the momentum floor is dominated by the omitted closure flux
rather than by truncation. This must be *measured*, not estimated — `nu_t` is a data
channel, so it is a 30-line script (§9, E1). If it confirms, the paper has a closed-form,
grid-independent floor and the novelty question is settled.

**Honest split.** The floor has two provably different components:

| Component | Measured (128^2) | Behaviour as `h -> 0` |
|---|---|---|
| momentum, `||r*_mom||` | 0.1448 | `-> ||2 S grad nu_t|| > 0` (consistency error, **does not vanish**) |
| continuity, `||r*_cont||` | 0.1226 | `-> 0` (representation/truncation error; but bounded below by the source-mesh interpolation error, and not below `~1e-1` at any `h` a surrogate uses) |

Report both. Claiming the *whole* floor is grid-independent would be false; claiming *none*
of it is would concede the reviewer's point. The split is the honest and the strongest
statement.

---

## 6. Does it support the weight the paper puts on it? — overreach list

The intended headline (`body.tex:56-59, 246-249, 269-270`) is: *the physics residual can
detect error but provably cannot certify or fix it.* Assessment:

**Licensed by a corrected theorem: the "cannot certify" half.** §7 Theorem B gives an
unconditional impossibility (unbounded sublevel sets), which is exactly "residual-space
smallness cannot certify solution-space accuracy". This is stronger than what the paper
currently claims and easier to defend.

**Overreach O1 — `body.tex:269`, "the *residual-floor theorem* quantifying the
operator-specific detection limit".** Nothing in the theorem quantifies a detection limit.
Leg (iii) is an upper bound on a displacement (§2.1) and is vacuous. There is no measured
`sigma_min`, `sigma_max`, `||P_{range L} r*||` or `dim ker L` anywhere in `results/`. Either
run E3/E4 and quantify, or delete the word "quantifying" and claim the *structural* result
(kernel dimension count + exact `nu_t` block), which needs no measurement.

**Overreach O2 — `body.tex:229-234, 335`, the theorem "marks the boundary" between regimes
where residual-driven correction succeeds and fails.** The theorem contains no statement
about the success regime. It does not prove that `r* = 0` implies residual-driven correction
works (it does not — leg (iv)'s cone degenerates when `r* = 0`, but ill-conditioning and
non-convexity remain). The positioning table (`tab:positioning`) rests a column on this. The
defensible version: *"the floor is a necessary condition for the failure mode we observe, and
prior successes operate where it is absent"* — a scoping claim, not a dichotomy theorem.

**Overreach O3 — the "Robustness to the boundary term" paragraph, claim (A): the wall-ring
zeroing makes (A) not a closed-form result about no-slip weighting.** The paragraph
asserts that `rho_lambda(u_inf) < rho_lambda(u*)` "for *every* `lambda >= 0`, in closed form
rather than on a grid of sampled values", and concludes "Up-weighting the boundary condition
cannot convert this residual into a usable correction objective." The closed form is correct
*arithmetic* — given `uniform_bc2 = 0.0053 < truth_bc2 = 0.0073`, no `lambda` can flip it —
but the input is contaminated by A3: **the monitor zeroes `bc` on precisely the cells where
the uniform field violates no-slip maximally** (penalty `= |U| = U_inf`, weight 1). The
surviving no-slip signal is the `exp(-|sdf|/3h)` band starting ~1.5 cells out, where the
weight is already `~0.6` and the *rasterised truth's* speed exceeds `U_inf` on the suction
side (sub-cell BL, A2) — which is why `truth_bc2 > uniform_bc2`.

[EST] Rough sizing: the `exp` band contributes `~430` cells at mean `w^2 ~ 0.2`, giving
`uniform_bc2 ~ 430*0.2/16384 = 0.0052` — matching the committed `0.00532` almost exactly, so
the band, not the far-field ring, is the whole of `uniform_bc2`. Un-masking the wall ring
(`~100` cells at weight 1) would add `~100/16384 = 0.0061` to `uniform_bc2` and only
`~100*(0.5)^2/16384 = 0.0015` to `truth_bc2`, i.e. `uniform_bc2 ~ 0.0114 > truth_bc2 ~ 0.0088`
— i.e. the sign would flip and claim (A) would fail. **This last inference is an estimate
only and must not be relied on until E2 reports.** What is *proved*, from the code alone, is
weaker but already sufficient to force the rewrite: the composite `bc` map whose sign drives
claim (A) has had its dominant no-slip cells deleted, so (A) is a numerical fact about a
masked composite, not a closed-form result about weighting the no-slip condition. The
paragraph cannot be defended as stated against a reviewer who reads `residuals.py:348`.

The *right* argument here is a dilemma, and it is robust to the outcome of E2:
> Either the wall ring is masked — and then the monitor is structurally blind to the no-slip
> condition, so no reweighting of the surviving band rescues it; or it is not masked — and
> then the monitor is dominated by the spurious `10-20x` near-wall residuals the code
> comments describe (`residuals.py:338-343`), which are a stencil artifact of a `u=v=0`
> discontinuity at a sub-cell boundary layer, not physics. Neither branch yields a usable
> correction objective.

This is stronger than the current claim and it does not depend on an unmeasured sign.

**Overreach O4 — the section's opening sentence.** "The iteration sweep of `tab:iters` shows
a counter-intuitive fact: driving the monitored physics residual *down* drives the field
error *up*." `tab:iters` shows the residual **rising** `0.113 -> 0.618` while error **falls**
`3.92 -> 2.29` (and then rises again to 2.58 at `n=10,15`, so it is not a clean
anti-correlation in either direction). The loop is not driving the residual down. The
observation is a *dissociation*; the causal claim is an inference from leg (iv), which the
system never exercises. Restate as: "error and monitored residual move in opposite directions
under iteration" and cite the acceptance-gate data for the causal leg.

**Overreach O5 — `body.tex:145`, "fails where the monitored residual carries a floor, as
here".** "Carries a floor" is not a sufficient condition for failure and the paper does not
prove it is. Every practical discrete monitor carries some floor. The operative quantity is
the **signal-to-floor ratio at the operating point** — and the paper has the number for it
(`||R_h(u_hat)||/||r*|| ~ 0.59` in the mean, `< 1` on 80% of cases). Use the ratio, not the
mere existence of a floor.

**Not overreach, and worth keeping:** the detector leg. `||Le||` two-sided-bounded by
`sigma_min/sigma_max ||e||` in the far-from-truth regime, claimed only as a positive rank
correlation (`0.83`), with tightness explicitly deferred to conditioning on `(ker L)^perp`.
That is correctly hedged.

---

## 7. The strongest honest version

Four results. (0) replaces (H2), (B) replaces (i)+(iii) and is the certification-impossibility
statement, (C) replaces the kernel paragraph, (D) is the empirical operating-point fact. Legs
(ii) and (iv) become a remark.

### Theorem A (consistency floor; closed form; grid-independent)

> Let `u*` be a solution of the steady incompressible RANS system with Boussinesq closure and
> modified pressure `P`, with `u*, nu_t*` in `C^2`, and let `R_h` be the monitored operator of
> §1.1. Then for every `h`,
> ```
> R_h(u*|_{Omega_h}) = ( tau_h^c , T_1 + tau_h^x , T_2 + tau_h^y ),
> T_i := (d_j nu_t*)(d_j u*_i + d_i u*_j) = 2 (S* grad nu_t*)_i ,
> ```
> where `tau_h^{.}` are the finite-difference truncation errors, `||tau_h|| = O(h^2)` on the
> smooth part. Consequently
> ```
> lim_{h -> 0} || R_h(u*|_{Omega_h}) || = || (0, T_1, T_2) ||  >  0
> ```
> whenever `grad nu_t*` is not orthogonal to the strain rate on a set of positive measure,
> which holds for any turbulent boundary layer or wake. In particular (H2) is a **theorem, not
> a hypothesis**, and the floor is **not removable by grid refinement**.

*Proof.* Substitute the momentum equation, expanded with `div u* = 0`, into the definition of
`r_x, r_y`; every term cancels except the omitted stress-divergence contribution
`(d_j nu_t)(d_j u_i + d_i u_j)`. Consistency of central differences on `C^2` fields gives the
`O(h^2)` remainder away from the wall; take `h -> 0`. Continuity contributes no `O(1)` term
because the code's `r_c` *is* a consistent discretisation of `div u`. QED

*Caveat to state explicitly (two parts).* (a) The measured `||r*|| = 0.192` is evaluated on
the **rasterised** `u*`, which is not `C^2` and carries interpolation error from the source
mesh; it therefore mixes `||T||` with representation error. (b) `nu_t` on the rasterised truth
is likewise a linear interpolant, so `D_j nu_t` computed with `operators.py` stencils is *not*
`d_j nu_t*` — it is the discrete gradient of an interpolant of a finite-volume field. E1
therefore measures `||2 S_h grad_h nu_t||` on the *same fields with the same stencils* as
`r*_mom`, which is the correct apples-to-apples comparison for the **dominance** claim at
`h = 0.0234c`, but it is **not** a measurement of the continuum limit `||T||` in the theorem.
Those are different claims; the one the paper needs is dominance at the deployed `h`, and E3
(refinement) is what speaks to the limit. Do not let E1's number be read as validating the
`h -> 0` statement.

### Theorem B (certification is impossible at every threshold; exact, no linearisation)

> Let `S(t) := { u : ||R_h(u)|| <= t }`. Let `e_p` be the field `(0,0,1,0)` (unit constant
> pressure). Then `R_h(u + c e_p) = R_h(u)` for all `u` and all `c ∈ R`; hence `S(t)` is an
> unbounded set for **every** `t >= 0`, and
> ```
> sup { ||u - u*|| : u ∈ S(t) } = +infinity   for all t >= 0.
> ```
> Combined with Theorem A, `u* ∉ S(t)` for any `t < ||r*||`. Therefore no threshold on the
> monitored residual, however small, bounds the distance to the truth; and the truth itself is
> rejected below the floor.

*Proof.* `p` enters `R_h` only through `D_x p, D_y p`; every stencil in `operators.py`,
including the one-sided edge stencils, annihilates a constant. QED

*Pre-empting the obvious objection ("that is only the pressure gauge — quotient it out"):*
Theorem C gives the gauge-free version, and Theorem D gives a gauge-free measurement
(`mse_u` is a velocity error).

### Theorem C (the kernel: counted, and characterised in closed form)

> **(C1) Dimension count.** Restrict the state to the fluid, `L : R^{4|M|} -> R^{3|A|}`
> (the solid degrees of freedom are handled separately in C2, so that this count and C2 add
> rather than overlap). Since `A = M \ W`,
> ```
> dim ker L >= 4|M| - 3|A| = 4|M| - 3(|M| - |W|) = |M| + 3|W| >= |M|.
> ```
> **At least one quarter of the fluid state space is invisible to the monitor at first
> order**, for the structural reason that the monitor imposes three equations per cell on four
> unknowns per cell — plus three further dimensions for every wall-ring cell, whose residuals
> are zeroed outright. E4 returns `|M|` and `|W|` and hence this bound as an exact integer.
>
> **(C2) Solid-interior invisibility (exact).** No active cell is 4-adjacent to a solid cell
> (by definition of `W`), and every interior stencil is 4-local. Hence, for cases where the
> body does not touch the outer border (all of AirfRANS: the crop is `3c x 3c` about a unit
> chord), all `4|solid|` degrees of freedom on solid cells are **exactly** in `ker R_h` —
> not merely `ker L` — as a nonlinear statement.
>
> **(C3) The `nu_t` block is diagonal, with closed-form entries (exact, not first order).**
> `R_h` is **affine** in `nu_t`. For any perturbation `delta` supported on `nu_t`,
> ```
> R_h(u* + delta) - R_h(u*) = ( 0, -delta ⊙ Lap_h u*, -delta ⊙ Lap_h v* )|_A   EXACTLY,
> so  || R_h(u* + delta) - R_h(u*) ||^2 = sum_{k ∈ A} c_k^2 delta_k^2,
> c_k := sqrt( (Lap_h u*_k)^2 + (Lap_h v*_k)^2 ) / (U^2/L).
> ```
> Consequently, for `Omega_eps := { k ∈ A : c_k <= eps }` and any `delta` supported in
> `Omega_eps`, `||R_h(u*+delta) - R_h(u*)|| <= eps ||delta||`. **The sublevel set `S(t)`
> therefore contains a ball of radius `(t - ||r*||)/eps` inside a coordinate subspace of
> dimension `|Omega_eps|`,** for every `t > ||r*||`.
>
> *Physical reading:* eddy viscosity is unobservable from a momentum residual wherever there
> is nothing to diffuse. In the freestream `Lap_h u* ~ 0`, so `c_k ~ 0` and `nu_t` may be set
> arbitrarily there without moving the monitor.

*Proof.* (C1) rank-nullity. (C2) locality of the stencils plus the definition of `W`; note
that on the outer border the one-sided stencils reach two cells, which is why the "body does
not touch the border" hypothesis is stated. (C3) `nu_eff = nu + nu_t` multiplies `Lap_h u`
pointwise and appears nowhere else, so `R_h` is affine in `nu_t` with the stated
(diagonal) linear part; the `clip(nu_eff, 0, inf)` is inactive wherever `nu + nu_t > 0`,
which holds on the AirfRANS truth (`nu_t >= 0` by construction, `airfrans_loader.py:204`).
QED

**This is the "explicit characterisation of the kernel for the specific discrete operator"
the paper claims but does not have.** It is exact, closed-form, physically interpretable,
and its size is one histogram away from being a number (E4).

### Theorem D (the operating-point fact — already measured)

> On the 200 AirfRANS `full` test cases, `||R_h(u_hat)|| < ||R_h(u*)||` on **160/200** cases,
> with means `0.114` vs `0.192`. The monitor therefore assigns a *lower* residual to a field
> with large solution error than to the reference solution, on 80% of the test set. Since the
> error metric is a velocity error, this is invariant to the pressure gauge of Theorem B.

*This is a measurement, not a theorem, and should be labelled as such.* It is the single most
persuasive line in the section.

### Remark (replacing legs (ii) and (iv))

`grad J(u*) = L^T r*`, so `u*` is stationary for the residual objective only in the
non-generic case `r* ∈ ker L^T`. Under residual-gradient flow,
`d/dt (1/2)||e||^2|_0 = -||Le||^2 - (Le)·r*`, which is positive on the open cone
`(Le)·r* < -||Le||^2` — non-empty whenever `P_{range L} r* != 0`, and reached for
`Le = -alpha P_{range L} r* / ||P_{range L} r*||` with `0 < alpha < ||P_{range L} r*||`.
*This characterises a residual-descent corrector, which this system does not deploy;* the
deployed gate uses the residual as a one-bit veto with backtracking and empirically admits
steps that reduce error on 89.3% of cases. The two are consistent: a veto on a truth-supervised
step is a much weaker use of the residual than a search direction, and only the latter is
subject to the cone.

---

## 8. Relation to known theory — honest placement

**The classical chain.** `||e|| <= C_stab ||R(u_hat)||` (Trefethen-Bau; a-posteriori residual
bounds generally) is *calibrated by the identity `R(u*) = 0`*. Our setting is the
**inconsistent-operator** case in which that calibration point is absent: `R(u*) != 0` with a
nonzero continuum limit (Theorem A). So the correct placement is **complement, not special
case** — the classical bound is not false here, it is uninformative, because the best
achievable right-hand side is bounded below by `||r*||`.

**Becker & Rannacher 2001 (DWR).** DWR exists precisely because residual magnitude is not an
error estimate; it weights the residual by an adjoint solution to produce a goal-oriented
estimate. The reviewer's obvious follow-up is: *"then do DWR for lift/drag."* The paper needs
an answer and should state it as an argument, not a theorem:
> A dual-weighted estimate `J(u*) - J(u_hat) ~ <z, R(u_hat)>` inherits the primal operator's
> inconsistency: with `R(u*) != 0` the identity acquires an uncontrolled `<z, r*>` term, so the
> estimate is biased by the floor weighted by the adjoint. DWR fixes *"magnitude is not
> error"*; it does not fix *"the operator is not the equation"*. Additionally, the adjoint
> requires a linearised RANS solve on the deployed grid, which is the cost the surrogate
> exists to avoid.
**Do not claim** that DWR provably fails here — that would require constructing `z` and
bounding `<z, r*>` from below.

**Hillebrecht 2025 (a-posteriori PINN error bounds) and Mukherjee 2026 (vanishing residuals
guarantee convergence under compactness).** I could not verify either from primary sources in
this session and I decline to paraphrase them from memory. State the relation
**conditionally**: *"a-posteriori PINN bounds of the form `||e|| <= C(residual)` are
conditional on the residual vanishing at the exact solution; our monitor does not satisfy that
premise."* That sentence is safe because it is a statement about the *structure* of such
bounds, which `body.tex:255-261` already makes correctly. Before submission, someone must read
both papers and confirm (i) that Hillebrecht's stability constant is for the *exact continuum*
operator and (ii) that Mukherjee's compactness hypothesis is what is claimed. **Flagging as an
unverified citation-level risk.**

**Where the genuine novelty is, stated narrowly enough to survive.** Not "a floor exists"
(elementary), not "residual bounds error only via a stability constant" (classical), not "the
truth is not a minimiser" (immediate from the pressure gauge). It is:
1. **A closed-form, grid-independent floor for a specific, widely-copied monitored operator**
   (the `nu_eff * Lap u` simplification of the RANS momentum equation is *the* standard choice
   in physics-informed CFD surrogates). Theorem A says everyone using that operator on a
   variable-`nu_t` closure has an `O(1)` floor they have not accounted for. That is a real
   message to a real audience.
2. **An exact, diagonal characterisation of the undetectable subspace** with computable
   singular values and a physical reading (Theorem C3).
3. **The measured dissociation on a competitive backbone** (Theorem D + `tab:iters` + the
   acceptance-gate data).

Points 1 and 2 are what should carry the originality argument that got desk-rejected twice.
Neither is currently in the theorem.

---

## 9. Ranked experiments and derivations

Ordered by (decisiveness / cost). E1 and E2 change verdicts; do them before resubmission.

**E1 — Measure the omitted closure flux `||2 S grad nu_t||` on the AirfRANS truth.**
*Cost: ~30 lines, minutes, existing `r128` cache.* `nu_t` is a data channel; compute
`T_i = (D_j nu_t)(D_j u_i + D_i u_j)` with the same `operators.py` stencils, same masking,
same `U^2/L` scaling, and compare to `norm_truth_momentum_mean = 0.1448`.
*Decides:* whether Theorem A's `h`-independent term dominates the momentum floor. If
`||T|| / ||r*_mom|| ~ 0.5-1`, the paper has a closed-form grid-independent floor and the
"just discretisation error" objection is dead. **This is the highest-value hour in the
project.**

**E2 — Decompose `bc^2` and re-run the `lambda` sweep without the wall-ring zeroing.**
*Cost: post-processing on the existing cache; the staged script `audit_a.py` does it.*
Report, for truth and uniform: no-slip-band contribution, wall-ring contribution, far-field-ring
contribution, with and without `bc[W] = 0`. *Decides:* whether "Robustness to the boundary
term" claim (A) survives, or must be rewritten as the dilemma in O3. My [EST] says the sign
flips. Either way the paragraph changes; better to find out internally.

**E3 — Grid refinement of `||r*||`, split into continuity and momentum, at
`res ∈ {32, 48, 64, 96, 128, 192, 256}`.** *Cost: moderate; feasible — `PointCloudCase`
retains `pos`, `features` (cols include `sdf`, normals) and `targets`, so
`airfrans_loader._sim_to_pair` can be re-driven at any resolution from
`data/cache/airfrans_pc_full_test_n200.pkl` by reassembling the 12-column array.* Run on ~20
cases. *Decides:* the split in §5. Expected and publishable either way: `||r*_cont||` should
decay (representation error), `||r*_mom||` should plateau at `||T||` (consistency error). A
plateau in momentum with decay in continuity is a *figure*, and it is the cleanest possible
refutation of "refine the grid".

**E4 — Histogram of `c_k = |Lap_h u*|, |Lap_h v*|` over the active set, and the exact
`|Omega_h|, |M|, |W|, |A|`.** *Cost: trivial; in `audit_a.py`.* **Schedule E1, E2 and E4 as a
single task:** all three run off the existing `data/cache/airfrans_full_test_r128_n200.pkl`
with no model loading and no forward passes, so one execution of `audit_a.py` (plus the ~30
lines of E1) returns the closure-flux number, the `bc` decomposition and Theorem C's exact
dimension count together. Report
`dim ker L >= 4|Omega_h| - 3|A|` as an integer, and `|Omega_eps| / |A|` for
`eps ∈ {1e-3, 1e-2, 1e-1} x max_k c_k`. *Turns Theorem C from existential into numerical*
and supplies the "how large a subspace" the audit brief asks for. No linear algebra needed —
the block is diagonal.

**E5 — Assemble `L` explicitly at `res = 32` or `48` and report `sigma_max`,
`||P_{range L} r*|| / sigma_max`, `dim ker L` numerically, spectrum.** *Cost: `L` is
`~3000 x 4000` at `res=32`; `R_h` is quadratic in the state so central finite differences
give the Jacobian to machine precision in `4N` cheap evaluations; dense SVD is seconds.*
*Gives:* the corrected lower bound of §2.1 as an actual number, and a verification of the
counting bound. **State plainly in the paper that magnitudes at `32^2` do not transfer to
`128^2`; only the structure does.** Lower priority than E1-E4 because the structural results
(C1-C3) do not need it.

**E6 — One derivation, no compute: the refinement `h` required to bring the floor below the
operating-point residual.** From E3's fitted `||r*_cont||(h)` and E1's `||T||`, solve for the
`h` at which `||r*|| < 0.114`. If (as expected) `||T||` alone exceeds it, the answer is
"no `h`", which is the strongest possible sentence and costs nothing once E1 and E3 are in.

**Routing to engineering (not experiments, but must be fixed):**
- `results/certificates/residual_floor_realdata.json` metadata string
  `"monitored_residual": "... matching physics_residual_torch"` is false (A6). Fix the string,
  or reconcile the two Laplacians.
- `residuals.py:200` (wall-layer no-slip boost) is dead code given `residuals.py:348`
  (`bc[wall_ring] = 0.0`). Either remove the boost or stop zeroing `bc` on the ring; the
  current pair silently deletes the monitor's only strong no-slip signal.
- Decide and document whether the monitored scalar is RMS over the grid
  (`Diagnostics.residual_norm`) or over the fluid (`probe_residual_floor.py`); the paper
  should name one.

---

## 10. Summary of required edits to `sections/residual_floor_theorem.tex`

1. **Add Theorem A as leg (0)** and demote (H2) from hypothesis to corollary. This is the
   single change that answers the originality objection.
2. **Delete leg (iii) as written.** Replace with the lower bound
   `||e_inf|| >= ||P_{range L} r*|| / sigma_max`, or — better — with Theorem B, which needs no
   linearisation, no pseudoinverse and no singular values.
3. **Replace the kernel paragraph with Theorem C** (C1 counting, C2 solid interior, C3 exact
   diagonal `nu_t` block). Delete "the residual-blind part of `nu_t`" as a phrase; it is
   now a formula.
4. **Promote the `160/200` result to the section's lead** (Theorem D). It is the strongest,
   most gauge-robust, already-measured fact in the section.
5. **Fix the opening sentence** (O4) and the `alpha < ||P_{range L} r*||` constant in (iv);
   demote (iv) to a remark scoped to residual-descent correctors, and reconcile it with the
   acceptance-gate numbers.
6. **Rewrite "Robustness to the boundary term"** as the dilemma of O3 after E2 reports.
7. **Soften `body.tex:229-234, 269, 335`** per O1, O2, O5: claim a scoping statement and a
   signal-to-floor ratio, not a dichotomy theorem and not an unquantified "detection limit".
8. **State the two hard numbers** that currently only appear implicitly: `h = 0.0234c` vs
   `delta ~ 0.019c` (the BL is sub-cell), and `||R_h(u_hat)|| / ||r*|| ~ 0.59`.

The corrected result is narrower than the current one in exactly one respect (it does not
"quantify the detection limit"), and substantially stronger in three (a grid-independent
closed-form floor, an unconditional certification-impossibility statement, and an exact
kernel characterisation with a countable dimension). That trade is worth making.
