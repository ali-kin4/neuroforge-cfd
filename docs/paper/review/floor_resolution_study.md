# Floor resolution study — the blocking question, answered

Date: 2026-09-07. This is the report `REFRAME_PLAN.md` §4 blocks on. It decides
which sentence is the paper's headline.

**Verdict: PLATEAU, decisively, and by the pre-registered rule.** The residual
floor does not decay under refinement — it *grows*. The strong branch of §4's
decision tree is the one that obtains.

---

## 0. The answer in one table

24 AirfRANS test cases, three levels, scored over a region fixed in **physical**
units (`sdf ≥ 0.0352` chord = 1.5 × the 128² cell) so that refining the grid
cannot manufacture a plateau by unmasking the steep near-wall cells.

| level | h | ‖r\*‖ | omitted ∇ν_t·∇u | ‖R_h(u_∞)‖ |
|---|---:|---:|---:|---:|
| 128² | 0.02344 | 0.0624 | 0.0018 | 0 |
| 256² | 0.01172 | 0.0779 | 0.0039 | 0 |
| 512² | 0.00586 | **0.1067** | **0.0054** | 0 |

Fitted `‖r*‖ ~ C h^p` gives **p = −0.387** — not merely inside the
pre-registered PLATEAU band (`p < 0.5`), but the wrong sign for decay
altogether. The floor **rose in 21 of 24 cases**. A 4× refinement multiplies it
by 1.71.

Producer: `scripts/floor_resolution_ladder.py` → `results/floor_resolution_ladder.json`.
The decision rule was committed in `bf5106c` **before** the ladder was run.

---

## 1. Why this is not an artifact — four independent checks

A rising floor is the sort of result that is usually somebody's bug. Four
things had to be ruled out, and were.

**The operator is second order.** Method of manufactured solutions on the same
monitor: `p = 2.06` in the interior band, `p = 1.65` native. The discretisation
is doing what it claims, so a non-decaying floor on real data is a statement
about the data, not about a broken stencil. (`gates.mms_order`, pass.)

**The pipeline reproduces the published number.** The committed floor is
reproduced to `3.3e-8` relative error. (`gates.reproduction`, pass.)

**The rise is not the mask unmasking near-wall cells.** That is exactly what the
physical-units exclusion is for. The default one-cell-ring mask — the policy the
paper's own numbers use — gives `p = −0.344`, the same verdict; it is reported
as the appendix curve, not as the result.

**The rise is not the pipeline itself.** The MMS-raster control pushes a
*manufactured analytic* solution through the identical rasteriser and residual;
its residual **falls** with refinement (fine/coarse 0.41–0.74) while the real
AirfRANS truth's **rises** (0.86–3.23). Same code, opposite direction. Claim
only what that shows: the rasteriser does not manufacture rising residuals out
of nothing. It does **not** prove the real data's rise is free of interpolation
error, because the manufactured field has no boundary layer and is genuinely
easier to interpolate. The argument against interpolation as the driver is §1a
below, not this control.

**A curve that was in the first draft of this report and should not be quoted.**
The ladder also scores a set restricted to cells holding ≥8 source points, which
fits `p = −0.909`. That number is not evidence: the qualifying cell count falls
1863 → 1529 → 732 as `h` shrinks, because cells that dense are found only near
the wall. The set *migrates* toward the wall rather than staying fixed, so part
of that exponent is "we moved the scoring region". It is retained in the JSON
and excluded from the argument.

### 1a. Continuity and momentum rise in lockstep — which names the mechanism

The decisive decomposition, and it does not go the way either candidate
explanation predicted:

| level | ‖r\*‖ | continuity | momentum |
|---|---:|---:|---:|
| 128² | 0.0624 | 0.0477 | 0.0401 |
| 256² | 0.0779 | 0.0589 | 0.0510 |
| 512² | 0.1067 | 0.0791 | 0.0717 |
| fitted `p` | −0.387 | **−0.365** | **−0.419** |
| rose in | 21/24 | 21/24 | 21/24 |

The two components grow at the same rate, in the same cases. The 5-rung study
agrees across all 16 of its cases: `p_cont` and `p_mom` track each other to
within 0.01–0.09 everywhere.

That rules out both single-term stories. Had the **closure omission** driven the
rise it would have hit momentum only — continuity contains no `ν_t`. Had
**divergence-freedom lost in linear interpolation** driven it (the domain-expert
report's §2d) it would have hit continuity only. Neither did; they move together,
which points at a cause common to every equation.

The cause consistent with that is the **reference-operator mismatch**. The
AirfRANS solution satisfies a different discrete operator on a body-fitted mesh.
Interpolated onto a Cartesian raster it satisfies neither operator, in *every*
equation, and refining the raster resolves more of the mismatch rather than less.
This is also the direct evidence for the §4 headline, which is a claim about
exactly that mismatch.

**Caveat, stated rather than buried.** The source cloud's median
nearest-neighbour spacing is `4.4e-5` chord near the wall but `4.6e-3` in the
far field. At 512² the raster spacing `5.9e-3` is only 1.27× the far-field
spacing, so that level is close to the point where the raster begins
differentiating a piecewise-linear interpolant rather than sampling a solution.
It clears the pre-registered exclusion, but barely, and 1024² would not. The
128²→256² leg — comfortably above the limit on its own — already shows the rise,
so the verdict does not rest on the marginal level.

---

## 2. The corroborating study, at more rungs and fewer cases

An independent pass (**16 cases**, **5 rungs**, three exclusion bands) reaches
the same verdict on every band of every case: **PLATEAU**, order `p` from −0.91
to +0.31. Bands of 0.05, 0.1 and 0.25 chord all agree, as do the
repaired-operator variants. Its per-case ladders are in
`results/certificates/floor_resolution_decomposition.json`, and §6 reports it in
full.

The two studies trade off the right way: 24 cases × 3 levels here, 16 cases × 5
rungs there, different exclusion policies, same verdict. Neither is a fluke of
its own design. Where they disagree — leg (A) at the finest rung — §7 settles it
in favour of the finer ladder.

---

## 3. What this does to the paper

**H-B held, but it is not the mechanism.** The omitted `∇ν_t·∇u` term was
predicted, before the run, to *grow* with refinement — at 128² the sub-cell
boundary layer smears `∇ν_t`, so the term is under-estimated there. It does:
0.0018 → 0.0039 → 0.0054, `p = −0.808`. That is a model omission behaving like
one, and it never vanishes under refinement.

It is nevertheless **~5% of the floor** at 512² (0.0054 against 0.1067), and
§1a shows the rise is not momentum-specific. So the closure omission is a real,
non-vanishing component and *not* the driver. An earlier draft of this report
called it "the cleanest component to point at"; that was wrong, and the paper
must not lead with it.

**H-C held at these three levels — but see §7, which supersedes this heading.** The
16-case × 5-rung run finds the margin crossing zero by 512² (−2.8%), so the
conclusion below ("needs no weakening") does not survive the finer ladder. What both
studies agree on — a margin that collapses monotonically under refinement, and so a
claim that must be stated at the deployed resolution rather than as
resolution-independent — is the paragraph's last sentence, which stands.

The λ-sweep's closed form requires
`r_bc²(u*) > r_bc²(u_∞)`, and the theorem file attributes that inequality to the
128² boundary layer being sub-cell. It survives at every level in both bands:

| level | framework band (truth vs uniform) | fixed band |
|---|---|---|
| 128² | 0.00594 > 0.00543 | 0.00585 > 0.00537 |
| 256² | 0.00279 > 0.00256 | 0.01383 > 0.01246 |
| 512² | 0.00128 > 0.00126 | 0.04489 > 0.04044 |

So leg (A) holds at all three levels *of this ladder*; §7's finer ladder does not
reproduce that at 512², and its reading is the one to use. **In both studies**,
under the framework's own no-slip band —
which decays over `3·min(dx,dy)` and therefore *shrinks* as the grid refines —
the margin narrows from 9.4% to 9.0% to **1.6%**. Under a band fixed in physical
units it is stable at 9–11%. The paper should say the closed form is verified at
the resolutions tested and note that the framework's band is itself
resolution-dependent, rather than implying resolution-independence it has not
shown.

**Leg (i) was never at risk.** `REFRAME_PLAN.md` §3 worried that the
exactly-zero residual of the uniform field might be a rasterisation artifact. It
cannot be: a spatially constant field has identically zero finite differences at
every `h`, which the ladder confirms numerically at all three levels. Only the
floor's *magnitude* was ever grid-dependent, and it grows.

---

## 4. The headline this licenses

Per §4 of the plan, the plateau branch gives:

> You cannot audit a surrogate with a residual operator different from the one
> that generated its training data.

That generalises past 2-D, past 128², and past RANS, and it is not desk-
rejectable as "no new methodology" because it is a measured constraint on a
practice the field is currently building on. The floor is not a resolution
problem to be engineered away; refining the grid makes it **worse**, because
refinement resolves more of the very structure the monitored operator omits.

---

## 5. The separate hole this study does not close

R2's worst objection is that **no experiment in the manuscript minimises the
residual**, so the negative half of the headline dissociation was never tested.
That is closed separately by `scripts/residual_descent.py`
(`results/residual_descent.json`), reported here because it bears on the same
claim:

| arm | residual | field error | numpy monitor |
|---|---|---|---|
| from truth, descend residual | 0.181 → 0.029, **24/24** | 0 → median 0.91 | 0.181 → 0.037, 24/24 |
| from truth, descend error | no motion (∇ = 0 exactly) | 0 → 0 | no motion |
| perturbed, descend residual | 0.503 → 0.038, 24/24 | 2.56 → 3.96 mean | 0.503 → 0.044, 24/24 |
| perturbed, descend error | 0.503 → 0.312 | 2.56 → 0.000, 24/24 | 0.503 → 0.311 |

Started at the **exact ground truth**, with no network anywhere in the loop, 300
steps of descent on the monitored residual cut it by 84% in 24/24 cases and take
the field error from zero to a median of 0.91 — the range a trained backbone
*starts* in. The compact-stencil monitor the paper reports falls in 24/24 too,
so this is not the differentiable twin's wider Laplacian being gamed.

**The claim to make, because it holds in 24/24 rather than 6/24.** A blanket
"descending the residual makes the field worse" does not survive this data:
from a perturbed start it *cuts* the error in 18/24 cases, typically by 60%,
blowing up only in the other 6. What holds everywhere is a statement about
**iterate selection**:

| | error-optimal iterate | residual-chosen iterate |
|---|---|---|
| from truth | step 0, error 0 — in 24/24 | step ~230, error 0.77 |
| perturbed | step ~62, error 0.573 | step ~192, error 1.044 |

The two never coincide — they differ in 24/24 cases on both arms — and the
residual-chosen iterate is **strictly worse in 24/24**. Descent overshoots: the
error bottoms out early, then climbs while the residual keeps falling. Stop
where the residual tells you and you land 1.3× worse than a point you had
already passed through on your own trajectory.

This is precisely the reading R2 steelmanned, now measured: the residual is not
a usable *selector* along a correction path. It survives the "but it helped most
cases" rebuttal, and it retires the n=1 exposure of `tab:iters`.

**Two supporting details, stated carefully.** Residual descent terminates at
0.028 while the exact truth sits at 0.110 — the objective ranks a field with
error 1.31 as four times better than the field with error 0. And the
`perturbed_error` arm does not "recover the truth exactly": it reaches error
`~1e-4` but a residual of 0.227, twice the truth's own 0.110. A field that is
numerically indistinguishable from the truth carries double its residual, so the
monitor is not a fine-grained selector — which is an asset for the detector
claim's honest framing, not an embarrassment.

**Not used: the λ = 100 boundary-inclusive arm.** Its optimisation largely
failed — the residual fell in only 10/24 cases and the compact monitor in 7/24,
while the error reached 210 and the distance to the uniform field 14.6. That is
a badly scaled objective driving the field somewhere wild, not evidence about
no-slip weighting. The λ question is settled by the closed form in §3, not here.

---

## 6. The decomposition study, completed (16 cases × 5 rungs)

§2 above referred to this pass while it was still running on 3 cases. It has since
finished at **n = 16 cases × 5 rungs**, and it settles *why* the floor rises.
Producer: `scripts/floor_resolution_decomposition.py` →
`results/certificates/floor_resolution_decomposition.json`. **17.8 min, CPU-only.**

Primary metric is `band_0.1` — RMS over fluid cells with `sdf > 0.1` chord, excluding
the monitor's zeroed ring: a **fixed physical region** at every rung, for the same
reason §0 fixes its exclusion in physical units.

| N | h | band0.05 | **band0.1** | band0.25 | cubic C1 | repaired | omit ∇ν_t | MMS analytic | MMS raster | h/s_local |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.02362 | 0.05743 | **0.04593** | 0.03768 | 0.05117 | 0.04592 | 0.00161 | 6.9e-4 | 6.9e-4 | 8.08 |
| 181 | 0.01667 | 0.05931 | **0.05590** | 0.05145 | — | 0.05600 | 0.00255 | 3.4e-4 | 3.7e-4 | 5.70 |
| 256 | 0.01176 | 0.07374 | **0.07192** | 0.07139 | 0.08046 | 0.07197 | 0.00345 | 1.7e-4 | 2.8e-4 | 4.02 |
| 362 | 0.00831 | 0.09107 | **0.09049** | 0.09197 | — | 0.09054 | 0.00426 | 8e-5 | 3.0e-4 | 2.84 |
| 512 | 0.00587 | 0.10711 | **0.10737** | 0.10933 | 0.11426 | 0.10740 | 0.00484 | 4e-5 | 3.8e-4 | 2.01 |

All three bands rise monotonically at every rung. `band0.25` — the best-resolved
region, furthest from every near-wall confound — rises **×2.90** and does so in
**16/16** cases; it is the strongest series, not a hand-picked one.

**Two ratios, both quoted below, so they are not confused.** Ratio-of-means over the
ladder is **×2.34** (0.10737/0.04593) on `band0.1` and **×2.90** on `band0.25`;
mean-of-per-case-ratios on `band0.1` is **×2.56**. The `fine/coarse` column in the
verdict table is the per-case mean (2.56); table arithmetic gives 2.34.

| series | DECAY | rising (p<0) | order p | fine/coarse |
|---|---:|---:|---:|---:|
| **band 0.10 (primary)** | **0/16** | **15/16** | **−0.64 ± 0.29** | **2.56** |
| band 0.25 (best-resolved) | 0/16 | **16/16** | −0.77 ± 0.16 | 2.97 |
| omitted ∇ν_t term | 0/16 | 16/16 | −0.93 ± 0.20 | 3.87 |
| MMS raster (pipeline null) | 0/16 | 3/16 | +0.32 ± 0.27 | 0.65 |
| MMS analytic (truncation) | **16/16** | 0/16 | **+2.02 ± 0.05** | 0.061 |

Per-case order on the primary metric spans −0.96 to +0.11; the one non-negative case
is flat, not decaying. This corroborates §0 at 5× the case count and adds two things.

### 6.1 A third artifact control: C¹ re-rasterisation

§1 rules out rasterisation error with the MMS-raster null. The remaining version of
that objection is sharper: *you are twice-differencing a C⁰ piecewise-linear
interpolant, whose kinks sit at fixed physical triangle edges, so a second difference
across one scales like Δs/h and blows up as h shrinks.* Re-rasterising with
**CloughTocher (C¹, same triangulation)** answers it: the floor comes out **higher**,
not lower, at every rung tested (0.0512 / 0.0805 / 0.1143 versus 0.0459 / 0.0719 /
0.1074). If kinks drove the rise, C¹ would remove it. **Refuted.**

### 6.2 The mechanism: it is not the closure term, it is failed cancellation

Splitting the momentum residual into its terms (band 0.1, means over 16 cases):

| term | 128² | 512² | change |
|---|---:|---:|---|
| convective `u·∇u` | 0.1832 | 0.1994 | +9% (grid-converged) |
| pressure gradient `∇p` | 0.1802 | 0.1807 | **+0.3% (grid-converged)** |
| viscous `ν_eff∇²u` | 0.0023 | 0.0073 | small throughout |
| continuity `∇·u` | 0.0351 | 0.0794 | +126% |
| **residual = their non-cancelling remainder** | **0.0459** | **0.1074** | **+134%** |

**The individual terms are grid-converged; their cancellation is not.** AirfRANS
balances convection against pressure gradient in *its* finite-volume operator on *its*
body-fitted mesh. Under our cell-centred central-difference Cartesian operator that
balance does not hold, and the imbalance grows from 25% to 54% of the convective scale
as our stencil resolves finer scales of the reference field. Continuity behaves the
same way: the reference is divergence-free in its FV sense, not in ours.

This refines §3's reading. The omitted `∇ν_t` term does grow as predicted (H-B holds),
but it is **4.5% of the floor** at 512², and **repairing the full stress divergence
`div(ν_eff(∇u+∇uᵀ)) − ν_eff∇²u` moves the floor by less than 0.1%** (0.10737 →
0.10740). So the floor is not "the structure the monitored operator omits" in the
closure sense — it is the residue of **operator provenance**. The `∇ν_t` term is the
cleanest *illustration* of a non-vanishing component; it is not the driver.

> **Honest bound.** The repaired floor is a **lower bound** on reference-operator
> mismatch: SA source terms, FV flux reconstruction and mesh non-orthogonality are not
> modelled. And the MMS null tests the pipeline on a field smooth at the cloud scale;
> §6.1 covers sub-cell content, but the ladder cannot be pushed past `h ≈ s_local`
> without differentiating a reconstruction. The defensible claim is **"does not
> decay"**, not "grows without bound".

## 7. Leg (A): a numerical disagreement with §3, and the wording both studies support

§3 concludes "H-C held — leg (A) needs no weakening". The 16-case decomposition run
does **not** reproduce that at the finest rung, and the discrepancy should be settled
in the paper's favour of caution rather than silently averaged away.

`bc²` under the **framework's own** no-slip band (the quantity `bc_weight_sweep.py`
uses), truth versus uniform:

| N | truth bc² | uniform bc² | margin | leg (A) per case |
|---:|---:|---:|---:|---:|
| 128 | 0.006018 | 0.005507 | **+9.3%** | 14/16 |
| 181 | 0.004113 | 0.003828 | +7.4% | 12/16 |
| 256 | 0.002790 | 0.002609 | +6.9% | 13/16 |
| 362 | 0.001904 | 0.001832 | +3.9% | 12/16 |
| 512 | 0.001281 | 0.001318 | **−2.8%** | 12/16 |

**Where the two studies agree:** the margin collapses monotonically with refinement.
§3 measures +9.4% → +9.0% → +1.6% over three levels; this run measures +9.3% → +7.4%
→ +6.9% → +3.9% → −2.8% over five. Same trend, same cause — the framework's band
decays over `3·min(dx,dy)` and so shrinks with `h`.

**Where they differ:** whether the *mean* has crossed zero by 512². §3 (n = 24) has
truth still above by 1.6%; this run (n = 16) has it 2.8% below. A ±2% mean difference
between two 16–24-case subsets at a margin that has already collapsed to ~2% is
subset noise, not a contradiction. **Neither study licenses a resolution-independent
claim**, and the per-case counts here show it was never universal anyway: leg (A)
holds at **14/16** cases at 128², at **all five rungs in only 11/16** cases, and at no
rung in 1/16.

Two further facts this run adds:

* **The theorem's attribution is CORRECT, and this was a live alternative.**
  `bc_violation` is two additive terms — the no-slip band *and* the far-field mismatch
  on the outer one-cell ring. The uniform field has an identically zero ring by
  construction, so the ordering could have been an artifact of the crop's outer
  boundary rather than of the boundary layer. Splitting them: the truth's ring is
  **5.3e-5 of 6.0e-3 — under 1%**. The ordering really is about the near-wall band.
* **The crossover weight is O(10²–10³), not O(1) and not O(10⁶).** Where the ordering
  flips, `λ* = ‖R_h(u*)‖²/(bc²_unif − bc²_truth)` has median **581** (range 359–8549;
  at 512²: median 599, n = 4). The framework's own trust map uses λ = 1, so λ* ≈ 600
  is a monitor in which the no-slip term outweighs the entire PDE residual by two
  orders of magnitude — no longer a PDE monitor at all. This is a **real technical
  break**, not a formality, and not a practical rescue either.

**Mitigation, already in hand.** Leg (B) — along the correction path *both* increments
are positive, so `ρ_λ` rises for every λ — does **not** depend on the truth-vs-uniform
ordering, and leg (B) is the leg that actually supports `tab:iters`. Leg (i) is
*strengthened*: `R_h(u_∞) = 0` exactly at every `h`, while `‖r*‖` grows, so the gap
the objective must be blamed for widens with refinement.

## 8. Exact text changes for `sections/residual_floor_theorem.tex`

**(a) §"Assumptions, stated plainly" — delete the grid concession.** Replace

> "(H2) holds here because $R_h$ omits the no-slip closure *and* the $128^2$ grid
> under-resolves the boundary layer and drops the spatially-varying-$\nu_t$ term
> $\nabla\nu_t\!\cdot\!\nabla u$."

with

> "(H2) holds here for a reason that is \emph{not} grid resolution. Re-rasterising the
> same AirfRANS fields from their source point clouds at $128^2$--$512^2$ makes the
> floor \emph{grow}, not decay (fitted order $p=-0.64\pm0.29$ on a fixed physical
> region, $0/16$ cases decaying and $15/16$ rising, $\times 2.6$ over the ladder;
> \texttt{results/certificates/floor\_resolution\_decomposition.json}). The convective
> and pressure-gradient terms are individually grid-converged ($0.183\!\to\!0.199$ and
> $0.180\!\to\!0.181$); the floor is their non-cancelling remainder, growing from
> $25\%$ to $54\%$ of the convective scale. Restoring the omitted stress divergence
> moves it by ${<}0.1\%$. (H2) is a statement about auditing a finite-volume,
> body-fitted reference with a different discrete operator, not about a sub-cell
> boundary layer."

**(b) §"Empirical confirmation" — one sentence after the 0.192/0.133 numbers.**

> "The floor is not an artifact of the deployed grid: refined to $512^2$ it
> \emph{increases} by $2.6\times$ on a fixed physical region, while a manufactured
> potential flow---an exact solution of the continuum operator for any $\nu_t$---pushed
> through the identical rasterisation pipeline converges at order $2.02\pm0.05$ and
> remains $280\times$ smaller, and a $C^1$ re-rasterisation \emph{raises} rather than
> lowers the floor."

**(c) §"Robustness to the boundary term", leg (A) — weaken the universal quantifier.**
Replace

> "\textbf{(A)}~The uniform field remains a spurious minimum: it has both a smaller
> interior residual and a smaller no-slip term than the truth
> ($\overline{r_{bc}^2}=0.0053$ versus $0.0073$---on a $128^2$ grid the boundary layer
> is sub-cell, so the rasterised truth itself carries near-wall velocity), hence
> $\rho_\lambda(u_\infty)<\rho_\lambda(u^\star)$ for all $\lambda$."

with

> "\textbf{(A)}~At the deployed resolution the uniform field remains a spurious
> minimum: it has both a smaller interior residual and a smaller no-slip term than the
> truth ($\overline{r_{bc}^2}=0.0053$ versus $0.0073$; $14/16$ cases in the resolution
> study), hence $\rho_\lambda(u_\infty)<\rho_\lambda(u^\star)$ for all $\lambda$
> \emph{at that resolution}. We claim no more, because this leg is
> resolution-contingent: the framework's no-slip band is scored over a width
> $3\min(dx,dy)$ that shrinks with $h$, and the margin collapses monotonically under
> refinement ($+9.3\%$ at $128^2$ to $-2.8\%$ at $512^2$), admitting a finite crossover
> weight $\lambda^\star=\|r^\star\|^2/(\overline{r_{bc}^2}(u_\infty)-\overline{r_{bc}^2}(u^\star))$
> of median ${\approx}600$---two orders of magnitude above the framework's own
> $\lambda=1$, i.e. a monitor in which no-slip outweighs the entire PDE residual. The
> far-field ring contributes under $1\%$ of $\overline{r_{bc}^2}(u^\star)$, so the
> inequality is genuinely about the near-wall band as stated. Leg~(B) carries the
> argument and does not depend on this ordering."

**(d) §"Relation to prior work", contribution (a) — restate the contribution.** Replace
the "quantified operator-specific floor" clause with

> "\textbf{(a)} the demonstration that this floor is a property of \emph{operator
> provenance} rather than of the mesh---it does not decay under refinement but grows,
> because the individually grid-converged terms of the reference solution fail to
> cancel under a different discrete operator, so a surrogate cannot be certified by a
> residual monitor that is not the one its training data solves;"

## 9. Rebuttal line

> **Reviewer:** *your residual floor is an artifact of a $128^2$ grid that cannot
> resolve the boundary layer.*
> **We did:** re-rasterised the same AirfRANS point clouds from the raw clouds at
> $128^2$–$512^2$ (one triangulation per case, `h` the only variable), gated on a
> manufactured solution (order $2.02\pm0.05$) and on reproducing the published
> per-case numbers ($3\times10^{-8}$).
> **Evidence:** the floor **rises** — $p = -0.64\pm0.29$, **0/16 cases decay**, 15/16
> rise, $\times2.6$ from $128^2$ to $512^2$. The pipeline null stays $280\times$
> smaller; a $C^1$ re-rasterisation raises the floor rather than lowering it;
> repairing the omitted closure term moves it by ${<}0.1\%$. The convective and
> pressure terms are individually grid-converged — the floor is their non-cancelling
> remainder, and refinement makes the mismatch worse.

## 10. Experiment 2 — seeding `tab:iters`

### 10.1 The part that cannot be fixed, stated first

`tab:iters` was produced by `scripts/run_sensitivity.py` on
**`checkpoints/certificates_deq.pt`** — a dropout-FNO backbone with a DEQ corrector,
`mse_u ≈ 3.92`, `n_eval = 80`. There is **exactly one such file** in the repository
and **no seeded siblings** (`find checkpoints -name '*.pt'`; the only other single
checkpoint is `audit_pilot/seed0_base.pt`, a different arm). **The n = 1 dependency
for that configuration cannot be removed without retraining that arm.** Nothing below
is a substitute for it, and the paper must not present it as one.

### 10.2 What seeded evidence does exist

| evidence | seeds | what it shows |
|---|---:|---|
| `results/control/w1_capture.json` | **5** | WITH-residual corrector beats the NULL (residual-zeroed) corrector on **0/5** seeds — the residual is not load-bearing *as a corrector input* |
| `results/control/bc_inclusive_sweep.json` | 1 | the dissociation survives adding the no-slip term (same n=1 checkpoint) |
| `scripts/iters_sweep_seeded.py` (this work) | **5** | the iteration sweep re-run on the 5-seed Transolver+DEQ arm |

Five reload-complete Transolver checkpoints *do* exist
(`checkpoints/v2_transolver/seed{0..4}.pt` + `seed{k}_corr_with.pt`, the headline v2
system, `mse_u ≈ 0.13`). Re-running the sweep there asks a **different, stronger**
question than the original could: is the dissociation a property of one checkpoint,
or of the method?

**Method deviations from `tab:iters`, stated so a diff of the two tables is not
mysterious.** `tab:iters` used `n_eval = 80` drawn through a *shuffled* index
(`idx[:N_EVAL]` in `run_sensitivity.py`); this run uses the **first 24 cases in cache
order** (CPU/GPU budget: ~18 min per seed × 5). Both the count and the selection
differ. The swept knob is identical — `DEQCorrector.max_iter`, with the control that
the engine-level `max_iters` is **inert** on the DEQ branch re-verified and logged
(`0.229216` at both 1 and 15).

### 10.3 Pre-declared reading rule (written before the 5-seed aggregate landed)

* **Sign count is the primary statistic.** Dissociation (residual higher at the last
  cap than at `k=0`, *and* best `mse_u` below `k=0`) on **≥ 4/5** seeds ⇒ backbone-
  agnostic and seed-robust.
* **Magnitude is reported against seed noise.** If the `k=0 → k=15` residual rise
  exceeds ~2× the across-seed std, it is reported as a measured rise; if not, the
  honest verdict is *"direction consistent across N/5 seeds, magnitude within seed
  noise"* — a robustness check, not "the residual rises".
* **The word "monotonically" is at risk and will not be smuggled through.** Seed 0
  already dips at `k=1` (0.2073 → **0.2024** → 0.2100 → 0.2116 → 0.2120). The script
  logs `residual_monotone_nondecreasing` per seed, and the report leads with it rather
  than with `dissociates`.

### 10.4 Result (seed 0 complete; full aggregate in `results/sensitivity/iters_seeded.json`)

Seed 0, 24 cases: residual `0.2073 → 0.2024 → 0.2100 → 0.2116 → 0.2120`,
`mse_u` `0.1547 → 0.1435 → 0.1424 → 0.1427 → 0.1429`.

The dissociation reproduces **in direction** — the error minimum is at `k=3` while the
residual keeps climbing past it — but the magnitude is nothing like the FNO arm's.
Residual rises **+2.3%** here against **+450%** in `tab:iters` (0.113 → 0.620), and
`mse_u` is flat after `k=1` where `tab:iters` shows real over-correction damage
(2.287 → 2.575).

**The coherent reading, and the one the paper should adopt:** dissociation magnitude
scales with how far the corrector moves the field. A well-fit backbone (`mse_u` 0.13)
barely moves, so its residual barely climbs. **Therefore: keep `tab:iters` with its
honest n = 1 caption and add the seeded sweep as a robustness table** — do not swap it
in as the headline. Swapping invites the question "why is your seeded version 100×
weaker?" with no answer prepared; presenting it as a robustness check answers it in
advance.

**Still to verify before this section is final** (both flagged rather than assumed):
`deq_iters_mean` per cap — on the FNO arm the DEQ never converged early and hit the
cap at every `k`, so if on the Transolver arm it reaches its fixed point at `k ≈ 2`
then the `k = 5, 10, 15` rows are near-duplicates and the flat tail means "converged",
not "residual plateaus"; and a **paired** per-case sign count (residual higher at the
error-minimising `k` than at `k = 0` in X/24 cases), without which a +2.3% mean over
24 unpaired cases is not credibly distinguishable from zero.

## 11. Status of the six reports

| Report | State |
|---|---|
| `novelty_hunt.md` | landed |
| `r2_holes.md` | landed |
| `domain_expert.md` | landed |
| `floor_resolution_study.md` | **this file** |
| `theorem_audit.md` | not run — superseded in part by §1's MMS and unit-test gates |
| `venue_plan.md` | not run — still open, and needed before submission |
