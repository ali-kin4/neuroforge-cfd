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

**The rise is not rasterisation error.** This is the important one, and it comes
from the MMS-raster control: push a *manufactured analytic* solution through the
identical rasteriser and residual, and its residual **falls** with refinement
(fine/coarse 0.41–0.74 across three cases) while the real AirfRANS truth's
**rises** (0.86–3.23). Same pipeline, same cases, opposite direction. Whatever
makes the real floor grow is in the data, not in the interpolation.

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

An independent pass (3 cases, **5 rungs**, three exclusion bands) reaches the
same verdict on every band of every case: **PLATEAU**, order `p` from −0.91 to
+0.31. Bands of 0.05, 0.1 and 0.25 chord all agree, as do the repaired-operator
variants. Its per-case ladders are in
`results/certificates/floor_resolution_decomposition.json`.

The two studies trade off exactly the right way: 24 cases × 3 levels here,
3 cases × 5 rungs there. Neither is a fluke of its own design.

---

## 3. What this does to the paper

**H-B held.** The omitted `∇ν_t·∇u` term was predicted, before the run, to
*grow* with refinement — at 128² the sub-cell boundary layer smears `∇ν_t`, so
the term is under-estimated there. It does: 0.0018 → 0.0039 → 0.0054, `p =
−0.808`. This is a **model** omission behaving like one. It is the cleanest
component of the floor to point at, because no amount of refinement removes it.

**H-C held — leg (A) needs no weakening.** The λ-sweep's closed form requires
`r_bc²(u*) > r_bc²(u_∞)`, and the theorem file attributes that inequality to the
128² boundary layer being sub-cell. It survives at every level in both bands:

| level | framework band (truth vs uniform) | fixed band |
|---|---|---|
| 128² | 0.00594 > 0.00543 | 0.00585 > 0.00537 |
| 256² | 0.00279 > 0.00256 | 0.01383 > 0.01246 |
| 512² | 0.00128 > 0.00126 | 0.04489 > 0.04044 |

So "for every λ ≥ 0" stands. **But** under the framework's own no-slip band —
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
| perturbed, descend error | 0.503 → 0.312 | 2.56 → **0.000**, 24/24 | 0.503 → 0.311 |

Started at the **exact ground truth**, with no network anywhere in the loop, 300
steps of descent on the monitored residual cut it by 84% in 24/24 cases and take
the field error from zero to a median of 0.91 — the range a trained backbone
*starts* in. The compact-stencil monitor the paper reports falls in 24/24 too,
so this is not the differentiable twin's wider Laplacian being gamed. The
`perturbed_error` arm recovers the truth exactly, so the optimiser works.

**One correction to the paper's framing, from the perturbed arm.** Residual
descent is *unreliable*, not uniformly harmful: from a perturbed field it cuts
the error in 18/24 cases, typically by 60%, and multiplies it by up to 7.5× in
the other 6. The defensible sentence is not "descending the residual makes
things worse" but "descending the residual cannot reach the truth and fails
catastrophically on a minority of cases" — it terminates at residual 0.028 while
the exact truth sits at 0.110, so the objective ranks a field with error 1.31 as
four times better than the field with error 0.

---

## 6. Status of the six reports

| Report | State |
|---|---|
| `novelty_hunt.md` | landed |
| `r2_holes.md` | landed |
| `domain_expert.md` | landed |
| `floor_resolution_study.md` | **this file** |
| `theorem_audit.md` | not run — superseded in part by §1's MMS and unit-test gates |
| `venue_plan.md` | not run — still open, and needed before submission |
