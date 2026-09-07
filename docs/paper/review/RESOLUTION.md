# Resolution ledger — every finding, and what was done about it

Date: 2026-09-07. Companion to `FINDINGS.md`, which states the problems; this
states the dispositions. Branch `paper1/reframe-after-jcp`.

Four dispositions are used and they mean different things. **Fixed** — the paper
now says something different and correct. **Measured** — an experiment was run
and the finding's question has an answer. **Conceded** — the finding is right,
the claim is withdrawn or narrowed, and the paper says so in its own voice.
**Refuted** — the finding is wrong on the evidence, and the evidence is given.

---

## The six numbered findings

| # | Finding | Disposition | Where |
|---|---|---|---|
| F1 | Nothing in the repository minimises `J = ½‖R_h‖²`, yet the abstract claims the residual fails as an objective | **Measured** | `sec:descent` |
| F2 | Theorem leg (iii) is an upper bound where a lower bound is needed, and numerically vacuous | **Fixed** | theorem (iii), (iv) |
| F3 | The omitted closure term explains the floor (`2S·∇ν_t ≈ 0.13`, ~90%) | **Refuted** | `floor_resolution_study.md` §11 |
| F4 | The regime boundary as drawn is falsified by Lei et al., uncited | **Fixed** | `sec:regime` |
| F5 | The floor observation is already in print (Zhang et al.), uncited | **Fixed** | `sec:regime` |
| F6 | A physics-free baseline may beat the physics residual | **Conceded** | `sec:selective` |

**F1.** The corrector whose iteration cap `tab:iters` sweeps is trained by
supervised regression and never sees the residual as an objective, so the sweep
could not support the claim. We now descend `J` directly, from the exact ground
truth with no network in the loop: the residual falls 84% in 24/24 cases while
field error goes from zero to a median of 0.91. The claim is stated in the form
that holds universally — the residual-selected iterate is never the error-optimal
one, 24/24 on both arms — rather than the form that does not: descent from a
perturbed start *helps* in 18/24 cases, and the paper says so.

**F2.** Leg (iii) now states the lower bound
`‖e_∞‖ ≥ ‖P_range(L) r*‖ / σ_max`, which is the direction the thesis needs, with
the old upper bound kept as a remark and its numerical vacuity admitted in the
text. Leg (iv)'s cone constant is corrected to the *projected* floor. Both
corrections are attributed to earlier drafts of this work in the prose rather
than silently applied.

**F3 is the one finding that did not survive contact with the data**, and it
mattered because `FINDINGS.md` nominated it as the theory to rebuild the paper
on. Adding the omitted term back moves the floor by −0.01% to +0.18% across five
rungs; the term is 3.5–4.8% of the floor, not 90%; and at `Re ≈ 2×10⁶` the entire
viscous term is 1–4% of convection, so nothing viscous can carry a floor of that
size. What survives is the *theorem* — an inconsistent operator has a nonzero
continuum limit, so refinement alone cannot remove the floor — not the
*attribution*. The paper keeps the first and states the second as measured: the
floor's size is set by the mismatch with the body-fitted discrete balance the
reference solved.

---

## F7's substantive holes, individually

| Hole | Disposition | Where |
|---|---|---|
| ρ is monotone in how bad the model is | **Conceded** | MeshGraphNet disclosure, `sec:v2` |
| `mgn_density_control.json` verdict `DENSITY-DRIVEN` never disclosed | **Conceded** | `sec:v2` |
| ρ_D = 0.84 is the integrator's ceiling, not a model result | already stated; **extended** | `sec:v2` |
| ρ_L = 0.998 is easy twice over (AoA is an input; Kutta–Joukowski from the outer ring) | **Conceded** | `sec:v2` |
| Viscous drag is *unavailable*, not merely inaccurate | **Conceded** | `sec:v2` |
| λ-sweep circularity: the wall ring zeroes the boosted no-slip cells | **Measured, partly refuted** | `sec:method` |
| Two Laplacian stencils in one pipeline | **Conceded** | `sec:method` |
| Internal contradiction: prediction below the floor, yet ρ = 0.61 there | **Fixed** | theorem, new paragraph |
| `norm_truth_continuity` ≈ half the floor may be rasterisation error | **Refuted** | `sec:selective` (C2) |

Two of these need their numbers stated, because the audit's version was not
quite right.

*The wall-ring objection.* The audit called the no-slip boost "dead code",
because `diagnose()` zeroes `bc` on exactly the ring where `bc_violation()`
weights it most. Measured over 8 cases, the zeroing removes a mean **46%** (range
35–51%) of the squared no-slip penalty — substantial, and worth disclosing, but
not all of it. The paper now states that any claim resting on the magnitude of
`r_bc`, leg (A) included, is a claim about the surviving 54%.

*The rasterisation confound.* The continuity residual contains no `ν_t`, no
closure and no boundary term, so roughly half the floor could have been pure
label-rasterisation error, which would make the detector a geometry-difficulty
proxy. Conditioning on each case's own floor leaves a partial rank correlation of
**0.561** against a raw **0.610** — 92% survives, and the drop's CI includes zero.
Independently, cubic interpolation of the same source cloud *raises* the floor by
6–12%, so linear interpolation's kinks are not what sets it. The confound is
refuted twice.

---

## The decisive controls (C1–C6)

| # | Question | Outcome |
|---|---|---|
| C1 | Physics vs physics-free trust score | Cannot be separated on field error (Δ +0.023, CI [−0.035, +0.083]); the fused number is not the residual's |
| C2 | Is the detector a difficulty proxy? | **Refuted** — partial ρ 0.561 vs raw 0.610 |
| C3 | Does the gate's accuracy come from the residual test? | **Conceded** — an ungated half step beats it, 95.8% vs 89.3% |
| C4 | Does the signal degrade on the engineering quantity? | It **improves** — AUROC 0.952 on \|ΔC_D\|, and here physics wins outright |
| C5 | MeshGraphNet density non-disclosure | Disclosure added; withdrawal not warranted on 4 cases of MSE |
| C6 | Omitted channels in the iteration sweep | Recoverable; non-monotonicity confirmed and now stated |

C3 is the most expensive concession in the paper and it is made in the paper's
own voice: the gate's *accuracy* is damping, and what it uniquely buys is the
**certificate** — an ungated step raises the monitored residual on 1.0% of
deployed cases (8.8% on the ensemble path) and the gate makes that impossible by
construction.

---

## Claims withdrawn or narrowed, collected

1. **"AUROC ≈ 0.9" attributed to the residual** — it is the fused score's number.
2. **The gate improves accuracy** — an ungated fixed half step does better.
3. **"For every λ ≥ 0"** in leg (A) — restated at the deployed resolution; the
   margin collapses monotonically and has crossed by 512², with λ* ≈ 600.
4. **Leg (iii) as the quantified floor** — it was an upper bound, and vacuous.
5. **`tab:iters` as the evidence for residual-as-objective failure** — wrong
   knob, non-monotone across the sweep, n = 1, two channels. Demoted to
   illustration with its three limits stated before its numbers.
6. **The three-backbone claim** — the third backbone is evaluated off its
   training density; the matched-density range is ρ = 0.40–0.61, not 0.40–0.85.
7. **A general "descending the residual makes the field worse"** — false; it
   helps in 18/24 cases. Replaced by the iterate-selection claim.

---

## Structural changes

- **Title** drops the coined system name.
- **Abstract and introduction** open on the measurement, not the method.
- **Contributions** reordered: floor → ladder → descent → operator boundary →
  what survives of the detector → what survives of the gate → package.
- **`tab:positioning`** rebuilt from a seven-column feature checklist into a
  two-column boundary (solver-consistent operator vs surrogate-side proxy), with
  the prior residual-correction successes placed on the side where they belong.
- **New sections**: `sec:floor_ladder`, `sec:descent`, `sec:regime`,
  and a `sec:selective` anchor for the physics-vs-physics-free comparison.
- **New citations**: Lei et al., Zhang et al., McGreivy & Hakim, Stetter,
  Brandt & Livne, Morin et al., Bochev & Gunzburger.

---

## Still open

- ~~**Venue formatting.**~~ **Closed 2026-09-07.** The guide and open-access pages
  were read directly in the browser (the 403 was an artifact of automated
  fetching), and `venue_plan.md` §12 records every value as verified. The
  headline results: **there is no manuscript length limit**, which was the
  single biggest open risk at ~14,700 words; the subscription route carries
  **no publication fee**, verbatim; review is **single anonymized**, so nothing
  needs anonymising. One requirement was missing entirely and is now met —
  **Highlights**, which the journal requires at submission (3–5 bullets, ≤85
  characters each): `docs/paper/highlights.txt`. `scripts/check_submission.py`
  verifies the abstract, keyword and highlight limits mechanically.
- **Two author decisions**, detailed in `venue_plan.md` §8: whether to disclose
  the two prior desk rejections in the cover letter (recommendation: no), and how
  to pre-empt the scope's request for comparison against traditional numerical
  methods.
- **The convection/pressure mechanism** is stated as the block the floor lives in
  (airtight, by elimination) and *not* as the sharpening narrative, which would
  need a per-cell correlation that was not run.
- **arXiv v4** after the reframe, not before.
