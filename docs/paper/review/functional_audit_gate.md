# The functional audit gate — `strongest_design.md` §4

Decides between **Design A** (the audit-operator ladder) and **Design D** (A + a
goal-oriented certificate). Three days of gate against a month of committed work.

The paper's established thesis is **rank / certify / descend**: the monitored
steady-RANS residual *ranks* predictions (AUROC 0.952 on worst-decile `|ΔC_d|`,
control 4) but cannot *certify* them (a floor at the reference solution that
refinement makes worse) and cannot be *descended* (24/24 divergence from the exact
truth). The open question is whether a **functional** (goal-oriented) reading of
the *same* residual recovers the certificate the norm-level reading cannot give.

Producer: `scripts/functional_audit_gate.py`.
Artifacts: `results/review/functional_audit_gate_validation.json`,
`results/review/functional_audit_gate.json`,
`results/review/functional_audit_gate_analysis.json`.

---

## 1. PRE-REGISTRATION — committed before the run

Everything in this section, **including the predictions in §1.6 and the decision
rule in §1.5**, was committed before the 200-case numbers existed. This mirrors
how the resolution ladder's decision rule was committed in `bf5106c`. The project
has already withdrawn five claims to post-hoc reading; this is exactly that shape
of risk.

Only §2 (plumbing validation) had been run at commit time, and it is code
correctness, not a hypothesis test. What §2 *did* incidentally reveal about
magnitudes is disclosed in §1.6 and is the basis of the stated prediction —
rather than being quietly used later as if it had been predicted.

### 1.1 The object

For a control volume `V` and the drag direction `e_D = (cos α, sin α)`,

```
I(w; V) = e_D · ∫_V R(w) dV
```

with `R` the **momentum** residual of the deployed monitor. Rendered in drag-
coefficient units, so that it is directly comparable with `|ΔC_d|`:

```
I = SIGN · (2/c²) · Σ_cells (e_D · r_scaled) dx dy
```

using `∫ r_phys dA / (½ u∞² c) = 2 ∫ r_scaled dA / c²` for `L = c`, where
`r_scaled` is the residual non-dimensionalised exactly as
`PhysicsChecker.diagnose` non-dimensionalises it (`residuals.py:355-359`).
`SIGN = −1`, so that a positive far-field flux is a positive drag; verified
numerically in §2, not assumed.

### 1.2 Residual form — the primary arm is the DEPLOYED monitor

| `form` | definition | role |
|---|---|---|
| `adv` | `u·∇u + ∇p − ν_eff ∇²u` — byte-for-byte `residuals.momentum_residual` | **PRIMARY**. This is the audited object. |
| `cons` | `∇·(uu + pI − ν_eff ∇u)` | Secondary. The only form for which the divergence-theorem reading exists. |

Leading with `cons` would change the operator under test, and the finding would
no longer be about the monitor the paper ships. It is reported as a declared
variant.

### 1.3 The immersed body's surface term — three treatments, not one

The design doc flags this as the main plumbing risk, and a **signed** integral is
not indifferent to it. The deployed monitor zeroes residuals inside the solid
*and* on the solid-adjacent fluid wall ring (`residuals.py:338-348`) — 0.51% of
fluid cells, sitting exactly where the surface term lives, carrying residuals
10–20× the bulk.

| | treatment |
|---|---|
| **T0** | as deployed: solid **and** wall ring zeroed. The residual fields *as they exist*. **PRIMARY.** |
| **T1** | wall ring **restored**; only the solid is zeroed. |
| **T2** | T0 **plus** the explicit body term `i_solid = e_D·Σ_{V∩solid} R dA` — what the *same* discrete operator attributes to the body. |

`i_solid` is not an approximation bolted on: for the conservative form the box
sum telescopes exactly, so
`φ_outer = I_T0 + i_solid + i_ring` holds to machine precision. The surface term
is therefore *accounted*, not dropped, and §2 checks that identity.

### 1.4 Control volumes — committed, with a multiplicity rule

Five nested boxes centred on the airfoil (chord runs `x = 0..1`), on the AirfRANS
crop `(−1, 2) × (−1.5, 1.5)` at 128², `h = 0.02362 c`:

| box | `x` | `y` |
|---|---|---|
| V1 | [−0.25, 1.25] | [−0.40, 0.40] |
| V2 | [−0.50, 1.50] | [−0.65, 0.65] |
| V3 | [−0.75, 1.75] | [−0.90, 0.90] |
| V4 | [−0.95, 1.95] | [−1.15, 1.15] |
| V5 | [−0.95, 1.95] | [−1.40, 1.40] |

Every box keeps ≥2 cells of margin from the array edge, so every cell uses the
central stencil and every stencil neighbour is in-array.

**Multiplicity rule, committed:** "the inversion rate falls below ~10% at *some*
`V`" is an open door — five boxes and one will oblige. A pass at a single box
with **no monotone trend in box size** is reported as **MARGINAL, not a pass**.

### 1.5 The tests and the decision rule

**(a) Does the floor cancel?** The norm-level pathology is an *inversion*: the
prediction's residual sits below the truth's, so no threshold on `‖R‖` implies an
error bound. Measure the inversion rate under `|I(·;V)|`.
- *Pass:* below ~10% at some `V`, with a monotone trend in box size.
- *Fail:* it stays high. Then no certificate exists.

**The bar is the MATCHED baseline, not 160/200.** The design doc's 160/200 comes
from `residual_floor_realdata.json`, whose `pred` arm is a *different model*
(`checkpoints/certificates_deq.pt`, dropout-FNO). We therefore compute the
norm-level inversion rate **on our own arms** and report the two side by side. A
functional inversion rate of 25% is not a triumph if the matched norm-level
baseline was already 30%.

**(b) Does it rank?** Spearman / AUROC / oracle recovery of `|I(w;V)|` against
per-case `|ΔC_d|` (target `vs_gt_nf`), against the residual-norm baseline.
Paired case-level bootstrap, 10⁴ draws, seed 0, with the top-decile threshold
held **fixed** at its full-sample value so "worst-decile case" stays a property
of the case — matching `decisive_controls.py` control 1. Statistics are lifted
verbatim from that file.

**(b′) Cancellation diagnostic — run ALONGSIDE (b), not after.** `I` is signed,
so a field with large but mutually cancelling errors scores near zero: the mirror
image of the floor problem, and a false-confidence mode in anything called a
certificate. Two probes: (i) do the low-`|I|` decile cases contain top-decile
`|ΔC_d|` cases; (ii) score the **unsigned** integrals `∫_V |e_D·R| dA` and
`∫_V |R| dA` in the same (b) framework. If the unsigned integral ranks materially
better than the signed one, cancellation is the *measured* mechanism rather than
a hand-wave.

**(c) Does it calibrate?** Split conformal `c` for `|ΔC_d| ≤ c·|I(w;V)|` at 90%
target coverage, 200 random half-splits. **Built only if (a) and (b) pass.**

**DECISION RULE.**

| outcome | branch |
|---|---|
| (a) **and** (b) pass | **Design D.** Build (c) and report it. |
| (b) passes, (a) fails | **Design A**, with the functional reported as a BETTER RANKER but still not a certificate. Honest, and it explains control 4. |
| both fail | **Design A only.** Control 4's AUROC 0.952 stays as a measured positive. |

(b) counts as passing only if the functional's paired ΔAUROC against the
residual-norm baseline is positive **with a CI excluding zero**, on every seed of
the primary arm.

### 1.6 Predictions, stated before the run

Stating these makes the run falsifiable rather than exploratory.

1. **(a) will FAIL.** Reasoning, and it is not a hunch — the §2 validation ran on
   6 truth cases for plumbing reasons and incidentally exposed the scale:
   `|I(u*; V5)|` came out at **0.003–0.052 in C_d units**, while `|ΔC_d|` is
   O(4×10⁻³) (case 0: gt 0.02798 vs seed-0 0.03220). The truth's own functional
   residual is the same size as, or larger than, the drag errors it would have to
   certify. That is the floor surviving integration, and it is what the floor
   study predicts: the residual **is** the non-cancelling remainder of
   individually grid-converged terms, so "it cancels under integration" was never
   plausible by default.
2. **(c) will therefore be dead on arrival** even if it is built: a conformal `c`
   would have to be small, but `|I|` already exceeds `|ΔC_d|`, so the bound is
   vacuous — it would be satisfied trivially and certify nothing useful.
3. **(b) is genuinely open.** Ranking is scale-invariant, and control 4 already
   shows the norm-level residual ranks drag at 0.952. Whether a signed functional
   beats that is not predictable from the magnitude argument, and (b′) is the
   reason to doubt it: cancellation destroys rank information that a norm keeps.
4. **The unsigned contrast will outrank the signed functional.** If true, that is
   the mechanism behind any (b) failure, measured rather than asserted.

If (a) passes, prediction 1 is wrong and the paper gains a certificate. Either
outcome is publishable; that is why the gate is worth three days.

### 1.7 Arms

| arm | score field | target | why |
|---|---|---|---|
| **S** (PRIMARY) | deployed v2_transolver seed-*k* prediction, `data/cache/acceptance_gate/seed{k}/*.npz` | `\|C_d(raw_k) − C_d(gt)\|` | Field, residual, functional and C_d all from the **same object**. No arm mismatch. |
| **E** | ensemble-mean field, `data/cache/w2/ensemble/*.npz` | same | Carries control 4's arm mismatch (ensemble-mean score, per-seed target); exists only so the functional is comparable with the committed AUROC 0.952. |

**Provenance verified, not assumed** (this is what makes arm S legitimate):
`force_coefficients(gt) == gt_nf_full_test_r128_n200.json` and
`force_coefficients(raw_seed0) == seed0_prednf.json` reproduce to full double
precision on the cases checked, so arm S's target **is** control 4's `vs_gt_nf`
target. The ensemble-mean field's residual norm reproduces
`selective_percase.json` exactly (0.13643930852413177). Masks from `encode_case`
are identical to the cached truth masks.

CPU-only, zero forward passes, committed caches only.

---

## 2. Plumbing validation — run before the pre-registration was committed

`results/review/functional_audit_gate_validation.json`, 6 cases. Split into
independently falsifiable checks, so a failure localises rather than leaving
"sign convention or masking?" ambiguous.

| check | what it tests | result |
|---|---|---|
| **V-A** | box entirely in fluid, no body: `Σ_V ∇·F dA` == telescoped edge flux | max rel. err **5.2×10⁻¹³** |
| **V-B** | box containing the body: `φ_outer == I_T0 + i_solid + i_ring` | max rel. err **5.9×10⁻¹⁵** |
| **V-C** | sign of `e_D·φ_outer` vs `cv_coefficients` (`design_force_integrator.py`) on the same field | agrees **6/6**; `SIGN = −1` |
| **V-D** | what the surface integrator reports on the same truth field | reported, see below |

V-A and V-B are exact identities of the discrete operator, satisfied to machine
precision, which is what licenses calling `i_solid` *the* surface term rather
than an estimate of it: for the second-order central stencil,
`Σ_{i=i0}^{i1} ∂_x F dx = (F_{i1} + F_{i1+1} − F_{i0−1} − F_{i0})/2`, so the box
sum reduces exactly to a two-point-averaged contour integral. Constant fields
telescope to zero, so the pressure gauge cancels and no `p_∞` subtraction is
needed — both contours close.

V-C sign and scale on the six validation cases (C_d units):

| case | `φ_outer` (V5) | `cv_coefficients` | surface integrator |
|---|---:|---:|---:|
| `..._31.812_1.334...` | +0.0128 | +0.0160 | +0.0280 |
| `..._39.741_14.642...` | +0.0262 | +0.0249 | +0.1943 |
| `..._85.752_6.341...` | +0.0093 | +0.0063 | +0.0529 |
| `..._80.247_1.078...` | +0.0077 | +0.0081 | +0.0033 |
| `..._61.728_9.994...` | +0.0153 | +0.0181 | +1.3003 |
| `..._65.202_-3.411...` | +0.0092 | +0.0139 | +0.0217 |

The far-field flux agrees with the existing control-volume integrator in sign on
6/6 and in magnitude to tens of percent — the residual disagreement is the
different quadrature, the box being 2 cells inside the domain edge, and the
viscous term which `cv_coefficients` neglects. The **surface** integrator is a
different story and it is the repo's known problem (B-ATTACK-3: 332% median
magnitude error on perfect fields); the 1.3003 row is that failure, on the ground
truth, with no model involved.

---
