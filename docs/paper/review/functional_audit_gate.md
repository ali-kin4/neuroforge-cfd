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

## 3. Results — 200 AirfRANS `full` test cases, 3 seeds, CPU-only

`results/review/functional_audit_gate.json` (311 s) and
`results/review/functional_audit_gate_analysis.json`.

Arm S's residual-norm baseline reproduces control 4: AUROC **0.9417 / 0.9567 /
0.9606** on the three seeds (mean **0.953** against control 4's 0.952), Spearman
+0.616 / +0.592 / +0.626, oracle recovery 0.821 / 0.806 / 0.788. Arm S removes
control 4's arm mismatch and the number does not move, which is itself worth a
sentence in the paper.

### 3.0 Scale check — can a certificate exist at all?

In C_d units, seed 0 (the other seeds agree to within 10%):

| quantity | value |
|---|---:|
| median &#124;ΔC_d&#124; (`vs_gt_nf`) | **0.00142** |
| mean &#124;ΔC_d&#124; | 0.01375 |
| top-decile &#124;ΔC_d&#124; threshold | 0.02133 |
| median &#124;I(u\*; V1)&#124; | **0.01356** |
| median &#124;I(u\*; V5)&#124; | **0.01453** |
| **ratio, truth's functional residual : median drag error** | **9.6× (V1) → 10.3× (V5)** |

**The floor does not cancel under integration.** The ground truth's own
drag-direction functional residual is an order of magnitude larger than the drag
errors it would have to certify, at every control volume. This is prediction 1 of
§1.6, and it is what the floor study predicts rather than a surprise: the residual
*is* the non-cancelling remainder of individually grid-converged terms, so there
was never a reason to expect it to cancel under a further integration.

### 3.1 Test (a) — does the floor cancel? **FAIL**

Inversion rate = fraction of cases where the prediction scores *below* the truth.
Low is good: it is the property a threshold needs before acceptance can imply
accuracy.

**First, the matched baseline, because the brief's bar was wrong.** The design
doc's 160/200 (80%) comes from `residual_floor_realdata.json`, whose prediction
arm is `checkpoints/certificates_deq.pt` — a **dropout-FNO** with `mse_u ≈ 3.92`;
verified: `norm_pred_median` 0.1038 against `norm_truth_median` 0.1325. On the
**deployed** v2_transolver (`mse_u ≈ 0.13`) the ordering reverses:

| arm | median ‖R‖ | norm-level inversion rate |
|---|---:|---:|
| ground truth | 0.1317 | — |
| `raw_seed0` | 0.1525 | **4.0%** (8/200) |
| `raw_seed1` | — | **2.5%** (5/200) |
| `raw_seed2` | — | **2.5%** (5/200) |
| `corrected_seed0` | 0.1648 | 6.0% (12/200) |
| `ensemble_mean` | 0.1394 | 6.5% (13/200) |

**The 80% inversion is a property of a weak, over-smoothing backbone, not of the
monitor.** A smooth wrong field has small derivatives and therefore a small
residual; an accurate field carries the truth's floor plus its own error. This is
a correction the paper must make (see §5.2).

Now the functional, primary variant `adv_I_T0`:

| arm | V1 | V2 | V3 | V4 | V5 | min |
|---|---:|---:|---:|---:|---:|---:|
| `raw_seed0` | 0.430 | 0.270 | 0.325 | 0.385 | 0.375 | **0.270** |
| `raw_seed1` | 0.505 | 0.435 | 0.365 | 0.310 | 0.310 | **0.310** |
| `raw_seed2` | 0.485 | 0.395 | 0.400 | 0.370 | 0.360 | **0.360** |
| `corrected_seed0` | 0.395 | 0.220 | 0.275 | 0.380 | 0.385 | 0.220 |
| `ensemble_mean` | 0.525 | 0.390 | 0.315 | 0.320 | 0.325 | 0.315 |

**Every cell is above the 10% pass bar, on every box, on every arm**, and there
is no monotone trend in box size, so the pre-registered MARGINAL branch does not
apply either. The best any *signed* variant reaches is `cons_phi_outer` at
0.170–0.235 on the larger boxes — still 3–9× the bar.

**The functional does not merely fail to fix the inversion; it manufactures one.**
Against a matched norm-level baseline of 2.5–6.5%, the drag-projected signed
integral inverts on 27–36% of cases — **7 to 15× worse**. Projecting onto `e_D`
and summing with sign discards the information that kept the truth at the bottom
of the ranking.

For contrast, the **unsigned** integrals `∫|e_D·R| dA` and `∫|R| dA` invert on
**0.000** of cases, at every box, on every arm.

### 3.2 Test (b) — does it rank? **FAIL**

AUROC on top-decile |ΔC_d| (`vs_gt_nf`), threshold fixed at its full-sample
value. Arm S (self-consistent: score, field and C_d from the same object):

| seed | ‖R‖ baseline | best signed I (box) | best unsigned J (box) |
|---|---:|---:|---:|
| 0 | **0.9417** | 0.6114 (V1) | 0.9353 (V1) |
| 1 | **0.9567** | 0.7025 (V1) | 0.9483 (V1) |
| 2 | **0.9606** | 0.6364 (V1) | 0.9581 (V1) |

Spearman tells the same story: the baseline is +0.59 to +0.63; `adv_I_T0` reaches
+0.28 to +0.33 at its best box and dips to +0.08. The conservative variants and
the ring-restored variants do not rescue it — `cons_I_T2` is *anti*-correlated on
several arms (AUROC 0.29–0.49). The best signed number anywhere in the sweep is
`cons_phi_outer` on arm E at 0.814–0.825, still 0.13 below the baseline, and it
is the far-field momentum-theorem drag rather than a residual functional.

**The functional is not a better ranker. It is a much worse one**, by roughly
0.25–0.35 AUROC, on every seed and every control volume.

### 3.3 Test (b′) — cancellation, and it is the measured mechanism

This is the part that explains 3.1 and 3.2 rather than merely reporting them.

**Probe (i) — the unsigned contrast.** Identical cells, identical operator,
identical boxes; the *only* difference is taking the absolute value before
summing:

| seed | signed &#124;I&#124; | unsigned ∫&#124;R&#124; | ‖R‖ baseline |
|---|---:|---:|---:|
| 0 | 0.611 | **0.935** | 0.942 |
| 1 | 0.703 | **0.948** | 0.957 |
| 2 | 0.636 | **0.958** | 0.961 |

Spearman, unsigned: **+0.693 / +0.724 / +0.685** — *higher* than the norm
baseline's +0.616 / +0.592 / +0.626. Removing the sign recovers the whole signal
and a little more. **Cancellation is therefore the mechanism, measured, not
inferred**: roughly 0.3 AUROC of ranking power is destroyed by the signed
projection and restored in full by the absolute value.

**Probe (ii) — the false-confidence mode, quantified.** Among the cases in the
lowest decile of |I| — the cases a certificate would be most confident about:

| seed | box | top-decile &#124;ΔC_d&#124; cases inside | max &#124;ΔC_d&#124; inside | top-decile threshold | ratio |
|---|---|---:|---:|---:|---:|
| 0 | V1 | 2 / 20 | 0.0848 | 0.0213 | **4.0×** |
| 0 | V3 | 2 / 20 | 0.0912 | 0.0213 | **4.3×** |
| 1 | V3 | 4 / 20 | 0.0851 | 0.0207 | **4.1×** |
| 2 | V5 | 2 / 20 | 0.0835 | 0.0236 | **3.5×** |

The median |ΔC_d| inside the low-|I| decile (0.0005–0.0025) is indistinguishable
from the overall median (0.0013–0.0015): the score carries no information there.
And the worst drag error the low-|I| decile hides is **3.5 to 4.3× the top-decile
threshold** — the functional assigns near-zero score to some of the worst drag
predictions in the dataset. That is precisely the false-confidence mode §1.5
pre-registered, and it is why a certificate built on a signed functional would be
unsafe even if the magnitudes had worked out.

### 3.4 Test (c) — not built

Per the pre-registered rule, (c) is built only if (a) and (b) pass. Both failed.
It would also have been vacuous: §3.0 shows |I(u\*;V)| is 9.6–10.3× the median
|ΔC_d|, so any conformal `c` at the truth already admits a bound an order of
magnitude wider than the quantity being bounded — the bound would hold, and
certify nothing. This is prediction 2 of §1.6.

---

## 4. VERDICT — the decision rule selects **Design A**

> **(a) FAILS** (inversion 27–36% against a 10% bar and a 2.5–6.5% matched
> baseline) and **(b) FAILS** (AUROC 0.61–0.70 against the baseline's
> 0.94–0.96). By the rule committed in `282b00e`: **both fail → Design A only.
> Control 4's AUROC 0.952 stays as a measured positive, which is a real result
> either way.**

Design B does not exist on these assets, and it does not fail for a fixable
plumbing reason — the plumbing is exact to 10⁻¹³ (§2). It fails because the
premise was wrong: **the floor does not cancel under integration, and the signed
projection that would let it cancel is the same operation that destroys the
ranking.** B-ATTACK-1's requirement that Design B's delta be "the floor/functional
interaction" is met — and the interaction is negative.

All four §1.6 predictions were borne out, including the one that mattered most
(the unsigned contrast outranking the signed functional), which is the difference
between a measured mechanism and a post-hoc story.

**Wall-clock.** 6 s validation, 311 s for the 200-case × 8-field sweep, ~20 min
for the 10⁴-draw paired bootstrap; total compute under half an hour, CPU-only,
zero forward passes, against the three days budgeted. The saving is entirely
because the prediction field caches were already committed. Analyst time, not
compute, was the cost.

---

## 5. What the paper may now claim

### 5.1 New claims this gate licenses

1. **The floor survives goal-oriented integration.** On 200 AirfRANS cases and
   five nested control volumes, the ground truth's drag-direction functional
   residual is **9.6–10.3× the median drag error** it would have to certify. A
   goal-oriented reading does not escape the operator-provenance floor.
2. **The signed projection destroys the ranking, and cancellation is why.** Same
   cells, same operator, same boxes: signed AUROC 0.61–0.70, unsigned 0.94–0.96,
   with the unsigned Spearman (+0.69) *above* the deployed norm's (+0.62). This
   is the cleanest available statement that what the monitor carries is
   **magnitude**, not direction.
3. **A signed functional certificate would be unsafe.** Its most-confident decile
   hides drag errors up to **4.3×** the worst-decile threshold.
4. **Design A's foil is now measured, not argued.** §3.5 of the design doc says
   goal-oriented estimation works for ROMs because the residual is evaluated with
   the full-order operator. This gate is the *measurement* of what happens when
   that assumption is removed: the technique does not transfer, and the number is
   0.61 against 0.94.

### 5.2 A correction the paper must make — the inversion premise is model-dependent

**Do not use the 160/200 inversion as the reason the monitor cannot certify.**
That figure is `checkpoints/certificates_deq.pt`, a dropout-FNO with
`mse_u ≈ 3.92`. On the **deployed** v2_transolver the norm-level inversion is
**2.5–6.5%**, i.e. the truth is the best-scoring field on 93.5–97.5% of cases. A
reviewer holding the repo can compute this in one command, and the paper's
"cannot certify" leg currently rests on it.

The **correct and model-independent** statement of the same leg is the floor
*magnitude*, which this gate now quantifies in engineering units:

> A threshold on the monitored residual cannot imply an error bound, because the
> monitor does not vanish at the reference solution: ‖r\*‖ = 0.192 where the error
> is zero, and in drag units the reference's own functional residual is **an order
> of magnitude larger than the drag errors to be certified**. The obstruction is
> the floor's magnitude, not the ordering — the ordering is model-dependent and
> reverses for a sufficiently over-smoothed backbone.

That is sharper, it is true on every arm, and it removes an attackable sentence.

### 5.3 Exact text for the manuscript

Add to the "certify" leg of the rank/certify/descend section:

> Nor does a goal-oriented reading escape the floor. Integrating the same
> monitored residual against the drag direction over five nested control volumes
> (`scripts/functional_audit_gate.py`; 200 cases, 3 seeds), the *reference
> solution's* functional residual is 9.6–10.3× the median |ΔC_d| it would have to
> certify, and the signed functional ranks worst-decile drag error at AUROC
> 0.61–0.70 against the residual norm's 0.94–0.96 on the same fields. The signed
> projection is the cause: the identical integral taken in absolute value scores
> 0.94–0.96 and a Spearman of +0.69 against the norm's +0.62, and the signed
> functional's most-confident decile conceals drag errors up to 4.3× the
> worst-decile threshold. What a provenance-mismatched residual carries is
> magnitude, not direction, so the classical goal-oriented machinery that makes
> dual-weighted residuals effective for projection-based reduced-order models —
> where the residual is evaluated with the full-order operator and vanishes at the
> truth by construction — does not transfer to a surrogate trained on a released
> dataset.

### 5.4 Rebuttal line

> **Reviewer:** *you show a residual norm cannot certify, but the goal-oriented
> literature would weight the residual for the functional of interest; you have
> not tried the thing that actually works.*
>
> **We did:** pre-registered the test and both branches before running
> (`282b00e`), then evaluated `I(w;V) = e_D·∫_V R dV` on 200 cases, 3 seeds, five
> nested control volumes, three treatments of the immersed-body surface term, and
> both the advective (deployed) and conservative forms — with the discrete
> divergence identity `φ_outer = I + i_solid + i_ring` verified to 6×10⁻¹⁵, so the
> surface term is accounted rather than dropped.
>
> **Evidence:** the floor does not cancel — the *reference solution's* functional
> residual is 9.6–10.3× the drag errors to be certified — and the signed
> functional ranks at AUROC 0.61–0.70 against the norm's 0.94–0.96. The unsigned
> integral of the same cells recovers 0.94–0.96, which identifies cancellation as
> the mechanism, and the signed functional's most-confident decile hides drag
> errors 4.3× the worst-decile threshold. We report this as a negative result and
> scope the paper accordingly.

---

## 6. Limitations of this gate, stated rather than buried

1. **Zeroth-order adjoint.** Constant `e_D` weighting over `V` is the zeroth-order
   drag adjoint, exact only in the continuum with exactly-satisfied conservation.
   The true adjoint costs ≈ one solve, which is the cost this gate exists to
   avoid; B-ATTACK-2 stands. A negative result at zeroth order does not prove the
   true adjoint would also fail — but §3.0's magnitude finding is
   adjoint-independent, since it is a statement about `R(u*)` itself, and §3.3's
   cancellation finding is a property of signed integration that a smoother
   weight would mitigate only in degree.
2. **AirfRANS crop.** The control volumes sit 0.4–1.4 chords from a lifting
   airfoil, where induced velocity is O(few %) — the scale being integrated
   (B-ATTACK-4). Unavoidable on the `(−1, 2, −1.5, 1.5)` crop; an owned corpus
   with a proper far field would fix it. The finding is robust across a 3.5×
   range of box size, which is the available mitigation.
3. **Viscous drag is unavailable at 128²**, not approximate: `Δ = 0.0234 c`,
   `y⁺ ≈ 10³`, wall ring zeroed. The functional is a pressure/momentum-flux
   object and is scoped as one.
4. **The surface integrator's magnitude bias** (332% median on perfect fields) is
   not in play for the target — both sides of `vs_gt_nf` go through the same
   integrator, so the target is model error — but it is why §2's V-D column shows
   a 1.30 C_d on a *ground-truth* field, and it bounds what any of this could
   certify against a real label.
5. **One dataset, one closure, 2-D, 128².** Conceded without argument.

---

## 7. The paired bootstrap

`results/review/functional_audit_gate_bootstrap.json`. 10⁴ case-level resamples,
seed 0, arm S, with the top-decile threshold held **fixed** at its full-sample
value so "worst-decile case" stays a property of the case — the construction
`decisive_controls.py` used for control 1. Δ = variant − residual-norm baseline.

| seed | variant | ΔAUROC [95% CI] | Δρ [95% CI] |
|---|---|---|---|
| 0 | `adv_I_T0` (V1) | **−0.330 [−0.487, −0.188]** ✔ | **−0.361 [−0.526, −0.187]** ✔ |
| 0 | `adv_I_T0` (V5) | **−0.394 [−0.536, −0.260]** ✔ | **−0.335 [−0.480, −0.186]** ✔ |
| 0 | `adv_J_mag` (V1) | −0.006 [−0.029, +0.014] | **+0.077 [+0.010, +0.144]** ✔ |
| 1 | `adv_I_T0` (V1) | **−0.254 [−0.392, −0.130]** ✔ | **−0.508 [−0.674, −0.340]** ✔ |
| 1 | `adv_I_T0` (V5) | **−0.470 [−0.610, −0.330]** ✔ | **−0.264 [−0.430, −0.095]** ✔ |
| 1 | `adv_J_mag` (V1) | −0.008 [−0.033, +0.014] | **+0.132 [+0.070, +0.199]** ✔ |
| 2 | `adv_I_T0` (V1) | **−0.324 [−0.459, −0.194]** ✔ | **−0.421 [−0.585, −0.252]** ✔ |
| 2 | `adv_I_T0` (V5) | **−0.391 [−0.542, −0.246]** ✔ | **−0.299 [−0.454, −0.140]** ✔ |
| 2 | `adv_J_mag` (V1) | −0.003 [−0.027, +0.019] | **+0.059 [+0.003, +0.119]** ✔ |

✔ = CI excludes zero.

Three readings, and the third is the only constructive one in this report.

1. **The signed functional is significantly worse, not merely nominally worse.**
   ΔAUROC −0.25 to −0.47 with every CI excluding zero, on every seed and both
   boxes. Test (b) does not fail for want of power; it fails decisively and in the
   wrong direction. The pre-registered pass condition — ΔAUROC positive with a CI
   excluding zero — is failed by a wide margin with the sign reversed.
2. **The unsigned integral is statistically indistinguishable from the deployed
   norm on AUROC** (Δ −0.003 to −0.008, all CIs including zero). Taking the
   absolute value does not merely help; it recovers the deployed monitor's
   detection performance in full.
3. **And it is significantly better on Spearman** (Δρ +0.059 to +0.132, all three
   CIs excluding zero). A plain L1 integral of the residual magnitude over a box
   around the body ranks per-case drag error better than the deployed RMS norm
   does, at zero marginal cost — the residual maps already exist. This is a small,
   real, deployable improvement to the *ranking* claim, and it belongs in the
   "what remains deployable" section rather than in the certificate section. It
   should be stated with its scope: it is an improvement in rank correlation only,
   the AUROC is unchanged, and it certifies nothing.
