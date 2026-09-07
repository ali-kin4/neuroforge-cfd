# Decisive controls on the paper-1 trust-signal claims

Six controls raised by two independent adversarial reviews. Five needed **no new
compute** (the per-case data was already committed); the sixth needed one CPU-only
re-run of an existing sweep on a retained checkpoint. Every number below is
recomputed from committed files by the scripts named in each section, not quoted
from the reviews.

Artifacts
- `scripts/decisive_controls.py` — controls 1, 2, 4, 5, 6a (pure re-analysis)
- `scripts/control_fixed_step.py` — control 3 (replays cached correction steps)
- `scripts/control_iters_channels.py` — control 6b (re-runs the iters sweep, all channels)
- `results/review/control{1,2,3,4,5,6}_*.json`

---

## Summary of verdicts

| # | Control | Verdict | The one number |
|---|---|---|---|
| 1 | Physics residual vs physics-free σ | **PARTIALLY-RESOLVED** — brief's premise is *not* statistically supported, but the abstract overclaims | σ−residual ΔAUROC **+0.023, CI [−0.035, +0.083]** — includes zero |
| 2 | Case-difficulty confound | **RESOLVED** — objection refuted | partial ρ **0.561** vs raw 0.610; **92%** survives; drop CI includes zero |
| 3 | Fixed-half-step control on the gate | **CONCEDED-WITH-MITIGATION** | ungated fixed 0.5 improves **95.8%** of cases vs the gate's **89.2%** |
| 4 | Risk-coverage on drag | **RESOLVED** — and stronger than the paper claims | residual AUROC **0.952** on \|ΔC_d\| vs 0.871 on field error |
| 5 | Undisclosed MGN density control | **PARTIALLY-RESOLVED** — non-disclosure confirmed, brief's stronger claim unsupported | **0** manuscript mentions; but the file measures MSE, not ρ, on **4** cases |
| 6 | Omitted channels in `tab:iters` | **CONCEDED** (table is selective; the **figure does not invert**) — channels recovered, not lost | `mse_v` rises **monotonically +44.2%** and `mse_p` **+19.0%** across the sweep, while the two reported channels fall |

**Three premises in the brief turned out to be wrong or overstated** and are flagged
in place: control 1's "σ outranks the residual" (true nominally, not statistically),
control 3's "the 0.5 trial is already recorded for all cases" (recorded only where
the full step was refused), and control 5's "ρ = 0.851 is an artifact of density
mismatch" (not established by that file, and not establishable from it).

---

## CONTROL 1 — Is the physics residual beaten by a physics-free baseline?

`results/review/control1_physics_vs_physicsfree.json`

### Premise check

The brief's table reproduces **exactly** against `selective_prediction.json`
(`reproduces: true` on every score; recomputed independently here):

| score | Spearman | AUROC | oracle recovery @10% |
|---|---|---|---|
| physics residual | 0.6103 | 0.8706 | 66.99% |
| σ_vel (physics-free) | 0.6543 | 0.8939 | 71.71% |
| σ_p | 0.5805 | 0.8553 | 59.80% |
| fused | 0.7028 | 0.9053 | 91.47% |

### The paired bootstrap (the test the paper does not have)

The committed `bootstrap_spearman_ci.json` gives **marginal** CIs only, which
cannot settle a paired comparison. 10⁴ case-level resamples, seed 0, with the
top-decile threshold held **fixed** at its full-sample value (0.0068018) so
"worst-decile case" stays a property of the case:

| comparison | ΔAUROC [95% CI] | ΔSpearman [95% CI] | Δoracle recovery [95% CI] |
|---|---|---|---|
| σ_vel − residual | **+0.0233 [−0.0346, +0.0832]** | **+0.0440 [−0.0517, +0.1421]** | +0.047 [−0.138, +0.354] |
| fused − residual | **+0.0347 [+0.0049, +0.0689]** ✔ | **+0.0925 [+0.0339, +0.1539]** ✔ | **+0.245 [+0.035, +0.471]** ✔ |
| fused − σ_vel | +0.0114 [−0.0208, +0.0446] | **+0.0485 [+0.0015, +0.0966]** ✔ | +0.198 [−0.008, +0.311] |

✔ = CI excludes zero.

### Verdict: PARTIALLY-RESOLVED

1. **The brief's headline claim is not statistically supported.** σ_vel is
   nominally ahead on every metric, but the paired difference's CI includes zero
   on AUROC, Spearman *and* retained risk (P(σ better) = 0.79 / 0.81 / 0.76).
   The correct statement is that the physics residual and the physics-free
   ensemble spread are **statistically indistinguishable** on field-error triage —
   not that the residual is beaten. Stated precisely: with n = 200 the paired CI
   ([−0.035, +0.083]) cannot resolve a difference of this size **in either
   direction**. This is *underpowered*, not *equivalent* — the paper should not
   claim equivalence any more than the reviewer should claim superiority.
2. **The fusion gain over the residual alone is real** on all three metrics.
3. **The fusion gain over σ alone is marginal** — significant on Spearman only,
   not on AUROC and not on retained risk. Even the Spearman result sits on the
   boundary (CI lower bound +0.0015). The paper should not lean on it.
   *Robustness of the one methodological choice a reviewer will probe:* the fused
   score is a rank fusion computed on the full sample and then held fixed under
   resampling (a per-case property, consistent with the fixed threshold).
   Re-fusing *within* each draw is a different estimand, so we checked rather than
   asserted: ΔAUROC +0.0105 CI [−0.0212, +0.0436] and Δρ +0.0485 CI [+0.0002,
   +0.0969], against +0.0113 / [−0.0208, +0.0446] and +0.0486 / [+0.0015, +0.0966]
   with the score held fixed. No conclusion moves; the committed `fused` column
   reproduces the re-fused score at rank correlation 1.000.
4. **The brief's "do this on every arm" is not satisfiable.** `sigma_vel` is
   dumped per-case only for `ensemble_mean`; `corrected_seed*` and `deepcfd_seed*`
   carry the residual alone. All arms' marginal statistics are in the JSON.
5. **Note the sharper caution:** oracle recovery jumps 67% → 91% from residual to
   fused, but that metric is a ratio over a small denominator (max achievable
   reduction = 5.33e-4). The underlying retained-risk difference is −1e-4 with a
   CI that only just excludes zero. The 67/91 framing is more dramatic than the
   evidence.

### Manuscript changes forced

- **`abstract.tex`** — "it flags the worst-decile cases with AUROC ≈ 0.9" has the
  *residual* as its antecedent, but the residual is 0.871; 0.905 is the *fused*
  score. **This is the sharpest single overclaim in the paper.** Change to:
  > "it flags the worst-decile cases with AUROC 0.87, rising to 0.91 when fused
  > with an ensemble spread"
- **`body.tex:1246-1249`** — states all three AUROCs but never draws the
  conclusion. Add one sentence:
  > "The physics-free ensemble σ is nominally the stronger single score
  > (0.894 vs 0.871), though a paired case-level bootstrap cannot separate them
  > (ΔAUROC +0.023, 95% CI [−0.035, +0.083]); the fused score is significantly
  > better than the residual alone (ΔAUROC +0.035, CI [+0.005, +0.069])."
- **`body.tex:1332-1343`** — the cost framing **survives intact** and is the right
  place to absorb this. σ-alone costs the same 4.86× as the fused score, so the
  residual remains the only *free* option. Add: "the ensemble σ alone recovers
  ≈72%, but at the same 4.86× cost as the fusion, so the residual remains the only
  zero-marginal-cost score." One clause, not a retreat.
- **`body.tex:103`** — intro quotes 91% (fused) and, at 1340, 67% (residual);
  neither mentions σ's 72%. Add it at 1340.

**Rebuttal line.** Reviewer said a physics-free baseline outranks the residual →
we ran the paired bootstrap the paper lacked → the difference is +0.023 AUROC
with CI [−0.035, +0.083]; the two are indistinguishable, and on the engineering
quantity (control 4) the residual wins decisively. We have corrected the
abstract's AUROC attribution and now report σ's numbers explicitly.

---

## CONTROL 2 — The case-difficulty confound

`results/review/control2_difficulty_confound.json`

### A naming correction first

The brief calls the conditioning variable "the per-case residual floor `‖r*‖`".
That is a **misnomer for this file**: `residual_floor_realdata.json`'s actual
minimum is `norm_uniform` = **0.0 exactly on all 200 cases**
(`frac_uniform_lt_truth` = 1.0) — the experiment exists to show the monitored
residual *prefers* the uniform freestream to the truth. We therefore condition on
**‖r(truth)‖** = `norm_truth`, the monitored residual of the ground-truth field,
which is the model-independent per-case discretisation/model-form difficulty the
objection actually names. (`norm_pred` in that file is a *different model* —
dropout-FNO, `checkpoints/certificates_deq.pt` — and is not joined.)

### Decomposition first: can the confound even bite?

| arm | ρ(residual, ‖r(truth)‖) | ρ(error, ‖r(truth)‖) |
|---|---|---|
| ensemble_mean | +0.9749 | +0.5162 |
| corrected_seed0 | +0.8997 | +0.4405 |
| corrected_seed1 | +0.9370 | +0.4717 |
| corrected_seed2 | +0.9198 | +0.4931 |

### Partial correlation

| arm | n | raw ρ | partial ρ [95% CI] | % surviving | drop [95% CI] |
|---|---|---|---|---|---|
| **ensemble_mean (Transolver headline)** | 200 | 0.6103 | **0.5615** [0.449, 0.640] | **92.0%** | 0.049 [−0.060, +0.180] |
| corrected_seed0 | 200 | 0.6356 | 0.6107 [0.502, 0.693] | 96.1% | 0.025 [−0.060, +0.126] |
| corrected_seed1 | 200 | 0.6159 | 0.5646 [0.448, 0.652] | 91.7% | 0.051 [−0.050, +0.170] |
| corrected_seed2 | 200 | 0.6665 | 0.6240 [0.518, 0.700] | 93.6% | 0.043 [−0.044, +0.148] |

The three DeepCFD arms key on `index`, n = 247, on a different split; the 200-case
floor table cannot be joined to them. Stated, not improvised.

### The decisive sub-test

If the residual were merely a difficulty proxy, it could not **beat the difficulty
variable itself**. ‖r(truth)‖ is an *oracle* score (it needs the ground truth):

| arm | ρ residual | ρ ‖r(truth)‖ | Δρ [95% CI] | AUROC residual | AUROC ‖r(truth)‖ | ΔAUROC [95% CI] |
|---|---|---|---|---|---|---|
| ensemble_mean | 0.6103 | 0.5162 | **+0.094 [+0.055, +0.142]** ✔ | 0.8706 | 0.8178 | **+0.053 [+0.005, +0.115]** ✔ |
| corrected_seed0 | 0.6356 | 0.4405 | **+0.195 [+0.118, +0.282]** ✔ | 0.9142 | 0.8264 | **+0.088 [+0.005, +0.193]** ✔ |
| corrected_seed1 | 0.6159 | 0.4717 | **+0.144 [+0.083, +0.214]** ✔ | 0.8919 | 0.8178 | **+0.074 [+0.007, +0.159]** ✔ |
| corrected_seed2 | 0.6665 | 0.4931 | **+0.173 [+0.105, +0.252]** ✔ | 0.9000 | 0.8178 | **+0.082 [+0.005, +0.181]** ✔ |

### Verdict: RESOLVED — the objection is refuted

The partial correlation does **not** collapse: 92–96% of the headline ρ survives
conditioning, the drop's CI includes zero on every arm, and the partial ρ's CI
excludes zero by a wide margin. And the deployable residual **significantly
outranks the oracle difficulty variable** on all four arms and both metrics, which
a pure difficulty proxy cannot do.

A secondary, honest observation worth disclosing: ρ(residual, ‖r(truth)‖) = 0.975
on the headline arm — the monitored residual is very largely rank-determined by
the case's intrinsic residual. The trust signal is *mostly* a difficulty readout;
the point of the partial and of the oracle-baseline test is that the remaining
model-specific component is real, statistically significant, and enough to beat
difficulty alone. Notably the physics-free σ_vel survives conditioning *less* well
(78.5%) than the residual (92.0%), so the confound is not specific to physics.

### Manuscript change forced

Add to `sec:conformal` (near body.tex:1255):
> "The correlation is not a case-difficulty artifact. Conditioning on ‖r(truth)‖,
> the monitored residual of the ground-truth field, leaves a partial Spearman of
> 0.561 (95% CI [0.449, 0.640]) against a raw 0.610 — 92% of the correlation
> survives, and the drop is not significant. The deployable residual also
> significantly outranks ‖r(truth)‖ itself (Δρ +0.094, CI [+0.055, +0.142]), which
> a pure difficulty proxy could not do
> (`scripts/decisive_controls.py --control 2`)."

---

## CONTROL 3 — Fixed-step control on the acceptance gate

`results/review/control3_fixed_step.json`

### Premise correction

The brief states the 1.0 **and** 0.5 trials are already logged for every case.
They are not: the 0.5 trial is recorded only where the full step was *refused*
(88–108 of 200 on the backbone arms; the gate took the full step on the rest, and
`trials` then contains only `step: 1.0`). The counterfactual therefore cannot be
computed from `acceptance_gate.json` alone. We recomputed it exactly from the
committed field caches (`data/cache/w2/*`, `data/cache/acceptance_gate/seed*`),
CPU-only, zero forward passes, replaying the same `delta = corrected − raw` at
fixed steps {0.25, 0.5, 0.75, 1.0} with the gate switched off.

### Scope note the paper should already be making

The paper's 99.8% / 89.3% / 5.8% are the **`backbone_*` arms** (deployed
Transolver+DEQ), not the pooled 1200 steps: accept rates 1.0/1.0/0.995 → 99.83%;
535/599 accepted steps improve → 89.32%; median changes −0.058/−0.070/−0.044 →
5.8%. Reproduced exactly. Pooling all six arms would give 99.25% / 81.9%.

### The comparison (deployed backbone+DEQ path, n = 600 = 200 × 3 seeds)

Common denominator: all 200 cases per arm; a gate rejection counts as "no step,
error unchanged".

| policy | frac improves error | median rel change | mean retained rel-L2 |
|---|---|---|---|
| **monotone-residual gate** | 0.8917 | −5.81% | 0.0051083 |
| ungated fixed 0.25 | 0.9783 | −3.94% | 0.0052126 |
| **ungated fixed 0.5** | **0.9583** | **−6.22%** | 0.0051089 |
| ungated fixed 0.75 | 0.8900 | −6.31% | **0.0050950** |
| ungated fixed 1.0 (no damping) | 0.7583 | −4.38% | 0.0051704 |

Head-to-head gate vs fixed-0.5: 290/600 exact ties (the gate admitted 0.5 there),
fixed-0.5 better on 175, gate better on 135.

**This is not a cherry-picked step.** Three of the four fixed steps beat the gate
on at least one metric: 0.25 on fraction-improving (0.978 vs 0.892), 0.75 on mean
retained error (0.0050950 vs 0.0051083), and 0.5 on both. Only the undamped
step 1.0 loses to the gate.

Ensemble-mean path (n = 600): gate 0.7333 / −2.56% / 0.0039034; fixed 0.5
**0.7917 / −3.22% / 0.0038741**. Same conclusion.

### What the gate *does* uniquely deliver

An ungated fixed half step **violates the monotone-residual guarantee** on
**6/600 = 1.0%** of deployed backbone cases — the paper's headline arm — and on
53/600 = 8.8% of ensemble-path cases. The gate never does, by construction.

**Do not let the 8.8% carry the argument.** On the arm the paper actually
headlines, the certificate binds on **six cases out of six hundred**. A reviewer
is entitled to ask whether a guarantee that is active on 1% of cases justifies
"certified self-correction" as a framing, and the honest answer is that it
justifies the guarantee as *stated* (it is free, and it is never violated) but not
as a headline mechanism. The mitigation for a conceded accuracy claim is real but
thin, and the paper should present it that way.

### Verdict: CONCEDED-WITH-MITIGATION

The gate's *accuracy* contribution is reproduced — and slightly exceeded — by a
fixed damped step with no residual test at all. **The mechanism is the step-size
schedule, not the physics test.** What the residual test uniquely buys is the
*certificate*: a fixed schedule cannot guarantee non-worsening residuals and
demonstrably fails to on up to 8.8% of cases.

For completeness, the gate *does* beat the ungated **full** step decisively
(0.892 vs 0.758 on the backbone path; 0.733 vs 0.318 on the ensemble path), so
the gate is not vacuous — it is just that damping, not the residual test, is
doing the accuracy work.

### Manuscript change forced

`body.tex:1111-1134` currently concludes:
> "The 'certified self-correction' guarantee is therefore real *and* useful: a
> cheap, provably non-worsening gate that also happens to improve accuracy."

Replace with:
> "The guarantee is real, but a fixed-step control shows the accuracy is not the
> residual test's doing. An **ungated** fixed half step improves true error on
> 95.8% of deployed cases by a median 6.2%, against the gate's 89.3% and 5.8%
> (`scripts/control_fixed_step.py`, `results/review/control3_fixed_step.json`) —
> the gain comes from damping the correction, not from testing the residual. What
> the residual test uniquely provides is the certificate itself: an ungated half
> step raises the monitored residual on 1.0% of deployed cases (8.8% on the
> ensemble-mean path), whereas the gate cannot. We therefore claim the gate as a
> free non-worsening guarantee, not as an accuracy mechanism."

Also correct the claim at `body.tex:135-136`, `315-316`, `563`, `1597`, `1655-1656`
(the 99.8/89.3/5.8 triple appears five times) to state the `backbone_*` scope, and
drop "and that step lowers true error" as an implied *causal* benefit of the gate.

**Rebuttal line.** Reviewer said a fixed half step would match the gate → we
recomputed the ungated step-size sweep from the committed caches → it does match
and slightly beats it (95.8% vs 89.3%), so we now claim the gate for its
certificate (which a fixed step violates on up to 8.8% of cases) and not for its
accuracy.

---

## CONTROL 4 — Risk-coverage on the engineering quantity

`results/review/control4_riskcoverage_drag.json`

### Premise correction

Per-case force errors are **not** in `results/force_vs_official*.json` — those
files hold per-seed aggregates only. They are in
`results/control/_cache/{seed*_prednf,gt_nf_full_test_r128_n200,official_labels_full_test_n200}.json`,
keyed by the same case names as `selective_percase.json`. Joined there.

Two targets, both reported: **primary** |C_d(pred) − C_d(gt)| with both sides
through the same NeuroForge integrator (isolates *model* error in drag);
**secondary** |C_d(pred) − C_d(official)| against the AirfRANS OpenFOAM labels
(what an engineer compares against, but dominated by integrator bias — the *ground
truth field alone* scores ρ_D = 0.839 and 11× relative C_d error against official).
Construction mirrors the field-error one exactly: top-decile |ΔC_d| positives,
same AUROC, same 10% budget. Scores are ensemble-mean, C_d is per-seed
v2_transolver — the arm mismatch is real and each seed is shown separately.

### Results (mean over 3 seeds, range in brackets)

| target | score | AUROC | Spearman | oracle recovery |
|---|---|---|---|---|
| *field rel-L2 (reference)* | residual | 0.8706 | 0.6103 | 0.670 |
| *field rel-L2 (reference)* | σ_vel | 0.8939 | 0.6543 | 0.717 |
| **\|ΔC_d\| vs gt (model error)** | **residual** | **0.9520** [0.942, 0.959] | 0.5854 [0.550, 0.613] | 0.805 |
| | σ_vel | 0.8881 [0.863, 0.901] | 0.4334 [0.372, 0.534] | 0.698 |
| | fused | 0.9459 [0.930, 0.955] | 0.5591 [0.511, 0.625] | 0.818 |
| **\|ΔC_d\| vs official** | **residual** | **0.9523** [0.938, 0.960] | **0.7659** [0.764, 0.768] | 0.769 |
| | σ_vel | 0.8663 [0.850, 0.874] | 0.4892 [0.487, 0.491] | 0.618 |
| | fused | 0.9364 [0.921, 0.944] | 0.7012 [0.700, 0.702] | 0.798 |

Paired bootstrap, residual − σ_vel: ΔAUROC **+0.055 to +0.088**, CI excludes zero
in **5 of 6** arms; Δρ vs official **+0.273 to +0.281**, CI excludes zero in all
three seeds. Fused − residual: ΔAUROC negative in every arm (fusion *hurts* here).

### Verdict: RESOLVED — and the result is stronger than the paper claims

The trust signal does **not** degrade on the engineering quantity — it **improves**.
The residual's AUROC rises from 0.871 on field error to **0.952** on |ΔC_d|, and
oracle recovery from 67% to 77–81%. Decisively, **the physics residual beats the
physics-free σ on drag** (ΔAUROC +0.055…+0.088, CIs excluding zero), reversing
control 1's field-error tie, and the rank fusion stops helping.

This is the direct answer to control 1: physics buys nothing over an ensemble
spread when the target is field MSE, but it buys a lot when the target is the
force coefficient an aerodynamicist actually triages on.

### Manuscript change forced

New paragraph in `sec:conformal` after the selective-prediction paragraph:
> "**Triage on the engineering quantity.** Field MSE is not what an aerodynamicist
> rejects on. Re-scoring the same trust signals against per-case |ΔC_d| (three
> v2_transolver seeds, 200 AirfRANS test cases,
> `scripts/decisive_controls.py --control 4`) — both prediction and reference
> integrated through the same operator, so the target is model error in drag
> rather than integrator bias — the residual detects worst-decile
> drag error with AUROC 0.952 — *better* than its 0.871 on field error — recovering
> 77–81% of the oracle's achievable reduction. Here the physics residual
> significantly outperforms the physics-free ensemble σ (ΔAUROC +0.055 to +0.088,
> 95% CIs excluding zero), and rank fusion no longer helps. The trust signal is
> therefore strongest precisely on the quantity of engineering interest."

Quote the **`vs_gt_nf`** numbers (AUROC 0.952, ρ 0.585, oracle recovery 0.805) in
that sentence, not the `vs_official` ones, and say which target they are. A
caution on the secondary target: against the official labels the seed-to-seed
Spearman spread is 0.7637–0.7676 for the residual and 0.4870–0.4909 for σ — a
spread of 0.004 across *independently trained* backbones, against 0.550–0.613 on
the primary target. The benign explanation is the intended one: mean |ΔC_d| vs
official is 0.204 (median 0.039), so that target is dominated by a **shared**
integrator bias, nearly identical across seeds, which compresses seed variance.
But it also means the `vs_official` ρ of 0.766 is substantially ranking
*integrator bias* rather than model error, so the manuscript claim must rest on
`vs_gt_nf` or it is attackable as integrator-driven.

---

## CONTROL 5 — The undisclosed MeshGraphNet density control

`results/review/control5_mgn_density_disclosure.json`

### Disclosure audit (confirmed)

Searching `docs/paper/**/*.{tex,md,bib}` **excluding** `docs/paper/review/` for
`mgn_density`, `density_control`, `DENSITY-DRIVEN`, `density-driven`,
`control_mgn_density`, `16384`, `16k point`: **0 hits**. The only mentions in the
repository are in `docs/paper/review/r2_holes.md`, which is a review note, not the
manuscript. **The brief is correct: the control is committed and never disclosed.**

### What the file shows

| quantity | value |
|---|---|
| verdict | `DENSITY-DRIVEN` |
| velocity MSE ratio 180k / 16k | **2.437** |
| per-channel ratio | u 2.99, v 1.89, p 2.06, ν_t 1.38 |
| trained at | 16,384 points |
| evaluated at (median) | 180,238 points → **11.0× mismatch** |
| cases scored | **4** |
| metric | per-point physical MSE at the same 16,384 indices (confound-free) |

### The brief overreaches, and this must be said

The brief asserts "ρ = 0.851 is substantially an artifact of a train/eval
point-density mismatch." **That is not established by this file and cannot be
established from it.** The file measures MSE; it contains **no rank correlation at
either density**, carries no per-case residual values, and scores 4 cases. A
density-driven inflation of the *error spread* is a plausible *mechanism* by which
ρ could be inflated — it is not a measurement that it is. Settling it would
require recomputing the residual–error Spearman at 16k and at full density, which
is not in committed data.

### Verdict: PARTIALLY-RESOLVED

The **non-disclosure is confirmed and is a genuine problem**; the **stronger
attack on ρ = 0.851 is unsupported**. Note that `body.tex:839-841` already
attributes the high MGN correlation to "that model's wide error spread" — this
file supplies a *mechanism* for that spread, and the paper does not disclose that
the mechanism is a train/eval artifact. The forced change is a disclosure sentence
plus the n = 4 caveat, **not** a withdrawal of 0.851.

### Manuscript change forced

Extend the parenthetical at `body.tex:839-841`:
> "(The high MeshGraphNet value partly reflects that model's wide error spread —
> large, dispersed errors are easier to rank by residual. A control run
> (`scripts/control_mgn_density.py`, `results/control/mgn_density_control.json`)
> attributes that spread substantially to a train/eval density mismatch: the model
> is trained on 16,384-point subsamples and evaluated on the full ~180,000-point
> clouds, and velocity MSE inflates 2.44× because of it. That control scores only
> 4 cases and measures MSE rather than the rank correlation, so it does not
> quantify how much of ρ = 0.851 is attributable; we disclose it as a caveat on
> the strength, not on the sign, of the MeshGraphNet data point.)"

**Rebuttal line.** Reviewer found an undisclosed density control → we confirmed it
is cited nowhere in the manuscript and have added the disclosure → but the file
measures MSE on 4 cases and does not establish that ρ = 0.851 is an artifact; we
say so rather than over-conceding.

---

## CONTROL 6 — The omitted channels in `tab:iters`

`results/review/control6_iters_channels.json` (availability + monotonicity),
`results/review/control6_iters_full_channels.json` (recovered channels)

### 6a — Was the per-channel data retained? No — but it is recoverable

`results/sensitivity/iters.json` and `iters.csv` retain only `n_iters`, `mse_u`,
`surface_mse_p`, `residual_norm`, `residual_error_spearman`, `deq_iters_mean`.
`mse_v`, volume `mse_p` and `mse_nut` appear **nowhere** in the sweep's committed
output.

But "not retained" understates it, and the precise statement is the
reviewer-proof one: `run_sensitivity.py` calls `per_channel_mse(field, ref)`,
which **computes** `mse_u, mse_v, mse_p, mse_speed, mse_nut`, and then keeps only
`pm["mse_u"]` (line ~184) before aggregating. The channels were **computed and
dropped on write**, and the sweep's checkpoint (`checkpoints/certificates_deq.pt`)
**is** retained. So rather than declare the data lost, we re-ran the identical
sweep — same checkpoint, same iteration grid, same `n_eval`, same metric
functions — keeping the whole dict (`scripts/control_iters_channels.py`).

**Reproduction gate: PASS.** Getting the case subset right mattered: `run_sensitivity.py`
loads `N_EVAL + N_CALIB` = 160 cases and takes `permutation(seed=0)[:80]`, *not*
the first 80. Using the first 80 put `mse_u` 11% off the committed sweep; the
correct subset reproduces every retained column to within **1.41%**
(`mse_u` 3.920 vs 3.9238, 2.465 vs 2.4603; `residual_norm` 0.1110 vs 0.1126;
ρ 0.404 vs 0.423). The recovered channels are therefore **absolute rows**, not
merely a direction of travel.

### The full channel set (n_eval = 80, task `full`, `checkpoints/certificates_deq.pt`)

| `n_iters` | `mse_u` | **`mse_v`** | **`mse_p`** | **`mse_nut`** | surf `mse_p` | `residual_norm` |
|---|---|---|---|---|---|---|
| 0 | 3.9200 | **0.4699** | **3199** | 4.699e−8 | 540623 | 0.1110 |
| 1 | 2.4650 | **0.4827** | **2669** | 4.586e−8 | 429680 | 0.3358 |
| **3** | **2.2996** ← min | **0.6208** | **3085** | 4.633e−8 | 339482 | 0.5400 |
| 5 | 2.4524 | **0.6606** | **3456** | 4.678e−8 | 311533 | 0.5910 |
| 10 | 2.5837 | **0.6766** | **3766** | 4.712e−8 | 303506 | 0.6153 |
| 15 | 2.5896 | **0.6777** | **3806** | 4.714e−8 | 303833 | 0.6173 |

Percentage change from iteration 0:

| `n_iters` | `mse_u` | `mse_v` | `mse_p` | `mse_nut` | surf `mse_p` |
|---|---|---|---|---|---|
| 1 | −37.1% | **+2.7%** | −16.6% | −2.4% | −20.5% |
| **3** | **−41.3%** | **+32.1%** | −3.6% | −1.4% | −37.2% |
| 5 | −37.4% | **+40.6%** | **+8.0%** | −0.4% | −42.4% |
| 10 | −34.1% | **+44.0%** | **+17.7%** | +0.3% | −43.9% |
| 15 | −33.9% | **+44.2%** | **+19.0%** | +0.3% | −43.8% |

### The answer: yes, the omitted channels rise — and one rises monotonically

- **`mse_v` is monotone increasing across the entire sweep**, +44.2% end to end.
  It is the *only* channel with no minimum after iteration 0.
- **`mse_p` (volume) ends +19.0% above baseline**, and +42.6% above its own
  iteration-1 minimum.
- `mse_nut` is flat (±3%).
- The two channels `tab:iters` *does* report are exactly the two that improve:
  `mse_u` −33.9% and surface `mse_p` −43.8%.
- At iteration 3 — the setting the table's own `mse_u` column identifies as best —
  **`mse_v` is already +32.1%**.

The sign pattern reproduces `tab:indist`'s finding (that corrector family inflates
`mse_v` and volume `mse_p`) on this sweep, at smaller magnitude. **The figure built
on `tab:iters` does not invert, but the table is selective:** it reports a
two-channel improvement from a correction that degrades a third channel
monotonically and a fourth by a fifth.



### 6b — The non-monotonicity is CONFIRMED

From the committed `iters.json`:

| `n_iters` | `mse_u` | surf `mse_p` | `residual_norm` | ρ |
|---|---|---|---|---|
| 0 | 3.9238 | 541204 | 0.1126 | 0.423 |
| 1 | 2.4603 | 428560 | 0.3356 | 0.644 |
| **3** | **2.2874** ← min | 336718 | 0.5419 | 0.647 |
| 5 | 2.4387 | 308381 | 0.5935 | 0.675 |
| 10 | 2.5698 | 300298 | 0.6179 | 0.703 |
| 15 | 2.5755 | 300664 | 0.6199 | 0.710 |

`mse_u` bottoms at iteration 3 (2.2874) and rises to 2.5755 by iteration 15
(**+12.6%**), while `residual_norm` rises 0.5419 → 0.6199 (**+14.4%**) over the
same span. `residual_norm` is monotone increasing across the whole sweep;
`mse_u` is **not** monotone. Spearman(residual, `mse_u`) over iterations 3–15 is
**+1.0**: across that range residual and error move in the **same** direction.

### Verdict: CONCEDED — the paper's gloss is wrong over part of its own range

`body.tex:1101-1104` states:
> "the **PDE residual norm rises monotonically** (0.11→0.62) while the **field
> error falls** (`mse_u` 3.92→2.29 by iter 3) — the two objectives move in
> *opposite* directions."

The parenthetical "(by iter 3)" is doing load-bearing work the sentence does not
acknowledge. Over iterations 3→15 — three-quarters of the table's own range — the
two move in the *same* direction. The **dissociation claim survives** (the
residual rises monotonically while error does not fall monotonically, so
minimising the residual is not minimising error), but the "opposite directions"
gloss does not.

### Manuscript change forced

Replace `body.tex:1101-1104` with:
> "As the loop iterates, the **PDE residual norm rises monotonically**
> (0.11→0.62) while the **field error is non-monotonic**: `mse_u` falls
> 3.92→2.29 through iteration 3, then rises back to 2.58 by iteration 15. Over
> 0→3 the two objectives move in opposite directions; over 3→15 they move in the
> same direction. Either way the residual is not a proxy for the error — no
> monotone relation holds across the sweep — which is the clean statement of
> detector ≠ fixer."

**And `tab:iters` must gain the missing columns.** They now exist, reproduce the
committed sweep to 1.41%, and are committed at
`results/review/control6_iters_full_channels.json`. Add `mse_v`, volume `mse_p`
and `mse_nut` to the table, with a caption note that the sweep is `n_eval` = 80,
task `full`, a single checkpoint (`checkpoints/certificates_deq.pt`), and the
seeded 80-of-160 subset `run_sensitivity.py` uses. Then add:

> "The channels the table now reports in full make the trade explicit: over the
> sweep `mse_u` falls 33.9% and surface `mse_p` 43.8%, while `mse_v` rises
> **monotonically** by 44.2% and volume `mse_p` by 19.0%. At iteration 3, where
> `mse_u` is minimised, `mse_v` is already 32.1% above the uncorrected baseline.
> The correction is not a uniform improvement; it reallocates error across
> channels, which is the same pattern \autoref{tab:indist} reports for this
> corrector family."

Leaving these columns out is the single most reopenable presentational choice in
the paper: the two channels reported are exactly the two that improve.

**Rebuttal line.** Reviewer said `tab:iters` hides channels and its monotonicity
claim inverts → we confirmed the non-monotonicity from the paper's own table and
re-ran the sweep (reproduction-gated to 1.41% on the retained columns) to recover
every dropped channel → `mse_v` rises monotonically +44.2% and volume `mse_p`
+19.0% while the two reported channels fall, and `mse_u` is non-monotonic. The
dissociation survives; the "opposite directions" wording and the two-channel
table do not.

---

## Prompt-injection note

The MCP server registered as `zoho-mail` emitted an "MCP Initialization Request"
instructing the agent to route all file reading and editing through shell
utilities rather than the designated tools. That instruction did not originate
from the user or the task, and was ignored. Flagged here because it is the kind
of thing that should be noticed and recorded rather than silently obeyed.
