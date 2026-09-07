# Paper 1 reframe plan — after two desk rejections on originality

> ## RESOLVED 2026-09-07 — read this before the body of the plan
>
> **The blocking question in §4 is answered: the floor PLATEAUS, and in fact
> grows under refinement.** `p = −0.387` over 24 cases × 3 levels, rising in
> 21/24; corroborated at 16 cases × 5 rungs across three exclusion bands. So the
> strong branch obtains and §4's decision tree is closed. Full report:
> `floor_resolution_study.md`.
>
> **§3 was wrong about which claim was at risk.** Leg (i) — the uniform field's
> exactly-zero residual — was never a rasterisation artifact and cannot be one:
> a constant field has identically zero finite differences at every `h`. Only
> the floor's *magnitude* was ever grid-dependent, and it grows. The claim that
> did break is leg (A) of the λ-sweep, which §3 did not flag: its margin
> collapses monotonically with refinement and crosses zero by 512², so "for
> every λ ≥ 0" must be restated at the deployed resolution.
>
> **The mechanism is the reference-operator mismatch, not the closure omission.**
> Continuity and momentum rise at the same rate in the same cases, which rules
> out both single-term stories. The omitted `∇ν_t·∇u` term does grow, as
> predicted, but is ~5% of the floor — real, non-vanishing, and not the driver.
>
> **§6's statistical exposure is retired, by a better experiment than the one
> proposed.** `tab:iters` no longer has to carry the negative half. Descending
> the monitored residual from the *exact ground truth*, with no network in the
> loop, cuts it by 84% in 24/24 cases and takes the field error from zero to a
> median of 0.91. The error-optimal and residual-chosen iterates differ in 24/24
> cases on both arms and the residual-chosen one is strictly worse in 24/24.
> This also supersedes `novelty_hunt.md` §0a experiment (1): there is no
> backbone to call over-smoothed.
>
> **The claim to make is about iterate selection, not about harm.** From a
> perturbed start, residual descent *cuts* the error in 18/24 cases, typically
> by 60%. "Descending the residual makes the field worse" does not survive that
> and must not be written. "The residual-selected iterate is not the
> error-minimising iterate" holds in 24/24 and does.
>
> **Venue: Computers & Fluids**, per `venue_plan.md`, with the no-APC route and
> the sanctions question verified there. That check is done; do not re-open it.
>
> Remaining before submission: the rewrite itself (§8 steps 3–6), and the
> `tab:positioning` decision in step 4.
>
> ### The outline to write against — supersedes FINDINGS' "The paper this should become"
>
> FINDINGS builds item 1 on F3's closed-form floor `2S·∇ν_t ≈ 0.13`. **That
> attribution is refuted** (`floor_resolution_study.md` §11): adding the omitted
> term back moves the floor by −0.01% to +0.18%, and the term is ~4% of it, not
> 90%. Do not write 0.13. The corrected outline:
>
> 1. The monitored residual is **inconsistent with the data generator**, so the
>    truth itself fails the check. Two separable statements, never conflated:
>    *(a)* the inconsistency is provable, so the floor is bounded away from zero
>    even as `h → 0` — this is F3's theorem and it survives; *(b)* the floor's
>    measured **size** is set by the discretisation-and-mesh mismatch, and it
>    **grows** under refinement (`p = −0.387`, 21/24), with the closure omission
>    a ~4% component. Neither linear-vs-cubic interpolation nor operator repair
>    moves it.
> 2. Therefore it cannot be minimised — shown by actual residual descent, and
>    stated as **iterate selection** (24/24) rather than as harm (6/24).
> 3. Where the operator IS solver-consistent, residual correction works (Lei et
>    al.), so the boundary is the operator, not the problem class.
> 4. It still ranks errors usefully, but physics earns its keep only in fusion.
> 5. What remains deployable: the conformal layer, and the gate's certificate —
>    conceding that the gate's *accuracy* is damping (C3).
>
> Write order, to avoid rework: contributions → abstract → title → intro →
> `tab:positioning`.


Created 2026-09-07. Living document: update as the six research reports land.

## 1. The situation, stated without euphemism

Two desk rejections, no review, both on the same axis:

| Date | Venue | Stated reason |
|---|---|---|
| 2026-08-02 | CMAME | "no new computational methodology within scope" |
| 2026-09-07 | Journal of Computational Physics | "originality with respect to other published papers is too questionable" |

JCP will not reconsider. Ever. Do not appeal, do not email the editor.

Two independent Elsevier editors-in-chief, reading only title, abstract, and
introduction, converged on the same verdict: **this does not present as new.**
The science is verified. The framing failed. A third submission in the current
framing gets the same word a third time.

## 2. Diagnosis: the paper nominates its weakest claim as its headline

- `body.tex:80`, Contribution 1, labelled **(headline)**: the trust layer.
  Residual correlates with error, plus off-the-shelf split conformal. This is
  the LEAST novel thing in the paper. Nearest prior art is dense and recent:
  gopakumar2025pre (residual AS the conformal nonconformity score),
  roy2025anchor, jia2026multigranularity, garg2025dfuq, ma2024uqno.
- The genuinely unpublished content is demoted: the falsification of
  residual-as-correction-objective, and the residual-floor theorem.
- The title opens with a coined system name, "NeuroForge:". Under an
  originality screen that reads as "here is our new engine".
- The abstract's first substantive move is "we compute the discretised
  steady-state RANS residual" — standard practice, before any finding.
- `tab:positioning` is a seven-column checkmark grid whose argument is "no
  prior work combines all of these". A combination-checklist is the
  signature of an incremental paper. It likely invited the rejection.

## 3. The candidate headline, and why it is not yet safe to write

The sharpest result in the paper is theorem leg (i), in
`sections/residual_floor_theorem.tex`:

> The uniform freestream field has EXACTLY ZERO monitored residual, while the
> true flow field has residual norm 0.192 (mean, 200/200 AirfRANS cases).
> The residual objective strictly prefers a physically wrong field to the truth.

That is arresting and quotable. **But it may be a rasterisation artifact.**
The lambda-sweep that closes the "you omitted no-slip" objection rests on the
rasterised truth violating no-slip MORE than the uniform field (0.0073 vs
0.0053), which the theorem file itself attributes to the boundary layer being
sub-cell on a 128^2 grid. A CFD reviewer finds this quickly, and then the
headline collapses to "you under-resolved your data".

**This is the blocking question and it is being measured now.**

## 4. Decision tree on the resolution study

Experiment: rasterise the same AirfRANS truth at 128^2 / 256^2 / 512^2 and
measure how the floor behaves, decomposed into truncation, the omitted
grad(nu_t).grad(u) term, and the body-fitted-reference mismatch.

- **Floor DECAYS as O(h^p) toward zero.** Leg (i) is a grid artifact. Demote it
  to a remark. The headline must then be the falsification plus the trust
  layer, and the theorem becomes supporting material.
- **Floor PLATEAUS.** Much stronger, and not about the grid at all. The claim
  becomes: *you cannot audit a surrogate with a residual operator different
  from the one that generated its training data.* That generalises past 2-D,
  past 128^2, past RANS, and is very hard to desk-reject on originality.

Two of the floor's components should not vanish with refinement: the omitted
turbulence term is a MODEL omission, and the AirfRANS reference was computed
with a different discrete operator on a body-fitted mesh, so the rasterised
truth never satisfies our operator at any h. So a plateau is plausible. It must
be measured, not assumed.

## 5. Structural rule for the reframe

A falsification-led paper has its own desk-reject mode: "negative result, no
method." Leading with only the negative trades one rejection for another.

The structure that survives:

1. **Problem** — the falsification. The residual cannot fix what it detects,
   and here is the theorem saying why it structurally cannot.
2. **Resolution** — what to do instead. The calibrated trust layer, and the
   acceptance gate, which admits a step on 99.8% of deployed cases and lowers
   true error on 89.3% of them.

Negative as the question, positive as the answer. That is a paper. The
negative alone is a note.

## 6. Statistical exposure to fix regardless of outcome

`tab:iters` — the central empirical falsification — is by its own caption a
"Single sweep (one checkpoint, not seeded)". If the falsification becomes the
headline, the headline rests on n=1. That survives a desk screen and dies in
review.

The W1 residual-input ablation is the statistically solid falsification: 5
seeds, WITH beats NULL on 0/5 (`results/control/w1_capture.json`). Promote W1
to primary evidence and keep the iteration sweep as illustration. Re-run the
sweep across seeds if the checkpoints still exist.

## 7. Research in flight (six agents, launched 2026-09-07)

All six landed 2026-09-07; `floor_resolution_study.md` and `decisive_controls.md`
carry the results that moved claims.

| Report | Question it settles |
|---|---|
| `novelty_hunt.md` | Which of the three candidate claims is actually novel, with the prior art that would kill each |
| `theorem_audit.md` | Is the floor theorem correct, non-vacuous, tight, and more than "discretisation error exists" |
| `floor_resolution_study.md` | **BLOCKING** — does the floor plateau or decay |
| `r2_holes.md` | Worst genuine objections, ranked, with what is fixable vs must be conceded |
| `domain_expert.md` | Technical correctness, and whether the desk rejections were defensible on the merits |
| `venue_plan.md` | The next journal, verified live against the no-APC and no-word-cap constraints |

## 8. Sequencing

1. Reports land. Reconcile them; expect the novelty and domain-expert reports
   to disagree, and resolve rather than average.
2. Resolve the plateau-vs-decay question. This picks the headline.
3. Rewrite title, abstract, introduction, contribution order. Nothing before
   this point, or the prose is written twice.
4. Kill or rebuild `tab:positioning`.
5. Fix the statistical exposure in section 6 above.
6. Format for the chosen venue, handle anonymisation if required, rewrite the
   cover letter to lead with what that venue's screen rewards.
7. arXiv v4 after the reframe, not before.

## Constraints that do not move

- **No article processing charges, ever.** Subscription or hybrid with a
  genuinely free route only. Take the subscription licence at the licensing
  step; choosing open access there is what triggers the fee.
- Manuscript is ~14,700 words. Any venue with a cap near 8,500 is out.
- A public arXiv preprint (2607.10333 v3) exists under the authors' real names
  with the current title. Any double-anonymised venue has a real tension here,
  not merely a URL to swap.
- Nothing was lost in the rejection. Priority is timestamped on arXiv and the
  Zenodo DOI is minted.
