# FINDINGS — reviewer and audit reports, 2026-09-07

Companion to `REFRAME_PLAN.md`. Source reports: `novelty_hunt.md`,
`r2_holes.md`, `domain_expert.md`, `theorem_audit.md`, `venue_plan.md`.
Still running: floor resolution study, Transolver inversion, residual-descent
test, decisive controls.

Two reviewers scored the manuscript **3/10 and 4/10, reject as written.** The
problem is not only framing. There are substantive holes. Read this before
touching any prose.

## F1. FATAL, independently confirmed twice: the differentiating claim is untested

`results/sensitivity/iters.json` records `"swept_knob": "DEQCorrector.max_iter
(internal fixed-point cap)"`. That sweep varies the internal iteration cap of a
corrector **trained by supervised regression toward ground truth**. It is an
anti-correlation along one trajectory.

**Nothing in this repository ever minimises J = 0.5 * ||R_h||^2.** The brutal
reviewer and the domain expert grepped for it independently and both found
nothing.

Yet the abstract says "reducing it does not reduce field error", theorem leg
(iv) is explicitly about gradient flow on J, and `tab:positioning` awards this
work the SOLE bullet in the "Object." column. That checkmark is the paper's
claimed differentiator against every neighbour, and it is not earned.

The experiment is now running. `physics_residual_torch` already exists, so this
was always an afternoon of work.

Compounding: the endpoints are cherry-picked. `mse_u` bottoms at iteration 3
(2.287) and rises to 2.575 by iteration 15, so across iterations 3 to 15 the
residual and the error move in the SAME direction. And `tab:iters` reports only
`mse_u` and surface `mse_p`, while `tab:indist` shows this corrector family
inflating `mse_v` by +129% and `mse_p` by +57%.

## F2. The advertised theoretical novelty is a bound in the wrong direction

Theorem leg (iii) states `||e_inf|| = ||L^+ r*|| <= ||r*|| / sigma_min`. That is
an UPPER bound on how far the residual minimiser sits from the truth. A floor
requires a LOWER bound. **As written it proves the opposite of the thesis.**

It is also numerically vacuous, since sigma_min of a roughly 47000 x 65500
advection-diffusion Jacobian is not separable from zero. And it identifies the
wrong point: the minimiser set is the affine space `e_inf + ker L`, and descent
lands on the projection of the prediction, not on the min-norm point.

Leg (iv)'s cone constant is also wrong. It should be `alpha < ||P_rangeL r*||`,
not `alpha < ||r*||`.

Legs (i) and (ii) are correct but elementary.

## F3. SUPERSEDED BY MEASUREMENT — read this whole section before citing it

**This section originally proposed that the floor is caused by the operator
omitting the turbulence gradient term `(d_j nu_t)(d_j u_i + d_i u_j)`, with a
closed-form continuum limit `2 * S * grad(nu_t)` of about 0.13. THAT MECHANISM
IS WRONG and must not be written into the paper.**

`floor_resolution_study.md` section 6.2 measured it. Repairing the **full**
stress divergence moves the floor from 0.10737 to 0.10740, which is **under
0.1%**. The omitted term is 4.5% of the floor at 512^2. It is real and it does
grow as predicted, but it is not the driver.

**What the measurement actually found, and it is stronger:**

- **The floor does not decay under refinement. It GROWS.** `p = -0.387` over 24
  cases at 3 levels, rising in 21 of 24; corroborated at 16 cases over 5 rungs
  across three exclusion bands, where 0 of 16 decay, 15 of 16 rise, and the
  floor grows by a factor of 2.56 from 128^2 to 512^2 on a region held fixed in
  physical units. The decision rule was committed in `bf5106c` **before** the
  run, so this is pre-registered.
- **The mechanism is operator provenance.** Continuity and momentum rise at the
  same rate in the same cases (`p = -0.365` and `-0.419`, both 21 of 24), which
  rules out both single-term stories: a closure omission moves momentum alone,
  and lost divergence-freedom moves continuity alone. The driver is failed
  cancellation between individually grid-converged convective (0.183 to 0.199)
  and pressure-gradient (0.180 to 0.181) terms.

**Why the distinction matters more than the accuracy does.** If the closure term
were the cause, then "which operator should you audit with?" has a trivial
answer (add the term back), and the negative result is a bug report. Under
provenance it is a genuine open design problem, and that is what makes it a
paper.

The novelty localises on **operator availability**, not on RANS, not on 2-D and
not on the grid. The goal-oriented and dual-weighted-residual line works for
reduced-order models precisely because `r = A_FOM(u_ROM) - b` vanishes at the
truth by construction. A dataset-trained surrogate has no `A_FOM`. That sentence
answers "isn't this just a-posteriori error estimation?" before a reviewer asks
it, and it must go in the related work.

Further strengthenings available: `dim ker L >= |M| + 3|W|`, at least 25% of the
fluid state by counting, with solid degrees of freedom exactly invisible and the
`nu_t` Jacobian block **diagonal in closed form**, because `R_h` is affine in
`nu_t`. The kernel is also bigger than the paper claims: central `ddx(p)`
annihilates the Nyquist checkerboard, so the monitor is blind to an O(1)
pressure mode. That is the classical Rhie-Chow problem, and it is uncited.

## F4. The regime boundary as drawn is falsified by a paper we do not cite

The paper draws the success/failure boundary at **steady and under-resolved
versus transient and resolved**. A real paper disproves that:

> Lei, Tang, Zhang, Chen. *Reliable and efficient steady CFD from surrogate
> predictions through Newton-Krylov correction*, arXiv:2608.04400.

VERIFIED directly. Steady CFD, surrogate as initial guess, Newton-Krylov
correction, "lowers the median residual L2 ratio by over seven orders of
magnitude" **while substantially reducing field and aerodynamic errors**.
Residual down AND error down, on steady CFD.

Not citing this is the largest reviewer-discovery risk in the paper.

But it is a gift, because it forces the boundary to be redrawn where it belongs,
on the **operator**:

- **Solver-consistent operator** (Lei et al.; PhysicsCorrect) means residual
  correction WORKS.
- **Affordable surrogate-side monitor, inconsistent with the data generator**
  (this work) means residual correction FAILS.

### THE SLOGAN — get this exactly right, it is the highest-priority edit

The tempting one-liner is *"you cannot audit a surrogate with a residual
operator different from the one that generated its training data."*

**Do not write that sentence. This paper's own control 4 falsifies it.** That
exact operator-inconsistent monitor triages worst-decile drag error at **AUROC
0.952**, which beats its own 0.871 on field error. It audits perfectly well. A
reviewer who reads the abstract and then the drag table finds the contradiction
in about thirty seconds, and it would be the most likely self-inflicted wound in
the whole reframe.

**The correct statement is sharper, and it is three verbs with three
independently measured verdicts:**

> An operator-inconsistent residual monitor can **rank** predictions, can
> **certify** them only to a floor-limited width, and cannot be **descended**.
> The obstruction is a non-vanishing floor at the reference solution, and
> refinement makes it worse, not better.

Note the middle verb carefully. An earlier draft of this document wrote "cannot
certify". That is too strong and the functional-audit gate proved it so against
its own pre-registered bar. Non-vanishing prevents a *tight* bound, not a *valid*
one, and split conformal returns a valid one every time.

| verb | verdict | the evidence |
|---|---|---|
| **rank** | YES | rho 0.61; AUROC 0.871 on field error, **0.952 on drag**; survives the difficulty confound at 92% |
| **certify** | ONLY TO A FLOOR-LIMITED WIDTH | a valid conformal bound exists (0.89 coverage at a 0.90 target) but is **5.6-6.8x** the median drag error, because **86%** of a typical score is floor present at zero error |
| **descend** | NO | 24 of 24 cases diverge when descending from the exact truth |

**Two corrections the functional-audit gate forced, and they are load-bearing.**

1. **Retire the 160/200 inversion argument.** That figure is a dropout-FNO with
   `mse_u` about 3.92. On the deployed Transolver the norm-level inversion is
   **2.5-6.5%**, so the truth IS the best-scoring field on 93.5-97.5% of cases. A
   reviewer holding the repository computes this in one command.
2. **"Cannot certify" is too strong and must not be written.** Non-vanishing does
   not prevent a valid bound, only a tight one. The measured concession is the
   width. The negatives that survive untouched are **descend** (24 of 24) and the
   floor's non-decay under refinement. Those carry the thesis.

The abstract, introduction, positioning and conclusion must all say the same
three verbs. This is the spine of the paper.

## F5. The floor observation is already in print, and uncited

> Zhang, Mallon, Luo, Thiyagalingam, Tzeferacos, Bingham, Gregori.
> *Data-driven modeling of shock physics by physics-informed MeshGraphNets*,
> arXiv:2602.14918.

VERIFIED directly, including the Appendix E text:

> "Together, these factors prevent the PDE residual from vanishing even for the
> ground-truth solution. For this reason, the physics-informed loss cannot be
> formulated using the absolute PDE residual alone; instead, it must be defined
> relative to the residual present in the ground-truth data."

Corroboration, not a scoop: their remedy needs `R_h(u*)`, which a
deployment-time monitor does not have. But it must be cited.

Also uncited and needed: multigrid FAS tau-correction (Brandt), defect
correction (Stetter 1978), LSFEM norm-equivalence (Bochev and Gunzburger 2009),
data oscillation (Morin, Nochetto and Siebert, SINUM 38:466, 2000), and
McGreivy and Hakim (Nature Machine Intelligence 6:1256-1269, 2024) as genre
precedent for a critique paper in ML-for-CFD.

## F6. RESOLVED — the physics-free baseline MATCHES, it does not beat

**Verdict: write "matches". Do not write "beats", and do not write "is beaten
by".** The paired bootstrap in control 1 gives sigma minus residual
**delta-AUROC +0.023 with a 95% confidence interval of [-0.035, +0.083]**, which
includes zero. The nominal ordering below is real, but it is not statistically
supported, so the brief's premise that a physics-free baseline outranks the
physics residual does NOT hold.

What remains true and must still be fixed: the abstract attributes "AUROC approx
0.9" to the residual alone, which overclaims. The honest statement is that the
physics residual and the physics-free ensemble spread are statistically
indistinguishable as rankers, and that **fusing them is what delivers the
headline number** (AUROC 0.905, about 91% oracle recovery against roughly 67%
and 72% separately).

The original, now-superseded framing follows for the record.

`results/selective/selective_prediction.json`, arm `ensemble_mean`:

| score | Spearman | AUROC | oracle recovery |
|---|---:|---:|---:|
| physics residual | 0.610 | 0.871 | ~67% |
| `sigma_vel`, no physics | **0.654** | **0.894** | **~72%** |
| fused | 0.703 | 0.905 | ~91% |

The manuscript states all three AUROCs in one sentence (`body.tex:1246-1249`)
and never draws the conclusion. The introduction quotes the residual's 67% and
omits sigma's 72%. The abstract attributes "AUROC approx 0.9" to the residual.

Under verification now with a paired bootstrap. If it holds, the honest claim is
that physics earns its place only in FUSION, and the paper must say so.

## F7. Other substantive holes

- **rho is monotone in how BAD the model is.** Transolver 0.611, weak FNO 0.397,
  damaged FNO+DEQ 0.827, MeshGraphNet 0.851 at `mse_u` 13.5 with 101% drag
  error. The detector may be measuring model quality rather than error.
- **`results/control/mgn_density_control.json` carries `"verdict":
  "DENSITY-DRIVEN"`** (ratio 2.44, trained at 16k points, evaluated at 180k) and
  is **never mentioned in the manuscript**. That is a non-disclosure.
- **Forces.** rho_D = 0.84 is the integrator's ceiling on PERFECT fields, with
  332% median magnitude error, and it drops to 0.31-0.51 on predictions.
  rho_L = 0.998 is easy twice over, since angle of attack is an input channel
  and the integrator recovers circulation via Kutta-Joukowski from the outer
  ring. With the wall ring zeroed at Re about 1e6, viscous drag is
  **unavailable**, not merely inaccurate.
- **The lambda-sweep is circular.** `residuals.py:348` zeroes `bc[wall_ring]` on
  the exact cells where `residuals.py:200` had just boosted the penalty, so the
  boost is dead code and claim (A) rests on it. The proximity weight
  `exp(-|sdf| / 0.15c)` penalises speed near the body, which the true
  accelerated flow maximises. And `residuals.py:203-213` penalises deviation
  from freestream on the outer ring, which the uniform field satisfies by
  construction.
- **Two different Laplacian stencils in one pipeline.** `residuals.py:524` is
  wide, `operators.py:186` is compact. The certificate JSON claims it matches
  `physics_residual_torch`; it does not. The wide stencil annihilates period-2
  modes, so the training operator has a strictly larger kernel than the
  monitored one. That is a live alternative explanation for the W1 null.
- **Internal contradiction.** The deployed model sits BELOW the floor
  (`norm_pred_mean` 0.1136 against `norm_truth_mean` 0.192, on 80% of cases),
  which is the regime the theorem says DECOUPLES residual from error. Yet
  rho = 0.61 is measured there.
- **`norm_truth_continuity_mean` = 0.1226** contains no nu_t, no closure and no
  boundary term. Roughly half the floor may be pure rasterisation error in the
  labels, which admits an untested confound: geometry that rasterises badly may
  also be hard to predict. The partial-correlation control now running settles
  this.

## F8. VENUE: Computers and Fluids, not EAAI

The incumbent plan had EAAI next. **Reverse that.**

Both rejections were on novelty of method, at desk, without review. So the
discriminating question is whether a venue makes novelty or RIGOR the bar.

**Computers and Fluids** names "uncertainty quantification in fluid flow
simulations, reduced-order and surrogate models for fluid flows" in scope, says
machine-learning papers "are welcome, provided they show excellent scientific
character", which is a rigor bar, and explicitly asks authors to "discuss the
limitations of the method as well as its merits". That is the closest thing to
an invitation for a falsification available inside the no-APC constraint.

Decisive practical point: **it is single anonymized.** That dissolves the whole
anonymization problem. The arXiv preprint stays up, the GitHub and Zenodo links
stay, the system name stays, there is zero rework, and the existing elsarticle
build and Editorial Manager account are reused. Hybrid, free by the subscription
route. The abstract is already 249 words against a 250 cap. The cost is impact
factor 3.0 against JCP's 3.9.

Order: **Computers and Fluids, then EAAI (IF 9.0), then RESS (IF 13.7, but a
13,000-word cap against our 14,700), then JOCS or Engineering with Computers.**

- **EAAI honestly judged is a coin flip, not the safest move.** Its scope wants
  "novel aspects of AI used for a real-world engineering application". This is a
  benchmark methods paper with no deployed system, and "novel" is the word that
  killed it twice.
- **Physics of Fluids is now OUT.** Two 2026 editorials from the new co-editors
  establish a "Physics First" policy requiring the contribution to "advance
  physical understanding". A trust-calibration paper cannot meet that.
- **Negative-results venues change nothing.** That category is structurally gold
  open access, which the no-APC constraint forbids.
- **Sanctions are settled.** Elsevier's trade-sanctions page exempts authors at
  government-controlled academic institutions publishing in a personal capacity,
  and both CMAME and JCP processed the submission on merits.

Verification caveat: ScienceDirect returned 403 to automated fetches, so guide
rows came from search extraction. **Eyeball the guide manually before filing**,
especially any length limit.

Two decisions for the author, detailed in `venue_plan.md`:

1. Whether to disclose the two prior desk rejections in the cover letter. The
   recommendation is not to. It is not discoverable, there is no obligation, and
   it hands the editor a pre-authorised reason. Disclose the preprint and keep
   the positioning sentence.
2. The scope asks for "comparisons with traditional numerical reconstruction
   methods". `body.tex:1358-1368` concedes the 286x figure is "not a controlled
   speed-up measurement", and `body.tex:1636-1638` concedes the OpenFOAM and SU2
   backends are unimplemented. Pre-empt both in the cover letter.

## The paper this should become

Not "here is our trust layer". That claim is crowded and it lost twice.

**An audit of what a surrogate-side physics residual can actually do**, with the
operator-level boundary as the organising result:

1. The monitored residual does not vanish at the reference solution, and the
   floor **grows** under refinement, so it is not a discretisation artifact. The
   mechanism is operator provenance, not a missing closure term (F3).
2. It therefore cannot be **descended**: from the exact truth, descent cuts the
   residual 84% while driving field error from zero to a median of 0.91, in 24
   of 24 cases. The precise claim is about **iterate selection** — the
   error-optimal and residual-chosen iterates never coincide and the
   residual-chosen one is strictly worse, 24 of 24. Do NOT write "descending the
   residual makes the field worse", because from a perturbed start it cuts error
   in 18 of 24 cases (F1).
3. It also cannot **certify**, because the floor sits above the prediction.
4. It CAN **rank**, and better on drag (AUROC 0.952) than on field error
   (0.871). A physics-free baseline matches it within noise; fusion is what
   delivers the headline (F6).
5. Where the operator IS solver-consistent, residual correction works (Lei et
   al.), so the boundary is the operator, not the problem class (F4).
6. What remains deployable: the calibrated conformal layer on the corrected
   field, and the acceptance gate — **for its certificate, not its accuracy**.
   An ungated fixed half step improves true error on 95.8% of cases against the
   gate's 89.2%, so the accuracy gain is damping. Concede this.

### Corrections still to apply to the manuscript

- **`tab:iters` must report all four channels.** Over iterations 0 to 3,
  `mse_v` rises 96% and `mse_p` rises 41% while `mse_u` falls 27%. Reporting two
  of four channels is what made the figure look clean.
- **Disclose the MeshGraphNet density control.** `mgn_density_control.json` has
  zero manuscript mentions. That is a non-disclosure regardless of what it does
  or does not establish, and it measures MSE on 4 cases, so the stronger claim
  that rho = 0.851 is a density artifact is NOT established by it either.
- **Restate the boundary-weight sweep at the deployed resolution.** Its margin
  collapses with refinement and crosses zero by 512^2, so "for every weight" is
  false as written.
- **Drop theorem leg (iii).** It is an upper bound sold as a lower bound and it
  is numerically vacuous. Do not try to repair it.

### Citation verification, closed this session

Verified directly against Crossref and arXiv, not taken on an agent's word:
`garg2025dfuq`, `yu2026conformalpinn`, `song2026structureaware`,
`rigotti2026gist`, `scherz2026evaluation` all check out exactly.
`ma2024uqno` is confirmed as TMLR 2024, OpenReview `cGpegxy12T`, **and its bib
entry was missing its second author, David Pitt — now fixed.** The two new
must-cites `lei2026newtonkrylov` and `zhang2026phymgn` were added to `refs.bib`
with verified metadata. Do not re-run this check.

Genre precedent: McGreivy and Hakim, Nature Machine Intelligence 2024. A
rigorous critique paper in machine learning for CFD, highly cited, and very hard
to desk-reject for lack of novelty, because its contribution is a measurement
nobody had made.
