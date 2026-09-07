# The strongest paper reachable from these assets

Written 2026-09-07, after `FINDINGS.md`, `floor_resolution_study.md`,
`decisive_controls.md`, `novelty_hunt.md`, `r2_holes.md`, `venue_plan.md`.

This document does four things: proposes candidate paper designs, attacks each
one hostilely, recommends one, and gives a week-by-week plan. It also states the
ceiling honestly. Read §0 first — the evidence base has moved since the reframe
plan was written, and two of the briefing's premises are now out of date.

---

## 0. What is actually true today (the briefing is behind)

The task brief lists the residual-descent test and the floor study as "running".
**Both have landed, and both came back strong.** The brief also leads with the
closed-form `2·S·∇ν_t` result as the paper's strongest asset. **The measurement
demoted it.** Correcting these two things changes which paper to write.

| Asset | Status | The number |
|---|---|---|
| Floor at the truth | **Landed, solid** | 0.192 mean / 0.133 median, 200/200 AirfRANS; uniform freestream **exactly 0** |
| Floor under refinement | **Landed, decisive** | Does **not** decay. `p = −0.64 ± 0.29`, **0/16 cases decay**, 15/16 rise, ×2.56 over 128²→512² on a fixed physical region |
| Mechanism | **Landed — and it is NOT the closure term** | Convective and pressure terms are individually grid-converged (0.183→0.199, 0.180→0.181); the floor is their **non-cancelling remainder**, 25%→54% of the convective scale. Repairing the full stress divergence moves the floor by **<0.1%**. Omitted `∇ν_t·∇u` is **4.5%** of the floor at 512² |
| Artifact controls on the ladder | **Landed, four of them** | MMS on the monitor `p = 2.06`; MMS-raster pipeline null `p = +0.32` and 280× smaller; **C¹ CloughTocher re-rasterisation raises the floor** (kills the piecewise-linear-kink objection); reproduction of committed numbers to 3.3e-8 |
| Residual descent (the missing experiment) | **Landed, 24/24** | From the **exact ground truth**, no network in the loop, 300 steps cut the monitored residual 0.181→0.029 (84%) in **24/24** while field error goes 0 → median **0.91**. Residual-selected iterate strictly worse than error-optimal iterate in **24/24 on both arms** |
| Difficulty confound | **Landed, refuted** | Partial ρ 0.561 vs raw 0.610 — **92% survives**; drop CI includes zero; and the deployable residual **outranks the oracle difficulty variable** ‖r(truth)‖ (Δρ +0.094, CI [+0.055, +0.142]) |
| Physics-free baseline on **field error** | **Landed, tie** | σ−residual ΔAUROC **+0.023, CI [−0.035, +0.083]** — underpowered, not equivalent, not a loss |
| Physics-free baseline on **drag** | **Landed, and the residual WINS** | residual AUROC **0.952** on \|ΔC_d\| vs σ's 0.888; Δ +0.055…+0.088, CI excludes zero in **5/6 arms**; fusion stops helping |
| Acceptance gate | **Landed, conceded** | Ungated fixed half-step improves 95.8% vs the gate's 89.2%. **Damping does the accuracy work, not the residual test.** The gate uniquely buys the guarantee — which binds on **1.0%** of deployed cases |
| λ-sweep leg (A) | **Landed, weakened** | Resolution-contingent: margin +9.3% at 128² → **−2.8%** at 512². Crossover weight λ* median **≈600** against the framework's λ=1 |
| `tab:iters` | **Landed, conceded** | `mse_v` **+96%** and `mse_p` **+41%** over iters 0→3 while `mse_u` falls 27%; and over 3→15 residual and error move in the **same** direction |

**Three consequences for the design.**

1. **The `2·S·∇ν_t` closed form cannot be the headline.** It is real, it is
   non-vanishing, it grows under refinement exactly as predicted — and it is
   4.5% of the effect. Leading with it, and then having a reviewer read §6.2 of
   the resolution study in the repo, is a self-inflicted credibility wound. It
   becomes *the cleanest illustration of a non-vanishing component*, one
   paragraph, and the headline mechanism becomes **operator provenance / failed
   cancellation**, which is what was measured.
2. **The residual-descent result is far stronger than the abstract currently
   claims, and it is the paper's best negative.** Starting at the *exact ground
   truth* with no network anywhere, and having the objective walk you off it in
   24/24 cases, is a cleaner demonstration than anything in the manuscript. The
   draft under-claims here badly (§6).
3. **Control 4 is the paper's best positive, and it is currently absent from the
   manuscript entirely.** AUROC 0.952 on the engineering quantity, beating a
   physics-free baseline with CIs excluding zero, is the result that makes the
   paper constructive rather than a complaint.

---

## 1. The slogan the evidence supports (fix this before anything else)

`floor_resolution_study.md` §4 licenses this headline:

> *You cannot audit a surrogate with a residual operator different from the one
> that generated its training data.*

**That sentence is falsified by this paper's own control 4.** That exact
inconsistent monitor triages worst-decile drag error at AUROC 0.952, beats a
physics-free ensemble on it, and survives the difficulty confound at 92%. It
audits perfectly well. A reviewer who reads the abstract and then the drag table
finds the contradiction in thirty seconds. This is the single most likely
self-inflicted wound in the reframe.

**The correct statement, and it is sharper:**

> An operator-inconsistent residual monitor can **rank** predictions but cannot
> **certify** them and cannot be **descended**. The obstruction is a
> non-vanishing floor at the reference solution, and refinement makes it worse,
> not better.

Three verbs, three verdicts, each independently measured: rank (ρ 0.61, AUROC
0.87 field / 0.952 drag), certify (floor 0.192 ≫ prediction 0.113 — no threshold
implies a bound), descend (24/24 divergence from the truth). That is a complete,
non-contradictory thesis, and it is the spine of every design below.

---

## 2. Candidate designs

### Design A — The audit-operator ladder (the measured boundary)

**Thesis (one sentence).** What a physics residual can tell you about a neural
surrogate is determined not by the physics but by the provenance of the discrete
operator you can afford to evaluate, and the boundary between "ranks", "certifies"
and "can be minimised" can be located by measurement.

**Title candidates.**
- *Operator provenance sets the limit of residual-based auditing for neural CFD surrogates*
- *What a surrogate-side residual can and cannot certify: a measured boundary on steady RANS*
- *The reference solution fails its own physics check, and refinement makes it worse*

**Contributions, in priority order.**
1. **A measured law.** The floor at the reference solution does not decay under
   refinement — it grows 2.6× from 128² to 512², 0/16 cases decaying — because
   the reference's individually grid-converged terms fail to cancel under a
   different discrete operator. Four artifact controls (MMS order 2.06, MMS-raster
   null, C¹ re-rasterisation, exact reproduction) close the "it's your grid" and
   "it's your interpolant" objections.
2. **The consequence, measured, not argued.** Descending that residual from the
   exact truth destroys the field in 24/24 cases; the residual-selected iterate is
   strictly worse than the error-optimal one in 24/24 on both arms.
3. **The boundary, measured on both sides in-house** (this is the new work): the
   same fields, the same cases, evaluated under (i) the affordable Cartesian
   surrogate-side monitor and (ii) the generating solver's own discrete operator
   on its own mesh. Floor ≈ 0 on one side, 0.19 on the other; descent converges on
   one side, diverges on the other. Redraws the boundary Lei et al. (arXiv:2608.04400)
   would otherwise be used to falsify.
4. **A cost ladder.** What each rung of operator fidelity costs, in wall-clock,
   against what it buys in floor magnitude and ranking quality. Turns the boundary
   from a label into a design rule.
5. **What survives as deployable**: case-level ranking (free, 1.13 ms), the drag
   triage result, the conformal layer, the gate's guarantee stated at its true size.

**Section structure.**
1. Intro — the practice (surrogate-side residual monitors and PINN residual losses
   are being built on right now), the question, the finding.
2. The monitor and its verification (MMS `p = 2.06`) — *code verification passes*.
3. The floor at the reference solution, and the refinement ladder — *solution
   verification fails, for the reference*.
4. Mechanism: failed cancellation, not closure omission; `∇ν_t` as illustration.
5. The three verbs: rank / certify / descend, each with its experiment.
6. The other side of the boundary: the solver-consistent operator, measured.
7. The cost ladder and the design rule.
8. What remains deployable (trust layer, drag triage, gate) — compressed.
9. Limitations, stated as scope.

**Carries over.** The floor measurement, the whole resolution ladder and its four
controls, the descent test, controls 1–6, the MMS gate, the λ-sweep and λ*≈600,
the three-backbone spread as evidence the phenomenon is not backbone-specific,
the cost table, the artifact manifest.

**Required new work.**
- **A1. An owned OpenFOAM corpus** (~30–50 2-D airfoil cases at AirfRANS-like
  conditions, own mesh, own schemes). *Compute trivial* (cold solves 88–239 s per
  paper-2 measurements); *plumbing is the cost*.
- **A2. A residual evaluator on the solver's own operator.** Write out residual
  fields explicitly. **Do not use OpenFOAM's logged initial residuals as a norm** —
  they carry a per-equation normalisation that is not comparable across operators
  or cases. Define and document your own norm.
- **A3. The ladder repeated on the owned corpus**, where `s_local` is known and
  controllable. This is what kills the ladder's one real weakness (see §3).
- **A4. Descent on the consistent operator** (SIMPLE/Newton iterations from the
  mapped prediction) — largely already exists in `openfoam_warm_start.py`.
- **A5. The cost ladder**: time each rung (Cartesian 128², Cartesian 512²,
  body-fitted raster, mesh-native FV assembly, full solver residual).

**Cost.** ~2.5 weeks, plumbing-dominated. Two of the five items are ports, not builds.

---

### Design B — The functional audit (the constructive turn)

**Thesis.** The pointwise residual norm is uncertifiable because of a floor that
refinement worsens; but the residual *weighted for a specific engineering
functional* may escape that floor, and if it does, the same free monitor that
cannot certify a field can certify a force.

**Title candidate.** *A free, goal-oriented certificate for neural CFD: the
residual cannot certify the field but may certify the force*

**Why this is the strongest possible version of the paper.** It converts the
negative into a method, it targets the quantity engineers actually use, it has a
respected classical anchor (Becker–Rannacher DWR — already cited; Giles & Pierce;
Venditti & Darmofal), and it explains control 4's otherwise-surprising AUROC 0.952
mechanistically rather than reporting it as a curiosity.

**Why it may not exist. Read this before investing.** The tempting argument is:
by the divergence theorem, `∫_V R(w) dV` equals a boundary momentum flux, so it
bounds force error, and its signed cancellation escapes the floor. **That argument
is wrong as stated.** The identity says that the volume integral of *one field's*
residual equals the discrepancy between two ways of computing the force *from that
same field*. It is a statement about a single field's internal inconsistency, not
about error against the truth. Bounding `|C_d(w) − C_d(u*)|` requires
`δJ ≈ −(ψ, R(w))` with `ψ` the drag adjoint — a nontrivial field costing roughly
one solve. Constant weighting over a control volume is the *zeroth-order* adjoint,
exact only in the continuum with exactly-satisfied conservation. And "the floor
cancels under integration" is not plausible-by-default: the floor study already
found the residual **is** the non-cancelling remainder of grid-converged terms.

**So Design B is gated on an experiment, not assumed.** See §4.

**Contributions if the gate passes.**
1. Everything in Design A, compressed to about half its length.
2. A functional-level audit: the floor's inversion (prediction scoring better than
   truth) disappears under the functional, so a threshold on the functional
   residual *does* imply a force-error bound.
3. A conformalised force-error certificate at zero marginal cost.
4. The classical framing: this is goal-oriented error estimation, imported into
   surrogate auditing, with the new content being the floor/functional interaction.

**Required new work.** The §4 gate first; then, if it passes, calibration and the
certificate. Plus the Design A infrastructure, because the certificate must be
validated against a solver-consistent reference.

**Cost.** Gate: 2–3 days. Full build if it passes: +1 week on top of Design A.

---

### Design C — Verification and validation for neural surrogates

**Thesis.** Map neural CFD surrogates onto the CFD community's formal V&V
machinery — code verification, solution verification, validation — and show what
each stage means when the "code" is a network.

**Verdict: kill as a standalone design.** It has no measurement. Computers & Fluids
sets a rigor bar; a mapping paper with no new number reads as a position piece and
is desk-rejectable on exactly the axis that has already killed this paper twice
("no new computational methodology"). A hostile reviewer's first sentence writes
itself: *what did you measure?*

**But keep the vocabulary.** V&V is the right organising language for Designs A/B,
and it lands *hard* on this evidence, in one subsection:

> Code verification of the monitor **passes** — method of manufactured solutions
> gives observed order 2.06. Solution verification of the *reference data under
> that monitor* **fails**, and fails worse under refinement. The surrogate is
> being audited by a verified operator against a field that is not a solution of it.

That is a precise, quotable, discipline-native statement of the whole finding. Use
it as the framing sentence of §3 in Design A. It is worth about a paragraph, not a
paper.

---

### Design D — A + B composite (the recommendation, conditional on the gate)

**Thesis.** *An operator-inconsistent residual monitor can rank a surrogate's
predictions but cannot certify or correct them; the obstruction is a floor that
refinement worsens; and the way out is not a better grid but either the generating
solver's own operator (which we cost) or a goal-oriented functional (which we
build).*

**Structure.** Design A's spine, with Design B slotted in as §7 in place of the
cost ladder's final rung, and the deployable material compressed to one section.

**Word budget.** Target **≤13,000 words**. This is not cosmetic: it keeps RESS
(IF 13.7) live as fallback #3 without a second rewrite, and it forces the cuts the
paper needs anyway. The current 14,700 words are dominated by material the
reframe demotes.

---

## 3. Hostile attack on each design

### Attacks on Design A

**A-ATTACK-1 (most dangerous). "Your floor rises because at your finest rung you
are differentiating a piecewise reconstruction, not sampling a solution."**
At 512², `h/s_local = 2.01`. The report itself concedes 1024² would not clear the
exclusion. A numerical analyst on a C&F board goes straight here.
- *Survivable?* **Yes, but only with A3.** Current defences: the 128²→256² leg
  alone shows the rise at `h/s_local` 8.1→4.0, comfortably clear; the MMS-raster
  null; the C¹ control that *raises* the floor. These are good but defensive.
- *What neutralises it:* the owned OpenFOAM corpus. You control the source mesh,
  so `s_local` is known everywhere, you can generate a *finer* source than any
  raster rung, and you can evaluate the identical field under the solver's own
  operator as an absolute reference. The attack disappears rather than being
  parried. **This alone justifies A1–A3.**

**A-ATTACK-2. "This is the local truncation error. Brandt's τ, Stetter's defect
correction, and Morin–Nochetto–Siebert's data oscillation own it. You have
rediscovered a 1977 result and measured it."**
- *Survivable?* **Yes**, and cheaply, but only if conceded in the first person and
  in the intro, not the appendix. The distinction is real and must be stated
  crisply: classical τ is *the same equation family on nested grids* — a
  discretisation-**order** gap that shrinks predictably under refinement. What is
  measured here is a **cross-provenance** mismatch that does **not** shrink, and
  0/16 cases decaying is precisely the evidence that it is not τ. One paragraph
  converts this from a scoop into a citation. Failing to write it is fatal.

**A-ATTACK-3. "Nobody audits with a Cartesian finite-difference monitor. You built
a straw man."**
- *Survivable?* **Yes**, but it must be pre-empted with citations, not assertion.
  Gopakumar et al. (arXiv:2502.04406) use convolutional layers as finite-difference
  stencils and the residual as a conformal nonconformity score. ANCHOR
  (arXiv:2512.19643) uses a residual-derived error signal to trigger a solver.
  Zhang et al. (arXiv:2602.14918, App. E) hit the floor and had to work around it.
  Every PINN residual loss is this monitor. Name three or four in the second
  paragraph of the intro; the straw-man charge dies there.

**A-ATTACK-4. "You cite Lei et al. for the good side of your boundary. Your
boundary is therefore half hearsay."**
- *Survivable only with A2/A4.* Measuring both sides on the same cases with the
  same predictions is what makes it a boundary rather than a juxtaposition. Without
  it, a reviewer is entitled to say the two regimes differ in a dozen ways besides
  operator consistency.

**A-ATTACK-5. "Your slogan says the monitor cannot audit; your Table N says it
triages drag at 0.952."**
- *Fatal if left in, free to fix.* See §1. Rank / certify / descend.

**A-ATTACK-6. "The floor is at least partly your rasteriser, and rasterisation is
your choice."**
- *Partly conceded, and it must be.* The honest position: yes, and that is the
  point — a deployed surrogate operating on a Cartesian grid *has* to rasterise,
  so the mismatch is a property of the deployment, not an avoidable mistake. The
  C¹ control and the MMS-raster null show the rasteriser does not manufacture the
  rise. The owned corpus quantifies the remaining part.

**Verdict on A: survives.** It is a solid, defensible specialist paper. Its
ceiling is "important measurement, no method".

### Attacks on Design B

**B-ATTACK-1 (potentially fatal). "Goal-oriented error estimation is twenty-five
years old. Adjoint-weighted residuals, DWR, Venditti–Darmofal output-based
adaptation. You have applied a standard technique."**
- *Survivable only if the new content is the floor/functional interaction*, i.e.
  the quantitative statement that the floor which destroys the norm-level audit
  does or does not survive the functional. That is new because the floor result is
  new. Frame it as: *goal-oriented estimation is the classical answer; what was
  unknown is whether it survives an operator-provenance floor, and here is the
  measurement.* If the gate returns "the floor does not cancel", there is no new
  content and this attack lands.

**B-ATTACK-2. "Your zeroth-order adjoint is an approximation of unmeasured
quality."** True. Must be stated, and the cost of the true adjoint (≈ one solve)
must be reported so the reader can see the trade. Survivable if honest.

**B-ATTACK-3 (serious, and it is the repo's own record). "Your force integrator
has 332% median magnitude error on *perfect* fields and ρ_D = 0.84 as a ceiling.
Any drag claim on this pipeline is suspect."**
- *Survivable only with scoping and, ideally, the owned corpus.* Control 4 is
  already constructed correctly (both sides through the same integrator, so the
  target is model error not integrator bias) and this must be stated in the caption,
  not the appendix. Additionally: **viscous drag is unavailable at 128², not
  approximate** — say so flatly and scope the claim to pressure/momentum-flux drag.
  The owned OpenFOAM corpus, where the wall is resolved, is the clean fix and is
  already being built for Design A.

**B-ATTACK-4. "Your control volume sits 1–1.5 chords from a lifting airfoil, where
induced velocity is O(few %) — exactly the scale of the effect you are
integrating."** Real, and the AirfRANS crop `(-1,2,-1.5,1.5)` makes it unavoidable
on the existing data. The owned corpus with a proper far field fixes it. Flag as a
known limitation of the AirfRANS arm.

**B-ATTACK-5. "AUROC 0.952 has 20 positives."** Control 4 already carries 3 seeds
and paired bootstrap CIs. Report them.

**Verdict on B: conditionally alive.** It is the highest-ceiling content available
and it is not yet known to exist. **Do not commit a month to it before §4's gate.**

### Attacks on Design C

**C-ATTACK-1. "Where is the measurement?"** No answer. **Killed as standalone.**

**C-ATTACK-2.** ASME V&V 20 is about validation uncertainty against *experiment*;
neural surrogates validated against simulation data do not map cleanly, and a
forced mapping reads as a survey. Confirms the kill.

---

## 4. The gate that decides between A and D (run this first — 2–3 days)

Everything in Design B hangs on one measurable question, and it is cheap because
the per-case residual fields already exist.

**Definition.** For a control volume `V` (a family of nested boxes around the
airfoil, plus the body surface term handled explicitly on masked cells), define
the drag-direction functional residual

`I(w; V) = e_D · ∫_V R(w) dV`

evaluated for the truth `u*`, for the raw prediction, and for the corrected field.

**Pre-register these three tests and both branches, before running.**

**(a) Does the floor cancel?** The norm-level pathology is the **inversion**: the
prediction's residual sits *below* the truth's in 160/200 cases, so no threshold
on `‖R‖` can imply an error bound. The direct functional analogue is the inversion
rate under `|I(·;V)|`.
- *Pass:* inversion rate falls below ~10% at some `V`, i.e. the truth reliably
  scores better than the prediction.
- *Fail:* inversion rate stays near 80%. The floor does not cancel; there is no
  certificate.

**(b) Does it rank?** Spearman and AUROC of `|I(w;V)|` against per-case `|ΔC_d|`,
against the existing baseline of `‖R‖` at **AUROC 0.952**. Paired bootstrap.
- *Pass:* within CI of 0.952 or better.
- *Fail:* materially worse — then the norm is already the better score and the
  functional adds nothing.

**(c) Does it calibrate?** Is `|ΔC_d| ≲ c·|I(w;V)|` with a conformalised `c`
holding target coverage on held-out cases? This is the actual certificate, and it
is the only one of the three that produces a deployable object.

**Decision rule.**
- **(a) and (b) pass → Design D.** The constructive headline is real. Build (c).
- **(b) passes, (a) fails → Design A**, with one subsection reporting the functional
  as a *better ranker but still not a certificate*. Honest, and it explains
  control 4.
- **Both fail → Design A only.** Control 4 stays as a measured, unexplained
  positive, which is fine — it is a result either way.

**Plumbing risks to budget for:** the immersed body's surface term on masked cells;
the zeroed wall ring (0.51% of fluid cells) sitting exactly where the surface term
lives; sign conventions between the CV formulation and the repo's existing
`design_force_integrator.py`. Budget three days, not one.

---

## 5. Recommendation

**Run the §4 gate first. Then build Design D if it passes, Design A if it does
not.** Either way, do §1's slogan fix and §0's demotion of the `∇ν_t` closed form
immediately, because they are free and they are currently load-bearing errors.

**Reasoning.**

1. **Design A is the floor of the outcome distribution and it is already almost
   paid for.** The measured law, its four artifact controls, and the 24/24 descent
   result are landed. The new work (A1–A5) is plumbing on infrastructure that
   already exists on the `paper2/openfoam-warm-start` branch. A month is enough.
2. **Design A alone has a real ceiling: it is a measurement paper with no method.**
   That is publishable at C&F — the scope text sets a rigor bar and explicitly asks
   for limitations — but it will not excite anyone.
3. **Design B is the only path to a paper someone cites for what it lets them do
   rather than what it stops them doing**, and its gate costs three days against a
   month of commitment. That ratio is the whole argument for running it first.
4. **Design C is a framing, not a design.** Absorb the vocabulary, discard the paper.
5. **Venue stays Computers & Fluids.** `venue_plan.md`'s reasoning holds and the
   reframe strengthens it: single-anonymised (no rework, preprint stays up), no
   length limit, scope names UQ and surrogate models, ML bar is "excellent
   scientific character" (rigor, not novelty), and the guide explicitly asks
   authors to discuss limitations. Design D over-satisfies that guide line by line.
   Cutting to ≤13,000 words additionally keeps RESS live as fallback #3.
6. **One title rule.** No coined system name in the title. That is what invited
   "no new computational methodology" twice. `neuroforge-cfd` belongs in the code
   availability section only.

---

## 6. Part 3 — where confidence is earned, and where it is a liability

### 6a. State these FLATLY. The evidence carries them.

Assertive language here is not overreach; hedging them is the defect.

1. **The monitored residual does not vanish at the reference solution.** 0.192
   mean / 0.133 median on **200/200** AirfRANS cases, model-independent. The
   physically wrong uniform freestream scores **exactly zero**, at every grid level.
2. **The floor does not decay under refinement.** `p = −0.64 ± 0.29`, **0 of 16
   cases decay**, 15/16 rise, ×2.56 from 128² to 512² on a region fixed in physical
   units. Two independent studies (24 cases × 3 levels; 16 cases × 5 rungs), three
   exclusion bands, same verdict.
3. **The monitor itself is verified.** Method of manufactured solutions gives
   observed order **2.06**. *This is a rigor asset the draft never sells.* Say it
   in the same breath as (2): the operator is correct; the reference field is not
   a solution of it.
4. **The rise is not a rasterisation artifact.** A manufactured analytic solution
   through the identical pipeline converges at order **2.02 ± 0.05** and stays 280×
   smaller; a **C¹** re-rasterisation *raises* the floor rather than lowering it.
5. **The mechanism is failed cancellation under a foreign operator, not a missing
   closure term.** Convective and pressure terms are individually grid-converged;
   repairing the full stress divergence moves the floor by **<0.1%**.
6. **Descending the monitored residual from the exact ground truth destroys the
   field.** 300 steps, no network in the loop, residual 0.181→0.029 (−84%) in
   **24/24 cases**, field error 0 → median **0.91**. The compact-stencil monitor
   falls in 24/24 too, so this is not the differentiable twin being gamed.
7. **The residual is not a valid iterate selector.** Error-optimal and
   residual-chosen iterates never coincide — **24/24 on both arms** — and the
   residual-chosen iterate is strictly worse in **24/24**.
8. **The objective ranks a wrong field above the truth.** Descent terminates at
   residual 0.028 with field error 1.31, while the exact truth sits at 0.110.
9. **The trust correlation is not a case-difficulty artifact.** Partial ρ 0.561
   against raw 0.610 — 92% survives; and the deployable residual **significantly
   outranks the oracle difficulty variable** ‖r(truth)‖ (Δρ +0.094, CI [+0.055,
   +0.142]) on all four arms and both metrics. *A pure difficulty proxy cannot do
   that.* This is the cleanest defensive result in the paper and it is currently
   invisible.
10. **On the engineering quantity the residual is decisively the better score.**
    AUROC **0.952** on worst-decile |ΔC_d| (vs 0.871 on field error), beating the
    physics-free ensemble σ by +0.055…+0.088 with CIs excluding zero in 5/6 arms,
    with rank fusion no longer helping.
11. **The audit is free.** 1.13 ms against ~10 s inference and a solver run.
12. **The gate's guarantee holds by construction and is never violated**, while an
    ungated fixed half-step violates it on 1.0% of deployed backbone cases and 8.8%
    on the ensemble path.
13. **λ\* ≈ 600.** Any no-slip weight that rescues the ordering is two orders of
    magnitude above the framework's λ = 1 — a monitor in which the boundary term
    outweighs the entire PDE residual, i.e. no longer a PDE monitor.

### 6b. These must stay scoped no matter how they are worded

Each of these is a place where confident phrasing is a *liability* — a reviewer
holding the repo can falsify it.

1. **Not "the residual cannot be minimised."** It can — it falls 84%. The true
   claim is that minimising it does not take you to the truth. Also: **not
   "descent always makes things worse"** — from a perturbed start it *improves*
   error in 18/24 cases, typically by 60%. Use the iterate-selection statement,
   which holds 24/24.
2. **Not "the floor grows."** Use **"does not decay, and grows by 2.6× over the
   measured range."** Past `h ≈ s_local` the ladder cannot be pushed, and the
   report's own honest bound says so.
3. **Not "the physics residual is the better trust signal."** On field error the
   paired CI is [−0.035, +0.083] — underpowered, resolving nothing in either
   direction. Say indistinguishable-and-underpowered. Do **not** claim equivalence
   either. (The abstract's current "AUROC ≈ 0.9" attributed to the residual is
   the single sharpest overclaim in the paper: the residual is 0.871.)
4. **Not "the gate improves accuracy."** Conceded: an ungated fixed half-step
   improves 95.8% vs the gate's 89.2%. Damping does the work. Claim the gate for
   the guarantee, and state that the guarantee binds on **1.0%** of deployed cases.
   A guarantee active on 1% of cases does not justify "certified self-correction"
   as a *framing*, only as a *statement*.
5. **Not "it tells you where."** Per-cell ρ = 0.166 ± 0.155; 16×16 patch 0.323.
   The slogan is **which case**, never *where*, never *how*. Also fix the
   mislabelled 0.22 ± 0.06 (superseded FNO checkpoint, n=15) quoted under the
   deployed field's name.
6. **Theorem leg (iii) must be withdrawn or restated.** As written it is an *upper*
   bound sold as a floor, `σ_min` is never computed and is not separable from zero
   on a 47000×65500 Jacobian, and it identifies the min-norm point rather than
   where descent actually lands. Leg (iv)'s cone constant is also wrong
   (`α < ‖P_rangeL r*‖`, not `α < ‖r*‖`). Legs (i)–(ii) are correct and elementary
   — keep them, labelled as elementary.
7. **λ-sweep leg (A) is resolution-contingent.** Margin +9.3% at 128² to **−2.8%**
   at 512²; holds at all five rungs in only 11/16 cases. Scope to the deployed
   resolution and lean on leg (B), which does not depend on the ordering.
8. **Conformal coverage is not evidence.** Split conformal attains marginal
   coverage by construction. The guarantee held is marginal over a random cell of a
   random case, pooled over ~1.6M dependent cells with effective n ≈ 100 — it is
   **not** the per-field guarantee the trust layer is sold for. Lead with ECE
   (0.074 with the deep ensemble) and adaptivity, not with coverage.
9. **Not "backbone-robust."** ρ 0.40–0.85 is positivity, not robustness, and it is
   monotone in how *bad* the model is. Report MeshGraphNet's absolute error
   (`mse_u` 13.5, 101% drag error) and **disclose the density control**
   (16k train / 180k eval, 2.44× MSE inflation, n=4) — currently zero mentions in
   the manuscript, which is a non-disclosure a repo-reading reviewer will punish.
10. **Wall quantities and viscous drag are unavailable, not approximate.** At
    Δ = 0.0234c and Re ≈ 2×10⁶ the first cell centre sits at y⁺ ≈ 10³ and the wall
    ring is zeroed. State this as a scope statement about what the monitor can
    certify, not as a blemish.
11. **`tab:iters` must report all channels.** `mse_v` +96% and `mse_p` +41% over
    iters 0→3 while `mse_u` falls 27%; and over 3→15 residual and error move in the
    *same* direction. Drop "opposite directions"; keep "no monotone relation holds".
12. **2-D, steady, one turbulence model (Spalart–Allmaras), two datasets.** Concede
    without argument.

### 6c. Where the CURRENT draft under-claims (under-claiming is also a defect)

1. **The abstract hedges "reducing it does not reduce field error"** on a sweep
   that never reduced it. It can now say something far stronger *and* true:
   descent from the exact truth, with no network in the loop, destroys the field in
   24/24 cases. Replace the weakest sentence with the strongest evidence.
2. **The floor is buried in an appended theorem** and framed as an assumption
   (H2). It is a measured law with a five-rung refinement ladder and four artifact
   controls. It belongs in the abstract's first two sentences.
3. **Control 4 does not appear at all.** AUROC 0.952 on drag, beating a
   physics-free baseline with CIs excluding zero, is the paper's best positive.
4. **Control 2 does not appear at all.** "The deployable residual outranks the
   oracle difficulty variable" pre-empts the confound objection a reviewer would
   otherwise raise first.
5. **The MMS verification (p = 2.06) is not sold.** It is the sentence that
   separates this from "your stencil is broken".
6. **The λ-sweep closed form is invisible**, and λ* ≈ 600 quantifies it.
7. **The cost asymmetry is under-used.** σ alone costs the same 4.86× as fusion, so
   the residual is the only zero-marginal-cost score. That is the correct framing
   of control 1, and it turns a tie into a practical win.

---

## 7. Part 4 — what a practising engineer can DO with this

A result is practical when it changes a decision. Three decisions, in order of how
much money they save.

**Decision 1 — "Should I train my surrogate against a physics residual loss, or
post-correct by residual descent?"**
*Answer: no, if your residual operator is not the one that produced your training
data — and it almost never is, because public CFD datasets ship fields, not
operators.* The evidence: descent from the exact truth walks off it in 24/24;
refinement makes the floor worse, so "use a finer grid" is not a remedy.
*What the paper must show:* the descent experiment starting from ground truth (done),
the refinement ladder (done), and the mechanism (done).

**Decision 2 — "Which of my 10,000 surrogate design evaluations do I send back to
the solver?"**
*Answer: rank by the residual — it is free (1.13 ms), it is not a difficulty proxy,
and on drag it is the best score available.* AUROC 0.952 on worst-decile drag
error; rejecting the least-trusted 10% recovers 77–81% of the oracle's achievable
reduction. *What the paper must show:* control 4 with its seeds and CIs, the
partial-correlation control, and the cost table — plus a clear statement that this
is a **ranking**, with no error bound attached.

**Decision 3 — "How much operator do I have to buy before my audit means
anything?"**
*Answer: the cost ladder.* This is the deliverable that makes the paper an
engineering artifact rather than a warning. Each rung: what it costs (wall-clock,
per case), what floor it leaves, and what it can then support (rank / certify /
descend). The likely shape — and it must be measured, not assumed — is that
nothing between "Cartesian FD monitor" and "the solver's own operator on its own
mesh" buys enough, in which case the rule is *one solver residual evaluation
(≈ one SIMPLE iteration) is the cheapest audit that can certify*, which is still
three orders below a full solve. **That is a genuinely useful number and nobody
has published it.**

**A fourth, aimed at the community rather than an engineer, and worth a
recommendation paragraph in the conclusion:**
*Public CFD datasets must ship their discrete operator — mesh, schemes, and a
residual evaluator — or surrogates trained on them cannot be audited against them.*
This follows directly from the measurement (the floor is provenance, not
resolution), it is concrete, it is actionable by AirfRANS/DrivAerML/BLASTNet
maintainers, and it is the kind of claim that gets a critique paper cited.

---

## 8. The paper-2 collision — an explicit decision for the author

Designs A and D both run on the `paper2/openfoam-warm-start` infrastructure. Paper 2
is drafted, unsubmitted, on an unmerged branch, and shares: the OpenFOAM corpus
concept, the prediction→mesh mapping pipeline, and — critically — the *lesson* that
representation/operator provenance is what decides whether the coupling works
(`nf_mesh` resampled reads −58.8% against +33.9%).

**This is a live salami exposure** and it is not the reviewer's job to sort out.
Three branches:

| Branch | What paper 1 owns | What paper 2 owns | Risk |
|---|---|---|---|
| **(i) Sequence, paper 2 first** | The audit-operator boundary; cites paper 2 for the mapping pipeline | Warm-start speedup, the representation criterion | Low, but paper 1 slips by however long paper 2 takes |
| **(ii) Sequence, paper 1 first** (recommended) | The boundary, using OpenFOAM as a *measurement instrument* | Warm-start speedup as a *performance* result | Moderate — paper 2 must then be explicit about what it adds |
| **(iii) Merge** | One paper: "the operator is the boundary — auditing and correction both live or die on provenance" | — | Kills a publication; produces a 20k-word monster; but is the most intellectually honest and the most impactful single artifact |

**Recommendation: (ii), with a hard rule.** Paper 1 uses the solver operator only
as an *instrument* (residual evaluation, floor measurement, cost) and makes **no
speedup claim**. Paper 2 keeps the entire warm-start performance story. Cite paper
2's preprint from paper 1 explicitly for the pipeline, and state the split in one
sentence in each. If the author would rather have one strong paper than two
adequate ones, (iii) is defensible and should be decided **now**, not after the
corpus is built.

---

## 9. Week-by-week plan (four weeks, single workstation + WSL2 OpenFOAM)

Plumbing-dominated by design. Compute is not the constraint here; mapping,
conventions and masks are.

### Days 1–3 — free fixes and the gate
- Fix the slogan everywhere (§1): rank / certify / descend. **Do this first**; it
  changes what every later sentence must say.
- Demote the `∇ν_t` closed form to "cleanest illustration of a non-vanishing
  component, 4.5% of the floor"; promote failed cancellation to the mechanism.
- Withdraw/restate theorem leg (iii); fix leg (iv)'s cone constant.
- **Run the §4 gate.** Pre-register (a)/(b)/(c) and both branches in a commit
  *before* running, exactly as the ladder's decision rule was committed in `bf5106c`.
- **Decision point: Design A or Design D.**

### Week 1 — the owned corpus (the exposure-killer)
- A1: generate 30–50 2-D airfoil OpenFOAM cases at AirfRANS-like conditions, own
  mesh, own schemes. Compute is hours; setup is the week.
- A2: residual evaluator on the solver's own operator, writing residual **fields**.
  Define and document the norm; **do not** reuse OpenFOAM's normalised log
  residuals as a norm.
- Gate: reproduce the converged solution's residual under its own operator at
  solver tolerance. If that number is not ≈0, something is wrong with A2, not with
  the thesis.

### Week 2 — the two-operator measurement
- A3: repeat the refinement ladder on the owned corpus, where `s_local` is known.
  **This is the single most valuable new number in the plan** — it converts
  A-ATTACK-1 from a parry into a non-issue.
- Same fields, both operators: floor, ranking quality, inversion rate.
- A4: descent on the consistent operator from the mapped prediction (port from
  `openfoam_warm_start.py`), against the existing 24/24 divergence on the
  inconsistent one.
- A5: time every rung of the cost ladder.

### Week 3 — the constructive half, and freeze
- If gate passed: build the functional certificate (conformalise `c`, held-out
  coverage), validated against the solver-consistent reference.
- If gate failed: expand the cost ladder's intermediate rungs (body-fitted raster,
  mesh-native FV assembly) so the design rule is a curve, not two points.
- **Freeze all tables and figures by end of week 3.** No number may move after this.

### Week 4 — write
- Rewrite from the new skeleton. Do not edit the old draft; the section order has
  changed and editing will leave orphaned claims.
- Cut to **≤13,000 words**. The trust layer, conformal layer, multi-backbone and
  DeepCFD material compress into one "what remains deployable" section.
- Apply every item in §6a/6b/6c. Add the missing citations (Lei; Zhang App. E;
  Astral; ENS; Luo & Zhou; Wang NeurIPS 2022; McGreivy & Hakim; Brandt; Stetter;
  Bochev & Gunzburger; Morin–Nochetto–Siebert).
- Delete `tab:positioning`; insert the two-row regime table **with a cost column**.
- New title, no system name. Cover letter per `venue_plan.md` §8. arXiv v4 **after**
  the rewrite.

**If a week slips**, cut in this order: (1) the cost ladder's intermediate rungs,
(2) the functional certificate's conformalisation (keep the ranking result),
(3) corpus size 50→30. **Never cut A3** — it is the exposure-killer.

---

## 10. The ceiling, stated without flattery

**With the §4 gate failing, this is a solid specialist paper.** A well-controlled
measurement that constrains a practice the field is currently building on, plus a
free ranking signal with a documented scope. Computers & Fluids is the right home,
the rigor bar is met comfortably, and it will be cited by people who were about to
add a residual loss and did not. It is not a landmark and no amount of writing will
make it one.

**With the gate passing, it is a strong specialist paper** — a measured law plus a
constructive, goal-oriented audit with a design rule and a certificate on the
quantity engineers use. That is genuinely good work and it would survive review at
a considerably better venue than C&F. It is still not a landmark.

**A landmark would additionally require, and none of these is reachable in a
month:**
- **3-D.** Every claim here is 2-D. The operator-provenance argument should hold in
  3-D and the cost ladder would change shape, but "should" is not a measurement.
- **A dataset that ships its operator**, so the floor can be driven to zero by
  construction and the boundary crossed *continuously* rather than in two discrete
  points. This is the experiment that would turn the boundary into a curve.
- **A cheap consistent monitor that actually works** — i.e. Design B succeeding not
  just for a functional but for the field. Nothing in the current evidence suggests
  that exists, and the refinement ladder is mild evidence that it does not.
- **A second turbulence closure and a second solver**, to show the effect is about
  provenance rather than about Spalart–Allmaras or OpenFOAM specifically.

**The honest summary for the author:** the assets support a paper that is
*correct, well-controlled, and useful*, whose contribution is a measurement and a
design rule rather than a new method. That is exactly the genre Computers & Fluids'
scope text invites, and it is a genre with real precedent (McGreivy & Hakim, Nature
MI 2024). Aim there, execute the §4 gate honestly, and stop trying to make the
paper sound like a method paper — that framing has now been rejected twice, and it
was the framing, not the science, that failed.
