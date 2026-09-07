# The strongest paper reachable from these assets

Written 2026-09-07, after `FINDINGS.md`, `floor_resolution_study.md`,
`decisive_controls.md`, `novelty_hunt.md`, `r2_holes.md`, `venue_plan.md`.

This document proposes candidate paper designs, attacks each one hostilely,
recommends one, separates the claims that can be stated flatly from those that
cannot, and gives a week-by-week plan. It also states the ceiling honestly.

Read §0 first — the evidence base has moved since the reframe plan was written,
and two of the briefing's premises are now out of date. Read §3.5 before
investing in Design B; a prior-art line that `novelty_hunt.md` never searched
turns out to matter, in both directions.

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
| Artifact controls on the ladder | **Landed, four of them** | MMS on the monitor `p = 2.06`; MMS-raster pipeline null `p = +0.32`, 280× smaller; **C¹ CloughTocher re-rasterisation raises the floor** (kills the piecewise-linear-kink objection); reproduction of committed numbers to 3.3e-8 |
| Residual descent (the missing experiment) | **Landed, 24/24** | From the **exact ground truth**, no network in the loop, 300 steps cut the monitored residual 0.181→0.029 (84%) in **24/24** while field error goes 0 → median **0.91**. Residual-selected iterate strictly worse than error-optimal iterate in **24/24 on both arms** |
| Difficulty confound | **Landed, refuted** | Partial ρ 0.561 vs raw 0.610 — **92% survives**; drop CI includes zero; and the deployable residual **outranks the oracle difficulty variable** ‖r(truth)‖ (Δρ +0.094, CI [+0.055, +0.142]) |
| Physics-free baseline on **field error** | **Landed, tie** | σ−residual ΔAUROC **+0.023, CI [−0.035, +0.083]** — underpowered, not equivalent, not a loss |
| Physics-free baseline on **drag** | **Landed, the residual WINS** | residual AUROC **0.952** on \|ΔC_d\| (vs_gt_nf) against σ's 0.888; Δ +0.055…+0.088 with CIs excluding zero; fusion stops helping |
| Acceptance gate | **Landed, conceded** | Ungated fixed half-step improves 95.8% vs the gate's 89.2%. **Damping does the accuracy work, not the residual test.** The gate uniquely buys the guarantee — which binds on **1.0%** of deployed cases |
| λ-sweep leg (A) | **Landed, weakened** | Resolution-contingent: margin +9.3% at 128² → **−2.8%** at 512². Crossover weight λ* median **≈600** against the framework's λ=1 |
| `tab:iters` | **Landed, conceded** | `mse_v` **+96%** and `mse_p` **+41%** over iters 0→3 while `mse_u` falls 27%; over 3→15 residual and error move in the **same** direction |

**Three consequences for the design.**

1. **The `2·S·∇ν_t` closed form cannot be the headline.** It is real, non-vanishing,
   and grows under refinement exactly as predicted — and it is 4.5% of the effect.
   Leading with it, then having a reviewer read §6.2 of the resolution study in the
   repo, is a self-inflicted credibility wound. It becomes *the cleanest
   illustration of a non-vanishing component*, one paragraph. The headline mechanism
   is **operator provenance / failed cancellation**, which is what was measured.
2. **The residual-descent result is far stronger than the abstract claims, and it
   is the paper's best negative.** Starting at the *exact ground truth* with no
   network anywhere, and having the objective walk you off it in 24/24 cases, is
   cleaner than anything currently in the manuscript.
3. **Control 4 is the paper's best positive and it is absent from the manuscript
   entirely.** AUROC 0.952 on the engineering quantity, beating a physics-free
   baseline with CIs excluding zero, is what makes the paper constructive rather
   than a complaint.

---

## 1. The slogan the evidence supports (fix this before anything else)

`floor_resolution_study.md` §4 licenses this headline:

> *You cannot audit a surrogate with a residual operator different from the one
> that generated its training data.*

**That sentence is falsified by this paper's own control 4.** That exact
inconsistent monitor triages worst-decile drag error at AUROC 0.952, beats a
physics-free ensemble on it, and survives the difficulty confound at 92%. It
audits perfectly well. A reviewer who reads the abstract and then the drag table
finds the contradiction in thirty seconds. This is the most likely self-inflicted
wound in the reframe.

**The correct statement, and it is sharper:**

> An operator-inconsistent residual monitor can **rank** predictions but cannot
> **certify** them and cannot be **descended**. The obstruction is a non-vanishing
> floor at the reference solution, and refinement makes it worse, not better.

Three verbs, three verdicts, each independently measured: rank (ρ 0.61; AUROC 0.87
field, 0.952 drag), certify (floor 0.192 ≫ prediction 0.113 — no threshold implies
a bound), descend (24/24 divergence from the truth). That is a complete,
non-contradictory thesis, and it is the spine of every design below.

---

## 2. Candidate designs

### Design A — The audit-operator ladder (the measured boundary)

**Thesis.** What a physics residual can tell you about a neural surrogate is
determined not by the physics but by the provenance of the discrete operator you
can afford to evaluate, and the boundary between "ranks", "certifies" and "can be
minimised" can be located by measurement.

**Title candidates.**
- *Operator provenance sets the limit of residual-based auditing for neural CFD surrogates*
- *What a surrogate-side residual can and cannot certify: a measured boundary on steady RANS*
- *The reference solution fails its own physics check, and refinement makes it worse*

**Contributions, in priority order.**
1. **A measured law.** The floor at the reference solution does not decay under
   refinement — it grows 2.6× from 128² to 512², 0/16 cases decaying — because the
   reference's individually grid-converged terms fail to cancel under a different
   discrete operator. Four artifact controls close the "it's your grid" and "it's
   your interpolant" objections.
2. **The consequence, measured.** Descending that residual from the exact truth
   destroys the field in 24/24 cases; the residual-selected iterate is strictly
   worse than the error-optimal one in 24/24 on both arms.
3. **The boundary, measured on both sides in-house** (the new work): the same
   fields, the same cases, under (i) the affordable Cartesian surrogate-side
   monitor and (ii) the generating solver's own discrete operator on its own mesh.
   Floor ≈ 0 on one side, 0.19 on the other; descent converges on one side,
   diverges on the other. Redraws the boundary that Lei et al. (arXiv:2608.04400)
   would otherwise be used to falsify.
4. **The ROM/DWR foil** (see §3.5 — this is now a first-class contribution, not a
   defence). Goal-oriented error estimation is mature and effective for
   projection-based ROMs *precisely because* the ROM residual is evaluated with the
   full-order operator, `r = A_FOM(u_ROM) − b`, so its floor is zero by
   construction. A data-driven surrogate trained on a released dataset has no
   `A_FOM`. **That single structural difference is the subject of the paper**, and
   it localises the novelty exactly.
5. **A cost ladder.** What each rung of operator fidelity costs in wall-clock,
   against what it buys in floor magnitude and ranking quality. Turns the boundary
   from a label into a design rule.
6. **What survives as deployable**: case-level ranking (free, 1.13 ms), the drag
   triage result, the conformal layer, the gate's guarantee stated at its true size.

**Section structure.**
1. Intro — the practice (surrogate-side residual monitors and PINN residual losses
   are being built on right now), the question, the finding.
2. The monitor and its verification (MMS `p = 2.06`) — *code verification passes*.
3. The floor at the reference solution, and the refinement ladder — *solution
   verification fails, for the reference*.
4. Mechanism: failed cancellation, not closure omission; `∇ν_t` as illustration.
5. The three verbs: rank / certify / descend, each with its experiment.
6. The other side of the boundary: the solver-consistent operator, measured. The
   ROM/DWR comparison lives here.
7. The cost ladder and the design rule.
8. What remains deployable (trust layer, drag triage, gate) — compressed.
9. Limitations, stated as scope.

**Carries over.** The floor measurement, the resolution ladder and its four
controls, the descent test, controls 1–6, the MMS gate, the λ-sweep and λ*≈600, the
three-backbone spread as evidence the phenomenon is not backbone-specific, the
DeepCFD second dataset, the cost table, the artifact manifest.

**Required new work.**
- **A1. An owned OpenFOAM corpus** (~30–50 2-D airfoil cases at AirfRANS-like
  conditions, own mesh, own schemes). *Compute trivial* (cold solves 88–239 s per
  paper-2 measurements); *plumbing is the cost*.
- **A2. A residual evaluator on the solver's own operator.** Write out residual
  **fields** explicitly. **Do not use OpenFOAM's logged initial residuals as a
  norm** — they carry a per-equation normalisation that is not comparable across
  operators or cases. Define and document your own norm.
- **A3. The ladder repeated on the owned corpus**, where `s_local` is known and
  controllable, **across 2–3 source-mesh families** (see A-ATTACK-7).
- **A4. Descent on the consistent operator** (SIMPLE/Newton from the mapped
  prediction) — largely a port of `openfoam_warm_start.py`.
- **A5. The cost ladder**: time each rung (Cartesian 128², Cartesian 512²,
  body-fitted raster, mesh-native FV assembly, full solver residual).

**Cost.** ~2.5 weeks, plumbing-dominated. Two of the five items are ports, not builds.

---

### Design B — The functional audit (the constructive turn)

**Thesis.** The pointwise residual norm is uncertifiable because of a floor that
refinement worsens; but the residual weighted for a specific engineering functional
may escape that floor, and if it does, the same free monitor that cannot certify a
field can certify a force.

**Title candidate.** *A free, goal-oriented certificate for neural CFD: the
residual cannot certify the field but may certify the force*

**Why it is attractive.** It converts the negative into a method, targets the
quantity engineers actually use, has a respected classical anchor, and explains
control 4's AUROC 0.952 mechanistically rather than reporting it as a curiosity.

**Why it may not exist. Read this before investing.**

*First, the mathematics is weaker than it looks.* The tempting argument is that by
the divergence theorem `∫_V R(w) dV` equals a boundary momentum flux, so it bounds
force error and its signed cancellation escapes the floor. **That argument is wrong
as stated.** The identity says the volume integral of *one field's* residual equals
the discrepancy between two ways of computing the force *from that same field*. It
is a statement about a single field's internal inconsistency, not about error
against the truth. Bounding `|C_d(w) − C_d(u*)|` requires `δJ ≈ −(ψ, R(w))` with
`ψ` the drag adjoint — a nontrivial field costing roughly one solve. Constant
weighting over a control volume is the **zeroth-order** adjoint, exact only in the
continuum with exactly-satisfied conservation. And "the floor cancels under
integration" is not plausible-by-default: the floor study already found the residual
**is** the non-cancelling remainder of grid-converged terms.

*Second, the prior art is denser than `novelty_hunt.md` recorded.* See §3.5. Design
B's surviving delta is narrow even if the gate passes.

**So Design B is gated on an experiment, not assumed.** See §4.

**Contributions if the gate passes.**
1. Everything in Design A, compressed to about half its length.
2. A functional-level audit in which the floor's inversion (prediction scoring
   better than truth) disappears, so a threshold on the functional residual *does*
   imply a force-error bound.
3. A conformalised force-error certificate at zero marginal cost.
4. The honest framing: goal-oriented error estimation is the classical answer; what
   was unknown is whether it survives an **operator-provenance floor**, and here is
   the measurement.

**Cost.** Gate: 3 days. Full build if it passes: +1 week on top of Design A.

---

### Design C — Verification and validation for neural surrogates

**Thesis.** Map neural CFD surrogates onto the CFD community's formal V&V machinery
— code verification, solution verification, validation.

**Verdict: kill as a standalone design.** No measurement. Computers & Fluids sets a
rigor bar; a mapping paper with no new number reads as a position piece and is
desk-rejectable on exactly the axis that has killed this paper twice. A hostile
reviewer's first sentence writes itself: *what did you measure?* ASME V&V 20 is
additionally about validation uncertainty against *experiment*; surrogates validated
against simulation data do not map cleanly, and a forced mapping reads as a survey.

**But keep the vocabulary.** It lands hard on this evidence, in one subsection:

> Code verification of the monitor **passes** — method of manufactured solutions
> gives observed order 2.06. Solution verification of the *reference data under that
> monitor* **fails**, and fails worse under refinement. The surrogate is being
> audited by a verified operator against a field that is not a solution of it.

That is a precise, quotable, discipline-native statement of the whole finding. Use
it as the framing sentence of §3 in Design A. A paragraph, not a paper.

---

### Design D — A + B composite (the recommendation, conditional on the gate)

**Thesis.** *An operator-inconsistent residual monitor can rank a surrogate's
predictions but cannot certify or correct them; the obstruction is a floor that
refinement worsens; and the way out is not a better grid but either the generating
solver's own operator (which we cost) or a goal-oriented functional (which we
build).*

**Structure.** Design A's spine, with Design B slotted in as §7 in place of the cost
ladder's final rung, and the deployable material compressed to one section.

---

## 3. Hostile attack on each design

### Attacks on Design A

**A-ATTACK-1 (most dangerous). "Your floor rises because at your finest rung you are
differentiating a piecewise reconstruction, not sampling a solution."** At 512²,
`h/s_local = 2.01`; the report concedes 1024² would not clear the exclusion. A
numerical analyst on a C&F board goes straight here.
- *Survivable?* **Yes, but only with A3.** Current defences: the 128²→256² leg alone
  shows the rise at `h/s_local` 8.1→4.0, comfortably clear; the MMS-raster null; the
  C¹ control that *raises* the floor. Good, but defensive.
- *What neutralises it:* the owned corpus. You control the source mesh, so `s_local`
  is known everywhere, you can generate a source finer than any raster rung, and you
  can evaluate the identical field under the solver's own operator as an absolute
  reference. **This alone justifies A1–A3.**

**A-ATTACK-2. "This is the local truncation error. Brandt's τ, Stetter's defect
correction, Morin–Nochetto–Siebert's data oscillation own it."**
- *Survivable*, cheaply, but only if conceded in the first person and in the intro.
  Classical τ is *the same equation family on nested grids* — a discretisation-**order**
  gap that shrinks predictably. This is a **cross-provenance** mismatch that does
  **not** shrink, and 0/16 cases decaying is the evidence that it is not τ. One
  paragraph converts a scoop into a citation. Failing to write it is fatal.

**A-ATTACK-3. "Nobody audits with a Cartesian finite-difference monitor. Straw man."**
- *Survivable*, but pre-empt with citations, not assertion. Gopakumar et al.
  (arXiv:2502.04406) use convolutional layers as finite-difference stencils with the
  residual as conformal nonconformity score; ANCHOR (arXiv:2512.19643) triggers a
  solver from a residual-derived signal; Zhang et al. (arXiv:2602.14918, App. E) hit
  the floor and worked around it. Every PINN residual loss is this monitor. Name four
  in the second paragraph of the intro; the charge dies there.

**A-ATTACK-4. "You cite Lei et al. for the good side. Your boundary is half hearsay."**
- *Survivable only with A2/A4.* Measuring both sides on the same cases with the same
  predictions is what makes it a boundary rather than a juxtaposition.

**A-ATTACK-5. "Your slogan says the monitor cannot audit; your table says it triages
drag at 0.952."** *Fatal if left in, free to fix.* See §1.

**A-ATTACK-6. "The floor is partly your rasteriser, and rasterisation is your
choice."** *Partly conceded, and it must be.* The honest position: yes, and that is
the point — a deployed surrogate on a Cartesian grid *has* to rasterise, so the
mismatch is a property of the deployment, not an avoidable mistake. The C¹ control
and the MMS-raster null show the rasteriser does not manufacture the rise.

**A-ATTACK-7 (new, introduced by the fix to A-ATTACK-1). "Once you own the source
mesh, the floor depends on *your* mesh choice. You picked a mesh that maximises the
mismatch you wanted to find."**
- *Survivable, and cheaply, because solves are 88–239 s.* Run the floor measurement
  across **2–3 source-mesh families** — O-grid and C-grid, two near-wall stretching
  ratios — and show the non-decay is robust to the choice. That converts "we control
  provenance" into "we varied provenance and it did not matter." **Budget this into
  A3 from the start**; retrofitting it after a reviewer asks costs a revision cycle.

**Verdict on A: survives.** A solid, defensible specialist paper. Ceiling:
"important measurement, no method."

### Attacks on Design B

**B-ATTACK-1 (now upgraded to potentially fatal — see §3.5). "Goal-oriented error
estimation for surrogates and reduced-order models is an established line. Cao et
al. correct neural-operator predictions with a residual-based scheme explicitly
inspired by goal-oriented a posteriori error estimation. Jha's corrector — which you
already cite — is the variational version. ROMES maps residual-based error
indicators to error distributions. DWR-vs-machine-learning error estimation for
pROMs has been evaluated head-to-head in CMAME. What is new?"**
- *Survivable only if the delta is stated as the floor/functional interaction*, i.e.
  the quantitative statement that the floor which destroys the norm-level audit does
  or does not survive the functional. That is new *because the floor result is new*.
- *If the §4 gate returns "the floor does not cancel", there is no new content and
  this attack lands.* Do not build B on hope.

**B-ATTACK-2. "Your zeroth-order adjoint is an approximation of unmeasured
quality."** True. State it, and report the cost of the true adjoint (≈ one solve) so
the reader sees the trade. Survivable if honest.

**B-ATTACK-3 (serious, and it is the repo's own record). "Your force integrator has
332% median magnitude error on *perfect* fields and ρ_D = 0.84 as a ceiling."**
- *Survivable only with scoping and, ideally, the owned corpus.* Control 4 is already
  constructed correctly (both sides through the same integrator, so the target is
  model error not integrator bias) and that must be in the caption, not the appendix.
  Additionally: **viscous drag is unavailable at 128², not approximate** — say so
  flatly and scope to pressure/momentum-flux drag. The owned corpus, where the wall
  is resolved, is the clean fix and is already being built for Design A.

**B-ATTACK-4. "Your control volume sits 1–1.5 chords from a lifting airfoil, where
induced velocity is O(few %) — exactly the scale you are integrating."** Real, and
unavoidable on the AirfRANS crop `(-1,2,-1.5,1.5)`. The owned corpus with a proper
far field fixes it. Flag as a limitation of the AirfRANS arm.

**B-ATTACK-5. "AUROC 0.952 has 20 positives."** Control 4 carries 3 seeds and paired
bootstrap CIs. Report them.

**Verdict on B: conditionally alive, with a lower ceiling than first assessed.** Run
the gate; do not pre-commit.

### Attacks on Design C

Killed above. No measurement; wrong genre for the venue's failure mode.

---

### 3.5. The prior-art line `novelty_hunt.md` never searched — and why it helps more than it hurts

`novelty_hunt.md` covered residual-as-objective, residual-as-nonconformity-score,
and the floor. It did **not** search goal-oriented / adjoint-weighted error
estimation for surrogates and ROMs. That is Design B's exact territory. Searched
now; findings, and they cut both ways.

**Prior art that constrains Design B (all must-cite, none currently in `refs.bib`
except Jha):**

| Work | What it does | Effect |
|---|---|---|
| Cao, O'Leary-Roseberry, Ghattas et al., *Residual-Based Error Correction for Neural Operator Accelerated Infinite-Dimensional Bayesian Inverse Problems*, arXiv:2210.03008 | Corrects a neural-operator prediction by solving a linear error problem on the PDE residual at the prediction, **"inspired by goal-oriented a posteriori error estimation"** | **Closest hit to Design B.** Residual-based correction of a *neural operator*, goal-oriented lineage |
| Jha, CMAME 419:116595 (2024), arXiv:2306.12047 | Residual-based error-corrector operator for neural-operator surrogates of variational BVPs | **Already cited** (as `learned2023residualcorrection`). Variational form ⇒ floor zero by construction — the benign-regime anchor |
| *Evaluation of dual-weighted residual and machine learning error estimation for projection-based ROMs of steady PDEs*, CMAME (2023), S0045782523001111 | Head-to-head DWR vs learned error models for pROMs; DWR extrapolates, MLEM interpolates | **The direct "why is a neural surrogate different from a ROM?" paper.** Must be answered, not ignored |
| Drohmann & Carlberg, *The ROMES method*, arXiv:1405.5170 | Uses dual-weighted residuals as cheap indicators, GP-regresses indicator → error distribution | Dents the trust-layer claim further: cheap-residual-indicator → error-estimate is an established pattern |
| Roth, Schröder, Wick, arXiv:2102.12450; multigoal DWR with PINNs (Springer, 2025) | Neural networks *solve the adjoint* inside DWR | Adjacent, not competing — NN as solver, not as audited object |

**Now the part that helps, and it is worth more than the damage.**

Every one of these evaluates the residual with the **full-order model's own
operator**. For a projection-based ROM that is definitional: `r = A_FOM(u_ROM) − b`.
The floor is zero by construction, which is exactly why DWR works there. **A
data-driven surrogate trained on a released dataset has no `A_FOM`** — the dataset
ships fields, not operators.

That gives the paper its cleanest novelty sentence, and it should appear in the
introduction:

> Goal-oriented error estimation is mature and effective for reduced-order models
> because the ROM residual is evaluated with the full-order operator, so it vanishes
> at the truth by construction. A neural surrogate trained on a public dataset has no
> such operator available. We measure what happens to residual-based auditing when
> that single structural assumption is removed.

This does three things at once: it answers the ROM question a reviewer will ask, it
localises the novelty on **operator availability** rather than on the residual idea,
and it explains why an established technique has not simply been applied already.

**Action:** add all five works to `refs.bib`; write the paragraph above; and note
that this line strengthens **Design A** more than Design B. A's foil is now a mature,
respected literature that works *for a reason this paper identifies*.

---

## 4. The gate that decides between A and D (run this first — 3 days)

Everything in Design B hangs on one measurable question, and it is cheap because the
per-case residual fields already exist.

**Definition.** For a control volume `V` (nested boxes around the airfoil, plus the
body surface term handled explicitly on masked cells), define the drag-direction
functional residual

`I(w; V) = e_D · ∫_V R(w) dV`

evaluated for the truth `u*`, the raw prediction, and the corrected field.

**Pre-register these tests and both branches, in a commit, before running** — as the
ladder's decision rule was committed in `bf5106c`.

**(a) Does the floor cancel?** The norm-level pathology is the **inversion**: the
prediction's residual sits *below* the truth's in 160/200 cases, so no threshold on
`‖R‖` can imply an error bound. Measure the inversion rate under `|I(·;V)|`.
- *Pass:* inversion rate falls below ~10% at some `V`.
- *Fail:* it stays near 80%. No certificate exists.

**(b) Does it rank?** Spearman and AUROC of `|I(w;V)|` against per-case `|ΔC_d|`
(target `vs_gt_nf`), against the existing `‖R‖` baseline of **AUROC 0.952**. Paired
bootstrap.

**(b′) Cancellation diagnostic — run this alongside (b), not after.** `I(w;V)` is a
*signed* integral, so a field with large but mutually cancelling errors scores near
zero. That is the mirror image of the floor problem and it is a false-confidence mode
in anything called a certificate. Check explicitly whether the low-`|I|` cases include
any high-`|ΔC_d|` cases. If they do: that is your explanation if the ranking is weak,
and it is the honest limitation to state if the ranking is strong. A certificate that
can be fooled by cancellation must say so.

**(c) Does it calibrate?** Is `|ΔC_d| ≲ c·|I(w;V)|` with a conformalised `c` holding
target coverage on held-out cases? This is the actual certificate and the only one of
the three that yields a deployable object.

**Decision rule.**
- **(a) and (b) pass → Design D.** Build (c).
- **(b) passes, (a) fails → Design A**, with one subsection reporting the functional
  as a *better ranker but still not a certificate*. Honest, and it explains control 4.
- **Both fail → Design A only.** Control 4 stays as a measured positive, which is
  fine — it is a result either way.

**Plumbing risks to budget for:** the immersed body's surface term on masked cells;
the zeroed wall ring (0.51% of fluid cells) sitting exactly where that term lives;
sign conventions against the existing `design_force_integrator.py`. Three days, not one.

---

## 5. Recommendation

**Run the §4 gate first. Then build Design D if it passes, Design A if it does not.**
Either way, do §1's slogan fix, §0's demotion of the `∇ν_t` closed form, and §3.5's
ROM paragraph immediately — all three are free and two are currently load-bearing
errors.

**Reasoning.**

1. **Design A is the floor of the outcome distribution and is already almost paid
   for.** The measured law, its four artifact controls, and the 24/24 descent result
   are landed. The new work is plumbing on infrastructure that exists on the
   `paper2/openfoam-warm-start` branch. A month is enough.
2. **Design A alone has a real ceiling: a measurement paper with no method.**
   Publishable at C&F — the scope sets a rigor bar and explicitly asks for
   limitations — but it will not excite anyone.
3. **Design B is the only path to a paper cited for what it lets people do rather
   than what it stops them doing**, and its gate costs three days against a month of
   commitment. That ratio is the whole argument for running it first. §3.5 lowered
   its ceiling; it did not close it.
4. **Design C is a framing, not a design.** Absorb the vocabulary, discard the paper.
5. **Venue stays Computers & Fluids.** `venue_plan.md`'s reasoning holds and the
   reframe strengthens it: single-anonymised (no rework, preprint stays up), no
   length limit, scope names UQ and surrogate models, the ML bar is "excellent
   scientific character" (rigor, not novelty), and the guide explicitly asks authors
   to discuss limitations. Design D over-satisfies that guide line by line.
6. **Do not impose a 13,000-word target.** An earlier draft of this document set one
   to keep RESS live as fallback #3. That was wrong: C&F has no length limit, and
   `venue_plan.md` says RESS needs a genuine reliability reframe anyway — a rewrite,
   not a trim. Cutting to 13,000 while *adding* four experiments, ten citations and
   the ROM section would force out the multi-backbone and DeepCFD material, which is
   the evidence that the floor is not backbone-specific. **Write for C&F. If RESS
   happens, cut then, with the reframe.** Cut what the reframe demotes (the trust
   layer's prominence, `tab:positioning`, the pipeline figure's unimplemented
   fallback box), not what the reframe needs.
7. **One title rule.** No coined system name in the title. That is what invited "no
   new computational methodology" twice. `neuroforge-cfd` belongs in code
   availability only.

---

## 6. Part 3 — where confidence is earned, and where it is a liability

### 6a. State these FLATLY. The evidence carries them.

Assertive language here is not overreach; hedging them is the defect.

1. **The monitored residual does not vanish at the reference solution.** 0.192 mean /
   0.133 median on **200/200** AirfRANS cases, model-independent. The physically
   wrong uniform freestream scores **exactly zero**, at every grid level.
2. **The floor does not decay under refinement.** `p = −0.64 ± 0.29`, **0 of 16 cases
   decay**, 15/16 rise, ×2.56 from 128² to 512² on a region fixed in physical units.
   Two independent studies (24 cases × 3 levels; 16 cases × 5 rungs), three exclusion
   bands, same verdict.
3. **The monitor itself is verified.** Method of manufactured solutions gives observed
   order **2.06**. *A rigor asset the draft never sells.* Say it in the same breath as
   (2): the operator is correct; the reference field is not a solution of it.
4. **The rise is not a rasterisation artifact.** A manufactured analytic solution
   through the identical pipeline converges at order **2.02 ± 0.05** and stays 280×
   smaller; a **C¹** re-rasterisation *raises* the floor rather than lowering it.
5. **The mechanism is failed cancellation under a foreign operator, not a missing
   closure term.** Convective and pressure terms are individually grid-converged;
   repairing the full stress divergence moves the floor by **<0.1%**.
6. **Descending the monitored residual from the exact ground truth destroys the
   field.** 300 steps, no network in the loop, residual 0.181→0.029 (−84%) in **24/24
   cases**, field error 0 → median **0.91**. The compact-stencil monitor falls in
   24/24 too, so this is not the differentiable twin being gamed.
7. **The residual is not a valid iterate selector.** Error-optimal and residual-chosen
   iterates never coincide — **24/24 on both arms** — and the residual-chosen iterate
   is strictly worse in **24/24**.
8. **The objective ranks a wrong field above the truth.** Descent terminates at
   residual 0.028 with field error 1.31, while the exact truth sits at 0.110.
9. **The trust correlation is not a case-difficulty artifact.** Partial ρ 0.561 against
   raw 0.610 — 92% survives; and the deployable residual **significantly outranks the
   oracle difficulty variable** ‖r(truth)‖ (Δρ +0.094, CI [+0.055, +0.142]) on all four
   arms and both metrics. *A pure difficulty proxy cannot do that.* The cleanest
   defensive result in the paper, and currently invisible.
10. **On the engineering quantity the residual is decisively the better score.** On
    the **primary `vs_gt_nf` target** (prediction and reference through the same
    integrator, so the target is model error in drag, not integrator bias): AUROC
    **0.952**, Spearman **0.585**, oracle recovery **0.805** — against 0.871 on field
    error — beating the physics-free ensemble σ (0.888), with paired-bootstrap CIs
    excluding zero, and rank fusion no longer helping. *Report `vs_official` only as a
    secondary target, with the caveat that its ρ = 0.766 is substantially ranking a
    shared integrator bias (seed spread 0.004 across independently trained backbones
    is the tell), so the claim must rest on `vs_gt_nf`.*
11. **The audit is free.** 1.13 ms against ~10 s inference and a solver run.
12. **The gate's guarantee holds by construction and is never violated**, while an
    ungated fixed half-step violates it on 1.0% of deployed backbone cases and 8.8% on
    the ensemble path.
13. **λ\* ≈ 600.** Any no-slip weight that rescues the ordering is two orders of
    magnitude above the framework's λ = 1 — a monitor in which the boundary term
    outweighs the entire PDE residual, i.e. no longer a PDE monitor.

### 6b. These must stay scoped no matter how they are worded

Each is a place where confident phrasing is a *liability* — a reviewer holding the
repo can falsify it.

1. **Not "the residual cannot be minimised."** It can — it falls 84%. The claim is
   that minimising it does not take you to the truth. Also **not "descent always makes
   things worse"** — from a perturbed start it *improves* error in 18/24 cases,
   typically by 60%. Use the iterate-selection statement, which holds 24/24.
2. **Not "the floor grows."** Use **"does not decay, and grows by 2.6× over the
   measured range."** Past `h ≈ s_local` the ladder cannot be pushed, and the report's
   own honest bound says so.
3. **Not "the physics residual is the better trust signal."** On field error the paired
   CI is [−0.035, +0.083] — underpowered, resolving nothing in either direction. Say
   indistinguishable-and-underpowered; do **not** claim equivalence either. (The
   abstract's current "AUROC ≈ 0.9" attributed to the residual is the single sharpest
   overclaim in the paper: the residual is 0.871.)
4. **Not "the gate improves accuracy."** Conceded: an ungated fixed half-step improves
   95.8% vs the gate's 89.2%. Damping does the work. Claim the gate for the guarantee,
   and state that it binds on **1.0%** of deployed cases. A guarantee active on 1% does
   not justify "certified self-correction" as a *framing*, only as a *statement*.
5. **Not "it tells you where."** Per-cell ρ = 0.166 ± 0.155; 16×16 patch 0.323. The
   slogan is **which case**, never *where*, never *how*. Fix the mislabelled 0.22 ± 0.06
   (superseded FNO checkpoint, n=15) quoted under the deployed field's name.
6. **Theorem leg (iii) must be withdrawn or restated.** As written it is an *upper*
   bound sold as a floor; `σ_min` is never computed and is not separable from zero on a
   47000×65500 Jacobian; and it identifies the min-norm point rather than where descent
   lands. Leg (iv)'s cone constant is also wrong (`α < ‖P_rangeL r*‖`, not `α < ‖r*‖`).
   Legs (i)–(ii) are correct and elementary — keep them, labelled as elementary.
7. **λ-sweep leg (A) is resolution-contingent.** Margin +9.3% at 128² to **−2.8%** at
   512²; holds at all five rungs in only 11/16 cases. Scope to the deployed resolution
   and lean on leg (B), which does not depend on the ordering.
8. **Conformal coverage is not evidence.** Split conformal attains marginal coverage by
   construction. The guarantee held is marginal over a random cell of a random case,
   pooled over ~1.6M dependent cells with effective n ≈ 100 — **not** the per-field
   guarantee the trust layer is sold for. Lead with ECE (0.074, deep ensemble) and
   adaptivity, not coverage.
9. **Not "backbone-robust."** ρ 0.40–0.85 is positivity, not robustness, and it is
   monotone in how *bad* the model is. Report MeshGraphNet's absolute error (`mse_u`
   13.5, 101% drag error) and **disclose the density control** (16k train / 180k eval,
   2.44× MSE inflation, n=4) — currently zero mentions in the manuscript, which is a
   non-disclosure a repo-reading reviewer will punish.
10. **Wall quantities and viscous drag are unavailable, not approximate.** At
    Δ = 0.0234c and Re ≈ 2×10⁶ the first cell centre sits at y⁺ ≈ 10³ and the wall ring
    is zeroed. State as a scope statement about what the monitor can certify.
11. **`tab:iters` must report all channels.** `mse_v` +96% and `mse_p` +41% over iters
    0→3 while `mse_u` falls 27%; over 3→15 residual and error move in the *same*
    direction. Drop "opposite directions"; keep "no monotone relation holds".
12. **2-D, steady, one turbulence closure (Spalart–Allmaras), two datasets.** Concede
    without argument.

### 6c. Where the CURRENT draft under-claims (under-claiming is also a defect)

1. **The abstract hedges "reducing it does not reduce field error"** on a sweep that
   never reduced it. It can now say something far stronger *and* true: descent from the
   exact truth, no network in the loop, destroys the field in 24/24 cases. Replace the
   weakest sentence with the strongest evidence.
2. **The floor is buried in an appended theorem** and framed as an assumption (H2). It
   is a measured law with a five-rung ladder and four artifact controls. It belongs in
   the abstract's first two sentences.
3. **Control 4 does not appear at all.** AUROC 0.952 on drag, beating a physics-free
   baseline with CIs excluding zero, is the paper's best positive.
4. **Control 2 does not appear at all.** "The deployable residual outranks the *oracle*
   difficulty variable" pre-empts the confound objection a reviewer raises first.
5. **The MMS verification (p = 2.06) is not sold.** It is the sentence that separates
   this from "your stencil is broken".
6. **The λ-sweep closed form is invisible**, and λ* ≈ 600 quantifies it.
7. **The cost asymmetry is under-used.** σ alone costs the same 4.86× as fusion, so the
   residual is the only zero-marginal-cost score. That turns control 1's tie into a
   practical win.
8. **The ROM/DWR contrast (§3.5) is not made at all**, and it is the paper's cleanest
   novelty sentence.

---

## 7. Part 4 — what a practising engineer can DO with this

A result is practical when it changes a decision. Three decisions, plus one
community recommendation.

**Decision 1 — "Should I train against a physics residual loss, or post-correct by
residual descent?"** *No, if your residual operator is not the one that produced your
training data — and it almost never is, because public CFD datasets ship fields, not
operators.* Evidence: descent from the exact truth walks off it in 24/24; refinement
makes the floor worse, so "use a finer grid" is not a remedy. *The paper must show:*
the descent experiment from ground truth (done), the refinement ladder (done), the
mechanism (done).

**Decision 2 — "Which of my 10,000 surrogate design evaluations do I send back to the
solver?"** *Rank by the residual — it is free (1.13 ms), it is not a difficulty proxy,
and on drag it is the best score available.* AUROC 0.952 on worst-decile drag error;
rejecting the least-trusted 10% recovers 77–81% of the oracle's achievable reduction.
*The paper must show:* control 4 with seeds and CIs on `vs_gt_nf`, the
partial-correlation control, the cost table — plus a clear statement that this is a
**ranking**, with no error bound attached.

**Decision 3 — "How much operator do I have to buy before my audit means anything?"**
*The cost ladder.* This is the deliverable that makes the paper an engineering
artifact rather than a warning. Each rung: what it costs (wall-clock, per case), what
floor it leaves, what it can then support (rank / certify / descend). The likely shape
— and it must be measured, not assumed — is that nothing between "Cartesian FD
monitor" and "the solver's own operator on its own mesh" buys enough, in which case
the rule is *one solver residual evaluation (≈ one SIMPLE iteration) is the cheapest
audit that can certify*, still three orders below a full solve. **That is a genuinely
useful number and nobody has published it.**

**A fourth, aimed at the community, and worth a recommendation paragraph in the
conclusion:** *Public CFD datasets must ship their discrete operator — mesh, schemes,
and a residual evaluator — or surrogates trained on them cannot be audited against
them.* This follows directly from the measurement (the floor is provenance, not
resolution), it explains why ROM goal-oriented estimation works and dataset-trained
surrogate auditing does not (§3.5), it is actionable by AirfRANS / DrivAerML /
BLASTNet maintainers, and it is the kind of claim that gets a critique paper cited.

---

## 8. The paper-2 collision — an explicit decision for the author

Designs A and D both run on the `paper2/openfoam-warm-start` infrastructure. Paper 2
is drafted, unsubmitted, on an unmerged branch, and shares: the OpenFOAM corpus
concept, the prediction→mesh mapping pipeline, and — critically — the *lesson* that
representation/operator provenance decides whether the coupling works (`nf_mesh`
resampled reads −58.8% against +33.9%).

**This is a live salami exposure** and it is not the reviewer's job to sort out.

| Branch | Paper 1 owns | Paper 2 owns | Risk |
|---|---|---|---|
| **(i) Paper 2 first** | The audit-operator boundary; cites paper 2 for the pipeline | Warm-start speedup, the representation criterion | Low, but paper 1 slips by however long paper 2 takes |
| **(ii) Paper 1 first** (recommended) | The boundary, using OpenFOAM as a *measurement instrument* | Warm-start speedup as a *performance* result | Moderate — paper 2 must then be explicit about what it adds |
| **(iii) Merge** | One paper: "the operator is the boundary — auditing and correction both live or die on provenance" | — | Kills a publication and produces a ~20k-word monster, but is the most honest and the most impactful single artifact |

**Recommendation: (ii), with a hard rule.** Paper 1 uses the solver operator only as
an *instrument* (residual evaluation, floor measurement, cost) and makes **no speedup
claim**. Paper 2 keeps the entire warm-start performance story. Cite paper 2's
preprint from paper 1 explicitly for the pipeline, and state the split in one sentence
in each. If the author would rather have one strong paper than two adequate ones,
(iii) is defensible and must be decided **now**, not after the corpus is built.

---

## 9. Week-by-week plan (four weeks, single workstation + WSL2 OpenFOAM)

Plumbing-dominated by design. Compute is not the constraint; mapping, conventions and
masks are. **Writing is deliberately parallelised** — see the Week 1–2 writing track.

### Days 1–3 — free fixes and the gate
- Fix the slogan everywhere (§1): rank / certify / descend. **Do this first**; it
  changes what every later sentence must say.
- Demote the `∇ν_t` closed form to "cleanest illustration of a non-vanishing
  component, 4.5% of the floor"; promote failed cancellation to the mechanism.
- Withdraw/restate theorem leg (iii); fix leg (iv)'s cone constant.
- Add the five §3.5 references to `refs.bib` and draft the ROM/DWR paragraph.
- **Run the §4 gate**, with (a)/(b)/(b′)/(c) and both branches pre-registered in a
  commit *before* running.
- **Decision point: Design A or Design D.**

### Week 1 — the owned corpus (the exposure-killer)
- A1: generate 30–50 2-D airfoil OpenFOAM cases at AirfRANS-like conditions, **across
  2–3 mesh families** (O-grid, C-grid, two near-wall stretchings — A-ATTACK-7).
  Compute is hours; setup is the week.
- A2: residual evaluator on the solver's own operator, writing residual **fields**.
  Define and document the norm; do **not** reuse OpenFOAM's normalised log residuals.
- Gate: the converged solution's residual under its own operator must come out at
  solver tolerance. If it does not, A2 is wrong, not the thesis.

### Week 2 — the two-operator measurement
- A3: repeat the refinement ladder on the owned corpus, where `s_local` is known, and
  across the mesh families. **The single most valuable new number in the plan** — it
  converts A-ATTACK-1 from a parry into a non-issue.
- Same fields, both operators: floor, ranking quality, inversion rate.
- A4: descent on the consistent operator from the mapped prediction, against the
  existing 24/24 divergence on the inconsistent one.
- A5: time every rung of the cost ladder.

### Weeks 1–2, parallel writing track (do not skip — this is the slip insurance)
The intro, the monitor-verification section (MMS), the floor section, the mechanism
section, the descent section and the "what remains deployable" section depend on
**landed evidence only**. Nothing in them waits on Weeks 1–3. Draft them while the
corpus builds. Week 4 then writes only the new-experiment sections and the
recommendation, which is a week's work rather than a month's.

### Week 3 — the constructive half, and freeze
- If the gate passed: build the functional certificate (conformalise `c`, held-out
  coverage), validated against the solver-consistent reference, with the (b′)
  cancellation limitation stated.
- If it failed: expand the cost ladder's intermediate rungs (body-fitted raster,
  mesh-native FV assembly) so the design rule is a curve, not two points.
- **Freeze all tables and figures by end of week 3.** No number moves after this.

### Week 4 — finish
- Write the new-experiment sections onto the Week 1–2 draft. Do not edit the old
  manuscript in place; the section order has changed and editing leaves orphaned claims.
- Apply every item in §6a/6b/6c.
- Add the remaining missing citations (Lei; Zhang App. E; Astral; ENS; Luo & Zhou;
  Wang NeurIPS 2022; McGreivy & Hakim; Brandt; Stetter; Bochev & Gunzburger;
  Morin–Nochetto–Siebert) alongside §3.5's five.
- Delete `tab:positioning`; insert the two-row regime table **with a cost column**.
  Grey out or remove the pipeline figure's unimplemented fallback box.
- New title, no system name. Cover letter per `venue_plan.md` §8. Clear the
  must-eyeball list (`venue_plan.md` §7) in a browser. arXiv v4 **after** the rewrite.

**If a week slips**, cut in this order: (1) the cost ladder's intermediate rungs,
(2) the functional certificate's conformalisation (keep the ranking result),
(3) corpus size 50→30, (4) mesh families 3→2. **Never cut A3** — it is the
exposure-killer.

---

## 10. The ceiling, stated without flattery

**With the §4 gate failing, this is a solid specialist paper.** A well-controlled
measurement that constrains a practice the field is currently building on, plus a free
ranking signal with a documented scope, plus the ROM/DWR contrast that explains why
the established technique does not transfer. Computers & Fluids is the right home, the
rigor bar is met comfortably, and it will be cited by people who were about to add a
residual loss and did not. It is not a landmark and no amount of writing will make it
one.

**With the gate passing, it is a strong specialist paper** — a measured law plus a
constructive, goal-oriented audit with a design rule and a certificate on the quantity
engineers use. Genuinely good work; it would survive review at a considerably better
venue than C&F. Still not a landmark, and §3.5 is why: the goal-oriented machinery is
borrowed, and the new part is its interaction with the floor.

**A landmark would additionally require, and none is reachable in a month:**
- **3-D.** Every claim here is 2-D. The provenance argument should hold in 3-D and the
  cost ladder would change shape, but "should" is not a measurement.
- **A dataset that ships its operator**, so the floor can be driven to zero by
  construction and the boundary crossed *continuously* rather than at two discrete
  points. This is the experiment that turns the boundary into a curve.
- **A cheap consistent monitor that actually works** — Design B succeeding not just
  for a functional but for the field. Nothing in the current evidence suggests it
  exists, and the refinement ladder is mild evidence that it does not.
- **A second turbulence closure and a second solver**, to show the effect is about
  provenance rather than about Spalart–Allmaras or OpenFOAM specifically.

**The honest summary for the author:** the assets support a paper that is *correct,
well-controlled, and useful*, whose contribution is a measurement and a design rule
rather than a new method. That is exactly the genre Computers & Fluids' scope text
invites, and it has real precedent (McGreivy & Hakim, Nature MI 2024). Aim there,
execute the §4 gate honestly, and stop trying to make the paper sound like a method
paper — that framing has been rejected twice, and it was the framing, not the science,
that failed.
