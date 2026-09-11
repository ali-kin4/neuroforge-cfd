# Idea portfolio — candidate directions, attacked

Date: 2026-09-11. Companion to `strongest_design.md` (which paper to write from the
*current* evidence) and `whitespace.md` (where the field is empty). This document
answers a different question: **what should the next month buy, optimised for
novelty × a positive outcome**, given that six claims did not survive measurement
in the week of 2026-09-07.

Nothing here edits a manuscript file. Every asset named was verified in-repo this
session; every probability carries its reasoning; every candidate is attacked.

---

## 0. Two definitions, fixed before any number is quoted

**0a. "Positive" is defined per candidate, in advance, or the probability is
meaningless.** The failure mode this portfolio is built to avoid is a candidate
with a 90% chance of a *strong effect* whose strong effect is another
demolition. C1 is exactly that shape: it is near-certain to produce a large,
clean effect, and the most likely large clean effect is "the AirfRANS force
metric measures angle of attack". That is a seventh negative wearing a
contribution's coat. So each candidate below states **what counts as positive**
before it states odds, and the odds are odds *of that*, not of "an effect".

**0b. A cheap discriminating test is one whose output is the probability
estimate**, not one that merely reduces variance. Three of the candidates below
have gates that cost under three days and run on data already on disk. The right
move is to run all three and then commit, exactly as the three dead ends were
killed last week. Do not pre-commit on the basis of the priors in this document —
they are priors, and two of them are near 0.5 by construction.

---

## 1. The ranked table

Costs are working days on the existing single 4070 Ti + WSL2 OpenFOAM. "Gate"
is the cheap discriminating test; "full" is the remaining cost if the gate passes.

| # | Candidate | Claim if positive | Gate (days) | Full (days) | P(positive *as defined*) | Novelty / nearest prior art | Strongest objection |
|---|---|---|---|---|---|---|---|
| **C8** | **Ship the operator: an owned 2-D RANS corpus that releases mesh + schemes + a residual evaluator, and the measured crossing of the audit boundary** | *With the generating operator in hand, residual-based auditing and correction work — floor at solver tolerance, descent stays put at the solution, the residual becomes a valid iterate selector. The obstruction paper 1 measured is a property of dataset release practice, and we remove it.* | **3** | 18–20 | **0.60–0.65** (see §2.1) | **High.** No public CFD dataset ships its discrete operator (null N4/N3 in `whitespace.md`). Nearest: AirfRANS (Bonnet 2022), The Well, PDEBench, CFDONEval (IJCAI 2025) — all fields-only. **DiFVM, arXiv:2603.15920 — VERIFIED this session** (Du, Li, Xu, Wang): differentiable unstructured FV in JAX, integrates with OpenFOAM, ships differentiable operators, **no dataset release** — a tool, and the live scoop risk. **ECO, CMAME S0045782526004962 / arXiv:2504.13422 — VERIFIED and NOT relevant**: super-resolution operator learning in *mechanics of materials*, not CFD; `whitespace.md` over-weights it | "Another 2-D airfoil dataset in a world that has AirfRANS." The answer must be the *operator*, not the fields — and it only lands if the boundary crossing is measured, not asserted |
| **C1** | **The parametric baseline audit of geometry-parametric CFD benchmarks — plus the split on which operators earn their keep** | *(a) On AirfRANS in-distribution, a regression on the case parameters — which are in the file name — matches or beats every field-based force number, and most field metrics inherit the same confound. (b) On the `reynolds` and `aoa` extrapolation splits the operators clear it decisively. Here is the covariate-controlled protocol that separates the two.* | **2–3** (a) + **5–6** (b, needs retraining) | 12–15 | (a) **0.85**; (b) **unknown — the gate is the estimate; prior 0.4–0.6** | **Moderate–high, and timing-critical.** `drag_covariate_control.json` already has ρ=0.874 from `(U,α,α²)`. Genre precedent: McGreivy & Hakim, Nature MI 6:1256 (2024); Duraisamy arXiv:2604.20061. Scoop-adjacent: DD-RNO arXiv:2608.13490 is one control away. Classical foil: kriging on airfoil parameters (Forrester & Keane) is 30 years old and is the objection, not the contribution | "You have rediscovered that Cd is a smooth function of α. Aerodynamicists have kriged this since 1995." Survives **only** if (b) is run and the constructive half is the headline |
| **C2** | **Trust-gated fallback, measured end to end against a production solver — where the rejected prediction pays for its own fallback** | *Rejecting the least-trusted decile and handing those cases to OpenFOAM, warm-started by the very prediction that was rejected, traces a risk–coverage–cost frontier that dominates random rejection; and the rejection score and the warm-start saving do not fight each other.* | **1** (correlation, data on disk) | 10–12 | **0.45** | **Moderate.** ANCHOR arXiv:2512.19643 already does residual-signal → solver fallback; PCGBandit arXiv:2509.08765 owns "never worse than the default". The genuinely new wrinkle is that the fallback's cost is *reduced by the failed prediction itself* — selective prediction assumes constant rejection cost | "This is selective prediction plus your own paper 2, and the cost saving is 'we only solved 10% of the cases', which is arithmetic." Also re-opens the paper-1/paper-2 split (§4b) |
| C7 | **RESERVE — do not start before the three gates return.** Residual-gated design search: use the 0.952 drag-triage score to decide which candidate shapes get a solver call | *A physics-residual-gated search reaches the drag optimum in fewer solver calls than σ-gated, random, or pure-surrogate search.* | 2 (pool-based, offline, no new solves) | 10 | **0.40** | Moderate. Residual-driven adaptive sampling exists for PINNs; residual-as-acquisition for *design* with a solver in the loop is thinner. Enabler is C4/`decisive_controls.md`'s AUROC 0.952 on \|ΔC_d\| | "Bayesian optimisation with a different acquisition function, benchmarked against weak alternatives." And σ is only 0.06 AUROC behind. Held in reserve because its gate competes for the same week as three better ones |
| C6 | Sub-claim of C1. A surface-native output head (c_p, c_f on the polyline) that clears the full-parameter baseline on partial ρ_D where raster integration cannot | *Drag becomes observable by changing the output representation, not the model size.* | 0 (gated on C1a) | 8 | **0.35**, and **conditional**: if C1a's full-parameter regression reaches ρ>0.98 there is no headroom and this is dead on arrival | Moderate; DD-RNO arXiv:2608.13490 occupies the constructive slot already (learned quadrature, 0.250→0.997) | "DD-RNO did this in August and got 0.997" |
| C10 | Sub-claim of paper 2. A conformal stopping rule for the solver: certify that the force has entered its band, with coverage | *X% fewer iterations than `residualControl` at a guaranteed force-band coverage.* | 1 (re-score existing trees) | 7 | 0.50 | Low–moderate; convergence-detection heuristics are an old literature and `solver/scoring.py` already contains the readability machinery | Collides head-on with paper 2's metric definitions; reads as a re-scoring of an existing study |
| C11 | Sub-claim of C1. Data-efficiency crossover: how much data an operator needs before it beats the parametric baseline | *The crossover point, per metric, on two benchmarks.* | 0 | 3 | 0.6 | Low as a standalone | Not a paper. Keep as C1's §5 |

**Read the table with §0a in hand.** C1's 0.85 is the probability of a *negative*
finding about the field with a constructive wrapper. C8's 0.60–0.65 is the
probability of a result the author would call positive. They are not comparable
numbers and the table would mislead if that were not said out loud.

---

## 2. The top three, worked out

### 2.1 C8 — Ship the operator, then cross the boundary

**Why this one exists.** Everything paper 1 lost, it lost for one structural
reason: a dataset-trained surrogate has no `A_FOM`. `strongest_design.md` §3.5
identifies that as the paper's cleanest novelty sentence, and §10 names
"a dataset that ships its operator, so the boundary can be crossed *continuously*"
as one of four things that would make the work a landmark and declares it out of
reach in a month. It is closer to reach than that assessment allowed, because the
compute is trivial (cold solves 88–239 s, `PLANS.md` §1) and the two expensive
pieces — body-fitted mesh generation (`solver/{ogrid,cgrid}.py`) and mesh-native
surrogate evaluation (`solver/surrogate_seed.py`) — are built and tested.

**The exact experiment.**

*Corpus (A1).* 250–400 2-D airfoil cases, NACA 4-digit × α × Re spanning the
AirfRANS envelope, on body-fitted C-grids, **across 2–3 mesh families** (O-grid,
C-grid, two near-wall stretchings) so the A-ATTACK-7 charge — "you chose the mesh
that maximises the mismatch you wanted" — is answered by construction rather than
by argument. Release: fields **plus** the mesh, the `fvSchemes`/`fvSolution`, and
a Python residual evaluator that reproduces the solver's own discrete operator.

*Instrument (A2).* A residual evaluator on the solver's own operator, writing
residual **fields**. Do not reuse OpenFOAM's logged initial residuals — they carry
a per-equation normalisation that is not comparable across operators or cases.
Document your own norm, reduced identically to `Diagnostics.residual_norm()` so
the numbers are commensurable with paper 1's.

*The four measurements that constitute the crossing*, each run on the same cases
with the same predictions under (i) the affordable Cartesian surrogate-side
monitor and (ii) the generating solver's own operator:

1. **Floor.** `‖R_solver(u*)‖` at the converged solution.
2. **Descend.** Descent under the solution's own operator, and defect-correction
   descent from the mapped prediction, against paper 1's results on the
   inconsistent operator.
3. **Rank.** Spearman and AUROC of `‖R_solver(û)‖` against per-case drag and
   field error.
4. **Certify.** Split-conformal width ratio, using the exact
   `⌈(1−α)(n+1)⌉` order statistic (the defect fixed in
   `mechanism_decision.md` §3 — do not reintroduce it).

**Gate — 3 days, ~20 cases, and it is not the plumbing check.** The obvious gate
("the converged solution's residual comes out at solver tolerance") proves only
that A2 is correctly implemented. It is necessary and it is not the science. The
scientific gate, and the reason it is a real gate:

> `mechanism_decision.md` §4.3 measured that the conformal width tracks
> `Q_0.9/median` of the nonconformity ratio `E/σ` almost exactly — 15.1 for an
> uninformative score, 6.3 for the deployed monitor, 10.5 after floor
> subtraction — and that removing the floor **exactly** made the certificate
> **wider**, 7.4× → 11.3×, on 3/3 seeds. "No floor ⇒ tight certificate" is not
> implied. This project has already measured floor removal making it worse once.

So the gate must measure, on ~20 owned cases:

| pre-registered test | pass | fail |
|---|---|---|
| **(G1) floor** | `‖R_solver(u*)‖` at solver tolerance, ≥2 orders below `‖R_solver(û)‖` on ≥90% of cases | it is not — A2 is wrong, or mapping error re-introduces a floor comparable to the signal, which is the interesting-but-fatal outcome |
| **(G2a) descent from the converged solution** | descent under the solution's **own** operator leaves it where it is: field error stays at ~0 on ≥90% of cases | it walks off, as it does on the inconsistent monitor. The boundary does not cross and the headline is gone |
| **(G2b) iterate selection** | the residual-chosen iterate **coincides** with the error-optimal iterate on ≥90% of cases | it does not — paper 1's universal claim survives the operator change, which kills the thesis |
| **(G3) dispersion** | `Q_0.9/median` of `E/σ` for `‖R_solver(û)‖` vs drag error comes in **below 3.0** (against the deployed raster monitor's 6.3) | it comes in near 6.3 — the certificate sub-claim is dead, **but C8 survives without it** (see below) |

**Why G2 is specified this way, and why the obvious version would have been
worthless.** The tempting gate is "defect-correction descent from the mapped
prediction reduces field error on ≥80% of cases". That is **not** the inverse of
paper 1's result and it sits barely above the null. Paper 1's 24/24 divergence was
measured **from the exact ground truth**; its companion number, stated three times
in the repo (`strongest_design.md` §6b, `RESOLUTION.md` F1), is that on the
**inconsistent** operator descent **from a perturbed start already improves error
in 18/24 cases, typically by 60%**. 18/24 is 75%. An ≥80% gate on the consistent
operator could therefore pass by a single case over what the operator we are
trying to beat already does — and a reviewer finds that in one pass through paper
1's own descent section.

G2a and G2b are the actual inverses. (a) is a clean binary against 24/24
divergence to median error 0.91 and needs no threshold argument. (b) reverses
paper 1's *universal* claim — "the error-optimal and residual-chosen iterates
never coincide, 24/24 on both arms" — which makes it the sharpest single number
C8 can produce. Specifying the gate this way **raises** P(pass), since G2a is
close to guaranteed if A2 is correct, while making a pass mean something.

*Descent from the mapped prediction stays in the report* as supporting
measurement (c), but it is scored as a **paired** contrast against the
inconsistent operator on the same cases — win rate and median improvement against
18/24 and ~60% — never against an absolute percentage.

**Success criterion, declared now.** C8 is a GO if **G1, G2a and G2b pass**. G3 is
a bonus, not a gate. If G3 fails, the paper's certificate section reports the
honest finding — *the floor is not what sets the certificate's width; the heavy
tail of the drag-error distribution is, and it survives the operator change* —
which is a second measured negative that strengthens rather than weakens the
dataset's motivation, because it says the remaining obstruction is statistical
rather than structural.

**Probability, with reasoning.** P(G1) ≈ 0.9: near-tautological if A2 is right,
and the one real risk is that querying the surrogate at cell centres carries an
error whose residual is comparable to the floor — but `PLANS.md` §3.1 measures
that mesh-native evaluation removes 98% of the wall-gradient error, so the
prediction's residual should sit far above the converged solution's. P(G2a, G2b |
G1) ≈ 0.8: Lei et al. (arXiv:2608.04400, verified) already demonstrate that
residual correction works on steady CFD with a solver-consistent operator,
reporting residual down *and* error down; our version is the cheap variant on a
corpus we own, so the physics is in our favour and the risk is implementation.
P(G3) ≈ 0.35, for the reason quoted above. **P(positive as defined) ≈ 0.6–0.65**,
and the residual risk is schedule, not science: A1's plumbing is the month.

**What the paper claims.** *Public CFD datasets ship fields, not operators, and
that single omission is what makes surrogate-side auditing fail. We release a
corpus that ships its operator, and measure the boundary crossing on both sides
with the same cases and the same predictions: floor 0.19 → solver tolerance,
iterate selection invalid 24/24 → valid N/N, and a design rule for how much
operator you must buy before an audit means anything.* Paper 1 becomes the
negative half and is cited, not repeated.

**Venue.** NeurIPS Datasets & Benchmarks (no APC; the artifact-plus-measurement
shape is exactly what D&B rewards) or TMLR (free, rigor bar, no novelty bar).
Computers & Fluids is the journal fallback and takes it happily. **Verify the D&B
deadline against the calendar before committing** — if it has passed, TMLR first
and D&B next cycle.

**Hostile attack.**
- *"Another 2-D airfoil dataset."* The only answer is the operator, and the
  answer has to be executable: a reader must be able to `pip install`, load a
  case, and get `R(u)` for their own field. If the release is fields plus a tar
  of `system/`, the objection lands. Budget the evaluator as a first-class
  deliverable, not an appendix.
- *"AirfRANS already exists and is bigger."* True, and the corpus should be
  positioned as complementary and deliberately small: it is an *audit* corpus,
  not a training corpus. Say the case count in the abstract rather than hiding it.
- **The live scoop risk, now verified: DiFVM (arXiv:2603.15920).** Du, Li, Xu and
  Wang ship a GPU differentiable finite-volume solver on unstructured meshes that
  reformulates FV operators as graph message-passing primitives and *integrates
  with OpenFOAM workflows*. They release no dataset. That group is one release
  away from shipping exactly this corpus, and a reviewer will ask **"why did you
  build an evaluator instead of using DiFVM?"** There is a good answer and it
  should be decided before the build, not in rebuttal: the evaluator must
  reproduce *the operator that generated the labels*, so adopting a different
  differentiable solver reintroduces the very provenance gap the corpus exists to
  close — unless DiFVM is used as **both** generator and evaluator, which is a
  legitimate and possibly cheaper design. **Make that build-vs-adopt call during
  the 3-day gate.**
- *ECO is not a competitor.* `whitespace.md` lists CMAME S0045782526004962 as a
  nearest constructive hit; verified this session, it is *Equilibrium conserving
  neural operators for super-resolution operator learning in mechanics of
  materials* (arXiv:2504.13422). Solid mechanics, super-resolution, no CFD corpus.
  Downgrade it in the related work.
- *"Who will use it?"* Gopakumar (arXiv:2502.04406), ANCHOR (arXiv:2512.19643),
  PhysicsCorrect (Huang & Perdikaris, AAAI 2026), Zhang et al. (arXiv:2602.14918
  App. E, who explicitly work around the missing `R_h(u*)`), and the ASME VVUQ 70
  committee that has been chartered since 2022 with nothing published. That is a
  named, verified constituency, not a hypothetical one.
- *Does it depend on anything fragile?* **No.** It depends on paper 1's negative
  only as motivation, and the parts of paper 1 that eroded this week (the
  provenance attribution, the floor-subtraction causality, the 0.839 ceiling) are
  not load-bearing for it. That independence is the strongest structural argument
  for C8 in this document.

---

### 2.2 C1 — The parametric baseline, and the split where operators win

**Why this one exists.** `drag_observability.md` §4 found that a three-parameter
OLS on `(U, α, α²)` — parsed from the simulation *file name* — ranks official cd
at **ρ = 0.874**, above every integrator arm ever reported on this benchmark
including on ground-truth fields (0.611–0.845). Geometry (the NACA digits, also
in the name) was **deliberately not controlled for**, so 0.874 is a stated lower
bound. That is the single most exploitable measurement this project owns and it
was found six days ago.

**The two halves, and they must be run in this order.**

*C1a — the demolition (2–3 days, no training).* Fit a GP and a gradient-boosted
tree on the **full** case parameter vector `(U, α, NACA digits)` → official `cd`,
`cl`, `cdp`, `cdv`. Then recompute every force number this project and the
literature report on AirfRANS as a **partial** correlation given those parameters.

> **Protocol discipline, and it is the whole credibility of the candidate.** This
> paper's force is "nobody reports the baseline on the same footing." The
> regression must therefore be fit on the **identical train split** the operator
> numbers use and scored on the **identical n=200 test cases against the identical
> official labels** — `results/control/_cache/official_labels_full_test_n200.json`
> makes this mechanical. **Report the held-out test number, not a cross-validated
> figure over the pooled set.** A CV number is not comparable to a test-split
> number, and that substitution is precisely the first thing a hostile reviewer
> will look for in a paper accusing the field of sloppy comparison.

*Pre-registered decision rule, to be committed before the run:*

| held-out test ρ_cd of the full-parameter regression | reading |
|---|---|
| **> 0.98** | every in-distribution field-based force metric on this benchmark is uninformative. C6 is dead (no headroom). C1 becomes a pure protocol/critique paper and the constructive half **must** come from C1b or it is a seventh negative |
| **0.90–0.98** | headroom exists; C6 becomes live and C1 gains a constructive model arm |
| **< 0.90** | the confound is real but bounded; the paper is a protocol note, not a headline |

My prior on the first row is **0.85**: 800 training cases over a smooth ~6-dim
design space is a textbook kriging problem, and `(U,α,α²)` alone already reaches
0.874 with no geometry at all.

*C1b — the constructive half, and the only thing that makes this a positive
paper (5–6 days, needs training).* Repeat the comparison on AirfRANS's own
**`reynolds` and `aoa` extrapolation splits**, for forces *and* for volume
fields, against a parameter-space **field** baseline (per-node regression /
nearest-neighbour interpolation in parameter space). The claim to hunt:

> *In distribution, a parametric regression matches or beats neural operators on
> every force metric and much of the field metric. Under parameter extrapolation
> it degrades sharply while the operators hold. The operators' value is
> extrapolation in the design space, and the standard in-distribution protocol
> cannot see it.*

**Success criterion, declared now.** C1b is positive if, on ≥1 of the two
extrapolation splits, the trained operator beats the best parametric baseline on
the pre-declared primary metric (volume rel-L2 on `u`) by a margin whose paired
bootstrap CI excludes zero, **while** the same baseline wins or ties
in-distribution. Both directions of the same comparison are required; one alone
is not the finding.

**Probability.** Per §0b, do not quote one — the gate is the estimate. Prior band
**0.4–0.6**, and the reasoning is genuinely two-sided: Cd(α) is smooth and
quadratic extrapolation in α may be embarrassingly strong, but *fields* under
α-extrapolation involve separation onset, which parameter interpolation cannot
represent and an operator plausibly can. The field arm is the more likely
positive and should be the primary metric for that reason.

**Two things to verify before writing a word of this.**

1. **Does Bonnet et al.'s own AirfRANS D&B paper already report a trivial or
   parametric baseline?** "Nobody reports it" is a strong claim about the dataset
   paper being critiqued, and it must be checked against the source rather than
   inferred from the surrogate literature that cites it. Thirty minutes.
2. **Split composition.** Confirm what `task='reynolds'` and `task='aoa'`
   actually hold out before designing around them; the loader exposes them
   (`airfrans_loader.py:344`) but the extrapolation direction should be read from
   the dataset paper, not assumed.

**Cost note, verified.** There are **no checkpoints trained on the `reynolds` or
`aoa` splits** — `checkpoints/v2_transolver/` holds five seeds on the standard
split, and `results/full_research/ood/` is an older FNO-backbone ablation. C1b
therefore needs ~1.7 h GPU per seed per split (Transolver at ~78.6 s/epoch × 80
epochs), so ≈10 h for 3 seeds × 2 splits. Cheap, but it is not inference-only —
plan it as such. Official force labels are cached only for `full` test n=200; the
OOD splits need their labels pulled, which is minutes via the `airfrans` package.

**What the paper claims, and the venue.** *A covariate-controlled evaluation
protocol for geometry-parametric CFD benchmarks, a strong parametric baseline
suite, and the identification of the split on which learned operators
demonstrably earn their keep.* NeurIPS D&B or TMLR; the genre precedent is
McGreivy & Hakim. **Add DeepCFD as a second benchmark** (the loader exists) — a
one-dataset version of this argument reads as one dataset's problem, and two
makes it a protocol result.

**Hostile attack.**
- *"Kriging on airfoil parameters is 1995 technology; you have shown that a
  parametric family is parametric."* This is the killer and it must be answered
  in the first paragraph, not the discussion. The contribution is not that
  kriging works; it is that **the benchmark used to rank neural operators does
  not measure anything kriging cannot do, and nobody reports the baseline.** The
  AASM/AIAA 2025-0036 PALMO case — 52,480 OVERFLOW simulations of NACA 4-series
  airfoils — is the institutional aerospace community shipping the same design
  and inherits the same critique, which is what makes this worth writing rather
  than grumbling about.
- *"Who is scooped by this?"* DD-RNO (arXiv:2608.13490, Aug 2026) is actively
  working AirfRANS forces and reports ρ 0.250 → 0.997 with no covariate baseline
  and no ground-truth control. They are one referee question away. **Timestamp
  early.**
- *"This is a negative result."* Correct, unless C1b returns positive. That is
  the whole reason C1b is not optional.
- *Does it depend on anything fragile?* Partly, and this is a real cost:
  0.874 is currently **also** the strongest new result in the diagnostic paper.
  See §4a.

---

### 2.3 C2 — Trust-gated fallback, end to end, where the rejection pays for itself

**Why this one exists.** `GOALS.md` states the project's ultimate goal as "hands
the untrustworthy cases to a classical solver it has already warm-started". Every
piece exists — AUROC 0.952 on worst-decile drag error, a calibrated conformal
layer, a production OpenFOAM backend, and a warm-start recipe worth +18.4% on
Cd_v across 13/13 cases — and the composition has **never been run**. It is the
one direction where this project's assets already form a system and only the
measurement is missing.

**The genuinely novel wrinkle, stated precisely.** Selective prediction assumes
the cost of a rejection is a constant: you abstain and pay the full price of the
expert. Here the rejected prediction is *not discarded* — it warm-starts its own
fallback, so the price of abstention is reduced by the very artifact that was
judged untrustworthy. That is a new object in the selective-prediction framing
and it is what separates C2 from ANCHOR and from plain risk-coverage curves.

**The gate — 1 day, data already on disk, four outcomes.** Compute the per-case
correlation between the trust score (residual norm and/or `σ_vel`) and the
warm-start iteration saving, over the 13-case corpus in
`results/depth_corpus.json` plus the placement and mechanism trees.

| outcome | reading | action |
|---|---|---|
| **strongly positive correlation** | the cases you reject are the ones the warm start helps least; selection and warm start fight each other and the headline collapses to plain selective prediction | **kill** |
| **uncorrelated** | the composition is clean and additive; both effects are real and independent | **pass, with scope** |
| **anti-correlated** | super-additive: the cases you reject are the ones the warm start helps *most*, so rejection is cheap exactly where it is needed | **pass, and this is the paper's headline** |
| **inconclusive** (n=13 cannot resolve a modest ρ) | expected, and pre-declared | **pass, with scope** — the additive claim needs only "not strongly positive" |

Pre-declaring that inconclusive counts as a pass is important and must be
committed before the run, because n=13 has very little power and the temptation
after the fact would be to read a weak positive as a weak negative or vice versa.

**The full experiment if the gate passes.** On the owned corpus (≥25 cases, which
C8 would produce anyway — these two candidates share infrastructure): sweep the
rejection fraction 0 → 100%, and at each point report (i) population drag error,
(ii) total wall-clock including inference, audit, seed construction and every
solver call, (iii) conformal coverage of the retained set. Three arms: reject by
physics residual, reject by ensemble σ, reject at random. The deliverable is a
**risk–coverage–cost Pareto surface** with the cold-solve-everything and
trust-nothing corners as the two endpoints.

**Success criterion, declared now — and deliberately not an absolute threshold.**
This project has already paid for one arbitrary pre-registered bar (+30% on a
force metric, missed at +18.4%, held to correctly and at cost, `PLANS.md` §0).
Do not set a second one that cannot be derived from anything. The pre-registered
primary is the **three-arm comparison**: the residual arm must dominate the
random arm across the rejection sweep, with a paired bootstrap CI on the
area-between-curves excluding zero, and the full Pareto surface must be reported
whatever it shows. An absolute read-out — *"≥90% of full-solver drag accuracy at
≤35% of full-solver wall-clock"* — is worth stating as a **secondary**,
illustrative operating point, and is explicitly not the pass condition. If the
residual arm merely ties σ, that is reportable and is *not* a failure: the
residual is the only zero-marginal-cost score (1.65 ms against σ's 4.86×
inference), which is the practical win `strongest_design.md` §6c item 7 says is
currently under-used.

**Probability: 0.45.** P(the Pareto surface is producible and clean) ≈ 0.85;
P(the composition claim survives the gate) ≈ 0.55; P(a reviewer nonetheless reads
it as "selective prediction plus your own paper 2") — high, and that is the
ceiling problem rather than a risk of a null.

**Hostile attack.**
- *"Your cost saving is 'we only solved 10% of the cases'."* Arithmetic, and it
  is the reduction a reviewer will reach for. The only defence is the
  anti-correlation arm — if the gate returns merely "uncorrelated", the paper is
  honest, useful and unexciting.
- *"ANCHOR did this."* arXiv:2512.19643 triggers a solver from a residual-derived
  signal. Our delta is a *production* solver, wall-clock accounting charged in
  full, and the warm-started rejection. That delta is real but narrow.
- *Fragility — and this one is verified and serious.*
  `results/control/fallback_cost_smoke.json` carries verdict **"INCONCLUSIVE —
  the full-domain solve DIVERGED on every case (0 accepted by the no-harm
  gate)"**. The in-repo 128² local fallback is not a usable expert. C2 therefore
  *must* run on the OpenFOAM backend, which means it runs on paper 2's
  infrastructure and re-opens the paper-1/paper-2 split (§4b). Budget that as a
  cost of the candidate, not a detail.
- *Second verified fragility.* `ClassicalFallback`'s `openfoam` backend is
  `NotImplementedError` on `main`, and paper 1 **states so in the manuscript**
  (`GOALS.md` hard constraints). C2 must land on a branch, and whoever resubmits
  paper 1 must check that sentence is still true of `main` at submission time.

---

## 3. The kill list

Recorded so these are not re-proposed. Each has its reason.

| Killed | Reason |
|---|---|
| **A predictive law for the warm-start saving** (regress iteration saving on seed error, "provably reduces iterations under a consistent operator") | **Two candidate mediators have already been falsified by direct measurement on this project's own data**: magnitude (`nf_proj_fix`, gradient error 1254% → 55% moves Cd_v the *wrong* way) and smoothness (`nf_proj_smooth`, matches the working seed on both diagnostics and still lands at −31%). `PLANS.md` §0.03 states the causal chain does not survive. A third mediator at n=13 is not supportable, and this is precisely the "depends on a result we know is fragile" test |
| **A learned preconditioner / initial guess inside the pressure Poisson solve** | Requires C++ inside OpenFOAM. Infeasible on this schedule with this toolchain, and the feasibility failure is certain rather than probabilistic |
| **AirfRANS → own-mesh transfer as a headline** | Already demonstrated and already spent: `nf_bl` *is* positive zero-shot transfer from an AirfRANS-trained Transolver to our own C-grids, at +18.4% on Cd_v, 13/13. Re-proposing it is re-labelling paper 2 |
| **"Use the solver's own residual to certify the solver-warm-started solve"** | Trivially true by construction. Not a finding |
| **A trust-ranked CFD benchmark (G4 in `whitespace.md`)** | RealPDE/NeurIPS 2026 publishes results 2026-11-10 with a "Safe Prediction Score" scoring coverage and tightness. Building a benchmark into that window is a losing race. Report the *finding* (physics-free σ ties the residual on field error, loses on drag) and cite RealPDE as convergent evidence of demand |
| **A standalone V&V-mapping paper (Design C / G3)** | Killed in `strongest_design.md` §2 and it stays killed: no measurement, and "what did you measure?" is the exact axis that desk-rejected this work twice. Keep the vocabulary as one subsection |
| **3-D, DrivAerML, a second turbulence closure, a second solver** | Each is the right long-term move and none fits in a month on one 4070 Ti. `whitespace.md` §3 already declines them and nothing has changed |
| **Re-deriving or repairing theorem leg (iii)** | An upper bound sold as a lower bound, numerically vacuous on a 47000×65500 Jacobian. `FINDINGS.md` F2: delete, do not repair |
| **Any candidate resting on the operator-provenance *attribution*** | Withdrawn 2026-09-07 by `floor_cloud_decimation.json`: at fixed `h`, thinning the cloud raises the floor in 16/16 cases, so provenance is refuted as the sole mechanism. Anything that needs "the floor is provenance" as a premise is building on sand |
| **Anything whose success criterion is "the certificate gets tighter when the floor goes away"** | Measured and **backwards**: exact floor removal widened the bound 7.4× → 11.3×, 3/3 seeds (`mechanism_decision.md` §4). This is why C8's G3 is a bonus and not a gate |
| **Any gate of the form "descent from a perturbed start improves error on ≥N% of cases"** | The inconsistent operator — the one we are trying to beat — already does this in **18/24 cases at ~60% improvement**. Any threshold near 75–80% is a null in disguise. See §2.1's G2 discussion |
| C6, C10, C11 as standalone papers | Demoted to sub-claims (of C1, of paper 2, and of C1 respectively). None carries a paper; C6 additionally has DD-RNO sitting in its slot |
| C7 | **Not killed — reserved.** Real and plausible, but its 2-day gate competes for the same week as three better ones, and σ trails the residual by only 0.06 AUROC, so the margin it needs may not exist. Do not start before the three gates return |

---

## 4. Decisions the author must make, which this document cannot make for them

**4a. The covariate baseline cannot headline two papers.** ρ = 0.874 is
simultaneously (i) the strongest new result in the diagnostic paper's force
section and (ii) the entire premise of C1. This is the same shape as the paper-1
/ paper-2 collision in `strongest_design.md` §8 and needs the same treatment: an
explicit allocation decided **now**, before anything is written, not after. The
cleanest split is that the diagnostic paper reports the baseline as a *control on
its own numbers* (one paragraph, scoped to its own arms), and C1 owns the
*protocol claim* across benchmarks and splits. If the author would rather have
one strong paper, merging is defensible — but decide before the corpus exists.

**4b. C2 re-opens the paper-1/paper-2 split.** `strongest_design.md` §8
recommended "paper 1 first, and paper 1 makes no speedup claim". C2 *is* a
speedup claim and it runs entirely on paper 2's machinery
(`solver/surrogate_seed.py`, `solver/scoring.py`, `scripts/certificate.py`, the
13-case corpus). Choosing C2 means either paper 2 ships first or the boundary
between them is redrawn. That cost belongs in the decision, not in the discussion
section of whatever gets written.

**4c. Is the 7-in/4-out freeze still binding?** `GOALS.md` freezes the channel
spec "while [paper 1] is under review". Paper 1 was desk-rejected 2026-09-07 and
is being reframed, not currently under review. If the freeze is released, C6 gets
materially cheaper; if the author prefers to keep `main` stable for resubmission,
C6's surface head must live outside the frozen grid registry as a point model
(which the Transolver and MeshGraphNet backbones already do, so the path exists
either way).

**4d. Build or adopt, for C8's evaluator.** DiFVM (arXiv:2603.15920) exists,
is differentiable, runs on unstructured meshes and integrates with OpenFOAM. Using
it as both generator and evaluator may be cheaper than writing A2, at the cost of
moving off `simpleFoam` and off paper 2's validated rig. Decide during the 3-day
gate, not in week 2.

**4e. Venue calendar.** C8 and C1 are both best served by NeurIPS D&B or TMLR,
both APC-free. TMLR has rolling submission and no novelty bar; D&B has a deadline
that must be checked against the calendar before it is planned around. Computers
& Fluids remains the journal home for the diagnostic paper and is not in tension
with either.

---

## 5. The single recommendation

**Ship the diagnostic paper, but do not spend the month on it. Spend the month on
C8 — after running all three gates first, which costs under one working week.**

**On the diagnostic paper.** Finish it and send it to Computers & Fluids. Its
spine after this week — the floor's non-decay (0/16 decaying, ×2.56), the 24/24
residual-descent divergence from the exact ground truth with no network in the
loop, AUROC 0.952 on worst-decile drag error beating a physics-free baseline with
CIs excluding zero, and MMS observed order 2.06 on the monitor itself — still
clears C&F's rigor bar comfortably. The week's damage is disclosure-level, not
fatal: the mechanism *attribution* is withdrawn but the measurement stands; the
floor-subtraction causality is backwards but the concession is itself a result;
0.839 was never a ceiling but the lift/drag observability asymmetry that replaced
it is stronger and covariate-controlled. It is roughly 80% paid for, the
corrections in `mechanism_decision.md` §6 are a list of specific sentence
replacements rather than a rewrite, and finishing it converts a demoralising week
into a submitted artifact. What it cannot be is the month's *investment*: its
ceiling is stated without flattery in `strongest_design.md` §10, and this week
lowered it.

**On the month.** Run the three gates before committing anything:

| gate | cost | what it decides |
|---|---|---|
| **C2's correlation** | 1 day, data on disk | whether the hybrid composes or fights itself |
| **C1a's full-parameter regression** | 2–3 days, no training | whether any in-distribution force metric on AirfRANS is informative — and whether C6 has headroom |
| **C8's G1/G2a/G2b/G3 on ~20 solves** | 3 days, under an hour of compute plus plumbing | whether the audit boundary actually crosses |

Under one working week, all three, and every one of them is the structure that
killed three dead ends last week rather than a month.

**If forced to state a prior before the gates return: C8.** Three reasons.

1. **It is the only candidate whose defined-positive outcome is genuinely
   positive.** C1's likeliest strong effect is a demolition; C2's ceiling is "a
   careful systems measurement of two known things composed". C8's headline is
   *here is the thing that makes it work*, and it is the exact converse of
   everything paper 1 lost.
2. **It is independent of every fragile result.** It needs paper 1's negative
   only as motivation, and none of the four claims that eroded this week —
   provenance attribution, floor-subtraction causality, the 0.839 ceiling, the
   MMS null's fine-end behaviour — is load-bearing for it. After a week in which
   six claims fell to measurement, a candidate that cannot be damaged by any of
   them is worth a premium that a raw probability does not capture.
3. **It has the highest ceiling and the lowest compute risk.** The corpus is
   hours of CPU; the month is plumbing, and two of the five pieces are ports of
   code that already exists and is tested. `strongest_design.md` §10 named "a
   dataset that ships its operator" as one of four things separating this work
   from a landmark and declared it out of reach. It is the only one of the four
   that is not.

**The two honest caveats.** G2 is a real gate: if descent under the solver's own
operator does not stay put at its own converged solution and the residual does not
become a valid iterate selector, the boundary does not cross, the headline is
gone, and what remains is a dataset release with a null — a thin month. That is a
genuine ~20% branch and the author should know it before starting, not after. And
DiFVM's authors are one release away from the same corpus. Both are exactly why
the gate costs three days and not three weeks.
