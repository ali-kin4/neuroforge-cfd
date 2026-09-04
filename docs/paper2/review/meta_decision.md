# Editor's meta-decision — Paper 2

**Manuscript:** "A perfect flow field on a raster is worth no initialisation:
separating representation, region and accuracy in RANS warm starting"
(A. Jabbary, K. Ghanavati)

**Target venue:** Computers & Fluids (Elsevier; subscription route)

**Panel:** R2-brutal — reject, 3/10 · fair — major revision, 5/10 ·
domain expert — major revision, 4/10

**Basis of this decision:** the revised `docs/paper2/DRAFT.md` as it stands on
2026-09-04, the three reports, `docs/PLANS.md` §0.001, and my own spot-checks
against `results/decomposition.json`, `results/depth_corpus2.json`,
`results/placement2.json`, `results/depth_perturb.json` and
`results/mesh_verification.json`. I did not take the authors' close-out on
trust; every credited fix below I confirmed is in the current draft, and the
items in §5 are ones I confirmed are *not*.

---

## 1. DECISION: **MAJOR REVISION**

Not reject. Not minor. Major revision at the low end of that band: **no new
solver campaign is required.** Every remaining blocker is post-processing of
histories already on disk, one optional five-case arm, or editorial
restructuring.

---

## 2. What the panel actually agreed on, and what the revision did about it

Strip the three reports of their individual style and they made **one**
load-bearing charge in three voices:

> The contrast that carries the title is not a controlled contrast, and it was
> never run on the case set where the study has statistical power.

R2 called it W1 + W2 and rejected on it. The domain expert called it M1 + M2
and made it blocking conditions 2 and 3. The fair reviewer reached it from the
other side (W1: the ladder is reported for one family only). This is the
consensus, and consensus on substance decides.

**The revision answers it, and the answer is real.** I verified
`results/decomposition.json` against §5.2.3 line by line: `oracle_bl` — the
missing mesh-native, boundary-layer-masked, four-channel control the domain
expert demanded in M1 — was run, and the whole ladder was re-run on the
thirteen-case corpus. Every link now moves one variable, on the row the paper's
own readability rule admits, paired within case, with an exact sign test that
can reach p = 0.0002:

| what moves | effect on `C_d,v`@1% | cases | p |
|---|---:|---|---:|
| representation, raster | −90.2 [−96.3, −84.7] | 0/13 improve | 0.0002 |
| accuracy | −54.5 [−58.9, −49.7] | 0/13 | 0.0002 |
| region | −20.7 [−24.2, −17.6] | 0/13 | 0.0002 |
| representation, body-fitted | +6.7 [−2.6, +14.8] | 10/13 | 0.09 (null) |

That is the paper now. It is a legitimate primary contribution for a methods
journal, and it did not exist when the panel reviewed.

I also confirmed, and credit:

- **W1 (headline and contrast on different quantities).** Closed. Headline and
  contrast are both `Cd_v`@1% at n = 13; the wall-fitted projected arm the
  corpus lacked (`or_proj_coarse`) is there at +79.6%, 13/13.
- **W2 / fixed-point confound.** Closed by §5.9. The matched-norm smooth
  perturbation is a genuine control — 12.2 points against the raster's 82.4,
  non-overlapping intervals, and the authors correctly state what it does *not*
  establish. This was R2's "fatal-adjacent" and it is now answered by
  measurement rather than argument.
- **F1 (the "1.8%" characterisation).** Closed, and better than the domain
  expert asked. `results/placement2.json` now reads `or_proj_fine` at 8.1% in
  `u` and 23.1% in `nut` inside the layer (the reviewer's 28.3%/40.0% came from
  the pre-fix `placement.json`), and §5.2.1's "three honest qualifications" box
  concedes the scalar-vs-state distinction, the growth-ratio confound, and the
  structural no-op — all three, unprompted, in the authors' own words.
- **W3 / M4 (the pressure mechanism).** Withdrawn in §5.2.2 with the three
  objections spelled out, including the one that hurts most (the winner is
  −116.1% on `C_d,p` too). Correct call.
- **W5 ("representation, not accuracy").** Withdrawn, because their own new
  control contradicts it. This is the single most creditable act in the
  revision: they built the control that killed their own title.
- **W6 / M3 (grid sequencing charged wrongly).** Re-charged at the coarse
  solve's own force band — 53 fine-equivalent iterations, not 1486 — and the
  conclusion *reversed against the authors*. §5.7 now states the classical
  method wins on both axes. Exactly what all three reviewers asked for and the
  opposite of what the incentives point at.
- **W8 / journal-standard verification.** §3 now carries measured y⁺ (median
  0.59–0.63, max 1.12 — I checked `mesh_verification.json`), a two-level grid
  comparison, and a literature check on `C_d`. Adequate. The observation that
  the headline quantity is the least mesh-sensitive and the demoted one the
  most is a good independent corroboration of §4's readability rule.
- **W10 (wall-clock on an unreadable row).** Re-scored on `Cd_v`: +9.0%
  end-to-end, and they state that moving to the readable row cost them two
  thirds of their own number.
- **W11 (gate direction).** Per-seed verdicts added; the gate admits the
  recommended seed 5/5 on `Cd_v` and rejects it 5/5 on `Cd`, and §8 reports the
  failure rather than only the success.
- **M5 / M6 (degenerate rows, saturated-regime vacuity).** Both conceded
  explicitly, in the sections that make the claims, not in a footnote.

That is eleven substantive items closed, several of them by running the
experiment that damaged the authors' own position. R2's report contains its own
escape hatch — *"the honest paper hiding inside this one is narrower and, I
suspect, publishable"* — and the current draft is close to being that paper.
Rejecting now would punish compliance and I will not do it.

---

## 3. Adjudicating the disagreements

**R2's reject is not sustained, and it was the correct review.** Every strength
R2 conceded is real, and W1/W2 were correctly identified as the load-bearing
defects. But R2's own remedy has been executed. A reject verdict does not
survive its own repair list being completed. Weight: high on diagnosis, spent
on verdict.

**Where R2 overreached:** W9's implication that rule 8 was an outcome-motivated
amendment. Rule 8 (a converged run is finished, not truncated) is a correct
scorer fix that penalised fast-converging arms; the domain expert independently
called it correct and non-obvious. Disclosure is the right treatment here and
the authors gave it. Not a blocker.

**The domain expert is weighted heaviest on correctness and he is largely
satisfied.** He checked `placement.u_plus` against Spalding's law, checked
`friction_velocity`, checked `iterations_to_force_band` implements
enters-and-stays rather than first-crossing, and found them right. That
independent code-level verification is worth more than any of the three
verdicts, and it means I do not have a correctness worry about the apparatus.
His remaining open items are §5 below.

**The fair reviewer's W1 is the one item the panel raised that the revision
did not touch, and I am charging it.** See §5.

**Discounted as noise:** R2's W7 objection to bootstrap CIs at n = 5 is
technically right but now largely moot, since the load-bearing claims moved to
n = 13 with an exact sign test. The fair reviewer's W13 (too few figures) is
correct for C&F but not a blocker — three figures for a paper this length is
still thin and I have folded it into §5 as a should-fix.

---

## 4. Answers to the five questions put to me

**(1) Is the contribution now sufficient for C&F? Yes — but not the one the
authors think they are selling.**

The +18.4% speedup is not the contribution and must stop being presented as the
headline result. It is a modest number, it is beaten by the classical baseline
in the same paper, and — as the domain expert noted and the draft still does
not concede — it sits below the closest prior art (Zhou et al., 11×/16× on a
force-error metric with a mesh-native seed on a comparable wall-resolved
configuration). If this paper's claim to C&F rested on the speedup it would be
below bar and I would be writing a different decision.

What is above bar for C&F is the combination of:

- the n = 13 one-variable decomposition, which prices representation, region
  and accuracy separately for the first time and is a design constraint on
  every surrogate intended to precede a solver;
- §4's scoring protocol, which all three reviewers independently said they
  would cite on its own and which the domain expert said he would accept as a
  shorter paper by itself;
- the classical baseline run honestly and reported as a loss;
- the closed-form pre-flight criterion, reduced to what it is (a fidelity
  diagnostic of a format, not a forecast of a solve) and shipped as a tool.

C&F is a methods-and-verification venue, not a leaderboard. "A criterion plus a
well-controlled negative result" is a C&F paper. **A modest speedup does not
sink it. Presenting the paper as a warm-start-recommendation paper would.**

**(2) Is it over-hedged? In substance, no. In staging, badly — and this is now
the single largest presentation risk.**

Blunt: every one of the five withdrawals is forced by the authors' own data and
every one is correct. This is not a study that established nothing. It
established a well-powered decomposition, a verified null, a falsified
mechanism and a transferable protocol. Do not remove a single withdrawal.

But the reader meets the withdrawals *before* the findings. Two are in the
abstract. The contributions list is ordered "by what we think survives", which
telegraphs that some did not. The conclusion's final paragraph is a confession.
The fair reviewer named this precisely — *"honesty is an asset; autobiography
is a liability"* (W8) — and it is the one substantive editorial item that went
unaddressed; with five withdrawals it is now worse than when he wrote it. I am
charging it as an unaddressed reviewer item, not as taste. A reviewer who reads
the abstract and the contributions list and stops will conclude the authors
lost confidence in their own paper, and that reader would be wrong.

The fix costs nothing and removes 15–20% of the length: withdrawals stay in the
paper, consolidated into one clearly-labelled "Corrections and withdrawn
claims" subsection, and leave the abstract, the contributions list and the
conclusion.

**(3) Does the decomposition carry the paper? Yes — and it should lead it.**

It is the only claim in the paper made at n = 13, with one variable moved,
paired, with a test that can reach significance. Everything else is n = 5. It
is not "we ran controls": it is a quantitative pricing of three independently
settable properties, with a null in it that the authors could have quoted as a
gain from the five-case tree (+16.2, 5/5) and did not, because thirteen cases
said otherwise. That single decision is the strongest evidence in the file that
these numbers can be trusted.

It does not have a mechanism, and the paper is right not to invent one. A
decomposition with a verified null and a falsified mediator is a finding. What
is missing is not a mechanism but *structure*: §5.1 should be the
decomposition, with +18.4% as a row inside it. At present the weaker result
leads and the strong one is buried at §5.2.3.

**(4) Remaining fatal flaw? No — but there is one defect no reviewer caught,
because the decomposition postdates the reviews.**

**The two representation contrasts are not baselined on the same arm.** The
raster contrast is `oracle_mesh` → `cartesian_128` (whole-field stratum,
−90.2). The body-fitted contrast is `oracle_bl` → `or_proj_coarse`
(boundary-layer stratum, +6.7). Each *link* moves one variable and that is
sound. But the paper's headline sentence — bolded in the abstract, in Highlight
3, and made the pivot of the conclusion ("the same 16,384 values, and no
measurable cost" against the raster's 90.2) — compares the two *across strata*.
That cross-stratum comparison is not itself one-variable, and the paper never
says so. It is the same species of error R2 charged the first time, and it now
sits inside the paper's best result.

Compounding it: the two grids also differ tangentially — the raster is
Δx ≈ 3/128 ≈ 0.023 chord along the surface, the body-fitted grid ≈ 0.008 chord
— which is the domain expert's `n_s` point, unaddressed, now living inside the
headline row. §6 attributes the whole difference to y⁺.

**Neither is fatal.** The gap is 97 points, far beyond any plausible stratum
interaction, and the authors already have the tangential control without
knowing it: §7.1's ladder shows a 421² raster — Δx ≈ 0.007 chord, tangentially
*finer* than the body-fitted grid — is still flat and negative. That connection
is not drawn anywhere in the draft. Drawing it converts a hole into a control.

**(5) Venue.** C&F is the right first choice on fit, on the no-APC constraint
(hybrid; confirm the subscription route at submission), and on bar. Honest
fallback if C&F declines: **International Journal for Numerical Methods in
Fluids** (Wiley, hybrid, subscription route available) — it publishes
initialisation, acceleration and verification methodology at a bar this clears
comfortably. Do not send this to JCP or CMAME: a withdrawn mechanism plus a
conceded loss to grid sequencing will not clear either. Do not send it to
Computer Physics Communications — it ships a CLI but it is not a software
paper. Avoid the OA-only venues in this space (EACFM, Data-Centric
Engineering) on the APC constraint.

---

## 5. Required for the next stage

**Blocking — reviewer items still open.**

- **B1. Report the network placement arms (fair W1).** `nf_proj_fine` and
  `nf_proj_half` do not appear anywhere in the current draft; I grepped. To be
  precise about why this still blocks: §5.2.1 no longer carries a load-bearing
  claim — the decomposition does — so the *result* does not depend on these
  arms. It blocks because this paper trades on completeness of reporting, and a
  two-family ladder reported as a one-family ladder is the one norm it cannot
  be seen to break. Report the full oracle × network ladder on both readable
  rows. The reviewer is right that this *strengthens* the negative: the sign of
  the placement effect flips between families and neither effect is large,
  which is better evidence that placement does not control convergence than
  either family alone.
- **B2. First entry alongside last exit, with traces (domain expert M2b,
  his blocking condition 3).** `iterations_to_force_band` is a last-exit
  statistic. A seed that starts near the answer, is pushed out of the band by
  the solver's own transient and returns late scores catastrophically while
  never being far from the answer. This is unaddressed and it is free —
  post-processing of histories on disk. Report both statistics for every arm in
  §5.2/§5.2.3, plus `C_d`(iteration) traces for cold / `oracle_mesh` /
  `or_proj_coarse` on one case.
- **B3. A causal stopping rule (fair W2).** `iterations_to_force_band` is
  defined against a reference that requires the converged answer. For a C&F
  readership this is the first question asked. Re-score the corpus under a rule
  using only run-time information (trailing-window peak-to-peak on `C_d,v`),
  and report saving *and* accuracy at stop for cold against `nf_bl`. If the
  saving does not survive, report that — it is an important negative for the
  whole warm-start literature and this author has already shown he will publish
  one.
- **B4. Baseline-match the representation contrasts (my finding, §4 above).**
  Either run `cartesian_128` boundary-layer-masked on the five-case tree, or
  state the stratum mismatch explicitly wherever the two contrasts are
  compared. And wire §7.1's 421² row into §5.2.3 as the tangential-resolution
  control.

**Blocking — provenance and internal consistency (W12 is claimed swept; it is
not).**

- **B5. §3's tree table is wrong in three places, and one of them mis-indexes
  the new headline result.** It lists `placement2` as "used in §5.2.1, §5.2.3",
  but §5.2.3's own header now reads "Tree: `corpus` (13 cases, 5 arms)". The
  `perturb` tree carrying §5.9 — the entire answer to R2's fatal-adjacent W2 —
  is **absent from the table of "seven trees" altogether**. And `repr3` is
  listed at 6 cases against "five cases" in §5 and §5.3. In a paper whose rule
  is that every number names its arm set, the provenance table is the worst
  possible place for this.
- **B6. Commit `results/placement2.json`.** It is uncommitted in the working
  tree, and §5.2.1's corrected 8.1%/23.1% figures — the numbers that answer the
  domain expert's F1 — depend on it. The Data Availability statement points at
  a tag. Commit before tagging.

**Required, not blocking.**

- **B7. Restructure §5 so the decomposition leads** and +18.4% is a row inside
  it, not a separate headline (§4(3) above).
- **B8. Consolidate the withdrawals** out of the abstract, the contributions
  list and the conclusion into one labelled subsection (§4(2) above). Nothing
  is deleted.
- **B9. Fix the unit mismatch in the grid-sequencing comparison (domain expert
  M3, second prong).** §5.7 charges the classical baseline in fine-equivalent
  iterations; §9 charges the learned seed in end-to-end seconds. I am *not*
  treating this as blocking, because §5.7 has already withdrawn the price claim
  outright — the mismatch now props up a concession rather than a claim, and a
  conceded loss stated in the wrong unit does not mislead a reader in the
  authors' favour. But **Contribution 7's "better than the learned seed on both
  axes" is still a two-unit sentence and must go.** Either price both in
  seconds (re-measure the sequencing tree exclusively, since §10 currently
  forbids quoting wall-clock from it), or state the workflow argument with no
  implied price comparison at all.
- **B10. Fix §2.1's novelty position on Zhou et al.** "Consistent with, not
  contrary to, what we measure" reads as parity when the closest prior art
  reports 11×/16× on a force metric with a mesh-native seed on a comparable
  configuration. Supply the section/table citations the domain expert asked
  for, and state the capability position honestly in one sentence: this paper
  is below the closest prior art and below the classical baseline on speed, and
  its contribution is diagnostic and methodological. Saying that plainly costs
  nothing and forecloses the objection.
- **B11. Demote the total-drag figures in §1.** Leading the introduction with
  −548% / −182% on a quantity the paper's own protocol declares unreadable at
  n = 13 is disclosed, but it still invites exactly the charge R2 made. Lead
  with the readable contrast; keep the total-drag numbers, second.
- **B12. Add the two physics figures the fair reviewer asked for** (W13):
  `u⁺(y⁺)` as sampled by each representation with the mesh's first cell marked,
  and the convergence histories with bands drawn. Three figures is thin for
  C&F, and both of these carry arguments the prose currently has to make alone.

---

## 6. Honest probability estimate

- **Submitted today, unrevised, to C&F:** ~30–35% acceptance. The apparatus and
  the decomposition would carry it past most reviewers, but a reviewer of R2's
  disposition would find B1, B2, B3 and B5 unaddressed and return another major
  revision; the staging problem (§4(2)) raises the chance of a
  "sound-but-unconvincing" reject from a reviewer who reads the abstract as an
  admission of failure.
- **After a competent revision addressing B1–B12:** ~60–65% at C&F. The
  decomposition is genuinely new, the protocol is genuinely transferable, and
  the honest classical baseline is the kind of result C&F's readership values.
  The residual risk is a reviewer who declines to accept a paper whose own
  practical recommendation loses to `mapFields`.
- **Fallback, IJNMF, after the same revision:** ~75%.

**Do not read the first number as a licence to submit now.** B1, B2, B3 and B5
are post-processing and editing; they cost days, not months, and every one of
them is an objection a reviewer has already written down once. Submitting
before they are done spends a referee round to be told what is already in this
file.

The rigour and the honest limitations are **positive** signals in this file and
I have weighted them as such. They are also, at present, presented in a way
that works against the authors. Fixing that is free and is worth more to this
paper's fate than any remaining experiment.

---

*Filed 2026-09-04.*
