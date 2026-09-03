# Review — *A perfect flow field can be a bad initial condition: representation, not accuracy, decides neural warm starts for RANS*

**Venue:** Computers & Fluids
**Reviewer role:** senior reviewer, balanced/constructive
**Recommendation:** **Major revision**
**Score:** **5 / 10** (defensible range 5–6; a 7 is reachable with the five changes in §6 below)

---

## 0. The one-sentence verdict

After the paper's own withdrawals are taken at face value, the defensible paper is:

> *Fields stored on a grid are unusable as RANS initial conditions, no matter how accurate; mesh-native seeds give a modest but reliable and well-controlled gain; and the reason is **not** the near-wall state, which is where everyone (including the authors) would have looked.*

That is a real, publishable contribution and it is smaller than the paper as currently framed. The closed-form near-wall criterion (§6), which occupies roughly a third of the manuscript and four sentences of the abstract, is explicitly disowned as a predictor of convergence in §6.7 and §10 and should be re-weighted to match. Everything I flag below follows from that single mismatch between what is claimed structurally and what survives empirically.

I want to be clear at the outset that I found no fabrication, no metric shopping, and no unsupported headline. I re-derived the corpus headline from `results/depth_corpus.json` myself (`Cd_v@0.01`: `nf_bl` = +18.38%, 13/13 positive, worst case +2.7%, oracle control +93.6%, Cartesian control +3.4%, `settled_spread` = 0.001 against a 0.005 limit) and it is exactly as reported. This is unusually careful work.

---

## 1. Summary (steelmanned)

The paper asks whether a neural surrogate can be used as an initial condition for a production RANS solve, and answers with a control that the warm-start literature has not run: it removes the network entirely and hands the solver **its own converged answer**, differing only in how that answer was stored. Read at the solver's cell centres, the converged field saves 92.0% of a cold solve. Stored first on an equal-budget wall-fitted grid, it *costs* 181.6% (95% CI −318 to −45, 1/5 wins). Stored on a uniform 128² raster, 548%. Since the field is exact, no property of any network — accuracy, architecture, training — can be responsible. That single contrast is the paper, and it is clean.

The paper then does three further things that are individually worth publishing:

1. It supplies a **parameter-free, pre-flight, solver-free criterion** for what a given output format does to the near-wall state (`G = u⁺(y₁⁺)/u⁺(y_c⁺)`), validates it as a conservative upper bound across a fifty-fold range of first stations and 3.5 decades of Reynolds number, and ships it as a CLI tool. The criterion's counter-intuitive prediction (lower Re is *worse* on the same mesh) is checked and holds.
2. It **falsifies its own proposed mechanism three separate ways** — by grading, by wall-law repair, by smoothing the repair — having pre-registered the prediction in `docs/protocols/placement_prediction.md`, and reports the negative result as the finding it is. It then locates the surviving candidate (the pressure field) by decomposition rather than by assertion.
3. It measures the thing this literature almost never measures: a **classical baseline** (grid sequencing) with its coarse solve charged, and reports that the classical seed is *better* than the learned one.

Layered on top is a genuinely useful methods section (§4) in which eight scoring rules are each justified by a sign flip they prevented on this project's own data, a deployability gate with an arithmetic worst-case bound `(1 + K/N)·cold`, an end-to-end wall-clock accounting, and a limitations section that is longer and more specific than most papers' results sections. The corpus headline (+18.4% on viscous drag, 13/13, p = 0.0002) is supported by a positive control at +93.6% and a null negative control — a control structure I wish were standard.

The significance for C&F's readership is real: if the representation result holds, it is a design constraint on every surrogate that hopes to be used ahead of a solver, and it arrives just as the field standardises on pretrained general-purpose aerodynamic models whose output format is chosen once and inherited by everyone downstream (§2.3, ref [24]).

---

## 2. Strengths

**S1. The accuracy confound is removed, and that is the right experiment.** Handing the solver its own converged field through different representations is the control the warm-start literature has been missing. It converts "our seed did not work" from a statement about a network into a statement about a format. The 274-point swing on one field is the paper's durable result and it needs no network to state.

**S2. Control discipline throughout.** Every experiment carries a converged-field oracle that must post a large positive saving before any other arm is read (§3), and the paper reports that it discards experiments whose control fails. The corpus additionally carries a *null* negative control. Combined with the readability rule (§4 rule 5), this is a stronger internal-validity apparatus than I normally see in this area.

**S3. §4 is a genuine contribution, not padding.** Rules 1–8 are each attached to a specific sign flip on real data (dropping non-finishers turning −199.4% into −31.2%; grading the oracle against itself turning +73.5% into +1.0%; `nNonOrthogonalCorrectors` desynchronising the pressure history 3:1). Rule 8 — that OpenFOAM's early exit on `residualControl` makes a fast-converging arm look truncated, penalising exactly the arms that win — is a trap I expect many published warm-start numbers have fallen into. I would cite this section independently of the rest of the paper.

**S4. The residual/force dissociation is important and is stated against the authors' interest.** §5.6's table — on the *exact converged field*, so it cannot be blamed on the surrogate — shows the two arms best on residual (+40.6%, +36.6%) are the two worst on drag (−181.6%, −266.8%), while the recommended arm is the worst on residual (−80.5%) and the best on drag. Given that much of the cited literature (e.g. [13]) validates on residuals alone, this is a finding with consequences beyond this paper. The authors keep the damaging residue in the text ("on the residual, `nf_bl` is worse than a cold start") rather than burying it.

**S5. Reporting negative and withdrawn results.** The withdrawn +41.8%, the withdrawn "13%-accurate" closed form (traced to a nearest-vertex wall-distance bug), the re-measured Reynolds table, the falsified extrapolation explanation for condition 2, the falsified divergence explanation, the falsified pressure-seeding strategy (§7.2), the wake bound at +0.5% (§7.3), and the admission that the gate fails on lift (§8). Each is reported with the number that replaces it. This is the behaviour the field claims to want and rarely rewards; I am rewarding it here.

**S6. The classical baseline, honestly scored.** Running grid sequencing and reporting that it makes a *five times better seed* than the learned one (+75.9% vs +14.6% on `Cd_v`) is the opposite of what a paper is incentivised to do. The reframing that follows — "the learned seed's advantage is price, not quality" — is a more useful sentence for a practitioner than another speed-up ratio. (See W3: I think the conclusion drawn from it is not yet supported.)

**S7. Reproducibility.** Appendix A maps every result to one script and one committed result file; Appendix B names the scoring functions and the unit test that guards each. Re-scoring any tree at any depth is one command that also prints its own readability verdict and declared arm set. I spot-checked three tables against the JSON and they matched.

**S8. Falsifiable, cheap, transferable side-results.** The Reynolds transfer (§6.6) is obtained at zero compute from existing solves and makes a prediction opposite to practitioner intuition. The wake bound (§7.3) turns a scope limitation into a measured statement. The `(1 + K/N)` bound in §8 is arithmetic, not statistical, and correctly identified as such.

---

## 3. Weaknesses, ranked

### W1 — Placement arms that do not reverse are omitted from the draft. **Major (blocking).**

The paper's fourth contribution is "three independent demonstrations that the near-wall state is not the mediator", and the first of them is the placement ladder (§5.2.1). The draft reports that ladder for the **oracle** family only (`or_proj_coarse/fine/half`) plus `oracle_mesh`. The corresponding **network** arms exist — `results/depth_placement.json` contains `nf_proj_coarse`, `nf_proj_fine`, `nf_proj_half` — and on the readable `Cd_v@1%` row they read **+14.5% / +17.7% / +15.7%**. Correct placement is *mildly* better there, and non-significantly so (per-case spreads .105–.185 against .134–.215 overlap), but it is in the direction the mechanism predicts and it does **not** reverse, whereas the reported oracle family reverses strongly (+86.1% coarse against +69.1% fine). The strings `nf_proj_fine` and `nf_proj_half` do not appear anywhere in the manuscript. `docs/PLANS.md` §0.05 previously headlined this same ladder as "PLACEMENT — confirmed, exactly as pre-registered."

I am not claiming a suppressed effect — the network effect is small and not resolvable. I am claiming that a two-family ladder is reported as a one-family ladder, and that the omitted family is the one whose sign does not support the headline negative. In a paper whose entire credibility rests on completeness of reporting, that cannot stand. I also note the oracle family *is* the cleaner contrast (no accuracy confound), which is a good reason to lead with it and not a reason to omit the other.

*Fix, and it is a gift rather than a penalty:* report the full 2 × 3 ladder (oracle × network, coarse/fine/half) on both readable rows. Then make the point that is actually strongest — **the sign of the placement effect flips between the oracle and network families and neither effect is large**, which is better evidence that placement does not control convergence than either family alone. State also whether the earlier `depth_placement` scoring (PLANS §0.05: `or_proj_coarse` +41.7% vs `or_proj_fine` +62.2% on `Cd_v`, non-overlapping CIs, i.e. the *opposite* oracle ordering to the draft's +86.1% vs +69.1%) is superseded by the wall-distance fix; if so, add it to the withdrawal list, which the paper is otherwise scrupulous about.

### W2 — The headline metric is not measurable in deployment. **Major.**

`iterations_to_force_band` is defined against a reference that is a median over *settled arms* — that is, against knowledge of the converged answer. A practitioner starting a warm solve does not have that. They stop on a residual criterion, or on a causal plateau test, and §5.6 establishes that the recommended arm is **worse than cold on the residual at every depth**. So the paper's central practical claim ("+18.4%, 13/13") is a statement about a quantity the intended user cannot observe, obtained on a metric where the recommended arm wins, while on the metric they *do* use it loses.

The paper defends the choice of metric well (rules 1–2, pre-commitment, §5.6). That defence answers "is this metric honest?" but not "is the saving realisable?".

*Fix, and it is cheap — post-processing of histories you already have:* define a **causal** stopping rule that uses only information available at run time — e.g. "stop when the peak-to-peak variation of `C_d,v` over a trailing window of W iterations falls below b, for W = 100, 200" — and re-score every arm in the corpus under it. Report saving and *final accuracy at stop* (how far the stopped value is from converged) for cold vs `nf_bl`. If `nf_bl` still wins, this becomes the paper's strongest practical claim and materially raises my score. If it does not, say so; the paper's own ethos makes that reportable, and it would be an important negative for the whole warm-start literature.

### W3 — The grid-sequencing conclusion rests on a charge the paper itself calls pessimistic. **Major.**

§5.7 concludes "and it still is not worth running here" because the coarse solve is charged at 1486 fine-equivalent iterations against a cold run of 696. But the caveat box concedes two things that both point the other way: the coarse solve ran its **full 6000 iterations** (a practitioner would truncate it), and the mapper is the authors' own, leaving a 6–10× first-cell gradient overestimate where the coarse mesh's placement alone would permit ~2× — so this arm is explicitly a **lower bound** on grid sequencing. The paper acknowledges both and then draws the conclusion anyway. For a C&F readership, whose default warm start *is* grid sequencing, this is the most consequential unresolved question in the paper: the honest current state is "we have not established that the learned seed beats the classical one on price either."

*Fix:* do not assert a reversal — seed quality at a truncated coarse solve is unknown, and I would not accept a back-of-envelope 500-iteration charge either. Instead report the **trade-off curve**: coarse-solve length (say 250 / 500 / 1000 / 6000 iterations, from checkpoints of the existing coarse run) → resulting `Cd_v` saving on the fine mesh → total charged cost. Five cases, one arm, existing machinery. Then state the crossover honestly. Either outcome is publishable; the current unsupported "still not worth running" is not.

### W4 — Internal inconsistency in the seed-diagnostic numbers. **Major (easy to fix, but it lands hard given the paper's history).**

§5.5's table gives `nf_proj_fix` roughness **26.2×** converged and `nf_proj_smooth` **5.7×**. §10 ("On the repair", first bullet) says "the repaired seed is **11.1×** rougher along the wall than the converged field against 4.2× for mesh-native". These are the same quantity for the same arm, in the same document, and they disagree by a factor of 2.4. The supporting notes contain a third and fourth value for the same arms (PLANS §0.03: 18.1× and 3.9×; §0.05: 11.1×), and `nf_proj`'s gradient error appears as 877.7%, 1254% and 1583% across documents.

Given that this paper already withdrew a headline number because of a projection-code defect, a reader who spots this will assume the diagnostics were computed under mixed code versions. That is exactly the doubt the paper's transparency is meant to foreclose.

*Fix:* regenerate all seed-diagnostic tables from one run of the fixed code, state in the caption which `seed_gradient_*.json` each came from, and reconcile §10 to §5.5. If §10's 11.1× is a superseded number, delete it; if §5.5's 26.2× is, delete that.

### W5 — §6 is structurally over-weighted relative to what it is allowed to claim. **Major.**

§6.7 and §10 say plainly that the criterion "measures representations; it does not forecast solves" and "nothing follows from it about the speedup". I take the authors at their word — and then note that the abstract spends four of its sentences on the closed form, contribution 2 and contribution 3 are both about it, and §6 plus §7.1 is roughly a third of the manuscript. A reader who reads the abstract and stops will come away believing the paper offers a pre-flight predictor of warm-start success. It does not, by the authors' own account.

Worse, the "veto" framing overstates what the data support. The claim is "every representation that loses the gradient costs the solve, so the check rules formats *out* for free". But within this study, representations that **pass** the check also cost the solve (`or_proj_fine`, `or_proj_half`, both correctly placed, both strongly negative on total drag). There is no measured example of a *grid* format that passes the check and accelerates. The only passing-and-working representation is mesh-native, which passes trivially. So the check has **no demonstrated discriminative power for convergence in either direction**, and the actionable content of the paper reduces to "be mesh-native".

*Fix (any of three, in decreasing order of ambition):*
(a) supply the missing positive control — a graded wall-fitted format that passes the criterion *and* accelerates; that single arm would rescue §6's practical claim outright;
(b) if (a) is unavailable, re-scope §6 explicitly as a **fidelity diagnostic** of representations, move it after §5 or partly into an appendix, cut its abstract footprint to one sentence, and delete "veto"/"rules out" language in favour of "quantifies what a format discards";
(c) at minimum, add one sentence in §6.4 stating that no format in this study passes the check *and* accelerates, so the check must not be read as a screen for warm-start viability.

### W6 — The placement ladder is not the single-variable contrast it is described as, and the statistics understate the authors' own result. **Major (fixable with existing data).**

§5.2.1 states that "only the grading of the grid changes". In `warmstart.clustered_seed`, the stations are `np.geomspace(first, n_max, n_n)` with `n_max` fixed at 1.0 chord, so moving `first` from 2.5·10⁻⁴ to 5·10⁻⁶ at fixed `n_n = 64` also changes the growth ratio from ≈1.140 to ≈1.213 — the arm gains near-wall stations and loses roughly 30% of its stations per octave *everywhere above the wall region*. Three quantities (first station, growth ratio, level count) move across the three arms; the table reports only two of them.

I want to be fair about this: it is **not** an alternative explanation for the ordering, because `or_proj_half` has the coarsest outer distribution of all (growth ≈1.48) and converges *better* than `or_proj_fine`. So the confound does not rescue the near-wall mechanism. But the description is imprecise and a reviewer will spot it.

Separately — and this one *strengthens* the paper — the statistical presentation undersells the result. The marginal CIs on `Cd@1%` overlap heavily ([−318, −45] vs [−419, −114] vs [−333, −34]), so as presented, "the arm at 1.8% converges worse than the arm at 1219%" is not statistically resolved; a sceptical reader will say all three projections are indistinguishably bad. But the arms are run **on the same five cases**, and the per-case values in `depth_placement.json` show `or_proj_fine` worse than `or_proj_coarse` in **5 of 5 cases** (−4.78 vs −4.07, −3.56 vs −2.24, −0.38 vs +0.19, −3.93 vs −2.76, −0.69 vs −0.20). That is a paired result at the n = 5 sign-test floor, and it is much stronger evidence than the marginal intervals.

*Fix:* (i) add growth ratio and stations-inside-first-cell columns to the §5.2.1 table and reword "only the grading changes" to name all three co-varying quantities; (ii) report **paired** contrasts (per-case differences, paired bootstrap, Wilcoxon/sign test on differences) for every arm-vs-arm comparison in §5.2, §5.2.1 and §5.5 — the cases are common to all arms, so this is free and it is the correct analysis; (iii) if budget allows one solver run, add a decoupling arm (`n_n = 96` at coarse's growth ratio, adding near-wall levels without coarsening above) — that is the clean single-variable placement experiment.

### W7 — The mechanism claim ("the damage is to the pressure field") is inferred, never measured directly. **Major.**

§5.2.2 and the conclusion locate the damage in the pressure field. The evidence is entirely indirect: viscous drag survives projection (+69 to +86%), total drag does not (−183.9% on `C_d,p`@0.5%), therefore the pressure field is where the damage lives. That is a reasonable inference, but the paper never reports the one measurement that would make it a finding: **the error in the seed's own pressure field, per representation**. The seeds are on disk; ‖p_seed − p_conv‖ by wall-distance band (first-cell / boundary layer / mid-field / far field) for `oracle_mesh`, `or_proj_coarse/fine/half` and `cartesian_128` costs no solves at all.

This is the single highest-value experiment left undone. If projections show large mid-field pressure error while mesh-native does not — and if `or_proj_fine` is *worse* than `or_proj_coarse` there — the paper converts a negative result into a positive mechanism and recovers most of the ground §6.7 gives up.

*Fix:* add that table (and a contour figure). Also state explicitly which arms hand over `p` at all: `nf_bl` hands over `u, v, nut` and no pressure (§7.2), while the oracle projections appear to carry `p` (`warmstart.clustered_seed` returns `back(p, 0.0)`). If so, the pressure-damage claim is established on arms that seed pressure and the recommended arm does not — that asymmetry needs one sentence.

### W8 — The narration of the research process crowds out the result. **Minor in severity, high in value — the cheapest big improvement available.**

I want to separate two things the paper currently fuses. The **content** of the withdrawals belongs in the paper, absolutely — that is S5 and it is why I am recommending revision rather than rejection. The **staging** of them does not. Contributions "ordered by what we think survives, not by what is largest"; "**But that is not what costs the solve.**"; "We predicted the opposite, in writing, before running any of it"; "and the arithmetic says so"; "§6.7 states what that costs the paper's mechanism, which is a great deal". Read cumulatively, this narrates a research trajectory rather than reporting a finding, and it invites the reading the authors least want: that this is a project that did not work, written up in good faith.

It is not a project that did not work. It is a project with one clean positive result, one clean negative result, and a verified modest headline. Write it that way.

*Fix:* lead each section with the finding in its final form; collect the process history — the wall-vertex bug, the withdrawn +41.8%, the withdrawn 13% claim, the re-measured Reynolds table, the falsified extrapolation and divergence explanations — into one clearly-labelled subsection ("Corrections and withdrawn claims") plus the existing limitations section. Cut the rhetorical boldface by about half. My estimate is this removes 15–20% of the length with no loss of content, which C&F will also appreciate.

### W9 — The `G` table is internally inconsistent and the formula as printed does not reproduce it. **Minor, but concrete.**

§7.1 gives the 128² raster first station 1.17·10⁻², `y⁺` = 1677, `G` = 36.6×. §6.5 gives essentially the same raster (1.2·10⁻², `y⁺` = 1700) as `G` = 29.3×. Same stated inputs, different answer. From the printed formula I cannot reproduce either: `u⁺(1677) = ln(1677)/0.41 + 5 = 23.1`, `u⁺(0.72) = 0.72` (sublayer) → 32.1×.

Reading `solver/placement.py` explains it: `amplification()` applies a **freestream ceiling** `u⁺ ≤ u_inf/u_τ` (21.0 at `u_τ` = 0.0477 → 29.3×) and a buffer-layer blend between `y⁺` = 5 and 30, and §7.1 evidently uses per-case `u_τ` rather than the fixed 0.0477 of §6.5. Neither the cap nor the blend appears in §6.1's statement of the closed form.

There is a constructive upside here that the paper is missing. In the raster regime the ceiling does *all* the work — the log law is inert — so the criterion for uniform rasters is really the much simpler and stricter statement: **the first station sits above the boundary layer, so it reads freestream, and the seeded first-cell gradient is `u_∞/y_c`.** That is more robust than the law-of-the-wall framing (no κ, no B, no blend) and it is exactly where the paper's headline verdicts live ("512² fails").

*Fix:* state the cap and the blend in §6.1; give one fully worked example; reconcile the two tables by using one `u_τ` convention and saying which; and add the sentence above about the saturated regime.

### W10 — The Cartesian negative control differs on more than representation. **Minor.**

In `scripts/corpus_probe.py`, `cartesian_128` is a whole-field `plain_seed` rasterisation, while `nf_bl` is boundary-layer-masked with `ramp = 3.0`. So the corpus's negative control violates condition 1 *and* condition 2 (and carries a discontinuous fall-back to freestream at the crop boundary, which §5.4/§7.3's own consistency argument says should be damaging in itself). As a null control for "does the pipeline manufacture savings?" it is perfectly valid, and that is the use §5.1 makes of it. But §2.3's stronger claim — "a surrogate stored as a 128² image cannot be used for warm starting at all" — leans on an over-determined arm.

*Fix:* one sentence noting the arm differs on two conditions, and either add a BL-masked raster arm or point the strong claim at §5.2's wall-fitted oracle family, where the contrast is clean.

### W11 — Highlight 5 and the abstract's "every projection preserves viscous drag" are contradicted by the Cartesian arm. **Minor.**

"Projections preserve viscous drag and destroy total drag" is true of the *wall-fitted* projections (+69 to +86%). The 128² raster preserves nothing: +3.4% on `Cd_v` over thirteen cases, i.e. statistically identical to no seed (and +10.0% on the five-case study per PLANS §0.01). Since the raster is the most damaged representation in the paper, the generalisation as written is wrong at exactly its extreme.

*Fix:* say "wall-fitted projections preserve viscous drag; the uniform raster preserves neither" in the highlight, abstract and §5.2.2. This is also mechanistically informative and worth a sentence.

### W12 — Reference construction on thin arm sets. **Minor.**

Rule 5 makes the reference a median over settled arms, so a tree with few arms has a fragile reference. The corpus has four arms, one of which (`cartesian_128`) is severely damaged; §9's wall-clock tree has five and the paper honourably reports that two of five cases are unreadable there. The corpus's `settled_spread` of 0.001 on `Cd_v@1%` is reassuring and should be quoted in §5.1 rather than left in the JSON.

*Fix:* print `settled_spread` next to every headline row; add a one-line sensitivity check (reference recomputed with `cartesian_128` excluded).

### W13 — The paper is nearly all tables; C&F expects flow physics to be shown. **Minor but I would insist on it.**

Two figures for a manuscript of this length is too few for this journal. The physics being argued about is visualisable and would carry the argument far better than the current prose.

*Fix:* add (i) the wall-normal velocity profile `u⁺(y⁺)` as actually sampled by each representation, overlaid on the converged profile, with the mesh's first cell marked — this makes §6 immediate; (ii) pressure-error contours for `oracle_mesh`, `or_proj_coarse`, `or_proj_fine`, `cartesian_128` — this makes W7's new measurement visible; (iii) `C_d` and `C_d,v` convergence histories for cold / `oracle_mesh` / `nf_bl` / `or_proj_coarse` on one case, with the force bands drawn — this makes the metric concrete.

### W14 — Scope, stated but worth repeating. **Minor.**

2-D, incompressible, steady, one turbulence model, one solver, one mesh family, one Reynolds number for all convergence results, one trained surrogate, one seed; n = 5 for every mechanism contrast and n = 13 for the headline. §10 states all of this plainly, which is why it is a minor rather than a major. But the title's "decides neural warm starts for RANS" is broader than the evidence; the conclusion's "Query your surrogate where the solver lives" is the right scope.

*Fix:* soften the title's verb (e.g. "…representation, not accuracy, gates neural warm starts for RANS") and, in §1, name the single Reynolds number and single turbulence model in the same paragraph as the headline number.

---

## 4. Questions to the authors

These are the ones that could move my score.

1. **(W1)** Why are `nf_proj_fine` and `nf_proj_half` absent from the manuscript? On the readable `Cd_v@1%` row they read +17.7% and +15.7% against `nf_proj_coarse`'s +14.5% — small and not significant, but not the reversal the oracle family shows. Please report them and reconcile the earlier `depth_placement` oracle ordering (`or_proj_coarse` +41.7% vs `or_proj_fine` +62.2%) with the draft's (+86.1% vs +69.1%). If the earlier scoring is superseded by the wall-distance fix, please add it to the withdrawal list.

2. **(W4)** Which projection-code version produced each seed-diagnostic table? §5.5 says `nf_proj_fix` roughness 26.2×; §10 says 11.1×. Please regenerate from one run and state the source file per table.

3. **(W2)** Under a *causal* stopping rule — one using only information available during the run — does `nf_bl` still beat cold, and by how much? If not, what should a practitioner do given that `nf_bl` is worse on the residual at every depth?

4. **(W3)** What is the seed quality and total charge for grid sequencing when the coarse solve is truncated at 250 / 500 / 1000 iterations? Does the conclusion "still is not worth running" survive?

5. **(W7)** What is ‖p_seed − p_converged‖, by wall-distance band, for `oracle_mesh`, `or_proj_coarse`, `or_proj_fine`, `or_proj_half` and `cartesian_128`? Does `or_proj_fine` have larger mid-field pressure error than `or_proj_coarse` — i.e. does the direct measurement corroborate §5.2.2?

6. **(W5)** Is there *any* representation in your data that passes the §6 criterion, is not mesh-native, and accelerates the solve? If not, on what basis should a reader use the pre-flight tool as a veto rather than simply adopting mesh-native output?

7. **(W6)** Please give the growth ratio and the number of stations inside the first cell for each arm in §5.2.1, and report the paired per-case contrasts. Is `or_proj_fine` worse than `or_proj_coarse` in 5/5 cases as the raw values suggest?

8. **(W9)** Please state the freestream cap and buffer-layer blend in §6.1 and reconcile `G` = 36.6× (§7.1) with `G` = 29.3× (§6.5) for the same 128² raster.

9. **(§5.3)** `nf_mesh` is reported as "< −568.3%", i.e. censored at budget. How many of the five cases never reached the band? Please give the censoring counts wherever a bound is quoted.

10. **(§9)** 10.4–10.6 s for a Transolver forward pass at 31,700 points seems high. CPU or GPU, and which hardware? The "price, not quality" argument in §5.7 depends on this number.

---

## 5. The three questions the editor asked me to address

**Is a paper whose central mechanism is disproven still a contribution?** Yes, unambiguously — provided the paper is re-framed around what survived. Two things here are independent of the disproof. First, the representation result: the exact converged field, stored on a grid, is a worse initial condition than freestream, with a 274-point swing from format alone and no network anywhere in the argument. That result stands whatever the mechanism turns out to be, and it is a constraint on how surrogates should emit their output. Second, the corpus headline: I verified +18.38%, 13/13, worst case +2.7%, oracle control +93.6%, null Cartesian control, `settled_spread` 0.001 — a small but genuinely reliable effect with the controls to prove it is real. And the disproof itself is a contribution: "restore the near-wall gradient and the warm start will work" is precisely what a competent reader would assume, and the paper shows by three independent routes that it is false. Publishing that saves the field a predictable dead end. What is *not* yet a contribution is the section that survived the disproof only as a diagnostic (§6) but is still presented as if it were the mechanism.

**Is the honesty an asset or does it read as a failed project?** The honesty is the paper's principal asset and I have weighted it as such — S5 is a large part of why this is major revision rather than reject. But I have to give the authors the distinction they are currently missing: **honesty is an asset; autobiography is a liability.** Nothing should be removed. What must change is that the manuscript currently tells its findings in the order they were discovered, with the reversals dramatised in the abstract, the contribution list, the section headings and the conclusion. That is what makes it read as a failed project — not the negative results themselves, which are clean. A reader who encounters "the criterion is the second contribution", then "the criterion does not forecast solves", then "this is the paper's largest limitation", concludes the authors lost confidence mid-way. The same facts, stated once in final form with a dedicated corrections subsection, read as unusual rigour. I would rather review ten papers like this one than one that quietly deleted its three failed repairs — please do not respond to this review by removing any of them.

**The 3–5 highest-value changes before submission.** In order:

1. **Report the omitted placement arms and state the flip (W1).** Blocking, free, and it makes the negative result stronger than it currently is.
2. **Measure the seed's pressure error directly (W7).** No solves required. This is the change most likely to convert the paper's central negative into a positive mechanism, and it is the difference between "we located the damage by elimination" and "we measured it".
3. **Re-score under a causal, deployable stopping rule (W2).** Post-processing only. It answers the question a C&F reader will actually ask and it is the change most likely to move my score to 7.
4. **Resolve the grid-sequencing charge with a truncation curve (W3).** One arm, five cases. As it stands, the paper's own caveats undercut its stated conclusion about the classical baseline.
5. **Re-weight §6 and de-dramatise the narration (W5 + W8), and fix the internal inconsistencies (W4, W9, W11).** This is editorial, costs no compute, shortens the paper by roughly a fifth, and removes every easy target a hostile reviewer would reach for first.

---

## 6. Recommendation

**Major revision. Score 5/10.**

Not a reject: the central experiment is sound, I verified the headline against the raw result files myself, the control structure is better than the norm for this literature, and the negative result is one the field needs. Every blocking item above is addressable with **data the authors already have**, plus at most one or two additional solver runs (the decoupling placement arm in W6, the coarse-truncation curve in W3). That is what distinguishes this from a reject and the authors should hear it plainly.

Not a minor revision: W1 concerns arms omitted from a paper whose case rests on completeness of reporting; W3 concerns a stated conclusion that may reverse; W2 concerns whether the headline saving is realisable at all by the reader it is addressed to; and W4 is an internal numerical contradiction in a manuscript that has already withdrawn a headline for a related reason. The four blocking items are W1, W2, W3 and W4; W5–W7 are substantive but would not by themselves hold up publication.

**What would most raise the score.** A revision that (a) reports the full placement ladder, (b) measures the seed pressure error directly and finds it ordered as §5.2.2 predicts, and (c) shows the +18.4% survives a causal stopping rule, would in my judgement be a **7–8** and a paper I would be glad to see in *Computers & Fluids*. If (c) fails, report it — a paper that establishes "grid-stored surrogate fields cannot warm-start RANS, and mesh-native ones give a saving that a deployable stopping rule does not recover" is still worth publishing, and this author is evidently capable of writing it.

---

### Files consulted

- `D:\Codes\Github\neuroforge-cfd\docs\paper2\DRAFT.md` (full)
- `D:\Codes\Github\neuroforge-cfd\docs\PLANS.md` §0.01–§0.05
- `D:\Codes\Github\neuroforge-cfd\results\depth_corpus.json` (headline verified)
- `D:\Codes\Github\neuroforge-cfd\results\depth_placement.json` (placement ladder, readability, per-case values)
- `D:\Codes\Github\neuroforge-cfd\src\neuroforge\solver\warmstart.py` (`clustered_seed`, `masked_seed`)
- `D:\Codes\Github\neuroforge-cfd\src\neuroforge\solver\placement.py` (`u_plus`, `amplification`)
- `D:\Codes\Github\neuroforge-cfd\scripts\placement_probe.py`, `scripts\corpus_probe.py`
