# What makes a benchmark's covariate null strong or weak?

**Question.** `null_travels.md` measured a parameter-only null on five CFD-ML benchmarks in
one subfield and found it lands anywhere from R² 0.10 to R² 0.97. That is an observation.
The design question is: *what computable property of a benchmark's metadata determines
which end it lands on?*

**Answer, in one line.** **No label-free property tested here orders the five benchmarks,
and two of the most intuitive ones fail with the sign reversed** — so the headline is a
null result, reported as one. But the exercise did resolve the single most informative
case: **WindsorML's counterexample status is most likely a metadata defect, not a physics
fact.** One of its seven design parameters is not published, and a single design parameter
is empirically worth up to R² 0.71 on benchmarks whose metadata *is* complete.

**Artifacts.** Script `D:\Codes\Github\neuroforge-cfd\scripts\null_mechanism.py`
(candidates, predicted directions and adjudication rule committed at `ffbec98`, with the
script present but **unrun**). Results
`D:\Codes\Github\neuroforge-cfd\results\review\null_mechanism.json`. CPU only, no GPU
touched, no field data read, ~3 min wall clock.

---

## 1. First: the table being explained was not internally consistent

Three of the five entries in the table this work was handed measure different things. They
are replaced here, and the substitutions were declared in the pre-registration commit
before any number existed.

| Benchmark | as handed to us | problem | harmonised |
|---|---|---|---|
| DrivAerML | 0.973 | on a drag-sorted outer-decile split, which inflates R² | **0.9598** |
| AirfRANS | 0.932 | **that is a Spearman, not an R²** | **0.6833** |
| DrivAerNet++ | 0.737 | scored on 595 test designs that have *no published parameters at all* | **0.8248** |
| AhmedML | 0.684 | — | **0.6842** |
| WindsorML | 0.104 | — | **0.1045** |

Harmonised protocol: 10-fold OOS OLS, seed 0, per-fold standardisation, R² (sklearn
convention), constant reference area where both conventions are published.

**The harmonisation reorders the table.** AirfRANS falls from second to fourth and lands in
a statistical tie with AhmedML (0.6833 vs 0.6842). DrivAerNet++ rises above both. Any
"explanation" fitted to the original ordering would have been fitted to an artefact of
mixing a rank correlation with a variance ratio.

**Harmonised drag null, the quantity everything below is regressed on:**

| DrivAerML | DrivAerNet++ | AhmedML | AirfRANS | WindsorML |
|---|---|---|---|---|
| 0.9598 | 0.8248 | 0.6842 | 0.6833 | 0.1045 |

---

## 2. The headline: nothing predicts it

Every candidate that was pre-registered, with the direction it predicted and what happened.
Adjudication rule, committed in advance: a candidate SURVIVES only if it orders all five
benchmarks perfectly *and* in the predicted direction. Exact permutation p over all 5! = 120
orderings; |ρ| = 1 is p = 0.0167, which is p = 0.167 after correcting for the eleven
candidates tried. **It was committed in advance that no candidate could reach corrected
significance at n = 5.**

| # | Candidate | Predicted | ρ | p (exact) | ρ w/o AirfRANS | Survives |
|---|---|---|---|---|---|---|
| A1 | effective dimensionality `d_eff` | ρ < 0 | **+0.600** | 0.350 | +0.800 | no — **wrong sign** |
| A2 | sampling density `log n / d_eff` | ρ > 0 | **−0.600** | 0.350 | −0.800 | no — **wrong sign** |
| A2b | number of cases `n` | ρ > 0 | +0.600 | 0.350 | +0.800 | no |
| A3 | scale-vs-shape (frontal-area CV) | ρ > 0 | +0.500 | 1.000 | +0.500 | no (computable on only 3 of 5) |
| A4 | metadata completeness | ρ > 0 | +0.707 | 0.400 | +0.816 | no |
| A5 | sampling design type | *nothing* | — | — | — | no — as predicted, all five are space-filling |
| B1 | label dynamic range CV | ρ > 0 | **−0.600** | 0.350 | −0.400 | no — **wrong sign** |
| B2 | signal-to-noise | ρ > 0 | — | — | — | **not computable** (see §5) |
| B3 | dip statistic | ρ < 0 | −0.800 | 0.133 | −0.800 | no |
| B3 | bimodality coefficient | ρ < 0 | +0.300 | 0.683 | +0.400 | no — wrong sign |
| B3 | excess kurtosis | ρ < 0 | **−0.900** | **0.083** | −1.000 | no (one inversion) |
| C2 | local-slope instability | ρ < 0 | −0.700 | 0.233 | −0.800 | no |
| C1 | flexible-fit ceiling | ρ > 0 | +0.700 | 0.233 | +0.800 | no |

**Read this plainly.** The three properties a benchmark author could actually control
before running any simulation — how many parameters to vary, how densely to sample them,
which DoE to use — are A1, A2, A2b and A5. **All four fail, and two fail with the sign
reversed.** In this set of five benchmarks, *higher* effective dimensionality and *sparser*
sampling per dimension went with *stronger* nulls, not weaker. DrivAerML is the sparsest
benchmark here (16 effective dimensions, 484 cases, `log n / d_eff` = 0.39) and has the
strongest null; WindsorML is among the densest (5.4 effective dimensions, 355 cases, 1.08)
and has the weakest.

The best-ordering candidate is excess kurtosis of the label (ρ = −0.900, single inversion
between DrivAerML and DrivAerNet++, p = 0.083 uncorrected). It fails the pre-registered
bar. It is also close to mechanical — heavy label tails hurt a least-squares fit — so it
would be a weak mechanism even if it had ordered them perfectly. **It is reported as a
hypothesis for future benchmarks, not a finding.**

### Controls on the ordering itself

* **Sample size is not the driver.** Every benchmark subsampled to n = 200 (5 draws each):
  DrivAerML 0.955, DrivAerNet++ 0.811, AirfRANS 0.688, AhmedML 0.685, WindsorML **−0.018
  (sd 0.437)**. The ordering is preserved exactly.
* **The ordering is not even invariant to a reporting convention.** Switching from constant
  to per-geometry reference area moves AhmedML by **0.345** (0.684 → 0.340), DrivAerML by
  0.069 and WindsorML by 0.031. Under the per-geometry convention AhmedML drops below
  AirfRANS. Null strength is a property of the triple (benchmark, target, reference-area
  convention), not of a benchmark.
* **The unit of analysis is not the benchmark.** Null strength varies within a single
  benchmark, with the parameter matrix, n, d_eff, DoE and solver all held fixed and only the
  force channel changed: DrivAerML 0.960 (drag) to 0.602 (side force); AhmedML 0.684 to
  0.452 (lift); WindsorML 0.104 to 0.235 (lift). A one-way decomposition over the 14
  (benchmark, target) pairs puts 88.3% of the spread between benchmarks — so this does *not*
  rule out design-space explanations by itself. Those are ruled out empirically, in the
  table above, not by this bound. (The 14 pairs share a parameter matrix within each
  benchmark, so no 14-point correlation is reported; that was pre-registered.)

---

## 3. What the exercise *did* establish: one number is not enough

The pre-registered C1 decomposition splits the null into **determinacy** (how much of the
label the metadata pins down at all, lower-bounded by the best of a flexible family:
quadratic + interactions with a ridge path, kNN, kNN-local-linear, same folds) and
**linearity** (what share of that a linear fit captures).

| Benchmark | linear null | flexible ceiling (lower bound) | linearity share |
|---|---|---|---|
| DrivAerML | 0.9598 | 0.9883 | 0.97 |
| DrivAerNet++ | 0.8248 | 0.8811 | 0.94 |
| AhmedML | 0.6842 | **0.7748** | 0.88 |
| AirfRANS | 0.6833 | **0.9463** | **0.72** |
| WindsorML | 0.1045 | 0.2516 | 0.42 |

**AirfRANS and AhmedML have statistically indistinguishable linear nulls — 0.6833 and
0.6842 — arising from completely different causes.** On AirfRANS the metadata determines
the drag almost fully (0.946) and the linear model simply cannot express the map; the
deficit is a *model-class* failure, and a slightly better null would erase it. On AhmedML a
flexible learner gets only 0.775 on a dense 6-dimensional Latin hypercube with 499 cases,
so roughly a quarter of the drag variance is not recoverable from the published parameters
at all; the deficit is *informational*.

Those two benchmarks would be scored identically by a single-number null and they pose
opposite challenges to a surrogate. **This is the concrete, reusable output of the
exercise: report the pair, not the number.**

---

## 4. WindsorML — the counterexample is a metadata defect, not a physics fact

This was the most informative single case and it resolves further than expected.

**Finding 1 (source-verified).** The WindsorML paper (arXiv 2407.19320) states **seven**
design parameters, sampled by a Halton sequence. The published
`geo_parameters_all.csv` contains **six** of them — `ratio_length_front_rear`, range 0 to
0.8, is absent — plus a derived `frontal_area` column. Verified by fetching the live
HuggingFace file, so it is not an artefact of our download.

**Finding 2.** That seventh column carries almost no independent design information:
`frontal_area` is **97.1%** explained by `clearance` alone (regression coefficient +0.985).
The apparent seven columns encode six degrees of freedom. (The same check is a useful
sanity test: AhmedML's eight columns likewise encode six — `slant-surface-length` is 97.6%
redundant — which matches its paper's stated six, and DrivAerML's sixteen are all
independent, most redundant column R² 0.024.)

**Finding 3 — the accounting.** The WindsorML paper's own MeshGraphNet implies R² ≥ 0.79.
It reads only the geometry, and the geometry is generated from the seven parameters, so
**some** function of those seven parameters reaches ≥ 0.79. The flexible ceiling from the
published metadata is 0.252. **At least 0.538 of WindsorML's drag variance is reachable
from the geometry but not from the published metadata.**

**Finding 4 — is one withheld parameter enough to cover that?** This is what the
pre-registered leave-one-parameter-out calibration (C3/P4) was for. P4 predicted no single
parameter would be worth more than 0.5 R². **P4 was FALSIFIED, and its falsification is the
load-bearing result:**

| Benchmark | base R² | largest single-parameter drop | which |
|---|---|---|---|
| AirfRANS | 0.683 | **0.705** | `alpha` |
| DrivAerML | 0.960 | **0.567** | `Vehicle_Width` |
| AhmedML | 0.684 | 0.415 | `body-width` |
| DrivAerNet++ | 0.825 | 0.262 | `fam_F_S` |
| WindsorML | 0.104 | 0.120 | `ratio_height_fast_back` |

A single design degree of freedom is empirically worth up to 0.705 of R². The gap WindsorML
must close is 0.538. **One withheld degree of freedom is a quantitatively sufficient
explanation.** Note also that none of WindsorML's six *published* parameters is worth more
than 0.12 — the benchmark publishes its weak levers and withholds a candidate strong one.

**Finding 5 — it is not sampling sparsity (P2, HELD).** The flexible learning curve on
WindsorML runs 0.104 (n=50) → 0.211 (100) → 0.220 (200) → 0.252 (355); the tail slope from
200 to full is +0.032. It is not going to reach 0.79. The cleaner argument needs no
extrapolation: DrivAerML sits in 16 effective dimensions with 484 cases — *sparser per
dimension than WindsorML* — and its linear null is 0.96.

**Verdict: PARTIALLY-RESOLVED, and the honest statement is that this is a sufficient
explanation, not a proven one.** The alternative — that the Windsor body's drag is genuinely
a rough function of its shape parameters — is not excluded. It is however now a *testable*
alternative rather than a shrug.

**The test that would settle it, stated in advance:** publish `ratio_length_front_rear` for
the 355 runs, or recover it from the distributed STL geometries, and refit. If the null
rises into the 0.6–0.8 range the cause is metadata incompleteness. If it stays near 0.10
the cause is the physics of this body, and WindsorML becomes the field's best example of a
benchmark whose design space genuinely defeats a metadata null. Either outcome is
publishable; the current state is that nobody can tell, and the benchmark could resolve it
by publishing one column.

---

## 5. Every candidate that failed, including the ones that failed quietly

Reported because a candidate list is only honest if the failures are in it.

* **A5, sampling design type** — pre-registered as predicting *nothing*, and it predicted
  nothing. All five benchmarks use a space-filling DoE (extensible lattice, Latin hypercube,
  Halton, parametric morph DoE, randomised sampling). There is no curated-configuration
  benchmark in this set, so the "space-filling vs curated" axis is **untested**, not
  refuted. A benchmark built from genuinely distinct real configurations would be the
  missing sixth point and is the most valuable one to add.
* **A3, scale-vs-shape** — computable on only three of five. DrivAerML publishes morph
  *deltas* (its `Vehicle_Width` is −21.12 on run 1) and no reference area, so no frontal-area
  proxy exists; AirfRANS has no area. On the three where it is computable the values are
  suggestive in the predicted direction — WindsorML 0.0149, DrivAerNet++ 0.0731, AhmedML
  0.1604, i.e. WindsorML varies shape at essentially fixed frontal area (1.5% spread) — and
  P3 held (DrivAerML's three scale parameters alone reach R² 0.649 of its 0.960). But
  AhmedML has the *largest* area variation of the three and a mid-table null, which is the
  wrong direction. **Not adjudicable at n = 3.**
* **B2, signal-to-noise** — **not computable, and that is a finding.** Only one of the five
  benchmarks publishes a per-case label uncertainty (DrivAerNet++'s `Std Cd`; mean 0.00637
  against a label sd of 0.02202, SNR 3.46). The two benchmarks that publish a side-force
  channel cannot be used as zero-yaw symmetry probes: WindsorML is run at −2.5° yaw
  deliberately (its `cs` mean is −0.0365), and DrivAerML's `cs` has mean 0.0219 against sd
  0.0150, so it is a systematic force, not scatter. **Four of five benchmarks make their own
  label noise floor unmeasurable.**
* **P1, the AhmedML regime-crossing prediction — FALSIFIED on the committed threshold, and
  the near-miss is more interesting than a pass.** Two of its three clauses held. The binned
  drag profile shows the textbook Ahmed critical-slant behaviour exactly where predicted:
  C_d flat at ≈0.267 below 25°, rising to a peak of **0.341 in the 32–38° bin**, then falling
  back to ≈0.29. The `|slant − 30|` basis is worth **6× more than any other parameter's
  kink** (0.0489 vs 0.0078). But it buys 0.0489, and the pre-registered bar was 0.05. The bar
  binds. More importantly the *substantive* reading is unaffected: the regime crossing is
  real and visible, and it accounts for about 15% of AhmedML's 0.32 deficit. **Flow-regime
  crossing is a genuine effect and is not the mechanism.**
* **C2, local-slope instability** — ordered four of five (ρ = −0.700) but conflates the thing
  it was meant to detect with noise. A value near 0.5 is what uncorrelated local slopes give,
  which is what a noisy label produces; WindsorML's 0.50 therefore cannot distinguish "the
  sensitivity flips sign across a regime boundary" from "the local fits are fitting nothing".
  Uninformative as specified.

---

## 6. Verdict

**NULL RESULT on the predictive question, stated as plainly as a positive one.** Of the
label-free properties a benchmark author could compute before simulating anything —
dimensionality, sampling density, sample size, DoE type, scale variation, metadata
completeness — **none orders the five benchmarks**, and dimensionality and sampling density
order them *backwards*. At n = 5 nothing could have reached corrected significance even if
it had ordered them perfectly; that was committed before the run, so the result is a
genuine absence of signal and not a power excuse. **We cannot yet say what drives the
difference.**

Two things were nonetheless resolved:

1. **Reporting a single null number is insufficient**, demonstrated rather than asserted:
   AirfRANS 0.6833 and AhmedML 0.6842 are the same number with opposite causes (model-class
   failure vs missing information), separated only by the linear/flexible pair.
2. **WindsorML's counterexample status is most likely an artefact of its own metadata
   publication**, with a verified missing design parameter, a redundant substitute column, a
   0.538 accounting gap, and a calibration showing one parameter can be worth 0.705. This is
   a sufficient explanation with a one-column test that would settle it.

### Design guidance a benchmark author can act on today

1. **Publish every design degree of freedom you varied.** A single withheld parameter is
   empirically worth up to R² 0.71. WindsorML's null is currently uninterpretable for this
   reason alone.
2. **Report the pair (R²_linear, R²_flexible) of a metadata-only null**, not one number.
   The gap is what tells a reader whether their benchmark is trivially predictable, smoothly
   predictable but nonlinear, or genuinely underdetermined by its metadata.
3. **State the reference-area convention.** It moves AhmedML's null by 0.345 and reorders
   the field.
4. **Publish a per-case label uncertainty.** One of five does. Without it the
   signal-to-noise of a benchmark cannot be assessed by anyone, including its authors.
5. **Report the null per force channel.** It varies by up to 0.36 within one benchmark with
   everything else held fixed.
6. **Do not trust design-space intuitions about null strength.** In this set, more
   dimensions and sparser sampling went with *stronger* nulls.

### What would make this predictive rather than descriptive

The honest limit is n = 5, and the missing point is structural rather than numerical: all
five benchmarks use a space-filling DoE over a single morphed base body. The untested axis
is a benchmark built from genuinely distinct configurations. **The stated hypothesis, for
future benchmarks to falsify: null strength is set by whether the sweep's dominant design
degrees of freedom are published and vary the body's overall scale, not by how many
parameters there are or how densely they are sampled.** It predicts that a curated
multi-configuration benchmark and a fixed-scale shape morph both yield weak nulls for
different reasons, and that publishing a withheld scale parameter on WindsorML raises its
null into the 0.6–0.8 range. Neither prediction is testable on the five benchmarks that
exist; both are cheap to test on the sixth.

### Rebuttal line

> Reviewer: *you show the null varies ten-fold across benchmarks — so what determines it?*
> → We pre-registered eleven candidate explanations with their predicted directions and an
> adjudication rule, then measured all of them → **None survives, and the two most intuitive
> (dimensionality, sampling density) fail with the sign reversed; we report that as a null
> result.** What the exercise did establish is that a single null number conflates two
> different failures (AirfRANS 0.683 and AhmedML 0.684 are identical numbers with opposite
> causes), and that the field's one counterexample, WindsorML, most likely owes its status
> to a design parameter its own paper lists but its data release omits — with a 0.538
> accounting gap against a calibration showing one parameter is worth up to 0.705. Script,
> pre-registration commit and results JSON are in the artifact.

---

## 7. Cost

| | |
|---|---|
| New data downloaded | 0 bytes of field data; 4 CSV headers re-fetched to verify the WindsorML omission upstream |
| Compute | CPU only, no GPU touched; ~3 min wall clock |
| Source documents read | WindsorML, DrivAerML and AhmedML dataset papers, for parameter counts and DoE type |
