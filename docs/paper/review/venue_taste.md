# Venue taste, reverse-engineered from the published record

**Date of all verification: 2026-09-12** unless a row says otherwise.

**What this document is.** Five venues profiled against *what they actually published*, not
what their scope statements say. Section A is the raw record, B is the taste inferred from it,
C is editorial signal, D is what this specific paper would have to look like to land.

**The paper being placed.** A benchmark-evaluation paper for CFD machine learning:
ordinary least squares on published case metadata predicts the headline label across five
benchmarks at R² 0.10–0.97 (`null_travels.md`); the field-MSE ranking between a 7.35M-parameter
Transolver and a seven-scalar kernel interpolator *reverses* under a weighting no published
AirfRANS comparison states (`measure_asymmetry.md`, D3 = ARTIFACT at 0.440); the r128
rasterisation's own round-trip error at the native nodes exceeds the surrogate's by 418×
pooled and 495× inside 0.005c. It ships a tested harness (`nullbench_release.md`) and five
corrected leaderboards. ~14,700 words, 10 figures, 9 tables, 42 refs. Desk-rejected twice on
originality (CMAME 2026-08-02, JCP 2026-09-07).

---

## 0. Method and provenance

Provenance marks follow `venue_plan.md` §0:

- **[P]** primary fetch — the page itself was retrieved and read.
- **[S]** search-extraction — the primary page's content as surfaced by a search engine.
- **[V]** verified in a browser session (inherited rows only, dated).

**Published-record evidence** is from the **OpenAlex API** (`api.openalex.org`), queried
directly, per journal source id, sorted by `publication_date` descending. This is [P] for the
*bibliographic record* (title, date, reference count) and is the only route available because
**ScienceDirect returns HTTP 403 to every automated fetch** — a limitation already documented
in `venue_plan.md` §0 and re-confirmed today on
`sciencedirect.com/special-issue/316707/...` (403).

Source ids used, each resolved through `api.openalex.org/sources` [P]:

| Venue | OpenAlex id | ISSN | works indexed |
|---|---|---|---|
| Computers & Fluids | S195914795 | 0045-7930 | 7,292 |
| Engineering Applications of AI | S900972176 | 0952-1976 | 14,649 |
| Computer Physics Communications | S142305363 | 0010-4655 | 14,108 |
| Nature Machine Intelligence | S2912241403 | 2522-5839 | 1,304 |
| Nature Computational Science | S4210228084 | 2662-8457 | 1,048 |

**Sanity check on currency:** the top entry for every Elsevier source dates to 2026-09-05 or
later (EAAI to 2026-09-10), so OpenAlex is not lagging materially. Elsevier
`publication_date` tracks the online-first date here, not an issue date, because all recent
entries carry article numbers rather than page ranges.

### What I could not access, stated rather than guessed

1. **Figure counts and printed page lengths for any of the three Elsevier titles.** Elsevier
   has moved to article-number pagination (`107276-107276`), so `biblio` gives no page count,
   and ScienceDirect 403s block the article pages. Reference counts are available and are used
   instead. **Any statement below about Elsevier article *length* is marked UNVERIFIED.**
2. **OpenReview reviews and meta-reviews.** `api2.openreview.net` now returns a 302 to a
   browser-challenge page, and `openreview.net/forum` renders a verification interstitial. The
   *search* endpoint (`/notes/search`) still answers, so accepted-paper titles and venue labels
   below are [P]; the reviews behind them are **inaccessible**. This is a real loss: reviews
   were the single best taste evidence available for venue 2 and none of it could be read.
3. **nature.com listing pages** (`/articles?type=editorial`, `/content-types`) redirect into an
   identity-provider loop that does not resolve. Individual *article* pages resolve by
   appending `?error=cookies_not_supported` [P]. Editorial titles below are therefore [S].

---

## 1. Computers & Fluids (Elsevier) — the incumbent choice

### A. What they published recently

**A.1 — the ML/neural slice, most recent first.** OpenAlex, source S195914795,
`default.search: machine learning neural network`, `from_publication_date: 2025-09-01`,
18 results [P]. `refs` = `referenced_works_count`; `0` means OpenAlex has not resolved the
reference list, not that the paper has none.

| # | Title | Date | refs | one-line characterisation |
|---|---|---|---|---|
| 1 | Accelerating particle-resolved simulations of dense suspensions with hybrid GNN–U-Net neural networks | 2026-09-05 | 77 | learned *initialiser* inside an unchanged OpenFOAM solver; 2.13× net speed-up |
| 2 | Reynolds-number-adaptive symbolic regression for explainable wall-bounded turbulence modelling | 2026-09-01 | 66 | symbolic-regression wall model vs classical composite law, DNS-trained |
| 3 | Uncertainty quantification of reacting fluids interacting with porous media using a hybrid physics-based and data-driven approach | 2026-08-21 | — | operator-inference ROM for UQ in ablative heat shields |
| 4 | Machine-learned R13 moment closures for shock-dominated rarefied gas flows | 2026-08-13 | 44 | NN replaces analytic closure terms in a moment method |
| 5 | Drag modelling for flows through assemblies of spherical particles with machine learning: A comparison of approaches | 2026-07-25 | — | **comparison study** of ML drag-closure approaches |
| 6 | Supervised machine learning applied to bifurcation analysis of weakly compressible flow past a rotating cylinder | 2026-07-11 | 42 | ML as an analysis tool on a canonical flow |
| 7 | A 3D Machine Learning based Volume Of Fluid scheme without explicit interface reconstruction | 2026-04-27 | — | ML component inside a VOF numerical scheme |
| 8 | Neural-network closures for complex-shaped particles in the force-coupling method | 2026-04-23 | — | NN closure |
| 9 | An adaptive-learning framework for chemistry tabulation in turbulent reacting flows | 2026-04-21 | 36 | learned tabulation, combustion |
| 10 | Energy-conserving neural network closure model for long-time accurate and stable 2D LES | 2026-03-04 | 37 | structure-preserving NN closure |
| 11 | A robust data-free physics-informed neural network for compressible flows with shocks | 2026-01-18 | 20 | PINN with shock handling |
| 12 | Data-driven regression of thermodynamic models in entropic form using physics-informed machine learning | 2025-11-30 | 15 | PIML for EOS regression |
| 13 | Modeling advection-dominated flows with space-local reduced-order models | 2025-11-20 | 35 | ROM method |
| 14 | Data-driven RANS closures using a relative importance term analysis based classifier for 2D and 3D separated flows | 2025-11-17 | 37 | classifier-gated RANS closure |
| 15 | Calibration of Manning's roughness coefficients ... using optimization algorithms and surrogate neural network models | 2025-10-25 | 26 | surrogate-assisted calibration, applied hydraulics |
| 16 | Fast integration method for averaging polydisperse bubble population dynamics | 2025-10-24 | 22 | numerical method (matched by the ML query only incidentally) |
| 17 | Graph neural network based model of hydrodynamic closure laws in non-spherical particle–laden flows | 2025-10-13 | 60 | GNN closure |
| 18 | Invariant control strategies for active flow control using graph neural networks | 2025-10-01 | 56 | GNN flow control |

**A.2 — the wider recent record.** The same source, `from_publication_date: 2019-01-01`,
sorted descending, unconstrained by ML terms, returns a list dominated by lattice-Boltzmann
schemes, SPH/meshless formulations, WENO/WCNS scheme families, DNS studies and
invariant-domain-preserving discretisations [P]. C&F is first and foremost a
**numerical-methods-for-fluids** journal; the ML papers are a minority stream inside it.

### B. Taste, inferred from that evidence

**Dominant contribution type: a new method component, embedded in a solver, with a physics
validation.** 16 of the 18 rows introduce a model, closure, scheme or framework. Row 5
("A comparison of approaches") is the only comparison study in 12 months of ML output, and it
compares *methods for a modelling task*, not evaluation protocols.

**Do they publish critique or negative-leaning results? Essentially no.** Two sweeps, both [P]:

- `title_and_abstract.search: overoptimism OR "weak baseline" OR "weak baselines" OR cautionary
  OR "critical assessment" OR "reporting bias" OR "reality check"`, 2019-01-01 onwards:
  **meta.count = 2**, and both are assessments of *numerical methods*, not of ML evaluation:
  - *Critical assessment of wall model numerical implementation in LBM* (2023-03-20)
  - *A critical assessment of the immersed boundary method for modeling flow around fixed and
    moving bodies* (2023-03-09)
- `default.search: benchmark dataset evaluation protocol metric`, 2024-01-01 onwards:
  **meta.count = 1**, and it is row 1 above (a GNN–U-Net solver accelerator), matched
  incidentally.

**In seven years C&F has published zero papers whose contribution is a critique of an ML
evaluation protocol.** That is the single most important finding in this document about the
incumbent choice, and it is a *published-record* finding that the scope statement quoted in
`venue_plan.md` §12 does not predict.

**Titles.** Descriptive noun phrases naming the method and the flow regime. The dominant shape
is "*[method] for [flow problem]*" or "*A/An [adjective] [method] for [problem]*". **Zero
assertive claims, zero questions, zero coined system names in the 18.** Row 4 uses an internal
label ("R13-ML") but the title still reads as a method description.

**Abstract house style** (reconstructed verbatim from OpenAlex `abstract_inverted_index`, [P]).
The pattern in all four abstracts retrieved is identical and rigid:

1. The physical/computational problem and why its cost matters.
2. "In this work / We address this by …" — the method, in mechanism terms.
3. Quantified validation against a reference (DNS, DSMC, the unmodified solver) with explicit
   percentages and factors.
4. A generalisation sentence — held-out cases, extrapolation, "these results indicate that …".

Example (row 1): *"…the learned initializer cuts the mean forcing field error … Embedded in the
full solver, it yields an iteration-reduction factor of 2.54 or overall speedup of 2.13× after
accounting for inference cost. The time-averaged shear stress changes only −1.65% relative to
the unmodified solver."* **The abstract's job at C&F is to state a mechanism and then quantify
its accuracy against a physics reference.** Not one of the four leads with a claim about
the field.

**Length and figures: UNVERIFIED** (ScienceDirect 403; article-number pagination). What is
verified: `venue_plan.md` §12 [V] 2026-09-07, read in-browser — **no word, page or figure cap**,
abstract ≤250 words, 1–7 keywords, **3–5 highlights of ≤85 characters required at submission**,
**single-anonymized** review.

**References: 15–77, median ≈ 37** (n = 14 with resolved lists). A 42-entry bibliography is
squarely normal here — which also means **adding the eight missing lineage citations from
`r2_round2.md` §2.2 costs nothing against house norms.**

### C. Editorial signals

OpenAlex `type:editorial`, C&F + EAAI + CPC, `from_publication_date: 2024-06-01`:
**6 results total**, of which three are C&F [P]:

| Title | Date |
|---|---|
| Editorial for "**Fusing data and physics: Machine learning for computational fluid dynamics**" | **2026-07-14** |
| Editorial for "The 12th international conference on computational fluid dynamics (ICCFD12)" | 2026-04-08 |
| Editorial: Special Issue on "New Directions in Computational Fluid Dynamics" | 2025-05-28 |

**The 2026-07-14 ML special issue is the live signal, and it carries a serious complication.**
Special-issue scope [S]: *"original contributions that explore novel ways of combining
data-driven ML models with classical physics-based numerical methods … as well as methods
directly incorporating physics-driven approaches such as PINNS, PINOs"*; topics listed are
"Physics-informed approaches for CFD applications (e.g. PINNS, PINO)" and "Reduced-order/
surrogate modeling using machine learning (e.g. GNNs, CNNs, neural operators)"; the stated goal
is to *"help establish best practices for fusing data-driven and physics-driven ML approaches
for CFD."*

**The guest editors are Dr Neil Ashton (NVIDIA) and Prof. Richard Dwight (TU Delft)** [S],
verified against `neilashton.co.uk` [S]: Ashton *"leads and collaborates on the open AhmedML,
WindsorML, DrivAerML, and HiLiftAeroML datasets."*

That is **three of the four benchmarks this paper audits in `null_travels.md`** — and NVIDIA
is the publisher of the PhysicsNeMo-CFD benchmarking table (arXiv 2507.10747) whose DoMINO and
X-MeshGraphNet rows the paper's null straddles and beats.

**And one of the two guest editors is a standing Associate Editor of the journal.** [S]
`journals.elsevier.com/computers-and-fluids/editorial-board/r-p-dwight` lists **R. P. Dwight**
on the C&F Editorial Board as an Associate Editor (research interests: data-driven modelling,
data assimilation, uncertainty quantification). **No editorial-board listing was found for Neil
Ashton**, and the ScienceDirect board page itself 403s, so the board search is [S] and
**incomplete** — absence of an Ashton listing is weak evidence, not proof.

The consequence is that **an earlier draft of this document's mitigation was wrong and is
withdrawn.** "Submit to the regular issue so the special-issue editors are not in the chain"
does not hold: Dwight sits in the regular chain by virtue of the standing board. What is left:

- *For:* Dwight's own research area is data-driven turbulence modelling and UQ. A result about
  **measure dependence of an error metric on a wall-clustered mesh** is squarely inside his
  competence and is the kind of thing a UQ methodologist takes seriously rather than defensively.
  He is not an author of the four audited datasets. The SI's own stated goal includes
  *"establish best practices."*
- *Against:* the covariate-null half of the paper is, in substance, a criticism of how the
  automotive-aero benchmark community reports results — a community Ashton leads and Dwight
  is adjacent to. `null_travels.md` §2.1 names X-MeshGraphNet as below the null on Ashton's own
  DrivAerML split.

**Revised handling, and it is an action item rather than a reassurance:**

1. Accept that a C&F submission is read inside this community; there is no routing around it.
   This is a reason to lead with the **measure-asymmetry** result (a discretisation statement)
   and demote the covariate null, which is also what §1.D already recommends on taste grounds.
2. **Use Editorial Manager's "opposed reviewers" field.** Naming Ashton there is legitimate,
   standard, and costs nothing; the paper evaluates datasets he authored and a benchmarking
   table his employer published. Do not name Dwight — opposing a sitting Associate Editor is
   both futile and a bad signal.
3. Do not submit to the special issue and do not cite it in the cover letter.
4. **Desk-screen risk is revised from LOW–MEDIUM to MEDIUM**, and review-stage risk stays the
   highest of any candidate (`venue_plan.md` §6).

Note separately that Ashton's WindsorML was a **NeurIPS 2024 Datasets & Benchmarks poster** —
the same producer community is present at venue 2, which is a point in venue 2's favour rather
than a problem there, because the ED track's own guidelines make critique of existing benchmarks
an accepted category.

### D. Reverse-engineered strategy for C&F

**What an editor here would consider the contribution: a measurement about a solver-adjacent
protocol, expressed in fluids terms.** Not "the field's reporting is broken."

- **Title shape.** Abandon any assertive/claim title. Match the house: a descriptive noun
  phrase naming the object measured and the flow regime. E.g. *"Measure dependence of
  field-error metrics for machine-learning RANS surrogates on wall-clustered meshes"* or
  *"On the near-wall resolvability of rasterised error metrics for airfoil RANS surrogates."*
  The word "benchmark" should appear at most once and never first.
- **What the abstract leads with.** Follow the four-beat house structure exactly. Beat 1: RANS
  surrogates are scored on rasterised field MSE, and AirfRANS places 64% of its nodes inside
  0.05c of the wall against 1.2% of the raster's area. Beat 2: we re-score identical
  predictions under the dataset's own node measure. Beat 3: the quantified result —
  `mse_p` 8.39× → 0.440×, every channel flips, three seeds, both arms reproducing their
  published rows at relative error 0.00e+00. Beat 4: the representation ceiling, 418× pooled.
  **Lead with the measure asymmetry, not with the covariate null.** The measure asymmetry is a
  discretisation-and-metric statement a CFD reviewer owns; the covariate null is a
  machine-learning-sociology statement they do not.
- **What must be foregrounded.** The near-wall physics. The 906× → 0.31× monotone band ladder
  on `u` (span 2905×, zero inversions) is the best exhibit in the whole project for this
  audience, because it says *a learned surrogate earns the first cell off the wall and loses
  the far field* — a statement about boundary layers, which is what C&F readers care about.
  The r128 representation ceiling (495× inside 0.005c) is the second, because it is a
  grid-resolution argument. The 512² resolution ladder (PARTIAL verdict, near-wall shares
  0.879/0.842) should be **in**, verdict-first; C&F reviewers will demand a refinement study
  and `interpolation_resolution_ladder.md` already is one.
- **What must be buried.** The covariate null goes into a subsection near the end framed as a
  *reporting-protocol recommendation*, not a headline. The force-rank Spearman comparison
  against 2022 dataset-paper baselines is weak here (`r2_round2.md` §2.4: only two of four are
  decisive) and invites a fight the paper does not need. The trust layer / conformal /
  selective-prediction sections should be **cut** per `r2_round2.md` §2.6 — note this
  **invalidates two of the four scope-checklist mappings in `venue_plan.md` §12** (the UQ
  exhibit and part of the "physical consistency" exhibit go with them). Re-derive the cover
  letter's checklist against the paper as it now is.
- **Does the harness help or read as engineering?** **It reads as engineering, and it should be
  demoted to a data-availability statement plus a Zenodo DOI.** Zero of the 18 recent ML papers
  is a software contribution; C&F's Research Data policy (Option C, [V] 2026-09-07) wants a
  deposit and a citation, not a package as the object of the paper. `nullbench_release.md`'s
  CLI is a liability in the body and an asset in the back matter.
- **Is the negative framing survivable?** **Only if it is never framed as negative.** The
  published record has two "critical assessment of [numerical method]" papers in seven years,
  and both assess a *discretisation*, not a community. The survivable frame is identical:
  *this is a critical assessment of a scoring discretisation.* The scope sentence's
  "discuss the limitations of the method as well as its merits" is real but it is an
  instruction about how to write up your own method, not an invitation to a genre.
- **Rework cost: LOW.** Retitle, reorder the abstract (still ≤250 words, already built to that
  cap), insert the ladder, cut ~510 lines of trust layer, add the RSM lineage paragraph. No new
  compute. The `elsarticle` build and Editorial Manager account are reused. Highlights file
  already exists (`docs/paper/highlights.txt`).
- **The residual desk-reject risk.** Both prior desk rejections were on originality of method,
  and C&F's *published ML record* is 100% new-method papers. A screening editor pattern-matching
  on that record can still bounce a measurement paper. The retitle and the measure-asymmetry
  lead are the whole defence.

---

## 2. NeurIPS **Evaluations & Datasets** Track (formerly Datasets & Benchmarks)

### The name, the status, and the date — verify these before anything else

- **The track was renamed.** The Datasets & Benchmarks track is, as of 2026, the
  **Evaluations & Datasets (ED) track** [P],
  `neurips.cc/Conferences/2026/CallForEvaluationsDatasets`.
- **The 2026 cycle is closed.** Abstracts were due **2026-05-04**, full papers **2026-05-06**,
  author notification **2026-09-24** [P]. Today is 2026-09-12. **The next intake is NeurIPS
  2027, approximately May 2027 — eight months away.** The task did not ask for this and it is
  the most decision-relevant single fact in this document: **NeurIPS ED cannot be a next move.
  It can only be a 2027 move.**
- **Format** [P]: **nine pages** including figures, excluding references and appendix; main-track
  template with `\usepackage[eandd]{neurips_2026}`; **double-blind by default** (single-blind
  available only for datasets with non-anonymisation restrictions).

### A. What they published recently

OpenReview `/notes/search` against `NeurIPS.cc/2025/Datasets_and_Benchmarks_Track` [P]. Venue
strings distinguish accepted ("… Track poster/spotlight") from submitted ("Submitted to …").

**A.1 — CFD / physical-simulation accepted papers, NeurIPS 2025 D&B:**

| Title | Status | characterisation |
|---|---|---|
| **ML4CFD Competition: Results and Retrospective Analysis** | poster | a *retrospective* on a CFD surrogate competition — AirfRANS-family task |
| **DrivAerStar: An Industrial-Grade CFD Dataset for Vehicle Aerodynamic Optimization** | poster | large automotive CFD dataset |
| **UniFoil: A Universal Dataset of Airfoils in Transitional and Turbulent Regimes** | poster | airfoil dataset, directly adjacent to AirfRANS |
| **AutoHood3D: A Multi-Modal Benchmark for Automotive Hood Design and Fluid–Structure Interaction** | poster | FSI benchmark |
| **AneuG-Flow: Large-Scale Synthetic Dataset of Intracranial Aneurysm Geometries and Hemodynamics** | poster | haemodynamics dataset |
| **LithoSim: A Large, Holistic Lithography Simulation Benchmark** | poster | physics-simulation benchmark |
| **OceanBench: A Benchmark for Data-Driven Global Ocean Forecasting systems** | poster | geoscience forecasting benchmark |
| **CausalDynamics: a large-scale benchmark for structural discovery of dynamical causal models** | poster | dynamical-systems benchmark |
| CFDLLMBench: A Benchmark Suite for Evaluating LLMs in CFD | *submitted* | (not accepted) |
| Physics-Learning AI Datamodel (PLAID) datasets | *submitted* | (not accepted) |

**A.2 — the critique / meta-evaluation genre, accepted at NeurIPS 2025 D&B:**

| Title | Status | characterisation |
|---|---|---|
| **Common Task Framework For a Critical Evaluation of Scientific Machine Learning Algorithms** | **poster** | the closest genre precedent anywhere on this list — arXiv 2510.23166, Wyder, Goldfeder, Yermakov et al.; its own abstract names *"weak baselines, reporting bias, and inconsistent evaluations"* as the problem |
| **Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking** | poster | diagnoses defects in existing benchmark datasets |
| **ORBIT — Open Recommendation Benchmark for Reproducible Research with Hidden Tests** | poster | protocol fix for a field with a known simple-baseline problem |
| **MARS-VFL: A Unified Benchmark for Vertical Federated Learning with Realistic Evaluation** | spotlight | "realistic evaluation" reframe of an existing task |
| **Is This Tracker On? A Benchmark Protocol for Dynamic Tracking** | poster | protocol paper with a question title |
| **DAVE: Diagnostic benchmark for Audio Visual Evaluation** | poster | diagnostic-first benchmark |
| **BRACE: A Benchmark for Robust Audio Caption Quality Evaluation** | poster | evaluating the evaluators |

**And the object itself lives here.** AirfRANS (Bonnet et al.) was **NeurIPS 2022 Datasets &
Benchmarks** [S]. WindsorML was a **NeurIPS 2024 D&B poster** [S]. **This paper's primary
benchmark and three of its four cross-benchmark targets were published at this venue.**

### B. Taste, inferred from that evidence

**Dominant contribution type: a named artifact — dataset, benchmark suite, or protocol — with a
capitalised name.** The naming convention is near-universal and is the opposite of C&F's.

**Do they publish critique?** **Yes, demonstrably, and increasingly.** Seven accepted 2025
papers above are meta-evaluation or diagnostic rather than resource-release, and one of them
(`Common Task Framework…`) is a critical evaluation of *scientific* ML specifically. This is
not inferred from scope text; these are accepted posters.

**Titles.** `Name: Descriptive subtitle`. Occasionally a question (`Is This Tracker On?`). Never
a bare assertive claim. A paper titled "Ordinary least squares outranks four published
baselines" would be stylistically off; "NullBench: a metadata-null control for CFD surrogate
benchmarks" is exactly on.

**Length.** Nine pages plus unlimited appendix. **Against ~14,700 words this is a structural
rewrite, not a reformat** — roughly a 60% cut of the main text with the remainder moved to
appendices.

**References.** Conference norm, 40–70; not a constraint.

**Abstract house style.** Problem with current evaluation → the artifact, named → what running
it reveals → release statement (code, data, licence, Croissant metadata).

### C. Editorial signals — the richest on this list, and all of it recent

- **CFP, 2026** [P]: evaluation is defined as *"the full set of processes, tools, datasets,
  benchmarks, and practices used to test, stress-test, audit, compare, and interpret AI/ML
  systems."* Explicitly solicited: *"Analyze strengths, limitations, or failure modes of
  existing benchmarks or evaluation practices"*; *"Study benchmark saturation or overfitting and
  their impact on scientific conclusions."* And verbatim: **"Submissions need not introduce a
  new model or outperform prior work."** *"Negative results, critical analyses, and
  use-case-inspired evaluations"* are named as welcome.
- **FAQ, 2026** [P]: negative results *"are welcome in ED track … as long as they bring new
  insights and are thoroughly demonstrated via empirical evaluations"*, including
  **"failure modes of current benchmarks."** Out of scope: *"papers where evaluation is
  secondary to novel methodology."*
- **Reviewer guidelines, 2026** [P], the decisive text:
  > *"Originality does **not** necessarily require introducing an entirely new method."*
  > *"Originality may be achieved through novel task design, evaluation setup, or analysis that
  > reveals properties of existing benchmarks. **Beating a baseline is not required.**"*
  > *"Negative results are valuable when rigorously supported."*

  Code release is **mandatory** for executable artifacts; for analytical work it is *"highly
  encouraged"* with a justification required if omitted.
- **Track-introduction blog, 2026-03-23** [P]: the shift is to treat evaluation as *"a
  scientific object in its own right — one that can be studied, stress-tested, reproduced,
  audited, and improved."* What they want *less* of: purely descriptive data releases,
  *"data without clarifying the intended problem formulation, evaluation setup, or interpretive
  boundaries."*
- **Chairs' 2025 retrospective, 2025-09-30** [P]: 1,995 submissions (1,820 in 2024; 987 in
  2023), acceptance targeted at the main track's ~25.8%. Chairs flag **limited reviewer
  diversity and domain coverage** as an open problem — a genuine risk for a CFD paper here.
- **Track blog, 2025-12-05** [P]: focused on metadata/Croissant standardisation; **no** mention
  of critique, negative results, or physics. The critique mandate is a **2026 addition**, which
  means the 2026 CFP is a stronger signal than the 2025 record, and the first cohort selected
  under it will not be visible until December 2026.

### D. Reverse-engineered strategy for NeurIPS ED

**This is the only venue on the list whose *written acceptance criteria* match this paper's
contribution word-for-word.** The reviewer guidelines' "Originality does not necessarily require
introducing an entirely new method" is the precise inverse of the sentence that produced both
desk rejections.

- **Title shape.** `Name: subtitle.` The repository already has the name:
  *"NullBench: metadata-only controls and measure-dependence audits for CFD surrogate
  benchmarks."* Drop "NeuroForge" entirely — it belongs to the system paper.
- **What the abstract leads with.** The artifact and what running it reveals, in that order:
  a metadata-null control and a measure-re-weighting audit, run across five CFD benchmarks under
  each one's own split and metric; the null spans R² 0.10–0.97 and no benchmark reports it; the
  field-MSE ranking on AirfRANS reverses under the dataset's own node measure; corrected
  leaderboards and a tested harness are released.
- **Foreground.** (i) `null_travels.md` in full — five benchmarks, pre-registered decision rule
  committed at `8928087` before any number existed, **including the WindsorML counterexample**.
  The counterexample is not a weakness here; it is what converts an attack into a measurement,
  and it is the structural move that made Errica et al. (ICLR 2020) and Ferrari Dacrema et al.
  (RecSys 2019) land. (ii) The harness — `nullbench_release.md` is a **first-class contribution**
  at this venue, not engineering overhead, and code release is mandatory anyway. (iii) The
  pre-registration discipline and the "How I tried to break this" section (11 adversarial
  controls), which map directly onto "rigorously supported."
- **Bury.** The residual/trust-layer material entirely. The Transolver-vs-interpolator
  *localisation physics* becomes an appendix; the reviewer pool here will not adjudicate
  boundary-layer claims, and the chairs themselves name domain coverage as a weakness.
- **Is the negative framing survivable?** It is not merely survivable, it is **the solicited
  category.** This is the only venue on the list where that is true.
- **The two hard costs.** (1) **Eight-month wait.** (2) **Double-blind against a live arXiv
  preprint (2607.10333), a coined system name that runs through the manuscript, and a GitHub URL
  containing the author's username** — `venue_plan.md` §5 costs this at about half a day and the
  fix (a system-name macro in `preamble.tex`, an anonymous mirror) applies unchanged. A 9-page
  rewrite is the larger cost by far.
- **What an editor/AC here would consider the contribution:** evidence that a published
  evaluation protocol produces conclusions that do not survive a control the protocol omits —
  plus the reusable control.

---

## 3. Nature Machine Intelligence — the genre precedent

### A. What they published recently

**A.1 — the last 15 months of primary content.** OpenAlex S2912241403,
`from_publication_date: 2025-06-01`, 50 most recent [P]. The distribution, characterised:

- **Biology / chemistry / medicine (≈40%)**: NucleicBERT (RNA language modelling), VITAL
  (peptide–protein interaction), HelixFold-S1, single-cell response prediction, survival
  prediction under covariate shift, ImmunoStruct.
- **LLM behaviour and evaluation (≈20%)**: *Causal evidence that language models use confidence
  to drive behaviour* (2026-09-07), *Implicit-bias-like patterns in reasoning models*
  (2026-09-01), *Capable language models can outgrow the benefits of collaboration*
  (2026-07-24), *When large language models are reliable for judging empathic communication*
  (2026-02-11), *Benchmarking large language models on safety risks in scientific laboratories*
  (2026-01-14).
- **Robotics / neuroscience / cognition (≈20%)**: ergoCub, tensegrity robots, soft hand
  exoskeleton, free-recall models, cognitive maps.
- **Physics / PDE / operator learning — thinner than it first looks**. Primary research, 15
  months:
  - *Principled approaches for extending neural architectures to function spaces for operator
    learning* (2026-07-03)
  - *Enabling local neural operators to perform equation-free system-level analysis* (2026-07-15)
  - *Quantum neural operators with implicit quadratic frame and expressivity advantages*
    (2026-09-03)
  - *Harnessing implicit neural representations for scientific data compression* (2026-08-24)
  - *Learning the coupled dynamics of global climate modes* (2026-06-01)

  **Not primary research, and excluded from that count:** *Enhancing reproducibility in hybrid
  Earth system models* (2026-08-28, 147 refs) is a **Perspective** [S, verified via the article's
  own summary: *"This Perspective introduces a framework for assessing reproducibility…"*], and
  *Towards general auditory intelligence for machine listening and speaking* (2026-08-14, 214
  refs) is almost certainly a Review (**UNVERIFIED**). **Five primary physics/PDE papers in
  fifteen months, none of them CFD** — this strengthens, not weakens, the high desk-reject
  estimate in §3.D. The Perspective is still a useful datum for section C: NMI is actively
  publishing *reproducibility-methodology* front matter in a simulation-science domain.
- **Front-matter / opinion, a large and visible stream**: *The epistemic debt of generative AI*
  (2026-08-26), *Thinking and rethinking data AI readiness* (2026-07-24), *Four questions for
  AI-ready biological data* (2026-07-10), *Solutions, challenges and rising tensions in AI and
  mathematics* (2026-06-23), *The brain is a diverse place, why not computing?* (2026-07-13).
- **A dedicated reproduction lane**: *Reusability report: Exploring the utility and extensibility
  of an integrated modelling framework for liquid electrolyte design* (2026-07-30).

**A.2 — the critique record.** OpenAlex sweep on critique terms, S2912241403 [P]. Specific
critique-genre papers surfaced:

| Title | Date | characterisation |
|---|---|---|
| **Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations** | **2024-09-25** | **Analysis**; 182 refs; the genre precedent |
| *A flaw in using pretrained protein language models in protein–protein interaction inference models* | 2026-02-13 | a named methodological flaw in a published modelling practice |
| *Explainable AI reveals Clever Hans effects in unsupervised learning models* | 2025-03-17 | shortcut-learning exposure |
| *Large language models that replace human participants can harmfully misportray and flatten identity groups* | 2025-02-17 | negative-leaning finding about a practice |
| *Navigating molecular OOD-ness* | 2026-06-09 | front matter on evaluation under distribution shift |
| *Personalized uncertainty quantification in artificial intelligence* | 2025-04-23 | UQ |

**The genre precedent, verified at source** [P],
`nature.com/articles/s42256-024-00897-5?error=cookies_not_supported`:

- **Article type: Analysis.** Not an Article. Published **25 September 2024**. **182 references.**
- It attracted a **News & Views**: Brandstetter, *"Envisioning better benchmarks for machine
  learning PDE solvers"*, **13 December 2024**, `s42256-024-00962-z` [S].
- **No Matters Arising or reply was found.** The critique was published, amplified by the
  journal, and not contested in print.

### B. Taste, inferred from that evidence

**Dominant contribution type: a method or system with a demonstrated scientific or societal
consequence, in biology, chemistry, medicine, robotics or LLM behaviour.** ML-for-PDE and
fluids is a *minority* topic — five papers in fifteen months, none of them CFD.

**Do they publish critique? Yes, as a standing lane, and they publish it about this exact
subfield.** Three distinct instruments: the **Analysis** type (McGreivy), the **Reusability
Report** type, and an **unbylined Editorial** stream that takes positions.

**Titles.** Full declarative sentences stating the finding: *"Weak baselines and reporting
biases lead to overoptimism in …"*, *"Capable language models can outgrow the benefits of
collaboration"*, *"A flaw in using pretrained protein language models …"*, *"Causal evidence
that language models use confidence to drive behaviour"*. **This is the only venue on the list
where an assertive claim title is the house style rather than a liability.**

**Length — verified content-type limits** [S], `nature.com/natmachintell/content`:

| Type | Main text | Abstract | Display items | Refs |
|---|---|---|---|---|
| **Analysis** | **≤3,500 words** (excl. abstract, Methods, refs, legends) | 100–150 words, unreferenced | **≤6 figures/tables** | (McGreivy carried 182; no hard cap enforced) |
| Comment | 1,500–2,000 words | — | — | ≤15 |
| Matters Arising | ≤1,200 words | — | — | ≤15 |

**Note a contradiction I could not resolve:** one search extraction reports a 50-reference limit
on Analysis; McGreivy's published Analysis carries **182**. The published record wins. Treat the
50 as belonging to a different Nature title. **UNVERIFIED.**

**Abstract house style.** 100–150 unreferenced words that state the *conclusion about the field*
first, then the evidence base and its size, then the implication. The mechanism is never the
lead. This is the exact inverse of C&F.

**Cost model.** **Hybrid.** Subscription (non-OA) route available; Gold OA APC is
**£9,390 / $12,850 / €10,850** [S], `nature.com/natmachintell/submission-guidelines/publishing-options`.
**Passes the no-APC gate via the subscription route** — and note that non-primary content types
are *not eligible* for OA and can only be published subscription-route.

### C. Editorial signals

**The strongest and most on-point editorial signal of any venue profiled here.**

- **Unbylined Editorial, 27 January 2025**, `s42256-025-00989-w`, *"Machine learning solutions
  looking for PDE problems"* [P, verified at source]. Opening verbatim: *"Machine learning
  models are promising approaches to tackle partial differential equations… However, in speaking
  with several experts about progress in the area, questions are emerging over what realistic
  advantages machine learning models have and how their performance should be evaluated."*
  **The journal's own editorial board has publicly stated that how ML-for-PDE performance should
  be evaluated is an open and pressing question.**
- **News & Views, 13 December 2024**: Brandstetter, *"Envisioning better benchmarks for machine
  learning PDE solvers"* [S] — *"stronger benchmark problems are needed for the field to
  advance."*
- **Editorial, ~24 April 2026**, on reproducibility and code reporting [S]: renews the
  **Reusability Reports** format and states that *"a renewed focus on reproducibility and
  transparency in code reporting seems warranted"* as output accelerates under LLM adoption.
  Related: *Recognizing reproducibility and reusability in times of fast science*,
  `s42256-026-01219-7` [S].
- Other 2026 editorials [S]: *"Stop 'tokenmaxxing' and deploy AI sensibly instead"* (May 2026),
  *"On the troubling rise of generative AI suspicion in academic publishing"* (30 Jan 2026),
  *"Solutions, challenges and rising tensions in AI and mathematics"* (23 Jun 2026). The stream
  is monthly and consistently takes normative positions about research practice.

**Read together: NMI opened this conversation (Sep 2024), amplified it (Dec 2024), editorialised
on it (Jan 2025), and has renewed its reproducibility commitment (Apr 2026). It has not
published a follow-up Analysis in the ML-for-PDE genre since.** That is an open slot, not a
closed one.

### D. Reverse-engineered strategy for NMI

**What an editor here would consider the contribution: a general claim about how a scientific
field constructs its evidence, demonstrated at scale, of interest beyond CFD.**

- **Article type: Analysis.** Not Article. The type definition — *"a new analysis of existing
  data … in a comparative analysis that leads to novel and arresting conclusions of importance
  to a broad audience"* — is a literal description of `null_travels.md`.
- **Title shape: a declarative sentence stating the finding.** E.g. *"Metadata-only controls and
  unstated error measures explain much of the reported progress in machine-learning
  computational fluid dynamics."* Assertive, hedged by "much of," naming both mechanisms.
- **What the abstract leads with.** The field-level conclusion, not the mechanism: *across five
  CFD benchmarks, a regression on published metadata — no flow field read — matches or beats
  published surrogates on two, and the strength of that control varies by an order of magnitude
  across benchmarks without any of them reporting it.* Then the measure result as the second
  mechanism. Then the implication for reporting standards.
- **Foreground.** Breadth and generality. **Five benchmarks, not one.** The R² 0.10–0.97 spread
  *is* the result and it is stated that way in `null_travels.md` §4 already. The WindsorML
  counterexample must be prominent — NMI published McGreivy partly because he was scrupulous,
  and an unfalsifiable attack would not clear this bar. The DrivAerNet++ reporting-practice
  finding (§2.2: the authors ran the parametric null, put it in Figure 5 on a different split,
  and never placed it beside Table 4) is the most NMI-flavoured single paragraph in the
  repository — it is a claim about *reporting structure*, which is the genre.
- **Bury or cut entirely.** Everything about NeuroForge. The trust layer. The residual
  falsification. The harness becomes a code-availability statement. The AirfRANS-specific
  boundary-layer band ladder becomes a Methods/SI figure — at NMI the near-wall physics is
  *supporting detail*, the opposite of its role at C&F.
- **Does the harness help?** Neutral-to-helpful as an artifact under the April 2026
  reproducibility editorial, but it is **not** the contribution and must not be named in the
  title or abstract.
- **Is the negative framing survivable?** **Yes — it is the precedent.** McGreivy is a
  pure-critique Analysis with 182 references, published and amplified, uncontested in print.
- **Rework cost: SEVERE, and this is the honest headline for this venue.** 14,700 words → 3,500.
  10 figures + 9 tables → ≤6 display items. Abstract → 100–150 words. This is not a rewrite of
  the paper; it is a *different paper* built from the same evidence, and it would need the
  cross-benchmark material (`null_travels.md` + `nullbench_release.md`) as its spine with
  AirfRANS demoted to one case among five.
- **The honest risk.** McGreivy surveyed **82 articles** and made a claim about an entire
  literature. This paper measures **five benchmarks and one head-to-head**. NMI's bar is
  "importance to a broad audience," and a deep, careful audit of five CFD benchmarks may read to
  a professional editor as narrower than the precedent it is invoking. **Desk-rejection
  probability at NMI is high — I would put it above 80%** — but the cost of trying is a cover
  letter and a presubmission enquiry, and NMI answers presubmission enquiries. **Send a
  presubmission enquiry before building the 3,500-word version.** That single action is the
  highest-information, lowest-cost move available across all five venues.

---

## 4. Engineering Applications of Artificial Intelligence (Elsevier)

### A. What they published recently

**A.1 — the raw firehose.** OpenAlex S900972176, `from_publication_date: 2026-06-01`,
unfiltered: **meta.count = 847** [P]. That is ~847 items in ~14 weeks, roughly **60 papers per
week**. The 30 most recent (2026-09-07 to 2026-09-10) [P]:

Multi-modal knowledge graph reasoning survey · Few-shot PIV denoising in industrial aerodynamics
· maritime anomaly detection · hierarchical consensus optimisation for group decision-making ·
power-transformer diagnosis under unseen load-voltage conditions · **physics-embedded GNNs for
million-node thermal-fluid digital twins of converter transformers** · multimodal entity linking
· AUV formation tracking control · apple-leaf disease detection · fault-propagation path
identification · farmland remote-sensing retrieval with MLLMs · multi-finger force estimation
from sEMG · soil classification · in-cylinder pressure virtual sensing · QR-code validation ·
*Editorial Board* · DLP 3D-printing deformation compensation · AC optimal power flow · fuzzy
elderly health monitoring · road geometry/friction/visibility sensing · crude-oil price
forecasting from news sentiment · building energy management RL · circular intuitionistic fuzzy
MADM · HIV/HBV co-infection neuro-stochastic framework · ground-target intention recognition ·
Korean speech-recognition error correction · acupuncture diagnosis multi-agent framework ·
EV sentiment Markov chain · *Editorial Board* · landslide mapping.

**A.2 — the CFD/surrogate slice.** Two targeted searches [P] returned, in 2026 to date:
*A framework for realisable data-driven active flow control … simplified truck wake*
(2026-08-06); *Operator learning methods for modeling interfacial dynamics of rising bubble*
(2026-04-04); *A hybrid couple-task surrogate operator with Fourier space–time encoding …
reservoir seepage* (2026-03-11); *Accelerating Particle-in-Cell simulations in Tokamak
Scrape-off Layer using segmented surrogate models* (2026-02-27); *Noninvasive pressure
difference mapping … physics-informed conditional variational learning* (2026-06-24);
*Novel doughnut-based cooling device … CFD* (2026-05-31). Fluids is present but marginal, and
always as an *application*.

### B. Taste, inferred from that evidence

**Dominant contribution type: an application paper — a novel AI pipeline applied to a named
engineering problem, validated on a dataset, reporting an improvement.** Essentially 100% of
the record. Titles are long compound noun phrases naming method-then-application.

**Do they publish critique or negative-leaning results? No.** The critique sweep
(2019 onward, same terms as C&F) returns **meta.count = 4** [P], across 14,649 works, and none
is a critique of an evaluation protocol:

1. *When meaning meets purpose: A survey on semantic and task-oriented communications* (2026-08-31) — survey
2. *Microwave imaging methodologies for breast cancer detection and their progress with deep learning* (2026-04-11) — review
3. *Enhanced automated code vulnerability repair using large language models* (2024-09-10) — method
4. *Generation of synthetic full-scale burst test data for corroded pipelines using TGAN* (2022-08-13) — method

**References: 4–67, median ≈ 29** (n = 29) — noticeably shorter bibliographies than C&F or CPC.
A 42-entry reference list is above the EAAI median. **Length and figures: UNVERIFIED**;
`venue_plan.md` §7 item 4 [S, single source] lists no length cap but flags it as needing an
in-browser check, and that check has **not** been done.

### C. Editorial signals

Only two EAAI editorials since mid-2024, both special-issue introductions [P]:
*Metaheuristics for sustainable manufacturing* (2025-08-26) and *Emerging industrial
digitalisation and ML applications in maintenance engineering* (2024-12-03). Neither is a scope
statement.

**The binding signal is the guide's four desk-rejection conditions** [S], Editor-in-Chief
Prof. Patrick Siarry; the journal is an official IFAC publication:

> Papers which do not respect the following conditions will be desk-rejected without being sent
> for peer review: papers on new metaphor-based metaheuristics are very rarely accepted; **the
> abstract should clearly specify which is the contribution in AI and which is the application
> in engineering**; undefined acronyms in title and abstract are forbidden; single-column format.

Plus: *"Submitted papers should report some **novel aspects of AI** used for a **real world
engineering application** and also **validated using some public data sets** for easy
replicability"*, and **"novelty should be made clear in the first two pages."**

### D. Reverse-engineered strategy for EAAI

- **The desk screen asks a question this paper cannot answer.** "Which is the contribution in
  AI, and which is the application in engineering?" — the contribution is a *negative
  measurement about how AI results are scored*, and the application is *a public research
  benchmark, not a deployed system*. The published record contains no precedent for either
  half. **"Novelty clear in the first two pages"** is the phrase that produced two desk
  rejections already, restated as a formal screening condition.
- **What would have to change.** The paper would have to be rebuilt as an *engineering decision
  tool*: "before adopting a neural surrogate in an aerodynamic design loop, run this
  metadata-null control and this measure-sensitivity audit; here is the tool, here is what it
  returns on five public datasets, here is the acceptance criterion." Title: *"A metadata-null
  and measure-sensitivity screening framework for validating machine-learning surrogates in
  external aerodynamic design."* Contribution-in-AI = the control estimator and audit procedure;
  application-in-engineering = surrogate qualification for aerodynamic design. The **harness is
  an asset here** — it is the only venue besides NeurIPS where shipping a tool helps, because
  EAAI's own scope demands public-dataset replicability.
- **What must be buried.** Every sentence that reads as a claim about the literature. The
  corrected leaderboards. The word "null" in the title.
- **Is the negative framing survivable?** **No.** Four critique-adjacent papers in 14,649 works,
  none of them an evaluation critique, against an explicit novelty screen. The finding must be
  re-expressed as a *positive tool* or it will not pass the desk.
- **Rework cost: HIGH (reframe, not reformat)**, plus double-anonymised compliance
  (`venue_plan.md` §5), plus an **unverified** length cap against ~14,700 words.
- **Honest read: the IF 9.0 is a mirage for this paper.** It is earned by a very high-volume
  application literature that this paper is not part of.

---

## 5. Computer Physics Communications (Elsevier)

### A. What they published recently

OpenAlex S142305363, `default.search: machine learning neural network`,
`from_publication_date: 2024-01-01`, 30 most recent of 50 [P]:

| # | Title | Date | refs |
|---|---|---|---|
| 1 | **WaveDL**: A Scalable Deep Learning Framework for Wave-Based Inverse Problems | 2026-09-01 | 41 |
| 2 | One-shot acceleration of transient PDE solvers via online-learned preconditioners | 2026-07-10 | — |
| 3 | **PINNIES**: An efficient physics-informed neural network framework for integral operator problems | 2026-06-06 | — |
| 4 | Numerically optimizing shortcuts to adiabaticity: A hybrid control strategy | 2026-06-03 | 34 |
| 5 | **PhySimNet**: Accelerating PDE simulations with physics-integrated learning | 2026-05-20 | 23 |
| 6 | **NAS-PINNv2**: Improved neural architecture search framework for PINNs in low-temperature plasma simulation | 2026-04-21 | — |
| 7 | Physics-informed GNNs for transverse momentum estimation in CMS trigger systems | 2026-04-18 | — |
| 8 | **msmJAX**: Fast and differentiable electrostatics on the GPU in Python | 2026-04-12 | — |
| 9 | The software landscape for the density matrix renormalization group | 2026-03-20 | 181 |
| 10 | Extending automated potential development workflow for legacy DFT reuse | 2026-02-26 | 23 |
| 11 | **pySIMsalabim**: a Python package to extend drift-diffusion modelling | 2026-02-22 | 57 |
| 12 | Fast and accurate quasi-atom method for atomistic/continuum simulation of solids | 2026-02-11 | 37 |
| 13 | Scalable neural network driven molecular dynamics simulation | 2026-01-21 | 37 |
| 14 | Efficient GPU-accelerated training of a neuroevolution potential with analytical gradients | 2025-12-13 | 35 |
| 15 | **PyLIT**: Reformulation and implementation of the analytic continuation problem | 2025-10-24 | 92 |
| 16 | Converting sWeights to probabilities with density ratios | 2025-10-07 | 22 |
| 17 | Physics-informed neural networks for supersonic flow over cones | 2025-07-23 | 67 |
| 18 | **Mlacs**: Machine learning assisted canonical sampling | 2025-07-17 | 88 |
| 19 | **WaterLily.jl**: A differentiable and backend-agnostic Julia solver for incompressible viscous flow | 2025-07-15 | 38 |
| 20 | Incremental pressure correction method for subsonic compressible flows | 2025-07-15 | 79 |
| 21 | Spatio-temporal neural operator on complex geometries | 2025-07-15 | 56 |
| 22 | Physics aware ML for micromagnetic energy minimization | 2025-06-16 | 64 |
| 23 | **Tadah!** a Swiss army knife for developing and deployment of ML interatomic potentials | 2025-06-16 | 30 |
| 24 | PINNs with trainable sinusoidal activation functions for Navier–Stokes | 2025-05-15 | 58 |
| 25 | One-to-one correspondence reconstruction at the electron-positron Higgs factory | 2025-05-15 | 47 |
| 26 | Generalizable models of magnetic hysteresis via physics-aware RNNs | 2025-05-05 | 80 |
| 27 | **Hotspice**: design, verification and applications of a Monte Carlo simulator for artificial spin ice | 2025-04-30 | 97 |
| 28 | Discovery and inversion of the viscoelastic wave equation in inhomogeneous media | 2025-04-05 | 59 |
| 29 | Ensemble and deep learning methods for J/ψ mass estimation | 2025-02-13 | 102 |
| 30 | ML-enhanced predictors for accelerated convergence of partitioned FSI simulations | 2025-01-31 | 76 |

### B. Taste, inferred from that evidence

**Dominant contribution type: a named, released, archived scientific *program*, demonstrated on
a physics problem.** Of the 30, at least **11 carry a capitalised software name in the title.**
The remainder are method-plus-implementation papers. There is no "analysis of a literature"
category and no evaluation category.

**Do they publish critique? No — zero.** The same critique-term sweep over 2019–2026 returns
**meta.count = 0** [P] across 14,108 works. This is the cleanest negative finding in the
document.

**Titles.** `Name: what it does` or a plain method statement. The name is a feature, not a
liability — this is the only Elsevier venue here where "NeuroForge" would read as normal.

**References: 22–181, median ≈ 57** (n = 24) — the longest bibliographies of the three Elsevier
titles. A 42-entry list is *below* the CPC median.

**Article types** [S, from `venue_plan.md` §2 #4, 2026-09-07]: **CPiP** (Computer Programs in
Physics — program archived in the CPC Program Library, mandatory Program Summary, approved
open-source licence) and **CP** (Computational Physics). Scope bar, verbatim: effectiveness
*"evidenced by the author(s) within the context of a substantive problem in physics"*; CPiP
encourages *"a problem of contemporary interest in physics that cannot be solved by current
software."*

### C. Editorial signals

No CPC editorials appear in the OpenAlex `type:editorial` sweep since 2024-06 [P]. The only
recent scope-adjacent item found is a special issue *"Advances in physics aware machine
learning"* [S]. **No editorial-level scope narrowing or widening was found in the last 18
months.** CPC's ML engagement is real but is channelled through the software lane.

### D. Reverse-engineered strategy for CPC

- **What an editor here would consider the contribution: the program.** Not the finding.
- **This means filing the harness, not the paper.** Title: *"NullBench: a program for
  metadata-null and measure-sensitivity controls in machine-learning benchmarks for
  computational fluid dynamics."* Mandatory Program Summary, licence (repo is MIT), Program
  Library deposit. The five corrected leaderboards become the *demonstration of effectiveness*.
  The measure-asymmetry and covariate-null results become §4 "Results obtained with the
  program" — real content, but structurally subordinate to the code.
- **Foreground:** the software (CLI, tests, hash manifest, Zenodo DOI, 2.4 MB of metadata and
  zero field data to reproduce every number — that last figure is a genuine CPC-flavoured
  selling point). **Bury:** the critique framing, entirely. Every claim becomes "the program
  computes X; on benchmark Y it returns Z."
- **Is the negative framing survivable? Only if invisible.** Zero critique papers in seven
  years is not a gap to fill; it is a category the venue does not have.
- **Does the harness help?** It is the *only* thing that helps here. This is the one venue where
  `nullbench_release.md` is the paper.
- **The scope risk `venue_plan.md` §2 #4 already identified stands and is confirmed by the
  record:** the CPC record is DFT, molecular dynamics, lattice QCD, plasma, spin ice, HEP
  triggers. **A 2-D airfoil RANS benchmark audit is not obviously "a substantive problem in
  physics" to that board**, and CPC sits in the same Elsevier computational-physics family as
  JCP, which desk-rejected this paper five days ago.
- **Rework cost: HIGH** — restructuring around a Program Summary is a different paper.

---

## 6. Venues I judged relevant and added

### 6.1 Nature Computational Science (Nature Portfolio) — **the strongest addition**

OpenAlex S4210228084 [P]. Hybrid; subscription route available; Gold OA APC
£9,390/$12,850/€10,850 [S] — **passes the no-APC gate.** Article and Analysis are both ≤3,500
words [S].

**Why it is relevant: its published record is full of the benchmark-and-evaluation genre**,
where NMI's is not. Critique-term sweep, `from_publication_date: 2024-01-01`, meta.count = 72,
top entries [P]:

- *Rethinking privacy-preserving GWAS benchmarks for governance* (2026-07-23)
- *UniFFBench: evaluating universal machine learning force fields against experimental
  measurements* (2026-07-14) — **evaluating a model class against ground truth it was never
  scored on; the nearest structural analogue to this paper anywhere in a journal**
- *The Quantum Optimization Benchmarking Library* (2026-06-23) and *Setting benchmarks for
  practical quantum utility of combinatorial optimization* (2026-06-23)
- *Benchmarking alignment methods for spatial transcriptomics data* (2026-04-03)
- *Pitfalls and prospects of quantum machine learning* (2025-12-22)
- *Adaptive validation strategies for real-world clinical artificial intelligence* (2025-11-17)

**The catch, and it is a real one.** A fluids/PDE search on the same source
(`fluid dynamics simulation surrogate neural operator PDE`, 2024-06 onward) returns
**meta.count = 1** [P]: *A scalable framework for learning the geometry-dependent solution
operators of partial differential equations* (2024-12-09). **NCS publishes the genre but not the
domain**; NMI publishes the domain's critique but rarely the domain. Neither is a clean fit.

**Strategy if pursued:** identical to NMI §3.D (3,500-word Analysis, declarative title,
five-benchmark spine), but the framing shifts from "this field over-reports" to "benchmarks for
computational surrogates need a metadata-null control; here is what one reveals across five" —
matching the `UniFFBench` / `Pitfalls and prospects` register rather than the McGreivy register.
**Presubmission enquiries to NMI and NCS should go out together**; Nature Portfolio editors
routinely transfer between the two, and an NMI decline with an NCS suggestion is a good outcome.

### 6.2 Considered and rejected, with the reason

| Venue | Verdict | Reason |
|---|---|---|
| **Physical Review Research** | OUT | Published *Adversarial vulnerability of machine-learning surrogates for CFD* (2026-07-01) — right genre, wrong cost model: fully Gold OA with a mandatory APC. Fails `no-apc-venues-only`. |
| **Machine Learning: Science and Technology (IOP)** | OUT | Fully OA £2,500 + 8,500-word cap (`venue_plan.md` §3). |
| **Data-Centric Engineering (Cambridge)** | OUT | Fully OA. |
| **Physics of Fluids** | OUT | The 2026 "Physics First" editorials (`venue_plan.md` §3, [S] 2026-09-07) are a live scope narrowing against exactly this kind of paper. |
| **JCP, CMAME** | OUT | Already desk-rejected; JCP will not reconsider. |
| **TMLR** | OUT on the author's stated preference for journal identity + IF — but note it is free, has no length limit, reviews on *claim–evidence correctness rather than novelty*, and is the single best *criteria* match after NeurIPS ED. If the IF constraint ever softens, revisit it first. |
| **ICLR / ICML main track** | OUT | 9-page rewrite for a paper that is not a method contribution; ED track dominates on criteria. |
| **Journal of Computational Science / Engineering with Computers** | Held | `venue_plan.md` §5 scope-safe landings; low desk risk, no taste evidence gathered here. Unchanged. |

---

## 7. Cross-venue comparison

| | **Computers & Fluids** | **NeurIPS ED Track** | **Nature Machine Intelligence** | **EAAI** | **CPC** | *(add)* **Nature Comp. Sci.** |
|---|---|---|---|---|---|---|
| Dominant contribution | new method/closure in a solver | named dataset / benchmark / protocol | method or analysis with scientific consequence | AI applied to an engineering problem | named, archived physics **program** | method or benchmark in computational science |
| Critique papers in record | **2 in 7 yrs**, both on numerical methods | **7+ accepted in 2025 alone** | **yes, standing lane** (Analysis, Reusability Report) | **4 in 14,649 works**, none an eval critique | **0 in 14,108 works** | **yes, frequent** (Pitfalls/Rethinking/evaluating) |
| Critique *of ML evaluation* | **none found** | yes, incl. SciML (`Common Task Framework`) | **yes — McGreivy is the precedent** | none | none | yes, adjacent domains |
| Domain fit (external aero CFD) | **excellent** | good (DrivAerStar, UniFoil, ML4CFD, AirfRANS's home) | weak (5 physics papers/15 mo, none CFD) | marginal | weak (no CFD benchmark culture) | **very weak (1 PDE paper)** |
| Title house style | descriptive noun phrase, no names | `Name: subtitle` | **declarative claim sentence** | method-for-application compound | `ProgramName: what it does` | mixed; `Name:` and "Benchmarking X" |
| Length constraint | **none** [V 2026-09-07] | **9 pp + appendix** | **3,500 words, ≤6 items** | unverified | none stated | **3,500 words** |
| Refs (median, this scan) | ≈37 (15–77) | n/a (40–70 typical) | ≈44 non-review | ≈29 (4–67) | ≈57 (22–181) | n/a |
| Review model | **single-anonymized** [V] | **double-blind** | single (editor-led triage) | **double-anonymized** | assume single | single (editor-led triage) |
| Cost (no-APC gate) | **PASS** — subscription, no fee [V] | **PASS** — free | **PASS** — subscription route | **PASS** — subscription | **PASS** — subscription | **PASS** — subscription |
| Rework from current MS | **LOW** (retitle, reorder, cut trust layer) | **HIGH** (9 pp + anonymisation) | **SEVERE** (3,500 w; different paper) | **HIGH** (reframe as tool) | **HIGH** (restructure as program) | **SEVERE** (3,500 w) |
| Next possible submission | **now** | **~May 2027** | now (presub first) | now | now | now (presub first) |
| Does the harness help? | no — demote to data statement | **yes — first-class, mandatory** | neutral, back matter | **yes** | **yes — it *is* the paper** | neutral |
| Negative framing survivable? | only if reframed as assessing a *discretisation* | **yes — explicitly solicited** | **yes — precedent exists** | **no** | **no** | yes, in the "Pitfalls" register |
| Desk-reject risk for *this* paper | MEDIUM | LOW | **HIGH (>80%)** | **HIGH** | HIGH | HIGH |
| Prestige | IF 3.0 | top-tier ML | IF ~21 | IF 9.0 | IF 3.9 | IF ~12 |

---

## 8. Ranking — expected value under the constraint set, *not* taste fit

**Read the header literally, because the two orders differ and conflating them would be the
worst mistake a venue-choosing agent could make here.**

- **Taste fit alone** — which venues publish this paper's genre — ranks:
  **NeurIPS ED > NMI ≈ NCS > C&F > EAAI ≈ CPC.**
- **Ranked below is expected value**, which folds in the deadline, the no-APC gate, the
  ~14,700-word length, the double-anonymity cost against a live preprint, and the rework budget.

**The honest statement about the #1 pick: Computers & Fluids ranks first on expected value
*despite* being the worst taste fit among the venues that publish this genre at all.** Section
1.B found **zero** critiques of an ML evaluation protocol in seven years at C&F; the entire
precedent for placing this paper there is **two 2023 papers titled "A critical assessment of
[a numerical method]"**. C&F wins on availability, length tolerance, single-anonymized review,
domain competence and rework cost — not because its editors have shown appetite for this kind of
paper. If the question asked is purely "whose taste does this paper match," the answer is
NeurIPS ED, and it is not close.

**One prestige note:** the prestige order (NMI ≈ NCS > EAAI > CPC ≈ C&F, with NeurIPS
incommensurable) is close to the inverse of this ranking at the top. That is deliberate and it
is stated so it cannot be mistaken for an oversight.

### 1. Computers & Fluids — file next, with the retitle and the reframe

Not because its taste fits (it does not — zero ML-evaluation critiques in seven years) but
because it is the only venue where **the paper as it currently exists can be filed within days**,
the length is unconstrained, review is single-anonymized so nothing has to be anonymised, the
cost gate passes, and the *domain* fit is the best on the list. The measure-asymmetry result is
genuinely a CFD result — 64% of nodes inside 0.05c, a 418× representation ceiling, a monotone
906×→0.31× band ladder — and it can be written as an assessment of a scoring discretisation,
which is a category C&F *does* publish (twice, in 2023). **Conditions:** retitle away from any
claim about the field; lead the abstract with the measure asymmetry, not the covariate null;
insert the resolution ladder verdict-first; cut the trust layer; add the RSM lineage paragraph;
**submit to the regular issue, not the Ashton/Dwight special issue.**

### 2. NeurIPS Evaluations & Datasets Track — the best *criteria* match anywhere, and it is a 2027 plan

If the deadline were open this would rank first without qualification. The reviewer guidelines
say, in writing, that originality does not require a new method and that beating a baseline is
not required; the CFP solicits analysis of failure modes of existing benchmarks; negative results
are named as welcome; and the accepted 2025 cohort contains both the CFD-dataset genre
(DrivAerStar, UniFoil, ML4CFD retrospective) and the SciML-critique genre
(`Common Task Framework For a Critical Evaluation of Scientific Machine Learning Algorithms`).
AirfRANS itself was published here. **But the 2026 deadline passed on 6 May and the next intake
is ~May 2027.** The correct use of that eight months is not to wait: file at C&F now, and if C&F
declines, the 9-page `NullBench` version is the paper that goes to ED 2027 with
`null_travels.md` as its spine.

### 3. Nature Computational Science — send a presubmission enquiry this week

Ranked above NMI on fit despite lower prestige, because its *record* contains the genre at a
scale NMI's does not (`UniFFBench`, `Pitfalls and prospects of quantum machine learning`,
`Rethinking privacy-preserving GWAS benchmarks`, two quantum-benchmark papers in one week). The
domain gap is severe — one PDE paper in two years — so the enquiry costs a paragraph and either
opens a door or closes it cheaply.

### 4. Nature Machine Intelligence — presubmission enquiry, same week, expect a no

The genre precedent is exact and the editorial record is unambiguous: NMI published McGreivy as
an **Analysis**, amplified it with a **News & Views**, and **editorialised on it** in January
2025 with the sentence *"questions are emerging over … how their performance should be
evaluated."* Nothing in that lane has been filled since. But McGreivy surveyed 82 papers; this
paper audits five benchmarks. The scale gap is real and an NMI editor will feel it. Ranked below
NCS only because NCS's bar for "broad audience" is lower and its genre density is higher.
**Both enquiries should go out together and neither should trigger a 3,500-word rewrite until
one comes back positive.**

### 5. Computer Physics Communications — only if the decision is to ship the harness as the paper

Zero critique papers in 14,108 works is not a gap, it is an absent category. But CPC is the one
venue where `nullbench_release.md` *is* a publishable object, the named-program title style fits,
and the cost gate passes. This is a real option for the *artifact*, running in parallel with a
journal home for the *finding* — not a home for the finding itself.

### 6. Engineering Applications of Artificial Intelligence — do not file without a full reframe

IF 9.0 earned by ~60 application papers a week, a formal desk-screen condition demanding that the
abstract name "the contribution in AI and the application in engineering," an explicit "novelty
clear in the first two pages" rule, four critique-adjacent papers in 14,649 works, an unverified
length cap against ~14,700 words, and mandatory double-anonymisation against a live preprint and
a coined system name. **The IF is a mirage for this paper.** If it is pursued, it must become a
surrogate-qualification *tool* paper, and the harness leads.

### The one cheap, unambiguous action

**Send presubmission enquiries to Nature Machine Intelligence and Nature Computational Science.**
Each costs a paragraph, neither is a submission, both editorial offices answer them, and they are
the only low-cost way to test the highest-prestige branch before committing to a 3,500-word
rebuild. This recommendation stands on its own and does not depend on anything else in this
document.

**What I am deliberately not recommending: a submission *sequence*.** An earlier draft suggested
filing at C&F in parallel and withdrawing if an enquiry came back positive. That is withdrawn.
Withdrawing a manuscript from an Elsevier journal after submission is not free — it consumes
goodwill at a venue this project may need as a fallback, and this author has already exhausted
CMAME and JCP. The task asked what each venue would have to look like to accept this paper, and
that is what sections D answer; the ordering of filings is a decision for whoever owns the
submission, with the deadline facts in §2 and the rework costs in §7 as inputs.

---

## 9. Confidence and caveats

- **Strong [P] evidence:** every published-record list, every `meta.count`, every OpenAlex-
  derived reference count, all NeurIPS 2026 CFP/FAQ/reviewer-guideline quotes, the NeurIPS 2025
  accepted-paper titles and venue labels, the McGreivy article type (**Analysis**, 182 refs,
  25 Sep 2024), and the NMI editorial of 27 Jan 2025 read at source.
- **[S] only:** Elsevier guide-for-authors details, NMI/NCS content-type word limits and APCs,
  the C&F special-issue scope and guest-editor names, EAAI's desk-rejection conditions,
  Neil Ashton's authorship of AhmedML/WindsorML/DrivAerML, **R. P. Dwight's Associate
  Editorship of Computers & Fluids**, and the Perspective classification of
  *Enhancing reproducibility in hybrid Earth system models*.
- **Two corrections made after first draft, recorded rather than silently applied:** (i) the
  claim that submitting to C&F's regular issue routes around the special-issue editors is
  **withdrawn** — Dwight is a standing Associate Editor (§1.C); (ii) the NMI physics/PDE count
  is reduced from six to **five primary-research papers** after one was identified as a
  Perspective (§3.A). Both corrections cut *against* the venues concerned.
- **Not verified and material:** whether Neil Ashton holds any C&F editorial-board role. The
  ScienceDirect board page 403s; a search of it returned no Ashton listing, which is weak
  evidence of absence. **Check this in a browser before filing**, because it determines whether
  the "opposed reviewers" step in §1.C is sufficient.
- **[V] inherited, 2026-09-07:** C&F length/abstract/keyword/highlights/review-model/cost rows
  from `venue_plan.md` §12.
- **UNVERIFIED and flagged as such:** article page-lengths and figure counts at all three
  Elsevier titles; EAAI's word cap; the Analysis reference limit at NMI (50 vs McGreivy's 182).
- **Inaccessible, not inferred:** OpenReview reviews and meta-reviews (browser challenge);
  nature.com listing pages (IdP redirect loop); ScienceDirect article and special-issue pages
  (403).
- **A finding stated as a finding, not an impression:** across 35,000+ indexed works at
  C&F + EAAI + CPC, targeted seven-year sweeps returned **six** critique-adjacent papers and
  **zero** critiques of a machine-learning evaluation protocol. None of those three venues has
  published this paper's genre. That is the constraint the positioning has to survive, and it is
  not visible in any of their scope statements.
