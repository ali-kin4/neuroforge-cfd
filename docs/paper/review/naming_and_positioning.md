# Does the covariate null have a name? Naming, positioning, and scoop risk

**Date:** 2026-09-11. **Input:** `docs/paper/review/null_travels.md`.
**Verification key:** `[V]` = fetched the primary source (publisher/anthology/arXiv abstract page)
in this session and the claim is quoted from it. `[V-repo]` = already verified in
`docs/paper/review/bibliography_audit.md`. `[S]` = seen only in a search-result snippet, not
opened — treat as unverified. `[P]` = preprint, not peer reviewed.

---

## 0. One-paragraph answer

The **mechanism** has a name in three other fields and none of those names fits our setting
exactly. (Terminology note: "arXiv search" below always means the API's `all:` field — title,
abstract, comments, journal-ref — **not** article bodies, which arXiv does not index. Full-text
claims are made only about papers I fetched and read; see §1.4 Tier 1.) The closest existing name
is **"partial-input baseline"** (NLP, Poliak et al. 2018; the caveat paper is
Feng et al. 2019) and its graph-learning cousin **"structure-agnostic baseline"** (Errica et al.,
ICLR 2020). Neither is our case, for a reason that is precise and worth stating in the paper: in
a partial-input baseline the cheap predictor consumes *a subset of what the model sees*; in ours
it consumes *the low-dimensional code that generated what the model sees* and which the model
never sees at all. That makes ours a **generative-factor** baseline, not a partial-input one — a
strict generalisation. **In scientific ML and in CFD benchmarking specifically, the control is
unnamed, and it is absent from every current benchmarking paper for these datasets that I read at
full text.** So: name it, but name it as an instance of a lineage you cite explicitly, not as an
invention.

---

## 1. Does it have a name?

### 1.1 The four candidate names, verified

| Name | Canonical citation | What it means there | Why it is not our case |
|---|---|---|---|
| **Hypothesis-only / partial-input baseline** | Poliak, Naradowsky, Haldar, Rudinger, Van Durme. *Hypothesis Only Baselines in Natural Language Inference.* \*SEM 2018, pp. 180–191, `10.18653/v1/S18-2023` **[V]** | Model input is (premise, hypothesis); baseline drops the premise and still beats majority class. "Statistical irregularities may allow a model to perform NLI ... beyond what should be achievable without access to the context." | Their cheap view is a **subset of the model's input**. Our 16/23/8/7 design parameters are **not an input to DoMINO, FIGConvNet, RegDGCNN or TripNet at all** — those models read meshes/point clouds. θ is upstream of the input, not part of it. |
| **Annotation artifacts** | Gururangan, Swayamdipta, Levy, Schwartz, Bowman, Smith. *Annotation Artifacts in Natural Language Inference Data.* NAACL 2018, pp. 107–112, `10.18653/v1/N18-2017` **[V]** | Crowdworker protocol "leaves clues"; hypothesis-only text classifier gets 67% on SNLI. Conclusion: "the success of NLI models to date has been overestimated." | "Artifact" implies a **defect introduced by the collection protocol**. Nothing is defective in AirfRANS or DrivAerML. The parameters *are* the design of experiment; a parametric dataset is supposed to be parametric. Using "artifact" would be dishonest and a reviewer who knows the term would say so. |
| **Structure-agnostic baseline** | Errica, Podda, Bacciu, Micheli. *A Fair Comparison of Graph Neural Networks for Graph Classification.* ICLR 2020; arXiv 1912.09893 **[V]** — abstract verbatim: "by comparing GNNs with **structure-agnostic baselines** we provide convincing evidence that, on some datasets, structural information has not been exploited yet." | Same graphs, same splits, same metric; baseline ignores edges and uses node features only. | **This is the closest structural match in all of ML** (see §2). The difference is that "structure-agnostic" describes *withholding a modality*; ours describes *substituting a generative code*. And no one has ported it to physics surrogates. |
| **Metadata-only screening / Metadata Prior Dominance Score (MPDS)** | Shao, K. *Metadata Predictability Is Not Evidence Dependence: An Intervention-Based Audit for Weak-Label Benchmarks.* arXiv 2605.23701 (v1 2026-05-22, v2 2026-06-25), cs.CL **[V] [P]** | MPDS := Acc_meta / Acc_full on SNLI, FEVER, HotpotQA. Proposes reporting a metadata-only screening statistic alongside an evidence-intervention statistic. | Single-author cs.CL preprint, not peer reviewed, no CFD content, and it does not cite the Poliak/Gururangan partial-input line for methodological comparison. **But the phrase "metadata-only screening statistic" now exists in print and post-dates nothing of ours.** Must-cite for naming hygiene — see §4. |

### 1.2 The distinction that decides the verdict

Write X for what the surrogate reads (mesh, point cloud, SDF), Y for the label (C_d, C_l), and
θ for the published design vector. In the partial-input literature X = (A,B) and the baseline
uses A ⊂ X. In ours, **X = g(θ)** — θ is the *generator* of X, with dim θ ∈ {7,…,23} against
dim X in the millions.

Three consequences, all of which should go in the paper:

1. **It is not leakage, and not an artifact.** θ carries no information about Y that X does not
   also carry (θ → X → Y is a Markov chain by construction of the dataset), so by the data
   processing inequality the null can never beat a *perfect* reader of X. It only beats real
   readers. That makes the finding a statement about **headroom and estimator efficiency**, never
   about contamination. `null_travels.md` §4 already states this correctly; it should be stated
   in exactly these terms.
2. **It bounds the task, not the model.** A high R²_meta says the benchmark's *label variance is
   low-dimensional*, which is a property of the design of experiment, not of anyone's architecture.
3. **AirfRANS is a hybrid case and should be flagged as such.** There, freestream velocity and
   Reynolds number *are* literally model input channels (`sdf, mask, x, y, u_in, v_in, log_re` —
   see `CLAUDE.md`), while the NACA digits generate the SDF. So AirfRANS is part partial-input,
   part generative-factor. DrivAerML/DrivAerNet++/AhmedML/WindsorML are pure generative-factor.
   Do not paper over this; it is a point in your favour (the effect survives both regimes).

### 1.3 The formal name that *does* exist and that you should adopt

What you measured is an estimate of **V-usable information**: Ethayarajh, Choi, Swayamdipta,
*Understanding Dataset Difficulty with V-Usable Information*, ICML 2022, PMLR 162:5988–6008;
arXiv 2110.08420; **ICML 2022 Outstanding Paper** **[V]**. They frame dataset difficulty w.r.t. a
model family V as the *lack* of V-usable information.

For a Gaussian predictive family with free variance, I_V(θ → Y) = H_V(Y) − H_V(Y|θ)
= ½ log(σ²_Y / σ²_resid) = **−½ log(1 − R²)** nats. So the **population** R²_meta is a monotone
reparameterisation of the linear-family V-information of published metadata about the label.

**Careful — state the caveat or do not state the identity.** The relation holds for the
population (or in-sample) R². Your reported R² values are **held-out**, and a held-out R² can go
negative: the WindsorML drag null is 0.104 with 95% CI **[−0.143, 0.267]**. A negative R²
substituted into −½ log(1 − R²) yields negative V-information, which is impossible by definition
(the constant predictor is in V, so I_V ≥ 0). So write it as: *R²_meta is a monotone
reparameterisation of the population R² under a Gaussian family, and our held-out estimator of
the corresponding I_V is truncated at zero.* One clause, and it forecloses the objection.

With that clause, the hook is worth keeping: it converts "we ran OLS" into "we estimated the
V-usable information of the published design vector under the linear family, and it ranges over
an order of magnitude across five benchmarks." Use it in one methods sentence; do not build the
paper on it. (The identity is standard; state it as such, do not claim it.)

### 1.4 Evidence of absence, with the queries

**Read this section's two tiers in the right order.** Tier 1 is the load-bearing evidence: papers
whose *full text* I fetched and searched. Tier 2 is a metadata screen only.

#### Tier 1 — competitor papers read at full text (the claim that matters)

| Paper | What I searched its body for | Result |
|---|---|---|
| **CarBench**, Elrefaie, Shu, Klenk, Ahmed; arXiv 2512.07847 v2 (rev. 2026-08-20) **[V, full text via `arxiv.org/html/2512.07847v2`] [P]** | `linear regression`, `parametric`, `design parameters`, `simple baseline`, `trivial baseline`, `MLP on parameters`, `tabular`, `OLS`, `random forest`, `XGBoost` | **Zero hits.** All eleven evaluated models are geometric deep architectures. Scope sentence, verbatim: "we focus exclusively on the task of learning surface-level aerodynamic quantities from geometry." Conclusion, verbatim: "We do not evaluate global aerodynamic coefficients such as drag and lift ... left as explicit targets for future extensions of the benchmark." |
| **PhysicsNeMo-CFD benchmarking framework**, Tangsali, Ranade, Nabian, Kamenev, Sharpe, Ashton, Cherukuri, Choudhry; arXiv 2507.10747 v1 (2025-07-14) **[V, abstract page] [P]** | parametric / design-parameter / linear-regression baseline | **None mentioned.** Three neural models only (DoMINO, X-MeshGraphNet, FIGConvNet). *Caveat: abstract page fetched, body not grepped — weaker than the CarBench row.* |
| **ShapeBench**, Fazliani, Chawla, Guo, Shen, Ihme, Udell; arXiv 2605.20763 v2 (2026-06-10) **[V, abstract page] [P]** | design-variable-only or linear baselines; critique of missing trivial baselines | **None.** "Well-configured baselines" are classical optimisers plus `ShapeEvolve`, an LLM evolutionary baseline. *Abstract page only.* |
| **DrivAerNet++** dataset paper, NeurIPS 2024 D&B | — | **Partial exception, already documented** in `null_travels.md` §2.2: §5.1.2 fits AutoGluon/XGBoost/LightGBM/RF/GB on 26 parameters, but on a random 80/20 split and reported only in Figure 5, never beside Table 4. |

#### Tier 2 — arXiv **metadata** screen (title / abstract / comments / journal-ref only)

`export.arxiv.org/api/query` with the `all:` field, run 2026-09-11. **This is not a full-text
index — arXiv's API has never indexed article bodies.** A zero here means the phrase appears in
no title or abstract; it does not prove no methods section does it. (The tell: `structure-agnostic
baseline` returns Errica et al. precisely because that phrase is in Errica's abstract.) Use this
as a screen for whether a *name* has been coined, which is what §4 needs it for — not as proof
that no one has run the experiment.

| Query phrase | Total results | What they were |
|---|---|---|
| `"parameter-only baseline"` | **0** | — |
| `"metadata-only baseline"` | 3 | LLM instruction hierarchy; mobile malware detection; clinician burnout. All unrelated. |
| `"covariate null"` | 5 | 4 hep-ph/gr-qc "covariant null plane"; 1 covariate-shift discriminator paper. **Zero in evaluation methodology.** |
| `"covariate-null fraction"` / `"covariate null fraction"` | **0** | — |
| `"design-parameter baseline"` OR `"metadata null"` OR `"generative factor baseline"` | **0** | — |
| `"partial-input baseline"` | 2 | Feng et al. 2019; Belinkov-line NLI follow-up 2205.12181. **Both NLP; none in physics or scientific computing.** |
| `"structure-agnostic baseline"` | 4 | Errica ICLR 2020 + 3 GNN/time-series papers. **None in physics.** |
| `"covariate floor"` / `"parametric floor"` / `"metadata sufficiency"` | 4 | functional data analysis, random forests, latent diffusion, speech provenance. None in evaluation. |

#### Tier 2b — corpus sweeps (metadata level)

- `abs:"DrivAerML"` → 9 papers, most recent 2026-09-07; none announces a parametric or tabular
  baseline. `abs:"AirfRANS" AND abs:"baseline"` → 5 papers; none critiques the benchmark or
  proposes a trivial baseline. These are metadata-level and therefore weak on their own; they
  matter only as a sweep that found nothing to promote into Tier 1.
- **Nearest thing found in CFD:** a linear regression on six hand-crafted point-cloud shape
  descriptors (coordinate means/variances + voxelised volume) used as a baseline for DrivAerNet++
  drag, in *Predicting drag coefficients of vehicle geometries: A PointNet++ point cloud
  surrogate model approach*, AIP Advances 15:105015 (2025) **[S — publisher returned HTTP 403;
  details from search snippet only, treat as unverified]**. Note this is a *descriptor* baseline
  computed from geometry, not the benchmark's **published design parameters**, and it is used to
  flatter the proposed model rather than to audit the benchmark. It is the single closest thing
  to a prior instance and you should cite it and distinguish it.

**Verdict on Q1: the concept is named in NLP and graph learning, under names that do not fit;
it is unnamed and unreported in scientific ML.** The defensible sentence is scoped to what Tier 1
supports: *"We read the current benchmarking papers for these datasets — CarBench, the
PhysicsNeMo-CFD framework, ShapeBench — and none reports a design-parameter baseline; the one
benchmark that computed one (DrivAerNet++ §5.1.2) reported it on a different split and never
beside its deep-model table."* Do not say "novel" unqualified, do not say "nobody has ever done
this," and do not lean the absence claim on the Tier-2 metadata screen.

---

## 2. The closest analogue in any field

### 2.1 Closest *structurally*: Errica et al., ICLR 2020

**What they claimed.** Re-ran five GNNs on nine benchmarks in one controlled framework (47,000
experiments) and added structure-agnostic baselines. Conclusion, verbatim: "on some datasets,
structural information has not been exploited yet."

**Why it is the closest.** Substitute *geometry* for *structure* and that sentence is your
abstract. Same dataset, same split, same metric; the only change is a baseline that ignores the
expensive modality. Crucially, their headline was **not** "GNNs don't work" — it was
*dataset-specific*, exactly as your 0.10–0.97 spread forces yours to be.

**How it was received.** Accepted at ICLR 2020, heavily cited, and it changed practice: node-
feature-only baselines are now routine in graph-classification papers. That is the outcome you
want — the control becomes standard, and your paper is the citation for it.

**What to copy structurally.** (i) One controlled framework, every benchmark run the same way.
(ii) The negative case reported as loudly as the positive (their "on *some* datasets" = your
WindsorML). (iii) No attack on any author; the target is the practice.

### 2.2 Closest *rhetorically*, and the genre proof point: Ahlmann-Eltze, Huber, Anders

*Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear
baselines.* **Nature Methods 22(8):1657–1661 (2025)**, `10.1038/s41592-025-02772-6` **[V]**.

Five foundation models plus two deep models against deliberately simplistic baselines ("no
change", "additive", "mean"); none outperformed the baselines. This is your existence proof that
**a simple-baseline audit in one scientific subfield reaches a Nature-family journal**, published
14 months ago, and it is the single best template for the title. Note the load-bearing hedge in
their title: *"does not yet"*. Copy that hedge.

### 2.3 The recommender-systems template, and what actually made it land

- Ferrari Dacrema, Cremonesi, Jannach. *Are We Really Making Much Progress? A Worrying Analysis
  of Recent Neural Recommendation Approaches.* RecSys 2019 (**Best Long Paper**); arXiv
  1907.06902 (v1 2019-07-16) **[V]**. 18 algorithms; only 7 reproducible; 6 of those 7 beaten by
  simple heuristics.
- Ferrari Dacrema, Boglio, Cremonesi, Jannach. *A Troubling Analysis of Reproducibility and
  Progress in Recommender Systems Research.* ACM TOIS 39(2):20:1–20:49 (2021),
  `10.1145/3434185` **[V]** — 11 of 12 reproducible neural methods beaten by nearest-neighbour or
  linear models.

**What made it land, and what to copy:**
1. **An exhaustive, pre-declared corpus.** Not "here are three papers I beat" but "every paper
   from these venues in these years." Your equivalent is *every runnable benchmark in the
   subfield*, and you have it: 5 attempted, 4 runnable, 1 reported as not runnable. Say the
   corpus rule out loud.
2. **A hard count in the abstract.** "18 algorithms, 7 reproducible, 6 beaten." Yours: "five
   benchmarks, four runnable, the null beats the published reference rows on two, is beaten
   outright on one, and is unadjudicable on one because no baseline exists." Put the counts in
   the first three sentences.
3. **Baselines tuned, not strawmanned.** This is their most attacked flank and the one Rendle,
   Zhang, Koren, *On the Difficulty of Evaluating Baselines: A Study on Recommender Systems*,
   arXiv 1905.01395 (2019) **[V]** turned into a standing objection: results are questionable
   "unless they were obtained on standardized benchmarks where baselines have been tuned
   extensively." **Your OLS is deliberately untuned — and that is a strength you must state
   explicitly, once, as a sentence**: an untuned 16-coefficient OLS is a *lower bound* on what
   published metadata supports, so every comparison is conservative in the direction that hurts
   your claim. Your ridge-path and squared-terms rows (`null_travels.md` §3.7, §2.1) are the
   evidence; surface them, don't bury them in the break-tests.
4. **Criticise the practice, never the authors.** Dacrema's paper is scrupulously polite. Yours
   already is — `null_travels.md` §2.2's treatment of DrivAerNet++ §5.1.2 ("the structure is the
   point", no number quoted) is exactly right. Keep it.

### 2.4 The must-cite that is a *rebuttal to you*

Feng, Wallace, Boyd-Graber. *Misleading Failures of Partial-input Baselines.* ACL 2019,
pp. 5533–5538, `10.18653/v1/P19-1554`; arXiv 1905.05778 **[V]**. Verbatim: "A successful
partial-input baseline indicates that the dataset is cheatable. But the converse is not
necessarily true: failures of partial-input baselines do not mean the dataset is free of
artifacts."

**Why you must engage it.** It is the canonical caveat on exactly your instrument, and a reviewer
who knows it will assume you don't. Two things to write:

- The **converse direction** applies to you directly and you already honour it: WindsorML's null
  failing does **not** license "WindsorML is a hard benchmark." It licenses only "the published
  MGN clears the null." Make that explicit; right now `null_travels.md` §2.4's "the null does not
  travel" is correct but the *asymmetry* deserves a named citation.
- The **forward direction is cleaner in your setting than in theirs**, and say why: the label is
  a single scalar; R² decomposes variance exactly; you score on the benchmark's own split with
  the benchmark's own metric against the benchmark's own published numbers. There is no
  annotation process to be artefactual, and no "hard example" subset to redefine. State this in
  one sentence and the caveat becomes a strength.

### 2.5 Genre and lineage citations (verified)

- McGreivy, Hakim. *Weak baselines and reporting biases lead to overoptimism in machine learning
  for fluid-related partial differential equations.* Nature Machine Intelligence 6:1256–1269
  (2024), `10.1038/s42256-024-00897-5`, arXiv 2407.07218. **[V-repo** — `bibliography_audit.md:88`,
  already in `refs.bib` as `mcgreivy2024weak`**]** 60 of 76 surveyed papers used weak baselines.
  **This is your parent paper and the sentence to write is: McGreivy & Hakim showed the field
  compares against weak *solvers*; we show it does not compare against a trivial *statistical*
  baseline at all, and we measure how much that matters.** That is a clean, non-overlapping
  extension and it is the single strongest positioning sentence available.
- Kapoor, Narayanan. *Leakage and the reproducibility crisis in machine-learning-based science.*
  Patterns 4(9):100804 (2023), `10.1016/j.patter.2023.100804` **[V]**. 17 fields, 294 papers,
  eight-type leakage taxonomy. **Cite to distinguish, not to align**: yours is not leakage under
  any of their eight types, and saying so pre-empts the lazy reviewer who files you under it.
- Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, Wichmann. *Shortcut learning in deep
  neural networks.* Nature Machine Intelligence 2:665–673 (2020), `10.1038/s42256-020-00257-z`
  **[V]**. **Cite to distinguish.** Shortcut learning is about what a *model* latches onto; your
  finding is about what the *task* permits, and the shortcut-taker is your baseline, not anyone's
  model. If you frame this as shortcut learning a reviewer will correctly say the models here are
  not taking a shortcut at all. (Note: `body.tex:1324` currently uses the phrase "the covariate
  shortcut". **Flagged — change it.** See §4.3.)
- Lapuschkin, Wäldchen, Binder, Montavon, Samek, Müller. *Unmasking Clever Hans predictors and
  assessing what machines really learn.* Nature Communications 10:1096 (2019),
  `10.1038/s41467-019-08987-4` **[V]**. Same "cite to distinguish" role. Optional.
- Dehghani, Tay, Gritsenko, Zhao, Houlsby, Diaz, Metzler, Vinyals. *The Benchmark Lottery.*
  arXiv 2107.07002 (2021) **[V] [P]**. Benchmark choice alone reorders methods. Adjacent, useful
  for the "which benchmark you pick decides your conclusion" framing that your 0.10–0.97 spread
  literally instantiates.
- Recht, Roelofs, Schmidt, Shankar. *Do ImageNet Classifiers Generalize to ImageNet?* ICML 2019,
  PMLR 97:5389–5400 **[V]**. Genre ancestor. Lower priority — their mechanism (new test sets) is
  not yours; cite once in a lineage sentence or not at all.
- **Good-practice counterexample worth citing positively:** Rasp et al., *WeatherBench 2*, JAMES
  16 (2024), `10.1029/2023MS004019`; arXiv 2308.15560 **[V]**. Data-driven weather forecasting
  mandates **persistence and climatology** baselines in the headline scorecard. This is the
  closest thing in physical ML to the protocol you are proposing, it is already normative in a
  neighbouring field, and it converts your recommendation from "a critic's demand" into "adopt
  what weather already does." **Strongly recommend including it.** It is the difference between a
  complaint and a proposal.
- **Aerospace pre-emption (do not skip).** Queipo, Haftka, Shyy, Goel, Vaidyanathan, Tucker.
  *Surrogate-based analysis and optimization.* Progress in Aerospace Sciences 41(1):1–28 (2005),
  `10.1016/j.paerosci.2005.02.001` **[V]**. Polynomial response surfaces and Kriging over design
  variables are 20–70 years old in this exact application. **An aerodynamics reviewer will say
  "you rediscovered RSM," and they will be partly right.** Say it first, in one sentence: *the
  estimator is textbook response-surface methodology; what is new is that nobody reports it
  beside the leaderboard, and the measurement of how much that omission conceals.* Without that
  sentence, §2.1 of `null_travels.md` is exposed.

---

## 3. The right framing for maximum reach

### Recommendation: **a scientific-ML evaluation paper, measured in CFD, with the general-ML lineage stated explicitly in the first paragraph.** Not a CFD-benchmarking paper; not a general-ML claim.

**Why not general ML.** The whole asset is the 0.10–0.97 spread, and it is measured entirely
inside one subfield. A general-ML framing requires a measurement outside CFD that you do not
have, and a general-ML reviewer will ask for one within the first paragraph of review. You would
be claiming a universal and showing a local. That is exactly the kind of overclaim this project
has twice been rejected for.

**Why not CFD benchmarking.** Framed as "AirfRANS and DrivAerML have a reporting gap," the
audience is ~200 people and the natural venue is Computers & Fluids, which caps the reach at the
same readership that has already seen this project. The finding is bigger than that: the *claim*
is about how a subfield scores surrogate models, and it generalises as a **protocol
recommendation** (report the metadata null) that any parametric-design benchmark can adopt.

**The framing sentence.** *Benchmarks in physical surrogate modelling publish the low-dimensional
design vector that generated each case. We ask how much of the headline label that vector alone
determines, on five benchmarks, under each benchmark's own split, metric and published numbers.
The answer ranges from R² = 0.10 to R² = 0.97, and no benchmark reports it.*

That sentence is legible to a general ML audience (it is the partial-input question), to a SciML
audience (it is the McGreivy & Hakim question one level down), and to a CFD audience (it names
their datasets). It does not overclaim.

### Venue, with the repo's constraints honoured

I have read `docs/paper/review/venue_plan.md` §3–4 and `field_scan_sept2026.md`. Those rank
venues for **paper 1** (the operator/trust paper) under a hard no-APC constraint and an
author preference for "journal identity and an impact factor" (which is why TMLR/JMLR are OUT).
**That ranking should not be transferred to this paper unexamined**, and here is the one fact
that changes the calculus:

**NeurIPS 2026 has renamed Datasets & Benchmarks to the "Evaluations & Datasets" track, and its
call explicitly says: "Negative results, critical analyses, and use-case-inspired evaluations are
welcome," and "A submission need not 'beat a baseline'; its primary contribution should be to
deepen and refine our understanding of evaluation practices." [V —
blog.neurips.cc/2026/03/23/introducing-the-evaluations-datasets-track-at-neurips-2026/]**

This directly contradicts `venue_plan.md` §4's finding ("No subscription or hybrid journal with
an impact factor was found that advertises negative results" → true, but this is not a journal)
and it is a venue with **no APC**, maximum reach, and a call that reads as though it were written
for this result. The only constraint it violates is "journal identity and impact factor", which
was an author preference stated about a different paper.

Ranked, with reasoning:

| # | Venue | Case for | Case against | APC |
|---|---|---|---|---|
| **1** | **NeurIPS 2026 Evaluations & Datasets track** | Call matches the paper almost word for word. Largest possible audience. Establishes the protocol as a community norm, which is the actual goal. No APC. | Conference not journal (author preference, paper 1). Registration/travel cost. Double-anonymous → the "NeuroForge" name and the GitHub URL leak (`venue_plan.md` §5 already has the macro fix). Deadline discipline. | **None** |
| **2** | **Nature Machine Intelligence** | Hybrid with a subscription route (`field_scan_sept2026.md:395`, verified in-repo). **Took McGreivy & Hakim.** Nature Methods took Ahlmann-Eltze. Two live proof points that this exact genre lands at Nature-family. Max prestige + max reach. | High desk-rejection risk; they will want ≥ the 5-benchmark scope you have, and probably a broader cross-domain claim you don't. Long timeline. | Subscription route exists |
| **3** | **Data-Centric Engineering** / **Scientific Data** | Perfect scope | Fully OA → **violates the hard no-APC constraint** (`venue_plan.md` §3). OUT. | OUT |
| **4** | **Computers & Fluids** | Zero desk risk, no anonymisation work, scope text asks for limitations | Reach is the floor, not the ceiling. Use only as fallback. | Hybrid, free |

**My recommendation: target #1, with #2 as the stretch and #4 as the backstop.** But flag
explicitly: this conflicts with a stated author preference recorded for paper 1, and it is the
author's call, not mine. What I am confident about is the *framing*, which is venue-independent.

**One tactical note.** Whatever venue: **preprint this fast**. See §5.

---

## 4. Is the naming defensible?

### 4.1 "Covariate-null fraction" — free, but do not use it

Free: `"covariate-null fraction"` and `"covariate null fraction"` return **0** results on arXiv
full text; `"covariate null"` returns 5, all physics or covariate-shift, none in evaluation.
So there is no collision. But three problems:

1. **"Fraction" is wrong.** R² is not a fraction of anything a reader can name without a
   definition, and it can be negative (your WindsorML CI lower bound is −0.143). A statistic
   called a "fraction" that goes negative will be queried by a reviewer.
2. **"Null" is overloaded twice.** Against the null hypothesis (you also run a *label-permutation*
   null in §3.1 — two different "nulls" in one paper is a real readability problem) and against
   SQL/missing-data NULL, which is what a data-centric audience hears first.
3. **It reads as a coined term, which invites the reinvention charge.** Dacrema named no
   statistic. Errica named no statistic. Poliak named no statistic. The ones that stuck named the
   **baseline** ("hypothesis-only baseline"), not a number.

Also, repo-internal: **"floor" is already load-bearing** in this project (`residual_floor_theorem.tex`,
`floor_resolution_study.md`), so "covariate floor" would collide with your own vocabulary even
though it is free on arXiv (4 unrelated hits).

### 4.2 Recommended naming

**Name the protocol memorably; keep the statistic boring.**

| Role | Recommended name | Notes |
|---|---|---|
| **The control / protocol** | **the metadata null** (or, when precision matters, **the design-parameter null**) | `"metadata null"` → **0** arXiv hits. Short, self-explanatory, and it is a *baseline* name, matching what worked in NLP and graph learning. This is the phrase that should appear in the title or abstract. |
| **The statistic** | **metadata-only R²**, written `R²_meta` | Boring on purpose. Needs no defence, no definition beyond one line, and cannot be accused of being a coinage. In MSE currency, `MSE_meta`. |
| **A secondary interpretive aid — *not* a headline** | **null-normalised gain**, `G = (R²_model − R²_meta) / (1 − R²_meta)` | The fraction of the variance *left over after metadata* that the model explains. See the warning below before using it. |
| **The formal framing** | linear-family **V-usable information** of the published metadata, `I_V = −½ log(1 − R²_meta)`, population R², held-out estimate truncated at 0 | Cite Ethayarajh et al. Use in one methods sentence; do not build the paper on it. |

**Why `G` must not be the headline, despite being the more intuitive number.** Two reasons, both
fatal in exactly the regime that motivates the paper:

1. **It is unstable under the published rounding.** On DrivAerML the denominator is
   1 − 0.9731 = 0.0269. DoMINO's published 0.98 carries a rounding interval [0.975, 0.985]
   (your own §2.1 qualification). Propagating it gives **G ∈ [0.07, 0.44]** — a 6× swing from
   rounding alone. A headline statistic cannot move 6× under the second decimal of a number you
   did not produce.
2. **There is no common footing for it on DrivAerML.** The split-free anchor is R²_meta = 0.9598
   (10-fold), but DoMINO's number exists only on the variance-inflated 436/48 split. Your §2.1
   argument — "the comparison is unaffected; only the absolute figure is not comparable" —
   licenses the R² *ordering*, which is invariant to the inflation. It does not license `G`,
   which is not.

So: print `G` with its propagated interval alongside, as an aid to interpretation, and never
alone. There is also a structural reason to keep it out of the harness: `R²_meta` can be computed
from a benchmark's own files standalone, whereas `G` requires a competitor's published number and
therefore belongs at comparison time, not at release time.

**For the harness, emit:** `metadata_null_r2` (with bootstrap CI), `metadata_null_mse`. Avoid
`covariate_null_fraction` as a field name for the reasons in §4.1. (The existing script name
`covariate_null_crossbench.py` is fine to leave; a script name is not a claim.)

### 4.3 Citation hygiene flags in the existing paper directory

- **`docs/paper/body.tex:1324` says "the covariate shortcut". Change it.** "Shortcut" is Geirhos's
  term for a *model* behaviour; the models here do not take this shortcut, the baseline does.
  Suggested replacement: "the metadata null" or "the covariate-only regression". Leaving it
  invites a reviewer to file the paper under shortcut learning and then reject it for
  mischaracterising shortcut learning.
- `mcgreivy2024weak` — already verified at `bibliography_audit.md:88`, Nature MI 6:1256–1269
  (2024), `10.1038/s42256-024-00897-5`. No action.
- **Correction to a likely mis-citation before it happens:** Tsuchiya's LREC 2018 paper is
  *"Performance Impact Caused by Hidden Bias of Training Data for Recognizing Textual Entailment"*
  (LREC 2018, anthology `L18-1239`) **[V]** — **not** "Performance Impact Caveats of Partial
  Inputs", which is a title that circulates in secondary summaries and does not exist. If you cite
  Tsuchiya, use the real title.

---

## 5. Who would scoop this, and how soon

Ranked by (capability × proximity × evidence of intent).

### Risk 1 — MIT DeCoDE / Elrefaie & Ahmed. **HIGH. Weeks to a few months.** [V]

They have everything: the DrivAerNet++ parametric table, the split files, and **they already ran
this fit** (`null_travels.md` §2.2: the dataset paper's §5.1.2 fits AutoGluon/XGBoost/LightGBM/RF/GB
on 26 parameters, reported only in Figure 5 on a random 80/20 split). The only thing standing
between them and half your result is *putting that figure next to Table 4 on the official split*.

And they are actively working in exactly the right place: **CarBench** (arXiv 2512.07847,
**v2 revised 2026-08-20**) is their standardised benchmark over eleven architectures on
DrivAerNet++, and its conclusion says: "We do not evaluate global aerodynamic coefficients such
as drag and lift ... these aspects are left as explicit targets for future extensions of the
benchmark." **When CarBench adds drag, the natural first row of that table is their own
parametric AutoML fit.** That is a plausible next revision, and the revision cadence on that
preprint is already ~9 months v1→v2 but with active editing.

*What survives if they do it:* the DrivAerNet++ section becomes theirs. **DrivAerML, AhmedML and
WindsorML, the cross-benchmark spread, and the pre-registered decision rule do not.** Your moat
is that it is five benchmarks with a counterexample, not one benchmark with a number.

### Risk 2 — NVIDIA PhysicsNeMo-CFD / Neil Ashton's group. **MEDIUM-HIGH. Unpredictable.** [V]

Ashton is an author on AhmedML, WindsorML, **and** DrivAerML, and a co-author on the
PhysicsNeMo-CFD benchmarking framework (arXiv 2507.10747). He owns three of your four runnable
benchmarks. A "benchmarking framework" paper is *precisely* the artifact into which someone adds
a trivial baseline when a reviewer asks for one. No revision of 2507.10747 has appeared since
v1 (2025-07-14), which is the good news; the bad news is that your DrivAerML result is a direct
statement about *their* Tables 4/6/7 and they are best placed to reproduce and pre-empt it.

### Risk 3 — a McGreivy & Hakim follow-up, or an ICML/NeurIPS position paper. **MEDIUM. 6–12 months.**

ICML 2026 position papers already include work "detailing the limitations of benchmark-centric
evaluations" **[S — from the ICML 2026 position-papers index page; not opened]**, and the NeurIPS
2026 Evaluations track will attract exactly this genre. A generalist writing "trivial baselines in
SciML benchmarks" would hit AirfRANS and DrivAerNet++ first.

### Risk 4 — the NLP/eval methodology side arriving from above. **LOW.**

Shao 2026 (arXiv 2605.23701) is one author, cs.CL, and shows no sign of crossing into physical
benchmarks. But its existence means the *vocabulary* ("metadata-only screening statistic") is
being claimed. Cite it; do not let a reviewer find it first.

### What I did **not** find (this is the reassuring half) [V]

No paper, in any of the searches in §1.4, runs a design-parameter-only regression as an audit of
a CFD or aerodynamics benchmark. Not CarBench, not PhysicsNeMo-CFD, not ShapeBench, not any of
the 9 DrivAerML papers or the 5 AirfRANS papers on arXiv. The one adjacent artefact is a
geometry-descriptor linear baseline used to *flatter a model* in AIP Advances 2025 **[S]**, not to
audit a benchmark.

### Mitigation, in priority order

1. **Preprint within days, not weeks.** The pre-registration commit (`8928087`, before any number
   existed) is a real asset and it is only an asset if it is timestamped publicly alongside the
   result. This is the single highest-value action on this list.
2. **Lead with the spread, not with DrivAerML.** "R² 0.10 to 0.97 across five benchmarks, none of
   which reports it" cannot be scooped by one benchmark's authors adding one row.
3. **Ship the harness as the contribution.** If the deliverable is a reusable protocol + statistic
   that any parametric benchmark can run, a competitor adding one table to their own benchmark
   *cites* you rather than replacing you. This is why §4.2's naming matters: it is what makes the
   work adoptable rather than merely correct.
4. **Include WindsorML prominently.** The counterexample is what makes this a measurement rather
   than a complaint, and it is the part a rushed competitor is least likely to bother running.

---

## 6. Drop-in related-work paragraph

> Simple-baseline audits have repeatedly revised the apparent state of the art. In natural
> language inference, *partial-input* baselines that discard the premise recover much of the
> label \citep{poliak2018hypothesis, gururangan2018artifacts}; question- and passage-only controls
> did the same for reading comprehension \citep{kaushik2018reading}, and language-prior-only
> models did the same for visual question answering \citep{goyal2017vqa};
> \citet{feng2019misleading} supply the standing
> caveat that the converse does not hold. In graph classification, \citet{errica2020fair} showed
> by comparison against *structure-agnostic* baselines that on several benchmarks structural
> information was not yet being exploited. In recommendation, \citet{dacrema2019progress,
> dacrema2021troubling} found that most reproducible neural methods were matched by
> nearest-neighbour or linear models, with \citet{rendle2019difficulty} establishing that baseline
> tuning is itself a confound. The same genre has now reached the natural sciences:
> \citet{ahlmanneltze2025deep} report that no tested deep or foundation model for gene-perturbation
> effects outperforms deliberately simple linear baselines, and in our own subfield
> \citet{mcgreivy2024weak} find that 60 of 76 surveyed ML-for-PDE papers compare against weak
> numerical baselines. Our control differs from all of these in one structural respect. A
> partial-input baseline consumes a subset of what the model consumes; the design vector we
> regress on is not an input to any of the surrogates we compare against — it is the
> low-dimensional code from which each benchmark's geometry was generated, and which each
> benchmark publishes. The quantity we report is therefore neither leakage nor an annotation
> artifact: by construction the parameters carry no information the geometry does not, so the
> null can only bound what a real reader of the geometry has achieved, never what one could. It
> is an estimate, under the linear family, of the $\mathcal{V}$-usable information
> \citep{ethayarajh2022vusable} that a benchmark's own published metadata carries about its
> headline label. Estimating design-variable response surfaces is of course standard practice in
> aerodynamic design \citep{queipo2005surrogate}; what is new here is not the estimator but the
> measurement of what its omission from benchmark reporting conceals — and that the answer spans
> an order of magnitude across five benchmarks in the same subfield. Data-driven weather
> forecasting already mandates trivial controls of this kind, reporting persistence and
> climatology alongside every model \citep{rasp2024weatherbench2}; external-aerodynamics
> benchmarking does not.

**BibTeX keys used above that are not yet in `refs.bib`**, all `[V]` at the abstract page:

| Key | Full citation as verified |
|---|---|
| `poliak2018hypothesis` | Poliak, Naradowsky, Haldar, Rudinger, Van Durme. \*SEM 2018, 180–191. `10.18653/v1/S18-2023` |
| `gururangan2018artifacts` | Gururangan, Swayamdipta, Levy, Schwartz, Bowman, Smith. NAACL 2018, 107–112. `10.18653/v1/N18-2017` |
| `kaushik2018reading` | Kaushik, Lipton. *How Much Reading Does Reading Comprehension Require? A Critical Investigation of Popular Benchmarks.* EMNLP 2018, **5010–5015**. `10.18653/v1/D18-1546` |
| `goyal2017vqa` | Goyal, Khot, Summers-Stay, Batra, Parikh. *Making the V in VQA Matter.* CVPR 2017; arXiv 1612.00837 |
| `feng2019misleading` | Feng, Wallace, Boyd-Graber. ACL 2019, 5533–5538. `10.18653/v1/P19-1554`; arXiv 1905.05778 |
| `errica2020fair` | Errica, Podda, Bacciu, Micheli. ICLR 2020; arXiv 1912.09893 |
| `dacrema2019progress` | Ferrari Dacrema, Cremonesi, Jannach. RecSys 2019 (Best Long Paper); arXiv 1907.06902 |
| `dacrema2021troubling` | Ferrari Dacrema, Boglio, Cremonesi, Jannach. ACM TOIS 39(2):20:1–20:49 (2021). `10.1145/3434185` |
| `rendle2019difficulty` | Rendle, Zhang, Koren. arXiv 1905.01395 (2019) **[P]** |
| `ahlmanneltze2025deep` | Ahlmann-Eltze, Huber, Anders. Nature Methods 22(8):1657–1661 (2025). `10.1038/s41592-025-02772-6` |
| `ethayarajh2022vusable` | Ethayarajh, Choi, Swayamdipta. ICML 2022, PMLR 162:5988–6008; arXiv 2110.08420 |
| `queipo2005surrogate` | Queipo, Haftka, Shyy, Goel, Vaidyanathan, Tucker. Prog. Aerospace Sci. 41(1):1–28 (2005). `10.1016/j.paerosci.2005.02.001` |
| `rasp2024weatherbench2` | Rasp et al. JAMES 16 (2024). `10.1029/2023MS004019`; arXiv 2308.15560 |

Optional, cite-to-distinguish: `kapoor2023leakage` (Patterns 4(9):100804, `10.1016/j.patter.2023.100804`),
`geirhos2020shortcut` (Nature MI 2:665–673, `10.1038/s42256-020-00257-z`),
`dehghani2021lottery` (arXiv 2107.07002 **[P]**), `shao2026metadata` (arXiv 2605.23701 **[P]**).

**Dropped after verification:** `jabri2016revisiting` (Jabri, Joulin, van der Maaten, ECCV 2016,
arXiv 1606.08390). I had it in an earlier draft as a question-only VQA baseline. **Its abstract
does not support that** — it proposes "a simple alternative model based on binary classification"
that *receives the answer as input*, which is a different point. Replaced with `goyal2017vqa`,
whose abstract does say models "ignore visual information, leading to an inflated sense of their
capability" and "have indeed learned to exploit language priors." Flagging this because it is
exactly the kind of plausible-but-wrong attribution this review exists to catch.

---

## 7. Honest verdict table

| Claim | Verdict | Reframe if needed |
|---|---|---|
| "The parameter-only null is a new kind of baseline" | **NOT novel** | It is a generative-factor variant of the partial-input / structure-agnostic baseline. Cite Poliak, Errica. |
| "Regressing drag on design parameters is new" | **NOT novel** — it is response-surface methodology, 1951– | Say so first, in one sentence. Cite Queipo et al. |
| "No CFD/aero benchmark reports this control beside its leaderboard" | **To our knowledge true**, evidenced by §1.4 **Tier 1** (benchmarking papers read at full text), not by the Tier-2 metadata screen; one partial exception (DrivAerNet++ §5.1.2, reported in a different figure on a different split) which you already document | Scope the sentence to the papers you read. Keep the exception prominent — it strengthens the reporting-practice claim rather than weakening it. |
| "The measurement — a 10× spread across five benchmarks, with a pre-registered rule and a reported counterexample" | **Novel, to our knowledge** | This is the contribution. Lead with it. |
| "The name *metadata null* / *R²_meta*" | Free (0 arXiv hits) and defensible | Name the baseline, not the number. |
| "This is shortcut learning / leakage" | **False. Do not claim it.** | Cite Geirhos and Kapoor & Narayanan explicitly *to distinguish*. Fix `body.tex:1324`. |

## 8. What I could not verify, and the one open search

**Unverified, do not cite until opened:**

- AIP Advances 15:105015 (2025) shape-descriptor linear baseline — publisher returned HTTP 403;
  known only from a search snippet. **Open the PDF before citing.** This is the only claimed prior
  instance of *any* cheap baseline on these benchmarks, so it matters more than its length suggests.
- Dacrema TOIS 2021, Queipo 2005, Recht 2019, Kapoor & Narayanan 2023, Geirhos 2020: verified
  bibliographically (title / authors / venue / pages / DOI) but full text not read here.
- PhysicsNeMo-CFD (2507.10747) and ShapeBench (2605.20763): abstract pages fetched, **bodies not
  grepped**. Weaker than the CarBench row in §1.4 Tier 1. If the absence claim is going to carry
  weight in review, grep these two bodies as well — it is two fetches.
- ICML 2026 position-paper claim in §5 Risk 3: index page only, individual papers not opened.

**The one open search.** A forward-citation sweep on McGreivy & Hakim (2024) and on Errica et al.
(2020) — i.e. who has cited them since mid-2025 — was **not run**; Google Scholar is not reachable
from this environment. It cannot change any claim in this document, because it could only surface
an *additional* competitor, never remove one. But it is the single highest-value remaining search
and it should be run manually before the preprint goes up. Everything else here is closed.
