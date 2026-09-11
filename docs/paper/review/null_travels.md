# Does the covariate null travel beyond AirfRANS?

**Question.** On AirfRANS, ordinary least squares on the case parameters alone — no flow
field opened — ranks the official force coefficients at Spearman 0.9821 [0.9737, 0.9866]
for lift and 0.9318 [0.8981, 0.9531] for drag, fit on the official 800-case train split
and scored on the official 200-case test split. Four of five published baselines fall
below it on both. Is that a fact about AirfRANS, or about how this subfield scores
surrogate models?

**Answer, in one line.** It is not a fact about AirfRANS. A parameter-only regression
matches or beats published surrogates on **two** of the four runnable benchmarks, fails
decisively on a **third**, and sets an unmet bar on a **fourth** — and the ten-fold spread
across benchmarks is itself the finding, because no benchmark reports it.

**Artifacts.** Script `D:\Codes\Github\neuroforge-cfd\scripts\covariate_null_crossbench.py`
(decision rule committed at `8928087`, before any number in this report existed).
Results `D:\Codes\Github\neuroforge-cfd\results\review\covariate_null_crossbench.json`.

---

## 1. The headline table

Drag null, out of sample, in each benchmark's own published metric. Bracketed intervals
are 95% case-level percentile bootstrap.

| Benchmark | protocol | params | null R² (drag) | best published | verdict |
|---|---|---|---|---|---|
| **DrivAerML** | benchmark's own 436/48 split | 16 | **0.9731 [0.9581, 0.9826]** | DoMINO 0.98, FIGConvNet 0.97, X‑MGN 0.92 | null straddles the top two, **beats X‑MeshGraphNet** |
| **DrivAerNet++** | official split, full 1154-design test | 23 + 8 category tokens | **0.7365 [0.7050, 0.7643]** | dataset paper 0.596–0.643; TripNet 0.957 | **beats all three dataset-paper rows**, loses to SOTA |
| **AhmedML** | 10-fold OOS (no split published) | 8 | 0.684 [0.638, 0.727] const-area; 0.340 [0.277, 0.386] var-area | none found | sets a bar nobody has cleared |
| **WindsorML** | 10-fold OOS (60/20/20 recommended, no ids) | 7 | **0.104 [−0.143, 0.267]** const-area; 0.073 var-area | MGN, implied R² ≥ 0.79 | **null fails; published model clears easily** |
| DeepCFD | — | — | not runnable | — | fields only, no force labels |

The spread — 0.10 to 0.97 — is the result. An overall verdict of "it travels" or "it does
not" would throw away the most informative thing here.

---

## 2. Per benchmark

### 2.1 DrivAerML — the clean kill

This is the strongest case because nothing has to be adapted, matched or caveated.

**What was obtainable.** 16 published morph parameters per run
(`geo_parameters_all.csv`, 78 kB), published force coefficients under two normalisations
(`force_mom_all.csv`, `force_mom_constref_all.csv`), all from
`huggingface.co/datasets/neashton/drivaerml`. The dataset paper (arXiv 2408.11969)
publishes **no** split and **no** ML baseline. NVIDIA's PhysicsNeMo-CFD benchmarking
framework supplies both: `workflows/benchmarking/drivaer_ml_files/train.csv` (436 runs)
and `validation.csv` (48 runs), each carrying the drag-force label the published table is
scored on.

**Verification that the target is the right one.** The `drag` column in those split files
is proportional to the constant-reference-area C_d at **corr = 1.000000** over all 484
runs, and only 0.717 against the per-geometry-area C_d. So the null is fitting exactly the
quantity the published R² table reports.

**Published, read at source** — arXiv 2507.10747, *A Benchmarking Framework for AI models
in Automotive Aerodynamics* (NVIDIA PhysicsNeMo-CFD), Tables 4, 6 and 7. Preprint, not
peer reviewed. No seed spread reported.

| Model | R² drag (surface mesh, Tab. 6) | R² drag (10M point cloud, Tab. 7) | Spearman drag (Tab. 4) |
|---|---|---|---|
| X-MeshGraphNet | 0.92 | 0.85 | 0.96 |
| FIGConvNet | 0.97 | 0.97 | 0.99 |
| DoMINO | 0.98 | 0.97 | 0.99 |
| **null, 16 params, OLS** | **0.9731 [0.9581, 0.9826]** | same | **0.984 [0.956, 0.991]** |
| null, + squared terms | 0.9841 [0.9724, 0.9905] | same | 0.990 [0.972, 0.995] |

**Verdict: RESOLVED — the null travels, decisively.** Same 436 training runs, same 48
validation runs, same target, same metric, no subset mismatch and no metric adaptation.
X-MeshGraphNet is BELOW THE NULL under the pre-registered rule. FIGConvNet and DoMINO
straddle it: a 16-coefficient linear model is statistically indistinguishable from two
models that read multi-million-point geometry and cost GPU-days.

Two honest qualifications, neither of which touches the head-to-head:

* The published values are given to two decimals, so DoMINO's rounding interval
  [0.975, 0.985] overlaps the null's CI upper bound of 0.9826. "Straddles" is the
  conservative call and stays correct; third-decimal distinctions are not resolvable.
* The PhysicsNeMo split is built by sorting on drag and taking the top and bottom deciles.
  That **raises** the validation target's variance (val sd 70.2 N against train sd 46.9 N),
  which inflates R² relative to a random 10% — *for every model scored on it equally*. The
  split-free anchor is the 10-fold number on the same constant-area label: R² 0.9598
  [0.9523, 0.9659] linear, 0.9708 [0.9655, 0.9752] with squares. The comparison is
  unaffected; only the absolute figure is not comparable to a random-split R².

### 2.2 DrivAerNet++ — the null beats what the field cites, not what the field's best is

**What was obtainable.** 23 design parameters for 4,165 of the 8,121 designs
(`ParametricModels/DrivAerNet_ParametricData.csv`, 1.4 MB), drag for all 8,121
(`DrivAerNetPlusPlus_Cd_8k_Updated.csv`), official train/val/test id lists
(5819/1148/1154), frontal areas for 8,007. The remaining 3,956 designs are the
DrivAerNet-v1 fastbacks from a different 50-parameter scheme; **that parameter table is
not published** in the repository or on the Dataverse record, so those designs enter the
null through their categorical tokens only.

**The decisive row** is therefore a null that uses published parameters where they exist
and the category dummy elsewhere, fit on the 5,819 official train ids and scored on the
**whole** 1,154-design official test split — exactly the set the published models are
scored on. It is strictly weaker than a null with every design's parameters, so it is a
lower bound on what published metadata supports.

| Null feature set | n scored | R² | MSE | Spearman |
|---|---|---|---|---|
| intercept only | 1154 | −0.005 | 4.168e-4 | — |
| category tokens only | 1154 | 0.4480 [0.4039, 0.4865] | 2.290e-4 | 0.558 |
| **category + params where published** | **1154** | **0.7365 [0.7050, 0.7643]** | **1.093e-4** | 0.772 |
| + squared terms | 1154 | 0.7385 [0.7066, 0.7664] | 1.085e-4 | 0.772 |
| category + 23 params (parametric subset) | 559 | 0.8515 [0.8285, 0.8713] | 7.898e-5 | 0.923 |
| 23 params alone (subset) | 559 | 0.5160 [0.4608, 0.5636] | 2.575e-4 | 0.714 |
| + frontal area (**diagnostic only**) | 559 | 0.8510 [0.8279, 0.8709] | 7.924e-5 | 0.923 |

Frontal area is kept out of the headline deliberately: DrivAerNet++ normalises C_d by each
design's own effective frontal area, so it sits partly inside the label and needs the
geometry to compute. It buys nothing anyway (0.8515 → 0.8510).

**Published, read at source.**

| Model | R² | MSE | MAE | source |
|---|---|---|---|---|
| GCNN | 0.596 | 17.1e-5 | 10.43e-3 | NeurIPS 2024 D&B Table 4, proceedings PDF p. 8 |
| RegDGCNN | 0.641 | 14.2e-5 | 9.31e-3 | same |
| PointNet | 0.643 | 14.9e-5 | 9.60e-3 | same |
| TripNet | 0.957 | 9.1e-5 | 7.17e-3 | arXiv 2503.17400 **Table 5** |
| PointNet2D+BiLSTM | 0.9528 | 6.50e-5 | 6.046e-3 | arXiv 2601.02112 Table 1 (preprint) |

A trap worth flagging: TripNet's **Table 2** (TripNet 0.972, FIGConvNet 0.957) is the
DrivAerNet **v1** fastback task, not this one. Its Table 5 is the DrivAerNet++ table. The
two are easy to conflate and are kept apart here. On the v1 task the null cannot be run at
all, because the 50 parameters defining those 4,000 designs are not published.

**Verdict: PARTIALLY-RESOLVED, and the honest statement is the narrower one.** On the
identical full test split and in the benchmark's own metrics, the null is above all three
of the dataset paper's baselines on **both** R² and MSE, and the pre-registered rule
returns BELOW THE NULL for each. It is below TripNet and PointNet2D+BiLSTM, which CLEAR
it. So: *the reference rows this benchmark established — rows still reproduced verbatim as
baselines in 2025–26 papers, including TripNet's own Table 5 — are beaten by a linear
regression on published metadata; the current state of the art is not.*

**The reporting-practice finding.** The DrivAerNet++ paper **already ran this null**.
Section 5.1.2 fits AutoGluon, XGBoost, LightGBM, Random Forest and Gradient Boosting on 26
parameters. But the result appears **only in Figure 5**, on a **random 80/20 split**
(`AutoML_parametric.py` imports `sklearn.model_selection.train_test_split`, not the
design-id files), and is **never placed beside Table 4**. The authors ran the parametric
null and their own reporting structure made it impossible for a reader to compare it with
the deep models. No number is quoted from that figure here — the structure is the point.

### 2.3 AhmedML — a bar nobody has cleared

500 variants, 8 published shape parameters, published forces under both normalisations, no
recommended split, and **no published ML drag baseline found**. Searched: the dataset paper
(arXiv 2407.20801, which reports no ML results and whose datasheet says only that "limited
testing with various ML approaches has been undertaken by the author team"), NeuralCFD /
GP-UPT (arXiv 2502.09692 — uses AhmedML for *pressure* prediction; its R²≈0.97 drag result
is on DrivAerML), PhysicsNeMo-CFD (DrivAerML only), and FIGConvNet (its "Ahmed body" is the
Li et al. 2023 dataset, not AhmedML).

10-fold OOS null on drag: **R² 0.684 [0.638, 0.727]** at constant reference area,
**0.340 [0.277, 0.386]** at per-geometry reference area (linear; squares add ≈0.03–0.05).
Lift: 0.452 → 0.630 with squares. The normalisation matters enormously here, because
Ahmed's frontal area is essentially `body-height × body-width`, two of the eight published
parameters — so the constant-area figure is the one to quote as a bar.

**Verdict: not adjudicable, and that is the finding.** A benchmark cannot be said to have
been beaten by a model when no model's number has been published on it. Any future AhmedML
drag surrogate should be required to clear R² 0.684 from eight numbers.

### 2.4 WindsorML — the null does not travel, and I am reporting that plainly

355 variants, 7 published shape parameters (including `frontal_area`), published forces.
The WindsorML paper publishes its own MeshGraphNet evaluation (SI D.2, verbatim): *"using a
60/20/20 split of train, validation and test data, it is possible to obtain a MSE of less
than 0.00028 for the drag coefficient."*

10-fold OOS null on drag: **R² 0.104 [−0.143, 0.267]** at constant reference area and
**0.073 [−0.181, 0.239]** at per-geometry area. Adding squared terms makes it *worse*
(−0.263), i.e. the 7 parameters genuinely do not determine the drag of this body. The null's
interval includes zero. The published bound, converted to R² currency by dividing by the
target's own variance, implies the MGN attains **R² ≥ 0.79** (constant area) or **≥ 0.77**
(per-geometry area).

**Verdict: the null is decisively beaten. WindsorML is the counterexample.** Because the
published figure is a *bound*, the pre-registered asymmetry applies: a bound is adjudicable
only in the direction where the null loses — which is the direction it lands. The verdict
is therefore firm, not "suggestive".

Both normalisations were run precisely because the WindsorML SI names only "the drag
coefficient" without saying which file; the counterexample holds either way.

### 2.5 DeepCFD — not runnable

On disk at `data/deepcfd`. `dataX.pkl` is (981, 3, 172, 79) SDF/flow-region channel images
and `dataY.pkl` the corresponding (u, v, p) fields. There is **no per-case parameter table**
and, decisively, **no force labels of any kind** — the DeepCFD paper reports field MSE only.
No label was manufactured. Reported as not runnable, which is a fact about the benchmark's
reporting, not a failure of this experiment.

---

## 3. How I tried to break this

Every check below is in the committed JSON under `break_tests` or the per-benchmark
diagnostics.

1. **Label permutation.** Refit the DrivAerNet++ full-test null on shuffled training
   labels: R² = **−0.0198**. The machinery is not manufacturing signal.
2. **Metric switch.** R² divides by the test set's own variance, so it cannot settle a
   cross-subset question. Repeated everything in **MSE**, which is scale-absolute and also
   published. On the full official test set the null's MSE is 1.093e-4 against 1.42–1.71e-4
   for the three dataset-paper models — the R² verdict and the MSE verdict agree.
3. **Is the parametric subset an easier target?** No — it is *harder* in both currencies.
   Its C_d spread is **larger** than the full test set's (sd 0.02308 vs 0.02038; var 5.33e-4
   vs 4.15e-4), so its lower MSE (7.90e-5) is achieved against more variance, not less. This
   caveat is in any case moot for the headline, which is scored on the full 1,154.
4. **Near-duplicate leakage.** Standardised-parameter nearest-neighbour distance,
   test→train against train→train: DrivAerNet++ ratio **0.944**, DrivAerML ratio **0.955**.
   Test designs are no closer to the training set than training designs are to each other,
   so neither split leaks by near-duplication — which would have inflated the null *and*
   every published model.
5. **Split independence.** The DrivAerNet++ paper describes its split as 5600/1200/1200
   while the published id lists are 5819/1148/1154, so the number could in principle be a
   property of those particular lists. 10-fold over the pooled 6,973 train+test designs,
   using no official split at all: R² **0.7492** against 0.7365 official. It is not.
6. **Protocol swap.** On the parametric pool: 10-fold OOS R² 0.8288, random 80/20 (the
   protocol the benchmark's own `AutoML_parametric.py` uses) R² 0.8308, official split
   0.8515. Three protocols, same answer.
7. **Regularisation.** Ridge path λ ∈ {1e-8 … 1}: R² flat at 0.7365 ± 0.0003 (collapsing
   only at λ=100, as it must). The OLS fit is not rescued by lucky conditioning.
8. **Ablate each feature block.** Category tokens alone: 0.448. Parameters alone, without
   the category dummies: 0.378. Together: 0.7365. Neither block carries the result — which
   also rules out the reading that the null is "just" a lookup table of design family.
9. **Per-family decomposition.** MSE on the full test set by family: E_S 8.58e-5, F_S
   7.13e-5, N_S 7.88e-5 (all with published parameters), F_D **1.379e-4** (595 designs, no
   published parameters, predicted by group mean alone). The aggregate is exactly what the
   parts say it should be, and the weak half is weak for a stated, structural reason.
10. **Normalisation bracketing.** Windsor and Ahmed run under both published reference-area
    conventions; DrivAerML under both, plus a correlation check pinning the published
    target to the constant-area convention at corr = 1.000000.
11. **Searched for stronger published opponents rather than the weakest ones.** The
    DrivAerNet++ comparison was extended to TripNet and PointNet2D+BiLSTM specifically
    because comparing only against 2024 dataset-paper baselines would have been the easy,
    attackable move. Both clear the null, and that is reported as the headline qualification.

### What I could not break, and what remains open

* **Label version.** `DrivAerNetPlusPlus_Cd_8k_Updated.csv` may post-date the labels Table 4
  was scored against. There is no earlier version to compare, so this is unresolvable; the
  direction of any correction is unknown. Stated, not spun.
* **Conservatism, not a hole.** The null is fit on the 5,819 train ids only, while the deep
  models additionally had a 1,148-design validation set for early stopping. Fitting on
  train+val could only help the null.
* **v1-family parameters.** Half of DrivAerNet++ cannot be given a parameter row because
  those 50 parameters are unpublished. The full-test null is a lower bound on what published
  metadata supports; the true bound is higher.

---

## 4. Verdict

**RESOLVED: the finding generalises, and it does not generalise uniformly — which is the
more useful result.**

* AirfRANS is **not** unusually parametric. The same phenomenon appears at full strength on
  DrivAerML (null indistinguishable from DoMINO and FIGConvNet on their own split) and at
  reduced strength on DrivAerNet++ (null above all three of the benchmark's own reference
  baselines, below current SOTA).
* It is bounded, not universal. WindsorML is a clear counterexample: seven parameters
  explain 10% of drag variance and the published model beats the null outright. AhmedML sits
  in between with no baseline published at all.
* The scoring-practice problem is therefore real and general: **the null's strength varies
  by an order of magnitude across benchmarks in the same subfield, and not one of them
  reports it.** Where a benchmark did run it (DrivAerNet++ §5.1.2) it was reported in a
  different figure, on a different split, never beside the deep-model table.

**The claim the paper can make.** Not "deep surrogates do not beat trivial baselines" —
on WindsorML and on DrivAerNet++ against SOTA, they plainly do. The claim is that *whether
they do is unknown at the point of publication, because the parameter-only null is not part
of any of these benchmarks' reporting protocol, and when it is computed it lands anywhere
between R² 0.10 and R² 0.97.*

### Suggested paper sentence

> The same control generalises beyond AirfRANS. On DrivAerML, under the published 436/48
> benchmark split and the published metric, ordinary least squares on the 16 design
> parameters reaches R² 0.973 [0.958, 0.983] for drag — statistically indistinguishable
> from DoMINO (0.98) and FIGConvNet (0.97), and above X-MeshGraphNet (0.92). On
> DrivAerNet++, a regression on published metadata alone reaches R² 0.737 [0.705, 0.764]
> on the full official test split, above all three baselines the dataset paper reports
> (0.596–0.643) though below current state of the art (TripNet, 0.957). The control is not
> universal — on WindsorML the same null reaches only R² 0.10 and the published
> MeshGraphNet clears it comfortably — and that variability is the point: the null spans an
> order of magnitude across benchmarks in this subfield, and none of them reports it.

### Rebuttal line

> Reviewer: *is this an AirfRANS quirk?* → We ran the identical control on four further
> benchmarks, using each one's own split, own metric and own published parameters, with the
> decision rule committed to version control before the runs → It reproduces on DrivAerML
> (R² 0.973 vs DoMINO 0.98 on their split) and on DrivAerNet++ against the dataset paper's
> baselines, fails on WindsorML, and cannot be adjudicated on AhmedML because no baseline
> has ever been published there. Script, pre-registration commit and results JSON are in the
> artifact.

---

## 5. Cost

| | |
|---|---|
| Metadata downloaded | **2.4 MB** — parameters, force labels and split id lists only |
| Field data downloaded | **0 bytes** (the four benchmarks together distribute ≈ 40 TB) |
| Source-verification documents | 34.7 MB (NeurIPS proceedings PDF + 6 paper HTMLs; not inputs to any number) |
| Compute | CPU only, no GPU touched; script wall clock **152 s** at 10,000 bootstrap resamples |
| Session wall clock | ≈ 100 min including source verification of every published number |

Every published number in this report was read in its own source document — proceedings
PDF or the paper's own full text — never from a secondary table. That mattered at least
once: TripNet's two drag tables report different benchmarks, and taking the more prominent
one would have put a v1-fastback number into a DrivAerNet++ comparison.
