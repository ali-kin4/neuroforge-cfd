# Field scan — what is working in ML-for-PDEs, and where a positive result is cheap

Date: **2026-09-11**. Companion to `whitespace.md` (2026-09-07),
`drag_observability.md`, `mechanism_decision.md`, `FINDINGS.md`.

Every citation below is marked **[V]** verified against a primary source in this
session, **[R]** reused from a verification closed in `FINDINGS.md` §"Citation
verification" or `whitespace.md` (do not re-check), or **[U]** UNVERIFIED.
Nothing is cited without one of those three marks.

---

## 0. Two corrections to the premises this scan was handed

### 0a. The RealPDE dates in the brief are in the FUTURE, not the past

The brief states RealPDE results "were due 10 November 2026 and presentation 6
December 2026, which may already have landed." Today is **11 September 2026**.
Both dates are ahead of us. Fetched from `realpdecompetition.github.io` this
session **[V]**:

| milestone | date | status on 2026-09-11 |
|---|---|---|
| launch, Codabench opens | 2026-07-05 | past |
| registration closes | 2026-08-21 | **past — entry is no longer possible** |
| main development phase ends | 2026-09-27 | 16 days away |
| decision phase ends | 2026-10-25 | future |
| **results announced** | **2026-11-10** | future |
| code/fact-sheet deadline | 2026-11-25 | future |
| **NeurIPS workshop presentation** | **2026-12-06** | future |

Metrics confirmed: relative L2, TKE, mean velocity profile error, a time score,
and a **Safe Prediction Score** that "jointly evaluates physical accuracy,
uncertainty coverage, and interval tightness, penalizing intervals that miss the
ground truth." Two tracks: Sim2Real on NACA4418 PIV, and long-term test-time
adaptation. 100 paired trajectories, 64x128, Re 2968–27975, alpha 0–20 deg. Lead
organiser Tailin Wu (Westlake / Uniforce AI).

**Consequence, and it is the opposite of what the brief implies.** The
trust-evaluation window is open for roughly **eight more weeks**, not closed. Any
claim of the form "no public benchmark scores whether a CFD surrogate knows it is
wrong" is true today and false on 2026-12-06. Anything shipped before then cites
RealPDE as convergent evidence of demand. We cannot enter (registration closed
2026-08-21), so the competition is purely a clock, not an opportunity.

### 0b. What in `whitespace.md` (2026-09-07) is now stale

Five items. All superseded by work committed on the same day or since.

| `whitespace.md` item | status on 2026-09-11 |
|---|---|
| **G1 headline**: "no near-wall scheme recovers the official drag ranking beyond rho_D ~ 0.84" | **WITHDRAWN.** It reports a number *below* the 0.874 do-nothing covariate baseline and presents it as a ceiling. `drag_observability.md` §5. |
| **§0a mechanism**: floor is "operator provenance, not missing physics" | **REFUTED as sole cause.** At fixed `h`, thinning the source cloud x8 raises the floor in 16/16 cases, `q = +0.55 ± 0.27`. Provenance predicts `q = 0`. `mechanism_decision.md` §1.4. **G2 ("what operator should you audit with") rested on provenance being the mechanism; its premise is gone. Re-rank G2 down.** |
| **A0 and A1** listed as pending/blocking | **DONE**, both, in one 46-minute CPU run. 200 cases x 3 levels x 4 arms, four gates at exactly 0.0. |
| **Conformal numbers** 5.6–6.8x width, 0.891 coverage | **MOVED** to 6.5–7.9x and 0.901–0.903, after fixing the split-conformal order statistic. `mechanism_decision.md` §3. |
| **G1 fact 3** (lift/drag observability asymmetry) | **STRENGTHENED and de-confounded.** Both arms now come from one fixed integrator on one raster, with a y+ ~ 5 mechanism and three independent resolution estimates agreeing within 2x. This is the surviving asset. |

Also stale in emphasis: `whitespace.md` treats DD-RNO as "measuring the symptom,
no covariate control." **They do run a partial correlation controlling for alpha
and log10(Re)** — see §1.6 below. The gap is narrower and more precisely stated
than that report has it. Do not overclaim it.

---

## 1. The state of the field, by line

Window: roughly Oct 2025 – Sep 2026, with anchors further back where they define
the line.

### 1.1 Geometry-native operators and transformers for CFD — **INCREMENTAL, industrialising, do not compete**

| work | id / venue | mark | what it got |
|---|---|---|---|
| Transolver | arXiv:2402.02366, **ICML 2024** | [V] | AirfRANS SOTA at the time; rho_L = 0.9978 |
| Transolver++ | arXiv:2502.02414, **ICML 2025**, PMLR 267:41432–41449, Luo et al. | [V] | million-point single-GPU; +13% over six benchmarks, +20% on industrial scale |
| Transolver-3 | arXiv:2602.04940 | [U] | "industrial-scale geometries" — details unverified |
| AB-UPT | arXiv:2502.09692 v4, **TMLR** | [R] | automotive SOTA, divergence-free hard constraint, **no UQ** |
| DoMINO | NVIDIA PhysicsNeMo, shipped NIM container | [R] | production deployment, no uncertainty story |
| DD-RNO | arXiv:2608.13490 (2026-08-13), Mehta, Bhati, Akolekar, IIT Jodhpur | [V] | AirfRANS: velocity MSE 17x/12x better; drag rank rho 0.250 -> 0.997; ~144 ms/sample |
| CarBench | arXiv:2512.07847 (2025-11-25, rev 2026-08-20), Elrefaie, Shu, Klenk, Ahmed | [V] | 11 architectures on DrivAerNet++ 8k sims; FNO, PointNet, RegDGCNN, PointMAE, PointTransformer, Transolver, Transolver++, AB-UPT, TripNet |
| PhysicsBench | arXiv:2608.24056 (2026-08-25), leaderboard.narnia.ai | [V] | 66 models, 7 tasks, 9 datasets incl. AirfRANS, DrivAerNet/++, DrivAerML, PDEBench |

**Verdict: incremental.** The headline numbers are relative field-MSE gains on
fixed benchmarks. The line has moved into benchmark-and-leaderboard consolidation
(CarBench, PhysicsBench, both 2026), which is what a field does when architecture
gains saturate. On one 4070 Ti in one month, **beating any of these on AirfRANS
or DrivAerNet++ is not a realistic positive.** AB-UPT is TMLR-published at
industrial scale; Transolver++ was trained on multi-GPU million-point meshes.

### 1.2 Force-coefficient metrics on these benchmarks — **CONTESTED, and this is the live hole**

This is the one place where the survey turned up something the field has not
noticed. Verified directly from the NeurIPS 2022 PDF, Tables 3 and 5 **[V]**:

**AirfRANS (Bonnet et al., NeurIPS 2022 D&B), full task, 5 seeds:**

| model | rel. err. C_D | rel. err. C_L | **rho_D** | rho_L |
|---|---:|---:|---:|---:|
| MLP | 4.289 ± 0.679 | 0.767 ± 0.108 | **-0.117 ± 0.256** | 0.913 ± 0.018 |
| GraphSAGE | 4.050 ± 0.704 | 0.517 ± 0.162 | **-0.303 ± 0.124** | 0.965 ± 0.011 |
| PointNet | 14.637 ± 3.668 | 0.742 ± 0.186 | **-0.022 ± 0.097** | 0.938 ± 0.023 |
| Graph U-Net | 10.385 ± 1.895 | 0.489 ± 0.105 | **-0.138 ± 0.258** | 0.967 ± 0.019 |

**GraphSAGE across all four tasks (Table 5):** rho_D = -0.303 (full), -0.139
(scarce), **+0.013 (reynolds)**, **+0.055 (aoa)**; rho_L = 0.965 / 0.981 / 0.927
/ 0.908.

And the benchmark's own declared metric hierarchy, quoted verbatim from the
paper's conclusion **[V]**:

> "The Spearman's correlation for the force coefficients is the main metrics to
> maximize the recovery of the best airfoils in terms of lift-over-drag ratio as
> it quantifies the ability of models to preserve the force coefficient ranks."

Three facts follow, each independently verified, and together they are the
strongest opportunity in this scan.

**(a) Every original AirfRANS baseline scores at or below zero on the benchmark's
own primary drag metric.** Range -0.303 to +0.055 across four architectures and
four tasks.

**(b) A three-parameter regression on the case name scores 0.874 on drag — and
0.934 on lift.** From `results/control/drag_covariate_control.json`, read
directly this session, touching no flow field:

| null, fitted on case name only | vs official C_D | vs official C_L |
|---|---:|---:|
| `alpha` alone | 0.800 | **0.9335** |
| `|alpha|` alone | 0.818 | — |
| `U` alone | -0.195 | — |
| OLS `(U, alpha)` | 0.830 | **0.9343** |
| OLS `(U, alpha, alpha^2)` | **0.8742** | **0.9343** |

AirfRANS encodes `U` and `alpha` in the simulation filename and
`Simulation.reset()` parses them. The NACA digits are in the name too and are
deliberately *not* controlled for, so **both numbers are lower bounds** on what
case identity buys. Note the two metrics want different nulls: lift is linear in
alpha, so `(U, alpha)` is already maximal for C_L and the quadratic term adds
exactly nothing; drag is quadratic, so `alpha^2` adds +0.044. **Report each
metric against its own strongest null.**

**(b') The lift null is where the audit actually has yield, and this is the
correction that matters most in this scan.** Drag is the metric almost nobody
reports (§1.2e). Lift is the metric *everybody* reports. Joining the null above
to the Table 3 column verified from the NeurIPS PDF:

| model | published rho_L | vs the lift null, 0.934 |
|---|---:|---|
| **MLP** (AirfRANS, NeurIPS 2022 D&B) | 0.913 ± 0.018 | **below** |
| PointNet | 0.938 ± 0.023 | straddles |
| GraphSAGE | 0.965 ± 0.011 | clears |
| Graph U-Net | 0.967 ± 0.019 | clears |
| Transolver (ICML 2024) | 0.9978 | clears |

A published NeurIPS D&B baseline sitting **below a three-parameter regression on
the filename**, on the metric the benchmark declares primary and every subsequent
paper reports, is the finding. Two discipline constraints before this is written
anywhere:

- MLP's shortfall is 0.021 against a 0.018 seed standard deviation, i.e. **~1.2
  sigma**. The honest sentence is *"two of four do not clear the null at their
  point estimates, one of those within seed noise"* — not "the baselines fail."
- **The null itself currently has no confidence interval.** A bootstrap CI on
  rho_null is a prerequisite, not a nicety. Without it the comparison is a point
  estimate against a point estimate.

One further observation, to be checked not asserted: GraphSAGE's rho_L drops to
**0.927 (reynolds)** and **0.908 (aoa)** on the extrapolation splits (Table 5,
verified), both below the in-distribution lift null of 0.934. The null must be
refit within each split before this means anything, because the covariate
regression is extrapolating there too.

**(c) The strongest published number in the line is produced by a head that is
explicitly conditioned on exactly those covariates, and no covariate baseline is
reported.** Verified from the DD-RNO PDF **[V]**: the flow-condition vector is

```
c = [ sin(alpha), cos(alpha), sin(2*alpha), log10(Re/1e6) ]  in R^4      (eq. 11)
w = M_psi(c)  in R^64                                                     (eq. 15)
W_canon = f_LCQ(w)  in R^{2 x 1024}                                       (eq. 30)
[C_L_hat ; C_D_hat] = sum_j W_canon,j(w) . C_p^(j)                        (eq. 31)
```

and the paper's own justification for the feature set:

> "Aerodynamic forces and integrated coefficients are strongly tied to quadratic
> trigonometric relationships such as lift scaling with sin 2alpha across broad
> operating ranges, or induced drag scaling with sin^2 alpha = (1 - cos 2alpha)/2."

That is a hand-engineered superset of `(U, alpha, alpha^2)`, feeding a learned
2x1024 weight vector that is dotted with surface pressure to emit C_D directly.
The reported rho = 0.997 is marginal.

**Be precise here, and do not repeat the argument this repo already discarded.**
Conditioning a surrogate on inlet state is normal and necessary — it cannot
predict a field without knowing the flow condition, and
`drag_observability.md` §6 records testing and rejecting the stronger "this must
be label regression" reading. The defensible claim is narrower and survives:

> A drag *rank* metric cannot distinguish a better integrator from a head that
> has learned the covariate dependence, because a three-parameter regression on
> `(U, alpha, alpha^2)` alone reaches 0.874. No paper in this line reports that
> baseline, so 0.250 and 0.997 are both uninterpretable as evidence about
> integration quality.

**(d) DD-RNO is one step from this, and knows it.** They run a partial
correlation — but only on the *implicit viscous correction* sub-term, not on the
headline. Verbatim **[V]**:

> "a partial correlation analysis is run to control for the confounding effects
> of angle of attack (alpha) and Reynolds number (log10(Re)). This revealed a
> statistically significant positive relationship, yielding a partial Spearman
> correlation of rho = 0.3365 (p_val < 0.001, partial R^2 ~ 0.081). ... a lot of
> the uncorrected correlation is driven by flow-condition variables (alpha and
> Re) rather than a complete, standalone recovery of viscous physics."

They apply the right instrument to a sub-quantity and never to the number in the
abstract. **This is the single best citation available for "the field knows the
confound exists and still does not report the baseline."** It is also the highest
scoop risk in this scan.

**(e) The most-cited model on this benchmark does not report drag at all.**
Verified from the Transolver paper **[V]**: Table 3 columns for AirfRANS are
`Volume, Surf, C_L, rho_L`; Table 6 (OOD) is `C_L, rho_L`. No drag column
anywhere. Appendix B.1, verbatim:

> "air velocity is hard to estimate for airplanes, making all the deep models
> fail in drag coefficient estimation. Thus, in the main text, we focus on the
> lift coefficient estimation and the pressure quantity on the volume and
> surface."

An ICML paper silently dropping the benchmark's declared primary metric because
all models fail it is a textbook instance of the outcome-reporting bias McGreivy
and Hakim quantified — **Nature Machine Intelligence 6:1256–1269 (2024),
arXiv:2407.07218, 79% (60/76) of surveyed papers compare to a weak baseline
[V]** — applied to the specific benchmark this project owns a pipeline for.

**(f) Every relevant dataset publishes the generating design parameters**, so the
null is computable everywhere, in hours per dataset:

| dataset | parameters published | mark |
|---|---|---|
| AirfRANS | `U`, `alpha`, NACA digits, in the filename; parsed by `Simulation.reset()` | [V] |
| DrivAerNet++ | **26** design parameters; NeurIPS 2024 D&B, arXiv:2406.09624 | [V] |
| DrivAerML | **16** geometric parameters, MELS DoE; arXiv:2408.11969 | [V] |
| AhmedML | **6** parameters, Latin hypercube; arXiv:2407.20801 | [V] |

**Partial occupancy, and it must be scoped honestly.** DrivAerNet++ *does* report
an AutoML tabular baseline on its 26 parameters (AutoGluon / XGBoost / LightGBM /
Random Forest), reaching R^2 ~ 0.9 on a single car category and ~0.55 on the
combined set against RegDGCNN's 0.641 **[V]**. So "nobody publishes
parameter-space baselines" is **false as stated**. Three things survive that:

1. It is an **R^2-on-magnitude** comparison. AirfRANS's declared primary metric is
   **Spearman rank**, which is far more exposed to covariate dominance, because
   any monotone function of alpha captures most of it.
2. **The three newest benchmarks in the line all drop it.** Each checked
   individually this session **[V]**:
   - **CarBench**: eleven architectures on DrivAerNet++, no parametric baseline.
     Elrefaie is an author on both DrivAerNet++ and CarBench — *the same group
     omitting its own baseline in the follow-on benchmark* is the strongest
     one-sentence case that a standing protocol is needed rather than a one-off
     table.
   - **PhysicsBench**: does carry tabular models (Ridge, GP, Random Forest,
     LightGBM, XGBoost, TabPFN, FT-Transformer, NODE, TabNet) behind a
     scikit-learn adapter — **but confines them to the 1-D scalar track
     (Concrete Strength, UCI Airfoil Self-Noise)**. AirfRANS is
     `airfrans_2d_2d_flow`, evaluated only with field-prediction architectures
     (U-Net, FPN, dense-prediction transformers), and **no lift or drag
     coefficient metric appears anywhere**. The machinery to run the null exists
     inside the benchmark and is pointed at the wrong tasks.
   - **NeurIPS 2024 ML4CFD retrospective**: one baseline only, "Baseline solution
     (Fully Connected NN)", global score 11.09%, below both the winners and
     OpenFOAM. No parametric baseline, and no systematic discussion of metric
     robustness, confounds or leakage — despite a stated scope that includes
     "assess the robustness of our evaluation framework."
3. **Nobody has done the cross-paper audit**: taking published force-rank numbers
   and asking which clear their own dataset's covariate null.

### 1.3 Hybrid solver-surrogate coupling and learned correction — **STRONG POSITIVE, and it is where the wins are**

| work | id / venue | mark | result |
|---|---|---|---|
| Lei, Tang, Zhang, Chen | arXiv:2608.04400 | [R] | steady CFD, surrogate as Newton–Krylov initial guess: median residual L2 ratio down **>7 orders**, field *and* aerodynamic errors down too |
| Huang & Perdikaris, PhysicsCorrect | arXiv:2507.02227, **AAAI 2026 Oral** | [R] | transient, prediction-time correction |
| ARC-STAR | arXiv:2605.22222 (2026-05-21), Chengze Li et al. | [V] | frozen-host post-hoc correction of PDE foundation models: global corrector + blockwise local refiner + **label-free risk scoring that routes budget to high-risk blocks**; velocity rollout error cut **>=36x over raw Poseidon on every cell** across five benchmarks |
| ANCHOR | arXiv:2512.19643 | [R] | residual-to-error as a deployed data-free signal plus solver fallback |

**Verdict: strong positive, and the closest competitor to NeuroForge's original
thesis.** ARC-STAR in particular is "trust-gated selective correction with a
label-free score," which is NeuroForge's architecture minus the conformal
certificate, landing in May 2026 on foundation models. Note it carries **no
formal guarantee and no conformal layer** in the abstract, which is the only
remaining differentiator — and that differentiator is exactly what this project's
own `functional_audit_gate` and `floor_subtraction_gate` results have narrowed to
a width concession.

The line succeeds where the monitored operator is **solver-consistent**. That
boundary is the one real intellectual asset left in the residual work, and it is
already written into `FINDINGS.md` F4.

### 1.4 Foundation models for PDEs — **STRONG POSITIVE but out of reach on one GPU**

Poseidon (arXiv:2405.19101, **NeurIPS 2024**, 15 downstream tasks, sample
efficiency and accuracy gains) **[V via DL/OpenReview listing]**; DPOT; MORPH
(arXiv:2509.21670) **[U]**; Origo (**ICML 2026 poster**, neural operator
splitting) **[V, listing only]**; jNO (arXiv:2605.10159) **[U]**; Walrus (1.3B),
PhysicsX (4.5B) **[U]**. Downstream-transfer papers are appearing (OOD transfer
to material dynamics arXiv:2603.04354 **[U]**; Martian atmosphere emulation
arXiv:2602.15004 **[U]**).

**Verdict: converting into accepted papers, and closed to this project.** Model
sizes are 21M–4.5B with pretraining corpora that do not fit on a 4070 Ti. The
only affordable entry point is *evaluating* or *correcting* someone else's
foundation model, which is ARC-STAR's slot and is now occupied.

### 1.5 Data efficiency, few-shot, transfer — **POSITIVE but crowded; cheap to run, hard to be novel**

Active in 2026 with consistent positive headlines: leave-one-family-out transfer
for automotive aero with **20 fine-tuning samples** (arXiv:2605.27968) **[U]**;
generative multi-fidelity probabilistic surrogates with transfer
(arXiv:2602.00072) **[U]**; active transfer learning for UAV aero (MDPI Drones
10(4):290) **[U]**; multi-fidelity wind-farm transfer (OpenReview) **[U]**.

**Verdict: positive results are the norm, which is exactly the problem.** The
result "transfer learning improves data efficiency" is now the expected outcome,
so the marginal paper is incremental by construction. P(experiment works) is high;
P(the result is novel) is low. Treat "data efficiency" as a *section*, never as a
headline.

### 1.6 UQ, conformal, reliability — **SATURATED, roughly one paper a month**

2026 alone, all on neural operators or CFD surrogates: arXiv:2608.28515
(pointwise conformal bands, Darcy + Navier–Stokes) **[U, abstract surfaced]**;
arXiv:2606.09923 (split conformal, heat conduction, 89.1% at alpha=0.1) **[U]**;
arXiv:2509.04623 (split conformal in function space) **[U]**; arXiv:2607.17297
(multi-granularity conformal on DrivAerML, C_D + surface fields) **[R]**;
arXiv:2603.11052 (structure-aware epistemic UQ, geometry-shifted 3-D car CFD)
**[R]**; UQNO, TMLR 2024 **[R]**; conformal for mesh-based simulations, Phil.
Trans. R. Soc. A 384(2327):20250076 **[U]**.

**Verdict: saturated.** Do not headline anything whose novelty is "we apply
conformal prediction to X." This is the shape that drew "not novel" twice already.

### 1.7 Critique, evaluation and reporting standards — **LIVE GENRE, and the best genre fit for this project**

McGreivy & Hakim, Nature MI 6:1256–1269 (2024) **[V]**; Duraisamy,
arXiv:2604.20061 **[R]**; Jakeman, Barba, Martins, O'Leary-Roseberry,
arXiv:2502.15496 (V&V for trustworthy SciML, position paper, no experiments)
**[R]**; Koehler & Thuerey, arXiv:2510.23111 **[R]**; adversarial vulnerability of
CFD surrogates, Phys. Rev. Research (doi 10.1103/n5kp-jjcp) **[U — ScienceDirect-
style 403 on fetch; verify manually]**.

**Verdict: this genre is converting, and it is the only genre where this
project's asset base is an advantage rather than a liability.** McGreivy & Hakim
is highly cited and was very hard to desk-reject because its contribution was a
measurement nobody had made. The direction in §3.1 is that same shape, one level
more specific, and with a *constructive* deliverable attached.

### 1.8 Benchmarks and competitions — **ACTIVE, and the window closes in December**

CFDONEval, IJCAI 2025 pp. 5752–5760 (12 models, 7 problems, 22 datasets;
accuracy, efficiency, KE spectra, visualisation — **no UQ**) **[R]**; AIAA AASM
benchmark cases, AIAA 2025-0036 **[R]**; CarBench **[V]**; PhysicsBench **[V]**;
NVIDIA PhysicsNeMo-CFD benchmarking framework, arXiv:2507.10747 (Tangsali, Ranade,
Nabian et al.; DoMINO, X-MeshGraphNet, FIGConvNet on DrivAerML; **no parametric
baseline**) **[V]**; AirfoilAD, Adv. Eng. Software, S0955598626001548, March 2026
(experimental + CFD, RF and XGBoost baselines) **[V via abstract]**; NeurIPS 2024
ML4CFD competition retrospective, arXiv:2506.08516, Yagoubi + 17 (240+ teams,
2-D airfoils, multi-criteria: accuracy, physical fidelity, efficiency, OOD)
**[V]**; RealPDE / NeurIPS 2026 **[V, §0a]**.

**Verdict: benchmark contributions are converting at D&B tracks and in journals,
and none of them carries a covariate null.** AirfoilAD is a second site for the
protocol rather than a scoop: its baselines are Random Forest and XGBoost on
tabular conditions, which is a *parametric model as the method*, not a parametric
model as the *null for a field-based method*.

---

## 2. Venue map inside the no-APC constraint

The hard constraint is no article processing charge. Two sub-cases matter and the
project's memory conflates them: **"no fee"** and **"subscription only."** TMLR
and NeurIPS charge nothing and are open access; if the constraint is literally
"take the subscription licence," they are excluded, and if it is "do not pay,"
they are the best options in the list. **Flagging this for the author to
resolve — it changes the top of the table.**

| venue | fee route | what it rewards | fit for §3 directions |
|---|---|---|---|
| **Computers and Fluids** (Elsevier, hybrid, IF ~3.0, single-anonymized) | free by subscription route | **rigor, not novelty**; scope names "uncertainty quantification in fluid flow simulations, reduced-order and surrogate models"; ML papers welcome "provided they show excellent scientific character"; explicitly asks authors to discuss limitations | D1, D2, D3. Current incumbent target. Single-anonymized dissolves the preprint/anonymity problem. **[R from `FINDINGS.md` F8; guide rows came from search extraction, ScienceDirect 403s — eyeball the guide manually before filing.]** |
| **TMLR** | **zero fee**, open access | explicitly **"are the claims supported by accurate, convincing and clear evidence?"** and explicitly **not** novelty or impact **[V]** | D1 and D4 are near-perfect fits. This is the structural match for a project whose two rejections were both on novelty-at-desk. AB-UPT went here. |
| **NeurIPS Datasets & Benchmarks** | zero fee | datasets, benchmarks, evaluation protocols, reproducibility | D1's harness. AirfRANS itself went here; DrivAerNet++ went here. Cycle timing needs checking. |
| **Nature Machine Intelligence** | hybrid, subscription route exists | broad-interest measurement results; took McGreivy & Hakim | D1 only if the cross-paper audit is large (>= 40 papers) and multi-dataset. High risk, high return. |
| **AIAA SciTech 2027** (Orlando, 11–15 Jan 2027) | conference registration, no APC | applied aerodynamics; there is a **"Special Session: Applied Surrogate Modeling"** sub-topic tied to the AASM benchmark group **[V]** | D2 and D3 are a natural fit. **Abstract deadline NOT confirmed — the call is live but the date did not surface. SciTech 2026 used 22 May 2025. Check `scitech.aiaa.org/call-for-content/call-for-papers/` immediately; if the deadline has passed, this route is closed until 2028.** |
| EAAI (Elsevier, IF 9.0) | free by subscription route | "novel aspects of AI used for a real-world engineering application" — **"novel" is the word that killed this twice** | coin flip; keep as fallback |
| RESS (IF 13.7) | subscription route | reliability engineering, UQ | 13,000-word cap against a 14,700-word manuscript |
| Physics of Fluids | — | **OUT.** Two 2026 co-editor editorials establish a "Physics First" policy requiring the work to "advance physical understanding" **[R]** | — |
| negative-results venues | — | **OUT.** Structurally gold OA. **[R]** | — |

**What each rewards, in one line each.** C&F rewards a rigorous measurement with
stated limitations. TMLR rewards a claim matched exactly to its evidence. NeurIPS
D&B rewards a reusable artifact. Nature MI rewards a measurement that changes how
a field reports. AIAA rewards an engineering quantity an aerodynamicist cares
about. None of them rewards a new architecture from a one-GPU lab in 2026.

---

## 3. Ranked research directions

Probabilities are decomposed as
`P(positive) = P(the experiment returns a usable number) x P(nobody has published it)`.
Where the measurement is already in the repository the first factor is ~1 and the
stated probability *is* the occupancy risk. Costs assume one RTX 4070 Ti and the
existing scripts.

---

### D1. The covariate null for force-coefficient metrics — **RANK 1**

**The claim a positive result supports.**
*Force-coefficient rank metrics on aerodynamic ML benchmarks must be reported
against a design-parameter null. We supply that null — **for lift and for drag,
each at its own strongest form** — for four public benchmarks, show which
published numbers clear it, give the covariate-adjusted metric (partial rank
correlation) that discriminates, and re-rank the published results under it.*

**Build this on lift first, drag second.** The earlier draft of this scan built
D1 on drag alone; that was a mistake and §1.2(b') is the correction. Drag is
barely reported (Transolver omits it outright, the AirfRANS baselines are at
-0.30 to +0.06, DD-RNO is the only paper in the line whose drag number clears its
null), so a drag-only audit yields "hardly anyone reports it, and the one who
does passes." Lift is reported by everyone, the null is 0.934, and a NeurIPS D&B
baseline sits below it. **Audit both; lead with lift.**

**Why this is a positive, not a critique.** The deliverable is **the baseline and
the corrected leaderboard**, not the complaint. The null *discriminates*:
GraphSAGE, Graph U-Net and Transolver clear the lift null; MLP does not and
PointNet straddles it. DD-RNO's 0.997 clears the drag null of 0.874; our
Transolver's 0.845 and the original baselines do not. A null that separates
methods is a protocol, not nihilism. Headline: *"here is the floor every AirfRANS
force number must clear, and here is who clears it."*

**The experiment.**
1. Covariate null on AirfRANS, done properly: fit on the **train** split, evaluate
   on **test** (the committed 0.874 / 0.934 are test-split fits and must be redone
   out-of-sample — this is the one real technical risk in D1 and it is small).
   Include the NACA digits, which the current numbers deliberately omit, so the
   null is reported at full strength. Bootstrap CI on the null itself. Both C_L
   and C_D, each at its own strongest covariate form. All four tasks: full,
   scarce, reynolds, aoa, with the null **refit inside each split**.
   `scripts/drag_covariate_control.py` exists. **~1.5 days.**
2. Same for DrivAerNet++ (26 params), DrivAerML (16), AhmedML (6). Join published
   parameters to published force labels; gradient boosting; report rank and R^2.
   **~3 days**, dominated by data plumbing, not compute.
3. Partial rank correlation as the adjusted metric, with a permutation test.
   **~1 day.**
4. Cross-paper audit: tabulate every paper reporting a force metric on these four
   datasets — does it report drag at all, does it report a null, does its number
   clear the null. Target 30–50 papers. **~4 days.**
5. Ship a small pip-installable harness so a reviewer can run the null on their
   own numbers in one command. **~2 days.**

**Cost: 10–12 days**, almost all CPU and writing.

**P(positive) = 0.97 x 0.89 ~ 0.86.**
- *0.97*: the AirfRANS arm is already computed; the only way it fails is if the
  out-of-sample refit collapses 0.874 / 0.934, which would require the covariate
  dependence to be non-transferable across an i.i.d. split. It is not.
- *0.89*: occupancy risk, revised **upward** after checking the three newest
  benchmarks individually (§1.2.2) and after a separate search for *standalone
  audit/reanalysis* papers in aero surrogate modelling, which is the artifact
  class a competitor would actually use (McGreivy & Hakim was a standalone, not a
  benchmark). That search returns evaluation papers — Scherz, Hines, Bekemeyer,
  arXiv:2607.13866, 15 Jul 2026, four architectures on airfoil surface pressure
  and industrial 3-D aircraft **[V]**; ShapeBench, arXiv:2605.20763 **[U]** — but
  no reanalysis of published force numbers against a null. What is unoccupied is
  the rank-metric framing, the cross-paper audit, and the harness.

**A third factor the 0.86 does not cover, and it must be stated.** D1's *lift*
headline — that a published NeurIPS D&B baseline sits below the null — rests on
MLP's 0.913 ± 0.018 against a null of 0.934 **that has no confidence interval and
has not been refit out of sample**. If the bootstrap CI is wide, or the
train-fit/test-eval refit moves 0.934 down by 0.02, that single below-null point
evaporates and the lift audit's headline degrades to *"all published lift numbers
clear the null"* — still a usable protocol paper, but not the finding ranked
here. The honest decomposition is
`P(null refits near 0.934) x P(>= 1 published number below it once CIs are
accounted) x P(unoccupied)`, and **only the third factor has been measured.**
Step 1 of the experiment exists precisely to settle the first two; run it before
committing to the framing.

**Novelty risk, stated honestly.** Medium. A reviewer will say "DrivAerNet++
already did a tabular baseline." The answer must be prepared and is three-part:
their comparison is R^2-on-magnitude not rank; their own follow-on benchmark
(CarBench) drops it; and no paper anywhere audits published numbers against it.
If the author cannot say all three comfortably, D1 is weaker than its rank
suggests.

**What would scoop it.** (i) The DD-RNO group — they already run the right
partial correlation on a sub-term and are one paragraph from the headline. **This
is the highest scoop risk in this scan.** (ii) A McGreivy & Hakim follow-up.
(iii) PhysicsBench v2 adding a naive baseline row. Mitigation: the AirfRANS
covariate control is committed and timestamped (`da23297`); get a preprint up
before extending to the other three datasets if time gets tight.

---

### D2. Move the metric D1 defines: a surface wall-shear head for raster surrogates — **RANK 2**

**The claim.**
*Drag on a rasterised representation is destroyed by the viscous sublayer, not by
model capacity. Predicting wall shear stress directly as a surface output head,
instead of differencing velocity on the raster, recovers the viscous drag the
raster annihilates, at fixed backbone and fixed training budget.*

**BLOCKING GATE — run this before any training. It takes minutes and it decides
6–9 days.** The repository already records, in
`results/control/drag_observability_roundtrip.json` under `label_composition`
(read directly this session):

| official-label quantity | value |
|---|---:|
| median cdv / cd | **0.682** |
| median cdp / cd | 0.318 |
| rho(official cdp, official cd) | **0.983** |
| rho(official cdv, official cd) | **-0.204** |
| rho(official cdp, official cdv) | -0.327 |

Viscous drag carries **68% of the magnitude and essentially none of the rank**,
and a perfect surface-pressure integrator could reach 0.983 on total-drag ranking
with no viscous information at all. **The partial versions controlling for
`(U, alpha)` have never been computed** — the cached labels
(`results/control/_cache/official_labels_full_test_n200.json`) carry only `cl`
and `cd`, so this needs the cdp/cdv arrays from the round-trip run. Compute:

```
partial rho(official cdp, official cd | U, alpha)   -> ceiling for a perfect pressure head
partial rho(official cdv, official cd | U, alpha)   -> ceiling for a perfect shear head
```

Pre-register the branch **before** running, as `9e21823` did:
- if the cdv ceiling is **> 0.4**, D2's rank endpoint is live and P rises;
- if it is **near zero** — which the marginal -0.204 and DD-RNO's own partial
  R^2 ~ 0.081 both predict — then **D2 must be restated as a magnitude result**,
  not a rank result. That version is still positive and still mechanistically
  owned: the round trip currently returns **0.55% of the true cdv at 128^2** and
  the median relative cd error is **133%**, so a shear head that recovers even a
  third of the viscous magnitude is a large, reportable, first-of-its-kind gain
  on a quantity nobody currently recovers at all.

**Write the magnitude version as the primary endpoint and the rank version as the
gated secondary.** That way the direction cannot fail, only change its sentence.

**Why the mechanism is owned and strong.** `drag_observability.md` §2–3, all
gated: the raster round trip through AirfRANS's own integrator returns **0.55% /
1.05% / 1.91%** of the true viscous drag at 128^2 / 256^2 / 512^2, and viscous
drag is **68% of total drag magnitude**. Three independent routes — a y+ = 5
boundary-layer scaling, a magnitude extrapolation (ratio ~ N^0.893), and a rank
extrapolation — put the raster resolution needed for recovery at **N ~ 4e4–1.4e5**,
i.e. 10^5–10^6x the deployed cell count. Measured near-wall cloud spacing 3.31e-5
chord = y+ 5.1, agreeing with the scale argument to **1.03x**. Lift, which lives
on the chord scale, converges at first order over the same ladder (12.3% -> 6.0%
-> 3.2%).

That is a complete, pre-registered diagnosis with an obvious remedy that nobody
has applied in this representation class.

**The experiment.** Add a surface-supervised `tau_w` head to the existing
Transolver and FNO backbones (AirfRANS ships surface fields), train at fixed
budget, integrate forces from the head rather than from raster velocity
gradients, and report marginal **and partial** rho_D against the D1 null on all
four tasks. Ablate: head vs no head, surface-weighted loss vs uniform.

**Cost: 6–9 days** (3 backbone-seed trainings x ~2 h each on the 4070 Ti, plus
analysis).

**P(positive) = 0.78 x 0.65 ~ 0.51** on the magnitude endpoint, and materially
lower on the rank endpoint until the gate returns.
- *0.78*: the magnitude baseline is as weak as a baseline gets (0.55% of true
  cdv recovered), so direct surface supervision on the quantity that carries the
  magnitude should move it substantially. The 22% failure mass is the real
  possibility that AirfRANS's `tau_w` labels are themselves only as good as the
  post-processing that produced them, and that a raster-conditioned surface head
  inherits the same sublayer problem through its *inputs* even though its output
  is surface-native.
- *0.65*: AB-UPT predicts surface wall shear for cars, and DD-RNO predicts forces
  from a surface head for airfoils, so the *mechanism* is not new. What is new is
  the diagnosis-plus-remedy pairing and the covariate-adjusted metric.
- **The rank endpoint is the optimistic one and should not be promised.** The
  marginal rho(cdv, cd) = -0.204 and DD-RNO's partial R^2 ~ 0.081 both say the
  headroom above the covariate null in *rank* may be close to nil. An earlier
  draft of this scan put D2 at 0.56 on a rank endpoint; that was too generous.

**Novelty risk: high as a standalone method paper, low as the second half of
D1.** Do not ship D2 alone. Ship it as "we defined the metric, then we moved it,"
which is the strongest available shape: a critique that pays its own bill.

**What would scoop it.** DD-RNO's follow-up, or any AirfRANS paper adding a
surface shear head. Low urgency relative to D1.

---

### D3. The lift/drag observability ladder as a standalone measurement — **RANK 3**

**The claim.** *On a Cartesian raster of AirfRANS, lift is observable and drag is
not, with a quantified resolution requirement of N ~ 4e4–1.4e5 derived three
independent ways, and this is a property of the representation class (DeepCFD,
U-Net, FNO, any voxel-output surrogate), not of any one integrator.*

**Status: essentially complete.** 200 cases x 3 levels x 4 arms, four gates at
exactly 0.0, decision rule committed in `9e21823` before the run, 46 minutes CPU.

**Cost: 3–4 days**, write-up and figures only.

**P(positive) = 1.0 x 0.90 ~ 0.90** — but read the next paragraph before
believing that number.

**The honest problem: this is a limits result, and the author has barred negative
results.** "Your representation cannot see the thing you are reporting" is
information the field needs and is not a positive headline. It becomes positive
only when attached to D2 ("...and here is the head that fixes it") or to D1
("...and here is why your metric was measuring alpha"). **Recommendation: do not
ship D3 alone. It is section 3 of the D1+D2 paper.**

**What would scoop it.** DD-RNO explicitly names "the numerical instability of
computing wall-normal velocity gradients from continuous-field approximations" as
one of their two motivating bottlenecks **[V]**. They have the symptom. They do
not have the ladder, the ground-truth control, or the y+ scale law.

---

### D4. Trust-ranked evaluation across backbones, as a finding not a benchmark — **RANK 4, time-limited**

**The claim.** *On steady-RANS surrogates, a physics-free ensemble spread ranks
field errors as well as a physics residual (AUROC 0.894 vs 0.871, paired
bootstrap delta = +0.023, 95% CI [-0.035, +0.083]), and physics earns its place
only in fusion (0.905, ~91% oracle recovery vs ~67% and ~72% separately).*

**Owned outright**: `results/selective/selective_prediction.json`,
`results/control/bootstrap_spearman_ci.json`, three backbone families with seeds,
two datasets.

**Cost: 4–6 days** of analysis and writing. No new training.

**P(positive) = 0.95 x 0.65 ~ 0.62.** The 0.65 is the window: **RealPDE's Safe
Prediction Score lands 2026-12-06**, and the UQ line publishes roughly monthly
(§1.6). Any sentence of the form "nobody evaluates whether a CFD surrogate knows
it is wrong" expires in December.

**Why it is only rank 4 despite being cheap.** The headline result is that *our
own physics signal is not better than a physics-free baseline*. That is a
credibility asset and a genuine correction to the field's implicit assumption,
but it reads as a negative, and it is a finding about a trust layer rather than
about CFD. Report it inside a larger paper; do not build a month around it.

---

### D5. Data efficiency and few-shot transfer on the AirfRANS scarce split — **RANK 5, do not headline**

**Cost: 4–6 days.** `run_baselines.py`, `train_airfrans.py`, the scarce split and
three backbones already exist.

**P(experiment works) ~ 0.9. P(novel) ~ 0.25. P(novel positive) ~ 0.22.**

The line publishes positive results as a matter of course (§1.5), which is
precisely why the marginal contribution is incremental. **Listed so it is
explicitly ranked and explicitly declined, not because it is recommended.**

One sharpening that would raise it: run the D1 covariate null *on the scarce and
extrapolation splits*. The honest expectation, from the verified Table 5, is that
the null wins there too — GraphSAGE reaches rho_D = 0.013 (reynolds) and 0.055
(aoa), and a quadratic in alpha extrapolates fine. **Do not build a pitch on
"models earn their keep OOD"; the data does not support it.** Run it anyway and
report it inside D1.

---

### D6. Conformal validity under geometry or Reynolds shift — **RANK 6, declined**

**P(experiment works) ~ 0.9. P(novel) ~ 0.15. P(novel positive) ~ 0.14.**

Method side is mature and general; the PDE-specific side already has an occupant
evaluating on geometry-shifted 3-D car CFD (arXiv:2603.11052 **[R]**), plus at
least four 2026 conformal-for-neural-operator papers (§1.6). The gap is "apply an
existing method to our setting," which is the exact shape that drew "not novel"
at desk twice. `results/sensitivity/ood_coverage.json` exists — **report it as a
paragraph, never as a contribution.**

---

### Summary table

| # | direction | days | P(positive) | novelty risk | ship with |
|---|---|---:|---:|---|---|
| **D1** | covariate null (**lift first**) + corrected leaderboard + harness | 10–13 | **0.86** | medium | D2, D3 |
| **D2** | surface wall-shear head; **magnitude** endpoint primary, rank gated | 6–9 | **0.51** | high alone, low with D1 | D1 |
| **D3** | lift/drag observability ladder | 3–4 | 0.90 (but a limits result) | low | D1 + D2 |
| D4 | trust-ranked evaluation, physics-free baseline | 4–6 | 0.62 | low, but expires 2026-12-06 | anything |
| D5 | data efficiency / few-shot | 4–6 | 0.22 | very high | section only |
| D6 | conformal under shift | 3–4 | 0.14 | very high | paragraph only |

**The recommendation is one paper: D1 + D2 + D3, ~20–25 days.** D1 defines the
metric, D3 diagnoses why the old metric failed, D2 moves the new one. Every
component has a verified closest competitor and a one-sentence differentiator,
and the combined headline is positive: *we give the null, we show who clears it,
and we clear it.*

---

## 4. What to avoid, and why

1. **Beating any 2026 SOTA on AirfRANS or DrivAerNet++ field MSE.** Transolver++
   is ICML 2025 multi-GPU million-point; AB-UPT is TMLR at industrial scale;
   CarBench and PhysicsBench have already consolidated the leaderboard. One
   4070 Ti, one month. Not achievable.

2. **Anything whose novelty is "conformal prediction applied to X."** Four+
   papers in 2026 alone (§1.6). This is the shape that desk-rejected twice.

3. **Building a trust benchmark.** RealPDE's Safe Prediction Score presents
   2026-12-06 with 15 organisers and NVIDIA/Caltech/UCSD advisors. Write the
   *finding*; cite RealPDE as convergent evidence of demand.

4. **A trust-gated selective-correction system paper.** ARC-STAR (2026-05-21)
   occupies it: frozen host, label-free risk scoring, budget-aware routing, 36x
   on Poseidon. The only gap left is the formal certificate, and this project's
   own results have narrowed that to a width concession (6.5–7.9x the drag error
   it bounds, and removing the floor *exactly* makes it **wider**, 7.4x -> 11.3x,
   3/3 seeds).

5. **G2 in its `whitespace.md` form ("what operator should you audit with").** Its
   premise — that the floor is operator provenance — was refuted on 2026-09-07
   (16/16 at fixed h). Do not build on it.

6. **PDE foundation models.** Compute-closed.

7. **Data efficiency as a headline.** Positive results are the field's default, so
   a positive result is not evidence of a contribution.

8. **The "you cannot audit a surrogate with a foreign operator" slogan.** This
   project's own control 4 falsifies it: that exact operator-inconsistent monitor
   triages worst-decile drag error at AUROC 0.952. Already flagged in
   `FINDINGS.md` and repeated here because it is the most likely self-inflicted
   wound in any reframe.

9. **Claiming DD-RNO's 0.997 is label leakage.** Tested and discarded in
   `drag_observability.md` §6: official `cdp` alone ranks total `cd` at 0.983, so
   a perfect surface-pressure integrator can legitimately reach ~0.98 with no
   viscous information. The defensible claim is the measurement gap (§1.2), not
   an accusation.

10. **Publishing any AirfRANS drag number without its covariate baseline.** That
    includes our own. It is the first thing a referee should ask of any paper in
    this line, and D1's whole value depends on us being the ones who ask it.

---

## 5. Citation ledger

**Verified this session, primary source, not previously in `refs.bib`:**

| citation | how verified |
|---|---|
| Bonnet, Mazari, Cinnella, Gallinari. *AirfRANS*. NeurIPS 2022 D&B. Tables 2–5, rho_D and rho_L, metric-hierarchy quote | NeurIPS proceedings PDF, read directly, pages 7–10 |
| Mehta, Bhati, Akolekar. *DD-RNO*. arXiv:2608.13490v1, 13 Aug 2026, IIT Jodhpur | arXiv PDF read directly: abstract, eqs. 10–11, 15–16, 28–31, and the partial-correlation passage (rho = 0.3365, partial R^2 ~ 0.081) |
| Wu, Luo, Wang, Wang, Long. *Transolver*. arXiv:2402.02366, ICML 2024. **No drag column; Appendix B.1 omission quote** | arXiv HTML, tables and appendix text |
| Luo et al. *Transolver++*. arXiv:2502.02414, ICML 2025, PMLR 267:41432–41449 | arXiv + PMLR listing |
| Elrefaie, Shu, Klenk, Ahmed. *CarBench*. arXiv:2512.07847, 25 Nov 2025, rev 20 Aug 2026. 11 architectures, **no parametric baseline** | arXiv abstract page |
| Elrefaie et al. *DrivAerNet++*. arXiv:2406.09624, NeurIPS 2024 D&B. 26 params; AutoML tabular baseline R^2 ~0.9 single-category, ~0.55 combined vs RegDGCNN 0.641 | arXiv HTML |
| *DrivAerML*. arXiv:2408.11969. 16 geometric params, MELS DoE | search extraction + arXiv listing |
| *AhmedML*. arXiv:2407.20801. 6 params, Latin hypercube, 500 variants, ~20M cells | arXiv PDF listing |
| Lee, Jeong, Kang. *PhysicsBench*. arXiv:2608.24056, 25 Aug 2026, leaderboard.narnia.ai. 66 models, 7 tasks, 9 datasets. **Tabular baselines confined to the 1-D scalar track; AirfRANS is field-prediction only; no lift/drag coefficient metric** | arXiv HTML, two targeted fetches |
| Tangsali, Ranade, Nabian et al. *A Benchmarking Framework for AI Models in Automotive Aerodynamics*. arXiv:2507.10747, 14 Jul 2025 | arXiv abstract page |
| Li et al. *ARC-STAR: Auditable Post-Hoc Correction for PDE Foundation Models*. arXiv:2605.22222, 21 May 2026 | arXiv abstract page |
| Yagoubi + 17. *NeurIPS 2024 ML4CFD Competition: Results and Retrospective*. arXiv:2506.08516, 10 Jun 2025, 240+ teams. **Single baseline (fully connected NN, global score 11.09%); no parametric baseline; no confound/leakage analysis** | arXiv HTML, targeted fetch |
| McGreivy & Hakim. Nature MI 6:1256–1269 (2024), arXiv:2407.07218. **79% (60/76) weak baselines** | Nature + arXiv listings |
| Scherz, Hines, Bekemeyer. *Evaluation of State-of-the-Art Deep Learning Architectures for Aerodynamical Predictions*. arXiv:2607.13866, 15 Jul 2026, 44 pp. Four operator-learning models; Bi-Stride MS-GNN and Transolver(++) highlighted; airfoil surface pressure + industrial 3-D aircraft; **no naive/parametric baseline, no force-metric confound analysis**. **This closes the `scherz2026evaluation` UNVERIFIED flag from `whitespace.md` §6** | arXiv abstract page |
| RealPDE Competition, NeurIPS 2026. Full schedule, SPS metric, tracks, organiser | realpdecompetition.github.io |
| TMLR acceptance criteria and zero fee | jmlr.org/tmlr |
| AIAA SciTech 2027, 11–15 Jan 2027, Orlando; "Special Session: Applied Surrogate Modeling" | AIAA call-for-papers + AASM announcements |

**Reused, already verified elsewhere in `docs/paper/review/` — do not re-check:**
AB-UPT (arXiv:2502.09692, TMLR); UQNO (arXiv:2402.01960, TMLR 2024, OpenReview
`cGpegxy12T`); Lei et al. (arXiv:2608.04400); PhysicsCorrect (arXiv:2507.02227,
AAAI 2026 Oral); ANCHOR (arXiv:2512.19643); Gopakumar et al. (arXiv:2502.04406);
Jia et al. (arXiv:2607.17297); Song et al. (arXiv:2603.11052); Zhang et al.
(arXiv:2602.14918); Duraisamy (arXiv:2604.20061); Jakeman et al.
(arXiv:2502.15496); AIAA 2025-0036; CFDONEval IJCAI 2025; WeatherBench 2
(arXiv:2308.15560); arXiv:2606.25752.

**UNVERIFIED — do not cite without checking:** arXiv:2602.04940 (Transolver-3);
2509.21670 (MORPH); 2605.10159 (jNO); 2603.04354; 2602.15004; 2608.28515;
2606.09923; 2509.04623; 2605.27968; 2602.00072; Phys. Rev. Research
10.1103/n5kp-jjcp (403 on fetch); Phil. Trans. R. Soc. A 384(2327):20250076;
AirfoilAD S0955598626001548 (abstract only, paywalled); Origo ICML 2026
(listing only); Poseidon venue details (listing only); Walrus / PhysicsX model
sizes.

**Occupancy holes closed this session** (they were the load-bearing risk on D1
and both came back clean): ML4CFD retrospective and PhysicsBench, each fetched
and searched specifically for parametric/naive/trivial baselines on aerodynamic
force metrics. Neither has one. See §1.2.2.

**Two things to check before any submission:** (i) the AIAA SciTech 2027 abstract
deadline, which did not surface and may already have passed; (ii) the exact
Computers and Fluids guide-for-authors length limit, since ScienceDirect returns
403 to automated fetches and the current rows came from search extraction.
