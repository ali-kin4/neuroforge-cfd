# Whitespace analysis — where this project can claim something both needed and unoccupied

Date: 2026-09-07. Companion to `FINDINGS.md`, `novelty_hunt.md`, `floor_resolution_study.md`.

Every citation below was checked against a primary source in this session unless
explicitly marked **UNVERIFIED**. Citations already verified in `novelty_hunt.md`
this session are reused and marked *(reused)*.

---

## 0. Two corrections to the premises this analysis was handed

Read these first. They change two of the six leads before any survey matters.

### 0a. The floor's mechanism is NOT the omitted turbulence gradient term

The task brief states the mechanism is the omitted `(d_j nu_t)(d_j u_i + d_i u_j)`,
with closed form `2*S*grad(nu_t)` ≈ 0.13 against a momentum floor of 0.1448.

**The repository's own data contradicts this**, in `floor_resolution_study.md` §6.2
(16 cases × 5 rungs, `results/certificates/floor_resolution_decomposition.json`):

| quantity | 128² | 512² |
|---|---:|---:|
| floor, band 0.1 | 0.04593 | 0.10737 |
| floor with full stress divergence **repaired** | 0.04592 | **0.10740** |
| omitted `∇ν_t` term alone | 0.00161 | 0.00484 (**4.5% of the floor**) |
| convective `u·∇u` | 0.1832 | 0.1994 (grid-converged) |
| pressure gradient `∇p` | 0.1802 | 0.1807 (grid-converged) |

Repairing the closure term moves the floor by **less than 0.1%**. The floor is the
**non-cancelling remainder of two individually grid-converged terms**, i.e. the
AirfRANS solution balances convection against pressure gradient in *its*
finite-volume operator on *its* body-fitted mesh, and that balance does not survive
resampling onto a Cartesian central-difference stencil.

Why this matters more than a footnote:

- If the closure omission were the cause, monitor design would be **trivial** — add
  the term, done — and the negative result would read as a bug report. Under
  provenance, no term-level repair on a Cartesian stencil closes the gap. That is
  what makes gap **G2** below a real design problem with a nontrivial answer.
- It makes the headline claim *operator provenance*, not *missing physics*. The
  defensible wording, per the study's own honest bound, is **"does not decay"**, not
  "grows without bound", and not "because the closure term is missing".

**Do not write the closure-term mechanism into the paper.** It is the cleanest
*illustration* of a non-vanishing component and it is 4.5% of the effect.

### 0b. Two of the six leads are already partly occupied

- **Lead #6 (calibrated error bars on integrated forces) is occupied.** Jia et al.,
  arXiv:2607.17297 *(reused; verified 2026-07-19, 6 authors, DrivAerML +
  GeoTransolver)* do conformalized quantile regression on the **drag coefficient**
  and residual-normalised conformal on surface pressure and wall shear stress. The
  bare claim "nobody reports calibrated uncertainty on integrated force
  coefficients" is false as of July 2026. What survives is different and stronger —
  see **G1**.
- **Lead #3 (cross-discretisation mismatch) is not a separate lead.** Under 0a it is
  the *same* claim as lead #2. Listing both is padding. Merged into **G2**.

---

## 1. Map of the field, 2025–2026

### 1.1 CROWDED — do not compete here

**Geometry-native neural operators / transformers for CFD.** Saturated and
industrialising.

| Work | Verified id / venue | Note |
|---|---|---|
| AB-UPT | arXiv:2502.09692 v4 (2025-10-13), **TMLR**. Alkin, Bleeker, Kurle, Kronlachner, Sonnleitner, Dorfer, Brandstetter | SOTA automotive; divergence-free hard constraint. **No UQ, no error certification** (checked). |
| AB-UPT for automotive & aerospace | arXiv:2510.15808 | UNVERIFIED details |
| DoMINO | NVIDIA PhysicsNeMo; shipped as **`DoMINO-Automotive-Aero` NIM container** | Production deployment. Docs surface accuracy and drag-sensitivity; **no uncertainty or validation story** in the NIM overview/deployment guides. |
| Transolver / Transolver++ | *(reused)* | Still a strong AirfRANS baseline in 2026 benchmarks. |
| GAOT | arXiv:2505.18781 | UNVERIFIED details |
| DD-RNO | arXiv:2608.13490 (2026-08-13), Mehta, Bhati, Akolekar | AirfRANS; see G1 — **directly relevant**. |
| SMART, SATO, GeoTransolver, AB-SWIFT | arXiv:2601.18707, 2603.25635, 2607.17297 | UNVERIFIED details for the first two |

**Conformal prediction / distribution-free UQ for PDE surrogates.** Extremely
crowded; roughly one new paper a month.

| Work | Verified id | Owns |
|---|---|---|
| Ma, Azizzadenesheli, Anandkumar — UQNO | arXiv:2402.01960; **TMLR, published 2024-09-21**, OpenReview `cGpegxy12T` — *this resolves the `ma2024uqno` UNVERIFIED venue flag in `novelty_hunt.md` §6* | functional conformal for operator learning |
| Gopakumar et al. | arXiv:2502.04406 *(reused)* | **residual as the nonconformity score**, FD-stencil monitor |
| Roy, Nayak, Goswami — ANCHOR | arXiv:2512.19643 *(reused)* | residual↔error as deployed data-free signal + solver fallback |
| Jia et al. | arXiv:2607.17297 *(reused)* | conformal on **drag** + surface fields, DrivAerML |
| Song, Li, Deng, Li, Pan, Lai, Wang | arXiv:2603.11052 (2026-02-24) — **verified; resolves the `song2026structureaware` UNVERIFIED flag** | structure-aware epistemic UQ, **geometry-shifted 3-D car CFD** |
| Chin | arXiv:2606.09923 (2026-06-07), single author | split conformal for neural operators, heat conduction |
| Split conformal in function space | arXiv:2509.04623 | UNVERIFIED |
| Conformal UQ guarantees for neural operators | arXiv:2608.28515 | UNVERIFIED |

**A-posteriori error estimation for learned PDE approximations.** Active, and
almost entirely in the **benign variational/elliptic regime** where the residual
characterises the exact solution (floor = 0 by construction).

- Astral (error majorants), arXiv:2406.02645 *(reused)*
- Jha, CMAME 419:116595 (2024), arXiv:2306.12047 *(reused)*
- Hillebrecht et al., IEEE TNNLS 36(1):1583–1593 (2025) *(reused)*
- RBNO with a-posteriori estimation, arXiv:2512.21319 — UNVERIFIED
- A-posteriori certification for NN approximations to PDEs, arXiv:2502.20336 — UNVERIFIED
- Lower and upper a-posteriori bounds for PINNs, arXiv:2606.12050 — UNVERIFIED

**Residual-driven correction / hybrid solver-surrogate coupling.** Crowded and
*succeeding*, on solver-consistent operators.

- Lei, Tang, Zhang, Chen, arXiv:2608.04400 *(reused)* — steady CFD, Newton–Krylov, residual ↓7 orders **and** error ↓
- Huang & Perdikaris, PhysicsCorrect, arXiv:2507.02227, AAAI 2026 Oral *(reused)* — transient
- ANCHOR, arXiv:2512.19643 *(reused)*
- ENS, arXiv:2606.27354 *(reused)* — residual as network *input*

**"Residual ≠ error" as an insight.** Owned collectively: Astral (2406.02645),
Luo & Zhou (2310.18201), Wang et al. NeurIPS 2022 (2206.02016) — all *(reused)*.

**Critique / reporting-standards papers in ML-for-PDEs.** A live and growing genre —
which is good for us (precedent) and bad (competition for the same slot).

- McGreivy & Hakim, *Nature Machine Intelligence* 6:1256–1269 (2024) *(reused)*
- **Duraisamy, "Predictivity and Utility of Neural Surrogates of Multiscale PDEs",
  arXiv:2604.20061 (2026-04-21)** — verified, single author. Distinguishes
  predictivity from utility, argues conflation caused overoptimism, and "calls for
  better reporting standards". **Must-cite; not in `refs.bib`.**
- Koehler & Thuerey, arXiv:2510.23111 *(reused)*

### 1.2 EMPTY or near-empty — verified nulls

Each of these is a *named-target* null, not a "I searched and found nothing".

**(N1) No published ASME standard for VVUQ of machine learning.** The committee
exists — **ASME VVUQ 70, "Verification, Validation, and Uncertainty Quantification
of Artificial Intelligence and Machine Learning"**, chartered, chair Joshua Kaizer,
vice chair Gregory Banyay, public meetings since Nov 2022
(`cstools.asme.org/csconnect/CommitteePages.cfm?Committee=103099834`). The
**ASME VVUQ Standards Portfolio Brochure, March 2026** lists the entire published
portfolio: VVUQ 1-2022, 10-2019, 10.1-2012, 10.2-2021, **20-2009 (CFD & heat
transfer)**, 20.1-2024, 30.1-2024, 40-2018, 40.1-2026, 50.1-2025, 60.1-2025.
**VVUQ 70 does not appear.** Confirmed independently against
`asme.org/codes-standards/vvuq-standards`. So: the standards body has formally
recognised the need and has published nothing, four years on.

**(N2) The one serious V&V-for-SciML paper contains no experiments.**
Jakeman, Barba, Martins, O'Leary-Roseberry, *Verification and Validation for
Trustworthy Scientific Machine Learning*, arXiv:2502.15496 (v1 2025-02-21, v2
2025-04-25, Sandia report SAND2025-01935O). Verified. Sixteen recommendations, four
groups (problem definition, verification, validation, continuous credibility). It
is **purely a position paper — no case study, no benchmark**. It does not discuss
residual-based a-posteriori estimation, conformal prediction, AirfRANS, ASME V&V 20,
or the grid convergence index. Its own open-problems list names exactly our
territory: models that "do not generally admit known convergence rates", and
"community benchmarks and dissemination mechanisms" still needing development.
Authors: Sandia + GWU + Michigan (Martins is the aerodynamic-shape-optimisation
community's centre of gravity). **This is the single most important missing
citation in the paper.**

**(N3) No CFD-ML benchmark scores whether the model knows it is wrong.**
- **CFDONEval**, IJCAI 2025, pp. 5752–5760 — 12 operator-learning models, 7 fluid
  problems, 22 datasets (18 newly generated). Metrics: **accuracy, efficiency,
  kinetic-energy spectra, visualisation.** No uncertainty, no calibration, no error
  detection.
- **AIAA Applied Aerodynamics Surrogate Modeling (AASM) benchmark cases** —
  Bekemeyer (DLR), Hariharan (DoD HPCMP), Wissink (US Army DEVCOM), Cornelius
  (NASA Ames), AIAA SciTech 2025, **AIAA 2025-0036**, doi:10.2514/6.2025-0036.
  Verified from the PDF. Four cases including **PALMO**: 52,480 OVERFLOW/SA
  simulations of NACA 4-series airfoils. The AASM group's stated scope explicitly
  includes "uncertainty quantification" — yet the benchmark as introduced reports
  **Cl and Cd accuracy curves only**. The paper states the motivating gap in its own
  words: the SciTech 2024 open discussion identified "the absence of common,
  publicly available benchmark cases specifically tailored to applied aerodynamic
  challenges". This is the institutional aerospace community (NASA + DLR + Army +
  DoD) naming the need and shipping an accuracy benchmark.
- PDEBench (arXiv:2210.07182, NeurIPS 2022 D&B), CFDBench (arXiv:2310.05963), The
  Well (NeurIPS 2024 D&B): accuracy-only. UNVERIFIED in detail, but consistent.
- **Contrast — the adjacent field solved this three years ago.** WeatherBench 2
  (arXiv:2308.15560, 2023) puts **CRPS and spread-skill ratio** on its public
  leaderboard, alongside deterministic RMSE, with ECMWF ENS as the probabilistic
  baseline. ML-weather ranks calibration; ML-CFD does not.
- **Partial and imminent occupancy, see §4 risks:** the **RealPDE Competition,
  NeurIPS 2026** (`realpdecompetition.github.io`; launch 2026-07-05, results
  2026-11-10, NeurIPS 2026-12-06; lead Tailin Wu, Westlake/Uniforce, 15
  contributors, advisors from NVIDIA/Caltech/UCSD; NACA4418 PIV + CFD) scores a
  **"Safe Prediction Score" that jointly evaluates physical accuracy, uncertainty
  coverage and interval tightness**. It is coverage-and-width, not error-ranking,
  and it is transient and real-world-data — but it substantially narrows this gap
  and it lands in three months.

**(N4) Nobody has posed audit-operator design as a problem.** Targeted searches for
monitor/audit-operator selection under a cost budget, and for cross-discretisation
mismatch as a systematic auditing problem, return only our own preprint
(arXiv:2607.10333) plus adjacent work that treats Cartesian resampling as a
*data-processing step* rather than an error source. Nearest hits are constructive
(ECO equilibrium-conserving operators, CMAME 2026, S0045782526004962 — UNVERIFIED;
DiFVM differentiable unstructured FV, arXiv:2603.15920 — UNVERIFIED), not
diagnostic.

**(N5) Nobody has measured the observability ceiling of an engineering quantity
under a surrogate's output representation — for incompressible aero.** One paper
asks exactly this question in a different regime: *Quantity-Dependent Bulk-to-Wall
Observability of Surface Loading in Rarefied Hypersonic Flow over Triangular
Protrusions*, arXiv:2606.25752 (verified) — Mach-6 DSMC, 57 conditions, asks which
wall loads are recoverable from which bulk state representations, and concludes
higher moments alone are insufficient. **That framing is exactly right and it has
never been applied to RANS aerodynamics.** See G1.

---

## 2. The gaps, ranked

Five, not eight. Each states the need, the occupancy, why it is open, and
reachability from the existing asset.

---

### G1 — Observability of the engineering quantity under the surrogate's output representation. **RANK 1.**

**The claim.** *Before you report a drag correlation, measure what your integrator
achieves on the ground-truth field. On a 128² Cartesian raster of AirfRANS, no
near-wall surface-integration scheme recovers the official drag ranking beyond
ρ_D ≈ 0.84, and the deployed model attains that ceiling exactly — so improving the
model cannot improve the reported drag.*

**Who needs it and why.** Every aerodynamicist. Drag is the number that pays for the
simulation. AASM/AIAA 2025-0036 names integrated coefficients as the first benchmark
case. If reported drag error is dominated by post-processing rather than by the
model, an entire literature's headline metric is measuring the wrong thing.

**What we already own (verified in-repo, model-free where stated).**
`results/control/integrator_design.json` (n=200, ground-truth fields, seed-independent)
and `results/control/force_vs_official_multischeme.json` (Transolver, 3 seeds):

| integration scheme | ρ_D, **truth** through integrator (ceiling) | ρ_D, prediction | ρ_L truth | ρ_L prediction | median \|Δc_d\|/c_d, truth |
|---|---:|---:|---:|---:|---:|
| near-field, offset (deployed) | **0.839** | **0.839 ± 0.007** | 0.876 | 0.879 ± 0.000 | 332% |
| near-field, wall-extrapolated | **0.800** | **0.799 ± 0.026** | 0.882 | 0.902 ± 0.007 | 220% |
| far-field control volume | 0.611 | 0.400 ± 0.082 | **0.99999** | 0.998 ± 0.001 | **23.5%** |

Three facts fall out. **Fact 1 is currently over-stated and fact 3 is the safest**
— read A0 in §3 before quoting any of them.

1. **The two near-field schemes have no measurable headroom.** The prediction ranks
   drag as well as the exact truth does through the same integrator (0.839 vs 0.839;
   0.799 vs 0.800). *Careful:* equality of two Spearman correlations is **not**
   proof that the representation dominates — Spearman is not additive, and two
   comparable independent error sources can coincide. The defensible wording until
   the per-case check in A0(c) is done is **"the model is not the dominant term"**,
   not "improving the model cannot improve the number".
2. **An 18-variant design sweep does not escape it.** Pressure scheme
   (offset/extrapolated), shear scheme (field_abs/field_signed/no-slip) and standoff
   (1.0/1.5/2.0 cells) span ρ_D ∈ [0.796, 0.841] and median c_d error ∈ [214%, 369%].
   This is not a tuning failure.
3. **Lift and drag have opposite observability.** A far-field momentum balance
   recovers lift from the truth at ρ_L = 0.99999 and 0.05% median error, and from
   predictions at 0.998 — while its drag ranking collapses to 0.611/0.400. Lift is
   observable from the outer ring; drag is not observable from anywhere on this
   representation.

**Occupancy.** Effectively one group, from the opposite direction, and it is a
must-cite and a partial scoop risk:

> **DD-RNO**, arXiv:2608.13490 (2026-08-13), Mehta, Bhati, Akolekar. On AirfRANS,
> "learned canonical quadrature replaces unstable pressure integration with
> flow-conditioned, learned integration weights", reducing drag MSE **7.5×** and
> raising drag rank correlation **from 0.250 to 0.997**.

They **measure the symptom and treat it as a modelling opportunity**; they do not
report the ceiling on ground-truth fields, so they cannot distinguish "our quadrature
integrates better" from "our quadrature regresses drag from surface pressure using
the label". Our ground-truth ceiling is precisely the control that separates those
two, and their own 0.250 number is the strongest available evidence that the control
is needed. Note their 0.250 is on a different (mesh-native) representation from our
0.839 — **scope honestly, do not merge the numbers.**

The conceptual framing exists in one adjacent paper only: arXiv:2606.25752
(rarefied hypersonic, DSMC). Zero papers do this for incompressible RANS aero.

**Why still open.** Unnoticed rather than hard. Measuring your own integrator's
ceiling requires you to suspect it, and the incentive structure rewards reporting a
model number, not auditing the measurement instrument. It is also invisible unless
you have the *official solver-computed* labels to compare against, which AirfRANS
provides and most datasets do not.

**Reachability: highest in the report — with one blocking control.** The measurement
is already in the repository, seed-independent, n=200, with no model in the loop.
But it currently conflates **rasterisation loss** with **integrator-formulation
difference**: `gt_nf` is *our* `force_coefficients` on a *128² raster*, compared
against `Simulation.force_coefficient(reference=True)`, which is AirfRANS's own
integrator on the body-fitted cloud. The 18-variant sweep controls our integrator's
design choices but every variant is still our quadrature on a raster. **The
separation has never been run** — `force_vs_official.json` says so in its own meta:
`"pred_af_skipped": "airfrans' own integrator on our prediction skipped"`. That is
A0 in §3, and it blocks the ladder.

---

### G2 — What operator *should* you audit with? The constructive converse. **RANK 2.**

**The claim.** *Given the provenance mechanism, characterise the cheapest monitored
operator whose residual floor is small relative to the error signal it must detect —
and show that term-level repair is not the axis that matters.*

**Who needs it.** Anyone building the trust layers in §1.1 — Gopakumar, ANCHOR,
PhysicsCorrect, Jia, and every industrial deployment (DoMINO NIM) that will
eventually need one. They all pick a monitor by convenience and none reports its
floor.

**Occupancy: none found (N4).** Targeted searches return our own preprint.

**Why still open.** It could not be posed before the negative result. And under 0a it
is genuinely hard: the answer is not "add the missing terms" (that moves the floor
<0.1%), it is about mesh topology, flux reconstruction and where the reference
solution's own balance lives. That difficulty is *why* it is unoccupied and *why*
it is worth a paper rather than a paragraph.

**What we own.** The full ladder — floor vs h for the raw operator, the
stress-repaired operator, the C¹ re-rasterisation, MMS analytic and MMS raster nulls
(`floor_resolution_decomposition.json`), plus a measured audit cost of **1.65 ms
median per case** against 1500 s for the classical solve, i.e. **~9×10⁵×**
(`results/control/audit_cost.json`). The cost axis of the design problem is already
instrumented.

**Reachability: good, but this is the ambitious one.** A full "cheapest sufficient
monitor" theory is a second paper. What is reachable now is the *diagnostic*: a
floor-to-signal ratio reported alongside any residual monitor, with a recipe for
measuring it. See §3, A2.

---

### G3 — An executed V&V argument for a neural CFD surrogate, in the standards' own language. **RANK 3.**

**The claim.** *Instantiate ASME VVUQ 20-style code verification / solution
verification / validation-uncertainty on a neural surrogate, on a public benchmark,
and report what does and does not transfer.*

**Who needs it.** The engineer signing the design, and the ASME VVUQ 70 committee
that has been chartered since 2022 with nothing published (N1). Jakeman et al.
(N2) wrote the recommendations; **nobody has executed them on a case**.

**Occupancy.** One position paper (N2, no experiments), one chartered committee
(N1, no standard), one accuracy-only aerospace benchmark (N3, AIAA 2025-0036). The
execution slot is open.

**Why still open.** Unfashionable at ML venues (no new architecture) and
unfamiliar at CFD venues (no new physics). It is exactly the genre that lost this
paper two desk rejections — which is a warning, not a disqualification: the fix is
to lead with a *measurement*, per McGreivy & Hakim and Duraisamy.

**Reachability: partial, and this is where honesty bites.** We can execute maybe
five of the sixteen recommendations well:
- R5/R6 code and solution verification → **owned**: MMS order 2.06, reproduction to
  3.3e-8, the resolution ladder.
- R3 quantities of interest → **owned via G1**, and the finding is negative.
- R9 quantify prediction uncertainties → **owned**: split conformal at target
  coverage on the deployed corrected field.
- R15 compare against alternatives → **owned**: 3 backbones, 2 datasets, physics-free
  ensemble baseline.
We cannot do R7 probabilistic calibration against experiment, or R8 purpose-specific
requirements, without a design task. **Do not claim "we perform V&V".** Claim: *we
map a neural CFD surrogate onto the ASME/Jakeman V&V structure and report which
elements are computable today and which are not.* The negative half is the
contribution.

---

### G4 — Trust-ranked rather than accuracy-ranked evaluation for CFD surrogates. **RANK 4, and time-limited.**

**The claim.** *Rank surrogates by their ability to know when they are wrong —
risk-coverage curves, selective-prediction AUROC, oracle recovery — across
backbones, datasets and scoring functions.*

**Who needs it.** Same constituency as G3. WeatherBench 2 shows the demand is real
and the solution is standard practice one field over.

**Occupancy: partial and closing fast.** CFDONEval and AASM are accuracy-only (N3).
But **RealPDE / NeurIPS 2026's Safe Prediction Score lands 2026-12-06** and scores
coverage + tightness. Our angle differs (error *ranking* / selective prediction, not
interval width) and our finding is interesting — from
`results/selective/selective_prediction.json`:

| score | Spearman | AUROC | oracle recovery |
|---|---:|---:|---:|
| physics residual | 0.610 | 0.871 | ~67% |
| `sigma_vel`, physics-free | 0.654 | 0.894 | ~72% |
| fused | **0.703** | **0.905** | **~91%** |

**a physics-free ensemble spread outranks the physics residual, and physics earns
its keep only in fusion.** That is a genuinely useful negative for everyone building
§1.1's residual-based trust layers.

**Why partly still open.** It is infrastructure work with no architecture in it.
**Reachability: high, but the window is ~3 months.** Do not build a benchmark; state
the comparison as a finding and cite RealPDE as convergent evidence of the need.

---

### G5 — Conformal validity when exchangeability fails, for PDE surrogates. **RANK 5 — real need, but do not lead with it.**

**Who needs it.** Everyone deploying the §1.1 conformal layers, since deployment is
the covariate-shift case by definition.

**Occupancy.** Method side is mature and general (weighted conformal under covariate
shift; adaptive conformal inference; Lévy–Prokhorov robustness, arXiv:2502.14105 —
UNVERIFIED). PDE-specific side has at least one occupant already:
**arXiv:2603.11052 evaluates on geometry-shifted 3-D car CFD** (verified). We have
`results/sensitivity/ood_coverage.json`.

**Why lower.** The gap is "apply an existing method to our setting", which is the
exact shape that drew "not novel" twice. Report it; do not headline it.

---

## 3. Concrete additions, ranked by impact per unit of work

Costs assume the existing single-GPU workstation and the existing scripts.

---

### A0. Separate rasterisation loss from integrator formulation. **BLOCKING. DO THIS FIRST.**
**Why.** ρ_D = 0.839 on the ground-truth field is the load-bearing number of G1 and
of §4, and it currently mixes two mechanisms. Without A0, **the A1 ladder is
uninterpretable**: a flat ladder is equally consistent with "the raster destroys the
information" and with "our quadrature carries a raster-independent offset from
AirfRANS's".

**The discriminating arm, and note the trap.** Running AirfRANS's own
`Simulation.force_coefficient(reference=True)` on the native cloud is *not* a
control — it **is** the label by construction, so it returns ≈1.0 trivially. The
control that separates the mechanisms is a **raster round-trip**:

| | AirfRANS's integrator | our integrator |
|---|---|---|
| native cloud | 1.0 by construction — not informative | **(b) never measured** |
| rasterise → scatter back to mesh nodes | **(a) never measured** | 0.839 (have it) |

- **(a)** Rasterise the GT to 128², interpolate back onto the original mesh nodes,
  call `Simulation.force_coefficient` on *those* fields. If ρ_D collapses to ≈0.84,
  **the raster owns the loss** and G1's framing is clean. If it stays ≈0.98, most of
  the 0.839 is our quadrature and the "representation ceiling" claim must be
  substantially weakened to "our integrator's ceiling".
- **(b)** Our `force_coefficients` on the native cloud (no raster). Brackets (a) from
  the other side.
- **(c)** Per-case correlation between prediction drag error and ground-truth drag
  error, from the arrays in `results/control/_cache/`. This is what licenses fact 1's
  strong form; without it, use the weak form.

**Cost.** ~4–6 h. (a) reuses the existing rasteriser plus a scatter-back
interpolation; (b) is a coordinate change in an existing function; (c) is analysis.

**Risk.** *This is a risk-reduction experiment, so "failure" is success.* The bad
outcome is (a) ≈ 0.98, which costs G1 its headline but is far cheaper to learn now
than in review — and it would still leave the honest and useful finding "report your
integrator's ceiling", plus the far-field/near-field observability asymmetry (fact
3), which is a property of the *scheme family*, not of our quadrature, and survives
either way.

---

### A1. The drag-observability ladder. **Second, and gated on A0.**
**Claim it buys.** *The ceiling on integrated drag from a Cartesian raster of
AirfRANS is a property of the representation, not of the grid: it does not close
under refinement, while lift's does not need to.*

**Experiment.** Merge two scripts that already exist: rasterise the AirfRANS truth at
128²/181²/256²/362²/512² (`scripts/floor_resolution_ladder.py`, already validated,
MMS-gated) and run `force_coefficients` at each rung against the cached official
OpenFOAM labels (`scripts/design_force_integrator.py`,
`scripts/recompute_force_vs_official.py`, labels cached in
`results/control/_cache/official_labels_full_test_n200.json`). Report ρ_D, ρ_L and
median relative error per rung, for the three scheme families.

**Cost.** ~6–10 h wall-clock, CPU-dominated; the 5-rung floor ladder took 17.8 min on
16 cases, and force integration is cheaper than the residual. Add ~4 h of analysis.
**Under one working day.**

**Prior art it must cite/beat.** DD-RNO arXiv:2608.13490 (the 0.250→0.997 number is
the reason this matters); arXiv:2606.25752 for the observability framing; AirfRANS
(Bonnet et al. NeurIPS 2022) for the labels; AIAA 2025-0036 for the need.

**Risk of failure.** *Moderate.* If ρ_D climbs toward 0.99 by 512², the claim
weakens from "representational ceiling" to "resolution requirement" — which is
**still publishable and still useful** ("report the raster resolution at which your
drag metric becomes meaningful"), just less striking. Given that the *residual* floor
rises under the same refinement and by the same non-cancellation mechanism, a rising
or flat drag ceiling is the more likely outcome, but this is a prediction, not a
result. Pre-register the decision rule before running, exactly as `bf5106c` did for
the floor ladder.

**Second-order risk.** The near-wall band is where h approaches the source cloud's
local spacing (median 4.4e-5 chord near the wall). At 512² you are close to
differentiating a reconstruction. Use the same physical-units exclusion and state
the same honest bound.

---

### A2. The floor-to-signal ratio as a reportable diagnostic.
**Claim.** *A residual monitor is usable as a ranker only where
‖r*‖ ≪ ‖L e‖; we give the ratio on AirfRANS, show where it inverts, and specify how
to measure it for any monitor.*

**Why it is worth doing.** It converts the negative result into a **one-number test
other people can run**, which is what makes a critique paper get cited rather than
merely agreed with. It also directly addresses the internal contradiction flagged in
`FINDINGS.md` F7 — the deployed model sits *below* the truth's floor
(0.1136 vs 0.192, 80% of cases) yet ρ = 0.61 is measured there. Reporting the ratio
per-case, and showing ρ conditioned on it, either resolves that contradiction or
honestly names it as unexplained.

**Experiment.** Per case: ‖r*‖ (have it), ‖r(pred)‖ (have it), field error (have it).
Bin by ratio, report ρ within bins. Add the measured audit cost (1.65 ms vs 1500 s)
as the second axis, so "affordable" stops being an assertion.

**Cost.** ~3–4 h. Everything is already in JSON; this is analysis, not simulation.

**Prior art.** Zhang et al. arXiv:2602.14918 App. E *(reused)* — they need
`R_h(u*)` and so does this diagnostic at calibration time, which must be conceded
openly; Lei et al. arXiv:2608.04400 *(reused)* as the ratio≈0 endpoint.

**Risk.** *Low technically, moderate rhetorically.* The diagnostic needs the
ground-truth residual on a calibration set, so it is a *design-time* test, not a
deployment-time one. Say so plainly; it is still exactly the test a monitor's
designer needs.

---

### A3. Map the paper onto Jakeman et al.'s 16 recommendations, and report the misses.
**Claim.** *Here is what a V&V argument for a neural CFD surrogate can and cannot
contain today.*

**Work.** A table, plus honest prose. Green for R3/R5/R6/R9/R14/R15, red for
R7/R8/R16, with one sentence each on why. No new computation.

**Cost.** ~4–6 h of writing.

**Prior art.** arXiv:2502.15496 (verified); ASME VVUQ 20-2009 and VVUQ 20.1-2024;
ASME VVUQ 70 committee (chartered, unpublished) — the last is the sentence that
establishes the need in one line.

**Risk.** *Low.* The only failure mode is overclaiming. Do not write "we perform
V&V"; write "we map onto, and report the gaps".

---

### A4. Report the physics-free baseline honestly, with a paired bootstrap.
**Claim.** *On steady-RANS surrogates, a physics-free ensemble spread ranks errors
better than the physics residual (AUROC 0.894 vs 0.871), and physics contributes
only in fusion (0.905).*

**Work.** The paired bootstrap is already in flight
(`results/control/bootstrap_spearman_ci.json` exists). Finish it, and rewrite the
abstract's "AUROC ≈ 0.9 for the residual" attribution, which `FINDINGS.md` F6 flags
as currently misleading.

**Cost.** ~2–3 h.

**Risk.** *This one is a credibility asset regardless of outcome.* If the difference
is not significant, the honest statement — "we cannot distinguish them" — is still a
correction to the field's implicit assumption that the physics residual is the
better signal.

---

### A5. Disclose the MGN density control.
`results/control/mgn_density_control.json` carries `"verdict": "DENSITY-DRIVEN"`
(ratio 2.44, trained at 16k points, evaluated at 180k) and is never mentioned in the
manuscript. `FINDINGS.md` F7 correctly calls this a non-disclosure. **Cost: ~1 h.
Risk of not doing it: fatal if a reviewer finds it.** This is not optional.

---

### Deliberately NOT recommended
- A new dataset or 3-D. Weeks of work, no claim that A1 does not already buy.
- Building a trust benchmark. RealPDE lands 2026-12-06 (§4).
- A full "optimal audit operator" theory (G2's ambitious form). Second paper.
- Anything that requires re-deriving theorem leg (iii), which `FINDINGS.md` F2 shows
  bounds the wrong direction. Delete it rather than repair it.

---

## 4. The single strongest claim reachable in about a month

**State it in two independent parts. Do not stake the headline on the link between
them until A0 and A1 return.**

> **(Earned today.)** On AirfRANS, the steady-RANS residual a deployed surrogate can
> afford to monitor does not vanish at the ground truth and **grows** under grid
> refinement (p = −0.64 ± 0.29, 0/16 cases decaying, ×2.6 from 128² to 512²),
> because the reference solution's individually grid-converged convective and
> pressure-gradient terms fail to cancel under a different discrete operator —
> repairing the omitted closure term moves the floor by <0.1%. The obstruction is
> **operator provenance**, not resolution and not missing physics, so a surrogate
> cannot be certified by a residual monitor that is not the one its training data
> solves.

> **(Second, independent measurement — pending A0.)** On the same benchmark, no
> near-wall integration scheme recovers the official drag ranking from the **exact
> ground-truth field** beyond ρ_D ≈ 0.84 across an 18-variant design sweep, while a
> far-field momentum balance recovers **lift** at ρ_L = 0.99999 and 0.05% median
> error — so on this representation lift is observable and drag is not.

The residual half is fully earned: 16 cases × 5 rungs, three artifact controls (MMS
analytic, MMS raster null, C¹ re-rasterisation), and a term-level decomposition. The
drag half has **none** of that yet — it is one uncontrolled comparison plus a design
sweep. Two separately measured ceilings on one public benchmark is already a strong
paper. **Upgrade to the single "one mechanism, two consequences" sentence only if
A0(a) shows the raster owns the loss and A1's ladder does not close it** — at which
point the merged claim becomes the strongest thing this project can say.

Note what these sentences deliberately do *not* say: not that the closure term is the
cause (§0a), not that descent always diverges (`novelty_hunt.md` §2 wording
discipline), not that the floor grows without bound (`floor_resolution_study.md`
§6.2 honest bound), and not that improving the model cannot improve drag (G1 fact 1,
weak form).

---

## 5. Honest risks

**A1 / G1.**
- *Scooped by DD-RNO's follow-up.* They are actively working the AirfRANS force
  problem as of Aug 2026 and are one control away from our result. **This is the
  highest scoop risk in the report.** Mitigation: the ground-truth ceiling is already
  computed and committed; timestamp the ladder on arXiv quickly.
- *Scope attack:* "your ceiling is about 128² rasters; mesh-native models are
  unaffected." Partly true and must be conceded. The defensible scope is
  raster/voxel-output surrogates (DeepCFD, U-Net, FNO families) — a large class, not
  all of them. Do not compare our 0.839 against DD-RNO's 0.250 as if they were the
  same measurement.
- *Label attack:* the AirfRANS official coefficients are themselves an OpenFOAM
  post-process with their own error. We are measuring agreement with a reference,
  not with truth. Say so.

**A2 / G2.**
- *"You chose a bad monitor"* is the standing rebuttal, and Lei et al. arXiv:2608.04400
  is the ammunition. The only answer is the cost column: a solver-consistent monitor
  costs a body-fitted mesh and a solver, i.e. the thing the surrogate exists to
  avoid. 1.65 ms vs 1500 s is that answer; put it in the positioning table.
- *The floor-to-signal diagnostic needs `R_h(u*)`*, which Zhang et al. 2602.14918
  already exploit at training time. Concede priority on the workaround, claim the
  diagnostic use.

**A3 / G3.**
- *Desk-reject risk on "no new method" is the same one that already fired twice.*
  Mitigation is structural: the V&V mapping must be a *section*, never the framing.
  Lead with A1's measurement.
- *A VVUQ 70 draft could appear.* Committee has been silent four years, so the
  probability inside a month is low — but check `cstools.asme.org` before submission,
  and check the AIAA SciTech 2027 programme for an AASM UQ case.

**A4 / G4.**
- **RealPDE / NeurIPS 2026 results publish 2026-11-10 and present 2026-12-06.** Any
  sentence of the form "no benchmark evaluates whether a surrogate knows it is wrong"
  becomes false in December. Write the finding, not the benchmark, and cite RealPDE
  as convergent evidence that the field wants this.
- *The physics-free baseline result cuts against our own trust layer.* That is
  precisely why it should be reported by us and not by a reviewer.

**Cross-cutting.**
- *The shared-mechanism claim is two unverified steps deep.* It assumes (i) the drag
  ceiling is representational — unproven until A0 — and (ii) that it is the *same*
  non-cancellation as the residual floor — untested. §4 is now split so the headline
  does not depend on either. Do not re-merge it in prose before the evidence merges.
- *If A1 comes out flat-but-not-rising while the residual ladder rises*, the framing
  is "two related ceilings", not one mechanism. Fallback wording is in §4; fix it
  before running, not after.
- *`FINDINGS.md` F7's two-Laplacian-stencil discrepancy* (`residuals.py:524` wide vs
  `operators.py:186` compact) is a live alternative explanation for the W1 null and
  is unresolved. It does not touch A1, but it must be fixed or disclosed before any
  version of the residual claim ships.

---

## 6. Citation actions

**Must add (verified this session, none in `refs.bib`):**
1. Jakeman, Barba, Martins, O'Leary-Roseberry. *Verification and Validation for Trustworthy Scientific Machine Learning.* arXiv:2502.15496 (2025). **[the V&V-for-SciML anchor]**
2. Bekemeyer, Hariharan, Wissink, Cornelius. *Introduction of Applied Aerodynamics Surrogate Modeling Benchmark Cases.* AIAA SciTech 2025, AIAA 2025-0036, doi:10.2514/6.2025-0036. **[the aerospace community naming the need]**
3. Mehta, Bhati, Akolekar. *DD-RNO: A Domain-Decomposed Routed Neural Operator for Airfoil Flow Prediction.* arXiv:2608.13490 (2026). **[closest prior art to G1 — do not omit]**
4. Duraisamy. *Predictivity and Utility of Neural Surrogates of Multiscale PDEs.* arXiv:2604.20061 (2026). **[genre precedent, reporting standards]**
5. ASME VVUQ 20-2009 (and VVUQ 20.1-2024); ASME VVUQ 70 committee as the unpublished-standard citation.
6. Rasp et al. *WeatherBench 2.* arXiv:2308.15560 (2023). **[the adjacent field's probabilistic leaderboard]**
7. arXiv:2606.25752 — bulk-to-wall observability. **[the framing for G1]**
8. CFDONEval, IJCAI 2025 pp. 5752–5760. **[accuracy-only CFD-ML benchmark]**
9. Alkin et al. *AB-UPT.* arXiv:2502.09692, **TMLR**. **[2026 SOTA, and it has no UQ]**
10. RealPDE Competition, NeurIPS 2026. **[cite as convergent need; check status before submission]**

Plus the nine already listed in `novelty_hunt.md` §7, which stand.

**Flag resolutions (updating `novelty_hunt.md` §6):**
- `ma2024uqno` — venue **RESOLVED: TMLR, published 2024-09-21**, OpenReview `cGpegxy12T`.
- `song2026structureaware` — **RESOLVED: arXiv:2603.11052 (2026-02-24)**, 7 authors: Song, Li, Deng, Li, Pan, Lai, Wang. Evaluates on geometry-shifted 3-D car CFD.
- `garg2025dfuq`, `yu2026conformalpinn`, `rigotti2026gist`, `scherz2026evaluation` — still **UNVERIFIED**.

**Marked UNVERIFIED in this report and not to be cited without checking:**
arXiv:2510.15808, 2505.18781, 2601.18707, 2603.25635, 2509.04623, 2608.28515,
2512.21319, 2502.20336, 2606.12050, 2502.14105, 2603.15920, 2310.05963;
CMAME S0045782526004962; PDEBench/The Well metric details.
