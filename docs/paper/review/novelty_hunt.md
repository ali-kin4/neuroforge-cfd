# Novelty hunt — what in this paper cannot be desk-rejected on originality

Date: 2026-09-07. Prepared after two desk rejections on originality (CMAME 2026-08-02; JCP 2026-09-07).
Every citation below was verified against the arXiv/publisher abstract page in this session unless
explicitly marked UNVERIFIED.

---

## 0. One-paragraph answer

The novelty is **not** the trust layer, and it is **not** the theorem's mathematics. It is a
**measurement**: the discrete steady-RANS residual that a deployed neural surrogate can actually
afford to monitor does **not vanish at the dataset's ground truth** — on 200/200 real AirfRANS
cases $\|r^\star\|$ has mean $0.192$ / median $0.133$, against an *exact zero* for the physically
wrong uniform-freestream field, and the trained prediction sits *below* the truth's residual in
160/200 cases (`results/certificates/residual_floor_realdata.json`). Editors desk-reject claimed
*methodology*; they rarely desk-reject a measured, reproducible fact about a practice the field is
currently building on. The paper today buries this in an appended theorem and leads with the least
novel of its three claims.

**But two of the numbers you would most like to headline are not yet strong enough to carry it.**
See §0a before rewriting anything.

---

## 0a. Evidence-strength ledger (read this before choosing the headline)

Sorted by strength, because the most rhetorically striking result is the least supported.

| Evidence | Backbone | n | Seeds | Strength |
|---|---|---|---|---|
| $\|r^\star\|$ = 0.192 mean / 0.133 median at the truth; uniform field = exactly 0 on **200/200** | model-independent | 200 | n/a | **Solid.** Full test split, no model involved. |
| W1 null-corrector ablation: WITH beats NULL on **0/5** seeds | Transolver (SOTA) | 200 | 5 | **Solid.** |
| $\lambda$-sweep: no no-slip weight $\ge0$ rescues the objective | — | closed form | n/a | **Solid**, and currently invisible in the paper. |
| Acceptance gate: 99.8% admit, 89.3% error-reducing | Transolver + ensemble | 200 | 3 (1200 steps) | **Solid.** |
| Prediction below truth's floor on **160/200** | dropout-FNO — the *weak* backbone, described in-paper as "over-smoothed" | 200 | 1 checkpoint | **Weak.** See below. |
| Divergence sweep: residual $0.11\!\to\!0.62$ while mse_u $3.92\!\to\!2.29$ | weak grid backbone | 1 checkpoint | **unseeded** (`tab:iters` caption: "Single sweep (one checkpoint, not seeded)"; Limitations repeats it) | **Weakest.** n=1. |

**The obvious reviewer attack on 160/200:** *your surrogate has a lower residual than the truth
because it is over-smoothed, and a smooth field has small derivatives — that is a fact about your
weak model, not about residual monitors.* The paper hands over the ammunition itself
(`residual_floor_theorem.tex` l.114 calls the checkpoint "over-smoothed" and "distinct from the
Transolver headline"), and §1 below concedes that degenerate low-residual fields are a documented
failure mode. The data supports the attack: `norm_pred_std` = 0.034 against `norm_truth_std` = 0.162
— the prediction's residual distribution is flat and narrow, exactly what over-smoothing produces.

**Two experiments, in this order, before the headline is rewritten:**

1. **Recompute the inversion on the Transolver backbone.** The harness and checkpoints exist. If a
   *competitive* surrogate also sits below $\|r^\star\|$, the claim transforms: the monitor is beaten
   by SOTA-quality predictions, not merely by smooth ones. If it does not hold, the headline **must**
   be the floor magnitude itself (200/200, model-independent), not the inversion count.
2. **Seed the iteration sweep.** It is a sweep over `n_iters` on existing checkpoints — the cheapest
   high-value experiment available. Until then the falsification of *descent* is n=1. Note the W1
   ablation does **not** backfill it: W1 tests residual-as-*input*, not residual-as-*objective*.

Fallback headline if (1) fails: lead with the floor magnitude and the exact-zero uniform field
(200/200, no model), and present the divergence sweep as corroborating illustration rather than as
the claim.

---

## 1. Ranked verdict on the three candidates

### Rank 1 — Candidate 2: falsification of residual-as-correction-objective (**strongest**)

**Verdict: partially novel, and the novel part is the load-bearing part.**
The *thesis* ("residual is a poor proxy for error") is NOT new. The *measurement in this regime*
— monotone divergence of residual and error along a learned correction path on steady RANS, plus
the null-corrector ablation — is, to my knowledge, new. No prior work found reports residual and
error moving in opposite directions on a steady-state CFD surrogate.

Prior art that *dents* it (all must-cite, none currently cited):

| Work | Verified id | What it already says | Why it does not kill us |
|---|---|---|---|
| Fanaskov, Yu, Rudikov, Oseledets — *Astral: training PINNs with error majorants* | arXiv:2406.02645 (v1 2024-06-04, v2 2026-03-02; ICLR 2026 AI&PDE workshop) | Verbatim: "residual is, at best, an indirect measure of the error of approximate solution"; shows their majorant loss is *better correlated with error than residual* | PINN **training loss**, elliptic/Maxwell/magnetostatics, continuous residual, and they never observe residual and error moving *apart*. They weaken "residual≠error" as an insight, not our measurement. |
| Jiang, Wang, Chen, Kwak, Kim, Bell, Park — *Error-Conditioned Neural Solvers* (ENS) | arXiv:2606.27354 (2026-06-25) | "numerically minimizing the PDE residual can be an unreliable proxy for reconstruction accuracy in ill-conditioned systems"; passes residual fields as network **inputs** rather than optimisation targets | Agrees with us on residual-as-objective. On residual-as-*input* our W1 control is a **non-replication in a different setting** (different architecture, different problems, their claim scoped to ill-conditioned/extrapolation regimes) — report it that way, **not** as a head-on contradiction. Picking a fight with a concurrent preprint is a fight we do not need. |
| Luo, Zhou — *On Residual Minimization for PDEs: Failure of PINN, Modified Equation, and Implicit Bias* | arXiv:2310.18201 (2023-10-27) | Residual minimisation implicitly biases the solution *away from the exact one*, toward a "modified solution" | Continuous residual, elliptic PDEs with discontinuous coefficients, network-approximation mechanism — not a discrete-operator floor and not CFD. Same conclusion shape, different mechanism. |
| Wang, Li, He, Wang — *Is $L^2$ Physics-Informed Loss Always Suitable for Training PINNs?* | arXiv:2206.02016, **NeurIPS 2022** | Proves small $L^2$ residual does not control solution error for a class of HJB equations; $L^\infty$ needed | Stability-class result in function space; no CFD, no post-hoc correction. |

Prior art that *supports* the regime boundary (and is a **must-cite we are missing**):

- **Lei, Tang, Zhang, Chen — "Reliable and efficient steady CFD from surrogate predictions through
  Newton-Krylov correction", arXiv:2608.04400 (2026-08-05).** Steady CFD (transonic/supercritical
  airfoils, 3-D flying wing). Surrogate prediction as initial guess for Newton–Krylov on the
  **solver's own consistent discrete operator**: median residual $L_2$ ratio drops **>7 orders of
  magnitude** *and* field and aerodynamic errors fall; 15.5× generation-level speedup.
  This is far better evidence for the regime boundary than PhysicsCorrect, because it is
  **steady CFD** — the exact regime where we report failure — and it succeeds. The discriminator is
  not steady-vs-transient; it is **solver-consistent operator (floor ≈ 0) vs. affordable
  surrogate-side monitor (floor dominant)**. Currently the paper's boundary is drawn as
  "transient/resolved vs. steady/under-resolved", which Lei et al. falsifies. **Redraw the boundary
  on the operator, not on the time-dependence.**
- Huang, Perdikaris — PhysicsCorrect, arXiv:2507.02227, **AAAI 2026 Oral** (verified). Confirmed
  transient/time-marching only (Navier–Stokes, wave, Kuramoto–Sivashinsky; "long-term rollouts");
  up-to-100× error reduction; <5% inference overhead. The paper's scope claim about it is correct.
- Jha — CMAME 419:116595 (2024), arXiv:2306.12047 (verified; **single author Prashant K. Jha**).
  Variational/weak form, where the residual characterises the exact solution — i.e. floor = 0 by
  construction. Correctly used as the "benign regime" anchor.

**Honest scoping required.** State the claim as: *at the operator fidelity a deployed surrogate can
afford (coarse uniform grid, approximate closure, no solver mesh), residual descent points away
from the truth.* Unscoped ("the residual is a bad correction objective") it is both attackable and
already published in spirit.

---

### Rank 2 — Candidate 1: the residual-floor theorem

**Verdict: the mathematics is NOT new. The measurement and the duality framing are.**
This is the brutally honest answer you asked for. Leg by leg:

- **(i) uniform field has exactly zero residual → truth is not a minimiser.** Known. This is the
  documented "trivial / degenerate low-residual solution" failure mode of physics-only losses;
  the literature explicitly notes that physics losses "admit low-complexity or degenerate solutions
  … that yield small residual values but fail to represent meaningful physical states", and that
  the interior residual without side conditions controls only distance to the *solution set*.
  In its bare form the paper already concedes this ("elementary … the pressure gauge above"). Good.
- **(ii)/(iii) nonzero floor at the discrete truth, with $\|e_\infty\|=\|L^+r^\star\|\le\|r^\star\|/\sigma_{\min}$.**
  **Classical.** $R_h(u^\star)\neq0$ *is the definition of the local truncation / consistency error*.
  Two specific literatures own it:
  - **Multigrid FAS $\tau$-correction (Brandt).** The relative truncation error $\tau_h^H$ is
    *precisely* the coarse-grid operator applied to the restricted fine-grid solution — i.e. the
    quantified residual floor of a coarse operator at the (finer) truth. Dual-grid $\tau$-estimation
    is standard practice for exactly this quantity. Confirmed by search; anchor citations
    (verify page numbers before use, marked UNVERIFIED as bibliographic detail):
    A. Brandt, *Multi-level adaptive solutions to boundary-value problems*, Math. Comp. 31(138):333–390, 1977;
    Trottenberg, Oosterlee & Schüller, *Multigrid*, Academic Press, 2001.
  - **Defect / deferred correction (Stetter).** The whole method is premised on the exact solution
    having a nonzero defect under a low-order operator. H. J. Stetter, *The defect correction
    principle and discretization methods*, Numer. Math. 29:425–443, 1978 (page numbers UNVERIFIED).
  - The bound $\|e\|\le\|r\|/\sigma_{\min}$ is textbook backward-error analysis — the paper already
    cites Trefethen & Bau for it, which is the correct concession.

  **Do not over-concede, though.** Classical $\tau$ is the *same equation family on nested grids* —
  a discretization-**order** gap, and it shrinks predictably under refinement. What is measured here
  is a **cross-discretization operator mismatch**: a Cartesian $128^2$ finite-difference monitor with
  an approximate closure, evaluated against a **body-fitted finite-volume RANS reference** — different
  operator, different mesh topology, different closure treatment, plus the dropped
  $\nabla\nu_t\!\cdot\!\nabla u$ term (the JSON's own `caveat` field lists all three causes honestly).
  Nobody computed this quantity before because nobody previously had a reason to monitor residuals on
  a grid *other than the solver's* — that reason only appears once a surrogate replaces the solver.
  Name it *"the relative truncation error of a surrogate-side monitor against a body-fitted
  reference"* and the $\tau$ objection converts from a scoop into a citation.
- **The neural-surrogate-specific version is also already in print — and it corroborates us.** This
  is the closest hit of the entire search and it is **not cited**:
  > **Zhang, Mallon, Luo, Thiyagalingam, Tzeferacos, Bingham, Gregori — "Data-driven modeling of
  > shock physics by physics-informed MeshGraphNets", arXiv:2602.14918 (2026-02-16), Appendix E:**
  > *"these factors prevent the PDE residual from vanishing even for the ground-truth solution"* and
  > *"the physics-informed loss cannot be formulated using the absolute PDE residual alone; instead,
  > it must be defined relative to the residual present in the ground-truth data"*.

  They state the floor **and** a remedy (subtract the ground-truth residual). **This kills the bare
  claim "we identify that the monitored residual does not vanish at the dataset truth."**

  **But file it as corroboration, not as a dent.** Their remedy — define the physics loss *relative
  to the ground-truth residual* — is an admission that the **absolute** residual is unusable as an
  objective. The closest prior art already had to work around exactly the failure we characterise.
  Moreover their workaround requires $R_h(u^\star)$, which is available at *training* time and is
  precisely what a *deployment-time* trust monitor does not have. The right sentence is: *"the
  workaround is already in print (Zhang et al. 2026, App. E); we explain why it is needed, quantify
  it on a public CFD benchmark, and show what it costs when $u^\star$ is unavailable at inference."*
  That is a stronger position than conceding priority, and it must be said by us before a reviewer
  says it.
- **(iv) local divergence cone.** Follows in two lines from (iii); not independently novel.

**What survives as genuinely ours:**
(a) the **measurement** — $\|r^\star\|$ mean 0.192 / median 0.133 on 200/200 AirfRANS cases, with
the trained prediction *below* the truth's floor on **160/200**;
(b) the **detector/fixer duality from a single decomposition** — far-field $\|Le\|\gg\|r^\star\|$
gives the two-sided ranking proxy; near-field floor dominance gives the objective failure. I found
no prior work deriving both verdicts from one expansion;
(c) the **kernel as an audit blind spot** (constant-pressure gauge + residual-blind $\nu_t$) framed
as *what the certificate cannot certify*. Nearest analogue found: "a smoothing forward operator can
hide structured model error below a residual-magnitude threshold" (*Sequential Structure-Sensitive
Residual Diagnostics for PDE Inverse Problems*, arXiv:2607.02101) — inverse problems, not surrogate
auditing;
(d) the **closed-form $\lambda$-sweep** proving no no-slip weight rescues the objective. This is a
genuinely good piece of red-teaming and it is currently invisible.

**Recommendation:** demote the theorem from "contribution" to "explanation of the measurement", and
rewrite the "Relation to prior work" paragraph to concede $\tau$-correction, defect correction, and
Zhang et al. 2026 by name. Claim (a)+(b)+(d), drop (a)-as-currently-worded ("the *quantified*
operator-specific floor"), which is $\tau$ under a new name.

---

### Rank 3 — Candidate 3: the trust layer (**least novel — concede it**)

Your suspicion is correct, and the margin is not close. Concede rather than defend.

- **Gopakumar, Gray, Zanisi, Nunn, Giles, Kusner, Pamela, Deisenroth — "Calibrated Physics-Informed
  Uncertainty Quantification", arXiv:2502.04406 (2025-02-06, rev 2025-06-10).** Verified abstract:
  *"utilises convolutional layers as finite-difference stencils and leverages physics residual errors
  as nonconformity scores, enabling data-free UQ with marginal and joint coverage guarantees"*.
  They own residual-as-nonconformity-score **and** the finite-difference-stencil monitor. Same
  mechanism, same motivation.
- **Roy, Nayak, Goswami — ANCHOR, arXiv:2512.19643 (2025-12-22, rev 2026-06-14).** Verified:
  *"the EMA-based estimator correlates strongly with the true relative $L_2$ error, enabling
  data-free, instance-aware error control during inference"* and it triggers a classical solver.
  They own residual↔error correlation as a deployed, data-free error signal **and** the
  trust-gated-fallback pattern. Time-marching, but the pattern is theirs.
- Ma, Azizzadenesheli, Anandkumar — arXiv:2402.01960 (2024-02-02) — conformal for operator learning
  (Darcy + 3-D car surface pressure). *Venue "TMLR 2024" in refs.bib is UNVERIFIED from the arXiv
  page; check before submission.*
- Garg & Chakraborty, JCP 534:114012 (2025); Yu, Ho, Wang, JCP 561:114979 (2026);
  Jia, Xia, Vdovin, Jia, Sebben, Yang, arXiv:2607.17297 (2026-07-19, DrivAerML + GeoTransolver,
  residual-**scale**-normalised conformal — the paper's characterisation of it as data-driven
  nonconformity is accurate).

**What is left to us:** the multi-backbone / multi-dataset *audit* of how strong the correlation is
(ρ 0.40–0.85, reported with its spread), the honest case-level-vs-per-cell scoping, and calibration
on the *corrected* field. That is evaluation value, not methodological novelty — and an originality
screen is precisely the filter that rejects evaluation value dressed as method. **Leading with this
is what got the paper desk-rejected twice.**

---

## 2. The single strongest claim, in one sentence

> On a standard public steady-RANS benchmark, the physics residual that a deployed neural CFD
> surrogate can afford to monitor **does not vanish at the ground truth** ($\|r^\star\|$ = 0.192
> mean on 200/200 AirfRANS cases, against an exact zero for the physically wrong uniform field), so
> its minimiser is displaced from the truth by $\|L^{+}r^\star\|$ and descending it measurably
> *raises* the field error in our sweep — the residual can **rank** predictions but cannot be
> **minimised**, and we locate the boundary separating this regime from the solver-consistent one
> where residual descent succeeds.

**Wording discipline.** Do *not* write "descending it **provably** moves the field away from the
truth." Theorem leg (iv) proves an *open cone* of error directions on which a residual-reducing step
increases error, to first order, under gradient flow — not that descent always diverges. Correct
form: *"the residual's minimiser is displaced from the truth by $\|L^{+}r^\star\|$, and descent
measurably raises the error in our sweep."* Getting caught overstating in the headline sentence is
the one credibility hit this project cannot absorb. (And per §0a, "measurably" is currently n=1 —
seed it, or hedge it as an illustration.)

Why this survives an originality screen: it asserts a *measured fact about a benchmark*, not a new
method. It is falsifiable, reproducible from a committed script, and it constrains what a large and
growing literature is allowed to claim. Precedent that a top venue publishes exactly this genre:
**McGreivy & Hakim, "Weak baselines and reporting biases lead to overoptimism in machine learning
for fluid-related partial differential equations", *Nature Machine Intelligence* 6:1256–1269 (2024),
doi:10.1038/s42256-024-00897-5** (verified) — a pure critique/negative-result paper in ML-for-CFD.
**Cite it.** It legitimises the genre in the first paragraph and pre-empts "where is the new method?"

---

## 3. Prior art that would KILL the claim if a reviewer found it

Searched for specifically. Ranked by danger.

1. **Zhang et al. 2026 (arXiv:2602.14918), Appendix E — DANGEROUS.** States the floor and the
   remedy. Kills any phrasing of the form "we identify/discover that the residual does not vanish at
   the truth". Does **not** kill: the 160/200 inversion, the correction-path divergence, the
   detector/fixer duality, or the deployment-time consequence (their fix needs the ground-truth
   residual, which a monitor does not have). **Cite it in the Introduction, not the appendix.**
2. **Lei et al. 2026 (arXiv:2608.04400) — DANGEROUS as a rebuttal, VALUABLE as evidence.** A reviewer
   says: "steady CFD residual correction works; you just chose a bad operator." Correct answer, in
   the paper: yes — and it costs a solver-consistent operator and a body-fitted mesh, i.e. the thing
   the surrogate exists to avoid; their result and ours are the two sides of the boundary we draw.
   Failing to cite this is the single largest reviewer-discovery risk in the paper.
3. **Multigrid $\tau$-correction / defect correction — DANGEROUS to the theorem only.** Any numerical
   analyst on a JCP/CMAME board recognises $R_h(u^\star)\neq0$ as $\tau$ and will read an unscoped
   "residual-floor theorem" as a rediscovery. Concede in one sentence and the danger evaporates.
4. **Jiang et al. 2026 ENS (arXiv:2606.27354) — MODERATE.** Concurrent, opposite empirical verdict on
   residual-as-input. Report the disagreement explicitly; it is a strength if surfaced, a wound if found.
5. **Astral (arXiv:2406.02645) / Luo & Zhou (arXiv:2310.18201) / Wang et al. NeurIPS 2022
   (arXiv:2206.02016) — MODERATE.** They collectively own "residual ≠ error" as an insight. Cite all
   three in one sentence and pivot the claim to the *measured inversion and divergence*, which none
   of them report.
6. **Gopakumar 2025 / ANCHOR 2025 — FATAL to Candidate 3 only.** Already handled by demoting it.

Searched and found **nothing** for: a prior report that a neural surrogate's PDE residual is *lower
than the reference solution's* on a CFD benchmark; a prior report of residual rising while error
falls along a correction path; a prior kernel-of-the-monitor analysis framed as an audit blind spot.
Nearest adjacent: Koehler & Thuerey, *Neural Emulator Superiority: When Machine Learning for PDEs
Surpasses its Training Data*, arXiv:2510.23111 (2025-10-27, v2 2026-01-14) — emulators beating their
training data against a higher-fidelity reference; accuracy, not residual. Worth a sentence.

---

## 4. Candidate titles

Shape rule: **[measured fact] + [regime]**. No coined system name. Reject anything whose first six
words could describe a new method.

1. **"Lower residual than the truth: why physics-residual descent misleads under-resolved steady-RANS surrogates"**
2. **"The residual a neural CFD surrogate can afford to monitor ranks its errors but cannot be minimised"**
3. **"When the ground truth fails its own physics check: the residual floor of neural CFD audits on AirfRANS"**
4. **"Rank it, don't descend it: measuring what a steady-RANS residual can and cannot certify about a neural surrogate"**
5. **"A measured boundary for residual-driven correction in steady CFD: consistent operators succeed, affordable monitors fail"**

(1) and (3) lead hardest with the surprising fact; (5) is the safest for a numerics readership
because it names the boundary rather than only the negative. Keep `neuroforge-cfd` in the
code-availability section only.

---

## 5. `tab:positioning` — it HURTS. Replace it.

Your read is right. A 7-column checkmark grid whose bottom row is all bullets is the canonical
incremental-paper signature; the sentence it supports ("none combines a validated trust signal with
distribution-free calibration, measured selective prediction, a head-on test of the objective role,
and a theory of the limits") is a claim about **coverage**, not about **knowledge**, and it invites
verbatim the rejection received: *"this is X + Y + Z."* Two of the seven columns (≥3 backbones,
≥2 datasets) are not even scientific properties — they are effort.

**Replace with a two-row regime table that asserts a boundary.** The discriminating column is the
floor-to-signal ratio, and the caption of the current table already contains the idea:

| Regime | Monitored operator | $\|r^\star\|$ at truth | Residual descent | Evidence |
|---|---|---|---|---|
| Solver-consistent | solver's own discrete operator, body-fitted mesh | $\approx 0$ (machine/solve tolerance) | **succeeds** — residual $\downarrow$ 7 orders, error $\downarrow$ | Lei et al. 2026 (steady CFD); Huang & Perdikaris 2026 (transient); Jha 2024 (variational) |
| Affordable surrogate-side monitor | coarse uniform grid, approximate closure, no mesh | **0.192 mean, 0.133 median** (AirfRANS, 200/200) | **fails** — residual $\uparrow$ 0.11→0.62 while error $\downarrow$ 3.92→2.29 | this work |

Same underlying data. Asserts a scientific boundary rather than tallying features, and it puts our
headline number *inside the positioning*, where an editor reading only the intro will see it.

Add a **cost column**: "affordable" is currently an assertion. `sec:cost` times every stage of our
pipeline, and Lei et al. report 15.5× over CFD at generation level. One cost number on each row turns
the boundary from a label into a measurement, and it is the answer to the strongest rebuttal
("just use a consistent operator") — yes, at solver cost and with a body-fitted mesh, i.e. the thing
the surrogate exists to avoid.

---

## 6. Citation audit of the entries this report relies on

| Key | Status |
|---|---|
| `huang2025physicscorrect` | **Verified.** Huang & Perdikaris, arXiv:2507.02227, 2025-07-03 (rev 2025-12-25), AAAI 2026 Oral. Transient only — the paper's scope claim is correct. |
| `learned2023residualcorrection` | **Verified content, bad key.** Prashant K. Jha (sole author), CMAME 419:116595 (2024), arXiv:2306.12047. Rename to `jha2024corrector`; the current key reads as a placeholder and the year in the key (2023) disagrees with the year field (2024). Body calls it "the variational corrector" — accurate. |
| `gopakumar2025pre` | **Verified.** 8 authors as listed; arXiv:2502.04406. Abstract confirms residual-as-nonconformity-score. The body's phrase "in *residual* space, with the transfer to solution space left open as a set-propagation problem" is a *reasonable inference*, not a quote — the abstract says "coverage guarantees across prediction domains". Soften to "their nonconformity score is the residual itself" or quote precisely. |
| `roy2025anchor` | **Verified.** Roy, Nayak, Goswami; arXiv:2512.19643. |
| `mukherjee2026certification` | **Verified.** Mukherjee, Fitzsimmons, Del Rey Fernández, Liu; arXiv:2603.19165 (rev 2026-09-01). Compactness requirement confirmed. Preprint — mark as such. |
| `jia2026multigranularity` | **Verified.** 6 authors as listed; arXiv:2607.17297 (2026-07-19); DrivAerML + GeoTransolver. Preprint. |
| `hillebrecht2025posteriori` | **Verified.** IEEE TNNLS 36(1):1583–1593, 2025 (arXiv:2210.03426 as "Certified machine learning: …"). |
| `beckerrannacher2001dwr` | **Verified** (Acta Numerica 10:1–102, 2001). |
| `ma2024uqno` | Title/authors/arXiv:2402.01960 **verified**; **venue "TMLR 2024" UNVERIFIED** from primary source. Check. |
| `garg2025dfuq`, `yu2026conformalpinn` | DOIs present and well-formed; not independently re-verified this session — **UNVERIFIED**. |
| `song2026structureaware`, `rigotti2026gist`, `scherz2026evaluation` | Not verified this session — **UNVERIFIED**. All are 2026 arXiv preprints; label them as preprints in the text, not as established results. |

## 7. Must-add citations (none currently in `refs.bib`)

1. Zhang, Mallon, Luo, Thiyagalingam, Tzeferacos, Bingham, Gregori. *Data-driven modeling of shock physics by physics-informed MeshGraphNets.* arXiv:2602.14918 (2026). **[closest prior art to the floor]**
2. Lei, Tang, Zhang, Chen. *Reliable and efficient steady CFD from surrogate predictions through Newton-Krylov correction.* arXiv:2608.04400 (2026). **[the other side of the boundary, in steady CFD]**
3. Fanaskov, Yu, Rudikov, Oseledets. *Astral: training physics-informed neural networks with error majorants.* arXiv:2406.02645 (2024/2026).
4. Jiang, Wang, Chen, Kwak, Kim, Bell, Park. *Error-Conditioned Neural Solvers.* arXiv:2606.27354 (2026). **[contradicts our W1 ablation — engage it]**
5. Luo, Zhou. *On Residual Minimization for PDEs: Failure of PINN, Modified Equation, and Implicit Bias.* arXiv:2310.18201 (2023).
6. Wang, Li, He, Wang. *Is $L^2$ Physics-Informed Loss Always Suitable for Training PINNs?* NeurIPS 2022, arXiv:2206.02016.
7. McGreivy, Hakim. *Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related PDEs.* Nature Machine Intelligence 6:1256–1269 (2024). **[genre precedent — cite in the intro]**
8. Brandt (1977) and/or Trottenberg–Oosterlee–Schüller (2001) for the FAS $\tau$-correction; Stetter (1978) for defect correction. **[concede the classical floor]**
9. Optional: Koehler, Thuerey. *Neural Emulator Superiority.* arXiv:2510.23111 (2025).

## 8. Ordered action list

**Blocking experiments (do these before rewriting the framing):**

0a. Recompute the residual-floor inversion on the **Transolver** backbone (§0a item 1). This decides
    whether the headline is "the prediction beats the truth" or the safer, model-independent "the
    truth fails its own physics check by 0.192 while a wrong uniform field scores exactly 0."
0b. Seed the `n_iters` sweep (§0a item 2). Cheapest high-value run in the project; without it the
    descent falsification is n=1 and unseeded, on the backbone the paper itself calls weak.

**Framing edits:**

1. **Move the floor measurement into the abstract's first two sentences** and make the theorem its
   *explanation*, not a contribution. Add the figure the data already supports: iteration sweep with
   a horizontal line at $\|r^\star\|=0.192$ and error on the twin axis — the prediction starts
   *below* the truth's residual and error falls as the residual rises *through* the floor. Both
   series are logged (`results/sensitivity/iters.csv`,
   `results/certificates/residual_floor_realdata.json`). Caption it with the seed count, honestly.
2. **Redraw the regime boundary on the operator, not on steady-vs-transient.** Lei et al. 2026
   falsifies the current phrasing (steady CFD where residual descent *succeeds*). Bake the scoping
   into the theorem statement itself — "for a surrogate-side monitor with $\|r^\star\|\gtrsim\|Le\|$"
   — not into an "Assumptions, stated plainly" paragraph 100 lines later.
3. **Delete `tab:positioning`; insert the two-row regime table with a cost column.** Demote the trust
   layer to a supporting result and concede `gopakumar2025pre` owns residual-as-nonconformity-score
   and `roy2025anchor` owns residual↔error correlation as a deployed signal, both by name.
4. **Concede the classical floor in one sentence** ($\tau$-correction, defect correction) and
   immediately distinguish it as a *cross-discretization* mismatch (§1 Rank 2). One sentence of
   concession buys the whole objection.
5. **Retitle** per §4; keep `neuroforge-cfd` in code-availability only.
