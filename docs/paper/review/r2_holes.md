# Reviewer 2 report — "NeuroForge: a self-auditing, geometry-native neural CFD engine"

Adversarial pre-submission review. Every objection below is anchored to a line of
`docs/paper/body.tex`, `docs/paper/abstract.tex`, `docs/paper/sections/residual_floor_theorem.tex`,
a committed file under `results/`, or a line of `src/neuroforge/`. Two of the eight attack
surfaces I was asked to press are **factually wrong** and I retract them explicitly in §E —
please read that section, because a real reviewer will not make those two attacks and time
spent defending against them is wasted.

---

## A. The single worst objection

> **The paper's headline is a two-way dissociation, but the negative half of that dissociation
> is never actually tested. No experiment in this manuscript minimises the physics residual.**

The abstract (`abstract.tex:10-13`) calls the two-way dissociation "our central finding"; it is
contribution 3 (`body.tex:128-146`), it owns §5.5 (`sec:iters`), it is the paper's self-declared
"money figure" (`fig:fig4`, `body.tex:1478`), and it closes the Conclusion (`body.tex:1652-1654`).
The entire empirical support for it is `tab:iters` (`body.tex:1082-1100`), sourced from
`results/sensitivity/iters.csv` / `iters.json`.

That file records what was actually swept:

```json
"swept_knob": "DEQCorrector.max_iter (internal fixed-point cap)",
"engine_maxiters_inert_on_deq": true,
"n_eval": 80
```

and `scripts/run_sensitivity.py:10-25` states it in the authors' own words: the engine-level
`max_iters`, `gate_by_trust` and the backtracking acceptance test are **structural no-ops** on a
DEQ corrector; the only live knob is the DEQ's internal fixed-point cap. The DEQ corrector is
trained by supervised regression toward ground truth (`body.tex:587-590`) and is applied
**without** the acceptance test.

So the sweep varies the number of fixed-point iterations of a corrector that (i) never sees the
residual as an objective, (ii) has no residual-based stopping rule, and (iii) bypasses the only
residual-driven mechanism in the codebase. The conclusion drawn from it —
"reducing it does not reduce field error" (`abstract.tex:17`), "the very quantity one would
minimise to 'fix' the field grows while the field gets better" (`body.tex:1105-1106`) — does not
follow, because nothing in the experiment minimises that quantity.

I will steelman the defensible reading: *if you used the residual to select an iterate along this
path, you would select iterate 0 (residual 0.113, `mse_u` 3.92) over iterate 3 (residual 0.542,
`mse_u` 2.29).* That is a real and interesting statement — but it is a statement about
**model/iterate selection**, not about the residual as a **correction objective**, and it is not
what the abstract, contribution 3, or `fig:fig4` claim.

The experiment the paper owes the reader is trivial and the machinery is already shipped:
`physics_residual_torch` is a differentiable residual (`body.tex:630-632`). Run
`y_{k+1} = y_k - η ∇_y ½‖R_h(y)‖²` from the backbone prediction on the AirfRANS test split, and
plot residual vs. field error. That is an afternoon on one GPU and it is the only thing that can
license the sentence in the abstract. Until it exists, the paper's central negative is an
inference from a theorem plus a sweep that does not test it.

**Why this is fatal rather than fixable by rewriting.** After two desk-rejections on originality,
the paper's positioning table (`body.tex:343-351`) concedes that `gopakumar2025pre` already holds
*Trust*, *Calib.*, *≥3 backbones* and *≥2 datasets*. The entire remaining originality delta is
three columns: **Object.**, **Select.**, **Theory.** This objection destroys *Object.*
Objection B1 destroys *Select.* Objection B3 wounds *Theory.* A third editor will reach the same
arithmetic.

---

## B. Ranked weaknesses

### B1 — FATAL. Your own results file shows a physics-free baseline beats the physics residual at the task the paper claims for it

`results/selective/selective_prediction.json`, arm `ensemble_mean`, n=200:

| score | Spearman vs error | AUROC (worst decile) | oracle recovery @10% reject |
|---|---|---|---|
| physics residual | **0.6103** | **0.8706** | ≈67% |
| ensemble `sigma_vel` | **0.6543** | **0.8939** | ≈72% |
| rank-fused | 0.7028 | 0.9053 | ≈91% |

(the 67%/72% figures follow from each score's `risk_coverage["10"]` block: retained mean rel-L2
0.003983 → 0.003626 residual, → 0.003601 σ, oracle 0.003450.)

A standard deep-ensemble spread — no physics, no PDE operator, no residual — is a **strictly
better** trust signal than the physics residual on both the paper's ranking metric and its
decision metric. The paper reports all three AUROCs in a single sentence (`body.tex:1246-1249`)
and never draws the conclusion. Worse, the framing selectively omits it:

- `abstract.tex:15-16` attributes "flags the worst-decile cases with AUROC ≈0.9" to the residual.
  The residual is 0.871; ≈0.9 requires the ensemble.
- `body.tex:52-56` reports "the residual alone still recovers ≈67%" — and does not report that
  σ alone recovers ≈72%, i.e. **more**.
- Contribution 1 (`body.tex:100-104`) headlines the fused 0.905.

The fusion gain over σ alone is +0.011 AUROC on n=200 with 20 positives. I do not believe a
bootstrap CI on ΔAUROC(fused − σ) excludes zero, and the paper does not report one although it
bootstraps Spearman elsewhere (`results/control/bootstrap_spearman_ci.json`).

This is the objection that most directly guts the paper's reason to exist: if the physics residual
is dominated by the UQ baseline it is supposed to complement, the *Select.* column of
`tab:positioning` is not earned and the "physics-residual trust signal" framing is decoration on
a deep ensemble.

**Resolution:** report ΔAUROC and Δoracle-recovery with paired bootstrap CIs; if fused ≉ σ, say so
and reframe the residual as *the free component* (it costs 1.13 ms vs. 10.1 s — `tab:cost`) rather
than as *the signal*. That reframing is honest and still publishable; the current framing is not.

### B2 — FATAL/MAJOR. The detector's strength is an inverse index of model quality, which inverts the deployment argument

Four data points, all the authors':

| backbone | accuracy (`mse_u`) | resid↔err ρ |
|---|---|---|
| Transolver (best) | 0.133 (`tab:v2`) | 0.611 |
| weak grid FNO | 3.479 (`tab:indist`) | 0.397 |
| weak grid FNO + DEQ (the arm that inflates `mse_v` **+129%**, `mse_p` **+57%**) | — | **0.827** |
| MeshGraphNet | **13.55**, `cd_rel_err_mean` **1.01** (`results/mgn/mgn_results.json`) | **0.851** |

ρ is monotone in how *bad* the model is. The paper concedes this once, in a parenthesis, for MGN
only (`body.tex:87-89`, `body.tex:839-841`). The corollary it never states: on the SOTA backbone
where the corrector *improves* the field, ρ does not move (0.611 → 0.612, `tab:v2`); the celebrated
"the corrector doubles the trust signal" (`body.tex:951-956`) and the "regime-invariant trust
signal — the single strongest result" (`body.tex:1004-1011`, aoa 0.314 → 0.746) both occur **only
on the arm that damages the field**. An obvious alternative explanation — the corrector injects a
structured, residual-visible artefact that raises residual and error together, manufacturing rank
correlation — is never excluded.

The deployment consequence is severe: the signal is weakest exactly on models good enough to
deploy, which is the regime the paper's own motivation targets. "Backbone-robust" (`abstract.tex:11`,
contribution 1, Conclusion) is doing work the range 0.40–0.85 cannot support: a 2× spread in the
headline statistic across three models is *positivity*, not robustness, and the paper's own gloss
("the detector *works* on every architecture, not that its strength is uniform", `body.tex:838-839`)
concedes that the word is being stretched.

Compounding it: the "same verifier across three backbones" is not the same operator. ν_t supplies
≈84% of ν_eff (`body.tex:468`) and is learned at R²=0.996 on the grid backbone vs **0.67–0.70** on
MGN (`body.tex:1629-1635`). The MGN residual is therefore a materially different (and much worse)
momentum operator than the grid one, which the "backbone-robust monitor" claim silently elides.

**Undisclosed confound on the third backbone.** `results/control/mgn_density_control.json`:
MGN was trained on 16,384 subsampled points and the headline evaluated on the full ~180k cloud;
the control reports `"velocity_ratio": 2.437`, `"verdict": "DENSITY-DRIVEN"` — error is ~2.4×
worse at evaluation density than at training density. I grepped `docs/paper/` for
`density|16,?384|16k|subsampl`: **the manuscript never mentions this.** The control is n=4 cases,
so I am not claiming the published number is wrong — I am claiming that a committed artifact in
your own repo says the third backbone's error (and hence, per your own parenthesis, its ρ) is
inflated by a train/test density mismatch, and the paper is silent. Reviewers who look at the
repo — and the artifact manifest invites them to — will treat that silence badly.

### B3 — MAJOR (co-fatal for the *Theory* column). The residual floor is unattributed, and both possible attributions damage the theorem

The theorem's hypothesis (H2) is that `‖R_h(u*)‖ > 0`. The empirical support
(`results/certificates/residual_floor_realdata.json`) gives `norm_truth_mean = 0.192` — of which
`norm_truth_continuity_mean = 0.1226`, i.e. **the "ground truth" field violates discrete continuity
at RMS ≈12% of u∞/L** (that file's `non_dim_scaling` field documents the scaling as
`continuity / (u_inf/L)`). Nothing in the paper attributes that 0.12, and it has two possible
sources, both damaging:

- **It is your rasteriser.** `rasterize_point_cloud(..., method="linear")` from ~180k unstructured
  points onto a 128² Cartesian crop is not conservative and is unconstrained near the wall. Then
  (H2) is an artifact of your preprocessing, not a property of steady RANS, and the paper's central
  negative is scoped to its own pipeline rather than to "the deployment-realistic regime"
  (`body.tex:234-235`).
- **It is the reference data.** Then (H2) is a property of the AirfRANS release, and the
  manuscript's stated justification for (H2) — that "the 128² grid under-resolves the boundary
  layer and drops the ∇ν_t·∇u term" (`residual_floor_theorem.tex:148-153`) — is the wrong
  explanation for its own hypothesis.

A refinement study (128 → 256 → 512, or a body-fitted O-grid) distinguishes these in one run and is
the minimum needed to keep the theorem's empirical section. The artifact file itself concedes the
multi-causality in its `caveat` field, listing coarse-grid discretisation as cause (3); the
manuscript compresses that concession into a subordinate clause.

Second, and sharper: **the "no boundary weight can rescue it" argument is circular.**
`residual_floor_theorem.tex:131-146` closes the obvious objection in closed form using
`truth_bc2 = 0.0073 > uniform_bc2 = 0.0053` (`results/control/bc_weight_sweep.json`, **n=10 cases,
one checkpoint**). But `src/neuroforge/physics/residuals.py:203-213` shows `bc_violation` is
no-slip **plus** a far-field term penalising deviation from `(u∞,v∞)` on the outer one-cell ring —
of a crop that is `_CROP = (-1.0, 2.0, -1.5, 1.5)` (`airfrans_loader.py:64`), i.e. **1–1.5 chords
from a lifting airfoil**, where the true flow carries O(few %) induced velocity from circulation.
The uniform freestream field satisfies that term *exactly, by construction*. The truth cannot.
That is a large part of why `uniform_bc2 < truth_bc2`, and it is a mis-specified boundary
functional, not a fact about RANS. Note also that `residual_floor_theorem.tex:122-123` describes
r_bc as "the proximity-weighted no-slip penalty" — dropping the far-field half at the exact point
where the distinction decides the argument. (`body.tex:470-473` describes both terms correctly, so
this is an internal inconsistency, not a hidden term.)

Third, legs (i)–(ii) are textbook: a PDE residual with no boundary constraint has the uniform field
as an exact global minimiser. The paper says so itself ("in its bare form, elementary",
`residual_floor_theorem.tex:162-163`). What remains as claimed novelty is (a) the quantified floor
`‖L⁺r*‖` — which is **never evaluated**: no `σ_min`, no `‖L⁺r*‖`, no measurement of `‖Le‖` vs `‖r*‖`
appears anywhere in `results/` — and (b) the detector/fixer duality, which is a re-reading of
`R = r* + Le`. A JCP referee will call leg (a) an unexercised bound.

Fourth, the floor story does not fit the data it is invoked to explain. Truth floor 0.192; backbone
prediction 0.113 (below the truth in 160/200 cases); the correction sweep drives the residual to
**0.62** — 3× past the truth's floor. "The corrector moves toward truth so the residual rises
toward the floor" explains 0.113 → 0.19, not → 0.62. The unexplained remainder is consistent with
the corrector injecting high-frequency content — which is exactly what `tab:indist` documents it
doing on this backbone.

### B4 — MAJOR. `tab:iters` reports one of three volume channels, and the omitted two are known to regress under this corrector

`tab:iters` reports `mse_u` and `surface_mse_p` only. `run_sensitivity.py` computes only those two.
The same corrector family on the same weak grid backbone, in `tab:indist`, inflates `mse_v`
0.385 → 0.880 (**+129%**, 3/3 seeds) and `mse_p` 2445 → 3833 (**+57%**, 3/3).

I am not asserting the sweep's `mse_v`/`mse_p` rise — different run, different checkpoint. I am
asserting that the paper's "money figure" claims "field error falls" on the strength of **one of
three** volume channels, while the paper elsewhere documents that this corrector trades exactly the
two channels it did not plot. Add `mse_v`, `mse_p` and a total-field error to
`results/sensitivity/iters.csv`. If aggregate field error rises with iterations, `fig:fig4` inverts
and the paper has no dissociation at all — it has a corrector that degrades the field and a
residual that correctly says so, which is a *different paper* (and, ironically, a more coherent one).

Also: `tab:iters` is n=80 cases, **one checkpoint, unseeded** (the caption says so;
`body.tex:1639-1641` repeats it), on the **weak grid backbone only**. The dissociation is never
demonstrated on the SOTA Transolver backbone that the rest of §5.2 is built on. The paper's
single most-emphasised finding rests on n=1 in seeds and n=1 in backbones.

### B5 — MAJOR. The abstract's "where" is contradicted by the paper's own §3.3 and §5.4

`abstract.tex:11-12`: "it tells you **where** the prediction is wrong". Repeated at `body.tex:26-27`,
36-38, 227, 310, 366-367, 813-814, 1608-1609, 1648.

What the evidence says:

- `body.tex:493-514`: "Our quantitative trust result is **case-level**… the residual is a
  *case-level* ranker and a *moderate region-level* locator… **not a per-cell error rank**."
- `results/control/percell_localization.json`: raw per-cell ρ = **0.166 ± 0.155** (n=200, deployed
  ensemble); 16×16 patch pooling → **0.323 ± 0.232**; patch AUROC 0.68–0.72. At those standard
  deviations a substantial minority of cases have zero or negative spatial correlation.
- `body.tex:1066-1074` (cylinder control): the residual's spatial structure "**cannot be read as
  'where the prediction failed'**."

The abstract therefore asserts the precise claim the body refutes. Correct slogan: **which case**,
not *where* and not *how*.

Related factual mislabel: `body.tex:495-498` reports per-cell "0.22±0.06 (Spearman) though 0.60
(Pearson), **measured on the deployed corrected field**", citing
`results/control/percell_residual_error.json`. That file's own meta says
`"checkpoint": "checkpoints/certificates_deq.pt"`, `"n_cases": 15` — the *old weak dropout-FNO*, not
the deployed field; `percell_localization.json:36-41` states this explicitly ("published 0.22 was an
OLD FNO checkpoint at n=15"). The deployed field's value is 0.166±0.155. The paper quotes the more
favourable, superseded number under the deployed field's name.

### B6 — MAJOR. Conformal coverage is a tautology check reported as a result, and the guarantee is not the guarantee a user needs

`src/neuroforge/physics/calibration.py:119-134` + `:78-80`: nonconformity scores `|ŷ−y|/σ` are
computed **per fluid cell**, concatenated across all calibration cases, and one scalar
`q = quantile(scores, ceil((n+1)(1−α))/n)` is taken over the pooled array — n ≈ 100 cases × ~16k
cells ≈ 1.6M **strongly dependent** cells.

Two consequences the paper does not confront:

1. The finite-sample correction is computed at n≈1.6M while the effective sample size is ≈100
   (the paper half-concedes this by quoting "the ±0.03 binomial spread at n_test = 100",
   `body.tex:862`). The `Limitations` bullet (`body.tex:1586-1591`) notes the pooling but treats it
   as a sample-size caveat rather than as what it is.
2. `prop:coverage` as instantiated is **marginal over a randomly drawn cell of a randomly drawn
   case**. It says nothing about the field a given user receives. The paper's product claim is
   per-case — "a *calibrated error band* with a distribution-free guarantee" as component (ii) of
   *self-auditing* (`body.tex:46-47`), and a trust layer sold for per-case accept/reject. Those are
   different objects. A field-level (max-over-cells, or two-level per-function) construction is the
   one that matches the claim.

Separately, **coverage is not evidence.** Split conformal attains marginal coverage by construction
under exchangeability. Contribution 1 (`body.tex:94-96`) headlines "target coverage on the SOTA
backbone (0.902±0.008 at the 0.90 target)" — that arm has `q ≈ 1.6×10⁹` and ECE ≈ 0.31
(`tab:uq`), i.e. a near-constant band around a near-deterministic model. §5.2 says so honestly
("Coverage was never the difficulty — **adaptivity** was", `body.tex:850-853`) and then the
abstract sells the coverage anyway (`abstract.tex:16`). The genuinely informative number is
ECE 0.074 with the deep ensemble; lead with that and drop the coverage-as-achievement framing.

### B7 — MAJOR. The acceptance gate is a line search, and the control that would separate the two is already sitting in your results file

The gate result (contribution 3, `body.tex:133-137`; §5.5; Conclusion) is: 99.8% admitted, 89.3%
improve error, median −5.8%. Verified against `results/control/acceptance_gate.json`
(`backbone_seed0/1/2`: accept 1.0/1.0/0.995 → 0.998; improve 0.935/0.955/0.789 → 0.893). Good.

But on that deployed arm `median_residual_ratio_full_step` = 0.995 / 0.99987 / 1.003 — the full
step barely moves the residual, so the gate is near-inert there — and the median admitted step
fraction is 0.5–0.75. The mechanism being credited is **damping**, and the residual is only one
possible line-search criterion. The decisive control — *fixed half-step, no gate* — requires **zero
forward passes**, because `per_case[*].trials` already logs `residual_norm` and `rel_l2` at step
1.0 and 0.5 for every case. If a blind 0.5× step matches the gate, the residual contributes nothing
here either — precisely the W1 result one level up. You have the data and did not run the control.

Two smaller points on the same result: (a) "measured over **1200** gated steps it admits a step on
**99.8%**" (`body.tex:1115-1121`) juxtaposes the full-N framing with a 600-step sub-arm figure; over
all 1200 the file reports `overall_accept_rate 0.9925` and `frac_accepted_step_improves_error
0.8187`. You do report the ensemble path separately, so this is framing, not concealment — but it
reads as N-inflation. (b) On the ensemble path the gate **rejected** steps of which 60% (seed0) and
33% (seed2) would have *improved* error (`frac_rejected_that_improve_error`). A gate that rejects
error-reducing steps deserves a sentence.

### B8 — MAJOR. The system that is certified is not the system that is measured

`results/uq_ensemble/w2_conformal_corrected.json` `meta.corrector_choice`: DEQ correctors
`trained_on: "individual v2_transolver backbone (seed-matched)"` are applied to the **ensemble
mean** field, with σ from the 5 members. The file's own `reasoning` field argues this is
acceptable because it is "the SAME TYPE of Transolver-on-AirfRANS field" — an assumption, applied
to a field whose error distribution is ~40% tighter than the one the corrector was trained on.

Meanwhile `body.tex:872-878` concedes the ensemble mean (0.077/0.058/440) is **more accurate** than
the DEQ-corrected single backbone (0.116/0.079/471). So:

- the accuracy headline (−8/−21/−25%) is for backbone+DEQ, single member;
- the shipped certificate is for ensemble-mean+DEQ;
- **no table in the paper reports the accuracy of ensemble-mean+DEQ**, the configuration actually
  being certified.

Add that row, or state plainly which single configuration is "NeuroForge" and report everything on it.

### B9 — MAJOR. Selective prediction is demonstrated in a regime where nothing needs triaging, and never on the quantity engineers use

At 10% rejection the fleet mean rel-L2 goes 0.00398 → 0.00363 (residual) or 0.00350 (fused). The
worst-decile threshold is **0.68% rel-L2**. The paper is triaging predictions that are already
accurate to sub-1%, for an absolute improvement of ~5×10⁻⁴ in relative L2. "Near-oracle triage at
deployment-relevant rejection budgets" (`body.tex:1258-1259`) is true and operationally empty.

The risk functional is never the engineering quantity. The paper concedes drag is *not* recovered:
`ρ_D = 0.84` against official labels, ~11× magnitude bias with the near-field integrator, and
**≈2.7× median error even with the repaired control-volume integrator on predicted fields**
(`body.tex:766-796`). Risk–coverage on |ΔC_d| (and |ΔC_l|) is the experiment that would make the
trust layer matter, and the per-case force errors already exist in
`results/control/force_vs_official*.json`. If the residual cannot triage drag error, the audit does
not certify the quantity the application needs — and that connects directly to B10.

### B10 — MAJOR (must be conceded, but stated too gently). The grid cannot represent the physics the residual claims to monitor

`body.tex:1572-1578` concedes it. It concedes too little. With `_CROP = (-1,2,-1.5,1.5)` over 128
cells, Δ = 3/128 ≈ 0.0234 c. At Re ≈ 2.0×10⁶ (`residual_floor_realdata.json:53`), a turbulent
δ ≈ 0.37·x·Re_x^{−1/5} ≈ 0.012 c at mid-chord: **the entire boundary layer sits inside half a
cell.** Taking c_f ≈ 3×10⁻³, u_τ ≈ 0.039 U∞, the first cell centre is at **y⁺ ≈ 1.8×10³** — outside
the boundary layer entirely. (Constants stated so you can check them; the conclusion is not
sensitive to a factor of two in c_f.)

The momentum residual's viscous term `ν_eff ∇²u` is therefore evaluated on a field with no
represented viscous layer, and the wall-adjacent ring is additionally zeroed
(`residuals.py:297-319`; 0.51% of fluid cells, per `percell_localization.json:25`). The audit is
structurally blind where the error and all the skin friction live, which is consistent with the
paper's own drag failure (B9). The honest framing is not "wall quantities are approximate" — it is
**the monitored operator is a bulk-flow consistency check on an interpolated field, and it neither
sees nor can certify wall-bounded physics.** Say that, and the drag concession stops looking like
an isolated blemish and starts looking like a coherent scope statement.

### B11 — MINOR/MAJOR. Statistical status of headline numbers

| claim | n |
|---|---|
| `tab:iters` residual↔error divergence (the central finding) | **1 checkpoint, 1 sweep, unseeded**, 80 cases, weak backbone only |
| λ-sweep closing the BC objection (`residual_floor_theorem.tex:131-146`) | **n=10 cases, 1 checkpoint** |
| deep-ensemble certificate q=2.35, ECE 0.074 (`tab:uq`) | **1 ensemble** (5 genuine retrains); only the cal/test split is resampled |
| corrected-field certificate on the weak backbone (`body.tex:1172-1197`) | **15/15**, self-described "directional" |
| selective prediction / AUROC 0.905 | **1 arm, 1 ensemble, 20 positives** |
| cylinder cross-geometry OOD | **1 geometry, qualitative** (correctly labelled) |
| force-vs-official recovery | 3 seeds, 200 cases (fine) |
| `tab:v2` corrector deltas, W1 ablation | 5 seeds (fine — the strongest evidence in the paper) |

The pattern: the *positive* results are well-seeded; the *headline negative* and the *theory's
empirical support* are n=1. That asymmetry is exactly backwards for a paper whose selling point is
the negative.

DeepCFD generality (`body.tex:1031-1050`): per-seed ρ = 0.718 / 0.658 / **0.934**. A 0.28 spread on
n=3, with the mean carried by one seed, reported as "0.770 ± 0.119" and glossed "comparable to —
indeed above — the AirfRANS Transolver value". The figure caption itself concedes seed-dependent
residual offsets (`body.tex:1505-1507`). Do not lean on the "indeed above".

### B12 — MINOR. Numerical inconsistencies a copy-editor will not catch but a referee will

- `body.tex:148` says the corrector "delivers **−9** to −25%"; `body.tex:109` and `tab:v2` say
  **−8%**; `results/control/w1_capture.json` gate says `measured_delta mse_u = −0.0827` against
  `published_delta −0.087`. Pick −8% and use it everywhere.
- The audit cost is **1.7 ms** at `body.tex:1262` and **1.13 ms** in `tab:cost` (`body.tex:1317`)
  for the same object, from two different measurement scripts. Reconcile or label them.
- `tab:v2` lists `C_l` rel err 5.7% for the backbone; `w1_capture.json` per-seed values are
  0.053/0.057/0.058/0.067/0.050 (mean 5.7% ✓) — fine, but the surface-pressure column
  (9843 → 9899) appears only in prose (`body.tex:757-759`), not in `tab:v2`. Put the one channel
  that regresses in the table.
- MeshGraphNet's absolute performance (`mse_u` 13.5, `cd_rel_err` 101%) appears **nowhere** in the
  manuscript; only its ρ=0.851 is quoted. A reader cannot tell from the paper that the third
  backbone is ~100× worse than the first. That omission is what makes B2 look like advocacy.

### B13 — MINOR. Scope items to concede without argument

2-D only; two datasets; classical fallback is an acknowledged stub (`body.tex:602-610`,
`body.tex:1636-1638`) — which means the pipeline figure (`fig:pipeline`) advertises a component
that does not exist. Either grey it out in the figure or remove it. Nothing else in the paper is
harmed by dropping it, and its presence invites the "engineering demo, not a contribution"
reading that has already cost two desk-rejections.

---

## C. Triage: rewrite / experiment / concede

**Fixable by rewriting only (do these regardless):**
- B5 "where" → "which case". Abstract, intro thesis, §3.1, §7, Conclusion. Also fix the
  `percell_residual_error.json` mislabel.
- B6 coverage-as-achievement → lead with ECE/adaptivity; state the pooled-cell marginal guarantee
  precisely and say it is not per-field.
- B2 "backbone-robust" → "consistently positive across three backbones, with strength inversely
  related to backbone quality"; report MGN's absolute error; disclose the density control.
- B10 rewrite the resolution limitation into a scope statement about what the monitor can certify.
- B12 all of it.
- B13 remove/grey the fallback box.

**Fixable by an experiment you can actually run (in priority order):**
1. **Direct residual minimisation** (A). `physics_residual_torch` + gradient descent from the
   prediction; residual and error vs. step. Without this the abstract's central claim is unsupported.
2. **All-channel iteration sweep** (B4). Add `mse_v`, `mse_p`, total field error to
   `run_sensitivity.py`. Cheap; possibly paper-inverting; must be done before submission.
3. **ΔAUROC / Δoracle-recovery bootstrap, residual vs σ vs fused** (B1). Zero new compute.
4. **Fixed-half-step control for the gate** (B7). Zero new compute — `per_case[*].trials` has it.
5. **Risk–coverage on |ΔC_d| and |ΔC_l|** (B9). Zero new compute; per-case force errors exist.
6. **Partial correlation ρ(residual, error | ‖r*‖)** — the confound control nobody has run. Both
   arrays exist (`residual_floor_realdata.json` per-case truth floors,
   `selective_percase.json` per-case errors). If per-case truth floor alone predicts model error,
   part of ρ=0.61 is case difficulty, not error detection. This is the control I would demand first
   in a rebuttal round.
7. **Resolution refinement 128 → 256 → 512** (B3), which decides whether (H2) is your grid or the
   data, and simultaneously answers the sub-cell-boundary-layer objection.
8. **Evaluate ‖L⁺r*‖ and σ_min** for at least a handful of cases (B3), or downgrade theorem leg
   (iii) to a remark. Seed `tab:iters` (3 seeds) and re-run the λ-sweep at n=200 rather than n=10.

**Must be conceded:** 2-D; sub-cell boundary layer and the consequent inability to certify wall
quantities or drag; stub fallback; single-ensemble certificate; MGN as a weak instantiation.

---

## D. Overreaching claims, verbatim, with corrections

1. `abstract.tex:11-12` — "the physics residual is a reliable, backbone-robust **trust signal** (it
   tells you *where* the prediction is wrong)"
   → **"…a reliable case-level trust signal (it tells you *which* predictions to distrust; spatial
   localisation is weak, ρ = 0.17 per cell and 0.32 at 16-cell patch scale)."**

2. `abstract.tex:15-16` — "it flags the worst-decile cases with AUROC ≈ 0.9"
   → **"…with AUROC 0.87, marginally below the 0.89 obtained by the ensemble standard deviation
   alone; their rank fusion reaches 0.91."**

3. `abstract.tex:16` — "a distribution-free split-conformal layer attains its target coverage"
   → **"…attains its target coverage (as split conformal must under exchangeability); with a
   deep-ensemble σ the band is additionally input-adaptive (ECE 0.074), which MC-dropout on this
   near-deterministic backbone is not (ECE 0.31)."**

4. `abstract.tex:17-18` — "*As a correction objective*, the residual fails: reducing it does not
   reduce field error."
   → Either run the minimisation experiment, or: **"Along the corrector's fixed-point path the
   monitored residual rises while `mse_u` falls, so the residual is not a valid criterion for
   selecting an iterate; we do not test direct residual minimisation."**

5. `body.tex:81-82` (contribution 1) — "A backbone-agnostic trust layer over a **consistently
   positive residual detector (headline)** … so the detector is consistently *positive* across all
   three"
   → keep, but add: **"strength is inversely related to backbone accuracy (ρ = 0.61 on the most
   accurate model, 0.85 on the least), so the signal is weakest where deployment matters most."**

6. `body.tex:100-104` — "rank-fused with the ensemble σ reaches AUROC 0.905, recovering ≈91% of the
   oracle's achievable error reduction"
   → **"…0.905, against 0.894 for the ensemble σ alone (recovering ≈72% vs the residual's ≈67% of
   the oracle); the fusion gain over σ alone is within bootstrap noise at n = 200."**

7. `body.tex:128-129` (contribution 3) — "**Residual-as-objective fails**… Sweeping the correction
   iterations *raises* the PDE residual (0.11→0.62) while *lowering* field error (`mse_u`
   3.92→2.29)"
   → **"Sweeping the DEQ corrector's internal fixed-point cap raises the monitored residual while
   `mse_u` falls (single checkpoint, 80 cases, unseeded, weak grid backbone; `mse_v` and `mse_p`
   not measured along this path)."**

8. `body.tex:148` — "delivers **−9** to −25% on the SOTA backbone" → **−8 to −25%.**

9. `body.tex:52-56` — "near-oracle triage, recovering ≈91% … the residual alone still recovers
   ≈67%" → add **"and the ensemble σ alone recovers ≈72%."**

10. `body.tex:494-498` — "0.22±0.06 … measured on the deployed corrected field"
    → **"0.166 ± 0.155 on the deployed field (n = 200); the 0.22 ± 0.06 figure is from a
    superseded dropout-FNO checkpoint at n = 15."**

11. `body.tex:267-272` — "Our distinct contributions are (i) the head-to-head *dissociation* …
    (ii) the *residual-floor theorem* … (iii) a split-conformal certificate calibrated on the
    *deployed, corrected* field."
    → (i) is not established until the minimisation experiment exists; (ii) legs (i)–(ii) are
    elementary and leg (iii)'s bound is never evaluated; (iii) is real but the corrector was trained
    on single-member fields and applied to the ensemble mean. Rewrite all three.

12. `body.tex:1004-1005` — "**The trust signal is regime-invariant — the single strongest result.**"
    → it is the result obtained on the arm that inflates `mse_v` by 129%. **"On the weak backbone
    the corrector both degrades the field and raises ρ; we cannot presently separate improved
    detection from the corrector injecting residual-visible error."**

---

## E. Two of the assigned attack surfaces are FACTUALLY WRONG — do not defend against them

**(E1) "The deep-ensemble rescue comes from one training run with member subsets rather than
independent retrains."** False. `scripts/run_ensemble_uq.py:147-149` sets
`torch.manual_seed(1000 + m)` / `np.random.seed(1000 + m)` per member with a fresh model and
optimiser — the M=5 members **are** independent retrains. The Limitations sentence
"Independently retrained ensembles (beyond member subsets of one training run) remain future work"
(`body.tex:1627-1628`) refers only to the **M-study** (M=2,3,4 are subsets of those same 5 members),
and that is stated correctly. The M-study caveat is legitimate and already disclosed; the ensemble
itself is sound. The only live criticism here is that the certificate rests on **one** ensemble
(B11).

**(E2) "The conformal certificate is calibrated on the raw backbone, not the corrected field."**
Outdated. §5.6 now calibrates on the corrected field directly on the deployed Transolver:
`results/uq_ensemble/w2_conformal_corrected.json`, production 100/100 split, all three corrector
seeds, with a faithfulness gate reproducing q=2.352 / coverage 0.9155 exactly
(`body.tex:1198-1213`). This is one of the better-executed parts of the paper. The live objection is
the **configuration mismatch** in B8 (corrector trained on individual backbones, applied to the
ensemble mean), not the raw-vs-corrected question.

Flagging these is worth more than adding two more attacks: a real R2 will not raise them, and
rebuttal space spent on them is wasted.

---

## F. Questions to the authors (trap-aware)

1. `results/sensitivity/iters.json` records `"swept_knob": "DEQCorrector.max_iter"` and
   `"engine_maxiters_inert_on_deq": true`. Identify the experiment in which the residual is the
   quantity being **minimised**. If there is none, on what basis does the abstract assert that
   "reducing it does not reduce field error"?
2. What are `mse_v` and `mse_p` at n_iters = 0, 1, 3, 5, 10, 15 in `tab:iters`? Does total volume
   field error fall, or only `mse_u`?
3. In `selective_prediction.json`, `sigma_vel` beats the residual on Spearman (0.654 vs 0.610) and
   AUROC (0.894 vs 0.871). What is the paired bootstrap CI on ΔAUROC(fused − σ)? If it contains
   zero, what does the physics residual add that a deep ensemble does not?
4. Across your four measurements ρ rises monotonically as backbone accuracy falls (Transolver 0.611
   → weak FNO 0.397 → damaged FNO+DEQ 0.827 → MGN 0.851, `mse_u` 13.5). Distinguish "the residual
   detects error" from "the residual ranks models by how dispersed their errors are."
5. `mgn_density_control.json` reports `"verdict": "DENSITY-DRIVEN"`, `velocity_ratio 2.44`, with MGN
   trained at 16,384 points and evaluated at ~180k. Why is this not in the manuscript, and what is
   ρ for MGN evaluated at its training density?
6. `norm_truth_continuity_mean = 0.1226`: the reference field violates discrete continuity at ~12%
   of u∞/L on your grid. Is that your rasteriser or the AirfRANS data? What fraction of the floor
   `‖r*‖ = 0.192` survives 128 → 256 → 512 refinement (or a body-fitted O-grid)? Does (H2) survive,
   and if it does, is the manuscript's stated cause for it correct?
7. `bc_violation` (`residuals.py:203-213`) penalises deviation from freestream on the outer ring of a
   crop 1–1.5 chords from a lifting airfoil, where the true flow is not freestream. The uniform field
   satisfies this term exactly by construction. Isn't argument (A) of the λ-sweep circular? What is
   `truth_bc2` vs `uniform_bc2` with the far-field term removed, or with a circulation-consistent
   far-field condition?
8. Report `σ_min(L)` and `‖L⁺r*‖` for at least a handful of cases. If they are not computed, what
   does theorem leg (iii) contribute beyond the classical `‖e‖ ≤ ‖R‖/σ_min`?
9. `per_case[*].trials` in `acceptance_gate.json` already contains step 1.0 and 0.5 for every case.
   What are the accept-rate and error-improvement statistics for a **fixed 0.5 step with no gate**?
   If they match, what does gating on the residual contribute?
10. Your conformal `q` is one quantile over ~1.6M pooled, spatially correlated cells. State the
    guarantee you actually hold: is it per-cell marginal, or per-field? If per-cell, how does it
    support the per-case accept/reject decisions the trust layer is sold for?
11. The certified path (`w2_conformal_corrected.json`) is ensemble-mean + a corrector trained on
    individual backbones; the accuracy headline is single-backbone + DEQ; the most accurate field is
    the uncorrected ensemble mean. What is "NeuroForge", and what is its accuracy?
12. Risk–coverage on |ΔC_d| against official labels: does rejecting the 10% least-trusted cases
    reduce drag error? If not, what design decision does the trust layer support?
13. At Δ = 0.0234 c and Re = 2×10⁶ the first cell centre is at y⁺ ≈ 10³. In what sense is
    `ν_eff ∇²u` on this grid a RANS momentum residual rather than an inviscid-plus-numerical-diffusion
    residual?
14. After the W1 ablation (0/5 seeds for WITH), the gate result (a damped step), and the failure of
    the objective role, which component of the deployed system depends on the physics residual for
    anything other than case-level ranking — and how does that ranking differ from
    `gopakumar2025pre`, which your own `tab:positioning` credits with Trust + Calib + ≥3bb + ≥2data?

---

## G. Score and verdict

**Score: 3 / 10 — Reject.**

I want to be fair about what is genuinely here, because it is not nothing: the W1 null-corrector
ablation (5 seeds, 0/5, `w1_capture.json`) is a model of self-refutation that most authors would
have buried; the force-integrator forensics (`ρ_L = 0.998`, median 3.6–3.9%, with drag error
correctly attributed to the model rather than the integrator) is careful measurement science; the
20-split conformal resampling, the M-study, the audit-cost measurement, the retraction of stale
single-seed numbers, and the artifact manifest are all above the norm for this literature. The
engineering and the record-keeping are better than the paper.

But the science does not survive its own repository. The paper's originality reduces, by its own
positioning table, to three columns. *Object.* rests on a sweep that never optimises the residual
and reports one of three channels. *Select.* rests on a signal that its own results file shows is
beaten by a physics-free ensemble baseline. *Theory* rests on a floor of unattributed origin,
argued at n=10 through a boundary functional that the spurious minimiser satisfies by construction.
Strip those and what remains is: a physics residual correlates with error at the case level
(established in the cited prior art), wrapped in off-the-shelf split conformal, on a 2-D benchmark
at a resolution that cannot represent the boundary layer, with a corrector whose gain the authors
themselves attribute to something other than the physics.

That is the same conclusion CMAME and JCP reached, and reframing will not change it. What could
change it is the list in §C: experiments 1, 2, 3, 4 and 6 are all cheap, several require **zero new
compute**, and any one of them could flip a headline. Run them before the third submission. If the
residual-minimisation experiment confirms the negative and the ΔAUROC bootstrap shows the residual
adds something to σ, this becomes a defensible paper — a narrower one, about a free case-level
error proxy and its documented limits, which is a real contribution honestly sized. If they do not,
the authors will have learned that before an editor does.

**The single fatal flaw:** the paper's central claim — that the residual fails as a correction
objective — is asserted from an experiment (`tab:iters`, `run_sensitivity.py`, n=1 checkpoint,
n=80 cases, one of three channels) in which the residual is never used as an objective.
