# Reviewer 2, round 2 — "Neural CFD surrogates earn their keep in the first cell off the wall"

Adversarial pre-submission review of `docs/paper/body.tex` + `abstract.tex` (read in full),
against `results/`, `src/neuroforge/`, `scripts/`, and the supporting notes in
`docs/paper/review/`. Every objection is anchored to a line, a number, or a committed file.
Round 1 (`docs/paper/review/r2_holes.md`) scored this project **3/10**. The comparison is in §6.

Two notes before the attacks:

* **The soft-spot list I was handed is stale.** It describes the *previous* manuscript
  (`mse_v` regression as a live issue, MeshGraphNet-as-strawman, cylinder-as-generality-claim,
  "the certificate is calibrated on the RAW backbone"). My own round-1 §E retracted the last of
  those, and the current `sec:conformal` calibrates on the corrected field directly on the
  deployed Transolver. I have not re-raised any of them. I reviewed against the four claims in
  the task brief plus what is actually in the file.
* **The brief's description of the manuscript is wrong about the manuscript.** It lists four
  claims, the fourth being "a short section on what a physics residual can and cannot do."
  In `body.tex`, `sec:residual` is 85 lines — but `sec:v2` through `sec:selective`
  (lines 744–1255) is ~510 lines of trust-layer material, plus `sec:cost` and
  `sec:indist-ablation`. That is roughly **40% of a 1683-line body** the authors did not
  mention when describing their own paper. §2.6 below treats that as a finding, not a slip.

---

## 1. The single worst genuine objection

> **The headline comparison is not a comparison between learning and interpolation. It is a
> comparison between two methods optimised against two different measures over the domain,
> scored in the one that favours the interpolator — and the paper asserts the asymmetry runs
> the other way.**
>
> `body.tex:450-452` states: *"The input asymmetry runs **against** the baseline: the surrogates
> receive the full 7-channel geometry and freestream encoding per case, the interpolator receives
> seven scalars."* That is true and it is not the asymmetry that matters. Here is the one that does.
>
> **The interpolator's hyperparameters were selected by cross-validation on the headline metric
> itself.** `interpolation_baseline.md:86-89`: 5-fold CV inside the train split on
> `S = mean_{u,v,p} MSE_channel / Var_train(channel)`, computed on the r128 rasterised stacks —
> i.e. on a uniform-area grid measure that is, by the paper's own `tab:interp_bands`,
> **98.7% a far-field measure** (cell fractions beyond 0.05c: 0.029+0.158+0.800).
>
> **The Transolver was trained against a different measure over the same domain.**
> `scripts/run_baselines.py:121-129`: `sel = rng.integers(0, m, size=min(n_pts, m))`, then
> `loss = torch.mean((pred - yb)**2)` on `normalizer.transform_out(pc.targets[sel])`. That is a
> *uniform* subsample, so it faithfully preserves **the cloud's own node measure** — and
> AirfRANS's cloud is wall-clustered: `interpolation_resolution_ladder.md:131-132` measures a
> far-field nearest-neighbour spacing of 0.004595c against a wall spacing of 4.43e-5c.
> The scoring measure is area-uniform. **The two measures therefore weight the near-wall region
> differently, and the manuscript never measures by how much.**
>
> The mechanism is visible in the adapter, and it is sharper than the paper's protocol-identity
> argument. `transolver_adapter.py:93-104` predicts at the source cloud's own node positions and
> then linearly rasterises — so the grid metric is a **linear re-weighting of per-node errors**,
> in which sparsely sampled far-field nodes are *upweighted* relative to the cloud measure and
> the densely sampled wall nodes are *downweighted*. Transolver is optimised under one weighting
> and scored under the other; the interpolator is selected and scored under the same one.
>
> So one arm is tuned on the scoreboard and the other is not. The paper's own decomposition is
> consistent with this: the only place the interpolator loses is the first grid cell, which is
> exactly where the cloud measure concentrates and where the grid metric barely looks.
> **"88% lower volume-pressure error" is, on the evidence in this manuscript, not distinguishable
> from "we changed the weighting."**
>
> The paper cannot settle this, and it knows it: `body.tex:1430-1441` concedes the point-space
> head-to-head was not run and calls it *"the single most valuable follow-up in the paper."*
> That was also its status in round 1. The most important experiment in the paper is still the
> one that was not done, and without it the abstract's lead number is unfalsifiable.

**The one-line measurement that would settle the size of the effect** — and which I am *not*
claiming to know the answer to: the fraction of each case's ~180k cloud nodes lying within 0.05c
of the wall, against the 1.3% of grid cells in the same band. One pass over the clouds, no
training. If that fraction is near 50% the objection is decisive; if it is 10% the objection
shrinks to a scoping caveat. **The paper should compute it before a reviewer does**, because right
now the direction of the asymmetry is established and the magnitude is unknown to both sides.

**Why this is fatal rather than fixable by rewriting.** Three of the paper's four claims sit on
this number. It cannot be rescued by a caveat, because the caveat ("scored on a 128² raster")
does not name the mechanism — the mechanism is *which points each method was asked to be right
at*, and that survives any change of scoring resolution as long as scoring is area-uniform.
Resolving it requires either a Transolver arm trained/selected against the grid measure, or the
head-to-head reported under the point measure. Both are new runs.

**The cheapest discriminator — separate from the fatal flaw, and not a substitute for fixing it.**
The paper never decomposes the *surrogate's* error by wall distance. `tab:interp_bands` is the
**interpolator's** error only. The central claim — `body.tex:546-550`, *"it says precisely what a
learned surrogate buys on this benchmark: the first cell off the wall"* — is inferred from (a)
where the interpolator's residual error lives and (b) channel-level agreement, and is never
measured on Transolver. Running `interpolation_band_control.band_decomposition` on the Transolver
prediction fields costs one evaluation pass over existing checkpoints and tells you which way this
breaks:

* If Transolver's `mse_p = 628.5` is *also* wall-concentrated, the localisation thesis is measured
  on both arms, and the §1 objection loses most of its bite — you would have shown that both
  methods are accurate in the same places and differ in degree.
* If it is far-field-spread, the honest statement is not "the surrogate buys the first cell" but
  "the two methods are accurate in different places and this metric weights one of them," which is
  the §1 objection confirmed from inside the paper.

Either outcome is worth having, and **its absence in a paper whose thesis is a localisation claim
is the kind of gap a hostile reviewer reads as choosing not to look.**

**What would resolve the fatal flaw, in ascending cost:**
1. Report the head-to-head under the *point-space* measure Transolver trains on (per-point
   standardised MSE on the native cloud), with the interpolator evaluated at cloud points. This
   needs no `PointNormalizer` recovery — the interpolator can be scattered to nodes.
2. Train one Transolver arm to the r128 grid measure, or re-select the interpolator by CV on the
   point measure. Either direction breaks the tie.
3. The full point-space head-to-head the paper names.

---

## 2. Ranked weaknesses

Severity: **[F]** fatal as-is, **[M]** major, **[m]** minor/polish.
Triage: **R** = fixable by rewriting, **E** = fixable by an experiment you can run, **C** = must be conceded.

### 2.1 [F][E] The measure asymmetry above, plus the un-decomposed surrogate error.
Evidence: `run_baselines.py:121-129,199`; `transolver_adapter.py:93-104`;
`interpolation_baseline.md:86-89`; `interpolation_resolution_ladder.md:131-132`;
`tab:interp_bands`; `body.tex:450-452`, `546-550`, `1430-1441`. Triage above.

### 2.2 [F][R] The originality defence your own repository wrote for you is not in the manuscript.
This paper has been desk-rejected twice on originality. `naming_and_positioning.md:271-278` tells
you, in writing, what the third rejection will say:

> *"Polynomial response surfaces and Kriging over design variables are 20–70 years old in this
> exact application. An aerodynamics reviewer will say 'you rediscovered RSM,' and they will be
> partly right. Say it first, in one sentence."*

And `naming_and_positioning.md:556-558` grades two of the paper's framings **NOT novel**.

I grepped `body.tex` and `refs.bib`. **`refs.bib` has 42 entries. Not one is Queipo (RSM),
Poliak/Gururangan (partial-input baselines), Feng (the standing caveat on partial-input
baselines), Errica (structure-agnostic baselines — the closest structural analogue in all of ML),
Ferrari Dacrema (the recommender-systems simple-baseline audit), Ahlmann-Eltze (the Nature Methods
genre proof point), Rasp/WeatherBench2 (the field that already mandates trivial controls), or
Ethayarajh (V-usable information).** The strings "response surface", "kriging" and "POD" appear
nowhere in `docs/paper/body.tex`. The Related Work benchmarking paragraph (`body.tex:224-245`)
cites exactly one evaluation-methodology paper, `mcgreivy2024weak`.

So the manuscript presents a Gaussian kernel-ridge response surface over seven airfoil design
variables as a novel baseline, without naming the 70-year-old literature it is drawn from, in a
paper submitted after two originality desk-rejects. `naming_and_positioning.md:486-518` even
contains the drop-in paragraph. It was not dropped in.

To be precise about what I am *not* claiming: the scoped sentence at `body.tex:92-98`
("no published parameter-space interpolation baseline for AirfRANS field metrics and no covariate
null for its force-rank metric") is probably true and I am not attacking it. I am attacking the
absence of the lineage, which makes the scoped sentence read as a novelty claim it is not.

**Triage: R, one hour.** Add the paragraph, add the eight bib entries, and add one sentence
stating that the estimator is textbook RSM and the contribution is the measurement of what its
omission conceals. This is the single highest value-per-minute change available.

### 2.3 [M][R+E] Two of the four headline numbers are not the quantities their names imply.
`abstract.tex:10-11`: *"force errors **4.9× lower on lift** and 3.8× lower on drag."*
`body.tex:38-40` and `tab:interp` repeat it as "$C_l$ err 1.18% against 5.81%".

What that number actually is: `physics/evaluation.py:108-147`, `coefficient_metrics(pred_coeffs,
ref_coeffs)` computes `|pred − ref| / |ref|` where **`ref` is the force integrated from the
rasterised ground-truth field through the same grid integrator** — not the official AirfRANS
label. `physics/metrics.py:203,238-248`: that integrator samples pressure 1.5 cells off the wall
and applies an *unsigned* wall-shear magnitude along the traversal tangent.

The paper declares this scope for $\rho_{C_l}/\rho_{C_d}$ (`body.tex:406-410`, and the `tab:interp`
caption) but **not for the relative-error columns**, which the caption leaves as bare "mean
relative error". And in `sec:v2` the paper demolishes that very integrator in its own words:

* `body.tex:798-800`: absolute drag *"is biased ($\approx 11\times$ mean relative error) by the
  integrator's 1.5-cell surface-sampling offset, and the same integrator on **perfect** fields also
  yields $\rho_D = 0.84$."*
* `body.tex:838-841`: *"the viscous component of drag is not merely inaccurate here---it is
  **unavailable**."* At 68% of true drag (`body.tex:819-820`).
* `interpolation_baseline.md:282-288`: against **official** labels the interpolator scores
  $\rho_{C_d}=0.8389$, *indistinguishable from the exact ground-truth field* (0.8394).

So the abstract's "3.8× lower on drag" is a 3.8× improvement in agreement with a quantity the
paper itself says is 11× wrong, missing 68% of its physics, and no better on the exact truth than
on the interpolation. An engineer reading the abstract will not read it that way.

There is also an unexplained internal tension the paper should resolve before a reviewer does:
the interpolator **loses** on surface pressure (`tab:interp`: 10989 vs 9110) yet wins lift by
4.9× through a *surface* integrator. Lift on an airfoil is surface pressure. Either the
surface-pressure MSE is dominated by cells the integral does not weight, or the force numbers are
not measuring what their name says. Say which.

**Triage: R for the labelling (mandatory); E for the fix.** Report $C_l/C_d$ magnitude error
against the **official** labels for both arms (you already have the official-label cache,
`results/control/_cache/official_labels_full_test_n200.json`, and the interpolator's ranks), or
drop the force-magnitude claims from the abstract and keep them as field-fidelity proxies in the
body with the scope attached. Also report median and a trimmed mean: `coefficient_metrics`
divides by a reference that crosses zero, and `interpolation_resolution_ladder.md:228-255` shows
one case owning a mean of 1.842 at 512².

### 2.4 [M][R] "Four of five published baselines fall below that null on lift" is true of point estimates and not of the spread you yourself report.
`abstract.tex:19-20`, `body.tex:59-61`, `tab:covariate`, Conclusion. Null:
0.9821 [0.9737, 0.9866]. Published (from your own source-verified
`published_baselines_verified.md:17-22`, five training runs each):

| model | ρ_L | gap to null point est. | gap in units of the model's own seed sd |
|---|---|---|---|
| MLP | 0.913 ± 0.018 | 0.069 | **3.8 sd** |
| PointNet | 0.938 ± 0.023 | 0.044 | **1.9 sd** |
| GraphSAGE | 0.965 ± 0.011 | 0.017 | **1.55 sd** |
| Graph U-Net | 0.967 ± 0.019 | 0.015 | **0.79 sd** |

Only MLP is decisively below. Graph U-Net is inside one published seed standard deviation.
Worse, the two intervals are not commensurable: yours is a **case-level bootstrap** over 200 test
cases; theirs is a **seed** spread over 5 training runs. Neither accounts for the other's variance
source, so "falls below" is a comparison of two different uncertainties. Your pre-registered rule
is satisfied, and I am not accusing you of moving it — I am saying the rule is weaker than the
sentence it licenses.

The paper is scrupulous about exactly this on the drag column (`body.tex:713-717`, three of four
intervals span zero). Apply the same standard to lift.

**Triage: R.** Rewrite as: *"the null's point estimate exceeds all four AirfRANS-paper lift
entries; the margin is 3.8 published seed standard deviations on MLP and 1.9 on PointNet, but
only 1.6 and 0.8 on GraphSAGE and Graph U-Net, so the decisive claim is for two of the four."*
That is a smaller headline and it is the one your numbers support.

**Related severity note.** The force-rank null's strongest instance is against *2022 dataset-paper
baselines that the benchmark's own authors already report as failing at drag*
(`body.tex:63-64`, Transolver App. B.1: "the deep models fail at drag"), and the one modern
entry, Transolver, **clears** on lift (0.9978). So in the manuscript's single benchmark, the null
beats what the field already knows is broken and loses to what the field actually uses. That is a
real limit on how much the result changes practice — see 2.7.

### 2.5 [M][R] A committed artifact in your own repo qualifies the headline's precision, and the manuscript does not report it.
`interpolation_resolution_ladder.md` runs the 128²/256²/512² ladder, gates the 128² rung against
the published row at **relative error 0.00e+00**, and returns a pre-registered verdict of
**PARTIAL** (line 21): the near-wall SE shares go **0.9238/0.8979 → 0.8792/0.8417**, and the $v$
share misses the pre-declared 0.85 floor at 512².

I grepped `body.tex` for `512|ladder|nearest_all`. The interpolation ladder appears **nowhere**;
lines 1335 and 1483 are the *residual-floor* ladder. The abstract and intro quote 92%/90% with no
resolution label, and `sec:limitations` still carries the pre-ladder text ("we cannot answer from
these runs"). The ladder document itself (§8.1–8.5) drafts the exact replacement paragraphs.

This is the round-1 pattern repeating. In round 1 the undisclosed artifact was
`mgn_density_control.json` (§B2). It is now disclosed (`body.tex:868-875`) — credit given. But the
*habit* recurs here, and a reviewer who reads your manifest (which you invite them to) will find
a PARTIAL verdict against a headline you present without qualification.

Note the ladder is **good news** for you: the localisation tightens under refinement (72% of the
$u$ error inside 0.005c at 512²), the outer-region MSE is flat to 1.5% across a 16× cell-count
increase, and $F(0.99)$ is stable at 0.995. Reporting it converts your most attackable statistic
into a defended one. Not reporting it converts a strength into a discoverable.

**Triage: R, the text is already written in `interpolation_resolution_ladder.md` §8.**

### 2.6 [M][R] Forty percent of the body is the paper you have twice been rejected for, and you state in it that you claim no priority over it.
`body.tex:260-263`: *"The trust-layer components this paper retains are individually well
developed in the recent literature \citep{gopakumar2025pre,jia2026multigranularity,roy2025anchor,
song2026structureaware,mukherjee2026certification} and we claim no priority over them."*
`body.tex:1570`: *"The durable contribution is a pair of reference points and the boundary they
draw, **not a system**."*

Then `sec:v2`, `sec:indist-ablation`, `sec:ood`, `sec:conformal`, `sec:selective`, `sec:cost`
(lines 744–1317) run the system anyway, and contributions **4, 5 and 6** of six are trust-layer.
Contribution 5 alone is 12 lines about conformal coverage, ensemble σ, an acceptance gate and an
oracle-recovery percentage.

An editor triaging on originality will read the contributions list top to bottom and see three of
six contributions that the paper itself disclaims priority over, on a 2-D benchmark, from a
project that has been desk-rejected twice. You are handing them the previous decision.

**Triage: R, by cutting.** Concretely: keep `sec:residual` (it is now the strongest negative in
the paper and it *supports* the evaluation thesis — "you cannot fix the metric with physics").
Keep one paragraph of `sec:v2` — the one that reads the corrected `mse_p = 485` against the
interpolator's 75.0 — because that is an evaluation statement. Move `sec:conformal`,
`sec:selective`, `sec:cost`, `sec:ood` and `sec:indist-ablation` to the companion paper you
already reference twice. Contributions 5 and 6 go with them. The paper gets shorter, the
originality surface gets smaller, and nothing in claims 1–4 is weakened.

### 2.7 [M][C→E] The manuscript's generality claim is the weakest instance of the evidence sitting in its own repository.
`body.tex:1452-1457` concedes: *"The covariate null is one benchmark, one metric family."*
Meanwhile `null_travels.md` reports the identical control on **five** benchmarks under each one's
own split and metric: DrivAerML $R^2 = 0.973$ [0.958, 0.983], an interval containing DoMINO (0.98)
and FIGConvNet (0.97) and above X-MeshGraphNet (0.92); DrivAerNet++ 0.737, above all three
dataset-paper baselines; AhmedML 0.684 with no published baseline; **WindsorML 0.104, a clean
counterexample where the published model wins outright**. `nullbench_release.md` ships this as a
tested package with a CLI and five corrected leaderboards.

I am not filing this as an overclaim — the paper is *narrower* than its evidence, which is the
honest direction. I am filing it as the reason the paper's claim 2 is weak where it did not have
to be. As submitted, the reviewer's question is "is this an AirfRANS quirk?" and the paper's
answer is a limitations bullet. With `null_travels.md` in it, the answer is a table with a
counterexample in it, which is exactly the structure that made Errica et al. (ICLR 2020) and
Ferrari Dacrema et al. (RecSys 2019 best paper) land — and `naming_and_positioning.md:190-210`
says so.

`nullbench_release.md:12-15` says this material was deliberately kept out because a concurrent
thread owned `body.tex`. That is a workflow reason, not a scientific one. **Decide explicitly
whether it is in or out, and if out, say in the paper that it is a companion release.**

**Triage: E if you want it (the runs exist; it is a §-insertion, not compute), C if you split it.**

### 2.8 [M][E] The density confound is disclosed for the arm that hurts a claim and assumed away for the arm that carries the headline.
`body.tex:868-875` discloses, correctly and to your credit, that MeshGraphNet *"was trained on
16k-point subsamples and evaluated at the full ≈180k points"* with a control returning
**DENSITY-DRIVEN**, MSE ratio **2.44**.

**The Transolver baseline has the same train/eval density mismatch.** `run_baselines.py:199`:
`--n-points` defaults to **16384**; line 17 of the same file documents the full run as
`--n-train 800 --n-val 200 --epochs 80 --n-points 16384`. `transolver_adapter.py:86-108`:
`make_predict_fn` *"predicts per-point fields on the **full cloud**"*. Identical setup.

The only place this is addressed is a **script docstring**: `control_mgn_density.py:15` asserts
*"Transolver (global attention) is density-robust, so the comparison would be unfair."* That is a
plausible architectural argument — slice attention aggregates by normalised weighted average, so
uniformly subsampling the same cloud should converge — and I am **not** claiming it is wrong.
I am claiming it is **asserted, never measured, and invisible to the reader**, on the arm whose
degradation directly inflates the 8.4× headline. The one arm where it *was* measured moved by 2.44×.

**Triage: E, and it is cheap.** Run `control_mgn_density.py`'s probe on Transolver: same
checkpoint, same normaliser, score per-point physical MSE at the same 16,384 indices under a 16k
graph and under the full cloud. If it is flat, you have converted a discoverable into a control
and strengthened the headline. One paragraph, no training.

### 2.9 [M][R] Contribution 3 states unconditionally what the evidence supports only conditionally, and the paper's own Table contradicts the unconditional form.
`body.tex:134-136`, contribution 3: *"**The `reynolds` split does not test Reynolds
generalisation.** The difficulty is dimensional scaling, not physics."* Conclusion and abstract
repeat it.

The measured content is solid and I accept it: 0/200 test cases have an in-range inlet velocity
(proved, not asserted); the nondimensional interpolator degrades 3% / −1%; the `raw` interpolator
degrades 5.2× / 10.1×. That is a clean, well-controlled demonstration.

But `sec:splits:648-649` states the claim in its correct conditional form — *"for any method that
**nondimensionalises**"* — and the contributions list, conclusion and abstract drop the condition.
And `tab:ood` in the same paper shows the authors' own grid backbone going `mse_u`
$3.479 \to 16.218$ on that split, a **4.7× degradation**. So the split evidently does test
something that a real learned model fails. What the paper has shown is that the split's difficulty
is *removable by dimensional analysis*, not that it is absent.

`body.tex:658-659` already carries the load-bearing caveat ("no matched *surrogate* row exists on
either split"). Promote it: the claim is n=1 in methods.

**Triage: R (restore the condition everywhere), or E (add a matched surrogate row on `reynolds`,
which would make this a two-method claim and is the same training run you already do).**

### 2.10 [m][R] The abstract's $R^2 \ge 0.9996$ is the more favourable of two statistics, and the only table in the paper shows the less favourable one.
`abstract.tex:14-15` and `body.tex:47-48`: *"98.7% of the domain, beyond 0.05 chord, is reproduced
at $R^2 \ge 0.9996$."* `tab:interp_bands` reports **only** $R^2_{\mathrm{pc}}$, whose values
beyond 0.05c are 0.9991 / 0.9983 / **0.9965** on $u$ — all below 0.9996. The caption says the
pooled figure "is higher than $R^2_{\mathrm{pc}}$ at every entry; both are in the artifact", so
the headline number appears **nowhere in the paper**. A reader checking the abstract against the
table concludes the abstract is wrong.

Additionally, `interpolation_resolution_ladder.md:403-408` records that the 0.9996 claim binds on
a **fourth-decimal margin** (p in 0.05–0.15c at 0.999606) and that under $R^2_{\mathrm{pc}}$
**no region clears 0.999 at any resolution**.

**Triage: R.** Either add the pooled column to `tab:interp_bands`, or quote the strict statistic
(0.9965) in the abstract. The strict number is still a remarkable claim and it costs you nothing
rhetorically.

### 2.11 [m][R] Two different "Transolver" rows with different numbers, and the comparison uses whichever is in scope.
`tab:interp` Transolver: `mse_u` 0.120, `mse_p` 628.5, $C_l$ 5.81%, $C_d$ 8.99%.
`tab:v2` backbone: 0.133, 644, 5.7%, 7.6%. `interpolation_baseline.md:102-103` discloses both and
the paper prints them in adjacent sections without saying they are different runs of the same
model. Separately, `tab:interp` reports $\rho_{C_l} = 0.9992$ (self-consistency) while
`tab:covariate` reports Transolver $\rho_{C_l} = 0.9978$ (vs official) — two numbers, both ~0.99,
both labelled $\rho_{C_l}$, in tables three pages apart. Label the runs and rename one of the two
correlations.

### 2.12 [m][C] Scope items to concede without argument.
2-D only; one benchmark for the null; `ClassicalFallback` is a stub (`body.tex:392-394`) — if the
trust-layer sections go per 2.6 this disappears; the interpolator carries 210 MB resident
(conceded, `body.tex:614-620`); no seed variance on the headline interpolation rows (deterministic,
conceded).

---

## 3. Direct answers to the questions in the brief

**Is the interpolation comparison fair?** Rasterising a point-cloud model does *not* invalidate
it, and this is a fairness argument you can win — but win it for the right reason.
`transolver_adapter.py:93-104` rasterises the **prediction at the source cloud's own node
positions** through `rasterize_point_cloud(..., method="linear")`, the identical call
`airfrans_loader._sim_to_pair` uses on the truth. Same points, same interpolant, so no *extra*
error is injected into the prediction relative to the reference. Say this explicitly; right now
you argue protocol identity but not mechanism identity, and the mechanism is the stronger argument.

But note what the same fact implies, because §1 turns on it: if the grid value is a linear
combination of per-node values, the grid metric is a **re-weighting** of per-node errors —
area-uniform rather than node-uniform. Sparse far-field nodes get upweighted; dense wall nodes get
downweighted. So "the rasterisation does not inject error" and "the rasterisation changes which
errors count" are both true, and the second is the live objection. Do not let the first be read as
answering it.

**What is not fair is the objective asymmetry in §1**, which the paper does not mention at all.

**Is the metadata null fair?** Yes, and the paper's defence is correct as far as it goes. The
digits generate the SDF the surrogate reads; $\theta \to X \to Y$ is a Markov chain by
construction, so the null carries no information the geometry does not, and it therefore bounds
what real readers achieve rather than what a perfect reader could
(`null_mechanism.md`/`naming_and_positioning.md:48-56` make this argument precisely and the paper
compresses it into one sentence at `body.tex:719-726`). Two things weaken it in practice:
(i) it is not *stated* in those information-theoretic terms in the manuscript, so a reviewer will
supply the leakage reading themselves; (ii) the fairness argument does not rescue §2.4 — the
comparison is still one seed sd wide for two of the four entries.

**Is this response-surface methodology from 2005 rediscovered?** The estimator is. The
*measurement* — what the omission of that estimator from benchmark reporting conceals — is not.
**Does the paper answer the objection? No.** It does not name RSM, does not cite Queipo, does not
cite the partial-input or simple-baseline-audit lineage, and does not contain the one sentence
`naming_and_positioning.md:271-278` instructs it to write. See §2.2. This is the most avoidable
weakness in the manuscript.

**Are the effects large enough to matter?** The field-metric effect, yes — 8.4× is not a margin
anyone can wave away, *if* §1 is answered. The force-rank effect, partially: the null beats four
2022 baselines that the benchmark's own authors already describe as failing at drag
(`body.tex:63-64`), and loses to the one modern entry on lift. Within this manuscript's single
benchmark, the practice-changing content is the *protocol recommendation*, not the margin.
`null_travels.md`'s DrivAerML row (null CI contains DoMINO) is the one that would make the effect
size undeniable, and it is not in the paper (§2.7).

**Does anything still overclaim?** Yes: §4 below, seven verbatim items.

**The `reynolds` split claim — established or asserted?** *Established* in its conditional form
(measured disjointness, measured `nd` vs `raw` contrast, a mechanism check at `body.tex:602-605`
showing $U$ enters as pure scaling). *Asserted* in the unconditional form that appears in the
contributions list, abstract and conclusion, and contradicted there by the paper's own `tab:ood`.
See §2.9.

---

## 4. Claims that overreach their evidence — verbatim, with corrections

1. `abstract.tex:6-9` — *"Kernel interpolation over seven scalars parsed from the AirfRANS case
   name---no network, no flow-field learning---scored through the identical protocol, gives 88%
   lower volume-pressure error than a Transolver trained on the same data (8.4×)."*
   → **"…gives 88% lower volume-pressure error under an area-uniform 128² grid measure, of which
   98.7% of cells lie beyond 0.05 chord. The interpolator's hyperparameters are selected on that
   measure; the Transolver minimises per-point error under AirfRANS's own wall-clustered node
   measure. The head-to-head under the point measure is not run."**

2. `abstract.tex:10-11` — *"and force errors 4.9× lower on lift and 3.8× lower on drag."*
   → **"and 4.9×/3.8× closer agreement with the forces integrated from the rasterised ground-truth
   field through the same grid integrator — a field-fidelity proxy, not agreement with the official
   labels; that integrator carries an ≈11× absolute drag bias even on exact fields and cannot
   represent the viscous component."** (`evaluation.py:108-147`; `body.tex:798-800,838-841`.)

3. `abstract.tex:19-20` — *"four of five published baselines fall below it on lift."*
   → **"the null's point estimate exceeds all four AirfRANS-paper lift entries, by 3.8 and 1.9 of
   each model's own published seed standard deviation for MLP and PointNet but only 1.6 and 0.8 for
   GraphSAGE and Graph U-Net; Transolver clears it."**

4. `abstract.tex:14-15` / `body.tex:47-48` — *"98.7% of the domain, beyond 0.05 chord, is reproduced
   at $R^2 \ge 0.9996$."*
   → quote the strict per-case-centred statistic the only table in the paper reports:
   **"…at $R^2_{\mathrm{pc}} \ge 0.9965$ (pooled $R^2 \ge 0.9996$)."**

5. `body.tex:134-136` (contribution 3) — *"**The `reynolds` split does not test Reynolds
   generalisation.** The difficulty is dimensional scaling, not physics."*
   → **"The `reynolds` split's difficulty is removable by dimensional analysis: a nondimensional
   parameter interpolator barely degrades where a dimensional one degrades 5–10×. It is not
   removed for methods that do not nondimensionalise — our own grid backbone degrades 4.7× on
   `mse_u` on that split (`tab:ood`)."**

6. `body.tex:546-550` — *"it says precisely what a learned surrogate buys on this benchmark: the
   first cell off the wall, the streamwise channel…"*
   → until the surrogate's own band decomposition exists: **"it says where the *interpolator's*
   remaining error lives; the corresponding decomposition of the surrogate's error is not
   measured here."**

7. `body.tex:450-452` — *"The input asymmetry runs against the baseline."*
   → **add**: *"The objective asymmetry runs the other way: the interpolator is selected by
   cross-validation on the area-uniform grid measure the tables report, while the surrogate
   minimises a per-point measure under the dataset's own wall-clustered node distribution."*

---

## 5. Questions to the authors (trap-aware)

1. `run_baselines.py:121-129` trains Transolver on per-point standardised MSE over a uniform
   subsample of AirfRANS's own node cloud, which is wall-clustered.
   `interpolation_baseline.md:86-89` selects the interpolator by CV on the area-uniform r128 grid
   MSE. `tab:interp_bands` puts 98.7% of grid cells beyond 0.05c. **What fraction of the ~180k
   cloud nodes lies inside 0.05c?** And on what basis is the 8.4× a statement about learning
   rather than about the ratio between those two numbers?
2. What is the wall-distance band decomposition of **Transolver's** squared error? You have the
   function (`interpolation_band_control.band_decomposition`) and the checkpoints. If Transolver's
   `mse_p = 628.5` is not wall-concentrated, does "what the surrogate buys is the first cell off
   the wall" survive?
3. `control_mgn_density.py:15` asserts *"Transolver (global attention) is density-robust."*
   Transolver was trained at `--n-points 16384` and evaluated on the full ~180k cloud
   (`run_baselines.py:17,199`; `transolver_adapter.py:93`). What is the probe MSE at the 16k
   training density versus the 180k deployment density, at the same point indices? MGN's ratio was
   2.44.
4. `coefficient_metrics` compares the predicted field's forces to the **ground-truth field's**
   forces through an integrator you describe as ≈11× biased on absolute drag and structurally
   unable to see the viscous component. Under what reading is "3.8× lower drag error" a statement
   about drag?
5. The interpolator is 1.21× **worse** on surface-pressure MSE (`tab:interp`) and 4.9× better on
   lift through a surface integrator. Lift is surface pressure. Reconcile these.
6. `refs.bib` contains 42 entries and none of Queipo (2005), Poliak (2018), Errica (2020),
   Ferrari Dacrema (2019/2021), Ahlmann-Eltze (2025), Rasp (2024) or Ethayarajh (2022), despite
   `naming_and_positioning.md` §2 and §6 supplying all of them and a drop-in paragraph. In what
   sense does this manuscript distinguish itself from response-surface methodology, in text a
   reader can find?
7. `interpolation_resolution_ladder.md` returns a pre-registered **PARTIAL** verdict with near-wall
   shares of 0.879/0.842 at 512². `body.tex` reports 92%/90% with no resolution attached and
   never mentions the ladder. Why?
8. Your null's interval is a case-level bootstrap; the published entries' intervals are seed
   spreads over 5 training runs. What is the correct paired comparison, and does Graph U-Net
   (0.967 ± 0.019 against 0.9821 [0.9737, 0.9866]) survive it?
9. Contribution 3 says the `reynolds` split does not test Reynolds generalisation; `tab:ood` shows
   your own backbone degrading 4.7× on `mse_u` on that split. Which is it?
10. You write that "the durable contribution is a pair of reference points… not a system"
    (`body.tex:1570`) and that you "claim no priority over" the trust-layer components
    (`body.tex:260-263`). Three of your six contributions are trust-layer. Why are they in this paper?
11. `null_travels.md` reports the same null on five benchmarks spanning $R^2$ 0.10–0.97 with a
    counterexample, and `nullbench_release.md` ships it as a tested package. `body.tex:1452` says
    "one benchmark, one metric family." Is the cross-benchmark result a companion paper, and if so
    why does the manuscript not say so?
12. `tab:interp` reports mean $C_d$ relative error. `coefficient_metrics` divides by a reference
    that reaches $6.13\times10^{-4}$ at 128² and crosses zero at 512²
    (`interpolation_resolution_ladder.md:237-255`). What are the median and 90%-trimmed values for
    both arms?

---

## 6. Score, verdict, and the comparison to round 1

### Score: **5 / 10 — reject in present form; borderline after §2.2, §2.3 and §2.5, which are all rewrites.**

### The single fatal flaw
**The headline is a measure choice, and the paper cannot currently distinguish that from a
finding.** The interpolator is selected by cross-validation on the area-uniform grid measure the
tables report; the Transolver minimises a per-point measure under AirfRANS's wall-clustered node
distribution; the grid metric is a linear re-weighting of per-node errors that upweights the far
field. Nothing in the manuscript separates "interpolation is better" from "we scored in the
weighting that suits it." Fixing this needs a new run — either the head-to-head under the point
measure, or a Transolver arm trained/selected against the grid measure — and the paper names that
follow-up itself, in Limitations, for the second submission in a row.

**Do not confuse this with the cheap diagnostic.** The Transolver band decomposition (one
evaluation pass over existing checkpoints) does *not* fix the flaw; it tells you which way it
breaks and how much of the localisation thesis survives. Run it first because it is cheap and
decision-relevant, then do the real fix.

### Comparison to my 3/10

**What genuinely improved — substance, not presentation.** I named a fatal flaw in round 1: the
central negative was asserted from a sweep (`tab:iters`) in which the residual was never minimised.
That experiment now exists and is stronger than I asked for: `body.tex:1357-1372` runs direct
Armijo-line-searched descent on $J = \frac12\|R_h\|^2$ from the **exact ground truth**, 500 steps,
**n = 200, 200/200 cases**, verified under three step rules and with boundary data both pinned and
unpinned. **My named fatal flaw is resolved.** Five more of my round-1 items were executed:

* B1 → `body.tex:1213-1221` now reports the paired bootstrap $\Delta$AUROC $[-0.035, +0.083]$ and
  concedes in a heading that *"on field error the physics does not earn its place."*
* B2 → `body.tex:868-875` discloses the MGN density control, returns DENSITY-DRIVEN, and narrows
  the backbone-robustness range to 0.40–0.61.
* B5 → `body.tex:336-348` replaces "where" with case-level ranking and attributes
  $0.166 \pm 0.155$ to the deployed field correctly.
* B7 → `body.tex:1244` reports that an **ungated fixed half-step beats the gate** (95.8%/6.2%
  against 89.3%/5.8%) and reframes the gate as buying the certificate, not the accuracy.
* B12 → −8% used consistently.

Two of those are self-refutations that most authors would have buried. On top of that, the paper
has an **entirely new and better thesis** — the interpolation baseline and the metadata null —
with pre-registered decision rules, five adversarial controls (name overlap, near-duplicates,
permuted-parameter negative control, silent drops, solid fill), a train-size ladder, a frozen
resolution ladder gated at relative error 0.00e+00, and a source-verified transcription of every
published baseline. That is genuinely good measurement science and it is why this is not a 3.

**What did not improve.** Three things, and they rhyme with each other:

1. **The most valuable experiment is still the one not run.** Round 1: "the residual is never
   minimised." Round 2: "the head-to-head is never run in the measure the surrogate optimises."
   Both times, the paper names the gap honestly in Limitations and proceeds to headline the claim
   anyway. Honest disclosure is not a substitute for the experiment.
2. **Selective disclosure recurs.** Round 1 it was the MGN density control (now fixed). Round 2 it
   is the interpolation resolution ladder's PARTIAL verdict (§2.5) and the Transolver density
   mismatch (§2.8) — one artifact in the repo, one line in a script docstring, neither in the
   paper, both cutting against the headline.
3. **The originality defence is still missing, after two originality desk-rejects**, despite being
   fully drafted in `naming_and_positioning.md` §2 and §6 (§2.2). And 40% of the body is still the
   system paper that produced those rejections, with three of six contributions in material the
   paper explicitly disclaims priority over (§2.6).

**Is the improvement substance or presentation?** **Substance, mostly.** A new empirical program
was built and the round-1 fatal experiment was run. But the *remaining* deficit is now
overwhelmingly presentational — §2.2, §2.3 (labelling), §2.4, §2.5, §2.6, §2.9, §2.10 are all
rewrites using material that already exists in this repository. That is a very different situation
from round 1, where the deficit was empirical.

### What moves the score

* **→ 6–7, by rewriting only (a day):** §2.2 (lineage paragraph + 8 bib entries), §2.3 (label the
  force columns; move magnitude claims off the abstract), §2.5 (insert the ladder, verdict first),
  §2.4 and §2.9 and §2.10 (restore the conditions), §2.6 (cut the trust-layer sections). At C&F or
  the NeurIPS Evaluations track I would be at weak accept here.
* **→ 7–8, with two cheap measurements:** the node-fraction inside 0.05c (§1) and the Transolver
  band decomposition (§1, question 2). Neither needs training. Together they convert the §1
  objection from unanswerable into quantified, and if the answers favour you the localisation
  thesis becomes measured on both arms.
* **→ 9, with the point-space head-to-head** (the real fix for the fatal flaw) and
  `null_travels.md` folded in.

The gap between 5 and 8 here is about two days of work, none of it new training. That is the most
useful thing I can tell you, and it is not something I could have said in round 1.
