# Reviewer 2, round 4 — "The barrier was a coordinate artifact. The gap is a different claim, and the manuscript states the wrong one."

Adversarial pre-submission review of `docs/paper/body.tex` + `abstract.tex` (read in full),
against `results/interpolation/{bodyfit_bound.json, point_space_oracle_ls_n200.json}`,
`scripts/bodyfit_bound.py`, `scripts/point_space_headtohead.py`, `docs/paper/refs.bib`,
and the committed titles in `neuroforge_cfd.tex` / `neuroforge_cfd_elsevier.tex`.

Round 1 (`r2_holes.md`) **3/10**; round 2 (`r2_round2.md`) **5/10**; round 3
(`r2_round3.md`) **6/10**. The comparison, and the explicit accounting of round 3's
"→ 9 with the wall-normal arm, whichever way it falls", is in §6. **I am honouring it.**
Read §6 before §1 if you want the verdict first.

---

## 0. What I owe the authors before the attacks

Round 3 named a three-legged fatal objection and priced three remedies. All three were run.

* **The rewrite list was executed.** `D3` and the word "withdraw" are in the text
  (`body.tex:722-726`). Contribution 3 no longer claims the 413× "bounds every published
  $r128$ number" and now states the cancellation argument against itself
  (`body.tex:198-209`) — my round-3 W2 is closed and closed *better* than I asked, because
  the projection-inversion reframe I suggested is in. The `p`-at-the-wall exception is named
  (`body.tex:896-910`). "Matched-budget" is gone from every `.tex` file (grepped: zero hits).
* **The n=3 bound became n=200.** `point_space_oracle_ls_n200.json` `meta.n_ls: 200`,
  21.3× → 25.0×, under gate **G7** requiring bit-identical reproduction of the frozen n=3
  artifact — achieved at `worst_rel_diff_vs_n3_artifact: 0.0`. My round-3 §1(ii) is closed.
* **The experiment I said decides the paper was built and run, and it reversed the
  headline for the third consecutive round.** `bodyfit_bound.py`'s docstring records that
  it was run without knowing which way it would fall, with rule **B1** fixed either way;
  it returned `BODY-FITTED-BOUND-OPEN` at 0.0044× and the title, abstract and contribution 1
  changed accordingly. I said in round 3 that this was "the most distinctive thing about this
  project". It now has a third instance and I will not pretend that is ordinary. I have
  reviewed ~200 papers; I have not seen this three times from one group.
* **My own round-3 W4 number was wrong and the authors corrected me in public.**
  `body.tex:898-908` names the error — `Var_train(p)=135,590` is an *area-uniform* statistic,
  the node-measure surface-band variance is $2.06\times10^7$, and the two arms sit at 0.73%
  and 0.48% of it, not my $R^2=0.27$. They stated the correction rather than silently
  applying it. I withdraw round-3 W4 in its stated form and it does not reappear below.

So §1 attacks the paper that now exists. Two of my round-3 weaknesses (W2, W4) are closed;
W1 is closed; W7's "matched-budget" and $R^2$ labelling items are closed. W5 is **not**
closed — see W6 below, and read it carefully, because the brief I was given says it is.

---

## 1. The single worst genuine objection

> **The new headline — "the near-wall gap is an artifact of the frame" — is carried by a
> quantity that cannot carry it, and credits a coordinate change that the paper's own named
> lineage has used as default practice for two decades without citing one line of it.
> `FAMILY-BOUND-CLOSED` at 25× is an impossibility result and is therefore informative.
> `BODY-FITTED-BOUND-OPEN` at 0.0044× is the *negation* of an impossibility result: it
> establishes that no barrier was *proved* in the new frame, not that the gap *closes*. The
> paper has the realizable number and it points the other way — the same family's deployed
> member in the same frame is $1.556\times$ on the case-mean, i.e. the surrogate still wins
> on average. The licensed claim is "the **barrier** was a coordinate artifact." Four
> sentences, a title and an abstract state "the **gap** was."**

### (i) The asymmetry between the two verdicts is logical, not rhetorical.

`body.tex:562-586` earns its 25×. It is a lower bound on what *any* weighting of the 800
fields can achieve, the argument (weights sum to one → redimensionalisation commutes →
prediction lies in the span → unconstrained LS projection is the exact minimum) is correct,
and I accepted it in round 3. Its force comes entirely from being an **impossibility**: no
member can do better, therefore the surrogate's near-wall advantage over that family is
real.

`bodyfit_bound.json` `B1.ratio: 0.004449987392641245` is the *same quantity* in the new
frame. But a bound that falls below 1 licenses nothing. It says a 800-dimensional subspace
of a smooth function space approximates a smooth function to five digits once the profiles
are aligned — which, in a coordinate system built precisely to align them, is close to an
approximation-theoretic identity. The minimum is attained by an unconstrained least-squares
weight vector with no convexity constraint and no relation to anything selectable from seven
scalars. Nothing in the paper exhibits a member of the family that achieves it.

The paper **has** the number that would test it, and reports it: `body.tex:638-644`,
published weights in the body-fitted frame, "a median of $\mathbf{0.38\times}$, with
$152/200$ cases beating Transolver near the wall… its *case-mean* ratio is $1.556\times$,
i.e. worse than Transolver, because a minority of cases carry a heavy tail." So the *only*
realizable body-fitted estimator the paper exhibits **loses to Transolver on the case-mean**
in the decisive band. Per-case `krr_mse` in the artifact confirms the tail is not small:
band-1 values in the body-fitted frame span at least $1.3\times10^{-1}$ to $7.5\times10^{1}$
across cases, a spread of ~600×.

`body.tex:644-646` states the restriction correctly and in the right place:
*"The coordinate-artifact finding therefore rests on the bound, which is unanimous, and not
on the deployed estimator, which wins at the median only. We do not claim the published
estimator beats a trained surrogate at the wall; we claim no member of this family is
barred from the wall by representation."* That sentence is exactly right and I will defend
it against anyone.

**Then four other places state the thing that sentence forbids:**

1. Contribution 1, `body.tex:185-186`: *"**The near-wall gap is an artifact of the frame**"*
   — bolded. Not "the bound". The gap.
2. Introduction, `body.tex:72-73`: *"What looked like something a surrogate represents and an
   interpolator cannot is a property of the frame the interpolator was asked in."*
3. Conclusion, `body.tex:1454`: *"Posed in the frame the flow is organised by, **it does not
   earn it at all**."* — a claim that the surrogate does not earn the boundary layer, resting
   on an 800-dof oracle projection while the actual estimator loses on the mean.
4. The **title** (`neuroforge_cfd.tex:10`, `neuroforge_cfd_elsevier.tex:41`): *"Near the
   wall, AirfRANS measures the coordinate system, not the model."*

And the abstract's comment block proves the split was noticed and the omission deliberate.
`abstract.tex:22-28`:

> *"The DEPLOYED estimator in the same frame wins at the median (0.38x) and on 152/200 cases
> but **LOSES on the case-mean (1.556×)**… The abstract claims the bound and never the
> estimator."*

As a guard against "parameter interpolation beats the surrogate", that is virtuous and I
credit it. As net effect: the abstract carries 0.0044× and 200/200 and **omits the one
number under which the surrogate still wins**. The reader is given the unanimous bound and
not the split estimator. This is the fourth consecutive round in which the careful form is
present in the manuscript and the strong form is what the abstract, title and contributions
say. In rounds 1–3 I called it a disclosure pattern; here the careful form is written down
in a standing comment and then kept out of the abstract, which is a stronger version of the
same finding and harder to attribute to oversight.

### (ii) The frame is textbook, and the paper cites nothing.

The winning frame is: arc length along the surface, wall distance normal to it, unrescaled
relative to each other (`bodyfit_bound.py:49-54`, "the standard boundary-layer coordinate").
The paper says so. That is Prandtl's coordinate system, and the paper's own docstring calls
it standard.

Worse for the novelty claim: **interpolating CFD fields across geometries on a common,
morphed mesh — precisely to obtain node correspondence so that boundary layers superpose —
is routine in POD/ROM aerodynamic databases.** I flagged this in round 3 as "unverified
because it is not established anywhere in this repository". It is now verified against
primary sources: mesh morphing to a common reference mesh is a standard remedy for geometric
variation across snapshots, and snapshot-POD requires the fields to be expressed on a common
mesh so that the reconstruction is a linear combination of the samples. See
[Elasticity-based morphing technique and application to reduced-order modeling](https://arxiv.org/pdf/2407.02433)
and
[Explicit interpolation-based CFD mesh morphing](https://internationalmeshingroundtable.com/assets/papers/2023/04-Malcevic-compressed.pdf).

`refs.bib` contains **exactly one** response-surface citation (`queipo2005surrogate`, line
397) and **zero** entries on mesh morphing, POD/ROM field interpolation across geometries,
snapshot correspondence, or body-fitted coordinates. Grepped: `morph|POD|proper
orthogonal|reduced.order|Amsallem|Iuliano|body.fitted|boundary.layer coordinate` — one hit,
`queipo2005surrogate`.

The consequence is severe and it hits both halves of contribution 1:

* The 25× `FAMILY-BOUND-CLOSED` bounds **physical-coordinate** parameter interpolation across
  geometries — a construction the response-surface/ROM community would not build for field
  data, for exactly the reason the paper measures (`body.tex:604-608`: 35% of the weight mass
  lands inside the training airfoil). An aerodynamics referee reads "we closed the family"
  and answers "you closed a family nobody uses."
* The 0.0044× `BODY-FITTED-BOUND-OPEN` recovers what that community does build. Framed as a
  discovery ("**we removed the mechanism**", `body.tex:66`, `:613`) it invites the reply that
  the mechanism was known and the fix is the default.

This project already took a desk reject on originality. A domain referee will reach this
paragraph within one page, and `refs.bib` will not defend it.

### (iii) Why this is worst rather than merely major

The carefully-scoped version survives (i) and (ii) intact, and it is a real contribution:

> *In physical coordinates, no weighting of the 800 training fields comes within 25× of the
> surrogate inside $0.005c$ — an exact bound, 200 cases, no free parameters. The mechanism is
> coordinate misalignment (7.8×, 35% inside the body), and re-posing the identical family in
> the standard boundary-layer frame removes the bound entirely (0.0044×), though the deployed
> estimator in that frame still loses on the case-mean. What the published protocol reports
> as a representational advantage is, at least in part, a statement about the frame the
> baseline was posed in.*

That is publishable and it is what the evidence supports. It is not what the title says, it
is not what contribution 1 says, and it is not what the abstract's closing sentence says.
**This is fixable by rewriting, which is why the score below is what it is** — but it is
fixable only by a rewrite that changes the title, and authors resist that, so I am stating
the cost plainly: leave the title as it is and a referee will read the paper against it and
find that the paper does not prove it.

**What would resolve it, in ascending cost.**
1. **Rewrite (mandatory).** Title → something the bound licenses ("Near the wall, AirfRANS
   measures the baseline's coordinate system" is *defensible*; "not the model" is not, see
   W4). Contribution 1's bold → "the near-wall **bound** is an artifact of the frame".
   `body.tex:1454` → the estimator-split sentence. Put `1.556×` in the abstract.
2. **One Related Work paragraph** on POD/ROM mesh morphing and snapshot correspondence,
   stating that the body-fitted arm is an import and that the contribution is the
   *measurement of its size on this benchmark* (0.0044/25.0 = $1.8\times10^{-4}$), not the
   idea. This is the same move the paper already makes beautifully for Kriging at
   `body.tex:325-340`. Do it again here and (ii) is neutralised.
3. **The experiment that would let you keep the strong claim:** a *realizable* body-fitted
   estimator — same seven scalars, same CV protocol, re-selected in the new frame — beating
   Transolver in band 1 on the case-mean, not the median. You are $1.556\times$ away. If it
   lands, contribution 1's bold sentence becomes true and the title becomes earned.

---

## 2. Every other substantive weakness, ranked

Severity: **[F]** fatal as-is · **[M]** major · **[m]** minor.
Triage: **R** rewrite · **E** experiment · **C** must be conceded.

### W1 [F][R] Three verified factual errors and one self-contradiction, all sediment from the three reversals. One of them is in the Conclusion.

These are not judgments. They are wrong numbers and stale sentences in the current file.

1. **`body.tex:1423` (Conclusion) quotes the retired n=3 number.**
   > *"the least-squares optimum over the span of all $800$ training fields, the exact lower
   > bound for any weighting of that family, is still $\mathbf{21\times}$ worse there."*

   The bound is $25.0\times$ (`abstract.tex:50`, `body.tex:62`, `:579`, `:1428` — five lines
   later in the *same paragraph*). $21.3\times$ is the superseded n=3 probe
   (`point_space_oracle_ls_n200.json` `meta.supersedes_n: 3`). The Conclusion states 21 and
   then 25 within one paragraph.

2. **`body.tex:662-665` says you did not run the experiment you just ran.**
   > *"The honest scope of the $25\times$ bound is therefore parameter interpolation of
   > AirfRANS fields *in physical coordinates*… **a body-fitted parameter interpolator is a
   > different method and we do not test it.**"*

   This is the round-3 sentence, left in place, sitting **three paragraphs after**
   `body.tex:613-660` reports the body-fitted arm on 200 cases. A referee who reads linearly
   hits "we do not test it" immediately after two pages of testing it. This one sentence will
   cost you a reviewer's trust in the whole document, because it is the kind of error that
   suggests the manuscript was not read end-to-end after the last reversal.

3. **`body.tex:1344` (Limitations) attributes the oracle's number to the estimator.**
   > *"a factor of a few against a measured $25\times$ for the whole family, and
   > $\mathbf{344\times}$ for the published estimator"*

   $343.6\times$ is the **best-single-field oracle** (`body.tex:566-568`). The published
   estimator is $175\times$ (`body.tex:638`) / $200.3$ in band 1 (`body.tex:571`). The paper
   spends a paragraph insisting the oracle is not a bound on the estimator and then labels it
   as the estimator in Limitations.

4. **The exclusion is misdescribed in the abstract and the Conclusion.**
   `abstract.tex:58`: *"it does not exist on $5.0\%$ of near-wall nodes, **behind the
   trailing edge**"*; `body.tex:1431`: same. But `bodyfit_bound.py:185-193`
   (`chain_endpoints`) and `bodyfit_bound.json:254` both name *"the sharp trailing edge**,
   leading-edge stagnation**"*, and the code drops any node whose nearest surface point is
   **either** chain endpoint. The leading-edge region is where the suction peak and the
   steepest surface-pressure gradients live. Describing a load-bearing exclusion by half its
   support, in the abstract, is the single most quotable error in this list.

**Triage: R**, one hour. But file nothing until a full linear read is done, because item 2
proves one has not happened since the reversal. Also fix `body.tex:1449-1454`, which now
carries sentences from two incompatible headlines back to back (*"a surrogate appears to earn
the boundary layer… Posed in the frame the flow is organised by, it does not earn it at
all"*).

### W2 [M][R+E] "The verdict is not carried by a tail" rests on four statistics, two of which are the same number by construction, and none of which is paired.

`body.tex:588-593`:
> *"Across the $200$ cases the ratio of means is $25.00$, the mean of ratios $25.00$, the
> node-weighted pooled ratio $25.04$ and the median $19.8$… These four estimators were named
> in the amendment before the run precisely so that a verdict resting on a handful of cases
> would be visible rather than averaged away; it does not."*

The Transolver denominator is a **single scalar**, not a per-case value:
`bodyfit_bound.json` `B1.transolver_band_mse_u: 1.157649054562416`. Therefore
`mean_of_ratios` is `ls_mse_casemean / 1.157649` — algebraically identical to the ratio of
means. The artifact shows it to the last bit:

```
"ratio":          0.004449987392641245
"mean_of_ratios": 0.004449987392641246     (bodyfit)
"ratio":          25.208590391775115
"mean_of_ratios": 25.208590391775115       (physical, identical)
```

Two of the four "independent estimators" differ in the 16th significant figure. They carry
zero robustness information and are presented as if they carried it.

The deeper consequence: **every per-case statistic in that paragraph is the interpolator's
own residual rescaled by a constant.** "161 cases clear the threshold of 10 on their own",
$p_5$ 4.7, $p_{95}$ 62.6, median 19.8 — none is a paired per-case comparison, and none
contains any information about case-to-case variation in Transolver's band error. Same on the
body-fitted side: "zero cases clear 10 and all 200 fall at or below 1", worst 0.093.

This is cheap to fix and the data exists: gate **G2** reproduces the deployed backbone's
per-band **per-node** MSE at $0.00\mathrm{e}{+}00$ (`body.tex:505-507`), so per-case
Transolver band errors are in hand. **Triage: R** (drop the duplicate statistic and say the
denominator is a band constant) **+ cheap E** (recompute the four statistics paired per case).
The distinction is between "the verdict is not carried by a tail" and "the interpolator's
residual is not carried by a tail", which is not the same claim.

### W3 [M][R] The exclusion criterion is parameter-free in form and symmetric in application, but post hoc in origin — and the manuscript says only the first two.

The brief asked whether the 4.98% exclusion survives me. Partly. Here is what does not.

`body.tex:654-656` presents it as design:
> *"Excluded by membership rather than by distance---a node is dropped iff its nearest
> surface point is a chain endpoint, with no tolerance---the frame is exactly faithful"*

The provenance is in the script and nowhere in the paper. `bodyfit_bound.py:34-41`: *"A
wall-normal coordinate does not exist everywhere in the strip, and **amendment B-1 measures
where**"*. `bodyfit_bound.py:111`: *"**GB2b** ROUND TRIP (amendment B-1, **replacing the
original GB2**)"*, with the original gate recorded as having *"fired at 4.97e8"*. So:

**the original fidelity gate failed; the amendment that responded to it simultaneously
(a) replaced the gate and (b) introduced the node exclusion.** The exclusion criterion was
*discovered from the failure*, not registered before it.

I verified the innocent reading and it holds. The docstring records the surviving GB2b
failures as *"a single node each (1 in 103,533, **at the nose stagnation point**)"* — a chain
endpoint, hence excluded — and `bodyfit_bound.json` `gates.GB2` reports
`GB2b_frac_above_tol: 0.0` and `GB2b_worst_scaled: 2.06e-13`. The paper's
`body.tex:657-658` claim ("*not one node above $10^{-9}$*") is accurate **post-exclusion**.
Note also that the replacement gate was written with a 0.01% failure allowance which then
went unused. That is to the authors' credit and I say so.

But the objection stands in its correct form, and it is the one place where this paper's
strongest asset is being claimed for something that did not earn it: **pre-registration is
the paper's credibility engine, and a criterion that removes 4.98% of the nodes — including
the leading-edge stagnation region and the entire trailing-edge wake fan — was shaped by
where the frame failed, and the manuscript presents it as construction.**

The innocence check (25.21× vs 25.0×, `body.tex:632-634`) is necessary and I credit it, but it
is asymmetric by construction: it shows the excluded nodes are not anomalous *for the
physical arm*. It cannot show anything about the body-fitted arm there, because the arm is
undefined there. The excluded set is selected by where the winner does not work.

**Triage: R** (three sentences: name amendment B-1, say the original GB2 failed, say the
criterion was derived from that failure and is parameter-free in form but post hoc in origin)
**+ E if you want the strong version** (report the band-1 exclusion fraction separately — the
JSON gives only a strip-wide 4.98% — and a fallback-blended arm scored on 100% of the strip).
Also reconcile `bodyfit_bound.py:35`'s "5.08%" with the paper's/JSON's 4.98%.

### W4 [M][R] The title generalises from one non-learned family to the benchmark, and to "the model".

*"Near the wall, AirfRANS measures the coordinate system, **not the model**."*
`abstract.tex:60`: *"Near the wall, this benchmark measures the coordinate system."*

Transolver was never re-posed in any frame. Its band-1 error is one scalar, 1.157649, used as
a fixed denominator in both arms. Nothing in this paper shows that *a model's* score depends
on its coordinate handling. What is shown is that *one non-learned family's achievable
bound* moves by $1.8\times10^{-4}$ under a reparametrisation. Generalising from that to "the
benchmark measures the coordinate system, not the model" is a claim about every method scored
on AirfRANS, evidenced on one method that is not a model.

There is a version that is earned: *near the wall, this benchmark rewards having the right
coordinates, and the baseline literature's frame is what its published deficit measures.*
**Triage: R.**

### W5 [M][R] The winning arm dissolves the paper's own input-asymmetry framing, and the paper does not say so.

`body.tex:474-476`: *"The input asymmetry runs **against** the baseline: the surrogates
receive the full $7$-channel geometry and freestream encoding per case, the interpolator
receives seven scalars."* That sentence is load-bearing throughout §`sec:interp` — it is why
the interpolator losing is meaningful at all.

The body-fitted arm receives, per test node: the surface point chain and normals of the test
case (`load_test_surface`), the arc-length parametrisation of both surfaces, and the wall
distance (`bodyfit_bound.py:212-224`). That is a hand-designed geometry encoder. The
information is derivable from the case name (the NACA digits determine the section), so I do
**not** allege leakage — but "no network, no geometry encoder, no flow-field learning"
(`body.tex:462-463`) is false of the arm that produces the new headline.

Stated honestly this is a *better* result, not a worse one: *a hand-built, parameter-free
geometric feature map recovers most of what a learned geometry encoder buys near the wall.*
That is a sentence an ML-for-PDE referee will find interesting. As written, the paper claims
the same seven-scalar austerity for both arms. **Triage: R**, one paragraph.

### W6 [M][E] There is still no evidence anywhere in the manuscript — or in the repository — that this Transolver is a competent Transolver. My round-3 W5 is open, and the brief says it is closed.

I was told: *"W5 answered: in the AirfRANS paper's own convention our Transolver is
11.3×/6.7× better than the best published baseline."* **I cannot find that number anywhere.**

* Not in the manuscript. Grepping `docs/paper/*.tex` for `11.3`, `6.7`, `relative.?L2`: the
  only live hits are unrelated (`body.tex:520`, `:653`, `:801`, `:1079`) plus
  `body.tex:1166`, which is the *declination*. `7.4→11.3` appears only in
  `sections/residual_floor_theorem.tex` and `sections/trust_layer_removed.tex`, which
  `body.tex:20` states are not `\input` by either build, and it is a certificate band width,
  not an AirfRANS accuracy figure.
* Not in `results/`. The 38 files matching `rel_l2|relative_l2|relL2` are all from the
  removed trust-layer / selective-prediction work (retained relative-$L_2$ under coverage) —
  `results/selective/`, `results/uq_ensemble/`, `results/residual_descent/`,
  `results/control/`. None is a comparison of this backbone against a published AirfRANS row.
* Not in `docs/paper/review/`. No review document carries the phrase, the number, or a
  calibration stage; the only `tab:airfrans-sota` discussions are about transcription
  (`consistency_audit.md:1161-1164`, `audit_closeout.md:141`).

So the triage is **E, not R**: this control does not exist. That matters, because it is one
of the three remedies I priced in round 3 and it is the one the programme did not deliver.

The manuscript's position is still `body.tex:1163-1169`:
> *"We deliberately do **not** place a row of our own in a cross-method leaderboard, because
> AirfRANS results are reported in mutually incompatible conventions… so any shared cell
> would mislead."*

and `tab:airfrans-sota`'s caption still reads *"Our rows are **not** comparable."* Since the
entire paper is a claim that measurement conventions decide conclusions, declining to place
your own arm on **any** commensurable scale — on convention-incompatibility grounds — is the
one methodological position this paper cannot hold. And it matters directionally:
`body.tex:876` reports 62.8% of Transolver's $p$ error beyond $0.5c$, so a better-trained
Transolver shrinks the $8.4\times$ that is the second finding's whole subject.

Note how close the manuscript already is. `tab:interp_std` reports per-channel **standardised**
volume MSE — nominally the AirfRANS paper's own convention — and `tab:airfrans-sota` reports
the published baselines in that convention, in the same document. The paper declines to relate
them, and `body.tex:1335-1338` gives the honest reason (its divisors are computed on the
$r128$ raster, not in point space, so the normalisation is "faithful in spirit but not
byte-identical"). **Fix the normaliser, report the row with the protocol differences named,
and W6 dies.** One evaluation pass on the three deployed checkpoints.

### W7 [M][E] The decisive experiment computes the pressure channel and the manuscript reports only $u$. This is the highest-value, zero-compute item on the list.

`bodyfit_bound.json` `meta.channels: ["u","p"]`, and every per-case row carries a `"p"` block
with `ls_mse` / `krr_mse` / `best_single_mse` in **both** frames (e.g. lines 1137, 1185, 1256,
1304 …). The summary blocks `B1`, `B1_physical_same_nodes`, `B2_vs_physical` and
`diagnostics` are $u$ only, and so is the manuscript.

Pressure is the channel the paper itself singles out as the one that matters and the one
where the surrogate's near-wall advantage does not exist: `body.tex:896-910`, *"the two arms
are within $1.51\times$ on $p$"*, and *"the surrogate's near-wall advantage is on $u$, $v$
and $\nu_t$ and **not** on $p$, so the force integrals do not inherit it."* So the decisive
experiment has the force-bearing channel in hand and does not report it.

I make no allegation about which way it falls — I have not computed the aggregate and the
per-case values I sampled do not settle it. I am saying an aerodynamics referee will ask for
it in one line, and that *"we computed it and did not report it"* is the worst available
answer and the one the artifact forces. **Triage: E**, zero new compute — the numbers are in
the file. Report `B1` on $p$ in both frames, with whatever verdict the same threshold returns.

### W8 [m][R] The $\lambda$-sensitivity control exists for one arm, is unreported, and is absent from the other.

`point_space_headtohead.py:1029` and `:1243` compute both `ls_mse` ($\lambda = 10^{-10}\,
\bar{t}$) and `ls_mse_ridge` ($\lambda = 10^{-6}\,\bar{t}$). In band 1 of the first case,
`point_space_oracle_ls_n200.json` gives 6.354 vs 6.490 — 2.1%, immaterial to the 25× verdict.
In band 2 it is 0.00103 vs 0.00179 — 73%. Neither appears in the manuscript.

`bodyfit_bound.py:359` computes **only** the $10^{-10}$ solve; there is no ridge arm for the
body-fitted bound at all.

I want to be precise about direction, because the naive version of this attack is wrong: a
ridge returns a residual $\geq$ the unregularised minimum, so ridge inflation can only mean
the body-fitted true minimum is *below* 0.0044× — it cannot overturn
`BODY-FITTED-BOUND-OPEN`. Conditioning bears on a **verdict** only for the *physical* arm,
where the minimum must exceed 10 — and that is the arm that has the control. So this is an
asymmetry in the reported control set, not a threat to a number. **Triage: R**, one clause:
"the bound is insensitive to the regularisation over four decades of $\lambda$ (6.354 →
6.490 in the verdict band)". It is free and it removes an obvious referee question.

### W9 [m][R] The availability statement is still falsified by `git status`.

`body.tex:1510`: *"Every headline number maps to a committed script and result file"*, plus a
SHA-256 manifest. At the start of this session `git status` shows
`?? scripts/make_fig_bandratio.py`, `?? results/figures/fig_bandratio.pdf`,
`?? results/figures/fig_bandratio.png` — the paper's **only** figure and its generator, both
untracked. This is the third consecutive round in which I have flagged an uncommitted
load-bearing artifact against a manifest that invites a referee to check. It is trivial and
it is still open. Also confirm `bodyfit_bound.json` and `point_space_oracle_ls_n200.json` are
tracked and hashed before filing.

### W10 [m][R] "Better on 200/200 cases" is ambiguous in the abstract.

`abstract.tex:55-56`: *"that bound falls to $\mathbf{0.0044\times}$ and is better on
$\mathbf{200/200}$ cases"*. In the artifact, `B2_vs_physical.n_cases_bodyfit_better: 200` is
better than the **physical arm**; the separate fact that all 200 fall at or below 1 is better
than **Transolver**. Both are true, so this is not an error — but the two claims are 200/200
for different reasons and the abstract does not say which. `body.tex:626-630` disambiguates;
the abstract should too.

### W11 [m][C] The frame relabels positions but not the vector basis.

`bodyfit_bound.py:330-333` transfers `yhat_j` — the nondimensional field components — in the
**global** Cartesian basis. $u$ and $v$ are not rotated into wall-tangent/normal components at
the matched $(s,n)$. So "the frame the flow is organised by" (`body.tex:1454`) relabels where
a value is read, not what the value is. I do not think this is a defect — declining the
rotation keeps the arm parameter-free and the result is *stronger* for holding without it —
but the rhetoric outruns the construction, and a referee who reads the code will notice.
One clause. **Triage: C/R.**

### W12 [m][C] Scope, conceded.

One benchmark, two dimensions, one learned architecture (three seeds), the `full` split only
for the native head-to-head (`body.tex:1354-1355`), one band and one channel for every bound,
the wall band not read, no 3-D. All stated in Limitations, accurately. None is individually
fatal for a measurement paper at a generalist computational-science venue.

---

## 3. Triage summary

**Blockers — must be fixed before filing (one day, all rewriting):**
W1 (four factual errors + the double-headline Conclusion — and a full linear read, because
`body.tex:662-665` proves one has not happened), §1's title/contribution-1/abstract rescope,
§1(ii)'s Related Work paragraph, W3's provenance sentences, W4, W5, W9.

**Cheap experiments that materially strengthen it, in priority order:**
1. **W7** — report the $p$ channel from `bodyfit_bound.json`. Zero new compute. This is the
   one a domain referee names, and the one you cannot decline.
2. **W2** — the paired per-case recompute. The data exists behind gate G2.
3. **W6** — the relative-$L_2$ / standardised row. It does **not** exist in the repository;
   this is a run, not a paste.
4. **§1(3)** — a realizable body-fitted estimator that beats Transolver on the case-mean.
   Only needed if you want to keep the strong form of contribution 1.

**Must be conceded:** 2-D; one benchmark; one architecture; one band, one channel; the
body-fitted bound is a bound and not an estimator; the frame is undefined on ~5% of the strip
including the leading edge.

---

## 4. Claims overreaching their evidence — verbatim, with corrections

1. `body.tex:185-186` (contribution 1) — *"**The near-wall gap is an artifact of the frame**"*
   → **"The near-wall *bound* is an artifact of the frame: re-posed in the standard
   boundary-layer coordinate the family is no longer barred from the wall (0.0044×,
   200/200). The deployed estimator in that frame wins at the median (0.38×) and loses on
   the case-mean (1.556×), so we claim the removal of the barrier and not the closure of the
   gap."** The corrected form is already written at `body.tex:644-646`; propagate it.

2. **Title** (`neuroforge_cfd.tex:10`, `neuroforge_cfd_elsevier.tex:41`) — *"Near the wall,
   AirfRANS measures the coordinate system, not the model"*
   → the "not the model" half is unevidenced; Transolver is a fixed scalar denominator in
   both arms and was never re-posed. **"Near the wall, AirfRANS measures the baseline's
   coordinate system"**, or a form that names the family.

3. `body.tex:1454` (Conclusion) — *"Posed in the frame the flow is organised by, **it does not
   earn it at all**."*
   → **"Posed in that frame, the family is no longer barred from the wall by representation;
   what a surrogate buys there is the coordinates, and a realizable member of the family
   still loses on the case-mean."**

4. `body.tex:1423` (Conclusion) — *"is still $\mathbf{21\times}$ worse there"*
   → **$25.0\times$.** Stale n=3 number, contradicted five lines later at `:1428`.

5. `body.tex:662-665` — *"a body-fitted parameter interpolator is a different method and **we
   do not test it**."*
   → **delete.** You tested it three paragraphs earlier.

6. `body.tex:1344` (Limitations) — *"and $\mathbf{344\times}$ for the published estimator"*
   → **"$175\times$ for the published estimator; $344\times$ is the best-single-field oracle,
   which is not a bound on the estimator."**

7. `abstract.tex:58` / `body.tex:1431` — *"it does not exist on $5.0\%$ of near-wall nodes,
   **behind the trailing edge**"*
   → **"…at both ends of each surface chain — behind the sharp trailing edge and at the
   leading-edge stagnation point"** (`bodyfit_bound.py:185-193`; `bodyfit_bound.json:254`).

8. `body.tex:588-593` — *"the ratio of means is $25.00$, the mean of ratios $25.00$… **These
   four estimators**…"*
   → they are three, not four: the denominator is a band constant (1.157649), so mean-of-
   ratios equals ratio-of-means identically. Report three, and say the comparison is unpaired.

9. `body.tex:654-656` — *"Excluded by membership rather than by distance… with no tolerance"*
   → add: **"The criterion was introduced by amendment B-1 in response to the original
   round-trip gate failing at the nose stagnation point; it is parameter-free in form and
   applied identically to both arms, but it was derived from that failure rather than
   registered before it."**

10. `body.tex:462-463` / `:474-476` — *"There is no network, no geometry encoder and no
    flow-field learning"* / *"the interpolator receives seven scalars"*
    → true of the physical arm; **false of the body-fitted arm**, which consumes the test
    case's surface chain, normals and wall distance. Say so, and say the information is
    still a function of the case name.

11. `body.tex:1510` — *"Every headline number maps to a committed script and result file"*
    → not while `scripts/make_fig_bandratio.py` is untracked.

---

## 5. Questions to the authors (trap-aware)

1. `bodyfit_bound.json` gives `B1.ratio = 0.0044` for the bound and `body.tex:642` gives
   $1.556\times$ for the deployed estimator's case-mean — a factor of 350 between what is
   achievable and what is achieved. On what basis does contribution 1 assert that the
   near-wall **gap** is an artifact of the frame, rather than the **barrier**?
2. Name one member of the family — any kernel, any bandwidth, selected by any protocol that
   does not see the test answer — that attains anything within two orders of magnitude of
   0.0044×. If none exists, what does a bound below 1 establish?
3. Interpolating flow fields across geometries on a common morphed mesh, to obtain node
   correspondence so that boundary layers superpose, is standard practice in POD/ROM
   aerodynamic databases. `refs.bib` cites one response-surface paper and nothing on
   morphing, POD field interpolation or body-fitted coordinates. Is the physical-coordinate
   family you bounded at 25× a family that community builds — and if not, what does
   `FAMILY-BOUND-CLOSED` close?
4. `bodyfit_bound.json` `transolver_band_mse_u` is a single scalar. Given that, in what sense
   are "the ratio of means" and "the mean of ratios" two estimators, and what do the p5/p95
   and the "161 cases clear 10" figures tell a reader about Transolver's per-case variation?
5. `bodyfit_bound.py:111` records GB2b as "replacing the original GB2", which fired. The same
   amendment introduced the interior-projection exclusion. Was the exclusion criterion chosen
   before or after the original gate failed, and why does the manuscript describe it only as
   construction?
6. What fraction of the **verdict band** (0–0.005c) is excluded, as opposed to the 4.98% over
   the whole strip? And what does the body-fitted bound read if the excluded nodes are
   scored with a physical-frame fallback, so that both arms cover 100% of the strip?
7. `bodyfit_bound.json` carries the `p` channel in both frames for all 200 cases. What is
   `B1` on `p`, and what verdict does the same threshold return? Why is it not in the paper?
8. `body.tex:1163-1169` declines any commensurable row on convention-incompatibility grounds,
   while `tab:interp_std` and `tab:airfrans-sota` sit in the same document in nominally the
   same convention. Your entire thesis is that conventions decide conclusions. What is this
   Transolver's number against the published baselines, and why is the one calibration your
   reviewer priced in round 3 absent from `results/` entirely?
9. `body.tex:474-476` says the input asymmetry runs against the baseline. The body-fitted arm
   consumes the test case's surface chain, normals and wall distance. Is that arm still
   "seven scalars, no geometry encoder"?
10. `body.tex:1423` says $21\times$ and `body.tex:1428` says $25\times$, in the same
    paragraph; `body.tex:662-665` says you did not run the experiment reported at
    `body.tex:613-660`. When was this file last read end to end?

---

## 6. Score, verdict, and the round-3 price

### Score: **8 / 10.** Accept after minor-to-moderate revision at the Journal of Computational Science. Now genuinely arguable at JCP *if and only if* the title, contribution 1 and the Related Work gap are fixed. Still no at NeurIPS/ICML.

Against my **3**, **5** and **6**: this is the largest single jump this project has had from me,
and it is earned by experiments, not by prose.

### On round 3's "→ 9 with the wall-normal interpolator arm, whichever way it falls" — I am honouring it, with one point of named deductions.

I refused round 2's price on the grounds that the result was the opposite of what I
anticipated. Refusing twice on the same grounds would make me useless as a pricer of
remedies — and, decisively, my round-3 text pre-committed to *"whichever way it falls"* and
pre-described this exact outcome as valuable: *"if it closes, the paper has a better result
about coordinates and a third consecutive honoured self-refutation."* It closed, and that is
what happened. **So the price is honoured.**

What I priced was the **experimental programme**, and the programme mostly delivered:

* the wall-normal arm, pre-registered, run blind, reversing the headline → **9-grade work**;
* the n=200 least-squares bound under a bit-identical reproduction gate → **delivered**;
* the relative-$L_2$ calibration → **not delivered**, and not merely unwritten: it is absent
  from `docs/paper/*.tex`, from `results/` and from `docs/paper/review/` (W6). One of the
  three remedies I named did not happen.

What I am scoring is the **manuscript**, and the manuscript is one point behind the
programme, for reasons I can itemise line by line rather than by feel:

| deduction | evidence |
|---|---|
| the bound/estimator conflation in the title, contribution 1, intro and Conclusion | `body.tex:185-186`, `:72-73`, `:1454`, title; corrected form already present at `:644-646` |
| a coordinate claim with zero coordinate/ROM citations | `refs.bib` — one hit on the whole family of search terms |
| three stale numbers and one self-contradiction, one of them in the Conclusion | `body.tex:1423`, `:662-665`, `:1344`, `:1449-1454` |
| a load-bearing exclusion misdescribed in the abstract, with undisclosed post-hoc provenance | `abstract.tex:58` vs `bodyfit_bound.py:185-193`, `:111` |
| the decisive experiment's `p` channel computed and unreported | `bodyfit_bound.json` `meta.channels` |
| the robustness paragraph double-counting one statistic | `bodyfit_bound.json` `ratio` == `mean_of_ratios` |
| one of the three priced remedies not run | W6 — absent from `.tex`, `results/` and `review/` |

Six of the seven are a day's work or less, and four are pure text. **Fix them, run W7's
zero-compute `p` arm, and I am at 9 without argument**, and I would say so in a signed report.

### "Three headline reversals — rigour or instability?"

**Rigour in the science; instability in the document, and only there.**

The reversals themselves are the best thing about this project and I will not hedge it. Three
times I named the experiment that would kill the lead claim; three times it was built, run
under a rule fixed in advance, and reported when it fired against the authors. The
`bodyfit_bound.py` docstring even records, before the run, that a reviewer had named it and
that the two previous such experiments reversed the headline — and it ran anyway. That is a
research practice almost nobody in this field actually has, and an editor should be told it
exists.

But each reversal has left textual sediment, and the sediment is now load-bearing: a
Conclusion that quotes the retired bound and the current one in one paragraph, a scope
paragraph that denies an experiment reported two pages earlier, a Limitations entry that
mislabels the oracle as the estimator, a Conclusion that states both the old headline and the
new one in consecutive sentences. **A referee cannot distinguish "reversed three times under
rigour" from "unstable" by reading the science — they will distinguish it by reading the
document, and right now the document reads as the second.** That is the entire remaining
difference between 8 and 9, and it is the cheapest point this project has ever been offered.

### "What is left that is novel, given the baseline is decades-old RSM and the frame is textbook?"

Answer honestly and it is enough for the venue; answer defensively and it collapses. What is
novel, in descending order of durability:

1. **The measurement of the node-versus-area weighting on a benchmark whose literature never
   states it** — $312\times$ in the innermost band, $51.5\times$ inside $0.05c$, a sign
   reversal, five discretisations with the least favourable reported, and the checkable
   observation that no published AirfRANS field comparison states its weighting. Nobody has
   this number. It is actionable and portable.
2. **The representation ceiling**, $413\times$/$495\times$, with the projection-inversion
   reframe now correctly stated at `body.tex:198-209`, shipped as a reusable diagnostic for
   any cloud+raster pair.
3. **The size of the coordinate effect**: $1.8\times10^{-4}$ between the two frames' bounds
   on identical nodes in one run. The *idea* that body-fitted coordinates help is textbook;
   the *magnitude*, measured against a trained SOTA surrogate on 200 cases with a
   pre-registered rule, is not published anywhere.
4. **The negative on surrogate-side residual descent** (200/200, Armijo, two stencils, three
   step rules, pinned and unpinned) — still the single strongest experiment in this project's
   history, and still carrying controls it says live in another paper (`body.tex:1231-1232`).

What is **not** novel and must be said first, in the paper's own voice, exactly as it already
does for Kriging at `body.tex:325-340`: the estimator, the frame, and the observation that
boundary layers must be aligned before fields are combined.

### The single flaw that most justifies rejection

**The paper proves that the near-wall *barrier* is a coordinate artifact and asserts, in its
title, its first contribution, its introduction and its conclusion, that the near-wall *gap*
is — while its own artifact shows the only realizable member of the family in that frame
still losing to the surrogate on the case-mean, and its own abstract comment block records
the decision to omit that number.** Everything else on this list is a day of editing. This one
is a claim, and it is stated four times in the four places a referee reads first.
