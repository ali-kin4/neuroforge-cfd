# Reviewer 2, round 3 — "What a neural flow surrogate buys is near-wall representation, and the scoring measure decides the ranking"

Adversarial pre-submission review of `docs/paper/body.tex` + `abstract.tex` (read in full),
against `results/interpolation/`, `docs/paper/review/{measure_asymmetry,point_space_headtohead,jocs_rebuild,restructure}.md`,
and the committed artifacts. Round 1 (`r2_holes.md`) scored **3/10**; round 2
(`r2_round2.md`) scored **5/10**. The comparison, and an explicit accounting of round 2's
"→9/10" promise, is in §6.

Two things I owe the authors before the attacks, because they change what is worth arguing about:

* **Round 2's named fatal flaw was run, and it went against the paper.** That is not a small
  thing and I will not pretend otherwise. The 8.4× was withdrawn as a standalone claim on
  the strength of a pre-registered rule (`D3`) that fired against the authors' own headline.
  Round 1's fatal flaw was also run (direct residual descent), and also honoured. **This
  project has now, twice in a row, run the experiment I said would kill its lead claim, and
  reported the kill.** I have reviewed ~200 papers and I can count on one hand the ones that
  did that once.
* **Three of my round-2 items are closed.** §2.2 (the RSM/Queipo lineage paragraph and eight
  bib entries) is in at `body.tex:276-319`. §2.5 (the resolution ladder's PARTIAL verdict) is
  in at `body.tex:958-994` and in Limitations. §2.3 (the force columns' labelling) is in at
  `body.tex:656-665` and `sec:forces`. §2.6 (cut the trust layer) was done. §2.10 is
  half-done (see W7). I do not re-raise any of them and a real R2 will not either.

So the review below is against the paper that now exists, not the one I rejected twice.

---

## 1. The single worst genuine objection

> **The paper's surviving positive claim — "a surrogate earns the boundary layer and no
> interpolation reaches it" — is a claim about a family the paper itself defines to exclude
> the fix, proved by an instrument the paper itself concedes is invalid, and bounded validly
> only at n = 3. Strip those three layers and the finding is: a 7.35M-parameter model that is
> given the geometry beats a seven-scalar model that is not, in the one region where geometry
> is the whole problem. That is not a finding. It is arithmetic.**

Three separable defects compound into one objection. Each is anchored.

**(i) The oracle that carries the headline number is not a bound, and the paper says so.**
`body.tex:535-544` runs the `P3` oracle — best single training field per case and band,
chosen knowing the test answer — gets 397.7 on $u$ against Transolver's 1.158 in $0$–$0.005c$,
reads $Q = 343.6$ against a pre-registered threshold of 10, and returns the verdict
**`NO-PARAMETER-COMBINATION`**. Then, four lines later, at `body.tex:543-546`:

> *"a minimum over **single** fields is not a lower bound on linear **combinations** of them,
> and in that innermost band the fitted combination indeed beats the oracle (200.3 against
> 397.7)"*

The instrument is beaten by the un-oracled estimator it is supposed to upper-bound. A
pre-registered rule does not rescue an invalid estimator — pre-registration protects against
*p*-hacking, not against measuring the wrong quantity. The verdict string
`NO-PARAMETER-COMBINATION` is a claim about parameter combinations licensed by a statistic
the authors concede says nothing about parameter combinations. And the abstract
(`abstract.tex:35-37`) leads with the 344×, not with the 173× the actual estimator achieves
in that band (`point_space_headtohead.md` §3, `KRR combination` column: 200.3 / 1.158) and
not with the 21× that is the only valid number.

**(ii) The one valid bound is n = 3 cases, one channel, one band.**
`results/interpolation/point_space_oracle_ls.json`, `meta.n_ls: 3`. The least-squares
projection argument at `body.tex:544-551` is correct and I accept it: because the weights sum
to one, redimensionalisation commutes, the prediction lies in the span of the 800
redimensionalised fields, and the unconstrained LS projection is a genuine lower bound for
any weighting, convex or not. That is a good argument. It is the load-bearing control for the
entire positive claim — it is what converts "our baseline lost" into "no baseline of this kind
can win" — and it is run on **three test cases**.

*The authors themselves* price the full version, at `jocs_rebuild.md:417-419`: *"it cost 301 s
for 3 cases, so 200 cases is ~5.6 h on the same path"* — against the 29 CPU-core-hours already
spent on the head-to-head. I have not verified that scaling and I am not asserting it: the
same document's §8 reports 7350 s for the 160,000 cloud-to-cloud transfers the interp stage
required, so if the n=200 projection needs transfers it does not already have cached, the true
cost is nearer that figure than theirs. Either way the objection stands in the authors' own
units: **they costed the scale-up, it is the single most important number in the paper, and
they shipped n=3.** If their own estimate is wrong, that belongs in the manuscript — an
expensive control honestly priced is a limitation; an undersized one sitting unmarked beside
four 200-case figures is an exposure.

There is also **no write-up of this stage in any review document** — `jocs_rebuild.md:206-207`
states it plainly: *"There is no write-up of this stage in `point_space_headtohead.md` — that
file has no section on amendment 2."* So the paper's load-bearing control has an n of 3, a
JSON, a script docstring, and no report.

**(iii) The family is defined to exclude the coordinate system in which the measured
mechanism disappears — and the refusal is contradicted by the paper's own feature vector.**
`body.tex:562-572` measures the mechanism beautifully: inside $0.005c$ a query lands on
average $7.8\times$ further from the training case's wall than from its own, and $35\%$ of
the weight mass falls **inside** the training airfoil ($47\%$ at the wall row,
`point_space_headtohead.md` §2.4). That is a complete diagnosis, and it is a diagnosis of
**coordinate misalignment**, not of "representation". Then `body.tex:572-576`:

> *"What we deliberately did **not** build is a body-fitted or wall-aligned variant of the
> interpolator: it would need a blend length scale and an arc-length correspondence with no
> published provenance, and a reviewer would reopen it by attacking the tuning."*

This will not survive contact. **The paper already computes the analytic camber line.**
`body.tex:432-433` lists among its seven features *"the camber line at
$x/c\in\{0.25,0.5,0.75\}$"* — so the NACA 4-digit mean line is already in the code path. The
released package ships a NACA generator and an SDF/mask module (`src/neuroforge/geometry/`),
and the whole paper is built on a signed distance function that AirfRANS itself also
distributes as column 4 (`measure_asymmetry.md` §2, gate G6, agreement 99.51%/99.90%). A
wall-normal coordinate $(s, n)$ — normalised arc length along the analytic surface, signed
distance normal to it — requires no blend length and no free parameter: it is exactly
determined by the four digits. **The refusal is therefore a choice, not a constraint**, and it
is precisely the control that decides whether the near-wall gap is "representational" (the
paper's word, `body.tex:64`) or merely coordinate-systemic.

*A secondary line, which I flag as unverified because it is not established anywhere in this
repository.* My understanding is that the aerodynamic-database / POD-ROM tradition this paper
claims as lineage builds flow databases over shape parameters on a **common morphed mesh**
with node correspondence, precisely so that boundary layers superpose — in which case the
family the paper bounded is not the family that community uses, and an aerodynamics referee
will say so. I cannot cite that from anything committed here: `refs.bib` carries exactly one
RSM entry (`queipo2005surrogate`), and it is also possible that the tradition interpolates
*integrated coefficients* rather than fields, which would make field interpolation unusual in
any coordinate system. **The authors should check this, and they should note that the
objection does not need it.** Objection (iii) rests entirely on the coordinate-misalignment
mechanism the paper measures itself; the norm claim, if it holds, only sharpens the framing.

The consequence: the family the paper closed is *physical-coordinate* parameter
interpolation. `body.tex:576-579` says so correctly — *"The honest scope of this result is
therefore parameter interpolation of AirfRANS fields **in physical coordinates**"* — and then
`abstract.tex:37` widens it to *"the exact least-squares bound over **the whole family**"*
and `body.tex:1309` to *"**no member of the interpolation family** reaches it."*

**Why this is the worst objection rather than merely a major one.** It is not fixable by
rewriting: rewriting produces the scoped claim *"physical-coordinate kriging over seven
scalars cannot represent a boundary layer, for the reason that its constituent fields'
boundary layers are in the wrong place"* — which is true, measured, and not a result anyone
will cite. It is fixable only by the body-fitted arm. And that arm is the **third consecutive
round** in which the paper's most valuable experiment is the one named in Limitations and not
run (round 1: direct residual minimisation; round 2: the point-space head-to-head; round 3:
the wall-normal interpolator). In both previous rounds the experiment, once run, **reversed
the headline**. The prior on this one is not favourable to the current claim, and the authors
should want to know before an editor does.

**What would resolve it, in ascending cost.**
1. **Rewrite, mandatory regardless:** propagate the `in physical coordinates` scope into
   `abstract.tex:37` and `body.tex:1309`, and demote 344× behind 21× everywhere.
2. **Scale the LS bound to n = 200** (no training, no new code). This converts the one valid
   number from a probe into a result.
3. **Build the wall-normal arm.** $(s,n)$ from the analytic NACA surface, no free parameter;
   if a blend is needed, sweep the blend length over a decade and report the envelope so
   "you tuned it" is unavailable. Run the same `P1`/`P3` rules. Either outcome is a better
   paper: if it still loses at the wall, finding 1 becomes decisive and non-obvious; if it
   closes the gap, the paper's headline becomes *"what a surrogate buys is the right
   coordinates, and here is what that costs"* — which is a more interesting claim than the
   one currently made.

---

## 2. Every other substantive weakness, ranked

Severity: **[F]** fatal as-is · **[M]** major · **[m]** minor.
Triage: **R** rewrite · **E** experiment · **C** must be conceded.

Ordered by what a referee encounters first, not by how hard each is to fix. W1 and W2 are both
one-paragraph rewrites; they are at the top because they are the two an adversary finds
immediately and cannot be argued out of.

### W1 [M][R] The measure-dependence "finding" is, by the authors' own pre-registered rule, a retraction — and it is the only registered verdict in the paper that is not named.

`measure_asymmetry.md` §6, rule **D3**, committed in `cdf08f5` before the number existed:

> `>=4` SURVIVES; `>=1.5` SCOPED; `>1` WITHDRAW; `<=1` **ARTIFACT: the 8.4x is a measure
> choice; say so plainly and withdraw it.** Measured: **0.440**.

The manuscript names, in small caps, every other registered verdict it holds:
`not-competitive-at-native` (×3), `family-bound-closed` (×2), `no-parameter-combination`,
`confirmed on both arms`, `partial` (×5), `inconclusive`, `dissolves`. I grepped
`body.tex` for `D1`, `D3`, `ARTIFACT` and `WITHDRAW`: **`D3` and its verdict appear
nowhere.** `D1`'s `LARGE` verdict is also absent. Instead, `body.tex:636` presents the same
number as *"That is the finding: the aggregate field ranking on this benchmark is not a
property of the methods."*

To be precise about what I am and am not alleging: **nothing is concealed.** All three
ratios (8.39 / 0.44 / 0.21) are in the abstract, the intro, two tables and the conclusion,
and the withdrawal is real. This is not the round-1 `mgn_density_control.json` pattern or the
round-2 resolution-ladder pattern. What I am alleging is **promotion**: a rule the authors
wrote in advance, which instructed them to read this number as *"our claim was an artifact"*,
has been reported as a discovery, with the verdict label — the one word that would tell a
reader which of the two it is — omitted from the only document a referee reads.

This is the direct answer to *"is measure-dependence a real contribution?"* My answer:
**partly, and less than the paper claims.** "Different metrics rank differently" is a truism.
What is not a truism, and what is genuinely worth publishing, is the *magnitude* ($312\times$
node/area disparity in the innermost band; $51.5\times$ inside $0.05c$; a sign reversal, not a
shuffle) and the *omission* (no published AirfRANS comparison states its weighting —
`body.tex:86-87`, `253-263`). Lead with those two. Do not lead with "the ranking reverses",
which a referee will read as the retraction it was registered as.

**Triage: R.** Name D1 and D3 by verdict, in the same small-caps register as the others.
Reframe the contribution as magnitude + omission. Costs nothing and removes the only
remaining instance of the disclosure pattern I have now named in three consecutive rounds.

### W2 [M][R] Contribution 3 does not bound what it says it bounds, the paper contradicts itself about it two pages apart — and taken literally the sentence destroys the paper.

Contribution 3, `body.tex:183-188`:

> *"**A representation ceiling that bounds every published $r128$ field number on this
> benchmark, ours included.** The scoring raster's own round-trip error at the native nodes
> exceeds the surrogate's per-node error by $413\times$ pooled and $495\times$ near the wall."*

The 413× is the error of rasterising the truth and resampling it back at the nodes. **The
published $r128$ metric never incurs that error.** It compares a rasterised prediction to a
*rasterised truth*; the round-trip is not in the comparison path at all. An error a metric
does not commit cannot bound the metric. What the number *does* establish — correctly and
usefully — is that the $r128$ representation cannot *express* near-wall structure, so an
$r128$ metric cannot reward a method for getting it right. That is a blindness statement, not
a bound, and the difference is exactly the difference between "published numbers are
meaningless" (what "bounds" implies) and "published numbers measure far-field skill" (what is
true, and what `body.tex:96` actually says).

**The paper knows this.** `body.tex:822-824`, explaining its own `P2` PARTIAL verdict:

> *"the grid metric scores a rasterised prediction against a **rasterised** truth, so the
> representation error partly cancels; the node metric does not."*

Contribution 3 asserts the bound; §4.1 invokes the cancellation that voids it. Pick one.

**This answers the brief's question "does the 413× bound the paper's own grid numbers into
uselessness, and does the paper survive it?" directly: it does not bound them, and the paper
survives *because* the sentence is wrong.** Taken at face value, "bounds every published
$r128$ field number on this benchmark, ours included" voids `tab:interp`, `tab:interp_std`,
`tab:measure`, `tab:bandratio` and `tab:interp_bands` — five of seven exhibits — and leaves a
one-table paper. That is the strongest reason the authors should *want* this correction: the
literal reading of their own contribution 3 destroys their own manuscript, and a referee who
applies it will say so in one line.

**And there is a sharper version of the point the paper is missing, which would strengthen
it.** Rasterisation is a projection. A projection does not merely go blind to what it
discards — it can *invert* rankings, by rewarding a prediction whose error lies in the
projection's null space over one that captures real sub-grid structure. The paper has the
evidence for exactly this and cannot explain it: the far-field $u$ entry flips from
$3.2\times$ ahead of the surrogate on the grid to $3.8\times$ behind at the nodes, and
`body.tex:817-819` rules out the obvious explanation (*"resampling the published $r128$
output at those same far-field nodes costs the interpolator only $1.27\times$ there, far
short of the $\approx10\times$ the swing requires"*). An unexplained rank inversion between a
projected and an unprojected measure of the same two predictions is what projection-induced
reweighting looks like. Reframing contribution 3 as *"the scoring projection is not
rank-preserving, and here is a measured instance"* is both defensible and stronger than the
word "bounds", which is not.

**Triage: R**, one paragraph, plus the honest reframe. Do it — this is currently the most
attackable single sentence in the contributions list and the fix makes the contribution
better.

### W3 [M][R] "Three defensible measures" is incoherent with the paper's own third finding.

`abstract.tex:30-32`: *"one pair of predictions, three defensible measures, and the ranking
reverses."* `body.tex:82-83`: *"Neither grid measure is obviously the right one, and that is
the point."*

Then `abstract.tex:38-40`: *"The raster cannot adjudicate it for anyone: its own round-trip
error at those nodes exceeds the surrogate's by $413\times$."* And `body.tex:96`: *"An $r128$
field metric on AirfRANS is a measurement of far-field skill."*

Two of the three measures operate on $r128$ fields. If those fields cannot express the region
where the two methods differ by three orders of magnitude, then two of the three measures are
not defensible *for this comparison* and the "reversal" is not an epistemic puzzle — it is a
broken instrument being correctly diagnosed. The paper wants both readings at once: the
sophisticated one ("the ranking is a property of the measure; measures are choices") and the
plain one ("the standard protocol is broken"). The plain one is the stronger paper and the
one the evidence supports. As written, a referee will quote the abstract against itself.

**Triage: R.** Say: *there is one measure without a representation ceiling and two with one;
the ranking reverses between them, which is why the ceiling matters.*

### W4 [M][C→E] "A surrogate earns the boundary layer" is true of the channels that do not set the forces and false of the one that does.

`body.tex:804-808`, the new wall row:

> *"at $\mathrm{sdf}=0$ the surrogate is $4060\times$ better on $u$, $2834\times$ on $v$ and
> $382\times$ on $\nu_t$, while the two arms are within $1.51\times$ on $p$ ($149\,203$
> against $99\,227$) and **both are poor there in absolute terms** relative to
> $\mathrm{Var}_{\text{train}}(p)=135\,590$."*

Credit where due: the paper states this plainly and unprompted. Now read the consequence.
Transolver's surface-node pressure MSE is $99\,227$ against a train variance of $135\,590$ —
it explains roughly **27%** of surface-pressure variance at the nodes where lift and drag are
integrated. Surface pressure is the *only* near-wall quantity an aerodynamicist cares about.
And `sec:forces` concedes the rest: viscous drag is *"**unavailable**"* on this representation
(`body.tex:1119-1121`), $68\%$ of drag is viscous (`body.tex:1099-1100`), and the repaired
control-volume estimator still reaches only $\approx2.7\times$ median drag error on predicted
fields (`body.tex:1111-1113`).

So "earns the boundary layer" cashes out as: the surrogate is three orders of magnitude
better than a baseline that is catastrophically wrong, on $u$, $v$ and $\nu_t$ at the wall;
it is $1.5\times$ better on the channel that determines every engineering quantity, and both
arms are bad there. The claim is **relative, and the paper's title and conclusion state it
absolutely**. `abstract.tex:42-43`: *"A neural surrogate earns the boundary layer."*

**Triage: R** for the claim (state it relatively and name the $p$ exception in the abstract,
not four pages in). **E** if the authors want the absolute version: report surface-pressure
$R^2$ at the nodes for both arms, and a $C_p(x)$ comparison on a handful of cases. That is
one evaluation pass and it is the exhibit an aerodynamics referee will ask for by name.

### W5 [M][R/E] There is no evidence anywhere in the paper that this Transolver is a competent Transolver, and finding 2's magnitude depends on it.

`body.tex:412-417`: 7.35M parameters, 80 epochs, per-point standardised MSE over a uniform
subsample of each case's cloud (the subsample size — 16,384 against a ~180k cloud — is no
longer stated; it was in the previous draft's MGN control, which left with the trust layer).
`body.tex:1062-1067` then declines a leaderboard row on the grounds that AirfRANS results use
incompatible conventions, and `tab:airfrans-sota` is explicitly *"field context only… Our rows
are **not** comparable."*

So the paper's single learned arm is never calibrated against any external number. Gates G1/G2
prove it is *the deployed backbone*; they say nothing about whether the deployed backbone is a
*good* Transolver. The asymmetry matters and runs in two directions:

* **Undertraining makes finding 1 conservative.** A weaker Transolver would understate the
  near-wall advantage. Fine.
* **Undertraining inflates finding 2.** The $8.4\times$ area-uniform pressure win is a
  far-field statement — `body.tex:791-793` says so: *"the area-uniform $8.4\times$ on
  $\texttt{mse\_p}$ is a far-field statement about $80\%$ of the cells and $11\%$ of the
  nodes"* — and $62.8\%$ of Transolver's own $p$ error lies beyond $0.5c$
  (`measure_asymmetry.md` §3a). A better-trained Transolver with a cleaner far field shrinks
  the $8.4\times$, and with it the size of the reversal that is the paper's second headline.

The paper's whole thesis is that measurement conventions decide conclusions. It cannot then
decline, on convention-incompatibility grounds, to place its own model on any commensurable
scale. **Triage: R + cheap E.** Report relative-$L_2$ in the Transolver-lineage convention on
the same three checkpoints, next to the published Transolver number, with the protocol
differences named. If it lands in the published range, W5 dies and finding 2 gets much
harder to attack. If it does not, the authors need to know.

### W6 [M][R] `sec:residual` is contribution 4 and its controls are in a different paper.

`body.tex:1129-1130`: *"This section reports the measured answer in compressed form; **the
theorem behind it and the full control set are deferred to a companion paper**."* Contribution
4 (`body.tex:190-195`) and a full Conclusion paragraph (`body.tex:1338-1345`) rest on it, and
the Conclusion states it unconditionally: *"Fixing the metric with physics does not work, and
the reason is measurable."*

Either it is a contribution of this paper, in which case the controls come with it, or it is
deferred, in which case it is not a contribution. This is round 2's §2.6 in a new costume: the
paper again carries a claim whose evidentiary base it says lives somewhere else.

**And there is a substantive problem underneath the structural one.** The residual floor
($\|r^\star\|$ mean 0.192) is measured on the $r128$ rasterised truth — the representation
contribution 3 declares unfit to adjudicate anything near the wall. The paper's own controls
point at the rasteriser as a major contributor: `body.tex:1158-1159`, *"at fixed $h$, thinning
the source cloud eightfold raises the floor in $16/16$ cases"*; `body.tex:1141`, the floor
*rises* under refinement, $0.0624\to0.0779\to0.1067$; `body.tex:1155-1159` names reconstruction
as one of two contributions and *"separate[s] neither."* So the paper's fourth contribution is
a negative result measured through an instrument its third contribution says commits $413\times$
the signal, and it says so itself and then states the conclusion without the condition.

I am not claiming the negative is wrong — the descent result (Armijo, $n=200$, 200/200, two
Laplacian stencils, pinned and unpinned, three step rules) is the strongest single experiment
in the paper's history and I said so in round 2. I am claiming the **scope** sentence is
missing from the Conclusion: this is a statement about a surrogate-side Cartesian operator on
a rasterised reconstruction of a body-fitted solve, not about physics-informed correction.

**Triage: R** (add the condition to `body.tex:1338` and contribution 4), or **cut** the
section to a Related-Work paragraph pointing at the companion paper. Cutting costs the paper
nothing that a referee will miss and removes a whole attack surface.

### W7 [m][R] Residual round-2 items, part-fixed.

* **The $R^2 \ge 0.9996$ two-statistic problem is 2/3 fixed.** `body.tex:763-767` now
  explains the pooled vs per-case-centred distinction properly and gives the strict floor
  (0.9965), and `tab:interp_bands`'s caption says both are in the artifact. But the
  Introduction (`body.tex:120`) and the Conclusion (`body.tex:1334`) still quote the pooled
  $0.9996$ **unlabelled**, and `tab:interp_bands` still reports only $R^2_{\mathrm{pc}}$, so
  the headline number still appears in no table. Add the pooled column or label the two
  remaining quotes. `body.tex:985-986` also concedes the claim binds at $0.99961$ — a
  fourth-decimal margin — at every rung. One sentence of the three should carry that.
* **"Matched-budget" was reinstated against the project's own written decision.**
  `abstract.tex:29`, `body.tex:48`, `body.tex:1311`. `restructure.md:179-186` records the
  deliberate choice to avoid the word: *"a referee reading it next to `0 params (210 MB)`
  will read it as a capacity claim and object."* No compute budget is matched anywhere —
  the interpolator is 15 minutes of CPU (`body.tex:270`), the Transolver is 80 GPU epochs.
  Use "trained on the same data".
* **The paper's sole figure is generated by an uncommitted script.** `git status` shows
  `scripts/make_fig_bandratio.py`, `results/figures/fig_bandratio.pdf` and `.png` as
  untracked, while the availability statement (`body.tex:1392`) claims *"Every headline
  number maps to a committed script and result file"* and ships a SHA-256 manifest. Trivial
  to fix; a referee who takes the manifest invitation at face value will find it.
  `jocs_rebuild.md:54-58` records the same class of problem for
  `point_space_oracle_ls.json` (*"currently untracked and the auditor now reads it… a SKIP
  suppresses the 'every checked number matches its source file' line exactly as a MISMATCH
  does"*) — verify that one is committed before filing, because it is the n=3 artifact.
* **`jocs_rebuild.md` §7 lists every build and audit check as UNRUN.** `pdflatex`, `bibtex`
  (required — 13 citations dropped), `audit_paper_numbers.py`, `check_submission.py`. Later
  commits suggest some of this was done; confirm all four before filing, and confirm the
  auditor prints no SKIP.

### W8 [m][C] The Transolver density mismatch now runs *for* the paper, and the paper should say so.

Round 2 §2.8 attacked the train/eval density mismatch (trained on uniform subsamples,
evaluated on the full ~180k cloud) as an uncontrolled confound inflating the 8.4×. In the
current draft it inverts: the training subsample is uniform over the cloud, so the training
measure *is* the node measure, and any density degradation makes `tab:native` **conservative**
for the surrogate. I withdraw the objection in its round-2 form. One sentence in
`sec:experiments` saying this converts a discoverable into a defence, and it is free.

### W9 [m][C] Scope, conceded without argument.

One benchmark. Two dimensions. One learned architecture, three seeds. One baseline family, in
one coordinate system. The `full` split only for the native head-to-head
(`body.tex:1248-1249`). No body-fitted arm. The 21× bound at n=3. All of these are stated in
Limitations, most of them accurately. None is individually fatal for a measurement paper at
this venue; together they mean the paper's claims are about **AirfRANS**, and the abstract's
closing sentence (*"A neural surrogate earns the boundary layer, and the protocol that reports
it cannot see there"*) is written as though they were about the field. `body.tex:1296-1303`
argues the mechanism generalises to any wall-clustered-cloud-plus-raster benchmark and ships
the diagnostic as a reusable routine — that is the right argument, it is well made, and it
should be *in the abstract's last sentence* rather than the eighth Limitations bullet.

---

## 3. Triage summary

**Must be fixed by rewriting before submission (a day, all of it):** W1 (name D1/D3, reframe
finding 2 as magnitude + omission), W2 (contribution 3's "bounds"), W3
(three-defensible-measures), W4 (relativise "earns the boundary layer", name the $p$ exception
in the abstract), W6 (condition the residual conclusion or cut the section), W7 (all of it),
§1(i) (demote 344× behind 21×), §1 scope propagation into `abstract.tex:37` and
`body.tex:1309`.

**Fixable by an experiment worth running, in priority order:**
1. **The wall-normal / body-fitted interpolator arm** (§1(iii)). The decisive one. No free
   parameter if built from the analytic NACA surface; sweep the blend length if one is needed.
2. **Scale the LS family bound to n = 200** (§1(ii)). No training, no new code; the authors
   price it at 5.6 h and should state the true figure either way. It is the only valid number
   behind the paper's positive claim.
3. **Relative-$L_2$ in the Transolver convention** (W5). One evaluation pass; kills the
   "is this Transolver any good" attack outright if it lands in range.
4. **Surface-pressure $R^2$ at the nodes, both arms, plus $C_p(x)$ on a few cases** (W4).
   One pass; it is the exhibit an aerodynamics referee will name.

**Must be conceded:** 2-D; one benchmark; one architecture; the physical-coordinate scope of
the family bound if (1) is not run; viscous drag unavailable; the $p$ channel at the wall.

---

## 4. Claims that overreach their evidence — verbatim, with corrections

1. `abstract.tex:35-37` — *"an oracle shown the test answer, allowed the single best of all
   $800$ training fields, is $344\times$ worse inside the first $0.005$ chord, where a
   three-case probe of the exact least-squares bound over **the whole family** still gives
   $21\times$."*
   → **"the exact least-squares lower bound over every weighting of the $800$ training fields
   *in physical coordinates* is $21\times$ worse there (three cases, $u$); the fitted
   estimator itself is $173\times$ worse, and a best-single-field oracle scores $344\times$
   but is not a bound on combinations and is beaten by the fitted estimator in this band."**
   The current sentence leads with the invalid instrument and lets "the whole family" carry a
   scope the body explicitly disclaims.

2. `body.tex:1309` (Conclusion) — *"**no member of the interpolation family reaches it**."*
   → **"no weighting of the $800$ training fields **in physical coordinates** reaches it; a
   body-fitted parameter interpolator is a different method and we do not test it."**
   (`body.tex:576-579` already says exactly this. Propagate it.)

3. `body.tex:183-188` (contribution 3) — *"A representation ceiling that **bounds every
   published $r128$ field number** on this benchmark, ours included."*
   → **"A representation blindness: the $r128$ raster's own round-trip error at the native
   nodes is $413\times$ the surrogate's, so no $r128$ field metric can reward near-wall
   accuracy for any method. The published metric compares a rasterised prediction to a
   rasterised truth, so this is not a bound on its value — it is a bound on what its value can
   mean."** As written it is contradicted by `body.tex:822-824` in the same paper, and taken
   literally it voids five of this paper's own seven exhibits.

4. `abstract.tex:30-32` — *"one pair of predictions, **three defensible measures**, and the
   ranking reverses."*
   → **"one pair of predictions, three published-or-defensible measures — two of them on a
   raster whose own round-trip error exceeds the model's by $413\times$ — and the ranking
   reverses between the two that cannot see the wall and the one that can."**

5. `abstract.tex:42-43` / title — *"A neural surrogate earns the boundary layer, and the
   protocol that reports it cannot see there."*
   → **"A neural surrogate earns the boundary layer on $u$, $v$ and $\nu_t$ — by three orders
   of magnitude at the surface — while both methods remain poor on surface pressure, the
   channel the forces are integrated from; and the protocol that reports it cannot see
   there."** The unqualified form is contradicted by `body.tex:804-808` and `sec:forces`.

6. `body.tex:541` — *"against a pre-registered threshold of $10$
   (\textsc{no-parameter-combination})."*
   → the verdict string claims what `body.tex:543-546` concedes the statistic cannot show.
   Rename the verdict, or attach the concession to the same sentence rather than the next
   paragraph. **"…(\textsc{no-single-field}; the corresponding statement about combinations
   is the least-squares bound below)."**

7. `body.tex:636` — *"**That is the finding**: the aggregate field ranking on this benchmark
   is not a property of the methods."*
   → **"Our pre-registered rule \textbf{D3} read $R_p^{\text{node}} \le 1$ as
   \textsc{artifact} — 'the $8.4\times$ is a measure choice; withdraw it' — and it returned
   $0.440$. We withdraw it. What survives as a finding is the *size* of the disparity
   ($312\times$ between node and area weight in the innermost band) and the fact that no
   published AirfRANS comparison states which weighting it uses."**

8. `body.tex:1338` (Conclusion) — *"**Fixing the metric with physics does not work, and the
   reason is measurable**."*
   → **"Fixing the metric with an *affordable surrogate-side* physics residual, monitored on
   a Cartesian reconstruction of a body-fitted solve, does not work; we measure the floor but
   do not separate its operator-provenance and reconstruction components, and the full control
   set is in a companion paper."**

9. `abstract.tex:29` / `body.tex:48` / `body.tex:1311` — *"a **matched-budget** Transolver"*
   → **"a Transolver trained on the same data"**, per the project's own
   `restructure.md:179-186`.

---

## 5. Questions to the authors (trap-aware)

1. `body.tex:432-433` puts the analytic NACA camber line in your own feature vector, the
   package ships a NACA generator and an SDF module, and AirfRANS distributes wall distance as
   column 4. In what sense does a wall-normal $(s,n)$ coordinate for a NACA 4-digit section
   have *"no published provenance"* (`body.tex:574`)? What is $R_u$ inside $0.005c$ for the
   identical kernel weights applied in that coordinate system?
2. `point_space_headtohead.md` §3 shows the fitted estimator at **200.3** and the oracle at
   **397.7** in the band that carries the claim. On what basis does a verdict named
   `NO-PARAMETER-COMBINATION` follow from a statistic that the parameter combination beats?
3. `point_space_oracle_ls.json` has `n_ls: 3`, and `jocs_rebuild.md:417-419` prices the
   200-case run at 5.6 h. The 21.3× is the only valid bound behind your central positive
   claim. Why is it n=3 — and is 5.6 h the real figure, given §8 of the same document reports
   7350 s for the interp stage's cloud-to-cloud transfers?
4. Contribution 3 says the raster's round-trip error *"bounds every published $r128$ field
   number"*; `body.tex:822-824` says the grid metric compares a rasterised prediction to a
   rasterised truth so *"the representation error partly cancels"*. Which is it — and if the
   first, which of your own seven exhibits survives?
5. `measure_asymmetry.md` §6 records rule **D3** returning **ARTIFACT: "the 8.4× is a measure
   choice; say so plainly and withdraw it."** Every other registered verdict in the paper is
   named in small caps. Why is this one not named, and is finding 2 a discovery or a
   retraction?
6. At the surface nodes Transolver's $p$ MSE is $99\,227$ against
   $\mathrm{Var}_{\text{train}}(p)=135\,590$ — roughly $R^2 = 0.27$ on the channel every force
   is integrated from, and within $1.5\times$ of the baseline you say does not reach the wall.
   In what sense does the surrogate "earn the boundary layer"?
7. What is your Transolver's relative-$L_2$ on AirfRANS `full` in the Transolver-lineage
   convention, against the published value? If the conventions are incompatible, on what
   basis is your arm a fair instantiation of the architecture whose near-wall advantage is
   your headline?
8. The far-field $u$ entry flips from $3.2\times$ ahead to $3.8\times$ behind between two
   measures of the same two predictions, and `body.tex:817-819` rules out the raster as the
   cause ($1.27\times$ of a $\approx10\times$ swing). If a projection can invert a ranking by
   a factor of ten for reasons you cannot decompose, what protects the *other* band entries in
   `tab:bandratio` — including the $906\times$ over 4489 cells — from the same effect?
9. `body.tex:1129-1130` defers the residual theorem and *"the full control set"* to a
   companion paper, and `body.tex:1155-1159` says you name two contributions to the floor and
   *"separate neither"*. Why is contribution 4 in this paper?
10. `restructure.md:179-186` records a deliberate decision not to write "matched-budget"
    because a referee would read it as a capacity claim. It is now in the abstract, the
    introduction and the conclusion. Which compute budget is matched?

---

## 6. Score, verdict, and the comparison to my 3 and my 5

### Score: **6 / 10.** Major revision at the Journal of Computational Science; **reject** at JCP, CMAME, NeurIPS or ICML.

### On round 2's "→ 9 with the point-space head-to-head" — I am not honouring it, and here is why.

Round 2 §6 wrote that the point-space head-to-head moves this to 9. That promise was priced on
a premise that did not hold. I expected the experiment to **confirm the measure objection
while leaving a headline standing** — i.e. that the paper would emerge with a defended
$8.4\times$ or a defended reversal *plus* its original interpretation. What actually happened
is that the headline inverted entirely, and the positive claim that replaced it — "a surrogate
earns the boundary layer and no interpolation reaches it" — is a *new* claim that my round-2
review never evaluated, because it did not exist. It has its own load-bearing control (the
family bound), and that control is n=3 over a family defined to exclude its own fix. A
reviewer who pays out a pre-committed score on an experiment whose result was the opposite of
what he anticipated is not being consistent; he is being lazy. The 9 was for a paper that
would have had a defended cross-method result. This paper has something different and, in
important respects, better — but its new centre of gravity has not been stress-tested, and I
am stress-testing it now for the first time.

### What genuinely improved between 5 and 6 — substance, not presentation.

* **The fatal flaw was run at real cost and reported against interest.** 2.3 h wall,
  ~29 CPU-core-hours, 35.8M nodes, 200/200 cases, two gates (G1 at $\le1.6\times10^{-8}$, G2
  at exactly $0.00\mathrm{e}{+}00$), decision rules committed in `a54cb75` before the first
  number. The pre-registered rule fired against the authors' own headline and they withdrew
  it. This is the second consecutive round in which that happened.
* **Three of my round-2 rewrites are done and done well:** the RSM/Queipo lineage paragraph
  (`body.tex:304-319`) is better than the one `naming_and_positioning.md` drafted; the
  resolution ladder's PARTIAL is in the body, in Limitations and in the abstract's standing
  comment; the force columns are labelled in three places including `tab:interp`'s caption.
* **The trust layer is gone.** Six contributions became four; 40% of the body that the paper
  disclaimed priority over is in a companion file. The paper is now about one thing.
* **Adverse verdicts are reported where they used to be omitted.** `P2` PARTIAL with its two
  inversions and the far-field $u$ reversal — the single most inconvenient number in the
  paper — is in the intro, the body, the figure caption, the Limitations and the abstract,
  with the honest statement that it cannot be explained. The n=3 scope of the LS bound is
  written *into the sentence* rather than a footnote, with a standing comment at the top of
  `abstract.tex` forbidding a later compression pass from removing it. The wall-row arm
  inconsistency was corrected in both the manuscript and the committed review document rather
  than silently in one. **The round-1 and round-2 disclosure pattern did not recur in its
  concealment form.** That is worth saying, because I have named it twice and it would be
  dishonest not to record that it stopped. W1 is its residue — a framing choice, not a
  concealment — and it is the mildest instance of the three.
* **The measure-dependence and representation-blindness measurements are real, cleanly
  instrumented, pre-registered, and useful to the field.** $312\times$, $51.5\times$, the
  across-case spread (min 0.6045 / max 0.6238), the G6 sdf-agreement check, the five node-
  measure discretisations with the least favourable one reported — this is good measurement
  science, and the observation that **no published AirfRANS comparison states its cell
  weighting** is a genuine, checkable, actionable contribution.

### What did not improve, and it is the same shape as both previous rounds.

1. **The most valuable experiment is still the one not run — for the third consecutive
   review.** Round 1: the residual was never minimised. Round 2: the head-to-head was never
   run in the measure the surrogate optimises. Round 3: the interpolator was never built in
   the coordinate system whose absence the paper *itself measures* as the failure mechanism
   ($7.8\times$, $35\%$ inside the training body). Each time the gap is disclosed honestly in
   Limitations and the claim is headlined anyway. Each time it was run, **the headline
   reversed.** The authors should treat that as a two-for-two prior, not a coincidence.
2. **The load-bearing control is again undersized.** Round 1's pattern was that the positives
   were well-seeded and the headline negative was n=1. Here the positives are 200 cases and
   3 seeds; the one control that converts "our baseline lost" into "no such baseline can win"
   is n=3, against a scale-up the authors themselves have costed.
3. **A contribution again asserts more than its instrument delivers** (W2's "bounds"), and
   again the paper contains the sentence that refutes it (`body.tex:822-824`). This is the
   same species as round 1's B3 and round 2's §2.3 — the claim is defensible in its careful
   form and is stated in its strong form.

### Is it publishable at the Journal of Computational Science?

**Yes, after major revision, and probably without the new experiment** — provided the
revision demotes finding 1 to its scoped form and leads with findings 2 and 3. JoCS is a
generalist computational-science journal, not JCP. A paper that (a) measures a $51.5\times$
node-versus-area weighting disparity nobody in the benchmark's literature states, (b)
measures a $413\times$ representation blindness in the standard scoring path and ships the
diagnostic as a reusable routine for arbitrary cloud+raster pairs, (c) runs a 35.8M-node
raster-free head-to-head with committed decision rules, and (d) reports its own
pre-registered rules firing against it — is a contribution that venue should take, and it is
more rigorous than most of what it will print. The revision that gets it there is §3's
rewrite list, which is a day's work using material already in this repository.

**It is not publishable at JCP, CMAME, NeurIPS or ICML**, and I would not soften that. At
those venues finding 1 is the paper, and finding 1 as scoped is arithmetic; findings 2 and 3
are protocol hygiene on a single 2-D benchmark, which is a workshop or a benchmark-track
contribution, not a JCP paper. The version that clears JCP is the one where the wall-normal
arm has been run and finding 1 is decisive either way.

### On "does a retracted headline read as rigour or as instability?"

**Rigour, decisively, and the authors should stop worrying about it** — but they must control
the framing. A paper that withdraws its own lead number on a rule it wrote in advance is
doing what the field says it wants and almost never does. The risk is not the withdrawal; it
is that the withdrawal is currently dressed as a discovery (W1) while the number it retracted
is still the abstract's first statistic. Name the rule, name the verdict, say "we withdrew
it", and the retraction becomes the paper's strongest credibility asset. Hide the verdict
label and a referee who reads `measure_asymmetry.md` — which the manifest invites them to —
finds the word `ARTIFACT` and supplies the uncharitable reading themselves.

### The single flaw that most justifies rejection at the higher venues

**The paper's surviving positive claim is bounded only within a family it defines to exclude
the fix its own mechanism section identifies, by an instrument it concedes is invalid, at
n = 3.** Everything else on this list is a day of rewriting. That one is an experiment, it is
the third such experiment in three rounds, and on both previous occasions running it reversed
the paper's headline.

### What moves the score

* **→ 7, by rewriting only (one day):** §3's rewrite list. At JoCS I would be at accept.
* **→ 8, with two cheap runs:** the n=200 LS bound and the relative-$L_2$ calibration (one
  pass). Finding 1's valid number stops being a probe and the learned arm stops being
  unvouched.
* **→ 9, with the wall-normal interpolator arm**, whichever way it falls. If the gap survives
  in body-fitted coordinates, the paper has a genuinely non-obvious result about
  representation and I would send it to JCP myself. If it closes, the paper has a better
  result about coordinates and a third consecutive honoured self-refutation, which at this
  point is the most distinctive thing about this project.
