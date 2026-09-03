# Reviewer 2 — Computers & Fluids

**Manuscript:** "A perfect flow field can be a bad initial condition: representation, not accuracy, decides neural warm starts for RANS"
**Recommendation:** Reject (resubmission possible after major new experiments)
**Score:** 3 / 10

---

## 1. Summary

The paper asks whether a neural surrogate's flow field is useful as an initial condition for a
production steady-RANS solve, and argues that the answer is decided by the surrogate's *output
representation* rather than by its accuracy. The central design is genuinely clever: the contrast is
made on the exact converged field, so no property of any network enters. Read at the solver's own
cell centres that field saves 92.0% of a cold solve (§5.2); stored first on an equal-budget
wall-fitted 256x64 grid it costs 181.6% [-318, -45], winning 1 case of 5; stored on a uniform 128²
raster it costs 548.4%. A parameter-free closed form, `G = u+(y1+)/u+(yc+)`, is offered for what a
projection does to the first-cell wall gradient; it is validated as an upper bound over a fifty-fold
range of first stations (§6.2) and across Re = 10³–3·10⁶ (§6.6), and shipped as a pre-flight tool
(§6.4). The paper then does something unusual and creditable: it reports that its own proposed
mediator is false. Placing the first station inside the mesh's first cell takes the gradient error
from 1218.8% to 1.8% and makes convergence *worse* (-266.8%, 0/5, §5.2.1); a wall-law repair moves
convergence 0.1 points and smoothing it 0.6 more (§5.5). The damage is relocated to the pressure
field (§5.2.2). A trained mesh-native boundary-layer seed is then shown to accelerate viscous-drag
convergence by +18.4%, 13/13 cases, p = 0.0002 on a disjoint thirteen-case corpus (§5.1), with a
converged-field control at +93.6% and a Cartesian negative control at +3.4% (p = 0.27). Grid
sequencing is run as a classical baseline and reported to be the better seed (+75.9% vs +14.6%) that
loses only on price (§5.7). An acceptance gate bounds the worst case at (1 + K/N) x cold (§8).

## 2. Strengths (conceded)

1. **The accuracy-free design is the right idea.** Handing the solver its own converged field through
   different representations is the cleanest way to isolate representation, and I have not seen it done
   in this literature. If the measurement supported the claim, this would be a paper.
2. **The scoring discipline in §4 is better than the field's norm**, and the withdrawals (the +41.8%
   at Cd@0.5% becoming -4.2%; the 13%-agreement claim; the wall-distance bug in §6.2 and §6.6) are
   reported rather than buried. Rules 1, 3, 4 and 5 each name a real trap; the readability rule is a
   contribution in itself.
3. **Grid sequencing as a baseline** (§5.7) is the comparison the ML warm-start literature refuses to
   make, and the paper reports that the classical method makes the better seed. That is honest and rare.
4. **Reporting three failed attempts to establish its own mechanism** (§5.2.1, §5.5, §6.7) is
   scientifically correct behaviour, and the pre-registration of the repair prediction is real evidence.
5. **The closed form is parameter-free and cheap**, and §5.7's correct prediction for grid sequencing
   is an out-of-sample use of it.

None of this is enough. The measurements do not support the sentence on the title page.

---

## 3. Weaknesses — ranked, with the experiment that would resolve each

### W1 (FATAL). The headline contrast and the headline result are measured on two different quantities, and each is null on the other's.

This is the objection I would reject on, and it is assembled entirely from the paper's own numbers.

- The representation claim — the title, the highlights, the abstract's "274-point swing", §5.2, §5.3,
  §11 — is measured on **total drag, Cd@1%, n = 5**: +92.0% / -181.6% / -548.4%.
- The generality claim — "the recipe works", §5.1, and the only number with a corpus behind it — is
  measured on **viscous drag, Cd_v@1%, n = 13**: +18.4%, 13/13, p = 0.0002.

Now cross them.

**On viscous drag the representation effect nearly vanishes.** §5.2.1's own Cd_v column: the
mesh-native oracle reads +92.5% and the three projected oracles read **+86.1%, +69.1%, +71.0%**, all
5/5. That is a 6-to-23 point spread, not 274 — and the *worst-placed* arm (`or_proj_coarse`,
1218.8% gradient error) is the **best** of the three. For the network arms the effect is not merely
small but absent: §5.5 reports `nf_proj` at +14.5% against `nf_bl` at +14.6%. So on the metric the
paper's own corpus can read, storing the field on a grid first costs essentially nothing, and the
advice in §11 — "query your surrogate where the solver lives" — has no support.

**On total drag the corpus says the measurement cannot be read.** I checked `results/depth_corpus.json`:
`Cd@0.01` carries `"settled_spread": 0.01274` and `"readable": false` (1.27% against a 0.5% limit),
as do every Cd, Cl, Cd_p and Cl_p row; only the three `Cd_v` rows are readable. §5.1 says this
plainly. Figure 2's caption says it more strongly: on total drag "every band is rejected by the
readability rule, *and the converged-field control itself swings +49.7% → -42.6% → +12.8%* — a
control that is not flat indicates a measurement that cannot be read." §4 says the same in general
terms: "Viscous drag is monotone across bands and total drag is not. Monotone stability *is* the
evidence that a number is a convergence-rate measurement rather than an artifact of where a wandering
curve happens to cross a line."

So the paper argues, in its own methodological section, that total-drag savings are not rate
measurements — and then puts a total-drag saving in the title, the abstract, all five highlights and
the conclusion. The defence available is that on the five-case tree the row *is* readable, and I
verified that: `results/depth_placement.json`, `Cd@0.01`, `"settled_spread": 0.00212`,
`"readable": true`. I am not accusing the authors of quoting an unreadable row here. The point is
worse than that. **The verdict is readable at n = 5 and unreadable at n = 13.** The paper's own
"Provisional" box warns that one additional case can flip a readability verdict; the corpus has
already flipped this one, and the flip goes against the headline. A reviewer must therefore read
Cd@1% at n = 5 as a quantity that does not survive its own study's expansion.

**And the thesis is never tested at n = 13.** The corpus carries four arms — cold, `oracle_mesh`,
`cartesian_128`, `nf_bl` (I checked the arm list in `depth_corpus.json`). There is **no wall-fitted
projected arm anywhere in the thirteen-case study**. The one representation arm that is there reads
**+3.4% (p = 0.27) on the readable row** — harmless, not catastrophic. The paper calls that a "null
negative control" and treats it as evidence of pipeline hygiene; read the other way, it is the
paper's central claim failing to replicate on the only case set with statistical power.

*What would resolve it.* Run the wall-fitted oracle arms (`or_proj_coarse`, `or_proj_fine`) and
`cartesian_128` on all thirteen corpus cases and report Cd_v (readable) and Cd (with its verdict).
Either the 274-point swing survives at n = 13 on a readable metric, in which case this is a strong
paper, or it does not, in which case the title must be rewritten around viscous drag, where the
effect is 6–23 points and non-monotone in placement. Extending the budget past 6000 iterations so
total drag becomes readable at n = 13 would be an acceptable alternative route.

### W2 (FATAL-adjacent). The oracle arms are a *discrete fixed point*, so the contrast confounds "representation" with "any departure from the solver's own fixed point".

The oracle seed is the cold run's own converged solution re-injected (§3, "Arms"). That field is not
merely accurate; it is, to solver tolerance, a fixed point of the discrete operator. It converges in
~50 iterations because the residual is already at the floor, not because it is physically excellent.
Any perturbation — of any kind, in any region, of any physical significance — restarts a transient.
The paper's projected arms are exactly such perturbations, and no control isolates *which property*
of the perturbation matters.

The paper's own §6.2 makes this concrete and, I think, fatal to its reading of §5.2.1: "The
implementation clips a query below its first station rather than extrapolating, so once y1+
approaches the mesh's own first cell the grid's first row is populated *from* that cell and the
damage collapses to 1.00x." That is, `or_proj_fine` is **identical to `oracle_mesh` at the wall by
construction**, and differs from it essentially only away from the wall. Its -266.8% is therefore
evidence about the *outer and mid-field* round trip, not about the boundary layer at all. The
paper's marquee puzzle — "a near-perfect near-wall seed converges worst of all" (Highlight 4) — is
substantially an artifact of what the projection does everywhere else, and the paper never measures
that.

This confound explains everything the paper cannot: why fixing the gradient by placement does not
help (§5.2.1), why repairing it does not help (§5.5), why smoothing the repair does not help (§5.5),
and why the network arms show no representation effect at all (§5.2: `nf_proj_coarse` -32.1%
[-144.1, +60.9], 3/5 — that field is far off the manifold to begin with, so a round trip changes
little).

*What would resolve it.* One cheap control with machinery already in the repository: perturb the
converged field by a smooth field of the **same L2 magnitude as the projection error** but with no
near-wall structural change (e.g. a low-pass filter applied outside the boundary layer, or smooth
noise at matched norm), and re-run. If that arm also reads -180%, "representation" is not what is
being measured — *distance from the discrete fixed point* is, and the paper's thesis is a different
and much weaker one. If it converges fine, the representation claim is enormously strengthened.
Until this is run I do not believe §5.2 measures what its heading says.

### W3 (MAJOR). The pressure claim — the paper's surviving mechanism — is arithmetic, censored, and internally contradicted.

§5.2.2, §6.7 and §11 relocate the damage to the pressure field. The evidence is three items, and each
fails on inspection.

**(a) It is an identity, not a localisation.** `Cd = Cd_p + Cd_v`. Showing that Cd_v is preserved and
Cd is destroyed *entails* that Cd_p is destroyed. No measurement of the pressure field is reported
anywhere in the paper — not an L2 error of the seeded `p`, not a spatial map, not a decomposition by
region. The paper measures wall-gradient error (§5.2.1, §5.5) and tangential roughness (§5.5) for its
seeds; the one field-level diagnostic that would support its new claim is absent.

**(b) The single quoted number is a censoring artifact.** §5.2.2 and §6.7 rest on "`or_proj` reads
-183.9% on Cd_p@0.5%." In `results/depth_repair.json`, `Cd_p@0.005`: `or_proj` has
`"saving": -1.8430`, `"saving_reached_only": -0.3782`, `"n_reached": 3`, `"n_censored": 1`, with the
censored case contributing a budget bound of **-623.8%**. So the headline pressure number is a
four-case mean of which one entry is a bound, and the value over the cases that actually reached the
band is **-37.8%** — a fifth of the quoted figure, never reported. The row is also over only four of
the five cases (`cold_n: 4`; `naca2412_aoa2` never reached the pressure band cold). The paper reports
per-case values for §5.1 and §7.3; it does not for the number carrying its mechanism.

**(c) The comparator undercuts it.** In the same row, the *recommended* seed `nf_bl` reads
**-116.1%** on Cd_p while reading +34.3% on Cd. A quantity on which the paper's winner and its worst
loser are both catastrophically negative cannot be the property that separates them as reported. I
did the fair version: on the three cases where both arms reach the band, `nf_bl` averages +35.6% and
`or_proj` -37.8% — a real 73-point directional signal, which I concede. But that is n = 3, it is a
third of the 216-point gap on Cd that it is supposed to explain, and it is not what the paper says.

**(d) §7.2 contradicts §5.2.2 outright.** §7.2 reports that `fitted_p` — a pressure-only seed — is
**inert (+0.2%)**, and explains it by SIMPLE's structure: "a pressure seed inconsistent with U is
overwritten within a few iterations." If a seeded pressure field is overwritten within a few
iterations and pressure-only seeding does nothing, then a *corrupted* seeded pressure field cannot
cost the solver 180%. Either §7.2's mechanism is wrong or §5.2.2's is. The reconciliation the authors
presumably intend — that the projected *velocity* field drives the solver to a bad pressure field —
makes the mediator velocity again, just not the near-wall gradient, and that hypothesis is untested
anywhere in the paper.

*What would resolve it.* Report ||p_seed − p_conv|| and ||u_seed − u_conv|| by region (boundary layer
/ near field / wake / far field) for `oracle_mesh`, `or_proj_coarse`, `or_proj_fine` and `nf_bl`. The
fields exist. If the projected seeds' pressure error is small while their outer velocity error is
large, §5.2.2 is dead and W2's explanation is confirmed. Also report Cd_p per case with censoring
declared, and use `saving_reached_only` alongside the bounded mean, as §5.1 does.

### W4 (MAJOR). The one piece of positive corroboration for the pressure mechanism is not a like-for-like comparison.

§5.5, §5.2.2 and §11 all lean on this: "the only arm in this study that is *positive* on pressure
drag is the one that smoothed the reconstructed wall shear: **+19.8%** at 0.5% and **+57.8%** at
0.2%, against -115.4% for the recommended seed."

In `results/depth_repair.json`, `Cd_p@0.005`:

| arm | `saving` | `saving_reached_only` | `n_reached` | `n_censored` |
|---|---:|---:|---:|---:|
| `nf_proj` | -1.4111 | **+0.1978** | 3 | 1 |
| `nf_proj_fix` | -1.4111 | **+0.1978** | 3 | 1 |
| `nf_proj_smooth` | **+0.1996** | +0.1996 | 3 | **0** |

All three arms are `null` on `naca0012_aoa0`. For `nf_proj` and `nf_proj_fix` that null was bounded
at the budget (-623.8%, per §4 rule 3); for `nf_proj_smooth` it was **dropped**. On the three cases
all three arms actually reached, the unsmoothed repair reads **+19.78%** and the smoothed repair
**+19.96%** — a **0.2-point** difference, presented in the paper (and in PLANS §0.02) as a 160-point
one. The identical pattern holds at `Cd_p@0.002` (+0.1982 vs +0.1989), which additionally carries
`"readable": false` in the same file — yet §5.5 and §11 quote a 0.2%-band pressure number from it.

I am not asserting a scorer bug: either the scorer dropped a censored case for this arm alone, or
that arm's run was incomplete when the tree was scored (`docs/PLANS.md` §0.03 records the tree being
scored while an arm was "still solving"). Either way the arms are scored over different case sets,
which is precisely the failure §4 rule 3 exists to prevent — "the failing arm was rewarded for
failing" — applied here to the arm the paper singles out. Smoothing moved nothing.

*What would resolve it.* Finish or bound that case and re-score; state the case set and the censoring
for every Cd_p number; drop the claim if it does not survive, since §5.2.2, §5.5 and §11 all cite it.

### W5 (MAJOR). "Representation, not accuracy" is contradicted by the paper's own arms.

The title asserts a negative — that accuracy is not what decides — and §1 states "it does not claim
the surrogate is accurate — accuracy is a separate question and irrelevant here." But the paper
contains an accuracy contrast at fixed representation: mesh-native oracle **+92.0%** against
mesh-native network **+34.1%** (§5.2). That is a 58-point effect from accuracy alone, on the same
metric, in the same tree. Meanwhile the representation contrast on the network's own field — the
only place where the claim is about a real surrogate — is explicitly reported as unresolvable
(§5.2: `nf_proj_coarse` -32.1% [-144.1, +60.9], 3/5, "the mean set by a single case at -231%").

So: accuracy matters by 58 points and is measured; representation matters by 274 points on a field no
network can produce and by an unresolvable amount on fields networks do produce. The honest title is
"both matter, and we can only measure representation on an oracle." No accuracy ladder is run (e.g.
seeds at 1%, 5%, 20% field error at fixed mesh-native representation), so the paper cannot support
its negative claim at all.

*What would resolve it.* An accuracy ladder at fixed representation — the converged field corrupted
to a controlled sequence of L2 errors — plotted against the representation ladder on the same axes.
That is the figure the title promises and the paper does not contain.

### W6 (MAJOR, and specific to this journal). The classical baseline is charged a cost the paper admits is wrong.

§5.7 concedes that grid sequencing is the better seed (+75.9% vs +14.6% on Cd_v — five times the
headline effect) and rescues the paper's recommendation only via price: "the coarse solve costs 1486
fine-equivalent iterations against a cold run of 696." The caveat box then dismantles that number:
the coarse solve "ran its full 6000 iterations, so the charge above is pessimistic; a practitioner
would stop it earlier", and the mapper "is ours, not a production `mapFields`", leaving a 6–10x
first-cell gradient overestimate where "the coarse mesh's placement alone would permit about 2x", so
"this arm is a **lower bound**".

A practitioner-facing claim in *Computers & Fluids* cannot rest on a baseline that the authors
themselves say is handicapped in both quality and cost. If the coarse solve is stopped at its own
`residualControl` — say a few hundred coarse iterations — the charge could fall below the fine cold
run and grid sequencing would beat the learned seed outright on both axes. The paper's only practical
recommendation then disappears. This is not an expensive experiment: it is one coarse solve per case
with `residualControl` active, plus OpenFOAM's shipped `mapFields`.

*What would resolve it.* Re-run §5.7 with (i) the coarse solve terminated by its own convergence
criterion, charged at the cell-count ratio, and (ii) `mapFields` rather than the in-house mapper.
Report the result even if it kills the recommendation. Also: the surrogate's training cost (an
AirfRANS dataset that is itself thousands of RANS solves) is nowhere amortised, which makes the
"price" argument incomplete on its face.

### W7 (MAJOR). n = 5, bounded/censored savings, bootstrap CIs, and no paired contrast statistic.

The mechanism study carries every claim in the title. At n = 5:

- The paper correctly states that the smallest attainable two-sided sign-test p is 0.0625, i.e.
  **no result in §5.2, §5.2.1, §5.5 or §7.3 can reach significance by its own pre-declared test.**
  The inferential weight is then quietly transferred to percentile bootstrap CIs ("the intervals on
  the projected arms exclude zero", §5.2), which at n = 5 on a left-skewed, ratio-scale, *budget-
  censored* quantity have poor and unquantified coverage: the resample space is 5⁵ = 3125 and the
  endpoints are little more than functions of the extreme values. A CI of [-318.0, -45.2] from five
  numbers is not evidence of the precision it implies.
- Savings are bounded at the budget (rule 3) and several are censored (`< -80.5%`, `< -568.3%`, and
  the -623.8% bounds in the Cd_p rows). Bootstrapping a mean of partly-censored values and reporting
  a percentile interval is not a defined procedure; a survival-analysis or rank-based treatment is
  needed, or the censored cases must be reported separately as §5.1 does.
- **No arm-vs-arm paired statistic appears anywhere.** Every CI is a saving against cold. The claims
  are *contrasts* (mesh-native vs projected; fine vs coarse placement). A paired per-case difference
  with a sign or rank test on the five paired differences is the correct test and is absent. §5.2.1's
  key assertion — that -266.8% is worse than -181.6% — is stated from two means with overlapping
  intervals ([-419.5, -114.1] and [-318.0, -45.2]) and never tested.
- "Winning 1 case of 5" means one case improved under an arm described as catastrophic; per-case
  values for the oracle projection arms are never shown, though they are for the corpus (§5.1) and
  the wake bound (§7.3).

*What would resolve it.* Per-case tables for every arm in §5.2/§5.2.1/§5.5; paired difference tests
for each declared contrast; either raise the mechanism study to n >= 10 or state every §5.2 claim as
descriptive.

### W8 (MAJOR, journal-standard). Verification of the CFD setup is missing.

The entire dependent variable is convergence behaviour on one homemade C-grid, and the paper offers:
no mesh-independence study; no validation of converged Cd/Cl against experiment or against AirfRANS's
own solutions for the same sections; no y+ distribution (only the design claim "y+ < 1"); no
`divSchemes`/`gradSchemes`/`laplacianSchemes` (§3 gives under-relaxation, `nNonOrthogonalCorrectors`
and the budget, nothing about discretisation); no statement of the wake-cut or far-field boundary
conditions beyond "20-chord far field". Three cases are admitted to have no unique steady drag
(§5.1, §10). Aspect ratios of 2·10⁵ with SIMPLEC and a residual floor of 3.5·10⁻⁷ are quoted, but
nothing establishes that the converged fields are correct rather than merely stationary. For a
journal whose readership will replicate this, that is not optional. It is, however, entirely fixable.

### W9 (MODERATE). The scoring protocol was amended after the corpus ran, in a direction that rescued a supporting number.

§4 states that rules 7 and 8 were "added by the thirteen-case sweep". Rule 8's effect is stated
explicitly: it "silently dropped the oracle control on one case, turning a +93% control into a +49.7%
bound." The +93.6% oracle control in §5.1 — one of the three properties offered as evidence that the
headline is a rate measurement — therefore exists because a scoring rule was changed after the data
were seen. Disclosure is not repair. Combined with the other researcher degrees of freedom in the
pipeline (arm set via `--drop-arm`, `MIN_DEPTH_OVER_FLOOR = 5.0`, `MAX_SPREAD_FRACTION = 0.5`, choice
of band, choice of coefficient, case admission gate that §4 rule 7 admits was mis-specified), the
surviving positive result is one of a large number of analysis paths.

§4 claims the pipeline is stable — "removing our most recently added arm moves every headline number
by at most 0.5 percentage points" — but the manuscript itself reports `nf_bl` at Cd@1% as
**+33.9%** (§5, §5.3, §5.7), **+34.1%** (§5.2, §5.6), **+34.3%** (§5.5) and **+34.0%** (§9), and
`or_proj` at **-181.6%** (§5.2), **-182.1%** (§5.5) and **-172.6%** (§5.3, as `fitted_256x64`), with
no row naming its arm set — which §4's own rule ("a number must name the arm set it was computed
over") requires.

*What would resolve it.* Report the corpus scored under the pre-registered scorer alongside the
current one; label every table row with its tree and arm set; state the sensitivity of §5.1 to the
scorer amendment explicitly.

### W10 (MODERATE). The wall-clock evidence does not support the headline.

§9 reports +34.0% iterations → +28.8% seconds for `nf_bl` on five cases, and its own readability box
concedes two of those five cases exceed the 0.5% spread limit, leaving three readable cases
(+29.7% → +25.0%). Which metric and tree does §9's "iterations" column use? If it is Cd@1% on the
mechanism tree — as the numbers suggest — then the **+18.4% corpus headline has no wall-clock support
anywhere in the paper**, and the only seconds reported ride the metric the corpus declares unreadable.
§9 also reports that seed construction costs ~11 s against cold solves of 88–239 s, i.e. 5–12% of the
solve, which is a large fraction of an 18.4% saving on a component quantity.

*What would resolve it.* Report end-to-end seconds for `nf_bl` on the thirteen corpus cases, scored
on Cd_v, with the 11 s construction charged.

### W11 (MODERATE). The acceptance certificate gates on a signal the paper shows is inverted.

§5.6's table is unambiguous: `or_proj_coarse` is +40.6% on the residual and -181.6% on drag;
`nf_bl` is **-80.5% on the residual** and +34.1% on drag. "A study that scored the residual alone
would have selected exactly the wrong seed." §8 then builds the acceptance rule on "two scalars from
the residual history — the level log10 r_K and the drop." I cannot reconcile these. Either the
threshold operates in the counter-intuitive direction (a *worse* residual at K = 25 is evidence of a
*good* seed), which needs to be stated and explained, or the gate is exploiting something other than
what §5.6 describes.

Further: the 70-seed population is 15 arms x 5 cases, so it is neither independent nor
representative — it is a set of arms the authors constructed, two thirds of which they designed to
fail. "0 / 46 harmful admitted" on such a population is close to uninformative. And the gated mean of
+1.5% against `nf_bl`'s own +34% means the gate discards most of the value it is protecting; the
paper concedes 12% capture. The (1 + K/N) bound is arithmetic and trivially true, as the paper says.

*What would resolve it.* State the learned threshold and its sign, and answer directly: **does the
K = 25 gate admit `nf_bl` on all five cases?** If it rejects the recommended seed, §8 and §5 are
incompatible. Re-evaluate the gate on the thirteen corpus cases with a seed population not authored
for the purpose.

### W12 (MODERATE-FIXABLE). The manuscript is not internally consistent after its rewrite.

Individually minor; collectively they read as a rewrite that was not swept, and in a paper whose
credibility rests on its corrections that matters.

1. **§10 still asserts the withdrawn claim.** "the predicted factor should be read as indicative
   rather than as the 13%-accurate number it is on the attached cases measured here" — contradicted
   by §6.2 and by §10's own final bullet ("An earlier version of this work reported 13% agreement;
   that figure ... is withdrawn").
2. **§10's repair bullet predates §5.5.** It says the repaired seed is "11.1x rougher", where §5.5
   reports 26.2x and 5.7x, and calls "a tangentially-smoothed repair ... the experiment that would
   settle it" — the experiment §5.5 ran.
3. **§7.4 asserts an explanation §5.3 falsifies.** §7.4 states `nf_mesh` fails "because the model's
   training `sdf` distribution is centred on 0.23 chords ... so the outer field is extrapolation";
   §5.3 states that explanation "is **falsified by §5.7**".
4. **§1 and the abstract disagree on the headline.** §1: "On an equal-budget wall-fitted 256x64 grid,
   **173%** slower." Abstract, §5.2 and §11: **181.6%**. (§5.3's `fitted_256x64` is -172.6%; the
   reader is not told these are different trees.)
5. **The closed form gives two different answers for the same format.** §6.5: 128² raster, first
   station 1.2·10⁻², y+ 1700, predicted G **29.3x**. §7.1: 128² raster, first station 1.17·10⁻²,
   y+ 1677, predicted G **36.6x**. Same expression, same mesh, same Re, 25% apart. Similarly the
   wall-fitted 256x64 from 2.5·10⁻⁴ is predicted at **23.3x** in §6.2 and **19.2x** in §6.5, and the
   mesh's first-cell y+ is quoted as **0.8** (§6.2), **0.72** (§6.5) and **0.58** (§6.6). If u_tau
   differs between per-case and single-case evaluations, say so and pick one.
6. **§5.2.1 vs §6.2.** A 1218.8% gradient error is G = 13.2x; §6.2 measures 14.4x at the same station.
7. Tables are numbered from "Table 5" (§6.2) with no Tables 1–4; **Figure 1 is never referenced in
   the text** though Appendix A lists it; §5.5's "+34.3%" table is captioned as if it were §5.2's.

---

## 4. Questions to the authors

1. Table §5.2.1 shows the three projected oracle arms at +86.1% / +69.1% / +71.0% on Cd_v against the
   native oracle's +92.5%. On the metric your corpus declares readable, what is the representation
   effect? Please state it as a number and reconcile it with "a 274-point swing from representation
   alone."
2. Why does the thirteen-case corpus contain no wall-fitted projected arm? What does
   `or_proj_coarse` read on Cd_v@1% over the thirteen?
3. §6.2 says the implementation clips below the first station, so a projection whose first station
   sits inside the mesh's first cell reproduces that cell from the cell itself. Given that,
   in what sense is `or_proj_fine` a *near-wall* manipulation at all, and what is its L2 velocity and
   pressure error **outside** the boundary layer relative to `oracle_mesh`?
4. What is ||p_seed − p_conv|| and ||u_seed − u_conv||, by region, for `oracle_mesh`,
   `or_proj_coarse`, `or_proj_fine` and `nf_bl`? Without it, on what basis other than
   `Cd = Cd_p + Cd_v` is the damage "located" in the pressure field?
5. §7.2 reports that a seeded pressure field is overwritten within a few iterations and that
   `fitted_p` is inert at +0.2%. How can corruption of the seeded pressure field then cost 180% of a
   cold solve? If the answer is that the projected *velocity* drives a bad pressure, please say so
   and test it — and note that the mediator is then velocity, not pressure.
6. `Cd_p@0.005` in `depth_repair.json`: `nf_proj_smooth` has `n_reached` 3, `n_censored` 0, while
   `nf_proj` and `nf_proj_fix` have `n_reached` 3, `n_censored` 1 with a -623.8% bound. On the three
   shared cases the savings are +19.78%, +19.75% and +19.96%. On what basis is the smoothed repair
   "the only seed in this study that is positive on pressure drag"?
7. Your recommended seed reads -116.1% on Cd_p@0.5% and +34.3% on Cd@1%. If pressure-drag damage is
   the mediator, why does the winner have it too?
8. What are the per-case savings for `or_proj_coarse`, `or_proj_fine` and `or_proj_half` at Cd@1%,
   and what is the paired per-case difference between `or_proj_fine` and `or_proj_coarse` with a
   rank or sign test on those five differences?
9. Does the K = 25 acceptance gate admit `nf_bl` on all five mechanism cases? What is the threshold,
   in which direction does it point, and how is that consistent with §5.6's finding that the residual
   ranking is inverted relative to drag?
10. What happens to §5.7 if the coarse solve is terminated by `residualControl` and mapped with
    OpenFOAM's `mapFields`? Is grid sequencing then cheaper as well as better?
11. Which tree and which metric does §9's "iterations" column use, and what are the end-to-end
    seconds for the thirteen-case corpus result?
12. Please supply mesh-independence and a validation of Cd/Cl against experiment or AirfRANS for at
    least two sections, plus the numerical schemes used.
13. §10 still describes the closed form as "the 13%-accurate number it is on the attached cases
    measured here", and still nominates a smoothed repair as the experiment that would settle §5.5.
    Are these residues of the rewrite, or claims you intend to keep?

---

## 5. Verdict

**Score: 3 / 10. Reject.**

Not a strong reject: the accuracy-free oracle design, the parameter-free closed form, the grid-
sequencing baseline and the discipline of §4 are real, and the reporting of three failed mechanism
attempts is the kind of behaviour this field needs more of. I would read a resubmission.

**The single fatal flaw:** the paper's headline contrast and its headline result are measured on
different quantities, and each is null on the other's. The 274-point representation swing exists only
on total drag at n = 5 — a metric the paper's own §4 and Figure 2 argue is not a convergence-rate
measurement, and which `depth_corpus.json` marks `readable: false` at n = 13 with the converged-field
control swinging +49.7% / -42.6% / +12.8%. On the metric that *is* readable at n = 13, the same
projections read +69% to +86% against the native oracle's +92.5%, the badly-placed projection is the
best of them, the network's projected and mesh-native seeds are indistinguishable (+14.5% vs +14.6%),
and the Cartesian control is +3.4% (p = 0.27) — harmless. The corpus contains no wall-fitted arm at
all, so the thesis is never tested where the paper has power. Compounding this, the contrast is drawn
between a discrete fixed point of the solver and perturbations of it, with no control isolating
representation from mere departure from that fixed point (W2), and the mechanism offered in its place
is an arithmetic identity supported by one censored mean and one non-like-for-like comparison
(W3, W4).

**Fatal vs fixable, for triage.** Fatal without new solves: W1, W2. Fatal to specific claims but
resolvable by re-scoring existing runs: W3, W4, W7. Fatal to the practical recommendation but cheap
to test: W6. Journal-standard and cheap: W8, W10, W11. Presentation and consistency: W9, W12.

The honest paper hiding inside this one is narrower and, I suspect, publishable: *a mesh-native
boundary-layer seed accelerates viscous-drag convergence by 18.4% across thirteen cases with a
verified control structure; the near-wall gradient is computable in closed form and is demonstrably
not the mediator; here is a scoring protocol that catches six ways of fooling yourself.* That paper
does not need the 274-point swing, and it would survive review. The current one stakes its title on a
number that its own corpus cannot read.
