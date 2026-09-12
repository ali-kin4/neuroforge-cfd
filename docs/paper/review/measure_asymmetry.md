# The measure asymmetry, measured — the 8.4x does not survive

Answers the objection scored **fatal** in `docs/paper/review/r2_round2.md` §1 and §2.1:
the headline is a *measure choice* and the paper cannot distinguish it from a finding.

Producer: `scripts/measure_asymmetry.py` (decision rules **D1–D3** and gates **G1–G6**
pre-registered in the docstring, committed in **`cdf08f5`** before any number existed;
two recorded amendments, `5746f58` and the §4b sensitivity block, both diagnostics only, no threshold touched).
Artifacts: `results/interpolation/measure_asymmetry_nodes.json`,
`results/interpolation/measure_asymmetry.json`, `results/interpolation/measure_asymmetry_sensitivity.json`.
Zero training. One inference pass over the committed `checkpoints/v2_transolver`
backbones, 200 test cases, 10.7 min on one GPU + 22 s of numpy.

---

## 0. The verdict, stated first and plainly

**The reviewer is right. The 8.4x is a weighting.**

Scored on the *identical* grid errors, the *identical* 200 test cases, the *identical*
rasterisation and fluid mask, changing **only the cell weights** from area-uniform to
the number of native AirfRANS cloud nodes each cell contains:

| `mse_p` | interpolation | Transolver (3 seeds) | ratio |
|---|---:|---:|---|
| area-uniform (the paper's measure) | 75.03 | 629.6 | **8.39x in the interpolator's favour** |
| node-weighted (the dataset's own measure) | 6012 | 2646 | **0.44x — 2.3x in Transolver's favour** |

(and 0.43x–1.61x across four discretisations of the node measure, §4b — in none of them
does the interpolator keep a meaningful aggregate advantage.)

The pre-registered rule (D3) reads `R_p_node <= 1.0` as **ARTIFACT: the 8.4x is a
measure choice; say so plainly and withdraw it.** It returns **0.440**. No threshold was
moved; the rule was committed before the number existed and it fires against us.

"Weight by node count" admits more than one discretisation, so §4b re-derives the same
ratio under four of them rather than quoting only the pre-registered one. The range on
`mse_p` is **[0.43, 1.61]** against **8.39** area-uniform — but the endpoints are not
equally credible. The two faithful cell-level estimators ("sample the grid error field at
each node, average over nodes", pooled or per-case) give **0.431** and **0.440**. The
largest value any construction gives is **1.61**, and it comes from the one construction
§4b documents as internally inconsistent (it pairs node masses with cell MSEs drawn from
a different spatial set); the coarse but consistent band-level construction gives
**1.10**. **No construction leaves the interpolator an aggregate win**: even at 1.61 on
`p` it is $85\times$ worse on `u` and $6\times$ worse on `v`.

**Every channel the interpolator was winning flips, and the channel it was losing stays
lost**, on every seed:

| ratio (Transolver MSE / interpolation MSE; `>1` = interpolator better) | `u` | `v` | `p` | `nut` |
|---|---:|---:|---:|---:|
| area-uniform | 0.162 | **2.971** | **8.391** | 0.070 |
| node-weighted | 0.012 | 0.066 | **0.440** | 0.108 |

(`nut` is the one entry that neither flips nor widens: Transolver's lead narrows from
$14.4\times$ to $9.2\times$. It was never a channel the interpolator won.)

Under AirfRANS's own node measure the parameter interpolator **wins nothing**. The
paper's standardised three-channel summary reverses with it: "the interpolator is
**1.9x better** averaged over `u, v, p`" becomes **8.3x worse** (same `Var_train`
divisors, §4).

**And the paper's central thesis is confirmed, on both arms, for the first time.** The
per-band ratio of interpolation error to Transolver error on `u` runs **906x → 0.31x**
monotonically from the first band to the far field — a span of **2905x** (D2:
`CONFIRMED ON BOTH ARMS`). The two methods are accurate in different places, by three
orders of magnitude, and which one an aggregate field-MSE table declares the winner is
decided by where you look.

**What must change:** the abstract's lead number, the `mse_v`/`mse_p` win claims, and
the standardised-mean claim. **What this audit does not touch:** the localisation thesis
(now measured on both arms instead of one), the far-field claim, the data-efficiency
ladder, the `reynolds` finding and the covariate null — none of them is a cross-method
aggregate field-MSE comparison. §7 proposes what should replace the headline; that is a
recommendation, not a measurement.

---

## 1. What was run, and the gates that make it checkable

| block | question | file |
|---|---|---|
| **A** | node fractions vs area fractions per wall band, three measures | `measure_asymmetry_nodes.json` |
| **B** | band decomposition of **both** arms + the per-band ratio | `measure_asymmetry.json` → `B_band_decomposition_{5,7}` |
| **C** | the same grid errors re-scored under node-count weights | `C_case_mean_{area_uniform,node_weighted}` |
| **D** | the r128 representation ceiling at the native nodes | `D_node_space` |

Gates, all asserted in the script (the run aborts on failure), all passed:

| gate | what it proves | result |
|---|---|---|
| **G1** | the interpolator arm *is* the published one | `mse_u/v/p` reproduce `interp_full.json` at **rel err 0.00e+00** |
| **G2** | the Transolver arm *is* the deployed backbone | all three seeds reproduce `v2_results.json` `backbone_per_seed` at **rel err 0.00e+00** |
| **G3** | the band accumulator *is* the published one | reproduces `interp_band_control_full.json` §A over **95 entries**, max rel **6.9e-08** |
| **G4** | the band table reconstructs the aggregate | `sum_b n_b · MSE_b == ` pooled SE, every arm, every channel, every band grid |
| **G5** | the two speed-ups change nothing | primed geometry cache **bitwise** equal to `signed_distance`/`solid_mask` on 3 cases; the one-Delaunay-per-case reuse **bitwise** equal to `rasterize_point_cloud` on 3 cases × 3 seeds |
| **G6** | the band edges do not depend on which sdf you believe | §2 |

G1 and G2 together are the load-bearing pair: this is not *a* Transolver and *an*
interpolator, it is **the two rows in the paper's own tables**, re-weighted.

### A correction the manuscript owes, found on the way in

`body.tex`'s limitations bullet and `interpolation_resolution_ladder.md:13` both state
that the point-space head-to-head cannot be run because *"the checkpoints do not store
the normaliser."* **That is false.** `checkpoints/v2_transolver/seed{m}.pt` contains
`point_norm` (`mean_in`, `std_in`, `mean_out`, `std_out`, `eps`) **and** `grid_norm`;
`scripts/recompute_force_vs_official.py:134-158` has been loading both since June. The
sentence excusing the single most important un-run experiment in the paper is wrong on
its facts, and it must come out whatever else is decided.

---

## 2. Block A — the node measure against the area measure

200 test cases, **35.85 M** native nodes (158k–203k per case), **95.4 %** of them inside
the 3c × 3c crop the metric scores. Bands are signed distance in chord units, identical
to `tab:interp_bands`.

| band | **A1** area frac (r128 crop) | **A2** node frac (crop) | **A3** node frac (full cloud) | **g = A2/A1** |
|---|---:|---:|---:|---:|
| 0–0.005c | 0.00138 | 0.4322 | 0.4119 | **312x** |
| 0.005–0.01c | 0.00118 | 0.0612 | 0.0584 | **52x** |
| 0.01–0.02c | 0.00251 | 0.0619 | 0.0590 | **25x** |
| 0.02–0.05c | 0.00742 | 0.0884 | 0.0842 | **12x** |
| 0.05–0.15c | 0.02942 | 0.1118 | 0.1066 | 3.8x |
| 0.15–0.5c | 0.15831 | 0.1311 | 0.1250 | 0.83x |
| >0.5c | 0.79978 | 0.1134 | 0.1549 | **0.14x** |

Cumulatively — the number the reviewer asked for, and it is not close:

| | area (A1) | node, crop (A2) | node, full cloud (A3) | **g** |
|---|---:|---:|---:|---:|
| inside **0.02c** | 0.00507 | **0.5553** | 0.5293 | **110x** |
| inside **0.05c** | 0.01249 | **0.6437** | 0.6136 | **52x** |

> The reviewer guessed "if that fraction is near 50% the objection is decisive; if it is
> 10% the objection shrinks to a scoping caveat." **It is 64% inside 0.05c and 56% inside
> 0.02c.** AirfRANS is a boundary-layer mesh: 41% of every cloud's nodes sit inside
> **0.005c** of the wall, a region that is **0.14%** of the r128 crop's area.

D1 verdict: **LARGE** (rule: `g >= 10` LARGE, `>= 2` MODERATE, `< 2` DISSOLVES).

The spread across cases is negligible — this is a property of the mesh generator, not of
a few cases: node fraction inside 0.05c has min 0.6045, median 0.6163, max 0.6238 over
the 200 cases (full cloud).

**G6 — the bands do not depend on which distance function you use.** The grid sdf comes
from `signed_distance` on a reconstructed surface polyline; the cloud sdf is AirfRANS's
own column 4. Over all 34.2 M in-crop nodes: median `|Δ| = 1.3e-4 c`, mean `1.3e-3 c`,
p99 `1.1e-2 c`. Seven-band assignment agrees on 87.1% — but that statistic is dominated
by the 0.005c/0.01c interior edges, which D1 never reads. At the two cumulative
thresholds D1 *is* read on, agreement is **99.51%** (0.02c) and **99.90%** (0.05c), and
recomputing the headline fractions under the *grid* sdf instead of the cloud sdf gives
0.5546 and 0.6454 against 0.5579 and 0.6458. The 110x and 52x are robust to the choice.

---

## 3. Block B — the two band decompositions, side by side

Same 200 cases, same rasterisation, same fluid mask, same accumulator (G3). Transolver
columns are the mean over the three deployed seeds. `cell frac` is area; `node frac` is
the node weight the same cells carry.

### 3a. Where each method's error lives (share of total squared error, `u` then `p`)

| band | cell frac | **interp** SE share `u` | **Transolver** SE share `u` | **interp** SE share `p` | **Transolver** SE share `p` |
|---|---:|---:|---:|---:|---:|
| 0–0.005c | 0.00138 | **0.8242** | 0.0057 | 0.3001 | 0.0614 |
| 0.005–0.01c | 0.00118 | 0.0538 | 0.0045 | 0.0937 | 0.0045 |
| 0.01–0.02c | 0.00251 | 0.0457 | 0.0094 | 0.1409 | 0.0107 |
| 0.02–0.05c | 0.00742 | 0.0180 | 0.0210 | 0.2375 | 0.0232 |
| 0.05–0.15c | 0.02942 | 0.0052 | 0.0560 | 0.1537 | 0.0710 |
| 0.15–0.5c | 0.15831 | 0.0167 | 0.1847 | 0.0518 | 0.2010 |
| >0.5c | 0.79978 | 0.0364 | **0.7188** | 0.0223 | **0.6281** |

**This is the reviewer's question 2, answered, and it breaks the way his second bullet
predicted.** The interpolator's error is wall-concentrated (92.4% of `u` inside 0.02c);
**Transolver's error is far-field spread** — 71.9% of its `u` squared error and 62.8% of
its `p` squared error lie beyond 0.5c, in the 80% of cells the interpolator handles
almost perfectly. The two methods are accurate in **different places**, and the
area-uniform metric weights one of them.

### 3b. The per-band ratio — the localisation claim, measured properly

`r_b = MSE_interp(b) / MSE_Transolver(b)`. `> 1` means Transolver is better by that factor.

| band | `u` | `v` | `p` | `nut` |
|---|---:|---:|---:|---:|
| 0–0.005c | **906.2** | 48.3 | 0.589 | 393.1 |
| 0.005–0.01c | **74.1** | 3.30 | 2.49 | 65.6 |
| 0.01–0.02c | **30.4** | 1.17 | 1.58 | 15.7 |
| 0.02–0.05c | **5.33** | 0.391 | 1.22 | 4.44 |
| 0.05–0.15c | 0.572 | 0.113 | 0.259 | 3.70 |
| 0.15–0.5c | 0.559 | 0.030 | 0.031 | 9.45 |
| >0.5c | 0.312 | 0.007 | 0.004 | 21.1 |

On `u` the ratio is **monotone across all seven bands**, spanning `S = 2905x` from the
innermost to the outermost (D2 rule: `S >= 10` with at most one inversion →
**`CONFIRMED ON BOTH ARMS`**). The crossover is between 0.02–0.05c and 0.05–0.15c.

The two endpoint bands are very different sample sizes, and the smaller one carries the
906x, so state it rather than let a reviewer find it. Pooled over the 200 cases, the
`0-0.005c` band holds **4489 fluid cells** (22.4 per case) against **2 595 931** beyond
0.5c (12 980 per case) — the 906x is a paired ratio over 4489 cell-samples, both arms
evaluated on exactly the same cells. Two things make it hold up anyway: the ratio is
monotone across all seven bands, so the innermost value sits on a trend rather than
alone, and `0.005-0.01c` (3824 cells) and `0.01-0.02c` (8143 cells) independently give
74x and 30x. It is also the band `interpolation_resolution_ladder.md` already refined:
at $512^2$, where it is 3.4 cells across instead of sub-cell, the interpolator's own
near-wall error *falls* 45% while the far field moves 0.6%, so a finer grid would narrow
this ratio rather than inflate it. `n_cells` is in the artifact for every band.

> **The thesis "a learned surrogate earns its keep in the first cell off the wall" is
> now measured on the surrogate, not inferred from the baseline.** Transolver is
> **906x** more accurate than parameter interpolation inside 0.005c and **3.2x less**
> accurate beyond 0.5c. That is a sharper statement than the paper currently makes and
> it is the one thing in this audit that makes the manuscript stronger.

Read the `p` row carefully, because it is where the headline came from: Transolver is
better in the innermost band (0.589), the interpolator wins modestly through 0.005–0.05c,
and then wins by **32x / 250x** in `0.15–0.5c` and `>0.5c`. **The 8.4x on `mse_p` is a
far-field statement.** It is 80% of the cells and 11% of the nodes.

---

## 4. Block C — re-scoring under the node measure

The weights are the only thing that changes: each fluid cell is weighted by the number
of native cloud nodes that bin to it, per case, then the same per-case-mean-of-MSE
protocol the paper uses is applied. Same predictions, same truth, same mask.

| channel | interp (area) | Transolver (area) | **R (area)** | interp (node) | Transolver (node) | **R (node)** |
|---|---:|---:|---:|---:|---:|---:|
| `mse_u` | 0.7816 | 0.1267 | 0.162 | 30.69 | 0.3578 | **0.012** |
| `mse_v` | 0.03362 | 0.09985 | **2.971** | 4.258 | 0.2821 | **0.066** |
| `mse_p` | 75.03 | 629.6 | **8.391** | 6012 | 2646 | **0.440** |
| `mse_nut` | 8.844e-8 | 6.147e-9 | 0.070 | 4.722e-7 | 5.119e-8 | 0.108 |

Per seed, so the result is not a seed artifact:

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| `R_p` area-uniform | 8.90 | 8.66 | 7.61 |
| `R_p` node-weighted | 0.418 | 0.433 | 0.469 |
| `R_u` node-weighted | 0.012 | 0.011 | 0.012 |
| `R_v` node-weighted | 0.062 | 0.063 | 0.074 |

Standardised MSE, dividing by the same `Var_train` the paper uses (u 341.88, v 52.45,
p 135590, nut 1.3199e-6 — a common divisor, so ratios are unaffected):

| | mean over `u,v,p` | mean over all four |
|---|---|---|
| area-uniform | interpolator **1.99x better** (the paper's "1.9x") | Transolver 6.1x better |
| **node-weighted** | **interpolator 8.30x worse** | interpolator 8.85x worse |

### The three honest caveats on this block, each of which runs *against* the interpolator

1. **Node weighting fixes the weighting, not the representation.** Both arms are still
   scored on r128 fields. A method that is right at sub-grid scales earns no credit
   under either weighting. Block D measures how big that remaining gap is.
2. **The re-weighting is deliberately conservative and still reverses the sign.** A
   0.0234c cell cannot sit inside a 0.005c band, so node mass from the innermost region
   is redistributed onto cells whose own sdf is larger: the node-weight fraction the
   grid assigns to 0–0.005c is **0.098** against the true **0.432**. The re-weighting
   therefore moves weight *out of* the band where Transolver is 906x better and *into*
   bands where the interpolator wins on `p` (2.49x, 1.58x). **The true node measure
   would be more adverse to the interpolator than 0.440, not less.**
3. **13.4% of in-crop nodes (4.58 M) bin to cells the geometry mask calls solid** and
   are dropped, because the metric scores fluid cells only. Those are the nodes nearest
   the wall. Dropping them also favours the interpolator.

### 4b. Sensitivity to the discretisation of the node measure

`results/interpolation/measure_asymmetry_sensitivity.json` (amendment 2; D3's rule and
threshold untouched, verdict still read on the pre-registered estimator). Four
constructions of "weight by node count", all from committed numbers, no new inference:

| construction | `R_u` | `R_v` | `R_p` | `R_nut` |
|---|---:|---:|---:|---:|
| area-uniform (the paper's) | 0.162 | **2.97** | **8.38** | 0.070 |
| **cell-level node counts, case-mean** (pre-registered, finest) | 0.0117 | 0.0663 | **0.440** | 0.108 |
| cell-level node counts, pooled | 0.0115 | 0.0649 | 0.431 | 0.103 |
| band-level, grid-binned node mass | 0.0061 | 0.159 | 1.102 | 0.042 |
| band-level, true node mass (crop) | 0.0018 | 0.041 | 1.610 | 0.018 |
| band-level, true node mass (full cloud) | 0.0018 | 0.041 | 1.613 | 0.019 |

The cell-level estimator is the finest and the most faithful — it is exactly "sample the
grid error field at each node, then average over nodes". The band-level rows are coarser:
they assume the error is uniform inside a band, which washes out the fact that within
each near-wall band the node mass concentrates on the cells closest to the wall, where
Transolver's advantage is largest. The last two rows are coarser still and internally
inconsistent (they pair node masses with cell MSEs drawn from a different spatial set);
they are reported because they are what a reader would compute by hand from §2 and §3a,
and they are the *least* favourable to this audit's conclusion.

Across all five node-measure rows: `mse_p` falls from $8.4\times$ to between $0.43\times$
and $1.61\times$; `mse_u` from $6.2\times$ against the interpolator to $85$–$560\times$
against it; `mse_v` from $3.0\times$ in its favour to $6$–$24\times$ against it. **No
construction leaves the interpolator with an aggregate win.**

### What this block is not

It is **not** a native point-space head-to-head. Scoring the interpolator at the native
nodes would change the measure *and* the representation at once, and would penalise it
for sampling a 0.0234c grid at wall nodes — a handicap that has nothing to do with the
objection. That experiment remains open. What is no longer open is whether the objection
has teeth: it does, and the direction and magnitude are now measured.

---

## 5. Block D — the representation ceiling the grid metric imposes

Per-node squared error against the native AirfRANS targets, by band, over the in-crop
nodes. `r128 ceiling` is the *rasterised ground truth* resampled at those same nodes —
i.e. the error the scoring representation itself commits, independent of any model.

Read from the artifact's `D_node_space.mse_in_crop` block, **not** `mse_full`: outside the
crop `_bilinear_sample` clamps query points to the domain boundary, so the outer bands of
`mse_full` are clamp-contaminated and must not be quoted.

| band | node frac | Transolver `u` @nodes | r128 ceiling `u` | ratio |
|---|---:|---:|---:|---:|
| 0–0.005c | 0.4322 | 1.158 | **572.8** | 0.0020 |
| 0.005–0.01c | 0.0612 | 0.6035 | 315.8 | 0.0019 |
| 0.01–0.02c | 0.0619 | 0.5069 | 109.5 | 0.0046 |
| 0.02–0.05c | 0.0884 | 0.3988 | 27.38 | 0.0146 |
| 0.05–0.15c | 0.1118 | 0.2627 | 9.608 | 0.0273 |
| 0.15–0.5c | 0.1311 | 0.1910 | 1.299 | 0.1470 |
| >0.5c | 0.1134 | 0.1104 | 0.1559 | 0.708 |
| **pooled** | 1.000 | **0.664** | **277.3** | **0.0024** |

Transolver's per-node error in the first band is **1.16**; the r128 raster's own
round-trip error there is **573**. The scoring representation destroys, in the region
holding 43% of the dataset's nodes, structure **495x larger than the model error it is
being used to adjudicate**. Pooled over the crop the factor is **418x**.

This is the half of the objection a re-weighting cannot reach, and it says the same
thing: *the r128 protocol is not a measurement of boundary-layer skill.* It is a
measurement of far-field skill, in which a 7-scalar interpolator is excellent.

---

## 6. Verdict against each pre-registered rule

| rule | threshold, committed in `cdf08f5` | measured | verdict |
|---|---|---|---|
| **D1** node/area gap inside 0.05c | `>=10` LARGE, `>=2` MODERATE, `<2` DISSOLVES | **51.5** | **LARGE** |
| **D2** span of `r_b` on `u` | `>=10` + monotone → CONFIRMED; `<3` → INTERPOLATOR-ONLY | **2905**, 0 inversions | **CONFIRMED ON BOTH ARMS** |
| **D3** `R_p` node-weighted | `>=4` SURVIVES; `>=1.5` SCOPED; `>1` WITHDRAW; `<=1` ARTIFACT | **0.440** (range 0.43–1.61 over four constructions, §4b) | **ARTIFACT** on the pre-registered estimator; **WITHDRAW** on the least favourable one. The headline goes either way. |
| **D4** representation ceiling | diagnostic, no threshold | 418x pooled | reported |

---

## 7. Exact manuscript changes

**I did not touch `body.tex`, `abstract.tex` or `refs.bib`.** These are the changes,
keyed to text, in priority order. Items 1–3 are not optional: the current sentences are
contradicted by a committed artifact in the same repository.

### 7.1 `abstract.tex:6-9` — the lead number. **Replace, do not caveat.**

Currently:

> Kernel interpolation over seven scalars parsed from the AirfRANS case name---no
> network, no flow-field learning---scored through the identical protocol, gives 88%
> lower volume-pressure error than a Transolver trained on the same data (8.4×).

Proposed:

> Kernel interpolation over seven scalars parsed from the AirfRANS case name---no
> network, no flow-field learning---gives $8.4\times$ lower volume-pressure error than a
> Transolver trained on the same data when both are scored on an area-uniform $128^2$
> raster---and no advantage at all on any channel when the identical predictions are
> re-weighted by AirfRANS's own node measure, which places $64\%$ of its mesh inside
> $0.05$ chord of the wall against $1.2\%$ of the raster's area. Re-weighting alone takes
> the pressure ratio from $8.4\times$ in the interpolator's favour to $0.44\times$
> (at most $1.6\times$ under a coarser discretisation of that measure), and the
> streamwise-velocity ratio from $6\times$ against it to $85$--$560\times$ against it.
> The benchmark's field-MSE ranking is
> not a property of the methods; it is a property of the weighting, and no published
> AirfRANS comparison states which one it uses.

That is the finding. It is stronger than the number it replaces and it cannot be
re-opened by the objection that killed the old one.

### 7.2 Everywhere `8.4\times`, `2.6\times` on `mse_v`, or "1.9x on the three reported channels" appears

(intro, contributions list, `tab:interp` discussion, conclusion.) Every one of these is
measure-contingent with a **sign flip**. They must be quoted as a pair or not at all:

> $\texttt{mse\_p}$ $75.0$ against $629.6$ area-uniform, $6012$ against $2646$
> node-weighted; $\texttt{mse\_v}$ $0.0336$ against $0.0999$ area-uniform, $4.26$ against
> $0.282$ node-weighted. Over $u,v,p$ in standardised units the interpolator is
> $1.99\times$ better area-uniform and $8.30\times$ worse node-weighted.

### 7.3 New table `tab:measure`, the defensive exhibit

| measure | `mse_u` | `mse_v` | `mse_p` | `mse_nut` |
|---|---|---|---|---|
| area-uniform (r128 crop), interp / Transolver | 0.782 / 0.127 | **0.0336** / 0.0999 | **75.0** / 629.6 | 8.84e-8 / **6.15e-9** |
| node-weighted (same cells, same errors) | 30.7 / **0.358** | 4.26 / **0.282** | 6012 / **2646** | 4.72e-7 / **5.12e-8** |

Caption must carry: only the cell weights differ; both arms reproduce their published
rows at relative error $0.00\mathrm{e}{+}00$; node weights are node counts per fluid
cell; the re-weighting is conservative (§4 caveat 2) so the true node measure is more
adverse to the interpolator; three seeds, per-seed values in the artifact.

### 7.4 `body.tex:450-452` — add the asymmetry that runs the other way

Append to "The input asymmetry runs **against** the baseline":

> The objective asymmetry runs the other way, and we measure it rather than concede it.
> The interpolator is selected by cross-validation on the area-uniform grid measure our
> tables report, while the surrogate minimises a per-point loss under the dataset's own
> node distribution, which places $55.5\%$ of its nodes inside $0.02c$ against $0.51\%$
> of the raster's cells---a factor of $110$. Re-scoring the identical predictions under
> node-count weights reverses every channel
> (\autoref{tab:measure}; \texttt{scripts/measure\_asymmetry.py}).

### 7.5 `body.tex:546-550` — the localisation claim, now measured on both arms

Currently *"it says precisely what a learned surrogate buys on this benchmark: the first
cell off the wall…"* — inferred. Replace with the measurement:

> We now measure this on the surrogate rather than inferring it. Band by band, the ratio
> of interpolation error to Transolver error on $u$ runs
> $906\times$ / $74\times$ / $30\times$ / $5.3\times$ / $0.57\times$ / $0.56\times$ /
> $0.31\times$ from $0$–$0.005c$ outward: Transolver is nearly three orders of magnitude
> more accurate inside the first half-percent of a chord and a factor of three
> \emph{less} accurate beyond $0.5c$. The two methods are accurate in different places,
> and which one a field-MSE table declares the winner is decided by the weighting.
> Transolver's own error is far-field spread---$71.9\%$ of its $u$ squared error lies
> beyond $0.5c$, where the interpolator carries $3.6\%$ of its own.

### 7.6 `sec:limitations` — delete the false sentence, replace the open item

The bullet stating the point-space head-to-head needs a `PointNormalizer` *"the
checkpoints do not store"* is **factually wrong** (§1). Replace with:

> The deployed checkpoints do store both normalisers, and we use them: the head-to-head
> under the dataset's node measure is run in \autoref{tab:measure}. What remains open is
> a head-to-head at native \emph{resolution}: both arms there are still scored on $r128$
> fields, and the raster's own round-trip error at the native nodes exceeds Transolver's
> per-node error by $418\times$ pooled and $495\times$ inside $0.005c$, so the $r128$
> protocol cannot adjudicate boundary-layer skill for either method. Scoring the
> interpolator at native nodes would change representation and measure together and is
> not equivalent to the re-weighting reported here.

### 7.7 Correct the same sentence in `interpolation_resolution_ladder.md:13` and §1 of that file

It repeats the normaliser claim. Both should point at this document.

### 7.8 What does **not** change

The train-size ladder, the `reynolds`/`aoa` split findings, the covariate null, the
force-coefficient block, the resolution ladder, and the five adversarial controls are
untouched: none of them is a cross-method field-MSE comparison. The corrector deltas
(`-9/-21/-25%`) are within-backbone and untouched. The far-field claim is untouched and
in fact now has a second witness: beyond $0.5c$ the interpolator is $3.2\times$ better
than Transolver on $u$ and $250\times$ better on $p$.

---

## 8. Rebuttal lines

> **Reviewer:** the 8.4x is a measure choice and you cannot distinguish it from a
> finding.
>
> **Response:** you were right, and we measured it rather than argued about it. AirfRANS
> places $64\%$ of its nodes inside $0.05c$, which is $1.2\%$ of the raster's area — a
> factor of $52$, and $110$ at $0.02c$. Re-weighting the *identical* grid errors of the
> *identical* published rows by node count reverses the sign on every channel:
> $\texttt{mse\_p}$ goes from $8.39\times$ in the interpolator's favour to $0.44\times$.
> We withdrew the number from the abstract. What replaced it is the measurement itself:
> on this benchmark the field-MSE ranking of a $7.35$M-parameter Transolver against a
> seven-scalar interpolator is decided by a weighting choice no published comparison
> states. `results/interpolation/measure_asymmetry.json`, rule pre-registered in
> `cdf08f5` before the run.

> **Reviewer:** you never decompose the *surrogate's* error by wall distance, so the
> localisation thesis is measured on one arm.
>
> **Response:** it is now measured on both. Band by band the ratio of interpolation
> error to Transolver error on $u$ runs $906\times \to 0.31\times$, monotone over seven
> bands, a span of $2905\times$. Transolver is $906\times$ better inside $0.005c$ and
> $3.2\times$ worse beyond $0.5c$; its own squared error is $71.9\%$ beyond $0.5c$ where
> the interpolator's is $3.6\%$. The thesis holds and is sharper than we stated it; what
> does not hold is reading an area-uniform aggregate as a verdict on it.

> **Reviewer:** so is any of the interpolation baseline left?
>
> **Response:** all of it except the aggregate ranking. Parameter interpolation
> reproduces $98.7\%$ of the domain at $R^2 \ge 0.9996$ from seven scalars, reaches that
> from $100$ training cases, does not degrade on the `reynolds` split, and is
> indistinguishable from the exact ground-truth field on official-label force ranking.
> Those claims are not cross-method field-MSE comparisons and none of them moves. What
> moved is our willingness to call $8.4\times$ a win.

---

## 9. Cost and reproduction

| step | wall clock |
|---|---|
| block A (`--stage nodes`, CPU, numpy only) | **22 s** (7 s warm) |
| §4b sensitivity (`--stage sensitivity`, reads the artifacts) | **< 1 s** |
| blocks B/C/D (`--stage full`, 1 GPU, inference only, 3 seeds × 200 cases) | **645 s** (3.13 s/case) |
| **total scientific compute** | **under 12 minutes** |

No training, no dataset download, no write to `results/mgn`, `checkpoints/mgn` or
`mgn_run.log`. Reproduce:

```
.venv/Scripts/python.exe scripts/measure_asymmetry.py --stage nodes
.venv/Scripts/python.exe scripts/measure_asymmetry.py --stage full --device auto
.venv/Scripts/python.exe scripts/measure_asymmetry.py --stage sensitivity
```

Per-case rows for both blocks are in the artifacts (`per_case`,
`per_case_area_uniform`, `per_case_node_weighted`), so every aggregate here can be
recomputed without re-running inference.
