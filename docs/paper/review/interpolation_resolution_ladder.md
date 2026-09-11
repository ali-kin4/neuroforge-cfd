# Interpolation resolution ladder — does the localisation survive refinement?

Date: 2026-09-11. Attacks the vulnerability `sec:limitations` names itself:

> every interpolation number is scored on the $128^2$ rasterised crop; the native
> AirfRANS point clouds resolve the boundary layer far better than $0.0234c$ cells,
> and our own decomposition says the entire remaining gap lives inside the first
> cell. A reviewer is entitled to ask whether the interpolation advantage survives
> at native resolution.

Producer: `scripts/interpolation_resolution_ladder.py` →
`results/interpolation/interp_resolution_ladder.json`.
The decision rule was committed in **`10f71a6` (17:44)**; the result JSON was written
at **22:07** the same day. Two later commits (`d009cc2`, `24a4528`) are recorded
amendments, both described below and both visible in the log.

---

## 0. The verdict, stated as the rule returns it

**PARTIAL.** Not `LOCALISATION HOLDS`, not `LOCALISATION DISSOLVES`.

The rule required the near-wall squared-error share to stay at or above `0.85` on
**both** $u$ and $v$ at every rung. At $512^2$ the $v$ share is **0.8417**. That is
the only clause that fails; $u$ holds at 0.8792, and nothing in the `DISSOLVES`
branch fires. **No threshold was moved.** Had I written `0.80` instead of `0.85`
the rule would read `HOLDS`; I wrote `0.85`, before seeing any number, and it
stands.

So the one-line answer a reviewer should get is:

> The localisation survives refinement in every sense the objection asserts — the
> domain fraction, the $R^2$ floor beyond $0.05c$, and the absolute far-field error
> are all flat under a $4\times$ refinement — but the near-wall *share* of the total
> error erodes from 92%/90% to 88%/84%, because the near-wall error itself falls
> by 45% while the far field stays put. The share statistic dilutes; the structure
> does not dissolve. One channel, surface pressure, genuinely degrades.

The three things the paper actually claims, at all three rungs:

| claim as printed | $128^2$ | $256^2$ | $512^2$ |
|---|---:|---:|---:|
| "$99.5\%$ of the domain" reproduced without learning ($\tau=0.99$) | 0.99493 | 0.99491 | **0.99497** |
| "beyond $0.05c$ it reproduces every channel at $R^2\ge0.9996$" | holds | holds | **holds** |
| absolute MSE beyond $0.05c$, $u$ | 0.04617 | 0.04548 | **0.04577** |

and the two that move:

| | $128^2$ | $256^2$ | $512^2$ |
|---|---:|---:|---:|
| SE share of $u$ inside $0.02c$ | 0.9238 | 0.9010 | **0.8792** |
| SE share of $v$ inside $0.02c$ | 0.8979 | 0.8552 | **0.8417** |

---

## 1. What this can and cannot settle

There is **no Transolver row at $256^2$ or $512^2$**. The point-space head-to-head
still needs Transolver inference with a `PointNormalizer` fitted on 800 train
clouds, which the checkpoints do not store. **Nothing here may be quoted as
"the interpolation advantage survives at native resolution."**

What it does settle is the separable, and separately attackable, half: the claim
that the *localisation* — 92% of the $u$ error inside a band of $0.02c$ holding
0.5% of cells — is an artifact of a grid whose cells ($0.0234c$) are **wider than
the band the claim is about**. At $512^2$ that band is 3.4 cells across. The
claim's structure is measured there, not inferred.

---

## 2. Protocol identity — the gate everything rests on

The estimator is not re-selected per rung. `build_weights` is a function of the
case **names** only (it never sees a field), so the weight matrix is bit-identical
at every resolution; the script hashes it and asserts equality:

| rung | `W` sha256 (first 32) | CV *at this rung* would select | geometry verified |
|---|---|---|---|
| $128^2$ | `739c588bb8230e26c123a73131b9c2e5` | `krr_s1.0_l0.001_wU0.25` | 3 cases, bitwise |
| $256^2$ | `739c588bb8230e26c123a73131b9c2e5` | `krr_s1.0_l0.001_wU0.25` | 3 cases, bitwise |
| $512^2$ | `739c588bb8230e26c123a73131b9c2e5` | `krr_s1.0_l0.001_wU0.25` | 3 cases, bitwise |

Identical case lists in identical order at every rung (asserted), zero train/test
name overlap (asserted), `n_cases == 200` from `evaluate_cases` (asserted). The
re-run cross-validation independently re-selects the frozen config at all three
rungs — reported, never used.

**The $128^2$ rung reproduces the published row exactly.** This gates the run; a
failure aborts before any 256/512 number is written.

| key | published | reproduced | rel. error |
|---|---:|---:|---:|
| `mse_u` | 0.7816026776 | 0.7816026776 | **0.00e+00** |
| `mse_v` | 0.03361523305 | 0.03361523305 | **0.00e+00** |
| `mse_p` | 75.03145038 | 75.03145038 | **0.00e+00** |
| `surface_mse_p` | 10988.62221 | 10988.62221 | **0.00e+00** |
| band `0-0.02c` `r2_u` | 0.83952 | 0.83951654 | 4.1e-06 |
| band `0-0.02c` `se_share_u` | 0.9238 | 0.92379515 | 5.3e-06 |
| band `0-0.02c` `se_share_v` | 0.8979 | 0.89789626 | 4.2e-06 |
| band `0-0.02c` `cell_frac` | 0.0051 | 0.00506991 | 5.9e-03 † |
| $F_{\text{band}}(0.99)$ | 0.995 | 0.99493009 | 7.0e-05 |

† compared at the two significant figures it is published to. The first dry run
failed this check at a five-figure tolerance; `d009cc2` fixed the tolerance, not
the protocol, and says so.

**One permitted optimisation, proven exact.** `signed_distance` costs 25.7 s/case
at $512^2$ — 86 minutes for 200 cases, dominating everything else.
`airfrans_loader._sim_to_pair` built the cached `FlowField.sdf`/`.mask` by calling
*those same two functions on those same two arguments*, so the script primes
`_GEOM_CACHE` from the cached arrays and **recomputes from scratch on 3 cases per
rung, asserting exact array equality** (dtype included). The cache is cleared
between rungs; its key is the case name, which carries no resolution.

---

## 3. Per-rung headline, with the `h/s` column

`s` is the source cloud's median nearest-neighbour spacing, measured on this same
test split by `scripts/floor_resolution_ladder.py`:
$s_{\text{far}} = 0.004595c$, $s_{\text{wall}} = 4.43\times10^{-5}c$.

| level | $h$ | $h/s_{\text{far}}$ | $h/s_{\text{wall}}$ | flag | `mse_u` | `mse_v` | `mse_p` | `mse_nut` | `surface_mse_p` |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| $128^2$ | 0.02344 | 5.10 | 529 | OK | 0.7816 | 0.03362 | 75.03 | 8.844e-08 | 10989 |
| $256^2$ | 0.01172 | 2.55 | 264 | OK | 0.6109 | 0.02455 | 80.25 | 8.798e-08 | 93105 |
| $512^2$ | 0.00586 | **1.28** | 132 | **NEAR LIMIT** | 0.5302 | 0.02289 | 75.49 | 8.746e-08 | 56126 |

Two readings of the `h/s` column, and the asymmetry is the defensive point:

* **The interpolation-floor caveat binds only in the far field, and only at 512.**
  At $h/s_{\text{far}} = 1.28$ the $512^2$ raster is resolving the source cloud's own
  spacing, so far-field derivatives there are partly those of a piecewise-linear
  interpolant. Pre-declared consequence: the outer-MSE arm of the verdict is read
  on the **$128\to256$ pair**, where both rungs sit at $h/s_{\text{far}}\ge2.55$.
* **It never binds where this finding lives.** In the near-wall band the
  body-fitted mesh is $132\times$ finer than the raster even at $512^2$. The
  $0$–$0.02c$ band is never over-resolved at any rung, so the localisation
  statistics are not contaminated by the ladder's own limit.

Three of the four volume channels **improve or hold** under refinement: `mse_u`
falls 32% ($0.782\to0.530$), `mse_v` falls 32% ($0.0336\to0.0229$), `mse_nut` is
flat, and `mse_p` returns to its $128^2$ value after a $7\%$ excursion at $256^2$
($75.03\to80.25\to75.49$). Refining the grid does not hurt the
parameter interpolator; the $128^2$ raster was, if anything, making its job
*harder* by aliasing the near-wall truth it had to match.

### 3b. Surface pressure — the one channel that genuinely degrades

`surface_mse_p` moves $10989\to93105\to56126$. That absolute number overstates the
effect, because at $128^2$ the $0.02c$ wall band is sub-cell so the bilinear
sampler smooths the **truth** as well as the prediction. Dividing by the variance
of the sampled truth is the resolution-comparable statistic (added post-hoc in
`24a4528`, no threshold touched):

| level | pooled MSE | Var(sampled truth) | MSE/Var | case-mean MSE |
|---|---:|---:|---:|---:|
| $128^2$ | 11190.8 | 3.846e+06 | **0.00291** | 10988.6 |
| $256^2$ | 94529.3 | 1.118e+07 | **0.00845** | 93104.6 |
| $512^2$ | 57381.2 | 1.053e+07 | **0.00545** | 56125.8 |

The case-mean column reproduces `evaluate_cases`' `surface_mse_p` exactly at every
rung — a free cross-check that the block samples the same field the paper's metric
does. Standardised, the degradation is $1.9\times$ from 128 to 512, not $5\times$,
and it is **non-monotone** (256 is worse than 512). But it is real, and it lands on
the one channel where the surrogate wins. **This must be conceded in the paper.**

---

## 4. Band decomposition, bands fixed in **physical** units

Every edge is a signed distance in chord units, identical at every rung — the
lesson `floor_resolution_ladder.py` records: a band defined in cells is physically
thinner on a finer grid and compares different regions across levels.

**Cell fraction** (the convergence witness; on a uniform crop cell fraction *is*
area fraction, which is why these numbers are comparable across rungs at all):

| band | $128^2$ | $256^2$ | $512^2$ |
|---|---:|---:|---:|
| $0$–$0.02c$ | 0.00507 | 0.00509 | 0.00503 |
| $0.02$–$0.05c$ | 0.00742 | 0.00753 | 0.00763 |
| $0.05$–$0.15c$ | 0.02942 | 0.02988 | 0.02996 |
| $0.15$–$0.5c$ | 0.15831 | 0.15956 | 0.16015 |
| $>0.5c$ | 0.79978 | 0.79794 | 0.79722 |

The $0$–$0.02c$ fraction is stable at $\approx0.005$ and sits just above the
geometric area fraction a $0.02c$ band around a median perimeter of $2.07c$ implies
($0.0046$). At $128^2$ that band is sub-cell and its fraction is discretisation
inflated; the stability confirms the inflation is small and the band is genuinely
resolved by $512^2$ ($3.4$ cells across).

**Pooled $R^2$**:

| band | $u$ 128 / 256 / 512 | $v$ 128 / 256 / 512 | $p$ 128 / 256 / 512 |
|---|---|---|---|
| $0$–$0.02c$ | 0.83952 / 0.87945 / **0.89693** | 0.97333 / 0.98249 / **0.98346** | 0.99680 / 0.99667 / 0.99691 |
| $0.02$–$0.05c$ | 0.99687 / 0.99655 / 0.99587 | 0.99908 / 0.99905 / 0.99900 | 0.99868 / 0.99867 / 0.99868 |
| $0.05$–$0.15c$ | 0.99971 / 0.99970 / 0.99968 | 0.99975 / 0.99975 / 0.99975 | 0.99961 / 0.99960 / 0.99961 |
| $0.15$–$0.5c$ | 0.99977 / 0.99978 / 0.99978 | 0.99995 / 0.99995 / 0.99995 | 0.99992 / 0.99992 / 0.99992 |
| $>0.5c$ | 0.99989 / 0.99989 / 0.99989 | 0.99999 / 0.99999 / 0.99999 | 0.99996 / 0.99996 / 0.99996 |

Near-wall $R^2$ **improves** with refinement on $u$ and $v$; everything beyond
$0.05c$ is flat in the fifth decimal. The strict per-case-centred $R^2_{\text{pc}}$
in the wall band likewise improves, $0.7539\to0.8127\to0.8408$ on $u$ — the
statistic the paper already flags as the harsh one.

**SE share** (the load-bearing statistic, unaffected by the centring choice):

| band | $u$ 128 / 256 / 512 | $v$ 128 / 256 / 512 | $p$ 128 / 256 / 512 |
|---|---|---|---|
| $0$–$0.02c$ | **0.9238 / 0.9010 / 0.8792** | **0.8979 / 0.8552 / 0.8417** | 0.5346 / 0.5603 / 0.5324 |
| $0.02$–$0.05c$ | 0.0180 / 0.0256 / 0.0358 | 0.0402 / 0.0594 / 0.0663 | 0.2375 / 0.2237 / 0.2383 |
| $0.05$–$0.15c$ | 0.0052 / 0.0069 / 0.0085 | 0.0298 / 0.0413 / 0.0445 | 0.1537 / 0.1461 / 0.1548 |
| $0.15$–$0.5c$ | 0.0167 / 0.0205 / 0.0235 | 0.0180 / 0.0250 / 0.0269 | 0.0518 / 0.0489 / 0.0522 |
| $>0.5c$ | 0.0364 / 0.0460 / 0.0531 | 0.0141 / 0.0192 / 0.0206 | 0.0223 / 0.0209 / 0.0223 |

This is the clause that fails, and the finer band grid says exactly why.

---

## 5. Why the share erodes — the numerator shrank

$R^2$'s denominator moves between rungs, which is why the rule reads a triple and
not $R^2$ alone. The finer bands separate numerator from denominator:

| band | cell frac 128/256/512 | `mse_u` 128 / 256 / 512 | share $u$ 128/256/512 | $R^2_u$ 128/256/512 |
|---|---|---:|---|---|
| $0$–$0.005c$ | .00138/.00145/.00149 | **466.7 / 327.9 / 257.7** | .824/.778/.722 | .507/.655/**.723** |
| $0.005$–$0.01c$ | .00118/.00121/.00118 | 35.79 / 32.32 / 38.49 | .054/.064/.086 | .953/.955/.948 |
| $0.01$–$0.02c$ | .00251/.00242/.00236 | 14.28 / 14.87 / 16.08 | .046/.059/.072 | .980/.979/.977 |
| $0.02$–$0.03c$ | .00222/.00247/.00248 | 3.537 / 3.792 / 4.471 | .010/.015/.021 | .994/.994/.993 |
| $0.03$–$0.05c$ | .00521/.00506/.00515 | 1.199 / 1.248 / 1.534 | .008/.010/.015 | .998/.998/.997 |
| $0.05$–$0.075c$ | .00663/.00681/.00687 | 0.3142 / 0.3183 / 0.3501 | .003/.004/.005 | .9994/.9994/.9993 |
| $0.075$–$0.1c$ | .00730/.00723/.00724 | 0.1300 / 0.1178 / 0.1220 | .001/.001/.002 | .9997/.9998/.9998 |
| $0.1$–$0.15c$ | .01549/.01584/.01586 | 0.0653 / 0.0758 / 0.0775 | .001/.002/.002 | .9999/.9998/.9998 |
| $0.15$–$0.25c$ | .03704/.03685/.03700 | 0.0736 / 0.0728 / 0.0718 | .004/.004/.005 | .9998/.9998/.9998 |
| $0.25$–$0.5c$ | .12127/.12272/.12315 | 0.0854 / 0.0803 / 0.0797 | .013/.016/.019 | .9998/.9998/.9998 |
| $>0.5c$ | .79978/.79794/.79722 | **0.0356 / 0.0353 / 0.0354** | .036/.046/.053 | .99989 ×3 |

Read the two bold rows together. In the innermost $0.005c$ — $0.15\%$ of the domain
— the absolute $u$ error falls **45%** ($466.7\to257.7$) while its $R^2$ rises from
$0.507$ to $0.723$. In the far field, $80\%$ of the domain, the absolute $u$ error
moves by **0.6%** ($0.0356\to0.0354$). The far field did not get worse; the wall got
better. A share is a ratio, so when the dominant term shrinks by half and the rest
holds, every other band's share must rise even though none of them degraded.

That is the honest mechanism of the `PARTIAL`. It is not a defence of the
threshold — the threshold was missed — it is the decomposition that says which
direction the miss points.

### The outer region, which is the arm the rule actually tests

| level | $n$ cells | `mse_u` | `mse_v` | `mse_p` |
|---|---:|---:|---:|---:|
| $128^2$ | 3 205 263 | 0.0461686 | 0.00211255 | 17.360 |
| $256^2$ | 12 818 409 | 0.0454756 | 0.00212803 | 17.597 |
| $512^2$ | 51 268 033 | 0.0457729 | 0.00213751 | 17.579 |

$128\to256$ growth: $u$ **0.985**, $v$ **1.007**, $p$ **1.014** — against a
pre-registered failure threshold of $2.0$. The far field is flat to within 1.5%
across a $16\times$ increase in cell count. This is the single most direct refutation
of "the interpolator only looks good because the coarse raster hides its error".

### The oversampled-cell control, reported and then discounted on its own evidence

| level | $n$ cells | frac of outer | mean $sdf$ | `mse_u` | `mse_v` | `mse_p` |
|---|---:|---:|---:|---:|---:|---:|
| $128^2$ | 362 160 | 0.113 | 0.3700c | 0.2000 | 0.01211 | 121.1 |
| $256^2$ | 267 304 | 0.021 | 0.2091c | 0.3735 | 0.02686 | 281.3 |
| $512^2$ | 108 401 | **0.002** | **0.1592c** | 0.8283 | 0.05899 | 160.1 |

Restricting to cells holding $\ge8$ source points was intended to remove the
$h/s$ confound. It cannot, and the two added columns prove why rather than leaving
it arguable: the source cloud is densest near the body, so as $h$ shrinks the
qualifying cells retreat toward the wall — from 11.3% of the outer region at mean
distance $0.370c$ to **0.2%** at mean distance $0.159c$. Its rising MSE is a region
change, not a resolution effect. **It is not evidence of far-field degradation and
must not be cited as such.** The pre-registered outer-MSE arm was defined on all
outer cells and still is.

---

## 6. The fraction of the domain reproduced without learning

Definition, pinned before the run so it reproduces the sentence the paper already
prints: $F_{\text{band}}(\tau)$ is the cumulative fluid-cell fraction of the
outermost run of bands, every one of which has $\min_{u,v,p} R^2 \ge \tau$. At
$\tau=0.99$ on the paper's five bands this must return $0.995$ at $128^2$ — and it
does ($0.99493$), which is part of the gate.

**Pooled $R^2$, paper's five bands:**

| $\tau$ | $128^2$ | cut | $256^2$ | cut | $512^2$ | cut |
|---|---:|---|---:|---|---:|---|
| 0.9 | 0.99493 | $0.02$–$0.05c$ | 0.99491 | $0.02$–$0.05c$ | **0.99497** | $0.02$–$0.05c$ |
| 0.99 | 0.99493 | $0.02$–$0.05c$ | 0.99491 | $0.02$–$0.05c$ | **0.99497** | $0.02$–$0.05c$ |
| 0.999 | 0.98751 | $0.05$–$0.15c$ | 0.98739 | $0.05$–$0.15c$ | 0.98733 | $0.05$–$0.15c$ |
| 0.9996 | 0.98751 | $0.05$–$0.15c$ | 0.98739 | $0.05$–$0.15c$ | 0.98733 | $0.05$–$0.15c$ |

The $\tau=0.9996$ row clears the bar **narrowly at every rung and by the same
margin**: the binding entry is $p$ in $0.05$–$0.15c$, at $0.999606$ / $0.999604$ /
$0.999607$. That is a thin claim in the published paper already, and refinement
neither strengthens nor weakens it — at $512^2$ it is marginally the largest of the
three. It is worth knowing that the sentence rests on a fourth-decimal margin,
whatever the resolution.

**Finer band grid** (localises the cut better than five bands allow):

| $\tau$ | $128^2$ | cut | $256^2$ | cut | $512^2$ | cut |
|---|---:|---|---:|---|---:|---|
| 0.9 | 0.99862 | $0.005$–$0.01c$ | 0.99855 | $0.005$–$0.01c$ | 0.99851 | $0.005$–$0.01c$ |
| 0.99 | 0.99493 | $0.02$–$0.03c$ | 0.99491 | $0.02$–$0.03c$ | 0.99497 | $0.02$–$0.03c$ |
| 0.999 | 0.98751 | $0.05$–$0.075c$ | 0.98739 | $0.05$–$0.075c$ | 0.98733 | $0.05$–$0.075c$ |
| 0.9996 | 0.98087 | $0.075$–$0.1c$ | 0.98058 | $0.075$–$0.1c$ | 0.98047 | $0.075$–$0.1c$ |

**Strict, per-case-centred $R^2_{\text{pc}}$** — reported because it is the harsh
statistic and it must not be buried:

| $\tau$ | $128^2$ | $256^2$ | $512^2$ |
|---|---:|---:|---:|
| 0.9 (five bands) | 0.99493 | 0.99491 | 0.99497 |
| 0.99 (five bands) | 0.99493 | 0.99491 | 0.99497 |
| 0.99 (fine bands) | 0.99271 | 0.99245 | 0.99248 |
| 0.999 / 0.9996 | **0** | **0** | **0** |

The zeros are not a regression: they are the statistic the paper already
acknowledges when it writes "$\ge0.9965$ with each case's own band mean removed".
Under $R^2_{\text{pc}}$ the outermost band sits at $0.9965$ on $u$, so no region
clears $\tau=0.999$ at **any** resolution — including the published $128^2$. The
$0.9996$ claim is a pooled-$R^2$ claim, it is labelled as one in `tab:interp_bands`,
and it is resolution-stable as a pooled claim. Nothing new is broken; the
refinement did not create this and does not remove it.

**Headline: the paper's most graspable number, $99.5\%$ of the domain, is stable to
four decimal places under a $4\times$ refinement, and at $512^2$ it is marginally
larger, not smaller.**

---

## 7. Direct answers to the question as posed

**Does the structure of the finding survive refinement?** Yes, with one conceded
channel and one honest caveat about the share statistic.

* *The gap stays confined to a physically thin near-wall band* — **yes, and it
  tightens.** At $512^2$, where the $0.02c$ band is 3.4 cells wide rather than
  sub-cell, $72\%$ of the $u$ error is inside $0.005c$, which is $0.15\%$ of the
  domain. The localisation is sharper at fine resolution than the $128^2$ table
  could show.
* *The rest of the domain is reproduced as well or better* — **yes.** Outer-region
  MSE flat within 1.5%; $R^2$ beyond $0.05c$ identical in the fifth decimal;
  $F_{\text{band}}(0.99)$ flat at $0.995$.
* *Does the interpolator degrade broadly as the grid refines?* — **no, the
  opposite.** `mse_u` and `mse_v` each fall about a third; `mse_p` and `mse_nut`
  are flat.
* *Does the near-wall share drop?* — **yes, from 0.92/0.90 to 0.88/0.84**, which is
  the pre-registered miss. It drops because the near-wall error fell 45% while the
  far field moved 0.6%.
* *Is there anything that gets genuinely worse?* — **yes: surface pressure.**
  Standardised by its own sampled-truth variance it rises $1.9\times$ from 128 to
  512 (non-monotonically). This is the channel the surrogate wins, and the paper
  should say that refinement widens, not narrows, that particular gap — while
  noting there is no surrogate row at 256/512 to quantify by how much.

**Is the finding partly a rasterisation effect?** For the domain fraction and the
$R^2$ floor, no. For the exact share figures 92%/90%, yes to the extent that those
are resolution-specific numbers: they are 88%/84% at $512^2$. The paper should
quote them with the resolution attached, and should quote the resolution-stable
statements as the headline.

---

## 8. Exact changes the paper should make

**I did not edit `body.tex`.** These are the changes, keyed to text rather than
line numbers because `body.tex` is being compacted concurrently.

### 8.1 `sec:limitations`, the bullet beginning "The point-space head-to-head was not run"

Replace the middle of the bullet. Currently:

> The native AirfRANS point clouds resolve the boundary layer far better than
> $0.0234c$ cells, and our own decomposition says the entire remaining gap lives
> inside the first cell. A reviewer is entitled to ask whether the interpolation
> advantage survives at native resolution, and we cannot answer from these runs:
> doing so needs Transolver inference with the \texttt{PointNormalizer} fitted on
> the $800$ train point clouds, and the checkpoints do not store the normaliser.
> This is the single most valuable follow-up in the paper.

Proposed:

> The native AirfRANS point clouds resolve the boundary layer far better than
> $0.0234c$ cells, and our own decomposition says the entire remaining gap lives
> inside the first cell. We can now answer half of this. A resolution ladder at
> $128^2$, $256^2$ and $512^2$ on the same $800/200$ split, with the estimator
> frozen (the weight matrix is a function of the case names alone and hashes
> identically at every rung) and the bands fixed in chord units, finds the
> \emph{localisation} resolution-stable: the domain fraction reproduced without
> learning is $0.9949/0.9949/0.9950$, every channel still clears $R^2\ge0.9996$
> beyond $0.05c$ at all three rungs, and absolute MSE beyond $0.05c$ is flat within
> $1.5\%$ across a $16\times$ increase in cell count. At $512^2$, where the $0.02c$
> band is $3.4$ cells wide rather than sub-cell, the error is \emph{more} tightly
> localised: $72\%$ of the $u$ error lies within $0.005c$, $0.15\%$ of the domain.
> What does move is the share itself, $0.92/0.90$ to $0.88/0.84$ on $u$/$v$, because
> the near-wall error falls $45\%$ while the far field moves $0.6\%$; our
> pre-registered rule reads that as \textsc{partial}, not as the localisation
> holding, and we report it as \textsc{partial}. Surface pressure is the one channel
> that genuinely degrades under refinement ($1.9\times$ once standardised by the
> sampled truth's own variance), and it is the channel the surrogate wins. What the
> ladder cannot settle is the \emph{advantage}: there is no Transolver row at
> $256^2$ or $512^2$, because that needs Transolver inference with the
> \texttt{PointNormalizer} fitted on the $800$ train point clouds and the checkpoints
> do not store the normaliser. The point-space head-to-head remains the single most
> valuable follow-up in the paper
> (\texttt{scripts/interpolation\_resolution\_ladder.py},
> \texttt{results/interpolation/interp\_resolution\_ladder.json}).

### 8.2 `sec:interp`, the paragraph "The boundary is one grid cell wide, and that is the contribution"

Two edits.

1. Attach the resolution to the share figures. Replace

   > \textbf{$92\%$ of its $u$ error and $90\%$ of its $v$ error lie within $0.02c$
   > of the wall---a band holding $0.5\%$ of cells, thinner than the $0.0234c$ grid
   > spacing---and beyond $0.05c$ it reproduces every channel at $R^2\ge0.9996$}

   with

   > \textbf{$92\%$ of its $u$ error and $90\%$ of its $v$ error lie within $0.02c$
   > of the wall---a band holding $0.5\%$ of cells, thinner than the $0.0234c$ grid
   > spacing---and beyond $0.05c$ it reproduces every channel at $R^2\ge0.9996$}.
   > Both statements survive refinement, and the first is resolution-dependent in a
   > direction worth stating: at $512^2$, where that band is $3.4$ cells wide, the
   > shares are $88\%$ and $84\%$ and $72\%$ of the $u$ error sits inside $0.005c$
   > ($0.15\%$ of the domain), because the near-wall error falls $45\%$ under
   > refinement while the far field moves $0.6\%$ (\autoref{tab:interp_ladder}).

2. Add a new table `tab:interp_ladder`, the defensive exhibit:

   | level | $h$ | $h/s_{\text{far}}$ | `mse_u` | SE share $u$, $0$–$0.02c$ | $R^2_u$, $0$–$0.02c$ | MSE $u$, $>0.05c$ | $F(0.99)$ |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | $128^2$ | 0.0234 | 5.10 | 0.782 | 0.924 | 0.840 | 0.0462 | 0.9949 |
   | $256^2$ | 0.0117 | 2.55 | 0.611 | 0.901 | 0.879 | 0.0455 | 0.9949 |
   | $512^2$ | 0.0059 | 1.28 | 0.530 | 0.879 | 0.897 | 0.0458 | 0.9950 |

   Caption should carry: bands fixed in chord units; estimator frozen (identical
   weight-matrix hash at every rung); the $128^2$ row reproduces
   \autoref{tab:interp} at relative error $0$; $s$ is the source cloud's far-field
   spacing, so $512^2$ is flagged as approaching the point where the raster samples
   the interpolant rather than the solution, and the outer-MSE trend is read on
   $128\to256$; the pre-registered verdict is \textsc{partial}, missing only the
   $0.85$ share floor on $v$ at $512^2$ ($0.8417$).

### 8.3 Intro paragraph "Learning earns its keep in the first cell off the wall" and the contributions list

Both quote "$92\%$ … $90\%$ … $R^2\ge0.9996$ … $80\%$ of the domain carries $3.6\%$".
Leave the numbers; append one clause once, at the first occurrence:

> These hold at $256^2$ and $512^2$ as well, with the near-wall share easing to
> $88\%$/$84\%$ as the near-wall error itself falls by $45\%$.

Do **not** add "the coarse-raster objection is answered" anywhere. It is answered
for the localisation and not for the head-to-head, and the limitations bullet is
the only place that distinction can be carried accurately.

### 8.4 Where surface pressure is discussed

Add, wherever `surface_mse_p` is compared to Transolver's $9110$:

> This is also the one channel where refinement hurts the interpolator: standardised
> by the variance of its own sampled truth, its surface-pressure error rises
> $1.9\times$ from $128^2$ to $512^2$. We have no surrogate row at those resolutions,
> so we do not claim the gap widens by a stated factor---only that the channel the
> surrogate wins is the channel a finer grid favours it in.

---

## 9. Rebuttal lines

> **Reviewer:** your whole decomposition rests on a $0.02c$ band measured on a grid
> whose cells are $0.0234c$. The band is thinner than one cell. This is a
> rasterisation artifact.
>
> **Response:** we ran the ladder. At $512^2$ that band is $3.4$ cells wide, on the
> same $800/200$ split, with the estimator frozen (identical weight-matrix hash at
> every rung) and the bands fixed in chord units. The domain fraction reproduced
> without learning is $0.9949/0.9949/0.9950$; every channel still clears
> $R^2\ge0.9996$ beyond $0.05c$; absolute MSE beyond $0.05c$ is flat within $1.5\%$
> across a $16\times$ increase in cell count; and the localisation *tightens* — $72\%$
> of the $u$ error is inside $0.005c$ at $512^2$. The $128^2$ rung reproduces our
> published numbers at relative error $0.00\mathrm{e}{+}00$, so the finer rows are
> anchored to the table you are reading. Our pre-registered rule nonetheless returns
> \textsc{partial}, because the near-wall share reaches $0.8417$ on $v$ at $512^2$
> against a $0.85$ floor we set before running; we report that, and we report why —
> the near-wall error fell $45\%$ while the far field moved $0.6\%$, so the share
> diluted. `results/interpolation/interp_resolution_ladder.json`.

> **Reviewer:** so the interpolation advantage survives at native resolution?
>
> **Response:** we do not claim that and the ladder cannot show it. There is no
> Transolver row at $256^2$ or $512^2$. What the ladder shows is that the
> *localisation* is not an artifact of the deployed grid. The point-space
> head-to-head remains an open follow-up and is still listed as one.

---

## 10. Cost

| stage | wall clock |
|---|---|
| rasterise `test r256 n200` | 288 s |
| rasterise `test r512 n200` | 3 231 s (54 min, thread pool) |
| rasterise `train r256 n800` | ~1 320 s (22 min; process killed before it logged its own figure) |
| rasterise `train r512 n800` | 41 min for the final 671 cases in 8 independent shard processes (3.3 s/case), on top of ~90 min lost to two OOMs and one self-inflicted bug (see below) |
| assemble the `train r512` cache | 35 s |
| **the ladder itself, all three rungs** | **290 s** (128: 24 s, 256: 42 s, 512: 127 s) |
| 128-only dry run + 128/256 stage | 98 s + ~90 s |

Total session ≈ 5 h, of which the scientific computation is **under 10 minutes**.
Everything else is cache construction. Disk: $+7.9$ GB of rasterised caches
(`train r512` alone is 5.04 GB; `test r512` 1.26, `train r256` 1.27,
`test r256` 0.32).

Three engineering notes worth keeping, because they will recur:

* `geometry.sdf.signed_distance` costs **25.7 s/case at $512^2$** and
  `_points_inside` allocates a single $(P,S)$ float64 array of $1.98$ GiB. It is
  called twice per case inside `_sim_to_pair`, so it dominates cache building and
  caps worker concurrency at about 4–8 processes on 68 GB. A chunked
  `_points_inside` (as `_unsigned_distance` already has) would remove both limits.
* `_rasterize_pairs` uses **threads** and scales only $2.15\times$ on 16 of them
  (GIL-bound). Independent OS processes over disjoint index strides reached
  $3.3$ s/case at $512^2$ versus $16$ s/case threaded. The process path was
  validated against the serially-built cache: **bit-identical** over 24 cases
  $\times$ 6 arrays plus geometry.
* I lost ~40 minutes to a file named `signal.py` in the scratchpad, which shadowed
  the stdlib `signal` module for every process launched from that directory
  (`subprocess` imports `signal`), so every detached worker hung silently at
  `import neuroforge`. Never name a helper after a stdlib module.

---

## 11. Artifacts

* `scripts/interpolation_resolution_ladder.py` — pre-registration in the docstring,
  committed in `10f71a6` before the run; amendments `d009cc2` (gate tolerance) and
  `24a4528` (two added diagnostics) both recorded in the docstring.
* `results/interpolation/interp_resolution_ladder.json` — per-rung metrics, both
  band grids, all four $\tau$ under both $R^2$ flavours, outer-region MSE with and
  without the oversampled restriction, the surface-pressure block, the weight-matrix
  hashes, the geometry bit-equality proof, the protocol gate, timings.
* `results/MANIFEST.json` — regenerated, 81 files, hashes verified with
  `make_manifest.py --check`.
* Reuses `parameter_interpolation_baseline.py` (weights, stacks, scoring),
  `interpolation_band_control.band_decomposition` (verbatim, which is what makes the
  $128^2$ gate meaningful) and `floor_resolution_ladder.{source_positions,
  oversampled_cells}`. None of the three was edited.

Reproduce (CPU only, no GPU, no training), given the caches:

```
.venv/Scripts/python.exe scripts/interpolation_resolution_ladder.py --levels 128 256 512
```
