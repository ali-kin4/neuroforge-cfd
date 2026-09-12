# The point-space head-to-head — the interpolation headline is dead in every measure

The experiment `r2_round2.md` §6 calls "the real fix for the fatal flaw" and scores as the
step from **5/10 to 9/10**; the experiment `interpolation_baseline.md` §6 calls "the single
most valuable follow-up and it is not done here"; the experiment `measure_asymmetry.md` §4
left open. It is now run.

**Transolver against parameter interpolation at the native AirfRANS cloud nodes, 200 `full`
test cases, 35 849 332 nodes, scored per node, with no rasterisation anywhere in the
comparison path.**

Producer: `scripts/point_space_headtohead.py` — rules **P1–P4** and gates **G1–G5**
pre-registered in the module docstring and committed in **`a54cb75`** before any number
existed (one recorded amendment, `4ba4dba`, a gate *tolerance* only; no decision rule and no
threshold touched). Artifacts: `results/interpolation/point_space_headtohead.json`,
`point_space_interp.json`, `point_space_transolver.json`, `point_space_r128resample.json`,
`point_space_pilot.json`. Zero training. 2.3 h wall clock.

---

## 0. The verdict, stated first and plainly

**At native resolution the parameter interpolator loses on every channel, including the two
it won on the grid.** Per-node MSE against the native AirfRANS targets, all 200 test cases,
`R = MSE_interp / MSE_Transolver` (`>1` = Transolver better):

| | `mse_u` | `mse_v` | `mse_p` | `mse_nut` |
|---|---:|---:|---:|---:|
| interpolation @ native nodes | 93.11 | 66.52 | 41 740 | 6.920e-7 |
| Transolver @ native nodes (3 seeds) | **0.6458** | **0.4968** | **8 605** | **3.791e-8** |
| **`R`** | **144.2x** | **133.9x** | **4.85x** | **18.3x** |

**The `8.4x` pressure win does not merely vanish at native resolution. It inverts to
`4.85x` against the interpolator.** The same two arms, the same 200 cases, scored three
ways:

| measure of the SAME two predictions | `mse_p` ratio |
|---|---|
| area-uniform $128^2$ raster (the paper's headline) | **8.39x in the interpolator's favour** |
| node-count re-weighting of those raster cells (`measure_asymmetry.md` D3) | 0.44x |
| **per node, native cloud, no raster anywhere (this document)** | **0.21x — i.e. 4.85x against it** |

Pre-registered rule **P1** (`R_c > 2` on all three of `u, v, p` → `NOT-COMPETITIVE-AT-NATIVE`)
returns **NOT-COMPETITIVE-AT-NATIVE**. It was committed before the run and it fires against
the interpolator on all three channels at once, with room to spare on two of them. In
standardised units (the paper's own `Var_train` divisors) the interpolator is **24.7x worse**
over `u, v, p` — against **1.99x better** area-uniform and **8.30x worse** node-weighted.

Per seed, so this is not a seed artifact: `R_u` **147.3 / 141.0 / 144.4**, `R_p`
**5.23 / 4.17 / 5.34**, `R_v` **142.3 / 130.7 / 129.4**. Paired per case, Transolver is better
on **200/200** cases on `u`, **200/200** on `v`, **200/200** on `nut`, and **142/200** on `p`
(one-sided binomial $p = 1.3\times10^{-9}$; per-case `R_p` median 2.21, p10 0.44, p90 9.2).

**And this is the interpolator's best case, not its worst.** Giving it the native clouds
*helps* it enormously against the grid protocol it was published under: the identical
published r128 prediction resampled at the same nodes scores **400.2** on `u` and
**3.06e6** on `p`, against **97.6** and **43 777** for the native construction — the raster
was costing the interpolator a factor of **4.1x** on `u` and **69.9x** on `p` (§2.3). No
one can say the baseline was down-sampled into losing.

**What is left of the interpolation result, stated without spin.** The far-field advantage
survives on `v` and `p` and is larger than the paper claims: beyond $0.5c$ the interpolator
is **17.7x** better on `v` and **265x** better on `p`. What does **not** survive is (i) any
aggregate win on any channel, (ii) the far-field advantage **on `u`**, where at native
resolution Transolver is better in **every one of the eight bands** (§4), and (iii) the
`measure_asymmetry.md` §7.5 sentence built on that `u` far-field number.

---

## 1. What was run, and the gates that make it checkable

| arm | construction | file |
|---|---|---|
| **Transolver** | the three deployed `checkpoints/v2_transolver/seed{m}.pt`, loaded with their own `point_norm`, inferred on the native cloud | `point_space_transolver.json` |
| **interpolation, `bridge`** | the published weight matrix applied to each train case's field **on that train case's own native cloud**, Delaunay-linear, in-body queries left to the Delaunay bridge | `point_space_interp.json` |
| **interpolation, `nearfill`** | identical, except in-body queries take train case $j$'s nearest cloud-node value | same |
| **interpolation, `r128_resample`** | the published r128 grid prediction bilinearly sampled at the native nodes — the *control*, not the comparison | `point_space_r128resample.json` |
| **oracle** | per case and band, the single best of the 800 training fields, chosen knowing the test answer | `point_space_headtohead.json` §`P3` |

Gates, all asserted in the script (the run aborts on failure), all passed:

| gate | what it proves | result |
|---|---|---|
| **G1** | the weight matrix **is** the published one | rebuilt `W @ Y` reproduces `interp_full.json` at rel **1.8e-10 / 2.4e-9 / 1.6e-8** on `u/v/p` (amendment 1, §1.1) |
| **G2** | the Transolver arm **is** the deployed backbone | per-node, per-band MSE reproduces `measure_asymmetry.json` `D_node_space.mse_full` at rel **0.00e+00**, all 3 seeds, all 4 channels, all 7 bands |
| **G3** | the construction is faithful, not a handicap | the point-space prediction rasterised back to r128 scores **1.074x / 0.769x / 0.996x** of the published grid arm on `u/v/p` (§2.2) |
| **G4** | the band tables reconstruct the aggregate | `sum_b n_b · MSE_b == ` pooled SE, max rel **2.0e-16**, every arm, every channel |
| **G5** | the 14-worker partition is a disjoint cover | union == the 200 cached test names, in cache order |

G1 and G2 are the load-bearing pair, exactly as in `measure_asymmetry.md`: this is not *a*
Transolver and *an* interpolator, it is **the two rows in the paper's own tables**, evaluated
where the data actually lives.

### 1.1 Amendment 1, recorded

G1 was committed at `< 1e-9`, copied from `measure_asymmetry.py`, which reaches `0.00e+00`
because it reuses the published metric path verbatim. This script rebuilds `W @ Y` and
computes the same per-case MSE inline, so the float32 GEMM over $800\times65536$ terms and
the reduction associate differently: `mse_v` 2.371e-09 (`0.03361523297` vs `0.03361523305`),
`mse_p` 1.624e-08 (`75.03145160` vs `75.03145038`) — eight to ten significant figures. The
tolerance was raised to `1e-7` in `4ba4dba` and the measured values recorded. **P1–P4 and
their thresholds are untouched.**

### 1.2 The correction the manuscript owes, restated because it is still in the file

`body.tex`'s limitations bullet, `interpolation_baseline.md:388-391` and
`interpolation_resolution_ladder.md:13` all state that this experiment cannot be run because
*"the checkpoints do not store the normaliser."* **That is false, and this run is the proof.**
`checkpoints/v2_transolver/seed{m}.pt` contains `point_norm` (`mean_in`, `std_in`, `mean_out`,
`std_out`, `eps`) **and** `grid_norm`; `scripts/recompute_force_vs_official.py:134-158` has
loaded both since June; this script loads them through that same function and G2 reproduces
the deployed backbone bit for bit. The sentence excusing the single most important un-run
experiment in the paper is wrong on its facts, and the experiment it excused is now done.

---

## 2. The construction, and why it is the fairest available

### 2.1 What it is

The interpolator's published definition is a weight matrix over the 800 train cases applied
to the *nondimensional* field and redimensionalised with the **test** case's own $(U,\alpha)$:
$\mathrm{pred}_t = \sum_j W[t,j]\,\hat f_j$. The weights depend only on the seven
name-parsed scalars — nothing about them is grid-specific, and G1 proves they are the
published ones. What *is* grid-specific in the published run is the representation of
$\hat f_j$: a $128^2$ raster. At native nodes the identical estimator is evaluated without
any raster,

$$\mathrm{pred}_t(x) \;=\; \sum_{j=1}^{800} W[t,j]\,\hat f_j(x), \qquad x \in \text{the native nodes of case } t,$$

where $\hat f_j(x)$ is the Delaunay-barycentric interpolation of **train case $j$'s own
nondimensional field on train case $j$'s own native cloud**. That is the same interpolant
`rasterize_point_cloud` and `airfrans_loader._sim_to_pair` use to build every grid field in
this project, so no new numerical operator enters the comparison.

Four properties make this the fairest construction available, and the fourth is the one
that matters most:

1. **It has no representation ceiling.** Every train field is evaluated on its own native
   cloud, at the resolution AirfRANS actually simulated. This is the decisive contrast with
   the r128 protocol, whose *own* round-trip error is 418x Transolver's per-node error
   (`measure_asymmetry.md` Block D). Nothing here discards sub-grid structure from either arm.
2. **It is the published estimator, not a new one** (G1).
3. **It is measurably better for the interpolator than the grid protocol** (§2.3).
4. **The one genuinely ambiguous choice is disclosed and resolved in the interpolator's
   favour.** P1 reads the better of `bridge` and `nearfill` per channel. It chose `nearfill`
   on all four; on `p` that mattered — 41 740 against `bridge`'s 58 020, a 28% gift to the
   interpolator that changes `R_p` from 6.74 to 4.85 and does not change the verdict.

### 2.2 G3 — the construction is faithful (the check that could have killed it)

Declared in the docstring before the run: rasterise the point-space prediction back to r128
with the identical `rasterize_point_cloud` call the Transolver arm uses, and score it on the
published grid protocol. Exact agreement is **not** expected — the published arm rasterises
then combines with `--fill nearest` *on the raster*; this arm combines then rasterises across
clouds with different triangulations. On the first three test cases:

| | published grid arm | point-space arm, rasterised back | ratio |
|---|---:|---:|---|
| `mse_u` | 0.6540 | 0.7022 | 1.074 |
| `mse_v` | 0.02614 | 0.02011 | **0.769** |
| `mse_p` | 33.451 | 33.322 | **0.996** |

Within 7% on the worst channel and *better* on the other two. The construction reproduces
the published estimator's behaviour on the published measure. (Pre-registered stop rule: `>3x`
worse on any channel → treat as a bug and stop. Not triggered.)

### 2.3 What the grid protocol was costing the interpolator — the control that kills "you crippled the baseline"

The alternative construction — bilinearly sample the *published r128 prediction* at the
native nodes — is exactly what a reviewer would propose, and it is the one this document
refuses to use as primary, because it hands the interpolator the raster's own error as a
floor. That refusal is a claim about a number, so here is the number (in-crop pooled, the
only region `_bilinear_sample` reports without clamp contamination):

| | `u` | `v` | `p` | `nut` |
|---|---:|---:|---:|---:|
| published r128 prediction, resampled at nodes | 400.2 | 189.7 | 3.058e6 | 2.075e-6 |
| the r128 raster's own round-trip error (no model) | 281.5 | 272.6 | 3.012e6 | 1.539e-6 |
| **interpolation at native nodes (this document)** | **97.6** | **69.8** | **43 777** | **5.355e-7** |
| Transolver at native nodes | 0.668 | 0.516 | 8 990 | 3.777e-8 |

The grid representation was costing the interpolator **4.1x** on `u`, **2.7x** on `v`,
**69.9x** on `p` and **3.9x** on `nut` — and its r128 output at the nodes is, on `p`,
indistinguishable from the raster's own round-trip error (3.058e6 against 3.012e6), i.e.
**entirely representation**. The native construction is not a handicap; it is a rescue, and
the interpolator still loses on all four channels by 4.9x–146x.

### 2.4 What the construction costs the interpolator, measured rather than argued

Physical coordinates. A test node at wall distance $d_t$ sits at wall distance $d_j \neq d_t$
under train case $j$'s geometry, because the airfoils differ. Per band, $|W|$-weighted over
all 800 train cases:

| band | node frac | mean $d_t$ | mean $|d_j - d_t|$ | ratio | $|W|$-mass **inside** a train body | $|W|$-mass outside a train hull |
|---|---:|---:|---:|---:|---:|---:|
| wall ($sdf=0$) | 0.0056 | 0 | 0.00783c | $\infty$ | **47.0%** | 0 |
| 0–0.005c | **0.4096** | 0.000942c | 0.00732c | **7.77x** | **35.4%** | 0 |
| 0.005–0.01c | 0.0581 | 0.00724c | 0.00595c | 0.82x | 14.4% | 0 |
| 0.01–0.02c | 0.0587 | 0.0145c | 0.00575c | 0.40x | 7.6% | 0 |
| 0.02–0.05c | 0.0838 | 0.0328c | 0.00633c | 0.19x | 2.3% | 0 |
| 0.05–0.15c | 0.1059 | 0.0916c | 0.00638c | 0.070x | 0.11% | 0 |
| 0.15–0.5c | 0.1243 | 0.293c | 0.00589c | 0.020x | 0 | 0 |
| >0.5c | 0.1540 | 1.179c | 0.00434c | 0.0037x | 0 | 0.094% |

**This is the mechanism in one number.** Inside $0.005c$ — where **41% of every AirfRANS
cloud lives** — a query at $0.00094c$ off its own wall lands, on average, $0.0073c$ from the
train case's wall: **7.8 times further away than the boundary-layer position it is trying to
sample**, and 35% of the weight mass lands *inside* the train airfoil altogether. The
$0.0234c$ raster hid this by averaging over it. **No extrapolation is involved** — the
outside-hull mass is zero in every band but the outermost, where it is 0.09% — so P4 trigger
(a) does not fire in any band, and neither does trigger (b): `bridge` and `nearfill` differ
by $0.2$ on `u` inside $0.005c$ against an interpolator–Transolver gap of $199$. **No band is
INCONCLUSIVE.**

### 2.5 What was deliberately NOT built, and why

No body-fitted or wall-aligned variant of the interpolator. Such an arm needs a blend length
scale and an arc-length correspondence with no published provenance — a straw steelman a
reviewer reopens by attacking the tuning. §3's oracle does that job with no free parameters
at all. The honest scope of this result is therefore: **parameter interpolation of AirfRANS
fields in physical coordinates**, which is what the paper published and what the RSM
literature does. A body-fitted parameter interpolator is a different and interesting method
and this document does not test it; §5 says so in the manuscript text.

---

## 3. P3 — is the near-wall gap the weights, or the coordinates?

For each test case and band, the **single best of the 800 training fields, chosen with
knowledge of the test answer**, on `u` (case-mean; $Q_b$ = that over Transolver's band MSE):

| band | oracle best single field | Transolver | $Q_b$ | the fitted KRR combination |
|---|---:|---:|---:|---:|
| wall | 23.87 | 0.1654 | **144.3** | 670.1 |
| 0–0.005c | 397.7 | 1.158 | **343.6** | 200.3 |
| 0.005–0.01c | 21.95 | 0.6035 | 36.4 | 91.51 |
| 0.01–0.02c | 8.560 | 0.5069 | 16.9 | 22.19 |
| 0.02–0.05c | 3.518 | 0.3988 | 8.82 | 5.739 |
| 0.05–0.15c | 1.453 | 0.2627 | 5.53 | 0.4795 |
| 0.15–0.5c | 0.4568 | 0.1910 | 2.39 | 0.4373 |
| >0.5c | 0.0736 | 0.1356 | **0.54** | 0.5141 |

Pre-registered rule: $Q_b > 10$ in `0-0.005c` → **`NO-PARAMETER-COMBINATION`**. It returns
**343.6**.

**The honest caveat, stated before a reviewer states it.** A minimum over single fields is
*not* a formal lower bound on linear combinations — and in the innermost band the fitted
combination indeed beats the oracle (200.3 against 397.7), because averaging 800 misaligned
profiles cancels variance. The claim the number supports is therefore the weaker but still
decisive one: **the best training field in the entire training set, selected by an oracle
that is shown the answer, is 344x worse than Transolver inside $0.005c$, and the fitted
combination is 173x worse.** Both are two orders of magnitude off, in the region holding 41%
of the dataset's nodes, and §2.4 says why: the fields being combined are not sampled at
comparable wall distances. This is not a tuning deficit. Re-selecting the kernel, the
bandwidth or the ridge by cross-validation on the node measure moves a factor of a few; it
does not move a factor of 344. (That said — see §5 — the interpolator's hyperparameters here
*are* the grid-CV-selected ones, and that is a real limitation of scope, not of magnitude.)

Note the last row: beyond $0.5c$ the oracle single field is **better than Transolver**
($Q = 0.54$). The representation is not globally deficient. It is deficient exactly where the
boundary layer is, which is the paper's own thesis, now measured on the surrogate, on the
baseline, and on an oracle.

---

## 4. The band decomposition, no raster anywhere

$r_b = \mathrm{MSE}_{\text{interp}}(b) / \mathrm{MSE}_{\text{Transolver}}(b)$; `>1` means
Transolver is better by that factor. Bands are the cloud's own sdf (AirfRANS column 4), edges
identical to `tab:interp_bands` and `measure_asymmetry.md` Block B, plus an explicit `wall`
row for the 201 444 nodes with $sdf = 0$ that the seven-band grid drops.

| band | node frac | `u` | `v` | `p` | `nut` |
|---|---:|---:|---:|---:|---:|
| wall | 0.0056 | **4052** | **2823** | 1.50 | 382 |
| 0–0.005c | **0.4096** | **173.0** | **167.2** | **5.42** | 195 |
| 0.005–0.01c | 0.0581 | 151.6 | 17.1 | 4.11 | 23.2 |
| 0.01–0.02c | 0.0587 | 43.8 | 5.76 | 2.40 | 6.01 |
| 0.02–0.05c | 0.0838 | 14.4 | 1.37 | 1.27 | 4.92 |
| 0.05–0.15c | 0.1059 | 1.83 | **0.226** | **0.321** | 5.40 |
| 0.15–0.5c | 0.1243 | 2.29 | **0.069** | **0.040** | 8.72 |
| >0.5c | 0.1540 | 3.79 | **0.057** | **0.0038** | 39.5 |

Read three things off it.

**(a) The localisation thesis is confirmed in magnitude and is now raster-free.** On `p`, the
ratio runs $5.42 \to 0.0038$ across the seven non-wall bands — a span of **1435x**, monotone
after the wall row, with the crossover between $0.02$–$0.05c$ and $0.05$–$0.15c$, exactly
where the grid put it. On `v` the span is **2958x**. The two methods are accurate in
different places, by three orders of magnitude, and which one an aggregate table declares the
winner is decided by where you look. **That statement is now true with no rasterisation in
it at all.**

**(b) The pre-registered P2 verdict on `u` is `PARTIAL`, and the reason is a correction the
project owes itself.** P2 reads $S = r_{\text{first}}/r_{\text{last}}$ on `u` over BANDS7 with
at most one inversion. It returns $S = 45.6$ (≥ 10) but **two** inversions, because on `u`
the ratio bottoms out at $1.83$ in $0.05$–$0.15c$ and then *rises* again — **the interpolator
never beats Transolver on `u` in any band at native resolution**, including the far field.
`measure_asymmetry.md` §3b reported the opposite from the grid ($r_b = 0.312$ beyond $0.5c$,
"Transolver is a factor of three *less* accurate beyond $0.5c$"). Block D of that same
document already contained the explanation: the r128 raster's own far-field round-trip error
on `u` is $0.156$ against Transolver's $0.110$, so **the grid's far-field `u` comparison was
itself raster-limited**. At the nodes, Transolver wins the far field on `u` by **3.2x**
(in-crop) / **3.8x** (full cloud). §7.5's sentence must be corrected; the `v` and `p`
far-field claims stand and get *larger*.

**(c) The wall row is new and it is the one the force claims sit on.** At $sdf = 0$ — the
surface nodes, where lift and drag are integrated — Transolver is **4052x** better on `u`,
**2823x** on `v` and **382x** on `nut`, and the two arms are within **1.5x** on `p`
(149 200 against 99 227). Both are poor there in absolute terms relative to
$\mathrm{Var}_{\text{train}}(p) = 135\,590$, which is worth saying plainly next to any
force-coefficient claim made by either method.

---

## 5. Verdict against each pre-registered rule

| rule | threshold, committed in `a54cb75` | measured | verdict |
|---|---|---|---|
| **P1** primary | `R_c <= 2` on all of `u,v,p` → COMPETITIVE; `> 2` on all three → NOT-COMPETITIVE | `R_u` **144.2**, `R_v` **133.9**, `R_p` **4.85** | **NOT-COMPETITIVE-AT-NATIVE** |
| **P2** localisation on `u` | `S >= 10` + ≤1 inversion → CONFIRMED; `< 3` → NOT CONFIRMED | `S = 45.6`, **2** inversions | **PARTIAL** (§4b) |
| **P3** oracle floor | `Q > 10` inside 0.005c → NO-PARAMETER-COMBINATION; `<= 1` → WEIGHTS-BOUND | `Q = 343.6` | **NO-PARAMETER-COMBINATION** (with §3's caveat) |
| **P4** inconclusive triggers | outside-hull mass `> 0.20`, or bridge/nearfill spread `>` the method gap | max outside-hull mass **0.094%**; spread 0.2 against a gap of 199 | **no band inconclusive** |

**Limitations, stated because they are real.** (i) The interpolator's hyperparameters are the
**grid**-CV-selected ones (`krr`, $\sigma = 1.0\times$ median, $\lambda = 10^{-3}$,
$w_U = 0.25$); re-selecting them by cross-validation *on the node measure* would require
$800\times800$ cloud-to-cloud transfers and was not run. §3's oracle bounds how much that can
buy at the wall — a factor of a few against a factor of 344 — but the scope of P1 is
correctly "this estimator", not "every possible parameter-space estimator".
(ii) `full` split only; `reynolds`/`aoa` not run at native nodes. (iii) No body-fitted arm
(§2.5). (iv) 2-D, one benchmark, as everywhere else in this paper.

---

## 6. Exact manuscript changes

**I did not touch `body.tex`, `abstract.tex` or `refs.bib`.** Keyed to text, priority order.
Items 1–4 are not optional: the current sentences are contradicted by a committed artifact in
the same repository.

### 6.1 `sec:limitations` — delete the false sentence and CLOSE the open item

The bullet stating that the point-space head-to-head needs a `PointNormalizer` *"the
checkpoints do not store"* is factually wrong (§1.2). `measure_asymmetry.md` §7.6 proposed a
replacement that still described the native-resolution head-to-head as open. **That
replacement is now stale.** Use:

> The deployed checkpoints store both normalisers and we use them. The head-to-head is run
> twice: under the dataset's node measure on the identical rasterised predictions
> (\autoref{tab:measure}), and at the native cloud nodes with no rasterisation anywhere in
> the comparison path (\autoref{tab:native}, `scripts/point_space_headtohead.py`). What
> remains open is a parameter interpolator built in body-fitted rather than physical
> coordinates, and a re-selection of its hyperparameters under the node measure; the oracle
> control in that script bounds what the second can buy near the wall at a factor of a few
> against a measured factor of $344$.

### 6.2 The abstract's lead number — the third and final measure

`measure_asymmetry.md` §7.1 proposed replacing the `8.4\times` with the re-weighting result.
That proposal can now be completed rather than hedged:

> Kernel interpolation over seven scalars parsed from the AirfRANS case name---no network, no
> flow-field learning---gives $8.4\times$ lower volume-pressure error than a Transolver
> trained on the same data when both are scored on an area-uniform $128^2$ raster. Scored
> where the data actually lives---per node on AirfRANS's own cloud, with no rasterisation
> anywhere in the comparison path---the same two predictions reverse on every channel: the
> interpolator is $4.9\times$ worse on pressure, $144\times$ worse on streamwise velocity and
> $134\times$ worse on cross-stream velocity, and $24.7\times$ worse in standardised units.
> The benchmark's field-MSE ranking is not a property of the methods; it is a property of the
> measure, and no published AirfRANS comparison states which one it uses.

### 6.3 New table `tab:native` — the exhibit that closes the objection

| per-node MSE, 200 `full` test cases, 35.8M native nodes | `mse_u` | `mse_v` | `mse_p` | `mse_nut` |
|---|---:|---:|---:|---:|
| parameter interpolation @ native nodes | 93.11 | 66.52 | 41 740 | 6.92e-7 |
| Transolver @ native nodes, 3 seeds | **0.6458** | **0.4968** | **8 605** | **3.79e-8** |
| ratio (interp / Transolver) | 144.2 | 133.9 | 4.85 | 18.3 |
| *memo:* published r128 prediction resampled at the same nodes | 400.2 | 189.7 | 3.06e6 | 2.08e-6 |
| *memo:* the r128 raster's own round-trip error, no model | 281.5 | 272.6 | 3.01e6 | 1.54e-6 |

Caption must carry: both arms are the published rows (G1 rel $\le1.6\mathrm{e}{-}8$, G2 rel
$0.00\mathrm{e}{+}00$); the interpolator is given the better of two in-body conventions per
channel; per-seed ratios are 141–147 / 129–142 / 4.2–5.3; Transolver wins 200/200 cases on
$u$, $v$, $\nu_t$ and 142/200 on $p$; the last two rows show that the grid protocol was
costing the *interpolator* $4.1\times$ on $u$ and $69.9\times$ on $p$, so the native
construction favours it.

### 6.4 Everywhere the interpolator is said to "beat" the Transolver

Every such sentence (abstract, intro, contributions, `tab:interp` discussion, conclusion)
must carry the measure. The defensible form is:

> On the area-uniform $128^2$ raster the interpolator attains $\texttt{mse\_p} = 75.0$
> against $629.6$; re-weighted by node count, $6012$ against $2646$; per node on the native
> cloud, $41\,740$ against $8605$. The ranking reverses between the first and the third.

### 6.5 `body.tex:546-550` and `measure_asymmetry.md` §7.5 — correct the far-field `u` claim

§7.5 of `measure_asymmetry.md` proposes the sentence *"Transolver is nearly three orders of
magnitude more accurate inside the first half-percent of a chord and a factor of three
\emph{less} accurate beyond $0.5c$."* **The second half is a grid artifact** (§4b). Replace
with:

> Band by band at the native nodes, the ratio of interpolation error to Transolver error runs
> $173\times \to 3.8\times$ on $u$, $167\times \to 0.057\times$ on $v$ and $5.4\times \to
> 0.0038\times$ on $p$, from $0$–$0.005c$ outward. On the two channels where the interpolator
> owns the far field it owns it by $18\times$ and $265\times$; on $u$ it is behind in every
> band, because the $128^2$ raster's own far-field round-trip error ($0.156$) exceeds
> Transolver's ($0.110$) and the grid comparison there was raster-limited. The two methods
> are accurate in different places and the metric decides which one wins.

### 6.6 `interpolation_baseline.md` §6 and `interpolation_resolution_ladder.md:13`

Both carry the false normaliser claim and both list this experiment as not done. Both should
point at this document.

### 6.7 What does **not** change

The train-size ladder, the `reynolds`/`aoa` split findings, the covariate null, the
force-coefficient block, the resolution ladder, the five adversarial controls, and the
corrector deltas ($-9/-21/-25\%$, within-backbone) are untouched: none is a cross-method
field-MSE comparison. The *measure-dependence* finding is now much stronger than it was —
the same two predictions on the same 200 cases rank $8.39\times$ one way, $0.44\times$ the
other, and $0.21\times$ the other still, depending only on where you look.

---

## 7. Rebuttal lines

> **Reviewer:** the head-to-head under the point measure is never run, so the abstract's lead
> number is unfalsifiable.
>
> **Response:** it is run, on all 200 test cases and all 35.8M native nodes, with the decision
> rule committed in `a54cb75` before the first number existed. The interpolator loses on all
> four channels: $144\times$ on $u$, $134\times$ on $v$, $4.85\times$ on $p$, $18\times$ on
> $\nu_t$; $24.7\times$ in standardised units; 200/200 cases on three channels. We withdrew
> the number from the abstract and replaced it with the measurement.

> **Reviewer:** you crippled the baseline by evaluating a grid method at points.
>
> **Response:** the opposite. We evaluated each training field on its own native cloud, so the
> comparison path has no representation ceiling at all. The construction rasterises back to
> the published grid row at $1.07\times$/$0.77\times$/$1.00\times$ (G3), and the alternative
> a critic would propose — resampling the published r128 output at the nodes — is $4.1\times$
> worse on $u$ and $69.9\times$ worse on $p$ than the construction we used. We handed the
> interpolator the better representation and it still lost by 4.9x–146x.

> **Reviewer:** then tune it. The hyperparameters were selected on the grid measure.
>
> **Response:** an oracle that is shown the test answer and allowed to pick the single best of
> all 800 training fields is still $344\times$ worse than Transolver inside $0.005c$. The
> mechanism is measured, not argued: inside $0.005c$, where $41\%$ of AirfRANS's nodes live, a
> query sits on average $7.8$ times further from the training case's wall than from its own,
> and $35\%$ of the weight mass lands inside the training airfoil. Re-tuning moves a factor of
> a few. It does not move $344$.

> **Reviewer:** so is any of the interpolation baseline left?
>
> **Response:** everything except the cross-method aggregate ranking, and the far-field claim
> is now larger, not smaller: beyond $0.5c$ the interpolator is $18\times$ better on $v$ and
> $265\times$ better on $p$. Parameter interpolation reproduces the outer field from seven
> scalars, from $100$ training cases, without degrading on the `reynolds` split, and is
> indistinguishable from the exact ground-truth field on official-label force ranking. What
> moved is our willingness to call $8.4\times$ a win — and, with this run, our ability to say
> exactly what a learned surrogate buys: the boundary layer, measured on the surrogate, the
> baseline and an oracle, with no raster in the comparison.

---

## 8. Cost and reproduction

| step | wall clock | resource |
|---|---|---|
| `--stage cache` (200 test clouds + 800 train `.vtu` → compact `.npz`, 4.8 GB scratch) | 525 s | 1 CPU |
| `--stage pilot` (G1 + G3, 3 cases × 800 train) | 217 s | 8 CPU |
| `--stage transolver` (200 cases × 3 seeds, per-node) | **263 s** | 1 GPU |
| `--stage interp` (800 train × 200 test = 160 000 cloud-to-cloud transfers) | **7350 s** | 14 CPU |
| `--stage r128resample` (the representation control) | 40 s | 1 CPU |
| `--stage reduce` | < 1 s | 1 CPU |
| **total** | **~2.3 h wall, ≈ 29 CPU-core-hours + 4.4 GPU-minutes** | |

No training, no dataset download, no write to `results/mgn`, `checkpoints/mgn` or
`mgn_run.log`. The interpolator sweep checkpoints every 25 train cases per worker and resumes
(`c0ee75a`, added after a first attempt was stopped at train 325/800).

```
.venv/Scripts/python.exe scripts/point_space_headtohead.py --stage cache
.venv/Scripts/python.exe scripts/point_space_headtohead.py --stage pilot
.venv/Scripts/python.exe scripts/point_space_headtohead.py --stage transolver --device auto
.venv/Scripts/python.exe scripts/point_space_headtohead.py --stage interp --n-proc 14 --kd-workers 1
.venv/Scripts/python.exe scripts/point_space_headtohead.py --stage r128resample
.venv/Scripts/python.exe scripts/point_space_headtohead.py --stage reduce
```

Per-case rows for both arms, the per-case-per-band oracle minima and argmins, and the
geometry-mismatch diagnostics are all in the artifacts, so every aggregate here can be
recomputed without re-running the sweep.
