# Withdrawing the 8.4x headline, and what replaces it

Companion to `docs/paper/review/measure_asymmetry.md`, which is the measurement.
This file is the **manuscript-side record**: every site changed, every auditor row
added or removed, the new headline, and the build/script status.

> **Status.** All manuscript, highlights and auditor edits are on disk. **`git`,
> `pdflatex`, `audit_paper_numbers.py` and `check_submission.py` could not be run
> in this session — the Bash tool was disabled.** §6 lists exactly what must be
> run before this is considered done, and §5 lists the two places most likely to
> break.

---

## 0. The new headline

> On this benchmark the field-MSE ranking of a 7.35M-parameter Transolver against a
> seven-scalar parameter interpolator is decided by a weighting choice that no published
> AirfRANS comparison states — and the r128 protocol both arms are scored on cannot
> adjudicate near-wall skill for **either** of them.

Three supporting facts, all from
`results/interpolation/measure_asymmetry{,_nodes,_sensitivity}.json`:

1. **The measure gap.** 64.4% of AirfRANS's native nodes lie inside 0.05c of the wall,
   which is 1.25% of the r128 crop's area — a factor of 51.5 (109.5 at 0.02c, 312 at
   0.005c, inverting to 0.14 beyond 0.5c). Re-weighting the *identical* grid errors of the
   *identical* published rows by node count takes `mse_p` from 8.39x in the interpolator's
   favour to 0.44x, and every channel it was winning flips. Pre-registered rule D3
   (committed in `cdf08f5` before the number existed) returns **ARTIFACT**.
2. **The adjudication limit.** The r128 raster's own round-trip error at the native nodes
   exceeds Transolver's per-node error by **413x** pooled and **495x** inside 0.005c. The
   grid protocol measures far-field skill, for either method. (**413, not the report's
   418** — see §5.1.)
3. **The localisation thesis, now on both arms.** Band by band outward from the wall the
   ratio of interpolation error to Transolver error on `u` runs
   906x / 74x / 30x / 5.3x / 0.57x / 0.56x / 0.31x, monotone over seven bands, span 2905x,
   zero inversions (D2: `CONFIRMED ON BOTH ARMS`). 71.9% of Transolver's own `u` squared
   error lies beyond 0.5c, where the interpolator carries 3.6% of its own.

---

## 1. Every manuscript site changed

### `docs/paper/abstract.tex` — rewritten lead

- **Out:** "gives 88% lower volume-pressure error … (8.4x), 62% lower cross-flow error, and
  4.9x/3.8x closer agreement"; "loses on streamwise velocity (6.5x) and eddy viscosity
  (15x), hence 6.4x on the four-channel training objective"; the 92%/98.7%/512^2 sentence;
  the closing residual sentence.
- **In:** the measure pair (8.4x area-uniform / 0.44x node-weighted, 64% vs 1.2%), the
  413x raster ceiling, the 906x→0.31x band ratio with Transolver's 71.9% far-field share,
  "A surrogate buys the first cell and the closure", and the unchanged covariate-null and
  `reynolds` sentences.
- **Word count.** Hand-counted under `check_submission.py`'s `strip_tex` semantics:
  **246 of 250**, four under the cap (was 244). The header comment records the new count
  and the counting rules. *This is the single most likely thing to be off by a word or
  two — run `check_submission.py`.*

### `docs/paper/body.tex`

| site | what changed |
|---|---|
| Intro, "Learning earns its keep…" paragraph | **Split into three paragraphs.** New lead paragraph *"The field-MSE ranking is decided by a weighting nobody states"* carries the 8.4x/0.44x pair, the u and v reversals, the 1.99x/8.30x standardised pair, the 64.4%/1.25%/51.5 asymmetry, and the two readings that do **not** move (the 4-channel objective, the force columns). New paragraph *"The deployed protocol cannot adjudicate the boundary layer for either method"* carries 413x/495x. The retitled *"Learning earns its keep in the first cell off the wall"* paragraph now leads with the **band ratio measured on both arms** and keeps the 92%/90%/98.7% and the `PARTIAL` ladder verdict. |
| Intro, "The four findings above are the contributions" | → "The findings above…" (there are now more paragraphs than four) |
| Contributions, item 1 | Retitled *"The missing baseline, the measure that decides it, and the boundary it draws"*. States that the aggregate head-to-head is not a stable quantity ([0.43, 1.61] over five node constructions against 8.39 area-uniform, no construction an interpolator win), that the decomposition is what is stable, and restates the 100-case ladder claim as reaching Transolver's `mse_p` **on the area-uniform measure**. |
| Positioning | Adds the measure-dependence result as the thing the paper now leads with. |
| `sec:experiments` metric definition | Volume MSE is now explicitly declared **area-uniform over the fluid cells of the r128 crop** in every table except `tab:measure`. |
| `sec:interp` title | "Where learning earns its keep: the parameter-interpolation baseline" → "**Where learning earns its keep, and the measure that decides the ranking**" |
| `sec:interp`, "The baseline" paragraph | Adds the **objective asymmetry** (55.5% of nodes inside 0.02c against 0.51% of cells, 109.5x; 312x inside 0.005c; inverting to 0.14x beyond 0.5c) alongside the existing input asymmetry, and points at `tab:measure`. |
| `tab:interp` caption + interpolation row | Bold on `mse_v` (`0.0336`) and `mse_p` (`75.03`) **removed** — the bolding asserted "best", and that assertion is what the node measure reverses. Caption now states bolding is read under the area-uniform weighting and names the two columns that reverse. `mse_u`, surface and force bolds kept (they do not reverse). |
| `sec:interp`, "Result, with its counterweight" | Retitled *"Result, and the weighting it is contingent on"*. Every volume ratio is now stated as an explicit pair; the force paragraph is split out and marked as not reachable by a cell weighting; the `mse_u` loss is given under both weightings (6.5x / 86x); `nut` gets its area/node pair (14.4x / 9.2x); the standardised means are given under both weightings with the 1.9x-vs-1.99x provenance difference named rather than hidden. |
| `sec:interp`, new paragraph | *"'Weight by node count' is not a single number, and none of its readings rescues the aggregate"* — the five-construction sensitivity, the [0.43, 1.61] range, why the endpoints are not equally credible, and the two ways the re-weighting is conservative (0.098 vs 0.432 grid-binned innermost mass; 13.4% solid-binned nodes dropped). |
| `tab:interp_std` caption + interpolation row | Bold on the 3-channel mean `1.16e-3` removed (it reverses). Caption declares every cell area-uniform and gives the node-weighted pair. |
| **New `tab:measure`** | The defensive exhibit: the same four channels under both weightings, both arms, plus the two ratio rows, with the full provenance caption (G1/G2, per-seed `R_p`, the 629.6-vs-628.5 provenance note, and the conservatism note). |
| `sec:interp`, new paragraph after `tab:interp_bands` | *"The same decomposition on the surrogate, and the boundary becomes a crossover"* — the band ratio on both arms, the mirror-image SE shares (92.4% inside 0.02c vs 71.9% beyond 0.5c), the reading that the 8.4x on `mse_p` is a far-field statement (32x and 237x in the two outer bands), and the sample-size caveat on the 906x (4489 cells, on a monotone trend, with 74x/30x independently). |
| **New `tab:bandratio`** | Per-band `r_b` on all four channels with cell fraction, node fraction, and Transolver's own `u` SE share. |
| `sec:interp`, new paragraph | *"Neither arm's near-wall skill is observable at r128"* — Block A (35.9M nodes, the per-band and cumulative asymmetry, the across-case spread, G6) and Block D (413x/495x, and the in-crop-only reading rule). |
| `sec:interp`, "It needs the shape…" | The train-size ladder's cross-method sentence now says the 100-case result is past Transolver **on the area-uniform measure**, and separates the ladder's measure-free internal result from the measure-contingent comparison. |
| `sec:v2`, "Learned correction works on a strong backbone" | The corrector deltas are explicitly flagged as **within-backbone and untouched**; the `485` vs `75.0` cross-method reading is now given as area-uniform with the node-weighted ordering named. |
| `sec:limitations`, first bullet | **The false sentence is gone.** Rewritten around what is true: the checkpoints *do* store `point_norm` and `grid_norm`, the obstacle is that native-node scoring changes measure and representation together, and the r128 round-trip ceiling bounds both arms. See §3. |
| Conclusion, first paragraph | Rewritten to lead with the measure-dependence finding and the adjudication limit, then the band ratio, then the surviving localisation and covariate-null claims. |
| Conclusion, last paragraph | Adds: report which cell weighting a field metric is computed under, "because on this benchmark that choice moves a pressure ratio by a factor of 19 and reverses its sign." |

### `docs/paper/submission/highlights.txt`

All five bullets replaced (character counts hand-verified ≤ 85):

```
- The AirfRANS field ranking flips with the cell weighting: 8.4x to 0.44x on pressure   (83)
- 64% of AirfRANS nodes sit inside 0.05 chord, which is 1.2% of the raster's area       (79)
- The raster's own round-trip error exceeds the surrogate's per-node error 413x         (77)
- Interpolation-to-Transolver error on u falls 906x to 0.31x from wall to far field     (80)
- Every published AirfRANS drag rank falls below a regression on the case name          (76)
```

Lengths were counted against `check_submission.py`'s own parser, which takes lines starting
with `- `, strips that prefix, and measures the remainder: 83 / 79 / 77 / 81 / 76. The
header block contains no `- ` line, so it is not counted; the file still has exactly five
bullets.

Dropped: the standalone "beats a matched Transolver by 8.4x" claim, the eddy-viscosity
counterweight bullet, the 92%/0.5% bullet, and the `reynolds` bullet. The counterweight is
now carried by bullets 1 and 4, which show the ranking flipping and Transolver winning at
the wall.

### `docs/paper/submission/SUBMISSION_CHECKLIST.md`

Two recorded counts refreshed: highlights character range 76–79 → 76–83, and the abstract
word count 241 → 246 (hand count, to be confirmed by `check_submission.py`).

### `docs/paper/review/interpolation_resolution_ladder.md`

§1 gets a dated **CORRECTION** block retracting the "the checkpoints do not store the
normaliser" claim, naming `point_norm`/`grid_norm` in
`checkpoints/v2_transolver/seed{m}.pt` and
`scripts/recompute_force_vs_official.py:134-158`, and marking §8.1's proposed `body.tex`
text as superseded. The same false clause is deleted from §8.1's proposed wording.

---

## 2. Auditor rows: `scripts/audit_paper_numbers.py`

### Removed (2 rows)

| row | value | why |
|---|---|---|
| `interp mse_p lower than Transolver by (%)` | 88.0 | The "88% lower" phrasing is withdrawn from the abstract, intro, `sec:interp` and the conclusion. It is an area-uniform statement whose **sign** reverses under the node measure; the manuscript now quotes a pair of ratios instead. |
| `interp mse_v lower than Transolver by (%)` | 62.0 | Same. |

`interp_pct_lower()` is retained (with a comment) because nothing else computes a
percentage form; only its two rows are gone. **No row was left dangling.**

### Kept deliberately (and why)

`interp beats Transolver on mse_p by` (8.4), `… on mse_v by` (2.6),
`Transolver beats interp on mse_u by` (6.5), `Transolver beats interp on nu_t by` (15.4),
`interp better on the 3-channel std mean by` (1.9),
`TRANSOLVER better on the 4-channel std mean by` (6.4), and both force ratios (4.9, 3.8).
Every one of these is still quoted verbatim in the manuscript — the first five now
explicitly as the *area-uniform* half of a stated pair, the force ratios because no cell
weighting reaches a surface integral.

### Added (54 rows) and the readers behind them

New readers: `ma`, `ma_nodes`, `ma_sens`, `ma_ratio`, `ma_ratio_inv`, `ma_cell`,
`ma_std_mean`, `ma_band_ratio`, `ma_band_span`, `ma_band_inversions`,
`ma_transolver_se_share`, `ma_gap`, `ma_band_mass`, `ma_band_mass_gap`, `ma_ceiling`,
`ma_ceiling_abs`, `ma_node_mse_u_inner`, `ma_sens_p`, `ma_sens_worst`, plus the
`MA_BANDS` constant.

| group | rows |
|---|---|
| `tab:measure` ratios | `R_p` area (8.39) / node (0.440); `R_v` area (2.97) / node (0.0663); `mse_u` 6.2x / 86x against; `nut` 14.4x / 9.2x against |
| `tab:measure` cells | Transolver area `mse_p` 629.6, `mse_u` 0.127, `mse_v` 0.0999; interp node 30.7 / 4.26 / 6012; Transolver node 0.358 / 0.282 / 2646 |
| standardised means | area 1.99x and 6.1x, node 8.30x and 8.85x — computed in `ma_std_mean` from `tab:measure`'s rows against `interp_full.json`'s **own** `var_train` block, so no divisor is hard-coded |
| D1 asymmetry | node/area fractions and gap inside 0.05c (0.6437 / 0.01249 / 51.5) and 0.02c (0.5553 / 0.00507 / 109.5); inside 0.005c (0.4322 / 0.00138 / 312); the inversion beyond 0.5c (0.14); the grid-binned innermost mass (0.098) |
| §4b sensitivity | lowest and highest `R_p` over the five node constructions (0.431, 1.61); least/most adverse `mse_u` (86x, 551x) and `mse_v` (6.3x, 24x) |
| D2 band ratio | `u` at 0-0.005c (906.2), 0.005-0.01c (74.1), 0.01-0.02c (30.4), 0.02-0.05c (5.33), >0.5c (0.312); span (2905); inversions (0); `p` innermost (0.589) and the two outer inverses (32x, 237x); Transolver's own SE share beyond 0.5c on `u` (71.9%) and `p` (62.8%) |
| Block D ceiling | pooled ratio (413.4), innermost ratio (495), Transolver's per-node `u` inside 0.005c (1.16), the raster's own (572.8) |

Every added row was hand-checked against the artifact before being written; the arithmetic
is in this session's transcript. **They have not been executed** — see §6.

---

## 3. The false limitations sentence

The old bullet said the point-space head-to-head could not be run because *"the checkpoints
do not store the normaliser."* That is false: `checkpoints/v2_transolver/seed{m}.pt`
carries `point_norm` (`mean_in`, `std_in`, `mean_out`, `std_out`, `eps`) **and**
`grid_norm`, and `scripts/recompute_force_vs_official.py:134-158` has loaded both since
June.

The replacement bullet states, in order: (a) every field number in `sec:interp` is scored
on r128 and the representation cannot adjudicate the boundary layer for either arm
(413x/495x); (b) therefore the weighting is reported both ways (`tab:measure`); (c) no
re-weighting reaches the representation; (d) the resolution ladder settles localisation but
not advantage, verdict `PARTIAL`; (e) what is open is a head-to-head at native
**resolution**, and the obstacle is *not* a missing normaliser — it is that native-node
scoring changes measure and representation together.

**This bullet will need another pass when the point-space head-to-head lands.** That
experiment is owned by another agent (`docs/paper/review/point_space_headtohead.md`,
`scripts/point_space_headtohead.py`), which this session did not touch. When it reports:
the clause "That experiment is the single most valuable follow-up in the paper and is not
reported here" must be replaced by the result, and `sec:interp` will need a paragraph for
it. Nothing else in the bullet depends on that outcome.

---

## 4. What was deliberately NOT weakened

The instinct after a withdrawal is to soften everything adjacent. These were left alone on
purpose, and each has a reason:

- **The covariate null (`sec:covariate`) and `tab:covariate`.** Spearman rank correlations
  of predicted against *official* force labels. No field-error measure is involved, so the
  node/area choice cannot reach them. Unchanged, including the four qualifications.
- **The `reynolds`/`aoa` split findings (`sec:splits`).** The interpolator's own
  in-distribution-to-OOD degradation, one arm, one measure held fixed throughout.
  Unchanged.
- **The force-coefficient block (`sec:v2`), the drag-observability analysis, the
  control-volume recovery.** Surface and control-volume integrals against a common
  rasterised reference. A cell weighting has nothing to act on. Unchanged.
- **The corrector deltas (-8/-21/-25%) and the W1 residual-input null.** Within-backbone,
  same architecture against itself, same measure on both sides. Unchanged; `sec:v2` now
  says so explicitly rather than leaving it implied.
- **The residual audit, the conformal/UQ block, the selective-prediction results.**
  Untouched by anything here.
- **The resolution ladder and the five adversarial controls.** The ladder is the
  interpolator against itself across three grids; the controls are the interpolator against
  its own floors. Both unchanged, and the ladder gains a role: it is now the reason the
  906x would *narrow* rather than inflate at finer resolution.
- **The far-field claim.** Strengthened, not weakened — it now has a second witness, since
  beyond 0.5c the interpolator is 3.2x better than Transolver on `u` and 237x on `p`.
- **The title.** See §5.2.

---

## 5. Two things a reader of `measure_asymmetry.md` will notice

### 5.1 The manuscript says **413x**, not the report's **418x**

`measure_asymmetry.md` §5 prints a pooled row of `0.664 | 277.3 | 0.0024`, hence 418x.
The ceiling side (277.3) reproduces exactly from the artifact. The model side does not:
node-weighting the seed-mean in-crop band MSEs of
`D_node_space.mse_in_crop.transolver_seed{0,1,2}.u` by `D_node_space.n_nodes_in_crop`
gives **0.6708**, not 0.664 — and 277.335 / 0.6708 = **413.4**.

The bands are the full partition of the in-crop nodes and G4 asserts the band
reconstruction, so this is not a partitioning issue; 0.664 appears to be a slip in the
report's prose table. The manuscript and the auditor both use **413x**, recomputed from the
artifact by `ma_ceiling()`, so the paper tracks the artifact rather than the summary.
**`measure_asymmetry.md` §5 should be corrected** — it is not this file's to edit without
the owner's say-so, but it is wrong as printed. The 495x inside 0.005c is unaffected
(572.789 / 1.1576 = 494.8).

A second, smaller slip: §0 of the report says "even at 1.61 on `p` it is 85x worse on `u`
and 6x worse on `v`", which mixes constructions. The construction that reaches 1.61 on `p`
(`band_true_node_crop`) is **551x** on `u` and **24x** on `v`; 86x and 6.3x are the
*least* adverse values, from two different constructions. The manuscript states the range
(86–551x on `u`, 6.3–24x on `v`) and names the 1.61 construction's own pair.

### 5.2 The title is unchanged, and that is a decision

> *Neural CFD surrogates earn their keep in the first cell off the wall: a
> parameter-interpolation baseline and a covariate null for AirfRANS*

The main clause asserts **localisation**, which is the one claim this measurement made
*stronger*: it went from inferred on one arm to measured on both, 906x inside 0.005c,
monotone over seven bands, zero inversions. The colon clause promises two artifacts, both
of which are still delivered. Nothing in the title leans on the withdrawn ratio.

The open question is whether the colon clause should now name the measure-dependence
finding, since that is what the abstract leads with. A candidate, if the venue wants the
lead in the title:

> *Neural CFD surrogates earn their keep in the first cell off the wall: the AirfRANS field
> ranking is a weighting choice*

Recommendation: **keep the current title**. It is already committed to, the abstract's
first two sentences carry the new lead, and the second version trades away the "here are
two reusable reference points" promise that is the paper's durable contribution.

---

## 6. Build and script status — WHAT STILL MUST BE RUN

Nothing below was executed; the Bash tool was unavailable for the whole session.

```
git branch --show-current          # must be paper1/reframe-after-jcp; nothing was committed
cd docs/paper
pdflatex neuroforge_cfd.tex          # twice
pdflatex neuroforge_cfd_elsevier.tex # twice
# bibtex not needed: no citation was added or removed
cd ../..
./.venv/Scripts/python.exe scripts/audit_paper_numbers.py --verbose
./.venv/Scripts/python.exe scripts/check_submission.py
```

Run the auditor with `--verbose` the first time. Bare, a reader that raises prints
`SKIP (no file)`, which is misleading when the file exists and a key path is wrong;
`--verbose` prints the exception type and message. **A SKIP suppresses the success line
just as a MISMATCH does** (`if not fails and not skips`), so every one of the 54 new rows
must resolve.

Files changed, for staging individually (never `git add -A`):

```
docs/paper/abstract.tex
docs/paper/body.tex
docs/paper/submission/highlights.txt
docs/paper/submission/SUBMISSION_CHECKLIST.md
docs/paper/review/interpolation_resolution_ladder.md
docs/paper/review/headline_withdrawal.md      (new)
scripts/audit_paper_numbers.py
```

Required outcomes: 0 overfull, 0 underfull, 0 undefined, 0 errors, no font-shape warnings
in both builds; `audit_paper_numbers.py` ending "every checked number matches its source
file"; `check_submission.py` ending "all mechanical requirements satisfied".

**Where this is most likely to fail, in order:**

1. **The abstract word count.** Hand-counted **twice**, token by token, against
   `strip_tex`'s actual semantics (drop `%`-leading lines; `re.sub(r"\\[a-zA-Z]+\*?")`;
   delete `{}$~\`; keep tokens containing an alphanumeric): **246 of 250**. Sentence
   totals 18 / 57 / 35 / 21 / 36 / 9 / 42 / 28. Note `\%` survives the control-sequence
   regex (`%` is not `[a-zA-Z]`) and its backslash is then deleted, so `$64\%$` is the
   one token `64%` and does count. If `check_submission.py` still reports over, the
   cheapest cuts are "against $1.2\%$ of the raster's area" (6 words, restated in the
   body) or "A surrogate buys the first cell and the closure." (9 words, restated in the
   title).
2. **An auditor reader path.** Every accessor was written against key names read directly
   out of the three JSON files in this session, but none was executed. A wrong path returns
   `None` and the success line will not print. The two least-standard readers are
   `ma_std_mean` (which joins `measure_asymmetry.json`'s cells to `interp_full.json`'s
   `var_train` block) and `ma_ceiling` (which pools with `n_nodes_in_crop` weights).
3. **`tab:measure` and `tab:bandratio` width.** Both use the same
   `adjustbox{max width=\textwidth}` skeleton as `tab:interp_std` and only booktabs
   primitives (`\cmidrule(lr)` is booktabs, already loaded), so they should shrink to the
   narrower Elsevier measure rather than overflow. `tab:measure` is nine columns and is the
   wider of the two.
4. **Two new float insertions** in an already float-dense section could shift page breaks
   and surface a previously latent underfull box. `hbadness`/`vbadness` are already 10000
   in `preamble.tex`, which suppresses underfull reporting, so this is unlikely to bite.

### Not done, and flagged rather than silently skipped

- `docs/REPRODUCE.md`'s per-claim map was not updated for the new `tab:measure` and
  `tab:bandratio` claims. `results/MANIFEST.json` **already** carries all three
  `measure_asymmetry*` artifacts with their producing commands, so no manifest change is
  needed.
- `docs/paper/review/interpolation_baseline.md:389-390` and
  `docs/paper/review/restructure.md:266` still repeat the false normaliser claim. They are
  superseded planning documents, not inputs to the build; corrected in
  `interpolation_resolution_ladder.md` (which the manuscript's own limitations bullet used
  to mirror) and recorded here.
- `measure_asymmetry.md` §5's pooled row and §0's mixed-construction sentence — see §5.1.
