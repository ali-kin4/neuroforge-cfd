# NullBench: turning `null_travels.md` into something the field can run

`null_travels.md` measured that the predictive power of published metadata
alone spans an order of magnitude across five CFD-ML benchmarks (R² 0.10 to
0.97), and that not one of the five reports this. That finding stays inert
if it lives only in this repository's review notes. This report is the
artifact that turns it into something a benchmark maintainer or a referee
can run in one command: a named statistic, a tested package
(`src/neuroforge/nullbench/`), a CLI (`neuroforge nullbench`), and five
corrected leaderboards.

Everything below was produced on branch `paper1/reframe-after-jcp`. **This
report does not touch the manuscript** (`docs/paper/body.tex`,
`abstract.tex`, `sections/`) — those files are owned by a concurrent thread
running the interpolation ladder.

## Reconciliation note (post-review revision)

The first version of this artifact named its headline statistic the
"covariate-null fraction" and reported cross-benchmark numbers computed
under each benchmark's own protocol side by side. A naming/positioning
review (`docs/paper/review/naming_and_positioning.md`) and a mechanism study
(`docs/paper/review/null_mechanism.md`), both completed on this branch after
this artifact's first draft, forced two corrections, made below and
recorded here rather than silently:

1. **Renamed.** The protocol is now **the metadata null**; its headline
   statistic is **metadata-only R²** (`R²_meta`, emitted as
   `metadata_null_r2`, alongside `metadata_null_mse`). The demoted ratio
   (`null / published`) is now `published_relative_ratio` — see §1.
2. **Harmonised.** A new module, `neuroforge.nullbench.harmonised`, adds the
   cross-benchmark-comparable view — one protocol (10-fold OOS OLS, R²,
   constant reference area) applied identically to all five benchmarks — as
   a *separate* artifact from the five per-benchmark worked examples, which
   keep their own protocols precisely because that is what makes each
   verdict against its own leaderboard admissible. See §2.4 and §4.

Both the renamed field names and the harmonised numbers are pinned by tests
(`tests/test_nullbench.py`); nothing here was hand-edited without a
reproducing test backing it.

---

## 1. The statistic

### 1.1 Name

**The metadata null** (protocol) / **metadata-only R²**, `R²_meta`, emitted
as `metadata_null_r2` (statistic). Settled by
`docs/paper/review/naming_and_positioning.md` §4.2, replacing an earlier
"covariate-null fraction" name this artifact used in its first draft, for
three reasons that review gives and this module accepts:

1. **"Fraction" is the wrong word for a quantity that can be negative.**
   R² is not bounded below; this project's own WindsorML interval reaches
   −0.143. A "fraction" that goes negative invites a reviewer's objection
   for a word choice, not a finding.
2. **"Null" was already doing two jobs.** This package also runs a
   label-permutation null (`harness.permutation_check`) to check the fitting
   machinery isn't manufacturing signal. Two different "nulls" answering two
   different questions, both called just "null", is a real readability
   problem once both appear in the same report.
3. **"Floor" is already load-bearing elsewhere in this repository** (the
   residual floor, `residual_floor_theorem.tex`, `floor_resolution_study.md`).
   A name built from "covariate" + "floor" would collide with that
   vocabulary even though it is unclaimed in the wider literature.

**Uniqueness check — with an explicit limitation, carried over from the
positioning review and not independently re-verified here.** Neither this
task nor the positioning review had a live literature-search tool available
in this environment; the review's evidence is an arXiv API metadata screen
(title/abstract/comments only, not full text) plus full-text reads of a
small set of directly competing benchmarking papers (CarBench, PhysicsNeMo-CFD,
ShapeBench). Within that evidence: `"metadata null"` returns zero arXiv
hits; `"covariate-null fraction"` also returns zero, but the review argues
against reusing it regardless, for the three reasons above. Full detail,
including what was and was not read at full text, is in
`naming_and_positioning.md` §1.4 and §4.1 — not repeated here to avoid a
second, potentially drifting copy of that evidence.

### 1.2 Definition — the primary statistic

`R²_meta` **is** the out-of-sample R² (or, on a benchmark that reports
Spearman, the out-of-sample rank correlation) of an OLS regression on a
benchmark's own published per-case parameters. It is reported directly, not
as a ratio of anything:

* **R²-style** (`metric="r2"`): floor 0 (an intercept-only / constant
  predictor scores exactly 0 by the definition of R², `1 − SSE/SST`),
  ceiling 1.
* **Rank-style** (`metric="spearman"`): floor 0 (a random ranking has
  expected Spearman correlation 0), ceiling 1.

Every `R²_meta` ships with a 95% case-level percentile bootstrap CI and,
alongside it, `metadata_null_mse` (scale-absolute, its own bootstrap CI) —
never a bare point estimate. Implementation:
`src/neuroforge/nullbench/stats.py::r2_score` / `mse_score`,
`harness.py::run_null`.

### 1.3 The demoted comparison aid: `published_relative_ratio`

A benchmark maintainer will still want to know "how much of my model's score
does the metadata null already reach?" — that is
`stats.published_relative_ratio`: `null / published`, on the same
floor-0/ceiling-1 scale, undefined (not clipped) when the published value
does not exceed its own floor (flag `published_at_or_below_floor`), never
clipped above 1 (`null_exceeds_published`) or below 0
(`null_at_or_below_floor`) when it occurs.

**This is explicitly not the headline**, for two reasons the positioning
review gives (§4.2) and this module now enforces structurally: it needs a
competitor's published number, so it cannot be computed from a benchmark's
own files standalone the way `R²_meta` can; and it inherits that number's
reporting precision. The review demonstrates this concretely for a
*different* ratio it also considered and rejected, null-normalised gain
`G = (R²_model − R²_meta) / (1 − R²_meta)`: on DrivAerML the denominator
`1 − R²_meta = 0.0269` is close to zero, so propagating DoMINO's published
`0.98` through its own 2-decimal rounding interval `[0.975, 0.985]` swings
`G` six-fold, `[0.07, 0.44]`.

**Checked, not assumed, for `published_relative_ratio` specifically** (this
task's own obligation, not carried over from the review): this ratio's
denominator is the published value itself, not `1 − R²_meta`, so the
dangerous regime is the *opposite* end — a published value reported to few
decimals **and** close to the metric's own floor, not close to 1. Every
precision-annotated published entry in the five worked examples was checked
(`tests/test_nullbench.py::test_rounding_sensitivity_*`,
`scripts covariate_null_crossbench.py`-equivalent hand check reproduced
below); none swings more than ~1.1%, three orders of magnitude below `G`'s
6× on the same DrivAerML row:

| benchmark | target | model | published | precision | ratio | rounding interval | swing |
|---|---|---|---|---|---|---|---|
| DrivAerML | drag_force_N | X-MeshGraphNet | 0.92 | 2dp | 1.0577 | [1.0520, 1.0635] | 1.09% |
| DrivAerML | drag_force_N | FIGConvNet | 0.97 | 2dp | 1.0032 | [0.9980, 1.0084] | 1.03% |
| DrivAerML | drag_force_N | **DoMINO** | 0.98 | 2dp | 0.9929 | [0.9879, 0.9980] | **1.02%** |
| DrivAerML | drag_force_N_rank | X-MeshGraphNet | 0.96 | 2dp | 1.0246 | [1.0193, 1.0300] | 1.04% |
| DrivAerML | drag_force_N_rank | FIGConvNet/DoMINO | 0.99 | 2dp | 0.9935 | [0.9886, 0.9986] | 1.01% |
| DrivAerNet++ | cd | PointNet | 0.643 | 3dp | 1.1454 | [1.1445, 1.1463] | 0.16% |
| DrivAerNet++ | cd | GCNN | 0.596 | 3dp | 1.2357 | [1.2347, 1.2367] | 0.17% |
| DrivAerNet++ | cd | RegDGCNN | 0.641 | 3dp | 1.1489 | [1.1480, 1.1498] | 0.16% |
| DrivAerNet++ | cd | TripNet | 0.957 | 3dp | 0.7696 | [0.7692, 0.7700] | 0.10% |
| DrivAerNet++ | cd | PointNet2D+BiLSTM | 0.9528 | 4dp | 0.7730 | [0.7729, 0.7730] | 0.01% |
| AirfRANS | cl | MLP/GraphSAGE/PointNet/Graph U-Net | 0.913–0.967 | 3dp | 1.02–1.08 | (width <0.001) | ~0.11% |
| AirfRANS | cl | Transolver | 0.9978 | 4dp | 0.9843 | [0.9843, 0.9844] | 0.01% |

The DoMINO row is the direct comparison the review makes for `G`: **1.02%
here against `G`'s 600%**, because this ratio's denominator (≈0.98) sits far
from zero while `G`'s (0.0269) sits close to it. On this evidence the ratio
is kept — demoted, renamed, and now with an automatic
`rounding_sensitive` flag (threshold 15% relative swing, chosen to sit
well above what is observed here and well below `G`'s failure mode) should
a future entry land in the dangerous regime. `published_relative_ratio`
accepts an optional `published_precision` and computes
`rounding_interval` whenever it is supplied; every entry above three of
these five benchmarks' published tables now carries a `precision` value
(`benchmarks.py`) so the check runs by default rather than needing to be
requested.

### 1.4 Thresholds — and the honest limit of five data points

Unchanged from the first draft, restated for the renamed statistic. The
five `R²_meta` values measured here (§4, per benchmark's own protocol) span
**0.10 (WindsorML) to beyond 1.0 relative to published** (DrivAerML: the
null's CI covers two of three published entries). That is the
interpretation this artifact can defend:

* **`R²_meta` near or above a published entry** (DrivAerML, and the *lift*
  target on AirfRANS) means a regression on published metadata is
  statistically indistinguishable from, or better than, at least one
  deployed model on this benchmark's own protocol.
* **`R²_meta` for a target whose published metric does not clear its own
  floor** (AirfRANS *drag*: every published ρ_D is negative) is a stronger
  and different finding than any ratio — the metric itself does not
  currently discriminate the thing it is named after.
* **`R²_meta` well below a published entry that clears decisively**
  (WindsorML, 0.10 against an implied floor of 0.79) is the genuine
  counterexample: the phenomenon is not universal.

**No numeric pass/fail band is proposed**, for the reason already stated in
the first draft: five benchmarks, drawn from one subfield, four of five from
two related dataset families, is not a basis for a cardinal threshold. What
would sharpen this is unchanged: more benchmarks outside automotive/
aerodynamic CFD; published-side seed spread on more than one of the five;
a benchmark with a much richer metadata block.

### 1.5 Relationship to `null_mechanism.md` — largely superseded by §2.4/§4

The first draft of this report reconciled with `null_mechanism.md` by
citing its harmonised numbers in prose while keeping the per-benchmark
tables as the only computed artifact. That reconciliation is now **built**,
not narrated: `neuroforge.nullbench.harmonised` computes the harmonised
`R²_meta` natively (§2.4) and reproduces `null_mechanism.json`'s
`null_r2_linear` to floating-point precision for all five benchmarks
(`tests/test_nullbench.py::test_harmonised_reproduces_null_mechanism_json`).
Three items from the first draft's reconciliation remain and are restated
precisely:

1. **The flexible ceiling is sourced, not recomputed** — see §2.4 and §3 for
   the explicit scope decision on why.
2. **WindsorML's config `description` is corrected** (`benchmarks.py`): the
   paper describes seven design parameters; the published
   `geo_parameters_all.csv` ships six plus a derived `frontal_area` column
   97.1% explained by `clearance` alone. Cited to
   `null_mechanism.md` §4.
3. **The per-benchmark DrivAerML target's `note` field now points to the
   harmonised split-free anchor** (0.9598) directly, rather than only in
   prose here.

### 1.6 Rounding-sensitivity: full table computed in §1.3, no changes needed here

(Content moved into §1.3 as the check the coordinator asked for; kept as a
numbered anchor for cross-reference from `docs/NULLBENCH.md`.)

---

## 2. The package and CLI

`src/neuroforge/nullbench/` — pure numpy/scipy, no torch import, so
`neuroforge nullbench` is cheap even before the solver stack is touched.

| file | contents |
|---|---|
| `stats.py` | `r2_score`, `spearman_score`, `mse_score`, `bootstrap_ci`/`bootstrap_ci_raw`, `published_relative_ratio`, `verdict_higher`/`verdict_lower`/`verdict_bound_lower`/`verdict_bound_higher` |
| `fit.py` | OLS `fit_predict`, `kfold_oos`, `run_official_split`/`run_kfold` |
| `io.py` | `Table`, `load_table` — the entire CSV contract |
| `harness.py` | `run_null` (protocol selection + scoring + comparisons), `permutation_check`, `PublishedEntry`, `NullResult` |
| `benchmarks.py` | the five worked-example configs, own protocol, with every published number's source |
| `harmonised.py` | the cross-benchmark-comparable view — one protocol, all five, sourced flexible ceiling |
| `leaderboard.py` | `build_leaderboard`/`build_all_leaderboards`/`build_harmonised_table`, JSON + markdown rendering |
| `cli.py` | `run` / `bench` / `leaderboard` (`--name` / `--all` / `--harmonised`) subcommands |
| `data/<benchmark>/*.csv` | the five distilled, committed metadata tables (own protocol) |
| `data/harmonised/*.csv` | the five distilled, committed metadata tables (harmonised protocol) + `flexible_ceiling_sourced.json` |

Wired into the top-level CLI: `neuroforge nullbench <run|bench|leaderboard> ...`
(`src/neuroforge/cli.py::cmd_nullbench`, forwarding argv via
`argparse.REMAINDER` — heavy imports stay lazy, matching every other
subcommand in that file).

### 2.1 The CSV contract (no field-data dependency, by construction)

`io.load_table(path, id_col, label_col, feature_cols=None, split_col=None)`
reads a plain CSV with the standard library `csv` module (no pandas
dependency, matching `scripts/covariate_null_crossbench.py`'s existing
convention). `feature_cols=None` auto-detects every non-id/label/split
column; benchmarks whose CSV carries more than one possible label (AirfRANS:
`cl`+`cd`; AhmedML/WindsorML: `cd`+`cl`; DrivAerML-harmonised:
`cd`+`cl`+`clf`+`clr`+`cs`) must pass `feature_cols` explicitly so the
*other* target(s) cannot silently leak in as a feature. This was caught by
the harness's own reproduction tests during development (both the original
per-benchmark build and again while building the harmonised CSVs) — see
`AIRFRANS_PARAM_COLS` / `AHMEDML_PARAM_COLS` / `WINDSORML_PARAM_COLS` in
`benchmarks.py`, the equivalent explicit lists in `harmonised.py`, and the
regression test `test_kfold_protocol_used_when_no_split_col`.

### 2.2 Split handling

`harness.run_null` inspects `Table.has_split()`. If the CSV carries a split
column, rows tagged with `train_values` are fit on and rows tagged with
`test_values` are scored (`protocol = "official_split"`). Otherwise the null
is refit by `folds`-fold out-of-sample cross-validation
(`protocol = "kfold_oos_k{folds}"`), and **every** output record — CLI JSON,
`NullResult.to_dict()`, and the leaderboard JSON — carries the `protocol`
string. Three of the five per-benchmark worked examples use `official_split`
(AirfRANS, DrivAerML, DrivAerNet++); two use K-fold (AhmedML, WindsorML).
The harmonised view (§2.4) uses K-fold uniformly for all five, including the
three that have an official split — that is the point of it.

### 2.3 One command, worked example

```
neuroforge nullbench bench --name drivaerml --target drag_force_N --boot 10000
neuroforge nullbench leaderboard --all --out-dir results/nullbench/leaderboards
neuroforge nullbench leaderboard --harmonised --out-dir results/nullbench/leaderboards
```

A benchmark maintainer with their own data runs:

```
neuroforge nullbench run --csv cases.csv --id-col id --label-col cd \
  --split-col split --metric r2 --published published.json --permute-check
```

### 2.4 The harmonised cross-benchmark view

`neuroforge.nullbench.harmonised` answers a question `benchmarks.py`
structurally cannot: *which benchmark's metadata null is strongest, compared
on equal footing?* Fixed protocol, named as a field in every emitted record
(not only in prose): **10-fold out-of-sample OLS, seed 0, R² (sklearn
convention), constant reference area where both conventions are published**
— mirroring `scripts/null_mechanism.py`'s own loaders (`load_airfrans`,
`load_simple`, `load_drivaernet`) byte-for-byte in row order and column set,
built by `scripts/nullbench_build_data.py`'s `build_harmonised_*`
functions. Row order matters here even though OLS-with-intercept is
mathematically invariant to `null_mechanism.py`'s per-fold standardisation
(verified: standardising every column by an invertible per-fold affine
transform cannot change fitted values when an intercept is included) —
because the 10-fold split itself is **position-based**
(`np.random.default_rng(seed).permutation(n)`), so matching row order is
what makes `kfold_oos` reproduce `null_r2_linear` bit-for-bit rather than
merely approximately. Verified empirically before the other four benchmarks
were built (AhmedML first, diff `5.6e-16`) and pinned for all five in
`tests/test_nullbench.py`.

**No published comparison in this view, and the emitted JSON says so
directly** (not only in this document): every `HarmonisedResult` carries
`"published_comparison": null` with a `published_comparison_note` field
stating that published entries were scored under each benchmark's own
protocol and comparing them here would repeat the exact mixing error this
artifact exists to flag.

**The flexible ceiling is sourced, not recomputed — an explicit scope
decision.** `null_mechanism.md` §3 shows a linear null alone conflates two
different failures: AirfRANS (`R²_meta` 0.6833) and AhmedML (0.6842) are
statistically indistinguishable linear nulls with opposite causes, separated
only by how much higher a flexible model can go (0.9463 vs 0.7748 — a
model-class limitation vs a genuine information deficit). Reproducing that
flexible-model family natively inside NullBench — quadratic expansion +
ridge path + kNN/local-linear variants with model selection over the path,
handling near-collinear columns numerically (WindsorML's `frontal_area` is
97% redundant with `clearance`) — is a materially larger unit of work than
packaging an existing, already-verified number, with its own numerical-
stability edge cases that would need their own test suite. **Declined for
this release.** Instead, `harmonised.py` reads the already-computed value
from `nullbench/data/harmonised/flexible_ceiling_sourced.json`, a small
distillation of `results/review/null_mechanism.json` committed with the
package (so it survives a `pip install`, unlike reading the top-level
`results/` tree directly). Every sourced field is named `*_sourced`
(`flexible_ceiling_r2_sourced`, `flexible_ceiling_best_model_sourced`,
`linearity_share_sourced`) so it is never mistaken for something this
harness computed, and a transcription-guard test
(`test_flexible_ceiling_matches_null_mechanism_json_verbatim`) asserts every
value still matches the source file exactly. **A native flexible-ceiling
implementation, so a maintainer running NullBench on their OWN benchmark
gets one too (not just the five pre-computed here), is real follow-up work
and is not scoped into this release.**

The harmonised table, reproduced exactly (`tests/test_nullbench.py`):

| benchmark | n | `metadata_null_r2` | 95% CI | flexible ceiling (sourced) | linearity share (sourced) |
|---|---|---|---|---|---|
| DrivAerML | 484 | **0.9598** | [0.9523, 0.9659] | 0.9883 | 0.97 |
| DrivAerNet++ | 4165 | **0.8248** | [0.8151, 0.8342] | 0.8811 | 0.94 |
| AhmedML | 499 | **0.6842** | [0.6377, 0.7273] | 0.7748 | 0.88 |
| AirfRANS | 200 | **0.6833** | [0.6197, 0.7372] | 0.9463 | 0.72 |
| WindsorML | 355 | **0.1045** | [−0.1427, 0.2668] | 0.2516 | 0.42 |

Read the AirfRANS/AhmedML pair together, as `null_mechanism.md` recommends:
identical linear null, opposite cause. AirfRANS's flexible ceiling (0.9463)
is far above its linear number — the metadata nearly determines the label,
linear regression just cannot express the map. AhmedML's flexible ceiling
(0.7748) is barely above its linear number — most of the shortfall is
information the published parameters do not carry at all, not a model-class
limitation.

---

## 3. Test coverage

`tests/test_nullbench.py`, **56 tests** (up from 29 in the first draft — 27
new: rounding-sensitivity checks, renamed-field assertions, harmonised
reproduction ×5 + n_cases + ranking + no-comparison, flexible-ceiling
transcription guard ×2), fast suite (no `-m slow` needed; slowest individual
test 4.3 s — the AirfRANS full-boot reproduction; every other worked-example
and harmonised reproduction runs in well under a second at `n_boot=10000`
because the design matrices are small).

* **Statistic correctness**: R² floor = 0 / ceiling = 1 by construction;
  Spearman of a constant prediction is `nan`, not silently mapped to the
  floor; ratio undefined when published ≤ floor; ratio undefined when
  published is `None`; ratio `> 1` and `< 0` reported, never clipped, with
  the correct flags in both directions; `NullResult.to_dict()` emits the
  metric-tagged `metadata_null_{r2,spearman}` and `metadata_null_mse` fields
  and never the retired `covariate_null_fraction` key.
* **Rounding sensitivity** (new): the DrivAerML DoMINO row swings <5% (it
  measures ~1.02%), not the 6× the positioning review demonstrates for a
  different ratio; a synthetic published value near the floor at 2-decimal
  precision is correctly flagged `rounding_sensitive`; a rounding band that
  itself straddles the floor is flagged `rounding_interval_crosses_floor`
  rather than silently computing a nonsense (near-infinite) interval.
* **No-drift check against the pre-registered rule**: `verdict_higher`,
  `verdict_lower`, and `verdict_bound_lower` are *re-implemented* in
  `stats.py` (not imported — this package has no import dependency on the
  top-level `scripts/` tree) and cross-checked against
  `scripts/covariate_null.py::verdict` and
  `scripts/covariate_null_crossbench.py::verdict_lower`/`verdict_bound_lower`
  on a 200-case randomised grid plus the `published=None` edge case.
* **Generic harness**: official-split protocol selection and labelling,
  K-fold protocol selection and labelling, auto feature-column detection,
  a clear `ValueError` on a missing column, published-entry comparison
  (verdict + ratio) end to end on a synthetic fixture.
* **Permutation test**: a synthetic dataset with real signal (R² > 0.8)
  collapses to R² < 0.1 when refit on label-shuffled training data.
* **Reproduction, own protocol, exact to `rel=1e-9`**, for all four
  R²-metric worked examples (DrivAerML, DrivAerNet++, AhmedML, WindsorML)
  and the Spearman worked example (AirfRANS, both `cl` and `cd`), including
  the bootstrap **interval** — see §3.1's note on why this is not automatic.
* **Reproduction, harmonised protocol, exact to floating-point precision**
  (new): all five benchmarks' `metadata_null_r2` against
  `null_mechanism.json`'s `null_r2_linear`; n_cases match; the published
  ranking (DrivAerML > DrivAerNet++ > AhmedML > AirfRANS > WindsorML) is
  reproduced from the harness's own output, not hard-coded.
* **Flexible-ceiling transcription guard** (new): every sourced value
  matches `null_mechanism.json` verbatim (`rel=1e-12`); a sanity check that
  the sourced flexible ceiling is never below the harness's own computed
  linear null (within numerical tolerance) for any benchmark.
* **Registry smoke test**: every one of the five configs loads and runs for
  every declared target.

### 3.1 A reproducibility trap this work found and fixed (both protocols)

Reproducing a bootstrap **interval** bit-for-bit (not just the point
estimate) requires the distilled CSV's row order to match the *source*
script's row order exactly, because the percentile bootstrap and the 10-fold
split both resample/partition cases **by array position**. This bit twice:

* **Own-protocol AirfRANS** (first draft): `scripts/covariate_null_trainfit.py`
  scores in the AirfRANS dataset's own (non-alphabetical) case order;
  `scripts/nullbench_build_data.py` preserves it explicitly for the test
  split rather than re-sorting.
* **Harmonised AhmedML/WindsorML/DrivAerML** (this revision):
  `scripts/null_mechanism.py::load_simple` iterates the **geo file**, not
  the **force file** — the opposite of `build_ashton`'s own-protocol
  builder. `scripts/nullbench_build_data.py::_load_simple_harmonised`
  mirrors `load_simple` exactly (geo-file iteration, `int(float(id))`
  keying) rather than adapting the existing own-protocol builder; the
  harmonised AirfRANS CSV additionally drops the `alpha²` term the
  own-protocol worked example includes, and uses alphabetical case-name
  order (`sorted(lab)`), matching `load_airfrans` exactly, not
  `covariate_null_trainfit.py`'s dataset order.

Both traps are documented in `nullbench_build_data.py`'s docstrings at the
point they would bite a maintainer who assumed "same source files" implies
"same row order".

---

## 4. The five corrected leaderboards (per-benchmark, own protocol) + the harmonised table

Machine-readable: `results/nullbench/leaderboards/{name}.json` (and
`ALL.md`/per-benchmark `.md` for the rendered form; `harmonised.json` /
`harmonised.md` for the cross-benchmark view), regenerated by
`neuroforge nullbench leaderboard --all` (which now also writes the
harmonised table) or `--harmonised` alone — never hand-edited. Headline row
below; full per-model tables (every verified published entry, its
`published_relative_ratio`, its verdict) are in the linked markdown files.

**Read this per-row, in its own metric and protocol — not as a cross-benchmark
ranking.** For that ranking, use §2.4's harmonised table instead; mixing the
two is exactly the error this project exists to flag.

| Benchmark | target | protocol | `metadata_null_{metric}` | best verified published | ratio | verdict |
|---|---|---|---|---|---|---|
| [AirfRANS](../../../results/nullbench/leaderboards/airfrans.md) | `cl` (Spearman) | official_split (800/200) | 0.9821 [0.9737, 0.9866] | Transolver 0.9978 | 0.9843 | clears |
| [AirfRANS](../../../results/nullbench/leaderboards/airfrans.md) | `cd` (Spearman) | official_split (800/200) | 0.9318 [0.8981, 0.9531] | none above floor (best: PointNet −0.022) | **undefined** — every published entry ≤ floor | BELOW THE NULL (all 4) |
| [DrivAerML](../../../results/nullbench/leaderboards/drivaerml.md) | `drag_force_N` (R²) | official_split (436/48) | 0.9731 [0.9581, 0.9826] | DoMINO 0.98 | 0.9929 | straddles |
| [DrivAerNet++](../../../results/nullbench/leaderboards/drivaernet.md) | `cd` (R²) | official_split (5819/1154) | 0.7365 [0.7050, 0.7643] | TripNet 0.957 | 0.7696 | clears |
| [AhmedML](../../../results/nullbench/leaderboards/ahmedml.md) | `cd` (R²) | kfold_oos_k10 | 0.6842 [0.6377, 0.7273] | none published | -- | not adjudicable |
| [WindsorML](../../../results/nullbench/leaderboards/windsorml.md) | `cd` (R²) | kfold_oos_k10 | 0.1045 [−0.1427, 0.2668] | MeshGraphNet ≥0.7916 (bound) | -- (bound) | clears (published bound beats the null outright) |
| [harmonised](../../../results/nullbench/leaderboards/harmonised.md) (all five, one protocol) | `cd`/`cl` (R²) | kfold_oos_k10, all five | see §2.4 table | *(none — see §2.4)* | *(none — see §2.4)* | *(ranking only)* |

Every published number in every per-benchmark table traces to a row already
verified at source in `docs/paper/review/published_baselines_verified.md`
(AirfRANS) or `docs/paper/review/null_travels.md` (the other four). No entry
was added here that is not already in one of those two files; where no
verified baseline exists (AhmedML entirely; WindsorML beyond the one bound)
the leaderboard says so rather than filling a row in.

---

## 5. What could not be verified / what was declined, and why

* **Literature-uniqueness of the name** — carried over from
  `naming_and_positioning.md`, not independently re-verified: stated as a
  limitation there and here, because neither session had a live search tool.
* **A native flexible-ceiling implementation** (§2.4) — declined for this
  release. Scope: quadratic expansion, a ridge path (needed for numerical
  stability on near-collinear metadata — WindsorML's `frontal_area` is 97%
  explained by `clearance` alone), a kNN/local-linear family, model
  selection over that family by out-of-fold performance, and a test suite
  covering the near-collinear edge case specifically. This is a materially
  larger unit of work than packaging an existing verified number; the five
  already-computed values are shipped (sourced, labelled, transcription-
  guarded) instead. If wanted, it is real follow-up work, not a small
  addition to this release.
* **AhmedML and WindsorML `cl` (lift) targets carry no published baseline**
  either; both are shipped as runnable targets but contribute nothing to
  either headline table.
* **The DrivAerNet++ own-protocol row is an acknowledged LOWER bound** (zero-
  fills 595 unparametrised test designs); the harmonised row (§2.4, 0.8248)
  excludes them entirely by construction and is the tighter number.
* **No sixth benchmark was added.** DeepCFD remains not runnable (no
  per-case force labels at all; `null_travels.md` §2.5).
* **This report and the harness were not reviewed against any new source
  document beyond `null_travels.md`, `published_baselines_verified.md`,
  `null_mechanism.md`, and `naming_and_positioning.md`.**

---

## 6. Commands to reproduce everything in this report

```bash
# rebuild the ten worked-example CSVs (five own-protocol + five harmonised)
# from committed/cached metadata
.venv/Scripts/python.exe scripts/nullbench_build_data.py

# fast test suite, including the 56 nullbench tests
.venv/Scripts/python.exe -m pytest -q

# rebuild the five per-benchmark leaderboards AND the harmonised table
PYTHONPATH=src .venv/Scripts/python.exe -m neuroforge.cli nullbench leaderboard --all \
  --boot 10000 --out-dir results/nullbench/leaderboards

# harmonised table alone
PYTHONPATH=src .venv/Scripts/python.exe -m neuroforge.cli nullbench leaderboard --harmonised \
  --boot 10000 --out-dir results/nullbench/leaderboards

# top-level artifact manifest still verifies
.venv/Scripts/python.exe scripts/make_manifest.py --check
```
