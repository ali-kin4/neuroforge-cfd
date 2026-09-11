# NullBench: turning `null_travels.md` into something the field can run

`null_travels.md` measured that the predictive power of published per-case
metadata alone spans an order of magnitude across five CFD-ML benchmarks
(R² 0.10 to 0.97), and that not one of the five reports this. That finding
stays inert if it lives only in this repository's review notes. This report
is the artifact that turns it into something a benchmark maintainer or a
referee can run in one command: a named statistic, a tested package
(`src/neuroforge/nullbench/`), a CLI (`neuroforge nullbench`), and five
corrected leaderboards.

Everything below was produced on branch `paper1/reframe-after-jcp`, HEAD
`e393981` at the time of writing. **This report does not touch the
manuscript** (`docs/paper/body.tex`, `abstract.tex`, `sections/`) — those
files are owned by a concurrent thread running the interpolation ladder.

---

## 1. The statistic

### 1.1 Name

**Covariate-null fraction (CNF).** Kept as the working name in the task
brief rather than replaced, for a concrete reason: continuity. It is the
same object already named "the covariate null" three times over in this
repository's own provenance chain — `scripts/covariate_null.py`,
`scripts/covariate_null_trainfit.py`, `scripts/covariate_null_crossbench.py`,
and `docs/paper/review/null_travels.md`'s pre-registered decision rule
(commit `8928087`) — and "fraction" is the one word that has to be added to
turn a comparison ("the null beats/loses to model X") into a single
reportable number (the *share* of X's value the null already reaches). A
clever alternative name would sever that link for no benefit.

**Uniqueness check — with an explicit limitation.** This environment has no
live web/literature search tool. I cannot run a database query against the
ML literature and confirm zero prior use with certainty. What I can state
from the material available in-repository and in training data: "covariate
null" is a standard statistics term (a model using only covariates, not the
outcome-generating mechanism) and is not itself a proprietary metric name;
"covariate-null fraction" as a *specific, defined, reported statistic* does
not appear in any of the ~40 papers, preprints, and benchmark documents read
at source for this project (listed in `null_travels.md` and
`published_baselines_verified.md`). Adjacent but distinct existing terms
that a reviewer might confuse this with, and why CNF is not a duplicate:

* **"Null model" / "baseline"** — the model itself, not a normalised
  fraction of a *reported metric's* value.
* **Permutation feature importance** — measures the drop in a *trained*
  model's own score when a feature is shuffled; CNF never trains the model
  being evaluated at all.
* **"Skill score"** (meteorology/forecasting) — compares a model to a
  reference *forecast* (e.g. climatology) in the *same* metric space and is
  close in spirit, but is not defined with a floor/ceiling convention
  spanning both R²-style and rank-style metrics, is not bootstrapped by
  convention, and is not applied to this literature. If a reviewer or a
  benchmark maintainer identifies a prior use, the name should change; that
  risk is stated here rather than hidden.

### 1.2 Definition

For a target metric family `m ∈ {r2, spearman}` with floor `F_m` and ceiling
`C_m`:

* **R²-style** (`m = r2`): `F = 0` (an intercept-only / constant predictor
  scores exactly 0 by the definition of R², `1 − SSE/SST`), `C = 1`.
* **Rank-style** (`m = spearman`): `F = 0` (the expected rank correlation of
  a random ranking is 0), `C = 1`.

Given the null's out-of-sample point estimate `n` (with bootstrap interval
`[n_lo, n_hi]`) and a published value `p`:

```
CNF = n / p                      if p > F     (the only regime where a "share of p" is meaningful)
CNF = undefined                  if p <= F    (flag: published_at_or_below_floor)
```

`CNF` is **never clipped**. `CNF > 1` (the null exceeds the published value)
is reported and flagged `null_exceeds_published`; `n <= F` is reported and
flagged `null_at_or_below_floor`. The reported interval is
`[n_lo/p, n_hi/p]` — it propagates only the null's own case-level bootstrap
uncertainty; a published standard deviation, where one exists, is carried
alongside (`published_std`) rather than folded in, because combining a
resampled distribution with an assumed-Gaussian one from a different study
and a different (usually smaller, non-bootstrapped) sample would manufacture
precision this artifact does not have.

Implementation: `src/neuroforge/nullbench/stats.py::covariate_null_fraction`.
Every one of these design choices is exercised by a test in
`tests/test_nullbench.py` (undefined-below-floor, never-clipped-above-one,
never-clipped-below-zero, published-not-reported).

### 1.3 Why the R² and Spearman branches are not unified into one currency

R² answers "share of explained variance"; Spearman answers "share of
achieved ranking". A benchmark that reports Spearman (AirfRANS, following
Bonnet et al. 2022) and one that reports R² (DrivAerML, DrivAerNet++,
AhmedML, WindsorML, following the PhysicsNeMo-CFD and DrivAerNet++ papers)
are not automatically comparable just because both metrics are
floor-0/ceiling-1 on the surface. `stats.py`'s `Metric` type keeps every CNF
record tagged with which branch produced it (`"metric": "r2"` or
`"spearman"`), and nothing in the harness collapses the two into a single
cross-benchmark scalar. The headline table in §4 makes this explicit per row.

### 1.4 Thresholds — and the honest limit of five data points

The five CNF headline values measured here (§4) span **0.10 (WindsorML) to
beyond 1.0 (DrivAerML: the null's CI covers two of three published
entries)**. That is the interpretation this artifact can defend:

* **A CNF near or above 1** (DrivAerML, and the *lift* target on AirfRANS)
  means a regression on published metadata is statistically
  indistinguishable from, or better than, at least one deployed model on
  this benchmark's own protocol — a strong signal the benchmark's current
  leaderboard is not yet separating "learned from the flow field" from
  "learned from the file name".
* **A CNF that is undefined because the published metric does not clear its
  own floor** (AirfRANS *drag*: every published ρ_D is negative) is a
  stronger and different finding than any fraction — it says the metric
  itself does not currently discriminate the thing it is named after.
* **A CNF well below 1 with the published entry clearing decisively**
  (WindsorML, 0.10 against an implied floor of 0.79) is the genuine
  counterexample: it demonstrates the phenomenon is not universal, and that
  a benchmark *can* be constructed (or a body geometry chosen) where
  metadata alone is nowhere near sufficient.

**No numeric pass/fail band (e.g. "CNF > 0.5 is concerning") is proposed.**
Five benchmarks, drawn from one subfield (external vehicle/airfoil
aerodynamics, four of five from two related dataset families — DrivAer* and
the Ashton et al. Ahmed/Windsor pair), is not a basis for a cardinal
threshold, and inventing one would misrepresent the evidence. What would
sharpen this: (a) more benchmarks outside automotive/aerodynamic CFD, to
test whether the spread is a property of this subfield or of tabular-CFD
benchmarking generally; (b) seed spread reported on the *published* side for
more than one of the five benchmarks (only AirfRANS's Table 3 and
DrivAerNet++'s MSE/MAE currently carry one), so `value_ci95` could be a true
combined interval rather than a null-only one; (c) a benchmark where the
metadata block itself is much richer (dozens to hundreds of parameters) to
see whether CNF saturates.

### 1.5 Relationship to `null_mechanism.md`

This branch already contains a follow-up to `null_travels.md`
(`docs/paper/review/null_mechanism.md`, committed before this task started)
that searched for what *predicts* covariate-null strength across the same
five benchmarks. It is a separate, already-completed research thread — this
task did not re-run or extend it, and none of CNF's numbers here were
changed to match it — but three of its findings bear directly on artifacts
shipped in this release and are recorded here rather than left as a silent
inconsistency between two files in the same directory:

1. **CNF v1 reports one number; `null_mechanism.md` argues that is
   insufficient.** Its §3 shows AirfRANS (0.6833) and AhmedML (0.6842) have
   statistically indistinguishable *linear* nulls with opposite causes — a
   model-class failure on AirfRANS (a flexible learner reaches 0.946) versus
   a genuine information deficit on AhmedML (a flexible learner reaches only
   0.775) — and recommends reporting the pair `(R²_linear, R²_flexible)`.
   NullBench v1 (this release) does **not** implement a flexible-fit ceiling;
   `covariate_null_fraction` is linear-null-only. This is a stated limitation
   of the statistic as shipped, not an oversight: adding a flexible second
   number is future work, tracked here rather than silently deferred.
2. **WindsorML's parameter list is corrected in `benchmarks.py` accordingly.**
   `null_mechanism.md` §4 establishes, source-verified against the live
   HuggingFace file, that the WindsorML paper describes seven design
   parameters but `geo_parameters_all.csv` ships only six plus a derived
   `frontal_area` column that is 97.1% explained by `clearance` alone. The
   `WINDSORML` config's `description` field originally said "7 published
   shape parameters"; it now says what is actually true (6 + 1 derived, one
   design parameter missing from the release) and cites the finding that
   this metadata gap, not the body's physics, is the more likely explanation
   for WindsorML being the one counterexample in §4 below.
3. **The DrivAerML headline (0.9731, §4) is scored on PhysicsNeMo-CFD's own
   split, which sorts on drag and takes the outer deciles** — appropriate and
   necessary for the exact head-to-head against DoMINO/FIGConvNet/
   X-MeshGraphNet (they were scored on the same split), and already flagged
   as such in that benchmark's `target_note` in the leaderboard. The
   split-free anchor `null_mechanism.md` reports on the same metadata under
   plain 10-fold OOS is **0.9598** — lower, and not comparable to the
   published entries (they were never scored under that protocol), but the
   more representative number if this CNF is ever quoted outside a
   head-to-head context. Both figures are legitimate; they answer different
   questions, and conflating them would be the error.

---

## 2. The package and CLI

`src/neuroforge/nullbench/` — pure numpy/scipy, no torch import, so
`neuroforge nullbench` is cheap even before the solver stack is touched.

| file | contents |
|---|---|
| `stats.py` | `r2_score`, `spearman_score`, `bootstrap_ci`, `covariate_null_fraction`, `verdict_higher`/`verdict_lower`/`verdict_bound_lower`/`verdict_bound_higher` |
| `fit.py` | OLS `fit_predict`, `kfold_oos`, `run_official_split`/`run_kfold` |
| `io.py` | `Table`, `load_table` — the entire CSV contract |
| `harness.py` | `run_null` (protocol selection + scoring + comparisons), `permutation_check`, `PublishedEntry`, `NullResult` |
| `benchmarks.py` | the five worked-example configs, with every published number's source |
| `leaderboard.py` | `build_leaderboard`/`build_all_leaderboards`, JSON + markdown rendering |
| `cli.py` | `run` / `bench` / `leaderboard` subcommands |
| `data/<benchmark>/*.csv` | the five distilled, committed metadata tables (parameters + labels only) |

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
`cl`+`cd`; AhmedML/WindsorML: `cd`+`cl`) must pass `feature_cols` explicitly
so the *other* target cannot silently leak in as a feature. This was caught
by the harness's own reproduction tests during development (§3) — see
`AIRFRANS_PARAM_COLS` / `AHMEDML_PARAM_COLS` / `WINDSORML_PARAM_COLS` in
`benchmarks.py` and the regression test
`test_kfold_protocol_used_when_no_split_col`, which fixture-reproduces
exactly this failure mode.

### 2.2 Split handling

`harness.run_null` inspects `Table.has_split()`. If the CSV carries a split
column, rows tagged with `train_values` are fit on and rows tagged with
`test_values` are scored (`protocol = "official_split"`). Otherwise the null
is refit by `folds`-fold out-of-sample cross-validation
(`protocol = "kfold_oos_k{folds}"`), and **every** output record — CLI JSON,
`NullResult.to_dict()`, and the leaderboard JSON — carries the `protocol`
string, so which one ran is never ambiguous downstream. Three of the five
worked examples use `official_split` (AirfRANS, DrivAerML, DrivAerNet++);
two use K-fold because no split id list is published (AhmedML, WindsorML).

### 2.3 One command, worked example

```
neuroforge nullbench bench --name drivaerml --target drag_force_N --boot 10000
neuroforge nullbench leaderboard --all --out-dir results/nullbench/leaderboards
```

A benchmark maintainer with their own data runs:

```
neuroforge nullbench run --csv cases.csv --id-col id --label-col cd \
  --split-col split --metric r2 --published published.json --permute-check
```

---

## 3. Test coverage

`tests/test_nullbench.py`, 29 tests, fast suite (no `-m slow` needed; slowest
individual test 3.7 s — the AirfRANS full-boot reproduction; the other four
worked-example reproductions run in 0.1–0.25 s each at `n_boot=10000`
because the design matrices are small).

* **Statistic correctness**: R² floor = 0 / ceiling = 1 by construction;
  Spearman of a constant prediction is `nan`, not silently mapped to the
  floor; CNF undefined when published ≤ floor; CNF undefined when published
  is `None`; CNF `> 1` and `< 0` reported, never clipped, with the correct
  flags in both directions.
* **No-drift check against the pre-registered rule**: `verdict_higher`,
  `verdict_lower`, and `verdict_bound_lower` are *re-implemented* in
  `stats.py` (not imported — this package has no import dependency on the
  top-level `scripts/` tree, which is not installed with the package) and
  cross-checked against `scripts/covariate_null.py::verdict` and
  `scripts/covariate_null_crossbench.py::verdict_lower`/`verdict_bound_lower`
  on a 200-case randomised grid plus the `published=None` edge case, so the
  two implementations cannot silently diverge.
* **Generic harness**: official-split protocol selection and labelling,
  K-fold protocol selection and labelling, auto feature-column detection,
  a clear `ValueError` on a missing column, published-entry comparison
  (verdict + CNF) end to end on a synthetic fixture.
* **Permutation test**: `test_permutation_check_collapses_to_the_floor` — a
  synthetic dataset with real signal (R² > 0.8) collapses to R² < 0.1 when
  refit on label-shuffled training data, confirming the pipeline is reading
  the metadata rather than manufacturing signal from array indices or a
  leaking feature.
* **Reproduction of committed numbers, exact to `rel=1e-9`**, for all four
  R²-metric worked examples (DrivAerML, DrivAerNet++, AhmedML, WindsorML)
  and the Spearman worked example (AirfRANS, both `cl` and `cd`), including
  the bootstrap **interval**, not just the point estimate — see the note
  on ordering below. Source: `results/review/covariate_null_crossbench.json`
  and `results/review/covariate_null_trainfit.json`.
* **Registry smoke test**: every one of the five configs loads and runs for
  every declared target.

### 3.1 A reproducibility trap this work found and fixed

Reproducing the AirfRANS bootstrap **interval** bit-for-bit (not just the
point estimate) required the distilled CSV's test-row order to match the
*original* AirfRANS dataset order used by
`scripts/covariate_null_trainfit.py`, because the percentile bootstrap
resamples cases **by array position** (`rng.integers(0, n, n)`), and the
point estimate (order-invariant) will match regardless while the CI
(position-dependent) silently will not. `scripts/nullbench_build_data.py`
documents this explicitly and verifies at commit-build time that the
committed test-label cache's key order is not alphabetically sorted before
relying on it. This is exactly the kind of trap a maintainer copying this
harness would hit silently; it is called out here so the pattern is
recognisable elsewhere.

---

## 4. The five corrected leaderboards

Machine-readable: `results/nullbench/leaderboards/{name}.json` (and
`ALL.md`/per-benchmark `.md` for the rendered form), regenerated by
`neuroforge nullbench leaderboard --all` — never hand-edited, so the tables
and the code that produced them cannot diverge. Headline row below; full
per-model tables (every verified published entry, its CNF, its verdict) are
in the linked markdown files.

| Benchmark | target | protocol | null (metric) | best verified published | CNF | verdict |
|---|---|---|---|---|---|---|
| [AirfRANS](../../../results/nullbench/leaderboards/airfrans.md) | `cl` (Spearman) | official_split (800/200) | 0.9821 [0.9737, 0.9866] | Transolver 0.9978 | 0.9843 | clears |
| [AirfRANS](../../../results/nullbench/leaderboards/airfrans.md) | `cd` (Spearman) | official_split (800/200) | 0.9318 [0.8981, 0.9531] | none above floor (best: PointNet −0.022) | **undefined** — every published entry ≤ floor | BELOW THE NULL (all 4) |
| [DrivAerML](../../../results/nullbench/leaderboards/drivaerml.md) | `drag_force_N` (R²) | official_split (436/48) | 0.9731 [0.9581, 0.9826] | DoMINO 0.98 | 0.9929 | straddles |
| [DrivAerNet++](../../../results/nullbench/leaderboards/drivaernet.md) | `cd` (R²) | official_split (5819/1154) | 0.7365 [0.7050, 0.7643] | TripNet 0.957 | 0.7696 | clears |
| [AhmedML](../../../results/nullbench/leaderboards/ahmedml.md) | `cd` (R²) | kfold_oos_k10 | 0.6842 [0.6377, 0.7273] | none published | -- | not adjudicable |
| [WindsorML](../../../results/nullbench/leaderboards/windsorml.md) | `cd` (R²) | kfold_oos_k10 | 0.1045 [−0.1427, 0.2668] | MeshGraphNet ≥0.7916 (bound) | -- (bound) | clears (published bound beats the null outright) |

**Read this table per-row, in its own metric — not as a ranking.** The
"null (metric)" column mixes Spearman (AirfRANS) and R² (the other four);
§1.3 explains why those are not the same currency and are never collapsed
into one scale anywhere in this package. If a single R²-only ranking is
wanted for a cross-benchmark comparison, `null_mechanism.md` §1 computes one
(AirfRANS harmonises to R² 0.6833, which *reorders* this table — see §1.5
above); that harmonised number is not reproduced here because it is a
different, already-completed piece of work with its own pre-registration,
not part of this release's worked examples.

Every published number in every table traces to a row already verified at
source in `docs/paper/review/published_baselines_verified.md` (AirfRANS) or
`docs/paper/review/null_travels.md` (the other four). No entry was added
here that is not already in one of those two files; where no verified
baseline exists (AhmedML entirely; WindsorML beyond the one bound) the
leaderboard says so rather than filling a row in.

---

## 5. What could not be verified / is out of scope here

* **Literature-uniqueness of the name** (§1.1) — stated as a limitation, not
  resolved, because this environment has no live search tool.
* **AhmedML and WindsorML `cl` (lift) targets carry no published baseline**
  either; both are shipped as runnable targets (so a future paper reporting
  an AhmedML/WindsorML lift number has something to check against
  immediately) but contribute nothing to the headline table above.
* **The DrivAerNet++ row is a acknowledged LOWER bound**, not the tightest
  possible null: it zero-fills the ~3956 DrivAerNet-v1 fastback designs
  whose 50-parameter table is not published (see `null_travels.md` §2.2).
  The true covariate-null fraction for that benchmark, given full metadata,
  is higher than 0.7696.
* **No sixth benchmark was added.** `null_travels.md` §2.5 records that
  DeepCFD is not runnable (no per-case force labels at all); it is not
  included in NullBench's registry for the same reason, and is not silently
  dropped — this is the explanation.
* **This report and the harness were not reviewed against a live copy of any
  paper past what `null_travels.md` and `published_baselines_verified.md`
  already captured** — no new source document was opened for this task.

---

## 6. Commands to reproduce everything in this report

```bash
# rebuild the five worked-example CSVs from committed/cached metadata
.venv/Scripts/python.exe scripts/nullbench_build_data.py

# fast test suite, including the 29 nullbench tests
.venv/Scripts/python.exe -m pytest -q

# rebuild the five leaderboards (JSON + markdown)
PYTHONPATH=src .venv/Scripts/python.exe -m neuroforge.cli nullbench leaderboard --all \
  --boot 10000 --out-dir results/nullbench/leaderboards

# top-level artifact manifest still verifies (this task added no HEADLINE entry
# that changes an existing hash)
.venv/Scripts/python.exe scripts/make_manifest.py --check
```
