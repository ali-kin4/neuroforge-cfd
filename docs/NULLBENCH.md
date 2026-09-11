# NullBench — report the metadata null

For a benchmark maintainer or a referee. Full definition, API and the five
worked examples: `docs/paper/review/nullbench_release.md`. Finding this
operationalises: `docs/paper/review/null_travels.md`. Naming rationale:
`docs/paper/review/naming_and_positioning.md`.

**The protocol** is called **the metadata null**: a regression on a
benchmark's own published per-case parameters, no simulation output opened.
**The statistic** is **metadata-only R²**, `R²_meta`, emitted as
`metadata_null_r2` (with a bootstrap CI), alongside `metadata_null_mse`.

## What to run

You need one CSV: one row per case, a case-id column, your published per-case
parameters, and your label column. No mesh, no field, no point cloud.

```bash
pip install neuroforge-cfd
neuroforge nullbench run \
  --csv your_cases.csv --id-col case_id --label-col cd \
  --split-col split --metric r2 \
  --published published.json \
  --permute-check --out result.json
```

* Omit `--split-col` to use 10-fold out-of-sample instead of an official split
  (state in your paper which one ran — the output's `split_used` field says
  so explicitly).
* `--metric r2` for an R²-style metric, `--metric spearman` for a
  rank-correlation metric — pick whichever your benchmark already reports.
* `--published published.json`: a JSON list of
  `{"name": ..., "value": ..., "std": ..., "source": ..., "precision": ...}`
  — the entries you want the null checked against. `precision` (decimal
  digits the published value was reported to) drives a rounding-sensitivity
  check on the comparison ratio below; omit `--published` entirely to just
  get the null and its interval.
* `--permute-check` refits on label-shuffled training data; the printed
  `permutation_check` value should land at or below the metric's floor (0).
  If it doesn't, something in your feature construction is leaking the label.

## What you get back

```json
{
  "metric": "r2",
  "protocol": "official_split",
  "metadata_null_r2": 0.973,
  "metadata_null_r2_ci95": [0.958, 0.983],
  "metadata_null_mse": 130.0,
  "metadata_null_mse_ci95": [95.2, 168.4],
  "comparisons": [
    {"model": "YourModel", "published": 0.98, "verdict": "straddles",
     "published_relative_ratio": {"published_relative_ratio": 0.993, "flags": []}}
  ]
}
```

Report **`metadata_null_r2`** (or `metadata_null_spearman`), its CI, and the
**protocol** for your headline model, in the table you already publish. One
sentence is enough:

> A regression on the [N] published case parameters alone (no field data)
> reaches R²_meta = X.XX [lo, hi] under [protocol].

The **`published_relative_ratio`** field (`metadata_null / published`, a
demoted comparison aid — never the headline; see "Naming" below) is useful
context but needs a competitor's published number and inherits that number's
precision. If `published_relative_ratio` is `null`, check `flags`:
`published_at_or_below_floor` means your own reported number does not clear
random/constant prediction, so no ratio is meaningful — report that fact
plainly, it is the more important finding. If `flags` contains
`rounding_sensitive`, the ratio moves by more than 15% across the published
value's own rounding band — quote the `rounding_interval`, not the point
ratio, or drop the comparison.

## Two views — do not build a cross-benchmark table from your own protocol's numbers

If you are comparing your null across *multiple* benchmarks (not just
checking your own against published baselines), do **not** put each
benchmark's own-protocol number in one table: a Spearman next to an R²
computed under a variance-inflating split is exactly the mixing error this
project exists to flag (`docs/paper/review/null_mechanism.md`). Use one
protocol identically across every benchmark you compare — NullBench ships
this as `neuroforge.nullbench.harmonised` / `neuroforge nullbench
leaderboard --harmonised`, with the fixed protocol string in every emitted
record.

## Ship it as a worked example

Contribute your benchmark's CSV + config to
`src/neuroforge/nullbench/benchmarks.py` (see the five existing entries) so
future authors on your benchmark never have to run the download-then-fit
pipeline again.

## Definition, thresholds, and what NOT to conclude from one number

See `docs/paper/review/nullbench_release.md` §1. In short: floor = 0
(intercept-only / random-rank), ceiling = 1, `R²_meta` is reported directly
(not as a ratio); the demoted `published_relative_ratio` is undefined when
the published value doesn't clear its own floor. Five benchmarks give
`R²_meta` from 0.10 to 0.97 under their own protocols — report the number, do
not round it into a fixed pass/fail bar; there is not yet enough
cross-benchmark evidence to defend one.

**One number is a start, not the whole picture.** A companion analysis in
this repository (`docs/paper/review/null_mechanism.md`) shows a *linear*
null alone can conflate two different failures that look identical as one
number and are not: AirfRANS (R²_meta 0.68) and AhmedML (R²_meta 0.68) are
statistically indistinguishable linear nulls, but a flexible model reaches
0.95 on AirfRANS (a model-class limitation — the metadata determines the
label almost fully, linear regression just cannot express the map) against
0.77 on AhmedML (a genuine information deficit — a quarter of the label's
variance is not recoverable from the published metadata at all). NullBench
v1's harness computes the linear `metadata_null_r2` only; it does **not**
natively fit a flexible-model ceiling (that is a materially larger unit of
work — quadratic expansion + ridge path + kNN/local-linear family with model
selection, plus its own numerical-stability test suite — declined for this
release, see `nullbench_release.md` §3). If you can afford it, also fit a
flexible model on the same metadata under the same split and report both;
the five values already computed this way are shipped in
`neuroforge.nullbench.harmonised` (sourced from `null_mechanism.json`, not
recomputed).
