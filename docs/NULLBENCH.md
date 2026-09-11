# NullBench — report the covariate-null fraction

For a benchmark maintainer or a referee. Full definition, API and the five
worked examples: `docs/paper/review/nullbench_release.md`. Finding this
operationalises: `docs/paper/review/null_travels.md`.

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
  `{"name": ..., "value": ..., "std": ..., "source": ...}` — the entries you
  want the null checked against. Omit it to just get the null and its
  interval.
* `--permute-check` refits on label-shuffled training data; the printed
  `permutation_check` value should land at or below the metric's floor (0).
  If it doesn't, something in your feature construction is leaking the label.

## What you get back

```json
{
  "metric": "r2",
  "protocol": "official_split",
  "out_of_sample": 0.973,
  "ci95": [0.958, 0.983],
  "comparisons": [
    {"model": "YourModel", "published": 0.98, "verdict": "straddles",
     "covariate_null_fraction": {"covariate_null_fraction": 0.993, "flags": []}}
  ]
}
```

Report **out_of_sample**, **ci95**, **protocol**, and the
**covariate_null_fraction** for your headline model, in the table you already
publish. One sentence is enough:

> A regression on the [N] published case parameters alone (no field data)
> reaches [metric] = X.XX [lo, hi] under [protocol], a covariate-null
> fraction of Y.YY against our best model's Z.ZZ.

If `covariate_null_fraction` is `null`, check `flags`:
`published_at_or_below_floor` means your own reported number does not clear
random/constant prediction, so no fraction is meaningful — report that fact
plainly, it is the more important finding.

## Ship it as a worked example

Contribute your benchmark's CSV + config to
`src/neuroforge/nullbench/benchmarks.py` (see the five existing entries) so
future authors on your benchmark never have to run the download-then-fit
pipeline again.

## Definition, thresholds, and what NOT to conclude from one number

See `docs/paper/review/nullbench_release.md` §1. In short: floor = 0
(intercept-only / random-rank), ceiling = 1, CNF = null / published on that
scale, undefined when the published value doesn't clear its own floor. Five
benchmarks give values from 0.10 to beyond 1.0 (the null exceeding published)
— report the number, do not round it into a fixed pass/fail bar; there is
not yet enough cross-benchmark evidence to defend one.

**One number is a start, not the whole picture.** A companion analysis in
this repository (`docs/paper/review/null_mechanism.md`) argues a *linear*
null alone can conflate two different failures — a model-class limitation
versus a genuine information deficit — that look identical as one number and
are not. NullBench v1 reports the linear CNF only; if you can afford it,
also fit a flexible model (quadratic + interactions, or a small ensemble) on
the same metadata under the same split and report both.
