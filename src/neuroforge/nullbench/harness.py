"""Orchestrates one metadata-null run: table in, R2_meta (+ MSE) + CI + ratio out.

This is the function a CLI, a benchmark config, or a notebook all funnel
through, so the split-selection logic (official split vs K-fold) and the
in-sample/out-of-sample bookkeeping exist in exactly one place.

Naming (``docs/paper/review/naming_and_positioning.md``): the protocol run
here is **the metadata null**; its headline output is **metadata-only R2**
(``R2_meta``), emitted below as ``metadata_null_r2`` (or
``metadata_null_spearman`` when ``metric="spearman"``) with a bootstrap CI,
alongside ``metadata_null_mse``. The demoted comparison aid formerly called
the "covariate-null fraction" is now :func:`~neuroforge.nullbench.stats.published_relative_ratio`,
emitted per published entry as ``published_relative_ratio`` -- see
``stats.py``'s module docstring for why it is not the headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fit, stats
from .io import Table, design_matrix
from .stats import Metric


@dataclass
class PublishedEntry:
    """One row a benchmark maintainer or referee wants to check against the null.

    ``is_bound=True`` marks ``value`` as a BOUND rather than a point estimate
    (e.g. a paper stating only "MSE < 0.00028"). ``bound_kind`` then selects
    which mirrored adjudicability rule applies: ``"upper"`` for a bound on a
    lower-is-better quantity (MSE-style: clears only if the null is
    decisively worse than the bound), ``"lower"`` for a bound on a
    higher-is-better quantity already expressed in the run's own metric
    (e.g. an MSE bound pre-converted to an implied R2 floor: clears only if
    the null is decisively below the floor). A bound entry never receives a
    published-relative ratio -- there is no point estimate to divide by.

    ``precision``, if given (decimal digits the published ``value`` was
    reported to), drives the rounding-sensitivity check on the ratio -- see
    ``stats.published_relative_ratio``.
    """

    name: str
    value: float | None
    std: float | None = None
    source: str = ""
    note: str = ""
    is_bound: bool = False
    bound_kind: str = "upper"  # "upper" | "lower", only meaningful if is_bound
    precision: int | None = None  # decimal digits `value` was reported to


@dataclass
class NullResult:
    metric: Metric
    protocol: str            # "official_split" | "kfold_oos_k{k}"
    n_fit: int
    n_score: int
    out_of_sample: float
    in_sample: float
    in_sample_inflation: float
    ci95: tuple[float, float]
    n_boot_finite: int
    mse_out_of_sample: float
    mse_ci95: tuple[float, float]
    feature_names: list[str]
    source: str
    comparisons: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        metadata_null_key = f"metadata_null_{self.metric}"
        return {
            "metric": self.metric,
            "protocol": self.protocol,
            "n_fit": self.n_fit,
            "n_score": self.n_score,
            # generic (metric-tagged via the "metric" field above) -- used internally
            # and by every existing consumer of this record
            "out_of_sample": self.out_of_sample,
            "in_sample": self.in_sample,
            "in_sample_inflation": self.in_sample_inflation,
            "ci95": list(self.ci95),
            "n_boot_finite": self.n_boot_finite,
            # the settled headline names (naming_and_positioning.md): same numbers,
            # spelled out so a reader of the JSON does not have to cross-reference
            # the "metric" field to know what "out_of_sample" means
            metadata_null_key: self.out_of_sample,
            f"{metadata_null_key}_ci95": list(self.ci95),
            "metadata_null_mse": self.mse_out_of_sample,
            "metadata_null_mse_ci95": list(self.mse_ci95),
            "feature_names": list(self.feature_names),
            "source": self.source,
            "comparisons": self.comparisons,
        }


def run_null(
    table: Table,
    metric: Metric = "r2",
    train_values: tuple[str, ...] = ("train",),
    test_values: tuple[str, ...] = ("test",),
    folds: int = 10,
    n_boot: int = 10000,
    seed: int = 0,
    published: list[PublishedEntry] | None = None,
) -> NullResult:
    """Run the metadata null on ``table`` and score every entry in ``published``.

    Protocol selection is automatic and always reported: if ``table`` carries
    a split column, cases tagged with any of ``train_values`` are fit on and
    cases tagged with any of ``test_values`` are scored (``protocol =
    "official_split"``); otherwise the null is refit by ``folds``-fold
    out-of-sample cross-validation (``protocol = "kfold_oos_k{folds}"``) and
    every table row is scored, each by a fold that excluded it.
    """
    Xd = design_matrix(table.X)

    if table.has_split():
        train_mask = table.split_mask(train_values)
        test_mask = table.split_mask(test_values)
        if not train_mask.any():
            raise ValueError(f"no rows tagged with train_values={train_values!r}")
        if not test_mask.any():
            raise ValueError(f"no rows tagged with test_values={test_values!r}")
        sr = fit.run_official_split(Xd[train_mask], table.y[train_mask],
                                     Xd[test_mask], table.y[test_mask])
    else:
        sr = fit.run_kfold(Xd, table.y, k=folds, seed=seed)

    oos_point = stats.score(metric, sr.y_true, sr.y_pred_oos)
    in_point = stats.score(metric, sr.y_true_in_sample, sr.y_pred_in_sample)
    lo, hi, n_finite = stats.bootstrap_ci(sr.y_true, sr.y_pred_oos, metric,
                                          n_boot=n_boot, seed=seed)

    mse_point = stats.mse_score(sr.y_true, sr.y_pred_oos)
    mse_lo, mse_hi, _ = stats.bootstrap_ci_raw(sr.y_true, sr.y_pred_oos, stats.mse_score,
                                               n_boot=n_boot, seed=seed)

    result = NullResult(
        metric=metric,
        protocol=sr.protocol,
        n_fit=sr.n_fit,
        n_score=sr.n_score,
        out_of_sample=oos_point,
        in_sample=in_point,
        in_sample_inflation=in_point - oos_point,
        ci95=(lo, hi),
        n_boot_finite=n_finite,
        mse_out_of_sample=mse_point,
        mse_ci95=(mse_lo, mse_hi),
        feature_names=table.feature_names,
        source=table.source,
    )

    if published:
        result.comparisons = [compare(result, entry) for entry in published]
    return result


def compare(result: NullResult, entry: PublishedEntry) -> dict:
    """One published-vs-null row: verdict + the published-relative ratio (a
    demoted comparison aid -- see stats.py's module docstring; the headline
    is ``result``'s own ``metadata_null_{metric}``, not this ratio)."""
    lo, hi = result.ci95
    if entry.is_bound:
        bound_fn = (stats.verdict_bound_lower if entry.bound_kind == "upper"
                    else stats.verdict_bound_higher)
        verdict = bound_fn(entry.value, result.out_of_sample, lo, hi)
        ratio = None
    else:
        verdict_fn = stats.VERDICT_FNS[result.metric]
        verdict = verdict_fn(entry.value, result.out_of_sample, lo, hi)
        ratio = stats.published_relative_ratio(result.metric, result.out_of_sample,
                                               (lo, hi), entry.value, entry.std,
                                               entry.precision)
    return {
        "model": entry.name,
        "published": entry.value,
        "published_std": entry.std,
        "published_source": entry.source,
        "published_note": entry.note,
        "is_bound": entry.is_bound,
        "verdict": verdict,
        "published_relative_ratio": ratio.to_dict() if ratio is not None else None,
    }


def permutation_check(
    table: Table,
    metric: Metric = "r2",
    train_values: tuple[str, ...] = ("train",),
    test_values: tuple[str, ...] = ("test",),
    folds: int = 10,
    seed: int = 1,
) -> float:
    """Refit with the TRAINING labels shuffled and return the out-of-sample
    score. Should land at or below the metric floor; anything else means the
    fitting machinery is manufacturing signal rather than reading it off the
    metadata. Test-set labels are never touched.

    Named ``permutation_check`` (a diagnostic run against the label-permutation
    null), distinct from the metadata null this module otherwise computes --
    see stats.py's module docstring on why the two must stay terminologically
    separate."""
    Xd = design_matrix(table.X)
    rng = np.random.default_rng(seed)

    if table.has_split():
        train_mask = table.split_mask(train_values)
        test_mask = table.split_mask(test_values)
        y_perm = table.y[train_mask][rng.permutation(train_mask.sum())]
        pred = fit.fit_predict(Xd[train_mask], y_perm, Xd[test_mask])
        return stats.score(metric, table.y[test_mask], pred)

    y_perm = table.y[rng.permutation(len(table.y))]
    oos = fit.kfold_oos(Xd, y_perm, k=folds, seed=seed)
    return stats.score(metric, table.y, oos)
