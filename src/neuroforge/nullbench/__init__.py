"""NullBench: the covariate-null fraction for CFD-ML (and other tabular-label)
benchmarks.

Measures how much of a reported metric a regression on the benchmark's own
published per-case metadata already achieves, with no simulation output
opened. See ``docs/NULLBENCH.md`` for the one-page how-to and
``docs/paper/review/null_travels.md`` / ``nullbench_release.md`` for the
finding this package operationalises.

Quick start
-----------
>>> from neuroforge.nullbench import load_table, run_null, PublishedEntry
>>> table = load_table("cases.csv", id_col="id", label_col="cd",
...                     split_col="split")  # or split_col=None for K-fold
>>> result = run_null(table, metric="r2", published=[
...     PublishedEntry("SomeGNN", value=0.85, std=0.02, source="Table 3")])
>>> result.out_of_sample, result.ci95, result.protocol
"""

from __future__ import annotations

from .fit import fit_predict, kfold_oos, run_kfold, run_official_split
from .harness import NullResult, PublishedEntry, compare, permutation_check, run_null
from .io import Table, design_matrix, intercept_only, load_table
from .stats import (
    CEILING,
    FLOOR,
    CovariateNullFraction,
    bootstrap_ci,
    covariate_null_fraction,
    r2_score,
    spearman_score,
    verdict_bound_lower,
    verdict_higher,
    verdict_lower,
)

__all__ = [
    "Table",
    "load_table",
    "design_matrix",
    "intercept_only",
    "fit_predict",
    "kfold_oos",
    "run_official_split",
    "run_kfold",
    "NullResult",
    "PublishedEntry",
    "run_null",
    "compare",
    "permutation_check",
    "CovariateNullFraction",
    "covariate_null_fraction",
    "bootstrap_ci",
    "r2_score",
    "spearman_score",
    "verdict_higher",
    "verdict_lower",
    "verdict_bound_lower",
    "FLOOR",
    "CEILING",
]
