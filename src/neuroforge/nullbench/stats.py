"""Metrics, bootstrap intervals, and the covariate-null fraction.

The **covariate-null fraction** (CNF) is the share of a reported metric's value
that a regression on the benchmark's own published per-case metadata already
achieves, with no simulation output opened. It is defined per metric family:

* **R2-style** (``metric="r2"``): floor 0 (an intercept-only / constant
  predictor scores exactly 0 by the definition of R2), ceiling 1.
  ``CNF = null_R2 / published_R2``.
* **Rank-style** (``metric="spearman"``): floor 0 (a random ranking has
  expected Spearman correlation 0), ceiling 1.
  ``CNF = null_rho / published_rho``.

Both branches share one convention: **the denominator must be above the
floor for the fraction to be defined.** If a published metric is at or below
its own floor (e.g. AirfRANS drag rank correlation, which is *negative* for
every published model in Bonnet et al. 2022 Table 3), no fraction of it can
be attributed to anything — the null is reported to have won outright, and
``value`` is ``None`` with the flag ``published_at_or_below_floor``. The
numerator is not floored: a null that is itself at or below its own floor
(metadata carries no signal) reports a fraction at or below zero, flagged
``null_at_or_below_floor`` rather than silently clipped, and a null that
*exceeds* the published value reports a fraction above 1, flagged
``null_exceeds_published`` -- both are informative and neither is hidden.

Every CNF ships with a bootstrap interval derived from the null's own
case-level percentile bootstrap (never a bare point estimate). Where the
published entry itself carries a reported standard deviation, it is carried
alongside for the reader rather than combined into the interval -- combining
a resampled distribution with an assumed-Gaussian one from a different study
would manufacture precision this artifact does not have.

The two verdict rules below (``verdict_higher`` / ``verdict_lower``) restate,
independently, the pre-registered decision rule first committed in
``scripts/covariate_null.py`` (git commit ``8928087``, before any covariate-
null number existed) and extended for lower-is-better metrics in
``scripts/covariate_null_crossbench.py``. They are re-implemented here rather
than imported so this package has no dependency on the top-level ``scripts/``
tree (which is not installed with the package); ``tests/test_nullbench.py``
cross-checks both implementations against the ``scripts`` originals on a grid
of cases to guarantee they cannot silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import spearmanr

Metric = Literal["r2", "spearman"]

FLOOR: dict[str, float] = {"r2": 0.0, "spearman": 0.0}
CEILING: dict[str, float] = {"r2": 1.0, "spearman": 1.0}


# --------------------------------------------------------------------------- #
# point metrics
# --------------------------------------------------------------------------- #
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sklearn convention: ``1 - SSE / SST``, SST about the mean of ``y_true``
    on the set being scored (an intercept-only / constant predictor scores
    exactly 0 by construction, which is the R2 floor)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def spearman_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and labels. A constant
    prediction (e.g. the intercept-only model) yields undefined ranks and
    ``nan``, which callers must handle explicitly -- it is not silently
    mapped to the floor, because "undefined" and "floor" are different facts."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(spearmanr(y_pred, y_true).statistic)


METRIC_FNS = {"r2": r2_score, "spearman": spearman_score}


def score(metric: Metric, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        fn = METRIC_FNS[metric]
    except KeyError as exc:
        raise ValueError(f"unknown metric {metric!r}; expected one of {list(METRIC_FNS)}") from exc
    return fn(y_true, y_pred)


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Metric,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """Case-level percentile bootstrap CI for ``metric(y_true, y_pred)``.

    Every resample draws cases with replacement (never cells or predictions
    independently), matching ``scripts/covariate_null.py``'s ``boot_ci``.
    Non-finite draws (e.g. a degenerate resample with all-equal ranks under
    Spearman) are dropped and the surviving count is returned so silent
    shrinkage of the interval is visible.

    Returns ``(lo, hi, n_finite)``; ``(nan, nan, 0)`` if every draw was
    non-finite.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fn = METRIC_FNS[metric]
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = fn(y_true[idx], y_pred[idx])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi, int(len(vals))


# --------------------------------------------------------------------------- #
# pre-registered verdict rules (see module docstring: re-implemented, not
# imported, and cross-checked by test against scripts/covariate_null.py)
# --------------------------------------------------------------------------- #
def verdict_higher(published: float | None, null_point: float, ci_lo: float, ci_hi: float) -> str:
    """Higher-is-better rule (R2, Spearman). ``BELOW THE NULL`` iff the
    published mean is below the null point estimate AND the null's 95% CI
    excludes it. Otherwise ``straddles`` (CI covers it) or ``clears``."""
    if published is None:
        return "not reported"
    if published < null_point and not (ci_lo <= published <= ci_hi):
        return "BELOW THE NULL"
    if ci_lo <= published <= ci_hi:
        return "straddles"
    return "clears"


def verdict_lower(published: float | None, null_point: float, ci_lo: float, ci_hi: float) -> str:
    """Lower-is-better mirror (MSE). ``BELOW THE NULL`` iff the published
    mean is above the null point estimate AND the null's 95% CI excludes it."""
    if published is None:
        return "not reported"
    if published > null_point and not (ci_lo <= published <= ci_hi):
        return "BELOW THE NULL"
    if ci_lo <= published <= ci_hi:
        return "straddles"
    return "clears"


def verdict_bound_lower(bound: float | None, null_point: float, ci_lo: float, ci_hi: float) -> str:
    """Adjudicability rule for a published upper BOUND on a lower-is-better
    metric (e.g. "MSE < 0.00028"). A bound only adjudicates in the direction
    where the null loses: if the null's whole interval sits worse than the
    bound, the published entry clears outright. Otherwise the true published
    value could be arbitrarily far inside the bound, so the comparison is
    recorded as not adjudicable rather than guessed at."""
    if bound is None:
        return "not reported"
    if null_point > bound and not (ci_lo <= bound <= ci_hi):
        return "clears (published bound beats the null outright)"
    return "bound_only_not_adjudicable"


def verdict_bound_higher(bound: float | None, null_point: float, ci_lo: float, ci_hi: float) -> str:
    """Mirror of :func:`verdict_bound_lower` for a published LOWER bound on a
    higher-is-better metric (e.g. an MSE bound converted to an implied R2
    floor: "published R2 is at least this"). Adjudicable only where the null
    loses: if the null's whole interval sits below the bound, the published
    entry clears outright; otherwise the true published value could be
    arbitrarily far above the bound, so it is not adjudicable."""
    if bound is None:
        return "not reported"
    if null_point < bound and not (ci_lo <= bound <= ci_hi):
        return "clears (published bound beats the null outright)"
    return "bound_only_not_adjudicable"


VERDICT_FNS = {"r2": verdict_higher, "spearman": verdict_higher}


# --------------------------------------------------------------------------- #
# covariate-null fraction
# --------------------------------------------------------------------------- #
@dataclass
class CovariateNullFraction:
    """One CNF computation: the null's share of one published number.

    ``value`` is ``None`` exactly when the published score does not exceed
    its own floor -- see the module docstring. ``flags`` never suppresses a
    number; it annotates one that is still reported.
    """

    metric: Metric
    floor: float
    ceiling: float
    null_point: float
    null_ci95: tuple[float, float]
    published: float | None
    published_std: float | None
    value: float | None
    value_ci95: tuple[float, float] | None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "null_point": self.null_point,
            "null_ci95": list(self.null_ci95),
            "published": self.published,
            "published_std": self.published_std,
            "covariate_null_fraction": self.value,
            "covariate_null_fraction_ci95": list(self.value_ci95) if self.value_ci95 else None,
            "flags": list(self.flags),
        }


def covariate_null_fraction(
    metric: Metric,
    null_point: float,
    null_ci95: tuple[float, float],
    published: float | None,
    published_std: float | None = None,
) -> CovariateNullFraction:
    """Compute one CNF row. See the module docstring for the floor/ceiling
    convention and the undefined-fraction rule.

    The returned interval propagates only the null's own resampling
    uncertainty (``null_ci95 / published``); the published point is treated
    as fixed because this artifact does not have access to the raw
    predictions a published model's own bootstrap would need. Where a
    published standard deviation exists it is carried in ``published_std``
    for the reader, not folded into ``value_ci95``.
    """
    floor = FLOOR[metric]
    ceiling = CEILING[metric]
    flags: list[str] = []

    if published is None:
        return CovariateNullFraction(metric, floor, ceiling, null_point, tuple(null_ci95),
                                      published, published_std, None, None,
                                      ["published_not_reported"])
    if published <= floor:
        return CovariateNullFraction(metric, floor, ceiling, null_point, tuple(null_ci95),
                                      published, published_std, None, None,
                                      ["published_at_or_below_floor"])

    value = null_point / published
    lo, hi = null_ci95[0] / published, null_ci95[1] / published
    value_ci = (min(lo, hi), max(lo, hi))

    if null_point <= floor:
        flags.append("null_at_or_below_floor")
    if value > 1.0:
        flags.append("null_exceeds_published")

    return CovariateNullFraction(metric, floor, ceiling, null_point, tuple(null_ci95),
                                  published, published_std, value, value_ci, flags)
