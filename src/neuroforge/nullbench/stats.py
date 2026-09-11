"""Metrics, bootstrap intervals, and the published-relative ratio.

**Naming, settled after review** (``docs/paper/review/naming_and_positioning.md``):
the protocol is **the metadata null** — a regression on a benchmark's own
published per-case parameters, no simulation output opened. Its primary,
headline statistic is **metadata-only R2**, written :math:`R^2_{meta}` and
emitted as ``metadata_null_r2`` (with a bootstrap CI), alongside
``metadata_null_mse``. This module previously called that primary quantity
a "covariate-null fraction" — retired for three reasons the positioning
review gives and this module accepts: R2 can be negative (this project's own
WindsorML interval reaches -0.143), so "fraction" is the wrong word for it;
"null" was already doing two jobs (the metadata null itself, and the
label-permutation null in ``harness.permutation_check``); and "floor" is
already load-bearing elsewhere in this repository (the residual floor),
so a name built from "covariate" + "floor" collides with it. The
*ratio* this module used to call the covariate-null fraction still exists,
demoted: :func:`published_relative_ratio` (``null / published``, on the
metric's own floor-0/ceiling-1 scale) is a **comparison aid, not the
headline**, precisely because it needs a competitor's published number and
inherits that number's precision — see the rounding-sensitivity discussion
below and ``docs/paper/review/nullbench_release.md`` Sec 1.5.

Floor / ceiling convention (unchanged):

* **R2-style** (``metric="r2"``): floor 0 (an intercept-only / constant
  predictor scores exactly 0 by the definition of R2), ceiling 1.
* **Rank-style** (``metric="spearman"``): floor 0 (a random ranking has
  expected Spearman correlation 0), ceiling 1.

:func:`published_relative_ratio` shares one convention with its retired
predecessor: **the denominator (the published value) must be above the
floor for the ratio to be defined.** If a published metric is at or below
its own floor (e.g. AirfRANS drag rank correlation, which is *negative* for
every published model in Bonnet et al. 2022 Table 3), no ratio against it
means anything — the null is reported to have won outright, and ``value``
is ``None`` with the flag ``published_at_or_below_floor``. The numerator is
not floored: a null that is itself at or below its own floor (metadata
carries no signal) reports a ratio at or below zero, flagged
``null_at_or_below_floor`` rather than silently clipped, and a null that
*exceeds* the published value reports a ratio above 1, flagged
``null_exceeds_published`` -- both are informative and neither is hidden.

**Rounding sensitivity.** The positioning review demonstrates that a
*differently defined* ratio it also considered and rejected as a headline —
null-normalised gain, ``G = (R2_model - R2_meta) / (1 - R2_meta)`` — swings
6-fold on DrivAerML (0.07 to 0.44) once DoMINO's published 0.98 is
propagated through its own 2-decimal rounding interval [0.975, 0.985],
because ``G``'s denominator (``1 - R2_meta = 0.0269``) sits close to zero.
:func:`published_relative_ratio`'s denominator is the published value
itself, not ``1 - R2_meta``, so the dangerous regime is different: a
published value reported to few decimals *and* close to the metric's own
floor. On the same DrivAerML DoMINO row this ratio swings only
0.988-0.998 (checked, not just argued -- see
``tests/test_nullbench.py::test_published_relative_ratio_rounding_sensitivity``
and ``nullbench_release.md`` Sec 1.6 for the worked table across all five
benchmarks). Pass ``published_precision`` (decimal digits) to
:func:`published_relative_ratio` to get that interval computed rather than
assumed; entries close to the floor at low precision are flagged
``rounding_sensitive`` rather than silently under-reported.

Every ratio ships with a bootstrap interval derived from the null's own
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

# Above this fractional swing (hi - lo, relative to the point ratio), a rounding
# interval is flagged rather than left for the reader to notice. Chosen, not
# derived: it is well below the 6x (proportional swing >> 1) case the positioning
# review demonstrates for null-normalised gain, and well above the ~1% swing this
# ratio shows on the DrivAerML DoMINO row it was checked against -- see the module
# docstring. A benchmark maintainer with a different tolerance can recompute the
# interval directly from `published_relative_ratio`'s returned fields.
ROUNDING_SENSITIVITY_THRESHOLD = 0.15


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


def mse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error -- scale-absolute, reported alongside R2 (never run
    through :func:`published_relative_ratio`'s floor/ceiling machinery, which
    is defined only for the two floor-0/ceiling-1 metric families above)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def spearman_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and labels. A constant
    prediction (e.g. the intercept-only model) yields undefined ranks and
    ``nan``, which callers must handle explicitly -- it is not silently
    mapped to the floor, because "undefined" and "floor" are different facts."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(spearmanr(y_pred, y_true).statistic)


# Deliberately excludes mse_score: METRIC_FNS backs published_relative_ratio's
# floor/ceiling machinery and the bootstrap CI loop, both defined only for the
# two floor-0/ceiling-1 families. MSE is reported by NullResult directly (see
# harness.py) rather than routed through this table, so `metric="mse"` cannot be
# passed into run_null and silently get a nonsense verdict direction.
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
def bootstrap_ci_raw(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """Case-level percentile bootstrap CI for an arbitrary ``fn(y_true, y_pred)``
    scorer. Shared implementation behind :func:`bootstrap_ci` (r2/spearman, via
    :data:`METRIC_FNS`) and behind the MSE bootstrap in ``harness.py`` (MSE is
    kept out of :data:`METRIC_FNS` deliberately -- see that table's comment).

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


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Metric,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """Case-level percentile bootstrap CI for ``metric(y_true, y_pred)``,
    ``metric in {"r2", "spearman"}``. See :func:`bootstrap_ci_raw`."""
    return bootstrap_ci_raw(y_true, y_pred, METRIC_FNS[metric],
                            n_boot=n_boot, seed=seed, alpha=alpha)


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
# published-relative ratio (demoted comparison aid; NOT the headline -- see
# module docstring. R2_meta / metadata_null_r2, computed in harness.py, is.)
# --------------------------------------------------------------------------- #
@dataclass
class PublishedRelativeRatio:
    """One ratio computation: the metadata null's share of one published number.

    ``value`` is ``None`` exactly when the published score does not exceed
    its own floor -- see the module docstring. ``flags`` never suppresses a
    number; it annotates one that is still reported. When
    ``published_precision`` is supplied, ``rounding_interval`` and the
    ``rounding_sensitive`` flag report how much ``value`` would move if the
    published figure were anywhere inside its own last-reported-digit
    rounding band -- see the module docstring's worked DrivAerML example.
    """

    metric: Metric
    floor: float
    ceiling: float
    null_point: float
    null_ci95: tuple[float, float]
    published: float | None
    published_std: float | None
    published_precision: int | None
    value: float | None
    value_ci95: tuple[float, float] | None
    rounding_interval: tuple[float, float] | None
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
            "published_precision_decimals": self.published_precision,
            "published_relative_ratio": self.value,
            "published_relative_ratio_ci95": list(self.value_ci95) if self.value_ci95 else None,
            "rounding_interval": list(self.rounding_interval) if self.rounding_interval else None,
            "flags": list(self.flags),
        }


def published_relative_ratio(
    metric: Metric,
    null_point: float,
    null_ci95: tuple[float, float],
    published: float | None,
    published_std: float | None = None,
    published_precision: int | None = None,
) -> PublishedRelativeRatio:
    """Compute one ratio row. See the module docstring for the floor/ceiling
    convention, the undefined-ratio rule, and the rounding-sensitivity check.

    The returned interval propagates only the null's own resampling
    uncertainty (``null_ci95 / published``); the published point is treated
    as fixed because this artifact does not have access to the raw
    predictions a published model's own bootstrap would need. Where a
    published standard deviation exists it is carried in ``published_std``
    for the reader, not folded into ``value_ci95``.

    ``published_precision``, if given (decimal digits the published value was
    reported to), additionally computes ``rounding_interval``: the ratio's
    range if the published figure is anywhere inside
    ``published +/- 0.5 * 10**-published_precision``. If that range's width
    relative to the point ratio exceeds :data:`ROUNDING_SENSITIVITY_THRESHOLD`,
    the flag ``rounding_sensitive`` is added -- the dangerous regime is a
    published value reported to few decimals *and* close to the metric's own
    floor, since that is where a small absolute rounding band is a large
    relative one.
    """
    floor = FLOOR[metric]
    ceiling = CEILING[metric]
    flags: list[str] = []

    if published is None:
        return PublishedRelativeRatio(metric, floor, ceiling, null_point, tuple(null_ci95),
                                      published, published_std, published_precision,
                                      None, None, None, ["published_not_reported"])
    if published <= floor:
        return PublishedRelativeRatio(metric, floor, ceiling, null_point, tuple(null_ci95),
                                      published, published_std, published_precision,
                                      None, None, None, ["published_at_or_below_floor"])

    value = null_point / published
    lo, hi = null_ci95[0] / published, null_ci95[1] / published
    value_ci = (min(lo, hi), max(lo, hi))

    if null_point <= floor:
        flags.append("null_at_or_below_floor")
    if value > 1.0:
        flags.append("null_exceeds_published")

    rounding_interval = None
    if published_precision is not None:
        half_ulp = 0.5 * 10 ** (-published_precision)
        p_lo, p_hi = published - half_ulp, published + half_ulp
        if p_lo > floor:
            r_a, r_b = null_point / p_lo, null_point / p_hi
            rounding_interval = (min(r_a, r_b), max(r_a, r_b))
            width = rounding_interval[1] - rounding_interval[0]
            if abs(value) > 0 and width / abs(value) > ROUNDING_SENSITIVITY_THRESHOLD:
                flags.append("rounding_sensitive")
        else:
            flags.append("rounding_interval_crosses_floor")

    return PublishedRelativeRatio(metric, floor, ceiling, null_point, tuple(null_ci95),
                                  published, published_std, published_precision,
                                  value, value_ci, rounding_interval, flags)
