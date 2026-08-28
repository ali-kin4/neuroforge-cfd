"""Scoring rules for warm-start experiments.

An iteration saving looks like a simple quantity -- ``1 - warm / cold`` -- and it
is not. Measuring it on this project produced four mistakes, each of which
changed a sign or a magnitude rather than a decimal place, and each of which is
easy to make again. The rules that avoid them live here, tested, rather than in
whichever analysis script happened to need them:

1. **A threshold only measures a convergence rate while the residual is falling.**
   Below a few times the residual floor, an iteration count records where a flat
   curve crosses a line. :func:`readable_depth` says whether a threshold clears
   the floor by enough to be worth reading.
2. **An arm that never reaches the target must be counted, not dropped.** It is
   worse than one that reaches it late, so discarding it raises that arm's own
   mean -- the failing arm is rewarded for failing. :func:`bounded_saving` scores
   it with the arm's budget instead, which bounds it.
3. **All arms must be scored against one external reference.** Grading an arm
   against its own final value measures how it approached its own asymptote.
   :func:`shared_reference` takes the median across arms.
4. **That reference is only usable if the arms agree.** If they disagree about
   the answer by more than the band being measured, an arm can sit outside it
   forever and the metric reports a convergence failure that is really a budget
   failure. :func:`reference_spread` is the check.

None of this is specific to OpenFOAM, or to CFD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Saving",
    "bounded_saving",
    "shared_reference",
    "reference_spread",
    "readable_depth",
    "MIN_DEPTH_OVER_FLOOR",
    "MAX_SPREAD_FRACTION",
]

# A threshold this many times above the residual floor is far enough up the curve
# for the crossing iteration to mean something. Below it the curve has flattened
# and the crossing is set by noise: measured here, the same arm read +15%, +31%
# and +13% at 1.9x, 1.3x and 0.9x the floor.
MIN_DEPTH_OVER_FLOOR = 5.0

# The arms may disagree about the converged value by at most this fraction of the
# band being measured. At half a band, an arm can sit just outside the band for
# the whole run and score as never converging.
MAX_SPREAD_FRACTION = 0.5


@dataclass(frozen=True)
class Saving:
    """A mean iteration saving, and how much of it rests on a bound."""

    saving: float | None            # mean of 1 - arm/base, censored cases included
    saving_reached_only: float | None
    n_reached: int
    n_censored: int
    spread: tuple[float, float] | None

    @property
    def n(self) -> int:
        return self.n_reached + self.n_censored

    @property
    def is_bound(self) -> bool:
        """True when at least one case never reached the target.

        The reported saving is then an upper bound: the arm is *at least* this
        bad. Never present such a number as a measurement.
        """
        return self.n_censored > 0

    def __str__(self) -> str:
        if self.saving is None:
            return "--"
        return (f"{'<' if self.is_bound else ''}{100 * self.saving:+.1f}% "
                f"({self.n_reached}/{self.n})")


def bounded_saving(reached, censored) -> Saving:
    """Combine measured savings with bounds from arms that never got there.

    ``reached`` holds ``1 - arm/base`` for the cases where the arm met the
    target. ``censored`` holds the same expression evaluated at the arm's
    *budget* for the cases where it did not: the arm needed more than that many
    iterations, so the value is an upper bound on its saving.

    Passing only ``reached`` -- the natural thing to write -- is the bug this
    exists to prevent. On the measured data it turned -199.4% into -31.2%.
    """
    reached, censored = list(reached), list(censored)
    values = reached + censored
    return Saving(
        saving=float(np.mean(values)) if values else None,
        saving_reached_only=float(np.mean(reached)) if reached else None,
        n_reached=len(reached),
        n_censored=len(censored),
        spread=(float(min(values)), float(max(values))) if values else None,
    )


def shared_reference(finals) -> float | None:
    """The converged value every arm of a case should be scored against.

    ``finals`` is one final value per arm. They all solve the same steady problem
    on the same mesh, so they must land on the same answer and the median is the
    best estimate of it -- robust to a straggler, and external to every arm.

    Using one arm's own final instead makes that arm's score meaningless: it
    measures how the arm approached its own asymptote. Doing that to the *oracle*
    arm turns the experiment's control into an artifact, which is how a control
    that should read +73% read +1%.
    """
    values = [float(v) for v in finals if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def reference_spread(finals, reference: float | None) -> float:
    """Largest relative disagreement between the arms and their reference.

    Compare against the band being measured: see :func:`readable_depth`'s
    sibling rule, ``spread <= MAX_SPREAD_FRACTION * tolerance``. A wider spread
    means the budget was too short for the arms to agree, and any band tighter
    than the spread will report convergence failures that are budget failures.
    """
    values = [float(v) for v in finals if v is not None and np.isfinite(v)]
    if not values or not reference:
        return float("nan")
    return max(abs(v - reference) / abs(reference) for v in values)


def readable_depth(threshold: float, floor: float,
                   minimum: float = MIN_DEPTH_OVER_FLOOR) -> bool:
    """Is ``threshold`` far enough above the residual ``floor`` to be read?

    Below the minimum the residual has flattened and the iteration at which it
    crosses the threshold is noise, not a convergence rate.
    """
    if not np.isfinite(floor) or floor <= 0:
        return True   # no floor established: nothing to object to
    return threshold / floor >= minimum
