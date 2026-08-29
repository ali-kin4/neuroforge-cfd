"""Tests for the warm-start scoring rules.

Each of these encodes a mistake that was actually made while measuring this
project, and that changed a sign or a magnitude rather than a decimal place.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.solver import scoring as sc


# --------------------------------------------------------------------------- #
# Censoring: an arm that never reaches the target is worse, not absent
# --------------------------------------------------------------------------- #


def test_dropping_a_failed_arm_would_flatter_it():
    """The bug this exists to prevent, stated as a comparison."""
    # Four cases where the arm was 20% faster, two where it never arrived at all
    # inside a 3000-iteration budget against a 600-iteration cold start.
    reached = [0.2, 0.2, 0.2, 0.2]
    censored = [1.0 - 3000 / 600] * 2          # -400% each
    scored = sc.bounded_saving(reached, censored)
    assert scored.saving == pytest.approx((4 * 0.2 + 2 * -4.0) / 6)
    assert scored.saving_reached_only == pytest.approx(0.2)
    assert scored.saving < scored.saving_reached_only


def test_a_saving_with_censored_cases_is_flagged_as_a_bound():
    scored = sc.bounded_saving([0.5], [-1.0])
    assert scored.is_bound is True
    assert str(scored).startswith("<")
    assert scored.n == 2 and scored.n_reached == 1 and scored.n_censored == 1


def test_a_saving_with_nothing_censored_is_not_a_bound():
    scored = sc.bounded_saving([0.5, 0.3], [])
    assert scored.is_bound is False
    assert scored.saving == pytest.approx(0.4)
    assert str(scored) == "+40.0% (2/2)"


def test_saving_spread_covers_the_censored_values_too():
    scored = sc.bounded_saving([0.5], [-4.0])
    assert scored.spread == (-4.0, 0.5)


def test_saving_with_no_cases_at_all():
    scored = sc.bounded_saving([], [])
    assert scored.saving is None and scored.n == 0
    assert str(scored) == "--"


# --------------------------------------------------------------------------- #
# The shared reference
# --------------------------------------------------------------------------- #


def test_shared_reference_is_the_median_not_one_arm():
    # cold, oracle, and two warm arms; the last has not converged.
    assert sc.shared_reference([0.0104, 0.0104, 0.0105, 0.0121]) == pytest.approx(0.01045)


def test_shared_reference_ignores_missing_and_nonfinite_values():
    assert sc.shared_reference([1.0, None, np.nan, 3.0]) == pytest.approx(2.0)
    assert sc.shared_reference([None, None]) is None


def test_reference_spread_is_the_worst_disagreement():
    finals = [0.0104, 0.0104, 0.0105, 0.0121]
    ref = sc.shared_reference(finals)
    # 0.0121 is 15.8% above the median of 0.01045.
    assert sc.reference_spread(finals, ref) == pytest.approx(0.15789, rel=1e-3)


def test_reference_spread_decides_whether_a_band_is_measurable():
    tight = [1.000, 1.0005, 0.9995]
    ref = sc.shared_reference(tight)
    spread = sc.reference_spread(tight, ref)
    assert spread <= sc.MAX_SPREAD_FRACTION * 0.002    # a 0.2% band is fine
    loose = [1.0, 1.0, 1.016]
    assert sc.reference_spread(loose, sc.shared_reference(loose)) > 0.002


def test_reference_spread_without_a_reference_is_nan():
    assert np.isnan(sc.reference_spread([1.0], None))
    assert np.isnan(sc.reference_spread([], 1.0))


# --------------------------------------------------------------------------- #
# Depth: a threshold on a flat curve is not a measurement
# --------------------------------------------------------------------------- #


def test_a_threshold_near_the_floor_is_not_readable():
    floor = 1.3e-5
    # The three thresholds that produced the withdrawn "+13% to +30%" headline.
    assert not sc.readable_depth(3e-5, floor)
    assert not sc.readable_depth(2e-5, floor)
    assert not sc.readable_depth(1.5e-5, floor)
    # And the one that was fine.
    assert sc.readable_depth(1e-3, floor)


def test_readable_depth_is_the_ratio_not_the_absolute_level():
    # 1e-5 is unreadable against a 1e-5 floor and fine against a 1e-7 one.
    assert not sc.readable_depth(1e-5, 1e-5)
    assert sc.readable_depth(1e-5, 1e-7)


def test_readable_depth_without_an_established_floor_does_not_object():
    assert sc.readable_depth(1e-5, float("nan"))
    assert sc.readable_depth(1e-5, 0.0)


# --- rule 4, per arm: a straggler must not condemn the arms that agree -------

def test_a_flat_tail_counts_as_settled():
    values = list(np.linspace(0.5, 0.01, 200)) + [0.01000, 0.010005, 0.01000] * 20
    assert sc.has_settled(values, tol=0.01)


def test_a_still_moving_trace_is_not_settled():
    assert not sc.has_settled(list(np.linspace(0.5, 0.01, 300)), tol=0.01)


def test_too_short_a_trace_is_not_settled():
    assert not sc.has_settled([0.01] * 5, tol=0.01)


def test_the_reference_ignores_an_arm_that_never_settled():
    """The measured case: three arms agree to 0.1%, one diverged to +3%."""
    finals = {"cold": 0.01041, "oracle": 0.01040, "nf_bl": 0.01042, "nf_mesh": 0.0107}
    ref, spread, unsettled = sc.settled_reference(
        finals, settled_arms=["cold", "oracle", "nf_bl"])
    assert ref == pytest.approx(0.01041, rel=1e-6)
    assert spread < 0.002                     # blunt rule gave 0.031
    assert unsettled == ["nf_mesh"]


def test_the_excluded_arm_is_named_not_silently_dropped():
    finals = {"a": 1.0, "b": 1.001, "bad": 2.0}
    _, _, unsettled = sc.settled_reference(finals, settled_arms=["a", "b"])
    assert "bad" in unsettled


def test_nothing_settled_falls_back_to_every_arm():
    finals = {"a": 1.0, "b": 1.2}
    ref, spread, unsettled = sc.settled_reference(finals, settled_arms=[])
    assert ref == pytest.approx(1.1)
    assert spread > 0.09 and unsettled == []
