"""The pre-flight check, and the measurement it has to keep reproducing.

The test that matters most is
:func:`test_predicts_the_measured_overestimate_on_the_repr3_cases`: it holds the
closed form against six converged solves already on disk. If a change to the
law-of-the-wall blending moves that agreement, the criterion the paper sells has
moved, and it should not move silently.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from neuroforge.solver import placement as pl


def test_sublayer_is_linear():
    """u+ = y+ below y+ = 5, exactly -- this is where a resolved mesh lives."""
    y = np.array([0.1, 1.0, 3.0, 5.0])
    assert np.allclose(pl.u_plus(y), y)


def test_log_layer_matches_the_standard_form():
    for y in (30.0, 100.0, 1000.0):
        assert pl.u_plus(y) == pytest.approx(np.log(y) / pl.KAPPA + pl.B_LOG)


def test_u_plus_is_continuous_across_the_buffer_blend():
    """A discontinuity at y+ = 5 or 30 would put a step in every reported factor."""
    for edge in (5.0, 30.0):
        below, above = pl.u_plus(edge * 0.999), pl.u_plus(edge * 1.001)
        assert below == pytest.approx(above, rel=2e-3)


def test_u_plus_is_monotone():
    y = np.geomspace(1e-3, 1e4, 400)
    assert np.all(np.diff(pl.u_plus(y)) > 0)


def test_a_station_inside_the_first_cell_loses_nothing():
    """The criterion's pass condition: no amplification, and it says so."""
    got = pl.amplification(first_station=4e-6, cell_centre=5e-6,
                           u_tau=0.0477, nu=3.333e-7)
    assert got["factor"] == 1.0
    assert got["regime"] == "resolved"


def test_friction_velocity_inverts_the_wall_gradient():
    nu, u_tau = 3.333e-7, 0.0477
    gradient = u_tau ** 2 / nu
    assert pl.friction_velocity(gradient, nu) == pytest.approx(u_tau)


def test_more_values_barely_help_but_placement_does():
    """The paper's claim, as arithmetic.

    Refining a uniform raster 128 -> 512 spends 16x the values and moves the
    predicted damage by a few per cent, because u+ grows logarithmically.
    Moving the first station inside the first cell removes it entirely.
    """
    kw = dict(cell_centre=5e-6, u_tau=0.0477, nu=3.333e-7)
    coarse = pl.amplification(first_station=pl.uniform_stations(3.0, 128)[0], **kw)
    finer = pl.amplification(first_station=pl.uniform_stations(3.0, 512)[0], **kw)
    placed = pl.amplification(first_station=4e-6, **kw)

    assert finer["factor"] > 0.5 * coarse["factor"]   # 16x the values, same order
    assert placed["factor"] == 1.0                    # a grading change, not a budget


def test_half_the_budget_with_correct_placement_beats_double_with_wrong():
    """The discriminating contrast `placement_probe.py` runs, checked on paper first."""
    kw = dict(cell_centre=5e-6, u_tau=0.0477, nu=3.333e-7)
    wrong_but_big = pl.preflight(
        stations=pl.geometric_stations(2.5e-4, 1.0, 64), **kw)
    right_but_small = pl.preflight(
        stations=pl.geometric_stations(5.0e-6, 1.0, 32), **kw)

    assert wrong_but_big["n_stations"] == 2 * right_but_small["n_stations"]
    assert wrong_but_big["stations_inside_first_cell"] == 0
    assert right_but_small["stations_inside_first_cell"] >= 1
    assert right_but_small["factor"] < wrong_but_big["factor"] / 10


def test_saturated_regime_is_flagged_when_the_station_leaves_the_layer():
    got = pl.amplification(first_station=0.0118, cell_centre=5e-6,
                           u_tau=0.0477, nu=3.333e-7, delta=0.0187)
    assert got["regime"] == "wall_law"          # 0.0118 is still inside delta
    outside = pl.amplification(first_station=0.05, cell_centre=5e-6,
                               u_tau=0.0477, nu=3.333e-7, delta=0.0187)
    assert outside["regime"] == "saturated"


@pytest.mark.skipif(
    not os.path.isfile(os.path.join("results", "closed_form_validation.json")),
    reason="needs the committed closed-form validation")
def test_the_closed_form_is_an_upper_bound_that_tracks_the_measurement():
    """What the closed form actually does, measured against the *fixed* projection.

    An earlier version of this test pinned agreement at 1.13 +- 0.02. That number
    was measured against a ``clustered_seed`` that took its wall-normal
    coordinate from ``surface_coords`` -- the nearest *vertex* -- and so
    mis-placed every near-wall cell by three orders of magnitude. With that bug
    fixed, the expression **over-predicts in every row**, by 1.3x to 2.6x across a
    fifty-fold range of first-station heights.

    That is a weaker claim than the one it replaces and it is the true one. What
    the expression provides is a *parameter-free upper bound* on the damage,
    correct in direction and ordering, never optimistic. Both halves are asserted:
    it must never under-predict, and it must stay within a small factor.

    The bound is claimed over the **non-degenerate** rows only. Where fewer than
    two mesh rings lie below the representation's first station, the round trip
    is a structural no-op -- ``clustered_seed`` populates the station from the
    first ring by nearest-neighbour donor and maps it straight back -- so the
    measurement cannot exercise the clipping the closed form models. Those rows
    are labelled by ``scripts/validate_closed_form.py`` from the mesh alone,
    before any measurement, and are excluded here too.
    """
    with open(os.path.join("results", "closed_form_validation.json"), encoding="utf-8") as fh:
        data = json.load(fh)

    rows = data["rows"]
    assert len(rows) >= 4
    live = [r for r in rows if not r["degenerate"]]
    assert len(live) >= 3, "the bound needs rows in which the mechanism is active"
    ratios = np.array([r["ratio"] for r in live])

    # Never optimistic: a bound that under-predicts is not a bound.
    assert ratios.min() >= 1.0
    # And not vacuous.
    assert ratios.max() <= 3.0

    # The degenerate rows must be exactly the ones the mesh criterion picks out,
    # and they must in fact be no-ops -- that is what makes them uninformative.
    for row in rows:
        if row["degenerate"]:
            assert row["rings_below_station"] < data["min_rings_below_station"]
            assert abs(row["measured"] - 1.0) < 0.05

    # Monotone in the right direction: a station further out does more damage.
    # Over the live rows only -- the no-op rows all sit on 1.0 and their order
    # among themselves is numerical noise (they differ in the sixth decimal).
    measured = [r["measured"] for r in sorted(live, key=lambda r: r["first"])]
    assert measured == sorted(measured), "damage must grow with first-station height"
    assert min(measured) > 1.0 + 1e-3, "a live row must show real damage"


# --------------------------------------------------------------------------- #
# The repair: inverting the damage the closed form predicts
# --------------------------------------------------------------------------- #

NU, U_TAU = 3.333e-7, 0.0477


def test_u_tau_inversion_round_trips():
    """Recovering u_tau from a velocity at one height must return what made it."""
    for height in (2.5e-4, 1e-3, 5e-3):
        speed = U_TAU * pl.u_plus(height * U_TAU / NU)
        assert float(pl.invert_u_tau(speed, height, NU)) == pytest.approx(U_TAU, rel=1e-4)


def test_inversion_is_vectorised_over_stations():
    speeds = U_TAU * pl.u_plus(np.full(7, 2.5e-4) * U_TAU / NU)
    got = pl.invert_u_tau(speeds, 2.5e-4, NU)
    assert got.shape == (7,)
    assert np.allclose(got, U_TAU, rtol=1e-4)


def test_the_repair_restores_the_first_cell_gradient():
    """The point of the whole exercise, on a profile that obeys the law exactly."""
    d = np.array([5e-6, 1e-5, 5e-5, 2.5e-4])
    station = U_TAU * pl.u_plus(2.5e-4 * U_TAU / NU)
    # What a projection leaves behind: the station's value, everywhere below it.
    projected = (np.full(4, station), np.zeros(4), np.zeros(4), np.full(4, 1e-5))
    fixed, report = pl.wall_law_repair(projected, d, first_station=2.5e-4, nu=NU)

    truth = U_TAU * pl.u_plus(d * U_TAU / NU)
    assert np.allclose(fixed[0][:3], truth[:3], rtol=1e-3)
    assert report["repaired_cells"] == 3          # the cell *at* the station is untouched


def test_the_repair_leaves_cells_above_the_station_alone():
    d = np.array([1e-3, 5e-3])
    projected = (np.full(2, 0.6), np.zeros(2), np.zeros(2), np.full(2, 1e-5))
    fixed, report = pl.wall_law_repair(projected, d, first_station=2.5e-4, nu=NU)
    assert report["repaired_cells"] == 0
    assert np.allclose(fixed[0], projected[0])


def test_the_repair_preserves_velocity_direction():
    """Only magnitude is rescaled -- no surface-tangent geometry is involved."""
    d = np.array([5e-6, 1e-5])
    u0, v0 = np.array([0.3, 0.3]), np.array([0.4, 0.4])
    fixed, _ = pl.wall_law_repair((u0, v0, np.zeros(2), np.full(2, 1e-5)),
                                  d, first_station=2.5e-4, nu=NU)
    before = u0 / np.hypot(u0, v0)
    after = fixed[0] / np.hypot(fixed[0], fixed[1])
    assert np.allclose(before, after)


def test_the_repair_sends_nut_and_velocity_to_zero_at_the_wall():
    """SA requires it, and a repair that violated it would be worse than none."""
    d = np.array([1e-9, 1e-7, 5e-6])
    projected = (np.full(3, 0.5), np.zeros(3), np.zeros(3), np.full(3, 1e-3))
    fixed, _ = pl.wall_law_repair(projected, d, first_station=2.5e-4, nu=NU)
    assert fixed[0][0] < fixed[0][-1]
    assert fixed[3][0] < fixed[3][-1]
    assert (fixed[3] >= 0).all()


def test_the_repair_passes_pressure_through_untouched():
    """Pressure is constant across a boundary layer; touching it would be wrong."""
    d = np.array([5e-6, 1e-5])
    p = np.array([-0.3, -0.29])
    fixed, _ = pl.wall_law_repair((np.full(2, 0.5), np.zeros(2), p, np.full(2, 1e-5)),
                                  d, first_station=2.5e-4, nu=NU)
    assert np.allclose(fixed[2], p)
