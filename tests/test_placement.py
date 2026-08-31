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


@pytest.mark.skipif(not os.path.isfile(os.path.join("results", "seed_gradient.json")),
                    reason="needs the committed gradient diagnostic")
def test_predicts_the_measured_overestimate_on_the_repr3_cases():
    """The claim the paper makes, against six solves that already exist.

    Parameter-free: `u_tau` comes from each case's own converged wall gradient.
    The agreement is a systematic ~13% over-prediction with very little scatter,
    and both halves of that sentence are asserted -- a drift in either direction
    means the closed form no longer describes the measurement.
    """
    with open(os.path.join("results", "seed_gradient.json"), encoding="utf-8") as fh:
        data = json.load(fh)

    nu = 1.0 / data["re"]
    probe = data["heights"][0]
    ratios = []
    for row in data["rows"]:
        arm = row["arms"].get("fitted_256x64")
        if not arm:
            continue
        u_tau = pl.friction_velocity(row["converged_mean_gradient"], nu)
        predicted = pl.amplification(first_station=2.5e-4, cell_centre=probe,
                                     u_tau=u_tau, nu=nu)["factor"]
        measured = arm["mean_gradient"] / row["converged_mean_gradient"]
        ratios.append(predicted / measured)

    ratios = np.array(ratios)
    assert len(ratios) >= 5
    assert ratios.mean() == pytest.approx(1.13, abs=0.05)
    assert ratios.std() < 0.05
