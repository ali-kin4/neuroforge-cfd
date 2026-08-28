"""Tests for seeding a solve from a trained NeuroForge prediction.

The unit conversion and the normal orientation are the two places this can be
silently wrong: neither raises, and both produce a plausible-looking field that
is not what the model was trained to emit.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.solver import surrogate_seed as ss


# --------------------------------------------------------------------------- #
# Units: AirfRANS is dimensional, the solver is not
# --------------------------------------------------------------------------- #


def test_reynolds_maps_to_an_inlet_speed_the_model_has_seen():
    """AirfRANS fixes nu and varies the speed, so Re enters through |u_in|."""
    assert ss.dimensional_speed(3e6) == pytest.approx(46.8)
    assert ss.dimensional_speed(1e6) == pytest.approx(15.6)
    # Chord scales it: Re = U c / nu.
    assert ss.dimensional_speed(3e6, chord=2.0) == pytest.approx(23.4)


def test_the_speed_for_flight_reynolds_is_inside_the_training_range():
    # Training inlet speeds are centred on 61.6 with a standard deviation of
    # 17.8; a query more than about two sigma out is an extrapolation.
    sigma = (ss.dimensional_speed(3e6) - 61.6065) / 17.7573
    assert abs(sigma) < 1.0


# --------------------------------------------------------------------------- #
# Normals: AirfRANS's point outward
# --------------------------------------------------------------------------- #


def _circle(n=64, clockwise=False):
    t = np.linspace(0.0, 2 * np.pi, n)
    if clockwise:
        t = t[::-1]
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def test_normals_are_unit_length():
    n = ss.surface_normals(_circle())
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("clockwise", [False, True])
def test_normals_point_outward_whichever_way_the_polyline_winds(clockwise):
    """Rotating the tangent gives a sign that follows the winding.

    A caller building a body-fitted grid should not have to know which way its
    surface polyline runs, and feeding the model inward normals is feeding it the
    opposite of its training features.
    """
    surf = _circle(clockwise=clockwise)
    n = ss.surface_normals(surf)
    # On a unit circle about the origin the outward normal is the position.
    np.testing.assert_allclose(n, surf, atol=2e-2)


def test_normals_on_an_offset_body_still_point_outward():
    surf = _circle() + np.array([5.0, -3.0])
    n = ss.surface_normals(surf)
    radial = surf - surf.mean(axis=0)
    assert np.all(np.sum(n * radial, axis=1) > 0)


def test_normals_on_a_thin_section():
    """An airfoil is not a circle: thin, cusped at the trailing edge."""
    x = np.linspace(0.0, 1.0, 80)
    upper = np.stack([x, 0.06 * np.sqrt(np.clip(x, 0, 1)) * (1 - x)], axis=1)
    lower = np.stack([x[::-1], -0.06 * np.sqrt(np.clip(x[::-1], 0, 1)) * (1 - x[::-1])], axis=1)
    surf = np.concatenate([upper, lower])
    n = ss.surface_normals(surf)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, rtol=1e-6)
    # The upper surface's normals have a positive y component, the lower's negative.
    assert n[len(x) // 2, 1] > 0
    assert n[len(x) + len(x) // 2, 1] < 0


def test_predict_on_mesh_rejects_an_empty_checkpoint_list():
    with pytest.raises(ValueError, match="no checkpoints"):
        ss.predict_on_mesh([], np.zeros((4, 3)), np.zeros((4, 2)),
                           reynolds=3e6, aoa_deg=0.0)


def test_a_closed_polyline_is_differenced_periodically():
    """The two worst normals otherwise land exactly on the trailing edge."""
    surf = _circle(n=64)
    assert np.allclose(surf[0], surf[-1])          # closed, as an airfoil is
    n = ss.surface_normals(surf)
    # Every normal, endpoints included, matches the exact outward direction.
    np.testing.assert_allclose(n, surf, atol=5e-3)
    np.testing.assert_allclose(n[0], n[-1], atol=1e-12)


def test_an_open_polyline_still_works():
    surf = _circle(n=64)[:-8]                      # a broken ring
    n = ss.surface_normals(surf)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, rtol=1e-6)
