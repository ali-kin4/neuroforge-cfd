"""Tests for the wall-distance measure.

It has to be the distance to the surface, not to the polyline's *vertices*. On a
body-fitted mesh the two differ by orders of magnitude exactly where it matters:
the surface here is sampled every ~0.01 chord and the first cell ring sits 4e-6
chord off the wall, so nearest-vertex overestimates by ~370x in the boundary
layer that every seeding strategy in this package is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.solver.warmstart import wall_distance


def test_distance_to_a_segment_not_to_its_endpoints():
    """The case that was wrong: a point beside a coarsely sampled wall."""
    wall = np.array([[0.0, 0.0], [1.0, 0.0]])          # one long segment
    point = np.array([[0.5, 1e-5, 0.0]])               # 1e-5 above its middle
    assert wall_distance(point, wall)[0] == pytest.approx(1e-5, rel=1e-6)


def test_refining_the_polyline_does_not_change_the_answer():
    """A converged measure must not depend on how finely the wall is sampled."""
    point = np.array([[0.5, 1e-4, 0.0]])
    got = [wall_distance(point, np.stack([np.linspace(0, 1, n), np.zeros(n)], axis=1))[0]
           for n in (2, 10, 200, 2000)]
    np.testing.assert_allclose(got, 1e-4, rtol=1e-6)


def test_a_point_beyond_the_ends_measures_to_the_endpoint():
    wall = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert wall_distance(np.array([[-3.0, 4.0, 0.0]]), wall)[0] == pytest.approx(5.0)
    assert wall_distance(np.array([[4.0, 4.0, 0.0]]), wall)[0] == pytest.approx(5.0)


def test_matches_the_analytic_distance_to_a_circle():
    theta = np.linspace(0.0, 2 * np.pi, 400)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    radii = np.array([1.5, 2.0, 5.0])
    pts = np.stack([radii, np.zeros(3), np.zeros(3)], axis=1)
    # Slightly under the exact value: a chord cuts inside the arc.
    got = wall_distance(pts, circle)
    np.testing.assert_allclose(got, radii - 1.0, rtol=2e-4)


def test_a_near_wall_point_is_not_reported_at_the_vertex_spacing():
    """The regression, stated as the numbers that were measured."""
    x = np.linspace(0.0, 1.0, 200)                     # ~0.01 chord spacing
    wall = np.stack([x, np.zeros_like(x)], axis=1)
    point = np.array([[0.505, 4e-6, 0.0]])             # first cell ring
    got = wall_distance(point, wall)[0]
    assert got == pytest.approx(4e-6, rel=1e-3)
    assert got < 1e-5                                  # nearest-vertex gave 1.4e-3


def test_chunking_does_not_change_the_result():
    rng = np.random.default_rng(0)
    wall = np.stack([np.linspace(0, 1, 50), np.zeros(50)], axis=1)
    pts = np.concatenate([rng.uniform(-1, 2, (500, 2)), np.zeros((500, 1))], axis=1)
    np.testing.assert_allclose(wall_distance(pts, wall, chunk=7),
                               wall_distance(pts, wall, chunk=10_000))


def test_a_degenerate_one_point_wall_still_answers():
    got = wall_distance(np.array([[3.0, 4.0, 0.0]]), np.array([[0.0, 0.0]]))
    assert got[0] == pytest.approx(5.0)
