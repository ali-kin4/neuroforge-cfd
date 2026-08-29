"""Tests for the wall-gradient diagnostic.

The measurement it exists to make: a seed's error in ``du_t/dy`` at the first
cell, which is what viscous drag integrates. Two properties have to hold for the
number to mean anything -- the sampling must walk *away* from the wall rather
than along it (the near-wall cells are ~2500x wider than they are tall), and the
roughness measure must be blind to a uniform scale error, so that a seed which
is smoothly wrong is not reported as ragged.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.seed_gradient_diagnostic import (
    relative_l2,
    roughness,
    tangential_profile,
)


def _flat_wall(n_stations=40, heights=(1e-5, 1e-4, 1e-3)):
    """A flat plate along y=0 with cells stacked above it."""
    x = np.linspace(0.1, 0.9, n_stations)
    mid = np.stack([x, np.zeros_like(x)], axis=1)
    normal = np.tile(np.array([0.0, 1.0]), (n_stations, 1))
    gx, gy = np.meshgrid(x, np.asarray(heights), indexing="ij")
    centres = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    return mid, normal, centres


def test_the_profile_samples_away_from_the_wall_not_along_it():
    """Anisotropic cells: a plain nearest-neighbour query walks sideways."""
    heights = (1e-5, 1e-4, 1e-3)
    mid, normal, centres = _flat_wall(heights=heights)
    # u grows linearly with height, so each ring has one unambiguous value.
    u = centres[:, 1] * 1000.0
    prof = tangential_profile(u, np.zeros_like(u), centres, mid, normal,
                              np.asarray(heights))
    for j, h in enumerate(heights):
        np.testing.assert_allclose(prof[:, j], h * 1000.0, rtol=1e-9)


def test_a_uniformly_scaled_seed_is_not_called_rough():
    rng = np.random.default_rng(0)
    signal = 1.0 + 0.3 * np.sin(np.linspace(0, 6, 200)) + 0.01 * rng.standard_normal(200)
    assert roughness(20.0 * signal) == pytest.approx(roughness(signal), rel=1e-9)


def test_station_to_station_noise_raises_roughness():
    rng = np.random.default_rng(1)
    smooth = 1.0 + 0.3 * np.sin(np.linspace(0, 6, 200))
    ragged = smooth + 0.05 * rng.standard_normal(200)
    assert roughness(ragged) > 5 * roughness(smooth)


def test_roughness_of_a_straight_line_is_zero():
    assert roughness(np.linspace(1.0, 2.0, 50)) == pytest.approx(0.0, abs=1e-12)


def test_relative_l2_is_zero_for_an_exact_match():
    a = np.array([1.0, 2.0, 3.0])
    assert relative_l2(a, a) == pytest.approx(0.0)
    assert relative_l2(2 * a, a) == pytest.approx(1.0)
