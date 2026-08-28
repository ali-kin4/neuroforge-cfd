"""Tests for the selective seed.

The point of `masked_seed` is that a warm start does not have to be
all-or-nothing. Each test pins one of the two knobs.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.solver import warmstart as ws


@pytest.fixture
def geometry():
    """A flat wall along y = 0, with cells at a spread of wall distances."""
    surface = np.stack([np.linspace(0.0, 1.0, 41), np.zeros(41)], axis=1)
    d = np.array([1e-5, 1e-3, 5e-3, 0.02, 0.1, 1.0])
    centres = np.stack([np.full(d.size, 0.5), d], axis=1)
    return centres, surface, d


@pytest.fixture
def seed(geometry):
    _, _, d = geometry
    n = d.size
    return (np.full(n, 0.8), np.full(n, 0.1), np.full(n, -0.3), np.full(n, 1e-4))


FREE = dict(u_inf=1.0, v_inf=0.0, nut_freestream=1e-5)


def test_keeping_everything_and_masking_nothing_is_the_identity(geometry, seed):
    centres, surface, _ = geometry
    out, report = ws.masked_seed(seed, centres, surface, **FREE)
    for got, want in zip(out, seed):
        np.testing.assert_allclose(got, want)
    assert report["blended_fraction"] == 0.0


def test_pressure_only_seed_leaves_the_rest_at_freestream(geometry, seed):
    centres, surface, _ = geometry
    (u, v, p, nut), report = ws.masked_seed(
        seed, centres, surface, fields=("p",), **FREE)
    np.testing.assert_allclose(p, seed[2])          # the slow field is kept
    np.testing.assert_allclose(u, 1.0)              # everything else is a cold start
    np.testing.assert_allclose(v, 0.0)
    np.testing.assert_allclose(nut, 1e-5)
    assert report["fields"] == ["p"]


def test_masking_the_near_wall_velocity_leaves_pressure_alone(geometry, seed):
    centres, surface, d = geometry
    (u, _v, p, nut), report = ws.masked_seed(
        seed, centres, surface, free_within=0.005, ramp=3.0, **FREE)
    near, far = d <= 0.005, d >= 0.015
    np.testing.assert_allclose(u[near], 1.0)        # solver's job again
    np.testing.assert_allclose(nut[near], 1e-5)
    np.testing.assert_allclose(u[far], 0.8)         # prediction, untouched
    # Pressure is never masked: it has no wall layer to get wrong, and it is the
    # slowest field in a cold solve.
    np.testing.assert_allclose(p, seed[2])
    assert 0.0 < report["blended_fraction"] < 1.0


def test_the_blend_is_smooth_and_monotone(geometry, seed):
    surface = np.stack([np.linspace(0.0, 1.0, 41), np.zeros(41)], axis=1)
    d = np.geomspace(1e-4, 0.1, 60)
    centres = np.stack([np.full(d.size, 0.5), d], axis=1)
    (u, *_), _ = ws.masked_seed(
        (np.full(d.size, 0.8), np.zeros(d.size), np.zeros(d.size),
         np.full(d.size, 1e-4)),
        centres, surface, free_within=0.005, ramp=3.0, **FREE)
    # Freestream 1.0 falling to the prediction 0.8: monotone, no overshoot.
    assert np.all(np.diff(u) <= 1e-12)
    assert u.max() == pytest.approx(1.0)
    assert u.min() == pytest.approx(0.8)


def test_eddy_viscosity_never_goes_negative(geometry):
    centres, surface, d = geometry
    n = d.size
    negative = (np.full(n, 0.8), np.zeros(n), np.zeros(n), np.full(n, -1.0))
    (*_, nut), _ = ws.masked_seed(negative, centres, surface, **FREE)
    assert nut.min() >= 0.0


def test_an_unknown_field_name_is_rejected(geometry, seed):
    centres, surface, _ = geometry
    with pytest.raises(ValueError, match="unknown seed fields"):
        ws.masked_seed(seed, centres, surface, fields=("pressure",), **FREE)


def test_the_seed_is_not_modified_in_place(geometry, seed):
    centres, surface, _ = geometry
    before = [a.copy() for a in seed]
    ws.masked_seed(seed, centres, surface, fields=("p",), free_within=0.01, **FREE)
    for got, want in zip(seed, before):
        np.testing.assert_allclose(got, want)
