"""Tests for the downstream (wake) handover.

This exists to answer a competing result -- wake-extension initialisation
reports 26.3x iterations by seeding the far wake, which is exactly the region
every seed in this package hands back to the solver. The arm has to be an honest
bound, so what matters is that upstream cells are untouched freestream, that
downstream cells carry the field exactly, and that nothing jumps in between.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.solver.warmstart import wake_seed

FREE = dict(u_inf=1.0, v_inf=0.0, nut_freestream=1e-6)


def _mesh(n=400, lo=-3.0, hi=6.0):
    x = np.linspace(lo, hi, n)
    return np.stack([x, np.zeros(n), np.zeros(n)], axis=1)


def _field(n=400, value=7.0):
    return tuple(np.full(n, value) for _ in range(4))


def test_upstream_cells_are_left_at_freestream():
    centres = _mesh()
    (u, v, p, nut), _ = wake_seed(_field(), centres, x_start=1.0, ramp=0.5, **FREE)
    up = centres[:, 0] <= 1.0
    np.testing.assert_allclose(u[up], 1.0)
    np.testing.assert_allclose(v[up], 0.0)
    np.testing.assert_allclose(p[up], 0.0)
    np.testing.assert_allclose(nut[up], 1e-6)


def test_far_downstream_cells_carry_the_field_exactly():
    centres = _mesh()
    (u, _, p, _), _ = wake_seed(_field(value=7.0), centres, x_start=1.0,
                                ramp=0.5, **FREE)
    far = centres[:, 0] >= 1.5
    np.testing.assert_allclose(u[far], 7.0)
    np.testing.assert_allclose(p[far], 7.0)


def _largest_step(n):
    centres = _mesh(n=n)
    (u, *_), _ = wake_seed(_field(n=n, value=7.0), centres, x_start=1.0,
                           ramp=0.5, **FREE)
    return float(np.abs(np.diff(u[np.argsort(centres[:, 0])])).max())


def test_the_transition_has_no_jump():
    """Continuity, tested as a rate rather than a threshold.

    A hard cut steps the full amplitude (6.0) between two adjacent cells however
    fine the mesh is; a continuous ramp's largest step is set by the cell
    spacing, so halving the spacing halves it. That distinguishes the two
    without a magic constant that only holds at one resolution.
    """
    coarse, fine = _largest_step(1000), _largest_step(2000)
    assert fine == pytest.approx(coarse / 2, rel=0.1)
    assert fine < 0.05 * 6.0


def test_a_zero_ramp_is_a_hard_cut_and_still_valid():
    centres = _mesh()
    (u, *_), rep = wake_seed(_field(value=7.0), centres, x_start=1.0, ramp=0.0,
                             **FREE)
    assert set(np.unique(np.round(u, 9))) == {1.0, 7.0}
    assert rep["ramp"] == 0.0


def test_eddy_viscosity_never_goes_negative():
    centres = _mesh()
    values = (np.zeros(400), np.zeros(400), np.zeros(400), np.full(400, -5.0))
    (_, _, _, nut), _ = wake_seed(values, centres, x_start=1.0, **FREE)
    assert (nut >= 0).all()


def test_the_report_says_how_much_was_handed_over():
    centres = _mesh(n=1000, lo=0.0, hi=2.0)   # half the cells sit past x = 1
    _, rep = wake_seed(_field(n=1000), centres, x_start=1.0, ramp=0.0, **FREE)
    assert rep["seeded_fraction"] == pytest.approx(0.5, abs=0.01)
    assert rep["mode"] == "wake"


def test_moving_x_start_downstream_seeds_less():
    centres = _mesh()
    _, near = wake_seed(_field(), centres, x_start=1.0, **FREE)
    _, far = wake_seed(_field(), centres, x_start=3.0, **FREE)
    assert far["seeded_fraction"] < near["seeded_fraction"]
