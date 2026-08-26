"""Tests for the warm-start seeding strategies.

All run without OpenFOAM: seeding is interpolation plus a wall profile.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.core.types import Domain, FlowField
from neuroforge.solver import warmstart as ws

U_INF, NUT_INF = 1.0, 3e-6


@pytest.fixture
def domain() -> Domain:
    return Domain(bounds=(-1.0, 2.0, -1.5, 1.5), nx=128, ny=128)


@pytest.fixture
def uniform_field(domain) -> FlowField:
    """A prediction that is exactly freestream everywhere."""
    shp = domain.shape
    return FlowField(
        domain=domain,
        u=np.full(shp, U_INF, np.float32),
        v=np.zeros(shp, np.float32),
        p=np.zeros(shp, np.float32),
        nut=np.full(shp, NUT_INF, np.float32),
    )


@pytest.fixture
def surface() -> np.ndarray:
    """A flat plate along y = 0 from x = 0 to 1 -- distance is just |y|.

    Sampled densely, and with a node landing exactly on x = 0.5 where the tests
    probe: wall distance is nearest-point, so a coarse polyline puts a floor of
    half the sampling step under every distance and quietly breaks the no-slip
    checks.
    """
    return np.stack([np.linspace(0.0, 1.0, 2001), np.zeros(2001)], axis=1)


def centres_at(ys, x=0.5):
    return np.stack([np.full(len(ys), x), np.asarray(ys, float), np.zeros(len(ys))], axis=1)


# --------------------------------------------------------------------------- #
# Boundary-layer thickness
# --------------------------------------------------------------------------- #


def test_bl_thickness_matches_the_number_the_probe_used():
    """0.019 chord at Re 3e6 is what the whole null result is measured against."""
    assert ws.bl_thickness(3e6) == pytest.approx(0.0187, abs=5e-4)


def test_bl_thickness_shrinks_with_reynolds():
    assert ws.bl_thickness(1e4) > ws.bl_thickness(1e6) > ws.bl_thickness(3e6)


def test_bl_thickness_scales_with_chord():
    assert ws.bl_thickness(3e6, chord=2.0) == pytest.approx(2 * ws.bl_thickness(3e6))


# --------------------------------------------------------------------------- #
# Wall distance
# --------------------------------------------------------------------------- #


def test_wall_distance_on_a_flat_plate(surface):
    d = ws.wall_distance(centres_at([0.0, 0.01, 0.5]), surface)
    np.testing.assert_allclose(d, [0.0, 0.01, 0.5], atol=1e-6)


def test_wall_distance_is_never_negative(surface):
    d = ws.wall_distance(centres_at([-0.3, -0.01, 0.01, 0.3]), surface)
    assert np.all(d >= 0)


# --------------------------------------------------------------------------- #
# Plain seed
# --------------------------------------------------------------------------- #


def test_plain_seed_reproduces_the_prediction(domain, uniform_field):
    c = centres_at([0.05, 0.2, 0.6])
    (u, v, p, nut), rep = ws.plain_seed(
        uniform_field, domain, c, u_inf=U_INF, v_inf=0.0, nut_freestream=NUT_INF)
    np.testing.assert_allclose(u, U_INF, rtol=1e-6)
    assert rep["mode"] == "plain"
    assert rep["covered_fraction"] == 1.0


def test_plain_seed_falls_back_to_freestream_outside_the_crop(domain, uniform_field):
    """A 20-chord mesh reaches far beyond the prediction's 3-chord crop."""
    c = np.array([[0.5, 0.0, 0.0], [15.0, 9.0, 0.0]])
    (u, v, p, nut), rep = ws.plain_seed(
        uniform_field, domain, c, u_inf=7.0, v_inf=0.0, nut_freestream=NUT_INF)
    assert u[1] == pytest.approx(7.0)
    assert rep["covered_fraction"] == pytest.approx(0.5)


def test_plain_seed_keeps_the_wall_value_wrong(domain, uniform_field, surface):
    """The failure the hybrid exists to fix: freestream velocity at the wall."""
    c = centres_at([1e-5])
    (u, *_), _ = ws.plain_seed(
        uniform_field, domain, c, u_inf=U_INF, v_inf=0.0, nut_freestream=NUT_INF)
    assert u[0] == pytest.approx(U_INF, rel=1e-6)


# --------------------------------------------------------------------------- #
# Hybrid seed
# --------------------------------------------------------------------------- #


def _hybrid(field, domain, ys, surface, **kw):
    return ws.hybrid_seed(
        field, domain, centres_at(ys), surface, reynolds=3e6,
        u_inf=U_INF, v_inf=0.0, nut_freestream=NUT_INF, **kw)


def test_hybrid_respects_no_slip_at_the_wall(domain, uniform_field, surface):
    (u, v, p, nut), _ = _hybrid(uniform_field, domain, [0.0], surface)
    assert u[0] == pytest.approx(0.0, abs=1e-12)
    assert nut[0] == pytest.approx(0.0, abs=1e-12)


def test_hybrid_leaves_the_outer_field_untouched(domain, uniform_field, surface):
    """Beyond blend_to * delta the prediction must pass through exactly."""
    dl = ws.bl_thickness(3e6)
    (u, *_), _ = _hybrid(uniform_field, domain, [5 * dl, 0.5], surface)
    np.testing.assert_allclose(u, U_INF, rtol=1e-9)


def test_hybrid_matches_the_prediction_at_the_layer_edge(domain, uniform_field, surface):
    dl = ws.bl_thickness(3e6)
    (u, *_), _ = _hybrid(uniform_field, domain, [dl], surface)
    assert u[0] == pytest.approx(U_INF, rel=1e-9)


def test_hybrid_profile_is_monotonic_through_the_layer(domain, uniform_field, surface):
    dl = ws.bl_thickness(3e6)
    ys = np.linspace(0.0, 2.5 * dl, 60)
    (u, *_), _ = _hybrid(uniform_field, domain, ys, surface)
    assert np.all(np.diff(u) >= -1e-12)
    assert u[0] < u[-1]


def test_hybrid_is_continuous_across_the_blend(domain, uniform_field, surface):
    """A kink in the seed is work the solver has to undo."""
    dl = ws.bl_thickness(3e6)
    ys = np.linspace(0.2 * dl, 3.0 * dl, 400)
    (u, *_), _ = _hybrid(uniform_field, domain, ys, surface)
    steps = np.abs(np.diff(u))
    assert steps.max() < 20 * np.median(steps[steps > 0])


def test_hybrid_passes_pressure_through_untouched(domain, surface):
    """Pressure is constant across a boundary layer -- the one safe channel."""
    dom = Domain(bounds=(-1.0, 2.0, -1.5, 1.5), nx=64, ny=64)
    shp = dom.shape
    f = FlowField(domain=dom, u=np.ones(shp, np.float32), v=np.zeros(shp, np.float32),
                  p=np.full(shp, -0.37, np.float32), nut=np.full(shp, NUT_INF, np.float32))
    (u, v, p, nut), _ = ws.hybrid_seed(
        f, dom, centres_at([0.0, 1e-4, 0.5]), surface, reynolds=3e6,
        u_inf=U_INF, v_inf=0.0, nut_freestream=NUT_INF)
    np.testing.assert_allclose(p, -0.37, rtol=1e-6)


def test_hybrid_reports_what_it_rebuilt(domain, uniform_field, surface):
    dl = ws.bl_thickness(3e6)
    ys = np.concatenate([np.linspace(0, dl, 10, endpoint=False), [5 * dl] * 10])
    _, rep = _hybrid(uniform_field, domain, ys, surface)
    assert rep["mode"] == "hybrid"
    assert rep["profiled_fraction"] == pytest.approx(0.5)
    assert rep["delta"] == pytest.approx(dl)


def test_hybrid_honours_an_explicit_delta(domain, uniform_field, surface):
    (u, *_), rep = _hybrid(uniform_field, domain, [0.05], surface, delta=0.2)
    assert rep["delta"] == pytest.approx(0.2)
    assert u[0] < U_INF          # 0.05 is inside a 0.2-thick layer


def test_hybrid_rejects_a_nonpositive_delta(domain, uniform_field, surface):
    with pytest.raises(ValueError):
        _hybrid(uniform_field, domain, [0.01], surface, delta=0.0)


def test_hybrid_differs_from_plain_only_near_the_wall(domain, uniform_field, surface):
    dl = ws.bl_thickness(3e6)
    ys = np.array([1e-5, 0.3 * dl, dl, 4 * dl, 0.8])
    c = centres_at(ys)
    (pu, *_), _ = ws.plain_seed(uniform_field, domain, c, u_inf=U_INF, v_inf=0.0,
                                nut_freestream=NUT_INF)
    (hu, *_), _ = ws.hybrid_seed(uniform_field, domain, c, surface, reynolds=3e6,
                                 u_inf=U_INF, v_inf=0.0, nut_freestream=NUT_INF)
    assert hu[0] < 0.5 * pu[0]                       # deep in the layer: rebuilt
    np.testing.assert_allclose(hu[3:], pu[3:], rtol=1e-9)   # outside: identical
