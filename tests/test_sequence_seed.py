"""Grid sequencing -- the classical warm start, and the criterion's out-of-sample test.

Each test names the mistake it prevents, as the rest of the seed suite does.
The one that matters most is
:func:`test_wall_values_are_carried_into_the_map`: without the wall's own
boundary values in the interpolation, every fine cell nearer the wall than any
coarse centre falls outside the convex hull and gets extrapolated -- and that is
exactly where the wall gradient this project is about lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.solver import warmstart as ws


U_INF, V_INF, NUT_FS = 1.0, 0.0, 3e-5 * 3.0


@pytest.fixture
def flat_plate():
    """A wall along y = 0, and two structured meshes above it, coarse and fine."""
    surface = np.stack([np.linspace(0.0, 1.0, 41), np.zeros(41)], axis=1)

    def mesh(n_x, n_y, first):
        x = np.linspace(0.0, 1.0, n_x)
        y = np.geomspace(first, 0.5, n_y)
        X, Y = np.meshgrid(x, y, indexing="xy")
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    coarse = mesh(21, 16, 1e-5)
    fine = mesh(41, 32, 5e-6)
    return surface, coarse, fine


def linear_profile(centres):
    """u = y (so du/dy = 1 everywhere), v = 0, p = -x, nut = y."""
    y, x = centres[:, 1], centres[:, 0]
    return (y.copy(), np.zeros_like(y), -x.copy(), y.copy())


def test_a_linear_profile_survives_the_map(flat_plate):
    """A field linear in y is what linear interpolation reproduces exactly.

    If this drifts, the map is not doing what it claims and every gradient
    number downstream of it is suspect.
    """
    surface, coarse, fine = flat_plate
    out, _ = ws.sequence_seed(linear_profile(coarse), coarse, fine, surface,
                              u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    assert np.allclose(out[0], fine[:, 1], atol=1e-9)


def test_wall_values_are_carried_into_the_map(flat_plate):
    """The mistake: mapping cell centres alone.

    The fine mesh's first centre (5e-6) is below the coarse mesh's first centre
    (1e-5). Without the wall in the interpolation that point is outside the hull
    and gets nearest-neighbour -- which returns the coarse first-cell value and
    so **doubles** the near-wall velocity, destroying the gradient. With the
    wall carried in, it interpolates between no-slip and the first coarse centre
    and lands on the right value.
    """
    surface, coarse, fine = flat_plate
    out, report = ws.sequence_seed(linear_profile(coarse), coarse, fine, surface,
                                   u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    lowest = fine[:, 1] <= 5e-6 + 1e-12
    assert lowest.any()
    # The true value is y itself; nearest-neighbour would have given 1e-5.
    assert np.allclose(out[0][lowest], fine[lowest, 1], atol=1e-9)
    assert report["wall_points"] == len(np.unique(surface, axis=0))


def test_no_slip_holds_at_the_wall(flat_plate):
    """Velocity and nuTilda must go to zero at the surface, as SA requires."""
    surface, coarse, fine = flat_plate
    out, _ = ws.sequence_seed(linear_profile(coarse), coarse, fine, surface,
                              u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    at_wall = fine[:, 1] <= 1e-5
    assert out[0][at_wall].max() < 1e-4      # u small near the wall
    assert out[3][at_wall].max() < 1e-4      # nut likewise
    assert (out[3] >= 0).all()               # never negative


def test_the_first_cell_gradient_is_preserved(flat_plate):
    """The quantity the paper is about, checked directly.

    ``du/dy`` in the fine mesh's first cell must come back as 1, not as the
    ~2x overestimate a hull-less map produces.
    """
    surface, coarse, fine = flat_plate
    out, _ = ws.sequence_seed(linear_profile(coarse), coarse, fine, surface,
                              u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    first = fine[:, 1] <= 5e-6 + 1e-12
    gradient = out[0][first] / fine[first, 1]
    assert np.allclose(gradient, 1.0, rtol=1e-6)


def test_far_field_falls_back_to_freestream(flat_plate):
    """Beyond the coarse mesh there is nothing to map, and freestream is right."""
    surface, coarse, fine = flat_plate
    far = np.array([[0.5, 5.0], [2.0, 0.1]])
    dst = np.vstack([fine, far])
    out, report = ws.sequence_seed(linear_profile(coarse), coarse, dst, surface,
                                   u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    assert report["extrapolated_fraction"] > 0.0
    assert np.isfinite(out[0]).all()


def test_report_states_the_coarsening_it_performed(flat_plate):
    """A number must name the thing it was measured over -- rule 4, in miniature."""
    surface, coarse, fine = flat_plate
    _, report = ws.sequence_seed(linear_profile(coarse), coarse, fine, surface,
                                 u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    assert report["mode"] == "sequence"
    assert report["coarse_cells"] == len(coarse)
    assert report["fine_cells"] == len(fine)
    assert report["coarsening_ratio"] == pytest.approx(len(fine) / len(coarse))
