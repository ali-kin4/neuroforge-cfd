"""Grid sequencing -- the classical warm start this paper owes a reader.

The map is a **nearest-cell map in body-fitted coordinates**, which is what
``mapFields``-style tools do and is deliberately not cleverer than that: a fine
cell takes the value of the coarse cell above it, and nothing invents structure
the coarse mesh does not contain.

The test that matters most is
:func:`test_the_lookup_is_body_fitted_not_euclidean`. Near-wall cells here reach
aspect ratios of 2e5, so a Euclidean nearest-neighbour query a few microns off
the wall is decided by *tangential* distance and can return a cell far out in
the field. That mistake does not look like a bug -- it looks like grid
sequencing being bad -- which is exactly why it has a test.
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

    return surface, mesh(21, 16, 1e-5), mesh(41, 32, 5e-6)


def height_field(centres):
    """``u = y`` (so ``du/dy = 1``), ``v = 0``, ``p = -x``, ``nut = y``."""
    y, x = centres[:, 1], centres[:, 0]
    return (y.copy(), np.zeros_like(y), -x.copy(), y.copy())


def test_the_map_returns_values_the_coarse_mesh_actually_holds(flat_plate):
    """A nearest map may only hand over values that exist in its source.

    If a mapped value is not one of the coarse mesh's own, something is
    interpolating where this map promises not to, and every claim about what the
    coarse mesh does or does not contain becomes unverifiable.
    """
    surface, coarse, fine = flat_plate
    out, _ = ws.sequence_seed(height_field(coarse), coarse, fine, surface,
                              u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    source = set(np.round(coarse[:, 1], 12)) | {0.0, U_INF}
    assert set(np.round(out[0], 12)) <= source


def test_the_lookup_is_body_fitted_not_euclidean(flat_plate):
    """The mistake this whole implementation exists to avoid.

    A fine cell 5e-6 off the wall must match a coarse cell in the first wall
    ring, not one far out in the field that happens to be closer in ``(x, y)``
    because near-wall cells are thousands of times wider than they are tall.
    """
    surface, coarse, fine = flat_plate
    out, _ = ws.sequence_seed(height_field(coarse), coarse, fine, surface,
                              u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    d = ws.wall_distance(fine, surface[:, :2])
    lowest = d <= 5e-6 + 1e-12
    assert lowest.any()
    # The coarse mesh's own first two rings; anything beyond means the query
    # walked away from the wall.
    rings = np.unique(np.round(coarse[:, 1], 12))[:2]
    assert set(np.round(out[0][lowest], 12)) <= set(rings)


def test_the_wall_gradient_degrades_by_about_the_placement_ratio(flat_plate):
    """Grid sequencing loses the gradient *a little*, and the criterion says so.

    The coarse mesh's first cell centre is 1e-5; the fine mesh's is 5e-6. A cell
    that takes the value from the ring above it therefore reads a gradient a
    small factor too high -- nothing like the ~21x a raster projection of the
    same field costs. The bound here is deliberately loose: what the paper
    claims is the *order*, not a precise factor from a synthetic fixture.
    """
    surface, coarse, fine = flat_plate
    out, _ = ws.sequence_seed(height_field(coarse), coarse, fine, surface,
                              u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    d = ws.wall_distance(fine, surface[:, :2])
    lowest = d <= 5e-6 + 1e-12
    ratio = np.median(out[0][lowest] / d[lowest])
    assert 1.0 <= ratio <= 6.0


def test_nut_is_never_negative(flat_plate):
    surface, coarse, fine = flat_plate
    out, _ = ws.sequence_seed(height_field(coarse), coarse, fine, surface,
                              u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    assert (out[3] >= 0).all()


def test_far_field_falls_back_to_freestream(flat_plate):
    """Beyond the coarse mesh there is nothing to map, and freestream is right."""
    surface, coarse, fine = flat_plate
    far = np.array([[0.5, 5.0], [2.0, 3.0]])
    dst = np.vstack([fine, far])
    out, report = ws.sequence_seed(height_field(coarse), coarse, dst, surface,
                                   u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    assert report["extrapolated_fraction"] > 0.0
    assert np.allclose(out[0][-2:], U_INF)
    assert np.isfinite(out[0]).all()


def test_report_states_the_coarsening_and_the_scale_it_used(flat_plate):
    """A number must name what it was measured over -- scoring rule 4, in miniature."""
    surface, coarse, fine = flat_plate
    _, report = ws.sequence_seed(height_field(coarse), coarse, fine, surface,
                                 u_inf=U_INF, v_inf=V_INF, nut_freestream=NUT_FS)
    assert report["mode"] == "sequence"
    assert report["coarse_cells"] == len(coarse)
    assert report["fine_cells"] == len(fine)
    assert report["coarsening_ratio"] == pytest.approx(len(fine) / len(coarse))
    assert 0.0 < report["coarse_first_cell"] < 1e-3
    assert "body-fitted" in report["coordinates"]
