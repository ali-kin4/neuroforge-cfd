"""Tests for the body-fitted O-grid mesh generator.

All of these run **without OpenFOAM installed**: the curves, the grading and the
``blockMeshDict`` are pure geometry. The convexity test is the important one --
it is the check that ``blockMesh`` itself performs, reproduced in numpy, and it
is what the whole one-block-per-segment design exists to satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.core.types import FlowCase
from neuroforge.solver import ogrid as og

AIRFOILS = ["naca0012", "naca2412", "naca4412", "naca0015", "naca6409"]


# --------------------------------------------------------------------------- #
# Surface loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", AIRFOILS)
def test_loop_has_exactly_n_surface_points(code):
    assert len(og.airfoil_loop(code, n_surface=240, n_te=8)) == 240


def test_loop_layout_is_surface_then_trailing_edge_face():
    """Indices [0, n-n_te) walk the wetted surface; the last n_te walk the base."""
    n, n_te = 240, 8
    loop = og.airfoil_loop("naca0012", n_surface=n, n_te=n_te)
    face = loop[n - n_te:]
    # The base of a NACA section sits at x = 1 (chord normalised), so every
    # point on the trailing-edge face shares that x.
    assert np.allclose(face[:, 0], 1.0, atol=1e-6)
    # ...and it spans the full base thickness, not a single point.
    assert np.ptp(face[:, 1]) > 0.0


def test_trailing_edge_face_is_actually_populated():
    """Regression: the generator leaves the base as a bare segment."""
    sparse = og.airfoil_loop("naca0012", n_surface=240, n_te=2)
    dense = og.airfoil_loop("naca0012", n_surface=240, n_te=20)
    assert np.isclose(dense[-20:, 0], 1.0, atol=1e-6).sum() == 20
    assert np.isclose(sparse[-2:, 0], 1.0, atol=1e-6).sum() == 2


@pytest.mark.parametrize("code", AIRFOILS)
def test_loop_handedness_is_what_blockmesh_needs(code):
    """(tangent x outward normal) must be +z or every cell has negative volume."""
    loop = og.airfoil_loop(code, n_surface=240, n_te=8)
    n_out = og._outward_normals(loop)
    t = np.roll(loop, -1, axis=0) - loop
    assert float(np.sum(t[:, 0] * n_out[:, 1] - t[:, 1] * n_out[:, 0])) > 0.0


def test_loop_rejects_a_trailing_edge_larger_than_the_surface():
    with pytest.raises(ValueError):
        og.airfoil_loop("naca0012", n_surface=20, n_te=16)


# --------------------------------------------------------------------------- #
# Offset and far field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", AIRFOILS)
def test_offset_curve_stays_outside_the_body(code):
    from neuroforge.core.types import Geometry
    from neuroforge.geometry.sdf import _closed_loop, _points_inside

    loop = og.airfoil_loop(code, n_surface=240, n_te=8)
    off = og.offset_curve(loop, 0.08, 10)
    geom = Geometry(name=code, surface_points=loop.astype(np.float32))
    assert int(_points_inside(off, _closed_loop(geom)).sum()) == 0


def test_offset_distance_is_near_the_request():
    loop = og.airfoil_loop("naca0012", n_surface=240, n_te=8)
    d = np.linalg.norm(og.offset_curve(loop, 0.08, 10) - loop, axis=1)
    # Laplacian smoothing pulls the curve in at high-curvature stations; it must
    # not collapse onto the wall, which would wreck the near-wall grading.
    assert 0.5 * 0.08 < d.min() <= d.max() <= 0.08 * 1.05


def test_more_smoothing_relaxes_the_offset_curve():
    """Smoothing evens out the point spacing that the normal offset distorts.

    Total turning is *not* the measure to use: every convex closed curve turns
    through exactly 2*pi, so it cannot distinguish the two. Local roughness --
    the second difference of the vertex positions -- is what smoothing reduces.
    """
    loop = og.airfoil_loop("naca0012", n_surface=240, n_te=8)
    rough = og.offset_curve(loop, 0.08, 0)
    smooth = og.offset_curve(loop, 0.08, 60)

    def roughness(c):
        d2 = np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) - 2.0 * c
        return float(np.linalg.norm(d2, axis=1).sum())

    assert roughness(smooth) < roughness(rough)


def test_far_field_circle_is_a_circle_at_the_requested_radius():
    loop = og.airfoil_loop("naca0012", n_surface=240, n_te=8)
    off = og.offset_curve(loop, 0.08, 10)
    centre = (0.25, 0.0)
    far = og.far_field_circle(off, 20.0, centre)
    r = np.linalg.norm(far - np.asarray(centre), axis=1)
    np.testing.assert_allclose(r, 20.0, rtol=1e-9)


def test_far_field_points_keep_their_bearing():
    """Each outer line must point away from its own inner point, not twist."""
    loop = og.airfoil_loop("naca0012", n_surface=240, n_te=8)
    off = og.offset_curve(loop, 0.08, 10)
    centre = np.array([0.25, 0.0])
    far = og.far_field_circle(off, 20.0, tuple(centre))
    a = np.arctan2(*(off - centre)[:, ::-1].T)
    b = np.arctan2(*(far - centre)[:, ::-1].T)
    np.testing.assert_allclose(np.cos(a - b), 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Convexity: the check blockMesh performs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", AIRFOILS)
def test_every_cell_quad_is_convex(code):
    """``blockMesh`` rejects a concave topology hex ("negative pyramid volume").

    With one block per surface segment the topology quad *is* the cell, so this
    numpy check is exactly the check blockMesh will run.
    """
    loop = og.airfoil_loop(code, n_surface=240, n_te=8)
    off = og.offset_curve(loop, 0.08, 10)
    far = og.far_field_circle(off, 20.0, (0.25, 0.0))
    bad, worst = og._segment_quads_convex([loop, off, far])
    assert bad == 0, f"{bad} non-convex cells, worst cross product {worst:.2e}"
    assert worst > 0.0


def test_block_mesh_dict_refuses_a_folded_offset():
    """An *inward* offset past the half-thickness folds the curve; refuse.

    Note an outward offset of a convex body can never self-intersect, which is
    why the O-grid is safe in normal use -- this drives the guard the other way.
    """
    spec = og.OGridSpec(n_surface=120, n_te=6, offset=-0.05, n_smooth=0)
    loop = og.airfoil_loop("naca0012", n_surface=120, n_te=6)
    with pytest.raises(ValueError, match="non-convex"):
        og.block_mesh_dict(spec, loop)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def test_expansion_ratio_reproduces_the_requested_first_cell():
    length, first, n = 0.08, 1e-5, 60
    ratio = og.expansion_ratio(length, first, n)
    growth = ratio ** (1.0 / (n - 1))
    total = first * (growth**n - 1.0) / (growth - 1.0)
    assert total == pytest.approx(length, rel=1e-6)


def test_expansion_ratio_growth_factor_is_gentle():
    """The ratio is large but the per-cell growth must stay mesh-quality sane."""
    ratio = og.expansion_ratio(0.08, 1e-5, 60)
    assert 1.05 < ratio ** (1.0 / 59) < 1.25


def test_expansion_ratio_is_uniform_when_no_grading_is_needed():
    assert og.expansion_ratio(1.0, 1.0, 10) == 1.0
    assert og.expansion_ratio(0.08, 1e-5, 1) == 1.0


# --------------------------------------------------------------------------- #
# blockMeshDict structure
# --------------------------------------------------------------------------- #


@pytest.fixture
def dict_text():
    spec = og.OGridSpec(n_surface=120, n_te=6, n_inner=20, n_outer=15)
    loop = og.airfoil_loop("naca0012", n_surface=spec.n_surface, n_te=spec.n_te)
    return spec, og.block_mesh_dict(spec, loop)


def test_block_count_is_two_rings_of_segments(dict_text):
    spec, text = dict_text
    assert text.count("hex (") == 2 * spec.n_surface


def test_vertex_count_is_three_rings_two_levels(dict_text):
    spec, text = dict_text
    verts = text.split("vertices")[1].split(");")[0]
    pts = [ln for ln in verts.splitlines()
           if ln.strip().startswith("(") and ln.strip(" ()").strip()]
    assert len(pts) == 3 * spec.n_surface * 2


def test_no_edge_entries_are_needed(dict_text):
    """One block per segment means every block edge is already straight."""
    _, text = dict_text
    assert "polyLine" not in text and "arc " not in text


def test_patches_are_named_and_typed(dict_text):
    spec, text = dict_text
    assert "airfoil\n    {\n        type wall;" in text
    assert "farField\n    {\n        type patch;" in text
    assert "frontAndBack\n    {\n        type empty;" in text


def test_wall_and_far_field_have_one_face_per_segment(dict_text):
    spec, text = dict_text
    wall = text.split("airfoil")[1].split("}")[0]
    assert len([ln for ln in wall.splitlines() if ln.strip().startswith("(")]) - 1 == \
        spec.n_surface


def test_spec_cell_count_matches_the_blocks(dict_text):
    spec, _ = dict_text
    assert spec.n_cells == spec.n_surface * (spec.n_inner + spec.n_outer)


# --------------------------------------------------------------------------- #
# Case writing
# --------------------------------------------------------------------------- #


@pytest.fixture
def case():
    return FlowCase.from_airfoil(airfoil="naca0012", aoa=4.0, reynolds=3e6, resolution=64)


def test_write_ogrid_case_tree(tmp_path, case):
    import os

    spec = og.OGridSpec(n_surface=120, n_te=6, n_inner=20, n_outer=15)
    d = og.write_ogrid_case(case, str(tmp_path / "c"), spec=spec, n_iter=10)
    for rel in ["system/controlDict", "system/fvSchemes", "system/fvSolution",
                "system/blockMeshDict", "constant/transportProperties",
                "constant/turbulenceProperties", "0/U", "0/p", "0/nut", "0/nuTilda"]:
        assert os.path.isfile(os.path.join(d, rel)), rel
    # No topoSet/subsetMesh on this mesh -- the body is a boundary, not a cut-out.
    assert not os.path.isfile(os.path.join(d, "system", "topoSetDict"))
    assert not os.path.isfile(os.path.join(d, "0", "cellId"))


def test_far_field_uses_the_freestream_bc_family(tmp_path, case):
    """A circular far field needs a BC that switches inflow/outflow per face."""
    import os

    spec = og.OGridSpec(n_surface=120, n_te=6, n_inner=20, n_outer=15)
    d = og.write_ogrid_case(case, str(tmp_path / "c"), spec=spec, n_iter=10)
    assert "freestreamVelocity" in open(os.path.join(d, "0", "U"), encoding="utf-8").read()
    assert "freestreamPressure" in open(os.path.join(d, "0", "p"), encoding="utf-8").read()


def test_non_orthogonal_correctors_are_enabled(tmp_path, case):
    """checkMesh reports ~1300 severely non-orthogonal faces on this mesh."""
    import os

    spec = og.OGridSpec(n_surface=120, n_te=6, n_inner=20, n_outer=15)
    d = og.write_ogrid_case(case, str(tmp_path / "c"), spec=spec, n_iter=10)
    text = open(os.path.join(d, "system", "fvSolution"), encoding="utf-8").read()
    assert "nNonOrthogonalCorrectors 2;" in text


def test_turbulence_model_is_spalart_allmaras(tmp_path, case):
    import os

    spec = og.OGridSpec(n_surface=120, n_te=6, n_inner=20, n_outer=15)
    d = og.write_ogrid_case(case, str(tmp_path / "c"), spec=spec, n_iter=10)
    text = open(os.path.join(d, "constant", "turbulenceProperties"), encoding="utf-8").read()
    assert "SpalartAllmaras" in text


# --------------------------------------------------------------------------- #
# End-to-end (needs a real OpenFOAM installation)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_ogrid_solves_at_airfrans_reynolds(tmp_path, case):
    # Probed in the body, not a skipif: skipif is evaluated at collection time
    # even when `slow` is deselected, and the probe spins up WSL.
    if not og.of.openfoam_available():
        pytest.skip("OpenFOAM/WSL not installed")
    res = og.solve_ogrid(case, case_dir=str(tmp_path / "e2e"), n_iter=200, timeout=3600)
    mag = np.hypot(res.u, res.v)
    assert np.all(np.isfinite(mag))
    # An airfoil at 4 degrees has a suction peak near 1.5x freestream; a value
    # far outside that band means the solve diverged rather than converged.
    assert 1.2 < mag.max() < 3.0
    assert res.meta["mesh"]["ok"] is True
