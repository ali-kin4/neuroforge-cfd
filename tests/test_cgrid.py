"""Tests for the body-fitted C-grid mesh generator.

All run **without OpenFOAM installed**. The two that matter most are the wake-cut
vertex sharing (which is what removes the need for ``stitchMesh``) and the
convexity check (which is the check ``blockMesh`` itself performs).
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.core.types import FlowCase
from neuroforge.solver import cgrid as cg

AIRFOILS = ["naca0012", "naca2412", "naca4412", "naca0015"]


def _rings(code="naca0012", spec=None):
    spec = spec or cg.CGridSpec()
    inner, nw, ns = cg.inner_curve(code, spec)
    off = cg.offset_open(inner, spec.offset, spec.n_smooth,
                         smooth_range=(nw - 1 - spec.smooth_pad, nw + ns - 2 + spec.smooth_pad))
    far = cg.outer_curve(spec, nw, ns)
    return spec, inner, off, far, nw, ns


# --------------------------------------------------------------------------- #
# Section curve
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", AIRFOILS)
def test_section_starts_and_ends_at_a_sharp_trailing_edge(code):
    """A C-grid wants the cusp: the cut springs from it."""
    surf = cg.airfoil_open_curve(code, 200)
    assert len(surf) == 200
    np.testing.assert_allclose(surf[0], surf[-1], atol=1e-12)
    assert surf[0][0] == pytest.approx(1.0, abs=1e-6)


def test_section_runs_lower_surface_first():
    """Traversal order fixes the orientation the whole mesh depends on."""
    surf = cg.airfoil_open_curve("naca0012", 200)
    assert surf[len(surf) // 8][1] < 0.0     # early -> lower surface
    assert surf[-len(surf) // 8][1] > 0.0    # late  -> upper surface
    assert surf[:, 0].argmin() == pytest.approx(len(surf) // 2, abs=len(surf) // 8)


# --------------------------------------------------------------------------- #
# Wake line
# --------------------------------------------------------------------------- #


def test_wake_line_spans_trailing_edge_to_outlet():
    w = cg.wake_line(np.array([1.0, 0.0]), 20.25, 60, 2e-3)
    assert len(w) == 60
    assert w[0][0] == pytest.approx(1.0)
    assert w[-1][0] == pytest.approx(20.25)
    np.testing.assert_allclose(w[:, 1], 0.0)


def test_wake_line_is_clustered_at_the_trailing_edge():
    w = cg.wake_line(np.array([1.0, 0.0]), 20.25, 60, 2e-3)
    d = np.diff(w[:, 0])
    assert d[0] == pytest.approx(2e-3, rel=0.05)
    assert np.all(np.diff(d) > 0)   # monotonically stretching downstream


def test_wake_line_rejects_an_outlet_upstream_of_the_body():
    with pytest.raises(ValueError):
        cg.wake_line(np.array([1.0, 0.0]), 0.5, 20, 1e-3)


# --------------------------------------------------------------------------- #
# Inner boundary
# --------------------------------------------------------------------------- #


def test_inner_curve_length_matches_the_spec():
    spec, inner, *_ = _rings()
    assert len(inner) == spec.n_i == 2 * spec.n_wake + spec.n_surface - 2


def test_trailing_edge_appears_twice_at_the_same_point():
    """The two occurrences are the slit; they share a vertex only at j = 0."""
    spec, inner, _, _, nw, ns = _rings()
    np.testing.assert_allclose(inner[nw - 1], inner[nw + ns - 2], atol=1e-12)


def test_both_ends_of_the_inner_curve_sit_on_the_outlet():
    spec, inner, *_ = _rings()
    x_out = spec.centre[0] + spec.wake_length
    assert inner[0][0] == pytest.approx(x_out)
    assert inner[-1][0] == pytest.approx(x_out)


@pytest.mark.parametrize("code", AIRFOILS)
def test_orientation_is_what_blockmesh_needs(code):
    """(tangent x outward) must be +z everywhere, cut included."""
    spec, inner, *_ = _rings(code)
    n_out = cg._outward_open(inner)
    t = np.diff(inner, axis=0)
    t = np.vstack([t, t[-1:]])
    cross = t[:, 0] * n_out[:, 1] - t[:, 1] * n_out[:, 0]
    assert np.all(cross > 0)


def test_the_two_cut_sheets_offset_in_opposite_directions():
    """Lower cut must offset down, upper cut up, or there is no slit."""
    spec, inner, off, _, nw, ns = _rings()
    assert off[nw - 1][1] < 0.0        # lower side of the trailing edge
    assert off[nw + ns - 2][1] > 0.0   # upper side


# --------------------------------------------------------------------------- #
# Offset
# --------------------------------------------------------------------------- #


def test_offset_distance_stays_near_the_request():
    spec, inner, off, *_ = _rings()
    d = np.linalg.norm(off - inner, axis=1)
    assert 0.7 * spec.offset < d.min() <= d.max() <= 1.05 * spec.offset


def test_smoothing_the_whole_curve_wrecks_the_offset():
    """Regression: the wake is geometrically stretched to ~a chord per cell, and
    Laplacian smoothing over it drags the offset far off its request."""
    spec, inner, _, far, nw, ns = _rings()
    whole = cg.offset_open(inner, spec.offset, spec.n_smooth)   # no range: everything
    d = np.linalg.norm(whole - inner, axis=1)
    assert d.max() > 2 * spec.offset


def test_offset_endpoints_stay_on_the_outlet_plane():
    """Smoothing must not drag the ends off the outlet or it stops being flat."""
    spec, inner, off, *_ = _rings()
    assert off[0][0] == pytest.approx(inner[0][0])
    assert off[-1][0] == pytest.approx(inner[-1][0])


# --------------------------------------------------------------------------- #
# Far boundary
# --------------------------------------------------------------------------- #


def test_outer_curve_is_matched_point_for_point():
    spec, inner, _, far, *_ = _rings()
    assert len(far) == len(inner)


def test_outer_curve_shape_is_a_c():
    spec, _, _, far, nw, ns = _rings()
    cx, cy = spec.centre
    bottom, arc, top = far[:nw], far[nw - 1:nw + ns - 1], far[nw + ns - 2:]
    np.testing.assert_allclose(bottom[:, 1], cy - spec.far_radius)
    np.testing.assert_allclose(top[:, 1], cy + spec.far_radius)
    r = np.linalg.norm(arc - np.array([cx, cy]), axis=1)
    np.testing.assert_allclose(r, spec.far_radius, rtol=1e-9)


# --------------------------------------------------------------------------- #
# Wake-cut vertex sharing: what removes stitchMesh
# --------------------------------------------------------------------------- #


def test_cut_nodes_share_vertex_labels_at_j_zero():
    spec, *_ , nw, ns = _rings()
    ids, _ = cg.vertex_ids(spec)
    for m in range(spec.n_wake):
        assert ids[0, nw - 1 - m] == ids[0, nw + ns - 2 + m], m


def test_the_two_sheets_separate_above_the_cut():
    """Sharing is j = 0 only; rings 1 and 2 must be all distinct."""
    spec, *_ = _rings()
    ids, _ = cg.vertex_ids(spec)
    for ring in (1, 2):
        assert len(set(ids[ring].tolist())) == spec.n_i


def test_vertex_sharing_actually_saves_labels():
    spec, *_ = _rings()
    ids, n_ids = cg.vertex_ids(spec)
    assert n_ids == 3 * spec.n_i - (spec.n_wake - 1)


def test_every_index_gets_a_label():
    spec, *_ = _rings()
    ids, n_ids = cg.vertex_ids(spec)
    assert ids.shape == (3, spec.n_i)
    assert ids.min() >= 0 and ids.max() == n_ids - 1


# --------------------------------------------------------------------------- #
# Convexity: the check blockMesh performs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", AIRFOILS)
def test_every_cell_quad_is_convex(code):
    spec, inner, off, far, *_ = _rings(code)
    bad, worst = cg._cell_quads_convex([inner, off, far])
    assert bad == 0, f"{bad} non-convex cells, worst cross product {worst:.2e}"
    assert worst > 0.0


def test_smoothing_that_does_not_span_the_trailing_edge_folds_the_offset():
    """Regression: the normal turns ~90 degrees across the sharp cusp, so
    smoothing confined to the section leaves the offset folded there."""
    spec, inner, _, far, nw, ns = _rings()
    narrow = cg.offset_open(inner, spec.offset, 10, smooth_range=(nw - 1, nw + ns - 2))
    bad, _ = cg._cell_quads_convex([inner, narrow, far])
    assert bad > 0


def test_block_mesh_dict_refuses_a_folded_offset():
    spec = cg.CGridSpec(n_surface=80, n_wake=20, n_smooth=0, smooth_pad=0)
    with pytest.raises(ValueError, match="non-convex"):
        cg.block_mesh_dict(spec, "naca0012")


# --------------------------------------------------------------------------- #
# blockMeshDict structure
# --------------------------------------------------------------------------- #


@pytest.fixture
def dict_text():
    spec = cg.CGridSpec(n_surface=80, n_wake=24, n_inner=20, n_outer=15)
    return spec, cg.block_mesh_dict(spec, "naca0012")


def test_block_count_is_two_layers_of_segments(dict_text):
    spec, text = dict_text
    assert text.count("hex (") == 2 * (spec.n_i - 1)


def test_wall_patch_covers_only_the_section(dict_text):
    spec, text = dict_text
    wall = text.split("airfoil")[1].split("}")[0]
    faces = [ln for ln in wall.splitlines() if ln.strip().startswith("(")]
    assert len(faces) - 1 == spec.n_surface - 1


def test_the_wake_cut_is_in_no_patch(dict_text):
    """It must become internal by vertex sharing, not be a boundary."""
    spec, text = dict_text
    wall = text.split("airfoil")[1].split("}")[0]
    n_wall = len([ln for ln in wall.splitlines() if ln.strip().startswith("(")]) - 1
    # Inner-boundary segments total n_i-1; only the section is a wall patch.
    assert n_wall < spec.n_i - 1


def test_outlet_patch_has_both_ends(dict_text):
    spec, text = dict_text
    outlet = text.split("outlet")[1].split("}")[0]
    faces = [ln for ln in outlet.splitlines() if ln.strip().startswith("(")]
    assert len(faces) - 1 == 4  # two radial layers x two ends


def test_no_edge_entries_are_needed(dict_text):
    _, text = dict_text
    assert "polyLine" not in text and "arc " not in text


def test_patches_are_named_and_typed(dict_text):
    _, text = dict_text
    assert "airfoil\n    {\n        type wall;" in text
    assert "farField\n    {\n        type patch;" in text
    assert "outlet\n    {\n        type patch;" in text
    assert "frontAndBack\n    {\n        type empty;" in text


# --------------------------------------------------------------------------- #
# Case writing
# --------------------------------------------------------------------------- #


@pytest.fixture
def case():
    return FlowCase.from_airfoil(airfoil="naca0012", aoa=4.0, reynolds=3e6, resolution=64)


def test_write_cgrid_case_tree(tmp_path, case):
    import os

    spec = cg.CGridSpec(n_surface=80, n_wake=24, n_inner=20, n_outer=15)
    d = cg.write_cgrid_case(case, str(tmp_path / "c"), spec=spec, n_iter=10)
    for rel in ["system/controlDict", "system/fvSchemes", "system/fvSolution",
                "system/blockMeshDict", "constant/transportProperties",
                "constant/turbulenceProperties", "0/U", "0/p", "0/nut", "0/nuTilda",
                "neuroforge.json"]:
        assert os.path.isfile(os.path.join(d, rel)), rel


def test_outlet_gets_a_pressure_boundary(tmp_path, case):
    """The C's open end needs a fixed pressure; the far field is freestream."""
    import os

    spec = cg.CGridSpec(n_surface=80, n_wake=24, n_inner=20, n_outer=15)
    d = cg.write_cgrid_case(case, str(tmp_path / "c"), spec=spec, n_iter=10)
    p = open(os.path.join(d, "0", "p"), encoding="utf-8").read()
    assert "freestreamPressure" in p
    outlet = p.split("outlet")[1]
    assert "fixedValue" in outlet.split("}")[0]


def test_turbulence_model_is_spalart_allmaras(tmp_path, case):
    import os

    spec = cg.CGridSpec(n_surface=80, n_wake=24, n_inner=20, n_outer=15)
    d = cg.write_cgrid_case(case, str(tmp_path / "c"), spec=spec, n_iter=10)
    text = open(os.path.join(d, "constant", "turbulenceProperties"), encoding="utf-8").read()
    assert "SpalartAllmaras" in text


# --------------------------------------------------------------------------- #
# End-to-end (needs a real OpenFOAM installation)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_cgrid_solves_at_airfrans_reynolds(tmp_path, case):
    # Probed in the body, not a skipif: skipif runs at collection time even when
    # `slow` is deselected, and the probe spins up WSL.
    from neuroforge.solver import openfoam as of

    if not of.openfoam_available():
        pytest.skip("OpenFOAM/WSL not installed")
    res = cg.solve_cgrid(case, case_dir=str(tmp_path / "e2e"), n_iter=200, timeout=5400)
    mag = np.hypot(res.u, res.v)
    assert np.all(np.isfinite(mag))
    assert 1.2 < mag.max() < 3.0
