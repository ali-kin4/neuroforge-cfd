"""Tests for streamline/vector plots and VTK (ParaView) export."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import neuroforge as nf
from neuroforge.data.synthetic import SyntheticRANS
from neuroforge.viz import plot_streamlines, plot_vectors, to_vtk


def _field():
    case = nf.FlowCase.from_airfoil("naca2412", aoa=6, reynolds=3e6, u_inf=30.0, resolution=32)
    return case, SyntheticRANS(32).solve(case)


def test_streamlines_and_vectors_return_axes():
    _, f = _field()
    assert plot_streamlines(f) is not None
    assert plot_vectors(f, step=4) is not None


def test_to_vtk_is_paraview_readable(tmp_path):
    _, f = _field()
    p = str(tmp_path / "field.vtk")
    to_vtk(f, p)
    text = (tmp_path / "field.vtk").read_text(encoding="utf-8")
    assert text.startswith("# vtk DataFile Version")
    assert "DATASET STRUCTURED_POINTS" in text
    assert "VECTORS velocity float" in text
    assert "SCALARS pressure float" in text
    # one point-data value per grid cell for a scalar line
    assert f"POINT_DATA {f.domain.nx * f.domain.ny}" in text
