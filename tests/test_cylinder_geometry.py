"""Cylinder (bluff-body) geometry path: SDF/mask, lifting gate, synthetic solve.

These cover the cross-geometry OOD path: a non-lifting circular cylinder must be
a valid drop-in geometry for the same SDF/mask/encoding/synthetic-solver pipeline
as the airfoils, and the airfoil behaviour must stay unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.core.types import (
    BoundaryConditions,
    Domain,
    FlowCase,
    FluidProperties,
)
from neuroforge.data.synthetic import SyntheticRANS
from neuroforge.geometry.airfoil import naca_airfoil
from neuroforge.geometry.sdf import signed_distance, solid_mask
from neuroforge.geometry.shapes import cylinder_geometry

# --------------------------------------------------------------------------- #
# Geometry: orientation, normals, SDF sign convention
# --------------------------------------------------------------------------- #


def test_cylinder_geometry_metadata_and_normals():
    geom = cylinder_geometry(cx=0.5, cy=0.0, r=0.5, n=200)
    assert geom.meta["family"] == "cylinder"
    assert geom.meta["lifting"] is False
    pts = geom.surface_points
    assert pts.shape == (200, 2)
    # Points lie on the circle of radius r about (cx, cy).
    radii = np.hypot(pts[:, 0] - 0.5, pts[:, 1] - 0.0)
    assert np.allclose(radii, 0.5, atol=1e-4)
    # Outward unit normals (radial), unit length, pointing away from centre.
    n = geom.surface_normals
    assert np.allclose(np.hypot(n[:, 0], n[:, 1]), 1.0, atol=1e-4)
    radial = pts - np.array([0.5, 0.0], dtype=pts.dtype)
    assert np.all(np.einsum("ij,ij->i", n, radial) > 0.0)


def test_cylinder_loop_is_ccw():
    geom = cylinder_geometry(n=200)
    p = np.asarray(geom.surface_points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    # Shoelace signed area is positive for a CCW loop.
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area > 0.0


def test_cylinder_sdf_sign_and_mask():
    geom = cylinder_geometry(cx=0.5, cy=0.0, r=0.5, n=200)
    dom = Domain(bounds=(-1.0, 2.0, -1.5, 1.5), nx=64, ny=64)
    sdf = signed_distance(geom, dom)
    mask = solid_mask(geom, dom)
    assert sdf.shape == dom.shape

    X, Y = dom.grid()
    inside = np.hypot(X - 0.5, Y - 0.0) < 0.4   # safely inside the cylinder
    fluid = np.hypot(X - 0.5, Y - 0.0) > 0.7    # safely outside
    # Negative inside the solid, positive in the fluid.
    assert np.all(sdf[inside] < 0.0)
    assert np.all(sdf[fluid] > 0.0)
    # Mask and SDF agree: solid cells <-> negative sdf.
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert np.all((sdf < 0.0) == (mask < 0.5))


# --------------------------------------------------------------------------- #
# Lifting gate: cylinder gamma == 0; airfoil unchanged
# --------------------------------------------------------------------------- #


def test_lifting_gate_zeroes_cylinder_circulation():
    # Use NONZERO inflow (v_inf != 0) so the gate is actually exercised: at AoA=0
    # a symmetric body would give gamma~0 regardless of the gate, masking the test.
    geom = cylinder_geometry(cx=0.5, cy=0.0, r=0.5, n=200)
    chord = geom.chord()
    gamma = SyntheticRANS._circulation(geom, u_inf=30.0, v_inf=10.0, chord=chord)
    assert gamma == 0.0


def test_lifting_gate_airfoil_unchanged():
    # The airfoil has no "lifting" key -> default True -> nonzero circulation at
    # incidence (the gate must NOT touch the airfoil path).
    geom = naca_airfoil("naca2412", n_points=160)
    chord = geom.chord()
    gamma = SyntheticRANS._circulation(geom, u_inf=30.0, v_inf=10.0, chord=chord)
    assert abs(gamma) > 1e-3


# --------------------------------------------------------------------------- #
# Synthetic solve: symmetry, solid interior, shoulder speed-up
# --------------------------------------------------------------------------- #


def _cylinder_case(resolution: int, u_inf: float = 30.0, reynolds: float = 1.0e6):
    geom = cylinder_geometry(cx=0.5, cy=0.0, r=0.5, n=200)
    length = geom.chord()
    nu = u_inf * length / reynolds
    return FlowCase(
        geometry=geom,
        bc=BoundaryConditions(u_inf=u_inf, aoa_deg=0.0, reynolds=reynolds),
        fluid=FluidProperties(density=1.0, kinematic_viscosity=nu),
        domain=Domain(bounds=(-1.0, 2.0, -1.5, 1.5), nx=resolution, ny=resolution),
        name="cyl_test",
    )


@pytest.mark.parametrize("resolution", [48, 64])
def test_cylinder_solve_top_bottom_symmetry(resolution):
    case = _cylinder_case(resolution)
    field = SyntheticRANS(resolution=resolution, seed=0).solve(case)
    u_inf = case.bc.u_inf
    # cy=0 and symmetric y-axis (linspace(-1.5,1.5)) -> row reversal is y -> -y.
    u_sym = float(np.max(np.abs(field.u - field.u[::-1, :])))
    v_anti = float(np.max(np.abs(field.v + field.v[::-1, :])))
    # Bounded relative to the freestream (float32 + panel solve leave a residue).
    assert u_sym < 1e-3 * u_inf
    assert v_anti < 1e-3 * u_inf
    # Guard against a vacuous antisymmetry check: v must be physically non-trivial
    # (the cylinder deflects flow), else v+v[::-1]~0 would hold simply because v~0.
    assert float(np.max(np.abs(field.v))) > 0.05 * u_inf


@pytest.mark.parametrize("resolution", [48, 64])
def test_cylinder_solve_solid_interior(resolution):
    case = _cylinder_case(resolution)
    field = SyntheticRANS(resolution=resolution, seed=0).solve(case)
    solid = field.mask < 0.5
    assert np.any(solid)
    # u, v, nut exactly zero inside; pressure is KEPT (stagnation 0.5*U^2, not 0).
    assert np.all(field.u[solid] == 0.0)
    assert np.all(field.v[solid] == 0.0)
    assert np.all(field.nut[solid] == 0.0)
    u_mag = case.bc.u_inf
    assert np.allclose(field.p[solid], 0.5 * u_mag * u_mag, atol=1e-3)


@pytest.mark.parametrize("resolution", [48, 64])
def test_cylinder_shoulder_speedup(resolution):
    case = _cylinder_case(resolution)
    field = SyntheticRANS(resolution=resolution, seed=0).solve(case)
    u_inf = case.bc.u_inf
    X, Y = case.domain.grid()
    speed = field.speed()
    # Sample the cylinder "shoulders" (top/bottom of the circle, x~cx): inviscid
    # potential flow over a cylinder accelerates to ~2*U_inf there; even with the
    # near-wall damping and grid resolution we expect a clear overspeed > U_inf.
    shoulder = (np.abs(X - 0.5) < 0.1) & (np.abs(np.abs(Y) - 0.6) < 0.15)
    assert np.any(shoulder)
    assert float(np.max(speed[shoulder])) > u_inf


def test_cylinder_dataset_distinct_from_airfoil():
    # The synthetic cylinder family yields cylinder geometries; the default
    # family is still airfoils. Same seed/idx, different geometry -> no collision.
    cyl = SyntheticRANS(resolution=32, seed=0, family="cylinder").sample_case(0)
    air = SyntheticRANS(resolution=32, seed=0, family="airfoil").sample_case(0)
    assert cyl.geometry.meta["family"] == "cylinder"
    assert cyl.geometry.meta["lifting"] is False
    assert air.geometry.meta["family"] in ("naca4", "naca5")
    assert cyl.bc.aoa_deg == 0.0   # cylinders never vary AoA
