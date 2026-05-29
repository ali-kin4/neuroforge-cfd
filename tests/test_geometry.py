"""Geometry tests: NACA airfoils, signed distance, masks, and case encoding."""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.core.types import INPUT_CHANNELS, N_IN, Domain
from neuroforge.geometry.airfoil import naca_airfoil
from neuroforge.geometry.encode import encode_case
from neuroforge.geometry.sdf import signed_distance, solid_mask


@pytest.mark.parametrize("code", ["naca2412", "naca0012", "naca23012"])
def test_naca_shape_chord_closure(code):
    geom = naca_airfoil(code, n_points=120)
    pts = geom.surface_points
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert pts.shape[0] >= 50
    # Chord = 1 with LE near x=0 and TE near x=1.
    xmin, xmax, _, _ = geom.bounding_box()
    assert xmin == pytest.approx(0.0, abs=1e-3)
    assert xmax == pytest.approx(1.0, abs=1e-3)
    assert geom.chord() == pytest.approx(1.0, abs=1e-3)
    # Surface normals present, unit length.
    assert geom.surface_normals is not None
    assert geom.surface_normals.shape == pts.shape
    mags = np.hypot(geom.surface_normals[:, 0], geom.surface_normals[:, 1])
    assert np.allclose(mags, 1.0, atol=1e-4)
    # Closed loop: first and last points effectively adjacent (loop wraps).
    perim = np.sum(np.hypot(*np.diff(np.vstack([pts, pts[0]]), axis=0).T))
    assert 1.5 < perim < 4.0  # a unit-chord airfoil perimeter is ~2


def test_naca_4_and_5_digit_distinct():
    g4 = naca_airfoil("naca2412", n_points=100)
    g5 = naca_airfoil("naca23012", n_points=100)
    assert g4.meta["family"] == "naca4"
    assert g5.meta["family"] == "naca5"


def test_naca_bad_code_raises():
    with pytest.raises(ValueError):
        naca_airfoil("nacaXYZ")


def test_signed_distance_sign_convention():
    geom = naca_airfoil("naca0012", n_points=160)
    dom = Domain(bounds=(-1.0, 2.0, -1.5, 1.5), nx=64, ny=64)
    sdf = signed_distance(geom, dom)
    assert sdf.shape == dom.shape

    # A point clearly inside the body (mid-chord, on the camber line) is negative.
    X, Y = dom.grid()
    inside_pt = (np.abs(X - 0.5) + np.abs(Y - 0.0))
    j, i = np.unravel_index(np.argmin(inside_pt), inside_pt.shape)
    assert sdf[j, i] < 0.0

    # A far-field corner is positive and large.
    assert sdf[0, 0] > 0.1

    # Some cells negative (interior) and most positive (fluid dominates domain).
    assert np.any(sdf < 0.0)
    assert np.mean(sdf > 0.0) > 0.9

    # On the surface the SDF magnitude is small near a surface point.
    pts = geom.surface_points
    p = pts[len(pts) // 4]
    fx = np.argmin(np.abs(dom.axes()[0] - p[0]))
    fy = np.argmin(np.abs(dom.axes()[1] - p[1]))
    assert abs(sdf[fy, fx]) < 0.1


def test_solid_mask_fraction_sane():
    geom = naca_airfoil("naca2412", n_points=160)
    dom = Domain(bounds=(-1.0, 2.0, -1.5, 1.5), nx=64, ny=64)
    mask = solid_mask(geom, dom)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    solid_frac = float(np.mean(mask < 0.5))
    # A thin airfoil in a 3x3 domain occupies a small but nonzero fraction.
    assert 0.0 < solid_frac < 0.2
    # Mask and SDF agree: solid cells <-> negative sdf.
    sdf = signed_distance(geom, dom)
    assert np.all((sdf < 0.0) == (mask < 0.5))


def test_encode_case_channels(tiny_case):
    stack = encode_case(tiny_case)
    ny, nx = tiny_case.domain.shape
    assert stack.shape == (N_IN, ny, nx)
    assert stack.dtype == np.float32
    assert INPUT_CHANNELS == ("sdf", "mask", "x", "y", "u_in", "v_in", "log_re")

    sdf, mask, xn, yn, u_in, v_in, log_re = stack
    # mask channel in {0,1}
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    # x/y normalised coords roughly in [-1, 1]
    assert xn.min() == pytest.approx(-1.0, abs=1e-5)
    assert xn.max() == pytest.approx(1.0, abs=1e-5)
    assert yn.min() == pytest.approx(-1.0, abs=1e-5)
    assert yn.max() == pytest.approx(1.0, abs=1e-5)
    # u_in / v_in equal the broadcast inlet vector.
    u0, v0 = tiny_case.bc.inlet_vector()
    assert np.allclose(u_in, u0)
    assert np.allclose(v_in, v0)
    # log_re channel == log10(reynolds), constant.
    assert np.allclose(log_re, np.log10(tiny_case.bc.reynolds))
