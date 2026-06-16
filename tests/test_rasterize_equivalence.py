"""Numerical-equivalence tests for the single-pass rasteriser (Change A).

The refactored :func:`rasterize_point_cloud` builds ONE Delaunay triangulation
and interpolates all channels in a single vector-valued pass. These tests pin it
to the *old* per-channel ``scipy.interpolate.griddata`` reference (which is what
the function used to do) so any drift in results is caught immediately. A Domain
that extends well beyond the point-cloud's convex hull is used on purpose so the
out-of-hull fill path (the place old/new could silently diverge) is exercised.
"""

from __future__ import annotations

import neuroforge  # noqa: F401  (caps BLAS/OMP threads before numpy)
import numpy as np
import pytest

from neuroforge.core.types import Domain
from neuroforge.data.rasterize import rasterize_point_cloud

griddata = pytest.importorskip("scipy.interpolate").griddata


def _reference(points, values, domain, fill, method):
    """Literal copy of the OLD per-channel griddata implementation."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    X, Y = domain.grid()
    xi = np.stack([X.ravel(), Y.ravel()], axis=1)
    ny, nx = domain.shape
    k = vals.shape[1]
    out = np.empty((k, ny, nx), dtype=np.float32)
    for c in range(k):
        grid_c = griddata(pts, vals[:, c], xi, method=method, fill_value=float(fill))
        grid_c = np.asarray(grid_c, dtype=np.float64)
        nan = np.isnan(grid_c)
        if nan.any():
            nn = griddata(pts, vals[:, c], xi[nan], method="nearest")
            grid_c[nan] = np.asarray(nn, dtype=np.float64)
        out[c] = grid_c.reshape(ny, nx).astype(np.float32)
    return out


def _random_cloud(seed=0, m=400):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.5, 0.5, size=(m, 2))  # cloud hull is inside the Domain
    vals = rng.standard_normal((m, 4))
    return pts, vals


# Domain deliberately larger than the cloud's hull -> out-of-hull cells exist,
# which is the path where a nearest-vs-fill mistake would show up.
_DOMAIN = Domain(bounds=(-1.0, 1.0, -1.0, 1.0), nx=20, ny=18)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_matches_reference_interp(method):
    pts, vals = _random_cloud()
    got = rasterize_point_cloud(pts, vals, _DOMAIN, fill=0.0, method=method)
    ref = _reference(pts, vals, _DOMAIN, fill=0.0, method=method)
    assert got.shape == (4, 18, 20)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_out_of_hull_cells_use_fill():
    """Cells outside the cloud's convex hull must equal ``fill`` (not nearest)."""
    pts, vals = _random_cloud()
    fill = 7.0
    got = rasterize_point_cloud(pts, vals, _DOMAIN, fill=fill, method="linear")
    ref = _reference(pts, vals, _DOMAIN, fill=fill, method="linear")
    # There must actually be out-of-hull cells for this test to mean anything.
    assert np.any(np.isclose(ref, fill))
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_nearest_is_exact():
    pts, vals = _random_cloud()
    got = rasterize_point_cloud(pts, vals, _DOMAIN, fill=0.0, method="nearest")
    ref = _reference(pts, vals, _DOMAIN, fill=0.0, method="nearest")
    np.testing.assert_array_equal(got, ref)  # nearest is exact, no interpolation


@pytest.mark.parametrize("m", [0, 1, 3])
def test_few_points_fallback(m):
    """<4 points -> robust nearest-bin scatter fallback (same as before)."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.5, 0.5, size=(m, 2))
    vals = rng.standard_normal((m, 4))
    got = rasterize_point_cloud(pts, vals, _DOMAIN, fill=0.0, method="linear")
    assert got.shape == (4, 18, 20)
    assert np.isfinite(got).all()


def test_1d_values_single_channel():
    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.5, 0.5, size=(50, 2))
    vals = rng.standard_normal(50)
    got = rasterize_point_cloud(pts, vals, _DOMAIN, fill=0.0, method="linear")
    assert got.shape == (1, 18, 20)
