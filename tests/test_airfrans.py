"""Tests for the AirfRANS loader paths that do NOT need the airfrans package:
manifest resolution, idempotent download skip, simulation -> (case, field)
conversion on a synthetic AirfRANS-format array, and the rasterisation cache.
"""

from __future__ import annotations

import pickle

import numpy as np

from neuroforge.core.types import FlowCase
from neuroforge.data.airfrans_loader import (
    _airfrans_present,
    _cache_path,
    _resolve_data_root,
    _sim_to_pair,
    download_airfrans,
    load_airfrans,
)
from neuroforge.data.synthetic import SyntheticRANS
from neuroforge.geometry.sdf import surface_normals


def test_resolve_data_root(tmp_path):
    assert _resolve_data_root(str(tmp_path)) is None
    ds = tmp_path / "Dataset"
    ds.mkdir()
    (ds / "manifest.json").write_text("{}")
    assert _resolve_data_root(str(tmp_path)) == str(ds)
    assert _airfrans_present(str(tmp_path))
    # A manifest directly at root takes precedence (checked first).
    (tmp_path / "manifest.json").write_text("{}")
    assert _resolve_data_root(str(tmp_path)) == str(tmp_path)


def test_download_skips_when_present(tmp_path):
    # Data already present -> returns without importing the airfrans package.
    (tmp_path / "manifest.json").write_text("{}")
    assert download_airfrans(str(tmp_path)) == str(tmp_path)


def _mock_airfrans_array(case: FlowCase) -> np.ndarray:
    """Build a synthetic (M, 11) AirfRANS-format array from a synthetic field."""
    field = SyntheticRANS(case.domain.nx).solve(case)
    X, Y = case.domain.grid()
    m = field.mask > 0.5
    u_inf, v_inf = case.bc.inlet_vector()

    vol = np.zeros((int(m.sum()), 11))
    vol[:, 0], vol[:, 1] = X[m], Y[m]
    vol[:, 2], vol[:, 3] = u_inf, v_inf
    vol[:, 4] = np.abs(field.sdf[m])
    vol[:, 7], vol[:, 8] = field.u[m], field.v[m]
    vol[:, 9], vol[:, 10] = field.p[m], field.nut[m]

    sp = case.geometry.surface_points
    sn = surface_normals(case.geometry)
    surf = np.zeros((sp.shape[0], 11))
    surf[:, 0], surf[:, 1] = sp[:, 0], sp[:, 1]
    surf[:, 2], surf[:, 3] = u_inf, v_inf
    surf[:, 5], surf[:, 6] = sn[:, 0], sn[:, 1]
    return np.vstack([vol, surf])


def test_sim_to_pair():
    case = FlowCase.from_airfoil("naca2412", aoa=5.0, reynolds=3e6, u_inf=30.0, resolution=32)
    arr = _mock_airfrans_array(case)
    c2, f2 = _sim_to_pair(arr, "airFoil2D_SST_x", resolution=32)
    assert f2.u.shape == (32, 32)
    assert f2.as_array().shape == (4, 32, 32)
    assert abs(c2.bc.u_inf - 30.0) < 1.0
    assert abs(c2.bc.aoa_deg - 5.0) < 1.0
    assert c2.geometry.surface_points.shape[0] >= 3
    assert np.isfinite(f2.u).all()


def test_load_airfrans_cache_hit(tmp_path):
    # A pre-written cache pickle is returned without touching the airfrans package.
    case = FlowCase.from_airfoil("naca0012", aoa=0.0, reynolds=1e6, u_inf=10.0, resolution=16)
    pairs = [(case, SyntheticRANS(16).solve(case))]
    cpath = _cache_path(str(tmp_path), "scarce", True, 16, 1)
    with open(cpath, "wb") as fh:
        pickle.dump(pairs, fh)

    got = load_airfrans(
        root="unused", task="scarce", train=True, resolution=16, limit=1,
        cache_dir=str(tmp_path),
    )
    assert len(got) == 1
    assert got[0][1].u.shape == (16, 16)
