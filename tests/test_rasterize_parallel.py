"""Parallel rasterise loop (Change B): results match serial and preserve order.

Uses synthetic AirfRANS-format arrays (no dataset download / no ``airfrans``
package). We drive :func:`_rasterize_pairs` directly with a tiny in-memory
``dataset`` list, comparing the process-pool path against the serial path.
"""

from __future__ import annotations

import neuroforge  # noqa: F401  (caps threads before numpy)
import numpy as np

from neuroforge.core.types import FlowCase
from neuroforge.data.airfrans_loader import _rasterize_pairs, _resolve_workers
from neuroforge.geometry.sdf import surface_normals
from neuroforge.data.synthetic import SyntheticRANS


def _mock_array(seed: int) -> np.ndarray:
    """A small (M, 11) AirfRANS-format array derived from a synthetic field."""
    case = FlowCase.from_airfoil(
        "naca2412", aoa=2.0 + seed, reynolds=2e6, u_inf=20.0 + seed, resolution=24
    )
    field = SyntheticRANS(24).solve(case)
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


def test_parallel_matches_serial_and_order():
    n = 6
    dataset = [_mock_array(i) for i in range(n)]
    names = [f"case_{i}" for i in range(n)]

    serial = _rasterize_pairs(dataset, names, n, resolution=24, n_workers=1, progress=False)
    parallel = _rasterize_pairs(dataset, names, n, resolution=24, n_workers=3, progress=False)

    assert len(serial) == len(parallel) == n
    for i in range(n):
        cs, fs = serial[i]
        cp, fp = parallel[i]
        # Order preserved: names line up with input order.
        assert cs.name == names[i] == cp.name
        # Numerically identical (same pure function, just different executor).
        np.testing.assert_array_equal(fs.u, fp.u)
        np.testing.assert_array_equal(fs.v, fp.v)
        np.testing.assert_array_equal(fs.p, fp.p)
        np.testing.assert_array_equal(fs.nut, fp.nut)


def test_parallel_refill_window_preserves_order():
    """Force the sliding-window refill path (n > 2*n_workers) and check order.

    With n_workers=2 the in-flight window is 4, so the 10 cases exercise the
    refill+reassembly branch that only fires at scale (the ~3000-case run).
    """
    n = 10
    dataset = [_mock_array(i) for i in range(n)]
    names = [f"case_{i}" for i in range(n)]

    serial = _rasterize_pairs(dataset, names, n, resolution=24, n_workers=1, progress=False)
    parallel = _rasterize_pairs(dataset, names, n, resolution=24, n_workers=2, progress=False)

    assert [c.name for c, _ in parallel] == names  # input order preserved
    for i in range(n):
        np.testing.assert_array_equal(serial[i][1].u, parallel[i][1].u)
        np.testing.assert_array_equal(serial[i][1].p, parallel[i][1].p)


def test_resolve_workers_env(monkeypatch):
    monkeypatch.setenv("NEUROFORGE_RASTERIZE_WORKERS", "5")
    assert _resolve_workers(None) == 5
    assert _resolve_workers(2) == 2  # explicit arg wins over env
    monkeypatch.delenv("NEUROFORGE_RASTERIZE_WORKERS", raising=False)
    assert _resolve_workers(None) >= 1
