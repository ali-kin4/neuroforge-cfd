"""Rasterise scattered point-cloud fields onto the structured grid.

AirfRANS (and other mesh-based) datasets store solutions on unstructured point
clouds. :func:`rasterize_point_cloud` interpolates ``(M, K)`` per-point values
onto a ``(K, ny, nx)`` structured stack on a :class:`Domain`. SciPy's
``griddata`` is preferred; if SciPy is unavailable a pure-NumPy nearest-bin
scatter is used so the function never hard-fails.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE, Domain

__all__ = ["rasterize_point_cloud"]


def _nearest_bin_scatter(
    points: np.ndarray, values: np.ndarray, domain: Domain, fill: float
) -> np.ndarray:
    """Pure-NumPy nearest-cell scatter with a nearest-neighbour fill.

    Each point is snapped to its nearest grid cell; cells receiving multiple
    points are averaged. Empty cells are filled by a coarse nearest-occupied
    propagation, falling back to ``fill`` if the grid is entirely empty.

    Parameters
    ----------
    points : numpy.ndarray
        ``(M, 2)`` point coordinates.
    values : numpy.ndarray
        ``(M, K)`` per-point values.
    domain : Domain
        Target grid.
    fill : float
        Value for cells with no nearby data.

    Returns
    -------
    numpy.ndarray
        ``(K, ny, nx)`` float32 raster.
    """
    ny, nx = domain.shape
    xmin, xmax, ymin, ymax = domain.bounds
    k = values.shape[1]

    xspan = max(xmax - xmin, 1e-12)
    yspan = max(ymax - ymin, 1e-12)
    ix = np.clip(np.round((points[:, 0] - xmin) / xspan * (nx - 1)), 0, nx - 1).astype(np.int64)
    iy = np.clip(np.round((points[:, 1] - ymin) / yspan * (ny - 1)), 0, ny - 1).astype(np.int64)
    flat = iy * nx + ix

    acc = np.zeros((ny * nx, k), dtype=np.float64)
    cnt = np.zeros(ny * nx, dtype=np.float64)
    np.add.at(acc, flat, values.astype(np.float64))
    np.add.at(cnt, flat, 1.0)

    filled = cnt > 0
    out = np.full((ny * nx, k), float(fill), dtype=np.float64)
    out[filled] = acc[filled] / cnt[filled, None]

    # Nearest-occupied fill for empty cells (cheap KD-free search over occupied).
    if filled.any() and not filled.all():
        occ_idx = np.nonzero(filled)[0]
        occ_y, occ_x = np.divmod(occ_idx, nx)
        empty_idx = np.nonzero(~filled)[0]
        emp_y, emp_x = np.divmod(empty_idx, nx)
        # Chunk to bound the (E, O) distance matrix.
        chunk = max(1, int(2_000_000 // max(occ_idx.size, 1)))
        for s in range(0, empty_idx.size, chunk):
            e = empty_idx[s:s + chunk]
            ey = emp_y[s:s + chunk][:, None]
            ex = emp_x[s:s + chunk][:, None]
            d2 = (ey - occ_y[None, :]) ** 2 + (ex - occ_x[None, :]) ** 2
            nn = occ_idx[np.argmin(d2, axis=1)]
            out[e] = out[nn]

    return out.reshape(ny, nx, k).transpose(2, 0, 1).astype(DTYPE)


def rasterize_point_cloud(
    points: np.ndarray,
    values: np.ndarray,
    domain: Domain,
    fill: float = 0.0,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate scattered point values onto a structured ``(K, ny, nx)`` grid.

    Parameters
    ----------
    points : numpy.ndarray
        ``(M, 2)`` point coordinates ``(x, y)``.
    values : numpy.ndarray
        ``(M,)`` or ``(M, K)`` per-point values. A 1-D input is treated as a
        single channel ``K = 1``.
    domain : Domain
        Target structured grid.
    fill : float, optional
        Value for query points outside the convex hull of the data (passed to
        ``griddata``) or for empty cells in the NumPy fallback.
    method : str, optional
        Interpolation method for ``scipy.interpolate.griddata``
        (``"linear"``, ``"nearest"`` or ``"cubic"``). Ignored by the fallback.

    Returns
    -------
    numpy.ndarray
        ``(K, ny, nx)`` float32 raster.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    if vals.shape[0] != pts.shape[0]:
        raise ValueError(
            f"points ({pts.shape[0]}) and values ({vals.shape[0]}) length mismatch"
        )

    # Too few points for a Delaunay triangulation (linear/cubic griddata raises
    # a QhullError on < 3 points and a ValueError on 0): use the robust scatter.
    if pts.shape[0] < 4:
        return _nearest_bin_scatter(pts, vals, domain, fill)

    try:
        from scipy.interpolate import griddata  # type: ignore
    except Exception:  # pragma: no cover - exercised only without scipy
        return _nearest_bin_scatter(pts, vals, domain, fill)

    X, Y = domain.grid()
    xi = np.stack([X.ravel(), Y.ravel()], axis=1)
    ny, nx = domain.shape
    k = vals.shape[1]
    out = np.empty((k, ny, nx), dtype=DTYPE)

    for c in range(k):
        grid_c = griddata(pts, vals[:, c], xi, method=method, fill_value=float(fill))
        grid_c = np.asarray(grid_c, dtype=np.float64)
        # `linear`/`cubic` leave NaN outside the hull; backfill with nearest.
        nan = np.isnan(grid_c)
        if nan.any():
            nn = griddata(pts, vals[:, c], xi[nan], method="nearest")
            grid_c[nan] = np.asarray(nn, dtype=np.float64)
        out[c] = grid_c.reshape(ny, nx).astype(DTYPE)

    return out
