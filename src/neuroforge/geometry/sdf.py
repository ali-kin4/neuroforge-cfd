"""Signed distance fields, solid masks and surface normals.

Pure-NumPy geometry rasterisation: no SciPy dependency. The signed distance is
the unsigned point-to-polygon distance with the sign set by a robust
crossing-number (even-odd) inside test, so it is **negative inside the solid**
body and positive in the fluid, matching the core contract.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE, Domain, Geometry

__all__ = ["signed_distance", "solid_mask", "surface_normals"]


# --------------------------------------------------------------------------- #
# Geometric primitives (vectorised, pure numpy)
# --------------------------------------------------------------------------- #


def _closed_loop(geom: Geometry) -> np.ndarray:
    """Return the surface points as an explicitly closed loop ``(N+1, 2)``."""
    p = np.asarray(geom.surface_points, dtype=np.float64)
    if p.shape[0] >= 1 and not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    return p


def _point_segment_distance_sq(
    px: np.ndarray, py: np.ndarray, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray
) -> np.ndarray:
    """Squared distance from query points to segments ``a -> b``.

    All inputs broadcast against one another. Used with query points shaped
    ``(P, 1)`` and segment endpoints shaped ``(1, S)`` to form a ``(P, S)``
    distance matrix.
    """
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    seg_len_sq = abx * abx + aby * aby
    seg_len_sq = np.where(seg_len_sq < 1e-30, 1e-30, seg_len_sq)
    t = (apx * abx + apy * aby) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


def _unsigned_distance(points_xy: np.ndarray, loop: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Minimum distance from each query point to the polygon boundary.

    Parameters
    ----------
    points_xy : numpy.ndarray
        ``(P, 2)`` query coordinates.
    loop : numpy.ndarray
        ``(N+1, 2)`` closed polygon vertices.
    chunk : int, optional
        Number of query points processed per block to bound peak memory of the
        ``(P, S)`` broadcast.

    Returns
    -------
    numpy.ndarray
        ``(P,)`` minimum Euclidean distances.
    """
    ax = loop[:-1, 0][None, :]
    ay = loop[:-1, 1][None, :]
    bx = loop[1:, 0][None, :]
    by = loop[1:, 1][None, :]

    px_all = points_xy[:, 0]
    py_all = points_xy[:, 1]
    out = np.empty(points_xy.shape[0], dtype=np.float64)
    for start in range(0, points_xy.shape[0], chunk):
        stop = start + chunk
        px = px_all[start:stop, None]
        py = py_all[start:stop, None]
        d2 = _point_segment_distance_sq(px, py, ax, ay, bx, by)
        out[start:stop] = np.sqrt(d2.min(axis=1))
    return out


def _points_inside(points_xy: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Boolean inside test via the crossing-number (even-odd) rule.

    Vectorised ray casting along ``+x``. A point is inside when an odd number
    of polygon edges straddle its ``y`` and lie to its right.

    Parameters
    ----------
    points_xy : numpy.ndarray
        ``(P, 2)`` query coordinates.
    loop : numpy.ndarray
        ``(N+1, 2)`` closed polygon vertices.

    Returns
    -------
    numpy.ndarray
        ``(P,)`` boolean array, ``True`` inside the polygon.
    """
    x1 = loop[:-1, 0]
    y1 = loop[:-1, 1]
    x2 = loop[1:, 0]
    y2 = loop[1:, 1]

    px = points_xy[:, 0][:, None]
    py = points_xy[:, 1][:, None]

    # Edge straddles the horizontal ray through py?
    cond = (y1[None, :] > py) != (y2[None, :] > py)
    # x-coordinate of the edge at height py (guard against horizontal edges).
    dy = (y2 - y1)[None, :]
    dy = np.where(np.abs(dy) < 1e-30, 1e-30, dy)
    x_cross = (x2 - x1)[None, :] * (py - y1[None, :]) / dy + x1[None, :]
    hit = cond & (px < x_cross)
    inside = (np.count_nonzero(hit, axis=1) % 2) == 1
    return inside


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def signed_distance(geom: Geometry, domain: Domain) -> np.ndarray:
    """Signed distance from every grid cell to the body surface.

    Parameters
    ----------
    geom : Geometry
        Closed 2-D body.
    domain : Domain
        Target grid.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` float32 signed distance. **Negative inside the solid**,
        positive in the fluid, zero on the surface.
    """
    X, Y = domain.grid()
    pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)

    # Degenerate geometry guard: a loop needs >= 2 points to define an interior.
    surf = np.asarray(geom.surface_points, dtype=np.float64).reshape(-1, 2)
    if surf.shape[0] < 2:
        if surf.shape[0] == 1:
            d = np.sqrt(((pts - surf[0]) ** 2).sum(axis=1))
        else:  # no surface at all -> everywhere is fluid, far from any body
            xmin, xmax, ymin, ymax = domain.bounds
            d = np.full(pts.shape[0], float(np.hypot(xmax - xmin, ymax - ymin)))
        return d.reshape(domain.shape).astype(DTYPE)

    loop = _closed_loop(geom)

    dist = _unsigned_distance(pts, loop)
    inside = _points_inside(pts, loop)
    dist[inside] *= -1.0

    return dist.reshape(domain.shape).astype(DTYPE)


def solid_mask(geom: Geometry, domain: Domain) -> np.ndarray:
    """Fluid/solid indicator field.

    Parameters
    ----------
    geom : Geometry
        Closed 2-D body.
    domain : Domain
        Target grid.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` float32: ``1.0`` in the fluid, ``0.0`` inside the body.
    """
    X, Y = domain.grid()
    pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)
    loop = _closed_loop(geom)
    inside = _points_inside(pts, loop)
    mask = (~inside).astype(DTYPE)
    return mask.reshape(domain.shape)


def surface_normals(geom: Geometry) -> np.ndarray:
    """Outward unit normals at the geometry's surface points.

    Uses the geometry's stored ``surface_normals`` when available; otherwise
    estimates them from the averaged edge tangents of the (assumed CCW) loop.

    Parameters
    ----------
    geom : Geometry
        Body whose boundary normals are required.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` outward unit normals.
    """
    if geom.surface_normals is not None and geom.surface_normals.shape[0] == geom.n_points:
        n = np.asarray(geom.surface_normals, dtype=np.float64)
        mag = np.hypot(n[:, 0], n[:, 1])
        mag[mag < 1e-12] = 1.0
        return (n / mag[:, None]).astype(DTYPE)

    p = np.asarray(geom.surface_points, dtype=np.float64)
    fwd = np.roll(p, -1, axis=0) - p
    bwd = p - np.roll(p, 1, axis=0)
    tang = fwd + bwd
    norm = np.hypot(tang[:, 0], tang[:, 1])
    norm[norm < 1e-12] = 1.0
    tang /= norm[:, None]
    normals = np.stack([tang[:, 1], -tang[:, 0]], axis=1)

    centroid = p.mean(axis=0)
    radial = p - centroid
    flip = np.einsum("ij,ij->i", normals, radial) < 0.0
    normals[flip] *= -1.0
    mag = np.hypot(normals[:, 0], normals[:, 1])
    mag[mag < 1e-12] = 1.0
    normals /= mag[:, None]
    return normals.astype(DTYPE)
