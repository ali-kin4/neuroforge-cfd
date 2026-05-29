"""NACA airfoil generation and ``.dat`` loading.

This module produces :class:`~neuroforge.core.types.Geometry` objects for NACA
4- and 5-digit airfoils, and loads Selig/Lednicer coordinate files. Every body
is returned as an ordered counter-clockwise loop of surface points with
``chord = 1``, the leading edge at ``x = 0`` and the trailing edge at ``x = 1``,
together with outward unit ``surface_normals``.

Notes
-----
The point distribution uses a cosine clustering so the curvature-heavy leading
edge is well resolved with relatively few points.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE, Geometry

__all__ = ["naca_airfoil", "airfoil_from_dat"]


# --------------------------------------------------------------------------- #
# Thickness / camber primitives
# --------------------------------------------------------------------------- #


def _cosine_spacing(n: int) -> np.ndarray:
    """Cosine-clustered abscissae on ``[0, 1]`` (dense near the endpoints)."""
    beta = np.linspace(0.0, np.pi, n)
    return (1.0 - np.cos(beta)) * 0.5


def _naca_thickness(x: np.ndarray, t: float, closed: bool) -> np.ndarray:
    """Half-thickness distribution ``yt(x)`` of the NACA 4-digit family.

    Parameters
    ----------
    x : numpy.ndarray
        Chordwise stations on ``[0, 1]``.
    t : float
        Maximum thickness as a fraction of chord.
    closed : bool
        If ``True`` use the modified trailing-edge coefficient so the section
        closes exactly at ``x = 1``; otherwise use the original open TE.

    Returns
    -------
    numpy.ndarray
        Half-thickness ``yt`` at each station.
    """
    a4 = -0.1015 if not closed else -0.1036
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.clip(x, 0.0, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        + a4 * x**4
    )
    return yt


def _camber_4digit(x: np.ndarray, m: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean camber line ``yc`` and slope ``dyc/dx`` for a 4-digit section."""
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m <= 0.0 or p <= 0.0:
        return yc, dyc
    fore = x < p
    aft = ~fore
    yc[fore] = (m / p**2) * (2.0 * p * x[fore] - x[fore] ** 2)
    yc[aft] = (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x[aft] - x[aft] ** 2)
    dyc[fore] = (2.0 * m / p**2) * (p - x[fore])
    dyc[aft] = (2.0 * m / (1.0 - p) ** 2) * (p - x[aft])
    return yc, dyc


def _camber_5digit(x: np.ndarray, code3: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean camber line and slope for a NACA 5-digit section.

    Parameters
    ----------
    x : numpy.ndarray
        Chordwise stations on ``[0, 1]``.
    code3 : str
        First three digits, e.g. ``"230"``. The first digit encodes the design
        lift coefficient (``cl = 0.15 * digit``), the second is twice the
        position of maximum camber, the third flags a reflexed mean line.

    Returns
    -------
    tuple of numpy.ndarray
        ``(yc, dyc/dx)``.
    """
    ld = int(code3[0])           # design-lift index
    pd = int(code3[1])           # 2 * position of max camber (tenths)
    reflex = int(code3[2])       # 0 = standard, 1 = reflexed

    cl_design = 0.15 * ld
    p = 0.05 * pd                # position of max camber along the chord

    # Standard (non-reflexed) tables keyed by p; values m (transition) and k1.
    standard = {
        0.05: (0.0580, 361.4),
        0.10: (0.1260, 51.64),
        0.15: (0.2025, 15.957),
        0.20: (0.2900, 6.643),
        0.25: (0.3910, 3.230),
    }
    # Reflexed tables: m, k1, k2/k1.
    reflexed = {
        0.10: (0.1300, 51.99, 0.000764),
        0.15: (0.2170, 15.793, 0.00677),
        0.20: (0.3180, 6.520, 0.0303),
        0.25: (0.4410, 3.191, 0.1355),
    }

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    scale = cl_design / 0.3  # tables are tabulated for a design cl of 0.3

    if reflex == 0:
        keys = np.array(list(standard))
        pk = keys[np.argmin(np.abs(keys - p))]
        m, k1 = standard[pk]
        fore = x < m
        aft = ~fore
        yc[fore] = (k1 / 6.0) * (x[fore] ** 3 - 3 * m * x[fore] ** 2 + m**2 * (3 - m) * x[fore])
        yc[aft] = (k1 * m**3 / 6.0) * (1.0 - x[aft])
        dyc[fore] = (k1 / 6.0) * (3 * x[fore] ** 2 - 6 * m * x[fore] + m**2 * (3 - m))
        dyc[aft] = -(k1 * m**3 / 6.0)
    else:
        keys = np.array(list(reflexed))
        pk = keys[np.argmin(np.abs(keys - p))]
        m, k1, k2k1 = reflexed[pk]
        fore = x < m
        aft = ~fore
        yc[fore] = (k1 / 6.0) * (
            (x[fore] - m) ** 3
            - k2k1 * (1.0 - m) ** 3 * x[fore]
            - m**3 * x[fore]
            + m**3
        )
        yc[aft] = (k1 / 6.0) * (
            k2k1 * (x[aft] - m) ** 3
            - k2k1 * (1.0 - m) ** 3 * x[aft]
            - m**3 * x[aft]
            + m**3
        )
        dyc[fore] = (k1 / 6.0) * (3 * (x[fore] - m) ** 2 - k2k1 * (1.0 - m) ** 3 - m**3)
        dyc[aft] = (k1 / 6.0) * (3 * k2k1 * (x[aft] - m) ** 2 - k2k1 * (1.0 - m) ** 3 - m**3)

    return yc * scale, dyc * scale


# --------------------------------------------------------------------------- #
# Normals
# --------------------------------------------------------------------------- #


def _outward_normals(points: np.ndarray) -> np.ndarray:
    """Outward unit normals for a CCW-ordered closed loop ``(N, 2)``.

    The normal at vertex ``i`` is built from the averaged edge tangent of the
    two incident segments, rotated by ``-90 deg`` (which points outward for a
    counter-clockwise loop), then flipped if it happens to point toward the
    centroid (robustness for non-convex sections).
    """
    p = np.asarray(points, dtype=np.float64)
    n = p.shape[0]
    # Forward/backward edge tangents, averaged per vertex (periodic).
    fwd = np.roll(p, -1, axis=0) - p
    bwd = p - np.roll(p, 1, axis=0)
    tang = fwd + bwd
    norm = np.hypot(tang[:, 0], tang[:, 1])
    norm[norm < 1e-12] = 1.0
    tang /= norm[:, None]
    # Rotate tangent by -90 deg: (tx, ty) -> (ty, -tx) is outward for CCW.
    normals = np.stack([tang[:, 1], -tang[:, 0]], axis=1)

    # Robustness: ensure each normal points away from the centroid.
    centroid = p.mean(axis=0)
    radial = p - centroid
    flip = np.einsum("ij,ij->i", normals, radial) < 0.0
    normals[flip] *= -1.0
    mag = np.hypot(normals[:, 0], normals[:, 1])
    mag[mag < 1e-12] = 1.0
    normals /= mag[:, None]
    return normals.astype(DTYPE)


def _order_ccw(points: np.ndarray) -> np.ndarray:
    """Return ``points`` ordered counter-clockwise (positive signed area)."""
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    # Shoelace signed area; positive => CCW.
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0.0:
        p = p[::-1]
    return p


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def naca_airfoil(code: str = "naca2412", n_points: int = 200, closed: bool = True) -> Geometry:
    """Generate a NACA 4- or 5-digit airfoil as a :class:`Geometry`.

    Parameters
    ----------
    code : str, optional
        Airfoil designation, with or without the ``"naca"`` prefix, e.g.
        ``"naca2412"`` (4-digit) or ``"naca23012"`` (5-digit).
    n_points : int, optional
        Approximate number of surface points in the returned loop. The loop is
        built from ``n_points // 2`` chordwise stations per surface and the
        shared trailing/leading-edge points are de-duplicated, so the final
        count may differ by one or two.
    closed : bool, optional
        If ``True`` the trailing edge is closed exactly (modified thickness
        coefficient); otherwise the standard open TE is used.

    Returns
    -------
    Geometry
        Body with ordered counter-clockwise ``surface_points`` ``(N, 2)`` and
        outward unit ``surface_normals`` ``(N, 2)``. ``meta`` records the code
        and family.

    Raises
    ------
    ValueError
        If ``code`` does not contain a 4- or 5-digit NACA designation.
    """
    raw = code.strip().lower().replace("naca", "").replace("-", "").replace(" ", "")
    if not raw.isdigit() or len(raw) not in (4, 5):
        raise ValueError(
            f"unsupported NACA code {code!r}: expected 4 or 5 digits "
            "(e.g. 'naca2412' or 'naca23012')"
        )

    n_half = max(int(n_points) // 2, 3)
    xc = _cosine_spacing(n_half)

    if len(raw) == 4:
        m = int(raw[0]) / 100.0          # max camber (fraction of chord)
        p = int(raw[1]) / 10.0           # position of max camber
        t = int(raw[2:]) / 100.0         # max thickness (fraction of chord)
        yc, dyc = _camber_4digit(xc, m, p)
        family = "naca4"
    else:  # 5-digit
        t = int(raw[3:]) / 100.0
        yc, dyc = _camber_5digit(xc, raw[:3])
        family = "naca5"

    yt = _naca_thickness(xc, t, closed)
    theta = np.arctan(dyc)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Upper and lower surfaces offset perpendicular to the camber line.
    xu = xc - yt * sin_t
    yu = yc + yt * cos_t
    xl = xc + yt * sin_t
    yl = yc - yt * cos_t

    # Assemble a single CCW loop. Trace lower surface from TE -> LE, then upper
    # LE -> TE. For a standard airfoil this orientation is already CCW.
    x_loop = np.concatenate([xl[::-1], xu[1:]])
    y_loop = np.concatenate([yl[::-1], yu[1:]])
    pts = np.stack([x_loop, y_loop], axis=1)

    # Drop a duplicated closing TE point if present.
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    pts = _order_ccw(pts)
    normals = _outward_normals(pts)

    return Geometry(
        name=f"naca{raw}",
        surface_points=pts.astype(DTYPE),
        surface_normals=normals,
        meta={"code": f"naca{raw}", "family": family, "thickness": float(t), "closed": bool(closed)},
    )


def airfoil_from_dat(path: str, name: str | None = None) -> Geometry:
    """Load a Selig or Lednicer ``.dat`` airfoil coordinate file.

    Both common formats are handled:

    * **Selig** — a single header line followed by one continuous loop of
      ``x y`` pairs (TE -> upper -> LE -> lower -> TE).
    * **Lednicer** — a header line whose two numbers are point counts, then the
      upper surface (LE -> TE) and the lower surface (LE -> TE) as two blocks.

    The result is normalised to unit chord with the leading edge at ``x = 0``,
    re-ordered counter-clockwise, and given outward unit normals.

    Parameters
    ----------
    path : str
        Path to the ``.dat`` file.
    name : str, optional
        Name for the returned :class:`Geometry`; defaults to the file stem.

    Returns
    -------
    Geometry
        The loaded body.

    Raises
    ------
    ValueError
        If fewer than three coordinate pairs can be parsed.
    """
    import os
    import re

    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    rows: list[tuple[float, float]] = []
    header_counts: tuple[float, float] | None = None
    for ln in lines:
        nums = num_re.findall(ln)
        if len(nums) < 2:
            continue
        a, b = float(nums[0]), float(nums[1])
        # A Lednicer header has two integer-ish counts both > 1.
        if header_counts is None and not rows and a > 1.5 and b > 1.5 \
                and abs(a - round(a)) < 1e-6 and abs(b - round(b)) < 1e-6:
            header_counts = (a, b)
            continue
        rows.append((a, b))

    if len(rows) < 3:
        raise ValueError(f"could not parse at least 3 coordinate pairs from {path!r}")

    arr = np.asarray(rows, dtype=np.float64)

    if header_counts is not None:
        # Lednicer: two LE->TE blocks. Reverse the upper block so the combined
        # path runs TE(lower-end) ... and de-duplicate the shared LE point.
        n_up = int(round(header_counts[0]))
        upper = arr[:n_up]
        lower = arr[n_up:]
        loop = np.concatenate([upper[::-1], lower[1:]])
    else:
        loop = arr

    # Normalise to unit chord with LE at x = 0.
    xmin = loop[:, 0].min()
    xmax = loop[:, 0].max()
    chord = max(xmax - xmin, 1e-12)
    loop = loop.copy()
    loop[:, 0] = (loop[:, 0] - xmin) / chord
    loop[:, 1] = loop[:, 1] / chord

    if np.allclose(loop[0], loop[-1]):
        loop = loop[:-1]

    loop = _order_ccw(loop)
    normals = _outward_normals(loop)

    stem = name or os.path.splitext(os.path.basename(path))[0]
    return Geometry(
        name=stem,
        surface_points=loop.astype(DTYPE),
        surface_normals=normals,
        meta={"source": "dat", "path": str(path),
              "format": "lednicer" if header_counts is not None else "selig"},
    )
