"""Analytic non-airfoil bodies (cylinder and bluff-body-ready helpers).

These build :class:`~neuroforge.core.types.Geometry` objects the same way as the
airfoil generators — an ordered **counter-clockwise** closed loop of surface
points with outward unit normals — so the whole geometry-general pipeline (SDF,
masks, encoding, the synthetic solver) consumes them unchanged.

The only flow-model difference between a lifting airfoil and a bluff body is the
Kutta bound-vortex: bodies created here carry ``meta["lifting"] = False`` so the
synthetic solver skips circulation (see :mod:`neuroforge.data.synthetic`).
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE, Geometry

__all__ = ["cylinder_geometry"]


def cylinder_geometry(
    cx: float = 0.5,
    cy: float = 0.0,
    r: float = 0.5,
    n: int = 200,
) -> Geometry:
    """Build a circular cylinder as a CCW closed loop :class:`Geometry`.

    The default ``(cx, cy, r) = (0.5, 0.0, 0.5)`` places a unit-diameter cylinder
    centred where the unit-chord airfoils sit (LE at ``x=0``, TE at ``x=1``), so a
    cylinder is a drop-in geometry for the same domain/encoding as the airfoils.
    The streamwise extent (``Geometry.chord()``) is the diameter ``2r``, which is
    the conventional reference length for cylinder Reynolds numbers.

    Parameters
    ----------
    cx, cy : float, optional
        Centre of the circle.
    r : float, optional
        Radius.
    n : int, optional
        Number of surface points (sampled at exact, evenly-spaced angles so the
        loop is symmetric about ``y = cy``).

    Returns
    -------
    Geometry
        Body with CCW ``surface_points`` ``(n, 2)`` and outward unit
        ``surface_normals`` ``(n, 2)``. ``meta`` records
        ``{"family": "cylinder", "lifting": False, ...}`` so the synthetic solver
        treats it as a non-lifting bluff body.
    """
    n = max(int(n), 8)
    # endpoint=False -> no duplicated closing point; CCW for increasing theta.
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    pts = np.stack([x, y], axis=1)

    # Outward unit normals are radial for a circle: (cos, sin).
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    return Geometry(
        name=f"cylinder_r{r:g}",
        surface_points=pts.astype(DTYPE),
        surface_normals=normals.astype(DTYPE),
        meta={
            "family": "cylinder",
            "lifting": False,
            "radius": float(r),
            "center": (float(cx), float(cy)),
        },
    )
