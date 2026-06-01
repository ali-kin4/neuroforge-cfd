"""Export a :class:`FlowField` to VTK for ParaView (interactive, ANSYS-style viz).

:func:`to_vtk` writes a legacy ``STRUCTURED_POINTS`` ``.vtk`` file (ASCII, no
dependencies) that ParaView / VisIt / PyVista open directly — giving interactive
contours, streamlines, slices and 3-D rendering of the predicted field. Scalar
fields (``p``, ``speed``, ``nut``, ``mask``, ``sdf``) and the velocity vector are
all written as point data.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import FlowField

__all__ = ["to_vtk"]


def to_vtk(field: FlowField, path: str) -> str:
    """Write ``field`` to a legacy VTK ``STRUCTURED_POINTS`` file at ``path``.

    Returns the path written. Open it in ParaView for interactive contours,
    vectors, and streamlines.
    """
    dom = field.domain
    nx, ny = dom.nx, dom.ny
    xmin, _xmax, ymin, _ymax = dom.bounds
    dx, dy = dom.dx, dom.dy

    def flat(a: np.ndarray) -> np.ndarray:
        # (ny, nx) C-order ravel = x (i) fastest, matching VTK point ordering.
        return np.asarray(a, np.float64).ravel()

    u, v, p = flat(field.u), flat(field.v), flat(field.p)
    speed = np.sqrt(u * u + v * v)
    npts = nx * ny

    lines = [
        "# vtk DataFile Version 3.0",
        "NeuroForge CFD field",
        "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {nx} {ny} 1",
        f"ORIGIN {xmin} {ymin} 0",
        f"SPACING {dx} {dy} 1",
        f"POINT_DATA {npts}",
    ]

    def scalar(name: str, arr: np.ndarray) -> None:
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.append(" ".join(f"{x:.6g}" for x in arr))

    scalar("pressure", p)
    scalar("speed", speed)
    if field.nut is not None:
        scalar("nut", flat(field.nut))
    if field.mask is not None:
        scalar("mask", flat(field.mask))
    if field.sdf is not None:
        scalar("sdf", flat(field.sdf))

    # Velocity as a 3-component vector (w = 0).
    lines.append("VECTORS velocity float")
    vec = np.empty((npts, 3), np.float64)
    vec[:, 0], vec[:, 1], vec[:, 2] = u, v, 0.0
    lines.append("\n".join(f"{a:.6g} {b:.6g} {c:.6g}" for a, b, c in vec))

    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
