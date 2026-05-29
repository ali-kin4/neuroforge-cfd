"""Mesh import (STL / OBJ) into 2-D :class:`Geometry` cross-sections.

Three-dimensional surface meshes are reduced to a single 2-D body by slicing /
projecting onto the ``z = 0`` plane. The heavy mesh libraries (``pyvista`` /
``trimesh``) are optional and imported lazily inside each function, so this
module always imports even when none of them are installed.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE, Geometry

__all__ = ["load_stl", "load_obj"]


def _ordered_loop_from_points(points2d: np.ndarray) -> np.ndarray:
    """Order a scattered 2-D point cloud into a single CCW boundary loop.

    A convex-hull-free angular sort about the centroid; adequate for the
    star-shaped silhouettes typical of an airfoil/wing cross-section.

    Parameters
    ----------
    points2d : numpy.ndarray
        ``(M, 2)`` boundary point cloud.

    Returns
    -------
    numpy.ndarray
        ``(K, 2)`` de-duplicated, angularly ordered loop.
    """
    p = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    # De-duplicate near-coincident points on a coarse grid.
    keyed = np.round(p, 9)
    _, idx = np.unique(keyed, axis=0, return_index=True)
    p = p[np.sort(idx)]
    centroid = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - centroid[1], p[:, 0] - centroid[0])
    order = np.argsort(ang)
    return p[order]


def _slice_mesh_z0(vertices: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    """Project mesh vertices to the ``z = 0`` plane and return a 2-D loop.

    The current implementation projects all vertices onto the ``xy`` plane and
    extracts an angularly ordered outline. A true planar section is deferred to
    a later release.

    Parameters
    ----------
    vertices : numpy.ndarray
        ``(V, 3)`` (or ``(V, 2)``) vertex coordinates.
    faces : numpy.ndarray or None
        Face connectivity (unused in the projection fallback).

    Returns
    -------
    numpy.ndarray
        ``(K, 2)`` ordered boundary loop.
    """
    v = np.asarray(vertices, dtype=np.float64)
    xy = v[:, :2]
    return _ordered_loop_from_points(xy)


def load_stl(path: str) -> Geometry:
    """Load a 2-D cross-section from an STL surface mesh.

    Parameters
    ----------
    path : str
        Path to a binary or ASCII ``.stl`` file.

    Returns
    -------
    Geometry
        The projected 2-D body.

    Raises
    ------
    NotImplementedError
        If no supported mesh backend (``pyvista`` or ``trimesh``) is installed.
        Planar slicing of arbitrary 3-D meshes is planned for v0.2.
    """
    try:  # optional backend
        import trimesh  # type: ignore
    except Exception:  # pragma: no cover - exercised only without trimesh
        trimesh = None

    if trimesh is not None:
        mesh = trimesh.load(path, force="mesh")
        loop = _slice_mesh_z0(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        return Geometry(
            name="stl_section",
            surface_points=loop.astype(DTYPE),
            meta={"source": "stl", "path": str(path)},
        )

    raise NotImplementedError(
        "STL slicing requires an optional mesh backend (planned for v0.2). "
        "Install 'trimesh' or 'pyvista' to enable STL import."
    )


def load_obj(path: str) -> Geometry:
    """Load a 2-D cross-section from a Wavefront OBJ mesh.

    A minimal pure-Python ``v ...`` vertex parser is used when no mesh backend
    is available, projecting vertices onto ``z = 0``.

    Parameters
    ----------
    path : str
        Path to a ``.obj`` file.

    Returns
    -------
    Geometry
        The projected 2-D body.

    Raises
    ------
    NotImplementedError
        If the file contains no vertices and no mesh backend is available
        (full OBJ handling is planned for v0.2).
    """
    try:  # optional backend
        import trimesh  # type: ignore
    except Exception:  # pragma: no cover - exercised only without trimesh
        trimesh = None

    if trimesh is not None:
        mesh = trimesh.load(path, force="mesh")
        loop = _slice_mesh_z0(np.asarray(mesh.vertices), np.asarray(getattr(mesh, "faces", None)))
        return Geometry(
            name="obj_section",
            surface_points=loop.astype(DTYPE),
            meta={"source": "obj", "path": str(path)},
        )

    # Fallback: parse only vertex lines, no external dependency.
    verts: list[tuple[float, float, float]] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("v "):
                parts = ln.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise NotImplementedError(
            "OBJ import found no vertices and no mesh backend is installed "
            "(full OBJ handling planned for v0.2). Install 'trimesh' or 'pyvista'."
        )
    loop = _slice_mesh_z0(np.asarray(verts), None)
    return Geometry(
        name="obj_section",
        surface_points=loop.astype(DTYPE),
        meta={"source": "obj", "path": str(path)},
    )
