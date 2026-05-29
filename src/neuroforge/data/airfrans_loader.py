"""Loader for the AirfRANS dataset → NeuroForge ``(FlowCase, FlowField)`` pairs.

AirfRANS (Bonnet et al., NeurIPS 2022) provides incompressible RANS solutions
over NACA 4/5-digit airfoils as point clouds. ``airfrans.dataset.load`` returns,
per simulation, an ``(M, 11)`` array with the column layout:

    [0] x            [1] y                 (position)
    [2] u_in_x       [3] u_in_y            (freestream / inlet velocity, constant)
    [4] sdf          (implicit distance to the airfoil; >= 0)
    [5] n_x          [6] n_y               (surface normals; 0 away from the wall)
    [7] u            [8] v                 (target velocity)
    [9] p            (target kinematic pressure p/rho)
    [10] nut         (target turbulent kinematic viscosity)

We rasterise the target fields onto a structured crop around the airfoil and
reconstruct the body geometry from the on-wall points. This module **lazily
imports** ``airfrans`` so the package installs and imports without it; install
with ``pip install neuroforge-cfd[data]`` and download the data first.
"""

from __future__ import annotations

import os

import numpy as np

from neuroforge.core.types import (
    DTYPE,
    BoundaryConditions,
    Domain,
    FlowCase,
    FlowField,
    FluidProperties,
    Geometry,
)
from neuroforge.data.rasterize import rasterize_point_cloud

__all__ = ["download_airfrans", "load_airfrans", "AIRFRANS_NU"]

# AirfRANS uses air at this kinematic viscosity with unit chord.
AIRFRANS_NU: float = 1.56e-5
_CHORD: float = 1.0
# Default ML crop around the airfoil (chord on [0, 1]).
_CROP = (-1.0, 2.0, -1.5, 1.5)


def _require_airfrans():
    try:
        import airfrans as af  # type: ignore

        return af
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "The 'airfrans' package is required for the AirfRANS loader. "
            "Install it with `pip install neuroforge-cfd[data]` (or `pip install airfrans`), "
            "then download the dataset via download_airfrans(root)."
        ) from exc


def download_airfrans(root: str = "data") -> str:
    """Download + unzip the AirfRANS dataset under ``root``. Returns the path."""
    af = _require_airfrans()
    os.makedirs(root, exist_ok=True)
    af.dataset.download(root=root, unzip=True)
    return root


def _order_surface_loop(points: np.ndarray) -> np.ndarray:
    """Order scattered airfoil-surface points into a single CCW loop.

    Greedy nearest-neighbour walk starting from the trailing edge (max-x point),
    then oriented counter-clockwise via the signed area. Robust enough for the
    thin, non-convex airfoil sections in AirfRANS (where angular sorting fails).
    """
    pts = np.asarray(points, np.float64)
    n = pts.shape[0]
    if n < 4:
        return pts.astype(DTYPE)
    start = int(np.argmax(pts[:, 0]))
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        d2 = np.sum((pts - pts[cur]) ** 2, axis=1)
        d2[visited] = np.inf
        nxt = int(np.argmin(d2))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    loop = pts[order]
    # Orient CCW (positive signed area via the shoelace formula).
    area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
    if area < 0:
        loop = loop[::-1]
    return loop.astype(DTYPE)


def _sim_to_pair(data: np.ndarray, name: str, resolution: int) -> tuple[FlowCase, FlowField]:
    """Convert one AirfRANS simulation array into a (FlowCase, FlowField)."""
    data = np.asarray(data, np.float64)
    pos = data[:, 0:2]
    u_in = data[:, 2:4]
    normals = data[:, 5:7]
    targets = data[:, 7:11]  # u, v, p, nut

    # Boundary conditions from the (constant) inlet column.
    uin = u_in[np.isfinite(u_in).all(axis=1)]
    u_inf_vec = uin.mean(axis=0) if uin.size else u_in[0]
    u_inf = float(np.hypot(*u_inf_vec))
    aoa = float(np.degrees(np.arctan2(u_inf_vec[1], u_inf_vec[0])))
    reynolds = float(u_inf * _CHORD / AIRFRANS_NU)

    # Reconstruct geometry from on-wall points (nonzero normal).
    on_wall = np.linalg.norm(normals, axis=1) > 1e-8
    wall_pts = pos[on_wall]
    if wall_pts.shape[0] < 4:  # fall back: points nearest the surface (sdf ~ 0)
        order = np.argsort(data[:, 4])[: max(64, data.shape[0] // 50)]
        wall_pts = pos[order]
    loop = _order_surface_loop(wall_pts)
    geom = Geometry(name=name, surface_points=loop)

    domain = Domain(bounds=_CROP, nx=resolution, ny=resolution)
    case = FlowCase(
        geometry=geom,
        bc=BoundaryConditions(u_inf=u_inf, aoa_deg=aoa, reynolds=reynolds),
        fluid=FluidProperties(density=1.0, kinematic_viscosity=AIRFRANS_NU),
        domain=domain,
        name=name,
    )

    # Rasterise targets onto the structured crop.
    raster = rasterize_point_cloud(pos, targets, domain, fill=0.0, method="linear")
    # Geometry-consistent mask/sdf (the data's interior is empty).
    from neuroforge.geometry.sdf import signed_distance, solid_mask

    sdf = signed_distance(geom, domain)
    mask = solid_mask(geom, domain)
    solid = mask < 0.5
    u = np.where(solid, 0.0, raster[0])
    v = np.where(solid, 0.0, raster[1])
    p = raster[2]
    nut = np.where(solid, 0.0, np.maximum(raster[3], 0.0))

    field = FlowField(
        domain=domain, u=u, v=v, p=p, nut=nut, mask=mask, sdf=sdf,
        meta={"source": "airfrans", "case": name},
    )
    return case, field


def _cache_path(cache_dir: str, task: str, train: bool, resolution: int, limit) -> str:
    split = "train" if train else "test"
    return os.path.join(cache_dir, f"airfrans_{task}_{split}_r{resolution}_n{limit}.pkl")


def load_airfrans(
    root: str = "data",
    task: str = "scarce",
    train: bool = True,
    resolution: int = 128,
    limit: int | None = None,
    cache_dir: str | None = None,
    progress: bool = True,
) -> list[tuple[FlowCase, FlowField]]:
    """Load AirfRANS simulations as ``(FlowCase, FlowField)`` pairs.

    Parameters
    ----------
    root : str
        Directory containing the downloaded dataset.
    task : str
        AirfRANS split: ``'full'``, ``'scarce'``, ``'reynolds'`` or ``'aoa'``.
    train : bool
        Load the train split (else the test split).
    resolution : int
        Structured-grid resolution to rasterise onto.
    limit : int, optional
        Cap the number of simulations loaded (useful for quick experiments).
    cache_dir : str, optional
        If given, the rasterised pairs are pickled here keyed by
        ``(task, split, resolution, limit)`` and reused on the next call. This
        makes repeated Colab sessions fast (rasterisation is the slow step).
    progress : bool
        Show a tqdm bar while rasterising.
    """
    # Fast path: reuse a cached rasterisation.
    if cache_dir is not None:
        cpath = _cache_path(cache_dir, task, train, resolution, limit)
        if os.path.exists(cpath):
            import pickle

            with open(cpath, "rb") as fh:
                return pickle.load(fh)

    af = _require_airfrans()
    dataset, names = af.dataset.load(root=root, task=task, train=train)
    n = len(names) if limit is None else min(limit, len(names))

    iterator = range(n)
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(iterator, desc=f"rasterise airfrans/{task}/{'train' if train else 'test'}")
        except Exception:
            pass

    pairs: list[tuple[FlowCase, FlowField]] = []
    for i in iterator:
        pairs.append(_sim_to_pair(np.asarray(dataset[i]), str(names[i]), resolution))

    if cache_dir is not None:
        import pickle

        os.makedirs(cache_dir, exist_ok=True)
        with open(_cache_path(cache_dir, task, train, resolution, limit), "wb") as fh:
            pickle.dump(pairs, fh)
    return pairs
