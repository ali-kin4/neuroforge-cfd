"""Loader for the DeepCFD dataset -> NeuroForge ``(FlowCase, FlowField)`` pairs.

DeepCFD (Ribeiro et al., 2020; https://github.com/mdribeiro/DeepCFD,
https://zenodo.org/records/3666056) is a set of **981** 2-D **steady-state laminar
incompressible** channel-flow simulations (OpenFOAM ``simpleFoam``) over randomly
shaped bluff obstacles. It is a genuinely different geometry AND regime from
AirfRANS (turbulent external aerodynamics over airfoils), which is exactly why it
is used here to test whether the physics-residual **trust signal** generalises.

Data layout (``dataX.pkl`` / ``dataY.pkl``, each a pickled ``(981, 3, 172, 79)``
``float32`` numpy array; axis 2 = streamwise x (172), axis 3 = perpendicular y
(79) -- verified empirically against the region labels):

``dataX`` (inputs)
    [0] SDF from the obstacle surface (negative inside the solid),
    [1] multi-label flow-region channel: ``0=obstacle, 1=free fluid,
        2=top/bottom no-slip wall, 3=constant-velocity inlet (left),
        4=zero-gradient outlet (right)``,
    [2] SDF from the top/bottom channel walls.

``dataY`` (targets)
    [0] ``Ux`` (streamwise velocity), [1] ``Uy`` (cross-stream velocity),
    [2] ``p`` (kinematic pressure ``p/rho`` -- simpleFoam is incompressible).

Simulation parameters (DeepCFD paper, Section 3 "Problem Setup" + Figure 5):

* **Kinematic viscosity** ``nu = 1e-4 m^2/s`` ("the laminar dynamic viscosity is
  set to ``1 x 10^-4 m^2/s``"; confirmed by Figure 5: ``nu = 1e-4 m^2/s``).
* **Inlet velocity** ``U = 0.1 m/s`` ("a constant radial velocity of 0.1 m/s on
  the inlet (left wall)").
* **Domain** ``260 mm`` (streamwise) ``x 120 mm`` (perpendicular) =
  ``0.26 m x 0.12 m``.
* Five convex primitive families (circle, square, forward-/backward-facing
  triangle, rhombus), randomly perturbed; **laminar** (Re ~ U*L/nu ~ 40), so the
  eddy viscosity ``nut`` is **physically exactly zero**.

The flow fields are already on a structured grid, so we **resample** the targets
(scipy ``RegularGridInterpolator``) onto a square ``resolution x resolution``
NeuroForge grid (so the finite-difference residual stencils have a well-defined
``dx``/``dy``), and we rebuild ``mask``/``sdf`` from a reconstructed
:class:`Geometry` using the *same* ``solid_mask``/``signed_distance`` functions
that :func:`~neuroforge.geometry.encode.encode_case` will use at train/eval time
(so the network's input mask and the evaluation reference mask never diverge --
mirrors the AirfRANS loader's discipline).
"""

from __future__ import annotations

import os
import pickle

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

__all__ = [
    "DEEPCFD_NU",
    "DEEPCFD_U_INF",
    "DEEPCFD_DOMAIN_M",
    "load_deepcfd",
]

# --- Authoritative physical parameters (DeepCFD paper, Sec. 3 + Fig. 5) ------ #
#: Laminar molecular kinematic viscosity used in the OpenFOAM simpleFoam runs.
DEEPCFD_NU: float = 1.0e-4          # m^2/s  (paper Sec.3 / Fig.5)
#: Constant inlet velocity on the left wall.
DEEPCFD_U_INF: float = 0.1         # m/s    (paper Sec.3)
#: Physical channel size: 260 mm streamwise x 120 mm perpendicular.
DEEPCFD_DOMAIN_M: tuple[float, float] = (0.26, 0.12)

#: Native DeepCFD array shape per sample: (3 channels, 172 streamwise, 79 perp).
_NATIVE_NX: int = 172
_NATIVE_NY: int = 79

# Region-channel labels (dataX channel 1).
_REGION_OBSTACLE = 0
# 1 = fluid, 2 = wall, 3 = inlet, 4 = outlet -> all of {1,2,3,4} are fluid domain.


def _load_raw(root: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and orient the raw ``(N, 3, ny, nx)`` arrays.

    The on-disk arrays are ``(N, 3, x=172, y=79)``; NeuroForge stores fields as
    ``(ny, nx)`` so we transpose the last two axes to ``(N, 3, y=79, x=172)``.
    """
    xp = os.path.join(root, "dataX.pkl")
    yp = os.path.join(root, "dataY.pkl")
    if not (os.path.isfile(xp) and os.path.isfile(yp)):
        raise FileNotFoundError(
            f"DeepCFD dataX.pkl/dataY.pkl not found under '{root}'. Download "
            "DeepCFD.zip from https://zenodo.org/records/3666056 and extract it "
            f"into '{root}' (it contains dataX.pkl and dataY.pkl)."
        )
    with open(xp, "rb") as fh:
        X = np.asarray(pickle.load(fh), dtype=np.float32)
    with open(yp, "rb") as fh:
        Y = np.asarray(pickle.load(fh), dtype=np.float32)
    if X.ndim != 4 or X.shape[1] != 3:
        raise ValueError(f"unexpected dataX shape {X.shape}; expected (N,3,172,79)")
    # (N, C, x, y) -> (N, C, y, x) so spatial axes are (ny, nx).
    X = np.transpose(X, (0, 1, 3, 2))
    Y = np.transpose(Y, (0, 1, 3, 2))
    return X, Y


def _boundary_pixels(obst: np.ndarray) -> np.ndarray:
    """Boolean map of obstacle cells with at least one non-obstacle 4-neighbour."""
    b = np.zeros_like(obst)
    b[:-1, :] |= obst[:-1, :] & ~obst[1:, :]
    b[1:, :] |= obst[1:, :] & ~obst[:-1, :]
    b[:, :-1] |= obst[:, :-1] & ~obst[:, 1:]
    b[:, 1:] |= obst[:, 1:] & ~obst[:, :-1]
    return b


def _reconstruct_geometry(
    region_yx: np.ndarray, name: str, domain_m: tuple[float, float]
) -> Geometry:
    """Reconstruct a closed obstacle loop from the region-label map.

    ``region_yx`` is the ``(ny, nx)`` region channel (``0`` = obstacle). The
    DeepCFD primitives are convex (and the perturbed variants star-convex about
    their centroid), so ordering the obstacle's boundary-pixel centres by polar
    angle around the centroid yields a clean, non-self-crossing CCW loop without
    any contour-tracing dependency (skimage is not required). Pixel indices are
    mapped to physical metres via the channel dimensions.

    Parameters
    ----------
    region_yx : numpy.ndarray
        ``(ny, nx)`` region-label map.
    name : str
        Geometry name.
    domain_m : tuple of float
        ``(Lx, Ly)`` physical domain size in metres.

    Returns
    -------
    Geometry
        CCW-ordered surface loop in physical coordinates.
    """
    ny, nx = region_yx.shape
    lx, ly = domain_m
    obst = region_yx == _REGION_OBSTACLE
    b = _boundary_pixels(obst)
    jy, jx = np.where(b)  # row (y) index, col (x) index
    if jx.size < 4:
        # Degenerate / empty obstacle: a tiny centred square so downstream code
        # (SDF/mask, force integrals) stays well-defined.
        cx, cy = lx * 0.5, ly * 0.5
        d = 0.01 * min(lx, ly)
        loop = np.array(
            [[cx - d, cy - d], [cx + d, cy - d], [cx + d, cy + d], [cx - d, cy + d]],
            dtype=np.float64,
        )
        return Geometry(name=name, surface_points=loop.astype(DTYPE))

    # Pixel-centre physical coordinates.
    px = (jx + 0.5) / nx * lx
    py = (jy + 0.5) / ny * ly
    pts = np.stack([px, py], axis=1).astype(np.float64)

    centroid = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    loop = pts[np.argsort(ang)]

    # Orient counter-clockwise (positive shoelace area).
    area = 0.5 * np.sum(
        loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1]
    )
    if area < 0:
        loop = loop[::-1]
    return Geometry(name=name, surface_points=loop.astype(DTYPE))


def _resample_targets(targets_yx: np.ndarray, domain: Domain) -> np.ndarray:
    """Resample the native ``(3, ny0, nx0)`` targets onto ``domain`` (``RegularGridInterpolator``).

    Returns ``(3, ny, nx)`` for ``(Ux, Uy, p)``. Linear interpolation; out-of-range
    queries clamp to the nearest edge value (``bounds_error=False``).
    """
    from scipy.interpolate import RegularGridInterpolator

    _, ny0, nx0 = targets_yx.shape
    lx, ly = domain.bounds[1] - domain.bounds[0], domain.bounds[3] - domain.bounds[2]
    # Native pixel-centre coordinates (match the geometry reconstruction).
    xs0 = (np.arange(nx0) + 0.5) / nx0 * lx + domain.bounds[0]
    ys0 = (np.arange(ny0) + 0.5) / ny0 * ly + domain.bounds[2]

    X, Y = domain.grid()  # (ny, nx)
    qpts = np.stack([Y.ravel(), X.ravel()], axis=1)  # (M, 2) in (y, x) order

    out = np.empty((targets_yx.shape[0], domain.ny, domain.nx), dtype=DTYPE)
    for c in range(targets_yx.shape[0]):
        interp = RegularGridInterpolator(
            (ys0, xs0), targets_yx[c].astype(np.float64),
            method="linear", bounds_error=False, fill_value=None,
        )
        out[c] = interp(qpts).reshape(domain.ny, domain.nx).astype(DTYPE)
    return out


def _sample_to_pair(
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    name: str,
    resolution: int,
    domain_m: tuple[float, float],
) -> tuple[FlowCase, FlowField]:
    """Convert one oriented ``(3, ny, nx)`` (dataX, dataY) sample to a pair."""
    lx, ly = domain_m
    domain = Domain(bounds=(0.0, lx, 0.0, ly), nx=resolution, ny=resolution)

    region = x_sample[1]  # (ny, nx) region labels (native resolution)
    geom = _reconstruct_geometry(region, name, domain_m)

    # Reynolds number from the constant inlet speed and the body chord (for the
    # log_re input channel); physically O(40) in this laminar regime.
    chord = max(float(geom.chord()), 1e-6)
    reynolds = float(DEEPCFD_U_INF * chord / DEEPCFD_NU)

    case = FlowCase(
        geometry=geom,
        bc=BoundaryConditions(u_inf=DEEPCFD_U_INF, aoa_deg=0.0, reynolds=reynolds),
        fluid=FluidProperties(density=1.0, kinematic_viscosity=DEEPCFD_NU),
        domain=domain,
        name=name,
    )

    # Geometry-consistent mask/sdf via the SAME functions encode_case uses, so the
    # network input mask matches the evaluation reference mask.
    from neuroforge.geometry.sdf import signed_distance, solid_mask

    sdf = signed_distance(geom, domain)
    mask = solid_mask(geom, domain)
    solid = mask < 0.5

    resampled = _resample_targets(y_sample, domain)  # (3, ny, nx): Ux, Uy, p
    u = np.where(solid, 0.0, resampled[0]).astype(DTYPE)
    v = np.where(solid, 0.0, resampled[1]).astype(DTYPE)
    p = resampled[2].astype(DTYPE)
    nut = np.zeros(domain.shape, dtype=DTYPE)  # laminar: eddy viscosity is exactly 0

    field = FlowField(
        domain=domain, u=u, v=v, p=p, nut=nut, mask=mask, sdf=sdf,
        meta={"source": "deepcfd", "case": name},
    )
    return case, field


def load_deepcfd(
    root: str = "data/deepcfd",
    resolution: int = 128,
    limit: int | None = None,
    indices: np.ndarray | list[int] | None = None,
    cache_dir: str | None = None,
    progress: bool = False,
) -> list[tuple[FlowCase, FlowField]]:
    """Load DeepCFD simulations as ``(FlowCase, FlowField)`` pairs.

    Parameters
    ----------
    root : str
        Directory containing ``dataX.pkl`` and ``dataY.pkl``.
    resolution : int
        Square structured-grid resolution to resample onto (default 128, to give
        the finite-difference residual stencils a well-defined ``dx``/``dy``).
    limit : int, optional
        Cap the number of samples loaded (ignored when ``indices`` is given).
    indices : array-like of int, optional
        Explicit sample indices to load (used for the train/test/OOD splits). If
        given, ``limit`` is ignored and the pairs are returned in this order.
    cache_dir : str, optional
        If given, the built pairs are pickled here keyed by
        ``(resolution, index-set)`` and reused on the next call.
    progress : bool
        Show a tqdm bar while building (best-effort; silent if tqdm missing).

    Returns
    -------
    list of (FlowCase, FlowField)
    """
    # Cache key from the resolution + a stable hash of the requested index set.
    cpath = None
    if cache_dir is not None:
        if indices is not None:
            tag = f"idx{abs(hash(tuple(int(i) for i in indices))) & 0xffffffff:08x}_n{len(indices)}"
        else:
            tag = f"limit{limit}"
        cpath = os.path.join(cache_dir, f"deepcfd_r{resolution}_{tag}.pkl")
        if os.path.exists(cpath):
            with open(cpath, "rb") as fh:
                return pickle.load(fh)

    X, Y = _load_raw(root)
    n_total = X.shape[0]

    if indices is not None:
        idx_list = [int(i) for i in indices]
    else:
        n = n_total if limit is None else min(int(limit), n_total)
        idx_list = list(range(n))

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            bar = tqdm(total=len(idx_list), desc="build deepcfd")
        except Exception:
            bar = None

    pairs: list[tuple[FlowCase, FlowField]] = []
    for i in idx_list:
        name = f"deepcfd_{i:04d}"
        pairs.append(
            _sample_to_pair(X[i], Y[i], name, resolution, DEEPCFD_DOMAIN_M)
        )
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    if cpath is not None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cpath, "wb") as fh:
            pickle.dump(pairs, fh)
    return pairs


def obstacle_areas(root: str = "data/deepcfd") -> np.ndarray:
    """Per-sample obstacle pixel area (region==0 count), for the OOD area split.

    Returns an ``(N,)`` int array. Cheap (only reads the region channel logic via
    the full load), so it is computed directly from the raw arrays.
    """
    X, _ = _load_raw(root)
    return (X[:, 1] == _REGION_OBSTACLE).sum(axis=(1, 2)).astype(np.int64)
