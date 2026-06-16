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

__BENCH_NOTE__ = """\
Backend choice for the rasterise loop (Change B): ThreadPoolExecutor.

We benchmarked ProcessPool vs ThreadPool as instructed. On REAL AirfRANS data
(task='full', 16 cases, res 128, M~180k points each) the picture differs sharply
from small synthetic clouds:

  ProcessPool: OOM-crashes (BrokenProcessPool) -- the full split is already
    resident in the parent (af.dataset.load loads ALL cases; `limit` only caps
    rasterisation), and each spawned worker re-imports torch (~0.5-1 GB x N), so
    the box runs out of memory. Its serial fallback then restarts the whole
    batch -- the opposite of crash-safe.
  ThreadPool: shares the parent's already-resident arrays (no pickling, no
    per-worker torch), and SciPy's Qhull / LinearNDInterpolator release the GIL,
    so it speeds up the loop with a bounded, predictable memory footprint.

Threads are therefore the correct, crash-safe choice for this memory profile.
We still bound concurrency and fall back to a serial loop on any pool failure.
"""

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


def _resolve_data_root(root: str) -> str | None:
    """Directory that actually contains ``manifest.json``.

    ``af.dataset.load`` reads ``<root>/manifest.json`` plus the simulation
    folders. ``af.dataset.download`` unzips ``Dataset.zip`` into a ``Dataset/``
    subfolder, so the manifest usually lives one level down. We accept either
    layout and return whichever directory holds the manifest (or ``None``).
    """
    for cand in (root, os.path.join(root, "Dataset")):
        if os.path.isfile(os.path.join(cand, "manifest.json")):
            return cand
    return None


def _airfrans_present(root: str) -> bool:
    """True if an extracted AirfRANS dataset (with manifest) exists under ``root``."""
    return _resolve_data_root(root) is not None


def download_airfrans(root: str = "data", force: bool = False) -> str:
    """Download + unzip the AirfRANS dataset under ``root`` (idempotent).

    Skips the (multi-GB) download when the dataset is already extracted under
    ``root`` unless ``force=True``. Returns ``root``.
    """
    os.makedirs(root, exist_ok=True)
    # Skip (and avoid even importing airfrans) when the data is already here.
    if not force and _airfrans_present(root):
        return root
    af = _require_airfrans()
    af.dataset.download(root=root, unzip=True)
    return root


def _order_surface_loop(points: np.ndarray) -> np.ndarray:
    """Order scattered airfoil-surface points into a single non-crossing CCW loop.

    Splits points into upper/lower surfaces by the sign of their offset from the
    leading-edge->trailing-edge chord line, orders the upper surface LE->TE and
    the lower surface TE->LE by streamwise position, then concatenates. This
    traces one continuous loop even on thin sections where the two surfaces are
    sub-cell apart — unlike a greedy nearest-neighbour walk, which zig-zags
    between the surfaces and self-crosses, corrupting the surface integral.
    """
    pts = np.asarray(points, np.float64)
    n = pts.shape[0]
    if n < 4:
        return pts.astype(DTYPE)

    # Chord axis: leading edge (min x) -> trailing edge (max x).
    le = pts[int(np.argmin(pts[:, 0]))]
    te = pts[int(np.argmax(pts[:, 0]))]
    axis = te - le
    chord = float(np.hypot(*axis))
    if chord < 1e-12:
        return pts.astype(DTYPE)
    axis = axis / chord

    rel = pts - le
    s = rel @ axis                                        # streamwise position
    perp = rel[:, 0] * (-axis[1]) + rel[:, 1] * axis[0]   # signed offset (side)

    upper = perp >= 0
    if upper.any() and (~upper).any():
        up = pts[upper][np.argsort(s[upper])]            # LE -> TE
        lo = pts[~upper][np.argsort(-s[~upper])]         # TE -> LE
        loop = np.vstack([up, lo])
    else:  # degenerate split -> fall back to streamwise sort
        loop = pts[np.argsort(s)]

    # Orient counter-clockwise (positive shoelace area).
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
    u_inf_vec = uin.mean(axis=0) if uin.size else np.array([1.0, 0.0])
    # Guard against an all-non-finite or zero inlet column (default to unit +x).
    if not np.all(np.isfinite(u_inf_vec)) or float(np.hypot(*u_inf_vec)) < 1e-9:
        u_inf_vec = np.array([1.0, 0.0])
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


def _sim_to_pair_packed(arg: tuple[np.ndarray, str, int]) -> tuple[FlowCase, FlowField]:
    """Wrapper of :func:`_sim_to_pair` taking a single packed ``(array, name,
    resolution)`` tuple, for use as the worker callable in the thread pool.

    ``_sim_to_pair`` is pure given its array input, so it is thread-safe; the
    package already caps BLAS/OMP/MKL to 1 thread (see ``neuroforge/__init__``)
    so the SciPy calls inside don't oversubscribe across pool threads.
    """
    data, name, resolution = arg
    return _sim_to_pair(np.asarray(data), str(name), int(resolution))


def _resolve_workers(n_workers: int | None) -> int:
    """Resolve the worker count from arg, then env, then a CPU-based default."""
    if n_workers is None:
        env = os.environ.get("NEUROFORGE_RASTERIZE_WORKERS")
        if env is not None and env.strip():
            try:
                n_workers = int(env)
            except ValueError:
                n_workers = None
    if n_workers is None:
        n_workers = min(16, max(1, (os.cpu_count() or 4) - 4))
    return int(n_workers)


def _rasterize_pairs(
    dataset, names, n: int, resolution: int, n_workers: int, progress: bool
) -> list[tuple[FlowCase, FlowField]]:
    """Rasterise ``n`` simulations into ordered ``(FlowCase, FlowField)`` pairs.

    Uses a thread pool (see ``__BENCH_NOTE__`` for why threads beat processes on
    this memory profile) with a *bounded* sliding window of in-flight futures, so
    only a few results accumulate ahead of consumption and the tqdm bar advances
    on completion. Results are reassembled in input order. Any pool failure falls
    back to a plain serial loop.
    """
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            bar = tqdm(total=n, desc="rasterise airfrans")
        except Exception:
            bar = None

    def _serial() -> list[tuple[FlowCase, FlowField]]:
        out: list[tuple[FlowCase, FlowField]] = []
        for i in range(n):
            out.append(_sim_to_pair(np.asarray(dataset[i]), str(names[i]), resolution))
            if bar is not None:
                bar.update(1)
        return out

    # Tiny n or a single worker: pool overhead isn't worth it.
    if n_workers <= 1 or n < 4:
        try:
            return _serial()
        finally:
            if bar is not None:
                bar.close()

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    results: list[tuple[FlowCase, FlowField] | None] = [None] * n
    max_inflight = max(1, 2 * n_workers)  # bound how far rasterisation runs ahead
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_idx: dict = {}
            next_submit = 0
            done_count = 0
            # Prime the window.
            while next_submit < n and len(future_to_idx) < max_inflight:
                arg = (np.asarray(dataset[next_submit]), str(names[next_submit]), resolution)
                future_to_idx[ex.submit(_sim_to_pair_packed, arg)] = next_submit
                next_submit += 1
            while future_to_idx:
                done, _ = wait(list(future_to_idx), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = future_to_idx.pop(fut)
                    results[idx] = fut.result()  # re-raises worker exceptions
                    done_count += 1
                    if bar is not None:
                        bar.update(1)
                    # Refill the window to keep workers busy.
                    if next_submit < n:
                        arg = (np.asarray(dataset[next_submit]), str(names[next_submit]), resolution)
                        future_to_idx[ex.submit(_sim_to_pair_packed, arg)] = next_submit
                        next_submit += 1
        if any(r is None for r in results):  # defensive: should not happen
            raise RuntimeError("rasterise pool returned incomplete results")
        return [r for r in results if r is not None]
    except Exception as exc:  # pool crash / pickling / OOM -> robust serial path
        import warnings

        warnings.warn(
            f"parallel rasterise failed ({exc!r}); falling back to serial.",
            RuntimeWarning,
            stacklevel=2,
        )
        if bar is not None:
            bar.reset(total=n)
        return _serial()
    finally:
        if bar is not None:
            bar.close()


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
    download: bool = False,
    n_workers: int | None = None,
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
    n_workers : int, optional
        Number of worker processes for the rasterise loop. ``None`` (default)
        reads the ``NEUROFORGE_RASTERIZE_WORKERS`` env var, else uses
        ``min(16, max(1, cpu_count - 4))``. ``<= 1`` forces the serial path.
    """
    # Fast path: reuse a cached rasterisation.
    if cache_dir is not None:
        cpath = _cache_path(cache_dir, task, train, resolution, limit)
        if os.path.exists(cpath):
            import pickle

            with open(cpath, "rb") as fh:
                return pickle.load(fh)

    # Cache miss: we need the raw dataset. Optionally fetch it (idempotent) so a
    # fresh session with no local copy doesn't fail with a missing manifest.json.
    if download:
        download_airfrans(root)

    af = _require_airfrans()
    # af.dataset.load wants the directory that holds manifest.json — which is
    # usually <root>/Dataset after the zip is extracted, not <root> itself.
    data_root = _resolve_data_root(root)
    if data_root is None:
        raise FileNotFoundError(
            f"AirfRANS manifest.json not found under '{root}' or '{root}/Dataset'. "
            "Pass download=True, or run download_airfrans(root, force=True)."
        )
    dataset, names = af.dataset.load(root=data_root, task=task, train=train)
    n = len(names) if limit is None else min(limit, len(names))

    # Parallel (or serial) rasterise — order-preserving. See __BENCH_NOTE__.
    pairs = _rasterize_pairs(
        dataset, names, n, resolution, _resolve_workers(n_workers), progress
    )

    # The raw split (all sims' point clouds) can be several GB; release it now
    # that we have the compact rasterised grids.
    del dataset, names
    import gc
    gc.collect()

    if cache_dir is not None:
        import pickle

        os.makedirs(cache_dir, exist_ok=True)
        with open(_cache_path(cache_dir, task, train, resolution, limit), "wb") as fh:
            pickle.dump(pairs, fh)
    return pairs
