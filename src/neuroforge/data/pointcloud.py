"""Native AirfRANS point-cloud datamodule for point/slice-attention baselines.

Unlike :mod:`neuroforge.data.airfrans_loader` (which **rasterises** each
simulation onto a structured grid for the grid backbones), this module keeps the
AirfRANS simulations as their **native irregular point clouds** — exactly what a
Transolver / GINO / MeshGraphNet baseline must consume so the comparison in the
paper's Table 2 is on the representation the baselines were designed for (see
``docs/EXPERIMENTS.md`` and ``docs/baseline_plan.md``).

AirfRANS raw arrays (airfrans 0.1.2) are ``(M, 12)``::

    [0,1] x, y           position
    [2,3] u_in_x, u_in_y inlet velocity (constant per sim)
    [4]   sdf            distance to airfoil (>= 0)
    [5,6] n_x, n_y       surface normals (0 away from the wall)
    [7,8] u, v           target velocity
    [9]   p              target kinematic pressure p/rho
    [10]  nut            target turbulent kinematic viscosity
    [11]  on_airfoil     boolean flag (1 on the body surface)

Per-point **input** features (``F_IN = 7``): ``x, y, u_in_x, u_in_y, sdf, n_x,
n_y``. Per-point **targets** (``F_OUT = 4``): ``u, v, p, nut`` — the same output
channels as the grid contract, so predictions can be scored by the shared
``evaluate_cases`` metrics after rasterisation (see
:func:`neuroforge.models.baselines.transolver_adapter`).

The :class:`PointNormalizer` fits per-feature mean/std on the **train split
only** (no leakage) and is persisted for use on the test split.
"""

from __future__ import annotations

import os

import numpy as np

from neuroforge.core.types import DTYPE
from neuroforge.data.airfrans_loader import _require_airfrans, _resolve_data_root

__all__ = [
    "PointCloudCase",
    "PointNormalizer",
    "load_airfrans_pointclouds",
    "POINT_IN_FEATURES",
    "POINT_OUT_FEATURES",
]

# Raw-array column slices (airfrans 0.1.2 layout, (M, 12)).
_POS = slice(0, 2)
_UIN = slice(2, 4)
_SDF = 4
_NRM = slice(5, 7)
_TGT = slice(7, 11)  # u, v, p, nut

#: Per-point input feature names, in order.
POINT_IN_FEATURES: tuple[str, ...] = ("x", "y", "u_in_x", "u_in_y", "sdf", "n_x", "n_y")
#: Per-point target names, in order (matches OUTPUT_CHANNELS).
POINT_OUT_FEATURES: tuple[str, ...] = ("u", "v", "p", "nut")

F_IN = len(POINT_IN_FEATURES)
F_OUT = len(POINT_OUT_FEATURES)


class PointCloudCase:
    """One AirfRANS simulation as a native point cloud, keyed by name.

    Attributes
    ----------
    name : str
        Simulation name (matches the rasterised ``FlowCase.name`` for alignment).
    features : numpy.ndarray
        ``(M, F_IN)`` per-point input features (raw, un-normalised).
    targets : numpy.ndarray
        ``(M, F_OUT)`` per-point targets ``u, v, p, nut`` (raw).
    pos : numpy.ndarray
        ``(M, 2)`` point positions (kept for rasterising predictions back to grid).
    """

    __slots__ = ("name", "features", "targets", "pos")

    def __init__(self, name: str, features: np.ndarray, targets: np.ndarray, pos: np.ndarray) -> None:
        self.name = str(name)
        self.features = np.asarray(features, DTYPE)
        self.targets = np.asarray(targets, DTYPE)
        self.pos = np.asarray(pos, DTYPE)

    @property
    def n_points(self) -> int:
        return int(self.features.shape[0])


def _array_to_pointcase(data: np.ndarray, name: str) -> PointCloudCase:
    """Slice one raw ``(M, 12)`` AirfRANS array into a :class:`PointCloudCase`."""
    data = np.asarray(data, np.float64)
    pos = data[:, _POS]
    feats = np.concatenate(
        [data[:, _POS], data[:, _UIN], data[:, _SDF:_SDF + 1], data[:, _NRM]],
        axis=1,
    )  # (M, 7): x, y, u_in_x, u_in_y, sdf, n_x, n_y
    targets = data[:, _TGT]  # (M, 4): u, v, p, nut
    return PointCloudCase(name, feats, targets, pos)


class PointNormalizer:
    """Per-feature z-score standardisation for point features and targets.

    Fit on the **train split only**; ``transform_*`` is then applied to both
    train and test (no leakage). Mirrors the grid :class:`Normalizer` policy.
    """

    def __init__(
        self,
        mean_in: np.ndarray | None = None,
        std_in: np.ndarray | None = None,
        mean_out: np.ndarray | None = None,
        std_out: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.eps = float(eps)
        self.mean_in = None if mean_in is None else np.asarray(mean_in, DTYPE)
        self.std_in = None if std_in is None else np.asarray(std_in, DTYPE)
        self.mean_out = None if mean_out is None else np.asarray(mean_out, DTYPE)
        self.std_out = None if std_out is None else np.asarray(std_out, DTYPE)

    def fit(self, cases: list[PointCloudCase]) -> PointNormalizer:
        """Fit feature/target statistics by pooling all points across train cases.

        Uses a streaming (sum, sum-of-squares) pass so we never materialise the
        full concatenated point set in memory.
        """
        n = 0
        s_in = np.zeros(F_IN, np.float64)
        ss_in = np.zeros(F_IN, np.float64)
        s_out = np.zeros(F_OUT, np.float64)
        ss_out = np.zeros(F_OUT, np.float64)
        for c in cases:
            f = c.features.astype(np.float64)
            t = c.targets.astype(np.float64)
            n += f.shape[0]
            s_in += f.sum(0)
            ss_in += (f * f).sum(0)
            s_out += t.sum(0)
            ss_out += (t * t).sum(0)
        n = max(n, 1)
        self.mean_in = (s_in / n).astype(DTYPE)
        self.mean_out = (s_out / n).astype(DTYPE)
        var_in = np.maximum(ss_in / n - (s_in / n) ** 2, 0.0)
        var_out = np.maximum(ss_out / n - (s_out / n) ** 2, 0.0)
        std_in = np.sqrt(var_in)
        std_out = np.sqrt(var_out)
        self.std_in = np.where(std_in < self.eps, 1.0, std_in).astype(DTYPE)
        self.std_out = np.where(std_out < self.eps, 1.0, std_out).astype(DTYPE)
        return self

    def transform_in(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, DTYPE) - self.mean_in) / self.std_in).astype(DTYPE)

    def transform_out(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, DTYPE) - self.mean_out) / self.std_out).astype(DTYPE)

    def inverse_out(self, y) -> np.ndarray:
        """De-standardise model outputs back to physical units (numpy or torch).

        Accepts a torch tensor (returns torch) or numpy (returns numpy).
        """
        try:
            import torch

            if isinstance(y, torch.Tensor):
                m = torch.as_tensor(self.mean_out, dtype=y.dtype, device=y.device)
                s = torch.as_tensor(self.std_out, dtype=y.dtype, device=y.device)
                return y * s + m
        except Exception:
            pass
        return (np.asarray(y, DTYPE) * self.std_out + self.mean_out).astype(DTYPE)

    @property
    def fitted(self) -> bool:
        return self.mean_in is not None and self.mean_out is not None

    def state_dict(self) -> dict:
        def lst(a):
            return None if a is None else a.tolist()

        return {
            "mean_in": lst(self.mean_in), "std_in": lst(self.std_in),
            "mean_out": lst(self.mean_out), "std_out": lst(self.std_out),
            "eps": self.eps,
        }


def _pc_cache_path(cache_dir: str, task: str, train: bool, limit) -> str:
    split = "train" if train else "test"
    return os.path.join(cache_dir, f"airfrans_pc_{task}_{split}_n{limit}.pkl")


def load_airfrans_pointclouds(
    root: str = "data",
    task: str = "full",
    train: bool = True,
    limit: int | None = None,
    cache_dir: str | None = None,
    progress: bool = True,
    download: bool = False,
) -> list[PointCloudCase]:
    """Load AirfRANS simulations as native :class:`PointCloudCase` objects.

    The order matches ``af.dataset.load`` (and therefore the rasterised
    ``load_airfrans`` order, since both consume the same backend), so the i-th
    point cloud aligns with the i-th cached ``(FlowCase, FlowField)`` pair by name.

    Parameters mirror :func:`neuroforge.data.airfrans_loader.load_airfrans`.
    Caching pickles the lightweight feature/target arrays (no rasterisation).
    """
    if cache_dir is not None:
        cpath = _pc_cache_path(cache_dir, task, train, limit)
        if os.path.exists(cpath):
            import pickle

            with open(cpath, "rb") as fh:
                return pickle.load(fh)

    if download:
        from neuroforge.data.airfrans_loader import download_airfrans

        download_airfrans(root)

    af = _require_airfrans()
    data_root = _resolve_data_root(root)
    if data_root is None:
        raise FileNotFoundError(
            f"AirfRANS manifest.json not found under '{root}' or '{root}/Dataset'. "
            "Pass download=True, or run download_airfrans(root, force=True)."
        )
    dataset, names = af.dataset.load(root=data_root, task=task, train=train)
    n = len(names) if limit is None else min(limit, len(names))

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            bar = tqdm(total=n, desc="load airfrans point clouds")
        except Exception:
            bar = None

    cases: list[PointCloudCase] = []
    for i in range(n):
        cases.append(_array_to_pointcase(np.asarray(dataset[i]), str(names[i])))
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    del dataset, names
    import gc

    gc.collect()

    if cache_dir is not None:
        import pickle

        os.makedirs(cache_dir, exist_ok=True)
        with open(_pc_cache_path(cache_dir, task, train, limit), "wb") as fh:
            pickle.dump(cases, fh)
    return cases
