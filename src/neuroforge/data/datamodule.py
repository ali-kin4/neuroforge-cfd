"""Datasets, normalisation, and dataloaders — the bridge to training.

* :class:`Normalizer` — per-channel standardisation that works on both NumPy
  arrays and torch tensors, and (de)serialises into a checkpoint.
* :class:`FlowDataset` — wraps ``list[(FlowCase, FlowField)]`` and yields the
  encoded network input, the target stack, and the physical mask/SDF.
* :func:`build_dataloaders` — turns a :class:`DataConfig` into train/val loaders
  plus a fitted :class:`Normalizer`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neuroforge.core.config import DataConfig
from neuroforge.core.types import DTYPE, FlowCase, FlowField
from neuroforge.geometry.encode import encode_case

__all__ = ["Normalizer", "FlowDataset", "collate_flow", "build_dataloaders"]


def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


class Normalizer:
    """Per-channel mean/std standardisation for inputs (N_IN) and outputs (N_OUT).

    Statistics are stored as NumPy vectors and broadcast over the channel axis
    (assumed at position ``-3`` for ``(B, C, H, W)`` / ``(C, H, W)`` data). All
    methods accept and return the same array type they are given (NumPy or
    torch), preserving device and dtype for tensors.
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

    # -- fitting ---------------------------------------------------------- #
    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> Normalizer:
        """Fit statistics from stacked ``(N, N_IN, H, W)`` / ``(N, N_OUT, H, W)`` arrays."""
        inputs = np.asarray(inputs, np.float64)
        outputs = np.asarray(outputs, np.float64)
        self.mean_in = inputs.mean(axis=(0, 2, 3)).astype(DTYPE)
        self.std_in = inputs.std(axis=(0, 2, 3)).astype(DTYPE)
        self.mean_out = outputs.mean(axis=(0, 2, 3)).astype(DTYPE)
        self.std_out = outputs.std(axis=(0, 2, 3)).astype(DTYPE)
        # Guard against zero-variance channels (e.g. a constant mask).
        self.std_in = np.where(self.std_in < self.eps, 1.0, self.std_in).astype(DTYPE)
        self.std_out = np.where(self.std_out < self.eps, 1.0, self.std_out).astype(DTYPE)
        return self

    # -- helpers ---------------------------------------------------------- #
    def _bcast(self, vec: np.ndarray, like):
        """Reshape a ``(C,)`` stat vector to broadcast over ``(..., C, H, W)``."""
        v = vec.reshape(vec.shape[0], 1, 1)
        if _is_torch(like):
            return torch.as_tensor(v, dtype=like.dtype, device=like.device)
        return v.astype(like.dtype if hasattr(like, "dtype") else DTYPE)

    def norm_in(self, x):
        return (x - self._bcast(self.mean_in, x)) / self._bcast(self.std_in, x)

    def denorm_in(self, x):
        return x * self._bcast(self.std_in, x) + self._bcast(self.mean_in, x)

    def norm_out(self, y):
        return (y - self._bcast(self.mean_out, y)) / self._bcast(self.std_out, y)

    def denorm_out(self, y):
        return y * self._bcast(self.std_out, y) + self._bcast(self.mean_out, y)

    # -- (de)serialisation ----------------------------------------------- #
    def state_dict(self) -> dict:
        return {
            "mean_in": None if self.mean_in is None else self.mean_in.tolist(),
            "std_in": None if self.std_in is None else self.std_in.tolist(),
            "mean_out": None if self.mean_out is None else self.mean_out.tolist(),
            "std_out": None if self.std_out is None else self.std_out.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> Normalizer:
        def arr(v):
            return None if v is None else np.asarray(v, DTYPE)

        return cls(
            mean_in=arr(d.get("mean_in")),
            std_in=arr(d.get("std_in")),
            mean_out=arr(d.get("mean_out")),
            std_out=arr(d.get("std_out")),
            eps=float(d.get("eps", 1e-6)),
        )

    @property
    def fitted(self) -> bool:
        return self.mean_in is not None and self.mean_out is not None


class FlowDataset(Dataset):
    """A torch dataset over ``(FlowCase, FlowField)`` pairs.

    Inputs are encoded once at construction (the SDF/encoding is non-trivial, so
    we cache rather than recompute every epoch). ``__getitem__`` returns a dict
    with normalised ``input``/``target`` tensors and physical ``mask``/``sdf``.
    """

    def __init__(
        self,
        pairs: list[tuple[FlowCase, FlowField]],
        normalizer: Normalizer | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("FlowDataset received an empty list of pairs")
        self.cases: list[FlowCase] = []
        inputs, targets, masks, sdfs = [], [], [], []
        for case, field in pairs:
            inputs.append(encode_case(case))                  # (N_IN, H, W)
            targets.append(field.as_array())                  # (N_OUT, H, W)
            masks.append(field.mask[None].astype(DTYPE))       # (1, H, W)
            sdfs.append(field.sdf[None].astype(DTYPE))         # (1, H, W)
            self.cases.append(case)
        self.inputs = np.stack(inputs).astype(DTYPE)
        self.targets = np.stack(targets).astype(DTYPE)
        self.masks = np.stack(masks).astype(DTYPE)
        self.sdfs = np.stack(sdfs).astype(DTYPE)
        self.normalizer = normalizer

    def raw_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Un-normalised ``(inputs, targets)`` — used to fit a Normalizer."""
        return self.inputs, self.targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, i: int) -> dict:
        x = self.inputs[i]
        y = self.targets[i]
        if self.normalizer is not None and self.normalizer.fitted:
            x = self.normalizer.norm_in(x)
            y = self.normalizer.norm_out(y)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(x, DTYPE)),
            "target": torch.from_numpy(np.ascontiguousarray(y, DTYPE)),
            "mask": torch.from_numpy(np.ascontiguousarray(self.masks[i], DTYPE)),
            "sdf": torch.from_numpy(np.ascontiguousarray(self.sdfs[i], DTYPE)),
            "case": self.cases[i],
        }


def collate_flow(batch: list[dict]) -> dict:
    """Collate that stacks tensors and keeps ``case`` objects as a python list."""
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "sdf": torch.stack([b["sdf"] for b in batch]),
        "case": [b["case"] for b in batch],
    }


def _load_pairs(cfg: DataConfig) -> tuple[list, list]:
    """Build (train_pairs, val_pairs) for the configured source."""
    if cfg.source in ("synthetic", "synthetic_cylinder"):
        from neuroforge.data.synthetic import SyntheticRANS

        # ``synthetic`` -> airfoils (default), ``synthetic_cylinder`` -> non-lifting
        # cylinders. Both are generated in memory and are deterministic given
        # (seed, family, idx); nothing is cached to disk, so the two families
        # cannot collide. (DataConfig.source is a free string, so this needs no
        # change to the frozen config schema.)
        family = "cylinder" if cfg.source == "synthetic_cylinder" else "airfoil"
        gen = SyntheticRANS(resolution=cfg.resolution, seed=cfg.seed, family=family)
        train = [
            (c, gen.solve(c))
            for c in (gen.sample_case(i) for i in range(cfg.n_train))
        ]
        val = [
            (c, gen.solve(c))
            for c in (gen.sample_case(i) for i in range(cfg.n_train, cfg.n_train + cfg.n_val))
        ]
        return train, val

    if cfg.source == "airfrans":
        from neuroforge.data.airfrans_loader import load_airfrans

        # Thread resolution/limit/cache through so the rasterised grid matches
        # the physics-loss dx/dy and repeated sessions reuse the cache.
        train = load_airfrans(
            root=cfg.root, task=cfg.task, train=True,
            resolution=cfg.resolution, limit=cfg.n_train, cache_dir=cfg.cache_dir,
            download=cfg.download,
        )
        val = load_airfrans(
            root=cfg.root, task=cfg.task, train=False,
            resolution=cfg.resolution, limit=cfg.n_val, cache_dir=cfg.cache_dir,
            download=cfg.download,
        )
        return train, val

    raise ValueError(
        f"unknown data source '{cfg.source}' "
        "(expected 'synthetic', 'synthetic_cylinder', or 'airfrans')"
    )


def build_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Normalizer]:
    """Create train/val :class:`DataLoader`s and a fitted :class:`Normalizer`."""
    import gc

    train_pairs, val_pairs = _load_pairs(cfg)

    # Build the train dataset ONCE; fit the normaliser from its cached arrays
    # and attach it (normalisation is applied lazily in __getitem__), instead of
    # constructing FlowDataset twice (which doubled peak RAM and encode time).
    normalizer = Normalizer()
    train_ds = FlowDataset(train_pairs, normalizer=None)
    if cfg.normalize:
        xi, yi = train_ds.raw_arrays()
        normalizer.fit(xi, yi)
        train_ds.normalizer = normalizer
    val_ds = FlowDataset(val_pairs, normalizer=normalizer if cfg.normalize else None)

    # The datasets now hold their own stacked copies; free the source pairs.
    del train_pairs, val_pairs
    gc.collect()

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_flow, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_flow, drop_last=False,
    )
    return train_loader, val_loader, normalizer
