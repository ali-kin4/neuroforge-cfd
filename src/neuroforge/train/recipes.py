"""High-level training recipes — one call from dataset to evaluated checkpoint.

``train_recipe`` ties together :func:`build_dataloaders`, :class:`Trainer`
(backbone + residual-driven corrector) and field-error evaluation, returning
everything needed to build a :class:`~neuroforge.solver.engine.NeuroForgeEngine`.
It is the shared code path for the ``neuroforge train`` CLI, the AirfRANS script
(``scripts/train_airfrans.py``) and the Colab notebook, and it is GPU-aware via
``cfg.train.device`` (``'auto'`` selects CUDA when available).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from neuroforge.core.config import Config
from neuroforge.data.datamodule import build_dataloaders
from neuroforge.models.base import build_model
from neuroforge.models.correction import LocalCorrectionNet
from neuroforge.train.trainer import Trainer, _model_kwargs

__all__ = ["train_recipe", "evaluate_fields"]

# AirfRANS uses air at this kinematic viscosity (unit chord); synthetic ~ air too.
_AIRFRANS_NU = 1.56e-5
_DEFAULT_NU = 1.5e-5


def _is_cuda_oom(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory error (so we can advise a fix)."""
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _rebuild_loader(loader, batch_size: int, shuffle: bool):
    """A new DataLoader over the same dataset with a different batch size."""
    from torch.utils.data import DataLoader

    from neuroforge.data.datamodule import collate_flow

    return DataLoader(
        loader.dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=getattr(loader, "num_workers", 0), collate_fn=collate_flow,
    )


def _fit_backbone(trainer, train_loader, val_loader, out, ckpt_every, verbose):
    """Run ``trainer.fit`` with automatic batch-halving on CUDA OOM.

    On an out-of-memory error the GPU cache is cleared and training restarts at
    half the batch size, down to 1, so a too-large batch degrades gracefully
    instead of crashing the run. Returns ``(history, train_loader, val_loader)``
    with the (possibly smaller-batch) loaders so the corrector reuses them.
    """
    while True:
        try:
            history = trainer.fit(
                train_loader, val_loader, ckpt_path=out, ckpt_every=ckpt_every
            )
            return history, train_loader, val_loader
        except RuntimeError as exc:
            bs = train_loader.batch_size or 1
            if _is_cuda_oom(exc) and bs > 1:
                torch.cuda.empty_cache()
                new_bs = max(1, bs // 2)
                if verbose:
                    print(f"[recipe] CUDA OOM at batch_size={bs} -> retrying at {new_bs}")
                train_loader = _rebuild_loader(train_loader, new_bs, shuffle=True)
                val_loader = _rebuild_loader(val_loader, new_bs, shuffle=False)
            elif _is_cuda_oom(exc):
                raise RuntimeError(
                    "CUDA out of memory even at batch_size=1. Reduce "
                    "cfg.model.width/modes or cfg.data.resolution, then re-run."
                ) from exc
            else:
                raise


@torch.no_grad()
def evaluate_fields(model, loader, normalizer, device) -> dict[str, float]:
    """Masked relative-L2 field errors over a loader (physical units).

    Returns ``{'u','v','p','speed'}`` relative-L2 errors aggregated across the
    whole loader (fluid cells only), plus mean inference time per sample.
    """
    if loader is None:
        return {}
    model.eval()
    # Accumulate squared error / squared signal per channel across batches.
    num = {"u": 0.0, "v": 0.0, "p": 0.0, "speed": 0.0}
    den = {"u": 0.0, "v": 0.0, "p": 0.0, "speed": 0.0}
    n_samples = 0
    t_total = 0.0
    for batch in loader:
        inp = batch["input"].to(device)
        tgt_norm = batch["target"].to(device)
        mask = (batch["mask"].to(device) > 0.5).to(inp.dtype)  # (B,1,H,W)

        t0 = time.time()
        pred_norm = model(inp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total += time.time() - t0
        n_samples += inp.shape[0]

        pred = normalizer.denorm_out(pred_norm)
        tgt = normalizer.denorm_out(tgt_norm)
        m = mask
        for ci, name in ((0, "u"), (1, "v"), (2, "p")):
            d = (pred[:, ci:ci + 1] - tgt[:, ci:ci + 1]) * m
            s = tgt[:, ci:ci + 1] * m
            num[name] += float((d ** 2).sum())
            den[name] += float((s ** 2).sum())
        # speed magnitude
        ps = torch.sqrt(pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2)
        ts = torch.sqrt(tgt[:, 0:1] ** 2 + tgt[:, 1:2] ** 2)
        num["speed"] += float((((ps - ts) * m) ** 2).sum())
        den["speed"] += float(((ts * m) ** 2).sum())

    out = {k: float(np.sqrt(num[k] / max(den[k], 1e-12))) for k in num}
    out["infer_ms_per_sample"] = 1000.0 * t_total / max(n_samples, 1)
    return out


def train_recipe(
    cfg: Config,
    *,
    download: bool = False,
    corrector_epochs: int = 5,
    corrector_width: int = 32,
    corrector_layers: int = 3,
    ckpt_every: int = 5,
    out: str | None = None,
    nu: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a backbone (+corrector) per ``cfg`` and evaluate on the val split.

    Parameters
    ----------
    cfg : Config
        Full configuration. ``cfg.data.source`` selects ``'synthetic'`` or
        ``'airfrans'``; ``cfg.train.device='auto'`` uses CUDA if available.
    download : bool
        For the AirfRANS source, download the dataset first if missing.
    corrector_epochs : int
        Epochs for the second-stage corrector (0 disables it).
    out : str, optional
        If given, save the checkpoint (backbone + corrector + normaliser) here.
    nu : float, optional
        Laminar viscosity for the physics loss; defaults to the AirfRANS value
        for that source else a generic air value.

    Returns
    -------
    dict
        ``{history, corrector_history, val_errors, n_train, n_val, device,
        n_params, checkpoint, model, normalizer, corrector, trainer}``.
    """
    # Auto-download AirfRANS lazily, only on a cache miss (so a session that
    # already has the rasterised cache never re-fetches the multi-GB dataset).
    if cfg.data.source == "airfrans" and download:
        cfg.data.download = True

    if nu is None:
        nu = _AIRFRANS_NU if cfg.data.source == "airfrans" else _DEFAULT_NU

    if verbose:
        print(f"[recipe] building dataloaders (source={cfg.data.source}, "
              f"res={cfg.data.resolution}, n_train={cfg.data.n_train}, n_val={cfg.data.n_val}) ...")
    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)

    model = build_model(cfg.model.name, **_model_kwargs(cfg.model))
    if verbose:
        print(f"[recipe] model={cfg.model.name} params={model.num_params():,}")

    trainer = Trainer(model, cfg, normalizer, nu=nu)
    if verbose:
        print(f"[recipe] training on {trainer.device} for {cfg.train.epochs} epochs ...")
    history, train_loader, val_loader = _fit_backbone(
        trainer, train_loader, val_loader, out, ckpt_every, verbose
    )

    if trainer.device.type == "cuda":
        torch.cuda.empty_cache()

    corrector = None
    corrector_history = None
    if corrector_epochs and corrector_epochs > 0:
        corrector = LocalCorrectionNet(width=corrector_width, n_layers=corrector_layers)
        if verbose:
            print(f"[recipe] training corrector for {corrector_epochs} epochs ...")
        try:
            corrector_history = trainer.fit_corrector(
                train_loader, corrector, epochs=corrector_epochs
            )
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                raise RuntimeError(
                    "CUDA out of memory while training the corrector. Lower "
                    "corrector_width or cfg.data.batch_size, then re-run."
                ) from exc
            raise
        if trainer.device.type == "cuda":
            torch.cuda.empty_cache()

    if verbose:
        print("[recipe] evaluating field errors on val split ...")
    val_errors = evaluate_fields(model, val_loader, normalizer, trainer.device)

    checkpoint = None
    if out is not None:
        trainer.save(out, corrector=corrector)
        checkpoint = out
        if verbose:
            print(f"[recipe] saved checkpoint -> {out}")

    if verbose and val_errors:
        print("[recipe] val rel-L2: " + ", ".join(
            f"{k}={v:.3f}" for k, v in val_errors.items() if k in ("u", "v", "p", "speed")
        ) + f"  ({val_errors.get('infer_ms_per_sample', 0):.1f} ms/sample)")

    return {
        "history": history,
        "corrector_history": corrector_history,
        "val_errors": val_errors,
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset) if val_loader is not None else 0,
        "device": str(trainer.device),
        "n_params": int(model.num_params()),
        "checkpoint": checkpoint,
        "model": model,
        "normalizer": normalizer,
        "corrector": corrector,
        "trainer": trainer,
    }
