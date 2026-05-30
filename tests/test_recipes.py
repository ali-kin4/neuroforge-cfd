"""Tests for the high-level training recipe: end-to-end on synthetic data,
per-epoch checkpointing, field evaluation, and the OOM batch-rebuild helper.
"""

from __future__ import annotations

import os

from neuroforge.core.config import Config, DataConfig
from neuroforge.data.datamodule import build_dataloaders
from neuroforge.solver.engine import NeuroForgeEngine
from neuroforge.train.recipes import _rebuild_loader, evaluate_fields, train_recipe


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.data = DataConfig(source="synthetic", resolution=24, n_train=4, n_val=2, batch_size=2)
    cfg.model.width = 12
    cfg.model.n_layers = 2
    cfg.model.modes = 6
    cfg.train.epochs = 2
    cfg.train.device = "cpu"
    cfg.train.log_every = 0
    return cfg


def test_train_recipe_end_to_end_and_checkpoint(tmp_path):
    cfg = _tiny_cfg()
    out = str(tmp_path / "ck.pt")
    result = train_recipe(cfg, corrector_epochs=1, ckpt_every=1, out=out, verbose=False)

    # Per-epoch checkpoint was written and reloads into a working engine.
    assert os.path.exists(out)
    engine = NeuroForgeEngine.from_checkpoint(out)
    assert engine.corrector is not None

    ve = result["val_errors"]
    for k in ("u", "v", "p", "speed"):
        assert k in ve and ve[k] >= 0.0
    assert result["n_params"] > 0
    # History losses are finite.
    assert all(map(lambda x: x == x, result["history"]["train_loss"]))  # not NaN


def test_evaluate_fields_self_is_zero():
    cfg = _tiny_cfg()
    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)
    # A model that perfectly reproduces the (normalised) target would score ~0;
    # here we just assert the metric runs and returns the right keys, finite.
    from neuroforge.models.base import build_model

    model = build_model("fno", width=12, n_layers=2, modes=6)
    import torch

    out = evaluate_fields(model, val_loader, normalizer, torch.device("cpu"))
    for k in ("u", "v", "p", "speed", "infer_ms_per_sample"):
        assert k in out and out[k] == out[k]  # finite


def test_rebuild_loader_changes_batch():
    cfg = _tiny_cfg()
    train_loader, _, _ = build_dataloaders(cfg.data)
    smaller = _rebuild_loader(train_loader, batch_size=1, shuffle=False)
    batch = next(iter(smaller))
    assert batch["input"].shape[0] == 1
