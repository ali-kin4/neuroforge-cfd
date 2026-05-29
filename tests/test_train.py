"""Training tests: Trainer.fit history, save/load round-trip, CompositeLoss."""

from __future__ import annotations

import os

import numpy as np
import torch

from neuroforge.core.config import Config, DataConfig
from neuroforge.data.datamodule import build_dataloaders
from neuroforge.models.base import build_model
from neuroforge.solver.engine import NeuroForgeEngine
from neuroforge.train.losses import CompositeLoss
from neuroforge.train.trainer import Trainer


def _tiny_setup():
    cfg = Config()
    cfg.data = DataConfig(
        source="synthetic", resolution=24, n_train=4, n_val=2, batch_size=2
    )
    cfg.train.epochs = 2
    cfg.train.log_every = 0
    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)
    return cfg, train_loader, val_loader, normalizer


def test_trainer_fit_finite_history():
    cfg, train_loader, val_loader, normalizer = _tiny_setup()
    model = build_model("fno", width=12, n_layers=2, modes=8)
    trainer = Trainer(model, cfg, normalizer, nu=1.5e-5)
    history = trainer.fit(train_loader, val_loader)
    assert "train_loss" in history and "val_loss" in history
    assert len(history["train_loss"]) == cfg.train.epochs
    for v in history["train_loss"]:
        assert np.isfinite(v)
    for v in history["val_loss"]:
        assert np.isfinite(v)


def test_trainer_save_load_and_from_checkpoint(tmp_path):
    cfg, train_loader, val_loader, normalizer = _tiny_setup()
    model = build_model("fno", width=12, n_layers=2, modes=8)
    trainer = Trainer(model, cfg, normalizer, nu=1.5e-5)
    trainer.fit(train_loader, val_loader)

    ckpt = os.path.join(tmp_path, "model.pt")
    trainer.save(ckpt)
    assert os.path.exists(ckpt)

    # Trainer.load round-trips weights.
    loaded, norm2, model_cfg = Trainer.load(ckpt)
    assert model_cfg.name == "fno"
    x = torch.randn(1, model_cfg.in_channels, 24, 24)
    with torch.no_grad():
        a = model(x)
        b = loaded(x)
    assert torch.allclose(a, b, atol=1e-5)

    # Engine loads the same checkpoint.
    engine = NeuroForgeEngine.from_checkpoint(ckpt)
    assert engine.predictor is not None
    assert engine.checker is not None


def test_composite_loss_finite():
    cfg, train_loader, _, normalizer = _tiny_setup()
    loss_fn = CompositeLoss(cfg, normalizer, nu=1.5e-5)
    model = build_model("fno", width=12, n_layers=2, modes=8)
    batch = next(iter(train_loader))
    pred = model(batch["input"])
    total, parts = loss_fn(pred, batch["target"], batch["input"], batch["mask"])
    assert torch.isfinite(total)
    assert set(parts) == {"loss", "data", "physics", "bc"}
    for v in parts.values():
        assert np.isfinite(v)
