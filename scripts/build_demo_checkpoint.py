"""Build the bundled demo checkpoint shipped with the package.

Trains a modest FNO backbone + local correction network on the synthetic
pseudo-RANS dataset and saves it to ``src/neuroforge/assets/demo.pt`` so that
``NeuroForgeEngine.pretrained()`` / ``neuroforge.demo()`` are instant on first
use (no training required by the end user).

Run from the repo root:  PYTHONPATH=src python scripts/build_demo_checkpoint.py
"""

from __future__ import annotations

import os
import time

import neuroforge  # noqa: F401  (sets the math-library thread caps on import)
from neuroforge.core.config import Config, DataConfig
from neuroforge.data.datamodule import build_dataloaders
from neuroforge.models.base import build_model
from neuroforge.models.correction import LocalCorrectionNet
from neuroforge.train.trainer import Trainer

OUT = os.path.join("src", "neuroforge", "assets", "demo.pt")


def main() -> None:
    t0 = time.time()
    cfg = Config()
    cfg.data = DataConfig(
        source="synthetic", resolution=64, n_train=24, n_val=6, batch_size=4, seed=7
    )
    cfg.train.epochs = 25
    cfg.train.lr = 1e-3
    cfg.train.log_every = 0

    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)
    print(f"[{time.time()-t0:5.1f}s] data ready ({len(train_loader.dataset)} train)")

    model = build_model("fno", width=32, n_layers=4, modes=14)
    trainer = Trainer(model, cfg, normalizer, nu=1.5e-5)
    trainer.fit(train_loader, val_loader)
    print(f"[{time.time()-t0:5.1f}s] backbone trained ({model.num_params():,} params)")

    corrector = LocalCorrectionNet(width=24, n_layers=3)
    trainer.fit_corrector(train_loader, corrector, epochs=8)
    print(f"[{time.time()-t0:5.1f}s] corrector trained")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    trainer.save(OUT, corrector=corrector)
    size = os.path.getsize(OUT) / 1e6
    print(f"[{time.time()-t0:5.1f}s] saved {OUT} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
