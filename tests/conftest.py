"""Shared pytest fixtures and path setup for the NeuroForge CFD test suite.

This conftest puts the repo ``src`` directory on ``sys.path`` so the suite runs
from a checkout without an editable install, registers the ``slow`` marker, and
provides small, fast, deterministic fixtures (tiny grids, a single tiny trained
engine reused across the solver/engine tests).
"""

from __future__ import annotations

import os
import sys

import pytest

# --------------------------------------------------------------------------- #
# Path setup: make `import neuroforge` work from the repo root with no install.
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# Also expose the repo root so `import benchmarks` works.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import neuroforge eagerly: it caps BLAS/OMP/MKL threads to 1 at import, which
# is what keeps the suite fast on this box.
import neuroforge  # noqa: E402,F401


def _seed_everything(seed: int = 0) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_seed_everything(0)


# --------------------------------------------------------------------------- #
# Marker registration (mirrors pyproject; harmless if duplicated).
# --------------------------------------------------------------------------- #
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselected by default)"
    )


@pytest.fixture(autouse=True)
def _deterministic():
    """Re-seed before every test for determinism."""
    _seed_everything(0)
    yield


# --------------------------------------------------------------------------- #
# Lightweight shared fixtures.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def tiny_case():
    """A small NACA airfoil FlowCase (resolution 32)."""
    from neuroforge import FlowCase

    return FlowCase.from_airfoil(
        "naca2412", aoa=4.0, reynolds=3.0e6, u_inf=30.0, resolution=32
    )


@pytest.fixture(scope="session")
def synthetic_field(tiny_case):
    """A synthetic pseudo-RANS solution for ``tiny_case`` (resolution 32)."""
    from neuroforge.data.synthetic import SyntheticRANS

    gen = SyntheticRANS(resolution=32, seed=0)
    return gen.solve(tiny_case)


@pytest.fixture(scope="session")
def trained_engine():
    """Train ONE tiny FNO + corrector on synthetic data and wrap an engine.

    Session-scoped so the (small) training cost is paid exactly once and reused
    across solver / engine tests. Sized to stay well under a few seconds:
    resolution 24, 4 train / 2 val cases, 2 backbone epochs (FNO width 12), and
    2 corrector epochs.
    """
    from neuroforge.core.config import Config, DataConfig
    from neuroforge.data.datamodule import build_dataloaders
    from neuroforge.models.base import build_model
    from neuroforge.models.correction import LocalCorrectionNet
    from neuroforge.physics.residuals import PhysicsChecker
    from neuroforge.solver.engine import NeuroForgeEngine, Predictor

    _seed_everything(0)

    cfg = Config()
    cfg.data = DataConfig(
        source="synthetic", resolution=24, n_train=4, n_val=2, batch_size=2
    )
    cfg.train.epochs = 2
    cfg.train.lr = 1e-3
    cfg.train.log_every = 0

    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)
    model = build_model("fno", width=12, n_layers=2, modes=8)

    from neuroforge.train.trainer import Trainer

    trainer = Trainer(model, cfg, normalizer, nu=1.5e-5)
    trainer.fit(train_loader, val_loader)

    corrector = LocalCorrectionNet(width=12, n_layers=2)
    trainer.fit_corrector(train_loader, corrector, epochs=2)

    predictor = Predictor(model, normalizer, device="cpu")
    checker = PhysicsChecker(cfg.physics)
    engine = NeuroForgeEngine(predictor, checker, corrector=corrector, config=cfg)
    # Stash the trainer + loaders so checkpoint tests can reuse the trained pieces.
    engine._test_trainer = trainer
    engine._test_corrector = corrector
    return engine
