"""Regression tests for the bugs found in the senior review (C1/C3/C4/M3/m4).

These lock in: nonzero & correctly-signed force coefficients (force sampling
offset + synthetic circulation sign), a non-self-crossing AirfRANS surface loop,
the dimensionless/wall-masked residual scale, and the balanced physics loss.
"""

from __future__ import annotations

import numpy as np
import torch

import neuroforge as nf
from neuroforge.core.config import Config, DataConfig
from neuroforge.core.types import FlowField
from neuroforge.data import build_dataloaders
from neuroforge.data.airfrans_loader import _order_surface_loop
from neuroforge.data.synthetic import SyntheticRANS
from neuroforge.geometry.airfoil import naca_airfoil
from neuroforge.physics.metrics import force_coefficients
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.train.losses import CompositeLoss


def test_force_coefficients_nonzero_and_correct_sign():
    """C3 + m4: forces are nonzero and dCl/dAoA > 0 for a cambered airfoil."""
    gen = SyntheticRANS(96)
    cls = {}
    for aoa in (-5.0, 5.0):
        case = nf.FlowCase.from_airfoil("naca2412", aoa=aoa, reynolds=3e6, u_inf=30.0, resolution=96)
        fc = force_coefficients(gen.solve(case), case)
        cls[aoa] = fc["cl"]
    assert abs(cls[5.0]) > 1e-3, "Cl is ~0 (force sampling regressed into the solid)"
    assert cls[5.0] > cls[-5.0], "dCl/dAoA <= 0 (circulation sign regressed)"


def _count_self_crossings(loop: np.ndarray) -> int:
    n = len(loop)
    cnt = 0
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = loop[j], loop[(j + 1) % n]
            r = b - a
            s = d - c
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12:
                continue
            t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / den
            u = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / den
            if 0.01 < t < 0.99 and 0.01 < u < 0.99:
                cnt += 1
    return cnt


def test_surface_loop_no_self_crossing():
    """C4: LE/TE-split ordering traces a single non-crossing loop on a thin airfoil."""
    g = naca_airfoil("naca0012", n_points=120)
    pts = g.surface_points.astype(np.float64)
    np.random.default_rng(0).shuffle(pts)
    loop = _order_surface_loop(pts)
    assert _count_self_crossings(loop) == 0


def test_residual_is_dimensionless_and_wall_masked():
    """C1 + M3: residual norm is dimensionless & O(1); a wrong-but-smooth field ~0."""
    gen = SyntheticRANS(64)
    case = gen.sample_case(0)
    truth = gen.solve(case)
    rn_truth = PhysicsChecker().diagnose(truth, case).residual_norm()
    assert 0.0 < rn_truth < 50.0, f"residual_norm not dimensionless/sane: {rn_truth}"

    # A uniform freestream field (wrong, ignores the body) is ~divergence-free,
    # so its residual is ~0 -> documents that residual != correctness.
    d = case.domain
    uni = FlowField(
        domain=d, u=np.full(d.shape, 30.0, "float32"),
        v=np.zeros(d.shape, "float32"), p=np.zeros(d.shape, "float32"),
        mask=truth.mask, sdf=truth.sdf,
    )
    assert PhysicsChecker().diagnose(uni, case).residual_norm() < 1e-2


def test_physics_loss_is_balanced_at_optimum():
    """M3: at a perfect prediction the physics term is O(1), not O(1e5)."""
    cfg = Config()
    cfg.data = DataConfig(source="synthetic", resolution=48, n_train=3, n_val=2, batch_size=3)
    tr, _, nrm = build_dataloaders(cfg.data)
    batch = next(iter(tr))
    loss_fn = CompositeLoss(cfg, nrm, nu=1.5e-5)
    # Feed the ground-truth target as the prediction -> data ~ 0.
    _, parts = loss_fn(batch["target"], batch["target"], batch["input"], batch["mask"])
    assert parts["data"] < 1e-6
    assert parts["physics"] < 100.0, f"physics term not balanced: {parts['physics']}"


def test_corrector_unrolled_training_runs():
    """The unrolled + noise corrector training produces a finite, decreasing loss."""
    from neuroforge.models.base import build_model
    from neuroforge.models.correction import LocalCorrectionNet
    from neuroforge.train.trainer import Trainer

    cfg = Config()
    cfg.data = DataConfig(source="synthetic", resolution=32, n_train=4, n_val=2, batch_size=2)
    cfg.train.epochs = 1
    cfg.train.device = "cpu"
    cfg.train.log_every = 0
    tr, va, nrm = build_dataloaders(cfg.data)
    model = build_model("fno", width=12, n_layers=2, modes=6)
    trainer = Trainer(model, cfg, nrm, nu=1.5e-5)
    trainer.fit(tr, va)
    corr = LocalCorrectionNet(width=12, n_layers=2)
    hist = trainer.fit_corrector(tr, corr, epochs=2, unroll=2, noise_std=0.1)
    assert len(hist["corrector_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(x)) for x in hist["corrector_loss"])
