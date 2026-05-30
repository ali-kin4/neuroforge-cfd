"""Tests for the contractive Deep-Equilibrium corrector and its guarantees."""

from __future__ import annotations

import torch

from neuroforge.core.types import N_IN, N_OUT
from neuroforge.models.deq_corrector import DEQCorrector


def _inputs(b=2, h=32, w=32):
    return (
        torch.randn(b, N_OUT, h, w),
        torch.randn(b, 3, h, w),
        torch.randn(b, N_IN, h, w),
    )


def test_operator_is_a_contraction():
    """Measured Lipschitz of T in delta must be <= kappa < 1 (Banach)."""
    torch.manual_seed(0)
    deq = DEQCorrector(width=24, n_layers=3, lipschitz=0.9)
    field, res, geom = _inputs()
    cond = torch.cat([field, res, geom], dim=1)
    a = torch.randn(2, N_OUT, 32, 32)
    b = torch.randn(2, N_OUT, 32, 32)
    with torch.no_grad():
        ratio = float(
            (deq._operator(a, cond) - deq._operator(b, cond)).norm() / ((a - b).norm() + 1e-12)
        )
    assert ratio <= deq.lipschitz + 1e-3, f"not a contraction: L={ratio} > kappa={deq.lipschitz}"


def test_fixed_point_converges_geometrically():
    torch.manual_seed(0)
    deq = DEQCorrector(width=24, n_layers=3, lipschitz=0.9, max_iter=60, tol=1e-6)
    field, res, geom = _inputs()
    deq.eval()
    delta = deq(field, res, geom)
    rep = deq.convergence_report()
    assert rep["rel_residual"] < 1e-4
    assert rep["iters"] < 60
    assert 0.0 < rep["empirical_contraction"] < 1.0
    # delta is a genuine fixed point: ||T(delta) - delta|| is tiny.
    cond = torch.cat([field, res, geom], dim=1)
    with torch.no_grad():
        t = deq._operator(delta, cond)
        nxt = (1 - deq.damping) * delta + deq.damping * t
        fp_err = float((nxt - delta).norm() / (delta.norm() + 1e-8))
    assert fp_err < 1e-3


def test_jfb_gradient_is_finite_and_nonzero():
    deq = DEQCorrector(width=16, n_layers=2)
    deq.train()
    field, res, geom = _inputs()
    out = deq(field, res, geom)
    loss = (out ** 2).mean()
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in deq.parameters() if p.grad is not None)
    assert torch.isfinite(loss) and g > 0.0


def test_deq_drop_in_shape():
    deq = DEQCorrector(width=16, n_layers=2)
    deq.eval()
    field, res, geom = _inputs(b=3, h=24, w=24)
    out = deq(field, res, geom)
    assert out.shape == (3, N_OUT, 24, 24)


def test_deq_end_to_end_via_recipe(tmp_path):
    """Train a tiny DEQ corrector, load it through the engine, and solve."""
    import neuroforge as nf
    from neuroforge.core.config import Config, DataConfig
    from neuroforge.solver.engine import NeuroForgeEngine
    from neuroforge.train.recipes import train_recipe

    cfg = Config()
    cfg.data = DataConfig(source="synthetic", resolution=24, n_train=4, n_val=2, batch_size=2)
    cfg.model.width, cfg.model.n_layers, cfg.model.modes = 12, 2, 6
    cfg.train.epochs, cfg.train.device, cfg.train.log_every = 1, "cpu", 0
    out = str(tmp_path / "deq.pt")
    train_recipe(
        cfg, corrector_type="deq", corrector_epochs=1, corrector_width=12,
        corrector_layers=2, ckpt_every=0, out=out, verbose=False,
    )
    eng = NeuroForgeEngine.from_checkpoint(out)
    assert type(eng.corrector).__name__ == "DEQCorrector"
    case = nf.FlowCase.from_airfoil("naca2412", aoa=5, reynolds=3e6, u_inf=30.0, resolution=24)
    res = eng.solve(case)
    h = res.history[-1]
    assert "deq_contraction" in h and h["deq_contraction"] < 1.0
