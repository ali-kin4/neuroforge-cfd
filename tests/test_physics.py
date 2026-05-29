"""Physics tests: operators, residuals, metrics, trust, differentiability."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuroforge.core.types import Domain
from neuroforge.physics.metrics import (
    field_errors,
    force_coefficients,
    pressure_coefficient,
)
from neuroforge.physics.operators import ddx, ddy, divergence, laplacian
from neuroforge.physics.residuals import (
    PhysicsChecker,
    continuity_residual,
    momentum_residual,
    physics_residual_torch,
)


# --------------------------------------------------------------------------- #
# Operators — analytic checks, numpy AND torch backends.
# --------------------------------------------------------------------------- #
def _grid(nx=32, ny=32):
    dom = Domain(bounds=(0.0, 1.0, 0.0, 1.0), nx=nx, ny=ny)
    # Rebuild coordinates in float64 (Domain.grid() returns float32) so the
    # analytic operator checks aren't dominated by float32 coordinate rounding.
    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return dom, X, Y


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_ddx_linear_ramp(backend):
    dom, X, Y = _grid()
    f = 3.0 * X + 0.0 * Y  # df/dx = 3 everywhere
    if backend == "torch":
        f = torch.from_numpy(f)
    out = ddx(f, dom.dx)
    out = out.numpy() if backend == "torch" else out
    assert np.allclose(out, 3.0, atol=1e-5)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_laplacian_linear_zero(backend):
    dom, X, Y = _grid()
    f = 2.0 * X - 5.0 * Y  # linear -> laplacian 0
    if backend == "torch":
        f = torch.from_numpy(f)
    out = laplacian(f, dom.dx, dom.dy)
    out = out.numpy() if backend == "torch" else out
    assert np.allclose(out, 0.0, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_divergence_uniform_zero(backend):
    dom, X, Y = _grid()
    u = np.full_like(X, 2.5)
    v = np.full_like(Y, -1.0)
    if backend == "torch":
        u = torch.from_numpy(u)
        v = torch.from_numpy(v)
    out = divergence(u, v, dom.dx, dom.dy)
    out = out.numpy() if backend == "torch" else out
    assert np.allclose(out, 0.0, atol=1e-6)


def test_ddy_returns_same_type():
    dom, X, Y = _grid()
    assert isinstance(ddy(X, dom.dy), np.ndarray)
    assert isinstance(ddy(torch.from_numpy(X), dom.dy), torch.Tensor)


# --------------------------------------------------------------------------- #
# Residual maps + PhysicsChecker.diagnose.
# --------------------------------------------------------------------------- #
def test_continuity_and_momentum_shapes(synthetic_field, tiny_case):
    cont = continuity_residual(synthetic_field)
    assert cont.shape == synthetic_field.shape
    r_x, r_y = momentum_residual(synthetic_field, tiny_case.fluid)
    assert r_x.shape == synthetic_field.shape
    assert r_y.shape == synthetic_field.shape
    assert np.all(np.isfinite(cont))
    assert np.all(np.isfinite(r_x)) and np.all(np.isfinite(r_y))


def test_diagnose_returns_valid_diagnostics(synthetic_field, tiny_case):
    checker = PhysicsChecker()
    diag = checker.diagnose(synthetic_field, tiny_case)
    shape = synthetic_field.shape
    for m in (diag.continuity, diag.momentum_x, diag.momentum_y,
              diag.bc_violation, diag.uncertainty, diag.trust, diag.trust_class):
        assert m.shape == shape
    # trust in [0,1]
    assert diag.trust.min() >= -1e-6 and diag.trust.max() <= 1.0 + 1e-6
    # trust_class subset of {0,1,2}
    assert set(np.unique(diag.trust_class)).issubset({0.0, 1.0, 2.0})
    # uncertainty defaults to zeros when None passed
    assert np.allclose(diag.uncertainty, 0.0)
    # residual_norm is a finite scalar
    assert np.isfinite(diag.residual_norm())


# --------------------------------------------------------------------------- #
# Differentiable torch residuals.
# --------------------------------------------------------------------------- #
def test_physics_residual_torch_differentiable():
    B, H, W = 1, 16, 16
    pred = torch.randn(B, 4, H, W, requires_grad=True)
    inp = torch.zeros(B, 7, H, W)
    inp[:, 1] = 1.0  # all-fluid mask
    res = physics_residual_torch(pred, inp, dx=0.1, dy=0.1, nu=1.5e-5)
    for k in ("continuity", "momentum_x", "momentum_y"):
        assert res[k].shape == (B, 1, H, W)
    loss = sum((res[k] ** 2).mean() for k in res)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def test_force_coefficients_finite(synthetic_field, tiny_case):
    coeffs = force_coefficients(synthetic_field, tiny_case)
    assert set(coeffs) == {"cl", "cd", "cm"}
    for v in coeffs.values():
        assert np.isfinite(v)


def test_field_errors_self_is_zero(synthetic_field):
    errs = field_errors(synthetic_field, synthetic_field)
    assert set(errs) == {"u", "v", "p", "speed"}
    for v in errs.values():
        assert v == pytest.approx(0.0, abs=1e-5)


def test_pressure_coefficient_shape(synthetic_field, tiny_case):
    cp = pressure_coefficient(synthetic_field, tiny_case)
    assert cp.shape == synthetic_field.shape
    assert np.all(np.isfinite(cp))
