"""Model tests: backbone forward shapes, correction net, and UQ wrappers."""

from __future__ import annotations

import pytest
import torch

from neuroforge.core.types import N_IN, N_OUT
from neuroforge.models.base import available_models, build_model
from neuroforge.models.correction import LocalCorrectionNet
from neuroforge.models.ensemble import DeepEnsemble, MCDropoutUQ


@pytest.mark.parametrize("name", available_models())
def test_backbone_forward_shape_and_params(name):
    model = build_model(name, width=12, n_layers=2)
    x = torch.randn(2, N_IN, 32, 32)
    y = model(x)
    assert y.shape == (2, N_OUT, 32, 32)
    assert torch.isfinite(y).all()
    assert model.num_params() > 0


def test_local_correction_net_shape():
    corr = LocalCorrectionNet(width=12, n_layers=3)
    B, H, W = 2, 32, 32
    field = torch.randn(B, N_OUT, H, W)
    residual = torch.randn(B, 3, H, W)
    geom = torch.randn(B, N_IN, H, W)
    delta = corr(field=field, residual=residual, geom=geom)
    assert delta.shape == (B, N_OUT, H, W)
    assert corr.num_params() > 0 if hasattr(corr, "num_params") else True
    # Near-zero init head -> tiny initial correction.
    assert float(delta.abs().max()) < 1e-3


def test_deep_ensemble_uncertainty():
    members = [build_model("fno", width=10, n_layers=2) for _ in range(3)]
    ens = DeepEnsemble(members)
    x = torch.randn(2, N_IN, 16, 16)
    mean, std = ens.predict_with_uncertainty(x)
    assert mean.shape == (2, N_OUT, 16, 16)
    assert std.shape == (2, N_OUT, 16, 16)
    assert (std >= 0).all()


def test_mc_dropout_uncertainty():
    model = build_model("fno", width=10, n_layers=2, dropout=0.3)
    mc = MCDropoutUQ(model)
    x = torch.randn(1, N_IN, 16, 16)
    mean, std = mc.predict_with_uncertainty(x, n_samples=4)
    assert mean.shape == (1, N_OUT, 16, 16)
    assert std.shape == (1, N_OUT, 16, 16)
    assert (std >= 0).all()
    assert float(std.sum()) > 0.0  # dropout produces nonzero spread
