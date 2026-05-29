"""Data-layer tests: dataloaders, Normalizer round-trips, FlowDataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuroforge.core.config import DataConfig
from neuroforge.core.types import N_IN, N_OUT
from neuroforge.data.datamodule import FlowDataset, Normalizer, build_dataloaders
from neuroforge.data.synthetic import SyntheticRANS


def _tiny_cfg():
    # build_dataloaders takes a DataConfig (per the frozen contract signature).
    return DataConfig(
        source="synthetic", resolution=24, n_train=4, n_val=2, batch_size=2, seed=0
    )


def test_build_dataloaders_batch_shapes():
    cfg = _tiny_cfg()
    train_loader, val_loader, normalizer = build_dataloaders(cfg)
    assert normalizer.fitted
    batch = next(iter(train_loader))
    b = batch["input"].shape[0]
    assert batch["input"].shape == (b, N_IN, cfg.resolution, cfg.resolution)
    assert batch["target"].shape == (b, N_OUT, cfg.resolution, cfg.resolution)
    assert batch["mask"].shape == (b, 1, cfg.resolution, cfg.resolution)
    assert batch["sdf"].shape == (b, 1, cfg.resolution, cfg.resolution)
    assert isinstance(batch["case"], list) and len(batch["case"]) == b
    # val loader yields the expected number of samples.
    n_val = sum(bt["input"].shape[0] for bt in val_loader)
    assert n_val == cfg.n_val


def test_normalizer_roundtrip():
    rng = np.random.default_rng(0)
    inputs = rng.normal(2.0, 3.0, size=(5, N_IN, 8, 8)).astype(np.float32)
    outputs = rng.normal(-1.0, 0.5, size=(5, N_OUT, 8, 8)).astype(np.float32)
    norm = Normalizer().fit(inputs, outputs)

    x = inputs[0]
    back = norm.denorm_in(norm.norm_in(x))
    assert np.allclose(back, x, atol=1e-4)

    y = outputs[0]
    back_y = norm.denorm_out(norm.norm_out(y))
    assert np.allclose(back_y, y, atol=1e-4)

    # Works for torch tensors too.
    xt = torch.from_numpy(x)
    back_t = norm.denorm_in(norm.norm_in(xt))
    assert torch.allclose(back_t, xt, atol=1e-4)


def test_normalizer_state_dict_roundtrip():
    rng = np.random.default_rng(1)
    inputs = rng.normal(size=(4, N_IN, 6, 6)).astype(np.float32)
    outputs = rng.normal(size=(4, N_OUT, 6, 6)).astype(np.float32)
    norm = Normalizer().fit(inputs, outputs)
    norm2 = Normalizer.from_state_dict(norm.state_dict())
    assert np.allclose(norm.mean_in, norm2.mean_in)
    assert np.allclose(norm.std_in, norm2.std_in)
    assert np.allclose(norm.mean_out, norm2.mean_out)
    assert np.allclose(norm.std_out, norm2.std_out)
    x = inputs[0]
    assert np.allclose(norm.norm_in(x), norm2.norm_in(x), atol=1e-6)


def test_flowdataset_length_and_items():
    gen = SyntheticRANS(resolution=24, seed=0)
    pairs = gen.generate(3)
    norm = Normalizer().fit(*FlowDataset(pairs).raw_arrays())
    ds = FlowDataset(pairs, normalizer=norm)
    assert len(ds) == 3
    item = ds[0]
    assert item["input"].shape == (N_IN, 24, 24)
    assert item["target"].shape == (N_OUT, 24, 24)
    assert item["mask"].shape == (1, 24, 24)


def test_flowdataset_empty_raises():
    with pytest.raises(ValueError):
        FlowDataset([])
