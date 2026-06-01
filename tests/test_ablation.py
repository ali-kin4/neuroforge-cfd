"""Smoke test for the reproducible ablation harness (marked slow: trains models)."""

from __future__ import annotations

import os

import pytest

from benchmarks.ablation import run_ablation


@pytest.mark.slow
def test_run_ablation_produces_table(tmp_path):
    r = run_ablation(
        "synthetic", n_train=3, n_val=2, resolution=16, seeds=(0,),
        epochs=1, corrector_epochs=1, width=10, modes=4, n_layers=2,
        batch_size=2, download=False, device="cpu", out_dir=str(tmp_path), verbose=False,
    )
    arms = r["table"]
    assert "backbone" in arms and "backbone + DEQ corrector" in arms
    for arm in arms.values():
        assert "rho_cd" in arm and "residual_error_spearman" in arm
    assert os.path.exists(tmp_path / "ablation.md")
    assert os.path.exists(tmp_path / "ablation.csv")
