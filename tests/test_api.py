"""Tests for the high-level NeuroForge estimator API."""

from __future__ import annotations

import neuroforge as nf


def _tiny_model(corrector="deq", dropout=0.0):
    return nf.NeuroForge(
        backbone="fno", width=12, modes=6, n_layers=2, corrector=corrector,
        dropout=dropout, epochs=2, corrector_epochs=1, resolution=24,
        batch_size=2, device="cpu",
    )


def test_fit_predict_solve_evaluate():
    model = _tiny_model().fit("synthetic", n_train=4, n_val=3, download=False, verbose=False)
    assert "fitted" in repr(model)

    case = nf.FlowCase.from_airfoil("naca2412", aoa=5, reynolds=3e6, u_inf=30.0, resolution=24)
    field = model.predict(case)
    assert field.u.shape == (24, 24)

    result = model.solve(case)
    assert "residual_norm" in result.metrics

    metrics = model.evaluate(limit=3)
    for k in ("mse_u", "rho_cd", "residual_error_spearman"):
        assert k in metrics


def test_ablation_returns_both():
    model = _tiny_model().fit("synthetic", n_train=4, n_val=3, download=False, verbose=False)
    abl = model.ablate_corrector(limit=3)
    assert "backbone" in abl and "corrected" in abl
    assert "rho_cd" in abl["backbone"]


def test_save_load_roundtrip(tmp_path):
    model = _tiny_model().fit("synthetic", n_train=4, n_val=2, download=False, verbose=False)
    p = str(tmp_path / "m.pt")
    model.save(p)
    loaded = nf.NeuroForge.load(p)
    case = nf.FlowCase.from_airfoil("naca0012", aoa=0, reynolds=1e6, u_inf=10.0, resolution=24)
    assert loaded.solve(case).metrics["residual_norm"] >= 0.0


def test_calibrate_requires_dropout():
    # dropout=0 -> calibrate returns None with a message (no usable uncertainty).
    model = _tiny_model(dropout=0.0).fit("synthetic", n_train=4, n_val=3, download=False, verbose=False)
    assert model.calibrate(limit=3) is None
    # dropout>0 -> a calibrator with a positive multiplier.
    model2 = _tiny_model(dropout=0.05).fit("synthetic", n_train=4, n_val=3, download=False, verbose=False)
    cal = model2.calibrate(alpha=0.1, limit=3)
    assert cal is not None and cal.q > 0.0
