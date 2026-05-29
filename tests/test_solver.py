"""Solver tests: Predictor, the self-correcting engine, fallback, NRI loop.

The headline assertion is that the residual_norm history is non-increasing
across accepted iterations of Neural Residual Iteration.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.core.types import FlowField, SolveResult
from neuroforge.solver.correction_loop import neural_residual_iteration
from neuroforge.solver.fallback import ClassicalFallback

_RES_TOL = 1e-9


def test_predictor_predict_shapes(trained_engine, tiny_case):
    # tiny_case is res 32; the engine was trained at res 24. The predictor is
    # resolution-agnostic (FNO), so predict on a fresh res-24 case.
    from neuroforge import FlowCase

    case = FlowCase.from_airfoil("naca2412", aoa=3.0, reynolds=3e6,
                                 u_inf=30.0, resolution=24)
    field = trained_engine.predictor.predict(case)
    assert isinstance(field, FlowField)
    assert field.shape == case.domain.shape
    assert field.u.shape == case.domain.shape
    assert np.all(np.isfinite(field.u))
    assert field.mask is not None and field.sdf is not None


def test_engine_solve_returns_result_and_metrics(trained_engine):
    from neuroforge import FlowCase

    case = FlowCase.from_airfoil("naca2412", aoa=4.0, reynolds=3e6,
                                 u_inf=30.0, resolution=24)
    result = trained_engine.solve(case)
    assert isinstance(result, SolveResult)
    for key in ("cl", "cd", "residual_norm", "n_iters"):
        assert key in result.metrics
        assert np.isfinite(result.metrics[key])
    assert len(result.history) >= 1


def test_residual_history_non_increasing(trained_engine):
    """The breakthrough claim: residual_norm never increases across iterations."""
    from neuroforge import FlowCase

    case = FlowCase.from_airfoil("naca2412", aoa=5.0, reynolds=3e6,
                                 u_inf=30.0, resolution=24)
    result = trained_engine.solve(case, max_iters=5)
    res = [h["residual_norm"] for h in result.history]
    assert len(res) >= 1
    for prev, nxt in zip(res, res[1:]):
        assert nxt <= prev + _RES_TOL, f"residual increased: {prev} -> {nxt}"


def test_neural_residual_iteration_direct(trained_engine):
    from neuroforge import FlowCase

    case = FlowCase.from_airfoil("naca0012", aoa=2.0, reynolds=3e6,
                                 u_inf=30.0, resolution=24)
    field0 = trained_engine.predictor.predict(case)
    field, history = neural_residual_iteration(
        field0, case, trained_engine.checker, trained_engine.corrector,
        trained_engine.config.correction, trained_engine.predictor,
    )
    assert isinstance(field, FlowField)
    assert isinstance(history, list) and len(history) >= 1
    assert history[0]["iter"] == 0
    res = [h["residual_norm"] for h in history]
    for prev, nxt in zip(res, res[1:]):
        assert nxt <= prev + _RES_TOL


def test_nri_no_corrector_diagnoses_once(trained_engine):
    from neuroforge import FlowCase

    case = FlowCase.from_airfoil("naca0012", aoa=0.0, reynolds=3e6,
                                 u_inf=30.0, resolution=24)
    field0 = trained_engine.predictor.predict(case)
    field, history = neural_residual_iteration(
        field0, case, trained_engine.checker, None,
        trained_engine.config.correction, trained_engine.predictor,
    )
    assert field is field0
    assert len(history) == 1


def test_classical_fallback_stub(synthetic_field, tiny_case):
    fb = ClassicalFallback("stub")
    region = np.asarray(synthetic_field.mask) > 0.5
    out = fb.patch(synthetic_field, tiny_case, region)
    assert isinstance(out, FlowField)
    assert "fallback" in out.meta
    meta = out.meta["fallback"]
    assert meta["backend"] == "stub"
    assert meta["ran"] is False
    assert meta["region_cells"] == int(region.sum())


def test_classical_fallback_unknown_backend_raises():
    with pytest.raises(ValueError):
        ClassicalFallback("nonsense")


def test_classical_fallback_openfoam_not_implemented(synthetic_field, tiny_case):
    fb = ClassicalFallback("openfoam")
    region = np.ones(synthetic_field.shape, dtype=bool)
    with pytest.raises(NotImplementedError):
        fb.patch(synthetic_field, tiny_case, region)
