"""Physics layer: finite-difference operators, RANS residuals, metrics, trust.

This package is the *verifier* that makes NeuroForge CFD trustworthy. It turns a
predicted :class:`~neuroforge.core.types.FlowField` into physics residuals,
engineering metrics, and a per-cell trust map, and provides a differentiable
residual for the physics-informed training loss.

Public API
----------
operators
    :func:`ddx`, :func:`ddy`, :func:`laplacian`, :func:`divergence`,
    :func:`gradient`, plus optional :func:`spectral_ddx` / :func:`spectral_ddy`.
residuals
    :func:`continuity_residual`, :func:`momentum_residual`,
    :func:`bc_violation`, :class:`PhysicsChecker`, :func:`physics_residual_torch`.
metrics
    :func:`pressure_coefficient`, :func:`force_coefficients`,
    :func:`wall_shear_stress`, :func:`field_errors`.
trust
    :func:`trust_map`.
"""

from __future__ import annotations

from .calibration import (
    ConformalCalibrator,
    calibrate_from_cases,
    cases_to_error_sigma,
    corrected_predict_fn,
    expected_calibration_error,
    reliability,
)
from .evaluation import (
    coefficient_metrics,
    evaluate_calibration,
    evaluate_cases,
    per_channel_mse,
    residual_error_correlation,
    surface_pressure_mse,
)
from .metrics import (
    field_errors,
    force_coefficients,
    pressure_coefficient,
    wall_shear_stress,
)
from .operators import (
    ddx,
    ddy,
    divergence,
    gradient,
    laplacian,
    spectral_ddx,
    spectral_ddy,
)
from .residuals import (
    PhysicsChecker,
    bc_violation,
    continuity_residual,
    momentum_residual,
    physics_residual_torch,
)
from .trust import trust_map

__all__ = [
    # operators
    "ddx",
    "ddy",
    "laplacian",
    "divergence",
    "gradient",
    "spectral_ddx",
    "spectral_ddy",
    # residuals
    "continuity_residual",
    "momentum_residual",
    "bc_violation",
    "PhysicsChecker",
    "physics_residual_torch",
    # metrics
    "pressure_coefficient",
    "force_coefficients",
    "wall_shear_stress",
    "field_errors",
    # evaluation (AirfRANS-protocol metrics)
    "per_channel_mse",
    "surface_pressure_mse",
    "coefficient_metrics",
    "residual_error_correlation",
    "evaluate_cases",
    "evaluate_calibration",
    # calibration (conformal coverage guarantees)
    "ConformalCalibrator",
    "calibrate_from_cases",
    "cases_to_error_sigma",
    "corrected_predict_fn",
    "reliability",
    "expected_calibration_error",
    # trust
    "trust_map",
]
