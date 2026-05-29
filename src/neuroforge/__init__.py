"""NeuroForge CFD — a self-correcting, geometry-native AI CFD engine.

Public API (stable):

    from neuroforge import FlowCase, NeuroForgeEngine
    case = FlowCase.from_airfoil("naca2412", aoa=5, reynolds=3e6, u_inf=30.0)
    engine = NeuroForgeEngine.from_checkpoint("checkpoints/airfoil.pt")
    result = engine.solve(case)
    result.save_report("report.html")

Lightweight contract types (``FlowCase``, ``FlowField``, ``Domain``, ...) import
eagerly because they are torch-free. Heavy objects (engine, models, trainer) are
exposed via a module-level ``__getattr__`` so that importing the package never
forces the deep-learning stack to load before it is needed, and a partially
built source tree still imports.
"""

from __future__ import annotations

from .core.config import Config, default_config
from .core.types import (
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    BoundaryConditions,
    Diagnostics,
    Domain,
    FlowCase,
    FlowField,
    FluidProperties,
    Geometry,
    SolveResult,
)

__version__ = "0.1.0"

# Names resolved lazily: attribute -> "submodule:qualname".
_LAZY: dict[str, str] = {
    "NeuroForgeEngine": "neuroforge.solver.engine:NeuroForgeEngine",
    "Predictor": "neuroforge.solver.engine:Predictor",
    "NeuralSolver": "neuroforge.models.base:NeuralSolver",
    "build_model": "neuroforge.models.base:build_model",
    "available_models": "neuroforge.models.base:available_models",
    "Trainer": "neuroforge.train.trainer:Trainer",
    "naca_airfoil": "neuroforge.geometry.airfoil:naca_airfoil",
    "SyntheticRANS": "neuroforge.data.synthetic:SyntheticRANS",
    "PhysicsChecker": "neuroforge.physics.residuals:PhysicsChecker",
    "demo": "neuroforge.solver.engine:demo",
}


def __getattr__(name: str):  # PEP 562 lazy attribute access
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'neuroforge' has no attribute '{name}'")
    import importlib

    mod_name, _, qual = target.partition(":")
    obj = getattr(importlib.import_module(mod_name), qual)
    globals()[name] = obj  # cache for next time
    return obj


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    "__version__",
    "Config",
    "default_config",
    "FlowCase",
    "FlowField",
    "Domain",
    "Geometry",
    "BoundaryConditions",
    "FluidProperties",
    "Diagnostics",
    "SolveResult",
    "INPUT_CHANNELS",
    "OUTPUT_CHANNELS",
    # lazy
    "NeuroForgeEngine",
    "Predictor",
    "NeuralSolver",
    "build_model",
    "available_models",
    "Trainer",
    "naca_airfoil",
    "SyntheticRANS",
    "PhysicsChecker",
    "demo",
]
