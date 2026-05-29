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

# --- Math-library thread guard (must run before numpy/torch are imported) --- #
# Prebuilt numpy/torch wheels ship OpenBLAS/MKL compiled with a high MAX_THREADS
# and DYNAMIC_ARCH. On low-core machines (laptops, VMs, CI runners) this
# oversubscribes catastrophically:
#   * a 200x200 ``np.linalg.solve`` (synthetic source-panel generator) takes
#     >20 s instead of ~1 ms, and
#   * a 32x32 FFT inside the Fourier Neural Operator takes ~0.15 s instead of
#     ~0.3 ms, making one training step ~30x slower.
# Both collapse to instant once the math libraries use a single thread, and our
# small models / dense solves don't benefit from BLAS/FFT threading on CPU
# anyway (torch parallelises across the batch at the op level regardless).
# We use ``setdefault`` so a power user training a large model on a many-core
# CPU can override any of these (e.g. ``OMP_NUM_THREADS=8``) before import.
import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

_BLAS_LIMITER = None  # held for the process lifetime so the cap is not GC-restored
try:  # best-effort runtime cap if the math libs were already imported elsewhere
    import threadpoolctl as _tpc  # type: ignore

    _BLAS_LIMITER = _tpc.threadpool_limits(limits=1)
except Exception:  # pragma: no cover - threadpoolctl is optional
    pass
import sys as _sys

if "torch" in _sys.modules:  # torch imported before us: cap at runtime
    try:
        _sys.modules["torch"].set_num_threads(1)
    except Exception:  # pragma: no cover
        pass

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
