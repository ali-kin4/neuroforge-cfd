"""Neural backbones, baselines, correction networks, and UQ wrappers.

The ``base`` imports below are the frozen contract. The model-building agent
appends concrete model imports (FNO2d, GeoFNO, PhysicsTransformer, UNet,
DeepONet, LocalCorrectionNet, DeepEnsemble, ...) without removing these.
"""

from .base import (
    MODEL_REGISTRY,
    CorrectionNetwork,
    NeuralSolver,
    available_models,
    build_model,
    register_model,
)

__all__ = [
    "NeuralSolver",
    "CorrectionNetwork",
    "MODEL_REGISTRY",
    "register_model",
    "build_model",
    "available_models",
]

# --- concrete implementations (registered on import) ---------------------- #
# Imported lazily-tolerant: a partially built tree should still expose the ABCs.
try:  # pragma: no cover - import wiring
    from .fno import FNO2d  # noqa: F401
    from .geo_fno import GeoFNO  # noqa: F401
    from .transformer import PhysicsTransformer  # noqa: F401
    from .unet import UNet  # noqa: F401
    from .deeponet import DeepONet  # noqa: F401
    from .correction import LocalCorrectionNet  # noqa: F401
    from .ensemble import DeepEnsemble, MCDropoutUQ  # noqa: F401

    __all__ += [
        "FNO2d",
        "GeoFNO",
        "PhysicsTransformer",
        "UNet",
        "DeepONet",
        "LocalCorrectionNet",
        "DeepEnsemble",
        "MCDropoutUQ",
    ]
except ImportError:  # modules not built yet
    pass
