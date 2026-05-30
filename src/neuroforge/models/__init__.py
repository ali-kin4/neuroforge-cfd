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
    from .correction import LocalCorrectionNet  # noqa: F401
    from .deeponet import DeepONet  # noqa: F401
    from .deq_corrector import DEQCorrector  # noqa: F401
    from .ensemble import DeepEnsemble, MCDropoutUQ  # noqa: F401
    from .fno import FNO2d  # noqa: F401
    from .geo_fno import GeoFNO  # noqa: F401
    from .transformer import PhysicsTransformer  # noqa: F401
    from .unet import UNet  # noqa: F401

    __all__ += [
        "FNO2d",
        "GeoFNO",
        "PhysicsTransformer",
        "UNet",
        "DeepONet",
        "LocalCorrectionNet",
        "DEQCorrector",
        "DeepEnsemble",
        "MCDropoutUQ",
        "build_corrector",
    ]
except ImportError:  # modules not built yet
    pass


def build_corrector(config: dict):
    """Instantiate a correction network from a serialised config dict.

    ``config['type']`` selects ``'deq'`` (:class:`DEQCorrector`, the principled
    convergent fixed point) or ``'local'`` (:class:`LocalCorrectionNet`, the
    feed-forward baseline). Remaining keys are constructor kwargs. Used by the
    checkpoint loader so the right corrector is rebuilt.
    """
    from .correction import LocalCorrectionNet
    from .deq_corrector import DEQCorrector

    cfg = dict(config or {})
    kind = str(cfg.pop("type", "local")).lower()
    if kind == "deq":
        return DEQCorrector(**{k: cfg[k] for k in (
            "width", "n_layers", "lipschitz", "damping", "max_iter", "tol",
        ) if k in cfg})
    return LocalCorrectionNet(**{k: cfg[k] for k in ("width", "n_layers") if k in cfg})
