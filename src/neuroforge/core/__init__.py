"""Core contracts: data types, configuration, channel definitions."""

from .config import (
    Config,
    CorrectionConfig,
    DataConfig,
    ModelConfig,
    PhysicsConfig,
    TrainConfig,
    default_config,
)
from .types import (
    DTYPE,
    INPUT_CHANNELS,
    N_IN,
    N_OUT,
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

__all__ = [
    "DTYPE",
    "INPUT_CHANNELS",
    "OUTPUT_CHANNELS",
    "N_IN",
    "N_OUT",
    "BoundaryConditions",
    "Diagnostics",
    "Domain",
    "FlowCase",
    "FlowField",
    "FluidProperties",
    "Geometry",
    "SolveResult",
    "Config",
    "ModelConfig",
    "PhysicsConfig",
    "CorrectionConfig",
    "DataConfig",
    "TrainConfig",
    "default_config",
]
