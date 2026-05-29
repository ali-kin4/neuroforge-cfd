"""Configuration schema for NeuroForge CFD.

Plain dataclasses (no pydantic dependency) with YAML (de)serialisation. Defaults
are tuned to be **CPU-friendly** so the package runs end-to-end on a laptop
without a GPU; scale ``width``/``modes``/``epochs`` up when a GPU is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from .types import N_IN, N_OUT


@dataclass
class ModelConfig:
    """Backbone / correction network hyper-parameters."""

    name: str = "fno"                 # registry key: fno | geo_fno | transformer | unet | deeponet
    in_channels: int = N_IN
    out_channels: int = N_OUT
    width: int = 32                   # latent width / hidden dim
    n_layers: int = 4
    modes: int = 16                   # Fourier modes kept (FNO family)
    n_heads: int = 4                  # transformer
    n_slices: int = 32                # physics-attention slices (transformer)
    dropout: float = 0.0              # >0 enables MC-dropout uncertainty
    activation: str = "gelu"


@dataclass
class PhysicsConfig:
    """Physics residual / trust-map settings."""

    enabled: bool = True
    weight_continuity: float = 1.0
    weight_momentum: float = 1.0
    weight_bc: float = 1.0
    derivative: str = "finite_difference"   # finite_difference | spectral
    # Trust-map combination weights and thresholds (operate on normalised maps).
    trust_residual_weight: float = 0.6
    trust_uncertainty_weight: float = 0.4
    trust_green_below: float = 0.15         # combined error below -> green
    trust_red_above: float = 0.45           # combined error above -> red


@dataclass
class CorrectionConfig:
    """Neural Residual Iteration (the self-correction loop) settings."""

    enabled: bool = True
    max_iters: int = 5
    residual_tol: float = 1e-3              # stop when residual_norm < tol
    min_improvement: float = 1e-3           # stop when relative improvement stalls
    gate_by_trust: bool = True              # apply corrections only in low-trust regions
    step_size: float = 1.0                  # damping on the correction delta
    # Uncertainty-gated classical fallback.
    fallback_enabled: bool = False
    fallback_uncertainty_threshold: float = 0.5
    fallback_solver: str = "stub"           # stub | openfoam | su2


@dataclass
class DataConfig:
    """Dataset / dataloader settings."""

    source: str = "synthetic"               # synthetic | airfrans
    root: str = "data"
    resolution: int = 128
    n_train: int = 64
    n_val: int = 16
    batch_size: int = 4
    num_workers: int = 0
    normalize: bool = True
    seed: int = 0
    # AirfRANS-specific
    task: str = "scarce"                    # full | scarce | reynolds | aoa
    cache_dir: str | None = None            # cache rasterised AirfRANS pairs here


@dataclass
class TrainConfig:
    """Training loop settings (data loss + physics-informed loss)."""

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    physics_weight: float = 0.1             # weight on the PDE residual loss term
    bc_weight: float = 0.1
    grad_clip: float = 1.0
    device: str = "auto"                    # auto | cpu | cuda
    ckpt_dir: str = "checkpoints"
    log_every: int = 10
    amp: bool = False                       # mixed precision (GPU only)


@dataclass
class Config:
    """Top-level config aggregating every sub-config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 0

    # --- (de)serialisation ------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            physics=PhysicsConfig(**d.get("physics", {})),
            correction=CorrectionConfig(**d.get("correction", {})),
            data=DataConfig(**d.get("data", {})),
            train=TrainConfig(**d.get("train", {})),
            seed=d.get("seed", 0),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})


def default_config() -> Config:
    """A ready-to-run, CPU-friendly configuration."""
    return Config()
