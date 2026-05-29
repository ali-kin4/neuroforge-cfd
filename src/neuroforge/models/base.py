"""Model contracts: the :class:`NeuralSolver` / :class:`CorrectionNetwork` ABCs
and a tiny string-keyed registry.

Every backbone maps an input tensor ``(B, C_in, H, W)`` to an output tensor
``(B, C_out, H, W)`` where ``C_in == len(INPUT_CHANNELS)`` and
``C_out == len(OUTPUT_CHANNELS)``. Models must be **device-agnostic** (no
hard-coded ``.cuda()``) and run on CPU.

This file is a frozen contract; module builds extend the registry by importing
:func:`register_model`, they do not edit the ABCs here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

from neuroforge.core.types import N_IN, N_OUT

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

MODEL_REGISTRY: dict[str, type["NeuralSolver"]] = {}


def register_model(name: str) -> Callable[[type], type]:
    """Class decorator registering a :class:`NeuralSolver` under ``name``."""

    def _wrap(cls: type) -> type:
        key = name.lower()
        if key in MODEL_REGISTRY:
            raise KeyError(f"model '{key}' is already registered")
        MODEL_REGISTRY[key] = cls
        cls.registry_name = key
        return cls

    return _wrap


def build_model(name: str, **kwargs) -> "NeuralSolver":
    """Instantiate a registered model by name with keyword hyper-parameters."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](**kwargs)


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


# --------------------------------------------------------------------------- #
# Abstract bases
# --------------------------------------------------------------------------- #


class NeuralSolver(nn.Module, ABC):
    """Base class for flow-field backbones.

    Subclasses set ``in_channels`` / ``out_channels`` and implement
    :meth:`forward`. The default channel counts come from the core contract.
    """

    registry_name: str = "base"

    def __init__(self, in_channels: int = N_IN, out_channels: int = N_OUT) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C_in, H, W)`` -> ``(B, C_out, H, W)``."""
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience eval-mode forward pass."""
        was_training = self.training
        self.eval()
        out = self(x)
        if was_training:
            self.train()
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CorrectionNetwork(nn.Module, ABC):
    """Base class for the local correction model used in Neural Residual Iteration.

    Takes the current solution, its physics-residual maps and the geometry
    channels, and predicts an additive correction ``delta`` of shape
    ``(B, C_out, H, W)`` to be applied (optionally trust-gated) to the field.
    """

    @abstractmethod
    def forward(
        self,
        field: torch.Tensor,      # (B, C_out, H, W) current solution
        residual: torch.Tensor,   # (B, 3, H, W) [continuity, mom_x, mom_y]
        geom: torch.Tensor,       # (B, C_in, H, W) geometry/BC channels
    ) -> torch.Tensor:
        """Return additive correction ``delta`` of shape ``(B, C_out, H, W)``."""
        raise NotImplementedError
