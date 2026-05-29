"""Training layer: composite loss + the two-stage trainer."""

from __future__ import annotations

from .losses import CompositeLoss
from .trainer import Trainer

__all__ = ["Trainer", "CompositeLoss"]
