"""Training layer: composite loss + the two-stage trainer."""

from __future__ import annotations

from .losses import CompositeLoss
from .recipes import evaluate_fields, train_recipe
from .trainer import Trainer

__all__ = ["Trainer", "CompositeLoss", "train_recipe", "evaluate_fields"]
