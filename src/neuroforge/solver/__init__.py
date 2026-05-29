"""Solver layer: predictor, self-correcting engine, correction loop, fallback."""

from __future__ import annotations

from .correction_loop import neural_residual_iteration
from .engine import NeuroForgeEngine, Predictor, demo
from .fallback import ClassicalFallback

__all__ = [
    "Predictor",
    "NeuroForgeEngine",
    "neural_residual_iteration",
    "ClassicalFallback",
    "demo",
]
