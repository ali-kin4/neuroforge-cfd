"""Learning-rate schedule helpers for the trainer.

A tiny, dependency-free cosine schedule with linear warm-up, exposed both as a
plain ``lr(step)`` function and as a thin wrapper around a torch optimizer that
sets ``param_group['lr']`` each step. CPU-friendly and stateless beyond a step
counter.
"""

from __future__ import annotations

import math

__all__ = ["cosine_warmup_lr", "WarmupCosineScheduler"]


def cosine_warmup_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> float:
    """Cosine-annealed learning rate with an optional linear warm-up.

    Parameters
    ----------
    step : int
        Current (0-based) global step.
    total_steps : int
        Total number of steps over the whole training run.
    base_lr : float
        Peak learning rate (reached at the end of warm-up).
    warmup_steps : int, optional
        Number of linear-warm-up steps from ``0`` to ``base_lr``.
    min_lr : float, optional
        Floor learning rate reached at ``total_steps``.

    Returns
    -------
    float
        The learning rate for ``step``.
    """
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    # Cosine decay over the post-warm-up span.
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


class WarmupCosineScheduler:
    """Apply :func:`cosine_warmup_lr` to a torch optimizer, one step at a time.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose param-group learning rates are updated in place.
    total_steps : int
        Total number of optimizer steps planned for the run.
    base_lr : float
        Peak learning rate.
    warmup_frac : float, optional
        Fraction of ``total_steps`` spent in linear warm-up.
    min_lr : float, optional
        Final learning rate floor.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        base_lr: float,
        warmup_frac: float = 0.05,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(max(0.0, float(warmup_frac)) * self.total_steps)
        self._step = 0

    def get_lr(self) -> float:
        """Learning rate that *will* be applied at the current step."""
        return cosine_warmup_lr(
            self._step, self.total_steps, self.base_lr, self.warmup_steps, self.min_lr
        )

    def step(self) -> float:
        """Advance one step, set the optimizer LR, and return the applied LR."""
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._step += 1
        return lr
