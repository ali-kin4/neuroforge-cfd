"""Uncertainty-quantification wrappers: deep ensembles and MC-dropout.

Two complementary epistemic-uncertainty estimators for the neural solvers:

* :class:`DeepEnsemble` aggregates the predictions of several independently
  parameterised :class:`NeuralSolver` members; the disagreement between members
  gives the uncertainty.
* :class:`MCDropoutUQ` wraps a single dropout-enabled solver and draws several
  stochastic forward passes with dropout left active at inference time; the spread
  of those samples gives the uncertainty.

Both expose ``predict_with_uncertainty`` returning a ``(mean, std)`` pair where
both tensors are ``(B, N_OUT, H, W)`` (per-cell, per-channel statistics). A scalar
``(B, 1, H, W)`` aggregate (the channel-mean std) is available via
:meth:`aggregate_uncertainty` for convenience.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import NeuralSolver

__all__ = ["DeepEnsemble", "MCDropoutUQ", "aggregate_uncertainty"]


def aggregate_uncertainty(std: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """Collapse a per-channel std map ``(B, C, H, W)`` to a scalar map ``(B, 1, H, W)``.

    Parameters
    ----------
    std:
        Per-cell, per-channel standard deviation.
    keepdim:
        If ``True`` keep the channel dimension (size 1); otherwise drop it.
    """
    return std.mean(dim=1, keepdim=keepdim)


class DeepEnsemble(nn.Module):
    """Ensemble of :class:`NeuralSolver` members with uncertainty from disagreement.

    Parameters
    ----------
    members:
        A list of (typically independently initialised / trained) solvers. They
        must share input/output channel counts so their outputs can be stacked.
    """

    def __init__(self, members: list[NeuralSolver]) -> None:
        super().__init__()
        if len(members) == 0:
            raise ValueError("DeepEnsemble requires at least one member")
        self.members = nn.ModuleList(members)

    def __len__(self) -> int:
        return len(self.members)

    def _stack(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-member predictions stacked as ``(M, B, C, H, W)``."""
        return torch.stack([m(x) for m in self.members], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ensemble mean prediction ``(B, N_OUT, H, W)``."""
        return self._stack(x).mean(dim=0)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` over members, each ``(B, N_OUT, H, W)``.

        Members are evaluated in eval mode; the standard deviation across members
        is the (epistemic) uncertainty. Uses the unbiased estimator when more than
        one member is present.
        """
        x = x.to(next(self.parameters()).device)
        was_training = self.training
        self.eval()
        preds = self._stack(x)  # (M, B, C, H, W)
        if was_training:
            self.train()
        mean = preds.mean(dim=0)
        unbiased = preds.shape[0] > 1
        std = preds.std(dim=0, unbiased=unbiased)
        return mean, std

    def aggregate_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: scalar ``(B, 1, H, W)`` uncertainty map for ``x``."""
        _, std = self.predict_with_uncertainty(x)
        return aggregate_uncertainty(std)


class MCDropoutUQ(nn.Module):
    """Monte-Carlo dropout uncertainty for a single dropout-enabled solver.

    Parameters
    ----------
    model:
        A :class:`NeuralSolver` containing dropout layers (e.g. built with
        ``dropout > 0``). Dropout is forced active during sampling.
    """

    def __init__(self, model: NeuralSolver) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single deterministic (eval-mode) forward pass through the wrapped model."""
        return self.model(x)

    @staticmethod
    def _enable_dropout(module: nn.Module) -> None:
        """Put only the dropout layers of ``module`` into training mode."""
        for m in module.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                m.train()

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` over ``n_samples`` MC-dropout passes.

        The model is otherwise in eval mode (so BatchNorm/running stats are fixed),
        but its dropout layers are kept active to inject stochasticity. Both
        returned tensors are ``(B, N_OUT, H, W)``.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        x = x.to(next(self.model.parameters()).device)
        was_training = self.model.training
        self.model.eval()
        self._enable_dropout(self.model)

        samples = torch.stack([self.model(x) for _ in range(n_samples)], dim=0)

        if was_training:
            self.model.train()

        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=n_samples > 1)
        return mean, std

    def aggregate_uncertainty(self, x: torch.Tensor, n_samples: int = 16) -> torch.Tensor:
        """Convenience: scalar ``(B, 1, H, W)`` uncertainty map for ``x``."""
        _, std = self.predict_with_uncertainty(x, n_samples=n_samples)
        return aggregate_uncertainty(std)
