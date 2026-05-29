"""Local correction network for Neural Residual Iteration.

A small residual CNN that predicts an additive correction ``delta`` to the
current solution given the physics-residual maps and the geometry context. It is
the learnable component of the self-correcting loop: applied (optionally
trust-gated) to nudge the field towards lower physics residuals.

The three inputs — the current field ``(B, N_OUT, H, W)``, the residual maps
``(B, 3, H, W)`` for ``[continuity, momentum_x, momentum_y]``, and the geometry/BC
channels ``(B, N_IN, H, W)`` — are concatenated along the channel axis and passed
through a stack of convolutions. The final convolution is initialised at (near)
zero so that an untrained corrector produces a negligible correction, keeping the
first iteration of the loop stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuroforge.core.types import N_IN, N_OUT

from .base import CorrectionNetwork

__all__ = ["LocalCorrectionNet"]

#: Number of residual map channels: [continuity, momentum_x, momentum_y].
N_RESIDUAL = 3


class LocalCorrectionNet(CorrectionNetwork):
    """Small residual CNN predicting an additive field correction.

    Parameters
    ----------
    width:
        Hidden channel width of the convolution stack.
    n_layers:
        Number of hidden convolution layers (each ``3x3`` + activation).
    out_channels:
        Number of solution channels to correct (defaults to ``N_OUT``).
    in_geom:
        Number of geometry/BC channels supplied to ``forward`` (defaults to
        ``N_IN``).
    dropout:
        Dropout probability inside the stack.
    """

    def __init__(
        self,
        width: int = 24,
        n_layers: int = 3,
        out_channels: int = N_OUT,
        in_geom: int = N_IN,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.n_layers = max(1, int(n_layers))
        self.out_channels = int(out_channels)
        self.in_geom = int(in_geom)

        in_ch = self.out_channels + N_RESIDUAL + self.in_geom
        layers: list[nn.Module] = [nn.Conv2d(in_ch, self.width, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(self.n_layers - 1):
            layers += [
                nn.Conv2d(self.width, self.width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout2d(float(dropout)),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(self.width, self.out_channels, kernel_size=3, padding=1)

        # Initialise the final layer near zero -> tiny initial correction.
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        field: torch.Tensor,      # (B, N_OUT, H, W)
        residual: torch.Tensor,   # (B, 3, H, W)
        geom: torch.Tensor,       # (B, N_IN, H, W)
    ) -> torch.Tensor:
        """Return the additive correction ``delta`` of shape ``(B, out_channels, H, W)``."""
        x = torch.cat([field, residual, geom], dim=1)
        return self.head(self.body(x))
