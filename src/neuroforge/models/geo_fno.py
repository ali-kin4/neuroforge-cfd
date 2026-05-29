"""Geometry-aware Fourier Neural Operator (Geo-FNO).

A pragmatic, robust take on Geo-FNO (Li et al., 2022). A small CNN
"deformation/encoder" reads the geometry channels (the SDF and solid mask, which
live at fixed positions in the input stack) and produces a learned per-cell
gating/warp field. This conditions the lifted features so the body geometry is
explicitly injected before the spectral core operates on what behaves like a
latent, geometry-normalised uniform field. An internal :class:`FNO2d` core does
the global spectral mixing, and a light convolutional head maps the result back
to the physical output channels.

The model deliberately stays simple and CPU-cheap; it composes :class:`FNO2d`
rather than reimplementing the spectral machinery, and falls back gracefully when
fewer geometry channels are present than the default contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuroforge.core.types import N_IN, N_OUT

from .base import NeuralSolver, register_model
from .fno import FNO2d, _make_activation

__all__ = ["GeoFNO"]


class _GeometryEncoder(nn.Module):
    """Small CNN that turns the geometry channels into a per-cell deformation field.

    It outputs ``2 * width`` maps interpreted as a multiplicative gate and an
    additive bias applied to the lifted features (a soft, learned feature warp
    conditioned on the body geometry).
    """

    def __init__(self, geom_channels: int, width: int, activation: str = "gelu") -> None:
        super().__init__()
        hidden = max(width // 2, 8)
        self.net = nn.Sequential(
            nn.Conv2d(geom_channels, hidden, kernel_size=3, padding=1),
            _make_activation(activation),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            _make_activation(activation),
            nn.Conv2d(hidden, 2 * width, kernel_size=1),
        )
        self.width = int(width)

    def forward(self, geom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(gate, bias)`` each of shape ``(B, width, H, W)``."""
        out = self.net(geom)
        gate, bias = out[:, : self.width], out[:, self.width :]
        # Center the gate at 1.0 so an untrained encoder is near the identity warp.
        gate = 1.0 + torch.tanh(gate)
        return gate, bias


@register_model("geo_fno")
class GeoFNO(NeuralSolver):
    """Geometry-conditioned FNO mapping ``(B,N_IN,H,W) -> (B,N_OUT,H,W)``.

    Parameters
    ----------
    in_channels, out_channels:
        Network I/O channel counts (default to the core contract).
    width:
        Latent channel width shared by the encoder, FNO core, and decoder.
    n_layers:
        Number of Fourier layers in the internal :class:`FNO2d` core.
    modes:
        Low-frequency Fourier modes per axis in the core (clamped per call).
    dropout:
        Dropout probability (kept active for MC-dropout uncertainty).
    activation:
        Activation name used throughout.
    geom_channels:
        Number of leading input channels treated as geometry/BC context for the
        deformation encoder. Defaults to ``min(4, in_channels)`` which covers the
        contract's ``(sdf, mask, x, y)`` block; clamped to ``in_channels``.
    """

    def __init__(
        self,
        in_channels: int = N_IN,
        out_channels: int = N_OUT,
        width: int = 32,
        n_layers: int = 4,
        modes: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        geom_channels: int | None = None,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.width = int(width)
        # Fall back gracefully if the stack is smaller than the default contract.
        if geom_channels is None:
            geom_channels = min(4, self.in_channels)
        self.geom_channels = max(1, min(int(geom_channels), self.in_channels))

        # Lift the full input stack into the latent width.
        self.lift = nn.Conv2d(self.in_channels, self.width, kernel_size=1)
        # Geometry-conditioned deformation/encoder.
        self.encoder = _GeometryEncoder(self.geom_channels, self.width, activation=activation)

        # Internal FNO core operates on the warped latent field (latent -> latent).
        self.core = FNO2d(
            in_channels=self.width,
            out_channels=self.width,
            width=self.width,
            n_layers=n_layers,
            modes=modes,
            dropout=dropout,
            activation=activation,
        )

        # Decoder: map the latent field back to physical output channels.
        self.decoder = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1),
            _make_activation(activation),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(self.width, self.out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H, W)`` to ``(B, out_channels, H, W)``."""
        geom = x[:, : self.geom_channels]
        gate, bias = self.encoder(geom)

        h = self.lift(x)
        # Warp the lifted features into the geometry-normalised latent field.
        h = gate * h + bias
        # Global spectral mixing on the latent field.
        h = self.core(h)
        return self.decoder(h)
