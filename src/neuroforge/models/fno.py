"""2-D Fourier Neural Operator (FNO).

A real, spectral FNO backbone following Li et al. (2021): a point-wise lifting
convolution raises the input channel stack to a latent ``width``, several Fourier
layers mix information globally in the frequency domain (each combined with a
parallel ``1x1`` convolution residual path and a non-linearity), and a two-layer
point-wise projection maps the latent field down to ``out_channels``.

The spectral convolution uses :func:`torch.fft.rfft2` / :func:`torch.fft.irfft2`,
keeps only the ``modes`` lowest frequencies in each spatial dimension, applies a
learned complex linear mixing across channels, and transforms back. Arbitrary
(non power-of-two) ``H``/``W`` are supported, and the requested number of modes
is clamped to the frequencies actually available for the given grid size, so a
forward pass never indexes out of bounds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroforge.core.types import N_IN, N_OUT

from .base import NeuralSolver, register_model

__all__ = ["FNO2d", "SpectralConv2d"]


def _make_activation(name: str) -> nn.Module:
    """Return an activation module for a short name (``gelu``/``relu``/``tanh``/``silu``)."""
    name = (name or "gelu").lower()
    table: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
    }
    if name not in table:
        raise ValueError(f"unknown activation '{name}'. Options: {sorted(table)}")
    return table[name]()


class SpectralConv2d(nn.Module):
    """2-D spectral convolution: FFT, low-mode complex mixing, inverse FFT.

    Parameters
    ----------
    in_channels, out_channels:
        Channel counts of the latent field.
    modes1, modes2:
        Maximum number of Fourier modes kept along the height and width axes.
        These are upper bounds; at call time they are clamped to the frequencies
        available for the actual grid resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        # Two complex weight blocks: one for the low +ve frequency corner and one
        # for the low -ve frequency (wrap-around) corner along the first axis.
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weight1 = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Complex batched matmul over channels: ``(B,Ci,M1,M2),(Ci,Co,M1,M2)->(B,Co,M1,M2)``."""
        return torch.einsum("bixy,ioxy->boxy", inp, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the spectral convolution to ``(B, in_channels, H, W)``."""
        batch, _, height, width = x.shape

        # Real FFT: width axis becomes (W // 2 + 1) one-sided frequencies.
        x_ft = torch.fft.rfft2(x, norm="ortho")
        w_half = x_ft.shape[-1]

        # Clamp modes to what this resolution can actually provide.
        m1 = min(self.modes1, height // 2)
        m1 = max(m1, 1)
        m2 = min(self.modes2, w_half)
        m2 = max(m2, 1)

        out_ft = torch.zeros(
            batch, self.out_channels, height, w_half, dtype=torch.cfloat, device=x.device
        )
        # Low positive frequencies (top-left corner of the spectrum).
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weight1[:, :, :m1, :m2]
        )
        # Low negative frequencies along the first axis (bottom-left corner).
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weight2[:, :, :m1, :m2]
        )

        # Inverse real FFT back to the original spatial size.
        out = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return out


@register_model("fno")
class FNO2d(NeuralSolver):
    """2-D Fourier Neural Operator mapping ``(B,N_IN,H,W) -> (B,N_OUT,H,W)``.

    Parameters
    ----------
    in_channels, out_channels:
        Network I/O channel counts (default to the core contract).
    width:
        Latent channel width of the Fourier layers.
    n_layers:
        Number of Fourier layers.
    modes:
        Number of low-frequency Fourier modes retained per spatial axis (clamped
        per call to the grid resolution).
    dropout:
        Dropout probability applied after each Fourier layer (kept active for
        MC-dropout uncertainty estimation).
    activation:
        Activation name used between layers (``gelu`` by default).
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
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.modes = int(modes)
        self.dropout_p = float(dropout)

        # Point-wise lifting from the input channel stack to the latent width.
        self.lift = nn.Conv2d(self.in_channels, self.width, kernel_size=1)

        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes, self.modes) for _ in range(self.n_layers)]
        )
        # Parallel point-wise (1x1) convolution residual path per Fourier layer.
        self.w_layers = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(self.n_layers)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout2d(self.dropout_p) for _ in range(self.n_layers)]
        )
        self.activation = _make_activation(activation)

        # Two-layer point-wise projection head to the output channels.
        self.project = nn.Sequential(
            nn.Conv2d(self.width, self.width * 2, kernel_size=1),
            _make_activation(activation),
            nn.Conv2d(self.width * 2, self.out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H, W)`` to ``(B, out_channels, H, W)``."""
        h = self.lift(x)
        for spectral, pointwise, drop in zip(self.spectral_layers, self.w_layers, self.dropouts):
            h_spec = spectral(h)
            h_local = pointwise(h)
            h = self.activation(h_spec + h_local)
            h = drop(h)
        return self.project(h)
