"""Grid-to-grid DeepONet baseline.

A DeepONet (Lu et al., 2021) factorises an operator into a *branch* network that
encodes the input function and a *trunk* network that encodes the query
coordinates; the output is the inner product of their ``p``-dimensional feature
vectors. Here the operator maps the input field stack to the solution fields on
the same structured grid:

* The **branch** is a small CNN that reads the whole ``(B, N_IN, H, W)`` stack and
  produces ``out_channels * p`` basis coefficients per sample (global, one set per
  image).
* The **trunk** is an MLP applied point-wise to the ``(x, y)`` coordinate grid,
  producing ``p`` basis functions at every cell.
* The output channel ``c`` at cell ``(i, j)`` is
  ``sum_k branch[c, k] * trunk[k, i, j] + bias[c]``.

The coordinate grid is taken from the ``x``/``y`` input channels when present
(contract positions 2 and 3) and otherwise synthesised on a normalised ``[-1, 1]``
lattice, so the model is robust to smaller input stacks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuroforge.core.types import N_IN, N_OUT

from .base import NeuralSolver, register_model

__all__ = ["DeepONet"]


class _BranchNet(nn.Module):
    """CNN encoding the input stack into ``out_channels * p`` coefficients per sample."""

    def __init__(self, in_channels: int, width: int, n_layers: int, out_channels: int, p: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, width, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(max(0, n_layers - 1)):
            layers += [
                nn.Conv2d(width, width, kernel_size=3, padding=1, stride=2),
                nn.GELU(),
                nn.Dropout2d(float(dropout)),
            ]
        self.features = nn.Sequential(*layers)
        # Pool to a fixed vector regardless of resolution, then map to coefficients.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, out_channels * p),
        )
        self.out_channels = int(out_channels)
        self.p = int(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return branch coefficients of shape ``(B, out_channels, p)``."""
        b = x.shape[0]
        h = self.features(x)
        h = self.pool(h).flatten(1)
        coeff = self.head(h)
        return coeff.view(b, self.out_channels, self.p)


class _TrunkNet(nn.Module):
    """Point-wise MLP mapping ``(x, y)`` coordinates to ``p`` basis functions."""

    def __init__(self, width: int, n_layers: int, p: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.GELU()]
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(width, width), nn.GELU(), nn.Dropout(float(dropout))]
        layers += [nn.Linear(width, p)]
        self.net = nn.Sequential(*layers)
        self.p = int(p)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Map ``(B, H*W, 2)`` coordinates to basis functions ``(B, H*W, p)``."""
        return self.net(coords)


@register_model("deeponet")
class DeepONet(NeuralSolver):
    """Grid-to-grid DeepONet mapping ``(B,N_IN,H,W) -> (B,N_OUT,H,W)``.

    Parameters
    ----------
    in_channels, out_channels:
        Network I/O channel counts (default to the core contract).
    width:
        Hidden width of both the branch CNN and the trunk MLP.
    n_layers:
        Depth of the branch and trunk networks.
    dropout:
        Dropout probability (kept active for MC-dropout uncertainty).
    p:
        Number of latent basis functions per output channel.
    """

    def __init__(
        self,
        in_channels: int = N_IN,
        out_channels: int = N_OUT,
        width: int = 32,
        n_layers: int = 4,
        dropout: float = 0.0,
        p: int = 64,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.width = int(width)
        self.p = int(p)
        self.branch = _BranchNet(self.in_channels, self.width, n_layers, self.out_channels, self.p, dropout)
        self.trunk = _TrunkNet(self.width, n_layers, self.p, dropout)
        # Per-output-channel bias added after the branch/trunk inner product.
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def _coords(self, x: torch.Tensor) -> torch.Tensor:
        """Build the ``(B, H*W, 2)`` coordinate grid for the trunk.

        Uses the ``x``/``y`` input channels (contract positions 2 and 3) when the
        stack is large enough; otherwise synthesises a normalised ``[-1, 1]`` grid.
        """
        b, _, h, w = x.shape
        if self.in_channels >= 4:
            coords = x[:, 2:4]  # (B, 2, H, W)
            return coords.flatten(2).transpose(1, 2)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).reshape(1, h * w, 2)
        return grid.expand(b, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H, W)`` to ``(B, out_channels, H, W)``."""
        b, _, h, w = x.shape
        branch = self.branch(x)             # (B, out_channels, p)
        coords = self._coords(x)            # (B, H*W, 2)
        trunk = self.trunk(coords)          # (B, H*W, p)

        # Inner product over the basis dimension p -> (B, out_channels, H*W).
        out = torch.einsum("bcp,bnp->bcn", branch, trunk)
        out = out + self.bias.view(1, -1, 1)
        return out.reshape(b, self.out_channels, h, w)
