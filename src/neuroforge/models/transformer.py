"""Transolver-style physics-attention backbone.

Implements the core idea of Transolver (Wu et al., 2024): instead of paying the
quadratic cost of token-to-token attention over every grid point, each point is
softly assigned to one of ``n_slices`` learnable "physics slices". Tokens are
aggregated into a small set of slice tokens (a weighted mean), full multi-head
self-attention runs **only among the slice tokens** (cost linear in the number of
grid points, quadratic only in the small ``n_slices``), and the attended slice
tokens are scattered back to the points using the same soft assignments. Several
such physics-attention blocks are stacked and the result is projected back to the
output channels.

Designed to run comfortably on CPU at modest resolutions (e.g. 64x64).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroforge.core.types import N_IN, N_OUT

from .base import NeuralSolver, register_model

__all__ = ["PhysicsTransformer", "PhysicsAttention"]


class PhysicsAttention(nn.Module):
    """One physics-attention block operating on flattened tokens ``(B, N, C)``.

    The block learns per-token soft assignments to ``n_slices`` slices, builds
    slice tokens by a normalised weighted mean of the point tokens, runs
    multi-head self-attention across the slice tokens, then scatters the result
    back to every point. Complexity is ``O(N * n_slices)`` plus
    ``O(n_slices^2)`` — i.e. linear in the number of grid points ``N``.
    """

    def __init__(self, width: int, n_heads: int, n_slices: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width ({width}) must be divisible by n_heads ({n_heads})")
        self.width = int(width)
        self.n_heads = int(n_heads)
        self.n_slices = int(n_slices)
        self.head_dim = self.width // self.n_heads

        self.norm = nn.LayerNorm(self.width)
        # Learned projection producing one logit per slice for every token.
        self.slice_proj = nn.Linear(self.width, self.n_slices)

        # Q/K/V projections over slice tokens, plus an output projection.
        self.to_q = nn.Linear(self.width, self.width)
        self.to_k = nn.Linear(self.width, self.width)
        self.to_v = nn.Linear(self.width, self.width)
        self.to_out = nn.Linear(self.width, self.width)
        self.dropout = nn.Dropout(float(dropout))
        self.temperature = self.head_dim ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return updated tokens of the same shape ``(B, N, C)`` (residual added)."""
        b, n, c = tokens.shape
        x = self.norm(tokens)

        # Soft assignment of each token to slices: (B, N, S).
        weights = F.softmax(self.slice_proj(x), dim=-1)

        # Aggregate tokens into slice tokens via a normalised weighted mean:
        # slice_tokens = (W^T X) / (sum of weights per slice).  (B, S, C)
        denom = weights.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)  # (B, S, 1)
        slice_tokens = torch.einsum("bns,bnc->bsc", weights, x) / denom

        # Multi-head self-attention among the (few) slice tokens.
        q = self._split_heads(self.to_q(slice_tokens))  # (B, H, S, d)
        k = self._split_heads(self.to_k(slice_tokens))
        v = self._split_heads(self.to_v(slice_tokens))
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        attn = self.dropout(attn)
        attended = attn @ v  # (B, H, S, d)
        attended = self._merge_heads(attended)  # (B, S, C)
        attended = self.to_out(attended)

        # Scatter slice tokens back to points using the same soft assignments.
        out = torch.einsum("bns,bsc->bnc", weights, attended)  # (B, N, C)
        out = self.dropout(out)
        return tokens + out

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, s, _ = t.shape
        return t.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, h, s, d = t.shape
        return t.transpose(1, 2).contiguous().view(b, s, h * d)


class _FeedForward(nn.Module):
    """Token-wise MLP with a residual connection and pre-norm."""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width * 2, width),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


@register_model("transformer")
class PhysicsTransformer(NeuralSolver):
    """Physics-attention transformer mapping ``(B,N_IN,H,W) -> (B,N_OUT,H,W)``.

    Parameters
    ----------
    in_channels, out_channels:
        Network I/O channel counts (default to the core contract).
    width:
        Token embedding width.
    n_layers:
        Number of stacked physics-attention + feed-forward blocks.
    n_heads:
        Attention heads used among the slice tokens (``width`` must be divisible
        by ``n_heads``).
    n_slices:
        Number of learnable physics slices.
    dropout:
        Dropout probability (kept active for MC-dropout uncertainty).
    """

    def __init__(
        self,
        in_channels: int = N_IN,
        out_channels: int = N_OUT,
        width: int = 32,
        n_layers: int = 4,
        n_heads: int = 4,
        n_slices: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_slices = int(n_slices)

        # Per-cell tokenisation (1x1 conv == point-wise linear embedding).
        self.embed = nn.Conv2d(self.in_channels, self.width, kernel_size=1)

        self.attn_blocks = nn.ModuleList(
            [PhysicsAttention(self.width, self.n_heads, self.n_slices, dropout) for _ in range(self.n_layers)]
        )
        self.ffn_blocks = nn.ModuleList(
            [_FeedForward(self.width, dropout) for _ in range(self.n_layers)]
        )

        self.out_norm = nn.LayerNorm(self.width)
        # Project tokens back to physical output channels (point-wise).
        self.head = nn.Conv2d(self.width, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H, W)`` to ``(B, out_channels, H, W)``."""
        b, _, h, w = x.shape
        feat = self.embed(x)  # (B, width, H, W)

        # Flatten the grid to tokens: (B, N, width) with N = H * W.
        tokens = feat.flatten(2).transpose(1, 2)
        for attn, ffn in zip(self.attn_blocks, self.ffn_blocks):
            tokens = attn(tokens)
            tokens = ffn(tokens)
        tokens = self.out_norm(tokens)

        # Unpatch back to a spatial feature map and project to outputs.
        feat = tokens.transpose(1, 2).reshape(b, self.width, h, w)
        return self.head(feat)
