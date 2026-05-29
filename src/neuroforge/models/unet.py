"""U-Net baseline backbone.

A standard encoder-decoder U-Net with skip connections, used as a strong
convolutional baseline against the neural-operator models. The number of
downsampling stages is ``n_layers``; channel widths double at each stage. To
support arbitrary input resolutions the input is reflection-padded up to the next
multiple of ``2 ** n_layers`` before the encoder and cropped back to the original
``H``/``W`` after the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroforge.core.types import N_IN, N_OUT

from .base import NeuralSolver, register_model

__all__ = ["UNet"]


class _DoubleConv(nn.Module):
    """Two ``3x3`` convolutions, each followed by GroupNorm + activation, with dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        # GroupNorm is batch-size agnostic (works for batch size 1 on CPU).
        g1 = _num_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g1, out_ch),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g1, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _num_groups(channels: int) -> int:
    """Pick a GroupNorm group count that divides ``channels`` (<= 8)."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


@register_model("unet")
class UNet(NeuralSolver):
    """Encoder-decoder U-Net mapping ``(B,N_IN,H,W) -> (B,N_OUT,H,W)``.

    Parameters
    ----------
    in_channels, out_channels:
        Network I/O channel counts (default to the core contract).
    width:
        Base channel width; doubles at every downsampling stage.
    n_layers:
        Number of down/up sampling stages (the network depth).
    dropout:
        Dropout probability inside each conv block (kept active for MC-dropout).
    """

    def __init__(
        self,
        in_channels: int = N_IN,
        out_channels: int = N_OUT,
        width: int = 32,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.width = int(width)
        self.depth = max(1, int(n_layers))

        # Encoder path: progressively double the channel width.
        self.inc = _DoubleConv(self.in_channels, self.width, dropout)
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        chans = [self.width]
        ch = self.width
        for _ in range(self.depth):
            self.pools.append(nn.MaxPool2d(2))
            self.downs.append(_DoubleConv(ch, ch * 2, dropout))
            ch *= 2
            chans.append(ch)

        # Decoder path: upsample and fuse with the matching encoder skip.
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for _ in range(self.depth):
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2))
            # After concatenating the skip connection the channel count is ch.
            self.up_convs.append(_DoubleConv(ch, ch // 2, dropout))
            ch //= 2

        self.outc = nn.Conv2d(ch, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H, W)`` to ``(B, out_channels, H, W)``."""
        _, _, h, w = x.shape
        x, (ph, pw) = self._pad_to_multiple(x)

        # Encoder, stashing skip features.
        skips = []
        h_feat = self.inc(x)
        skips.append(h_feat)
        for pool, down in zip(self.pools, self.downs):
            h_feat = down(pool(h_feat))
            skips.append(h_feat)

        # Decoder, fusing skips from the encoder (deepest skip is the bottleneck).
        h_feat = skips[-1]
        for i, (up, up_conv) in enumerate(zip(self.ups, self.up_convs)):
            h_feat = up(h_feat)
            skip = skips[-(i + 2)]
            # ConvTranspose can mismatch by 1 on odd sizes; align to the skip.
            h_feat = self._match(h_feat, skip)
            h_feat = up_conv(torch.cat([skip, h_feat], dim=1))

        out = self.outc(h_feat)
        # Crop away any padding added for divisibility.
        return out[..., :h, :w] if (ph or pw) else out

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Reflection-pad ``H``/``W`` up to the next multiple of ``2 ** depth``."""
        mult = 2 ** self.depth
        _, _, h, w = x.shape
        ph = (mult - h % mult) % mult
        pw = (mult - w % mult) % mult
        if ph or pw:
            # Reflection padding needs pad < dim; fall back to replicate otherwise.
            mode = "reflect" if (ph < h and pw < w) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return x, (ph, pw)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Crop/pad ``x`` so its spatial size matches ``ref`` (guards odd sizes)."""
        _, _, h, w = x.shape
        _, _, th, tw = ref.shape
        if h == th and w == tw:
            return x
        if h >= th and w >= tw:
            return x[..., :th, :tw]
        return F.pad(x, (0, max(0, tw - w), 0, max(0, th - h)), mode="replicate")
