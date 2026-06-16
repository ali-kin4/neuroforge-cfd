"""Transolver (ICML 2024) baseline for AirfRANS — native point-cloud surrogate.

This is a FAITHFUL vendoring of the authors' official irregular-mesh
physics-attention block, adapted to the AirfRANS point-cloud features used by the
NeuroForge baselines. It is deliberately NOT registered in the frozen
``models.base`` registry: the registry's contract is grid-shaped
``(B, C, H, W) -> (B, C, H, W)`` solvers, whereas Transolver maps a point cloud
``(B, N, F_IN) -> (B, N, F_OUT)``. Keeping it standalone avoids misrepresenting
the contract (see ``docs/reviews/baseline_plan.md`` §3).

Attribution
-----------
The :class:`PhysicsAttentionIrregularMesh` block, the :class:`MLP` and the
:class:`TransolverBlock` below are transcribed from the official Transolver
implementation:

    Wu, Luo, Wang, Wang, Long. "Transolver: A Fast Transformer Solver for PDEs
    on General Geometries." ICML 2024.
    Repo: https://github.com/thuml/Transolver  (MIT License,
    Copyright (c) 2024 THUML @ Tsinghua University)
    File: Car-Design-ShapeNetCar/models/Transolver.py
          (Physics_Attention_Irregular_Mesh)

The defining Transolver features are reproduced verbatim: **per-head** soft
slice assignment (``(B, H, N, G)``), the dual ``in_project_x`` / ``in_project_fx``
projections, per-slice normalisation by the weight sum, the **learnable** per-head
temperature initialised to 0.5, and the orthogonal init of the slice projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "PhysicsAttentionIrregularMesh",
    "TransolverBlock",
    "TransolverPointModel",
]

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class PhysicsAttentionIrregularMesh(nn.Module):
    """Official Transolver physics-attention for irregular meshes / point clouds.

    Verbatim transcription (variable names preserved) of
    ``Physics_Attention_Irregular_Mesh`` from the THUML Transolver repo (MIT).
    Input/Output: ``(B, N, dim)``.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):  # x: (B, N, C)
        B, N, C = x.shape
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        # Per-head soft slice assignment: (B, H, N, G).
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)  # (B, H, G)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v)  # (B, H, G, c)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        # rearrange("b h n d -> b n (h d)") without the einops dependency.
        out_x = out_x.permute(0, 2, 1, 3).contiguous().reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out_x)


class MLP(nn.Module):
    """Transolver feed-forward MLP (verbatim from the official repo)."""

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        if act not in _ACT:
            raise NotImplementedError(act)
        act_cls = _ACT[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_cls())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_cls()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class TransolverBlock(nn.Module):
    """One Transolver block: physics-attention + MLP, pre-norm residual (verbatim)."""

    def __init__(self, num_heads, hidden_dim, dropout, act="gelu", mlp_ratio=4, slice_num=32):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregularMesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class TransolverPointModel(nn.Module):
    """Transolver wrapped as a point-cloud regressor for AirfRANS.

    Maps per-point input features ``(B, N, in_features)`` to per-point predictions
    ``(B, N, out_features)`` = ``(u, v, p, nut)``. A linear embedding lifts the
    raw point features to ``width``; ``n_layers`` Transolver blocks follow; a
    final norm + linear head projects to the outputs.

    The default ``width=256, n_layers=10`` is chosen to match the NeuroForge FNO
    backbone budget (7,350,420 params vs the FNO's 7,387,684; the exact count is
    reported by :meth:`num_params` and printed in ``run_baselines.py``).
    """

    def __init__(
        self,
        in_features: int = 7,
        out_features: int = 4,
        width: int = 256,
        n_layers: int = 10,
        n_heads: int = 8,
        n_slices: int = 32,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        act: str = "gelu",
    ) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width ({width}) must be divisible by n_heads ({n_heads})")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_slices = int(n_slices)

        self.embed = MLP(in_features, width, width, n_layers=0, res=False, act=act)
        self.blocks = nn.ModuleList(
            [TransolverBlock(n_heads, width, dropout, act=act, mlp_ratio=mlp_ratio, slice_num=n_slices)
             for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, in_features) -> (B, N, out_features)``."""
        fx = self.embed(x)
        for blk in self.blocks:
            fx = blk(fx)
        return self.head(self.out_norm(fx))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
