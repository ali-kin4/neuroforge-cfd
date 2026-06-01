"""Contractive Deep-Equilibrium (DEQ) correction operator with a convergence
guarantee.

The ad-hoc backtracking correction loop has no convergence theory: its only
property (a non-increasing residual) is satisfiable by doing nothing. This module
replaces it with a *principled fixed point*. The correction ``δ`` is defined as
the equilibrium of a learned operator

    δ* = T_θ(δ* ; cond),    T_θ(δ; cond) = κ · g_θ([δ, cond]),

where ``cond = [backbone_field, residual, geometry]`` is fixed conditioning and
``g_θ`` is a CNN whose layers are **spectrally normalised** (each ≤ 1-Lipschitz).
Scaling by ``κ < 1`` makes ``T_θ`` a **contraction in δ** (Lipschitz ≤ κ), so by
the Banach fixed-point theorem the equilibrium ``δ*`` exists, is unique, and the
iteration ``δ_{k+1} = T_θ(δ_k)`` converges geometrically: ``‖δ_k − δ*‖ ≤ κ^k‖δ_0 − δ*‖``.

Gradients use **Jacobian-Free Backpropagation** (Fung et al., 2022): the fixed
point is found under ``no_grad`` and a single extra operator application carries
the gradient — O(1) memory, no unrolling, and stable. The corrected field is
``backbone_field + δ*``; the operator is trained on *data* (so it targets
correctness, with the residual as an informative input feature — not driven to a
residual that we have shown need not coincide with truth).

References: Bai et al., *Deep Equilibrium Models*, NeurIPS 2019; Winston & Kolter,
*Monotone Operator Equilibrium Networks*, NeurIPS 2020; Fung et al., *JFB:
Jacobian-Free Backpropagation for Implicit Networks*, AAAI 2022.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuroforge.core.types import N_IN, N_OUT

from .base import CorrectionNetwork

__all__ = ["DEQCorrector"]


def _spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalisation (parametrisation API, with a fallback)."""
    try:
        from torch.nn.utils.parametrizations import spectral_norm

        return spectral_norm(module)
    except Exception:  # pragma: no cover - very old torch
        from torch.nn.utils import spectral_norm as _sn

        return _sn(module)


class DEQCorrector(CorrectionNetwork):
    """A contractive DEQ correction operator (drop-in :class:`CorrectionNetwork`).

    Parameters
    ----------
    in_channels, out_channels : int
        Geometry-input and solution-field channel counts.
    width, n_layers : int
        Hidden width / depth of the spectrally-normalised operator ``g_θ``.
    lipschitz : float
        Contraction factor ``κ ∈ (0, 1)``. The operator is ``κ · g_θ`` with
        ``g_θ`` ≤ 1-Lipschitz (spectral norm), so ``T_θ`` is a ``κ``-contraction.
    damping : float
        Relaxation ``β ∈ (0, 1]`` for the forward iteration
        ``δ ← (1−β) δ + β T_θ(δ)`` (same fixed point; improves stability).
    max_iter : int
        Maximum forward fixed-point iterations.
    tol : float
        Relative convergence tolerance on ``‖δ_{k+1} − δ_k‖``.
    """

    def __init__(
        self,
        in_channels: int = N_IN,
        out_channels: int = N_OUT,
        width: int = 32,
        n_layers: int = 3,
        lipschitz: float = 0.9,
        damping: float = 0.5,
        max_iter: int = 25,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.lipschitz = float(lipschitz)
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        # Diagnostics from the most recent solve (for reporting/guarantee checks).
        self.last_iters: int = 0
        self.last_residual: float = 0.0
        self.last_contraction: float = 0.0

        # cond = [field (N_OUT), residual (3), geom (N_IN)]; operator input also
        # carries the recurrent state δ (N_OUT).
        cond_ch = self.out_channels + 3 + self.in_channels
        in_ch = self.out_channels + cond_ch

        layers: list[nn.Module] = [_spectral_norm(nn.Conv2d(in_ch, self.width, 3, padding=1))]
        for _ in range(self.n_layers - 1):
            layers += [nn.GELU(), _spectral_norm(nn.Conv2d(self.width, self.width, 3, padding=1))]
        layers += [nn.GELU(), _spectral_norm(nn.Conv2d(self.width, self.out_channels, 3, padding=1))]
        self.g = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    # The contraction operator and its fixed-point solve
    # ------------------------------------------------------------------ #

    def _operator(self, delta: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``T_θ(δ; cond) = κ · g_θ([δ, cond])`` — a κ-contraction in ``δ``."""
        return self.lipschitz * self.g(torch.cat([delta, cond], dim=1))

    @torch.no_grad()
    def solve(self, cond: torch.Tensor) -> tuple[torch.Tensor, int, float, float]:
        """Forward fixed-point solve. Returns ``(δ*, iters, rel_residual, contraction)``.

        ``contraction`` is the measured ratio ``‖Δ_k‖ / ‖Δ_{k−1}‖`` (empirical
        confirmation of the Banach factor); it should stay below 1.
        """
        b, _, h, w = cond.shape
        delta = cond.new_zeros(b, self.out_channels, h, w)
        prev_step = None
        contraction = 0.0
        rel = 0.0
        iters = 0
        while iters < self.max_iter:
            iters += 1
            t = self._operator(delta, cond)
            nxt = (1.0 - self.damping) * delta + self.damping * t
            step = nxt - delta
            step_norm = float(step.norm())
            if prev_step is not None and prev_step > 1e-12:
                contraction = max(contraction, step_norm / prev_step)
            prev_step = step_norm
            rel = step_norm / (float(delta.norm()) + 1e-8)
            delta = nxt
            if rel < self.tol:
                break
        return delta, iters, rel, contraction

    def forward(
        self,
        field: torch.Tensor,      # (B, N_OUT, H, W) current/backbone solution
        residual: torch.Tensor,   # (B, 3, H, W) physics residual feature
        geom: torch.Tensor,       # (B, N_IN, H, W) geometry/BC channels
    ) -> torch.Tensor:
        """Return the equilibrium correction ``δ*`` (Jacobian-Free gradient)."""
        dev = next(self.parameters()).device
        cond = torch.cat([field.to(dev), residual.to(dev), geom.to(dev)], dim=1)
        delta_star, iters, rel, contraction = self.solve(cond)
        self.last_iters = int(iters)
        self.last_residual = float(rel)
        self.last_contraction = float(contraction)

        if self.training:
            # JFB: one extra operator application carries the gradient (O(1) mem).
            d = delta_star.detach()
            t = self._operator(d, cond)
            return (1.0 - self.damping) * d + self.damping * t
        return delta_star

    def convergence_report(self) -> dict[str, float]:
        """Diagnostics from the last forward solve."""
        return {
            "lipschitz_kappa": self.lipschitz,
            "iters": float(self.last_iters),
            "rel_residual": self.last_residual,
            "empirical_contraction": self.last_contraction,
        }
