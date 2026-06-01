"""Composite training loss: data + physics-informed + boundary-condition terms.

:class:`CompositeLoss` combines

1. a **data** term — masked MSE between the predicted and target fields in the
   *normalised* space (fluid cells only),
2. a **physics** term — the steady-RANS PDE residual evaluated on the
   *denormalised (physical)* fields via
   :func:`neuroforge.physics.physics_residual_torch`, and
3. a **boundary-condition** term — a penalty on the velocity magnitude inside /
   at the solid (where the no-slip condition demands ``|U| -> 0``).

The physics residual needs the grid spacing ``(dx, dy)`` but the loss only sees
batched tensors, so the spacing is fixed at construction time. By default it is
derived from ``cfg.data.resolution`` and the default :class:`Domain` bounds
``(-1, 2, -1.5, 1.5)``; pass ``dx`` / ``dy`` explicitly to override (e.g. when
training on a non-default domain).
"""

from __future__ import annotations

import torch

from neuroforge.core.config import Config
from neuroforge.core.types import Domain

__all__ = ["CompositeLoss", "per_channel_balanced_mse"]


def per_channel_balanced_mse(
    delta: torch.Tensor,
    target_delta: torch.Tensor,
    fluid: torch.Tensor,
    var_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel variance-normalised masked MSE over fluid cells.

    A plain masked MSE over all output channels is dominated by the
    large-magnitude channels (e.g. ``u``/``p``), so the near-zero small-magnitude
    channels (e.g. the cross-stream velocity ``v``) are effectively ignored and
    can be *over-corrected*. Here each channel's squared error is divided by that
    channel's target variance over the fluid cells (computed per batch), so every
    channel contributes comparably and the corrector learns to improve every
    field rather than only the dominant ones.

    Parameters
    ----------
    delta : torch.Tensor
        Predicted correction, shape ``(B, C, H, W)``.
    target_delta : torch.Tensor
        Target correction (``target - state``), shape ``(B, C, H, W)``.
    fluid : torch.Tensor
        Fluid mask (1 fluid, 0 solid), broadcastable to ``(B, 1, H, W)``.
    var_floor : float
        Lower clamp on the per-channel variance, keeping the normalisation finite
        for (near-)constant channels.

    Returns
    -------
    (loss, per_channel) : tuple[torch.Tensor, torch.Tensor]
        Scalar mean-over-channels balanced loss, and the per-channel
        (already variance-normalised) loss contributions of shape ``(C,)`` for
        diagnostics. Both share the input dtype/device.
    """
    c = delta.shape[1]
    # Flatten to (C, B*H*W) so per-channel reductions are a single dim-1 sum.
    m = fluid.expand(-1, c, -1, -1).permute(1, 0, 2, 3).reshape(c, -1)
    n_per_ch = m.sum(dim=1).clamp_min(1.0)

    err = target_delta.permute(1, 0, 2, 3).reshape(c, -1)
    # Per-channel target mean / variance over fluid cells only.
    mean = (err * m).sum(dim=1) / n_per_ch
    var = (((err - mean.unsqueeze(1)) ** 2) * m).sum(dim=1) / n_per_ch
    var = var.clamp_min(var_floor)

    diff2 = ((delta - target_delta) ** 2).permute(1, 0, 2, 3).reshape(c, -1)
    per_channel = (diff2 * m).sum(dim=1) / n_per_ch / var
    loss = per_channel.mean()
    return loss, per_channel


class CompositeLoss:
    """Weighted data + physics + BC loss for flow-field training.

    Parameters
    ----------
    cfg : Config
        Full config; ``cfg.train.physics_weight`` / ``cfg.train.bc_weight`` set
        the physics and BC term weights, ``cfg.data.resolution`` fixes the grid.
    normalizer : Normalizer
        Used to denormalise ``pred`` (``denorm_out``) and ``inp`` (``denorm_in``)
        into physical units for the physics residual.
    nu : float
        Laminar kinematic viscosity. The effective viscosity used in the
        residual is ``nu + nut`` (clamped ``>= nu`` inside ``physics_residual_torch``).
    dx, dy : float, optional
        Grid spacing. Defaults derive from ``cfg.data.resolution`` and the
        default :class:`Domain` bounds ``(-1, 2, -1.5, 1.5)``.
    """

    def __init__(
        self,
        cfg: Config,
        normalizer,
        nu: float,
        dx: float | None = None,
        dy: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.normalizer = normalizer
        self.nu = float(nu)
        self.physics_weight = float(cfg.train.physics_weight)
        self.bc_weight = float(cfg.train.bc_weight)

        if dx is None or dy is None:
            # Derive from a representative Domain at the configured resolution.
            res = int(cfg.data.resolution)
            dom = Domain(bounds=(-1.0, 2.0, -1.5, 1.5), nx=res, ny=res)
            dx = dom.dx if dx is None else dx
            dy = dom.dy if dy is None else dy
        self.dx = float(dx)
        self.dy = float(dy)

    def __call__(
        self,
        pred: torch.Tensor,    # (B, 4, H, W) normalised
        target: torch.Tensor,  # (B, 4, H, W) normalised
        inp: torch.Tensor,     # (B, 7, H, W) normalised
        mask: torch.Tensor,    # (B, 1, H, W) physical (1 fluid, 0 solid)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(total_loss, {'loss','data','physics','bc'})``.

        ``pred``/``target`` are in normalised output space, ``inp`` in normalised
        input space, ``mask`` is the physical fluid mask. The physics term is
        skipped (contributes 0) if its residual comes out non-finite.
        """
        fluid = (mask > 0.5).to(pred.dtype)
        n_fluid = fluid.sum().clamp_min(1.0)

        # Per-sample reference speed U from the inlet channels (4,5), for
        # non-dimensionalising the physics/BC residuals (chord length L = 1).
        inp_phys = self.normalizer.denorm_in(inp)
        u_ref = torch.sqrt(
            inp_phys[:, 4:5] ** 2 + inp_phys[:, 5:6] ** 2
        ).clamp_min(1e-6)
        length = 1.0

        # 1) Data term: masked MSE over fluid cells (normalised space).
        diff2 = (pred - target) ** 2
        data = (diff2 * fluid).sum() / (n_fluid * pred.shape[1])

        # 2) Physics term: residual on denormalised (physical) fields, with each
        #    component non-dimensionalised (continuity ~ U/L, momentum ~ U^2/L)
        #    so it is O(1) and comparable to the data term. Previously the raw
        #    residual was ~1e5 vs data ~1, so `physics_weight` did not mean a
        #    fractional weight; now it does.
        physics = pred.new_tensor(0.0)
        physics_val = 0.0
        if self.physics_weight > 0.0:
            from neuroforge.physics.residuals import physics_residual_torch

            pred_phys = self.normalizer.denorm_out(pred)
            res = physics_residual_torch(
                pred_phys, inp_phys, self.dx, self.dy, self.nu
            )
            cont = res["continuity"] / (u_ref / length)
            momx = res["momentum_x"] / (u_ref * u_ref / length)
            momy = res["momentum_y"] / (u_ref * u_ref / length)
            r = cont ** 2 + momx ** 2 + momy ** 2
            # Residuals are already mask-scoped inside physics_residual_torch.
            phys_term = (r * fluid).sum() / n_fluid
            if torch.isfinite(phys_term):
                physics = phys_term
                physics_val = float(phys_term.detach())
            # else: skip (leave physics at 0) — guards against NaN/inf blow-ups.

        # 3) BC term: penalise the (non-dimensional) velocity magnitude in the solid.
        solid = (mask <= 0.5).to(pred.dtype)
        n_solid = solid.sum().clamp_min(1.0)
        pred_phys_v = self.normalizer.denorm_out(pred)[:, 0:2]
        vel2 = (pred_phys_v ** 2).sum(dim=1, keepdim=True) / (u_ref * u_ref)
        bc = (vel2 * solid).sum() / n_solid

        total = data + self.physics_weight * physics + self.bc_weight * bc

        return total, {
            "loss": float(total.detach()),
            "data": float(data.detach()),
            "physics": physics_val,
            "bc": float(bc.detach()),
        }
