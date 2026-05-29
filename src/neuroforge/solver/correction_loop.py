"""Neural Residual Iteration — the self-correcting loop (the breakthrough).

Given a predicted :class:`FlowField`, its physics :class:`Diagnostics`, and a
trained :class:`CorrectionNetwork`, this routine iteratively nudges the field
toward lower physics residual. The crucial guarantee is an **acceptance test**:
a candidate correction is applied only if it does *not* increase the scalar
residual norm; otherwise the step is backtracked (halved) a few times and, if
still no improvement, the loop stops. This makes the residual norm provably
non-increasing across accepted iterations — the demonstrable claim of the engine.

Corrections are computed in normalised space (matching the corrector's training)
and denormalised before being applied to the physical field. With
``cfg.gate_by_trust`` the correction is scaled by ``(1 - trust)`` so it acts
mainly where the field is least trustworthy, and it is always zeroed inside the
solid.
"""

from __future__ import annotations

import numpy as np
import torch

from neuroforge.core.config import CorrectionConfig
from neuroforge.core.types import DTYPE, Diagnostics, FlowCase, FlowField

__all__ = ["neural_residual_iteration"]

_EPS = 1e-12
_MAX_BACKTRACK = 4


def _diag_uncertainty(uq, predictor, case: FlowCase) -> np.ndarray | None:
    """Get a ``(ny, nx)`` uncertainty map from a UQ estimator, or ``None``.

    Uses the predictor's normaliser + encoding so the UQ model sees the same
    normalised input the backbone was trained on. Aggregates the per-channel std
    to a scalar map. Any failure degrades gracefully to ``None``.
    """
    if uq is None:
        return None
    try:
        from neuroforge.geometry.encode import encode_case

        x = encode_case(case)
        xn = predictor.normalizer.norm_in(x)
        xt = torch.from_numpy(np.ascontiguousarray(xn, DTYPE))[None]
        mean, std = uq.predict_with_uncertainty(xt)
        unc = std.mean(dim=1, keepdim=False)[0].cpu().numpy().astype(DTYPE)
        return unc
    except Exception:
        return None


def _residual_tensor(field: FlowField, case: FlowCase, predictor) -> torch.Tensor:
    """Compute the normalised 3-channel residual map ``(1, 3, H, W)`` for ``field``."""
    from neuroforge.physics.residuals import physics_residual_torch

    pred_phys = torch.from_numpy(np.ascontiguousarray(field.as_array(), DTYPE))[None]
    from neuroforge.geometry.encode import encode_case

    inp_phys = torch.from_numpy(
        np.ascontiguousarray(encode_case(case), DTYPE)
    )[None]
    dx, dy = field.domain.dx, field.domain.dy
    nu = float(case.fluid.kinematic_viscosity)
    res = physics_residual_torch(pred_phys, inp_phys, dx, dy, nu)
    residual = torch.cat(
        [res["continuity"], res["momentum_x"], res["momentum_y"]], dim=1
    )
    std = residual.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-6)
    return residual / std.unsqueeze(-1)


def _apply_delta(field: FlowField, delta_phys: np.ndarray, step: float) -> FlowField:
    """Return a new FlowField = field + step * delta_phys (channels u,v,p,nut)."""
    arr = field.as_array() + DTYPE(step) * delta_phys.astype(DTYPE)
    return FlowField.from_array(
        arr, field.domain, mask=field.mask, sdf=field.sdf, meta=dict(field.meta)
    )


def neural_residual_iteration(
    field: FlowField,
    case: FlowCase,
    checker,
    corrector,
    cfg: CorrectionConfig,
    predictor,
    uq=None,
) -> tuple[FlowField, list[dict]]:
    """Run the trust-gated, residual-monotone correction loop.

    Parameters
    ----------
    field : FlowField
        Initial prediction.
    case : FlowCase
        Problem definition.
    checker : PhysicsChecker
        Produces :class:`Diagnostics` (residuals + trust).
    corrector : CorrectionNetwork or None
        Learned residual corrector. If ``None`` the field is diagnosed once and
        returned unchanged.
    cfg : CorrectionConfig
        Loop settings (``max_iters``, ``residual_tol``, ``step_size``,
        ``gate_by_trust``, ``min_improvement``).
    predictor : Predictor
        Supplies the normaliser used to map fields/residuals to the corrector's
        normalised input space.
    uq : optional
        Uncertainty estimator with ``predict_with_uncertainty``.

    Returns
    -------
    tuple
        ``(final_field, history)`` where ``history`` is a list of per-iteration
        dicts ``{'iter','residual_norm','max_uncertainty','trust_mean'}``. The
        ``residual_norm`` values are non-increasing across accepted iterations.
    """
    history: list[dict] = []
    normalizer = predictor.normalizer

    # Initial diagnose (iter 0).
    unc = _diag_uncertainty(uq, predictor, case)
    diag = checker.diagnose(field, case, uncertainty=unc)
    cur_norm = diag.residual_norm()
    history.append(
        {
            "iter": 0,
            "residual_norm": cur_norm,
            "max_uncertainty": float(np.max(diag.uncertainty)) if diag.uncertainty.size else 0.0,
            "trust_mean": float(np.mean(diag.trust)) if diag.trust.size else 1.0,
        }
    )

    if corrector is None:
        return field, history

    corrector.eval()
    mask = field.mask if field.mask is not None else np.ones(field.shape, dtype=DTYPE)
    solid = (np.asarray(mask) <= 0.5)

    for it in range(1, int(cfg.max_iters) + 1):
        if cur_norm < cfg.residual_tol:
            break

        # Build corrector inputs in normalised space.
        field_arr = field.as_array()
        field_norm = normalizer.norm_out(field_arr)
        field_t = torch.from_numpy(np.ascontiguousarray(field_norm, DTYPE))[None]
        residual_t = _residual_tensor(field, case, predictor)
        from neuroforge.geometry.encode import encode_case

        geom_phys = encode_case(case)
        geom_norm = normalizer.norm_in(geom_phys)
        geom_t = torch.from_numpy(np.ascontiguousarray(geom_norm, DTYPE))[None]

        with torch.no_grad():
            delta_norm = corrector(field=field_t, residual=residual_t, geom=geom_t)
        # Denormalise the additive correction back to physical units. The output
        # normaliser is affine (y = std*y_norm + mean), so a *delta* maps by the
        # std only (the mean offset cancels): delta_phys = std * delta_norm.
        delta_norm_np = delta_norm[0].cpu().numpy().astype(DTYPE)
        std_out = normalizer.std_out.reshape(-1, 1, 1).astype(DTYPE)
        delta_phys = delta_norm_np * std_out

        # Trust gating: apply corrections mainly in low-trust regions.
        if cfg.gate_by_trust:
            gate = (1.0 - np.asarray(diag.trust, dtype=DTYPE))[None, :, :]
            delta_phys = delta_phys * gate
        # Never modify the solid interior.
        delta_phys[:, solid] = 0.0

        # Acceptance test with backtracking line search: accept only if the
        # residual norm does NOT increase. Guarantees monotone non-increase.
        step = float(cfg.step_size)
        accepted = False
        for _bt in range(_MAX_BACKTRACK + 1):
            candidate = _apply_delta(field, delta_phys, step)
            cand_unc = _diag_uncertainty(uq, predictor, case)
            cand_diag = checker.diagnose(candidate, case, uncertainty=cand_unc)
            cand_norm = cand_diag.residual_norm()
            if np.isfinite(cand_norm) and cand_norm <= cur_norm + _EPS:
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # No step reduced (or held) the residual -> stop.
            break

        rel_improve = (cur_norm - cand_norm) / max(cur_norm, _EPS)
        field = candidate
        diag = cand_diag
        cur_norm = cand_norm
        history.append(
            {
                "iter": it,
                "residual_norm": cur_norm,
                "max_uncertainty": float(np.max(diag.uncertainty)) if diag.uncertainty.size else 0.0,
                "trust_mean": float(np.mean(diag.trust)) if diag.trust.size else 1.0,
            }
        )

        if rel_improve < cfg.min_improvement:
            break

    return field, history
