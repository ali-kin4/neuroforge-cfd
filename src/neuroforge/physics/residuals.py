"""Steady incompressible RANS residuals — the physics *verifier*.

These functions quantify how well a :class:`~neuroforge.core.types.FlowField`
satisfies the governing equations and boundary conditions. They operate on
**physical (denormalised)** fields with kinematic pressure (``p = p/rho``), so
the momentum balance carries no explicit density. The effective viscosity is
``nu_eff = fluid.kinematic_viscosity + field.nut`` (laminar + turbulent eddy
viscosity), evaluated pointwise.

Governing equations (steady, incompressible, 2-D RANS in primitive form)::

    continuity:  du/dx + dv/dy = 0
    x-momentum:  u du/dx + v du/dy + dp/dx - nu_eff * lap(u) = 0
    y-momentum:  u dv/dx + v dv/dy + dp/dy - nu_eff * lap(v) = 0

The residual of each equation is its left-hand side (zero for an exact
solution). A separate :func:`bc_violation` map captures no-slip on the body and
free-stream mismatch on the outer boundary. :class:`PhysicsChecker` assembles
all of these into a :class:`~neuroforge.core.types.Diagnostics` (including a
trust map). :func:`physics_residual_torch` is the differentiable twin used in
the training loss.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.config import PhysicsConfig
from neuroforge.core.types import (
    DTYPE,
    Diagnostics,
    FluidProperties,
    FlowCase,
    FlowField,
)

from .operators import ddx, ddy, laplacian
from .trust import trust_map

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Field-level residuals (numpy, physical fields)
# --------------------------------------------------------------------------- #


def _nu_eff(field: FlowField, fluid: FluidProperties) -> np.ndarray:
    """Pointwise effective viscosity ``nu_eff = nu_laminar + nut`` (``>= 0``)."""
    nut = field.nut
    if nut is None:
        nut = np.zeros(field.shape, dtype=DTYPE)
    nu_eff = float(fluid.kinematic_viscosity) + np.asarray(nut, dtype=DTYPE)
    return np.clip(nu_eff, 0.0, None).astype(DTYPE)


def continuity_residual(field: FlowField) -> np.ndarray:
    """Continuity residual ``du/dx + dv/dy``.

    Parameters
    ----------
    field : FlowField
        Physical flow field on the structured grid.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` residual map ``float32``.
    """
    dx, dy = field.domain.dx, field.domain.dy
    res = ddx(field.u, dx) + ddy(field.v, dy)
    return res.astype(DTYPE)


def momentum_residual(
    field: FlowField, fluid: FluidProperties
) -> tuple[np.ndarray, np.ndarray]:
    """Steady incompressible RANS momentum residuals ``(r_x, r_y)``.

    ``r_x = u du/dx + v du/dy + dp/dx - nu_eff * lap(u)`` and the analogous
    ``r_y``. Pressure is kinematic, so there is no division by density.

    Parameters
    ----------
    field : FlowField
        Physical flow field.
    fluid : FluidProperties
        Supplies the laminar kinematic viscosity; combined with ``field.nut``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(r_x, r_y)`` each ``(ny, nx)`` ``float32``.
    """
    dx, dy = field.domain.dx, field.domain.dy
    u, v, p = field.u, field.v, field.p
    nu_eff = _nu_eff(field, fluid)

    dudx = ddx(u, dx)
    dudy = ddy(u, dy)
    dvdx = ddx(v, dx)
    dvdy = ddy(v, dy)
    dpdx = ddx(p, dx)
    dpdy = ddy(p, dy)
    lap_u = laplacian(u, dx, dy)
    lap_v = laplacian(v, dx, dy)

    r_x = u * dudx + v * dudy + dpdx - nu_eff * lap_u
    r_y = u * dvdx + v * dvdy + dpdy - nu_eff * lap_v
    return r_x.astype(DTYPE), r_y.astype(DTYPE)


# --------------------------------------------------------------------------- #
# Boundary-condition violation
# --------------------------------------------------------------------------- #


def _surface_proximity_weight(field: FlowField) -> np.ndarray:
    """Weight in ``[0, 1]`` that is largest for fluid cells touching the wall.

    The weight decays with the absolute signed distance ``|sdf|`` over a length
    scale of a few grid cells, so the no-slip penalty concentrates in the thin
    band of fluid immediately adjacent to the body surface (where velocity
    should be approaching zero).
    """
    sdf = field.sdf
    if sdf is None:
        return np.zeros(field.shape, dtype=DTYPE)
    sdf = np.asarray(sdf, dtype=np.float64)

    dx, dy = field.domain.dx, field.domain.dy
    # Length scale: ~3 cells. Avoids an all-zero weight on coarse grids.
    length = 3.0 * float(min(dx, dy))
    if length < _EPS:
        length = _EPS
    w = np.exp(-np.abs(sdf) / length)
    return w.astype(DTYPE)


def _solid_adjacent_fluid(mask: np.ndarray) -> np.ndarray:
    """Boolean map of fluid cells (mask==1) that have a solid 4-neighbour.

    These are exactly the cells where the mask transitions solid -> fluid, i.e.
    the discrete wall layer on the fluid side.
    """
    m = np.asarray(mask) > 0.5  # True in fluid
    solid = ~m
    neigh_solid = np.zeros_like(m)
    # 4-connected dilation of the solid region, intersected with fluid.
    neigh_solid[:-1, :] |= solid[1:, :]
    neigh_solid[1:, :] |= solid[:-1, :]
    neigh_solid[:, :-1] |= solid[:, 1:]
    neigh_solid[:, 1:] |= solid[:, :-1]
    return m & neigh_solid


def bc_violation(field: FlowField, case: FlowCase) -> np.ndarray:
    """Boundary-condition violation map (no-slip on body + far-field mismatch).

    Two contributions are summed into one ``(ny, nx)`` map:

    1. **No-slip near the body.** In the fluid band adjacent to the surface the
       velocity magnitude should go to zero. We penalise ``|U|`` weighted by a
       proximity factor that decays with ``|sdf|`` (largest at the wall), and we
       additionally emphasise the discrete solid->fluid transition cells.
    2. **Far-field inlet mismatch.** On the outer domain boundary the velocity
       should equal the freestream ``case.bc.inlet_vector()``. We penalise the
       magnitude of the deviation ``(u - u_inf, v - v_inf)`` on the one-cell
       border ring.

    Parameters
    ----------
    field : FlowField
        Physical flow field (uses ``field.mask`` and ``field.sdf``).
    case : FlowCase
        Supplies the inlet/freestream vector.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` non-negative violation magnitude ``float32``.
    """
    ny, nx = field.shape
    u = np.asarray(field.u, dtype=DTYPE)
    v = np.asarray(field.v, dtype=DTYPE)
    speed = np.sqrt(u * u + v * v)

    mask = field.mask
    if mask is None:
        mask = np.ones(field.shape, dtype=DTYPE)
    fluid = np.asarray(mask, dtype=DTYPE)

    viol = np.zeros((ny, nx), dtype=np.float64)

    # 1) No-slip near the body: proximity-weighted speed in the fluid.
    prox = _surface_proximity_weight(field).astype(np.float64)
    noslip = prox * speed.astype(np.float64) * (fluid > 0.5)
    # Emphasise the discrete wall layer (solid-adjacent fluid cells).
    wall_layer = _solid_adjacent_fluid(fluid)
    noslip[wall_layer] = np.maximum(noslip[wall_layer], speed[wall_layer].astype(np.float64))
    viol += noslip

    # 2) Far-field inlet mismatch on the outer one-cell border ring.
    u_in, v_in = case.bc.inlet_vector()
    dev = np.sqrt((u - DTYPE(u_in)) ** 2 + (v - DTYPE(v_in)) ** 2).astype(np.float64)
    border = np.zeros((ny, nx), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    # Only count outer-boundary cells that lie in the fluid.
    border &= fluid > 0.5
    viol[border] += dev[border]

    return viol.astype(DTYPE)


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #


class PhysicsChecker:
    """Turn a :class:`FlowField` + :class:`FlowCase` into :class:`Diagnostics`.

    Parameters
    ----------
    cfg : PhysicsConfig, optional
        Trust-map weights/thresholds and residual settings. Defaults to a fresh
        :class:`~neuroforge.core.config.PhysicsConfig`.
    """

    def __init__(self, cfg: PhysicsConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else PhysicsConfig()

    # -- raw residual maps -------------------------------------------------- #
    def residuals(self, field: FlowField, case: FlowCase) -> dict[str, np.ndarray]:
        """Compute the four residual maps on physical fields.

        Returns
        -------
        dict
            Keys ``'continuity'``, ``'momentum_x'``, ``'momentum_y'``, ``'bc'``;
            each value is a ``(ny, nx)`` ``float32`` map.
        """
        cont = continuity_residual(field)
        r_x, r_y = momentum_residual(field, case.fluid)
        bc = bc_violation(field, case)
        return {
            "continuity": cont,
            "momentum_x": r_x,
            "momentum_y": r_y,
            "bc": bc,
        }

    # -- full diagnostics --------------------------------------------------- #
    def diagnose(
        self,
        field: FlowField,
        case: FlowCase,
        uncertainty: np.ndarray | None = None,
    ) -> Diagnostics:
        """Assemble full :class:`Diagnostics` including the trust map.

        Residuals are computed on the physical field and then **zeroed inside
        the solid** (where the governing equations do not apply). The trust map
        fuses the combined residual magnitude with ``uncertainty`` (zeros if
        ``None``) via :func:`neuroforge.physics.trust.trust_map`. ``summary``
        records the mean/max of each residual (over the fluid) and the fraction
        of fluid cells in each trust class.

        Parameters
        ----------
        field : FlowField
            Physical flow field.
        case : FlowCase
            Problem definition.
        uncertainty : numpy.ndarray, optional
            Per-cell predictive uncertainty ``(ny, nx)``. Defaults to zeros.

        Returns
        -------
        Diagnostics
        """
        res = self.residuals(field, case)
        cont = res["continuity"]
        r_x = res["momentum_x"]
        r_y = res["momentum_y"]
        bc = res["bc"]

        mask = field.mask
        if mask is None:
            mask = np.ones(field.shape, dtype=DTYPE)
        fluid = np.asarray(mask, dtype=DTYPE)
        fluid_bool = fluid > 0.5

        # Zero residuals inside the solid (equations do not apply there).
        solid = ~fluid_bool
        cont = cont.copy()
        r_x = r_x.copy()
        r_y = r_y.copy()
        cont[solid] = 0.0
        r_x[solid] = 0.0
        r_y[solid] = 0.0
        # bc_violation is already fluid-scoped, but keep it consistent.
        bc = bc.copy()
        bc[solid] = 0.0

        if uncertainty is None:
            unc = np.zeros(field.shape, dtype=DTYPE)
        else:
            unc = np.asarray(uncertainty, dtype=DTYPE)
            if unc.shape != field.shape:
                unc = np.zeros(field.shape, dtype=DTYPE)

        # Combined PDE-residual magnitude drives the trust map (BC folded in).
        residual_mag = np.sqrt(
            cont.astype(np.float64) ** 2
            + r_x.astype(np.float64) ** 2
            + r_y.astype(np.float64) ** 2
            + bc.astype(np.float64) ** 2
        ).astype(DTYPE)

        # Physical reference scale for normalising residuals so the trust
        # thresholds are absolute, not self-relative. The momentum residual
        # scales like the advective term U^2 / L; continuity like U / L. We use
        # the advective scale as the reference and add the continuity scale so a
        # near-zero residual reads as high trust regardless of the field's own
        # spread.
        # CONTRACT-NOTE: PhysicsConfig has no absolute residual-scale field, so
        # the reference scale is derived here from the case (U_inf, chord). If a
        # configurable scale is desired, add e.g. `trust_residual_scale` to
        # PhysicsConfig and thread it through.
        u_inf = float(case.bc.u_inf)
        length = max(float(case.reference_length()), 1e-9)
        residual_scale = (u_inf * u_inf / length) + (u_inf / length)
        if residual_scale < 1e-12:
            residual_scale = None  # fall back to percentile self-scaling

        trust, trust_class = trust_map(
            residual_mag, unc, self.cfg, mask=fluid, residual_scale=residual_scale
        )
        # Inside the solid there is nothing to trust/distrust: mark neutral.
        trust = trust.copy()
        trust_class = trust_class.copy()
        trust[solid] = 1.0
        trust_class[solid] = 2.0

        summary = self._summarise(
            cont, r_x, r_y, bc, residual_mag, trust_class, fluid_bool
        )

        return Diagnostics(
            continuity=cont,
            momentum_x=r_x,
            momentum_y=r_y,
            bc_violation=bc,
            uncertainty=unc,
            trust=trust,
            trust_class=trust_class,
            summary=summary,
        )

    @staticmethod
    def _summarise(
        cont: np.ndarray,
        r_x: np.ndarray,
        r_y: np.ndarray,
        bc: np.ndarray,
        residual_mag: np.ndarray,
        trust_class: np.ndarray,
        fluid_bool: np.ndarray,
    ) -> dict[str, float]:
        """Build the scalar summary dict (means/maxes + trust fractions)."""

        def _mean(a: np.ndarray) -> float:
            vals = np.abs(a[fluid_bool]) if np.any(fluid_bool) else np.abs(a).ravel()
            return float(vals.mean()) if vals.size else 0.0

        def _max(a: np.ndarray) -> float:
            vals = np.abs(a[fluid_bool]) if np.any(fluid_bool) else np.abs(a).ravel()
            return float(vals.max()) if vals.size else 0.0

        n_fluid = int(fluid_bool.sum())
        if n_fluid > 0:
            tc = trust_class[fluid_bool]
            frac_green = float(np.mean(tc == 2.0))
            frac_yellow = float(np.mean(tc == 1.0))
            frac_red = float(np.mean(tc == 0.0))
        else:
            frac_green = frac_yellow = frac_red = 0.0

        return {
            "continuity_mean": _mean(cont),
            "continuity_max": _max(cont),
            "momentum_x_mean": _mean(r_x),
            "momentum_x_max": _max(r_x),
            "momentum_y_mean": _mean(r_y),
            "momentum_y_max": _max(r_y),
            "bc_mean": _mean(bc),
            "bc_max": _max(bc),
            "residual_mag_mean": _mean(residual_mag),
            "residual_mag_max": _max(residual_mag),
            "trust_green_frac": frac_green,
            "trust_yellow_frac": frac_yellow,
            "trust_red_frac": frac_red,
        }


# --------------------------------------------------------------------------- #
# Differentiable residuals for the training loss (torch only)
# --------------------------------------------------------------------------- #


def _ddx_t(f, dx: float):
    """Differentiable ``df/dx`` on ``(B, C, H, W)`` torch tensors (last axis)."""
    return ddx(f, dx)


def _ddy_t(f, dy: float):
    """Differentiable ``df/dy`` on ``(B, C, H, W)`` torch tensors (axis -2)."""
    return ddy(f, dy)


def physics_residual_torch(
    pred,
    inp,
    dx: float,
    dy: float,
    nu,
) -> dict:
    """Differentiable steady-RANS residuals for the physics-informed loss.

    Fully differentiable (torch ops only) so gradients flow back to ``pred``.
    ``pred`` holds **physical** fields (already denormalised by the caller).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted fields ``(B, 4, H, W)`` in ``OUTPUT_CHANNELS`` order
        ``(u, v, p, nut)``; physical units, kinematic pressure.
    inp : torch.Tensor
        Network input stack ``(B, 7, H, W)`` in ``INPUT_CHANNELS`` order. Only
        the ``mask`` channel (index 1) is used here, to zero residuals inside
        the solid; the geometry/coordinate channels are accepted for signature
        compatibility.
    dx, dy : float
        Grid spacings.
    nu : float or torch.Tensor
        Laminar kinematic viscosity. Effective viscosity is
        ``nu_eff = nu + nut`` clamped to be ``>= nu`` (so eddy viscosity never
        reduces total diffusion).

    Returns
    -------
    dict
        ``{'continuity', 'momentum_x', 'momentum_y'}`` each ``(B, 1, H, W)``.
    """
    import torch  # local import: this function is torch-only by contract.

    if not isinstance(pred, torch.Tensor):
        raise TypeError("physics_residual_torch expects a torch.Tensor `pred`.")

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]
    nut = pred[:, 3:4]

    # Effective viscosity nu_eff = nu + nut, clamped so the eddy viscosity can
    # never *reduce* the total diffusion below the laminar value (nu_eff >= nu).
    # Works for a scalar nu (fast path) or a broadcastable tensor nu.
    if isinstance(nu, torch.Tensor):
        nu_b = nu.to(device=pred.device, dtype=pred.dtype)
        nu_eff = torch.maximum(nu_b + nut, nu_b + torch.zeros_like(nut))
    else:
        nu_f = float(nu)
        nu_eff = torch.clamp(nu_f + nut, min=nu_f)

    dudx = _ddx_t(u, dx)
    dudy = _ddy_t(u, dy)
    dvdx = _ddx_t(v, dx)
    dvdy = _ddy_t(v, dy)
    dpdx = _ddx_t(p, dx)
    dpdy = _ddy_t(p, dy)

    lap_u = _ddx_t(_ddx_t(u, dx), dx) + _ddy_t(_ddy_t(u, dy), dy)
    lap_v = _ddx_t(_ddx_t(v, dx), dx) + _ddy_t(_ddy_t(v, dy), dy)

    cont = dudx + dvdy
    mom_x = u * dudx + v * dudy + dpdx - nu_eff * lap_u
    mom_y = u * dvdx + v * dvdy + dpdy - nu_eff * lap_v

    # Zero residuals inside the solid using the mask channel (index 1).
    if inp is not None and isinstance(inp, torch.Tensor) and inp.shape[1] > 1:
        mask = inp[:, 1:2]
        mask = (mask > 0.5).to(pred.dtype)
        cont = cont * mask
        mom_x = mom_x * mask
        mom_y = mom_y * mask

    return {"continuity": cont, "momentum_x": mom_x, "momentum_y": mom_y}


__all__ = [
    "continuity_residual",
    "momentum_residual",
    "bc_violation",
    "PhysicsChecker",
    "physics_residual_torch",
]
