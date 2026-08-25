"""Local classical incompressible CFD solver for the uncertainty-gated fallback.

When the engine's trust map flags a sub-region of the flow as unreliable, this
module runs a small, self-contained **steady incompressible** solve on just the
bounding box of that region (plus a halo of surrounding cells), using the
neural prediction on the box border as Dirichlet boundary conditions. The intent
is a *local* classical patch that couples to the surrounding learned solution
without re-solving the whole domain.

The scheme is a deliberately simple SIMPLE/projection iteration on a collocated
grid:

1. **Momentum predictor.** Advance ``u, v`` one pseudo-step of the steady
   momentum balance (convection ``u.grad(u)`` + diffusion ``nu_eff * lap`` -
   ``grad(p)``) with under-relaxation ``relax_u``.
2. **Pressure correction.** Solve a Poisson problem ``lap(p') = div(u*)/dt`` on
   the interior fluid cells (assembled as a sparse system, solved with
   :func:`scipy.sparse.linalg.spsolve`), correct the velocity to be (discretely)
   divergence-free, and under-relax the pressure by ``relax_p``.

The effective viscosity ``nu_eff = nu + nut`` is frozen from the input
prediction (the turbulence field is not evolved). Solid cells (``mask < 0.5``)
inside the box are held at the no-slip condition ``u = v = 0`` and excluded from
the pressure update.

**Scope: small patches only.** The scheme is validated for sub-regions whose box
border carries Dirichlet values from a surrounding solution. It is *not* a
general-purpose solver: run over the whole domain, where that border is the real
far-field and wall boundary, it diverges (measured on 10/10 AirfRANS cases,
``scripts/smoke_fallback_cost.py``). Two known limits drive this -- the collocated
grid has no Rhie-Chow coupling, and ``nu_eff`` is frozen rather than evolved, so
the patch does not solve the RANS system its input came from. Accepted patches
lower the region *residual*; measured against ground truth they do not lower true
field error (``scripts/probe_patch_acceptance.py``, 0/140 trials, and 0 again when
handed exact boundary data). Treat this backend as a residual-reducing local
smoother, not as a correction method.

**No-harm guarantee.** The patch is only accepted if it *reduces* the combined
physics residual norm over the flagged region (continuity + momentum, measured
with the same verifier the engine trusts, scoped to fluid cells). Otherwise the
original field is returned unchanged. This makes the fallback monotone in the
region residual — it can never make the prediction worse — matching the engine's
monotone-residual philosophy.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE, FlowCase, FlowField, FluidProperties
from neuroforge.physics.residuals import continuity_residual, momentum_residual

__all__ = ["local_incompressible_solve", "region_residual_norm"]

_EPS = 1e-12

# Velocity magnitudes beyond this multiple of the incoming field's peak speed
# mean the explicit predictor has gone unstable, not that the flow accelerated.
_DIVERGENCE_FACTOR = 1.0e3

# Below this, the baseline region residual is an exact (or near-exact) zero and
# the no-harm test is vacuous: nothing can beat it. A uniform freestream is the
# canonical case -- it is an exact zero of this operator without being a
# solution of the boundary-value problem.
_DEGENERATE_BASELINE = 1.0e-12


# --------------------------------------------------------------------------- #
# Region residual norm (the no-harm metric)
# --------------------------------------------------------------------------- #


def region_residual_norm(
    field: FlowField, fluid: FluidProperties, region: np.ndarray
) -> float:
    """Combined PDE-residual norm over the fluid cells of ``region``.

    ``N_region = sqrt(mean over (region & fluid) of
    (r_continuity^2 + r_x^2 + r_y^2))`` using the verifier functions
    :func:`~neuroforge.physics.residuals.continuity_residual` and
    :func:`~neuroforge.physics.residuals.momentum_residual` (computed on the
    physical field with ``nu_eff = nu + nut``). Solid cells are excluded.

    Note
    ----
    Unlike :class:`~neuroforge.physics.residuals.PhysicsChecker`, this metric
    does **not** zero the solid-adjacent fluid "wall ring". This matches the
    task's definition of ``N_region`` exactly (raw residuals over
    ``region & fluid``). The caveat is that if a flagged region hugs the body
    the spurious near-wall residual (central differences reaching into the
    ``u = v = 0`` body) can dominate ``N_region``; the solver cannot reduce that
    sub-grid term, so the no-harm test would simply reject the patch there (safe,
    but a no-op). Regions in the bulk flow are unaffected.

    Parameters
    ----------
    field : FlowField
        Physical flow field.
    fluid : FluidProperties
        Supplies the laminar kinematic viscosity for the momentum residual.
    region : numpy.ndarray
        Boolean ``(ny, nx)`` map selecting the cells of interest.

    Returns
    -------
    float
        The region residual norm (``0.0`` if no fluid cell is selected).
    """
    mask = np.asarray(field.mask, dtype=DTYPE) > 0.5
    sel = (np.asarray(region) > 0.5) & mask
    if not np.any(sel):
        return 0.0

    cont = continuity_residual(field).astype(np.float64)
    r_x, r_y = momentum_residual(field, fluid)
    r_x = r_x.astype(np.float64)
    r_y = r_y.astype(np.float64)

    sq = cont[sel] ** 2 + r_x[sel] ** 2 + r_y[sel] ** 2
    return float(np.sqrt(np.mean(sq)))


# --------------------------------------------------------------------------- #
# Sub-box extraction
# --------------------------------------------------------------------------- #


def _bounding_box(region: np.ndarray, halo: int, shape: tuple[int, int]):
    """Inclusive ``(j0, j1, i0, i1)`` bounding box of ``region`` padded by ``halo``.

    Clamped to the grid ``shape == (ny, nx)``. ``j`` is the row (y) index and
    ``i`` the column (x) index. Returns ``None`` if ``region`` is empty.
    """
    ny, nx = shape
    rows = np.any(region, axis=1)
    cols = np.any(region, axis=0)
    if not rows.any() or not cols.any():
        return None
    j_idx = np.where(rows)[0]
    i_idx = np.where(cols)[0]
    j0 = max(0, int(j_idx[0]) - halo)
    j1 = min(ny - 1, int(j_idx[-1]) + halo)
    i0 = max(0, int(i_idx[0]) - halo)
    i1 = min(nx - 1, int(i_idx[-1]) + halo)
    return j0, j1, i0, i1


# --------------------------------------------------------------------------- #
# The local SIMPLE / projection solve
# --------------------------------------------------------------------------- #


def _pressure_poisson_matrix(active: np.ndarray, dx: float, dy: float):
    """Assemble a sparse 5-point Poisson operator on the ``active`` cells.

    Builds the discrete Laplacian (Neumann at non-active neighbours, i.e. a
    zero-flux wall) over the boolean ``(h, w)`` mask ``active`` and returns the
    CSR matrix together with the mapping from active-cell linear index to the
    matrix row. Inactive cells (box border + solid) are excluded; their pressure
    correction is held at zero (Dirichlet ``p' = 0``), which anchors the system
    so it is non-singular.

    Parameters
    ----------
    active : numpy.ndarray
        Boolean ``(h, w)`` map of interior fluid cells to solve for.
    dx, dy : float
        Grid spacings.

    Returns
    -------
    tuple
        ``(A, index_map)`` where ``A`` is a :class:`scipy.sparse.csr_matrix`
        ``(K, K)`` and ``index_map`` is an ``(h, w)`` int array giving the row
        for each active cell (``-1`` for inactive cells).
    """
    from scipy import sparse

    h, w = active.shape
    index_map = np.full((h, w), -1, dtype=np.int64)
    act_idx = np.argwhere(active)
    for k, (j, i) in enumerate(act_idx):
        index_map[j, i] = k
    n = len(act_idx)

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for k, (j, i) in enumerate(act_idx):
        diag = 0.0
        # 4-connected neighbours; a Dirichlet p'=0 is imposed where the
        # neighbour is inactive (border / solid), keeping the operator SPD.
        for jj, ii, coef in (
            (j - 1, i, inv_dy2),
            (j + 1, i, inv_dy2),
            (j, i - 1, inv_dx2),
            (j, i + 1, inv_dx2),
        ):
            diag -= coef
            if 0 <= jj < h and 0 <= ii < w and active[jj, ii]:
                rows.append(k)
                cols.append(index_map[jj, ii])
                vals.append(coef)
        rows.append(k)
        cols.append(k)
        vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A, index_map


def local_incompressible_solve(
    field: FlowField,
    case: FlowCase,
    region_mask: np.ndarray,
    *,
    max_outer: int = 200,
    tol: float = 1e-4,
    relax_u: float = 0.5,
    relax_p: float = 0.3,
    halo: int = 2,
) -> FlowField:
    """Run a local SIMPLE-style incompressible solve on the flagged region.

    A steady incompressible projection iteration is run on the bounding box of
    ``region_mask`` (padded by ``halo`` cells). The neural prediction supplies
    the Dirichlet boundary values on the box border (coupling the patch to the
    surrounding solution) and the frozen ``nu_eff = nu + nut``. The result is
    written back into a copy of the full field over the box interior.

    The solve is only kept if it reduces the region residual norm (see
    :func:`region_residual_norm`); otherwise the original field is returned
    unchanged (a true no-op). Either way ``meta['fallback']`` records the attempt.

    Parameters
    ----------
    field : FlowField
        Current (neural) flow field.
    case : FlowCase
        Problem definition (supplies the fluid viscosity).
    region_mask : numpy.ndarray
        Boolean / ``{0,1}`` ``(ny, nx)`` map of low-trust fluid cells to re-solve.
    max_outer : int, optional
        Maximum number of SIMPLE outer iterations. Default 200.
    tol : float, optional
        Relative-improvement tolerance on the continuity residual used for early
        stopping. Default ``1e-4``.
    relax_u : float, optional
        Velocity under-relaxation factor in ``(0, 1]``. Default 0.5.
    relax_p : float, optional
        Pressure under-relaxation factor in ``(0, 1]``. Default 0.3.
    halo : int, optional
        Number of cells to pad around the region's bounding box. Default 2.

    Returns
    -------
    FlowField
        A new field with the patched region (if accepted) and a populated
        ``meta['fallback']`` dict; the same field (copy) unchanged otherwise.
    """
    region = np.asarray(region_mask) > 0.5
    full_mask = np.asarray(field.mask, dtype=DTYPE) > 0.5
    region = region & full_mask
    n_cells = int(region.sum())

    fluid_props = case.fluid

    def _attach_meta(out: FlowField, ran: bool, before: float, after: float, note: str,
                     diverged: bool = False):
        # A blown-up solve produces a non-finite residual; report it as None so
        # callers (and JSON) get an explicit "no value" rather than a NaN that
        # silently compares False against every threshold.
        after_f = float(after)
        before_f = float(before)
        out.meta = dict(out.meta)
        out.meta["fallback"] = {
            "backend": "numpy",
            "ran": bool(ran),
            "region_cells": n_cells,
            "residual_before": before_f if np.isfinite(before_f) else None,
            "residual_after": after_f if np.isfinite(after_f) else None,
            "diverged": bool(diverged),
            # The no-harm test cannot fire against a zero baseline; flag it so a
            # rejection here is not mistaken for "the patch made things worse".
            "degenerate_baseline": bool(
                np.isfinite(before_f) and before_f <= _DEGENERATE_BASELINE
            ),
            "note": note,
        }
        return out

    # Nothing flagged -> nothing to do.
    if n_cells == 0:
        out = FlowField.from_array(
            field.as_array(), field.domain, mask=field.mask, sdf=field.sdf,
            meta=dict(field.meta),
        )
        return _attach_meta(out, False, 0.0, 0.0, "empty region: no fluid cells flagged")

    bbox = _bounding_box(region, halo, field.shape)
    if bbox is None:  # pragma: no cover - guarded by n_cells above.
        out = FlowField.from_array(
            field.as_array(), field.domain, mask=field.mask, sdf=field.sdf,
            meta=dict(field.meta),
        )
        return _attach_meta(out, False, 0.0, 0.0, "empty bounding box")

    j0, j1, i0, i1 = bbox
    dx, dy = float(field.domain.dx), float(field.domain.dy)

    # Baseline region residual (no-harm reference), computed on the FULL field
    # with the verifier so the metric exactly matches the engine's.
    before = region_residual_norm(field, fluid_props, region)

    # ---- Extract the sub-box -------------------------------------------- #
    sl = (slice(j0, j1 + 1), slice(i0, i1 + 1))
    u = np.asarray(field.u, dtype=np.float64)[sl].copy()
    v = np.asarray(field.v, dtype=np.float64)[sl].copy()
    p = np.asarray(field.p, dtype=np.float64)[sl].copy()
    nut = np.asarray(field.nut, dtype=np.float64)[sl]
    sub_fluid = full_mask[sl]
    h, w = u.shape

    nu = float(fluid_props.kinematic_viscosity)
    nu_eff = np.clip(nu + nut, 0.0, None)

    # No-slip on solid cells inside the box.
    solid = ~sub_fluid
    u[solid] = 0.0
    v[solid] = 0.0

    # Frozen Dirichlet boundary values on the box border (the surrounding
    # neural solution). These rows/cols are never updated below.
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    # Interior fluid cells we actually solve for.
    interior = sub_fluid & ~border
    if not np.any(interior):
        out = FlowField.from_array(
            field.as_array(), field.domain, mask=field.mask, sdf=field.sdf,
            meta=dict(field.meta),
        )
        return _attach_meta(
            out, False, before, before, "no interior fluid cells in box"
        )

    u_bc = u.copy()
    v_bc = v.copy()

    # Pseudo-time step for the explicit predictor: a CFL/diffusion-stable choice.
    # Convective limit ~ min(dx,dy)/|U|; diffusive limit ~ min(dx,dy)^2/(4 nu_eff).
    speed = np.sqrt(u * u + v * v)
    umax = float(speed.max()) + _EPS
    nu_max = float(nu_eff.max()) + _EPS
    hmin = min(dx, dy)
    dt_conv = 0.4 * hmin / umax
    dt_diff = 0.2 * hmin * hmin / nu_max
    dt = float(min(dt_conv, dt_diff))
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.1 * hmin * hmin / nu_max

    # Assemble the pressure-correction operator once (geometry is fixed).
    A, index_map = _pressure_poisson_matrix(interior, dx, dy)
    from scipy.sparse.linalg import spsolve

    def _ddx(f):
        out = np.zeros_like(f)
        out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
        return out

    def _ddy(f):
        out = np.zeros_like(f)
        out[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dy)
        return out

    def _lap(f):
        out = np.zeros_like(f)
        out[1:-1, 1:-1] = (
            (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / (dx * dx)
            + (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dy * dy)
        )
        return out

    # Divergence guard. The explicit predictor can run away long before it
    # produces a non-finite value: magnitudes overflow to something huge but
    # still finite, the loop happily burns every remaining iteration, and the
    # residual check downstream then overflows to inf/NaN. Bail as soon as the
    # velocity leaves any physically plausible band around the field we were
    # handed, so a blown-up solve costs a few iterations instead of all of them
    # and is reported as divergence rather than as a failed no-harm test.
    speed_cap = _DIVERGENCE_FACTOR * (umax + 1.0)
    diverged = False

    prev_cont = np.inf
    iters_run = 0
    for it in range(int(max_outer)):
        iters_run = it + 1

        # ---- Momentum predictor (explicit, under-relaxed) ------------- #
        conv_u = u * _ddx(u) + v * _ddy(u)
        conv_v = u * _ddx(v) + v * _ddy(v)
        diff_u = nu_eff * _lap(u)
        diff_v = nu_eff * _lap(v)
        dpdx = _ddx(p)
        dpdy = _ddy(p)

        u_star = u + relax_u * dt * (-conv_u + diff_u - dpdx)
        v_star = v + relax_u * dt * (-conv_v + diff_v - dpdy)

        # Re-impose boundary / solid conditions on the predictor.
        u_star[border] = u_bc[border]
        v_star[border] = v_bc[border]
        u_star[solid] = 0.0
        v_star[solid] = 0.0

        if not (np.all(np.isfinite(u_star)) and np.all(np.isfinite(v_star))):
            iters_run = it  # last good state was `u, v, p`
            break

        # ---- Pressure correction (projection) ------------------------- #
        div = _ddx(u_star) + _ddy(v_star)
        rhs = (div / dt)[interior]

        try:
            p_corr_flat = spsolve(A, rhs)
        except Exception:  # pragma: no cover - singular/ill-conditioned guard.
            break
        if not np.all(np.isfinite(p_corr_flat)):
            break

        p_corr = np.zeros((h, w), dtype=np.float64)
        p_corr[interior] = p_corr_flat

        # Correct velocity to enforce discrete continuity, then update pressure.
        u_new = u_star - dt * _ddx(p_corr)
        v_new = v_star - dt * _ddy(p_corr)
        p_new = p + relax_p * p_corr

        # Re-impose boundary / solid conditions on the corrected fields.
        u_new[border] = u_bc[border]
        v_new[border] = v_bc[border]
        u_new[solid] = 0.0
        v_new[solid] = 0.0
        p_new[border] = p[border]
        p_new[solid] = p[solid]

        if not (
            np.all(np.isfinite(u_new))
            and np.all(np.isfinite(v_new))
            and np.all(np.isfinite(p_new))
        ):
            diverged = True
            break

        if max(np.abs(u_new).max(), np.abs(v_new).max()) > speed_cap:
            diverged = True
            break

        u, v, p = u_new, v_new, p_new

        # ---- Convergence check on interior continuity ----------------- #
        cont_now = float(np.sqrt(np.mean((div[interior]) ** 2)))
        if cont_now < _EPS:
            break
        if np.isfinite(prev_cont) and abs(prev_cont - cont_now) <= tol * (
            prev_cont + _EPS
        ):
            break
        prev_cont = cont_now

    # ---- Write the solved sub-box back into a copy of the full field --- #
    arr = field.as_array().astype(np.float64)
    patched = arr.copy()
    patched[0][sl] = u
    patched[1][sl] = v
    patched[2][sl] = p
    # Keep solid cells at zero velocity in the patched region.
    pj, pi = np.where(solid)
    patched[0][j0 + pj, i0 + pi] = 0.0
    patched[1][j0 + pj, i0 + pi] = 0.0

    candidate = FlowField.from_array(
        patched.astype(DTYPE), field.domain, mask=field.mask, sdf=field.sdf,
        meta=dict(field.meta),
    )

    after = region_residual_norm(candidate, fluid_props, region)

    # ---- No-harm acceptance test -------------------------------------- #
    # Accept only if the region residual strictly improved (tiny slack to avoid
    # rejecting on float noise when already near machine precision).
    improved = bool(
        np.isfinite(after) and after < before - 1e-12 * max(before, 1.0)
    )
    if improved:
        return _attach_meta(
            candidate,
            True,
            before,
            after,
            (
                f"local SIMPLE solve on a {h}x{w} sub-box ({iters_run} iters); "
                f"region residual {before:.4e} -> {after:.4e}"
            ),
        )

    # Reject: return the ORIGINAL field unchanged (true no-op), but record it.
    out = FlowField.from_array(
        field.as_array(), field.domain, mask=field.mask, sdf=field.sdf,
        meta=dict(field.meta),
    )
    if diverged:
        why = (
            f"local SIMPLE solve on a {h}x{w} sub-box DIVERGED after {iters_run} "
            "iters (velocity left the plausible band or went non-finite); patch "
            "rejected, field returned unchanged"
        )
    elif np.isfinite(before) and before <= _DEGENERATE_BASELINE:
        why = (
            f"local SIMPLE solve on a {h}x{w} sub-box ({iters_run} iters): the "
            f"baseline region residual is {before:.4e}, an exact zero of this "
            "operator, so the no-harm test cannot fire -- rejection here says "
            "nothing about the patch's quality; field returned unchanged"
        )
    else:
        why = (
            f"local SIMPLE solve on a {h}x{w} sub-box ({iters_run} iters) did not "
            f"reduce the region residual ({before:.4e} -> {after:.4e}); "
            "patch rejected (no-harm guarantee), field returned unchanged"
        )
    return _attach_meta(out, False, before, after, why, diverged=diverged)
