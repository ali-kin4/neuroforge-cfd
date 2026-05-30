"""Engineering metrics derived from a predicted flow field.

Includes the pressure coefficient field, integrated force coefficients
(lift / drag / moment) obtained by sampling the grid solution onto the body
surface, the wall-shear-stress field, and masked relative-L2 field errors.

All routines are pure numpy and operate on **physical** fields with kinematic
pressure (``p = p/rho``); the fluid density therefore appears only as an
explicit ``fluid.density`` factor in the dimensional wall shear stress.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE, FlowCase, FlowField

from .operators import gradient

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Grid <-> surface sampling (bilinear, pure numpy)
# --------------------------------------------------------------------------- #


def _bilinear_sample(field2d: np.ndarray, domain, pts: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a ``(ny, nx)`` field at world points ``pts`` (M, 2).

    Points outside the domain are clamped to the boundary (nearest valid cell),
    which is appropriate for sampling a body that sits inside the domain.

    Parameters
    ----------
    field2d : numpy.ndarray
        Scalar field on the grid, shape ``(ny, nx)``.
    domain : Domain
        Provides bounds and resolution.
    pts : numpy.ndarray
        ``(M, 2)`` world coordinates ``(x, y)``.

    Returns
    -------
    numpy.ndarray
        ``(M,)`` interpolated values ``float64``.
    """
    field2d = np.asarray(field2d, dtype=np.float64)
    xmin, xmax, ymin, ymax = domain.bounds
    nx, ny = domain.nx, domain.ny
    dx, dy = domain.dx, domain.dy

    x = np.asarray(pts[:, 0], dtype=np.float64)
    y = np.asarray(pts[:, 1], dtype=np.float64)

    # Fractional grid indices (column = x, row = y).
    fx = (x - xmin) / dx if dx > _EPS else np.zeros_like(x)
    fy = (y - ymin) / dy if dy > _EPS else np.zeros_like(y)

    fx = np.clip(fx, 0.0, nx - 1.0)
    fy = np.clip(fy, 0.0, ny - 1.0)

    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)

    tx = fx - i0
    ty = fy - j0

    # field2d is indexed [row=j (y), col=i (x)].
    f00 = field2d[j0, i0]
    f10 = field2d[j0, i1]
    f01 = field2d[j1, i0]
    f11 = field2d[j1, i1]

    return (
        f00 * (1 - tx) * (1 - ty)
        + f10 * tx * (1 - ty)
        + f01 * (1 - tx) * ty
        + f11 * tx * ty
    )


def _surface_points_normals(case: FlowCase) -> tuple[np.ndarray, np.ndarray]:
    """Ordered surface points ``(N, 2)`` and outward unit normals ``(N, 2)``.

    Uses ``geometry.surface_normals`` if present; otherwise estimates outward
    normals from the counter-clockwise point ordering: for a CCW loop the
    outward normal of the local tangent ``t = (tx, ty)`` is ``(ty, -tx)``.
    """
    geom = case.geometry
    pts = np.asarray(geom.surface_points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if geom.surface_normals is not None and len(geom.surface_normals) == n:
        nrm = np.asarray(geom.surface_normals, dtype=np.float64).reshape(-1, 2)
    else:
        # Central-difference tangents around the closed loop.
        t = np.zeros_like(pts)
        t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)  # outward for CCW ordering
    # Normalise.
    mag = np.sqrt((nrm ** 2).sum(axis=1, keepdims=True))
    nrm = nrm / np.maximum(mag, _EPS)
    return pts, nrm


# --------------------------------------------------------------------------- #
# Pressure coefficient
# --------------------------------------------------------------------------- #


def pressure_coefficient(field: FlowField, case: FlowCase) -> np.ndarray:
    """Pressure-coefficient field ``Cp = (p - p_inf) / (0.5 * U_inf^2)``.

    Pressure is kinematic, so the freestream kinematic pressure ``p_inf`` is
    taken as ``0`` (gauge). The dynamic head uses ``U_inf = case.bc.u_inf``.

    Parameters
    ----------
    field : FlowField
        Physical flow field.
    case : FlowCase
        Supplies ``U_inf``.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` ``Cp`` map ``float32``.
    """
    u_inf = float(case.bc.u_inf)
    q = 0.5 * u_inf * u_inf
    p_inf = 0.0
    cp = (np.asarray(field.p, dtype=np.float64) - p_inf) / max(q, _EPS)
    return cp.astype(DTYPE)


# --------------------------------------------------------------------------- #
# Wall shear stress
# --------------------------------------------------------------------------- #


def wall_shear_stress(field: FlowField, case: FlowCase) -> np.ndarray:
    """Wall-shear-stress magnitude field ``tau_w ~ rho * nu_eff * du_t/dn``.

    Computed on the whole grid as ``rho * nu_eff * |grad U_t . n|`` where the
    wall-normal direction ``n`` is taken from the (normalised) gradient of the
    signed distance field and ``U_t`` is the velocity component tangential to
    the wall. The physically meaningful values sit in the near-wall band; the
    field is returned everywhere for convenience and masking by the caller.

    Parameters
    ----------
    field : FlowField
        Physical flow field (uses ``sdf`` for the wall-normal direction).
    case : FlowCase
        Supplies fluid density and laminar viscosity.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` ``tau_w`` magnitude ``float32`` (dimensional, includes rho).
    """
    dx, dy = field.domain.dx, field.domain.dy
    u = np.asarray(field.u, dtype=np.float64)
    v = np.asarray(field.v, dtype=np.float64)

    rho = float(case.fluid.density)
    nut = np.asarray(field.nut, dtype=np.float64) if field.nut is not None else 0.0
    nu_eff = np.clip(float(case.fluid.kinematic_viscosity) + nut, 0.0, None)

    # Wall-normal direction from the SDF gradient (points outward from body).
    sdf = np.asarray(field.sdf, dtype=np.float64)
    nx_, ny_ = gradient(sdf, dx, dy)
    nmag = np.sqrt(nx_ ** 2 + ny_ ** 2)
    nmag = np.where(nmag < _EPS, 1.0, nmag)
    nx_ /= nmag
    ny_ /= nmag

    # Tangential direction (rotate normal by 90 deg).
    tx_, ty_ = -ny_, nx_

    # Tangential velocity and its wall-normal derivative (grad . n).
    u_t = u * tx_ + v * ty_
    dut_dx, dut_dy = gradient(u_t, dx, dy)
    dut_dn = dut_dx * nx_ + dut_dy * ny_

    tau = rho * nu_eff * np.abs(dut_dn)
    return tau.astype(DTYPE)


# --------------------------------------------------------------------------- #
# Force coefficients
# --------------------------------------------------------------------------- #


def force_coefficients(field: FlowField, case: FlowCase) -> dict[str, float]:
    """Integrate surface pressure + wall shear into ``{'cl', 'cd', 'cm'}``.

    Method
    ------
    The body surface is traced by the ordered (CCW) ``geometry.surface_points``
    with outward unit normals ``n``. For each surface segment of length ``ds``
    centred at midpoint ``x_m`` we sample the grid pressure and wall shear by
    bilinear interpolation and accumulate the sectional force per unit span

    ``dF = (-p * n + tau_w * t) * ds``

    where ``t`` is the unit tangent (CCW). Pressure is kinematic, so the
    resulting force is *per unit density* (i.e. ``F/rho``); dividing by the
    kinematic dynamic head ``0.5 * U_inf^2 * chord`` cancels the density and
    yields dimensionless coefficients.

    Sign conventions
    ----------------
    * ``cd`` (drag) is positive in the freestream flow direction.
    * ``cl`` (lift) is positive perpendicular to the freestream, rotated +90 deg
      (i.e. for ``AoA = 0`` and flow in +x, positive lift is in +y).
    * ``cm`` (pitching moment) is taken about the quarter-chord point
      ``(x_le + 0.25 * chord, y_le)``, positive nose-up (counter-clockwise,
      ``r x F`` with ``z`` out of plane), non-dimensionalised by an extra factor
      of ``chord``.

    Parameters
    ----------
    field : FlowField
        Physical flow field.
    case : FlowCase
        Geometry + freestream conditions.

    Returns
    -------
    dict
        ``{'cl': float, 'cd': float, 'cm': float}``.
    """
    pts, nrm = _surface_points_normals(case)
    n = pts.shape[0]
    if n < 3:
        return {"cl": 0.0, "cd": 0.0, "cm": 0.0}

    # Segment geometry: segment i connects pts[i] -> pts[i+1] (closed loop).
    p_next = np.roll(pts, -1, axis=0)
    seg = p_next - pts
    ds = np.sqrt((seg ** 2).sum(axis=1))
    ds = np.maximum(ds, _EPS)
    mid = 0.5 * (pts + p_next)
    # Outward normal at the midpoint (average of the two endpoint normals).
    nrm_next = np.roll(nrm, -1, axis=0)
    nmid = 0.5 * (nrm + nrm_next)
    nmid /= np.maximum(np.sqrt((nmid ** 2).sum(axis=1, keepdims=True)), _EPS)
    # Unit tangent along the (CCW) traversal direction.
    tmid = seg / ds[:, None]

    # Sample pressure / wall shear a small distance OUTWARD along the normal so
    # the bilinear stencil stays in the fluid. Sampling exactly on the surface
    # line straddles solid cells (whose pressure is an interior fill), which
    # collapses the pressure variation and drives Cl -> 0. Offsetting by ~1.5
    # cells recovers the true near-wall pressure (verified ~10-20x more spread).
    cell = 0.5 * (field.domain.dx + field.domain.dy)
    sample_pts = mid + (1.5 * cell) * nmid
    p_s = _bilinear_sample(field.p, field.domain, sample_pts)        # kinematic p
    tau_field = wall_shear_stress(field, case)
    tau_s = _bilinear_sample(tau_field, field.domain, sample_pts)    # dimensional
    rho = float(case.fluid.density)
    tau_kin = tau_s / max(rho, _EPS)  # per-density, to match kinematic pressure

    # Sectional force per unit span, per unit density:
    #   pressure acts along -n (compression onto the surface),
    #   viscous shear acts along +t (in the traversal/flow-aligned direction).
    fx = (-p_s * nmid[:, 0] + tau_kin * tmid[:, 0]) * ds
    fy = (-p_s * nmid[:, 1] + tau_kin * tmid[:, 1]) * ds

    Fx = float(fx.sum())
    Fy = float(fy.sum())

    # Moment about the quarter-chord, positive counter-clockwise (nose-up).
    xmin, xmax, ymin, ymax = case.geometry.bounding_box()
    chord = max(case.reference_length(), _EPS)
    x_ref = xmin + 0.25 * chord
    y_ref = 0.5 * (ymin + ymax)
    rx = mid[:, 0] - x_ref
    ry = mid[:, 1] - y_ref
    Mz = float((rx * fy - ry * fx).sum())

    # Rotate body-axis forces into wind axes by the angle of attack.
    a = np.deg2rad(float(case.bc.aoa_deg))
    ca, sa = np.cos(a), np.sin(a)
    drag = Fx * ca + Fy * sa          # along freestream
    lift = -Fx * sa + Fy * ca         # perpendicular (+90 deg from freestream)

    u_inf = float(case.bc.u_inf)
    q = 0.5 * u_inf * u_inf           # kinematic dynamic head
    denom = max(q * chord, _EPS)

    cl = lift / denom
    cd = drag / denom
    cm = Mz / (denom * chord)
    return {"cl": float(cl), "cd": float(cd), "cm": float(cm)}


# --------------------------------------------------------------------------- #
# Field errors
# --------------------------------------------------------------------------- #


def field_errors(pred: FlowField, ref: FlowField) -> dict[str, float]:
    """Relative-L2 errors of ``pred`` vs ``ref``, masked to the fluid region.

    For each field ``relL2 = ||pred - ref||_2 / (||ref||_2 + eps)`` computed over
    fluid cells only. ``'speed'`` uses the velocity magnitude.

    Parameters
    ----------
    pred, ref : FlowField
        Predicted and reference fields on the same grid.

    Returns
    -------
    dict
        ``{'u', 'v', 'p', 'speed'}`` relative-L2 errors (``float``).
    """
    # Use the reference mask if available, else the prediction's, else all-fluid.
    mask = ref.mask if ref.mask is not None else pred.mask
    if mask is None:
        mask = np.ones(pred.shape, dtype=DTYPE)
    m = np.asarray(mask) > 0.5
    if not np.any(m):
        m = np.ones(pred.shape, dtype=bool)

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)[m]
        b = np.asarray(b, dtype=np.float64)[m]
        num = float(np.sqrt(np.sum((a - b) ** 2)))
        den = float(np.sqrt(np.sum(b ** 2)))
        return num / (den + _EPS)

    return {
        "u": rel(pred.u, ref.u),
        "v": rel(pred.v, ref.v),
        "p": rel(pred.p, ref.p),
        "speed": rel(pred.speed(), ref.speed()),
    }


__all__ = [
    "pressure_coefficient",
    "force_coefficients",
    "wall_shear_stress",
    "field_errors",
]
