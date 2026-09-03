"""Will this surrogate's output format survive being handed to a solver?

The question this module answers costs no solve, no network and no data. It
needs two numbers a CFD engineer already has -- the wall-normal position of the
mesh's first cell centre, and the wall-normal position of the surrogate
representation's first station -- and it returns how badly the representation
will misreport the first-cell wall gradient, which is what viscous drag
integrates.

**The mechanism, and why a closed form exists.** Resampling a field through a
grid gives every mesh cell nearer the wall than the grid's first station the
velocity belonging to that station: there is nothing else to give it. The
first-cell gradient ``u_t / y`` is then overestimated by the ratio of the true
velocity at the station to the true velocity at the cell centre. In wall units
that ratio is a property of the *law of the wall alone*:

    G = u+(y+ of the first station) / u+(y+ of the first cell centre)

There is no fitted parameter in it. Measured against five cases at five
first-station heights spanning a factor of fifty, it **over-predicts in every
row, by 1.3x to 2.6x** (``results/closed_form_validation.json``). So it is an
**upper bound** on the damage -- correct in direction and in ordering, and never
optimistic -- rather than a point estimate.

An earlier version of this docstring claimed agreement to 13%. That figure was
measured before :func:`neuroforge.solver.warmstart.clustered_seed` was fixed to
take its wall-normal coordinate from ``wall_distance`` (point-to-segment) rather
than ``surface_coords`` (nearest vertex, which overestimated the first cell ring
by a median 1147x). It is withdrawn.

**Where it holds.** While the first station lies in the viscous sublayer, buffer
or log region. Once the station is outside the boundary layer the log law is not
valid there and the velocity has saturated at freestream; the formula then
over-predicts further still. :func:`amplification` reports which regime it used
rather than leaving the caller to guess.

**What it is for.** Deciding, before committing to an output format, whether a
surrogate can be used to warm start a solver at all -- and if not, what to
change. The fix the formula points at is never "more values"; ``u+`` grows
logarithmically, so refining a raster buys almost nothing. It is "put the first
station inside the first cell", which is a grading choice, not a budget.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "u_plus",
    "wall_units",
    "friction_velocity",
    "amplification",
    "geometric_stations",
    "uniform_stations",
    "stations_inside",
    "preflight",
    "invert_u_tau",
    "wall_law_profile",
    "wall_law_repair",
    "smooth_along_wall",
    "KAPPA",
    "B_LOG",
]

# von Karman constant and the log-law intercept, in their standard smooth-wall
# values. They are not tuned here and must not be tuned to fit a result.
KAPPA = 0.41
B_LOG = 5.0

_SUBLAYER_TOP = 5.0
_LOG_BOTTOM = 30.0


def u_plus(y_plus: np.ndarray | float) -> np.ndarray | float:
    """Law of the wall, blended across the buffer layer.

    ``u+ = y+`` below y+ = 5 and ``u+ = ln(y+)/kappa + B`` above y+ = 30, joined
    logarithmically in between so the function is continuous. The blend is a
    convenience, not a turbulence model: every number this module reports is a
    *ratio* of two values of ``u+``, and the buffer-layer detail cancels to the
    extent that both arguments sit in the same region.
    """
    y = np.asarray(y_plus, dtype=np.float64)
    if np.any(y < 0):
        raise ValueError("y+ must be non-negative")
    log_top = np.log(np.maximum(y, 1e-30)) / KAPPA + B_LOG
    log_at_30 = np.log(_LOG_BOTTOM) / KAPPA + B_LOG

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.log(np.maximum(y, 1e-30) / _SUBLAYER_TOP) / np.log(
            _LOG_BOTTOM / _SUBLAYER_TOP)
    blended = (1.0 - frac) * _SUBLAYER_TOP + frac * log_at_30

    out = np.where(y <= _SUBLAYER_TOP, y,
                   np.where(y >= _LOG_BOTTOM, log_top, blended))
    return float(out) if np.isscalar(y_plus) or out.ndim == 0 else out


def wall_units(y: np.ndarray | float, u_tau: float, nu: float) -> np.ndarray | float:
    """Convert a wall-normal distance to ``y+``."""
    if u_tau <= 0 or nu <= 0:
        raise ValueError("u_tau and nu must be positive")
    return np.asarray(y, dtype=np.float64) * u_tau / nu


def friction_velocity(wall_gradient: float, nu: float) -> float:
    """``u_tau`` from the wall velocity gradient, for kinematic (``rho = 1``) fields.

    ``tau_w / rho = nu * du/dy|_wall`` and ``u_tau = sqrt(tau_w / rho)``. Pass the
    converged solution's own mean wall gradient; nothing here needs a
    correlation.
    """
    if wall_gradient < 0 or nu <= 0:
        raise ValueError("wall_gradient must be non-negative and nu positive")
    return float(np.sqrt(nu * wall_gradient))


def geometric_stations(first: float, outer: float, n: int) -> np.ndarray:
    """Wall-normal stations of a geometrically graded wall-fitted grid."""
    if n < 1 or first <= 0 or outer <= first:
        raise ValueError("need n >= 1 and 0 < first < outer")
    return np.geomspace(first, outer, n)


def uniform_stations(height: float, n: int) -> np.ndarray:
    """Wall-normal stations of a uniform raster of ``n`` rows spanning ``height``.

    Cell-centred, so the first station is half a spacing off the wall -- the
    most favourable reading of a uniform grid, which is the one worth quoting.
    """
    if n < 1 or height <= 0:
        raise ValueError("need n >= 1 and height > 0")
    edge = height / n
    return (np.arange(n, dtype=np.float64) + 0.5) * edge


def stations_inside(stations: np.ndarray, cell_centre: float) -> int:
    """How many stations fall inside the mesh's first cell.

    This is the whole criterion in one integer. Zero means the representation
    has no sample of the near-wall state at all and must invent it.
    """
    return int(np.count_nonzero(np.asarray(stations, dtype=np.float64) <= cell_centre))


def amplification(
    *,
    first_station: float,
    cell_centre: float,
    u_tau: float,
    nu: float,
    delta: float | None = None,
    u_inf: float | None = None,
) -> dict:
    """Predicted overestimate of the first-cell wall gradient, and its regime.

    Returns a dict carrying ``factor`` (the multiplier on the true gradient),
    the two ``y+`` values it came from, and ``regime`` -- one of ``"resolved"``
    (the station is at or inside the first cell, so nothing is lost),
    ``"wall_law"`` (the estimate is quantitative) or ``"saturated"`` (the law of
    the wall has run past the freestream or past the layer edge, so the number is
    an upper bound rather than an estimate).

    Passing ``u_inf`` caps ``u+`` at ``u_inf / u_tau``, because no station can
    carry more than freestream velocity. That cap is what separates a
    quantitative prediction from a bound: the uniform-raster arms of this study
    sit at ``y+ ~ 1700``, where the unbounded log law returns a velocity above
    freestream and over-predicts the measured damage.
    """
    if cell_centre <= 0 or first_station <= 0:
        raise ValueError("distances must be positive")

    y_cell = float(wall_units(cell_centre, u_tau, nu))
    y_station = float(wall_units(first_station, u_tau, nu))

    if first_station <= cell_centre:
        return {"factor": 1.0, "y_plus_station": y_station, "y_plus_cell": y_cell,
                "regime": "resolved"}

    station_u = float(u_plus(y_station))
    regime = "wall_law"

    if u_inf is not None:
        ceiling = float(u_inf) / u_tau
        if station_u >= ceiling:
            station_u, regime = ceiling, "saturated"
    if delta is not None and first_station > delta:
        regime = "saturated"

    return {"factor": station_u / float(u_plus(y_cell)),
            "y_plus_station": y_station, "y_plus_cell": y_cell, "regime": regime}


def preflight(
    *,
    stations: np.ndarray,
    cell_centre: float,
    u_tau: float,
    nu: float,
    delta: float | None = None,
    u_inf: float | None = None,
) -> dict:
    """The whole check, for one representation against one mesh.

    ``stations`` is the representation's wall-normal sample positions --
    :func:`geometric_stations` for a graded wall-fitted grid,
    :func:`uniform_stations` for a raster, or the mesh's own wall distances for
    a mesh-native surrogate, which trivially passes.
    """
    st = np.sort(np.asarray(stations, dtype=np.float64))
    if st.size == 0:
        raise ValueError("need at least one station")
    amp = amplification(first_station=float(st[0]), cell_centre=cell_centre,
                        u_tau=u_tau, nu=nu, delta=delta, u_inf=u_inf)
    return {
        "n_stations": int(st.size),
        "first_station": float(st[0]),
        "cell_centre": float(cell_centre),
        "stations_inside_first_cell": stations_inside(st, cell_centre),
        "first_station_over_cell_centre": float(st[0] / cell_centre),
        **amp,
    }


def invert_u_tau(speed: np.ndarray | float, height: float, nu: float,
                 *, iterations: int = 60) -> np.ndarray:
    """Recover ``u_tau`` from a velocity known at one off-wall height.

    Solves ``speed = u_tau * u+(height * u_tau / nu)`` for ``u_tau``. This is the
    standard wall-function inversion; it is here because it is what makes the
    repair in :func:`wall_law_profile` possible using **only what the
    representation actually carries** -- a velocity at its own first station --
    and nothing from the converged solution.

    The right-hand side is monotone increasing in ``u_tau``, so bisection is
    unconditionally safe and needs no derivative or initial guess.
    """
    s = np.asarray(speed, dtype=np.float64)
    if height <= 0 or nu <= 0:
        raise ValueError("height and nu must be positive")

    lo = np.full(s.shape, 1e-12)
    hi = np.maximum(np.abs(s), 1e-12)          # u+ >= 1 for y+ >= 1, so u_tau <= speed
    hi = np.maximum(hi, np.sqrt(np.abs(s) * nu / height))   # covers the sublayer branch
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        predicted = mid * u_plus(np.maximum(height * mid / nu, 1e-30))
        too_small = predicted < np.abs(s)
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


def wall_law_profile(speed_at_station: np.ndarray, height: float,
                     distance: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the near-wall speed a representation could not store.

    Given the speed the representation holds at its first station ``height`` and
    the wall distance of each mesh cell, return ``(speed, u_tau)`` with the speed
    evaluated from the law of the wall at each cell's own distance.

    **This is the repair the mechanism implies.** Section 6 says a projection
    overestimates the first-cell gradient by ``u+(y1+)/u+(yc+)``, a factor that
    is known in closed form -- so it can be divided back out. Nothing here is
    learned or fitted, and nothing uses the converged answer: the only input is
    the value the representation already carries.

    It is not new physics. Inverting a wall function to get ``u_tau`` from an
    off-wall velocity is what every wall-modelled LES does. What is new is the
    use: repairing a surrogate's *output format* so that a seed which would
    otherwise cost the solve can be handed to the solver.
    """
    d = np.asarray(distance, dtype=np.float64)
    u_tau = invert_u_tau(speed_at_station, height, nu)
    y_plus = np.maximum(d * u_tau / nu, 1e-30)
    return u_tau * u_plus(y_plus), u_tau


def smooth_along_wall(arclength: np.ndarray, values: np.ndarray,
                      window: float) -> np.ndarray:
    """Smooth a per-cell quantity along the surface, over a window in arclength.

    ``u_tau`` varies smoothly along an attached surface; a *reconstruction* of it
    need not, because each surface station is inverted independently from its own
    -- possibly noisy -- station value. That is the difference the repair of
    :func:`wall_law_repair` leaves behind: it restores the wall gradient's
    magnitude and leaves it 11x rougher along the wall than the converged field,
    against 4.2x for a seed that works.

    Binning on arclength and box-filtering keeps this O(n) and free of any mesh
    topology, which matters because the caller has cell centres and nothing else.
    """
    s = np.asarray(arclength, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if window <= 0 or s.size == 0:
        return v

    span = float(s.max() - s.min())
    if span <= 0:
        return np.full_like(v, float(v.mean()))

    n_bins = max(int(np.ceil(span / (0.25 * window))), 3)
    idx = np.clip(((s - s.min()) / span * n_bins).astype(int), 0, n_bins - 1)
    total = np.bincount(idx, weights=v, minlength=n_bins)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)

    width = max(int(round(window / (span / n_bins))), 1)
    kernel = np.ones(width)
    smoothed = np.convolve(total, kernel, mode="same")
    weight = np.convolve(count, kernel, mode="same")
    # Bins the window never covers keep their own value rather than a zero.
    filled = np.where(weight > 0, smoothed / np.maximum(weight, 1e-30),
                      np.where(count > 0, total / np.maximum(count, 1e-30), v.mean()))
    return filled[idx]


def wall_law_repair(
    values: tuple,
    distance: np.ndarray,
    *,
    first_station: float,
    nu: float,
    kappa_damping: float = 26.0,
    arclength: np.ndarray | None = None,
    smooth_window: float = 0.0,
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Repair a projected seed below the representation's first station.

    Takes ``(u, v, p, nut)`` as a projection left them -- every cell nearer the
    wall than ``first_station`` holding that station's value -- and rebuilds the
    velocity magnitude and eddy viscosity from the law of the wall at each
    cell's own wall distance. Direction is preserved; only magnitude is
    rescaled, which keeps the repair free of any surface-tangent geometry.

    ``nut`` is rebuilt as the mixing-length estimate ``kappa * u_tau * d`` with
    van Driest damping, which is what Spalart-Allmaras relaxes to in the log
    layer and which correctly goes to zero at the wall.

    Pressure is untouched: it is constant across a boundary layer, and it is the
    one channel a coarse representation does not damage.
    """
    u, v, p, nut = (np.asarray(a, dtype=np.float64).copy() for a in values)
    d = np.asarray(distance, dtype=np.float64)
    below = d < first_station
    if not below.any():
        return (u, v, p, nut), {"mode": "wall_law_repair", "repaired_cells": 0,
                                "first_station": float(first_station)}

    speed = np.hypot(u[below], v[below])
    u_tau = invert_u_tau(speed, first_station, nu)
    if arclength is not None and smooth_window > 0:
        # Each station is inverted independently, so the reconstructed `u_tau`
        # inherits whatever noise the projection left in the station value. The
        # physical quantity varies smoothly along an attached surface, so this
        # smooths it before the profile is rebuilt from it.
        u_tau = smooth_along_wall(np.asarray(arclength)[below], u_tau,
                                  smooth_window)
    rebuilt = u_tau * u_plus(np.maximum(d[below] * u_tau / nu, 1e-30))
    scale = rebuilt / np.maximum(speed, 1e-30)
    u[below] *= scale
    v[below] *= scale

    y_plus = np.maximum(d[below] * u_tau / nu, 1e-30)
    damping = (1.0 - np.exp(-y_plus / kappa_damping)) ** 2
    nut[below] = np.maximum(KAPPA * u_tau * d[below] * damping, 0.0)

    return (u, v, p, np.maximum(nut, 0.0)), {
        "mode": "wall_law_repair",
        "repaired_cells": int(below.sum()),
        "repaired_fraction": float(below.mean()),
        "first_station": float(first_station),
        "u_tau_median": float(np.median(u_tau)),
        "speed_scale_median": float(np.median(scale)),
        "smooth_window": float(smooth_window),
    }
