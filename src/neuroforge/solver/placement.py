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

There is no fitted parameter in it. Against six converged cases at Re = 3e6 the
closed form predicts 23.7x for a 256x64 wall-fitted grid whose first station
sits at y+ = 36, where the measured overestimate is 21.0x -- accurate to 13%
with 2% scatter (``docs/protocols/placement_prediction.md``).

**Where it holds.** While the first station lies in the viscous sublayer, buffer
or log region. Once the station is outside the boundary layer the log law is not
valid there and the velocity has saturated at freestream; the formula then
over-predicts and should be read as an upper bound. :func:`amplification`
reports which regime it used rather than leaving the caller to guess.

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
