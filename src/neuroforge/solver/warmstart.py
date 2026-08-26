"""Ways of turning a NeuroForge prediction into an OpenFOAM initial field.

Two strategies, shared by every mesh in this package:

``plain``
    Interpolate the prediction onto the mesh cell centres and use it as-is.
    This is what :func:`neuroforge.solver.ogrid.seed_warm_start` does, and at
    AirfRANS Reynolds it **does not work** -- ``scripts/ogrid_resolution_probe.py``
    measured that even the *exact* solution, round-tripped through the 128^2
    grid, saves no iterations at all.

``hybrid``
    Keep the prediction where it is good and rebuild the boundary layer where it
    is not. The measured error profile is unambiguous about where the line falls
    (``results/ogrid_resolution_bands.json``, velocity error of a round-tripped
    exact solution, binned by distance from the wall):

    ===================  ==========
    distance from wall   rel. error
    ===================  ==========
    0 - 1e-4                   440%
    1e-4 - 1e-3                352%
    1e-3 - 1e-2                 44%
    1e-2 - 0.019                15%
    0.019 - 0.1                  1%
    > 0.1                       ~0%
    ===================  ==========

    The outer field survives intact; only the boundary layer is destroyed, and
    handing SIMPLE a near-wall state that is 3-4x wrong is worse than handing it
    a uniform one. So outside the layer the prediction is used unchanged, and
    inside it the velocity is replaced by a profile that respects the wall.

Why the reconstruction is this simple
-------------------------------------
At Re 3e6 the layer is ~0.019 chord and the surrogate's cell is 0.0236 chord, so
the prediction is very nearly *constant* across the whole layer -- that is the
same fact that makes the plain seed fail. Its value at a near-wall cell is
therefore already the boundary-layer **edge** value, and no wall-normal search or
surface normal is needed: scale it by a profile in ``d / delta`` and the result
matches the prediction at the edge and satisfies no-slip at the wall.

Pressure is passed through untouched, because pressure really is constant across
a boundary layer -- it is the one channel the surrogate's resolution does not
harm. The eddy viscosity is ramped to zero at the wall, as Spalart-Allmaras
requires.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import Domain, FlowField

__all__ = [
    "wall_distance",
    "surface_coords",
    "clustered_seed",
    "bl_thickness",
    "sample_on_mesh",
    "plain_seed",
    "hybrid_seed",
]

# Turbulent boundary-layer profile exponent (the classic 1/7 power law).
PROFILE_EXPONENT = 1.0 / 7.0


def surface_coords(centres: np.ndarray, surface: np.ndarray) -> tuple:
    """Body-fitted coordinates of each cell: (arclength along the wall, distance).

    The mapping a wall-fitted surrogate would predict on. Arclength is taken from
    the nearest surface point, which is exact on the wall and degrades gracefully
    away from it -- and away from it the field is smooth enough not to care.
    """
    from scipy.spatial import cKDTree

    surf = np.asarray(surface)[:, :2]
    seg = np.linalg.norm(np.diff(surf, axis=0), axis=1)
    s_of = np.concatenate([[0.0], np.cumsum(seg)])
    d, idx = cKDTree(surf).query(np.asarray(centres)[:, :2])
    return s_of[idx], d, float(s_of[-1])


def wall_distance(centres: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Distance from each cell centre to the body surface polyline."""
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(surface)[:, :2]).query(np.asarray(centres)[:, :2])[0]


def bl_thickness(reynolds: float, chord: float = 1.0) -> float:
    """Turbulent boundary-layer thickness at the trailing edge, flat-plate scaling.

    ``delta = 0.37 c / Re^(1/5)`` -- the standard correlation. At Re 3e6 this is
    0.019 chord, which is the number the resolution probe compared against the
    surrogate's 0.0236-chord cell.
    """
    re = max(float(reynolds), 1.0)
    return 0.37 * float(chord) / re**0.2


def sample_on_mesh(
    field: FlowField,
    domain: Domain,
    centres: np.ndarray,
    *,
    u_inf: float,
    v_inf: float,
    nut_freestream: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a Cartesian prediction onto mesh cell centres.

    Returns ``(u, v, p, nut, inside)`` where ``inside`` flags the cells the
    prediction's crop actually covers; everything outside falls back to
    freestream. On a 20-chord mesh against a 3-chord crop that fallback is a
    large share of the domain, and it is reported rather than hidden.
    """
    from scipy.interpolate import RegularGridInterpolator

    c = np.asarray(centres)
    cx, cy = c[:, 0], c[:, 1]
    x, y = domain.axes()
    inside = (cx >= x[0]) & (cx <= x[-1]) & (cy >= y[0]) & (cy <= y[-1])
    pts = np.stack([np.clip(cy, y[0], y[-1]), np.clip(cx, x[0], x[-1])], axis=1)

    def sample(arr, fill):
        interp = RegularGridInterpolator(
            (y, x), np.asarray(arr, dtype=np.float64), bounds_error=False, fill_value=None
        )
        return np.where(inside, interp(pts), fill)

    nut_src = field.nut if field.nut is not None else np.full(domain.shape, nut_freestream)
    return (
        sample(field.u, u_inf),
        sample(field.v, v_inf),
        sample(field.p, 0.0),
        np.maximum(sample(nut_src, nut_freestream), 0.0),
        inside,
    )


def plain_seed(
    field: FlowField,
    domain: Domain,
    centres: np.ndarray,
    *,
    u_inf: float,
    v_inf: float,
    nut_freestream: float,
) -> tuple[tuple[np.ndarray, ...], dict]:
    """The prediction, interpolated onto the mesh and used as-is."""
    u, v, p, nut, inside = sample_on_mesh(
        field, domain, centres, u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_freestream
    )
    report = {"mode": "plain", "cells": int(len(u)),
              "covered_fraction": float(inside.mean())}
    return (u, v, p, nut), report


def hybrid_seed(
    field: FlowField,
    domain: Domain,
    centres: np.ndarray,
    surface: np.ndarray,
    *,
    reynolds: float,
    u_inf: float,
    v_inf: float,
    nut_freestream: float,
    delta: float | None = None,
    blend_to: float = 2.0,
) -> tuple[tuple[np.ndarray, ...], dict]:
    """The prediction outside the boundary layer, a wall profile inside it.

    Parameters
    ----------
    surface : numpy.ndarray
        The body polyline, used only to measure distance from the wall.
    delta : float, optional
        Boundary-layer thickness. Defaults to :func:`bl_thickness` at
        ``reynolds``.
    blend_to : float
        Where the prediction takes over completely, in multiples of ``delta``.
        Between ``delta`` and ``blend_to * delta`` the two are ramped smoothly so
        the seed has no kink for the solver to iron out.

    Returns
    -------
    (u, v, p, nut), report
        Mesh-order arrays, plus what was touched -- ``profiled_fraction`` is the
        share of cells whose velocity was rebuilt rather than taken from the
        prediction.
    """
    u, v, p, nut, inside = sample_on_mesh(
        field, domain, centres, u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_freestream
    )
    d = wall_distance(centres, surface)
    dl = float(delta if delta is not None else bl_thickness(reynolds))
    if dl <= 0:
        raise ValueError(f"boundary-layer thickness must be positive (got {dl})")

    t = np.clip(d / dl, 0.0, 1.0)
    # The prediction is essentially constant across the layer, so its value at a
    # near-wall cell already is the edge value; scaling it by the profile matches
    # the prediction at d = delta and gives no-slip at d = 0.
    profile = t**PROFILE_EXPONENT

    # Ramp between the rebuilt layer and the untouched prediction so the seed is
    # continuous; smoothstep rather than linear to avoid a slope discontinuity.
    s = np.clip((d - dl) / (max(blend_to, 1.0 + 1e-9) * dl - dl), 0.0, 1.0)
    w = s * s * (3.0 - 2.0 * s)          # 0 at the edge of the layer, 1 beyond

    u_out = w * u + (1.0 - w) * (u * profile)
    v_out = w * v + (1.0 - w) * (v * profile)
    # Pressure is constant across a boundary layer, so the prediction is valid
    # all the way to the wall -- the one channel resolution does not damage.
    p_out = p
    # Spalart-Allmaras wants nut -> 0 at the wall.
    nut_out = np.maximum(nut * np.minimum(t, 1.0), 0.0)

    report = {
        "mode": "hybrid",
        "cells": int(len(u)),
        "covered_fraction": float(inside.mean()),
        "delta": dl,
        "blend_to": float(blend_to),
        "profiled_fraction": float((d < dl).mean()),
        "blended_fraction": float(((d >= dl) & (d < blend_to * dl)).mean()),
    }
    return (u_out, v_out, p_out, nut_out), report


def clustered_seed(
    values: tuple,
    centres: np.ndarray,
    surface: np.ndarray,
    *,
    n_s: int = 256,
    n_n: int = 64,
    first: float = 2.5e-4,
    n_max: float = 1.0,
    u_inf: float,
    v_inf: float,
    nut_freestream: float,
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Round-trip a mesh solution through a **wall-fitted** surrogate grid.

    The counterpart to :func:`plain_seed`, and the reason it exists: every
    failure measured so far assumed the surrogate predicts on a *uniform
    Cartesian* grid. More points on that grid do not help
    (``scripts/resolution_ladder.py``), because a uniform grid must resolve the
    smallest scale everywhere and the near-wall scale collapses like
    ``nu / u_tau``. This spends the **same number of output values** on a grid
    that is clustered where the gradient is.

    At the default 256 x 64 that is 16,384 values -- exactly a 128^2 grid -- with
    the first station at ``first`` (2.5e-4 chord) instead of 0.0118, roughly 94x
    finer at the wall for identical model capacity.

    The projection is deliberately matched to the Cartesian arm so the comparison
    is like for like: nearest-neighbour from the mesh onto the surrogate grid
    (what ``to_grid`` does), then linear interpolation back (what
    :func:`sample_on_mesh` does). Cells beyond ``n_max`` fall back to freestream,
    just as the Cartesian arm does outside its crop.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.spatial import cKDTree

    u, v, p, nut = (np.asarray(a, dtype=np.float64) for a in values)
    s, d, s_max = surface_coords(centres, surface)

    s_grid = np.linspace(0.0, s_max, n_s)
    n_grid = np.geomspace(first, n_max, n_n)

    # Mesh -> surrogate grid, nearest neighbour in normalised (s, n).
    def norm(ss, dd):
        return np.stack([ss / max(s_max, 1e-12),
                         np.log10(np.clip(dd, first, n_max)) / np.log10(n_max / first)], axis=1)

    tree = cKDTree(norm(s, d))
    SS, NN = np.meshgrid(s_grid, n_grid, indexing="xy")
    _, idx = tree.query(norm(SS.ravel(), NN.ravel()))
    shp = (n_n, n_s)

    inside = d <= n_max
    query = np.stack([np.clip(d, first, n_max), np.clip(s, 0.0, s_max)], axis=1)

    def back(field_vals, fill):
        grid = field_vals[idx].reshape(shp)
        interp = RegularGridInterpolator((n_grid, s_grid), grid,
                                         bounds_error=False, fill_value=None)
        return np.where(inside, interp(query), fill)

    out = (back(u, u_inf), back(v, v_inf), back(p, 0.0),
           np.maximum(back(nut, nut_freestream), 0.0))
    report = {
        "mode": "clustered", "n_s": n_s, "n_n": n_n, "points": n_s * n_n,
        "first": first, "n_max": n_max,
        "covered_fraction": float(inside.mean()),
    }
    return out, report
