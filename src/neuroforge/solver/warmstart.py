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
    "masked_seed",
    "wake_seed",
    "sequence_seed",
    "FIELDS",
]

# Order of the four-tuple every seed in this module produces and consumes.
FIELDS = ("u", "v", "p", "nut")

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


def wall_distance(centres: np.ndarray, surface: np.ndarray, *,
                  chunk: int = 4096) -> np.ndarray:
    """Distance from each cell centre to the body surface polyline.

    To the nearest **segment**, not the nearest vertex. The difference is not
    cosmetic on a body-fitted mesh: the surface polyline here carries 200 points,
    so its vertices are ~0.01 chord apart, while the first cell ring sits 4e-6
    chord off the wall. Nearest-vertex then reports 1.4e-3 for a cell whose true
    distance is 3.8e-6 -- a **368x** overestimate, and worst exactly in the
    boundary layer that every seed in this module is about.

    Costs one pass over ``n_surface`` segments per chunk of query points, which
    is nothing against a solve: a couple of hundred segments against tens of
    thousands of cells.
    """
    surf = np.asarray(surface, dtype=np.float64)[:, :2]
    pts = np.asarray(centres, dtype=np.float64)[:, :2]
    if len(surf) < 2:
        from scipy.spatial import cKDTree

        return cKDTree(surf).query(pts)[0]

    a, b = surf[:-1], surf[1:]
    ab = b - a
    length2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30)

    out = np.empty(len(pts), dtype=np.float64)
    for lo in range(0, len(pts), chunk):
        block = pts[lo: lo + chunk]
        rel = block[:, None, :] - a[None]                       # (k, M, 2)
        t = np.clip(np.einsum("kmj,mj->km", rel, ab) / length2, 0.0, 1.0)
        closest = rel - t[..., None] * ab[None]
        out[lo: lo + chunk] = np.sqrt(np.einsum("kmj,kmj->km", closest, closest)).min(axis=1)
    return out


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


def masked_seed(
    values: tuple,
    centres: np.ndarray,
    surface: np.ndarray,
    *,
    fields: tuple[str, ...] = FIELDS,
    free_within: float = 0.0,
    ramp: float = 3.0,
    background: tuple | None = None,
    u_inf: float,
    v_inf: float,
    nut_freestream: float,
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Keep part of a seed and hand the rest back to the solver.

    A warm start is usually all-or-nothing: every field, everywhere. Measured on
    the C-grid at Re 3e6, that is the wrong granularity, because the quantities
    in a cold solve do not converge at the same rate. Iterations for each to
    settle within 1% of its converged value, averaged over four cases:

    ======================  =========
    quantity                cold
    ======================  =========
    viscous drag ``Cd_v``     ~700
    lift ``Cl``               ~950
    pressure drag ``Cd_p``   ~1850
    ======================  =========

    So the slow part of a cold start is the **pressure** field, and the fast part
    is the near-wall velocity gradient -- which is also the part a surrogate
    reconstructs worst, and the part that dominates total drag (``Cd_v`` is 60%
    to 84% of ``Cd`` here). Seeding everything therefore trades a large gain on
    the pressure-dominated quantities against a loss on the one the solver was
    going to get right by itself.

    This splits the seed two ways so that trade can be measured and then chosen:

    ``fields``
        Which of ``u, v, p, nut`` to take from the prediction. The rest start at
        freestream, exactly as a cold start does.
    ``free_within``
        Wall distance inside which the *velocity* reverts to freestream, ramped
        smoothly to the prediction by ``ramp * free_within`` so the seed has no
        jump in it. Pressure is never masked -- it is the field with no wall
        boundary layer to get wrong, and the slow one.

    ``background``
        What the masked-out parts fall back to. Defaults to uniform freestream --
        a cold start. Pass another seed and the two are composed instead, which
        is how the measured split turns into a recipe: at Re 3e6 the wall-fitted
        surrogate wins on friction drag (+37%) and lift (+87%) and *loses* on
        pressure drag (-150%), while potential flow is free, untrained, and good
        at exactly the global pressure field the surrogate ruins. Passing
        potential flow as ``values`` and the surrogate as ``background`` with
        ``free_within = delta`` gives classical pressure and outer flow with a
        learned boundary layer inside it.

    ``free_within=0`` and all four fields is the identity, which is what makes
    this usable as the control arm of its own ablation.
    """
    u, v, p, nut = (np.asarray(a, dtype=np.float64).copy() for a in values)
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown seed fields: {sorted(unknown)}; expected {FIELDS}")

    if background is None:
        free = {"u": np.float64(u_inf), "v": np.float64(v_inf),
                "p": np.float64(0.0), "nut": np.float64(nut_freestream)}
    else:
        free = dict(zip(FIELDS, (np.asarray(a, dtype=np.float64) for a in background)))
    out = {"u": u, "v": v, "p": p, "nut": nut}
    for name in FIELDS:
        if name not in fields:
            out[name] = np.broadcast_to(free[name], out[name].shape).astype(np.float64).copy()

    blended = 0.0
    if free_within > 0:
        d = wall_distance(centres, surface)
        # 0 at the wall, 1 beyond ramp * free_within, smooth in between.
        span = max(ramp * free_within - free_within, 1e-30)
        w = np.clip((d - free_within) / span, 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)          # smoothstep: zero slope at both ends
        for name in ("u", "v", "nut"):
            out[name] = w * out[name] + (1.0 - w) * free[name]
        blended = float((w < 1.0).mean())

    out["nut"] = np.maximum(out["nut"], 0.0)
    report = {
        "mode": "masked",
        "fields": list(fields),
        "free_within": free_within,
        "ramp": ramp,
        "blended_fraction": blended,
        "background": "freestream" if background is None else "seed",
    }
    return tuple(out[name] for name in FIELDS), report


def wake_seed(
    values: tuple,
    centres: np.ndarray,
    *,
    x_start: float = 1.0,
    ramp: float = 0.5,
    u_inf: float,
    v_inf: float,
    nut_freestream: float,
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Hand over the **downstream** region and nothing else.

    The counterpart to :func:`masked_seed`'s boundary-layer handover, and the
    reason it exists is a competing result rather than a hypothesis of ours. The
    largest speed-up in this literature -- 26.3x iterations and 16.4x wall clock,
    `arXiv:2501.14699 <https://arxiv.org/abs/2501.14699>`_ -- comes from
    initialising the far wake, which a cold RANS solve is slowest to develop.
    Every seed in this package deliberately does the opposite: the surrogate's
    training ``sdf`` distribution is centred on 0.23 chords, so the seed is cut
    off near the body and the wake is handed back to the solver.

    Whether that costs anything *here* is a question about this configuration,
    not about their method, and it is answerable with an oracle: seed the
    converged field downstream of ``x_start`` and freestream everywhere else. If
    the saving is small on a C-grid at Re 3e6, the wake is not where this
    solver's time goes and the comparison is between different regimes rather
    than between better and worse methods. If it is large, the near-body and wake
    seeds are complementary and worth composing.

    The transition is a smoothstep over ``ramp`` chords so the seed carries no
    jump, matching :func:`masked_seed`. All four fields are handed over together,
    because seeding a subset inconsistently is what makes ``nut``-only fail.
    """
    u, v, p, nut = (np.asarray(a, dtype=np.float64).copy() for a in values)
    x = np.asarray(centres, dtype=np.float64)[:, 0]

    # 0 upstream of x_start, 1 beyond x_start + ramp, smooth in between.
    w = np.clip((x - float(x_start)) / max(float(ramp), 1e-30), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)

    free = {"u": float(u_inf), "v": float(v_inf), "p": 0.0,
            "nut": float(nut_freestream)}
    out = {"u": u, "v": v, "p": p, "nut": nut}
    for name, arr in out.items():
        out[name] = w * arr + (1.0 - w) * free[name]
    out["nut"] = np.maximum(out["nut"], 0.0)

    report = {
        "mode": "wake",
        "x_start": float(x_start),
        "ramp": float(ramp),
        "seeded_fraction": float((w > 0).mean()),
        "fully_seeded_fraction": float((w >= 1.0).mean()),
    }
    return tuple(out[name] for name in FIELDS), report


def sequence_seed(
    values: tuple,
    coarse_centres: np.ndarray,
    fine_centres: np.ndarray,
    surface: np.ndarray,
    *,
    u_inf: float,
    v_inf: float,
    nut_freestream: float,
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Map a solution from a coarse mesh onto a fine one -- **grid sequencing**.

    This is the classical warm start, and the one the machine-learning
    initialisation literature does not measure itself against: solve on a
    coarsened mesh, map the result up, continue on the fine mesh. OpenFOAM ships
    it as ``mapFields``, and every production aerodynamics workflow has some
    form of it.

    It is in this package for a reason beyond fairness. A coarsened body-fitted
    mesh is *still body-fitted*: halving the wall-normal count doubles the first
    cell but leaves its stations clustered at the wall, so the representation
    keeps a sample point inside -- or within a factor of two of -- the fine
    mesh's first cell. The placement criterion therefore **predicts that grid
    sequencing helps**, on a method it was not derived from and which involves
    no network at all. That is the criterion's out-of-sample test.

    The map follows ``mapFields`` rather than a bare cell-to-cell interpolation:
    the wall's own boundary values are carried into the interpolation as data.
    Without them a fine cell centre nearer the wall than any coarse centre sits
    outside the convex hull and has to be extrapolated, which is precisely where
    the wall gradient lives. With them, ``u`` and ``v`` interpolate between
    no-slip at the surface and the first coarse centre, which is what makes the
    near-wall map faithful. ``nut`` goes to zero at the wall as Spalart-Allmaras
    requires; ``p`` is zero-gradient there, so the surface carries its nearest
    coarse value.

    Parameters
    ----------
    values:
        ``(u, v, p, nut)`` on ``coarse_centres``.
    coarse_centres, fine_centres:
        Cell centres of the two meshes, ``(N, 2)`` or ``(N, 3)``.
    surface:
        The wall polyline, used both as no-slip data and to anchor ``p``.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import cKDTree

    u, v, p, nut = (np.asarray(a, dtype=np.float64).ravel() for a in values)
    src = np.asarray(coarse_centres, dtype=np.float64)[:, :2]
    dst = np.asarray(fine_centres, dtype=np.float64)[:, :2]
    wall = np.unique(np.asarray(surface, dtype=np.float64)[:, :2], axis=0)

    # Pressure is zero-gradient at the wall, so the surface inherits the nearest
    # coarse value; velocity and nuTilda are zero there by the wall condition.
    p_wall = p[cKDTree(src).query(wall)[1]]

    points = np.vstack([src, wall])
    columns = {
        "u": np.concatenate([u, np.zeros(len(wall))]),
        "v": np.concatenate([v, np.zeros(len(wall))]),
        "p": np.concatenate([p, p_wall]),
        "nut": np.concatenate([nut, np.zeros(len(wall))]),
    }
    fill = {"u": float(u_inf), "v": float(v_inf), "p": 0.0,
            "nut": float(nut_freestream)}

    out = {}
    outside = np.zeros(len(dst), dtype=bool)
    for name, col in columns.items():
        linear = LinearNDInterpolator(points, col)
        got = linear(dst)
        gap = ~np.isfinite(got)
        outside |= gap
        if gap.any():
            # Beyond the coarse mesh's hull -- the far field, where freestream is
            # the right answer anyway. Nearest keeps it continuous.
            got[gap] = NearestNDInterpolator(points, col)(dst[gap])
        out[name] = got
    out["nut"] = np.maximum(out["nut"], 0.0)

    report = {
        "mode": "sequence",
        "coarse_cells": int(len(src)),
        "fine_cells": int(len(dst)),
        "coarsening_ratio": float(len(dst) / max(len(src), 1)),
        "wall_points": int(len(wall)),
        "extrapolated_fraction": float(outside.mean()),
        "fill": fill,
    }
    return tuple(out[name] for name in FIELDS), report
