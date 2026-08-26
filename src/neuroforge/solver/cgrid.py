"""Body-fitted C-grid meshes for the OpenFOAM backend.

The O-grid in :mod:`neuroforge.solver.ogrid` already reaches AirfRANS Reynolds,
but its radial lines fan out behind the body, so the wake is resolved only as
well as the far field -- poor for drag, and poor for any comparison against
AirfRANS force labels. A C-grid wraps the front of the section and carries a
*dense, aligned* block straight downstream, which is the standard topology for
2-D airfoil RANS. It also takes a **sharp** trailing edge natively, so no blunt
base is needed.

The wake cut without stitching
------------------------------
A C-grid's inner boundary runs from the outlet upstream along the lower side of
the wake cut, around the airfoil, and back downstream along the upper side. The
two cut runs are geometrically coincident, which is normally resolved with
``stitchMesh`` or ``mergeOrSplitBaffles``.

Neither is needed here. ``blockMesh`` identifies a block face by its *vertex
labels*, so if the lower-cut and upper-cut nodes at ``j = 0`` are emitted as the
same vertices, the faces coincide by construction and blockMesh joins the blocks
into internal faces on its own. Only ``j = 0`` is shared; from the first radial
station outwards the two sheets separate, which is exactly the slit a C-grid
needs. :func:`vertex_ids` builds that map.

Orientation
-----------
The inner boundary is traversed with the fluid consistently on the **left**:
downstream-to-trailing-edge along the lower cut (travelling -x, fluid below),
around the section from the lower surface via the leading edge to the upper
surface, then trailing-edge-to-downstream along the upper cut (travelling +x,
fluid above). Increasing ``j`` is therefore a +90-degree rotation of the tangent
everywhere, including across the cut, and ``(tangent x outward) = +z`` falls out
automatically -- which is what ``blockMesh`` needs for positive cell volumes.

Blocks
------
One block per inner-boundary segment per radial layer, for the reason recorded
in :mod:`neuroforge.solver.ogrid`: ``blockMesh`` builds a straight-sided
topology hex per block and rejects a concave one, and with wide blocks that check
fails at high-curvature features under every corner placement. One block per
segment makes the topology quad *be* the cell, so the convexity check runs in
numpy before a dictionary is ever written.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field as _dc_field

import numpy as np

from neuroforge.core.types import DTYPE, Domain, FlowCase, FlowField
from neuroforge.geometry.airfoil import naca_airfoil

from . import openfoam as of
from .ogrid import expansion_ratio

__all__ = [
    "CGridSpec",
    "CGridResult",
    "airfoil_open_curve",
    "wake_line",
    "inner_curve",
    "offset_open",
    "outer_curve",
    "vertex_ids",
    "block_mesh_dict",
    "write_cgrid_case",
    "solve_cgrid",
]


@dataclass(frozen=True)
class CGridSpec:
    """Knobs for the C-grid. Defaults target y+ ~ 1 at Re 3e6, chord 1."""

    n_surface: int = 200        # points around the section, trailing edge to trailing edge
    n_wake: int = 60            # points along each side of the wake cut
    n_inner: int = 60           # radial cells, wall -> offset curve
    n_outer: int = 40           # radial cells, offset curve -> far field
    first_cell: float = 1.0e-5  # wall-normal height of the first cell [chord]
    first_wake: float = 2.0e-3  # streamwise size of the first wake cell [chord]
    offset: float = 0.08        # thickness of the wall-normal inner layer
    n_smooth: int = 40          # Laplacian passes on the offset curve
    smooth_pad: int = 10        # wake nodes each side of the TE to include
    far_radius: float = 20.0    # radius of the upstream semicircle [chord]
    wake_length: float = 20.0   # outlet distance downstream of `centre` [chord]
    centre: tuple[float, float] = (0.25, 0.0)
    thickness: float = 0.1      # span in z (2-D: `empty` front/back)

    @property
    def n_i(self) -> int:
        """Inner-boundary nodes: lower cut + section + upper cut."""
        return 2 * self.n_wake + self.n_surface - 2

    @property
    def n_cells(self) -> int:
        return (self.n_i - 1) * (self.n_inner + self.n_outer)


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #


def airfoil_open_curve(code: str, n: int) -> np.ndarray:
    """Section from the trailing edge, along the lower surface, round the leading
    edge and back along the upper surface to the trailing edge.

    A **sharp** trailing edge (``closed=True``) -- the whole point of a C-grid is
    that the cut springs from a cusp, so no blunt base is wanted here. The first
    and last points are the same trailing-edge node.
    """
    loop = np.asarray(
        naca_airfoil(code, n_points=max(n, 8), closed=True).surface_points,
        dtype=np.float64,
    )
    # The generator leaves the trailing edge last; roll it to the front and close
    # the run, then reverse so the traversal is lower surface first.
    seq = np.vstack([np.roll(loop, 1, axis=0), loop[-1:]])[::-1]
    return _resample_open(seq, n)


def _resample_open(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline to ``n`` points, preserving its clustering."""
    m = len(pts)
    src = np.arange(m, dtype=np.float64) / (m - 1)
    tgt = np.arange(n, dtype=np.float64) / (n - 1)
    return np.stack(
        [np.interp(tgt, src, pts[:, 0]), np.interp(tgt, src, pts[:, 1])], axis=1
    )


def wake_line(te: np.ndarray, x_out: float, n: int, first: float) -> np.ndarray:
    """Cut nodes from the trailing edge downstream to the outlet.

    Geometrically stretched from ``first`` at the trailing edge, so the shear
    layer is resolved where it is thin and the mesh coarsens toward the outlet.
    """
    length = float(x_out - te[0])
    if length <= 0:
        raise ValueError(f"outlet at x={x_out} is not downstream of the trailing edge")
    ratio = expansion_ratio(length, first, n - 1)
    growth = ratio ** (1.0 / max(n - 2, 1)) if n > 2 else 1.0
    steps = first * growth ** np.arange(n - 1)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    s = s / s[-1] * length
    return np.stack([te[0] + s, np.full(n, te[1])], axis=1)


def inner_curve(code: str, spec: CGridSpec) -> tuple[np.ndarray, int, int]:
    """The C-grid inner boundary, plus the counts needed to index it.

    Returns ``(curve, n_wake, n_surface)`` where ``curve`` has
    ``spec.n_i`` points: the lower cut reversed (outlet -> trailing edge), the
    section (trailing edge -> trailing edge), then the upper cut (trailing edge
    -> outlet). The trailing edge appears at index ``n_wake - 1`` and again at
    ``n_wake + n_surface - 2``; those two nodes are the same point and, at
    ``j = 0`` only, the same vertex.
    """
    surf = airfoil_open_curve(code, spec.n_surface)
    te = surf[0]
    x_out = spec.centre[0] + spec.wake_length
    wake = wake_line(te, x_out, spec.n_wake, spec.first_wake)

    lower = wake[::-1]              # outlet -> trailing edge
    curve = np.vstack([lower, surf[1:], wake[1:]])
    assert len(curve) == spec.n_i, (len(curve), spec.n_i)
    return curve, spec.n_wake, spec.n_surface


def _outward_open(curve: np.ndarray) -> np.ndarray:
    """Unit normals of an open curve, +90 degrees from the tangent.

    The fluid is on the left of the traversal everywhere (see the module
    docstring), so this points into the domain on both sides of the cut and
    around the section, and gives blockMesh the positive orientation it wants.
    """
    t = np.empty_like(curve)
    t[1:-1] = curve[2:] - curve[:-2]
    t[0] = curve[1] - curve[0]
    t[-1] = curve[-1] - curve[-2]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def offset_open(
    curve: np.ndarray,
    distance: float,
    n_smooth: int = 10,
    smooth_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Normal offset of the inner boundary, Laplacian-smoothed over the section.

    Smoothing is confined to ``smooth_range`` -- in practice the section -- and
    that restriction is load-bearing. The wake cut is geometrically stretched
    (first cell 2e-3, last of order a chord), and Laplacian smoothing on a curve
    with spacing that uneven drags points toward their distant neighbours: run
    over the whole curve it pushed the offset from the requested 0.08 out to
    0.46 and produced twelve non-convex cells. The wake is a straight line
    anyway, so its normal offset is already exact and needs no smoothing.

    Endpoints are pinned regardless: they sit on the outlet plane and must stay
    there or the outlet stops being a straight vertical boundary.
    """
    out = curve + distance * _outward_open(curve)
    lo, hi = smooth_range if smooth_range is not None else (1, len(curve) - 1)
    lo, hi = max(1, lo), min(len(curve) - 1, hi)
    for _ in range(max(0, n_smooth)):
        lap = np.zeros_like(out)
        lap[lo:hi] = out[lo - 1 : hi - 1] + out[lo + 1 : hi + 1] - 2.0 * out[lo:hi]
        out = out + 0.25 * lap
    return out


def outer_curve(spec: CGridSpec, n_wake: int, n_surface: int) -> np.ndarray:
    """Far boundary: outlet -> bottom -> upstream semicircle -> top -> outlet.

    Point-for-point matched to the inner boundary's three runs, so each radial
    line joins a cut node to a far-field node directly above or below it, and a
    section node to the semicircle.
    """
    cx, cy = spec.centre
    R = spec.far_radius
    x_out = cx + spec.wake_length

    bottom = np.stack([np.linspace(x_out, cx, n_wake), np.full(n_wake, cy - R)], axis=1)
    top = np.stack([np.linspace(cx, x_out, n_wake), np.full(n_wake, cy + R)], axis=1)
    # Semicircle from straight down, round the front, to straight up.
    ang = np.linspace(-np.pi / 2, -3 * np.pi / 2, n_surface)
    arc = np.stack([cx + R * np.cos(ang), cy + R * np.sin(ang)], axis=1)

    curve = np.vstack([bottom, arc[1:], top[1:]])
    assert len(curve) == spec.n_i, (len(curve), spec.n_i)
    return curve


# --------------------------------------------------------------------------- #
# Vertex numbering: the wake cut
# --------------------------------------------------------------------------- #


def vertex_ids(spec: CGridSpec) -> tuple[np.ndarray, int]:
    """``ids[ring, i]`` -> vertex label for one z-level, and the label count.

    Ring 0 (the inner boundary) shares a label between the lower-cut node and its
    mirror on the upper cut, which is what makes the two sheets of the wake cut
    the *same* faces to blockMesh and removes any need for ``stitchMesh``. Rings
    1 and 2 are all distinct -- the sheets have separated by then.
    """
    ni, nw, ns = spec.n_i, spec.n_wake, spec.n_surface
    ids = np.full((3, ni), -1, dtype=np.int64)

    nxt = 0
    for i in range(nw + ns - 1):        # lower cut + section, up to the second TE
        ids[0, i] = nxt
        nxt += 1
    for m in range(nw):                 # upper cut mirrors the lower cut
        ids[0, nw + ns - 2 + m] = ids[0, nw - 1 - m]
    for ring in (1, 2):
        for i in range(ni):
            ids[ring, i] = nxt
            nxt += 1
    assert (ids >= 0).all()
    return ids, nxt


def _cell_quads_convex(rings: list[np.ndarray]) -> tuple[int, float]:
    """Count non-convex cell quads on an *open* ring stack, worst cross product."""
    bad, worst = 0, float("inf")
    n = len(rings[0])
    for r in range(len(rings) - 1):
        a, b = rings[r], rings[r + 1]
        for i in range(n - 1):
            q = np.array([a[i], a[i + 1], b[i + 1], b[i]])
            e = np.roll(q, -1, axis=0) - q
            e2 = np.roll(e, -1, axis=0)
            cr = e[:, 0] * e2[:, 1] - e[:, 1] * e2[:, 0]
            worst = min(worst, float(cr.min()))
            if (cr <= 0).any():
                bad += 1
    return bad, worst


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #


def block_mesh_dict(spec: CGridSpec, code: str) -> str:
    """``blockMeshDict`` for the C-grid."""
    inner, nw, ns = inner_curve(code, spec)
    # Smooth only across the section; see offset_open on why the wake must not be.
    # Span the trailing edge by `smooth_pad` wake nodes: the normal rotates ~90
    # degrees across the sharp cusp and folds the offset there, and smoothing that
    # only reaches the section cannot relax it. Those first wake cells are still
    # fine (2e-3 and up), so including them is safe; reaching further out drags
    # the offset toward the coarse far wake and inflates it.
    pad = int(spec.smooth_pad)
    off = offset_open(inner, spec.offset, spec.n_smooth,
                      smooth_range=(nw - 1 - pad, nw + ns - 2 + pad))
    far = outer_curve(spec, nw, ns)
    rings = [inner, off, far]
    ni = spec.n_i
    z0, z1 = 0.0, float(spec.thickness)

    bad, worst = _cell_quads_convex(rings)
    if bad:
        raise ValueError(
            f"{bad} non-convex cells in the C-grid (worst cross product {worst:.2e}); "
            f"blockMesh would reject this. Try a smaller `offset`, more `n_smooth`, "
            f"or a larger `first_wake`."
        )

    ids, n_ids = vertex_ids(spec)

    # One coordinate per distinct label, per z-level.
    pos = np.zeros((n_ids, 2), dtype=np.float64)
    for ring in range(3):
        for i in range(ni):
            pos[ids[ring, i]] = rings[ring][i]
    verts = []
    for z in (z0, z1):
        for k in range(n_ids):
            verts.append((pos[k, 0], pos[k, 1], z))
    vtxt = "\n".join(
        f"    ({of._num(x)} {of._num(y)} {of._num(z)})" for (x, y, z) in verts
    )

    def vid(level: int, ring: int, i: int) -> int:
        return level * n_ids + int(ids[ring, i])

    # The wall-normal distribution is uniform along the whole C, and it has to be.
    # Adjacent blocks share a radial face, and blockMesh places that face's points
    # from each block's own grading -- differing gradings make the two placements
    # disagree and it aborts with "Point merge failure ... inconsistent grading".
    # Per-segment grading was tried to cap the aspect ratio and fails for exactly
    # that reason.
    #
    # The consequence is a high aspect ratio at the downstream end of the cut,
    # where a 1e-5 wall cell meets a streamwise cell of order a chord (checkMesh
    # reports ~2e5 on ~7% of cells). That is inherent to combining y+ ~ 1 wall
    # spacing with a 20-chord wake on a structured C-grid -- production airfoil
    # C-grids carry the same -- and it sits in uniform far-wake flow, not in the
    # boundary layer. `first_wake` and `wake_length` trade it off if needed.
    grade_in = expansion_ratio(spec.offset, spec.first_cell, spec.n_inner)
    growth = grade_in ** (1.0 / max(spec.n_inner - 1, 1))
    last_inner = spec.first_cell * growth ** max(spec.n_inner - 1, 0)
    grade_out = expansion_ratio(spec.far_radius - spec.offset, last_inner, spec.n_outer)

    blocks = []
    for ring, (n_rad, grade) in enumerate(
        ((spec.n_inner, grade_in), (spec.n_outer, grade_out))
    ):
        for i in range(ni - 1):
            v = [
                vid(0, ring, i), vid(0, ring, i + 1),
                vid(0, ring + 1, i + 1), vid(0, ring + 1, i),
                vid(1, ring, i), vid(1, ring, i + 1),
                vid(1, ring + 1, i + 1), vid(1, ring + 1, i),
            ]
            blocks.append(
                f"    hex ({' '.join(str(k) for k in v)}) "
                f"(1 {n_rad} 1) simpleGrading (1 {of._num(grade)} 1)"
            )

    # The section occupies inner-boundary segments [nw-1, nw+ns-2); the segments
    # either side of it are the wake cut, whose j=0 faces are internal because
    # both sheets carry the same vertex labels.
    wall, outlet, farfield, front, back = [], [], [], [], []
    for i in range(ni - 1):
        if nw - 1 <= i < nw + ns - 2:
            wall.append(
                f"            ({vid(0, 0, i)} {vid(1, 0, i)} "
                f"{vid(1, 0, i + 1)} {vid(0, 0, i + 1)})"
            )
        farfield.append(
            f"            ({vid(0, 2, i)} {vid(0, 2, i + 1)} "
            f"{vid(1, 2, i + 1)} {vid(1, 2, i)})"
        )
        for ring in (0, 1):
            back.append(
                f"            ({vid(0, ring, i)} {vid(0, ring, i + 1)} "
                f"{vid(0, ring + 1, i + 1)} {vid(0, ring + 1, i)})"
            )
            front.append(
                f"            ({vid(1, ring, i)} {vid(1, ring + 1, i)} "
                f"{vid(1, ring + 1, i + 1)} {vid(1, ring, i + 1)})"
            )
    for ring in (0, 1):
        # i = 0 and i = ni-1 are the two outlet ends of the C.
        outlet.append(
            f"            ({vid(0, ring, 0)} {vid(0, ring + 1, 0)} "
            f"{vid(1, ring + 1, 0)} {vid(1, ring, 0)})"
        )
        outlet.append(
            f"            ({vid(0, ring, ni - 1)} {vid(1, ring, ni - 1)} "
            f"{vid(1, ring + 1, ni - 1)} {vid(0, ring + 1, ni - 1)})"
        )

    def patch(name: str, ptype: str, faces: list[str]) -> str:
        return (
            f"    {name}\n    {{\n        type {ptype};\n        faces\n        (\n"
            + "\n".join(faces)
            + "\n        );\n    }\n"
        )

    return (
        of._header("dictionary", "blockMeshDict", "system")
        + "scale   1;\n\nvertices\n(\n" + vtxt + "\n);\n\n"
        + "blocks\n(\n" + "\n".join(blocks) + "\n);\n\n"
        + "edges\n(\n);\n\n"
        + "boundary\n(\n"
        + patch("airfoil", "wall", wall)
        + patch("farField", "patch", farfield)
        + patch("outlet", "patch", outlet)
        + patch("frontAndBack", "empty", front + back)
        + ");\n\nmergePatchPairs\n(\n);\n"
    )


# --------------------------------------------------------------------------- #
# Case
# --------------------------------------------------------------------------- #


def write_cgrid_case(
    case: FlowCase,
    case_dir: str,
    *,
    spec: CGridSpec | None = None,
    n_iter: int = 3000,
    tol_p: float = 1e-6,
    tol_u: float = 1e-7,
    airfoil_code: str | None = None,
) -> str:
    """Write a ``simpleFoam`` C-grid case. The ``0/`` fields are a cold start."""
    import json as _json
    import shutil

    spec = spec or CGridSpec()
    code = airfoil_code or case.geometry.meta.get("code") or case.geometry.name
    u_inf, v_inf = of._freestream(case)
    nu = float(case.fluid.kinematic_viscosity)
    nut_inf = of.NUTILDA_FREESTREAM_RATIO * nu

    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)

    of._write(os.path.join(case_dir, "system", "controlDict"), of._control_dict(n_iter, n_iter))
    of._write(os.path.join(case_dir, "system", "fvSchemes"),
              of._header("dictionary", "fvSchemes", "system") + of._FV_SCHEMES)
    of._write(os.path.join(case_dir, "system", "fvSolution"),
              of._fv_solution(tol_p, tol_u, n_non_orth=2))
    of._write(os.path.join(case_dir, "system", "blockMeshDict"), block_mesh_dict(spec, code))

    of._write(
        os.path.join(case_dir, "constant", "transportProperties"),
        of._header("dictionary", "transportProperties", "constant")
        + f"transportModel  Newtonian;\nnu              {of._num(nu)};\n",
    )
    turb = (
        "simulationType  RAS;\n\nRAS\n{\n    RASModel        SpalartAllmaras;\n"
        "    turbulence      on;\n    printCoeffs     on;\n}\n"
    )
    for name in ("turbulenceProperties", "momentumTransport"):
        of._write(os.path.join(case_dir, "constant", name),
                  of._header("dictionary", name, "constant") + turb)

    free_u = f"uniform ({of._num(u_inf)} {of._num(v_inf)} 0)"
    of._write(
        os.path.join(case_dir, "0", "U"),
        of._header("volVectorField", "U", "0")
        + f"dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   {free_u}\n;\n\n"
        + of._boundary_field({
            "airfoil": "        type            noSlip;\n",
            "farField": "        type            freestreamVelocity;\n"
                        f"        freestreamValue {free_u};\n",
            "outlet": "        type            zeroGradient;\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )
    of._write(
        os.path.join(case_dir, "0", "p"),
        of._header("volScalarField", "p", "0")
        + "dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 0\n;\n\n"
        + of._boundary_field({
            "airfoil": "        type            zeroGradient;\n",
            "farField": "        type            freestreamPressure;\n"
                        "        freestreamValue uniform 0;\n",
            "outlet": "        type            fixedValue;\n"
                      "        value           uniform 0;\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )
    of._write(
        os.path.join(case_dir, "0", "nuTilda"),
        of._header("volScalarField", "nuTilda", "0")
        + f"dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform {of._num(nut_inf)}\n;\n\n"
        + of._boundary_field({
            "airfoil": "        type            fixedValue;\n"
                       "        value           uniform 0;\n",
            "farField": "        type            freestream;\n"
                        f"        freestreamValue uniform {of._num(nut_inf)};\n",
            "outlet": "        type            zeroGradient;\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )
    of._write(
        os.path.join(case_dir, "0", "nut"),
        of._header("volScalarField", "nut", "0")
        + f"dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform {of._num(nut_inf)}\n;\n\n"
        + of._boundary_field({
            "airfoil": "        type            nutUSpaldingWallFunction;\n"
                       "        value           uniform 0;\n",
            "farField": "        type            calculated;\n"
                        f"        value           uniform {of._num(nut_inf)};\n",
            "outlet": "        type            calculated;\n"
                      f"        value           uniform {of._num(nut_inf)};\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )

    of._write(
        os.path.join(case_dir, "neuroforge.json"),
        _json.dumps(
            {"case": case.name, "mesh": "cgrid", "airfoil": code, "start": "cold",
             "nu": nu, "u_inf": u_inf, "v_inf": v_inf, "nut_freestream": nut_inf,
             "n_iter": int(n_iter), "spec": spec.__dict__, "n_cells": spec.n_cells},
            indent=2, default=str,
        ) + "\n",
    )
    return case_dir


@dataclass
class CGridResult:
    """Outcome of a C-grid ``simpleFoam`` solve, in **mesh** order."""

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    nut: np.ndarray
    centres: np.ndarray
    iterations: int
    converged: bool
    wall_time: float
    execution_time: float
    start: str
    case_dir: str
    residuals: dict = _dc_field(default_factory=dict)
    meta: dict = _dc_field(default_factory=dict)

    def iterations_to(self, threshold: float) -> int | None:
        return of.iterations_to_threshold(self.residuals, threshold)

    @property
    def residual_floor(self) -> float:
        return of.residual_floor(self.residuals)

    def to_grid(self, domain: Domain) -> FlowField:
        """Rasterise back onto a NeuroForge Cartesian grid (nearest neighbour)."""
        from scipy.spatial import cKDTree

        X, Y = domain.grid()
        _, idx = cKDTree(self.centres[:, :2]).query(
            np.stack([X.ravel(), Y.ravel()], axis=1)
        )
        shp = domain.shape
        return FlowField(
            domain=domain,
            u=self.u[idx].reshape(shp).astype(DTYPE),
            v=self.v[idx].reshape(shp).astype(DTYPE),
            p=self.p[idx].reshape(shp).astype(DTYPE),
            nut=self.nut[idx].reshape(shp).astype(DTYPE),
            meta={"source": "openfoam-cgrid", "start": self.start},
        )


def solve_cgrid(
    case: FlowCase,
    *,
    mesh_initial: tuple | None = None,
    case_dir: str | None = None,
    spec: CGridSpec | None = None,
    n_iter: int = 3000,
    distro: str | None = None,
    timeout: float = 7200.0,
    check_mesh: bool = True,
    reuse: bool = True,
) -> CGridResult:
    """Mesh, optionally warm-start from mesh-order values, and solve on a C-grid."""
    import json as _json
    import re as _re
    import time as _time

    of.require_openfoam(distro)
    spec = spec or CGridSpec()
    start = "warm" if mesh_initial is not None else "cold"
    if case_dir is None:
        safe = _re.sub(r"[^A-Za-z0-9._-]+", "_", case.name or "case")
        case_dir = os.path.join("runs", "openfoam", f"{safe}_cgrid_{start}")
    case_dir = os.path.abspath(case_dir)

    if reuse:
        done = of.completed_run(case_dir, n_iter=n_iter, start=start)
        if done is not None:
            return _read(case_dir, done, start, spec, distro, reused=True)

    write_cgrid_case(case, case_dir, spec=spec, n_iter=n_iter)
    of.run_openfoam("blockMesh", case_dir, distro=distro, timeout=timeout,
                    log_name="log.blockMesh")
    mesh_report = {}
    if check_mesh:
        proc = of.run_openfoam("checkMesh -constant", case_dir, distro=distro,
                               timeout=timeout, log_name="log.checkMesh", check=False)
        text = proc.stdout or ""
        mesh_report = {"ok": "Mesh OK." in text,
                       "failed_checks": [ln.strip() for ln in text.splitlines()
                                         if "***" in ln]}

    if mesh_initial is not None:
        from .ogrid import seed_from_mesh

        seed_from_mesh(case_dir, *mesh_initial,
                       nut_freestream=of.NUTILDA_FREESTREAM_RATIO
                       * float(case.fluid.kinematic_viscosity))
        meta_path = os.path.join(case_dir, "neuroforge.json")
        with open(meta_path, encoding="utf-8") as fh:
            meta = _json.load(fh)
        meta["start"] = start
        of._write(meta_path, _json.dumps(meta, indent=2, default=str) + "\n")

    t0 = _time.perf_counter()
    proc = of.run_openfoam("simpleFoam", case_dir, distro=distro, timeout=timeout,
                           log_name="log.simpleFoam")
    info = of.parse_simple_foam_log(proc.stdout or "")
    info["wall_time"] = _time.perf_counter() - t0
    return _read(case_dir, info, start, spec, distro, mesh_report=mesh_report)


def _read(case_dir, info, start, spec, distro, *, mesh_report=None, reused=False):
    """Assemble a :class:`CGridResult` from a case directory on disk."""
    latest = of._latest_time(case_dir)
    if latest is None:
        raise RuntimeError(f"simpleFoam wrote no time directory > 0 in {case_dir}")
    U = of.read_volfield(os.path.join(case_dir, latest, "U"))
    p = of.read_volfield(os.path.join(case_dir, latest, "p"))
    nut_path = os.path.join(case_dir, latest, "nut")
    nut = of.read_volfield(nut_path) if os.path.isfile(nut_path) else np.zeros(len(U))
    if not os.path.isfile(os.path.join(case_dir, "0", "C")):
        of.run_openfoam("postProcess -func writeCellCentres -time 0", case_dir,
                        distro=distro, log_name="log.writeCellCentres")
    C = of.read_volfield(os.path.join(case_dir, "0", "C"))
    return CGridResult(
        u=U[:, 0], v=U[:, 1], p=p, nut=nut, centres=C,
        iterations=int(info["iterations"]), converged=bool(info["converged"]),
        wall_time=float(info.get("wall_time", float("nan"))),
        execution_time=float(info["execution_time"]),
        start=start, case_dir=case_dir, residuals=info["residuals"],
        meta={"time_dir": latest, "mesh": mesh_report or {}, "reused": reused,
              "n_cells": int(len(p)), "spec": spec.__dict__},
    )
