"""Body-fitted O-grid meshes for the OpenFOAM backend.

The uniform Cartesian mesh in :mod:`neuroforge.solver.openfoam` cannot run
external aerodynamics at realistic Reynolds numbers: measured on this repo,
``simpleFoam`` dies with SIGFPE above a cell Reynolds number of roughly 250,
which is ~300x short of AirfRANS (Re 2e6-6e6). That is not a defect in the case
writer -- a RANS airfoil needs the first cell at y+ ~ 1, about 1e-5 chord, and no
uniform grid can deliver that while also spanning a 20-chord far field.

This module builds a **graded, body-fitted O-grid** instead:

* an 8-block ring (4 arcs x 2 radial layers) written as a ``blockMeshDict``;
* an inner layer from the airfoil surface to a normal-offset curve, carrying the
  boundary-layer grading (first cell ~1e-5 chord, ~12% growth);
* an outer layer from that curve to a far-field circle ~20 chords out.

Two blocks radially rather than one is deliberate. A single block from the
airfoil straight to the far-field circle interpolates along lines joining
surface points to distant circle points; those lines are nowhere near normal to
the wall, so the near-wall cells are badly skewed exactly where the boundary
layer lives. The intermediate offset curve makes the inner layer's lines
wall-normal by construction.

O-grid rather than C-grid: a C-grid's wake cut needs coincident-face stitching
and a degenerate vertex pair at a sharp trailing edge. An O-grid on a **blunt**
trailing edge (``naca_airfoil(..., closed=False)``, which leaves a finite
~0.0025-chord base) has neither -- ``blockMesh`` matches the block faces of the
ring automatically. The price is coarse wake resolution, which does not affect
iterations-to-threshold, the metric the warm-start experiment reports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field as _dc_field

import numpy as np

from neuroforge.core.types import DTYPE, Domain, FlowCase, FlowField
from neuroforge.geometry.airfoil import naca_airfoil

from . import openfoam as of

__all__ = [
    "OGridSpec",
    "OGridResult",
    "airfoil_loop",
    "offset_curve",
    "far_field_circle",
    "expansion_ratio",
    "block_mesh_dict",
    "write_ogrid_case",
    "seed_from_mesh",
    "solve_ogrid",
]


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #


def _resample_by_index(loop: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed loop to ``n`` points, preserving its clustering.

    Parametrised by *index*, not arclength: the NACA generator already clusters
    points at the leading and trailing edges (cosine spacing in x), and
    arclength resampling would throw that away.
    """
    m = len(loop)
    src = np.arange(m + 1, dtype=np.float64) / m
    closed = np.vstack([loop, loop[:1]])
    tgt = np.arange(n, dtype=np.float64) / n
    return np.stack(
        [np.interp(tgt, src, closed[:, 0]), np.interp(tgt, src, closed[:, 1])], axis=1
    )


def _resample_open(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline to ``n`` points, preserving its clustering.

    Parametrised by index, not arclength: the NACA generator already clusters at
    the leading and trailing edges and arclength resampling would discard that.
    """
    m = len(pts)
    src = np.arange(m, dtype=np.float64) / (m - 1)
    tgt = np.arange(n, dtype=np.float64) / (n - 1)
    return np.stack(
        [np.interp(tgt, src, pts[:, 0]), np.interp(tgt, src, pts[:, 1])], axis=1
    )


def _outward_normals(loop: np.ndarray) -> np.ndarray:
    """Unit outward normals of a closed loop, sign fixed by the polygon area."""
    nxt = np.roll(loop, -1, axis=0)
    prv = np.roll(loop, 1, axis=0)
    t = nxt - prv
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    # (ty, -tx) is the right-hand normal; for a counter-clockwise loop that
    # points outward, for a clockwise one it points inward.
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    x, y = loop[:, 0], loop[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return n if area > 0 else -n


def airfoil_loop(
    code: str = "naca0012", n_surface: int = 240, n_te: int = 8, n_points: int = 400,
    n_blocks: int = 12,
) -> np.ndarray:
    """Closed surface loop with the blunt trailing-edge face resolved.

    ``naca_airfoil(..., closed=False)`` returns the upper surface from the
    trailing edge to the leading edge and back along the lower surface, ending at
    the opposite trailing-edge corner -- the ~0.0025-chord base between those two
    corners is a bare segment with no points on it. Left as-is, the O-grid puts a
    single cell across the whole trailing edge. ``n_te`` points are inserted
    along it here.

    The result is ordered so that the tangent crossed with the outward normal
    gives ``+z``, which is what ``blockMesh`` needs for positive cell volumes.
    """
    if n_surface % n_blocks:
        raise ValueError(
            f"n_surface must be divisible by n_blocks (got {n_surface} / {n_blocks})"
        )
    geom = naca_airfoil(code, n_points=n_points, closed=False)
    pts = np.asarray(geom.surface_points, dtype=np.float64)

    # Layout is deterministic: indices [0, n_surface - n_te) walk the wetted
    # surface between the two trailing-edge corners, and the final n_te indices
    # walk the blunt trailing-edge face back to the start.
    n_surf = n_surface - n_te
    if n_surf < n_te:
        raise ValueError(f"n_surface={n_surface} too small for n_te={n_te}")
    surface = _resample_open(pts, n_surf)
    lo, hi = surface[-1], surface[0]
    face = lo + np.linspace(0.0, 1.0, n_te + 2)[1:-1, None] * (hi - lo)
    loop = np.vstack([surface, face])

    # blockMesh wants (tangent x outward-normal) = +z; flip the traversal if not.
    # Reversing moves the trailing-edge face to the front, so roll it back to the
    # end and keep the layout above true whichever way the generator ordered the
    # section.
    n_out = _outward_normals(loop)
    t = np.roll(loop, -1, axis=0) - loop
    if float(np.sum(t[:, 0] * n_out[:, 1] - t[:, 1] * n_out[:, 0])) < 0.0:
        loop = np.roll(loop[::-1], -n_te, axis=0).copy()
    return loop


def offset_curve(loop: np.ndarray, distance: float, n_smooth: int = 40) -> np.ndarray:
    """Normal offset of ``loop`` at ``distance``, Laplacian-smoothed.

    An outward offset of a convex curve cannot self-intersect, and an airfoil is
    convex to within negligible curvature on the aft lower surface; the smoothing
    is belt-and-braces for the high-curvature leading edge and the trailing-edge
    corners, and it also relaxes the point distribution so the inner block's
    radial lines stay close to wall-normal.
    """
    out = loop + distance * _outward_normals(loop)
    for _ in range(max(0, n_smooth)):
        out = out + 0.25 * (np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0) - 2.0 * out)
    return out


def far_field_circle(
    ref: np.ndarray, radius: float, centre: tuple[float, float] = (0.25, 0.0)
) -> np.ndarray:
    """Far-field circle, one point per ``ref`` point at that point's bearing.

    Sampling by bearing (rather than by uniform angle) keeps each outer-block
    line pointing away from its own inner point, which keeps the outer layer
    close to radial.
    """
    c = np.asarray(centre, dtype=np.float64)
    d = np.asarray(ref, dtype=np.float64) - c
    theta = np.arctan2(d[:, 1], d[:, 0])
    return c + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _segment_quads_convex(rings: list[np.ndarray]) -> tuple[int, float]:
    """Count non-convex (ring, ring+1) segment quads and report the worst cross."""
    bad, worst = 0, float("inf")
    n = len(rings[0])
    for ri in range(len(rings) - 1):
        a, b = rings[ri], rings[ri + 1]
        for i in range(n):
            j = (i + 1) % n
            q = np.array([a[i], a[j], b[j], b[i]])
            e = np.roll(q, -1, axis=0) - q
            e2 = np.roll(e, -1, axis=0)
            cr = e[:, 0] * e2[:, 1] - e[:, 1] * e2[:, 0]
            worst = min(worst, float(cr.min()))
            if (cr <= 0).any():
                bad += 1
    return bad, worst


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def expansion_ratio(length: float, first_cell: float, n: int) -> float:
    """``blockMesh`` expansion ratio (last cell / first cell) for a geometric run.

    Solves ``first_cell * (r^n - 1) / (r - 1) = length`` for the growth factor
    ``r`` by bisection, then returns ``r^(n-1)`` -- which is what ``simpleGrading``
    actually takes. Growth factor is the number that governs mesh quality; the
    expansion ratio it implies is large (O(1e2-1e3)) and that is normal for a
    boundary-layer mesh.
    """
    if n < 2:
        return 1.0
    if first_cell * n >= length:  # uniform or finer than requested: no grading
        return 1.0
    lo, hi = 1.0 + 1e-12, 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        total = first_cell * (mid**n - 1.0) / (mid - 1.0)
        if total < length:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    return float(r ** (n - 1))


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OGridSpec:
    """Knobs for the O-grid. Defaults target y+ ~ 1 at Re 3e6, chord 1."""

    n_surface: int = 240          # points around the airfoil (divisible by n_blocks)
    n_blocks: int = 12            # arcs around the ring; see block_mesh_dict
    n_te: int = 8                 # points across the blunt trailing-edge face
    n_inner: int = 60             # radial cells, wall -> offset curve
    n_outer: int = 40             # radial cells, offset curve -> far field
    first_cell: float = 1.0e-5    # wall-normal height of the first cell [chord]
    offset: float = 0.08          # thickness of the wall-normal inner layer
    n_smooth: int = 10            # Laplacian passes on the offset curve
    far_radius: float = 20.0      # far-field radius [chord]
    centre: tuple[float, float] = (0.25, 0.0)
    thickness: float = 0.1        # span in z (2-D: `empty` front/back)

    @property
    def n_cells(self) -> int:
        return self.n_surface * (self.n_inner + self.n_outer)


def _poly_line(a: int, b: int, pts: np.ndarray, z: float) -> str:
    """A ``polyLine`` edge between two vertices through the interior points."""
    if len(pts) == 0:
        return ""
    body = "\n".join(f"        ({of._num(p[0])} {of._num(p[1])} {of._num(z)})" for p in pts)
    return f"    polyLine {a} {b}\n    (\n{body}\n    )\n"


def block_mesh_dict(spec: OGridSpec, loop: np.ndarray) -> str:
    """``blockMeshDict`` for the O-grid: **one block per surface segment**.

    ``blockMesh`` builds a straight-sided "topology" hex per block from its corner
    vertices alone, applies the curved ``edges`` afterwards, and rejects any
    topology hex that is concave ("zero or negative pyramid volume"). With a few
    wide blocks that check fails at the trailing edge, and it fails for a
    different reason under every corner placement tried:

    * corners evenly spaced -> one lands mid-TE-face, where the offset curve turns
      ~90 degrees while the surface barely moves;
    * corners anchored on the TE corners -> the corner sits on a ~90-degree turn
      of the surface, so the quad's interior angle there exceeds 180 degrees;
    * one block straddling the TE -> its two corners sit on *opposite* surfaces
      and the straight chord between them passes through the airfoil.

    All three are properties of a wide block near a high-curvature feature, so
    adding blocks only shrinks the violation asymptotically. Using one block per
    surface segment removes the failure mode by construction: the topology quad
    *is* the cell, so blockMesh sees exactly the geometry checked here, no
    ``edges`` entries are needed at all, and convexity is verified directly.
    The mesh is unchanged -- the block count is bookkeeping, not resolution.
    """
    n = spec.n_surface
    off = offset_curve(loop, spec.offset, spec.n_smooth)
    far = far_field_circle(off, spec.far_radius, spec.centre)
    rings = [loop, off, far]
    z0, z1 = 0.0, float(spec.thickness)

    bad, worst = _segment_quads_convex(rings)
    if bad:
        raise ValueError(
            f"{bad} non-convex cells in the O-grid (worst cross product {worst:.2e}); "
            f"blockMesh would reject this. Try a smaller `offset`, more `n_smooth`, "
            f"or more `n_surface`."
        )

    verts: list[tuple[float, float, float]] = []
    for z in (z0, z1):
        for ring in rings:
            for pt in ring:
                verts.append((float(pt[0]), float(pt[1]), z))
    vtxt = "\n".join(
        f"    ({of._num(x)} {of._num(y)} {of._num(z)})" for (x, y, z) in verts
    )

    def vid(level: int, ring: int, i: int) -> int:
        return level * (3 * n) + ring * n + (i % n)

    grade_in = expansion_ratio(spec.offset, spec.first_cell, spec.n_inner)
    growth = grade_in ** (1.0 / max(spec.n_inner - 1, 1))
    last_inner = spec.first_cell * growth ** max(spec.n_inner - 1, 0)
    grade_out = expansion_ratio(spec.far_radius - spec.offset, last_inner, spec.n_outer)

    blocks = []
    for ring_i, (n_rad, grade) in enumerate(
        ((spec.n_inner, grade_in), (spec.n_outer, grade_out))
    ):
        for i in range(n):
            j = i + 1
            v = [
                vid(0, ring_i, i), vid(0, ring_i, j),
                vid(0, ring_i + 1, j), vid(0, ring_i + 1, i),
                vid(1, ring_i, i), vid(1, ring_i, j),
                vid(1, ring_i + 1, j), vid(1, ring_i + 1, i),
            ]
            blocks.append(
                f"    hex ({' '.join(str(k) for k in v)}) "
                f"(1 {n_rad} 1) simpleGrading (1 {of._num(grade)} 1)"
            )

    airfoil_faces, far_faces, front, back = [], [], [], []
    for i in range(n):
        j = i + 1
        airfoil_faces.append(
            f"            ({vid(0, 0, i)} {vid(1, 0, i)} {vid(1, 0, j)} {vid(0, 0, j)})"
        )
        far_faces.append(
            f"            ({vid(0, 2, i)} {vid(0, 2, j)} {vid(1, 2, j)} {vid(1, 2, i)})"
        )
        for ring_i in (0, 1):
            back.append(
                f"            ({vid(0, ring_i, i)} {vid(0, ring_i, j)} "
                f"{vid(0, ring_i + 1, j)} {vid(0, ring_i + 1, i)})"
            )
            front.append(
                f"            ({vid(1, ring_i, i)} {vid(1, ring_i + 1, i)} "
                f"{vid(1, ring_i + 1, j)} {vid(1, ring_i, j)})"
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
        + patch("airfoil", "wall", airfoil_faces)
        + patch("farField", "patch", far_faces)
        + patch("frontAndBack", "empty", back + front)
        + ");\n\nmergePatchPairs\n(\n);\n"
    )


# --------------------------------------------------------------------------- #
# Case
# --------------------------------------------------------------------------- #


def write_ogrid_case(
    case: FlowCase,
    case_dir: str,
    *,
    spec: OGridSpec | None = None,
    n_iter: int = 3000,
    tol_p: float = 1e-6,
    tol_u: float = 1e-7,
    airfoil_code: str | None = None,
) -> str:
    """Write a ``simpleFoam`` O-grid case. The ``0/`` fields are a cold start.

    A warm start cannot be written here: the mesh does not exist yet, so there
    are no cell centres to interpolate onto. :func:`solve_ogrid` meshes first and
    then overwrites ``0/`` -- see :func:`seed_warm_start`.
    """
    import shutil

    spec = spec or OGridSpec()
    code = airfoil_code or case.geometry.meta.get("code") or case.geometry.name
    u_inf, v_inf = of._freestream(case)
    nu = float(case.fluid.kinematic_viscosity)
    nut_inf = of.NUTILDA_FREESTREAM_RATIO * nu

    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)

    loop = airfoil_loop(code, n_surface=spec.n_surface, n_te=spec.n_te,
                        n_blocks=spec.n_blocks)

    of._write(os.path.join(case_dir, "system", "controlDict"), of._control_dict(n_iter, n_iter))
    of._write(os.path.join(case_dir, "system", "fvSchemes"),
              of._header("dictionary", "fvSchemes", "system") + of._FV_SCHEMES)
    # The O-grid is markedly non-orthogonal where the wall-normal inner layer
    # meets the bearing-radial outer layer (checkMesh: max 75 deg, ~1300 severe
    # faces), so the pressure equation gets explicit non-orthogonal correctors.
    of._write(os.path.join(case_dir, "system", "fvSolution"),
              of._fv_solution(tol_p, tol_u, n_non_orth=2))
    of._write(os.path.join(case_dir, "system", "blockMeshDict"), block_mesh_dict(spec, loop))

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

    # Far field uses the freestream family: one patch that switches between
    # inflow and outflow per face from the local flux, so a circular far-field
    # boundary needs no inlet/outlet split.
    of._write(
        os.path.join(case_dir, "0", "U"),
        of._header("volVectorField", "U", "0")
        + "dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   "
        + f"uniform ({of._num(u_inf)} {of._num(v_inf)} 0)\n;\n\n"
        + of._boundary_field({
            "airfoil": "        type            noSlip;\n",
            "farField": "        type            freestreamVelocity;\n"
                        f"        freestreamValue uniform ({of._num(u_inf)} {of._num(v_inf)} 0);\n",
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
            "frontAndBack": "        type            empty;\n",
        }),
    )
    of._write(
        os.path.join(case_dir, "0", "nut"),
        of._header("volScalarField", "nut", "0")
        + f"dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform {of._num(nut_inf)}\n;\n\n"
        + of._boundary_field({
            # y+ ~ 1 is the design point, but Spalding's law is valid across the
            # whole y+ range, so an imperfect first cell degrades gracefully.
            "airfoil": "        type            nutUSpaldingWallFunction;\n"
                       "        value           uniform 0;\n",
            "farField": "        type            calculated;\n"
                        f"        value           uniform {of._num(nut_inf)};\n",
            "frontAndBack": "        type            empty;\n",
        }),
    )

    import json as _json

    of._write(
        os.path.join(case_dir, "neuroforge.json"),
        _json.dumps(
            {
                "case": case.name, "mesh": "ogrid", "airfoil": code,
                "nu": nu, "u_inf": u_inf, "v_inf": v_inf, "nut_freestream": nut_inf,
                "n_iter": int(n_iter), "spec": spec.__dict__, "n_cells": spec.n_cells,
            },
            indent=2, default=str,
        ) + "\n",
    )
    return case_dir


def seed_warm_start(case_dir: str, field: FlowField, domain: Domain, *, nut_freestream: float,
                    u_inf: float, v_inf: float, distro: str | None = None) -> dict:
    """Overwrite ``0/`` with ``field`` interpolated onto the meshed cell centres.

    Must run *after* ``blockMesh``. Cell centres come from
    ``postProcess -func writeCellCentres``; values are interpolated bilinearly
    from the NeuroForge Cartesian grid, and cells outside that crop (most of the
    20-chord far field) fall back to freestream.

    Returns a report including ``covered_fraction`` -- the share of cells the
    prediction actually reaches. On a 20-chord O-grid against a 3-chord crop this
    is small, and it is the honest denominator for any warm-start claim.
    """
    from scipy.interpolate import RegularGridInterpolator

    of.run_openfoam("postProcess -func writeCellCentres -time 0", case_dir,
                    distro=distro, log_name="log.writeCellCentres")
    C = of.read_volfield(os.path.join(case_dir, "0", "C"))
    cx, cy = C[:, 0], C[:, 1]

    x, y = domain.axes()
    inside = (cx >= x[0]) & (cx <= x[-1]) & (cy >= y[0]) & (cy <= y[-1])
    pts = np.stack([np.clip(cy, y[0], y[-1]), np.clip(cx, x[0], x[-1])], axis=1)

    def sample(arr, fill):
        interp = RegularGridInterpolator((y, x), np.asarray(arr, dtype=np.float64),
                                         bounds_error=False, fill_value=None)
        vals = interp(pts)
        return np.where(inside, vals, fill)

    u = sample(field.u, u_inf)
    v = sample(field.v, v_inf)
    p = sample(field.p, 0.0)
    nut_src = field.nut if field.nut is not None else np.full(domain.shape, nut_freestream)
    nut = np.maximum(sample(nut_src, nut_freestream), 0.0)

    def rewrite(name, cls, dims, body):
        path = os.path.join(case_dir, "0", name)
        text = open(path, encoding="utf-8").read()
        head, _, tail = text.partition("internalField")
        _, _, rest = tail.partition(";")
        of._write(path, head + "internalField   " + body + ";" + rest)

    rewrite("U", "volVectorField", None, of._vector_list(u, v))
    rewrite("p", "volScalarField", None, of._scalar_list(p))
    rewrite("nut", "volScalarField", None, of._scalar_list(nut))
    rewrite("nuTilda", "volScalarField", None, of._scalar_list(np.maximum(nut, nut_freestream)))

    return {
        "cells": int(C.shape[0]),
        "covered_cells": int(inside.sum()),
        "covered_fraction": float(inside.mean()),
    }


def seed_from_mesh(case_dir: str, u, v, p, nut, nut_freestream: float) -> None:
    """Overwrite ``0/`` with values already in **mesh** order (no interpolation).

    Used for the oracle control arm: warm-starting from a solution computed on
    this very mesh, so the start is exact and any failure to save iterations is a
    fault in the measurement rather than in the initial guess.
    """
    def rewrite(name: str, body: str) -> None:
        path = os.path.join(case_dir, "0", name)
        text = open(path, encoding="utf-8").read()
        head, _, tail = text.partition("internalField")
        _, _, rest = tail.partition(";")
        of._write(path, head + "internalField   " + body + ";" + rest)

    nut = np.maximum(np.asarray(nut, dtype=np.float64), 0.0)
    rewrite("U", of._vector_list(u, v))
    rewrite("p", of._scalar_list(p))
    rewrite("nut", of._scalar_list(nut))
    rewrite("nuTilda", of._scalar_list(np.maximum(nut, nut_freestream)))


@dataclass
class OGridResult:
    """Outcome of an O-grid ``simpleFoam`` solve, in **mesh** order."""

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    nut: np.ndarray
    centres: np.ndarray            # (n_cells, 3)
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
        tree = cKDTree(self.centres[:, :2])
        _, idx = tree.query(np.stack([X.ravel(), Y.ravel()], axis=1))
        shp = domain.shape
        return FlowField(
            domain=domain,
            u=self.u[idx].reshape(shp).astype(DTYPE),
            v=self.v[idx].reshape(shp).astype(DTYPE),
            p=self.p[idx].reshape(shp).astype(DTYPE),
            nut=self.nut[idx].reshape(shp).astype(DTYPE),
            meta={"source": "openfoam-ogrid", "start": self.start},
        )


def solve_ogrid(
    case: FlowCase,
    *,
    initial: FlowField | None = None,
    mesh_initial: tuple | None = None,
    case_dir: str | None = None,
    spec: OGridSpec | None = None,
    n_iter: int = 3000,
    distro: str | None = None,
    timeout: float = 7200.0,
    check_mesh: bool = True,
) -> OGridResult:
    """Mesh, (optionally) warm-start, and solve ``case`` on a body-fitted O-grid."""
    import re as _re
    import time as _time

    of.require_openfoam(distro)
    spec = spec or OGridSpec()
    start = "warm" if (initial is not None or mesh_initial is not None) else "cold"
    if case_dir is None:
        safe = _re.sub(r"[^A-Za-z0-9._-]+", "_", case.name or "case")
        case_dir = os.path.join("runs", "openfoam", f"{safe}_ogrid_{start}")
    case_dir = os.path.abspath(case_dir)

    write_ogrid_case(case, case_dir, spec=spec, n_iter=n_iter)
    of.run_openfoam("blockMesh", case_dir, distro=distro, timeout=timeout,
                    log_name="log.blockMesh")
    mesh_report = {}
    if check_mesh:
        proc = of.run_openfoam("checkMesh -constant", case_dir, distro=distro,
                               timeout=timeout, log_name="log.checkMesh", check=False)
        text = proc.stdout or ""
        mesh_report = {
            "ok": "Mesh OK." in text,
            "failed_checks": [ln.strip() for ln in text.splitlines() if "***" in ln],
        }

    seed = {}
    u_inf, v_inf = of._freestream(case)
    nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
    if mesh_initial is not None:
        seed_from_mesh(case_dir, *mesh_initial, nut_freestream=nut_fs)
        seed = {"mode": "mesh", "cells": int(len(mesh_initial[0]))}
    elif initial is not None:
        seed = seed_warm_start(
            case_dir, initial, case.domain,
            nut_freestream=of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity),
            u_inf=u_inf, v_inf=v_inf, distro=distro,
        )

    t0 = _time.perf_counter()
    proc = of.run_openfoam("simpleFoam", case_dir, distro=distro, timeout=timeout,
                           log_name="log.simpleFoam")
    info = of.parse_simple_foam_log(proc.stdout or "")
    wall = _time.perf_counter() - t0

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

    return OGridResult(
        u=U[:, 0], v=U[:, 1], p=p, nut=nut, centres=C,
        iterations=int(info["iterations"]), converged=bool(info["converged"]),
        wall_time=wall, execution_time=float(info["execution_time"]),
        start=start, case_dir=case_dir, residuals=info["residuals"],
        meta={"time_dir": latest, "mesh": mesh_report, "warm_start": seed,
              "n_cells": int(len(p)), "spec": spec.__dict__},
    )
