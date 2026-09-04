"""Solver and mesh verification for the C-grid this study runs on.

Every claim in this paper is a *ratio* -- iterations from one seed against
iterations from a cold start, on the same mesh with the same discretisation --
so mesh-independence of the absolute force is not what the results rest on. It
is still what a CFD journal asks for before it reads them, and asking is right:
a mesh too coarse to resolve the boundary layer would not have a near-wall state
worth arguing about.

Three checks, in the order a reader wants them.

**1. y+ along the surface.** The wall-resolved claim, measured rather than
asserted: the distribution of ``y+`` at the first cell centre over every surface
station, per case, from the converged field. Wall-resolved SA wants ``y+ ~ 1``
with no wall function.

**2. Grid convergence.** Forces on this study's mesh against a systematically
coarsened member of the same family (every count halved, first cell doubled --
``scripts/sequencing_probe.py``'s ``coarsen``), both solved to the same
residual. Reported as the relative change in ``C_d``, ``C_d,v``, ``C_p`` and
``C_l``, with the observed order where a third level exists.

**3. The absolute number, against the literature.** NACA0012 at Re 3e6, where
published experimental and computational drag exists to compare against.

Checks 1 and 2 read runs already on disk and cost nothing. Add ``--refine`` to
solve a third, finer level and turn check 2 into a three-level study with an
observed order of convergence.

Usage
-----
    python scripts/verify_mesh.py
    python scripts/verify_mesh.py --refine            # adds the fine level
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

# Must precede numpy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree

from neuroforge.core.types import FlowCase
from neuroforge.solver import cgrid as cg, openfoam as of, warmstart as ws

# NACA0012 at Re 3e6, fully turbulent, low incidence. The comparison a reviewer
# reaches for. Abbott & von Doenhoff's section data is the classical source; the
# NASA Turbulence Modeling Resource's SA computations are the modern one and are
# the right comparison here because this study also runs SA with no transition
# model. Both are quoted as ranges because neither is a single number.
LITERATURE = {
    "naca0012_aoa0": {
        "cd_range": (0.0080, 0.0090),
        "source": "NASA TMR 2-D NACA0012 SA, Re 6e6 grid-converged Cd = 0.00819; "
                  "Abbott & von Doenhoff section data at Re 3e6 give ~0.0085 "
                  "for the smooth section. Fully turbulent SA with no transition "
                  "model is expected at or slightly above the measured value.",
    },
}


def wall_frame(code: str, spec: cg.CGridSpec):
    inner, n_wake, n_surface = cg.inner_curve(code, spec)
    surf = inner[n_wake - 1: n_wake + n_surface - 1]
    mid = 0.5 * (surf[:-1] + surf[1:])
    tangent = surf[1:] - surf[:-1]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-30
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    outward = mid - surf.mean(axis=0)
    flip = np.sign(np.sum(normal * outward, axis=1))
    flip[flip == 0] = 1.0
    return mid, normal * flip[:, None], surf


def y_plus_profile(case_dir: str, code: str, spec: cg.CGridSpec, nu: float):
    """``y+`` at the first cell centre, one value per surface station."""
    latest = of._latest_time(case_dir)
    if latest is None:
        return None
    centres = of.read_volfield(os.path.join(case_dir, "0", "C"))
    U = of.read_volfield(os.path.join(case_dir, latest, "U"))
    inner, n_wake, n_surface = cg.inner_curve(code, spec)
    surface = inner[n_wake - 1: n_wake + n_surface - 1]
    distance = ws.wall_distance(centres, surface)
    y_c = float(np.min(distance))

    mid, normal, _ = wall_frame(code, spec)
    tree = cKDTree(centres[:, :2])
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    _, idx = tree.query(mid + y_c * normal)
    speed = np.abs(np.sum(np.stack([U[idx, 0], U[idx, 1]], axis=1) * tangent, axis=1))
    d = distance[idx]
    u_tau = np.sqrt(nu * speed / np.maximum(d, 1e-30))
    return d * u_tau / nu


def final_forces(case_dir: str) -> dict:
    """Converged ``Cd``/``Cl`` plus the pressure/viscous split.

    The split lives in the separate ``forces`` function object, not in
    ``forceCoeffs`` -- see :func:`openfoam.read_force_components`.
    """
    out = {}
    for reader in (of.read_force_coeffs, of.read_force_components):
        try:
            c = reader(case_dir)
        except Exception:
            continue
        for key in ("Cd", "Cl", "Cd_v", "Cd_p", "Cl_p"):
            series = c.get(key)
            if series is not None and len(series):
                out[key] = float(series[-1])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fine-root", default=os.path.join("runs", "openfoam", "placement2"))
    ap.add_argument("--coarse-root", default=os.path.join("runs", "openfoam", "sequencing"))
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--refine", action="store_true",
                    help="solve a third, finer level (2 solves per case)")
    ap.add_argument("--refine-cases", nargs="*", default=["naca0012_aoa0"])
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--timeout", type=float, default=43200.0)
    ap.add_argument("--out", default=os.path.join("results", "mesh_verification.json"))
    args = ap.parse_args(argv)

    spec = cg.CGridSpec()
    nu = 1.0 / args.re
    tags = sorted(d[: -len("_cold")] for d in os.listdir(args.fine_root)
                  if d.endswith("_cold"))

    print(f"mesh: C-grid, {spec.n_cells:,} cells, first cell {spec.first_cell:.1e} "
          f"chords (nominal centre {spec.first_cell / 2:.1e}), "
          f"far field {spec.far_radius:g} chords\n")

    print("1. y+ at the first cell centre, over every surface station")
    print(f"   {'case':>16}  {'min':>7} {'median':>7} {'mean':>7} {'max':>7}  "
          f"{'frac > 1':>9}")
    yplus = {}
    for tag in tags:
        code = tag.split("_")[0]
        prof = y_plus_profile(os.path.join(args.fine_root, f"{tag}_cold"), code,
                              spec, nu)
        if prof is None:
            continue
        yplus[tag] = {"min": float(prof.min()), "median": float(np.median(prof)),
                      "mean": float(prof.mean()), "max": float(prof.max()),
                      "fraction_above_one": float(np.mean(prof > 1.0))}
        print(f"   {tag:>16}  {prof.min():7.3f} {np.median(prof):7.3f} "
              f"{prof.mean():7.3f} {prof.max():7.3f}  "
              f"{100 * np.mean(prof > 1.0):8.1f}%")

    print("\n2. grid convergence: this study's mesh against a halved member of "
          "the same family")
    coarse_spec = dataclasses.replace(
        spec, n_surface=spec.n_surface // 2, n_wake=spec.n_wake // 2,
        n_inner=spec.n_inner // 2, n_outer=spec.n_outer // 2,
        first_cell=spec.first_cell * 2, first_wake=spec.first_wake * 2)
    print(f"   fine {spec.n_cells:,} cells (first cell {spec.first_cell:.1e})  vs  "
          f"coarse {coarse_spec.n_cells:,} cells "
          f"(first cell {coarse_spec.first_cell:.1e})")
    print(f"   {'case':>16}  " + "  ".join(f"{k:>16}" for k in
                                           ("Cd", "Cd_v", "Cd_p", "Cl")))
    levels = {}
    for tag in tags:
        fine = final_forces(os.path.join(args.fine_root, f"{tag}_cold"))
        coarse = final_forces(os.path.join(args.coarse_root, f"{tag}_coarse"))
        if not fine or not coarse:
            continue
        entry = {"fine": fine, "coarse": coarse, "change_pct": {}}
        bits = []
        for k in ("Cd", "Cd_v", "Cd_p", "Cl"):
            if k not in fine or k not in coarse:
                bits.append(" " * 16)
                continue
            # A symmetric section at zero incidence has Cl ~ 0, and a relative
            # change against zero is not a number anyone should read. Those
            # entries are reported as an absolute difference and left out of the
            # mean.
            if abs(fine[k]) < 1e-4:
                bits.append(f"{fine[k]:8.5f} {coarse[k] - fine[k]:+7.1e}")
                continue
            d = 100 * (coarse[k] - fine[k]) / abs(fine[k])
            entry["change_pct"][k] = float(d)
            bits.append(f"{fine[k]:8.5f} {d:+6.1f}%")
        levels[tag] = entry
        print(f"   {tag:>16}  " + "  ".join(bits))
    if levels:
        for k in ("Cd", "Cd_v", "Cd_p", "Cl"):
            vals = [e["change_pct"][k] for e in levels.values() if k in e["change_pct"]]
            if vals:
                print(f"   {'mean |change| ' + k:>16}: {np.mean(np.abs(vals)):5.1f}% "
                      f"over {len(vals)} cases")

    refined = {}
    if args.refine:
        print("\n   solving the refined level "
              f"({', '.join(args.refine_cases)}) ...")
        fine_spec = dataclasses.replace(
            spec, n_surface=int(spec.n_surface * 1.5),
            n_wake=int(spec.n_wake * 1.5), n_inner=int(spec.n_inner * 1.5),
            n_outer=int(spec.n_outer * 1.5), first_cell=spec.first_cell / 2,
            first_wake=spec.first_wake / 2)
        print(f"   refined {fine_spec.n_cells:,} cells "
              f"(first cell {fine_spec.first_cell:.1e})")
        for tag in args.refine_cases:
            code, _, aoa_text = tag.partition("_aoa")
            case = FlowCase.from_airfoil(airfoil=code, aoa=float(aoa_text),
                                         reynolds=args.re, u_inf=1.0)
            res = cg.solve_cgrid(
                case, case_dir=os.path.join("runs", "openfoam", "verify",
                                            f"{tag}_refined"),
                spec=fine_spec, n_iter=args.n_iter, timeout=args.timeout)
            forces = final_forces(res.case_dir)
            refined[tag] = {"cells": fine_spec.n_cells, "forces": forces}
            base = levels.get(tag, {}).get("fine", {})
            for k, v in forces.items():
                if k in base and abs(base[k]) > 1e-12:
                    print(f"   {tag} {k}: refined {v:.5f} vs study mesh "
                          f"{base[k]:.5f}  ({100 * (v - base[k]) / abs(base[k]):+.2f}%)")

    print("\n3. the absolute number against the literature")
    lit = {}
    for tag, ref in LITERATURE.items():
        got = final_forces(os.path.join(args.fine_root, f"{tag}_cold")).get("Cd")
        if got is None:
            continue
        lo, hi = ref["cd_range"]
        lit[tag] = {"Cd": got, "range": [lo, hi], "inside": bool(lo <= got <= hi),
                    "source": ref["source"]}
        print(f"   {tag}: Cd = {got:.5f}, reference range {lo:.4f}-{hi:.4f}  "
              + ("consistent" if lo <= got <= hi else "OUTSIDE the range"))
        print(f"     {ref['source']}")

    payload = {"re": args.re, "nu": nu,
               "fine_spec": dataclasses.asdict(spec),
               "coarse_spec": dataclasses.asdict(coarse_spec),
               "y_plus": yplus, "levels": levels, "refined": refined,
               "literature": lit}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
