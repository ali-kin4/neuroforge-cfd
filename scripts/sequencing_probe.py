"""Grid sequencing: the classical warm start, and the criterion's out-of-sample test.

Two things at once, and the second is why this script matters more than fairness.

**The baseline we owe a reviewer.** Every industrial aerodynamics workflow warm
starts *somehow*, and the way it does it is not a neural network -- it is grid
sequencing: solve on a coarsened mesh, map the result up, continue on the fine
one. OpenFOAM ships it as ``mapFields``. A paper that measures a learned seed
only against a uniform freestream has not answered "is this better than what I
already do". ``potentialFoam`` was measured and is inert (+0.6%); grid
sequencing is the harder and more honest comparator.

**The test the criterion has to pass.** The placement criterion says a seed
helps only if its representation carries a sample station into the solver's
first cell. A coarsened body-fitted mesh is still body-fitted: halving the
wall-normal count doubles the first cell but leaves the stations clustered at
the wall. So the criterion **predicts grid sequencing helps** -- on a method it
was not derived from, that involves no network, no training data and no
surrogate of any kind. If instead it fails, the criterion is a fact about our
seeds rather than a property of representations, and the paper must say so.

A criterion that only ever explains the experiments it was fitted to is a
description. One that predicts a method it never saw is a finding.

**The third question, which came free.** ``sequenced`` hands over the whole
mapped field; ``sequenced_bl`` hands over only the boundary layer, exactly as
``nf_bl`` does. Condition 2 of the paper says the outer field must be left cold,
and the reason given is that the *surrogate extrapolates badly* out there. If
that reason is right, it is a statement about the model's trust region and
should **not** apply to a coarse-mesh solution, which is a valid solution
everywhere. Then ``sequenced`` should beat ``sequenced_bl``. If instead BL-only
wins for grid sequencing too, condition 2 is about regions rather than about
trust, and the paper's explanation of it is wrong.

**Cost is charged, not assumed.** The coarse solve is a real cost. Its
iterations and its wall-clock are both recorded here so the analysis can charge
it; a coarse cell is cheaper than a fine one, so iterations are converted at the
cell-count ratio, which is the conservative direction. Nothing in this file
reports a saving -- it records what happened and ``reanalyse_depth.py`` scores it.

Fresh work directory, its own ``cold`` and its own ``oracle_mesh`` control:
scoring rule 4 re-scores every arm in a tree when a new arm joins it.

Usage
-----
    python scripts/sequencing_probe.py
    python scripts/reanalyse_depth.py --root runs/openfoam/sequencing
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import warnings

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.core.types import FlowCase
from neuroforge.solver import cgrid as cg, openfoam as of, warmstart as ws
from neuroforge.solver import surrogate_seed as ss

CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0),
         ("naca0012", 0.0), ("naca2415", 5.0)]
THRESHOLDS = (1e-3, 1e-4, 1e-5, 5e-6, 1e-6)
NEW_ARMS = ("oracle_mesh", "nf_bl", "sequenced", "sequenced_bl",
            "sequenced_vel", "sequenced_nut")


def coarsen(spec: cg.CGridSpec, factor: int = 2) -> cg.CGridSpec:
    """Halve every mesh count and double the first cell -- a true coarsening.

    The point is that this is what a practitioner would actually do, not a
    representation chosen to make a point: the coarse mesh is the fine mesh's
    own family, so its stations stay clustered at the wall.
    """
    return dataclasses.replace(
        spec,
        n_surface=max(spec.n_surface // factor, 8),
        n_wake=max(spec.n_wake // factor, 4),
        n_inner=max(spec.n_inner // factor, 4),
        n_outer=max(spec.n_outer // factor, 4),
        first_cell=spec.first_cell * factor,
        first_wake=spec.first_wake * factor,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--n-iter-coarse", type=int, default=6000,
                    help="budget for the coarse solve; it usually exits earlier "
                         "on residualControl, and what it actually spent is recorded")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA")
    ap.add_argument("--ckpt-dir", default=os.path.join("checkpoints", "v2_transolver"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--max-sdf", type=float, default=3.5)
    ap.add_argument("--ramp", type=float, default=3.0)
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "sequencing"))
    ap.add_argument("--out", default=os.path.join("results", "sequencing.json"))
    ap.add_argument("--timeout", type=float, default=43200.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    checkpoints = [os.path.join(args.ckpt_dir, f"seed{k}.pt") for k in args.seeds]
    missing = [p for p in checkpoints if not os.path.isfile(p)]
    if missing:
        print("missing checkpoint(s): " + ", ".join(missing))
        return 1

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cases = CASES
    if args.only:
        wanted = set()
        for text in args.only:
            code, _, aoa = text.partition("@")
            wanted.add((code.strip(), float(aoa or 0.0)))
        cases = [c for c in CASES if c in wanted]
        if not cases:
            print("not in CASES: " + ", ".join(f"{c}@{a:g}" for c, a in sorted(wanted)))
            return 1
        if out_path == os.path.abspath(ap.get_default("out")):
            stem, ext = os.path.splitext(out_path)
            out_path = f"{stem}_" + "_".join(f"{c}{a:g}" for c, a in cases) + ext

    tags = {f"{code}_aoa{aoa:g}" for code, aoa in cases}
    busy = [p for p in of.running_solvers(os.path.basename(os.path.normpath(args.work_dir)))
            if any(os.path.basename(p).startswith(t + "_") for t in tags)]
    if busy and not args.force:
        print("already being solved by another process: "
              + ", ".join(os.path.basename(p) for p in busy[:6]))
        return 1

    def checkpoint(rows):
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter,
                       "n_iter_coarse": args.n_iter_coarse, "factor": args.factor,
                       "checkpoints": checkpoints, "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    spec = cg.CGridSpec()
    coarse_spec = coarsen(spec, args.factor)
    delta = ws.bl_thickness(args.re)
    print(f"Re {args.re:.0e} | boundary layer {delta:.4f}\n"
          f"fine mesh {spec.n_cells} cells, first cell {spec.first_cell:.1e}\n"
          f"coarse mesh {coarse_spec.n_cells} cells, first cell "
          f"{coarse_spec.first_cell:.1e} "
          f"({spec.n_cells / coarse_spec.n_cells:.2f}x fewer)\n", flush=True)

    rows: list[dict] = []
    for code, aoa in cases:
        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=128)
        tag = f"{code}_aoa{aoa:g}"
        print(f"== {tag} ==", flush=True)

        def run(name, **kw):
            return cg.solve_cgrid(case, case_dir=os.path.join(args.work_dir, f"{tag}_{name}"),
                                  spec=spec, n_iter=args.n_iter, timeout=args.timeout, **kw)

        cold = run("cold")
        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]
        distance = ws.wall_distance(cold.centres, surface)
        truth = (cold.u, cold.v, cold.p, cold.nut)

        # --- the classical seed ------------------------------------------------
        # A full solve on the coarsened mesh. Its cost is real and is recorded.
        coarse = cg.solve_cgrid(
            case, case_dir=os.path.join(args.work_dir, f"{tag}_coarse"),
            spec=coarse_spec, n_iter=args.n_iter_coarse, timeout=args.timeout)
        mapped, map_report = ws.sequence_seed(
            (coarse.u, coarse.v, coarse.p, coarse.nut),
            coarse.centres, cold.centres, surface,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)

        # --- the learned seed --------------------------------------------------
        pred, rep = ss.predict_on_mesh(
            checkpoints, cold.centres, surface[:, :2], reynolds=args.re, aoa_deg=aoa,
            wall_distance=distance, max_sdf=args.max_sdf, u_inf=u_inf,
            nut_freestream=nut_fs)

        free = (np.full_like(pred[0], u_inf), np.full_like(pred[0], v_inf),
                np.zeros_like(pred[0]), np.full_like(pred[0], nut_fs))

        def bl_only(background):
            seed, _ = ws.masked_seed(free, cold.centres, surface,
                                     background=background, free_within=delta,
                                     ramp=args.ramp, u_inf=u_inf, v_inf=v_inf,
                                     nut_freestream=nut_fs)
            return seed

        # The channel split, for a seed with no network in it. Section 5.4 says
        # velocity and eddy viscosity must be handed over together, and explains
        # it by Spalart-Allmaras' production term: a `nut` inconsistent with the
        # strain that generated it is a wrong momentum sink. If that mechanism is
        # about *consistency* rather than about surrogate error, it must bite a
        # coarse-mesh solution too -- and a coarse mesh's turbulence field is
        # under-resolved, so its `nut` is inconsistent with the fine mesh's
        # strain in exactly the same way.
        free_map = (np.full_like(mapped[0], u_inf), np.full_like(mapped[0], v_inf),
                    np.zeros_like(mapped[0]), np.full_like(mapped[0], nut_fs))
        mapped_vel = (mapped[0], mapped[1], free_map[2], free_map[3])
        mapped_nut = (free_map[0], free_map[1], free_map[2], mapped[3])

        results = {
            "oracle_mesh": run("oracle_mesh", mesh_initial=truth),
            "nf_bl": run("nf_bl", mesh_initial=bl_only(pred)),
            "sequenced": run("sequenced", mesh_initial=mapped),
            "sequenced_bl": run("sequenced_bl", mesh_initial=bl_only(mapped)),
            "sequenced_vel": run("sequenced_vel", mesh_initial=bl_only(mapped_vel)),
            "sequenced_nut": run("sequenced_nut", mesh_initial=bl_only(mapped_nut)),
        }

        in_bl = distance <= delta

        def err(a, b, m):
            return float(100 * np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-30))

        row = {
            "case": tag, "re": args.re, "covered_fraction": rep["covered_fraction"],
            "map": map_report,
            "coarse": {
                "cells": coarse_spec.n_cells,
                "first_cell": coarse_spec.first_cell,
                "iterations": coarse.iterations,
                "converged": bool(coarse.converged),
                "wall_time_s": coarse.wall_time,
                "execution_time_s": coarse.execution_time,
                # Charged in fine-mesh-equivalent iterations at the cell ratio:
                # a coarse iteration touches fewer cells, so this is the
                # conservative conversion.
                "fine_equivalent_iterations":
                    coarse.iterations * coarse_spec.n_cells / spec.n_cells,
            },
            "cold_wall_time_s": cold.wall_time,
            "bl_error_pct_vs_converged": {
                "sequenced": {n: err(mapped[i], truth[i], in_bl)
                              for i, n in enumerate(ws.FIELDS)},
                "nf_bl": {n: err(pred[i], truth[i], in_bl)
                          for i, n in enumerate(ws.FIELDS)},
            },
        }
        for name, res in results.items():
            row[f"{name}_wall_time_s"] = res.wall_time
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        checkpoint(rows)

        c = row["coarse"]
        print(f"   coarse solve: {c['iterations']} it"
              f"{' (converged)' if c['converged'] else ''}, {c['wall_time_s']:.0f} s"
              f"  -> charged {c['fine_equivalent_iterations']:.0f} fine-equivalent it"
              f"   | cold {cold.wall_time:.0f} s"
              f"   | extrapolated {100 * row['map']['extrapolated_fraction']:.1f}%",
              flush=True)
        print("   boundary-layer error vs converged: "
              + "  ".join(f"{a}={row['bl_error_pct_vs_converged'][a]['u']:.1f}%"
                          for a in ("sequenced", "nf_bl")), flush=True)
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            base = row[f"cold@{k}"]
            bits = []
            for name in NEW_ARMS:
                v = row.get(f"{name}@{k}")
                s = (1 - v / base) if (base and v) else None
                bits.append(f"{name}={v}" + (f" ({100 * s:+.0f}%)" if s is not None else ""))
            print(f"   @{k}: cold={base}  " + "  ".join(bits), flush=True)

    print("\nper-threshold mean (raw iterations, coarse solve NOT yet charged)")
    for t in THRESHOLDS:
        k = f"{t:.0e}"
        bits = []
        for name in NEW_ARMS:
            vals = [1 - r[f"{name}@{k}"] / r[f"cold@{k}"] for r in rows
                    if r.get(f"cold@{k}") and r.get(f"{name}@{k}")]
            bits.append(f"{name} {100 * np.mean(vals):+6.1f}% (n={len(vals)})"
                        if vals else f"{name} --")
        print(f"  @{k}: " + "   ".join(bits))

    charged = [r["coarse"]["fine_equivalent_iterations"] for r in rows]
    if charged:
        print(f"\ncoarse solve charges {np.mean(charged):.0f} fine-equivalent "
              f"iterations on average -- subtract this from every `sequenced*` saving")
    checkpoint(rows)
    print(f"\nwrote {os.path.relpath(out_path)}")
    print("\nThe forces are what decides this. Score with:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
