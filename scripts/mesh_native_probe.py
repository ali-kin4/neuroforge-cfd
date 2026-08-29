"""The controlled test of the mesh-native claim, and of which channel does the work.

Two questions, one prediction, so nothing else can move.

**Is resampling the mechanism?** ``scripts/seed_gradient_diagnostic.py`` measured
that every 16,384-value grid projection of the *exact* converged field leaves
1700-1900% error in the first-cell wall gradient -- the quantity viscous drag
integrates -- while the trained backbone, queried directly at the C-grid cell
centres, leaves 54%. A 16k grid simply has no station 4e-6 chords off the wall.
That is a compelling explanation for why every projected arm is negative on drag
and the mesh-native one is positive, and it is not yet a controlled result:
``fitted_bl`` comes from the oracle and ``nf_bl`` from the network, so
representation and source of truth are confounded.

``nf_bl_proj`` removes the confound. It takes the **same network prediction**
``nf_bl`` uses, sends it through the same 256x64 wall-fitted round-trip the
``fitted_*`` arms use, and seeds with the result. Everything else is identical.
If it goes negative, resampling is the mechanism and the claim is established:
*a surrogate intended to accelerate a solver must be evaluable at the solver's
own cell centres.* If it stays positive, the mechanism is the source of the
field, not its representation, and the paper says that instead.

**Which channel is doing the work?** ``nf_bl`` hands over near-wall velocity
*and* eddy viscosity. The convergence decomposition predicted that seeding
pressure would be the win; that prediction was falsified (``fitted_p`` is inert
at +0.1%, ``composite`` and ``potential`` are negative), so the win is in the
other two and it is not known which. ``nf_bl_nut`` and ``nf_bl_vel`` split them.
``nut`` is the slow field to develop and the one the model predicts *worst*
(52-111% error), so either answer is worth having.

Four solves per case become three: ``cold``, ``oracle_mesh`` and ``nf_bl`` are
reused from ``repr3``.

Usage
-----
    python scripts/mesh_native_probe.py --only naca0012@4
    # one process per case is the sanctioned way to parallelise (see PLANS.md)
    python scripts/reanalyse_depth.py --root runs/openfoam/repr3 --exclude naca4412
"""

from __future__ import annotations

import argparse
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
NEW_ARMS = ("nf_bl_proj", "nf_bl_nut", "nf_bl_vel")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA")
    ap.add_argument("--ckpt-dir", default=os.path.join("checkpoints", "v2_transolver"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--max-sdf", type=float, default=3.5)
    ap.add_argument("--ramp", type=float, default=3.0)
    ap.add_argument("--n-s", type=int, default=256)
    ap.add_argument("--n-n", type=int, default=64)
    ap.add_argument("--first", type=float, default=2.5e-4)
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "repr3"))
    ap.add_argument("--out", default=os.path.join("results", "mesh_native.json"))
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
        print(f"running only: {', '.join(f'{c}@{a:g}' for c, a in cases)}\n"
              f"checkpointing to {out_path}\n")

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
                       "checkpoints": checkpoints, "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    print(f"Re {args.re:.0e} | boundary layer {delta:.4f} | "
          f"projection {args.n_s}x{args.n_n} = {args.n_s * args.n_n} values\n")

    rows = []
    for code, aoa in cases:
        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=128)
        tag = f"{code}_aoa{aoa:g}"
        print(f"== {tag} ==", flush=True)

        def run(name, **kw):
            return cg.solve_cgrid(case, case_dir=os.path.join(args.work_dir, f"{tag}_{name}"),
                                  spec=spec, n_iter=args.n_iter, timeout=args.timeout, **kw)

        cold = run("cold")            # reused from disk; no re-solve
        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]
        distance = ws.wall_distance(cold.centres, surface)

        pred, rep = ss.predict_on_mesh(
            checkpoints, cold.centres, surface[:, :2], reynolds=args.re, aoa_deg=aoa,
            wall_distance=distance, max_sdf=args.max_sdf, u_inf=u_inf,
            nut_freestream=nut_fs)

        # The same prediction through the same round-trip the fitted arms use.
        projected, proj_rep = ws.clustered_seed(
            pred, cold.centres, surface, n_s=args.n_s, n_n=args.n_n,
            first=args.first, u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)

        free = (np.full_like(pred[0], u_inf), np.full_like(pred[0], v_inf),
                np.zeros_like(pred[0]), np.full_like(pred[0], nut_fs))

        def bl_only(background):
            """Hand `background` over inside the boundary layer, cold outside."""
            seed, _ = ws.masked_seed(free, cold.centres, surface,
                                     background=background, free_within=delta,
                                     ramp=args.ramp, u_inf=u_inf, v_inf=v_inf,
                                     nut_freestream=nut_fs)
            return seed

        # Channel splits: the background carries one part of the prediction and
        # freestream for the rest, so `masked_seed` blends exactly that part in.
        nut_only = (free[0], free[1], free[2], pred[3])
        vel_only = (pred[0], pred[1], free[2], free[3])

        results = {
            "nf_bl_proj": run("nf_bl_proj", mesh_initial=bl_only(projected)),
            "nf_bl_nut": run("nf_bl_nut", mesh_initial=bl_only(nut_only)),
            "nf_bl_vel": run("nf_bl_vel", mesh_initial=bl_only(vel_only)),
        }

        # How much of the prediction the round-trip destroys, in the one quantity
        # viscous drag integrates. Reported next to the solve, because the solve
        # is only interpretable against it.
        in_bl = distance <= delta
        def err(a, b, m):
            return float(100 * np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-30))
        truth = (cold.u, cold.v, cold.p, cold.nut)
        damage = {n: err(projected[i], pred[i], in_bl) for i, n in enumerate(ws.FIELDS)}
        against_truth = {n: (err(pred[i], truth[i], in_bl),
                             err(projected[i], truth[i], in_bl))
                         for i, n in enumerate(ws.FIELDS)}

        row = {"case": tag, "re": args.re, "covered_fraction": rep["covered_fraction"],
               "projection": proj_rep, "round_trip_change_pct": damage,
               "bl_error_pct_direct_vs_projected": against_truth}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        checkpoint(rows)

        print("   in the boundary layer, error vs converged (direct | projected): "
              + "  ".join(f"{n}={a:.1f}%|{b:.1f}%"
                          for n, (a, b) in against_truth.items()), flush=True)
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            base = row[f"cold@{k}"]
            bits = []
            for name in NEW_ARMS:
                v = row[f"{name}@{k}"]
                s = (1 - v / base) if (base and v) else None
                bits.append(f"{name}={v}" + (f" ({100 * s:+.0f}%)" if s is not None else ""))
            print(f"   @{k}: cold={base}  " + "  ".join(bits), flush=True)

    print("\nper-threshold mean")
    for t in THRESHOLDS:
        k = f"{t:.0e}"
        bits = []
        for name in NEW_ARMS:
            vals = [1 - r[f"{name}@{k}"] / r[f"cold@{k}"] for r in rows
                    if r.get(f"cold@{k}") and r.get(f"{name}@{k}")]
            bits.append(f"{name} {100 * np.mean(vals):+6.1f}% (n={len(vals)})"
                        if vals else f"{name} --")
        print(f"  @{k}: " + "   ".join(bits))
    checkpoint(rows)
    print(f"\nwrote {os.path.relpath(out_path)}")
    print("\nThe forces are what decides this. Score with:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir} --exclude naca4412")
    return 0


if __name__ == "__main__":
    sys.exit(main())
