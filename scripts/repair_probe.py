"""If the damage is known in closed form, divide it back out.

Section 6 says a projection overestimates the first-cell wall gradient by
``u+(y1+)/u+(yc+)``, and that expression contains no fitted parameter. A factor
that is known can be removed. This script tests whether removing it turns a seed
that costs the solve into one that accelerates it.

**The repair.** Every mesh cell nearer the wall than the representation's first
station holds that station's velocity, because the representation stores nothing
in between. Invert the law of the wall at the station to recover ``u_tau`` --
this is the standard wall-function inversion, using **only what the
representation already carries** -- then re-evaluate the profile at each cell's
own wall distance, and rebuild ``nut`` as a damped mixing length. No retraining,
no extra output values, no access to the converged answer.

**Why it matters more than the criterion alone.** Most surrogates in this field
emit a raster or a graded grid, and §5.2 says such a seed is worse than starting
cold. If the criterion's only advice is "be mesh-native", it asks the field to
rebuild its models. If the damage can instead be repaired after the fact, every
existing grid-based surrogate becomes usable as a warm start, and the criterion
becomes a design *rule* rather than a veto.

**What is already known before any solve here.** Offline, against three
converged cases, the repair moves the projected seed's first-cell wall-gradient
error from **1803-1976% to 37-52%**, and the overestimate from **24.6-35.1x to
1.16-1.43x**. The mesh-native arm ``nf_bl`` sits at 53.7%. So on the quantity
viscous drag integrates, a repaired projection is already as good as a
mesh-native prediction. This script asks the only question that remains: does
the solver agree?

**It can fail, and the failure would be informative.** The law of the wall is a
statement about equilibrium boundary layers. These are airfoils with pressure
gradients, and the repair assumes a profile shape rather than measuring one. If
the gradient is restored and the solve is *not* accelerated, then the first-cell
gradient is not sufficient for a good seed and the paper says so -- §5.3 already
shows the criterion is necessary but not sufficient.

Arms, all boundary-layer-masked identically so region is held fixed:

* ``nf_bl``       -- the network at the solver's cell centres. The reference.
* ``nf_proj``     -- the same prediction through a 256x64 grid at first = 2.5e-4.
* ``nf_proj_fix`` -- that projection, repaired. **The arm this script exists for.**
* ``or_proj``     -- the *exact converged field* through the same grid.
* ``or_proj_fix`` -- that, repaired. The version that cannot be explained away
  by the network being inaccurate.

Fresh work directory, its own ``cold`` and ``oracle_mesh`` control.

Usage
-----
    python scripts/repair_probe.py --only naca0012@4
    python scripts/reanalyse_depth.py --root runs/openfoam/repair
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
from neuroforge.solver import placement as pl, surrogate_seed as ss

CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0),
         ("naca0012", 0.0), ("naca2415", 5.0)]
THRESHOLDS = (1e-3, 1e-4, 1e-5, 5e-6, 1e-6)
NEW_ARMS = ("oracle_mesh", "nf_bl", "nf_proj", "nf_proj_fix", "or_proj", "or_proj_fix")


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
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "repair"))
    ap.add_argument("--out", default=os.path.join("results", "repair.json"))
    ap.add_argument("--timeout", type=float, default=43200.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    checkpoints = [os.path.join(args.ckpt_dir, f"seed{k}.pt") for k in args.seeds]
    if [p for p in checkpoints if not os.path.isfile(p)]:
        print("missing checkpoint(s)")
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
            print("not in CASES")
            return 1
        stem, ext = os.path.splitext(out_path)
        out_path = f"{stem}_" + "_".join(f"{c}{a:g}" for c, a in cases) + ext

    tags = {f"{code}_aoa{aoa:g}" for code, aoa in cases}
    busy = [p for p in of.running_solvers(os.path.basename(os.path.normpath(args.work_dir)))
            if any(os.path.basename(p).startswith(t + "_") for t in tags)]
    if busy and not args.force:
        print("already being solved: " + ", ".join(os.path.basename(p) for p in busy[:6]))
        return 1

    def checkpoint(rows):
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter, "first": args.first,
                       "n_s": args.n_s, "n_n": args.n_n,
                       "checkpoints": checkpoints, "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    nu = 1.0 / args.re
    print(f"Re {args.re:.0e} | boundary layer {delta:.4f} | "
          f"projection {args.n_s}x{args.n_n} from {args.first:.1e}\n", flush=True)

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

        pred, rep = ss.predict_on_mesh(
            checkpoints, cold.centres, surface[:, :2], reynolds=args.re, aoa_deg=aoa,
            wall_distance=distance, max_sdf=args.max_sdf, u_inf=u_inf,
            nut_freestream=nut_fs)

        free = (np.full_like(pred[0], u_inf), np.full_like(pred[0], v_inf),
                np.zeros_like(pred[0]), np.full_like(pred[0], nut_fs))

        def bl_only(background):
            seed, _ = ws.masked_seed(free, cold.centres, surface, background=background,
                                     free_within=delta, ramp=args.ramp, u_inf=u_inf,
                                     v_inf=v_inf, nut_freestream=nut_fs)
            return seed

        seeds = {"oracle_mesh": truth, "nf_bl": bl_only(pred)}
        repairs = {}
        for family, values in (("nf", pred), ("or", truth)):
            projected, _ = ws.clustered_seed(
                values, cold.centres, surface, n_s=args.n_s, n_n=args.n_n,
                first=args.first, u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
            repaired, report = pl.wall_law_repair(
                projected, distance, first_station=args.first, nu=nu)
            repairs[family] = report
            seeds[f"{family}_proj"] = bl_only(projected)
            seeds[f"{family}_proj_fix"] = bl_only(repaired)

        results = {name: run(name, mesh_initial=seed) for name, seed in seeds.items()}

        row = {"case": tag, "re": args.re, "covered_fraction": rep["covered_fraction"],
               "repair": repairs}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        checkpoint(rows)

        print(f"   repaired {repairs['nf']['repaired_cells']} cells "
              f"({100 * repairs['nf']['repaired_fraction']:.1f}%), "
              f"u_tau median {repairs['nf']['u_tau_median']:.4f}", flush=True)
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            base = row[f"cold@{k}"]
            bits = []
            for name in NEW_ARMS:
                v = row.get(f"{name}@{k}")
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
    print(f"\nScore with:\n  python scripts/reanalyse_depth.py --root {args.work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
