"""Seed the field the solver is slow at, not the field it is fast at.

Every warm start measured in this repo hands the solver *everything* the
surrogate predicts. On the C-grid at Re 3e6 that is the wrong granularity,
because the quantities in a cold solve do not converge at the same rate.
Iterations to settle within 1% of the converged value, four cases, 6000-iteration
runs:

    quantity                    cold      seeded with the exact field
    viscous drag  Cd_v           ~700                  ~53
    lift          Cl             ~950                    1
    pressure drag Cd_p          ~1850                  1-2

So the **pressure field is what a cold start is slow at**, and the near-wall
velocity gradient is what it is fast at -- and that gradient is also what a
surrogate reconstructs worst and what dominates total drag (Cd_v is 60-84% of Cd
here). Seeding everything therefore buys a large gain on the pressure-dominated
quantities and pays for it on the one the solver was going to get right anyway,
which is exactly the split measured: lift +86%, drag -71%.

If that reading is correct, the fix is not a better surrogate. It is to stop
handing over the near-wall velocity. Five arms per case, same mesh, same budget,
all seeded from the same wall-fitted 256x64 projection so the information content
is identical and only its *use* differs:

* ``cold``           -- uniform freestream. Baseline.
* ``oracle_mesh``    -- the converged field at mesh resolution. Control.
* ``fitted``         -- the whole wall-fitted seed. The measured trade.
* ``fitted_p``       -- pressure only; velocity and nuTilda start cold.
* ``fitted_outer``   -- everything, but velocity reverts to freestream inside
  the boundary layer and ramps back to the prediction by 3 delta.

``cold`` and ``oracle_mesh`` are reused from the representation probe's tree when
one is pointed at, so this costs three solves per case rather than five.

A positive result here is worth more than the representation result on its own:
it is a recipe rather than an observation, it needs no new model, and it applies
to any surrogate anyone already has.

Usage
-----
    python scripts/selective_seed_probe.py --work-dir runs/openfoam/repr3
    python scripts/reanalyse_depth.py --root runs/openfoam/repr3 --per-case
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

CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0),
         ("naca0012", 0.0), ("naca4412", 3.0), ("naca2415", 5.0)]
THRESHOLDS = (1e-3, 1e-4, 1e-5, 5e-6, 1e-6)
FORCE_TOLS = (0.01, 0.005)
NEW_ARMS = ("fitted_p", "fitted_outer")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA",
                    help="run just this case (repeatable), e.g. naca0012@4")
    ap.add_argument("--n-s", type=int, default=256)
    ap.add_argument("--n-n", type=int, default=64)
    ap.add_argument("--first", type=float, default=2.5e-4)
    ap.add_argument("--ramp", type=float, default=3.0,
                    help="where the prediction takes over, in multiples of delta")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "repr3"),
                    help="shares the representation probe's tree, so cold and "
                         "oracle_mesh are reused rather than re-solved")
    ap.add_argument("--out", default=os.path.join("results", "selective_seed.json"))
    ap.add_argument("--timeout", type=float, default=43200.0)
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

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
        # One checkpoint per process; several sharing one path would interleave.
        if out_path == os.path.abspath(ap.get_default("out")):
            stem, ext = os.path.splitext(out_path)
            out_path = f"{stem}_" + "_".join(f"{c}{a:g}" for c, a in cases) + ext
        print(f"running only: {', '.join(f'{c}@{a:g}' for c, a in cases)}\n"
              f"checkpointing to {out_path}\n")

    def checkpoint(rows):
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter, "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    print(f"Re {args.re:.0e} | boundary layer {delta:.4f} chord | velocity handed "
          f"back to the solver inside it, ramped to the prediction by "
          f"{args.ramp:g} delta\n")

    rows = []
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

        # One projection, used three ways -- the arms differ in what they hand
        # over, never in what they know.
        fitted, rep = ws.clustered_seed(
            (cold.u, cold.v, cold.p, cold.nut), cold.centres, surface,
            n_s=args.n_s, n_n=args.n_n, first=args.first,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)

        p_only, rep_p = ws.masked_seed(fitted, cold.centres, surface, fields=("p",),
                                       u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        outer, rep_o = ws.masked_seed(fitted, cold.centres, surface,
                                      free_within=delta, ramp=args.ramp,
                                      u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        results = {"fitted_p": run("fitted_p", mesh_initial=p_only),
                   "fitted_outer": run("fitted_outer", mesh_initial=outer)}

        row = {"case": tag, "re": args.re, "delta": delta,
               "covered_fraction": rep["covered_fraction"],
               "blended_fraction": rep_o["blended_fraction"],
               "seeded_fields": rep_p["fields"]}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        checkpoint(rows)

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
    print("\nScore both metrics with:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir} --per-case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
