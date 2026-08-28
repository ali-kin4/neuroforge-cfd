"""At what Reynolds number does a 128^2 warm start stop working?

Three experiments have established that surrogate warm-starting fails at Re 3e6
and why: the boundary layer is thinner than one surrogate cell, the exact answer
fails identically (so it is not a training problem), and rebuilding the layer
afterwards does not rescue it. The Re-1e4 pilot, on the other hand, measured a
69.7% saving.

So the useful question is not *whether* it works but *where it stops*, and the
controlling parameter is not Reynolds number itself -- it is

    delta / h    =    boundary-layer thickness / surrogate cell size

with ``delta = 0.37 c / Re^0.2`` and ``h = 3 c / (N - 1)`` on the standard
3-chord crop. That ratio is what decides whether the prediction carries any
near-wall information at all, and it makes the result predictive: it applies to
any Reynolds number and any surrogate resolution, not just the ones measured.

This sweeps Reynolds so that delta/h runs from ~3.9 down to ~0.8, and at each
point runs the same three arms on the same C-grid:

* **cold**        -- uniform freestream.
* **oracle_mesh** -- the case's own converged field at mesh resolution. Control.
* **oracle_128**  -- the same field through the 128^2 grid. The measurement.

Because the arm under test is the *exact* answer degraded only in resolution, the
saving it achieves is an upper bound on any surrogate trained on that grid.

Usage
-----
    python scripts/reynolds_crossover.py --n-iter 800
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

REYNOLDS = (1e3, 1e4, 1e5, 1e6, 3e6)
CASES = [("naca0012", 4.0), ("naca2412", 2.0)]
THRESHOLDS = (1e-2, 1e-3, 1e-4)
ARMS = ("oracle_mesh", "oracle_128")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--reynolds", type=float, nargs="*", default=list(REYNOLDS))
    ap.add_argument("--only", action="append", metavar="AIRFOIL",
                    help="restrict to this airfoil (repeatable). Combined with a "
                         "single --reynolds this gives one process per (airfoil, "
                         "Re) cell, each owning its own case directories, so the "
                         "sweep can be spread across processes without two of "
                         "them writing the same run.")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "crossover"))
    ap.add_argument("--out", default=os.path.join("results", "reynolds_crossover.json"))
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def checkpoint(rows, summary=None):
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summary": summary or {"status": "in-progress"},
                       "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    cases = CASES
    if args.only:
        wanted = {a.strip() for a in args.only}
        cases = [c for c in CASES if c[0] in wanted]
        if not cases:
            print("not in CASES: " + ", ".join(sorted(wanted)))
            return 1
        # Give each process its own checkpoint; several sharing one --out would
        # write the same ".tmp" and os.replace the same destination, leaving the
        # file that is meant to survive a power cut holding one process's rows.
        if out_path == os.path.abspath(ap.get_default("out")):
            stem, ext = os.path.splitext(out_path)
            slug = "_".join([c for c, _ in cases]
                            + [f"re{r:.0e}" for r in args.reynolds])
            out_path = f"{stem}_{slug}{ext}"
        print(f"running only: {', '.join(c for c, _ in cases)}\n"
              f"checkpointing to {out_path}\n")

    spec = cg.CGridSpec()
    h = 3.0 / (args.resolution - 1)
    rows = []
    total = len(args.reynolds) * len(cases)
    done = 0

    for re in args.reynolds:
        delta = ws.bl_thickness(re)
        print(f"\n== Re {re:.0e} | delta {delta:.4f} | delta/h {delta / h:.2f} ==", flush=True)
        for code, aoa in cases:
            case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=re,
                                         u_inf=1.0, resolution=args.resolution)
            tag = f"{code}_aoa{aoa:g}_re{re:.0e}"
            done += 1

            def run(name, **kw):
                return cg.solve_cgrid(
                    case, case_dir=os.path.join(args.work_dir, f"{tag}_{name}"),
                    spec=spec, n_iter=args.n_iter, timeout=args.timeout, **kw)

            try:
                cold = run("cold")
                oracle = run("oracle_mesh",
                             mesh_initial=(cold.u, cold.v, cold.p, cold.nut))
                degraded = cold.to_grid(case.domain)
                u_inf, v_inf = of._freestream(case)
                nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
                vals, rep = ws.plain_seed(degraded, case.domain, cold.centres,
                                          u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
                coarse = run("oracle_128", mesh_initial=vals)
            except Exception as exc:
                print(f"   [{done}/{total}] {tag}: FAILED {str(exc)[:90]}", flush=True)
                continue

            row = {"case": tag, "airfoil": code, "aoa": aoa, "re": re,
                   "delta": delta, "h": h, "delta_over_h": delta / h,
                   "cold_floor": cold.residual_floor,
                   "covered_fraction": rep["covered_fraction"],
                   "cold_exec_s": cold.execution_time}
            for t in THRESHOLDS:
                k = f"{t:.0e}"
                row[f"cold@{k}"] = cold.iterations_to(t)
                row[f"oracle_mesh@{k}"] = oracle.iterations_to(t)
                row[f"oracle_128@{k}"] = coarse.iterations_to(t)
            rows.append(row)
            checkpoint(rows)

            k = "1e-03"
            print(f"   [{done}/{total}] {tag}: cold={row[f'cold@{k}']} "
                  f"mesh={row[f'oracle_mesh@{k}']} 128={row[f'oracle_128@{k}']} "
                  f"(floor {cold.residual_floor:.1e})", flush=True)

    # ---- summary: saving vs delta/h ---------------------------------------- #
    summary = {"resolution": args.resolution, "h": h, "n_iter": args.n_iter,
               "mesh": "cgrid", "by_reynolds": {}}
    print(f"\n{'Re':>9} {'d/h':>6} " +
          "  ".join(f"{a.replace('oracle_', ''):>12}" for a in ARMS))
    for re in args.reynolds:
        sub = [r for r in rows if r["re"] == re]
        if not sub:
            continue
        entry = {"delta_over_h": sub[0]["delta_over_h"], "n": len(sub)}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            for arm in ARMS:
                vals = [1.0 - r[f"{arm}@{k}"] / r[f"cold@{k}"]
                        for r in sub if r.get(f"cold@{k}") and r.get(f"{arm}@{k}")]
                entry[f"{arm}@{k}"] = float(np.mean(vals)) if vals else None
                entry[f"{arm}@{k}_n"] = len(vals)
        summary["by_reynolds"][f"{re:.0e}"] = entry
        k = "1e-03"
        print(f"{re:>9.0e} {entry['delta_over_h']:>6.2f} " + "  ".join(
            (f"{100 * entry[f'{a}@{k}']:>11.1f}%" if entry[f"{a}@{k}"] is not None
             else f"{'--':>12}") for a in ARMS))

    checkpoint(rows, summary)
    print(f"\nwrote {os.path.relpath(out_path)}")

    # Where does the coarse arm stop paying?
    k = "1e-03"
    pts = [(v["delta_over_h"], v[f"oracle_128@{k}"])
           for v in summary["by_reynolds"].values() if v.get(f"oracle_128@{k}") is not None]
    pts.sort()
    good = [d for d, s in pts if s > 0.15]
    if good:
        print(f"\ncrossover @1e-3: a 128^2 start pays while delta/h >= {min(good):.2f} "
              f"(saving > 15%)")
    else:
        print("\ncrossover @1e-3: no sweep point showed a saving above 15%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
