"""Can rebuilding the boundary layer rescue a 128^2 warm start at Re 3e6?

`scripts/ogrid_resolution_probe.py` established the null: the *exact* solution,
round-tripped through the 128^2 Cartesian grid, saves no iterations, because the
boundary layer is thinner than one surrogate cell and the resulting near-wall
state is 3-4x wrong precisely where SIMPLE spends its work.

The measured error profile also says the outer field survives the round trip
intact. So this asks whether keeping that outer field and *rebuilding* the layer
underneath it recovers the saving -- and it asks it without training anything,
by degrading the exact answer and reconstructing from that.

Four arms per case on one C-grid, identical schemes and budget:

* **cold**          -- uniform freestream.
* **oracle_mesh**   -- the case's own converged field at full mesh resolution.
  The control: if this does not collapse the iteration count, nothing else here
  can be read.
* **oracle_128**    -- the same field via the 128^2 grid, used as-is. Reproduces
  the null on this mesh.
* **oracle_128_hybrid** -- the same 128^2 field, with the boundary layer rebuilt
  by `solver.warmstart.hybrid_seed`. The arm under test.

Reading it: `hybrid` near `oracle_mesh` means the fix works and a surrogate is
worth training; `hybrid` near `oracle_128` means the outer field alone carries no
usable head start and the honest move is to reframe to a Reynolds number whose
layer the surrogate's grid resolves.

Usage
-----
    python scripts/hybrid_seed_probe.py --re 3e6 --n-iter 800
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

CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0)]
THRESHOLDS = (1e-2, 1e-3, 1e-4)
ARMS = ("oracle_mesh", "oracle_128", "oracle_128_hybrid")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--resolution", type=int, default=128,
                    help="the surrogate's Cartesian grid, i.e. what gets degraded to")
    ap.add_argument("--blend-to", type=float, default=2.0,
                    help="where the prediction takes over, in multiples of delta")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "hybrid"))
    ap.add_argument("--out", default=os.path.join("results", "hybrid_seed_probe.json"))
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def checkpoint(rows, summary=None):
        """Write after every case, atomically -- this machine loses power."""
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summary": summary or {"status": "in-progress"},
                       "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    spec = cg.CGridSpec()
    rows = []

    for code, aoa in CASES:
        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=args.resolution)
        tag = f"{code}_aoa{aoa:g}"
        print(f"== {tag} (Re {args.re:.0e}, C-grid) ==", flush=True)

        def run(name, **kw):
            return cg.solve_cgrid(case, case_dir=os.path.join(args.work_dir, f"{tag}_{name}"),
                                  spec=spec, n_iter=args.n_iter, timeout=args.timeout, **kw)

        cold = run("cold")
        print(f"   cold   floor={cold.residual_floor:.2e} exec={cold.execution_time:.0f}s",
              flush=True)

        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)

        # Arm 1: exact, at full mesh resolution.
        oracle = run("oracle_mesh", mesh_initial=(cold.u, cold.v, cold.p, cold.nut))

        # The same field, seen through the surrogate's grid.
        degraded = cold.to_grid(case.domain)

        # Arm 2: used as-is.
        plain_vals, plain_rep = ws.plain_seed(
            degraded, case.domain, cold.centres,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        coarse = run("oracle_128", mesh_initial=plain_vals)

        # Arm 3: boundary layer rebuilt underneath it.
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]
        hyb_vals, hyb_rep = ws.hybrid_seed(
            degraded, case.domain, cold.centres, surface,
            reynolds=args.re, u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs,
            blend_to=args.blend_to)
        hybrid = run("oracle_128_hybrid", mesh_initial=hyb_vals)

        row = {"case": tag, "re": args.re, "mesh": "cgrid",
               "cold_floor": cold.residual_floor,
               "covered_fraction": plain_rep["covered_fraction"],
               "delta": hyb_rep["delta"],
               "profiled_fraction": hyb_rep["profiled_fraction"],
               "cold_exec_s": cold.execution_time}
        results = {"oracle_mesh": oracle, "oracle_128": coarse,
                   "oracle_128_hybrid": hybrid}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for arm, res in results.items():
                row[f"{arm}@{k}"] = res.iterations_to(t)
        for arm, res in results.items():
            row[f"{arm}_exec_s"] = res.execution_time
        rows.append(row)
        checkpoint(rows)

        for t in THRESHOLDS:
            k = f"{t:.0e}"
            print(f"   @{k}: cold={row[f'cold@{k}']}  "
                  + "  ".join(f"{a.replace('oracle_', '')}={row[f'{a}@{k}']}" for a in ARMS),
                  flush=True)
        print(f"   layer rebuilt on {100 * hyb_rep['profiled_fraction']:.1f}% of cells "
              f"(delta = {hyb_rep['delta']:.4f} chord)", flush=True)

    summary = {"re": args.re, "n_iter": args.n_iter, "resolution": args.resolution,
               "mesh": "cgrid", "blend_to": args.blend_to, "per_threshold": {}}
    for t in THRESHOLDS:
        k = f"{t:.0e}"
        entry = {}
        for arm in ARMS:
            vals = [1.0 - r[f"{arm}@{k}"] / r[f"cold@{k}"]
                    for r in rows if r.get(f"cold@{k}") and r.get(f"{arm}@{k}")]
            entry[f"{arm}_saving"] = float(np.mean(vals)) if vals else float("nan")
            entry[f"{arm}_n"] = len(vals)
        summary["per_threshold"][k] = entry
        print(f"\nthreshold {k}: " + "   ".join(
            f"{a.replace('oracle_', '')} {100 * entry[f'{a}_saving']:6.1f}% (n={entry[f'{a}_n']})"
            for a in ARMS))

    checkpoint(rows, summary)
    print(f"\nwrote {args.out}")

    ctrl = summary["per_threshold"]["1e-03"]["oracle_mesh_saving"]
    if not (ctrl > 0.5):
        print("\n!! ORACLE CONTROL FAILED: an exact, full-resolution start does not save "
              ">50% of iterations. Do not read the other arms as results.")
        return 2

    plain = summary["per_threshold"]["1e-03"]["oracle_128_saving"]
    hyb = summary["per_threshold"]["1e-03"]["oracle_128_hybrid_saving"]
    print(f"\nverdict @1e-3: plain {100 * plain:.1f}% -> hybrid {100 * hyb:.1f}% "
          f"(control {100 * ctrl:.1f}%)")
    if hyb > plain + 0.15:
        print("  the boundary-layer rebuild recovers a real part of the saving.")
    else:
        print("  the rebuild does not help; the outer field alone carries no head start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
