"""Does a 128^2-resolution initial guess still warm-start a Re-3e6 RANS solve?

This is the question that decides whether training a surrogate for Paper 2 is
worth doing, and it is answered *without* training one.

At Re 3e6 the boundary layer is about 0.019 chord thick. The NeuroForge grid
spacing on the standard 3-chord crop is 0.0236 chord, so **the entire boundary
layer is sub-cell**: no Cartesian-grid surrogate, however well trained, carries
any near-wall information. Meanwhile the body-fitted O-grid puts its first cell
at 1e-5 chord, and that is where the SIMPLE iterations are actually spent. If a
128^2 initial guess cannot help, no amount of training changes it.

Three arms per case, identical mesh, schemes and budget:

* **cold**        -- uniform freestream.
* **oracle_mesh** -- the case's own converged field at full mesh resolution.
  The control: if this does not collapse the iteration count, the measurement is
  broken and neither other arm means anything.
* **oracle_128**  -- the *same* field, projected onto the 128^2 Cartesian grid
  and interpolated back onto the mesh. Nothing changed but the resolution, so
  the gap between this and ``oracle_mesh`` is exactly the price of the
  surrogate's grid -- an upper bound on any surrogate trained on that grid.

Reading the result:

* ``oracle_128`` ~ ``oracle_mesh``  -> resolution is not the bottleneck. Train
  the surrogate and run the full protocol.
* ``oracle_128`` ~ ``cold``         -> a Cartesian surrogate cannot warm-start
  this regime. The honest reframing is "warm-start the outer field and let the
  solver build the boundary layer", which may still pay and is worth reporting.

Usage
-----
    python scripts/ogrid_resolution_probe.py --re 3e6 --n-iter 800
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
from neuroforge.solver import ogrid as og, openfoam as of

CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--resolution", type=int, default=128,
                    help="the surrogate's Cartesian grid, i.e. what gets degraded to")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "resprobe"))
    ap.add_argument("--out", default=os.path.join("results", "ogrid_resolution_probe.json"))
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def checkpoint(rows, summary=None):
        """Write results after every case, atomically.

        The machine this runs on loses power without warning, and a Re-3e6 case
        is minutes. Writing only at the end would throw away everything finished
        so far; the temp-file rename means a cut mid-write cannot corrupt the
        file either. Solves themselves resume from `runs/` via
        `openfoam.completed_run`, so a restart re-reads rather than re-solves.
        """
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summary": summary or {"status": "in-progress"},
                       "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    rows = []
    for code, aoa in CASES:
        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=args.resolution)
        tag = f"{code}_aoa{aoa:g}"
        print(f"== {tag} (Re {args.re:.0e}) ==", flush=True)

        def run(name, **kw):
            return og.solve_ogrid(
                case, case_dir=os.path.join(args.work_dir, f"{tag}_{name}"),
                n_iter=args.n_iter, timeout=args.timeout, **kw
            )

        cold = run("cold")
        print(f"   cold        floor={cold.residual_floor:.2e} "
              f"exec={cold.execution_time:.0f}s", flush=True)

        # Arm 1: exact, full mesh resolution.
        oracle = run("oracle_mesh", mesh_initial=(cold.u, cold.v, cold.p, cold.nut))

        # Arm 2: the same field, round-tripped through the surrogate's grid.
        degraded = cold.to_grid(case.domain)
        coarse = run("oracle_128", initial=degraded)
        cover = coarse.meta["warm_start"].get("covered_fraction", float("nan"))

        # How much the round-trip actually destroyed, measured on the mesh.
        back = og.OGridResult(
            u=cold.u, v=cold.v, p=cold.p, nut=cold.nut, centres=cold.centres,
            iterations=0, converged=False, wall_time=0, execution_time=0,
            start="x", case_dir="x",
        ).to_grid(case.domain)
        roundtrip_err = float(
            np.abs(np.asarray(back.u) - np.asarray(degraded.u)).max()
        )

        row = {"case": tag, "re": args.re, "covered_fraction": cover,
               "roundtrip_linf": roundtrip_err,
               "cold_floor": cold.residual_floor}
        for t in (1e-2, 1e-3, 1e-4):
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            row[f"oracle_mesh@{k}"] = oracle.iterations_to(t)
            row[f"oracle_128@{k}"] = coarse.iterations_to(t)
        for k in ("cold", "oracle_mesh", "oracle_128"):
            pass
        row["cold_exec_s"] = cold.execution_time
        row["oracle_mesh_exec_s"] = oracle.execution_time
        row["oracle_128_exec_s"] = coarse.execution_time
        rows.append(row)
        checkpoint(rows)

        for t in (1e-2, 1e-3, 1e-4):
            k = f"{t:.0e}"
            print(f"   @{k}: cold={row[f'cold@{k}']}  "
                  f"oracle_mesh={row[f'oracle_mesh@{k}']}  "
                  f"oracle_128={row[f'oracle_128@{k}']}", flush=True)
        print(f"   grid covers {100 * cover:.1f}% of mesh cells", flush=True)

    summary = {"re": args.re, "n_iter": args.n_iter, "resolution": args.resolution,
               "per_threshold": {}}
    for t in (1e-2, 1e-3, 1e-4):
        k = f"{t:.0e}"

        def saving(arm):
            vals = [1.0 - r[f"{arm}@{k}"] / r[f"cold@{k}"]
                    for r in rows if r.get(f"cold@{k}") and r.get(f"{arm}@{k}")]
            return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

        m, mn = saving("oracle_mesh")
        c, cn = saving("oracle_128")
        summary["per_threshold"][k] = {"oracle_mesh_saving": m, "oracle_mesh_n": mn,
                                       "oracle_128_saving": c, "oracle_128_n": cn}
        print(f"\nthreshold {k}: oracle_mesh saves {100 * m:5.1f}% (n={mn}), "
              f"oracle_128 saves {100 * c:5.1f}% (n={cn})")

    checkpoint(rows, summary)
    print(f"\nwrote {args.out}")

    ctrl = summary["per_threshold"]["1e-03"]["oracle_mesh_saving"]
    if not (ctrl > 0.5):
        print("\n!! ORACLE CONTROL FAILED: an exact, full-resolution start does not save "
              ">50% of iterations. Do not read the oracle_128 number as a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
