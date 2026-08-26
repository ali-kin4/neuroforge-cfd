"""Pilot: does warm-starting simpleFoam save iterations at all, in this framework?

Runs before any body-fitted-mesh work, because that build is only justified if
the answer here is yes. Three arms per case, identical mesh/schemes/budget:

* **cold**      -- uniform freestream (the baseline).
* **oracle**    -- warm-started from the case's *own* converged field. This is a
                   control, not a result: if starting from the answer does not
                   show a large saving, the measurement apparatus is broken and
                   every other number is meaningless.
* **neighbour** -- warm-started from a *different* case's converged field (same
                   airfoil, angle of attack shifted). This is an upper bound on
                   what any surrogate could deliver: a real, physically
                   consistent flow field that is wrong in exactly the way a good
                   surrogate's prediction is wrong. If the neighbour arm shows no
                   saving, no amount of surrogate training will either.

Metric
------
Iterations to drive max(Ux, Uy, p) initial residual below a threshold -- **not**
``residualControl``. Steady SIMPLE stagnates at a nonzero residual floor on this
mesh (measured: Ux 6.2e-4, p 1.0e-3, bit-identical from iteration 500 to 1500 at
Re 1e4), so a convergence flag never fires. Thresholds are chosen per case from
the cold arm's measured floor, so they are reachable by construction.

Usage
-----
    python scripts/openfoam_warmstart_pilot.py --re 1e4 --resolution 128
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
from neuroforge.solver import openfoam as of

AIRFOILS = ["naca0012", "naca2412"]
AOAS = [0.0, 2.0, 4.0, 6.0, 8.0]


def _agreement(a, b) -> float:
    """L-inf velocity difference over the fluid cells shared by two fields."""
    m = (np.asarray(a.mask) > 0.5) & (np.asarray(b.mask) > 0.5)
    du = np.abs(np.asarray(a.u) - np.asarray(b.u))[m]
    dv = np.abs(np.asarray(a.v) - np.asarray(b.v))[m]
    return float(max(du.max(), dv.max())) if du.size else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=1e4)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--n-iter", type=int, default=1500, help="fixed budget, both arms")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "pilot"))
    ap.add_argument("--out", default=os.path.join("results", "openfoam_warmstart_pilot.json"))
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")  # the staircase warning is expected and known

    cases = [
        FlowCase.from_airfoil(airfoil=a, aoa=aoa, reynolds=args.re, u_inf=1.0,
                              resolution=args.resolution)
        for a in AIRFOILS
        for aoa in AOAS
    ]

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def checkpoint(rows, summary=None):
        """Write results after every case, atomically.

        The machine this runs on loses power without warning. Writing only at the
        end would discard every finished case; the temp-file rename means a cut
        mid-write cannot corrupt the file either. The solves themselves resume
        from ``runs/`` via :func:`openfoam.completed_run`, so a restart re-reads
        finished cases rather than re-solving them.
        """
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summary": summary or {"status": "in-progress"},
                       "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    def solve(case, initial, tag):
        return of.solve_case(
            case, initial=initial, n_iter=args.n_iter, timeout=args.timeout,
            case_dir=os.path.join(args.work_dir, f"{case.name}_{tag}"),
        )

    # ---- 1. cold reference for every case ---------------------------------- #
    print(f"== cold references ({len(cases)} cases, Re={args.re:.0e}, "
          f"res={args.resolution}, budget={args.n_iter}) ==", flush=True)
    cold = {}
    for c in cases:
        r = solve(c, None, "cold")
        cold[c.name] = r
        print(f"  {c.name:32s} floor={r.residual_floor:.2e} "
              f"to1e-2={r.iterations_to(1e-2)} to1e-3={r.iterations_to(1e-3)} "
              f"{r.execution_time:.1f}s", flush=True)

    floors = [cold[c.name].residual_floor for c in cases]
    floor = float(np.nanmax(floors))
    # Thresholds must sit above the stagnation floor to be reachable at all.
    ladder = [t for t in (1e-1, 1e-2, 1e-3, 1e-4) if t > 3.0 * floor]
    if not ladder:
        print(f"\nNo threshold above the residual floor {floor:.2e}: nothing is measurable "
              f"on this mesh at this Reynolds number.")
        return 2
    print(f"\nresidual floor over all cold runs = {floor:.2e}; thresholds = {ladder}\n")

    # ---- 2. oracle and neighbour arms -------------------------------------- #
    rows = []
    for k, c in enumerate(cases):
        ref = cold[c.name]
        # Neighbour: same airfoil, the adjacent angle of attack.
        same = [x for x in cases if x.geometry.name == c.geometry.name and x.name != c.name]
        if not same:
            continue
        nb = min(same, key=lambda x: abs(x.bc.aoa_deg - c.bc.aoa_deg))

        oracle = solve(c, ref.field, "oracle")
        neigh = solve(c, cold[nb.name].field, "neighbour")

        row = {
            "case": c.name,
            "neighbour": nb.name,
            "d_aoa": float(nb.bc.aoa_deg - c.bc.aoa_deg),
            "cold_exec_s": ref.execution_time,
            "oracle_exec_s": oracle.execution_time,
            "neighbour_exec_s": neigh.execution_time,
            "residual_floor": ref.residual_floor,
            "agreement_oracle": _agreement(ref.field, oracle.field),
            "agreement_neighbour": _agreement(ref.field, neigh.field),
            # How wrong the neighbour start was, as a stand-in for surrogate error.
            "start_error_neighbour": _agreement(ref.field, cold[nb.name].field),
        }
        for t in ladder:
            key = f"{t:.0e}"
            row[f"cold_it@{key}"] = ref.iterations_to(t)
            row[f"oracle_it@{key}"] = oracle.iterations_to(t)
            row[f"neighbour_it@{key}"] = neigh.iterations_to(t)
        rows.append(row)
        checkpoint(rows)

        t0 = ladder[-1]
        k0 = f"{t0:.0e}"
        print(f"  {c.name:32s} @{k0}: cold={row[f'cold_it@{k0}']} "
              f"oracle={row[f'oracle_it@{k0}']} neigh={row[f'neighbour_it@{k0}']} "
              f"(start err {row['start_error_neighbour']:.3f})", flush=True)

    # ---- 3. summary --------------------------------------------------------- #
    summary = {"re": args.re, "resolution": args.resolution, "n_iter": args.n_iter,
               "residual_floor": floor, "thresholds": ladder, "per_threshold": {}}
    for t in ladder:
        key = f"{t:.0e}"
        def saving(arm):
            vals = [
                1.0 - r[f"{arm}_it@{key}"] / r[f"cold_it@{key}"]
                for r in rows
                if r.get(f"cold_it@{key}") and r.get(f"{arm}_it@{key}")
            ]
            return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

        o_mean, o_n = saving("oracle")
        n_mean, n_n = saving("neighbour")
        summary["per_threshold"][key] = {
            "oracle_mean_saving": o_mean, "oracle_n": o_n,
            "neighbour_mean_saving": n_mean, "neighbour_n": n_n,
        }
        print(f"\nthreshold {key}: oracle saves {100 * o_mean:5.1f}% (n={o_n}), "
              f"neighbour saves {100 * n_mean:5.1f}% (n={n_n})")

    checkpoint(rows, summary)
    print(f"\nwrote {args.out}")

    worst = summary["per_threshold"][f"{ladder[-1]:.0e}"]["oracle_mean_saving"]
    if not (worst > 0.5):
        print("\n!! ORACLE CONTROL FAILED: starting from the exact answer does not save "
              ">50% of iterations. The measurement is not trustworthy; do not read the "
              "neighbour number as a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
