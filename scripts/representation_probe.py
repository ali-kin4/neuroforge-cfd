"""Is it the resolution, or the representation?

Every failure measured so far shares one assumption: the surrogate predicts on a
**uniform Cartesian grid**. `scripts/resolution_ladder.py` showed that adding
points to that grid does not help -- 128 through 421 all lose to a cold start at
Re 3e6 -- because a uniform grid must resolve the smallest scale everywhere, and
the near-wall scale collapses like ``nu / u_tau``.

What was never tested is the same number of output values *placed differently*.

Four arms per case on one C-grid, identical schemes and budget:

* **cold**          -- uniform freestream.
* **oracle_mesh**   -- the case's own converged field at mesh resolution. Control.
* **cartesian_128** -- that field through a 128x128 uniform grid: 16,384 values,
  first station 0.0118 chord from the wall. The measured failure.
* **fitted_256x64** -- that field through a wall-fitted grid: **also 16,384
  values**, first station 2.5e-4 chord. Same capacity, ~94x finer at the wall.

The projection is matched between the two surrogate arms -- nearest neighbour
onto the grid, linear interpolation back -- so the only difference is *where the
points are*.

If the fitted arm pays where the Cartesian one does not, the result is positive
and actionable: surrogate warm-starting works at flight Reynolds, but only on a
wall-fitted output representation. If it fails too, the negative result stands
and hardens, since capacity has been ruled out alongside resolution.

Usage
-----
    python scripts/representation_probe.py --re 3e6
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
# Deep enough to be worth reading only because the solve no longer stalls at
# 1.1e-5 (see openfoam.RELAX_U). The floor is now below 2e-6, so 1e-5 sits about
# six times above it; 1e-2 and 1e-3 describe the opening transient, not the run.
THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5, 5e-6)
ARMS = ("oracle_mesh", "cartesian_128", "fitted_256x64")
# Fraction of the converged coefficient the arm has to reach *and stay inside*.
FORCE_TOLS = (0.01, 0.002)


def force_scores(case_dir: str, reference: dict | None, tol: float) -> dict:
    """Iterations until Cd and Cl settle inside ``tol`` of ``reference``.

    ``reference`` is the converged coefficient for the *case*, shared by every
    arm, so a warm start that stops somewhere slightly different is not scored
    against its own resting place. Returns ``{}`` for a run written before the
    forceCoeffs function object existed.
    """
    data = of.read_force_coeffs(case_dir)
    if not data or "Time" not in data:
        return {}
    out = {}
    for name in ("Cd", "Cl"):
        if name not in data:
            continue
        ref = (reference or {}).get(name)
        out[name] = of.iterations_to_force_band(data["Time"], data[name],
                                                reference=ref, tol=tol)
        out[name + "_final"] = float(data[name][-1])
    return out


def converged_coefficients(case_dir: str) -> dict:
    """Final Cd/Cl of a run, for use as the shared reference of its case."""
    data = of.read_force_coeffs(case_dir)
    return {n: float(data[n][-1]) for n in ("Cd", "Cl") if n in data and len(data[n])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=2500)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA",
                    help="run just this case (repeatable), e.g. naca0012@4. "
                         "Cases are independent, so several processes can share "
                         "one --work-dir; a later full run reuses them all off "
                         "disk to aggregate.")
    ap.add_argument("--n-s", type=int, default=256)
    ap.add_argument("--n-n", type=int, default=64)
    ap.add_argument("--first", type=float, default=2.5e-4)
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "repr"))
    ap.add_argument("--out", default=os.path.join("results", "representation_probe.json"))
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

    spec = cg.CGridSpec()
    budget = args.n_s * args.n_n
    print(f"Re {args.re:.0e} · equal budget: 128² = {128 * 128:,} values vs "
          f"{args.n_s}x{args.n_n} = {budget:,} values")
    print(f"first station: {3.0 / 127 / 2:.5f} chord (uniform) vs "
          f"{args.first / 2:.6f} chord (fitted)\n")

    cases = CASES
    if args.only:
        wanted = set()
        for spec_text in args.only:
            code, _, aoa_text = spec_text.partition("@")
            wanted.add((code.strip(), float(aoa_text or 0.0)))
        cases = [c for c in CASES if c in wanted]
        missing = wanted - set(cases)
        if missing:
            print("not in CASES: " + ", ".join(f"{c}@{a:g}" for c, a in sorted(missing)))
            return 1
        print("running only: " + ", ".join(f"{c}@{a:g}" for c, a in cases) + "\n")

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
        oracle = run("oracle_mesh", mesh_initial=(cold.u, cold.v, cold.p, cold.nut))
        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]

        # Arm A: the uniform Cartesian grid (the known failure).
        cart_vals, cart_rep = ws.plain_seed(
            cold.to_grid(case.domain), case.domain, cold.centres,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        cart = run("cartesian_128", mesh_initial=cart_vals)

        # Arm B: the same number of values, wall-fitted.
        fit_vals, fit_rep = ws.clustered_seed(
            (cold.u, cold.v, cold.p, cold.nut), cold.centres, surface,
            n_s=args.n_s, n_n=args.n_n, first=args.first,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        fit = run("fitted_256x64", mesh_initial=fit_vals)

        res = {"oracle_mesh": oracle, "cartesian_128": cart, "fitted_256x64": fit}
        row = {"case": tag, "re": args.re, "budget": budget,
               "cartesian_covered": cart_rep["covered_fraction"],
               "fitted_covered": fit_rep["covered_fraction"],
               "cold_floor": cold.residual_floor}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for a, r in res.items():
                row[f"{a}@{k}"] = r.iterations_to(t)

        # Force convergence, the metric that does not sit on the residual floor.
        # The reference is the oracle arm's converged coefficient -- the arm that
        # started from the answer and is therefore the most settled -- and every
        # arm of this case is scored against that same number.
        def dir_of(name):
            return os.path.join(args.work_dir, f"{tag}_{name}")

        ref = converged_coefficients(dir_of("oracle_mesh")) or \
            converged_coefficients(dir_of("cold"))
        row["force_reference"] = ref
        for tol in FORCE_TOLS:
            for name in ["cold"] + list(ARMS):
                for coeff, value in force_scores(dir_of(name), ref, tol).items():
                    row[f"{name}.{coeff}@{tol:g}"] = value
        rows.append(row)
        checkpoint(rows)

        for t in THRESHOLDS:
            k = f"{t:.0e}"
            c = row[f"cold@{k}"]
            bits = []
            for a in ARMS:
                v = row[f"{a}@{k}"]
                sv = (1 - v / c) if (c and v) else None
                bits.append(f"{a.split('_')[0]}={v}"
                            + (f" ({100 * sv:+.0f}%)" if sv is not None else ""))
            print(f"   @{k}: cold={c}  " + "  ".join(bits), flush=True)
        for tol in FORCE_TOLS:
            for coeff in ("Cd", "Cl"):
                c = row.get(f"cold.{coeff}@{tol:g}")
                if not c:
                    continue
                bits = []
                for a in ARMS:
                    v = row.get(f"{a}.{coeff}@{tol:g}")
                    sv = (1 - v / c) if v else None
                    bits.append(f"{a.split('_')[0]}={v}"
                                + (f" ({100 * sv:+.0f}%)" if sv is not None else ""))
                print(f"   {coeff} within {100 * tol:g}%: cold={c}  "
                      + "  ".join(bits), flush=True)

    summary = {"re": args.re, "n_iter": args.n_iter, "budget": budget,
               "mesh": "cgrid", "per_threshold": {}}
    for t in THRESHOLDS:
        k = f"{t:.0e}"
        entry = {}
        for a in ARMS:
            vals = [1 - r[f"{a}@{k}"] / r[f"cold@{k}"]
                    for r in rows if r.get(f"cold@{k}") and r.get(f"{a}@{k}")]
            entry[f"{a}_saving"] = float(np.mean(vals)) if vals else float("nan")
            entry[f"{a}_n"] = len(vals)
        summary["per_threshold"][k] = entry
        print(f"\nthreshold {k}: " + "   ".join(
            f"{a} {100 * entry[f'{a}_saving']:+6.1f}% (n={entry[f'{a}_n']})" for a in ARMS))

    summary["per_force"] = {}
    for tol in FORCE_TOLS:
        for coeff in ("Cd", "Cl"):
            k = f"{coeff}@{tol:g}"
            entry = {}
            for a in ARMS:
                vals = [1 - r[f"{a}.{k}"] / r[f"cold.{k}"] for r in rows
                        if r.get(f"cold.{k}") and r.get(f"{a}.{k}")]
                entry[f"{a}_saving"] = float(np.mean(vals)) if vals else float("nan")
                entry[f"{a}_n"] = len(vals)
                entry[f"{a}_spread"] = ([float(min(vals)), float(max(vals))]
                                        if vals else None)
            summary["per_force"][k] = entry
            print(f"\nforce {k}: " + "   ".join(
                f"{a} {100 * entry[f'{a}_saving']:+6.1f}% (n={entry[f'{a}_n']})"
                for a in ARMS))

    checkpoint(rows, summary)
    print(f"\nwrote {args.out}")

    # The verdict reads the force metric when it is available, because a residual
    # threshold only measures a convergence rate while the residual is falling.
    e = summary["per_force"].get("Cd@0.002") or summary["per_threshold"]["1e-03"]
    ctrl, cart_s, fit_s = (e["oracle_mesh_saving"], e["cartesian_128_saving"],
                           e["fitted_256x64_saving"])
    if not (ctrl > 0.5):
        print("\n!! ORACLE CONTROL FAILED - do not read the other arms.")
        return 2
    print(f"\nverdict (control {100 * ctrl:.0f}%): "
          f"uniform {100 * cart_s:+.1f}%  ->  wall-fitted {100 * fit_s:+.1f}%")
    if fit_s > 0.15:
        print("  POSITIVE: at equal budget, a wall-fitted output representation "
              "warm-starts where a uniform grid cannot.")
    elif fit_s > cart_s + 0.15:
        print("  Partial: the representation helps materially but does not beat cold.")
    else:
        print("  Negative: representation does not rescue it either; capacity and "
              "resolution are both ruled out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
