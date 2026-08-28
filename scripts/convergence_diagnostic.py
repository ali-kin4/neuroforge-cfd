"""Why does the C-grid solve stagnate at a residual of ~1e-5?

Every warm-start saving in this repo is an *iteration count to a threshold*, and
the thresholds that produced the positive result (3e-5, 2e-5) sit barely above
where ``simpleFoam`` stops making progress. That is not a measurement, it is a
reading of where a flat curve happens to cross a line: on `runs/openfoam/repr`
the fitted arm scores +15%, +31%, +13% at 3e-5, 2e-5, 1.5e-5 -- the same arm,
three adjacent thresholds.

The floor is not a budget problem. The **oracle** arm, seeded with the converged
field itself, goes 3.1e-4 -> 1.2e-5 by iteration 100 and then sits at 1.0e-5
for the next 700 iterations. A longer run cannot dig below a level the exact
answer already sits on; something in the discretisation or the linear solves is
holding it up. This script finds out which, on one case, cheaply.

Variants (all on the same mesh unless noted)::

    base          as shipped -- p relTol 0.01, U/nuTilda relTol 0.1, relax 0.9
    tight         inner linear solves tightened: p 1e-3, U/nuTilda 1e-2
    relax         under-relaxation lowered: U and nuTilda 0.7
    tight_relax   both
    upwind        first-order div(phi,U): tests limiter cycling in linearUpwind
    long          base, but 4000 iterations -- decaying slowly, or truly floored?

Read the printed tail slope. A variant that reaches 1e-6 *with the slope still
negative* has removed the floor; one that lands on 1e-5 again has not.

Usage
-----
    python scripts/convergence_diagnostic.py --variant tight --n-iter 1500
    python scripts/convergence_diagnostic.py --summarise
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.core.types import FlowCase
from neuroforge.solver import cgrid as cg, openfoam as of

DEPTHS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

VARIANTS = {
    "base": {},
    "tight": {"p_reltol": 1e-3, "u_reltol": 1e-2},
    "relax": {"relax": 0.7},
    "tight_relax": {"p_reltol": 1e-3, "u_reltol": 1e-2, "relax": 0.7},
    "upwind": {"upwind": True},
    "long": {},
}


def fv_solution(*, p_reltol: float = 0.01, u_reltol: float = 0.1,
                relax: float = 0.9, n_non_orth: int = 2) -> str:
    """``system/fvSolution`` with the knobs this diagnostic varies exposed."""
    return (
        of._header("dictionary", "fvSolution", "system")
        + "solvers\n{\n"
        + "    p\n    {\n        solver          GAMG;\n"
        + "        smoother        GaussSeidel;\n"
        + f"        tolerance       1e-09;\n        relTol          {of._num(p_reltol)};\n    }}\n\n"
        + '    "(U|nuTilda)"\n    {\n        solver          smoothSolver;\n'
        + "        smoother        symGaussSeidel;\n"
        + f"        tolerance       1e-09;\n        relTol          {of._num(u_reltol)};\n    }}\n}}\n\n"
        + "SIMPLE\n{\n"
        + f"    nNonOrthogonalCorrectors {int(n_non_orth)};\n"
        + "    consistent      yes;\n"
        + "    residualControl\n    {\n"
        + "        p               1e-08;\n        U               1e-09;\n"
        + "        nuTilda         1e-09;\n    }\n}\n\n"
        + "relaxationFactors\n{\n    equations\n    {\n"
        + f"        U               {of._num(relax)};\n"
        + f"        nuTilda         {of._num(relax)};\n    }}\n}}\n"
    )


def tail_slope(history: np.ndarray, window: int = 200) -> float:
    """Decades per 100 iterations over the last ``window`` iterations.

    Negative means still converging; ~0 means floored. Reported rather than
    thresholded, because "has it stopped" is the whole question here.
    """
    tail = history[-window:]
    tail = tail[np.isfinite(tail) & (tail > 0)]
    if tail.size < 20:
        return float("nan")
    x = np.arange(tail.size, dtype=float)
    return float(np.polyfit(x, np.log10(tail), 1)[0] * 100.0)


def combined(residuals: dict, fields=("Ux", "Uy", "p")) -> np.ndarray:
    present = [f for f in fields if residuals.get(f)]
    n = min(len(residuals[f]) for f in present)
    return np.max(np.stack([np.asarray(residuals[f][:n], float) for f in present]), axis=0)


def report(name: str, case_dir: str) -> dict:
    log = os.path.join(case_dir, "log.simpleFoam")
    if not os.path.isfile(log):
        return {"variant": name, "status": "missing"}
    with open(log, encoding="utf-8", errors="replace") as fh:
        info = of.parse_simple_foam_log(fh.read())
    res = info["residuals"]
    if not res:
        return {"variant": name, "status": "no residuals"}
    m = combined(res)
    row = {
        "variant": name,
        "iterations": info["iterations"],
        "converged": info["converged"],
        "final": float(m[-1]),
        "min": float(np.nanmin(m)),
        "slope_per_100": tail_slope(m),
        "execution_time": info["execution_time"],
    }
    for f in ("Ux", "Uy", "p", "nuTilda"):
        if res.get(f):
            row[f"final_{f}"] = float(res[f][-1])
    for t in DEPTHS:
        row[f"to_{t:.0e}"] = of.iterations_to_threshold(res, t)
    return row


def run(name: str, args) -> dict:
    opts = dict(VARIANTS[name])
    upwind = opts.pop("upwind", False)
    n_iter = args.n_iter if name != "long" else max(args.n_iter, 4000)
    case = FlowCase.from_airfoil(airfoil=args.airfoil, aoa=args.aoa,
                                 reynolds=args.re, u_inf=1.0, resolution=128)
    case_dir = os.path.abspath(os.path.join(args.work_dir, name))

    done = of.completed_run(case_dir, n_iter=n_iter, start="cold")
    if done is not None:
        print(f"[{name}] reusing finished run ({done['iterations']} iterations)")
        return report(name, case_dir)

    cg.write_cgrid_case(case, case_dir, spec=cg.CGridSpec(), n_iter=n_iter)
    of._write(os.path.join(case_dir, "system", "fvSolution"), fv_solution(**opts))
    if upwind:
        schemes = of._header("dictionary", "fvSchemes", "system") + of._FV_SCHEMES.replace(
            "div(phi,U)      bounded Gauss linearUpwind grad(U);",
            "div(phi,U)      bounded Gauss upwind;")
        of._write(os.path.join(case_dir, "system", "fvSchemes"), schemes)

    print(f"[{name}] blockMesh", flush=True)
    of.run_openfoam("blockMesh", case_dir, timeout=args.timeout, log_name="log.blockMesh")
    print(f"[{name}] simpleFoam, {n_iter} iterations", flush=True)
    t0 = time.perf_counter()
    of.run_openfoam("simpleFoam", case_dir, timeout=args.timeout, log_name="log.simpleFoam")
    print(f"[{name}] done in {time.perf_counter() - t0:.0f} s", flush=True)
    return report(name, case_dir)


def summarise(args) -> int:
    rows = [report(n, os.path.join(args.work_dir, n)) for n in VARIANTS]
    rows = [r for r in rows if r.get("status") is None]
    if not rows:
        print(f"nothing finished under {args.work_dir}")
        return 1
    head = (f"{'variant':>12} {'iters':>6} {'final':>10} {'min':>10} "
            f"{'slope/100':>10} " + " ".join(f"{f'to {t:.0e}':>9}" for t in DEPTHS))
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = " ".join(f"{(r[f'to_{t:.0e}'] or '--'):>9}" for t in DEPTHS)
        print(f"{r['variant']:>12} {r['iterations']:>6} {r['final']:>10.2e} "
              f"{r['min']:>10.2e} {r['slope_per_100']:>10.3f} {cells}")
    print("\nper-field final initial residual")
    for r in rows:
        bits = " ".join(f"{f}={r.get(f'final_{f}', float('nan')):.2e}"
                        for f in ("Ux", "Uy", "p", "nuTilda"))
        print(f"{r['variant']:>12}  {bits}")
    print("\nslope/100 is decades per 100 iterations over the last 200; ~0 means floored.")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"airfoil": args.airfoil, "aoa": args.aoa, "re": args.re,
                   "rows": rows}, fh, indent=2)
    os.replace(tmp, out)
    print(f"\nwrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS),
                    help="run this variant (repeatable); default runs all")
    ap.add_argument("--summarise", action="store_true",
                    help="read finished runs off disk and print the table")
    ap.add_argument("--airfoil", default="naca0012")
    ap.add_argument("--aoa", type=float, default=4.0)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=1500)
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "convdiag"))
    ap.add_argument("--out", default=os.path.join("results", "convergence_diagnostic.json"))
    ap.add_argument("--timeout", type=float, default=14400.0)
    args = ap.parse_args(argv)

    if args.summarise:
        return summarise(args)
    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    for name in args.variant or sorted(VARIANTS):
        row = run(name, args)
        print(f"[{name}] final={row.get('final', float('nan')):.2e} "
              f"slope/100={row.get('slope_per_100', float('nan')):.3f}", flush=True)
    return summarise(args)


if __name__ == "__main__":
    sys.exit(main())
