"""Paper-2 experiment: does a NeuroForge prediction warm-start OpenFOAM?

Runs ``simpleFoam`` (Spalart-Allmaras, the AirfRANS closure) twice per case on an
identical mesh with identical schemes and convergence criteria:

* **cold** -- uniform freestream initial field (the classical baseline);
* **warm** -- the NeuroForge prediction written into ``0/``.

and reports iterations-to-convergence and wall-clock for each. Because both arms
solve the same boundary-value problem, the converged answer is the same; the only
thing the surrogate can buy is *cost*. That is the claim
``docs/ROADMAP_paper2.md`` targets, and it is what this script measures.

Usage
-----
    python scripts/openfoam_warm_start.py --check
    python scripts/openfoam_warm_start.py --smoke
    python scripts/openfoam_warm_start.py --n-cases 5 --resolution 128 --n-iter 3000

Requires OpenFOAM in WSL2 -- run ``--check`` for the exact install commands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.core.types import FlowCase
from neuroforge.solver import openfoam as of

AIRFOILS = ["naca0012", "naca2412", "naca4412", "naca0015", "naca6409"]
AOAS = [0.0, 4.0, -3.0, 8.0, 2.0]


def check_env() -> int:
    """Report what the host can see. Returns a shell exit code."""
    distros = of.list_wsl_distros()
    print(f"WSL distributions : {distros or '(none found)'}")
    env = of.detect_openfoam(refresh=True)
    if env is None:
        print("OpenFOAM          : NOT FOUND")
        print()
        print("Install it inside WSL2 (needs your sudo password, so run it yourself):")
        print("  wsl -d Ubuntu -- bash -c 'curl -fsSL "
              "https://dl.openfoam.com/add-debian-repo.sh | sudo bash && sudo apt-get update'")
        print("  wsl -d Ubuntu -- bash -c \"apt-cache search 'openfoam.*-default' | tail\"")
        print("  wsl -d Ubuntu -- bash -c 'sudo apt-get install -y <tag-from-above>'")
        return 1
    print(f"OpenFOAM bashrc   : {env.bashrc}")
    print(f"OpenFOAM version  : {env.version}")
    where = "(native linux)" if env.native else (env.distro or "Ubuntu (WSL default)")
    print(f"distro            : {where}")
    print("OK")
    return 0


def build_cases(n: int, resolution: int) -> list[FlowCase]:
    cases = []
    for i in range(n):
        cases.append(
            FlowCase.from_airfoil(
                airfoil=AIRFOILS[i % len(AIRFOILS)],
                aoa=AOAS[i % len(AOAS)],
                reynolds=1.0e6,
                u_inf=1.0,
                resolution=resolution,
            )
        )
    return cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="probe WSL/OpenFOAM and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="one small cold case (res 64, 100 iters) to prove the pipeline runs")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=128,
                    help="grid resolution; below ~128 the body is a coarse staircase")
    ap.add_argument("--n-iter", type=int, default=3000, help="SIMPLE iteration cap")
    ap.add_argument("--out", default=os.path.join("results", "openfoam_warm_start.json"))
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam"))
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument(
        "--agree-tol", type=float, default=1e-3,
        help="max L-inf velocity difference (in u_inf units) between the cold and warm "
             "solutions for a case to count toward the reported saving",
    )
    args = ap.parse_args(argv)

    if args.check:
        return check_env()

    if of.detect_openfoam() is None:
        check_env()
        return 1

    if args.smoke:
        case = FlowCase.from_airfoil(airfoil="naca0012", aoa=0.0, resolution=64)
        res = of.solve_case(
            case,
            case_dir=os.path.join(args.work_dir, "smoke"),
            n_iter=100,
            timeout=args.timeout,
        )
        print(f"smoke: {res.iterations} iters, converged={res.converged}, "
              f"{res.wall_time:.1f}s wall, case in {res.case_dir}")
        print(f"  |u| range: {float(np.abs(res.field.u).max()):.4g}")
        return 0

    from neuroforge.solver.engine import NeuroForgeEngine

    engine = NeuroForgeEngine.pretrained()
    cases = build_cases(args.n_cases, args.resolution)
    rows = []

    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case.name}", flush=True)
        pred = engine.predictor.predict(case)

        cold = of.solve_case(
            case, initial=None, n_iter=args.n_iter, timeout=args.timeout,
            case_dir=os.path.join(args.work_dir, f"{case.name}_cold"),
        )
        warm = of.solve_case(
            case, initial=pred, n_iter=args.n_iter, timeout=args.timeout,
            case_dir=os.path.join(args.work_dir, f"{case.name}_warm"),
        )

        # Both arms must land on the same steady solution; if they do not, the
        # comparison is meaningless and the run is flagged rather than reported.
        fluid = np.asarray(cold.field.mask) > 0.5
        du = float(np.abs(cold.field.u - warm.field.u)[fluid].max())
        dv = float(np.abs(cold.field.v - warm.field.v)[fluid].max())
        agree = max(du, dv)

        row = {
            "case": case.name,
            "resolution": args.resolution,
            "cold_iterations": cold.iterations,
            "warm_iterations": warm.iterations,
            "cold_converged": cold.converged,
            "warm_converged": warm.converged,
            "cold_wall_s": cold.wall_time,
            "warm_wall_s": warm.wall_time,
            "cold_exec_s": cold.execution_time,
            "warm_exec_s": warm.execution_time,
            "iteration_saving": (
                1.0 - warm.iterations / cold.iterations if cold.iterations else float("nan")
            ),
            "solution_agreement_linf": agree,
        }
        rows.append(row)
        # Flag the row inline: a printed "saving" on a case that will be excluded
        # reads as a result to anyone scanning the log.
        ok = cold.converged and warm.converged and agree <= args.agree_tol
        print(
            f"    cold {cold.iterations:5d} it / {cold.execution_time:7.1f}s   "
            f"warm {warm.iterations:5d} it / {warm.execution_time:7.1f}s   "
            f"saving {100 * row['iteration_saving']:5.1f}%   agree {agree:.2e}"
            + ("" if ok else "   [EXCLUDED]"),
            flush=True,
        )

    # Only cases where BOTH arms converged AND landed on the same field may enter
    # the headline mean. `residualControl` tests the initial residual of the
    # current outer iteration against the current field, so a smooth warm start
    # can satisfy it after a couple of iterations without being the solution --
    # which would manufacture a spectacular fake saving. And steady SIMPLE on a
    # shedding case can limit-cycle to different states from different starts.
    # Both failure modes show up as disagreement, so it is enforced, not assumed.
    both = [r for r in rows if r["cold_converged"] and r["warm_converged"]]
    usable = [r for r in both if r["solution_agreement_linf"] <= args.agree_tol]
    usable_ids = {id(r) for r in usable}
    excluded = [r["case"] for r in rows if id(r) not in usable_ids]
    for r in rows:
        if id(r) in usable_ids:
            continue
        why = (
            f"arms disagree by {r['solution_agreement_linf']:.3e} (> {args.agree_tol:g})"
            if r["cold_converged"] and r["warm_converged"]
            else "an arm hit the iteration cap without converging"
        )
        print(f"  ! {r['case']}: {why} -- excluded from the mean", flush=True)

    summary = {
        "n_cases": len(rows),
        "n_both_converged": len(both),
        "n_usable": len(usable),
        "excluded_cases": excluded,
        "agree_tol": args.agree_tol,
        "mean_iteration_saving": (
            float(np.mean([r["iteration_saving"] for r in usable])) if usable else float("nan")
        ),
        "mean_exec_saving": (
            float(np.mean([1.0 - r["warm_exec_s"] / r["cold_exec_s"] for r in usable]))
            if usable else float("nan")
        ),
        "max_solution_disagreement": (
            float(max(r["solution_agreement_linf"] for r in rows)) if rows else float("nan")
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
