"""What does a warm start actually cost, in seconds?

Every saving in this track is an iteration count, and an iteration count is not
the speed-up. A warm start shortens the **inner** linear solves as well as the
outer loop, so cost per iteration is not a constant across arms: on the shared
box the oracle arm ran at 0.134 s/iteration against the cold arm's 0.251 s, and
the wall-fitted arm at 0.407 s. If those ratios are real, the wall-fitted arm's
+35.8% iteration saving at residual 5e-6 becomes a 4% *loss* in wall-clock, and
the oracle arm's +92.4% stays a 96% win.

But those ratios were measured with a dozen solves competing for memory
bandwidth, and the arms did not all run under the same load. Only a strictly
sequential run answers the question. **Nothing else may be running on this
machine**; the script refuses to start if it finds another solver.

One case, four arms, one at a time, cost read at the iteration each arm met each
target rather than at the end. Iteration counts are unaffected by load and are
reported alongside, so this run also cross-checks the parallel sweeps.

Usage
-----
    python scripts/wallclock_control.py --n-iter 6000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.core.types import FlowCase
from neuroforge.solver import cgrid as cg, openfoam as of, warmstart as ws

DEPTHS = (1e-4, 1e-5, 5e-6, 1e-6)
FORCE_TOLS = (0.01, 0.005)
ARMS = ("cold", "oracle_mesh", "cartesian_128", "fitted_256x64")


def other_solvers_running() -> int:
    """Count ``simpleFoam`` processes, which would invalidate the timings."""
    env = of.detect_openfoam()
    if env is None:
        return 0
    try:
        proc = of._run_bash(env, "ps -eo args | grep -c '[s]impleFoam' || true", 60)
    except (OSError, subprocess.SubprocessError):
        return 0
    text = (proc.stdout or "0").strip().splitlines()
    try:
        return int(text[-1])
    except (ValueError, IndexError):
        return 0


def cost_at(info: dict, iteration: int | None) -> float | None:
    """Cumulative solver seconds at ``iteration`` (1-based), or ``None``."""
    if not iteration:
        return None
    elapsed = info.get("elapsed") or []
    if iteration > len(elapsed):
        return None
    value = elapsed[iteration - 1]
    return float(value) if np.isfinite(value) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--airfoil", default="naca0012")
    ap.add_argument("--aoa", type=float, default=4.0)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--n-s", type=int, default=256)
    ap.add_argument("--n-n", type=int, default=64)
    ap.add_argument("--first", type=float, default=2.5e-4)
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "wallclock"))
    ap.add_argument("--out", default=os.path.join("results", "wallclock_control.json"))
    ap.add_argument("--timeout", type=float, default=43200.0)
    ap.add_argument("--force", action="store_true",
                    help="run even though other solves are in flight (the timings "
                         "are then not a wall-clock measurement)")
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    busy = other_solvers_running()
    if busy and not args.force:
        print(f"{busy} simpleFoam process(es) already running. This measurement is\n"
              "only meaningful with the machine to itself -- wait for them, or pass\n"
              "--force and do not quote the seconds.")
        return 1
    warnings.simplefilter("ignore")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    spec = cg.CGridSpec()
    case = FlowCase.from_airfoil(airfoil=args.airfoil, aoa=args.aoa, reynolds=args.re,
                                 u_inf=1.0, resolution=128)
    tag = f"{args.airfoil}_aoa{args.aoa:g}"
    print(f"{tag} at Re {args.re:.0e}, {args.n_iter} iterations, one arm at a time\n")

    def dir_of(name):
        return os.path.abspath(os.path.join(args.work_dir, f"{tag}_{name}"))

    def run(name, **kw):
        t0 = time.perf_counter()
        result = cg.solve_cgrid(case, case_dir=dir_of(name), spec=spec,
                                n_iter=args.n_iter, timeout=args.timeout, **kw)
        print(f"  {name:>14}: {time.perf_counter() - t0:7.0f} s wall", flush=True)
        return result

    cold = run("cold")
    run("oracle_mesh", mesh_initial=(cold.u, cold.v, cold.p, cold.nut))

    u_inf, v_inf = of._freestream(case)
    nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
    inner, nw, ns = cg.inner_curve(args.airfoil, spec)
    surface = inner[nw - 1: nw + ns - 1]

    cart_vals, _ = ws.plain_seed(cold.to_grid(case.domain), case.domain, cold.centres,
                                 u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
    run("cartesian_128", mesh_initial=cart_vals)
    fit_vals, _ = ws.clustered_seed((cold.u, cold.v, cold.p, cold.nut), cold.centres,
                                    surface, n_s=args.n_s, n_n=args.n_n,
                                    first=args.first, u_inf=u_inf, v_inf=v_inf,
                                    nut_freestream=nut_fs)
    run("fitted_256x64", mesh_initial=fit_vals)

    # ---- read cost at the iteration each arm met each target ---------------- #
    info, forces = {}, {}
    for name in ARMS:
        log = os.path.join(dir_of(name), "log.simpleFoam")
        with open(log, encoding="utf-8", errors="replace") as fh:
            info[name] = of.parse_simple_foam_log(fh.read())
        forces[name] = of.read_force_coeffs(dir_of(name))

    finals = [f[-1] for f in (d.get("Cd") for d in forces.values()) if f is not None]
    reference = float(np.median(finals)) if finals else None
    spread = (max(abs(v - reference) / abs(reference) for v in finals)
              if reference else float("nan"))

    rows = []
    targets = ([("residual", f"{t:.0e}", t) for t in DEPTHS]
               + [("Cd", f"{100 * t:g}%", t) for t in FORCE_TOLS])
    for kind, label, value in targets:
        row = {"kind": kind, "target": label}
        for name in ARMS:
            if kind == "residual":
                it = of.iterations_to_threshold(info[name]["residuals"], value)
            else:
                d = forces[name]
                it = (of.iterations_to_force_band(d["Time"], d["Cd"],
                                                  reference=reference, tol=value)
                      if d and reference else None)
            row[f"{name}_iterations"] = it
            row[f"{name}_seconds"] = cost_at(info[name], it)
        rows.append(row)

    print(f"\narms' final Cd spread about the median: {100 * spread:.3f}%")
    head = f"{'target':>12} " + "  ".join(f"{a:>22}" for a in ARMS)
    print("\n" + head)
    print("-" * len(head))
    for row in rows:
        cells = []
        for name in ARMS:
            it, sec = row[f"{name}_iterations"], row[f"{name}_seconds"]
            cells.append(f"{it} it / {sec:.0f} s" if it and sec else "--")
        print(f"{row['target']:>12} " + "  ".join(f"{c:>22}" for c in cells))

    print(f"\n{'saving vs cold':>12} " + "  ".join(f"{a:>22}" for a in ARMS[1:]))
    print("-" * len(head))
    for row in rows:
        base_it, base_s = row["cold_iterations"], row["cold_seconds"]
        cells = []
        for name in ARMS[1:]:
            it, sec = row[f"{name}_iterations"], row[f"{name}_seconds"]
            if it and base_it and sec and base_s:
                cells.append(f"{100 * (1 - it / base_it):+.0f}% it "
                             f"{100 * (1 - sec / base_s):+.0f}% s")
            else:
                cells.append("--")
        print(f"{row['target']:>12} " + "  ".join(f"{c:>22}" for c in cells))

    summary = {
        "case": tag, "re": args.re, "n_iter": args.n_iter,
        "exclusive": not busy, "force_reference_Cd": reference,
        "force_reference_spread": spread,
        "seconds_per_iteration": {
            name: (info[name]["elapsed"][-1] / len(info[name]["elapsed"]))
            for name in ARMS if info[name]["elapsed"]},
        "rows": rows,
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, out_path)
    print(f"\nwrote {os.path.relpath(out_path)}")
    print("\nThe two columns can disagree in sign. Where they do, seconds is the "
          "one\na practitioner spends -- and iterations is the one that transfers "
          "to\nanother machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
