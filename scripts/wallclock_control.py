"""What does a warm start actually cost, in seconds, all of it included?

Every saving in this track is an iteration count, and an iteration count is not
a speed-up. Two things stand between them.

**Cost per iteration is not constant across arms.** A warm start shortens the
inner linear solves as well as the outer loop, so a good seed is cheaper per
iteration and a bad one is dearer. On a contended box the arms read 0.134 to
0.407 s/iteration, which if taken at face value would turn a +35.8% iteration
saving into a 4% *loss*. Measured serially on one case it is 1.14x, not 1.62x,
and the saving survives. That is n = 1 and this script exists to make it n = 5.

**The seed is not free.** Querying the backbone at 31,700 points, measuring wall
distance (O(N.M) and genuinely not free), projecting through a surrogate grid,
and writing the ``0/`` fields all happen before the solver starts. "Milliseconds"
is not an answer to a reviewer; every stage is timed here and charged to the arm
that used it, so the reported saving is end-to-end from a cold machine.

**Nothing else may be running.** Iteration counts are contention-proof and
seconds are not; the script refuses to start if it finds another solver, and
that refusal is the measurement working, not a failure.

Arms: ``cold``, ``oracle_mesh`` (the control), ``cartesian_128`` (the
comparator), ``fitted_bl`` (the oracle recipe) and ``nf_bl`` -- the deployed
recipe, and the only one whose seed cost includes a neural network.

Usage
-----
    python scripts/wallclock_control.py                      # five cases, serial
    python scripts/wallclock_control.py --only naca0012@4
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
from neuroforge.solver import cgrid as cg, openfoam as of, scoring as sc, warmstart as ws
from neuroforge.solver import surrogate_seed as ss

CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0),
         ("naca0012", 0.0), ("naca2415", 5.0)]
DEPTHS = (1e-4, 1e-5, 5e-6, 1e-6)
FORCE_TOLS = (0.01,)          # the readable band; see PLANS.md 3.3
ARMS = ("cold", "oracle_mesh", "cartesian_128", "fitted_bl", "nf_bl")


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


class Clock:
    """Charge each preparation stage to the arms that needed it."""

    def __init__(self):
        self.stages: dict[str, float] = {}

    def time(self, name, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0
        return out

    def total(self, *names) -> float:
        return float(sum(self.stages.get(n, 0.0) for n in names))


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
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "wallclock2"))
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

    checkpoints = [os.path.join(args.ckpt_dir, f"seed{k}.pt") for k in args.seeds]
    if any(not os.path.isfile(p) for p in checkpoints):
        print("missing checkpoint(s): " + ", ".join(checkpoints))
        return 1

    cases = CASES
    if args.only:
        wanted = set()
        for text in args.only:
            code, _, aoa = text.partition("@")
            wanted.add((code.strip(), float(aoa or 0.0)))
        cases = [c for c in CASES if c in wanted] or CASES

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    print(f"{len(cases)} case(s) at Re {args.re:.0e}, {args.n_iter} iterations, "
          f"one arm at a time, nothing else on the machine\n")

    all_rows, all_prep = [], []
    for code, aoa in cases:
        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=128)
        tag = f"{code}_aoa{aoa:g}"
        print(f"== {tag} ==", flush=True)
        clock = Clock()

        def dir_of(name):
            return os.path.abspath(os.path.join(args.work_dir, f"{tag}_{name}"))

        def run(name, **kw):
            t0 = time.perf_counter()
            result = cg.solve_cgrid(case, case_dir=dir_of(name), spec=spec,
                                    n_iter=args.n_iter, timeout=args.timeout, **kw)
            print(f"  {name:>14}: {time.perf_counter() - t0:7.0f} s wall", flush=True)
            return result

        cold = run("cold")
        truth = (cold.u, cold.v, cold.p, cold.nut)
        run("oracle_mesh", mesh_initial=truth)

        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]

        cart_vals, _ = clock.time(
            "rasterise", ws.plain_seed, cold.to_grid(case.domain), case.domain,
            cold.centres, u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        run("cartesian_128", mesh_initial=cart_vals)

        fit_vals, _ = clock.time(
            "project", ws.clustered_seed, truth, cold.centres, surface,
            n_s=args.n_s, n_n=args.n_n, first=args.first, u_inf=u_inf,
            v_inf=v_inf, nut_freestream=nut_fs)
        free = (np.full_like(fit_vals[0], u_inf), np.full_like(fit_vals[0], v_inf),
                np.zeros_like(fit_vals[0]), np.full_like(fit_vals[0], nut_fs))
        bl_vals, _ = clock.time(
            "mask", ws.masked_seed, free, cold.centres, surface,
            background=fit_vals, free_within=delta, ramp=args.ramp,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        run("fitted_bl", mesh_initial=bl_vals)

        # The deployed recipe. Its seed is the only one that costs a network.
        distance = clock.time("wall_distance", ws.wall_distance, cold.centres, surface)
        pred, _ = clock.time(
            "inference", ss.predict_on_mesh, checkpoints, cold.centres,
            surface[:, :2], reynolds=args.re, aoa_deg=aoa, wall_distance=distance,
            max_sdf=args.max_sdf, u_inf=u_inf, nut_freestream=nut_fs)
        nf_vals, _ = clock.time(
            "mask_nf", ws.masked_seed, free, cold.centres, surface,
            background=pred, free_within=delta, ramp=args.ramp, u_inf=u_inf,
            v_inf=v_inf, nut_freestream=nut_fs)
        run("nf_bl", mesh_initial=nf_vals)

        # What each arm must pay before the solver starts.
        prep = {
            "cold": 0.0,
            "oracle_mesh": 0.0,
            "cartesian_128": clock.total("rasterise"),
            "fitted_bl": clock.total("project", "mask"),
            "nf_bl": clock.total("wall_distance", "inference", "mask_nf"),
        }
        print("  seed construction: "
              + "  ".join(f"{k}={v:.2f}s" for k, v in clock.stages.items())
              + f"   -> charged: nf_bl {prep['nf_bl']:.2f}s, "
                f"fitted_bl {prep['fitted_bl']:.2f}s", flush=True)
        all_prep.append({"case": tag, "stages": dict(clock.stages), "charged": prep})

        info, forces = {}, {}
        for name in ARMS:
            with open(os.path.join(dir_of(name), "log.simpleFoam"),
                      encoding="utf-8", errors="replace") as fh:
                info[name] = of.parse_simple_foam_log(fh.read())
            forces[name] = of.read_force_coeffs(dir_of(name))

        finals = {n: float(d["Cd"][-1]) for n, d in forces.items()
                  if d and len(d.get("Cd", []))}
        settled = [n for n in finals if sc.has_settled(forces[n]["Cd"], min(FORCE_TOLS))]
        reference, spread, unsettled = sc.settled_reference(finals, settled)

        targets = ([("residual", f"{t:.0e}", t) for t in DEPTHS]
                   + [("Cd", f"{100 * t:g}%", t) for t in FORCE_TOLS])
        for kind, label, value in targets:
            row = {"case": tag, "kind": kind, "target": label,
                   "reference_spread": spread, "unsettled": unsettled}
            for name in ARMS:
                if kind == "residual":
                    it = of.iterations_to_threshold(info[name]["residuals"], value)
                else:
                    d = forces[name]
                    it = (of.iterations_to_force_band(d["Time"], d["Cd"],
                                                      reference=reference, tol=value)
                          if d and reference else None)
                sec = cost_at(info[name], it)
                row[f"{name}_iterations"] = it
                row[f"{name}_solver_seconds"] = sec
                # End to end: the seed has to be built before the solver runs.
                row[f"{name}_seconds"] = (sec + prep[name]) if sec is not None else None
            all_rows.append(row)

        print(f"  settled Cd spread {100 * spread:.3f}%"
              + (f"; unsettled: {', '.join(unsettled)}" if unsettled else ""),
              flush=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter, "exclusive": not busy,
                       "checkpoints": checkpoints, "arms": list(ARMS),
                       "preparation": all_prep, "rows": all_rows}, fh, indent=2)

    # ---- the table the paper needs ------------------------------------------
    print("\nmean saving against cold, over the cases that reached each target")
    print("seconds are end to end: seed construction + solver\n")
    head = f"{'target':>10} {'n':>3} " + "  ".join(f"{a:>24}" for a in ARMS[1:])
    print(head)
    print("-" * len(head))
    by_target = {}
    for row in all_rows:
        by_target.setdefault((row["kind"], row["target"]), []).append(row)
    for (kind, label), rows in by_target.items():
        cells, n_used = [], 0
        for name in ARMS[1:]:
            it_s, sec_s = [], []
            for r in rows:
                bi, bs = r["cold_iterations"], r["cold_seconds"]
                it, sec = r[f"{name}_iterations"], r[f"{name}_seconds"]
                if bi and bs and it and sec:
                    it_s.append(1 - it / bi)
                    sec_s.append(1 - sec / bs)
            n_used = max(n_used, len(it_s))
            cells.append(f"{100 * np.mean(it_s):+.0f}% it {100 * np.mean(sec_s):+.0f}% s"
                         if it_s else "--")
        print(f"{label:>10} {n_used:>3} " + "  ".join(f"{c:>24}" for c in cells))

    print("\nIterations are contention-proof; seconds are only meaningful because "
          "nothing\nelse ran. Seconds include building the seed -- for `nf_bl` that "
          "is the wall\ndistance, the backbone inference and the mask.")
    print(f"\nwrote {os.path.relpath(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
