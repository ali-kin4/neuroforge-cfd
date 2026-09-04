"""End-to-end seconds on the row the paper actually reports.

``scripts/wallclock_control.py`` times its arms against ``Cd@1%`` and residual
thresholds. Section 4's readability rule rejects every ``Cd`` row on the
thirteen-case corpus, so the paper's headline is ``Cd_v@1%`` -- and the seconds
in section 9 therefore rode a metric the paper itself declares unreadable. A
reviewer is right to call that out.

This re-scores the wall-clock control's *existing* runs on ``Cd_v``. Nothing is
re-solved: the force histories, the per-iteration cost and the measured seed
construction time are all already on disk, and the viscous/pressure split comes
from the ``forces`` function object via
:func:`openfoam.read_force_components`.

Seconds are end to end -- seed construction plus solver -- with construction
charged at what it actually cost on this machine (inference dominates it at
~10 s; a cold start is charged nothing).

Usage
-----
    python scripts/wallclock_cdv.py
    python scripts/wallclock_cdv.py --root runs/openfoam/wallclock2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import openfoam as of, scoring as sc

ARMS = ("cold", "oracle_mesh", "cartesian_128", "fitted_bl", "nf_bl")


def cost_at(info: dict, iterations) -> float | None:
    """Solver seconds to reach ``iterations``, from the run's own timing.

    ``parse_simple_foam_log`` records ``elapsed`` -- cumulative ExecutionTime per
    iteration -- which is what a partial run would actually have cost. Falls back
    to pro-rating the total only if that series is short.
    """
    if not iterations:
        return None
    elapsed = info.get("elapsed")
    if elapsed is not None and len(elapsed) >= iterations:
        return float(elapsed[iterations - 1])
    total = info.get("execution_time")
    n = info.get("iterations")
    if total and n:
        return float(total) * iterations / n
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "wallclock"))
    ap.add_argument("--control",
                    default=os.path.join("results", "wallclock_control.json"),
                    help="for the measured seed-construction cost per arm")
    ap.add_argument("--coeff", default="Cd_v")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join("results", "wallclock_cdv.json"))
    args = ap.parse_args(argv)

    prep_by_case = {}
    if os.path.isfile(args.control):
        with open(args.control, encoding="utf-8") as fh:
            for entry in json.load(fh).get("preparation", []):
                prep_by_case[entry["case"]] = entry["charged"]

    tags = sorted(d[: -len("_cold")] for d in os.listdir(args.root)
                  if d.endswith("_cold"))
    if not tags:
        print(f"no '<case>_cold' directories under {args.root}")
        return 1

    rows = []
    for tag in tags:
        def dir_of(name):
            return os.path.join(args.root, f"{tag}_{name}")

        info, comps = {}, {}
        for name in ARMS:
            log = os.path.join(dir_of(name), "log.simpleFoam")
            if not os.path.isfile(log):
                continue
            with open(log, encoding="utf-8", errors="replace") as fh:
                info[name] = of.parse_simple_foam_log(fh.read())
            comps[name] = of.read_force_components(dir_of(name))
        present = [a for a in ARMS if a in comps and comps[a].get(args.coeff) is not None]
        if "cold" not in present:
            print(f"{tag}: no cold arm with a {args.coeff} history; skipped")
            continue

        finals = {n: float(comps[n][args.coeff][-1]) for n in present}
        settled = [n for n in present
                   if sc.has_settled(comps[n][args.coeff], args.tol)]
        reference, spread, unsettled = sc.settled_reference(finals, settled)
        if reference is None:
            print(f"{tag}: no settled reference on {args.coeff}; skipped")
            continue

        prep = prep_by_case.get(tag, {})
        row = {"case": tag, "coeff": args.coeff, "tol": args.tol,
               "reference_spread": spread, "unsettled": unsettled}
        for name in present:
            d = comps[name]
            it = of.iterations_to_force_band(d["Time"], d[args.coeff],
                                             reference=reference, tol=args.tol)
            sec = cost_at(info[name], it)
            build = float(prep.get(name, 0.0))
            row[f"{name}_iterations"] = it
            row[f"{name}_solver_seconds"] = sec
            row[f"{name}_build_seconds"] = build
            row[f"{name}_seconds"] = (sec + build) if sec is not None else None
        rows.append(row)
        print(f"== {tag} ==  settled {args.coeff} spread {100 * spread:.3f}%"
              + (f"; unsettled: {', '.join(unsettled)}" if unsettled else ""))
        for name in present:
            sec, total = row[f"{name}_solver_seconds"], row[f"{name}_seconds"]
            if sec is None:
                print(f"   {name:>14}: never entered the band inside the budget")
                continue
            print(f"   {name:>14}: {row[f'{name}_iterations']} it  "
                  f"{sec:.1f} s solver  + {row[f'{name}_build_seconds']:.1f} s "
                  f"build  = {total:.1f} s")

    if not rows:
        print("nothing scored")
        return 1

    print(f"\nmean saving against cold on {args.coeff}@{100 * args.tol:g}%, "
          f"{len(rows)} cases, end to end (build + solver)")
    print(f"   {'arm':>14} {'iterations':>12} {'solver s':>10} {'end to end':>12}")
    summary = {}
    for name in ARMS[1:]:
        pairs = [(r, r.get(f"{name}_seconds")) for r in rows
                 if r.get(f"{name}_seconds") and r.get("cold_seconds")]
        if not pairs:
            continue
        it = [1 - r[f"{name}_iterations"] / r["cold_iterations"] for r, _ in pairs
              if r.get(f"{name}_iterations") and r.get("cold_iterations")]
        solver = [1 - r[f"{name}_solver_seconds"] / r["cold_solver_seconds"]
                  for r, _ in pairs]
        end = [1 - s / r["cold_seconds"] for r, s in pairs]
        summary[name] = {"iterations": float(np.mean(it)),
                         "solver_seconds": float(np.mean(solver)),
                         "end_to_end_seconds": float(np.mean(end)),
                         "n": len(pairs),
                         "wins_end_to_end": int(sum(1 for v in end if v > 0))}
        print(f"   {name:>14} {100 * np.mean(it):+11.1f}% "
              f"{100 * np.mean(solver):+9.1f}% {100 * np.mean(end):+11.1f}%  "
              f"({summary[name]['wins_end_to_end']}/{len(pairs)})")

    payload = {"root": args.root, "coeff": args.coeff, "tol": args.tol,
               "rows": rows, "summary": summary}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
