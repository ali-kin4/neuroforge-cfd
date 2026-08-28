"""Does the delta/h = 2 crossover hold when you vary the *grid* instead of Reynolds?

`scripts/reynolds_crossover.py` measured that a warm start pays while

    delta / h  =  boundary-layer thickness / surrogate cell size   >~  2

but it measured that by varying Reynolds number at a fixed 128^2 grid. Applying
it the other way -- fixing Reynolds and refining the grid -- is a *hypothesis*,
not a result: changing Re changes the flow, changing h changes only how the flow
is represented, and the two are not obviously equivalent.

This tests it. At Re 3e6 the criterion predicts the sign change at

    h = delta / 2 = 0.0094 chord   ->   N ~ 321

which sits inside what the AirfRANS point cloud can support (~421^2). So the
ladder brackets it: 128 (measured to fail), 192, 256, 320, 421.

Per case, ``cold`` and ``oracle_mesh`` do not depend on the surrogate grid at all
-- the C-grid is built from the geometry, not from the Cartesian crop -- so they
are solved once and reused for every rung by `openfoam.completed_run`.

Usage
-----
    python scripts/resolution_ladder.py --re 3e6
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

RESOLUTIONS = (128, 192, 256, 320, 421)
CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0)]
THRESHOLDS = (1e-2, 1e-3, 1e-4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--resolutions", type=int, nargs="*", default=list(RESOLUTIONS))
    ap.add_argument("--only", action="append", metavar="AIRFOIL",
                    help="restrict to this airfoil (repeatable). Split the "
                         "sweep by case, not by resolution: every rung of a "
                         "case shares its cold and oracle_mesh runs, so two "
                         "processes splitting resolutions would write the "
                         "same directories at the same time.")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "resladder"))
    ap.add_argument("--out", default=os.path.join("results", "resolution_ladder.json"))
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
    delta = ws.bl_thickness(args.re)
    print(f"Re {args.re:.0e} | delta {delta:.4f} chord | criterion predicts the sign "
          f"change at N ~ {int(round(3.0 / (delta / 2))) + 1}\n")

    cases = CASES
    if args.only:
        wanted = {a.strip() for a in args.only}
        cases = [c for c in CASES if c[0] in wanted]
        if not cases:
            print("not in CASES: " + ", ".join(sorted(wanted)))
            return 1
        if out_path == os.path.abspath(ap.get_default("out")):
            stem, ext = os.path.splitext(out_path)
            out_path = stem + "_" + "_".join(c for c, _ in cases) + ext
        print(f"running only: {', '.join(c for c, _ in cases)}\n"
              f"checkpointing to {out_path}\n")

    rows = []
    for code, aoa in cases:
        base = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=128)
        tag = f"{code}_aoa{aoa:g}"
        print(f"== {tag} ==", flush=True)

        def run(name, **kw):
            return cg.solve_cgrid(base, case_dir=os.path.join(args.work_dir, f"{tag}_{name}"),
                                  spec=spec, n_iter=args.n_iter, timeout=args.timeout, **kw)

        # Neither of these depends on the surrogate grid, so they are solved once
        # and every rung is compared against the same pair.
        cold = run("cold")
        oracle = run("oracle_mesh", mesh_initial=(cold.u, cold.v, cold.p, cold.nut))
        u_inf, v_inf = of._freestream(base)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(base.fluid.kinematic_viscosity)
        print(f"   cold {cold.iterations_to(1e-3)} it to 1e-3 | "
              f"oracle_mesh {oracle.iterations_to(1e-3)}", flush=True)

        for n in args.resolutions:
            case_n = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                           u_inf=1.0, resolution=n)
            h = 3.0 / (n - 1)
            degraded = cold.to_grid(case_n.domain)
            vals, rep = ws.plain_seed(degraded, case_n.domain, cold.centres,
                                      u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
            try:
                arm = run(f"oracle_{n}", mesh_initial=vals)
            except Exception as exc:
                print(f"   N={n:>4}: FAILED {str(exc)[:80]}", flush=True)
                continue

            row = {"case": tag, "airfoil": code, "aoa": aoa, "re": args.re,
                   "resolution": n, "h": h, "delta": delta, "delta_over_h": delta / h,
                   "covered_fraction": rep["covered_fraction"]}
            for t in THRESHOLDS:
                k = f"{t:.0e}"
                row[f"cold@{k}"] = cold.iterations_to(t)
                row[f"oracle_mesh@{k}"] = oracle.iterations_to(t)
                row[f"arm@{k}"] = arm.iterations_to(t)
            rows.append(row)
            checkpoint(rows)

            k = "1e-03"
            c, a = row[f"cold@{k}"], row[f"arm@{k}"]
            sv = (1 - a / c) if (c and a) else float("nan")
            print(f"   N={n:>4}  d/h={delta / h:>5.2f}  cold={c}  arm={a}  "
                  f"saving={100 * sv:>6.1f}%", flush=True)

    summary = {"re": args.re, "delta": delta, "n_iter": args.n_iter,
               "mesh": "cgrid", "by_resolution": {}}
    print(f"\n{'N':>5} {'d/h':>6} {'saving@1e-3':>12} {'n':>3}")
    for n in args.resolutions:
        sub = [r for r in rows if r["resolution"] == n]
        if not sub:
            continue
        entry = {"delta_over_h": sub[0]["delta_over_h"], "h": sub[0]["h"], "n": len(sub)}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            vals = [1 - r[f"arm@{k}"] / r[f"cold@{k}"]
                    for r in sub if r.get(f"cold@{k}") and r.get(f"arm@{k}")]
            entry[f"arm@{k}"] = float(np.mean(vals)) if vals else None
            entry[f"arm@{k}_n"] = len(vals)
            mv = [1 - r[f"oracle_mesh@{k}"] / r[f"cold@{k}"]
                  for r in sub if r.get(f"cold@{k}") and r.get(f"oracle_mesh@{k}")]
            entry[f"oracle_mesh@{k}"] = float(np.mean(mv)) if mv else None
        summary["by_resolution"][str(n)] = entry
        s = entry["arm@1e-03"]
        print(f"{n:>5} {entry['delta_over_h']:>6.2f} "
              f"{(f'{100 * s:>11.1f}%' if s is not None else '          --')} "
              f"{entry['arm@1e-03_n']:>3}")

    checkpoint(rows, summary)
    print(f"\nwrote {os.path.relpath(out_path)}")

    pts = sorted((v["delta_over_h"], v["arm@1e-03"])
                 for v in summary["by_resolution"].values() if v.get("arm@1e-03") is not None)
    sign = [d for d, s in pts if s > 0]
    if sign and len(sign) < len(pts):
        print(f"\nsign change between delta/h {max(d for d, s in pts if s <= 0):.2f} "
              f"and {min(sign):.2f} — the Reynolds sweep predicted 2.0")
    elif sign:
        print(f"\nevery rung tested pays (lowest delta/h {min(d for d, _ in pts):.2f})")
    else:
        print("\nno rung pays: refining the grid at fixed Reynolds does NOT reproduce "
              "the crossover, so delta/h does not transfer between the two axes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
