"""Is the wake where this solver's time goes?

The largest speed-up in the warm-start literature -- **26.3x iterations and
16.4x wall-clock**, `arXiv:2501.14699 <https://arxiv.org/abs/2501.14699>`_ --
comes from initialising the far wake. Ours is +34% on drag. On raw speed we lose,
and the honest response is not to reframe but to ask *why* they win.

Their gain comes from a region we deliberately never seed. The backbone's
training ``sdf`` distribution is centred on 0.23 chords, so every seed here is
cut off at 3.5 and the wake is handed back to the solver. On this C-grid the
downstream region is 28.5% of the cells, so the seed is structurally available;
what is unknown is whether it is worth anything.

This answers that with an **oracle**, not a method. ``oracle_wake`` seeds the
converged field downstream of the trailing edge and freestream everywhere else,
which bounds what any wake model could buy on these cases:

* **If the saving is small**, the wake is not where this configuration's time
  goes. 26.3x is then a fact about their geometry, their solver and their cold
  baseline rather than a better method, and the paper can say so with a
  measurement instead of conceding the comparison.
* **If it is large**, the near-body and wake seeds are complementary, and
  composing a boundary-layer surrogate with a wake model is worth building.

Deliberately *not* included yet: an arm that composes the trained model's
boundary layer with an oracle wake. Half-oracle composition measures a bound,
not whether the method composes, and it costs solves that only matter if the
bound above turns out to be large. See PLANS.md Phase B6.

One solve per case; ``cold`` and ``oracle_mesh`` are reused from ``repr3``.

Usage
-----
    python scripts/wake_probe.py --only naca0012@4
    python scripts/reanalyse_depth.py --root runs/openfoam/repr3 --exclude naca4412
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
         ("naca0012", 0.0), ("naca2415", 5.0)]
THRESHOLDS = (1e-3, 1e-4, 1e-5, 5e-6, 1e-6)
NEW_ARMS = ("oracle_wake",)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA")
    ap.add_argument("--x-start", type=float, default=1.0,
                    help="chords from the leading edge; 1.0 is the trailing edge")
    ap.add_argument("--ramp", type=float, default=0.5)
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "repr3"))
    ap.add_argument("--out", default=os.path.join("results", "wake_probe.json"))
    ap.add_argument("--timeout", type=float, default=43200.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cases = CASES
    if args.only:
        wanted = set()
        for text in args.only:
            code, _, aoa = text.partition("@")
            wanted.add((code.strip(), float(aoa or 0.0)))
        cases = [c for c in CASES if c in wanted]
        if not cases:
            print("not in CASES: " + ", ".join(f"{c}@{a:g}" for c, a in sorted(wanted)))
            return 1
        if out_path == os.path.abspath(ap.get_default("out")):
            stem, ext = os.path.splitext(out_path)
            out_path = f"{stem}_" + "_".join(f"{c}{a:g}" for c, a in cases) + ext

    tags = {f"{code}_aoa{aoa:g}" for code, aoa in cases}
    busy = [p for p in of.running_solvers(os.path.basename(os.path.normpath(args.work_dir)))
            if any(os.path.basename(p).startswith(t + "_") for t in tags)]
    if busy and not args.force:
        print("already being solved by another process: "
              + ", ".join(os.path.basename(p) for p in busy[:6]))
        return 1

    spec = cg.CGridSpec()
    print(f"Re {args.re:.0e} | seeding the converged field for x > {args.x_start:g} "
          f"(ramp {args.ramp:g}), freestream elsewhere\n")

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
        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * float(case.fluid.kinematic_viscosity)
        seed, rep = ws.wake_seed((cold.u, cold.v, cold.p, cold.nut), cold.centres,
                                 x_start=args.x_start, ramp=args.ramp,
                                 u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)
        result = run("oracle_wake", mesh_initial=seed)

        row = {"case": tag, "re": args.re, "wake": rep}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            row[f"oracle_wake@{k}"] = result.iterations_to(t)
        rows.append(row)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter,
                       "x_start": args.x_start, "rows": rows}, fh, indent=2)

        print(f"   seeded {100 * rep['seeded_fraction']:.1f}% of cells "
              f"({100 * rep['fully_seeded_fraction']:.1f}% fully)", flush=True)
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            base, v = row[f"cold@{k}"], row[f"oracle_wake@{k}"]
            s = (1 - v / base) if (base and v) else None
            print(f"   @{k}: cold={base}  oracle_wake={v}"
                  + (f" ({100 * s:+.0f}%)" if s is not None else ""), flush=True)

    print("\nper-threshold mean")
    for t in THRESHOLDS:
        k = f"{t:.0e}"
        vals = [1 - r[f"oracle_wake@{k}"] / r[f"cold@{k}"] for r in rows
                if r.get(f"cold@{k}") and r.get(f"oracle_wake@{k}")]
        print(f"  @{k}: " + (f"{100 * np.mean(vals):+6.1f}% (n={len(vals)})"
                             if vals else "--"))
    print(f"\nwrote {os.path.relpath(out_path)}")
    print("\nThe forces decide it:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir} --exclude naca4412")
    return 0


if __name__ == "__main__":
    sys.exit(main())
