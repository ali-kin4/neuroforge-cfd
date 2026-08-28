"""Does the *trained* model reach the bound the oracle set?

Every warm start measured in this track so far uses an oracle -- the converged
answer degraded only by projection. That bounds what any surrogate could achieve
and is deliberately not a claim about a model. Here the deployed NeuroForge point
backbone is queried directly at the C-grid cell centres and handed to the solver,
so the question becomes: how much of the achievable saving survives a real
prediction?

The oracle bound to beat, from `repr3` (five cases, 6000 iterations, control
+88% to +99.9%), and the wall-clock control that says those iterations are real
seconds:

    arm                   residual 5e-6      Cd_v@1%
    oracle_mesh                  +92.4%       +92.4%
    fitted_256x64 (bound)        +31.5%       +37.1%
    fitted_bl     (bound)        +34.3%       +41.7%
    cartesian_128                -72.9%       +10.0%

What the model actually predicts, measured against the converged field on
naca0012@4: **10.3% relative L2 in u**, 24.8% in p, 15.3% in u inside the
boundary layer. That is with `sdf` measured to the nearest wall *segment*; to the
nearest polyline vertex, as it was, the same numbers are 18.9% / 28.1% / 31.4%.
Against its own native AirfRANS clouds the backbone is at 0.8-2.8% in u, so the
remainder is the cost of asking it about a case built outside its data pipeline.

Two arms, both from one prediction, so they differ only in how much of it is
handed over:

* ``nf_mesh`` -- the prediction everywhere it is trustworthy (``max_sdf``),
  freestream beyond. The direct analogue of ``fitted_256x64``.
* ``nf_bl``   -- the prediction inside the boundary layer only. The recipe the
  ablations pointed at, applied to a real model.

``cold`` and ``oracle_mesh`` are reused from the representation probe's tree, so
this costs two solves per case.

Usage
-----
    python scripts/neuroforge_seed_probe.py --only naca0012@4
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
from neuroforge.solver import surrogate_seed as ss

CASES = [("naca0012", 4.0), ("naca2412", 2.0), ("naca0015", 6.0),
         ("naca0012", 0.0), ("naca2415", 5.0)]
THRESHOLDS = (1e-3, 1e-4, 1e-5, 5e-6, 1e-6)
NEW_ARMS = ("nf_mesh", "nf_bl")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA")
    ap.add_argument("--ckpt-dir", default=os.path.join("checkpoints", "v2_transolver"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0],
                    help="backbone seeds to average. More is the deployed "
                         "ensemble; measured on naca0012@4 it does not lower the "
                         "field error, so one is the default.")
    ap.add_argument("--max-sdf", type=float, default=3.5,
                    help="chords from the wall beyond which the prediction is "
                         "replaced by freestream. The training sdf distribution "
                         "reaches 3.5 and the C-grid reaches 20.")
    ap.add_argument("--ramp", type=float, default=3.0)
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "repr3"))
    ap.add_argument("--out", default=os.path.join("results", "neuroforge_seed.json"))
    ap.add_argument("--timeout", type=float, default=43200.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if of.detect_openfoam() is None:
        print("OpenFOAM not found -- run scripts/openfoam_warm_start.py --check")
        return 1
    warnings.simplefilter("ignore")

    checkpoints = [os.path.join(args.ckpt_dir, f"seed{k}.pt") for k in args.seeds]
    missing = [p for p in checkpoints if not os.path.isfile(p)]
    if missing:
        print("missing checkpoint(s): " + ", ".join(missing))
        return 1

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
        print(f"running only: {', '.join(f'{c}@{a:g}' for c, a in cases)}\n"
              f"checkpointing to {out_path}\n")

    tags = {f"{code}_aoa{aoa:g}" for code, aoa in cases}
    busy = [p for p in of.running_solvers(os.path.basename(os.path.normpath(args.work_dir)))
            if any(os.path.basename(p).startswith(t + "_") for t in tags)]
    if busy and not args.force:
        print("already being solved by another process: "
              + ", ".join(os.path.basename(p) for p in busy[:6]))
        return 1

    def checkpoint(rows):
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter,
                       "checkpoints": checkpoints, "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    print(f"Re {args.re:.0e} | {len(checkpoints)} checkpoint(s) | "
          f"trustworthy to {args.max_sdf:g} chords | boundary layer {delta:.4f}\n")

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
        inner, nw, ns = cg.inner_curve(code, spec)
        surface = inner[nw - 1: nw + ns - 1]
        distance = ws.wall_distance(cold.centres, surface)

        pred, rep = ss.predict_on_mesh(
            checkpoints, cold.centres, surface[:, :2], reynolds=args.re, aoa_deg=aoa,
            wall_distance=distance, max_sdf=args.max_sdf, u_inf=u_inf,
            nut_freestream=nut_fs)

        # How wrong the prediction is, against this case's own converged field.
        # A seeding result is only interpretable next to it.
        covered = distance <= args.max_sdf
        in_bl = distance <= delta
        truth = (cold.u, cold.v, cold.p, cold.nut)
        def err(a, b, m):
            return float(100 * np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-30))
        field_err = {n: err(pred[i], truth[i], covered)
                     for i, n in enumerate(ws.FIELDS)}
        field_err["u_in_bl"] = err(pred[0], truth[0], in_bl)

        bl_only, _ = ws.masked_seed(
            (np.full_like(pred[0], u_inf), np.full_like(pred[0], v_inf),
             np.zeros_like(pred[0]), np.full_like(pred[0], nut_fs)),
            cold.centres, surface, background=pred, free_within=delta, ramp=args.ramp,
            u_inf=u_inf, v_inf=v_inf, nut_freestream=nut_fs)

        results = {"nf_mesh": run("nf_mesh", mesh_initial=pred),
                   "nf_bl": run("nf_bl", mesh_initial=bl_only)}

        row = {"case": tag, "re": args.re, "covered_fraction": rep["covered_fraction"],
               "speed_m_s": rep["speed_m_s"], "speed_sigma": rep["speed_sigma"],
               "field_error_pct": field_err}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        checkpoint(rows)

        print("   field error vs converged: "
              + "  ".join(f"{n}={v:.1f}%" for n, v in field_err.items()), flush=True)
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            base = row[f"cold@{k}"]
            bits = []
            for name in NEW_ARMS:
                v = row[f"{name}@{k}"]
                s = (1 - v / base) if (base and v) else None
                bits.append(f"{name}={v}" + (f" ({100 * s:+.0f}%)" if s is not None else ""))
            print(f"   @{k}: cold={base}  " + "  ".join(bits), flush=True)

    print("\nper-threshold mean")
    for t in THRESHOLDS:
        k = f"{t:.0e}"
        bits = []
        for name in NEW_ARMS:
            vals = [1 - r[f"{name}@{k}"] / r[f"cold@{k}"] for r in rows
                    if r.get(f"cold@{k}") and r.get(f"{name}@{k}")]
            bits.append(f"{name} {100 * np.mean(vals):+6.1f}% (n={len(vals)})"
                        if vals else f"{name} --")
        print(f"  @{k}: " + "   ".join(bits))
    checkpoint(rows)
    print(f"\nwrote {os.path.relpath(out_path)}")
    print("\nScore both metrics, and against the oracle bound, with:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir} --exclude naca4412")
    return 0


if __name__ == "__main__":
    sys.exit(main())
