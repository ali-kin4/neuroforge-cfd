"""Does the recipe hold where it is most likely to break?

Every case measured in this track so far is attached flow on a thin NACA section
at 0-6 degrees. That is five airfoils, and no error bar turns five airfoils into
a study. The objection a reviewer raises first is not "widen the confidence
interval", it is **"you have only shown this where the solver was easy"**.

So this widens the corpus along the axis that can actually falsify the recipe.
"Seed the boundary layer and let the solver do the outer field" rests on the
outer field being something a cold solve gets quickly. That is true of attached
flow. As incidence rises the wake stops being quick, the separated shear layer
becomes the slow structure, and the recipe has no reason left to work. If it
survives to 10-12 degrees the paper can claim a range; if it degrades, *where*
it degrades is the criterion the paper reports, which is a better result than an
unqualified claim.

Four arms per case, the deployed set:

* ``cold``          -- the baseline every saving is measured against.
* ``oracle_mesh``   -- the converged field itself. **The control.** If it does
  not pass on a case, nothing else on that case may be read, and the case is
  reported with its residual floor and its arms' Cd spread rather than dropped
  silently (``naca4412@3`` is the precedent: no unique steady fixed point).
* ``nf_bl``         -- the recipe. Trained backbone, queried at the cell centres,
  handed over inside the boundary layer only.
* ``cartesian_128`` -- the comparator at equal output budget, and the arm the
  representation claim is *against*.
* ``oracle_bl``     -- the exact field, mesh-native, handed over inside the
  boundary layer only. Against ``oracle_mesh`` it prices the *region*
  restriction; against ``nf_bl`` it prices *accuracy* with everything else held
  fixed (same mask, same channels, same mesh-native delivery).
* ``or_proj_coarse``-- the same exact field, same mask, sent through a 256x64
  body-fitted grid (16,384 values, first station 2.5e-4). Against ``oracle_bl``
  it prices the *representation* with region, channels and accuracy all fixed --
  and it is the wall-fitted projected arm the five-case tree has and the corpus
  did not, which is what stopped the representation claim from being tested at
  the corpus's statistical power.

Those five arms plus ``cold`` make every step of the decomposition a
one-variable contrast on the one row this corpus can read (``Cd_v@1%``).

Ten new cases at Re 3e6: an incidence ladder to 12 degrees on two sections, two
thicker/cambered families the model has not been asked about here, and two
fillers in the attached regime so the ladder has a foot.

Usage
-----
    python scripts/corpus_probe.py --only naca0012@8          # one process per case
    python scripts/corpus_probe.py --re 1e6 --work-dir runs/openfoam/corpus_re1e6
    python scripts/reanalyse_depth.py --root runs/openfoam/corpus --per-case
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

# The incidence ladder is the point; the rest give it a foot and a second family.
#
# Deliberately over-provisioned in the attached regime. Steady RANS has no unique
# fixed point once separation becomes unsteady -- `naca4412@3` had to be excluded
# for exactly that -- so some of the high-incidence cases are expected to fail
# their control. With five cases already measured, the thirteen here leave the
# corpus above n = 12 even if every case past 8 degrees has to be dropped.
CASES = [
    ("naca0012", 8.0), ("naca0012", 10.0), ("naca0012", 12.0),
    ("naca2412", 8.0), ("naca2412", 10.0),
    ("naca0018", 4.0), ("naca0018", 8.0),
    ("naca4415", 4.0), ("naca2415", 8.0),
    ("naca0015", 2.0), ("naca2415", 2.0), ("naca4415", 2.0),
    ("naca0015", 4.0),
]
THRESHOLDS = (1e-3, 1e-4, 1e-5, 5e-6, 1e-6)
NEW_ARMS = ("oracle_mesh", "oracle_bl", "nf_bl", "cartesian_128", "or_proj_coarse")

# The wall-fitted projection, at the five-case tree's `coarse` placement so the
# two studies are comparable arm-for-arm. Kept as constants rather than flags:
# the corpus is the powered study and its arms should not be tunable per run.
PROJ_FIRST, PROJ_N_N, PROJ_N_S, PROJ_N_MAX = 2.5e-4, 64, 256, 1.0

# --- the exclusion rule, fixed before the sweep runs ------------------------- #
#
# Steady RANS has no unique fixed point once the separated flow becomes
# genuinely unsteady, and a warm-start saving measured against a solution that
# does not exist is meaningless. One case has already had to be excluded on this
# ground -- `naca4412@3`, whose arms landed 7% apart in final Cd against a
# residual floor of 1.6e-5 where every other case floors at 6e-8 to 1.7e-6.
#
# Deciding that *after* seeing which cases helped is the objection this pre-empts,
# so both thresholds are constants here rather than a judgement later. Each sits
# roughly an order of magnitude clear of both populations, so no case is decided
# by where exactly the line was drawn -- and any case near either threshold is
# printed with its numbers so a reader can disagree.
MAX_RESIDUAL_FLOOR = 1e-5      # naca4412@3: 1.6e-5. Everything kept: <= 1.7e-6.
MAX_ARM_CD_SPREAD = 0.02       # naca4412@3: 7%. Everything kept: <= 0.34%.


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--only", action="append", metavar="AIRFOIL@AOA")
    ap.add_argument("--ckpt-dir", default=os.path.join("checkpoints", "v2_transolver"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--max-sdf", type=float, default=3.5)
    ap.add_argument("--ramp", type=float, default=3.0)
    ap.add_argument("--resolution", type=int, default=128,
                    help="the Cartesian comparator's grid; 128^2 = 16,384 values")
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "corpus"))
    ap.add_argument("--out", default=os.path.join("results", "corpus.json"))
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
    print(f"Re {args.re:.0e} | boundary layer {delta:.4f} | "
          f"{len(cases)} case(s) x {1 + len(NEW_ARMS)} arms\n")

    rows = []
    for code, aoa in cases:
        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0, resolution=args.resolution)
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
        truth = (cold.u, cold.v, cold.p, cold.nut)

        pred, rep = ss.predict_on_mesh(
            checkpoints, cold.centres, surface[:, :2], reynolds=args.re, aoa_deg=aoa,
            wall_distance=distance, max_sdf=args.max_sdf, u_inf=u_inf,
            nut_freestream=nut_fs)

        free = (np.full_like(pred[0], u_inf), np.full_like(pred[0], v_inf),
                np.zeros_like(pred[0]), np.full_like(pred[0], nut_fs))
        def bl_only(background):
            """Hand `background` over inside the boundary layer, cold outside."""
            seed, _ = ws.masked_seed(free, cold.centres, surface,
                                     background=background, free_within=delta,
                                     ramp=args.ramp, u_inf=u_inf, v_inf=v_inf,
                                     nut_freestream=nut_fs)
            return seed

        # The exact field through the 256x64 body-fitted grid, then masked to the
        # boundary layer exactly as every other `_bl` arm is. `or_proj_coarse`
        # differs from `oracle_bl` in the representation and in nothing else.
        projected, _ = ws.clustered_seed(truth, cold.centres, surface,
                                         n_s=PROJ_N_S, n_n=PROJ_N_N,
                                         first=PROJ_FIRST, n_max=PROJ_N_MAX,
                                         u_inf=u_inf, v_inf=v_inf,
                                         nut_freestream=nut_fs)
        # The comparator: the same converged field, but seen only through a
        # `resolution`^2 uniform Cartesian grid. `to_grid` rasterises at the
        # case's own resolution, so 128 here is 16,384 values -- matched to the
        # wall-fitted 256x64 budget by construction.
        cartesian, _ = ws.plain_seed(cold.to_grid(case.domain), case.domain,
                                     cold.centres, u_inf=u_inf, v_inf=v_inf,
                                     nut_freestream=nut_fs)

        results = {"oracle_mesh": run("oracle_mesh", mesh_initial=truth),
                   "oracle_bl": run("oracle_bl", mesh_initial=bl_only(truth)),
                   "nf_bl": run("nf_bl", mesh_initial=bl_only(pred)),
                   "cartesian_128": run("cartesian_128", mesh_initial=cartesian),
                   "or_proj_coarse": run("or_proj_coarse",
                                         mesh_initial=bl_only(projected))}

        in_bl = distance <= delta
        covered = distance <= args.max_sdf
        def err(a, b, m):
            return float(100 * np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-30))
        field_err = {n: err(pred[i], truth[i], covered) for i, n in enumerate(ws.FIELDS)}
        field_err["u_in_bl"] = err(pred[0], truth[0], in_bl)
        # What the round trip did to the exact field it was handed, in the
        # boundary layer. Reported next to the solve because `or_proj_coarse`'s
        # convergence is only interpretable against how much it changed.
        proj_err = {n: err(projected[i], truth[i], in_bl)
                    for i, n in enumerate(ws.FIELDS)}

        # The pre-registered admission test, applied and recorded per case.
        finals = {}
        for name, d in [("cold", cold.case_dir)] + [
                (n, r.case_dir) for n, r in results.items()]:
            try:
                coeffs = of.read_force_coeffs(d)
                if len(coeffs.get("Cd", [])):
                    finals[name] = float(coeffs["Cd"][-1])
            except Exception:
                pass
        median = float(np.median(list(finals.values()))) if finals else float("nan")
        spread = (max(abs(v - median) / abs(median) for v in finals.values())
                  if finals and median else float("nan"))
        floor = float(cold.residual_floor)
        admitted = (floor <= MAX_RESIDUAL_FLOOR
                    and np.isfinite(spread) and spread <= MAX_ARM_CD_SPREAD)

        row = {"case": tag, "re": args.re, "aoa": aoa, "airfoil": code,
               "covered_fraction": rep["covered_fraction"],
               "cold_residual_floor": floor,
               "final_Cd_by_arm": finals,
               "arm_Cd_spread": spread,
               "admitted": bool(admitted),
               "field_error_pct": field_err,
               "or_proj_coarse_round_trip_pct": proj_err}
        print(f"   admission: floor {floor:.2e} (limit {MAX_RESIDUAL_FLOOR:.0e})  "
              f"arm Cd spread {100 * spread:.2f}% (limit "
              f"{100 * MAX_ARM_CD_SPREAD:g}%)  ->  "
              + ("ADMITTED" if admitted else "EXCLUDED: no unique steady fixed point"),
              flush=True)
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        checkpoint(rows)

        print(f"   cold residual floor {cold.residual_floor:.2e} | "
              "prediction error vs converged: "
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

    kept = [r for r in rows if r.get("admitted")]
    if len(kept) != len(rows):
        print("\nexcluded by the pre-registered admission test: "
              + ", ".join(r["case"] for r in rows if not r.get("admitted")))
    print(f"\nper-threshold mean over the {len(kept)} admitted case(s)")
    for t in THRESHOLDS:
        k = f"{t:.0e}"
        bits = []
        for name in NEW_ARMS:
            vals = [1 - r[f"{name}@{k}"] / r[f"cold@{k}"] for r in kept
                    if r.get(f"cold@{k}") and r.get(f"{name}@{k}")]
            bits.append(f"{name} {100 * np.mean(vals):+6.1f}% (n={len(vals)})"
                        if vals else f"{name} --")
        print(f"  @{k}: " + "   ".join(bits))
    checkpoint(rows)
    print(f"\nwrote {os.path.relpath(out_path)}")
    print("\nThe forces decide it, and the control decides whether to read them:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir} --per-case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
