"""Is it the *budget* of values, or *where they are put*?

The claim this repo has been making is that "no 16,384-value grid has a station
4e-6 chords off the wall". That sentence is **arithmetically false**, and a
reviewer with a calculator finds it in the abstract.

``clustered_seed`` builds its wall-fitted grid with ``first = 2.5e-4`` chords.
The C-grid's first cell *centre* is ``5e-6`` -- fifty times finer. A 64-level
geometric stack from 5e-6 to 1.0 needs a growth ratio of only 1.214, which is an
ordinary mesh. So a 16,384-value wall-fitted grid *can* have a station in the
first cell. Ours simply did not.

This script decides what the real mechanism is, on one prediction per case so
nothing else can move. Two families, each a projection of the same field through
a wall-fitted grid of the same construction, differing only in **where the
stations are**:

* ``*_coarse`` -- ``first = 2.5e-4``, ``n_n = 64``. **16,384 values**, first
  station 50x outside the first cell. This is the arm the paper has been
  reporting (``nf_bl_proj`` = -58.8%, ``fitted_bl`` = -206.1%).
* ``*_fine``   -- ``first = 5e-6``, ``n_n = 64``. **16,384 values**, first
  station *inside* the first cell. Same budget, correct placement.
* ``*_half``   -- ``first = 5e-6``, ``n_n = 32``. **8,192 values** -- the budget
  *halved* -- first station still inside the first cell.

``*_half`` is the arm that decides it. If correct placement at **half** the
budget recovers what correct budget at the wrong placement could not, then
placement beats budget with budget moving adversely, and the paper's thesis is
station placement rather than value count. If instead ``*_fine`` and ``*_half``
both stay negative, the round trip is destroying the field for some other reason
and we have isolated it rather than assumed it.

Both families are run twice: once on the **network prediction** (the practical
statement) and once on the **exact converged field** (the statement that cannot
be explained away by our model being bad).

Every arm is boundary-layer-masked exactly as ``nf_bl`` is, so region is held
fixed and representation is the only variable.

**Fresh work directory on purpose.** ``runs/openfoam/repr3`` is the evidence
behind every table in the draft, and scoring rule 4 re-scores every arm in a
tree when a new arm joins it (the settled reference is a median over arms). This
tree is self-contained: it carries its own ``cold`` and its own ``oracle_mesh``
control, and is scored on its own.

Usage
-----
    python scripts/placement_probe.py
    python scripts/placement_probe.py --only naca0012@4
    python scripts/reanalyse_depth.py --root runs/openfoam/placement
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

# (arm suffix, first station, wall-normal levels). The tangential count is held
# at 256 throughout, so `points` moves only with `n_n`.
PLACEMENTS = (
    ("coarse", 2.5e-4, 64),   # 16,384 values, first station 50x outside cell 1
    ("fine",   5.0e-6, 64),   # 16,384 values, first station inside cell 1
    ("half",   5.0e-6, 32),   # 8,192 values -- half the budget, still inside
)

NEW_ARMS = (("oracle_mesh", "nf_bl")
            + tuple(f"nf_proj_{s}" for s, _, _ in PLACEMENTS)
            + tuple(f"or_proj_{s}" for s, _, _ in PLACEMENTS))


def stations_below(first: float, n_n: int, n_max: float, cell_centre: float) -> int:
    """How many of the grid's wall-normal stations fall inside the first cell.

    This is the quantity the paper calls *placement*, and it is computable from
    the surrogate's output format and the target mesh alone -- no solve, no
    network, no data. It is the whole pre-flight check.
    """
    if n_n < 1:
        return 0
    stations = np.geomspace(first, n_max, n_n)
    return int(np.count_nonzero(stations <= cell_centre))


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
    ap.add_argument("--n-max", type=float, default=1.0)
    ap.add_argument("--work-dir", default=os.path.join("runs", "openfoam", "placement"))
    ap.add_argument("--out", default=os.path.join("results", "placement.json"))
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

    def checkpoint(rows, geometry):
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"re": args.re, "n_iter": args.n_iter,
                       "checkpoints": checkpoints, "n_s": args.n_s,
                       "n_max": args.n_max, "placements": geometry,
                       "rows": rows}, fh, indent=2)
        os.replace(tmp, out_path)

    spec = cg.CGridSpec()
    delta = ws.bl_thickness(args.re)
    print(f"Re {args.re:.0e} | boundary layer {delta:.4f}\n")

    rows: list[dict] = []
    geometry: dict[str, dict] = {}
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
        truth = (cold.u, cold.v, cold.p, cold.nut)

        # The mesh's own first cell centre, measured rather than assumed: the
        # smallest wall distance over all cells. Placement is defined against it.
        cell_centre = float(np.min(distance))

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

        def project(values, first, n_n):
            out, report = ws.clustered_seed(
                values, cold.centres, surface, n_s=args.n_s, n_n=n_n,
                first=first, n_max=args.n_max, u_inf=u_inf, v_inf=v_inf,
                nut_freestream=nut_fs)
            return out, report

        in_bl = distance <= delta

        def err(a, b, m):
            return float(100 * np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-30))

        seeds: dict[str, tuple] = {"oracle_mesh": truth, "nf_bl": bl_only(pred)}
        damage: dict[str, dict] = {}
        for suffix, first, n_n in PLACEMENTS:
            geometry[suffix] = {
                "first": first, "n_n": n_n, "n_s": args.n_s,
                "points": args.n_s * n_n,
                "stations_inside_first_cell":
                    stations_below(first, n_n, args.n_max, cell_centre),
                "first_station_over_cell_centre": first / max(cell_centre, 1e-30),
            }
            for family, values in (("nf", pred), ("or", truth)):
                projected, _ = project(values, first, n_n)
                seeds[f"{family}_proj_{suffix}"] = bl_only(projected)
                # What the round trip did to the field it was handed, in the
                # boundary layer -- reported next to the solve, because the
                # solve is only interpretable against it.
                damage[f"{family}_proj_{suffix}"] = {
                    n: err(projected[i], values[i], in_bl) for i, n in enumerate(ws.FIELDS)
                }

        results = {name: run(name, mesh_initial=seed)
                   for name, seed in seeds.items()}

        row = {"case": tag, "re": args.re, "covered_fraction": rep["covered_fraction"],
               "first_cell_centre": cell_centre,
               "round_trip_change_pct": damage}
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            row[f"cold@{k}"] = cold.iterations_to(t)
            for name, res in results.items():
                row[f"{name}@{k}"] = res.iterations_to(t)
        rows.append(row)
        checkpoint(rows, geometry)

        print(f"   first cell centre {cell_centre:.2e}; stations inside it: "
              + ", ".join(f"{s}={geometry[s]['stations_inside_first_cell']}"
                          f"/{geometry[s]['n_n']}" for s, _, _ in PLACEMENTS), flush=True)
        for t in THRESHOLDS:
            k = f"{t:.0e}"
            base = row[f"cold@{k}"]
            bits = []
            for name in NEW_ARMS:
                v = row.get(f"{name}@{k}")
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
    checkpoint(rows, geometry)
    print(f"\nwrote {os.path.relpath(out_path)}")
    print("\nThe forces are what decides this. Score with:\n"
          f"  python scripts/reanalyse_depth.py --root {args.work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
