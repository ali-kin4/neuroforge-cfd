"""Does the closed form of section 6.1 bound the damage it predicts?

This is the committed producer of ``results/closed_form_validation.json``, the
file behind panel A of the placement figure and behind every "upper bound"
statement in the paper. It runs **no solver**: it reads each case's converged
field from the cold arm already on disk, sends it through the same
``clustered_seed`` round trip the projected arms use, and compares the
first-cell wall gradient it comes back with against what
:func:`neuroforge.solver.placement.amplification` predicts.

Three things this version fixes, each a reviewer finding.

**The prediction is evaluated at the mesh's own first cell centre.** The earlier
file recorded ``"probe": 4e-06`` -- a fixed height shared by every case -- while
the mesh's first cell centre is 3.79e-6 to 4.01e-6 depending on the section.
The *measurement* never depended on it (it is a ratio of two gradients sampled
at the same height, so the height cancels identically), but the *prediction*
does, through ``u+(y_c+)``. Here each case predicts against its own
``min(wall_distance)``.

**The degenerate rows are labelled.** Once the representation's first station
falls within the mesh's first ring, ``clustered_seed`` populates that station
from the first cell itself and maps it straight back: the round trip is the
identity near the wall and the measured overestimate is exactly 1.000000 with
zero variance across five different airfoils. That is a property of the donor
mapping, not an agreement between prediction and measurement, and rows in which
it happens are marked ``degenerate`` and excluded from the bound statistic. The
bound is claimed over the non-degenerate rows only.

**Every row carries its regime.** ``resolved`` / ``wall_law`` / ``saturated``,
from ``amplification`` itself, so a reader can see which rows the law of the
wall is actually doing work in (see section 6.2 on the saturated cap).

Usage
-----
    python scripts/validate_closed_form.py --root runs/openfoam/placement2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree

from neuroforge.core.types import FlowCase
from neuroforge.solver import cgrid as cg, openfoam as of, warmstart as ws
from neuroforge.solver import placement as pl

# The ladder of first stations, spanning a factor of fifty. Fixed here rather
# than passed in: it is the ladder the paper reports and it should not move.
FIRST_STATIONS = (2.5e-4, 1.0e-4, 2.5e-5, 1.0e-5, 5.0e-6)
N_S, N_N, N_MAX = 256, 64, 1.0

# The budget ladder, measured at *every* first station rather than only at the
# finest one. This matters: the five-case solve tree tests budget by comparing
# `or_proj_fine` (16,384 values) against `or_proj_half` (8,192), but both sit at
# first = 5e-6, where the round trip is a structural no-op -- so it compares two
# no-ops and could not have shown a budget effect even if one existed. Measuring
# the ladder at 2.5e-4 and 1e-4, where the mechanism is live, is what actually
# separates placement from budget. It costs no solve.
BUDGETS = (64, 32, 16)

# When does the closed form's *mechanism* apply at all?
#
# Section 6.1 models the round trip as clipping: a mesh cell nearer the wall
# than the representation's first station receives the velocity belonging to
# that station. That is what `clustered_seed` does to a cell only when the
# station is genuinely above it. But the grid's own stations are populated by
# nearest-neighbour *donor from the mesh*, so if fewer than two mesh rings lie
# below the station, the station is populated from the first ring itself and
# mapping back returns that ring its own value: the round trip is a near no-op
# at the wall and the measurement carries no information about clipping.
#
# This is a structural property of the pair (mesh, representation), computable
# before any measurement, so it is stated as a criterion rather than inferred
# from a measured ratio landing near one.
MIN_RINGS_BELOW_STATION = 2


def rings_below(distance, centres, mid, normal, tree, station: float) -> int:
    """How many distinct mesh cell rings lie below ``station``.

    Counted by walking the wall normal and collecting the distinct cells it
    passes through, at each of a spread of surface stations, then taking the
    median. Clustering the raw wall distances does not work: within a single
    ring the point-to-segment distance varies by more than the ring-to-ring
    spacing does near the leading and trailing edges, so a distance histogram
    counts curvature, not rings.
    """
    counts = []
    for j in np.linspace(0, len(mid) - 1, 21).astype(int):
        heights = np.geomspace(0.05 * station, station, 400)
        _, idx = tree.query(mid[j] + heights[:, None] * normal[j])
        counts.append(len(np.unique(idx[distance[idx] < station])))
    return int(np.median(counts))


def wall_frame(code: str, spec: cg.CGridSpec):
    """Surface midpoints and outward unit normals, ``(M, 2)`` each."""
    inner, n_wake, n_surface = cg.inner_curve(code, spec)
    surf = inner[n_wake - 1: n_wake + n_surface - 1]
    mid = 0.5 * (surf[:-1] + surf[1:])
    tangent = surf[1:] - surf[:-1]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-30
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    outward = mid - surf.mean(axis=0)
    flip = np.sign(np.sum(normal * outward, axis=1))
    flip[flip == 0] = 1.0
    return mid, normal * flip[:, None], surf


def first_cell_speed(u, v, centres, mid, normal, height, tree=None):
    """``|u_t|`` in the first cell above each surface station.

    Sampled along the wall normal, not by nearest neighbour to the surface: the
    C-grid's near-wall cells are ~2500x wider than they are tall, so a plain
    nearest query walks along the wall instead of away from it.
    """
    tree = tree if tree is not None else cKDTree(centres[:, :2])
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    _, idx = tree.query(mid + height * normal)
    vel = np.stack([u[idx], v[idx]], axis=1)
    return np.abs(np.sum(vel * tangent, axis=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "placement2"))
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--only", action="append", metavar="CASE_TAG")
    ap.add_argument("--out", default=os.path.join("results",
                                                  "closed_form_validation.json"))
    args = ap.parse_args(argv)

    spec = cg.CGridSpec()
    tags = sorted(d[: -len("_cold")] for d in os.listdir(args.root)
                  if d.endswith("_cold"))
    if args.only:
        tags = [t for t in tags if t in set(args.only)]
    if not tags:
        print(f"no '<case>_cold' directories under {args.root}")
        return 1

    per_case = []
    for tag in tags:
        code, _, aoa_text = tag.partition("_aoa")
        aoa = float(aoa_text)
        cold_dir = os.path.join(args.root, f"{tag}_cold")
        latest = of._latest_time(cold_dir)
        if latest is None:
            print(f"{tag}: cold has no converged time directory; skipped")
            continue

        case = FlowCase.from_airfoil(airfoil=code, aoa=aoa, reynolds=args.re,
                                     u_inf=1.0)
        nu = float(case.fluid.kinematic_viscosity)
        u_inf, v_inf = of._freestream(case)
        nut_fs = of.NUTILDA_FREESTREAM_RATIO * nu

        centres = of.read_volfield(os.path.join(cold_dir, "0", "C"))
        U = of.read_volfield(os.path.join(cold_dir, latest, "U"))
        p = of.read_volfield(os.path.join(cold_dir, latest, "p")).reshape(-1)
        nut = of.read_volfield(os.path.join(cold_dir, latest, "nut")).reshape(-1)
        truth = (U[:, 0], U[:, 1], p, nut)

        inner, n_wake, n_surface = cg.inner_curve(code, spec)
        surface = inner[n_wake - 1: n_wake + n_surface - 1]
        distance = ws.wall_distance(centres, surface)
        cell_centre = float(np.min(distance))

        mid, normal, _ = wall_frame(code, spec)
        tree = cKDTree(centres[:, :2])
        # The height is the case's own first cell centre, so the sample lands in
        # the first ring by construction. It cancels from every ratio below.
        ref_speed = first_cell_speed(truth[0], truth[1], centres, mid, normal,
                                     cell_centre, tree)
        ref_grad = float(np.mean(ref_speed)) / cell_centre
        u_tau = pl.friction_velocity(ref_grad, nu)

        print(f"== {tag} ==  first cell centre {cell_centre:.3e}  "
              f"u_tau {u_tau:.4f}  y+ {cell_centre * u_tau / nu:.3f}", flush=True)

        def measure(first: float, n_n: int) -> float:
            seed, _ = ws.clustered_seed(truth, centres, surface, n_s=N_S, n_n=n_n,
                                        first=first, n_max=N_MAX, u_inf=u_inf,
                                        v_inf=v_inf, nut_freestream=nut_fs)
            speed = first_cell_speed(seed[0], seed[1], centres, mid, normal,
                                     cell_centre, tree)
            return (float(np.mean(speed)) / cell_centre) / ref_grad

        rows = []
        for first in FIRST_STATIONS:
            by_budget = {n: measure(first, n) for n in BUDGETS}
            measured = by_budget[N_N]
            pred = pl.amplification(first_station=first, cell_centre=cell_centre,
                                    u_tau=u_tau, nu=nu, u_inf=float(u_inf))
            n_rings = rings_below(distance, centres, mid, normal, tree, first)
            degenerate = n_rings < MIN_RINGS_BELOW_STATION
            rows.append({"first": first, "predicted": pred["factor"],
                         "measured": measured,
                         "ratio": pred["factor"] / measured,
                         "regime": pred["regime"],
                         "y_plus_station": pred["y_plus_station"],
                         "y_plus_cell": pred["y_plus_cell"],
                         "rings_below_station": n_rings,
                         "degenerate": bool(degenerate),
                         "measured_by_n_n": {str(n): v for n, v in by_budget.items()},
                         "values_by_n_n": {str(n): N_S * n for n in BUDGETS}})
            flag = ("  <- no-op: donor mapping returns the first ring its own "
                    "value" if degenerate else "")
            budget = "  ".join(f"n_n={n}:{by_budget[n]:.3f}" for n in BUDGETS)
            print(f"   first {first:.1e}  y+ {pred['y_plus_station']:7.2f}  "
                  f"rings below {n_rings:2d}  "
                  f"predicted {pred['factor']:8.3f}  measured {measured:8.3f}  "
                  f"ratio {pred['factor'] / measured:5.2f}  {pred['regime']}{flag}\n"
                  f"          budget ladder  {budget}", flush=True)

        per_case.append({"case": tag, "first_cell_centre": cell_centre,
                         "u_tau": u_tau, "nu": nu,
                         "y_plus_cell": cell_centre * u_tau / nu, "rows": rows})

    if not per_case:
        print("no cases measured")
        return 1

    # Aggregate across cases, one summary row per first station.
    summary = []
    for j, first in enumerate(FIRST_STATIONS):
        cells = [c["rows"][j] for c in per_case]
        pred = np.array([r["predicted"] for r in cells])
        meas = np.array([r["measured"] for r in cells])
        ratio = pred / meas
        summary.append({
            "first": first,
            "predicted": float(np.mean(pred)),
            "measured": float(np.mean(meas)),
            "ratio": float(np.mean(ratio)),
            "ratio_std": float(np.std(ratio)),
            "n": len(cells),
            "regime": sorted({r["regime"] for r in cells}),
            "rings_below_station": int(np.min([r["rings_below_station"]
                                               for r in cells])),
            "degenerate": bool(all(r["degenerate"] for r in cells)),
            "measured_by_n_n": {
                str(n): float(np.mean([r["measured_by_n_n"][str(n)] for r in cells]))
                for n in BUDGETS},
            "values_by_n_n": {str(n): N_S * n for n in BUDGETS},
        })

    live = [r for r in summary if not r["degenerate"]]
    bound_holds = all(r["ratio"] >= 1.0 for r in live)
    payload = {
        "root": args.root, "re": args.re, "n_s": N_S, "n_n": N_N, "n_max": N_MAX,
        "probe": "each case's own first cell centre (min wall distance)",
        "min_rings_below_station": MIN_RINGS_BELOW_STATION,
        "rows": summary,
        "per_case": per_case,
        "bound": {
            "non_degenerate_rows": len(live),
            "degenerate_rows": len(summary) - len(live),
            "holds_on_non_degenerate": bool(bound_holds),
            "ratio_min": float(min(r["ratio"] for r in live)) if live else None,
            "ratio_max": float(max(r["ratio"] for r in live)) if live else None,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nbound over the {len(live)} non-degenerate row(s): "
          f"over-predicts by {payload['bound']['ratio_min']:.2f}x to "
          f"{payload['bound']['ratio_max']:.2f}x"
          + ("" if bound_holds else "  <- BOUND VIOLATED: a row under-predicts"))
    print("\nplacement against budget, on the rows where the mechanism is live "
          "(the ones that can show a budget effect at all):")
    print(f"  {'first':>10}  " + "  ".join(f"{N_S * n:>7d} vals" for n in BUDGETS))
    for r in live:
        print(f"  {r['first']:10.1e}  "
              + "  ".join(f"{r['measured_by_n_n'][str(n)]:11.3f}" for n in BUDGETS))
    spread = max(
        max(r["measured_by_n_n"].values()) / max(min(r["measured_by_n_n"].values()), 1e-30)
        for r in live) if live else float("nan")
    across = (max(r["measured"] for r in live) / max(min(r["measured"] for r in live), 1e-30)
              if live else float("nan"))
    print(f"  cutting the budget {BUDGETS[0] // BUDGETS[-1]}x moves the damage by "
          f"at most {spread:.2f}x; moving the station over the same live rows "
          f"moves it {across:.1f}x.")

    print(f"\n{len(summary) - len(live)} row(s) excluded as no-ops: fewer than "
          f"{MIN_RINGS_BELOW_STATION} mesh rings lie below the representation's "
          "first station, so the donor mapping returns the first ring its own "
          "value and the round trip cannot exercise the clipping the closed "
          "form models.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
