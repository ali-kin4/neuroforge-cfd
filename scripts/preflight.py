"""Before you train anything: will this output format survive a solver?

The question costs nothing to ask. Give it your mesh's first cell height and the
shape of the representation your surrogate would emit, and it reports how badly
that representation misreports the first-cell wall gradient -- the quantity
viscous drag integrates, and the one that decides whether a warm start helps or
costs you the solve.

Examples
--------
A wall-resolved airfoil mesh at Re 3e6 (first cell 1e-5 chords, y+ ~ 1), against
the two output formats the field actually ships and the one that works::

    python scripts/preflight.py --first-cell 1e-5 --re 3e6 --raster 128
    python scripts/preflight.py --first-cell 1e-5 --re 3e6 --fitted 256x64@2.5e-4
    python scripts/preflight.py --first-cell 1e-5 --re 3e6 --fitted 256x32@5e-6
    python scripts/preflight.py --first-cell 1e-5 --re 3e6 --mesh-native

If you know your own ``u_tau`` -- and if you have a converged solve you do --
pass it with ``--u-tau`` instead of letting the flat-plate correlation estimate
it. The verdict does not depend much on it: it enters both ``y+`` values and
largely cancels in their ratio.
"""

from __future__ import annotations

import argparse
import sys

# Must precede numpy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import placement as pl

# Where the measured arms of this study land, so a user can read their own
# number against something real rather than against an invented scale.
LANDMARKS = (
    ("mesh-native (this paper, `nf_bl`)", 1.0, "+18.4% on viscous drag, 13/13 cases"),
    ("wall-fitted 256x64 @ 2.5e-4", 21.0, "-58.8% on total drag"),
    ("uniform Cartesian 128^2", 18.7, "-548% with the *exact* converged field"),
    ("uniform freestream (cold start)", 33.8, "the baseline being beaten"),
)


def parse_fitted(text: str) -> tuple[int, int, float]:
    """``256x64@2.5e-4`` -> (256, 64, 2.5e-4)."""
    grid, _, first = text.partition("@")
    n_s, _, n_n = grid.lower().partition("x")
    if not (n_s and n_n and first):
        raise argparse.ArgumentTypeError(
            "expected TANGENTIALxNORMAL@FIRST, e.g. 256x64@2.5e-4")
    return int(n_s), int(n_n), float(first)


def flat_plate_u_tau(re: float, u_inf: float = 1.0, x: float = 0.5) -> float:
    """Rough ``u_tau`` from the flat-plate skin-friction correlation.

    ``c_f = 0.0576 Re_x^{-1/5}`` at a representative station. Only a stand-in for
    a measured value, and the report says so.
    """
    c_f = 0.0576 * (re * x) ** -0.2
    return float(u_inf * np.sqrt(c_f / 2.0))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-cell", type=float, required=True,
                    help="wall-normal HEIGHT of the mesh's first cell [chord]; "
                         "the centre is taken at half of it")
    ap.add_argument("--re", type=float, default=3e6, help="chord Reynolds number")
    ap.add_argument("--u-inf", type=float, default=1.0)
    ap.add_argument("--chord", type=float, default=1.0)
    ap.add_argument("--u-tau", type=float, default=None,
                    help="measured friction velocity; overrides the correlation")
    ap.add_argument("--delta", type=float, default=None,
                    help="boundary-layer thickness [chord]; enables the "
                         "saturated-regime warning")
    ap.add_argument("--raster", type=int, metavar="N",
                    help="uniform NxN raster output")
    ap.add_argument("--raster-height", type=float, default=3.0,
                    help="wall-normal extent the raster spans [chord]")
    ap.add_argument("--fitted", type=parse_fitted, metavar="256x64@2.5e-4",
                    help="wall-fitted graded grid: TANGENTIALxNORMAL@FIRST")
    ap.add_argument("--outer", type=float, default=1.0,
                    help="outer extent of the wall-fitted grid [chord]")
    ap.add_argument("--mesh-native", action="store_true",
                    help="the surrogate is queried at the solver's own cell centres")
    args = ap.parse_args(argv)

    nu = args.u_inf * args.chord / args.re
    cell_centre = 0.5 * args.first_cell
    u_tau = args.u_tau if args.u_tau else flat_plate_u_tau(args.re, args.u_inf)
    delta = args.delta

    if not (args.raster or args.fitted or args.mesh_native):
        ap.error("give one of --raster, --fitted or --mesh-native")

    print(f"mesh        first cell {args.first_cell:.2e} chord, "
          f"centre {cell_centre:.2e}")
    print(f"flow        Re {args.re:.1e}, nu {nu:.3e}, u_tau {u_tau:.4f}"
          + ("" if args.u_tau else "  (flat-plate correlation -- pass --u-tau if known)"))
    print(f"            first cell centre is y+ "
          f"{pl.wall_units(cell_centre, u_tau, nu):.2f}\n")

    if args.mesh_native:
        stations = np.array([cell_centre])
        label = "mesh-native"
    elif args.fitted:
        n_s, n_n, first = args.fitted
        stations = pl.geometric_stations(first, args.outer, n_n)
        label = f"wall-fitted {n_s}x{n_n} from {first:.1e}, {n_s * n_n} values"
    else:
        stations = pl.uniform_stations(args.raster_height, args.raster)
        label = (f"uniform raster {args.raster}^2 over {args.raster_height:g} chord, "
                 f"{args.raster ** 2} values")

    got = pl.preflight(stations=stations, cell_centre=cell_centre,
                       u_tau=u_tau, nu=nu, delta=delta, u_inf=args.u_inf)

    print(f"representation  {label}")
    print(f"  first station        {got['first_station']:.2e} chord "
          f"(y+ {got['y_plus_station']:.2f})")
    print(f"  stations inside the first cell   "
          f"{got['stations_inside_first_cell']}")
    print(f"  first station / first cell centre "
          f"{got['first_station_over_cell_centre']:.1f}x\n")

    factor, regime = got["factor"], got["regime"]
    if regime == "resolved":
        verdict = ("PASSES. The representation samples inside the first cell, so "
                   "the wall gradient survives the round trip.")
    elif regime == "saturated":
        verdict = (f"FAILS. First station is outside the boundary layer, where the "
                   f"law of the wall does not apply; >{factor:.0f}x is a bound, not "
                   f"an estimate, and the near-wall state is simply absent.")
    else:
        verdict = (f"{'FAILS' if factor > 2 else 'MARGINAL'}. Expect the first-cell "
                   f"wall gradient to be overestimated by about {factor:.1f}x.")

    print(f"  predicted wall-gradient overestimate  {factor:.2f}x   [{regime}]")
    print(f"  {verdict}\n")

    if factor > 2:
        need = cell_centre
        print("  What to change: not the number of values -- u+ grows "
              "logarithmically, so\n"
              "  refining the grid barely moves this. Move the first station to "
              f"<= {need:.1e}\n  chord. On a geometric stack to "
              f"{args.outer:g} chord that costs nothing but grading:")
        for n_n in (32, 64):
            growth = (args.outer / need) ** (1.0 / max(n_n - 1, 1))
            print(f"    {n_n:3d} levels from {need:.1e} -> growth ratio {growth:.3f}")
        print()

    print("  for scale, measured in this study (Re 3e6, first cell 1e-5):")
    nearest = min(range(len(LANDMARKS)),
                  key=lambda i: abs(np.log10(max(factor, 1e-9) / LANDMARKS[i][1])))
    for i, (name, amp, outcome) in enumerate(LANDMARKS):
        mark = " <-- you are here" if i == nearest else ""
        print(f"    {amp:5.1f}x   {name:<34} {outcome}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
