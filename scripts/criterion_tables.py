"""The criterion, evaluated for every output format the paper tables.

Two tables in the manuscript apply the closed form of section 6.1 to uniform
rasters -- section 6.5's format comparison and section 7.1's resolution ladder --
and before this script they were computed separately and disagreed: the same
128^2 raster was quoted at 29.3x in one and 36.6x in the other. The difference
was the ``u_tau`` each had been evaluated at (0.0477 against 0.0427), and
neither matched the value the study actually measures.

So both tables now come from here, from one stated triple:

* ``u_tau`` -- the mean over the five placement cases' own converged wall
  gradients (``results/closed_form_validation.json``), not a correlation.
* ``y_c``   -- the mean measured first cell centre over the same five cases.
  The C-grid's *nominal* first cell centre is ``first_cell / 2 = 5e-6``; the
  measured minimum wall distance is smaller because the surface is polygonal.
* ``nu``    -- the case's kinematic viscosity at Re 3e6, chord 1, u_inf 1.

Every raster's first station is half its cell, which is where a cell-centred
rasteriser puts its nearest sample to the wall.

Usage
-----
    python scripts/criterion_tables.py
    python scripts/criterion_tables.py --u-tau 0.0477   # reproduce an old row
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import placement as pl

NU = 3.3333333333333335e-07
U_INF = 1.0
DELTA = 0.0187          # boundary-layer thickness at Re 3e6, chord 1

# Section 6.5: the formats a surrogate might emit. (label, values, first station)
FORMATS = [
    ("uniform raster 128^2, 3-chord crop", 128 * 128, 3.0 / 128 / 2),
    ("uniform raster 256^2, 3-chord crop", 256 * 256, 3.0 / 256 / 2),
    ("uniform raster 512^2, 3-chord crop", 512 * 512, 3.0 / 512 / 2),
    ("uniform raster 128^2, 1-chord crop", 128 * 128, 1.0 / 128 / 2),
    ("wall-fitted 256x64 from 2.5e-4", 256 * 64, 2.5e-4),
    ("wall-fitted 256x64 from 2.5e-5", 256 * 64, 2.5e-5),
    ("wall-fitted 256x64 from 5e-6", 256 * 64, 5.0e-6),
    ("wall-fitted 256x32 from 5e-6", 256 * 32, 5.0e-6),
    ("mesh-native, queried at cell centres", 0, None),
]

# Section 7.1: the resolution ladder that was actually solved.
LADDER = [128, 181, 256, 362, 421]


def defaults_from_validation(path: str) -> tuple[float, float]:
    """``(u_tau, y_c)`` averaged over the cases the study measured."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    cases = data["per_case"]
    return (float(np.mean([c["u_tau"] for c in cases])),
            float(np.mean([c["first_cell_centre"] for c in cases])))


# The verdict thresholds, stated rather than tuned. `G` is an upper bound that
# over-predicts the measured damage by 1.9x to 2.8x (Table 5), so a predicted
# value inside that factor of unity is consistent with no measurable damage at
# all -- and is measured at 1.00x where the study measures it. Above 10x the
# representation has no sample of the state viscous drag integrates.
PASS_BELOW, FAIL_ABOVE = 1.5, 10.0


def classify(r: dict) -> str:
    if r["factor"] <= PASS_BELOW:
        return "**passes**"
    if r["factor"] <= FAIL_ABOVE:
        return "degraded"
    return "fails (bound)" if r["regime"] == "saturated" else "fails"


def row(first: float, y_c: float, u_tau: float) -> dict:
    return pl.amplification(first_station=first, cell_centre=y_c, u_tau=u_tau,
                            nu=NU, delta=DELTA, u_inf=U_INF)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation",
                    default=os.path.join("results", "closed_form_validation.json"))
    ap.add_argument("--u-tau", type=float, default=None)
    ap.add_argument("--y-c", type=float, default=None)
    ap.add_argument("--out", default=os.path.join("results", "criterion_tables.json"))
    args = ap.parse_args(argv)

    u_tau, y_c = defaults_from_validation(args.validation)
    u_tau = args.u_tau if args.u_tau is not None else u_tau
    y_c = args.y_c if args.y_c is not None else y_c
    y_c_plus = float(pl.wall_units(y_c, u_tau, NU))

    print(f"u_tau {u_tau:.5f}   y_c {y_c:.4e}   y_c+ {y_c_plus:.3f}   "
          f"nu {NU:.4e}   cap u_inf/u_tau {U_INF / u_tau:.2f}\n")

    print("### Section 6.5 -- the formats a surrogate might emit\n")
    print("| output format | values | first station | `y+` | predicted `G` | regime | verdict |")
    print("|---|---:|---:|---:|---:|---|---|")
    formats = []
    for label, values, first in FORMATS:
        station = y_c if first is None else first
        r = row(station, y_c, u_tau)
        verdict = classify(r)
        budget = "native" if not values else f"{values:,}"
        print(f"| {label} | {budget} | {station:.2g} | {r['y_plus_station']:.0f} | "
              f"{r['factor']:.1f}x | {r['regime']} | {verdict} |")
        formats.append({"format": label, "values": values, "first_station": station,
                        **r, "verdict": verdict})

    print("\n### Section 7.1 -- the resolution ladder that was solved\n")
    print("| raster | values | first station | `y+` | predicted `G` |")
    print("|---|---:|---:|---:|---:|")
    ladder = []
    for n in LADDER:
        first = 3.0 / n / 2
        r = row(first, y_c, u_tau)
        print(f"| {n}^2 | {n * n:,} | {first:.3g} | {r['y_plus_station']:.0f} | "
              f"{r['factor']:.1f}x |")
        ladder.append({"n": n, "values": n * n, "first_station": first, **r})

    first_g, last_g = ladder[0]["factor"], ladder[-1]["factor"]
    print(f"\na {LADDER[-1] ** 2 / LADDER[0] ** 2:.1f}-fold increase in stored "
          f"values moves the predicted damage {first_g:.1f}x -> {last_g:.1f}x "
          f"({100 * (1 - last_g / first_g):.1f}% of it)")

    payload = {"u_tau": u_tau, "y_c": y_c, "y_c_plus": y_c_plus, "nu": NU,
               "u_inf": U_INF, "delta": DELTA,
               "source": "mean over results/closed_form_validation.json per_case",
               "formats": formats, "ladder": ladder}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
