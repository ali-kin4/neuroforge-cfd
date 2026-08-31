"""Does the criterion transfer across Reynolds number? Asked with no new solves.

The closed form of §6 is written in wall units, so a Reynolds sweep is the
natural test of it -- and it makes a prediction that is not obvious:

> **On the same mesh, a lower Reynolds number makes the projection worse.**

Both `y+` values fall together as `nu` rises, but they fall through different
parts of the profile. The mesh's first cell sinks deeper into the *linear*
sublayer, where `u+ = y+` falls in proportion; the representation's fixed
station at 2.5e-4 stays in the log or buffer layer, where `u+` falls only
logarithmically. The ratio -- which is the damage -- therefore **grows**.

That is a falsifiable prediction about a regime the paper never measured, and it
can be tested for free: `runs/openfoam/crossover2` already holds converged cold
solves on the *same* C-grid at Re 1e3, 1e4, 1e5, 1e6 and 3e6. Project each one
through the same wall-fitted grid, measure what happens to the first-cell wall
gradient, and compare with the closed form. No solver runs, no network.

Nothing here re-solves anything: the finished cases are read directly from disk.

Usage
-----
    python scripts/reynolds_transfer.py
    python scripts/reynolds_transfer.py --root runs/openfoam/crossover2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import cgrid as cg, openfoam as of, warmstart as ws
from neuroforge.solver import placement as pl

PROBE = 4e-6          # matches scripts/seed_gradient_diagnostic.py
FIRST_STATION = 2.5e-4


def read_case(case_dir: str):
    """Fields and cell centres of a finished solve, without re-solving it."""
    meta_path = os.path.join(case_dir, "neuroforge.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if of.completed_run(case_dir) is None:
        return None
    latest = of._latest_time(case_dir)
    if latest is None:
        return None
    U = of.read_volfield(os.path.join(case_dir, latest, "U"))
    p = of.read_volfield(os.path.join(case_dir, latest, "p"))
    nut_path = os.path.join(case_dir, latest, "nut")
    nut = of.read_volfield(nut_path) if os.path.isfile(nut_path) else np.zeros(len(U))
    C = of.read_volfield(os.path.join(case_dir, "0", "C"))
    return meta, np.asarray(U), np.asarray(p).ravel(), np.asarray(nut).ravel(), np.asarray(C)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "crossover2"))
    ap.add_argument("--out", default=os.path.join("results", "reynolds_transfer.json"))
    args = ap.parse_args(argv)

    sys.path.insert(0, "scripts")
    from seed_gradient_diagnostic import wall_stations, tangential_profile

    if not os.path.isdir(args.root):
        print(f"no such tree: {args.root}")
        return 1

    cases = sorted(d for d in os.listdir(args.root) if d.endswith("_cold"))
    rows = []
    for name in cases:
        got = read_case(os.path.join(args.root, name))
        if got is None:
            continue
        meta, U, p, nut, C = got
        nu = float(meta["nu"])
        code = meta["airfoil"]
        spec = cg.CGridSpec(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in meta["spec"].items()})
        u_inf, v_inf = float(meta["u_inf"]), float(meta["v_inf"])
        nut_fs = float(meta["nut_freestream"])

        mid, normal, surface = wall_stations(code, spec)
        u, v = U[:, 0], U[:, 1]
        truth = (u, v, p, nut)

        projected, _ = ws.clustered_seed(truth, C, surface, n_s=256, n_n=64,
                                         first=FIRST_STATION, u_inf=u_inf,
                                         v_inf=v_inf, nut_freestream=nut_fs)

        def gradient(fields):
            return tangential_profile(fields[0], fields[1], C, mid, normal, [PROBE])[:, 0] / PROBE

        g_true = gradient(truth)
        g_proj = gradient(projected)
        ok = g_true > 0
        if ok.sum() < 10:
            continue

        # Fraction of the surface carrying almost no wall shear. It is the
        # tell-tale of a laminar or separated layer, and therefore of the
        # regime where the law of the wall does not apply at all.
        weak = float((g_true[ok] < 0.1 * g_true[ok].max()).mean())
        u_tau = pl.friction_velocity(float(np.mean(g_true[ok])), nu)
        predicted = pl.amplification(first_station=FIRST_STATION, cell_centre=PROBE,
                                     u_tau=u_tau, nu=nu, u_inf=float(np.hypot(u_inf, v_inf)))
        ratios = g_proj[ok] / g_true[ok]
        measured = float(np.mean(ratios))
        measured_median = float(np.median(ratios))

        rows.append({
            "case": name[:-5], "airfoil": code, "nu": nu,
            "reynolds": float(np.hypot(u_inf, v_inf) / nu),
            "u_tau": u_tau,
            "y_plus_cell": predicted["y_plus_cell"],
            "y_plus_station": predicted["y_plus_station"],
            "regime": predicted["regime"],
            "predicted": predicted["factor"],
            "measured": measured,
            "measured_median": measured_median,
            "weak_shear_fraction": weak,
            "ratio": predicted["factor"] / measured if measured else None,
        })

    if not rows:
        print("no finished cases found")
        return 1

    rows.sort(key=lambda r: (r["reynolds"], r["case"]))
    print(f"{'case':<26}{'Re':>9}{'y+ cell':>9}{'y+ stn':>9}"
          f"{'predicted':>11}{'measured':>10}{'median':>9}{'ratio':>8}{'weak':>7}")
    for r in rows:
        print(f"{r['case']:<26}{r['reynolds']:>9.0e}{r['y_plus_cell']:>9.3f}"
              f"{r['y_plus_station']:>9.1f}{r['predicted']:>11.1f}{r['measured']:>10.1f}"
              f"{r['measured_median']:>9.1f}{(r['ratio'] or 0):>8.2f}"
              f"{100 * r['weak_shear_fraction']:>6.0f}%")
    print("\n'weak' is the fraction of surface stations carrying under a tenth of"
          " the peak wall shear -- the signature of a laminar or separated layer,"
          " and so of the regime where the law of the wall does not apply at all.")

    by_re: dict[float, list] = {}
    for r in rows:
        by_re.setdefault(round(r["reynolds"], -2), []).append(r)
    print("\nmean over cases at each Reynolds number")
    for re_val in sorted(by_re):
        group = by_re[re_val]
        print(f"  Re {re_val:8.0e}:  predicted {np.mean([g['predicted'] for g in group]):6.1f}x"
              f"   measured {np.mean([g['measured'] for g in group]):6.1f}x"
              f"   ratio {np.mean([g['ratio'] for g in group if g['ratio']]):.2f}"
              f"   (n={len(group)})")

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"probe": PROBE, "first_station": FIRST_STATION, "rows": rows},
                  fh, indent=2)
    print(f"\nwrote {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
