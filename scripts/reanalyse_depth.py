"""Re-score finished warm-start runs at several convergence depths.

The probes in this repo reported iteration savings at residual 1e-3. On the
C-grid at Re 3e6 a cold start reaches 1e-3 in **36 iterations** and the residual
floor at ~1.3e-5 after several hundred, so 1e-3 is roughly 12% of the way to
convergence -- an early checkpoint, not convergence. The warm-start literature
reports *iterations to convergence*, and the two disagree in sign:

    threshold   oracle_mesh   uniform 128^2   wall-fitted 256x64
    1e-3            +73.1%          -18.5%              -44.4%
    1e-4            +68.5%         -306.4%              -49.5%
    3e-5            +84.3%          -33.2%              **+13.0%**
    2e-5            +85.9%          -21.7%              **+30.4%**

Early iterations are dominated by the global transient, where a structured but
imperfect seed costs; deep convergence is dominated by the near-wall state, where
a wall-fitted seed is right and a uniform one is hopeless. Reporting only 1e-3
hid a positive result.

This reads the residual histories already on disk -- no re-solving -- and scores
every arm at a ladder of depths so the choice of threshold is visible rather than
buried.

Usage
-----
    python scripts/reanalyse_depth.py --root runs/openfoam/repr
"""

from __future__ import annotations

import argparse
import json
import os

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import openfoam as of

DEPTHS = (1e-2, 1e-3, 1e-4, 5e-5, 3e-5, 2e-5)


def residuals(case_dir: str):
    log = os.path.join(case_dir, "log.simpleFoam")
    if not os.path.isfile(log):
        return None
    with open(log, encoding="utf-8", errors="replace") as fh:
        return of.parse_simple_foam_log(fh.read())["residuals"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "repr"))
    ap.add_argument("--cold", default="cold", help="suffix of the baseline arm")
    ap.add_argument("--out", default=os.path.join("results", "depth_reanalysis.json"))
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"no such directory: {args.root}")
        return 1

    # Split "<case>_<arm>" on the known arm suffixes present in the tree.
    entries = sorted(os.listdir(args.root))
    arms, cases = set(), set()
    for e in entries:
        for arm in ("oracle_mesh", "cartesian_128", "fitted_256x64", "oracle_128",
                    "oracle_128_hybrid", "cold"):
            if e.endswith("_" + arm):
                cases.add(e[: -len(arm) - 1])
                arms.add(arm)
                break
    arms = [a for a in ("oracle_mesh", "cartesian_128", "fitted_256x64",
                        "oracle_128", "oracle_128_hybrid") if a in arms]
    cases = sorted(cases)
    if not cases:
        print(f"no recognisable case directories under {args.root}")
        return 1

    hist = {}
    for c in cases:
        for a in [args.cold] + arms:
            hist[(c, a)] = residuals(os.path.join(args.root, f"{c}_{a}"))

    out = {"root": args.root, "cases": cases, "arms": arms, "by_depth": {}}
    print(f"{len(cases)} cases · arms: {', '.join(arms)}\n")
    print(f"{'depth':>8} {'cold it':>9} " + "  ".join(f"{a:>18}" for a in arms))

    for t in DEPTHS:
        k = f"{t:.0e}"
        colds, savings = [], {a: [] for a in arms}
        for c in cases:
            hc = hist[(c, args.cold)]
            if hc is None:
                continue
            base = of.iterations_to_threshold(hc, t)
            if not base:
                continue
            colds.append(base)
            for a in arms:
                ha = hist[(c, a)]
                v = of.iterations_to_threshold(ha, t) if ha else None
                if v:
                    savings[a].append(1.0 - v / base)
        entry = {"cold_mean": float(np.mean(colds)) if colds else None,
                 "cold_n": len(colds)}
        for a in arms:
            entry[a] = float(np.mean(savings[a])) if savings[a] else None
            entry[a + "_n"] = len(savings[a])
        out["by_depth"][k] = entry
        def cell(a):
            if entry[a] is None:
                return f"{'--':>16}"
            return f"{100 * entry[a]:+9.1f}% (n={entry[a + '_n']})"

        cold_txt = f"{entry['cold_mean']:.0f}" if colds else "--"
        print(f"{k:>8} {cold_txt:>9} " + "  ".join(f"{cell(a):>18}" for a in arms))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    print("\nNote the sign changes with depth. Report the depth alongside any saving;\n"
          "a number quoted at 1e-3 is a statement about the first ~12% of the solve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
