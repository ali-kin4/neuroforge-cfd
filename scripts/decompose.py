"""What actually decides a warm start, separated into one-variable contrasts.

The study's arms differ from each other in more than one way at a time, and a
comparison that moves two variables cannot attribute the difference to either.
This script takes the arms that *do* differ in exactly one property and reports
each step of the decomposition with a paired confidence interval, a win count
and an exact sign test.

The four properties, and the pair that isolates each:

* **region** -- ``oracle_mesh`` (the exact converged field everywhere) against
  ``oracle_bl`` (the same field, handed over inside the boundary layer only).
  Same field, same accuracy, same mesh-native delivery.
* **representation, body-fitted** -- ``oracle_bl`` against ``or_proj_coarse``
  (the same field, same mask, sent through a 256x64 body-fitted grid of 16,384
  values first). Only the storage format moves.
* **representation, raster** -- ``oracle_mesh`` against ``cartesian_128`` (the
  same field, same region, all four channels, stored on a 128^2 uniform raster
  holding the same 16,384 values). Only the storage format moves.
* **accuracy** -- ``oracle_bl`` against ``nf_bl`` (the trained surrogate's
  prediction in place of the exact field, same mask, same mesh-native delivery).
  Only the source of the values moves.

Everything is measured on ``Cd_v@1%``, the row section 4's readability rule
admits on the thirteen-case corpus, and every contrast is **paired within case**
before it is averaged, because the cases differ more from each other than the
arms do.

Usage
-----
    python scripts/decompose.py --scored results/depth_corpus2.json
    python scripts/decompose.py --scored results/depth_placement2.json --row Cd_v@0.01
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import scoring as sc

# (label, baseline arm, arm, what moves)
CONTRASTS = [
    ("region", "oracle_mesh", "oracle_bl", "whole field -> boundary layer only"),
    ("representation (body-fitted)", "oracle_bl", "or_proj_coarse",
     "mesh-native -> 256x64 body-fitted grid, 16,384 values"),
    ("representation (raster)", "oracle_mesh", "cartesian_128",
     "mesh-native -> 128^2 uniform raster, 16,384 values"),
    ("accuracy", "oracle_bl", "nf_bl", "exact field -> surrogate prediction"),
]


def savings(per_case: dict, arm: str) -> dict[str, float]:
    """Per-case iteration saving against that case's own cold start."""
    out = {}
    for case, row in per_case.items():
        cold, got = row.get("cold"), row.get(arm)
        if cold and got:
            out[case] = 1.0 - got / cold
    return out


def report(name: str, values: np.ndarray, unit: str = "%") -> dict:
    lo, hi = sc.bootstrap_ci(values)
    p = sc.sign_test(values)
    wins = int(np.sum(values > 0))
    return {"mean": float(np.mean(values)), "ci": [float(lo), float(hi)],
            "wins": wins, "n": len(values), "p": float(p)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default=os.path.join("results", "depth_corpus2.json"))
    ap.add_argument("--row", default="Cd_v@0.01")
    ap.add_argument("--out", default=os.path.join("results", "decomposition.json"))
    args = ap.parse_args(argv)

    if not os.path.isfile(args.scored):
        print(f"missing {args.scored} -- run reanalyse_depth.py first")
        return 1
    with open(args.scored, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = (data.get("by_force") or {}).get(args.row)
    if not entry:
        print(f"{args.scored} has no row {args.row}; it has "
              + ", ".join(sorted(data.get("by_force") or {})))
        return 1
    per_case = entry.get("per_case") or {}
    if not entry.get("readable", True):
        print(f"WARNING: {args.row} is marked unreadable in {args.scored} "
              f"(settled spread {entry.get('settled_spread')}). Section 4 forbids "
              "reading it. Reported below only so the verdict is visible.")

    print(f"{args.scored}   row {args.row}   {len(per_case)} cases\n")

    print("absolute saving against a cold start, per arm")
    arms = ["oracle_mesh", "oracle_bl", "or_proj_coarse", "nf_bl", "cartesian_128"]
    levels = {}
    for arm in arms:
        s = savings(per_case, arm)
        if not s:
            print(f"   {arm:>16}: not in this tree")
            continue
        v = np.array(list(s.values()))
        r = report(arm, v)
        levels[arm] = {**r, "per_case": s}
        print(f"   {arm:>16}: {100 * r['mean']:+7.1f}%  "
              f"[{100 * r['ci'][0]:+6.1f}, {100 * r['ci'][1]:+6.1f}]  "
              f"{r['wins']}/{r['n']}  p = {r['p']:.4f}")

    print("\none-variable contrasts, paired within case")
    out = {}
    for label, base, arm, moves in CONTRASTS:
        a, b = savings(per_case, base), savings(per_case, arm)
        shared = sorted(set(a) & set(b))
        if not shared:
            print(f"   {label:>28}: needs both {base} and {arm}; not in this tree")
            continue
        delta = np.array([b[c] - a[c] for c in shared])
        r = report(label, delta)
        out[label] = {**r, "baseline": base, "arm": arm, "moves": moves,
                      "per_case": {c: float(b[c] - a[c]) for c in shared}}
        sign = "costs" if r["mean"] < 0 else "gains"
        print(f"   {label:>28}: {100 * r['mean']:+7.1f} points  "
              f"[{100 * r['ci'][0]:+6.1f}, {100 * r['ci'][1]:+6.1f}]  "
              f"{r['wins']}/{r['n']} cases improve  p = {r['p']:.4f}")
        print(f"   {'':>28}  {moves}  ({sign})")

    payload = {"scored": args.scored, "row": args.row,
               "readable": entry.get("readable"),
               "n_cases": len(per_case), "levels": levels, "contrasts": out}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
