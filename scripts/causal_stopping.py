"""Two things the headline metric does not tell you, both free from disk.

``iterations_to_force_band`` -- the metric behind every saving in this paper --
has two properties a reviewer is right to want stated rather than inferred.

**It is a last-exit statistic.** It walks back from the end of the run over the
maximal in-band run, so it reports the iteration after which the coefficient
*stays* inside the band. That is deliberate: a coefficient sweeping through its
final value on the way past would otherwise score as converged at the crossing.
But it means a large number can mean either "arrived late" or "arrived early and
wandered out once", and those are different failures. This reports **first
entry** alongside it, so the gap between them is visible.

**It is not causal.** The band is centred on the *converged* value, which in
production nobody has -- that is the entire point of warm starting. So the
metric answers "how many iterations did this seed need", which is the right
question for a mechanism, and not "when would a practitioner have stopped",
which is the right question for a user. This re-scores every arm under a
**causal** stopping rule that uses only the history up to the current iteration:

    stop at the first i >= W with  max |C(j) - C(i)| <= tol * |C(i)|  for
    j in [i - W, i]

-- the coefficient has not moved by more than ``tol`` over the trailing window.
Decidable at iteration ``i`` from that run alone, with no knowledge of the
answer and no cold run to compare against.

If the two orderings agree, the paper's savings survive translation into a rule
a practitioner could actually run. If they do not, that is the more important
result and it belongs in the paper.

No solver runs.

Usage
-----
    python scripts/causal_stopping.py --root runs/openfoam/corpus
    python scripts/causal_stopping.py --root runs/openfoam/corpus --coeff Cd_v --window 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import openfoam as of, scoring as sc


def read_series(case_dir: str, coeff: str):
    """``(time, values)`` for one coefficient, or ``None``."""
    for reader in (of.read_force_components, of.read_force_coeffs):
        try:
            d = reader(case_dir)
        except Exception:
            continue
        v = d.get(coeff)
        if v is not None and len(v):
            return np.asarray(d["Time"], dtype=float), np.asarray(v, dtype=float)
    return None


def first_entry(time, values, reference: float, tol: float):
    """First iteration at which the coefficient enters the band, ever."""
    inside = np.abs(values - reference) <= tol * abs(reference)
    hit = np.flatnonzero(inside)
    return int(round(float(time[hit[0]]))) if hit.size else None


def causal_stop(time, values, tol: float, window: int):
    """First iteration whose trailing window is flat to ``tol``. Uses no future."""
    n = len(values)
    if n <= window:
        return None
    for i in range(window, n):
        seg = values[i - window: i + 1]
        scale = abs(values[i])
        if scale < 1e-30:
            continue
        if np.max(np.abs(seg - values[i])) <= tol * scale:
            return int(round(float(time[i])))
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "corpus"))
    ap.add_argument("--coeff", default="Cd_v")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=50,
                    help="trailing window of the causal rule, in iterations")
    ap.add_argument("--cold", default="cold")
    ap.add_argument("--out", default=os.path.join("results", "causal_stopping.json"))
    args = ap.parse_args(argv)

    entries = sorted(os.listdir(args.root))
    tags = sorted(d[: -len("_" + args.cold)] for d in entries
                  if d.endswith("_" + args.cold))
    arms = sorted({d[len(t) + 1:] for d in entries for t in tags
                   if d.startswith(t + "_")} - {args.cold})
    if not tags:
        print(f"no '<case>_{args.cold}' directories under {args.root}")
        return 1
    print(f"{args.root}: {len(tags)} cases x {len(arms) + 1} arms | {args.coeff}"
          f"@{100 * args.tol:g}% | causal window {args.window}\n")

    rows = []
    for tag in tags:
        series = {}
        for arm in [args.cold] + arms:
            got = read_series(os.path.join(args.root, f"{tag}_{arm}"), args.coeff)
            if got is not None:
                series[arm] = got
        if args.cold not in series:
            continue
        # The same settled reference the paper uses, over the arms that settled.
        finals = {a: float(v[-1]) for a, (t, v) in series.items()}
        settled = [a for a in series if sc.has_settled(series[a][1], args.tol)]
        reference, spread, _ = sc.settled_reference(finals, settled)
        if reference is None:
            continue
        row = {"case": tag, "reference": reference, "spread": spread, "arms": {}}
        for arm, (t, v) in series.items():
            row["arms"][arm] = {
                "first_entry": first_entry(t, v, reference, args.tol),
                "last_exit": of.iterations_to_force_band(t, v, reference=reference,
                                                         tol=args.tol),
                "causal": causal_stop(t, v, args.tol, args.window),
            }
        rows.append(row)

    def savings(kind: str, arm: str):
        out = []
        for r in rows:
            a, c = r["arms"].get(arm, {}).get(kind), r["arms"][args.cold].get(kind)
            if a and c:
                out.append(1.0 - a / c)
        return np.array(out)

    print(f"   {'arm':>16} " + "  ".join(f"{k:>26}" for k in
                                         ("last exit (the paper's)",
                                          "first entry", "causal rule")))
    summary = {}
    for arm in arms:
        bits, entry = [], {}
        for kind in ("last_exit", "first_entry", "causal"):
            v = savings(kind, arm)
            if v.size:
                lo, hi = sc.bootstrap_ci(v)
                entry[kind] = {"mean": float(v.mean()), "ci": [float(lo), float(hi)],
                               "wins": int((v > 0).sum()), "n": int(v.size)}
                bits.append(f"{100 * v.mean():+7.1f}% [{100 * lo:+5.1f},{100 * hi:+5.1f}] "
                            f"{int((v > 0).sum())}/{v.size}")
            else:
                bits.append(" " * 26)
        summary[arm] = entry
        print(f"   {arm:>16} " + "  ".join(bits))

    # How far apart are the two non-causal statistics? That gap is the thing the
    # last-exit rule is protecting against, and it should be small if the curves
    # are settling rather than wandering.
    gaps = []
    for r in rows:
        for arm, e in r["arms"].items():
            if e["first_entry"] and e["last_exit"]:
                gaps.append(e["last_exit"] - e["first_entry"])
    if gaps:
        g = np.array(gaps, dtype=float)
        print(f"\nlast exit minus first entry, over {len(g)} (case, arm) pairs: "
              f"median {np.median(g):.0f} iterations, mean {g.mean():.0f}, "
              f"max {g.max():.0f}")
        print("a large gap means the coefficient entered the band and left it "
              "again; the paper's metric charges that, first entry does not.")

    payload = {"root": args.root, "coeff": args.coeff, "tol": args.tol,
               "window": args.window, "rows": rows, "summary": summary,
               "gap_iterations": {"median": float(np.median(gaps)) if gaps else None,
                                  "mean": float(np.mean(gaps)) if gaps else None,
                                  "max": float(np.max(gaps)) if gaps else None}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
