"""A warm start you can switch off before it costs you anything.

Every result in this track has the same shape: a seed helps on one quantity and
hurts on another, and *which* depends on the seed, the case and the depth. That
is honest and it is unusable. A CFD engineer cannot adopt a technique whose sign
they only learn after paying for the solve.

So bound it. Run ``K`` probe iterations from the seed, look at the residual, and
either continue or throw the seed away and start cold. The probe iterations are
not wasted when the seed is kept -- they are the first ``K`` iterations of the
warm solve -- so the accounting is:

    accepted   cost = warm iterations               (probe included)
    rejected   cost = K + cold iterations

which bounds the worst case at ``(1 + K/N_cold)`` times a cold solve, whatever
the seed does. That is the property
[PCGBandit](https://arxiv.org/html/2509.08765) sells as "never worse than the
default", and it is what makes a mixed result deployable.

**The rule may not look at the cold run.** In production there isn't one -- that
is the whole point of warm starting. So the decision uses only what the probe
itself produces: the residual after ``K`` iterations, and how far it has fallen
from the seed's own starting residual. A threshold on those is calibrated on
other cases and tested on the held-out one (leave-one-case-out), so the reported
capture is not the threshold fitting the answer it is scored on.

Reports, per ``K``:

* **capture** -- the fraction of the saving an oracle gatekeeper would get (one
  that knows each arm's outcome in advance and admits exactly the winners).
* **worst case** -- the largest loss any single (case, arm) suffers.
* **admitted harm** -- how many losing seeds slipped through.

No solver runs; the traces are already on disk.

Usage
-----
    python scripts/certificate.py --root runs/openfoam/repr3 --exclude naca4412
    python scripts/certificate.py --root runs/openfoam/repr3 --metric residual
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import openfoam as of, scoring as sc

PROBES = (25, 50, 100, 200, 400)
FIELD = "Ux"


def trace(case_dir: str):
    """Per-iteration ``Ux`` residuals and the force history for one run."""
    log = os.path.join(case_dir, "log.simpleFoam")
    if not os.path.isfile(log):
        return None, None
    with open(log, encoding="utf-8", errors="replace") as fh:
        parsed = of.parse_simple_foam_log(fh.read())
    res = parsed["residuals"].get(FIELD, [])
    try:
        forces = of.read_force_coeffs(case_dir)
        parts = of.read_force_components(case_dir)
        for k, v in parts.items():
            forces.setdefault(k, v)
    except Exception:
        forces = None
    return np.asarray(res, dtype=float), forces


def features(res: np.ndarray, k: int) -> dict | None:
    """What the probe knows at iteration ``k``. Never the cold run."""
    if res is None or len(res) < k + 1:
        return None
    window = res[:k + 1]
    finite = window[np.isfinite(window) & (window > 0)]
    if finite.size < 2:
        return None
    return {
        # Where the residual has got to. A seed that is genuinely closer to the
        # answer sits lower here than one that has injected an error the solver
        # must now remove.
        "level": float(np.log10(finite[-1])),
        # How far it has fallen from where the seed started it. Catches the seed
        # that starts low because it is smooth and rises once the solver touches
        # it.
        "drop": float(np.log10(finite[-1]) - np.log10(finite[0])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "repr3"))
    ap.add_argument("--cold", default="cold")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--arms", nargs="*", default=None,
                    help="default: every arm present except the cold reference")
    ap.add_argument("--metric", default="Cd", help="Cd, Cd_v, Cl, or 'residual'")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--residual-target", type=float, default=5e-6)
    ap.add_argument("--probes", type=int, nargs="*", default=list(PROBES))
    ap.add_argument("--json", default=os.path.join("results", "certificate.json"))
    args = ap.parse_args(argv)

    dirs = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d)))
    pairs = [(d[: -len("_" + args.cold)], args.cold) for d in dirs
             if d.endswith("_" + args.cold)]
    cases = [c for c, _ in pairs if not any(x in c for x in args.exclude)]
    if not cases:
        print(f"no '<case>_{args.cold}' directories under {args.root}")
        return 1
    arms = args.arms or sorted({d[len(c) + 1:] for d in dirs for c in cases
                                if d.startswith(c + "_")} - {args.cold})

    # --- what each run did, and how long it took to get there ------------------
    res, forces, budget = {}, {}, {}
    for c in cases:
        for a in [args.cold] + arms:
            r, f = trace(os.path.join(args.root, f"{c}_{a}"))
            if r is None:
                continue
            res[(c, a)], forces[(c, a)], budget[(c, a)] = r, f, len(r)

    def target(c, a):
        """Iterations for this run to reach the metric, or None."""
        if args.metric == "residual":
            return of.iterations_to_threshold(
                {FIELD: list(res[(c, a)])}, args.residual_target, fields=(FIELD,))
        d = forces.get((c, a))
        if not d or args.metric not in d:
            return None
        finals = {b: float(forces[(c, b)][args.metric][-1])
                  for b in [args.cold] + arms
                  if forces.get((c, b)) and args.metric in forces[(c, b)]}
        settled = [b for b, _ in finals.items()
                   if sc.has_settled(forces[(c, b)][args.metric], args.tol)]
        ref, _, _ = sc.settled_reference(finals, settled)
        if ref is None:
            return None
        return of.iterations_to_force_band(d["Time"], d[args.metric],
                                           reference=ref, tol=args.tol)

    cold_n = {c: target(c, args.cold) for c in cases}
    cases = [c for c in cases if cold_n.get(c)]
    if not cases:
        print("no case has a readable cold reference for this metric")
        return 1

    # One record per (case, arm): what it would cost with and without the gate.
    records = []
    for c in cases:
        for a in arms:
            if (c, a) not in res:
                continue
            n = target(c, a)
            warm = n if n else budget[(c, a)] + cold_n[c]   # never got there
            records.append({"case": c, "arm": a, "cold": cold_n[c],
                            "warm": float(warm), "reached": bool(n),
                            "helpful": bool(warm < cold_n[c])})
    if not records:
        print("no arms to certify")
        return 1

    print(f"{len(cases)} cases x {len(arms)} arms = {len(records)} seeds "
          f"| metric {args.metric}"
          + (f"@{100 * args.tol:g}%" if args.metric != "residual"
             else f" {args.residual_target:.0e}"))
    good = [r for r in records if r["helpful"]]
    print(f"{len(good)} of {len(records)} seeds are actually helpful; "
          f"ungated mean saving "
          f"{100 * np.mean([1 - r['warm'] / r['cold'] for r in records]):+.1f}%, "
          f"worst single seed "
          f"{100 * min(1 - r['warm'] / r['cold'] for r in records):+.1f}%")

    # The gatekeeper that knows the future -- the ceiling any rule can reach.
    oracle_saving = float(np.mean([max(0.0, 1 - r["warm"] / r["cold"])
                                   for r in records]))
    print(f"an oracle gatekeeper (admits exactly the winners, no probe cost) "
          f"would save {100 * oracle_saving:+.1f}%\n")

    print(f"{'K':>5} {'feature':>8} {'capture':>9} {'saving':>8} {'worst':>8} "
          f"{'admitted harm':>14} {'bound':>8}")
    print("-" * 68)
    out = {"root": args.root, "metric": args.metric, "tol": args.tol,
           "n_records": len(records), "oracle_saving": oracle_saving, "rules": []}

    for k in args.probes:
        for name in ("level", "drop"):
            feats, keep = {}, []
            for r in records:
                f = features(res[(r["case"], r["arm"])], k)
                if f is None:
                    continue
                feats[(r["case"], r["arm"])] = f[name]
                keep.append(r)
            if len(keep) < 4:
                continue

            # Leave-one-case-out: the threshold is chosen on the other cases, so
            # what is reported is not the rule fitting the data it is scored on.
            costs, admitted_harm = [], 0
            for r in keep:
                others = [q for q in keep if q["case"] != r["case"]]
                grid = sorted({feats[(q["case"], q["arm"])] for q in others})
                best, tau = -np.inf, None
                for t in grid:
                    total = 0.0
                    for q in others:
                        accept = feats[(q["case"], q["arm"])] <= t
                        total += (q["warm"] if accept else k + q["cold"]) / q["cold"]
                    score = 1 - total / len(others)
                    if score > best:
                        best, tau = score, t
                if tau is None:
                    continue
                accept = feats[(r["case"], r["arm"])] <= tau
                cost = r["warm"] if accept else k + r["cold"]
                costs.append(1 - cost / r["cold"])
                admitted_harm += int(accept and not r["helpful"])

            if not costs:
                continue
            saving = float(np.mean(costs))
            row = {"K": k, "feature": name, "saving": saving,
                   "capture": saving / oracle_saving if oracle_saving else float("nan"),
                   "worst": float(min(costs)), "admitted_harm": admitted_harm,
                   "n": len(costs),
                   "bound": -k / float(np.mean([r["cold"] for r in keep]))}
            out["rules"].append(row)
            print(f"{k:>5} {name:>8} {100 * row['capture']:8.1f}% "
                  f"{100 * saving:+7.1f}% {100 * row['worst']:+7.1f}% "
                  f"{admitted_harm:>9}/{len(costs) - sum(r['helpful'] for r in keep):<4} "
                  f"{100 * row['bound']:+7.1f}%")

    print("\ncapture  = fraction of the saving a gatekeeper that knew the outcome "
          "would get\nworst    = the largest loss any single seed still suffers "
          "under the rule\nbound    = -K/N_cold, the guarantee: no solve can cost "
          "more than this over cold\n         (a rejected seed pays K probe "
          "iterations and then runs cold)")

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(args.json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
