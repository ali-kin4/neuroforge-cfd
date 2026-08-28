"""Re-score finished warm-start runs at several convergence depths.

The probes in this repo first reported iteration savings at residual 1e-3. On the
C-grid at Re 3e6 a cold start reaches 1e-3 in **35 iterations** and then spends
hundreds more crawling towards a floor near 1e-5, so 1e-3 is roughly a tenth of
the way through the solve -- an early checkpoint, not convergence. The warm-start
literature reports *iterations to convergence*, and the two disagree in sign::

    depth    cold it   oracle_mesh   cartesian_128   fitted_256x64
    1e-2         13        +92.1%          -20.6%         -104.4%
    1e-3         35        +73.6%          -30.2%          -70.8%
    1e-4         74        +64.0%         -366.4%          -42.9%
    5e-5         95        +65.2%         -140.9%          -79.6%
    3e-5        273        +84.3%          -33.6%          +15.0%
    2e-5        347        +86.1%          -21.7%          +31.2%
    1.5e-5      458        +88.0%           -9.0%          +13.5%

Early iterations are dominated by the global transient, where a structured but
imperfect seed costs; deeper convergence is dominated by the near-wall state,
where a wall-fitted seed is right and a uniform one is hopeless.

**Read the `floor` column before believing any row.** The deepest thresholds sit
within a factor of two or three of where ``simpleFoam`` stops making progress on
this mesh, and an iteration count taken off a flat curve is a reading of where
the curve happens to cross a line, not a measurement of convergence. That the
same arm scores +15%, +31% and +13% at three adjacent thresholds is the
symptom. ``scripts/convergence_diagnostic.py`` chases the floor itself.

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

DEPTHS = (1e-2, 1e-3, 1e-4, 1e-5, 5e-6, 1e-6, 5e-7)
KNOWN_ARMS = ("oracle_mesh", "cartesian_128", "fitted_256x64",
              "oracle_128", "oracle_128_hybrid", "neighbour", "oracle", "cold")

# Relative bands for the force metric. Unlike a residual threshold this does not
# stop meaning anything when the residual stalls, so it is the one to read when
# the two disagree.
FORCE_TOLS = (0.01, 0.005, 0.002)
FORCE_REF_FLOOR = 1e-3   # a relative band around zero lift is numerical noise


def summarise(savings, censored, n_cases):
    """Mean saving over the arms that reached the target, plus the censored ones.

    An arm that never reaches the target inside its budget is *worse* than one
    that reaches it late, so dropping it silently raises that arm's own mean --
    the failing arm is rewarded for failing. Every such case is instead scored
    with the arm's full run length, which is a lower bound on the iterations it
    would have needed and therefore an **upper bound** on its saving: the arm is
    at least this bad. `n_reached` and `n_censored` are reported so a bound is
    never mistaken for a measurement.
    """
    values = list(savings) + list(censored)
    return {
        "saving": float(np.mean(values)) if values else None,
        "saving_reached_only": float(np.mean(savings)) if savings else None,
        "n": len(values),
        "n_cases": n_cases,
        "n_reached": len(savings),
        "n_censored": len(censored),
        "spread": [float(min(values)), float(max(values))] if values else None,
    }


def cell(entry: dict, width: int = 20) -> str:
    """One table cell: the bounded mean, with a mark when it rests on a bound."""
    if entry is None or entry["saving"] is None:
        return f"{'--':>{width}}"
    mark = "<" if entry["n_censored"] else " "
    return (f"{mark}{100 * entry['saving']:+8.1f}% "
            f"({entry['n_reached']}/{entry['n']})").rjust(width)


def key(threshold: float) -> str:
    """JSON key for a depth.

    ``f"{1.5e-5:.0e}"`` is ``"2e-05"`` -- one significant figure rounds 1.5e-5
    onto 2e-5 and silently overwrites that entry. One decimal place keeps every
    rung on the ladder distinct.
    """
    return f"{threshold:.1e}"


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
    ap.add_argument("--per-case", action="store_true",
                    help="also print each case separately, so the spread is visible")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"no such directory: {args.root}")
        return 1

    # Split "<case>_<arm>" on the known arm suffixes present in the tree.
    arms, cases = set(), set()
    for e in sorted(os.listdir(args.root)):
        for arm in KNOWN_ARMS:
            if e.endswith("_" + arm):
                cases.add(e[: -len(arm) - 1])
                arms.add(arm)
                break
    arms = [a for a in KNOWN_ARMS if a in arms and a != args.cold]
    cases = sorted(cases)
    if not cases:
        print(f"no recognisable case directories under {args.root}")
        return 1

    hist = {}
    for c in cases:
        for a in [args.cold] + arms:
            hist[(c, a)] = residuals(os.path.join(args.root, f"{c}_{a}"))

    floors = {c: of.residual_floor(hist[(c, args.cold)])
              for c in cases if hist[(c, args.cold)]}
    out = {"root": args.root, "cases": cases, "arms": arms,
           "cold_floor": {c: float(v) for c, v in floors.items()},
           "by_depth": {}}

    print(f"{len(cases)} cases | arms: {', '.join(arms)}")
    print("cold residual floor: " + "  ".join(f"{c}={v:.2e}" for c, v in floors.items()))
    print()
    header = (f"{'depth':>8} {'x floor':>8} {'cold it':>8} "
              + "  ".join(f"{a:>20}" for a in arms))
    print(header)
    print("-" * len(header))

    budgets = {(c, a): len(hist[(c, a)].get("Ux", ())) if hist[(c, a)] else 0
               for c in cases for a in [args.cold] + arms}

    mean_floor = float(np.mean(list(floors.values()))) if floors else float("nan")
    for t in DEPTHS:
        colds = []
        savings = {a: [] for a in arms}
        censored = {a: [] for a in arms}
        per_case = {}
        for c in cases:
            hc = hist[(c, args.cold)]
            if hc is None:
                continue
            base = of.iterations_to_threshold(hc, t)
            if not base:
                continue
            colds.append(base)
            per_case[c] = {"cold": base}
            for a in arms:
                ha = hist[(c, a)]
                v = of.iterations_to_threshold(ha, t) if ha else None
                per_case[c][a] = v
                if v:
                    savings[a].append(1.0 - v / base)
                elif budgets[(c, a)]:
                    # Never got there inside its budget: bound it, do not drop it.
                    censored[a].append(1.0 - budgets[(c, a)] / base)
        entry = {"cold_mean": float(np.mean(colds)) if colds else None,
                 "cold_n": len(colds), "per_case": per_case,
                 "threshold_over_floor": t / mean_floor if mean_floor else None}
        for a in arms:
            entry[a] = summarise(savings[a], censored[a], len(colds))
        out["by_depth"][key(t)] = entry

        ratio = f"{t / mean_floor:.1f}x" if mean_floor else "--"
        cold_txt = f"{entry['cold_mean']:.0f}" if colds else "--"
        print(f"{key(t):>8} {ratio:>8} {cold_txt:>8} "
              + "  ".join(cell(entry[a]) for a in arms))

    # --- the same comparison on the forces ------------------------------------
    forces = {(c, a): of.read_force_coeffs(os.path.join(args.root, f"{c}_{a}"))
              for c in cases for a in [args.cold] + arms}
    scored = any(forces.values())
    out["by_force"] = {}
    if scored:
        # One reference per case, shared by its arms: the oracle arm's converged
        # coefficient where there is one, else the cold arm's. Scoring each arm
        # against its own final would grade a warm start on wherever it stopped.
        refs = {}
        for c in cases:
            for a in ("oracle_mesh", "oracle", args.cold):
                d = forces.get((c, a))
                if d and "Cd" in d and len(d["Cd"]):
                    refs[c] = {n: float(d[n][-1]) for n in ("Cd", "Cl") if n in d}
                    break
        out["force_reference"] = refs

        print()
        header = f"{'force':>12} {'cold it':>8} " + "  ".join(f"{a:>20}" for a in arms)
        print(header)
        print("-" * len(header))
        for coeff in ("Cd", "Cl"):
            for tol in FORCE_TOLS:
                colds = []
                savings = {a: [] for a in arms}
                censored = {a: [] for a in arms}
                per_case = {}
                for c in cases:
                    ref = (refs.get(c) or {}).get(coeff)
                    if ref is None or abs(ref) < FORCE_REF_FLOOR:
                        continue

                    def band(arm):
                        d = forces.get((c, arm))
                        if not d or coeff not in d:
                            return None
                        return of.iterations_to_force_band(d["Time"], d[coeff],
                                                           reference=ref, tol=tol)

                    base = band(args.cold)
                    if not base:
                        continue
                    colds.append(base)
                    per_case[c] = {"cold": base}
                    for a in arms:
                        v = band(a)
                        per_case[c][a] = v
                        if v:
                            savings[a].append(1.0 - v / base)
                        elif budgets[(c, a)]:
                            censored[a].append(1.0 - budgets[(c, a)] / base)
                entry = {"cold_mean": float(np.mean(colds)) if colds else None,
                         "cold_n": len(colds), "per_case": per_case}
                for a in arms:
                    entry[a] = summarise(savings[a], censored[a], len(colds))
                out["by_force"][f"{coeff}@{tol:g}"] = entry
                if not colds:
                    continue

                print(f"{coeff + '@' + f'{100 * tol:g}%':>12} {entry['cold_mean']:>8.0f} "
                      + "  ".join(cell(entry[a]) for a in arms))

    if args.per_case:
        print("\niterations per case (spread is the point)")
        for t in DEPTHS:
            pc = out["by_depth"][key(t)]["per_case"]
            if not pc:
                continue
            print(f"  {key(t)}")
            for c, row in pc.items():
                print(f"    {c:>18} " + "  ".join(
                    f"{a}={row.get(a) or '--'}" for a in ["cold"] + arms))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = os.path.abspath(args.out) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, os.path.abspath(args.out))
    print(f"\nwrote {args.out}")
    print("\nCells read  saving (reached/total).  A leading '<' means at least one\n"
          "case never reached the target inside its budget and was scored with its\n"
          "full run length: the arm is at least that bad, so the number is a bound.\n"
          "\nThe sign changes with depth, so report the depth alongside any saving --\n"
          "and check 'x floor': a threshold within a few times the stagnation level\n"
          "is measuring where a flat curve crosses a line, not a convergence rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
