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

from neuroforge.solver import openfoam as of, scoring as sc

DEPTHS = (1e-2, 1e-3, 1e-4, 1e-5, 5e-6, 1e-6, 5e-7)
KNOWN_ARMS = ("oracle_mesh", "cartesian_128", "fitted_256x64", "fitted_outer",
              "fitted_p", "fitted_bl", "composite", "potential", "nf_mesh", "nf_bl",
              # The mesh-native control and the channel split. `nf_bl_proj` is
              # the same network prediction as `nf_bl`, sent through the same
              # 256x64 round-trip the fitted arms use, so the pair isolates
              # resampling from the source of the field.
              "nf_bl_proj", "nf_bl_nut", "nf_bl_vel", "oracle_wake",
              "oracle_128_hybrid", "oracle_128", "oracle_192",
              "oracle_256", "oracle_320", "oracle_421", "neighbour",
              "oracle",
              "cold")

# Relative bands for the force metric. Unlike a residual threshold this does not
# stop meaning anything when the residual stalls, so it is the one to read when
# the two disagree.
FORCE_TOLS = (0.01, 0.005, 0.002)
FORCE_REF_FLOOR = 1e-3   # a relative band around zero lift is numerical noise

# Total drag and lift, then their pressure and viscous parts. The split is
# what separates the two: at these Reynolds numbers drag is mostly wall shear
# and lift is mostly pressure, so a seed that gets the pressure field right
# while corrupting the near-wall velocity gradient converges one fast and the
# other slowly -- a prediction these columns test directly.
COEFFS = ("Cd", "Cl", "Cd_p", "Cd_v", "Cl_p")


def cell(entry, width: int = 20) -> str:
    """One table cell: the bounded mean, with a mark when it rests on a bound."""
    return (str(entry) if entry is not None else "--").rjust(width)


def key(threshold: float) -> str:
    """JSON key for a depth.

    ``f"{1.5e-5:.0e}"`` is ``"2e-05"`` -- one significant figure rounds 1.5e-5
    onto 2e-5 and silently overwrites that entry. One decimal place keeps every
    rung on the ladder distinct.
    """
    return f"{threshold:.1e}"


def parsed(case_dir: str):
    log = os.path.join(case_dir, "log.simpleFoam")
    if not os.path.isfile(log):
        return None
    with open(log, encoding="utf-8", errors="replace") as fh:
        return of.parse_simple_foam_log(fh.read())


def residuals(case_dir: str):
    info = parsed(case_dir)
    return info["residuals"] if info else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join("runs", "openfoam", "repr"))
    ap.add_argument("--cold", default="cold", help="suffix of the baseline arm")
    ap.add_argument("--out", default=os.path.join("results", "depth_reanalysis.json"))
    ap.add_argument("--per-case", action="store_true",
                    help="also print each case separately, so the spread is visible")
    ap.add_argument("--stats", nargs="*", default=[], metavar="ARM",
                    help="mean, bootstrap 95%% CI, win count and sign-test p for "
                         "these arms on every readable force row. Anything going "
                         "in a paper needs this, not the mean alone.")
    ap.add_argument("--exclude", action="append", metavar="SUBSTRING", default=[],
                    help="drop cases whose name contains this (repeatable). For a "
                         "case whose steady solve has no unique fixed point -- its "
                         "arms converge to different force coefficients -- which "
                         "is a property of the case, not of the seeds, and which "
                         "poisons every aggregate it appears in. Always report "
                         "the exclusion and the reason.")
    ap.add_argument("--filter", metavar="SUBSTRING",
                    help="score only the cases whose name contains this. A "
                         "Reynolds sweep must be scored one Reynolds number at a "
                         "time: the residual floor moves three orders of "
                         "magnitude across it, so a mean 'x floor' over the whole "
                         "tree describes none of the points in it.")
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
    dropped = sorted(c for c in cases if any(x in c for x in args.exclude))
    cases = sorted(c for c in cases
                   if (not args.filter or args.filter in c)
                   and not any(x in c for x in args.exclude))
    if dropped:
        print("excluded: " + ", ".join(dropped))
    if not cases:
        print(f"no recognisable case directories under {args.root}")
        return 1

    info = {(c, a): parsed(os.path.join(args.root, f"{c}_{a}"))
            for c in cases for a in [args.cold] + arms}
    hist = {k: (v["residuals"] if v else None) for k, v in info.items()}
    # Cumulative solver seconds per iteration. A warm start shortens the inner
    # linear solves too, so an iteration saving understates the cost saving --
    # but only a run with the machine to itself gives a clean wall-clock number,
    # and these sweeps run many solves at once. Reported, flagged, not claimed.
    elapsed = {k: (v["elapsed"] if v else None) for k, v in info.items()}

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

    # An arm that stopped short of the tree's budget is *unfinished*, not
    # censored: scoring it against its own truncated length would read a run the
    # power cut interrupted as a catastrophic failure. Only arms that used the
    # full budget can bound a target they never reached.
    lengths = {(c, a): len(hist[(c, a)].get("Ux", ())) if hist[(c, a)] else 0
               for c in cases for a in [args.cold] + arms}
    full = max(lengths.values(), default=0)
    budgets = {k: (v if v >= 0.9 * full else 0) for k, v in lengths.items()}
    unfinished = sorted(f"{c}_{a} ({lengths[(c, a)]})"
                        for (c, a), v in budgets.items() if v == 0 and lengths[(c, a)])
    if unfinished:
        print(f"\nstill short of the {full}-iteration budget, left unscored: "
              + ", ".join(unfinished))

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
            entry[a] = sc.bounded_saving(savings[a], censored[a])
        out["by_depth"][key(t)] = {k: (vars(v) if isinstance(v, sc.Saving) else v)
                                   for k, v in entry.items()}

        ratio = f"{t / mean_floor:.1f}x" if mean_floor else "--"
        cold_txt = f"{entry['cold_mean']:.0f}" if colds else "--"
        print(f"{key(t):>8} {ratio:>8} {cold_txt:>8} "
              + "  ".join(cell(entry[a]) for a in arms))

    # --- the same comparison on the forces ------------------------------------
    forces = {}
    for c in cases:
        for a in [args.cold] + arms:
            d = os.path.join(args.root, f"{c}_{a}")
            merged = dict(of.read_force_coeffs(d))
            # Cd(f)/Cd(r) in coefficient.dat are the front and rear halves about
            # CofR, not the viscous and pressure parts. The separate `forces`
            # object gives those, and they are what separates the two
            # coefficients: drag here is mostly wall shear, lift mostly pressure.
            parts = of.read_force_components(d)
            for name in ("Cd_p", "Cd_v", "Cl_p", "Cl_v"):
                if name in parts:
                    merged[name] = parts[name]
            forces[(c, a)] = merged
    scored = any(forces.values())
    out["by_force"] = {}
    if scored:
        # One reference per case, shared by its arms: the **median** of the arms'
        # final values. Every arm solves the same steady problem on the same mesh
        # and must land on the same coefficient, so the median is the best
        # estimate of it and is robust to one straggler.
        #
        # Taking the oracle arm's own final instead -- the obvious choice -- makes
        # the oracle's score an artifact: an arm graded against where it itself
        # stopped is measuring how it approached its own asymptote, not how fast
        # it converged. That reads as a failed control when nothing is wrong with
        # the data, which is worse than no control at all.
        # The cohort that defines the reference is the arms that have *stopped
        # moving* (`scoring.has_settled`), not every arm. One diverged arm used to
        # drag the spread to 3.1% and condemn the whole force ladder, including
        # arms that agree with each other to 0.1%. An unsettled arm is named
        # below and still scored, at its full budget, which bounds it rather than
        # excusing it; what it may not do is move the reference.
        tightest = min(FORCE_TOLS)
        refs, spread, unsettled = {}, {}, {}
        for c in cases:
            finals, settled = {}, {}
            for a in [args.cold] + arms:
                d = forces.get((c, a))
                if not d:
                    continue
                for n in COEFFS:
                    if n in d and len(d[n]):
                        finals.setdefault(n, {})[a] = float(d[n][-1])
                        if sc.has_settled(d[n], tightest):
                            settled.setdefault(n, []).append(a)
            refs[c], spread[c], unsettled[c] = {}, {}, {}
            for n, v in finals.items():
                r, sp, un = sc.settled_reference(v, settled.get(n, []))
                refs[c][n], spread[c][n], unsettled[c][n] = r, sp, un
        out["force_reference"] = refs
        out["force_reference_spread"] = spread
        out["force_unsettled"] = unsettled

        # The check that decides whether a band is measurable at all: if the arms
        # disagree about the answer by more than the band, an arm can sit outside
        # it forever and the metric reports a convergence failure that is really a
        # budget failure.
        worst = max((v.get("Cd", 0.0) for v in spread.values()), default=0.0)
        print(f"\nsettled arms' final Cd spread about the reference: {100 * worst:.3f}% "
              f"(tightest band {100 * tightest:g}%)"
              + ("  <-- too wide to score: raise the budget"
                 if worst > 0.5 * tightest else "  ok"))
        # Name the arms the cohort excluded. They are still scored, at their full
        # budget; what they are not allowed to do is move the reference.
        never = {}
        for c in cases:
            for a in unsettled[c].get("Cd", []):
                never.setdefault(a, []).append(c)
        if never:
            print("  Cd never settled (bounded at full budget, excluded from the "
                  "reference): "
                  + ";  ".join(f"{a} on {len(v)}/{len(cases)}"
                               for a, v in sorted(never.items())))

        print()
        header = f"{'force':>12} {'cold it':>8} " + "  ".join(f"{a:>20}" for a in arms)
        print(header)
        print("-" * len(header))
        for coeff in COEFFS:
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
                         "cold_n": len(colds), "per_case": per_case,
                         # Kept so the mean can be given an interval and a sign
                         # test below: on five to twenty cases the per-case
                         # spread is wider than the gap between arms, and a mean
                         # quoted bare reads as tighter than it is.
                         "values": {a: savings[a] + censored[a] for a in arms}}
                for a in arms:
                    entry[a] = sc.bounded_saving(savings[a], censored[a])
                # Readability, per coefficient and per band. The settled arms
                # have to agree about the answer to well inside the band they are
                # being timed against; where they do not, the row records how
                # long arms sat just outside a band placed on a reference that is
                # not yet pinned down that finely. Measured here, Cd@1% is
                # readable, Cd@0.5% is not, and the two disagree in sign.
                worst_c = max((spread[c].get(coeff, 0.0) for c in cases
                               if c in per_case), default=0.0)
                readable = worst_c <= sc.MAX_SPREAD_FRACTION * tol
                entry["settled_spread"] = float(worst_c)
                entry["readable"] = bool(readable)
                out["by_force"][f"{coeff}@{tol:g}"] = {
                    k: (vars(v) if isinstance(v, sc.Saving) else v)
                    for k, v in entry.items()}
                if not colds:
                    continue

                print(f"{coeff + '@' + f'{100 * tol:g}%':>12} {entry['cold_mean']:>8.0f} "
                      + "  ".join(cell(entry[a]) for a in arms)
                      + ("" if readable
                         else f"   ! unreadable: settled arms disagree by "
                              f"{100 * worst_c:.2f}% on a {100 * tol:g}% band"))

    # --- what the mean is hiding ---------------------------------------------
    # A headline saving is an average over a handful of cases whose per-case
    # spread is wider than the difference between arms. Anything going in a
    # paper needs its interval and its win count next to it.
    if args.stats and out.get("by_force"):
        print("\nper-case statistics for the arms named with --stats")
        print(f"{'row':>14} {'arm':>14} {'mean':>8} {'95% CI':>18} {'wins':>7} "
              f"{'p':>7}  per case")
        for row_name, entry in out["by_force"].items():
            if not entry.get("readable") or not entry.get("values"):
                continue
            for a in args.stats:
                vals = entry["values"].get(a)
                if not vals:
                    continue
                lo, hi = sc.bootstrap_ci(vals)
                p = sc.sign_test(vals)
                wins = sum(1 for v in vals if v > 0)
                print(f"{row_name:>14} {a:>14} {100 * np.mean(vals):+7.1f}% "
                      f"[{100 * lo:+6.1f}, {100 * hi:+6.1f}] {wins:>3}/{len(vals):<3} "
                      f"{p:7.3f}  "
                      + " ".join(f"{100 * v:+.0f}" for v in sorted(vals)))
        print("  Only readable rows are shown. 'wins' counts cases where the arm "
              "beat cold;\n  p is a two-sided exact sign test on that count, which "
              "answers 'does this\n  help at all' without letting one catastrophic "
              "case decide the mean.")
        # A p that cannot reach 0.05 is not weak evidence, it is an unanswerable
        # question, and the two read identically in a table. Say which it is.
        widest = max((len(v) for e in out["by_force"].values()
                      for v in (e.get("values") or {}).values()), default=0)
        if 0 < widest < 6:
            print(f"  With {widest} cases the smallest attainable two-sided p is "
                  f"{2 / 2 ** widest:.3f}: no result here\n  can be significant by "
                  "this test whatever the effect. That is a statement about the\n"
                  "  corpus, not about the effect -- see PLANS.md Phase B.")

    # --- cost per iteration, the part an iteration count cannot show ----------
    rates = {}
    for c in cases:
        for a in [args.cold] + arms:
            e = elapsed.get((c, a))
            if e and len(e) > 1 and np.isfinite(e[-1]):
                rates[(c, a)] = e[-1] / len(e)
    if rates:
        print("\nsolver seconds per iteration (machine was shared -- indicative only)")
        base = [rates[(c, args.cold)] for c in cases if (c, args.cold) in rates]
        line = f"{'':>12} {np.mean(base):>8.3f} " if base else f"{'':>12} {'--':>8} "
        for a in arms:
            v = [rates[(c, a)] for c in cases if (c, a) in rates]
            line += f"{(f'{np.mean(v):.3f} s' if v else '--'):>20}  "
        print(f"{'per iter':>12} {'cold':>8}" + "".join(f"{a:>22}" for a in arms))
        print(line)
        out["seconds_per_iteration"] = {
            f"{c}_{a}": float(v) for (c, a), v in rates.items()}

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
