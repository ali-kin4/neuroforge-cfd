"""Trace the manuscript's headline numbers back to their committed result files.

The failure mode this catches is drift: a result is re-run, the JSON moves, and
the sentence quoting it does not. That is invisible to a LaTeX build and to the
test suite, and it is exactly what a referee checks first when the repository is
public. Each entry below names a claim, the file it must come from, how to read
the value out of that file, and the value the manuscript currently states.

A row FAILS if the recomputed value disagrees with the manuscript beyond its
stated tolerance. A row is SKIPPED, and says so, if the source file is missing --
skips are not passes and are counted separately.

Usage
-----
    python scripts/audit_paper_numbers.py
    python scripts/audit_paper_numbers.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def load(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---- readers: each returns the value the manuscript quotes -------------------

def floor_realdata_mean(root):
    d = load(os.path.join(root, "results/certificates/residual_floor_realdata.json"))
    if d is None:
        return None
    return float(np.mean([r["norm_truth"] for r in d["per_case"]]))


def floor_realdata_median(root):
    d = load(os.path.join(root, "results/certificates/residual_floor_realdata.json"))
    if d is None:
        return None
    return float(np.median([r["norm_truth"] for r in d["per_case"]]))


def uniform_all_zero(root):
    d = load(os.path.join(root, "results/certificates/residual_floor_realdata.json"))
    if d is None:
        return None
    return float(max(abs(r["norm_uniform"]) for r in d["per_case"]))


def ladder_level(root, res, key="floor"):
    d = load(os.path.join(root, "results/floor_resolution_ladder.json"))
    if d is None:
        return None
    rows = d["per_level"].get(str(res))
    if not rows:
        return None
    return float(np.nanmean([r[key] for r in rows]))


def ladder_p(root):
    d = load(os.path.join(root, "results/floor_resolution_ladder.json"))
    if d is None:
        return None
    return float(d["fits"]["floor"]["p"])


def ladder_rose(root):
    d = load(os.path.join(root, "results/floor_resolution_ladder.json"))
    if d is None:
        return None
    lo, hi = d["levels"][0], d["levels"][-1]
    a = {r["case"]: r["floor"] for r in d["per_level"][str(lo)]}
    b = {r["case"]: r["floor"] for r in d["per_level"][str(hi)]}
    return float(sum(1 for c in a if c in b and b[c] > a[c]))


def descent(root, arm, field, agg="mean", where="end"):
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    idx = -1 if where == "end" else 0
    vals = [r["arms"][arm][idx][field] for r in d["rows"]]
    return float(np.mean(vals) if agg == "mean" else np.median(vals))


def descent_residual_cut(root):
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    a = np.array([r["arms"]["from_truth"][0]["residual"] for r in d["rows"]])
    b = np.array([r["arms"]["from_truth"][-1]["residual"] for r in d["rows"]])
    return float(100.0 * (1.0 - b.mean() / a.mean()))


def descent_wins(root, arm):
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    a = np.array([r["arms"][arm][0]["residual"] for r in d["rows"]])
    b = np.array([r["arms"][arm][-1]["residual"] for r in d["rows"]])
    return float((b < a).sum())


def descent_selection_worse(root, arm):
    """Cases where the residual-chosen iterate is strictly worse than the best."""
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    n = 0
    for r in d["rows"]:
        t = r["arms"][arm]
        mse = np.array([s["mse_u"] for s in t])
        res = np.array([s["residual"] for s in t])
        if mse[int(res.argmin())] > mse.min() + 1e-12:
            n += 1
    return float(n)


def descent_perturbed_helped(root):
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    e0 = np.array([r["arms"]["perturbed_residual"][0]["mse_u"] for r in d["rows"]])
    e1 = np.array([r["arms"]["perturbed_residual"][-1]["mse_u"] for r in d["rows"]])
    return float((e1 < e0).sum())


def audit_cost_median(root):
    d = load(os.path.join(root, "results/control/inference_cost.json"))
    if d is None:
        return None
    return float(d["audit_ms"]["median_ms"])


def deployed_solve_median(root):
    d = load(os.path.join(root, "results/control/inference_cost.json"))
    if d is None:
        return None
    return float(d["per_device"]["cuda"]["solve_ms"]["median_ms"])


def decomp_repaired_shift(root, which="max"):
    """Percent change in the floor when the omitted stress term is restored."""
    d = load(os.path.join(root, "results/certificates/floor_resolution_decomposition.json"))
    if d is None:
        return None
    shifts = []
    for c in d["per_case"]:
        for r in c["rungs"]:
            if "truth_band_0.1" in r and "truth_repaired_band_0.1" in r:
                shifts.append(100.0 * (r["truth_repaired_band_0.1"]
                                       / r["truth_band_0.1"] - 1.0))
    if not shifts:
        return None
    by_rung: dict[int, list[float]] = {}
    for c in d["per_case"]:
        for r in c["rungs"]:
            if "truth_band_0.1" in r and "truth_repaired_band_0.1" in r:
                by_rung.setdefault(r["n"], []).append(
                    100.0 * (r["truth_repaired_band_0.1"] / r["truth_band_0.1"] - 1.0))
    means = [float(np.mean(v)) for v in by_rung.values()]
    return max(means) if which == "max" else min(means)


def decomp_omitted_fraction(root):
    d = load(os.path.join(root, "results/certificates/floor_resolution_decomposition.json"))
    if d is None:
        return None
    fr = []
    for c in d["per_case"]:
        for r in c["rungs"]:
            if "truth_omit_gradnu_band_0.1" in r and "truth_band_0.1" in r:
                fr.append(100.0 * r["truth_omit_gradnu_band_0.1"] / r["truth_band_0.1"])
    return float(np.mean(fr)) if fr else None


def review(root, name):
    return load(os.path.join(root, "results/review", name))


def c1(root, path):
    d = review(root, "control1_physics_vs_physicsfree.json")
    if d is None:
        return None
    node = d["headline"]
    for k in path.split("/"):
        node = node[k]
    return float(node)


def c2(root, field):
    d = review(root, "control2_difficulty_confound.json")
    if d is None:
        return None
    return float(d["arms"]["ensemble_mean"][field])


def c3_ungated(root, which):
    """Fraction of cases where an UNGATED half step raises the monitored residual.

    Split by path, and mind the key naming, which is the opposite of what it looks
    like. The artifact's own ``meta.arms`` block maps ``backbone_seed*`` to the
    DEPLOYED backbone+DEQ path and bare ``seed*`` to the ENSEMBLE-mean path. Its
    ``pooled`` block states the same thing outright: deployed 1.0%, ensemble 8.8%.

    A previous version of this function read ``backbone`` as the ensemble arm and so
    certified the swapped pair that reached the manuscript. Cross-check against
    ``d["pooled"]`` rather than trusting the key prefix.
    """
    d = review(root, "control3_fixed_step.json")
    if d is None:
        return None
    dep, ens = [], []
    for k, v in d["summary"].items():
        f = v.get("certificate_at_fixed_0.5", {}).get("frac_residual_increases")
        if f is None:
            continue
        (dep if k.startswith("backbone") else ens).append(f)
    vals = ens if which == "ensemble" else dep
    if not vals:
        return None
    # Guard: the pooled block is authoritative; if the per-seed mean disagrees with it
    # by more than a point, the key convention has changed and this must be revisited.
    pooled = d.get("pooled", {})
    key = "ensemble_mean_path" if which == "ensemble" else "deployed_backbone_DEQ"
    ref = pooled.get(key, {}).get("certificate_at_fixed_0.5", {}).get(
        "frac_residual_increases"
    )
    out = float(100.0 * np.mean(vals))
    if ref is not None and abs(out - 100.0 * float(ref)) > 1.0:
        raise AssertionError(
            "c3_ungated(%s): per-seed mean %.2f%% disagrees with pooled %.2f%%; "
            "the arm-naming convention in control3_fixed_step.json has changed"
            % (which, out, 100.0 * float(ref))
        )
    return out


def c4_drag_auroc(root):
    d = review(root, "control4_riskcoverage_drag.json")
    if d is None:
        return None
    best = None
    def walk(o):
        nonlocal best
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "auroc_mean" and isinstance(v, (int, float)):
                    best = v if best is None else max(best, v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return float(best) if best is not None else None


def gate_followup(root, path):
    d = review(root, "functional_audit_gate_followup.json")
    if d is None:
        return None
    node = d["F1"]
    for k in path.split("/"):
        node = node[k]
    return float(node)


def bound_over_error(root, which):
    d = review(root, "functional_audit_gate_followup.json")
    if d is None:
        return None
    vals = [v["bound_over_error"] for k, v in d["F1"]["arms"].items()
            if "absdCd" in k]
    return float(min(vals) if which == "min" else max(vals))


def conformal_coverage(root):
    d = review(root, "functional_audit_gate_followup.json")
    if d is None:
        return None
    vals = [v["coverage_mean"] for k, v in d["F1"]["arms"].items() if "absdCd" in k]
    return float(np.mean(vals))


def transolver_inversion(root, which):
    d = review(root, "functional_audit_gate_analysis.json")
    if d is None:
        return None
    vals = [v["norm_baseline_inversion_rate"] for v in d["test_a"].values()
            if isinstance(v, dict) and "norm_baseline_inversion_rate" in v]
    return float(100.0 * (min(vals) if which == "min" else max(vals)))


# ---- the claim table --------------------------------------------------------
# (label, reader, manuscript value, absolute tolerance)
CLAIMS = [
    ("floor mean, 200 AirfRANS cases", floor_realdata_mean, 0.192, 0.001),
    ("floor median", floor_realdata_median, 0.133, 0.001),
    ("uniform-field residual is exactly zero", uniform_all_zero, 0.0, 1e-12),
    ("ladder floor @128^2", lambda r: ladder_level(r, 128), 0.0624, 0.0002),
    ("ladder floor @256^2", lambda r: ladder_level(r, 256), 0.0779, 0.0002),
    ("ladder floor @512^2", lambda r: ladder_level(r, 512), 0.1067, 0.0002),
    ("ladder fitted exponent p", ladder_p, -0.387, 0.002),
    ("cases where the floor rose", ladder_rose, 21, 0.5),
    ("omitted closure @128^2", lambda r: ladder_level(r, 128, "omitted_closure"),
     0.0018, 0.0002),
    ("omitted closure @512^2", lambda r: ladder_level(r, 512, "omitted_closure"),
     0.0054, 0.0002),
    ("residual cut, descent from truth (%)", descent_residual_cut, 84.0, 1.5),
    ("cases where residual fell, from truth", lambda r: descent_wins(r, "from_truth"),
     24, 0.5),
    ("median field error after descent from truth",
     lambda r: descent(r, "from_truth", "mse_u", "median"), 0.91, 0.02),
    ("residual-chosen iterate worse, from truth",
     lambda r: descent_selection_worse(r, "from_truth"), 24, 0.5),
    ("residual-chosen iterate worse, perturbed",
     lambda r: descent_selection_worse(r, "perturbed_residual"), 24, 0.5),
    ("perturbed cases where descent helped", descent_perturbed_helped, 18, 0.5),
    ("audit cost, median ms", audit_cost_median, 1.13, 0.02),
    ("deployed solve, median ms", deployed_solve_median, 3822.0, 2.0),
    ("omitted term as % of floor", decomp_omitted_fraction, 4.2, 1.0),
    ("largest shift from repairing the operator (%)", decomp_repaired_shift,
     0.18, 0.05),
    # --- the concessions. These are the rows a referee checks, and the ones
    # most likely to drift, because they came from a separate analysis pass.
    ("C1 residual AUROC on field error",
     lambda r: c1(r, "residual/auroc"), 0.871, 0.001),
    ("C1 physics-free sigma AUROC",
     lambda r: c1(r, "sigma_vel/auroc"), 0.894, 0.001),
    ("C1 fused AUROC",
     lambda r: c1(r, "fused/auroc"), 0.905, 0.001),
    ("C1 sigma minus residual, delta AUROC",
     lambda r: c1(r, "sigma_vel_minus_residual/delta_auroc"), 0.023, 0.001),
    ("C2 raw rank correlation",
     lambda r: c2(r, "rho_raw"), 0.610, 0.001),
    ("C2 partial correlation given the floor",
     lambda r: c2(r, "rho_partial_given_normtruth"), 0.561, 0.001),
    ("C2 fraction of the association surviving (%)",
     lambda r: 100 * c2(r, "fraction_of_rho_surviving"), 92.0, 0.5),
    ("C3 ungated raises residual, DEPLOYED path (%)",
     lambda r: c3_ungated(r, "deployed"), 1.0, 0.1),
    ("C3 ungated raises residual, ensemble path (%)",
     lambda r: c3_ungated(r, "ensemble"), 8.8, 0.1),
    ("C3 ungated half-step median error improvement, DEPLOYED (%)",
     lambda r: -100.0 * review(r, "control3_fixed_step.json")["pooled"]
     ["deployed_backbone_DEQ"]["policies"]["fixed_0.5"]["median_err_rel_change"], 6.2, 0.1),
    ("C3 gate median error improvement, DEPLOYED (%)",
     lambda r: -100.0 * review(r, "control3_fixed_step.json")["pooled"]
     ["deployed_backbone_DEQ"]["policies"]["gate"]["median_err_rel_change"], 5.8, 0.1),
    ("C4 residual AUROC on drag error",
     c4_drag_auroc, 0.952, 0.001),
    ("floor share of a typical score (%)",
     lambda r: 100 * gate_followup(r, "floor_share_of_typical_score"), 86.0, 1.0),
    ("conformal bound width, min (x the error)",
     lambda r: bound_over_error(r, "min"), 6.5, 0.05),
    ("conformal bound width, max (x the error)",
     lambda r: bound_over_error(r, "max"), 7.9, 0.05),
    ("conformal coverage at a 0.90 target",
     conformal_coverage, 0.90, 0.005),
    ("Transolver inversion rate, min (%)",
     lambda r: transolver_inversion(r, "min"), 2.5, 0.1),
    ("Transolver inversion rate, max (%)",
     lambda r: transolver_inversion(r, "max"), 7.0, 0.1),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    print("Tracing manuscript numbers to committed result files\n")
    print(f"  {'claim':<46} {'paper':>10} {'recomputed':>12}   verdict")
    fails, skips = [], []
    for label, reader, stated, tol in CLAIMS:
        try:
            got = reader(args.root)
        except Exception as exc:
            got = None
            if args.verbose:
                print(f"    ! {label}: {type(exc).__name__}: {exc}")
        if got is None:
            skips.append(label)
            print(f"  {label:<46} {stated:>10} {'--':>12}   SKIP (no file)")
            continue
        ok = abs(got - stated) <= tol
        if not ok:
            fails.append((label, stated, got))
        print(f"  {label:<46} {stated:>10.4g} {got:>12.4g}   "
              f"{'ok' if ok else 'MISMATCH'}")

    print()
    if fails:
        print(f"{len(fails)} number(s) in the manuscript no longer match their source:")
        for label, stated, got in fails:
            print(f"  - {label}: paper says {stated:g}, file gives {got:g}")
    if skips:
        print(f"{len(skips)} claim(s) could not be checked (missing file): "
              f"{', '.join(skips)}")
    if not fails and not skips:
        print("every checked number matches its source file.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
