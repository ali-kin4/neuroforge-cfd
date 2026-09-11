"""Covariate null for AirfRANS force-coefficient rank correlations.

The question
------------
AirfRANS papers report Spearman rank correlation of predicted vs official force
coefficients as a headline metric. Every case in the dataset is indexed by a small
parameter vector that ``Simulation.reset()`` parses out of the *file name*:

    airFoil2D_SST_<U>_<alpha>_<NACA digits...>

So a model that never looks at a flow field can still rank the test set, using only
the case name. That is the null this script measures, and it is the baseline the
benchmark's headline metric has to clear before a learned number means anything.

Why this script exists rather than reusing ``drag_covariate_control.py``
-----------------------------------------------------------------------
That script fits the regression on the same 200 cases it scores, which is in-sample
and inflates the null, and it reports a point estimate with no interval. Neither is
admissible for a comparison against published numbers that carry seed spreads. Here:

* the null is refit **out of sample** by K-fold cross-validation -- every prediction
  is made by a model that never saw that case -- and the in-sample value is reported
  alongside so the inflation is visible rather than hidden;
* a case-level bootstrap gives a percentile CI on every Spearman, so "below the null"
  is a statistical statement and not an eyeball one.

Feature sets are nested, so the marginal value of each piece of the file name is
readable off the table: angle alone, then (U, alpha), then the quadratic that drag
needs because induced drag scales with sin^2(alpha), then the NACA digits.

Pre-registered reading (fixed before the numbers were generated)
----------------------------------------------------------------
A published entry is BELOW THE NULL if its reported mean is below the null's
out-of-sample point estimate AND the null's bootstrap CI excludes that mean.
Anything else is STRADDLES or CLEARS. No other reading is claimed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy; see CLAUDE.md

import numpy as np
from scipy.stats import spearmanr

CACHE = os.path.join("results", "control", "_cache")
LABELS = os.path.join(CACHE, "official_labels_full_test_n200.json")

# Published AirfRANS `full` numbers, for the comparison table. Sources:
#   MLP / GraphSAGE / PointNet / Graph U-Net -- Bonnet et al., NeurIPS 2022 D&B, Table 3,
#   read at source from the proceedings PDF on 2026-09-11; see
#   docs/paper/review/published_baselines_verified.md. Note the drag spreads are large
#   enough that three of the four intervals span zero.
#   Transolver -- Wu et al., ICML 2024. It reports no drag column; Appendix B.1 states
#   the deep models fail at drag, so lift only.
PUBLISHED = [
    ("MLP",          {"cl": (0.913, 0.018), "cd": (-0.117, 0.256)}),
    ("GraphSAGE",    {"cl": (0.965, 0.011), "cd": (-0.303, 0.124)}),
    ("PointNet",     {"cl": (0.938, 0.023), "cd": (-0.022, 0.097)}),
    ("Graph U-Net",  {"cl": (0.967, 0.019), "cd": (-0.138, 0.258)}),
    ("Transolver",   {"cl": (0.9978, None), "cd": (None, None)}),
]

NAME_RE = re.compile(r"^airFoil2D_SST_(-?[\d.]+)_(-?[\d.]+)_(.+)$")


def parse_case(name):
    """(U, alpha, digits) from an AirfRANS case name.

    The trailing block is the NACA specification and has either 3 or 4 numbers
    (4-digit vs 5-digit series), so it is returned as a variable-length list and
    padded by the caller.
    """
    m = NAME_RE.match(name)
    if m is None:
        raise ValueError("unparsable case name: %r" % name)
    u = float(m.group(1))
    alpha = float(m.group(2))
    digits = [float(x) for x in m.group(3).split("_") if x != ""]
    return u, alpha, digits


def feature_sets(u, alpha, digits):
    """Nested feature matrices, cheapest first. Each includes an intercept."""
    n = len(u)
    one = np.ones(n)
    a = alpha
    sets = {
        "alpha": np.column_stack([one, a]),
        "U_alpha": np.column_stack([one, u, a]),
        "U_alpha_alpha2": np.column_stack([one, u, a, a * a]),
        "U_alpha_alpha2_naca": np.column_stack([one, u, a, a * a, digits]),
    }
    return sets


def kfold_oos(X, y, k=10, seed=0):
    """Out-of-sample predictions by K-fold OLS. Every y-hat is from a fold that
    excluded its own case, so the resulting correlation carries no in-sample
    advantage over a learned model evaluated on a held-out split."""
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    pred = np.empty(n)
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        pred[f] = X[f] @ beta
    return pred


def insample(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X @ beta


def boot_ci(a, b, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for Spearman(a, b), resampling CASES."""
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # a degenerate resample (all-equal ranks) yields nan; carry it and drop later
        vals[i] = spearmanr(a[idx], b[idx]).statistic
    vals = vals[np.isfinite(vals)]
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi, int(len(vals))


def verdict(published_mean, null_point, ci_lo, ci_hi):
    """Pre-registered reading; see the module docstring."""
    if published_mean is None:
        return "not reported"
    if published_mean < null_point and not (ci_lo <= published_mean <= ci_hi):
        return "BELOW THE NULL"
    if ci_lo <= published_mean <= ci_hi:
        return "straddles"
    return "clears"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/review/covariate_null.json")
    args = ap.parse_args()

    with open(args.labels, encoding="utf-8") as fh:
        labels = json.load(fh)

    names = sorted(labels)
    u, alpha, dig = [], [], []
    for nm in names:
        a_u, a_a, a_d = parse_case(nm)
        u.append(a_u)
        alpha.append(a_a)
        dig.append(a_d)
    width = max(len(d) for d in dig)
    digits = np.array([d + [0.0] * (width - len(d)) for d in dig])
    u = np.asarray(u)
    alpha = np.asarray(alpha)

    cl = np.array([labels[n]["cl"] for n in names])
    cd = np.array([labels[n]["cd"] for n in names])

    sets = feature_sets(u, alpha, digits)
    out = {
        "artifact": "covariate_null",
        "question": ("How well does a model that sees only the AirfRANS case name rank "
                     "the official force coefficients? This is the null the benchmark's "
                     "headline Spearman metric must clear."),
        "n_cases": len(names),
        "split": "full test (the split the published numbers are reported on)",
        "protocol": {
            "out_of_sample": "%d-fold CV; every prediction from a fold excluding its case" % args.folds,
            "bootstrap": "%d case-level resamples, percentile CI at 95%%" % args.boot,
            "naca_digit_width": int(width),
        },
        "prereg": ("BELOW THE NULL iff published mean < null OOS point estimate AND the "
                   "null's 95% CI excludes that mean."),
        "targets": {},
    }

    for tgt_name, y in (("cl", cl), ("cd", cd)):
        rec = {}
        for fs_name, X in sets.items():
            oos = kfold_oos(X, y, k=args.folds, seed=args.seed)
            ins = insample(X, y)
            r_oos = float(spearmanr(oos, y).statistic)
            r_ins = float(spearmanr(ins, y).statistic)
            lo, hi, nb = boot_ci(oos, y, n_boot=args.boot, seed=args.seed)
            rec[fs_name] = {
                "spearman_out_of_sample": r_oos,
                "spearman_in_sample": r_ins,
                "in_sample_inflation": r_ins - r_oos,
                "ci95": [lo, hi],
                "n_boot_finite": nb,
            }
        out["targets"][tgt_name] = rec

    # Comparison against the published leaderboard, using the (U, alpha) null for lift
    # -- lift is linear in alpha so the quadratic buys nothing -- and the quadratic
    # null for drag, which induced-drag scaling motivates.
    table = []
    null_cl = out["targets"]["cl"]["U_alpha"]
    null_cd = out["targets"]["cd"]["U_alpha_alpha2"]
    for model, vals in PUBLISHED:
        row = {"model": model}
        for tgt, null in (("cl", null_cl), ("cd", null_cd)):
            mean = vals[tgt][0]
            row[tgt] = {
                "published": mean,
                "published_std": vals[tgt][1],
                "null_oos": null["spearman_out_of_sample"],
                "null_ci95": null["ci95"],
                "verdict": verdict(mean, null["spearman_out_of_sample"], *null["ci95"]),
            }
        table.append(row)
    out["leaderboard_vs_null"] = table

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # --- console summary -------------------------------------------------
    print("Covariate null, AirfRANS full test, n=%d\n" % len(names))
    for tgt in ("cl", "cd"):
        print("target: %s" % tgt)
        print("  %-24s %8s %8s %8s  %s" % ("features", "OOS", "in-samp", "inflate", "95% CI"))
        for fs_name, r in out["targets"][tgt].items():
            print("  %-24s %8.4f %8.4f %8.4f  [%.4f, %.4f]" % (
                fs_name, r["spearman_out_of_sample"], r["spearman_in_sample"],
                r["in_sample_inflation"], r["ci95"][0], r["ci95"][1]))
        print()

    print("Published vs null (lift null = U_alpha, drag null = U_alpha_alpha2):")
    print("  %-13s %-26s %-26s" % ("model", "lift", "drag"))
    for row in table:
        def fmt(d):
            if d["published"] is None:
                return "not reported"
            s = "%.4f" % d["published"]
            if d["published_std"] is not None:
                s += " +/-%.3f" % d["published_std"]
            return "%s  %s" % (s, d["verdict"])
        print("  %-13s %-26s %-26s" % (row["model"], fmt(row["cl"]), fmt(row["cd"])))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
