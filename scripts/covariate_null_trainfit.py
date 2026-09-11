"""Airtight version of the covariate null: fit on TRAIN, score on TEST.

``covariate_null.py`` refits the null by K-fold on the 200 test cases, so every
prediction is out of sample. That is defensible, but it is not the protocol the
published models ran: they fit on the 800 `full` train cases and were scored on the
200 test cases. A reviewer is entitled to ask whether the null's advantage comes
from being fit on the test distribution.

This script closes that. The null is fit on the official 800-case train split, using
official AirfRANS force labels for those cases, and scored on the 200-case test split
-- the exact protocol the leaderboard numbers were produced under. If the null still
clears a published entry under this protocol, the comparison is admissible.

Reported alongside: the K-fold-on-test number from the sibling script, so the two
protocols can be compared directly rather than argued about.

Feature sets are nested, cheapest first, so the marginal value of each part of the
case name is readable. The NACA digits are included deliberately: a surrogate takes
the airfoil geometry as an input, and the digits ARE that geometry, exactly
specified. A regression on them is therefore "predict the force from the model's own
inputs, ignoring the flow field" -- which is precisely the null a field-predicting
model must beat to have earned anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy; see CLAUDE.md

import numpy as np
from scipy.stats import spearmanr

import airfrans.dataset as afds
from airfrans.simulation import Simulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covariate_null import PUBLISHED, boot_ci, feature_sets, parse_case, verdict  # noqa: E402

ROOT_DEFAULT = os.path.join("data", "Dataset")


def official_labels(root, names):
    """(cl, cd) from AirfRANS's own integrator, for each named case."""
    cl, cd, kept = [], [], []
    for nm in names:
        try:
            sim = Simulation(root=root, name=nm)
            (c_d, _, _), (c_l, _, _) = sim.force_coefficient(reference=True)
        except Exception as exc:  # a case missing from disk is reported, never skipped silently
            print("  WARN: %s -> %s" % (nm, exc))
            continue
        cl.append(float(c_l))
        cd.append(float(c_d))
        kept.append(nm)
    return np.asarray(cl), np.asarray(cd), kept


def design(names, width=None):
    u, alpha, dig = [], [], []
    for nm in names:
        a_u, a_a, a_d = parse_case(nm)
        u.append(a_u)
        alpha.append(a_a)
        dig.append(a_d)
    w = width or max(len(d) for d in dig)
    digits = np.array([d + [0.0] * (w - len(d)) for d in dig])
    return np.asarray(u), np.asarray(alpha), digits, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/review/covariate_null_trainfit.json")
    args = ap.parse_args()

    print("loading official train/test split names ...")
    _, train_names = afds.load(root=args.root, task="full", train=True)
    _, test_names = afds.load(root=args.root, task="full", train=False)
    print("  train=%d  test=%d" % (len(train_names), len(test_names)))

    print("reading official labels (train) ...")
    cl_tr, cd_tr, train_names = official_labels(args.root, train_names)
    print("reading official labels (test) ...")
    cl_te, cd_te, test_names = official_labels(args.root, test_names)
    print("  usable train=%d  test=%d" % (len(train_names), len(test_names)))

    # one shared digit width so train and test designs are conformable
    _, _, _, w_tr = design(train_names)
    _, _, _, w_te = design(test_names)
    w = max(w_tr, w_te)
    u_tr, a_tr, d_tr, _ = design(train_names, w)
    u_te, a_te, d_te, _ = design(test_names, w)

    sets_tr = feature_sets(u_tr, a_tr, d_tr)
    sets_te = feature_sets(u_te, a_te, d_te)

    out = {
        "artifact": "covariate_null_trainfit",
        "question": ("Does the case-name null still clear published AirfRANS entries when "
                     "fit on the official 800-case train split and scored on the 200-case "
                     "test split -- the exact protocol those entries were produced under?"),
        "n_train": len(train_names),
        "n_test": len(test_names),
        "protocol": {
            "fit": "OLS on the official full/train split, official labels",
            "score": "Spearman on the official full/test split, official labels",
            "bootstrap": "%d case-level resamples over the test set, percentile 95%%" % args.boot,
            "naca_digit_width": int(w),
        },
        "prereg": ("Reading fixed in covariate_null.py before any number existed: BELOW THE "
                   "NULL iff published mean < null point estimate AND the null's 95% CI "
                   "excludes that mean."),
        "targets": {},
    }

    for tgt, y_tr, y_te in (("cl", cl_tr, cl_te), ("cd", cd_tr, cd_te)):
        rec = {}
        for fs in sets_tr:
            beta, *_ = np.linalg.lstsq(sets_tr[fs], y_tr, rcond=None)
            pred_te = sets_te[fs] @ beta
            r = float(spearmanr(pred_te, y_te).statistic)
            lo, hi, nb = boot_ci(pred_te, y_te, n_boot=args.boot, seed=args.seed)
            rec[fs] = {"spearman_train_fit_test_score": r, "ci95": [lo, hi],
                       "n_boot_finite": nb}
        out["targets"][tgt] = rec

    null_cl = out["targets"]["cl"]["U_alpha_alpha2_naca"]
    null_cd = out["targets"]["cd"]["U_alpha_alpha2_naca"]
    table = []
    for model, vals in PUBLISHED:
        row = {"model": model}
        for tgt, null in (("cl", null_cl), ("cd", null_cd)):
            mean = vals[tgt][0]
            row[tgt] = {
                "published": mean,
                "published_std": vals[tgt][1],
                "null": null["spearman_train_fit_test_score"],
                "null_ci95": null["ci95"],
                "verdict": verdict(mean, null["spearman_train_fit_test_score"], *null["ci95"]),
            }
        table.append(row)
    out["leaderboard_vs_full_name_null"] = table

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\nNull fit on %d train, scored on %d test\n" % (len(train_names), len(test_names)))
    for tgt in ("cl", "cd"):
        print("target: %s" % tgt)
        for fs, r in out["targets"][tgt].items():
            print("  %-24s %8.4f   [%.4f, %.4f]" % (
                fs, r["spearman_train_fit_test_score"], r["ci95"][0], r["ci95"][1]))
        print()

    print("Published vs the FULL case-name null:")
    print("  %-13s %-30s %-30s" % ("model", "lift", "drag"))
    for row in table:
        def fmt(d):
            if d["published"] is None:
                return "not reported"
            s = "%.4f" % d["published"]
            if d["published_std"] is not None:
                s += " +/-%.3f" % d["published_std"]
            return "%s  %s" % (s, d["verdict"])
        print("  %-13s %-30s %-30s" % (row["model"], fmt(row["cl"]), fmt(row["cd"])))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
