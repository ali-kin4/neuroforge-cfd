"""Trace paper 2's headline numbers back to their committed result files.

The companion to ``audit_paper_numbers.py``, which does the same job for paper 1
and which found a reversed pair that had been in the manuscript for a day. This
paper is about to be tagged, so the same check runs here before the tag rather
than after it.

Each row names a claim, the file it must come from, how the value is read out,
and what ``docs/paper2/DRAFT.md`` currently states. A row FAILS on disagreement
beyond its tolerance and is SKIPPED, counted separately, if the file is missing.

Usage
-----
    python scripts/audit_paper2_numbers.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def load(root: str, rel: str):
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---- the four-way decomposition, the paper's spine --------------------------

def level(root, arm, field):
    d = load(root, "results/decomposition.json")
    if d is None:
        return None
    node = d["levels"][arm]
    if field == "mean":
        return 100.0 * node["mean"]
    if field == "ci_lo":
        return 100.0 * node["ci"][0]
    if field == "ci_hi":
        return 100.0 * node["ci"][1]
    if field == "wins":
        return float(node["wins"])
    if field == "p":
        return float(node["p"])
    raise KeyError(field)


def contrast(root, name, field):
    d = load(root, "results/decomposition.json")
    if d is None:
        return None
    node = d["contrasts"][name]
    if field == "mean":
        return 100.0 * node["mean"]
    if field == "wins":
        return float(node["wins"])
    if field == "p":
        return float(node["p"])
    raise KeyError(field)


# ---- the fixed-point control ------------------------------------------------

def perturb_contrast(root, arm, band="Cd_v@0.01"):
    """Paired drop, in points of saving, from ``oracle_mesh`` to ``arm``.

    The draft's -12.2 and -82.5 are contrasts against the oracle, not absolute
    savings: on the raw rows ``smooth_perturb`` converges much faster than cold.
    Recomputing the contrast is the only way to check the number the paper
    actually prints.
    """
    d = load(root, "results/depth_perturb.json")
    if d is None:
        return None
    per = d["by_force"].get(band, {}).get("per_case")
    if not per:
        return None
    drops = []
    for _, row in per.items():
        cold, orac, got = row.get("cold"), row.get("oracle_mesh"), row.get(arm)
        if not cold or not orac or not got:
            continue
        drops.append(100.0 * ((1 - got / cold) - (1 - orac / cold)))
    return float(np.mean(drops)) if drops else None


# ---- the causal stopping rule -----------------------------------------------

def causal(root, arm, field):
    d = load(root, "results/causal_stopping.json")
    if d is None:
        return None
    node = d["summary"][arm]["causal"]
    if field == "mean":
        return 100.0 * node["mean"]
    if field == "wins":
        return float(node["wins"])
    raise KeyError(field)


# ---- the closed form --------------------------------------------------------

def bound_ratio(root, which):
    """Over-prediction factor of the closed form on the rows where it is live.

    Read from the artifact's own ``bound`` block, which excludes the degenerate
    rows -- those with fewer than two mesh rings below the representation's first
    station, where ``clustered_seed``'s donor mapping hands the first ring its own
    value and the mechanism cannot act. Averaging over all five rows instead gives
    1.25--3.09 and is not what the paper claims; the distinction is the point of
    the degeneracy criterion, so the audit has to respect it.
    """
    d = load(root, "results/closed_form_validation.json")
    if d is None:
        return None
    return float(d["bound"][f"ratio_{which}"])


def bound_rows(root, kind):
    d = load(root, "results/closed_form_validation.json")
    if d is None:
        return None
    return float(d["bound"][f"{kind}_rows"])


CLAIMS = [
    # arms
    ("oracle_mesh saving (%)", lambda r: level(r, "oracle_mesh", "mean"), 93.6, 0.05),
    ("oracle_mesh CI low", lambda r: level(r, "oracle_mesh", "ci_lo"), 92.9, 0.05),
    ("oracle_mesh CI high", lambda r: level(r, "oracle_mesh", "ci_hi"), 94.3, 0.05),
    ("oracle_mesh wins", lambda r: level(r, "oracle_mesh", "wins"), 13, 0.5),
    ("oracle_bl saving (%)", lambda r: level(r, "oracle_bl", "mean"), 72.9, 0.05),
    ("or_proj_coarse saving (%)", lambda r: level(r, "or_proj_coarse", "mean"),
     79.6, 0.05),
    ("nf_bl saving (%)", lambda r: level(r, "nf_bl", "mean"), 18.4, 0.05),
    ("cartesian_128 saving (%)", lambda r: level(r, "cartesian_128", "mean"),
     3.4, 0.05),
    ("cartesian_128 CI low", lambda r: level(r, "cartesian_128", "ci_lo"),
     -2.3, 0.05),
    ("cartesian_128 CI high", lambda r: level(r, "cartesian_128", "ci_hi"),
     8.8, 0.05),
    ("cartesian_128 p-value", lambda r: level(r, "cartesian_128", "p"), 0.27, 0.005),
    # contrasts
    ("contrast: region", lambda r: contrast(r, "region", "mean"), -20.7, 0.05),
    ("contrast: accuracy", lambda r: contrast(r, "accuracy", "mean"), -54.5, 0.05),
    ("contrast: representation (raster)",
     lambda r: contrast(r, "representation (raster)", "mean"), -90.2, 0.05),
    ("contrast: representation (body-fitted)",
     lambda r: contrast(r, "representation (body-fitted)", "mean"), 6.7, 0.05),
    ("body-fitted contrast p-value",
     lambda r: contrast(r, "representation (body-fitted)", "p"), 0.09, 0.005),
    ("body-fitted contrast wins",
     lambda r: contrast(r, "representation (body-fitted)", "wins"), 10, 0.5),
    # the fixed-point control
    ("smooth_perturb drop from oracle (points)",
     lambda r: perturb_contrast(r, "smooth_perturb"), -12.2, 0.3),
    ("cartesian_128 drop from oracle (points)",
     lambda r: perturb_contrast(r, "cartesian_128"), -82.5, 0.3),
    # the causal stopping rule
    ("causal: nf_bl saving (%)", lambda r: causal(r, "nf_bl", "mean"), 7.0, 0.1),
    ("causal: nf_bl wins", lambda r: causal(r, "nf_bl", "wins"), 6, 0.5),
    ("causal: oracle_mesh saving (%)",
     lambda r: causal(r, "oracle_mesh", "mean"), 80.0, 0.1),
    ("causal: oracle_bl saving (%)",
     lambda r: causal(r, "oracle_bl", "mean"), 77.8, 0.1),
    ("causal: or_proj_coarse saving (%)",
     lambda r: causal(r, "or_proj_coarse", "mean"), 72.9, 0.1),
    # the closed form
    ("closed form over-prediction, min", lambda r: bound_ratio(r, "min"), 1.9, 0.05),
    ("closed form over-prediction, max", lambda r: bound_ratio(r, "max"), 2.8, 0.05),
    ("closed form live rows", lambda r: bound_rows(r, "non_degenerate"), 3, 0.5),
    ("closed form degenerate rows", lambda r: bound_rows(r, "degenerate"), 2, 0.5),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    print("Tracing paper 2's numbers to committed result files\n")
    print(f"  {'claim':<44} {'draft':>9} {'recomputed':>12}   verdict")
    fails, skips = [], []
    for label, reader, stated, tol in CLAIMS:
        try:
            got = reader(args.root)
        except Exception as exc:
            print(f"  {label:<44} {stated:>9} {'ERR':>12}   "
                  f"{type(exc).__name__}: {exc}")
            fails.append((label, stated, float("nan")))
            continue
        if got is None:
            skips.append(label)
            print(f"  {label:<44} {stated:>9} {'--':>12}   SKIP (no file)")
            continue
        ok = abs(got - stated) <= tol
        if not ok:
            fails.append((label, stated, got))
        print(f"  {label:<44} {stated:>9.4g} {got:>12.4g}   "
              f"{'ok' if ok else 'MISMATCH'}")

    print()
    if fails:
        print(f"{len(fails)} number(s) in the draft no longer match their source:")
        for label, stated, got in fails:
            print(f"  - {label}: draft says {stated:g}, file gives {got:g}")
    if skips:
        print(f"{len(skips)} claim(s) unchecked (missing file): {', '.join(skips)}")
    if not fails and not skips:
        print("every checked number matches its source file.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
