"""WindsorML: does restoring `ratio_length_front_rear` rescue the metadata null?

Decision question (written before the first run of this script, 2026-09-29):
docs/paper/review/null_mechanism.md Sec 4 argued WindsorML's weak null (R2 0.104,
harmonised protocol) is most likely a METADATA DEFECT: the aggregated
geo_parameters_all.csv omits `ratio_length_front_rear`. The per-run files
(run_i/geo_parameters_i.csv on HuggingFace neashton/windsorml) carry it.

Pre-stated reading (from null_mechanism.md Sec 4, unchanged):
  null rises into 0.6-0.8  -> metadata incompleteness was the cause;
  null stays near 0.1-0.3  -> WindsorML's drag is genuinely not a smooth low-dim
                              function of its published design parameters, and
                              the metadata-defect explanation is withdrawn.
Protocol: identical to neuroforge.nullbench.harmonised (10-fold OOS OLS, seed 0,
plain OLS via lstsq on an intercept design matrix, constant-reference-area cd),
run on the SAME case set with and without the restored column, so that the
only thing that changes is the column. Runs 350-354 have no per-run file and
are excluded from BOTH arms; the committed n=355 number is reproduced alongside.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import neuroforge  # noqa: F401  (thread cap before numpy)
import numpy as np

from neuroforge.nullbench import stats
from neuroforge.nullbench.fit import kfold_oos
from neuroforge.nullbench.io import design_matrix
from neuroforge.nullbench.benchmarks import WINDSORML_PARAM_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AGG = os.path.join(ROOT, "src", "neuroforge", "nullbench", "data", "windsorml", "windsorml_355.csv")
PERRUN = os.path.join(ROOT, "data", "crossbench", "windsor_perrun")
OUT = os.path.join(ROOT, "paper_rebuild", "results", "windsor_completed_metadata.json")


def load_agg():
    with open(AGG, newline="") as fh:
        return {int(r["run_id"]): r for r in csv.DictReader(fh)}


HF = "https://huggingface.co/datasets/neashton/windsorml/resolve/main/run_{i}/geo_parameters_{i}.csv"


def fetch_perrun(n_max=360):
    """Download the per-run parameter files (a few kB each) if absent. Runs
    350-354 have none upstream (HTTP 404 on 2026-09-29)."""
    import urllib.error
    import urllib.request
    os.makedirs(PERRUN, exist_ok=True)
    for i in range(n_max):
        dst = os.path.join(PERRUN, f"geo_{i}.csv")
        if os.path.exists(dst):
            continue
        try:
            urllib.request.urlretrieve(HF.format(i=i), dst)
        except urllib.error.HTTPError:
            pass


def load_perrun():
    if not glob.glob(os.path.join(PERRUN, "geo_*.csv")):
        fetch_perrun()
    out = {}
    for p in glob.glob(os.path.join(PERRUN, "geo_*.csv")):
        rid = int(os.path.basename(p)[4:-4])
        with open(p, newline="") as fh:
            out[rid] = next(csv.DictReader(fh))
    return out


def null(X, y, n_boot=10000):
    pred = kfold_oos(design_matrix(X), y, k=10, seed=0)
    r2 = stats.r2_score(y, pred)
    lo, hi, _ = stats.bootstrap_ci(y, pred, "r2", n_boot=n_boot, seed=0)
    return {"r2": r2, "ci95": [lo, hi], "n": int(len(y))}


def main():
    agg, per = load_agg(), load_perrun()
    # consistency: shared columns must agree between aggregate and per-run files
    maxdiff = 0.0
    for rid, r in per.items():
        for c in WINDSORML_PARAM_COLS:
            maxdiff = max(maxdiff, abs(float(r[c]) - float(agg[rid][c])))
    ids = sorted(set(per) & set(agg))
    res = {"per_run_files": len(per), "shared_ids": len(ids),
           "max_abs_diff_shared_columns": maxdiff}
    extra = np.array([float(per[i]["ratio_length_front_rear"]) for i in ids])
    res["ratio_length_front_rear"] = {"min": extra.min(), "max": extra.max(),
                                      "sd": extra.std(ddof=1)}
    for target in ("cd", "cl"):
        y = np.array([float(agg[i][target]) for i in ids])
        X7 = np.array([[float(agg[i][c]) for c in WINDSORML_PARAM_COLS] for i in ids])
        X8 = np.column_stack([X7, extra])
        yall = np.array([float(agg[i][target]) for i in sorted(agg)])
        Xall = np.array([[float(agg[i][c]) for c in WINDSORML_PARAM_COLS] for i in sorted(agg)])
        res[target] = {
            "committed_protocol_n355": null(Xall, yall),
            "published_cols_n350": null(X7, y),
            "restored_cols_n350": null(X8, y),
            "restored_quadratic_n350": null(np.column_stack([X8, X8 ** 2]), y),
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    json.dump(res, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
