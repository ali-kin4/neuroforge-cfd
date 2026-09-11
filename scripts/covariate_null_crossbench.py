"""Does the AirfRANS covariate null travel to other CFD-ML benchmarks?

Background
----------
``covariate_null.py`` / ``covariate_null_trainfit.py`` established, on AirfRANS, that
ordinary least squares on the *case parameters alone* -- the numbers in the case file
name, no flow field opened -- ranks the official force coefficients at Spearman 0.9821
[0.9737, 0.9866] for lift and 0.9318 [0.8981, 0.9531] for drag, fit on the official 800
train cases and scored on the official 200 test cases. Four of five published baselines
fall below that on both targets.

The obvious reviewer question is whether that is a fact about AirfRANS or a fact about
how this subfield scores surrogate models. This script answers it by running the same
null, under each benchmark's own protocol and in each benchmark's own published metric,
on four further benchmarks:

  DrivAerNet++   8121 car designs, official train/val/test id lists, 23 published design
                 parameters for the 4165 smooth-underbody designs, published drag-
                 prediction table (NeurIPS 2024 D&B Table 4).
  AhmedML        500 Ahmed-body variants, 8 published shape parameters, published force
                 coefficients, no recommended split, no published ML baseline.
  WindsorML      355 Windsor-body variants, 7 published shape parameters, published force
                 coefficients, recommended 60/20/20 split, one published MGN drag number.
  DrivAerML      484 DrivAer variants, 16 published morph parameters, published force
                 coefficients, no recommended split, no ML baseline in the dataset paper.

Everything this script needs is per-case metadata: a parameter row and a force label.
No mesh, no field, no point cloud is downloaded -- roughly 300 kB against the ~40 TB of
field data these benchmarks distribute.

Protocol (identical in spirit to the AirfRANS run; adaptations stated explicitly)
--------------------------------------------------------------------------------
1. Nested feature sets, cheapest first, so the marginal value of each parameter block
   is readable rather than asserted.
2. Official split where the benchmark defines one (DrivAerNet++: fit on the 5819 train
   ids, score on the 1154 test ids). Where no canonical split exists (AhmedML,
   WindsorML, DrivAerML) the null is refit by K-fold out-of-sample exactly as in
   ``covariate_null.py``, and every such number is labelled ``kfold_oos``. The two
   protocols are never silently mixed.
3. Case-level percentile bootstrap, 95%, on every reported correlation and score.
4. In-sample reported alongside out-of-sample so the optimism is visible.

ADAPTATION, pre-registered here before any number was generated: the AirfRANS comparison
ran in Spearman because Bonnet et al. report Spearman. DrivAerNet++ reports R-squared,
MSE, MAE and Max AE and does not report Spearman, so the DrivAerNet++ verdict runs on
R-squared (sklearn convention, test-set mean in the denominator -- the convention the
benchmark's own ``AutoML_parametric.py`` uses via ``sklearn.metrics.r2_score``). Spearman
and MSE/MAE/MaxAE are reported alongside as secondary columns so the two benchmarks stay
readable against one another. Scoring the null in a metric the benchmark does not publish
would make the comparison void, so the metric follows the benchmark, not our convenience.

Pre-registered reading (unchanged from ``covariate_null.py``)
-------------------------------------------------------------
For a higher-is-better metric, a published entry is BELOW THE NULL iff its reported mean
is below the null's out-of-sample point estimate AND the null's bootstrap 95% interval
excludes that mean. Anything else is ``straddles`` or ``clears``. ``verdict`` is imported
from ``covariate_null`` rather than reimplemented so the rule cannot drift.
``verdict_lower`` below is the same rule mirrored for a lower-is-better metric (MSE), and
is the only new rule introduced here.

A published number given only as a BOUND ("MSE of less than 0.00028", WindsorML) is
adjudicable in one direction only: if the null's interval lies entirely on the worse side
of the bound, the published entry CLEARS; if the null looks better, the comparison is
recorded as ``bound_only_not_adjudicable``, because the true published value may be
arbitrarily far below its stated bound. This asymmetry is fixed here, before the run.

Feature blocks, and what is deliberately kept OUT of the headline
------------------------------------------------------------------
DrivAerNet++ normalises Cd by each design's own effective frontal area (paper, App.
"Cd = Fd / (0.5 rho u^2 A_ref)", with A_ref the effective frontal area), and ships the
per-design areas as a separate CSV. Frontal area is therefore partly inside the label and
requires the geometry to compute. It is included only as a clearly labelled DIAGNOSTIC
block, last in the nesting, and never as the headline null. The headline claim must be
"design parameters alone, no geometry opened".

For AhmedML / WindsorML / DrivAerML the force file used is the one keyed to the benchmark's
own primary normalisation (``force_mom_all.csv``); WindsorML publishes ``frontal_area`` as
one of its seven design parameters and its ``force_mom_all.csv`` uses a CONSTANT reference
area, so there the area is an ordinary design parameter and not leakage.

Usage
-----
    .venv/Scripts/python.exe scripts/covariate_null_crossbench.py --download
    .venv/Scripts/python.exe scripts/covariate_null_crossbench.py

CPU-only, pure numpy/scipy, well under a minute. Metadata is cached under
``data/crossbench/`` (gitignored; the licences are CC BY-NC / CC BY-SA and the files are
re-fetchable from the URLs recorded in ``SOURCES`` and in the output JSON).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy; see CLAUDE.md

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covariate_null import kfold_oos, verdict  # noqa: E402

DATA = os.path.join("data", "crossbench")

# --------------------------------------------------------------------------------------
# Where every byte came from. Only per-case metadata: parameters and force labels.
# --------------------------------------------------------------------------------------
GH = "https://raw.githubusercontent.com/Mohamedelrefaie/DrivAerNet/main/"
DROPBOX_CD = ("https://www.dropbox.com/scl/fi/2rtchqnpmzy90uwa9wwny/"
              "DrivAerNetPlusPlus_Cd_8k_Updated.csv?rlkey=vjnjurtxfuqr40zqgupnks8sn&dl=1")
DROPBOX_AREA = ("https://www.dropbox.com/scl/fi/b7fenj0wmhzqx64bj82t1/"
                "DrivAerNetPlusPlus_CarDesign_Areas.csv?rlkey=usbunuupxwmx6g49r9r7dh8zk&dl=1")
HF = "https://huggingface.co/datasets/neashton/%s/resolve/main/%s"

SOURCES = {
    "drivaernet_params.csv": GH + "ParametricModels/DrivAerNet_ParametricData.csv",
    "drivaernet_train_ids.txt": GH + "train_val_test_splits/train_design_ids.txt",
    "drivaernet_val_ids.txt": GH + "train_val_test_splits/val_design_ids.txt",
    "drivaernet_test_ids.txt": GH + "train_val_test_splits/test_design_ids.txt",
    "drivaernet_cd.csv": DROPBOX_CD,
    "drivaernet_areas.csv": DROPBOX_AREA,
    "ahmedml_geo.csv": HF % ("ahmedml", "geo_parameters_all.csv"),
    "ahmedml_force.csv": HF % ("ahmedml", "force_mom_all.csv"),
    "windsorml_geo.csv": HF % ("windsorml", "geo_parameters_all.csv"),
    "windsorml_force.csv": HF % ("windsorml", "force_mom_all.csv"),
    "drivaerml_geo.csv": HF % ("drivaerml", "geo_parameters_all.csv"),
    "drivaerml_force.csv": HF % ("drivaerml", "force_mom_all.csv"),
    "drivaerml_force_constref.csv": HF % ("drivaerml", "force_mom_constref_all.csv"),
}

# --------------------------------------------------------------------------------------
# Published numbers. Every one read at source; see docs/paper/review/null_travels.md.
# --------------------------------------------------------------------------------------

# DrivAerNet++ NeurIPS 2024 D&B, Table 4, "All cars", test set of 1,200 designs. Read from
# the proceedings PDF (013cf29a9e68e4411d0593040a8a1eb3-Paper-Datasets_and_Benchmarks_Track.pdf,
# page 8) on 2026-09-11 and cross-checked against arXiv 2406.09624. The paper's own NeurIPS
# checklist item 3(c) answers "[No]" to error bars, so these carry no seed spread.
DRIVAERNET_PUBLISHED = [
    ("PointNet",   {"r2": 0.643, "mse": 14.9e-5, "mae": 9.60e-3, "maxae": 12.45e-3}),
    ("GCNN",       {"r2": 0.596, "mse": 17.1e-5, "mae": 10.43e-3, "maxae": 15.03e-3}),
    ("RegDGCNN",   {"r2": 0.641, "mse": 14.2e-5, "mae": 9.31e-3, "maxae": 12.79e-3}),
]

# WindsorML (arXiv 2407.19320) SI D.2, verbatim: "using a 60/20/20 split of train,
# validation and test data, it is possible to obtain a MSE of less than 0.00028 for the
# drag coefficient". A bound, not a point estimate -- see the adjudicability rule above.
WINDSOR_PUBLISHED_CD_MSE_BOUND = 2.8e-4

REPORT_SETS = {}  # filled per benchmark below


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def r2_score(y_true, y_pred):
    """sklearn convention: 1 - SSE / SST, SST about the mean of *y_true* on this set."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def all_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_true - y_pred
    return {
        "r2": r2_score(y_true, y_pred),
        "spearman": float(spearmanr(y_pred, y_true).statistic),
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "maxae": float(np.max(np.abs(err))),
    }


def boot_ci_metrics(y_true, y_pred, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for every metric, resampling CASES (as in covariate_null)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ("r2", "spearman", "mse", "mae", "maxae")
    acc = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        m = all_metrics(y_true[idx], y_pred[idx])
        for k in keys:
            acc[k][i] = m[k]
    out = {}
    for k in keys:
        v = acc[k][np.isfinite(acc[k])]
        out[k] = [float(np.percentile(v, 100 * alpha / 2)),
                  float(np.percentile(v, 100 * (1 - alpha / 2))),
                  int(len(v))]
    return out


def verdict_lower(published_mean, null_point, ci_lo, ci_hi):
    """The pre-registered rule, mirrored for a lower-is-better metric such as MSE.

    BELOW THE NULL (i.e. the published model is worse than the parameter regression) iff
    the published mean is ABOVE the null's point estimate and the null's 95% interval
    excludes it.
    """
    if published_mean is None:
        return "not reported"
    if published_mean > null_point and not (ci_lo <= published_mean <= ci_hi):
        return "BELOW THE NULL"
    if ci_lo <= published_mean <= ci_hi:
        return "straddles"
    return "clears"


def verdict_bound_lower(bound, null_point, ci_lo, ci_hi):
    """Adjudicability rule for a published upper BOUND on a lower-is-better metric."""
    if bound is None:
        return "not reported"
    if null_point > bound and not (ci_lo <= bound <= ci_hi):
        return "clears (published bound beats the null outright)"
    return "bound_only_not_adjudicable"


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------
def fit_predict(X_tr, y_tr, X_te):
    beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    return X_te @ beta


def nn_leak_check(X_tr, X_te, seed=0):
    """Nearest-neighbour distance from each test design to the train set, against the
    within-train nearest-neighbour distribution, on standardised features.

    If test designs sit at essentially zero distance from train designs the split leaks
    and BOTH the null and every published model on it are inflated. Reported either way.
    """
    mu = X_tr.mean(0)
    sd = X_tr.std(0)
    sd[sd == 0] = 1.0
    A = (X_tr - mu) / sd
    B = (X_te - mu) / sd
    # chunked to keep memory sane
    def nn(Q, R, exclude_self=False):
        out = np.empty(len(Q))
        step = 512
        for i in range(0, len(Q), step):
            d = np.sqrt(((Q[i:i + step, None, :] - R[None, :, :]) ** 2).sum(-1))
            if exclude_self:
                for j in range(d.shape[0]):
                    d[j, i + j] = np.inf
            out[i:i + step] = d.min(1)
        return out
    d_te = nn(B, A)
    d_tr = nn(A, A, exclude_self=True)
    return {
        "test_to_train_nn_median": float(np.median(d_te)),
        "test_to_train_nn_min": float(np.min(d_te)),
        "test_to_train_nn_p05": float(np.percentile(d_te, 5)),
        "train_to_train_nn_median": float(np.median(d_tr)),
        "train_to_train_nn_min": float(np.min(d_tr)),
        "ratio_median_test_over_train": float(np.median(d_te) / np.median(d_tr))
        if np.median(d_tr) > 0 else None,
        "reading": ("A ratio near 1 means test designs are no closer to the train set than "
                    "train designs are to each other, i.e. the split does not leak by "
                    "near-duplication. A ratio well below 1 would indicate leakage."),
    }


def score_official(X_tr, y_tr, X_te, y_te, n_boot, seed):
    pred_te = fit_predict(X_tr, y_tr, X_te)
    pred_tr = fit_predict(X_tr, y_tr, X_tr)
    m = all_metrics(y_te, pred_te)
    m_in = all_metrics(y_tr, pred_tr)
    ci = boot_ci_metrics(np.asarray(y_te, float), pred_te, n_boot=n_boot, seed=seed)
    return {"protocol": "official_split_trainfit_testscore",
            "n_fit": int(len(y_tr)), "n_score": int(len(y_te)),
            "out_of_sample": m, "in_sample_on_fit_set": m_in,
            "in_sample_inflation_r2": m_in["r2"] - m["r2"],
            "ci95": ci}


def score_kfold(X, y, k, n_boot, seed):
    oos = kfold_oos(X, y, k=k, seed=seed)
    ins = fit_predict(X, y, X)
    m = all_metrics(y, oos)
    m_in = all_metrics(y, ins)
    ci = boot_ci_metrics(np.asarray(y, float), oos, n_boot=n_boot, seed=seed)
    return {"protocol": "kfold_oos_k%d" % k,
            "n_fit": int(len(y)), "n_score": int(len(y)),
            "out_of_sample": m, "in_sample": m_in,
            "in_sample_inflation_r2": m_in["r2"] - m["r2"],
            "ci95": ci}


# --------------------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------------------
def download_all(force=False):
    os.makedirs(DATA, exist_ok=True)
    for name, url in SOURCES.items():
        dst = os.path.join(DATA, name)
        if os.path.exists(dst) and not force:
            print("  have %s" % name)
            continue
        print("  fetching %s" % name)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as fh:
            fh.write(r.read())
    return True


def read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # CSVs in this family carry stray spaces in header names and values
    clean = []
    for r in rows:
        clean.append({(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                      for k, v in r.items()})
    return clean


def numeric_matrix(rows, cols):
    return np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)


def onehot(labels):
    """One-hot with the first level dropped; the intercept carries it."""
    levels = sorted(set(labels))
    if len(levels) <= 1:
        return np.zeros((len(labels), 0)), levels
    M = np.zeros((len(labels), len(levels) - 1))
    for i, lab in enumerate(labels):
        j = levels.index(lab)
        if j > 0:
            M[i, j - 1] = 1.0
    return M, levels


# --------------------------------------------------------------------------------------
# benchmark 1: DrivAerNet++
# --------------------------------------------------------------------------------------
def run_drivaernet(n_boot, seed, k):
    params = read_csv("drivaernet_params.csv")
    cd_rows = read_csv("drivaernet_cd.csv")
    areas = read_csv("drivaernet_areas.csv")
    cd = {r["ID"]: float(r["Drag_Value"]) for r in cd_rows}
    area = {r["Car Design"]: float(r["Frontal Area (m²)"]) for r in areas}
    pcols = [c for c in params[0]
             if c != "Experiment" and not c.startswith(("Average ", "Std "))]
    pvec = {r["Experiment"]: np.array([float(r[c]) for c in pcols]) for r in params}

    split = {}
    for tag in ("train", "val", "test"):
        with open(os.path.join(DATA, "drivaernet_%s_ids.txt" % tag), encoding="utf-8") as fh:
            split[tag] = [ln.strip() for ln in fh if ln.strip()]

    def cat_tokens(i):
        t = i.split("_")
        return "|".join(t[:4])  # rear type, underbody, wheels, mirrors

    tr_all = [i for i in split["train"] if i in cd]
    te_all = [i for i in split["test"] if i in cd]
    tr_sub = [i for i in tr_all if i in pvec]
    te_sub = [i for i in te_all if i in pvec]

    y_tr_all = np.array([cd[i] for i in tr_all])
    y_te_all = np.array([cd[i] for i in te_all])
    y_tr_sub = np.array([cd[i] for i in tr_sub])
    y_te_sub = np.array([cd[i] for i in te_sub])

    # --- feature blocks -------------------------------------------------------------
    levels_all = sorted(set(cat_tokens(i) for i in tr_all))

    def cat_mat(ids):
        M = np.zeros((len(ids), len(levels_all) - 1))
        for r, i in enumerate(ids):
            j = levels_all.index(cat_tokens(i)) if cat_tokens(i) in levels_all else 0
            if j > 0:
                M[r, j - 1] = 1.0
        return M

    def one(ids):
        return np.ones((len(ids), 1))

    def P(ids):
        return np.array([pvec[i] for i in ids])

    def A(ids):
        return np.array([[area[i]] for i in ids])

    res = {"benchmark": "DrivAerNet++", "targets": {"cd": {}}}

    # (a) FULL official test set, category metadata only -- directly comparable to Table 4
    sets_full = {
        "intercept_only": (one(tr_all), one(te_all)),
        "category_tokens_only": (np.hstack([one(tr_all), cat_mat(tr_all)]),
                                 np.hstack([one(te_all), cat_mat(te_all)])),
    }
    for nm, (Xtr, Xte) in sets_full.items():
        r = score_official(Xtr, y_tr_all, Xte, y_te_all, n_boot, seed)
        r["scored_on"] = "full official test split (all 8k design families)"
        r["comparable_to_published"] = True
        res["targets"]["cd"][nm] = r

    # (b) PARAMETRIC SUBSET -- the 23 published design parameters exist only here
    Ptr, Pte = P(tr_sub), P(te_sub)
    sets_sub = {
        "sub__intercept_only": (one(tr_sub), one(te_sub)),
        "sub__category_tokens": (np.hstack([one(tr_sub), cat_mat(tr_sub)]),
                                 np.hstack([one(te_sub), cat_mat(te_sub)])),
        "sub__params23": (np.hstack([one(tr_sub), Ptr]), np.hstack([one(te_sub), Pte])),
        "sub__category_params23": (np.hstack([one(tr_sub), cat_mat(tr_sub), Ptr]),
                                   np.hstack([one(te_sub), cat_mat(te_sub), Pte])),
        "sub__category_params23_squares": (
            np.hstack([one(tr_sub), cat_mat(tr_sub), Ptr, Ptr ** 2]),
            np.hstack([one(te_sub), cat_mat(te_sub), Pte, Pte ** 2])),
        # DIAGNOSTIC ONLY: frontal area sits in the denominator of Cd.
        "sub__DIAG_category_params23_area": (
            np.hstack([one(tr_sub), cat_mat(tr_sub), Ptr, A(tr_sub)]),
            np.hstack([one(te_sub), cat_mat(te_sub), Pte, A(te_sub)])),
    }
    for nm, (Xtr, Xte) in sets_sub.items():
        r = score_official(Xtr, y_tr_sub, Xte, y_te_sub, n_boot, seed)
        r["scored_on"] = ("parametric subset of the official test split "
                          "(smooth-underbody designs only)")
        r["comparable_to_published"] = False
        r["caveat"] = ("Published Table 4 is scored on the full ~1200-design test set. "
                       "This row is scored on the %d of those designs that have published "
                       "parameters. See the label-spread diagnostic before reading it as a "
                       "head-to-head." % len(te_sub))
        res["targets"]["cd"][nm] = r

    # --- diagnostics ----------------------------------------------------------------
    res["subset_vs_full_label_spread"] = {
        "full_test_n": int(len(y_te_all)),
        "full_test_cd_sd": float(np.std(y_te_all, ddof=1)),
        "full_test_cd_var": float(np.var(y_te_all, ddof=1)),
        "full_test_cd_range": [float(y_te_all.min()), float(y_te_all.max())],
        "param_subset_test_n": int(len(y_te_sub)),
        "param_subset_test_cd_sd": float(np.std(y_te_sub, ddof=1)),
        "param_subset_test_cd_var": float(np.var(y_te_sub, ddof=1)),
        "param_subset_test_cd_range": [float(y_te_sub.min()), float(y_te_sub.max())],
        "reading": ("R-squared divides by the test set's own variance. If the parametric "
                    "subset's Cd variance is at or below the full test set's, the subset "
                    "R-squared is not inflated by an easier target; if it is larger, the "
                    "cross-subset comparison is suggestive only."),
    }
    res["near_duplicate_check_params23"] = nn_leak_check(Ptr, Pte)
    res["coverage"] = {
        "designs_with_published_drag": len(cd),
        "designs_with_published_parameters": len(pvec),
        "official_split_sizes": {k: len(v) for k, v in split.items()},
        "train_in_params": len(tr_sub), "test_in_params": len(te_sub),
        "note": ("The 3956 F_D_* designs are the DrivAerNet-v1 fastbacks, generated from a "
                 "different 50-parameter scheme. That parameter table is not published in "
                 "the repository or on the Dataverse record, so those designs enter the "
                 "null only through their categorical tokens."),
    }

    # --- published comparison -------------------------------------------------------
    null_full = res["targets"]["cd"]["category_tokens_only"]
    null_sub = res["targets"]["cd"]["sub__category_params23"]
    table = []
    for model, vals in DRIVAERNET_PUBLISHED:
        row = {"model": model, "published": vals, "published_std": None,
               "published_source": ("DrivAerNet++ NeurIPS 2024 D&B Table 4, test set of "
                                    "1200 designs; no seed spread reported")}
        for tag, null in (("vs_category_null_full_test", null_full),
                          ("vs_param_null_subset", null_sub)):
            row[tag] = {
                "null_r2": null["out_of_sample"]["r2"],
                "null_r2_ci95": null["ci95"]["r2"][:2],
                "verdict_r2": verdict(vals["r2"], null["out_of_sample"]["r2"],
                                      *null["ci95"]["r2"][:2]),
                "null_mse": null["out_of_sample"]["mse"],
                "null_mse_ci95": null["ci95"]["mse"][:2],
                "verdict_mse": verdict_lower(vals["mse"], null["out_of_sample"]["mse"],
                                             *null["ci95"]["mse"][:2]),
                "comparable_test_set": null["comparable_to_published"],
            }
        table.append(row)
    res["published_vs_null"] = table
    return res


# --------------------------------------------------------------------------------------
# benchmarks 2-4: the Ashton et al. family (AhmedML / WindsorML / DrivAerML)
# --------------------------------------------------------------------------------------
ASHTON = {
    "AhmedML": {
        "geo": "ahmedml_geo.csv", "force": "ahmedml_force.csv",
        "id_geo": "run", "id_force": "run",
        "targets": {"cd": "cd", "cl": "cl"},
        "split": None,
        "force_note": ("force_mom_all.csv: time-averaged coefficients on a CONSTANT "
                       "reference area of 0.112 m^2, so no area term enters the label."),
    },
    "WindsorML": {
        "geo": "windsorml_geo.csv", "force": "windsorml_force.csv",
        "id_geo": "run", "id_force": "run",
        "targets": {"cd": "cd", "cl": "cl"},
        "split": "60/20/20 recommended, but no id lists are published, so K-fold is used",
        "force_note": ("force_mom_all.csv: CONSTANT reference area, so the published "
                       "frontal_area design parameter is an ordinary covariate here."),
    },
    "DrivAerML": {
        "geo": "drivaerml_geo.csv", "force": "drivaerml_force.csv",
        "id_geo": "Run", "id_force": "run",
        "targets": {"cd": "cd", "cl": "cl"},
        "split": None,
        "force_note": ("force_mom_all.csv: per-geometry reference values. "
                       "force_mom_constref_all.csv is run as a robustness check."),
    },
}


def run_ashton(name, n_boot, seed, k, force_override=None):
    cfg = ASHTON[name]
    geo = read_csv(cfg["geo"])
    force = read_csv(force_override or cfg["force"])
    gid, fid = cfg["id_geo"], cfg["id_force"]
    pcols = [c for c in geo[0] if c != gid]
    gmap = {r[gid]: np.array([float(r[c]) for c in pcols]) for r in geo}
    ids = [r[fid] for r in force if r[fid] in gmap]
    X_raw = np.array([gmap[i] for i in ids])
    one = np.ones((len(ids), 1))

    sets = {
        "intercept_only": one,
        "params_linear": np.hstack([one, X_raw]),
        "params_linear_squares": np.hstack([one, X_raw, X_raw ** 2]),
    }

    res = {"benchmark": name, "n_cases": len(ids), "n_params": len(pcols),
           "param_names": pcols, "force_note": cfg["force_note"],
           "split_note": cfg["split"] or "no split published by the benchmark",
           "force_file": force_override or cfg["force"], "targets": {}}

    for tgt, col in cfg["targets"].items():
        if col not in force[0]:
            continue
        y = np.array([float(r[col]) for r in force if r[fid] in gmap])
        rec = {}
        for nm, X in sets.items():
            rec[nm] = score_kfold(X, y, k, n_boot, seed)
        res["targets"][tgt] = rec

    res["near_duplicate_check"] = nn_leak_check(X_raw, X_raw)
    res["near_duplicate_check"]["note"] = ("Computed within the single pool, since there is "
                                           "no published split; the test-to-train column is "
                                           "therefore leave-one-out and equals the train "
                                           "column by construction.")
    return res


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/review/covariate_null_crossbench.json")
    args = ap.parse_args()

    t0 = time.time()
    if args.download:
        print("downloading per-case metadata only (no fields, no meshes) ...")
        download_all()
    missing = [n for n in SOURCES if not os.path.exists(os.path.join(DATA, n))]
    if missing:
        print("missing cached metadata: %s\nrun with --download" % ", ".join(missing))
        return 2

    out = {
        "artifact": "covariate_null_crossbench",
        "question": ("Does the AirfRANS covariate null -- OLS on the published case "
                     "parameters alone, no flow field opened -- also match or beat "
                     "published surrogates on other CFD-ML benchmarks?"),
        "prereg": ("Higher-is-better: BELOW THE NULL iff published mean < null point "
                   "estimate AND the null's 95% CI excludes it (verdict() imported from "
                   "covariate_null.py, unchanged). Lower-is-better: the mirrored rule. "
                   "A published BOUND is adjudicable only against the null being worse."),
        "metric_adaptation": ("DrivAerNet++ publishes R-squared, not Spearman, so its "
                              "verdict runs on R-squared (sklearn convention). Spearman, "
                              "MSE, MAE and Max AE are reported alongside."),
        "data_sources": SOURCES,
        "download_policy": "per-case parameters and force labels only; no field data",
        "benchmarks": {},
    }

    print("\n== DrivAerNet++ ==")
    out["benchmarks"]["DrivAerNet++"] = run_drivaernet(args.boot, args.seed, args.folds)
    for nm in ("AhmedML", "WindsorML", "DrivAerML"):
        print("== %s ==" % nm)
        out["benchmarks"][nm] = run_ashton(nm, args.boot, args.seed, args.folds)
    print("== DrivAerML (constref robustness) ==")
    out["benchmarks"]["DrivAerML_constref"] = run_ashton(
        "DrivAerML", args.boot, args.seed, args.folds,
        force_override="drivaerml_force_constref.csv")

    # WindsorML published bound
    w = out["benchmarks"]["WindsorML"]["targets"]["cd"]["params_linear"]
    out["benchmarks"]["WindsorML"]["published_vs_null"] = [{
        "model": "MeshGraphNet (direct KPI head)",
        "published_cd_mse_bound": WINDSOR_PUBLISHED_CD_MSE_BOUND,
        "published_source": ("WindsorML arXiv 2407.19320 SI D.2: 'MSE of less than 0.00028 "
                             "for the drag coefficient' on the recommended 60/20/20 split"),
        "null_cd_mse": w["out_of_sample"]["mse"],
        "null_cd_mse_ci95": w["ci95"]["mse"][:2],
        "verdict": verdict_bound_lower(WINDSOR_PUBLISHED_CD_MSE_BOUND,
                                       w["out_of_sample"]["mse"], *w["ci95"]["mse"][:2]),
        "protocol_mismatch": ("published: 60/20/20 single split; null: %d-fold OOS over all "
                              "cases. Both are out of sample; neither is the other."
                              % args.folds),
    }]
    out["benchmarks"]["AhmedML"]["published_vs_null"] = [{
        "model": None,
        "verdict": "no published ML baseline",
        "published_source": ("AhmedML arXiv 2407.20801 reports no ML benchmark results; the "
                             "datasheet states only that 'limited testing with various ML "
                             "approaches has been undertaken by the author team'."),
    }]
    out["benchmarks"]["DrivAerML"]["published_vs_null"] = [{
        "model": None,
        "verdict": "no published ML baseline in the dataset paper",
        "published_source": ("DrivAerML arXiv 2408.11969 reports no ML benchmark table; it "
                             "positions the dataset as a future challenge case."),
    }]

    out["wall_clock_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # --- console summary ------------------------------------------------------------
    print("\n" + "=" * 78)
    d = out["benchmarks"]["DrivAerNet++"]
    print("DrivAerNet++  (target Cd, official split, metric R^2)")
    for nm, r in d["targets"]["cd"].items():
        print("  %-34s n=%4d  R2=%7.4f [%.4f, %.4f]  MSE=%.3e  rho=%6.3f" % (
            nm, r["n_score"], r["out_of_sample"]["r2"], r["ci95"]["r2"][0],
            r["ci95"]["r2"][1], r["out_of_sample"]["mse"], r["out_of_sample"]["spearman"]))
    print("\n  published (Table 4, n=1200 test):")
    for row in d["published_vs_null"]:
        print("    %-10s R2=%.3f  MSE=%.3e | vs category null (comparable): %-14s | "
              "vs param null (subset): %s" % (
                  row["model"], row["published"]["r2"], row["published"]["mse"],
                  row["vs_category_null_full_test"]["verdict_r2"],
                  row["vs_param_null_subset"]["verdict_r2"]))
    s = d["subset_vs_full_label_spread"]
    print("\n  label spread: full test sd=%.5f (n=%d) vs param subset sd=%.5f (n=%d)" % (
        s["full_test_cd_sd"], s["full_test_n"], s["param_subset_test_cd_sd"],
        s["param_subset_test_n"]))
    nnc = d["near_duplicate_check_params23"]
    print("  near-duplicate: test->train NN median %.3f vs train->train %.3f (ratio %.3f)"
          % (nnc["test_to_train_nn_median"], nnc["train_to_train_nn_median"],
             nnc["ratio_median_test_over_train"]))

    for nm in ("AhmedML", "WindsorML", "DrivAerML", "DrivAerML_constref"):
        b = out["benchmarks"][nm]
        print("\n%s  (n=%d cases, %d params, %s)" % (nm, b["n_cases"], b["n_params"],
                                                     b["targets"] and "kfold OOS"))
        for tgt, rec in b["targets"].items():
            for fs, r in rec.items():
                print("  %-4s %-24s R2=%7.4f [%.4f, %.4f]  MSE=%.3e  rho=%6.3f" % (
                    tgt, fs, r["out_of_sample"]["r2"], r["ci95"]["r2"][0],
                    r["ci95"]["r2"][1], r["out_of_sample"]["mse"],
                    r["out_of_sample"]["spearman"]))
    wv = out["benchmarks"]["WindsorML"]["published_vs_null"][0]
    print("\nWindsorML published MGN drag MSE bound %.2e vs null %.2e -> %s" % (
        wv["published_cd_mse_bound"], wv["null_cd_mse"], wv["verdict"]))
    print("\nwall clock %.1f s\nwrote %s" % (out["wall_clock_seconds"], args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
