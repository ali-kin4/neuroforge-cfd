"""What makes a CFD-ML benchmark's covariate null strong or weak?

Background
----------
``scripts/covariate_null_crossbench.py`` measured a parameter-only OLS null on five
benchmarks in one subfield and found a ten-fold spread in how well it predicts the
published force label (see ``docs/paper/review/null_travels.md``).  "Benchmarks vary"
is an observation.  This script asks the design question: **what property of a
benchmark's published metadata determines whether the null wins?**

=============================================================================
PRE-REGISTRATION.  Everything below was committed BEFORE the script was run.
=============================================================================

0.  The quantity being explained: a HARMONISED null
---------------------------------------------------
The table in ``null_travels.md`` mixes protocols, metrics and reference-area
conventions, so it cannot be regressed on as-is.  Three of its five entries are
replaced here, and the substitutions are declared now:

* **AirfRANS.**  ``null_travels.md`` reports Spearman (0.9318 drag / 0.9821 lift),
  not R^2.  Recomputed here as OOS R^2 under the common protocol.  It may land far
  from 0.93 and may reorder the table.
* **DrivAerML.**  0.9731 is on the PhysicsNeMo split, which is built by sorting on
  drag and taking the outer deciles -- that inflates R^2.  The split-free anchor
  already published in that report (10-fold, constant area) is used instead.
* **DrivAerNet++.**  0.7365 is scored on a test set where 595 of 1154 designs have
  NO published parameters at all.  "How well does metadata predict the label" is
  ill-posed on cases with no metadata, so the mechanism entry is the parametric
  pool.  The 0.7365 figure remains the correct *reporting-practice* number and is
  carried alongside, not used as the regressand.

Common protocol for every benchmark and every target:
  10-fold OOS OLS, seed 0, features = published parameters standardised on each
  training fold (+ family dummies for DrivAerNet++), metric = R^2 (sklearn
  convention), reference area = CONSTANT where a benchmark publishes both
  conventions.  Per-geometry-area is carried as a sensitivity.

Declared now, independent of any result: null strength is a property of the triple
(benchmark, target, reference-area convention), not of a benchmark.  AhmedML drag
moves from 0.684 to 0.340 on convention alone.

1.  Candidate explanations, with the direction each predicts
------------------------------------------------------------
Class A -- LABEL-FREE.  Computable from the parameter matrix and the dataset paper
alone, never touching the label.  These are the only candidates that could be
applied to a benchmark before it is simulated, so they are the only ones that
could become prospective design guidance.

  A1  Design-space dimensionality.  d_eff = participation ratio of the eigenvalues
      of the parameter correlation matrix, (sum L)^2 / sum L^2.
      PREDICTS: null R^2 falls as d_eff rises (rho < 0).
  A2  Sampling density per dimension, log(n) / d_eff.
      PREDICTS: rho > 0.
  A3  Scale-vs-shape.  Does the sweep move the body's overall size, or only its
      shape at fixed size?  Drag tracks frontal area at first order, so a sweep
      that moves area has a large, linear, easily-fit component.  Measured as
      CV of the published reference/frontal area where published, else CV of the
      product of the length-like parameters.
      PREDICTS: rho > 0.
  A4  Metadata completeness = (number of design degrees of freedom recoverable
      from the published metadata) / (number the dataset paper says were varied).
      PREDICTS: rho > 0.
      DISCLOSURE: this candidate was added after source-checking the four dataset
      papers and finding that WindsorML publishes six of its seven design
      parameters (``ratio_length_front_rear``, range 0-0.8, is absent from
      ``geo_parameters_all.csv``; verified against the live HuggingFace file, so
      it is not a download artefact).  That is a structural fact about the
      benchmark's publication, established before any fit in this script was run,
      but it is a fact discovered while looking for explanations and is flagged as
      such rather than presented as an independent prediction.
  A5  Sampling design type (space-filling DoE vs curated configurations).
      From the dataset papers: DrivAerML = extensible lattice sequence, AhmedML =
      Latin hypercube, WindsorML = Halton, DrivAerNet++ = parametric morph DoE,
      AirfRANS = randomised sampling of (U, alpha, NACA digits).
      PREDICTS: NOTHING -- all five are space-filling, so this candidate is
      pre-registered as unable to produce any ordering.  It is included so that
      the report can say it was tested rather than quietly dropped.

Class B -- LABEL-SIDE, but not using the parameter-to-label fit.

  B1  Label dynamic range, CV = sd(y) / |mean(y)|.
      PREDICTS: rho > 0 (a wider spread is easier to rank against a fixed noise
      floor).
  B2  Signal-to-noise, sd(y) / (solver noise floor).  The floor is estimated from
      a published per-design standard deviation where one exists (DrivAerNet++
      publishes ``Std Cd``), else from a force channel that symmetry forces to
      zero mean (side force at zero yaw).
      PREDICTS: rho > 0.
      CAVEAT DECLARED NOW: WindsorML is run at -2.5 degrees yaw, deliberately, to
      suppress wake bi-stability.  Its side force is therefore a physical signal,
      not a noise probe, and will not be used as one.
  B3  Regime crossing, read off the label distribution: Hartigan-style dip
      statistic, bimodality coefficient, excess kurtosis.
      PREDICTS: rho < 0 (a sweep that crosses a flow-regime boundary produces a
      multimodal or kinked label and defeats a linear fit).

Class C -- DECOMPOSITION.  Declared NOT independent of the null: these partition
the deficit rather than explain it from outside.  Reported as diagnosis, never as
a correlate.

  C1  Determinacy vs linearity.  R^2_flex from the best of {full quadratic +
      interaction OLS, kNN at k in {3,5,10}, kNN-local-linear}, same folds.
      R^2_flex is a LOWER bound on how much of the label the metadata determines.
      linearity share = R^2_lin / R^2_flex.
  C2  Local vs global linear fit, and a coefficient-instability index: the mean
      pairwise cosine distance between locally fitted slope vectors.  High =
      the sensitivity changes sign across the design space = regime change.
  C3  Leave-one-parameter-out.  Drop each published parameter in turn and record
      the fall in null R^2.  This CALIBRATES what a single missing design degree
      of freedom can cost, which is what candidate A4 needs to be adjudicated.
  C4  Learning curve for the flexible learner at n = 50, 100, 200, full.  Needed
      to separate "the label is genuinely rough in these parameters" from "n is
      too small for a nonparametric fit at this dimension".

2.  Falsifiable point predictions, made before the run
-------------------------------------------------------
The n = 5 correlation exercise cannot reach significance (see section 3), so the
real evidential weight is placed on these, which are within-benchmark and have
real degrees of freedom.

  P1  AHMEDML.  Its rear-window angle sweeps 10-60 deg and the Ahmed body's wake
      is known to change regime near 30 deg (C-pillar vortex breakdown).  PREDICT:
      C_d is non-monotone in ``slant-angle-degrees`` with an interior maximum in
      25-35 deg, and adding the single basis function |slant - 30| raises the
      AhmedML drag null R^2 by at least 0.05, by more than the same |x - median|
      basis added on any other single parameter.
      FALSIFIED IF: the label is monotone in slant angle, or the kink basis buys
      < 0.05, or another parameter's kink buys more.
  P2  WINDSORML.  PREDICT: the flexible learning curve (C4) has plateaued by
      n = 355 and still lies below R^2 = 0.5, i.e. the deficit is not sampling
      sparsity.
      FALSIFIED IF: R^2 is still climbing steeply at the full n.
  P3  DRIVAERML.  PREDICT: a single scale composite built from
      Vehicle_Length/Width/Height alone reaches R^2 >= 0.5 on drag.
      FALSIFIED IF: it does not.
  P4  WINDSORML, missing-DOF calibration.  PREDICT: on the benchmarks whose
      metadata IS complete, the largest single-parameter leave-one-out drop is
      smaller than the gap WindsorML would have to close, so one withheld
      parameter is NOT on its own sufficient to explain R^2 = 0.10.
      FALSIFIED IF: some benchmark shows a single parameter worth > 0.5 of R^2.

3.  Adjudication rule, committed now
-------------------------------------
Statistic: Spearman rho between each candidate property and the harmonised drag
null R^2 across the five benchmark-level entries.  Two-sided p by EXACT
permutation over all 5! = 120 orderings.

  * |rho| = 1.0  ->  p = 2/120 = 0.0167.  This is the smallest p attainable.
  * |rho| = 0.9  ->  p = 0.0833.
  * Roughly ten candidates are tested.

Therefore, committed in advance: **no candidate can reach a
multiplicity-corrected significance at n = 5.**  Even a perfect ordering is
p = 0.0167 uncorrected and p = 0.167 after correcting for ten candidates.  The
pre-registered claim ceiling for Class A and B is "ordering consistent with" or
"ordering inconsistent with" -- never "predicts".  Any candidate that fails to
order the five benchmarks perfectly is reported as NOT SURVIVING.  Every
candidate tried is reported, including the failures.

Also committed: rho is recomputed with AirfRANS excluded (n = 4, where nothing
can be significant at all), because AirfRANS varies operating conditions on 2-D
steady RANS while the other four are 3-D scale-resolving shape morphs, and a
reviewer will object to pooling them.

4.  Within-benchmark control, and the bound it gives
------------------------------------------------------
The null is recomputed for every force target each benchmark publishes (14 pairs
in all).  Within a benchmark, the parameter matrix, n, d_eff, sampling design and
solver are all held fixed and only the label changes.  A one-way variance
decomposition of null R^2 over benchmarks then gives a HARD CAP: any purely
design-space property (A1, A2, A5) can explain at most
sigma^2_between / (sigma^2_between + sigma^2_within) of the observed spread.

Declared now: the 14 pairs are NOT independent -- they share a parameter matrix
within each benchmark.  The decomposition is reported; a 14-point scatter
correlation is NOT, because it would claim degrees of freedom that do not exist.

Run:  PYTHONPATH=src python scripts/null_mechanism.py
CPU only; no GPU is touched and no field data is read.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy; see CLAUDE.md

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covariate_null import parse_case  # noqa: E402  -- read-only import

CB = os.path.join("data", "crossbench")
AIRFRANS_LABELS = os.path.join(
    "results", "control", "_cache", "official_labels_full_test_n200.json")

# Design degrees of freedom each dataset paper states it varied, read at source.
# Used only by candidate A4.
DOF_VARIED = {
    "DrivAerML": (16, "arXiv 2408.11969: 16 morphing parameters, extensible lattice DoE"),
    "AirfRANS": (6, "arXiv 2212.07564: U, angle of attack, and the NACA digit spec; the "
                    "case name carries the complete generative specification"),
    "DrivAerNet++": (26, "NeurIPS 2024 D&B sec. 5.1.2 fits 26 parameters; 23 shape "
                         "parameters + 3 categorical axes are published for the "
                         "parametric families"),
    "AhmedML": (6, "arXiv 2407.20801: 6 parameters, Latin hypercube"),
    "WindsorML": (7, "arXiv 2407.19320: 7 parameters, Halton sequence"),
}
DOF_PUBLISHED = {
    "DrivAerML": 16,
    "AirfRANS": 6,
    "DrivAerNet++": 26,
    "AhmedML": 6,   # 8 columns, of which the slant block is redundant; all 6 recoverable
    "WindsorML": 6,  # ratio_length_front_rear is NOT in geo_parameters_all.csv
}
SAMPLING_DESIGN = {
    "DrivAerML": "modified extensible lattice sequence (ANSA DoE), space-filling",
    "AirfRANS": "randomised sampling of U, alpha and NACA digits",
    "DrivAerNet++": "parametric morph DoE over 23-26 parameters, space-filling",
    "AhmedML": "Latin hypercube, space-filling",
    "WindsorML": "Halton quasi-random low-discrepancy sequence, space-filling",
}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def r2_score(y_true, y_pred):
    """sklearn convention, identical to covariate_null_crossbench.r2_score."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def folds_of(n, k, seed):
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k)


def _standardise(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ols_oos(X, y, k=10, seed=0):
    """10-fold OOS OLS with per-fold standardisation and an intercept."""
    n = len(y)
    pred = np.empty(n)
    for f in folds_of(n, k, seed):
        m = np.ones(n, bool)
        m[f] = False
        A, B = _standardise(X[m], X[f])
        A = np.column_stack([np.ones(len(A)), A])
        B = np.column_stack([np.ones(len(B)), B])
        beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        pred[f] = B @ beta
    return pred


def quad_expand(X):
    """Full quadratic: linear + squares + pairwise interactions."""
    cols = [X]
    cols.append(X * X)
    d = X.shape[1]
    for i in range(d):
        for j in range(i + 1, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.column_stack(cols)


def knn_pred(X, y, k_nn, k=10, seed=0, local_linear=False):
    """kNN (or kNN-weighted local-linear) regression on the same folds as ols_oos."""
    n = len(y)
    pred = np.empty(n)
    for f in folds_of(n, k, seed):
        m = np.ones(n, bool)
        m[f] = False
        A, B = _standardise(X[m], X[f])
        ytr = y[m]
        d2 = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d2, axis=1)[:, : min(k_nn, A.shape[0])]
        if not local_linear:
            pred[f] = ytr[idx].mean(1)
        else:
            out = np.empty(len(f))
            for r in range(len(f)):
                nb = idx[r]
                An = np.column_stack([np.ones(len(nb)), A[nb] - B[r]])
                beta, *_ = np.linalg.lstsq(An, ytr[nb], rcond=None)
                out[r] = beta[0]
            pred[f] = out
    return pred


def flexible_r2(X, y, k=10, seed=0, ns_local=(10, 20, 40)):
    """Lower bound on determinacy: best OOS R^2 over a small flexible family."""
    cands = {}
    try:
        cands["quadratic_ols"] = r2_score(y, ols_oos(quad_expand(X), y, k, seed))
    except np.linalg.LinAlgError:
        cands["quadratic_ols"] = float("nan")
    for kk in (3, 5, 10, 20):
        cands["knn_k%d" % kk] = r2_score(y, knn_pred(X, y, kk, k, seed))
    for kk in ns_local:
        if kk > X.shape[1] + 2:
            cands["locallin_k%d" % kk] = r2_score(
                y, knn_pred(X, y, kk, k, seed, local_linear=True))
    best = max((v for v in cands.values() if np.isfinite(v)), default=float("nan"))
    best_name = [nm for nm, v in cands.items() if v == best]
    return best, (best_name[0] if best_name else None), cands


# --------------------------------------------------------------------------- #
# candidate properties
# --------------------------------------------------------------------------- #
def d_eff(X):
    """Participation ratio of the parameter correlation eigenvalues (A1)."""
    Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-12, 1.0, X.std(0))
    C = np.cov(Z, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    if lam.sum() <= 0:
        return float("nan")
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def dip_statistic(y, n_boot=0, seed=0):
    """Hartigan's dip statistic: sup-norm distance from the ECDF to the closest
    unimodal (greatest convex minorant / least concave majorant) distribution.
    Implemented directly; only the statistic is used, as a relative index."""
    x = np.sort(np.asarray(y, float))
    n = len(x)
    ecdf_lo = np.arange(n) / n
    ecdf_hi = np.arange(1, n + 1) / n
    best = np.inf
    # A cheap, monotone-envelope surrogate: the dip is bounded below by the
    # smallest sup-distance to any unimodal cdf; evaluate the two-piece
    # convex/concave envelope anchored at each candidate mode.
    for mi in range(1, n - 1):
        gcm = _greatest_convex_minorant(x[: mi + 1], ecdf_lo[: mi + 1])
        lcm = _least_concave_majorant(x[mi:], ecdf_hi[mi:])
        d = max(np.abs(gcm - ecdf_hi[: mi + 1]).max(),
                np.abs(lcm - ecdf_lo[mi:]).max())
        best = min(best, d)
    return float(best / 2.0)


def _greatest_convex_minorant(x, f):
    n = len(x)
    out = f.copy()
    for _ in range(n):
        changed = False
        for i in range(1, n - 1):
            if x[i + 1] > x[i - 1]:
                t = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
                lin = out[i - 1] + t * (out[i + 1] - out[i - 1])
                if out[i] > lin + 1e-15:
                    out[i] = lin
                    changed = True
        if not changed:
            break
    return out


def _least_concave_majorant(x, f):
    return -_greatest_convex_minorant(x, -f)


def bimodality_coefficient(y):
    """BC = (g^2 + 1) / (k + 3(n-1)^2/((n-2)(n-3))); BC > 5/9 suggests bimodality."""
    y = np.asarray(y, float)
    n = len(y)
    m = y.mean()
    s = y.std(ddof=1)
    g = float(np.mean((y - m) ** 3) / s ** 3)
    kur = float(np.mean((y - m) ** 4) / s ** 4) - 3.0
    denom = kur + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g * g + 1.0) / denom), g, kur


def coefficient_instability(X, y, k_nn=40, n_probe=60, seed=0):
    """C2: mean pairwise cosine distance between locally fitted slope vectors.
    0 = one global linear law; near 1 = the sensitivity flips sign across the
    design space, the signature of a regime change."""
    rng = np.random.default_rng(seed)
    Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-12, 1.0, X.std(0))
    n, d = Z.shape
    k_nn = min(k_nn, n // 3)
    if k_nn <= d + 2:
        return float("nan")
    probes = rng.choice(n, size=min(n_probe, n), replace=False)
    slopes = []
    for p in probes:
        d2 = ((Z - Z[p]) ** 2).sum(1)
        nb = np.argsort(d2)[:k_nn]
        A = np.column_stack([np.ones(k_nn), Z[nb] - Z[p]])
        beta, *_ = np.linalg.lstsq(A, y[nb], rcond=None)
        v = beta[1:]
        nv = np.linalg.norm(v)
        if nv > 1e-14:
            slopes.append(v / nv)
    if len(slopes) < 2:
        return float("nan")
    S = np.array(slopes)
    G = S @ S.T
    iu = np.triu_indices(len(S), 1)
    return float(np.mean(1.0 - G[iu]) / 2.0)  # in [0, 1]


def exact_perm_p(prop, target):
    """Two-sided exact permutation p for Spearman at small n (n! orderings)."""
    prop = np.asarray(prop, float)
    target = np.asarray(target, float)
    ok = np.isfinite(prop) & np.isfinite(target)
    prop, target = prop[ok], target[ok]
    n = len(prop)
    if n < 3:
        return float("nan"), float("nan")
    obs = spearmanr(prop, target).statistic
    cnt = 0
    tot = 0
    for perm in itertools.permutations(range(n)):
        r = spearmanr(prop, target[list(perm)]).statistic
        tot += 1
        if abs(r) >= abs(obs) - 1e-12:
            cnt += 1
    return float(obs), float(cnt / tot)


# --------------------------------------------------------------------------- #
# data loading  (metadata only; no field data is read)
# --------------------------------------------------------------------------- #
def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    rows = [{k.strip(): v for k, v in r.items() if k is not None} for r in rows]
    return rows


def load_airfrans(labels_path):
    with open(labels_path, encoding="utf-8") as fh:
        lab = json.load(fh)
    names = sorted(lab)
    u, a, dig = [], [], []
    for nm in names:
        uu, aa, dd = parse_case(nm)
        u.append(uu)
        a.append(aa)
        dig.append(dd)
    w = max(len(d) for d in dig)
    D = np.array([d + [0.0] * (w - len(d)) for d in dig])
    X = np.column_stack([np.asarray(u), np.asarray(a), D])
    pnames = ["U", "alpha"] + ["naca_%d" % i for i in range(w)]
    targets = {"cd": np.array([lab[n]["cd"] for n in names]),
               "cl": np.array([lab[n]["cl"] for n in names])}
    return X, pnames, targets, names


def load_simple(geo_csv, force_csv, key_geo, key_force, tgt_cols, drop_params=()):
    g = _read_csv(os.path.join(CB, geo_csv))
    f = _read_csv(os.path.join(CB, force_csv))
    gcols = [c for c in g[0] if c != key_geo and c not in drop_params]
    fmap = {}
    for r in f:
        try:
            fmap[int(float(r[key_force]))] = r
        except (TypeError, ValueError):
            continue
    Xs, ys, ids = [], {c: [] for c in tgt_cols}, []
    for r in g:
        rid = int(float(r[key_geo]))
        if rid not in fmap:
            continue
        try:
            row = [float(r[c]) for c in gcols]
            tv = [float(fmap[rid][c]) for c in tgt_cols]
        except (TypeError, ValueError):
            continue
        Xs.append(row)
        ids.append(rid)
        for c, v in zip(tgt_cols, tv):
            ys[c].append(v)
    return (np.array(Xs, float), gcols,
            {c: np.array(v, float) for c, v in ys.items()}, ids)


def load_drivaernet():
    p = _read_csv(os.path.join(CB, "drivaernet_params.csv"))
    pcols = [c for c in p[0] if c != "Experiment"]
    design_cols = [c for c in pcols if not c.startswith(("Average", "Std"))]
    cd = {r["ID"]: float(r["Drag_Value"])
          for r in _read_csv(os.path.join(CB, "drivaernet_cd.csv"))}
    fams = sorted({"_".join(r["Experiment"].split("_")[:2]) for r in p})
    X, y, stdcd, ids = [], [], [], []
    for r in p:
        did = r["Experiment"]
        if did not in cd:
            continue
        try:
            row = [float(r[c]) for c in design_cols]
        except (TypeError, ValueError):
            continue
        fam = "_".join(did.split("_")[:2])
        X.append(row + [1.0 if fam == fm else 0.0 for fm in fams[1:]])
        y.append(cd[did])
        try:
            stdcd.append(float(r["Std Cd"]))
        except (TypeError, ValueError):
            stdcd.append(np.nan)
        ids.append(did)
    names = design_cols + ["fam_" + f for f in fams[1:]]
    return (np.array(X, float), names, {"cd": np.array(y, float)}, ids,
            np.array(stdcd, float), design_cols)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/review/null_mechanism.json")
    args = ap.parse_args()
    K, SEED = args.folds, args.seed

    out = {
        "artifact": "null_mechanism",
        "question": ("What computable property of a CFD-ML benchmark's published "
                     "metadata determines whether a parameter-only null beats the "
                     "published surrogates?"),
        "prereg": "see the module docstring; committed before this script was run",
        "protocol": {
            "null": "%d-fold OOS OLS, per-fold standardisation, R^2 (sklearn convention)"
                    % K,
            "seed": SEED,
            "reference_area": "CONSTANT where both conventions are published",
            "no_field_data": True,
        },
        "benchmarks": {},
    }

    B = {}

    # ---- AirfRANS ------------------------------------------------------- #
    Xa, pa, ta, _ = load_airfrans(AIRFRANS_LABELS)
    B["AirfRANS"] = dict(X=Xa, pnames=pa, targets=ta, area=None, noise=None,
                         note="official full/test 200 cases; the cached official "
                              "labels. Params are the complete generative spec in "
                              "the case name (U, alpha, NACA digits).")

    # ---- AhmedML (constant reference area) ------------------------------ #
    Xh, ph, th, _ = load_simple("ahmedml_geo.csv", "ahmedml_force.csv",
                                "run", "run", ["cd", "cl"])
    B["AhmedML"] = dict(X=Xh, pnames=ph, targets=th,
                        area=Xh[:, ph.index("body-height")] * Xh[:, ph.index("body-width")],
                        noise=None,
                        note="force_mom_all.csv = constant reference area 0.112 m^2.")

    # ---- WindsorML (constant reference area) ---------------------------- #
    Xw, pw, tw, _ = load_simple("windsorml_geo.csv", "windsorml_force.csv",
                                "run", "run", ["cd", "cl", "cs", "cmy"])
    B["WindsorML"] = dict(X=Xw, pnames=pw, targets=tw,
                          area=Xw[:, pw.index("frontal_area")], noise=None,
                          note="6 of the paper's 7 design parameters + frontal_area; "
                               "ratio_length_front_rear is not published. -2.5 deg yaw.")

    # ---- DrivAerML (constant reference area) ---------------------------- #
    Xd, pd_, td, _ = load_simple("drivaerml_geo.csv", "drivaerml_force_constref.csv",
                                 "Run", "run", ["cd", "cl", "clf", "clr", "cs"])
    B["DrivAerML"] = dict(X=Xd, pnames=pd_, targets=td,
                          area=Xd[:, pd_.index("Vehicle_Width")] *
                               Xd[:, pd_.index("Vehicle_Height")],
                          noise=None,
                          note="force_mom_constref_all.csv; 16 published morph params.")

    # ---- DrivAerNet++ (parametric pool) --------------------------------- #
    Xn, pn, tn, _, stdcd, dn_design = load_drivaernet()
    B["DrivAerNet++"] = dict(X=Xn, pnames=pn, targets=tn, area=None,
                             noise=float(np.nanmean(stdcd)),
                             note="parametric pool (designs with published "
                                  "parameters), 23 design params + family dummies.")

    # ---- harmonised nulls, all targets ---------------------------------- #
    for name, rec in B.items():
        X, y_all = rec["X"], rec["targets"]
        entry = {
            "n_cases": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "param_names": rec["pnames"],
            "note": rec["note"],
            "sampling_design": SAMPLING_DESIGN[name],
            "dof_varied_per_paper": DOF_VARIED[name][0],
            "dof_varied_source": DOF_VARIED[name][1],
            "dof_published": DOF_PUBLISHED[name],
            "A1_d_eff": d_eff(X),
            "A4_metadata_completeness": DOF_PUBLISHED[name] / DOF_VARIED[name][0],
            "targets": {},
        }
        entry["A2_density_logn_over_deff"] = math.log(X.shape[0]) / entry["A1_d_eff"]
        if rec["area"] is not None:
            a = rec["area"]
            entry["A3_area_cv"] = float(np.std(a, ddof=1) / abs(np.mean(a)))
        else:
            entry["A3_area_cv"] = None

        for tname, y in y_all.items():
            lin = r2_score(y, ols_oos(X, y, K, SEED))
            flex, flex_name, flex_all = flexible_r2(X, y, K, SEED)
            bc, skew, kur = bimodality_coefficient(y)
            entry["targets"][tname] = {
                "null_r2_linear": lin,
                "B1_label_cv": float(np.std(y, ddof=1) / abs(np.mean(y)))
                if abs(np.mean(y)) > 1e-12 else None,
                "label_mean": float(np.mean(y)),
                "label_sd": float(np.std(y, ddof=1)),
                "B3_dip": dip_statistic(y),
                "B3_bimodality_coef": bc,
                "B3_skew": skew,
                "B3_excess_kurtosis": kur,
                "C1_flex_r2_lower_bound": flex,
                "C1_flex_best_model": flex_name,
                "C1_flex_family": flex_all,
                "C1_linearity_share": (lin / flex) if (flex and flex > 0) else None,
                "C2_coef_instability": coefficient_instability(X, y, seed=SEED),
            }
        out["benchmarks"][name] = entry
        print("[%s] n=%d d=%d  d_eff=%.2f  " % (name, X.shape[0], X.shape[1],
                                                entry["A1_d_eff"]),
              {t: round(v["null_r2_linear"], 4) for t, v in entry["targets"].items()})

    # ================================================================== #
    # STAGE 2 -- the pre-registered tests
    # ================================================================== #

    # --- 4. within- vs between-benchmark variance of null R^2 ---------- #
    groups = {nm: [t["null_r2_linear"] for t in e["targets"].values()]
              for nm, e in out["benchmarks"].items()}
    allv = np.array([v for g in groups.values() for v in g], float)
    grand = allv.mean()
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups.values())
    ssw = sum(float(np.sum((np.asarray(g) - np.mean(g)) ** 2)) for g in groups.values())
    kgr = len(groups)
    ngr = len(allv)
    msb = ssb / (kgr - 1)
    msw = ssw / (ngr - kgr) if ngr > kgr else float("nan")
    out["within_vs_between"] = {
        "n_pairs": int(ngr),
        "n_benchmarks": int(kgr),
        "per_benchmark_null_r2": {k: [float(x) for x in v] for k, v in groups.items()},
        "SS_between": float(ssb),
        "SS_within": float(ssw),
        "MS_between": float(msb),
        "MS_within": float(msw),
        "share_of_total_SS_between": float(ssb / (ssb + ssw)) if (ssb + ssw) > 0 else None,
        "cap_on_design_space_only_explanations": (
            "SS_between / (SS_between + SS_within): the largest share of the observed "
            "spread in null R^2 that any property constant within a benchmark (d_eff, "
            "n, sampling design) could possibly account for."),
        "independence_warning": (
            "the 14 (benchmark, target) pairs share a parameter matrix within each "
            "benchmark; a 14-point correlation is NOT reported, by pre-registration."),
    }

    # --- 1/3. candidate properties vs harmonised DRAG null ------------- #
    order = ["DrivAerML", "AirfRANS", "DrivAerNet++", "AhmedML", "WindsorML"]
    tgt = np.array([out["benchmarks"][b]["targets"]["cd"]["null_r2_linear"]
                    for b in order])
    cand = {
        "A1_d_eff": ([out["benchmarks"][b]["A1_d_eff"] for b in order], -1),
        "A2_density_logn_over_deff": (
            [out["benchmarks"][b]["A2_density_logn_over_deff"] for b in order], +1),
        "A2b_n_cases": ([out["benchmarks"][b]["n_cases"] for b in order], +1),
        "A3_area_cv": ([out["benchmarks"][b]["A3_area_cv"] for b in order], +1),
        "A4_metadata_completeness": (
            [out["benchmarks"][b]["A4_metadata_completeness"] for b in order], +1),
        "B1_label_cv": ([out["benchmarks"][b]["targets"]["cd"]["B1_label_cv"]
                         for b in order], +1),
        "B3_dip": ([out["benchmarks"][b]["targets"]["cd"]["B3_dip"] for b in order], -1),
        "B3_bimodality_coef": (
            [out["benchmarks"][b]["targets"]["cd"]["B3_bimodality_coef"]
             for b in order], -1),
        "B3_excess_kurtosis": (
            [out["benchmarks"][b]["targets"]["cd"]["B3_excess_kurtosis"]
             for b in order], -1),
        "C2_coef_instability": (
            [out["benchmarks"][b]["targets"]["cd"]["C2_coef_instability"]
             for b in order], -1),
        "C1_flex_r2_lower_bound": (
            [out["benchmarks"][b]["targets"]["cd"]["C1_flex_r2_lower_bound"]
             for b in order], +1),
    }
    res = {}
    for nm, (vals, direction) in cand.items():
        v = np.array([np.nan if x is None else float(x) for x in vals], float)
        rho, p = exact_perm_p(v, tgt)
        ok = np.isfinite(v)
        rho4, p4 = (exact_perm_p(v[[i for i, b in enumerate(order)
                                    if b != "AirfRANS" and ok[i]]],
                                tgt[[i for i, b in enumerate(order)
                                     if b != "AirfRANS" and ok[i]]])
                    if ok.sum() >= 4 else (float("nan"), float("nan")))
        perfect = np.isfinite(rho) and abs(abs(rho) - 1.0) < 1e-9
        signed_ok = np.isfinite(rho) and (np.sign(rho) == direction)
        res[nm] = {
            "values_in_order": {b: (None if not np.isfinite(v[i]) else float(v[i]))
                                for i, b in enumerate(order)},
            "n_used": int(ok.sum()),
            "predicted_sign": direction,
            "spearman_rho": rho,
            "exact_perm_p_two_sided": p,
            "spearman_rho_without_AirfRANS": rho4,
            "exact_perm_p_without_AirfRANS": p4,
            "sign_matches_prediction": bool(signed_ok),
            "SURVIVES_prereg": bool(perfect and signed_ok),
        }
    out["candidate_vs_drag_null"] = {
        "benchmark_order": order,
        "harmonised_drag_null_r2": {b: float(t) for b, t in zip(order, tgt)},
        "adjudication": ("pre-registered: SURVIVES only if the candidate orders all "
                         "five benchmarks perfectly AND in the predicted direction. "
                         "|rho|=1 is p=2/120=0.0167 uncorrected, p=0.167 after "
                         "correcting for the ~10 candidates tried, so no candidate "
                         "can reach corrected significance at n=5."),
        "candidates": res,
    }

    # --- n-matched control: is the ordering an artefact of sample size? - #
    nmatch = {}
    nmin = min(out["benchmarks"][b]["n_cases"] for b in order)
    for b in order:
        X, y = B[b]["X"], B[b]["targets"]["cd"]
        vals = []
        for s in range(5):
            rng = np.random.default_rng(1000 + s)
            idx = rng.choice(len(y), size=nmin, replace=False)
            vals.append(r2_score(y[idx], ols_oos(X[idx], y[idx], K, SEED)))
        nmatch[b] = {"n": int(nmin), "mean_r2": float(np.mean(vals)),
                     "sd_r2": float(np.std(vals, ddof=1)),
                     "per_subsample": [float(x) for x in vals]}
    out["n_matched_control"] = {
        "purpose": ("every benchmark subsampled to the smallest n, 5 draws, so the "
                    "ordering cannot be an artefact of differing sample size"),
        "results": nmatch,
    }

    # --- C3 / P4. leave-one-parameter-out --------------------------------- #
    loo = {}
    for b in order:
        X, y = B[b]["X"], B[b]["targets"]["cd"]
        names = B[b]["pnames"]
        base = r2_score(y, ols_oos(X, y, K, SEED))
        drops = {}
        for j in range(X.shape[1]):
            keep = [c for c in range(X.shape[1]) if c != j]
            drops[names[j]] = float(base - r2_score(y, ols_oos(X[:, keep], y, K, SEED)))
        top = sorted(drops.items(), key=lambda kv: -kv[1])[:5]
        loo[b] = {"base_r2": float(base), "max_single_param_drop": float(top[0][1]),
                  "top5": [[k, float(v)] for k, v in top], "all_drops": drops}
    out["C3_leave_one_param_out"] = {
        "purpose": ("calibrates what ONE design degree of freedom is worth, which is "
                    "what candidate A4 (metadata completeness) needs in order to be "
                    "adjudicated on WindsorML"),
        "P4_prediction": ("on complete-metadata benchmarks the largest single-parameter "
                          "drop is smaller than the gap WindsorML must close; FALSIFIED "
                          "if any single parameter is worth > 0.5 of R^2"),
        "results": loo,
        "P4_verdict": ("FALSIFIED" if max(v["max_single_param_drop"] for k, v in loo.items()
                                          if k != "WindsorML") > 0.5 else "HELD"),
    }

    # --- P1. AhmedML slant-angle regime crossing -------------------------- #
    Xh_, ph_, yh_ = B["AhmedML"]["X"], B["AhmedML"]["pnames"], B["AhmedML"]["targets"]["cd"]
    base_h = r2_score(yh_, ols_oos(Xh_, yh_, K, SEED))
    kink = {}
    for j, nm in enumerate(ph_):
        med = float(np.median(Xh_[:, j]))
        Xk = np.column_stack([Xh_, np.abs(Xh_[:, j] - med)])
        kink[nm] = {"knot": med,
                    "gain": float(r2_score(yh_, ols_oos(Xk, yh_, K, SEED)) - base_h)}
    sj = ph_.index("slant-angle-degrees")
    Xk30 = np.column_stack([Xh_, np.abs(Xh_[:, sj] - 30.0)])
    gain30 = float(r2_score(yh_, ols_oos(Xk30, yh_, K, SEED)) - base_h)
    # binned label profile in slant angle, to see the interior maximum directly
    sa = Xh_[:, sj]
    edges = np.linspace(sa.min(), sa.max(), 11)
    prof = []
    for i in range(10):
        m = (sa >= edges[i]) & (sa <= edges[i + 1] if i == 9 else sa < edges[i + 1])
        if m.sum() >= 3:
            prof.append({"slant_lo": float(edges[i]), "slant_hi": float(edges[i + 1]),
                         "n": int(m.sum()), "mean_cd": float(yh_[m].mean())})
    interior_max = (len(prof) > 2 and
                    0 < int(np.argmax([p["mean_cd"] for p in prof])) < len(prof) - 1)
    argmax_bin = prof[int(np.argmax([p["mean_cd"] for p in prof]))] if prof else None
    best_other = max((v["gain"] for k, v in kink.items()
                      if k != "slant-angle-degrees"), default=float("nan"))
    out["P1_ahmed_regime_crossing"] = {
        "prediction": ("C_d non-monotone in slant-angle-degrees with an interior "
                       "maximum in 25-35 deg; the |slant-30| basis buys >= 0.05 R^2 "
                       "and more than any other parameter's |x-median| kink"),
        "base_r2": float(base_h),
        "gain_from_abs_slant_minus_30": gain30,
        "gain_from_median_kink_per_param": kink,
        "best_other_param_median_kink_gain": float(best_other),
        "binned_cd_vs_slant": prof,
        "interior_maximum": bool(interior_max),
        "argmax_bin": argmax_bin,
        "VERDICT": ("HELD" if (interior_max and gain30 >= 0.05 and gain30 > best_other)
                    else "FALSIFIED"),
    }

    # --- P2. WindsorML learning curve ------------------------------------- #
    Xw_, yw_ = B["WindsorML"]["X"], B["WindsorML"]["targets"]["cd"]
    curve = []
    for nsub in (50, 100, 200, len(yw_)):
        vals = []
        for s in range(5 if nsub < len(yw_) else 1):
            rng = np.random.default_rng(2000 + s)
            idx = (rng.choice(len(yw_), size=nsub, replace=False)
                   if nsub < len(yw_) else np.arange(len(yw_)))
            f, fname, _ = flexible_r2(Xw_[idx], yw_[idx], min(K, max(3, nsub // 20)), SEED)
            vals.append(f)
        curve.append({"n": int(nsub), "flex_r2_mean": float(np.mean(vals)),
                      "flex_r2_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0})
    slope_tail = curve[-1]["flex_r2_mean"] - curve[-2]["flex_r2_mean"]
    out["P2_windsor_learning_curve"] = {
        "prediction": ("the flexible learning curve has plateaued by n=355 and lies "
                       "below R^2=0.5, i.e. the deficit is not sampling sparsity"),
        "curve": curve,
        "tail_slope_200_to_full": float(slope_tail),
        "VERDICT": ("HELD" if (curve[-1]["flex_r2_mean"] < 0.5 and slope_tail < 0.10)
                    else "FALSIFIED"),
    }

    # --- P3. DrivAerML scale composite ------------------------------------ #
    Xd_, pd2, yd_ = (B["DrivAerML"]["X"], B["DrivAerML"]["pnames"],
                     B["DrivAerML"]["targets"]["cd"])
    sc = [pd2.index(c) for c in ("Vehicle_Length", "Vehicle_Width", "Vehicle_Height")]
    r2_scale = r2_score(yd_, ols_oos(Xd_[:, sc], yd_, K, SEED))
    out["P3_drivaerml_scale_only"] = {
        "prediction": "Vehicle_Length/Width/Height alone reach R^2 >= 0.5 on drag",
        "r2_scale_params_only": float(r2_scale),
        "r2_all_16_params": float(r2_score(yd_, ols_oos(Xd_, yd_, K, SEED))),
        "VERDICT": "HELD" if r2_scale >= 0.5 else "FALSIFIED",
    }

    # --- reference-area convention sensitivity ---------------------------- #
    conv = {}
    for nm, geo, f_const, f_var, kg in (
            ("AhmedML", "ahmedml_geo.csv", "ahmedml_force.csv",
             "ahmedml_force_varref.csv", "run"),
            ("WindsorML", "windsorml_geo.csv", "windsorml_force.csv",
             "windsorml_force_varref.csv", "run"),
            ("DrivAerML", "drivaerml_geo.csv", "drivaerml_force_constref.csv",
             "drivaerml_force.csv", "Run")):
        row = {}
        for lbl, ff in (("const_area", f_const), ("per_geometry_area", f_var)):
            Xc, _, tc, _ = load_simple(geo, ff, kg, "run", ["cd"])
            row[lbl] = float(r2_score(tc["cd"], ols_oos(Xc, tc["cd"], K, SEED)))
        row["delta"] = row["const_area"] - row["per_geometry_area"]
        conv[nm] = row
    out["reference_area_convention"] = {
        "claim": ("null strength is a property of (benchmark, target, reference-area "
                  "convention), not of a benchmark"),
        "results": conv,
    }

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwrote", args.out)

    print("\n--- harmonised drag null ---")
    for b in order:
        print("  %-14s %.4f" % (b, out["candidate_vs_drag_null"]
                                ["harmonised_drag_null_r2"][b]))
    print("\n--- candidates (SURVIVES needs a perfect, correctly-signed ordering) ---")
    for nm, r in res.items():
        print("  %-28s rho=%+.3f p=%.4f  n=%d  %s"
              % (nm, r["spearman_rho"], r["exact_perm_p_two_sided"], r["n_used"],
                 "SURVIVES" if r["SURVIVES_prereg"] else "no"))
    print("\n--- point predictions ---")
    for k in ("P1_ahmed_regime_crossing", "P2_windsor_learning_curve",
              "P3_drivaerml_scale_only"):
        print("  %-30s %s" % (k, out[k]["VERDICT"]))
    print("  %-30s %s" % ("P4_missing_dof_calibration",
                          out["C3_leave_one_param_out"]["P4_verdict"]))
    print("\n--- design-space cap ---")
    print("  SS_between share = %.3f"
          % out["within_vs_between"]["share_of_total_SS_between"])
    return out, B, K, SEED


if __name__ == "__main__":
    main()
