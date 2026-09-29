"""Frozen metadata-only reference for aerodynamic surrogate benchmarks (analysis A1).

PROTOCOL (fixed in this docstring and committed BEFORE the first run; any later
change is recorded in paper_rebuild/DEVIATIONS.md with both results reported).

Purpose
-------
S_ref = the score a regressor attains from a benchmark's published per-case
parameters alone (no geometry file, no field), on the benchmark's own split and
headline metric. A published model's margin is Delta = S_model - S_ref.
The reference is a legitimate surrogate of the parameter -> scalar map, not a
leak: where the parameters fully specify the simulation it IS a surrogate of it.

Families (identical recipe on every benchmark; no per-benchmark feature additions)
---------------------------------------------------------------------------------
Q  : ordinary least squares on [1, z, z^2], z = train-standardised published
     parameters; squares are omitted for binary (<= 2 distinct values) columns.
K  : RBF kernel ridge on z, target standardised on train.
     k(a,b) = exp(-||a-b||^2 / (2 l^2 d)), d = number of columns,
     l in {0.25, 0.5, 1, 2, 4, 8}, lambda in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
     (l, lambda) chosen by 5-fold CV MSE on the TRAINING cases only (seed 0).
REF: the family (Q or K) with the lower 5-fold CV MSE on the training cases.
     The test set never chooses the family or its hyperparameters.
Both families are always reported; REF is the headline.

Splits (the benchmark's own arrangement)
----------------------------------------
AirfRANS     official `full` 800/200; params U, alpha, NACA digits (alpha^2 is NOT
             added by hand -- family Q squares every column). Targets cd, cl.
             Headline metric: Spearman (AirfRANS Table 3 convention).
DrivAerML    PhysicsNeMo-CFD 436/48; 16 morph params; target drag force [N].
             Headline R^2 (2507.10747 Tables 6/7); Spearman secondary.
DrivAerNet++ official ids, full 1154-design test set; 7 family tokens + 23 params
             (zero where unpublished, as in covariate_null_crossbench.py). Target
             cd. Headline R^2; MSE secondary (both published in its Table 4).
AhmedML      no published split -> 10-fold OOS (seed 0; K/Q tuned inside each
             training fold). Targets cd, cl (constant reference area). R^2.
WindsorML    no published split -> 10-fold OOS. PRIMARY: complete metadata (the
             aggregate CSV's 7 columns + ratio_length_front_rear from the per-run
             files; n=350). SENSITIVITY: aggregate CSV only (n=355). cd, cl. R^2.

Statistics
----------
Unit = case. Point estimate on the scored set; 95% BCa case-bootstrap interval
(B = 10000, seed 0), percentile interval alongside. For K-fold protocols the
bootstrap resamples the pooled out-of-fold predictions (fold-assignment variance
is not included; stated as a limitation).

Reading
-------
This run re-measures benchmarks whose linear nulls were already seen, so its
values are FROZEN-PROTOCOL re-measurements, not confirmations. Confirmatory use
of this harness is reserved for benchmarks, splits and models scored after the
commit that introduces this file (A3 resplits, A4 new benchmarks).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import neuroforge  # noqa: F401  thread cap before numpy
import numpy as np
from scipy.stats import norm, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "src", "neuroforge", "nullbench", "data")
OUT = os.path.join(ROOT, "paper_rebuild", "results", "null_reference.json")

LS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAMS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
B = 10000


# ----------------------------------------------------------------------------- metrics
def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def spear(y, p):
    return spearmanr(y, p).correlation


def mse(y, p):
    return float(np.mean((y - p) ** 2))


METRICS = {"r2": r2, "spearman": spear, "mse": mse}


def bca(y, p, fn, b=B, seed=0):
    """BCa and percentile 95% intervals, case bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y)
    theta = fn(y, p)
    idx = rng.integers(0, n, size=(b, n))
    boot = np.array([fn(y[i], p[i]) for i in idx])
    boot = boot[np.isfinite(boot)]
    z0 = norm.ppf(np.clip(np.mean(boot < theta), 1e-6, 1 - 1e-6))
    jack = np.array([fn(np.delete(y, k), np.delete(p, k)) for k in range(n)])
    jm = jack.mean()
    a = np.sum((jm - jack) ** 3) / (6.0 * np.sum((jm - jack) ** 2) ** 1.5 + 1e-300)
    out = []
    for q in (0.025, 0.975):
        zq = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
        out.append(float(np.quantile(boot, adj)))
    pct = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return {"point": float(theta), "bca95": out, "pct95": pct}


# ----------------------------------------------------------------------------- families
def _standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _quad(Z, binary):
    cols = [np.ones(len(Z)), *Z.T] + [Z[:, j] ** 2 for j in range(Z.shape[1]) if not binary[j]]
    return np.column_stack(cols)


def fit_q(Xtr, ytr, Xte):
    binary = [len(np.unique(Xtr[:, j])) <= 2 for j in range(Xtr.shape[1])]
    Ztr, Zte = _standardise(Xtr, Xte)
    beta, *_ = np.linalg.lstsq(_quad(Ztr, binary), ytr, rcond=None)
    return _quad(Zte, binary) @ beta


def _sqdist(A, Bm):
    return np.maximum((A * A).sum(1)[:, None] + (Bm * Bm).sum(1)[None, :] - 2 * A @ Bm.T, 0.0)


def fit_k(Xtr, ytr, Xte, ls, lam):
    Ztr, Zte = _standardise(Xtr, Xte)
    d = Ztr.shape[1]
    ym, ys = ytr.mean(), ytr.std() or 1.0
    t = (ytr - ym) / ys
    K = np.exp(-_sqdist(Ztr, Ztr) / (2 * ls * ls * d))
    alpha = np.linalg.solve(K + lam * np.eye(len(K)), t)
    Ks = np.exp(-_sqdist(Zte, Ztr) / (2 * ls * ls * d))
    return Ks @ alpha * ys + ym


def _folds(n, k, seed=0):
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def tune(Xtr, ytr, k=5):
    """5-fold CV on training cases: best (ls, lam) for K, and CV MSE of Q and K.
    Uses one eigendecomposition per (fold, ls) so the lambda path is cheap."""
    n = len(ytr)
    folds = _folds(n, k)
    err_k = np.zeros((len(LS), len(LAMS)))
    err_q = 0.0
    for f in folds:
        m = np.ones(n, bool)
        m[f] = False
        err_q += np.sum((fit_q(Xtr[m], ytr[m], Xtr[f]) - ytr[f]) ** 2)
        Ztr, Zte = _standardise(Xtr[m], Xtr[f])
        d = Ztr.shape[1]
        ym, ys = ytr[m].mean(), ytr[m].std() or 1.0
        t = (ytr[m] - ym) / ys
        D, Ds = _sqdist(Ztr, Ztr), _sqdist(Zte, Ztr)
        for i, ls in enumerate(LS):
            w, V = np.linalg.eigh(np.exp(-D / (2 * ls * ls * d)))
            Vt = V.T @ t
            Ks = np.exp(-Ds / (2 * ls * ls * d)) @ V
            for j, lam in enumerate(LAMS):
                pred = Ks @ (Vt / (w + lam)) * ys + ym
                err_k[i, j] += np.sum((pred - ytr[f]) ** 2)
    i, j = np.unravel_index(np.argmin(err_k), err_k.shape)
    return {"ls": LS[i], "lam": LAMS[j], "cv_mse_k": float(err_k[i, j] / n),
            "cv_mse_q": float(err_q / n), "edge": bool(i in (0, len(LS) - 1) or j in (0, len(LAMS) - 1))}


def predict_both(Xtr, ytr, Xte):
    h = tune(Xtr, ytr)
    pq = fit_q(Xtr, ytr, Xte)
    pk = fit_k(Xtr, ytr, Xte, h["ls"], h["lam"])
    h["ref"] = "K" if h["cv_mse_k"] < h["cv_mse_q"] else "Q"
    return pq, pk, h


# ----------------------------------------------------------------------------- data
def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _arr(rows, cols):
    return np.array([[float(r[c]) for c in cols] for r in rows])


def load_airfrans():
    rows = read(os.path.join(DATA, "airfrans", "airfrans_full_800_200.csv"))
    cols = ["U", "alpha", "naca_0", "naca_1", "naca_2", "naca_3"]
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    return {t: ("split", _arr(tr, cols), _arr(tr, [t])[:, 0], _arr(te, cols), _arr(te, [t])[:, 0])
            for t in ("cd", "cl")}, cols


def load_split_csv(sub, fname, target, drop):
    rows = read(os.path.join(DATA, sub, fname))
    cols = [c for c in rows[0] if c not in drop]
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] != "train"]
    return {target: ("split", _arr(tr, cols), _arr(tr, [target])[:, 0], _arr(te, cols),
                     _arr(te, [target])[:, 0])}, cols


def load_kfold(sub, fname, idcol, targets, extra=None):
    rows = read(os.path.join(DATA, sub, fname))
    cols = [c for c in rows[0] if c not in (idcol, "cd", "cl")]
    if extra is not None:
        rows = [r for r in rows if int(r[idcol]) in extra]
        for r in rows:
            r["ratio_length_front_rear"] = extra[int(r[idcol])]
        cols = cols + ["ratio_length_front_rear"]
    return {t: ("kfold", _arr(rows, cols), _arr(rows, [t])[:, 0]) for t in targets}, cols


def windsor_extra():
    d = os.path.join(ROOT, "data", "crossbench", "windsor_perrun")
    if not os.path.isdir(d):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from windsor_completed_metadata import fetch_perrun
        fetch_perrun()
    out = {}
    for f in os.listdir(d):
        out[int(f[4:-4])] = next(iter(read(os.path.join(d, f))))["ratio_length_front_rear"]
    return out


# ----------------------------------------------------------------------------- run
def score(y, pq, pk, h, metrics):
    res = {"tuning": h, "n_scored": int(len(y))}
    for fam, p in (("Q", pq), ("K", pk)):
        res[fam] = {m: bca(y, p, METRICS[m]) for m in metrics}
    res["REF"] = dict(res[h["ref"]], family=h["ref"])
    return res


def run_one(spec, metrics):
    if spec[0] == "split":
        _, Xtr, ytr, Xte, yte = spec
        pq, pk, h = predict_both(Xtr, ytr, Xte)
        return score(yte, pq, pk, h, metrics)
    _, X, y = spec
    pq, pk = np.empty(len(y)), np.empty(len(y))
    hs = []
    for f in _folds(len(y), 10):
        m = np.ones(len(y), bool)
        m[f] = False
        a, b_, h = predict_both(X[m], y[m], X[f])
        pq[f], pk[f] = a, b_
        hs.append(h)
    # reference family by pooled training-CV MSE across outer folds
    cq = sum(h["cv_mse_q"] for h in hs)
    ck = sum(h["cv_mse_k"] for h in hs)
    agg = {"ref": "K" if ck < cq else "Q", "per_fold": hs,
           "any_edge": any(h["edge"] for h in hs)}
    return score(y, pq, pk, agg, metrics)


def main():
    t0 = time.time()
    plan = []
    air, _ = load_airfrans()
    plan += [("AirfRANS", t, air[t], ["spearman", "r2"]) for t in ("cd", "cl")]
    dml, _ = load_split_csv("drivaerml", "drivaerml_pn_436_48.csv", "drag_force_N", ("run_id", "split", "drag_force_N"))
    plan.append(("DrivAerML", "drag_force_N", dml["drag_force_N"], ["r2", "spearman", "mse"]))
    dnp, _ = load_split_csv("drivaernet", "drivaernet_pp_5819_1154.csv", "cd", ("design_id", "split", "cd"))
    plan.append(("DrivAerNet++", "cd", dnp["cd"], ["r2", "mse", "spearman"]))
    ahm, _ = load_kfold("ahmedml", "ahmedml_500.csv", "run_id", ("cd", "cl"))
    plan += [("AhmedML", t, ahm[t], ["r2", "spearman"]) for t in ("cd", "cl")]
    win, _ = load_kfold("windsorml", "windsorml_355.csv", "run_id", ("cd", "cl"), extra=windsor_extra())
    plan += [("WindsorML", t, win[t], ["r2", "spearman"]) for t in ("cd", "cl")]
    wagg, _ = load_kfold("windsorml", "windsorml_355.csv", "run_id", ("cd", "cl"))
    plan += [("WindsorML[aggregate-csv sensitivity]", t, wagg[t], ["r2", "spearman"]) for t in ("cd", "cl")]

    only = set(sys.argv[1:])
    results = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            results = json.load(fh).get("results", {})
    for bench, tgt, spec, metrics in plan:
        key = f"{bench}/{tgt}"
        if only and bench not in only:
            continue
        t1 = time.time()
        results[key] = run_one(spec, metrics)
        r = results[key]["REF"]
        print(f"{key:48s} REF={r['family']} " + " ".join(
            f"{m}={r[m]['point']:.4f}[{r[m]['bca95'][0]:.4f},{r[m]['bca95'][1]:.4f}]" for m in metrics),
            f"({time.time() - t1:.0f}s)", flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump({"protocol": "see paper_rebuild/analysis/null_reference.py docstring",
                       "results": results}, fh, indent=1)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
