"""Declared secondary (DEVIATIONS.md, 2026-09-29): a stronger flexible fit on the two
benchmarks where the frozen reference is claimed WEAK (WindsorML, AhmedML).

Why: A1's kernel-ridge grid hit its edge on these benchmarks, so A1 is anti-conservative
exactly where it is used to say "the case parameters do not determine the target". This
secondary is reported beside A1 and never replaces it.

Families (numpy/scipy only):
  KW : RBF kernel ridge, wider grid  l in {0.1,0.2,0.35,0.5,1,2,4,8,16,32},
       lambda in {1e-8,...,1}, tuned by inner 5-fold CV on the training fold.
  GP : Gaussian process, ARD RBF kernel (one length scale per parameter) + white noise,
       hyperparameters by maximising the log marginal likelihood (L-BFGS-B, 3 restarts).
  (tuned gradient boosting was declared; it is run only if scikit-learn is importable.)
Protocol: 5 x repeated 10-fold OOS (seeds 0-4); R^2 per repeat, mean and range reported.
Data: WindsorML complete metadata (n=350), AhmedML (n=500); targets cd, cl, constant
reference area.
Reading, fixed now: if the best family reaches R^2 >= 0.6 on WindsorML drag, WindsorML is
NOT described as a benchmark whose drag the case parameters fail to determine.
"""
import json
import os
import sys

import neuroforge  # noqa: F401
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from null_reference import ROOT, _folds, _sqdist, _standardise, load_kfold, r2, windsor_extra  # noqa: E402

OUT = os.path.join(ROOT, "paper_rebuild", "results", "strong_flexible_secondary.json")
LSW = (0.1, 0.2, 0.35, 0.5, 1, 2, 4, 8, 16, 32)
LAMW = tuple(10.0 ** k for k in range(-8, 1))


def kw_fit(Xtr, ytr, Xte):
    n = len(ytr)
    best = (np.inf, None, None)
    folds = _folds(n, 5)
    errs = np.zeros((len(LSW), len(LAMW)))
    for f in folds:
        m = np.ones(n, bool)
        m[f] = False
        Ztr, Zte = _standardise(Xtr[m], Xtr[f])
        d = Ztr.shape[1]
        ym, ys = ytr[m].mean(), ytr[m].std() or 1.0
        t = (ytr[m] - ym) / ys
        D, Ds = _sqdist(Ztr, Ztr), _sqdist(Zte, Ztr)
        for i, ls in enumerate(LSW):
            w, V = np.linalg.eigh(np.exp(-D / (2 * ls * ls * d)))
            Vt, Ks = V.T @ t, np.exp(-Ds / (2 * ls * ls * d)) @ V
            for j, lam in enumerate(LAMW):
                errs[i, j] += np.sum((Ks @ (Vt / (w + lam)) * ys + ym - ytr[f]) ** 2)
    i, j = np.unravel_index(np.argmin(errs), errs.shape)
    ls, lam = LSW[i], LAMW[j]
    Ztr, Zte = _standardise(Xtr, Xte)
    d = Ztr.shape[1]
    ym, ys = ytr.mean(), ytr.std() or 1.0
    K = np.exp(-_sqdist(Ztr, Ztr) / (2 * ls * ls * d))
    a = np.linalg.solve(K + lam * np.eye(n), (ytr - ym) / ys)
    edge = i in (0, len(LSW) - 1) or j in (0, len(LAMW) - 1)
    return np.exp(-_sqdist(Zte, Ztr) / (2 * ls * ls * d)) @ a * ys + ym, edge


def _ard_k(A, Bm, ell):
    return np.exp(-0.5 * _sqdist(A / ell, Bm / ell))


def gp_fit(Xtr, ytr, Xte, restarts=3):
    Ztr, Zte = _standardise(Xtr, Xte)
    n, d = Ztr.shape
    ym, ys = ytr.mean(), ytr.std() or 1.0
    t = (ytr - ym) / ys

    def nlml(th):
        ell, sf2, sn2 = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1]) + 1e-8
        K = sf2 * _ard_k(Ztr, Ztr, ell) + sn2 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        al = np.linalg.solve(L.T, np.linalg.solve(L, t))
        return 0.5 * t @ al + np.log(np.diag(L)).sum()

    rng = np.random.default_rng(0)
    best = None
    for r in range(restarts):
        th0 = np.concatenate([np.log(np.full(d, 1.0) * (1 if r == 0 else rng.uniform(0.3, 3, d))),
                              [0.0, np.log(0.1)]])
        res = minimize(nlml, th0, method="L-BFGS-B",
                       bounds=[(-4, 5)] * d + [(-5, 3), (-12, 1)])
        if best is None or res.fun < best.fun:
            best = res
    th = best.x
    ell, sf2, sn2 = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1]) + 1e-8
    K = sf2 * _ard_k(Ztr, Ztr, ell) + sn2 * np.eye(n)
    a = np.linalg.solve(K, t)
    return sf2 * _ard_k(Zte, Ztr, ell) @ a * ys + ym, ell.tolist()


def repeated(X, y, fam, reps=5):
    out = []
    for rep in range(reps):
        p = np.empty(len(y))
        for f in _folds(len(y), 10, seed=rep):
            m = np.ones(len(y), bool)
            m[f] = False
            p[f] = fam(X[m], y[m], X[f])[0]
        out.append(float(r2(y, p)))
    return {"r2_per_repeat": out, "r2_mean": float(np.mean(out)), "r2_min": min(out), "r2_max": max(out)}


def main():
    win, wcols = load_kfold("windsorml", "windsorml_355.csv", "run_id", ("cd", "cl"), extra=windsor_extra())
    ahm, acols = load_kfold("ahmedml", "ahmedml_500.csv", "run_id", ("cd", "cl"))
    res = {}
    for bench, dd in (("WindsorML", win), ("AhmedML", ahm)):
        for t, (_, X, y) in dd.items():
            key = f"{bench}/{t}"
            res[key] = {"KW": repeated(X, y, kw_fit), "GP": repeated(X, y, gp_fit)}
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
                res[key]["GBM"] = "not implemented in this pass"
            except ImportError:
                res[key]["GBM"] = "not run: scikit-learn unavailable (PyPI unreachable 2026-09-29)"
            print(key, {k: (round(v["r2_mean"], 3) if isinstance(v, dict) else v) for k, v in res[key].items()},
                  flush=True)
            with open(OUT, "w") as fh:
                json.dump(res, fh, indent=1)
    wd = max(res["WindsorML/cd"][k]["r2_mean"] for k in ("KW", "GP"))
    res["reading"] = ("WindsorML drag NOT determined by published parameters under these families"
                      if wd < 0.6 else "WindsorML drag reachable (>=0.6): do not call it a positive control")
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(res["reading"])


if __name__ == "__main__":
    main()
