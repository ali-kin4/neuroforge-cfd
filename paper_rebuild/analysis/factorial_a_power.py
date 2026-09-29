"""Factorial cell (a): reference-side power check on AirfRANS (PREREG_factorial.md).

The reference (frozen families of null_reference.py, chosen by training-CV) is fit on
the official `full` 800/200 split three times:
  R    : all case parameters
  R-t  : thickness withheld
  R-c  : camber and camber position withheld (5-digit reflex flag withheld too)
Encoding (implementation of "withhold thickness", logged in DEVIATIONS.md): the raw
AirfRANS name puts thickness in digit 3 for the 4-digit series and digit 4 for the
5-digit series, so the three fits use an aligned encoding
  [U, alpha, series5, camber, camber_pos, reflex, thickness].
Pass (design has power): withholding raises test drag MAE by >= 5 counts.
Outcomes: drag MAE in counts (1e-4), log(1 - rho_D), lift MAE, log(1 - rho_L), with
paired percentile intervals on the difference (withheld - full), case bootstrap B=10000.
"""
import csv
import json
import os
import sys

import neuroforge  # noqa: F401
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from null_reference import DATA, ROOT, bca, predict_both  # noqa: E402

OUT = os.path.join(ROOT, "paper_rebuild", "results", "factorial_a_power.json")
COLS = ["U", "alpha", "series5", "camber", "camber_pos", "reflex", "thickness"]
DROP = {"R": [], "R-t": ["thickness"], "R-c": ["camber", "camber_pos", "reflex"]}


def aligned(r):
    d = [float(r[f"naca_{i}"]) for i in range(4)]
    five = d[3] != 0.0
    if five:
        return [float(r["U"]), float(r["alpha"]), 1.0, d[0], d[1], d[2], d[3]]
    return [float(r["U"]), float(r["alpha"]), 0.0, d[0], d[1], 0.0, d[2]]


def mae_counts(y, p):
    return float(np.mean(np.abs(y - p)) * 1e4)


def log1m_rho(y, p):
    return float(np.log(1.0 - spearmanr(y, p).correlation))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main():
    with open(os.path.join(DATA, "airfrans", "airfrans_full_800_200.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    Xtr, Xte = np.array([aligned(r) for r in tr]), np.array([aligned(r) for r in te])
    out = {"encoding": COLS, "n_train": len(tr), "n_test": len(te)}
    for tgt, metrics in (("cd", {"mae_counts": mae_counts, "log1m_rho": log1m_rho}),
                         ("cl", {"mae": mae, "log1m_rho": log1m_rho})):
        ytr = np.array([float(r[tgt]) for r in tr])
        yte = np.array([float(r[tgt]) for r in te])
        preds, res = {}, {}
        for name, drop in DROP.items():
            keep = [i for i, c in enumerate(COLS) if c not in drop]
            pq, pk, h = predict_both(Xtr[:, keep], ytr, Xte[:, keep])
            preds[name] = pk if h["ref"] == "K" else pq
            res[name] = {"family": h["ref"], "tuning": h,
                         **{m: bca(yte, preds[name], f) for m, f in metrics.items()}}
        for name in ("R-t", "R-c"):
            for m, f in metrics.items():
                res[name][f"delta_vs_R_{m}"] = paired_diff(yte, preds[name], preds["R"], f)
        out[tgt] = res
    d = out["cd"]
    out["verdict"] = {n: ("PASS" if d[n]["delta_vs_R_mae_counts"]["point"] >= 5.0 else "FAIL")
                          for n in ("R-t", "R-c")}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    for tgt in ("cd", "cl"):
        for n in DROP:
            r = out[tgt][n]
            print(tgt, n, r["family"], {m: round(v["point"], 4) for m, v in r.items()
                                        if isinstance(v, dict) and "point" in v})
    print("verdict", out["verdict"])


def paired_diff(y, pa, pb, f, b=10000, seed=0):
    """Point and percentile 95% interval of f(y,pa) - f(y,pb), paired cases."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pt = f(y, pa) - f(y, pb)
    idx = rng.integers(0, n, size=(b, n))
    boot = np.array([f(y[i], pa[i]) - f(y[i], pb[i]) for i in idx])
    return {"point": pt, "pct95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


if __name__ == "__main__":
    main()
