"""Factorial cell (b) scoring: Transolver official-integrator forces vs the frozen reference R.

Reads paper_rebuild/results/factorial_b_forces.jsonl (factorial_b_official_forces.py).
Interpretation of PREREG_factorial.md "seeds enter as a random effect, seed-averaged per
case for the primary estimate", fixed here before the full results were read:
  MAE outcomes  : per-case error averaged over the 5 seeds, then paired with R per case.
  rank outcomes : log(1 - rho) computed per seed, averaged over seeds (per-seed values kept).
  secondary     : the seed-ensemble mean prediction, scored like a single model.
Gate: median |C_D(gt) - C_D(official)| < 1 count and rho_D(gt, official) > 0.999.
R   : frozen reference of null_reference.py on the official split (raw parameter
      encoding, family by training CV); negative control = R refit on permuted training
      labels (must be no better than intercept-only within the margin).
Paired case bootstrap (B = 10000, seed 0), percentile 95% intervals on differences.
Margin: 5 drag counts on MAE; log 1.25 on log(1 - rho).
"""
import csv
import json
import os
import sys

import neuroforge  # noqa: F401
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from null_reference import DATA, ROOT, predict_both  # noqa: E402

IN = os.path.join(ROOT, "paper_rebuild", "results", "factorial_b_forces.jsonl")
OUT = os.path.join(ROOT, "paper_rebuild", "results", "factorial_b_compare.json")
SEEDS = [f"seed{s}" for s in range(5)]
B = 10000


def l1r(y, p):
    return float(np.log(max(1.0 - spearmanr(y, p).correlation, 1e-12)))


def boot(fn, n, b=B, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(b)])
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def reference_preds(names, target, permute=False):
    with open(os.path.join(DATA, "airfrans", "airfrans_full_800_200.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = ["U", "alpha", "naca_0", "naca_1", "naca_2", "naca_3"]
    tr = [r for r in rows if r["split"] == "train"]
    te = {r["case_id"]: r for r in rows if r["split"] == "test"}
    Xtr = np.array([[float(r[c]) for c in cols] for r in tr])
    ytr = np.array([float(r[target]) for r in tr])
    if permute:
        ytr = np.random.default_rng(0).permutation(ytr)
    Xte = np.array([[float(te[n][c]) for c in cols] for n in names])
    pq, pk, h = predict_both(Xtr, ytr, Xte)
    return (pk if h["ref"] == "K" else pq), h["ref"]


def main():
    recs = [json.loads(l) for l in open(IN) if l.strip()]
    names = [r["name"] for r in recs]
    n = len(recs)
    off = {t: np.array([r["official"][t] for r in recs]) for t in ("cd", "cl")}
    gt = {t: np.array([r["gt"][t] for r in recs]) for t in ("cd", "cl")}
    gate = {"median_abs_dcd_counts": float(np.median(np.abs(gt["cd"] - off["cd"])) * 1e4),
            "rho_d": float(spearmanr(gt["cd"], off["cd"]).correlation)}
    gate["pass"] = gate["median_abs_dcd_counts"] < 1.0 and gate["rho_d"] > 0.999
    out = {"n": n, "gate": gate}
    if not gate["pass"]:
        json.dump(out, open(OUT, "w"), indent=1)
        sys.exit("GATE FAILED: model forces not scored")

    for t, scale, unit in (("cd", 1e4, "counts"), ("cl", 1.0, "abs")):
        y = off[t]
        P = np.array([[r[s][t] for s in SEEDS] for r in recs])      # (n, 5)
        ref, fam = reference_preds(names, t)
        perm, _ = reference_preds(names, t, permute=True)
        err_m = np.abs(P - y[:, None]).mean(1) * scale                 # seed-averaged per case
        err_r = np.abs(ref - y) * scale
        err_e = np.abs(P.mean(1) - y) * scale                          # ensemble secondary
        err_perm = np.abs(perm - y) * scale
        err_int = np.abs(y.mean() - y) * scale
        d = err_m - err_r
        res = {
            "reference_family": fam,
            f"mae_{unit}": {"transolver_seedavg": float(err_m.mean()),
                            "transolver_ensemble": float(err_e.mean()),
                            "reference": float(err_r.mean()),
                            "neg_control_permuted": float(err_perm.mean()),
                            "intercept_only": float(err_int.mean()),
                            "per_seed": [float(np.abs(P[:, k] - y).mean() * scale) for k in range(5)]},
            "delta_mae_transolver_minus_ref": {"point": float(d.mean()),
                                               "pct95": boot(lambda i: d[i].mean(), n),
                                               "cases_ref_better": int((d > 0).sum())},
            "log1m_rho": {"transolver_per_seed": [l1r(y, P[:, k]) for k in range(5)],
                          "transolver_seedavg": float(np.mean([l1r(y, P[:, k]) for k in range(5)])),
                          "transolver_ensemble": l1r(y, P.mean(1)),
                          "reference": l1r(y, ref)},
            "rho": {"transolver_per_seed": [float(spearmanr(y, P[:, k]).correlation) for k in range(5)],
                    "reference": float(spearmanr(y, ref).correlation)},
        }
        dl = lambda i: (np.mean([l1r(y[i], P[i, k]) for k in range(5)]) - l1r(y[i], ref[i]))  # noqa: E731
        res["delta_log1m_rho"] = {"point": res["log1m_rho"]["transolver_seedavg"] - res["log1m_rho"]["reference"],
                                  "pct95": boot(dl, n, b=2000)}
        if t == "cd":
            for comp in ("cdp", "cdv"):
                yc = np.array([r["official"][comp] for r in recs])
                Pc = np.array([[r[s][comp] for s in SEEDS] for r in recs])
                res[f"mae_counts_{comp}_transolver_seedavg"] = float(np.abs(Pc - yc[:, None]).mean() * 1e4)
        out[t] = res
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
