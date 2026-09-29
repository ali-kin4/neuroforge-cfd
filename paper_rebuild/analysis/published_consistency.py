"""Internal consistency of published DrivAerNet++ drag rows (analysis A6).

For each published row reporting both MSE and R^2 on the official DrivAerNet++
test split, compute the R^2 implied by the reported MSE and the variance of the
official 1154-design test labels: R2_implied = 1 - MSE / Var(test Cd), ddof 0.

A row whose reported R^2 and reported MSE disagree cannot be placed on one scale.
Such rows are compared in MSE, the scale-absolute quantity, and the discrepancy
is stated without attributing a cause: a different test subset, R^2 convention
or unit would each produce it.

Published values (read at source by the literature review; see
docs/paper/review/null_travels.md Sec 2.2):
  DrivAerNet++ NeurIPS 2024 D&B Table 4: GCNN, RegDGCNN, PointNet
  TripNet arXiv:2503.17400 Table 5; PointNet2D+BiLSTM arXiv:2601.02112 Table 1
"""
import csv
import json
import os

import neuroforge  # noqa: F401
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(ROOT, "src", "neuroforge", "nullbench", "data", "drivaernet",
                   "drivaernet_pp_5819_1154.csv")
OUT = os.path.join(ROOT, "paper_rebuild", "results", "published_consistency.json")

PUBLISHED = [  # name, MSE, R2, source
    ("GCNN", 17.1e-5, 0.596, "DrivAerNet++ Table 4"),
    ("RegDGCNN", 14.2e-5, 0.641, "DrivAerNet++ Table 4"),
    ("PointNet", 14.9e-5, 0.643, "DrivAerNet++ Table 4"),
    ("TripNet", 9.1e-5, 0.957, "arXiv:2503.17400 Table 5"),
    ("PointNet2D+BiLSTM", 6.50e-5, 0.9528, "arXiv:2601.02112 Table 1"),
]


def main():
    with open(CSV, newline="") as fh:
        y = np.array([float(r["cd"]) for r in csv.DictReader(fh) if r["split"] != "train"])
    v = float(y.var())
    rows = []
    for name, m, r2, src in PUBLISHED:
        rows.append({"model": name, "source": src, "mse": m, "r2_reported": r2,
                     "r2_implied_by_mse": 1 - m / v, "mse_implied_by_r2": (1 - r2) * v,
                     "abs_r2_gap": abs(r2 - (1 - m / v))})
    out = {"n_test": int(len(y)), "var_test_ddof0": v, "rows": rows}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        print(f"{r['model']:18s} R2 reported {r['r2_reported']:.3f}  implied {r['r2_implied_by_mse']:.3f}")


if __name__ == "__main__":
    main()
