"""Bootstrap confidence intervals for the residual-error Spearman headlines.

Case-level nonparametric bootstrap (resample cases with replacement, 10k
draws, percentile CI) for every arm with a persisted per-case
(score, error) dump:

  * ensemble-mean arm + corrected seed0/1/2 arms on AirfRANS
    (results/selective/selective_percase.json, produced by
    scripts/run_selective_prediction.py)
  * DeepCFD OOD arm, 3 seeds (results/deepcfd/deepcfd_trust_percase.json)

The per-seed backbone numbers (0.611/0.652/0.613) have no per-case dump in
the repo; the ensemble-mean arm (rho ~ 0.610, same 200 cases) is the
closest per-case-resolved proxy and is labelled as such.

Run:  .venv/Scripts/python.exe scripts/bootstrap_spearman_ci.py
Writes results/control/bootstrap_spearman_ci.json and prints a table.
"""

from __future__ import annotations

import json
import os

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)

N_BOOT = 10_000
SEED = 0
OUT = os.path.join("results", "control", "bootstrap_spearman_ci.json")


def boot_ci(scores: np.ndarray, errs: np.ndarray,
            rng: np.random.Generator) -> dict:
    n = len(scores)
    point = float(spearmanr(scores, errs).statistic)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws[b] = spearmanr(scores[idx], errs[idx]).statistic
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n_cases": int(n), "spearman": point,
            "ci95": [float(lo), float(hi)],
            "boot_mean": float(draws.mean()), "boot_std": float(draws.std()),
            "n_boot": int(len(draws))}


def main() -> int:
    rng = np.random.default_rng(SEED)
    out: dict = {"meta": {
        "method": f"case-level percentile bootstrap, {N_BOOT} draws, seed {SEED}",
        "note": ("Backbone per-seed arms have no per-case dump; the "
                 "ensemble-mean arm on the same 200 cases is the per-case-"
                 "resolved proxy for the AirfRANS trust-signal headline."),
    }, "arms": {}}

    sel = json.load(open(os.path.join(
        "results", "selective", "selective_percase.json"), encoding="utf-8"))
    for arm_name, rows in sel.items():
        if arm_name.startswith("_") or not isinstance(rows, list) or not rows:
            continue
        errs = np.array([r["rel_l2"] for r in rows], np.float64)
        for score_key in ("residual", "sigma_vel", "fused"):
            if score_key not in rows[0]:
                continue
            scores = np.array([r[score_key] for r in rows], np.float64)
            out["arms"][f"{arm_name}/{score_key}"] = boot_ci(scores, errs, rng)

    dc = json.load(open(os.path.join(
        "results", "deepcfd", "deepcfd_trust_percase.json"), encoding="utf-8"))
    for rec in dc.get("per_seed", []):
        seed = rec.get("seed")
        scores = np.array(rec["residual_norms"], np.float64)
        err_key = next(k for k in ("rel_l2", "rel_l2_errors", "rel_l2_speed",
                                   "errors", "rel_l2s") if k in rec)
        errs = np.array(rec[err_key], np.float64)
        out["arms"][f"deepcfd_seed{seed}/residual"] = boot_ci(scores, errs, rng)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)

    print(f"{'arm':40s} {'n':>4s} {'rho':>7s} {'95% CI':>18s}")
    for k, v in out["arms"].items():
        print(f"{k:40s} {v['n_cases']:4d} {v['spearman']:7.3f} "
              f"[{v['ci95'][0]:6.3f}, {v['ci95'][1]:6.3f}]")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
