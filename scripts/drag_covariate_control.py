"""How much of a reported AirfRANS drag rank correlation is FORCE information,
and how much is just knowing the case?

Companion control to ``scripts/drag_observability_roundtrip.py``. It applies to
the numbers ALREADY committed in ``results/control/`` (no new simulation), and it
gates the paper's drag-observability sentence.

===========================================================================
WHY THIS EXISTS
===========================================================================
``whitespace.md`` G1 reads ``rho_D = 0.839`` -- our integrator on the exact
ground-truth field, against the official labels -- as a CEILING imposed by the
representation. That reading assumes a marginal Spearman against official cd
measures drag observability. It does not, because cd is largely a smooth
function of case covariates and AirfRANS hands those covariates over in the
simulation NAME:

    airFoil2D_SST_<inlet_velocity>_<angle_of_attack>_<NACA digits...>

``Simulation.reset()`` itself parses fields 2 and 3 as U and alpha. So ANY
quantity that is monotone in alpha will rank-correlate with cd without
containing a single bit of integrated-force information.

===========================================================================
WHAT IS REPORTED
===========================================================================
* ``rho_cov``      -- Spearman(OLS fit of official cd on (U, alpha[, alpha^2]),
                      official cd). A trivial regression on the case label that
                      never touches a flow field. This is the number every
                      reported rho_D must be compared against.
* ``rho_partial``  -- Spearman-partial of each arm against official cd with
                      (U, alpha) linearly removed from the ranks. The part of
                      the arm's number that is genuinely force information.

Geometry (the NACA digits, also in the name) is deliberately NOT controlled for,
so ``rho_cov`` is a LOWER bound on what case identity alone buys.

**Reading rule (pre-registered with the round-trip script, P3).** If ``rho_cov``
>= an arm's marginal rho, that arm's marginal rho is not evidence of drag
observability, and the sentence "no scheme recovers drag beyond rho_D ~ 0.84"
must be withdrawn as stated.

    .venv/Scripts/python.exe scripts/drag_covariate_control.py
"""

from __future__ import annotations

import argparse
import json
import os

import neuroforge  # noqa: F401  -- MUST precede numpy/scipy (BLAS thread caps)

import numpy as np
from scipy.stats import rankdata, spearmanr

CACHE = os.path.join("results", "control", "_cache")


def covars_from_names(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Inlet velocity and angle of attack, parsed from the simulation name."""
    U = np.array([float(n.split("_")[2]) for n in names], np.float64)
    A = np.array([float(n.split("_")[3]) for n in names], np.float64)
    return U, A


def partial_spearman(x, y, covars) -> float:
    n = len(x)
    design = np.column_stack([np.ones(n)] + [rankdata(c) for c in covars])

    def resid(v):
        rv = rankdata(v)
        beta, *_ = np.linalg.lstsq(design, rv, rcond=None)
        return rv - design @ beta

    rx, ry = resid(x), resid(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_rho(cols, target) -> float:
    X = np.column_stack([np.ones(len(target))] + cols)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    return float(spearmanr(X @ beta, target).statistic)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="covariate control on AirfRANS force rankings")
    ap.add_argument("--out", default="results/control/drag_covariate_control.json")
    a = ap.parse_args(argv)

    labels = json.load(open(os.path.join(CACHE, "official_labels_full_test_n200.json"),
                            encoding="utf-8"))
    gt_nf = json.load(open(os.path.join(CACHE, "gt_nf_full_test_r128_n200.json"),
                           encoding="utf-8"))
    multi = json.load(open(os.path.join(CACHE, "gt_multischeme_full_test_r128_n200.json"),
                           encoding="utf-8"))

    names = [c["name"] for c in gt_nf]
    U, A = covars_from_names(names)
    off_cd = np.array([labels[n]["cd"] for n in names])
    off_cl = np.array([labels[n]["cl"] for n in names])

    baseline = {
        "rho_alpha_vs_cd": float(spearmanr(A, off_cd).statistic),
        "rho_absalpha_vs_cd": float(spearmanr(np.abs(A), off_cd).statistic),
        "rho_U_vs_cd": float(spearmanr(U, off_cd).statistic),
        "rho_fit_U_a_vs_cd": fit_rho([U, A], off_cd),
        "rho_fit_U_a_a2_vs_cd": fit_rho([U, A, A ** 2], off_cd),
        "rho_alpha_vs_cl": float(spearmanr(A, off_cl).statistic),
        "rho_fit_U_a_vs_cl": fit_rho([U, A], off_cl),
        "rho_fit_U_a_a2_vs_cl": fit_rho([U, A, A ** 2], off_cl),
    }

    # Arms: our integrator on ground-truth fields, three scheme families.
    arms = {"legacy_gt_nf": gt_nf}
    for k, v in multi.items():
        arms[f"multischeme_{k}"] = v
    # Predictions (Transolver, per seed) if the caches are present.
    for seed in (0, 1, 2):
        p = os.path.join(CACHE, f"seed{seed}_prednf.json")
        if os.path.exists(p):
            arms[f"pred_seed{seed}"] = json.load(open(p, encoding="utf-8"))

    per_arm = {}
    for tag, recs in arms.items():
        by = {c["name"]: c for c in recs}
        if not all(n in by for n in names):
            continue
        cd = np.array([by[n]["cd"] for n in names])
        cl = np.array([by[n]["cl"] for n in names])
        per_arm[tag] = {
            "rho_cd_marginal": float(spearmanr(cd, off_cd).statistic),
            "rho_cd_partial_U_a": partial_spearman(cd, off_cd, [U, A]),
            "rho_cd_partial_U_a_a2": partial_spearman(cd, off_cd, [U, A, A ** 2]),
            "rho_cl_marginal": float(spearmanr(cl, off_cl).statistic),
            "rho_cl_partial_U_a": partial_spearman(cl, off_cl, [U, A]),
        }

    beaten = [t for t, e in per_arm.items()
              if baseline["rho_fit_U_a_a2_vs_cd"] >= e["rho_cd_marginal"]]
    out = {
        "meta": {
            "purpose": "Separate force information from case identity in AirfRANS "
                       "drag/lift rank correlations. Covariates (U, alpha) are parsed "
                       "from the simulation NAME, exactly as Simulation.reset() does.",
            "n_cases": len(names),
            "geometry_not_controlled": "NACA digits are also in the name; rho_cov is "
                                       "therefore a LOWER bound on case identity.",
        },
        "covariate_baseline": baseline,
        "per_arm": per_arm,
        "arms_not_beating_covariate_baseline_on_cd": beaten,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[cov] wrote {a.out}")
    print(f"[cov] covariate baseline (U,a,a^2) -> cd : rho = "
          f"{baseline['rho_fit_U_a_a2_vs_cd']:.4f}   (alpha alone: "
          f"{baseline['rho_alpha_vs_cd']:.4f})")
    for t, e in per_arm.items():
        print(f"[cov] {t:22s} cd marginal={e['rho_cd_marginal']:+.4f} "
              f"partial={e['rho_cd_partial_U_a']:+.4f} | "
              f"cl marginal={e['rho_cl_marginal']:+.4f} "
              f"partial={e['rho_cl_partial_U_a']:+.4f}")
    print(f"[cov] arms NOT beating the covariate baseline on cd: {beaten}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
