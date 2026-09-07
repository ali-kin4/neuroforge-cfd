"""Is the conformal certificate wide BECAUSE of the residual floor?

THE CLAIM UNDER TEST
--------------------
`sections/residual_floor_theorem.tex` says the conformal bound on |dC_D| is wide
BECAUSE ~86% of a typical prediction's score is floor
(`floor_share_of_typical_score = 0.8636`). `docs/paper/review/theorem_targets.md`
sec 2.4 disputes the causal clause: |dC_D| has `Q_0.9/median = 15.1`, so an
UNINFORMATIVE score already gives 15.1x and the deployed monitor's measured width is
already several-fold better than no monitor at all. On that reading the width is the
heavy tail of the drag-error distribution, not the floor.

THE ORACLE COUNTERFACTUAL
-------------------------
Remove the floor exactly and re-run the identical protocol. The right instrument is a
FIELD DIFFERENCE, not a difference of norms: `||R_h(u_hat)|| - ||r*||` is not a valid
norm decomposition (it is negative on a few percent of cases). So

    sigma'_i := || R_h(u_hat_i) - R_h(u*_i) ||

with the SAME reduction as `Diagnostics.residual_norm()` -- RMS over all cells of
sqrt(c^2 + mx^2 + my^2) -- applied to the component-wise difference of the residual
maps. The invalid norm-difference version is computed too, and reported only to show
what it does, never as the instrument.

TWO CAVEATS, stated regardless of outcome
-----------------------------------------
1. sigma' uses r*, unavailable at deployment. It is an oracle counterfactual: the
   right instrument for an impossibility argument, the wrong thing to present as a
   method. It is also the CEILING on what any learned floor predictor r_hat* could
   buy (theorem_targets.md sec 3.4), so it closes the useful half of Target C too.
2. sigma' removes MORE than the floor. The deployed Transolver's u_hat is itself
   rasterised from a point cloud through the same pipeline as u*, so the field
   difference partially cancels SHARED rasterisation noise as well as the floor.
   A large improvement is therefore an UPPER bound on what floor removal could buy,
   not evidence that the floor alone was responsible.

PRE-REGISTERED DECISION RULE (committed before the run; from theorem_targets.md
sec 2.5's table, with the registered prediction "it drops, but to roughly 3-4x, not
to 1-2x"). Primary statistic: `bound_over_error = median_i(c sigma'_i) / median_i(E_i)`,
averaged over the three deployed seeds, 400 random half-splits, target coverage 0.90,
using the EXACT split-conformal order statistic (`conformal_quantile`).

  FLOOR-CAUSAL       mean bound_over_error <= 2.0
                     => the floor is causally responsible for the width; the "because
                     86% is floor" clause is supported and may stay.
  FLOOR-NOT-CAUSAL   mean bound_over_error >= 4.0
                     => the floor is not the cause; the width is the drag-error tail
                     plus the monitor's imperfect correlation with error. The causal
                     clause must be DELETED from the manuscript.
  PARTIAL            2.0 < mean < 4.0
                     => the floor is a contributor but not the explanation; the
                     clause must be softened from "because" to a measured share.

Reference points reported on the same axis so the number can be read:
  uninformative  sigma == const  -> the marginal split-conformal interval (~15x)
  deployed       sigma  = ||R_h(u_hat)||
  oracle-floor   sigma' = ||R_h(u_hat) - R_h(u*)||
  perfect        sigma propto E  -> 1x by construction

Usage
-----
    python scripts/floor_subtraction_gate.py            # 3 seeds, 200 cases, CPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import neuroforge  # noqa: F401
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functional_audit_gate import (  # noqa: E402
    AG_DIR,
    conformal_quantile,
    encode_case,
    load_pairs,
    spearman,
    wrap,
)

from neuroforge.core.config import Config  # noqa: E402
from neuroforge.core.types import DTYPE  # noqa: E402
from neuroforge.physics.residuals import PhysicsChecker  # noqa: E402

OUT_JSON = "results/review/floor_subtraction_gate.json"


def maps(field, case, checker):
    d = checker.diagnose(field, case)
    return (np.asarray(d.continuity, np.float64),
            np.asarray(d.momentum_x, np.float64),
            np.asarray(d.momentum_y, np.float64))


def rnorm(c, mx, my):
    """Exactly Diagnostics.residual_norm(): RMS over all cells of |(c, mx, my)|."""
    return float(np.sqrt(np.mean(c * c + mx * mx + my * my)))


def conformal(score, err, alpha=0.10, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    cs, cov, w = [], [], []
    for _ in range(reps):
        i = rng.permutation(len(err))
        cal, tst = i[: len(i) // 2], i[len(i) // 2:]
        c = conformal_quantile(err[cal] / np.maximum(score[cal], 1e-30), alpha=alpha)
        cs.append(c)
        cov.append(float((err[tst] <= c * score[tst]).mean()))
        w.append(float(np.median(c * score[tst])))
    med_e = float(np.median(err))
    return {"c_median": float(np.median(cs)),
            "coverage_mean": float(np.mean(cov)),
            "coverage_std": float(np.std(cov)),
            "median_bound_width": float(np.median(w)),
            "bound_over_error": float(np.median(w) / max(med_e, 1e-30)),
            "median_abs_err": med_e,
            "spearman_score_vs_err": spearman(score, err)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args(argv)

    t0 = time.time()
    checker = PhysicsChecker(Config().physics)
    pairs = load_pairs(a.limit)
    seeds = [s for s in a.seeds if os.path.isdir(os.path.join(AG_DIR, f"seed{s}"))]
    print(f"{len(pairs)} cases, seeds {seeds}", flush=True)

    per_case = {}
    for i, (case, gt) in enumerate(pairs):
        st = encode_case(case)
        sdf, mask = st[0].astype(DTYPE), st[1].astype(DTYPE)
        ct, mxt, myt = maps(gt, case, checker)
        rec = {"floor_norm": rnorm(ct, mxt, myt), "cd_truth": cd_of(gt, case),
               "fields": {}}
        for s in seeds:
            npz = os.path.join(AG_DIR, f"seed{s}", f"{case.name}.npz")
            if not os.path.exists(npz):
                continue
            z = np.load(npz)
            for tag in ("raw", "corrected"):
                f = wrap(z[tag], case, sdf, mask)
                c1, m1, m2 = maps(f, case, checker)
                rec["fields"][f"{tag}_seed{s}"] = {
                    "sigma_norm": rnorm(c1, m1, m2),
                    "sigma_diff": rnorm(c1 - ct, m1 - mxt, m2 - myt),
                    "cd": cd_of(f, case),
                }
        per_case[case.name] = rec
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)} ({(time.time() - t0) / (i + 1):.2f}s/case)",
                  flush=True)

    names = sorted(per_case)
    cd_t = np.array([per_case[n]["cd_truth"] for n in names])
    floor = np.array([per_case[n]["floor_norm"] for n in names])

    arms = {}
    for tag in ("raw", "corrected"):
        for s in seeds:
            fk = f"{tag}_seed{s}"
            if fk not in per_case[names[0]]["fields"]:
                continue
            g = lambda k: np.array([per_case[n]["fields"][fk][k] for n in names])  # noqa: E731
            e = np.abs(g("cd") - cd_t)
            sig = g("sigma_norm")
            sig2 = g("sigma_diff")
            signaive = np.maximum(sig - floor, 1e-12)
            arms[f"{fk}/uninformative"] = conformal(np.ones_like(e), e, reps=a.reps)
            arms[f"{fk}/deployed_norm"] = conformal(sig, e, reps=a.reps)
            arms[f"{fk}/oracle_floor_subtracted"] = conformal(sig2, e, reps=a.reps)
            arms[f"{fk}/invalid_norm_difference"] = conformal(signaive, e, reps=a.reps)
            arms[f"{fk}/perfect_oracle"] = conformal(e, e, reps=a.reps)
            arms[f"{fk}/deployed_norm"]["floor_share_of_typical_score"] = \
                float(np.median(floor) / np.median(sig))
            arms[f"{fk}/oracle_floor_subtracted"]["median_sigma_over_deployed"] = \
                float(np.median(sig2) / np.median(sig))
            arms[f"{fk}/deployed_norm"]["frac_cases_norm_below_floor"] = \
                float(np.mean(sig < floor))

    prim = [arms[f"raw_seed{s}/oracle_floor_subtracted"]["bound_over_error"]
            for s in seeds if f"raw_seed{s}/oracle_floor_subtracted" in arms]
    dep = [arms[f"raw_seed{s}/deployed_norm"]["bound_over_error"] for s in seeds
           if f"raw_seed{s}/deployed_norm" in arms]
    uni = [arms[f"raw_seed{s}/uninformative"]["bound_over_error"] for s in seeds
           if f"raw_seed{s}/uninformative" in arms]
    m = float(np.mean(prim))
    if m <= 2.0:
        br, why = "FLOOR-CAUSAL", f"oracle floor-subtracted width {m:.2f}x <= 2.0x"
    elif m >= 4.0:
        br, why = ("FLOOR-NOT-CAUSAL",
                   f"oracle floor-subtracted width {m:.2f}x >= 4.0x: removing the "
                   f"floor EXACTLY does not make the certificate usable, so the "
                   f"width is not caused by the floor")
    else:
        br, why = "PARTIAL", f"oracle floor-subtracted width {m:.2f}x is in (2, 4)"

    out = {"artifact": "floor_subtraction_gate",
           "question": ("Is the conformal certificate wide BECAUSE of the residual "
                        "floor? Oracle counterfactual with sigma' = ||R(u_hat) - R(u*)||."),
           "preregistration": "docstring of scripts/floor_subtraction_gate.py",
           "verdict": {"branch": br, "why": why,
                       "oracle_floor_subtracted_mean_x": m,
                       "deployed_mean_x": float(np.mean(dep)),
                       "uninformative_mean_x": float(np.mean(uni)),
                       "monitor_gain_over_no_monitor": float(np.mean(uni) / np.mean(dep)),
                       "floor_removal_gain": float(np.mean(dep) / m)},
           "caveats": [
               "sigma' uses r*, unavailable at deployment: an oracle counterfactual, "
               "and the ceiling on any learned floor predictor.",
               "sigma' cancels SHARED rasterisation noise as well as the floor, so a "
               "large gain is an UPPER bound on what floor removal could buy.",
           ],
           "metadata": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "cpu",
                        "n_cases": len(names), "seeds": seeds, "reps": a.reps,
                        "conformal": "exact ceil((1-alpha)(n+1)) order statistic",
                        "runtime_s": time.time() - t0},
           "arms": arms,
           "per_case": per_case}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="\n", encoding="utf-8") as fh:
        # Spearman against a constant score is undefined (the uninformative arm);
        # JSON has no NaN, so non-finite floats are written as null.
        json.dump(_clean(out), fh, indent=1, allow_nan=False)
    print(f"\nwrote {a.out} ({time.time() - t0:.0f}s)\n")

    hdr = f"{'arm':<46}{'cov':>7}{'width':>10}{'x med err':>11}{'rho':>8}"
    print(hdr)
    for k in sorted(arms):
        v = arms[k]
        print(f"{k:<46}{v['coverage_mean']:>7.3f}{v['median_bound_width']:>10.5f}"
              f"{v['bound_over_error']:>11.2f}{v['spearman_score_vs_err']:>8.3f}")
    print(f"\n=== VERDICT: {br} ===\n  {why}")
    return 0


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def cd_of(field, case):
    from neuroforge.physics.metrics import force_coefficients
    return float(force_coefficients(field, case)["cd"])


if __name__ == "__main__":
    raise SystemExit(main())
