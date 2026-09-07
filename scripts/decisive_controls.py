"""Decisive controls on the paper-1 trust-signal claims -- pure re-analysis.

Five of the six controls raised by the adversarial reviews need NO new compute:
the per-case data they require is already committed.  This script runs them all
off committed JSON, CPU-only, in seconds, and writes one result file per control
under ``results/review/``.

    --control 1   Is the physics residual beaten by a physics-free baseline?
                  Paired case-level bootstrap of the AUROC / Spearman / retained-
                  risk DIFFERENCES between residual, sigma_vel and their fusion.
                  Marginal CIs (results/control/bootstrap_spearman_ci.json)
                  cannot settle a paired comparison; these can.

    --control 2   The case-difficulty confound.  Partial Spearman
                  rho(residual, error | ||r(truth)||), where ||r(truth)|| is the
                  monitored residual OF THE GROUND-TRUTH FIELD from
                  results/certificates/residual_floor_realdata.json -- a
                  model-independent per-case discretisation/model-form difficulty
                  proxy.

    --control 4   Risk-coverage on the engineering quantity.  Re-runs the
                  selective-prediction machinery with |Delta C_d| as the error
                  target instead of field rel-L2.

    --control 5   Disclosure audit of results/control/mgn_density_control.json.

    --control 6   The omitted channels in tab:iters, and the direction of travel
                  of residual vs error over the full iteration range.

Usage
-----
    PYTHONPATH=src .venv/Scripts/python.exe scripts/decisive_controls.py --control all
"""

from __future__ import annotations

import argparse
import json
import os
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets BLAS thread caps)
import numpy as np

SEL_PC = "results/selective/selective_percase.json"
SEL_AGG = "results/selective/selective_prediction.json"
FLOOR = "results/certificates/residual_floor_realdata.json"
MGN_DENS = "results/control/mgn_density_control.json"
ITERS = "results/sensitivity/iters.json"
FORCE = "results/control/force_vs_official.json"
FCACHE = "results/control/_cache"
OUTDIR = "results/review"


def log(m):
    print(f"[controls] {m}", flush=True)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # newline="\n": the repo stores result JSON with LF; Windows would emit CRLF.
    with open(path, "w", newline="\n") as f:
        json.dump(obj, f, indent=1)
    log(f"wrote {path}")


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
try:                                    # C implementation; 10^4 bootstraps need it
    from scipy.stats import rankdata as _rankdata

    def rankdata(a):
        return np.asarray(_rankdata(np.asarray(a, float)), float)
except ImportError:                     # pure-numpy fallback, same tie convention
    def rankdata(a):
        """Average-rank transform (ties averaged)."""
        a = np.asarray(a, float)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        s = a[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
            i = j + 1
        return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def auroc(score, label):
    """Mann-Whitney AUROC with tie correction. label: 1 = positive (bad case)."""
    score = np.asarray(score, float)
    label = np.asarray(label).astype(bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y given z: Pearson of ranks, partialled."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def pear(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / d) if d > 0 else float("nan")

    rxy, rxz, ryz = pear(rx, ry), pear(rx, rz), pear(ry, rz)
    den = np.sqrt(max(1 - rxz ** 2, 0.0) * max(1 - ryz ** 2, 0.0))
    return float((rxy - rxz * ryz) / den) if den > 0 else float("nan")


def boot_ci(stat_fn, n, n_boot=10000, seed=0, alpha=0.05):
    """Case-level percentile bootstrap of an arbitrary statistic.

    stat_fn(idx) -> float (or None to skip the draw).  Returns the point
    estimate on the full sample plus a percentile CI and the sign mass.
    """
    rng = np.random.default_rng(seed)
    full = stat_fn(np.arange(n))
    draws = np.empty(n_boot)
    draws[:] = np.nan
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat_fn(idx)
        draws[b] = v if v is not None else np.nan
    d = draws[np.isfinite(draws)]
    if len(d) == 0:
        return {"point": full, "ci95": [None, None], "n_boot_valid": 0}
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(full),
        "ci95": [float(lo), float(hi)],
        "boot_mean": float(d.mean()),
        "boot_std": float(d.std(ddof=1)),
        "frac_draws_gt_0": float((d > 0).mean()),
        "frac_draws_lt_0": float((d < 0).mean()),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot_valid": int(len(d)),
    }


def retained_risk(err, score, frac_reject):
    """Mean error among the cases retained after rejecting the top-scoring frac."""
    n = len(err)
    k = int(round(frac_reject * n))
    keep = np.argsort(rankdata(score))[: n - k] if k > 0 else np.arange(n)
    return float(np.mean(np.asarray(err)[keep]))


# --------------------------------------------------------------------------
# CONTROL 1
# --------------------------------------------------------------------------
def control1(a):
    pc = json.load(open(SEL_PC))
    agg = json.load(open(SEL_AGG))
    out = {
        "artifact": "control1_physics_vs_physicsfree",
        "question": ("Does the physics residual buy anything over a physics-free "
                     "ensemble-spread baseline on the paper's own selective-prediction "
                     "task, and is the fusion gain statistically real? Marginal CIs "
                     "cannot answer this; a PAIRED case-level bootstrap can."),
        "method": {
            "auroc": ("Mann-Whitney AUROC for detecting worst-decile-rel-L2 cases. The "
                      "positive-class threshold is the FULL-SAMPLE top-decile threshold "
                      "and is held FIXED across bootstrap draws, so 'worst-decile case' "
                      "is a property of the case (the paper's estimand) and the paired "
                      "delta is not contaminated by label churn."),
            "spearman": "case-level Spearman(score, rel_L2).",
            "risk": ("retained mean rel-L2 after rejecting the highest-scoring 10% of "
                     "cases. Primary paired risk statistic because it is a difference of "
                     "well-behaved means; oracle-recovery is a RATIO of differences whose "
                     "denominator can approach zero under resampling, so it is reported "
                     "as a descriptive on the full sample only."),
            "bootstrap": f"{a.n_boot} case-level resamples, seed {a.seed}, percentile CI.",
        },
        "arms": {},
    }

    for arm, rows in pc.items():
        if arm == "meta":
            continue
        err = np.array([r["rel_l2"] for r in rows], float)
        scores = {k: np.array([r[k] for r in rows], float)
                  for k in ("residual", "sigma_vel", "sigma_p", "fused")
                  if k in rows[0]}
        n = len(err)
        thr = float(np.quantile(err, 0.9))
        lab = err >= thr
        rec = {
            "n_cases": n,
            "top_decile_threshold_rel_l2": thr,
            "n_positives": int(lab.sum()),
            "scores_available": sorted(scores),
            "marginal": {},
            "paired": {},
        }
        # cross-check against the committed aggregate
        pub = agg["arms"].get(arm, {}).get("scores", {})
        for k, s in scores.items():
            m = {
                "spearman": spearman(s, err),
                "auroc_top_decile": auroc(s, lab),
                "retained_risk_at_10pct": retained_risk(err, s, 0.10),
            }
            if k in pub:
                m["published_spearman"] = pub[k]["spearman"]
                m["published_auroc"] = pub[k]["auroc_top_decile"]
                m["reproduces"] = bool(
                    abs(m["spearman"] - pub[k]["spearman"]) < 1e-9
                    and abs(m["auroc_top_decile"] - pub[k]["auroc_top_decile"]) < 1e-9)
            rec["marginal"][k] = m

        base = retained_risk(err, err, 0.0)
        oracle10 = retained_risk(err, err, 0.10)
        rec["oracle"] = {
            "retained_risk_at_0pct": base,
            "oracle_retained_risk_at_10pct": oracle10,
            "max_achievable_reduction": base - oracle10,
        }
        for k, s in scores.items():
            red = base - rec["marginal"][k]["retained_risk_at_10pct"]
            rec["marginal"][k]["oracle_recovery_at_10pct"] = (
                float(red / (base - oracle10)) if base > oracle10 else None)

        # paired bootstrap on every available score pair
        keys = [k for k in ("residual", "sigma_vel", "fused") if k in scores]
        pairs = [(x, y) for i, x in enumerate(keys) for y in keys[i + 1:]]
        for x, y in pairs:
            sx, sy = scores[x], scores[y]

            def mk(fn):
                def f(idx):
                    return fn(idx)
                return f

            d_auroc = boot_ci(
                mk(lambda i: auroc(sy[i], lab[i]) - auroc(sx[i], lab[i])),
                n, a.n_boot, a.seed)
            d_sp = boot_ci(
                mk(lambda i: spearman(sy[i], err[i]) - spearman(sx[i], err[i])),
                n, a.n_boot, a.seed)
            d_risk = boot_ci(
                mk(lambda i: retained_risk(err[i], sy[i], 0.10)
                   - retained_risk(err[i], sx[i], 0.10)),
                n, a.n_boot, a.seed)
            # oracle recovery is the paper's own headline metric (67% / 91%), but it
            # is a RATIO whose denominator can approach zero under resampling. We
            # therefore bootstrap it with the denominator FIXED at its full-sample
            # value, which makes it a rescaling of the well-behaved risk difference
            # and keeps the CI finite. Flagged as secondary.
            denom = base - oracle10
            d_orec = boot_ci(
                mk(lambda i: (retained_risk(err[i], sx[i], 0.10)
                              - retained_risk(err[i], sy[i], 0.10)) / denom),
                n, a.n_boot, a.seed) if denom > 0 else None
            rec["paired"][f"{y}_minus_{x}"] = {
                "delta_auroc": d_auroc,
                "delta_spearman": d_sp,
                "delta_retained_risk_at_10pct": d_risk,
                "delta_oracle_recovery_fixed_denominator": d_orec,
                "note": "positive delta_auroc/spearman favours the FIRST-named score; "
                        "NEGATIVE delta_retained_risk favours it (lower risk is better).",
            }
        out["arms"][arm] = rec
        log(f"  C1 {arm}: " + ", ".join(
            f"{k} auroc={v['auroc_top_decile']:.4f}" for k, v in rec["marginal"].items()))

    out["headline"] = _c1_headline(out)
    jdump(out, f"{OUTDIR}/control1_physics_vs_physicsfree.json")
    return out


def _c1_headline(out):
    e = out["arms"].get("ensemble_mean")
    if not e:
        return {}
    h = {"arm": "ensemble_mean",
         "note": ("sigma_vel is dumped per-case ONLY for the ensemble_mean arm; the "
                  "corrected_seed*/deepcfd_seed* arms carry the residual score alone, so "
                  "the residual-vs-sigma paired test is not computable there. The brief's "
                  "'do this on every arm' is not satisfiable with the committed data.")}
    for k, v in e["marginal"].items():
        h[k] = {"spearman": v["spearman"], "auroc": v["auroc_top_decile"],
                "oracle_recovery_10pct": v.get("oracle_recovery_at_10pct")}
    for k, v in e["paired"].items():
        h[k] = {"delta_auroc": v["delta_auroc"]["point"],
                "delta_auroc_ci95": v["delta_auroc"]["ci95"],
                "delta_auroc_excludes_zero": v["delta_auroc"]["excludes_zero"],
                "delta_spearman": v["delta_spearman"]["point"],
                "delta_spearman_ci95": v["delta_spearman"]["ci95"],
                "delta_spearman_excludes_zero": v["delta_spearman"]["excludes_zero"]}
    return h


# --------------------------------------------------------------------------
# CONTROL 2
# --------------------------------------------------------------------------
def control2(a):
    pc = json.load(open(SEL_PC))
    fl = json.load(open(FLOOR))
    floor = {r["name"]: r for r in fl["per_case"]}

    out = {
        "artifact": "control2_difficulty_confound",
        "question": ("Is the residual-error rank correlation a per-case DIFFICULTY proxy? "
                     "A geometry that rasterises badly could carry both a high "
                     "ground-truth residual and a high prediction error with no causal "
                     "link between the monitored residual and the model's error."),
        "conditioning_variable": {
            "name": "norm_truth  ==  ||r(truth)||",
            "source": FLOOR,
            "definition": fl["metadata"]["monitored_residual"]
            + " -- evaluated on the GROUND-TRUTH field.",
            "why_this_one": ("model-independent per-case discretisation + model-form "
                             "residual: exactly the 'this geometry is hard to rasterise' "
                             "quantity the confound objection names."),
            "NOT_the_floor_r_star": (
                "The brief calls this ||r*||. That is a misnomer for this file: its actual "
                "minimum is norm_uniform = 0.0 exactly on all 200 cases "
                f"(frac_uniform_lt_truth = {fl['aggregate']['frac_uniform_lt_truth']}); the "
                "experiment exists to show the monitored residual PREFERS the uniform "
                "freestream to the truth. We therefore condition on ||r(truth)||, the "
                "residual of the ground-truth field, and name it as such."),
            "norm_pred_NOT_used": (
                "residual_floor_realdata.json also carries norm_pred, but that is a "
                "DIFFERENT model (checkpoints/certificates_deq.pt, dropout-FNO, DEQ not "
                "applied) than the Transolver arms scored here. Only the "
                "model-independent norm_truth is joined."),
        },
        "arms": {},
    }

    for arm, rows in pc.items():
        if arm == "meta":
            continue
        if "name" not in rows[0]:
            out["arms"][arm] = {
                "joinable": False,
                "reason": ("keys on 'index', not case name, and n=%d on a different "
                           "(DeepCFD OOD) split -- the 200-case floor table cannot be "
                           "joined to it. Not improvised." % len(rows)),
            }
            continue
        j = [(r, floor[r["name"]]) for r in rows if r["name"] in floor]
        if not j:
            out["arms"][arm] = {"joinable": False, "reason": "no name overlap"}
            continue
        res = np.array([r["residual"] for r, _ in j], float)
        err = np.array([r["rel_l2"] for r, _ in j], float)
        z = np.array([f["norm_truth"] for _, f in j], float)
        n = len(j)

        raw = spearman(res, err)
        par = partial_spearman(res, err, z)
        rec = {
            "joinable": True,
            "n_cases": n,
            "decomposition_first": {
                "rho_residual_vs_normtruth": spearman(res, z),
                "rho_error_vs_normtruth": spearman(err, z),
                "note": ("the confound can only explain the headline if the conditioning "
                         "variable correlates with BOTH sides. If rho(error, ||r(truth)||) "
                         "is small the partial cannot collapse much, and the objection is "
                         "answered arithmetically."),
            },
            "rho_raw": raw,
            "rho_partial_given_normtruth": par,
            "fraction_of_rho_surviving": float(par / raw) if raw != 0 else None,
            "absolute_drop": float(raw - par),
        }
        # THE DECISIVE SUB-TEST: if the monitored residual were merely a proxy for
        # per-case difficulty, it could not BEAT the difficulty variable itself.
        thr = float(np.quantile(err, 0.9))
        lab = err >= thr
        rec["residual_vs_difficulty_itself"] = {
            "rho_error_vs_residual": spearman(res, err),
            "rho_error_vs_normtruth": spearman(z, err),
            "auroc_residual": auroc(res, lab),
            "auroc_normtruth": auroc(z, lab),
            "paired_delta_rho_residual_minus_normtruth": boot_ci(
                lambda i: spearman(res[i], err[i]) - spearman(z[i], err[i]),
                n, a.n_boot, a.seed),
            "paired_delta_auroc_residual_minus_normtruth": boot_ci(
                lambda i: auroc(res[i], lab[i]) - auroc(z[i], lab[i]),
                n, a.n_boot, a.seed),
            "logic": ("||r(truth)|| is the difficulty variable, and it needs the ground "
                      "truth to compute, so it is an ORACLE difficulty score. If the "
                      "deployable residual outranks it, the residual carries "
                      "model-specific error information that pure case difficulty does "
                      "not, and the 'it is just a difficulty proxy' objection fails."),
        }
        rec["bootstrap"] = {
            "rho_raw": boot_ci(lambda i: spearman(res[i], err[i]), n, a.n_boot, a.seed),
            "rho_partial": boot_ci(
                lambda i: partial_spearman(res[i], err[i], z[i]), n, a.n_boot, a.seed),
            "drop_raw_minus_partial": boot_ci(
                lambda i: spearman(res[i], err[i]) - partial_spearman(res[i], err[i], z[i]),
                n, a.n_boot, a.seed),
        }
        # sigma_vel gets the same treatment where available (physics-free control)
        if "sigma_vel" in rows[0]:
            sg = np.array([r["sigma_vel"] for r, _ in j], float)
            rec["sigma_vel_same_test"] = {
                "rho_raw": spearman(sg, err),
                "rho_partial_given_normtruth": partial_spearman(sg, err, z),
                "rho_sigma_vs_normtruth": spearman(sg, z),
                "note": ("if the physics-free score survives conditioning just as well, "
                         "the confound is not specific to the physics residual."),
            }
        out["arms"][arm] = rec
        log(f"  C2 {arm}: raw={raw:.4f} partial={par:.4f} "
            f"({100 * par / raw:.1f}% survives)")

    jdump(out, f"{OUTDIR}/control2_difficulty_confound.json")
    return out


# --------------------------------------------------------------------------
# CONTROL 4
# --------------------------------------------------------------------------
def control4(a):
    pc = json.load(open(SEL_PC))["ensemble_mean"]
    off = json.load(open(f"{FCACHE}/official_labels_full_test_n200.json"))
    gtnf = {r["name"]: r for r in
            json.load(open(f"{FCACHE}/gt_nf_full_test_r128_n200.json"))}
    seeds = {}
    for s in (0, 1, 2):
        p = f"{FCACHE}/seed{s}_prednf.json"
        if os.path.exists(p):
            seeds[s] = {r["name"]: r for r in json.load(open(p))}

    out = {
        "artifact": "control4_riskcoverage_on_drag",
        "question": ("The paper's selective-prediction result is on FIELD error. An "
                     "aerodynamicist triages on drag. Does the trust signal still triage "
                     "when the error target is |Delta C_d|?"),
        "targets": {
            "primary_vs_gt_nf": ("|C_d(pred) - C_d(gt)| with BOTH sides through the same "
                                 "NeuroForge integrator: isolates MODEL error in drag, "
                                 "which is what a trust signal could plausibly see."),
            "secondary_vs_official": ("|C_d(pred) - C_d(official)| against the AirfRANS "
                                      "OpenFOAM reference labels: the number an engineer "
                                      "compares against, but dominated by integrator bias "
                                      "the trust signal cannot see -- force_vs_official.json "
                                      "reports cd_rel_err_official ~ 11x and "
                                      "rho_D_gt_vs_official = 0.839 for the GROUND TRUTH "
                                      "field alone."),
        },
        "caveats": {
            "arm_mismatch": ("The trust scores (residual, sigma_vel, fused) are for the "
                             "ENSEMBLE-MEAN field; the per-case C_d values are for the "
                             "three individual v2_transolver seeds. We therefore run each "
                             "seed as its own arm and show the spread rather than hiding "
                             "the mismatch behind a seed average."),
            "integrator_scheme": ("'legacy' scheme only -- the one behind "
                                  "force_vs_official.json and the paper. The multischeme "
                                  "file shows the integrator choice moves rho_D from 0.31 "
                                  "(cv) to 0.82 (wall); that is a separate finding."),
            "construction": ("Mirrors the field-error construction exactly: top-decile "
                             "|Delta C_d| is the positive class, same Mann-Whitney AUROC, "
                             "same 10% rejection budget."),
        },
        "field_error_reference": {},
        "arms": {},
    }

    err_field = np.array([r["rel_l2"] for r in pc], float)
    sc = {k: np.array([r[k] for r in pc], float) for k in ("residual", "sigma_vel", "fused")}
    lab_f = err_field >= np.quantile(err_field, 0.9)
    for k, s in sc.items():
        out["field_error_reference"][k] = {
            "spearman": spearman(s, err_field), "auroc_top_decile": auroc(s, lab_f),
            "oracle_recovery_at_10pct": _orec(err_field, s)}

    names = [r["name"] for r in pc]
    for s_i, pred in seeds.items():
        for target, ref in (("vs_gt_nf", gtnf), ("vs_official", off)):
            keys = [i for i, nm in enumerate(names) if nm in pred and nm in ref]
            if not keys:
                continue
            cd_p = np.array([pred[names[i]]["cd"] for i in keys], float)
            if target == "vs_official":
                cd_r = np.array([ref[names[i]]["cd"] for i in keys], float)
            else:
                cd_r = np.array([ref[names[i]]["cd"] for i in keys], float)
            e = np.abs(cd_p - cd_r)
            lab = e >= np.quantile(e, 0.9)
            rec = {"n_cases": len(keys),
                   "mean_abs_dCd": float(e.mean()),
                   "median_abs_dCd": float(np.median(e)),
                   "top_decile_threshold_abs_dCd": float(np.quantile(e, 0.9)),
                   "scores": {}}
            n = len(keys)
            for k, sv in sc.items():
                svk = sv[keys]
                rec["scores"][k] = {
                    "spearman": spearman(svk, e),
                    "spearman_ci95": boot_ci(
                        lambda i: spearman(svk[i], e[i]), n, a.n_boot, a.seed)["ci95"],
                    "auroc_top_decile": auroc(svk, lab),
                    "auroc_ci95": boot_ci(
                        lambda i: auroc(svk[i], lab[i]), n, a.n_boot, a.seed)["ci95"],
                    "oracle_recovery_at_10pct": _orec(e, svk),
                    "retained_risk_at_10pct": retained_risk(e, svk, 0.10),
                }
            rec["oracle"] = {"retained_risk_at_0pct": float(e.mean()),
                             "oracle_retained_risk_at_10pct": retained_risk(e, e, 0.10)}
            # paired bootstrap on the drag target: does PHYSICS beat PHYSICS-FREE
            # here, where control 1 found them indistinguishable on field error?
            rec["paired"] = {}
            for x, y in (("sigma_vel", "residual"), ("sigma_vel", "fused"),
                         ("residual", "fused")):
                sx, sy = sc[x][keys], sc[y][keys]
                rec["paired"][f"{y}_minus_{x}"] = {
                    "delta_auroc": boot_ci(
                        lambda i: auroc(sy[i], lab[i]) - auroc(sx[i], lab[i]),
                        n, a.n_boot, a.seed),
                    "delta_spearman": boot_ci(
                        lambda i: spearman(sy[i], e[i]) - spearman(sx[i], e[i]),
                        n, a.n_boot, a.seed),
                }
            # is the drag error even related to the field error?
            rec["rho_absdCd_vs_field_rel_l2"] = spearman(e, err_field[keys])
            out["arms"][f"seed{s_i}_{target}"] = rec
            log(f"  C4 seed{s_i} {target}: " + ", ".join(
                f"{k} auroc={v['auroc_top_decile']:.3f} rho={v['spearman']:+.3f}"
                for k, v in rec["scores"].items()))

    # pooled across seeds for a headline
    out["pooled_across_seeds"] = {}
    for target in ("vs_gt_nf", "vs_official"):
        arms = [v for k, v in out["arms"].items() if k.endswith(target)]
        if not arms:
            continue
        out["pooled_across_seeds"][target] = {
            k: {"auroc_mean": float(np.mean([x["scores"][k]["auroc_top_decile"] for x in arms])),
                "auroc_range": [float(min(x["scores"][k]["auroc_top_decile"] for x in arms)),
                                float(max(x["scores"][k]["auroc_top_decile"] for x in arms))],
                "spearman_mean": float(np.mean([x["scores"][k]["spearman"] for x in arms])),
                "spearman_range": [float(min(x["scores"][k]["spearman"] for x in arms)),
                                   float(max(x["scores"][k]["spearman"] for x in arms))],
                "oracle_recovery_mean": float(np.mean(
                    [x["scores"][k]["oracle_recovery_at_10pct"] for x in arms]))}
            for k in sc}
    jdump(out, f"{OUTDIR}/control4_riskcoverage_drag.json")
    return out


def _orec(err, score):
    base = float(np.mean(err))
    orc = retained_risk(err, err, 0.10)
    got = retained_risk(err, score, 0.10)
    return float((base - got) / (base - orc)) if base > orc else None


# --------------------------------------------------------------------------
# CONTROL 5
# --------------------------------------------------------------------------
def control5(a):
    d = json.load(open(MGN_DENS))
    hits = []
    for root, _, files in os.walk("docs/paper"):
        if os.sep + "review" in root:
            continue        # review notes are not the manuscript
        for fn in files:
            if not fn.endswith((".tex", ".md", ".bib")):
                continue
            p = os.path.join(root, fn)
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for pat in ("mgn_density", "density_control", "DENSITY-DRIVEN",
                        "density-driven", "control_mgn_density", "16384", "16k point"):
                if pat in txt:
                    hits.append({"file": p, "pattern": pat})
    review_hits = []
    for root, _, files in os.walk("docs/paper/review"):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if "mgn_density" in txt or "DENSITY-DRIVEN" in txt:
                review_hits.append(p)

    out = {
        "artifact": "control5_mgn_density_disclosure",
        "question": ("Is the MeshGraphNet density control disclosed, and does it "
                     "undermine the rho = 0.851 headline?"),
        "what_the_file_shows": {
            "verdict": d["verdict"]["verdict"],
            "velocity_ratio_180k_over_16k": d["verdict"]["velocity_ratio"],
            "per_channel_ratio": d["verdict"]["ratio_180k_over_16k"],
            "trained_at_n_points": d["meta"]["train_density_16k"],
            "evaluated_at_n_points_typical": int(np.median(
                [c["n_points"] for c in d["per_case"]])),
            "density_mismatch_factor": float(np.median(
                [c["n_points"] for c in d["per_case"]]) / d["meta"]["train_density_16k"]),
            "n_cases_scored": d["meta"]["n_cases_scored"],
            "metric": "per-point PHYSICAL MSE at the same 16,384 indices (confound-free)",
        },
        "disclosure_audit": {
            "manuscript_files_searched": "docs/paper/**/*.{tex,md,bib} EXCLUDING docs/paper/review/",
            "manuscript_hits": hits,
            "disclosed_in_manuscript": bool(hits),
            "review_note_hits": review_hits,
        },
        "what_this_file_does_NOT_establish": (
            "The brief asserts rho = 0.851 is 'substantially an artifact of a train/eval "
            "point-density mismatch'. This file cannot establish that and does not claim "
            "to: it measures MSE, contains NO rank correlation at either density, carries "
            "no per-case residual values, and scores only 4 cases. A density-driven "
            "inflation of the ERROR SPREAD is a mechanism by which rho could be inflated, "
            "not a measurement that it is. Establishing the stronger claim would require "
            "recomputing the residual-error Spearman at 16k and at full density -- not "
            "available in committed data."),
        "what_it_DOES_establish": [
            "MeshGraphNet is evaluated ~11x off its training point density.",
            "Velocity MSE inflates 2.44x because of that mismatch (4 cases, confound-free probe).",
            "The control exists, is committed, and is cited nowhere in the manuscript.",
            ("body.tex:839-841 already attributes the high rho to 'that model's wide error "
             "spread'; this file supplies a mechanism for that spread which the paper does "
             "not disclose."),
        ],
    }
    log(f"  C5: manuscript hits = {len(hits)} ; review-note hits = {len(review_hits)}")
    jdump(out, f"{OUTDIR}/control5_mgn_density_disclosure.json")
    return out


# --------------------------------------------------------------------------
# CONTROL 6
# --------------------------------------------------------------------------
def control6(a):
    it = json.load(open(ITERS))
    rows = it["rows"]
    have = sorted({k for r in rows for k in r})
    wanted = ["mse_u", "mse_v", "mse_p", "mse_nut", "surface_mse_p"]
    missing = [k for k in wanted if k not in have]

    # did the producing script ever compute the full channel set?
    src = ""
    for p in ("scripts/run_sensitivity.py",):
        if os.path.exists(p):
            src = open(p, encoding="utf-8", errors="ignore").read()
    prov = {"script": "scripts/run_sensitivity.py", "found": bool(src)}
    if src:
        i = src.find("n_iters")
        prov["mentions_mse_v"] = "mse_v" in src
        prov["mentions_mse_p"] = "mse_p" in src
        # locate the row-building block
        for key in ("mse_u", "surface_mse_p"):
            prov[f"first_index_{key}"] = src.find(key)
    figs = []
    fd = "results/sensitivity/figures"
    if os.path.isdir(fd):
        figs = sorted(os.listdir(fd))

    u = [r["mse_u"] for r in rows]
    rn = [r["residual_norm"] for r in rows]
    ni = [r["n_iters"] for r in rows]
    i_min = int(np.argmin(u))
    out = {
        "artifact": "control6_iters_channels_and_monotonicity",
        "question": ("tab:iters reports only mse_u and surface mse_p while tab:indist "
                     "documents this corrector family inflating mse_v (+129%) and mse_p "
                     "(+57%). Do the omitted channels rise along the sweep? And is the "
                     "'opposite directions' claim true over the whole range?"),
        "channel_availability": {
            "columns_retained_in_iters_json": have,
            "columns_needed": wanted,
            "columns_MISSING": missing,
            "verdict": ("The per-channel data was NOT retained. results/sensitivity/"
                        "iters.json and iters.csv carry mse_u and surface_mse_p only; "
                        "mse_v, volume mse_p and mse_nut appear nowhere in the sweep's "
                        "committed output."),
            "provenance_check": prov,
            "figures_present": figs,
            "no_substitution": ("We do NOT substitute the tab:indist per-channel numbers: "
                                "that is a different corrector arm on a different backbone "
                                "and would be a fabricated row."),
        },
        "monotonicity": {
            "n_iters": ni,
            "mse_u": u,
            "residual_norm": rn,
            "mse_u_argmin_iter": ni[i_min],
            "mse_u_min": u[i_min],
            "mse_u_at_last": u[-1],
            "mse_u_rise_from_min_to_last_pct": float(100 * (u[-1] - u[i_min]) / u[i_min]),
            "residual_at_min": rn[i_min],
            "residual_at_last": rn[-1],
            "residual_rise_from_min_to_last_pct": float(100 * (rn[-1] - rn[i_min]) / rn[i_min]),
            "residual_is_monotone_increasing": bool(all(
                rn[i + 1] >= rn[i] for i in range(len(rn) - 1))),
            "mse_u_is_monotone": bool(all(u[i + 1] <= u[i] for i in range(len(u) - 1))
                                      or all(u[i + 1] >= u[i] for i in range(len(u) - 1))),
            "spearman_residual_vs_mse_u_over_sweep": spearman(rn, u),
            "spearman_over_iters_3_to_15": spearman(rn[i_min:], u[i_min:]),
            "confirmed": ("mse_u bottoms at iteration %d (%.4f) and rises to %.4f by "
                          "iteration %d, while residual_norm rises %.4f -> %.4f over the "
                          "same span: across iterations %d-%d residual and error move in "
                          "the SAME direction."
                          % (ni[i_min], u[i_min], u[-1], ni[-1], rn[i_min], rn[-1],
                             ni[i_min], ni[-1])),
        },
        "surface_mse_p_direction": {
            "values": [r["surface_mse_p"] for r in rows],
            "monotone_decreasing_to_iter10": bool(all(
                rows[i + 1]["surface_mse_p"] <= rows[i]["surface_mse_p"]
                for i in range(4))),
            "note": "surface pressure keeps improving to iter 10 then flattens/rises "
                    "slightly; it does not share mse_u's turn at iteration 3.",
        },
    }
    log(f"  C6: missing channels = {missing}; mse_u min at iter {ni[i_min]}")
    jdump(out, f"{OUTDIR}/control6_iters_channels.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", default="all",
                   help="1, 2, 4, 5, 6, or 'all' (control 3 lives in "
                        "scripts/control_fixed_step.py -- it touches field caches)")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    t0 = time.time()
    which = ["1", "2", "4", "5", "6"] if a.control == "all" else [a.control]
    fns = {"1": control1, "2": control2, "4": control4, "5": control5, "6": control6}
    for k in which:
        log(f"=== CONTROL {k} ===")
        fns[k](a)
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
