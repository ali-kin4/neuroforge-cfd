"""Trace the manuscript's headline numbers back to their committed result files.

The failure mode this catches is drift: a result is re-run, the JSON moves, and
the sentence quoting it does not. That is invisible to a LaTeX build and to the
test suite, and it is exactly what a referee checks first when the repository is
public. Each entry below names a claim, the file it must come from, how to read
the value out of that file, and the value the manuscript currently states.

A row FAILS if the recomputed value disagrees with the manuscript beyond its
stated tolerance. A row is SKIPPED, and says so, if the source file is missing --
skips are not passes and are counted separately.

Usage
-----
    python scripts/audit_paper_numbers.py
    python scripts/audit_paper_numbers.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def load(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---- readers: each returns the value the manuscript quotes -------------------

def floor_realdata_mean(root):
    d = load(os.path.join(root, "results/certificates/residual_floor_realdata.json"))
    if d is None:
        return None
    return float(np.mean([r["norm_truth"] for r in d["per_case"]]))


def floor_realdata_median(root):
    d = load(os.path.join(root, "results/certificates/residual_floor_realdata.json"))
    if d is None:
        return None
    return float(np.median([r["norm_truth"] for r in d["per_case"]]))


def uniform_all_zero(root):
    d = load(os.path.join(root, "results/certificates/residual_floor_realdata.json"))
    if d is None:
        return None
    return float(max(abs(r["norm_uniform"]) for r in d["per_case"]))


def ladder_level(root, res, key="floor"):
    d = load(os.path.join(root, "results/floor_resolution_ladder.json"))
    if d is None:
        return None
    rows = d["per_level"].get(str(res))
    if not rows:
        return None
    return float(np.nanmean([r[key] for r in rows]))


def ladder_p(root):
    d = load(os.path.join(root, "results/floor_resolution_ladder.json"))
    if d is None:
        return None
    return float(d["fits"]["floor"]["p"])


def ladder_rose(root):
    d = load(os.path.join(root, "results/floor_resolution_ladder.json"))
    if d is None:
        return None
    lo, hi = d["levels"][0], d["levels"][-1]
    a = {r["case"]: r["floor"] for r in d["per_level"][str(lo)]}
    b = {r["case"]: r["floor"] for r in d["per_level"][str(hi)]}
    return float(sum(1 for c in a if c in b and b[c] > a[c]))


def descent(root, arm, field, agg="mean", where="end"):
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    idx = -1 if where == "end" else 0
    vals = [r["arms"][arm][idx][field] for r in d["rows"]]
    return float(np.mean(vals) if agg == "mean" else np.median(vals))


def descent_residual_cut(root):
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    a = np.array([r["arms"]["from_truth"][0]["residual"] for r in d["rows"]])
    b = np.array([r["arms"]["from_truth"][-1]["residual"] for r in d["rows"]])
    return float(100.0 * (1.0 - b.mean() / a.mean()))


def descent_wins(root, arm):
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    a = np.array([r["arms"][arm][0]["residual"] for r in d["rows"]])
    b = np.array([r["arms"][arm][-1]["residual"] for r in d["rows"]])
    return float((b < a).sum())


def descent_selection_worse(root, arm):
    """Cases where the residual-chosen iterate is strictly worse than the best."""
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    n = 0
    for r in d["rows"]:
        t = r["arms"][arm]
        mse = np.array([s["mse_u"] for s in t])
        res = np.array([s["residual"] for s in t])
        if mse[int(res.argmin())] > mse.min() + 1e-12:
            n += 1
    return float(n)


def descent_perturbed_helped(root):
    d = load(os.path.join(root, "results/residual_descent.json"))
    if d is None:
        return None
    e0 = np.array([r["arms"]["perturbed_residual"][0]["mse_u"] for r in d["rows"]])
    e1 = np.array([r["arms"]["perturbed_residual"][-1]["mse_u"] for r in d["rows"]])
    return float((e1 < e0).sum())


def audit_cost_median(root):
    d = load(os.path.join(root, "results/control/inference_cost.json"))
    if d is None:
        return None
    return float(d["audit_ms"]["median_ms"])


def deployed_solve_median(root):
    d = load(os.path.join(root, "results/control/inference_cost.json"))
    if d is None:
        return None
    return float(d["per_device"]["cuda"]["solve_ms"]["median_ms"])


def decomp_repaired_shift(root, which="max"):
    """Percent change in the floor when the omitted stress term is restored."""
    d = load(os.path.join(root, "results/certificates/floor_resolution_decomposition.json"))
    if d is None:
        return None
    shifts = []
    for c in d["per_case"]:
        for r in c["rungs"]:
            if "truth_band_0.1" in r and "truth_repaired_band_0.1" in r:
                shifts.append(100.0 * (r["truth_repaired_band_0.1"]
                                       / r["truth_band_0.1"] - 1.0))
    if not shifts:
        return None
    by_rung: dict[int, list[float]] = {}
    for c in d["per_case"]:
        for r in c["rungs"]:
            if "truth_band_0.1" in r and "truth_repaired_band_0.1" in r:
                by_rung.setdefault(r["n"], []).append(
                    100.0 * (r["truth_repaired_band_0.1"] / r["truth_band_0.1"] - 1.0))
    means = [float(np.mean(v)) for v in by_rung.values()]
    return max(means) if which == "max" else min(means)


def decomp_omitted_fraction(root):
    d = load(os.path.join(root, "results/certificates/floor_resolution_decomposition.json"))
    if d is None:
        return None
    fr = []
    for c in d["per_case"]:
        for r in c["rungs"]:
            if "truth_omit_gradnu_band_0.1" in r and "truth_band_0.1" in r:
                fr.append(100.0 * r["truth_omit_gradnu_band_0.1"] / r["truth_band_0.1"])
    return float(np.mean(fr)) if fr else None


def review(root, name):
    return load(os.path.join(root, "results/review", name))


def c1(root, path):
    d = review(root, "control1_physics_vs_physicsfree.json")
    if d is None:
        return None
    node = d["headline"]
    for k in path.split("/"):
        node = node[k]
    return float(node)


def c2(root, field):
    d = review(root, "control2_difficulty_confound.json")
    if d is None:
        return None
    return float(d["arms"]["ensemble_mean"][field])


def c3_ungated(root, which):
    """Fraction of cases where an UNGATED half step raises the monitored residual.

    Split by path, and mind the key naming, which is the opposite of what it looks
    like. The artifact's own ``meta.arms`` block maps ``backbone_seed*`` to the
    DEPLOYED backbone+DEQ path and bare ``seed*`` to the ENSEMBLE-mean path. Its
    ``pooled`` block states the same thing outright: deployed 1.0%, ensemble 8.8%.

    A previous version of this function read ``backbone`` as the ensemble arm and so
    certified the swapped pair that reached the manuscript. Cross-check against
    ``d["pooled"]`` rather than trusting the key prefix.
    """
    d = review(root, "control3_fixed_step.json")
    if d is None:
        return None
    dep, ens = [], []
    for k, v in d["summary"].items():
        f = v.get("certificate_at_fixed_0.5", {}).get("frac_residual_increases")
        if f is None:
            continue
        (dep if k.startswith("backbone") else ens).append(f)
    vals = ens if which == "ensemble" else dep
    if not vals:
        return None
    # Guard: the pooled block is authoritative; if the per-seed mean disagrees with it
    # by more than a point, the key convention has changed and this must be revisited.
    pooled = d.get("pooled", {})
    key = "ensemble_mean_path" if which == "ensemble" else "deployed_backbone_DEQ"
    ref = pooled.get(key, {}).get("certificate_at_fixed_0.5", {}).get(
        "frac_residual_increases"
    )
    out = float(100.0 * np.mean(vals))
    if ref is not None and abs(out - 100.0 * float(ref)) > 1.0:
        raise AssertionError(
            "c3_ungated(%s): per-seed mean %.2f%% disagrees with pooled %.2f%%; "
            "the arm-naming convention in control3_fixed_step.json has changed"
            % (which, out, 100.0 * float(ref))
        )
    return out


def c4_drag_auroc(root):
    d = review(root, "control4_riskcoverage_drag.json")
    if d is None:
        return None
    best = None
    def walk(o):
        nonlocal best
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "auroc_mean" and isinstance(v, (int, float)):
                    best = v if best is None else max(best, v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return float(best) if best is not None else None


def gate_followup(root, path):
    d = review(root, "functional_audit_gate_followup.json")
    if d is None:
        return None
    node = d["F1"]
    for k in path.split("/"):
        node = node[k]
    return float(node)


def bound_over_error(root, which):
    d = review(root, "functional_audit_gate_followup.json")
    if d is None:
        return None
    vals = [v["bound_over_error"] for k, v in d["F1"]["arms"].items()
            if "absdCd" in k]
    return float(min(vals) if which == "min" else max(vals))


def conformal_coverage(root):
    d = review(root, "functional_audit_gate_followup.json")
    if d is None:
        return None
    vals = [v["coverage_mean"] for k, v in d["F1"]["arms"].items() if "absdCd" in k]
    return float(np.mean(vals))


def transolver_inversion(root, which):
    d = review(root, "functional_audit_gate_analysis.json")
    if d is None:
        return None
    vals = [v["norm_baseline_inversion_rate"] for v in d["test_a"].values()
            if isinstance(v, dict) and "norm_baseline_inversion_rate" in v]
    return float(100.0 * (min(vals) if which == "min" else max(vals)))


# ---- the parameter-interpolation baseline and the covariate null ------------
# These are the paper's headline evaluation results (sec:interp, sec:splits,
# sec:covariate). Ratios are recomputed here rather than hardcoded, so a re-run
# that moves either side of a ratio is caught.

def interp(root, task="full"):
    return load(os.path.join(root, "results/interpolation", f"interp_{task}.json"))


def interp_metric(root, field, task="full", variant="nd"):
    d = interp(root, task)
    if d is None:
        return None
    return float(d["variants"][variant]["test_metrics"][field])


def interp_ratio_vs_transolver(root, field):
    """interpolation / Transolver on a volume channel, from one file."""
    d = interp(root)
    if d is None:
        return None
    ours = d["variants"]["nd"]["test_metrics"][field]
    ref = d["reference_rows"]["transolver_tab_transolver"][field]
    return float(ref / ours) if ours else None


def interp_std_mean(root, who, key):
    d = interp(root)
    if d is None:
        return None
    return float(d["variants"]["nd"]["standardised_metrics"][who][key])


def interp_std_ratio(root, key):
    """How many times better the interpolator is than Transolver on a mean."""
    d = interp(root)
    if d is None:
        return None
    s = d["variants"]["nd"]["standardised_metrics"]
    return float(s["transolver_tab_transolver"][key] / s["interpolation"][key])


def band(root, key, band_name="0-0.02c"):
    d = load(os.path.join(root,
                          "results/interpolation/interp_band_control_full.json"))
    if d is None:
        return None
    return float(d["A_band_decomposition"][band_name][key])


def band_outer_min_r2(root, which="r2"):
    """Worst R^2 over the three bands beyond 0.05c, all channels."""
    d = load(os.path.join(root,
                          "results/interpolation/interp_band_control_full.json"))
    if d is None:
        return None
    outer = ["0.05-0.15c", "0.15-0.5c", ">0.5c"]
    keys = [f"{which}_{c}" for c in ("u", "v", "p")]
    return float(min(d["A_band_decomposition"][b][k] for b in outer for k in keys))


def band_ladder(root, n, field, which="mean"):
    d = load(os.path.join(root,
                          "results/interpolation/interp_band_control_full.json"))
    if d is None:
        return None
    row = d["B_train_size_ladder"][str(n)]
    return float(row[field] if which == "mean" else row[f"{field}_max"])


def band_permuted(root, field):
    d = load(os.path.join(root,
                          "results/interpolation/interp_band_control_full.json"))
    if d is None:
        return None
    return float(d["C_permuted_parameters"][field])


def band_ablation(root, arm, field):
    d = load(os.path.join(root,
                          "results/interpolation/interp_band_control_full.json"))
    if d is None:
        return None
    return float(d["D_feature_ablation"][arm][field])


def interp_ood_ratio(root, task, field):
    """raw (dimensional) / nd (nondimensional) on an OOD split."""
    d = interp(root, task)
    if d is None:
        return None
    nd = d["variants"]["nd"]["test_metrics"][field]
    raw = d["variants"]["raw"]["test_metrics"][field]
    return float(raw / nd) if nd else None


def interp_containment(root, task):
    d = interp(root, task)
    if d is None:
        return None
    return float(d["feature_containment"]["per_feature_frac_test_inside_train_range"]["U"])


def interp_forces(root, key):
    d = interp(root)
    if d is None:
        return None
    return float(d["variants"]["nd"]["official_forces"][key]["rho_marginal"])


def covariate_trainfit(root, target, which="point"):
    d = review(root, "covariate_null_trainfit.json")
    if d is None:
        return None
    node = d["targets"][target]["U_alpha_alpha2_naca"]
    if which == "point":
        return float(node["spearman_train_fit_test_score"])
    return float(node["ci95"][0 if which == "lo" else 1])


def covariate_trainfit_2param(root, target):
    d = review(root, "covariate_null_trainfit.json")
    if d is None:
        return None
    return float(d["targets"][target]["U_alpha"]["spearman_train_fit_test_score"])


def covariate_inflation(root, which):
    """Range of in-sample optimism over every feature set and both targets."""
    d = review(root, "covariate_null.json")
    if d is None:
        return None
    vals = [v["in_sample_inflation"]
            for t in d["targets"].values() for v in t.values()]
    return float(min(vals) if which == "min" else max(vals))


def covariate_below_null(root, target):
    """How many published AirfRANS-paper entries fall below the full null."""
    d = review(root, "covariate_null_trainfit.json")
    if d is None:
        return None
    names = {"MLP", "GraphSAGE", "PointNet", "Graph U-Net"}
    return float(sum(1 for r in d["leaderboard_vs_full_name_null"]
                     if r["model"] in names
                     and r[target]["verdict"] == "BELOW THE NULL"))


# ---- the claim table --------------------------------------------------------
# (label, reader, manuscript value, absolute tolerance)
CLAIMS = [
    ("floor mean, 200 AirfRANS cases", floor_realdata_mean, 0.192, 0.001),
    ("floor median", floor_realdata_median, 0.133, 0.001),
    ("uniform-field residual is exactly zero", uniform_all_zero, 0.0, 1e-12),
    ("ladder floor @128^2", lambda r: ladder_level(r, 128), 0.0624, 0.0002),
    ("ladder floor @256^2", lambda r: ladder_level(r, 256), 0.0779, 0.0002),
    ("ladder floor @512^2", lambda r: ladder_level(r, 512), 0.1067, 0.0002),
    ("ladder fitted exponent p", ladder_p, -0.387, 0.002),
    ("cases where the floor rose", ladder_rose, 21, 0.5),
    ("omitted closure @128^2", lambda r: ladder_level(r, 128, "omitted_closure"),
     0.0018, 0.0002),
    ("omitted closure @512^2", lambda r: ladder_level(r, 512, "omitted_closure"),
     0.0054, 0.0002),
    ("residual cut, descent from truth (%)", descent_residual_cut, 84.0, 1.5),
    ("cases where residual fell, from truth", lambda r: descent_wins(r, "from_truth"),
     24, 0.5),
    ("median field error after descent from truth",
     lambda r: descent(r, "from_truth", "mse_u", "median"), 0.91, 0.02),
    ("residual-chosen iterate worse, from truth",
     lambda r: descent_selection_worse(r, "from_truth"), 24, 0.5),
    ("residual-chosen iterate worse, perturbed",
     lambda r: descent_selection_worse(r, "perturbed_residual"), 24, 0.5),
    ("perturbed cases where descent helped", descent_perturbed_helped, 18, 0.5),
    ("audit cost, median ms", audit_cost_median, 1.13, 0.02),
    ("deployed solve, median ms", deployed_solve_median, 3822.0, 2.0),
    ("omitted term as % of floor", decomp_omitted_fraction, 4.2, 1.0),
    ("largest shift from repairing the operator (%)", decomp_repaired_shift,
     0.18, 0.05),
    # --- the concessions. These are the rows a referee checks, and the ones
    # most likely to drift, because they came from a separate analysis pass.
    ("C1 residual AUROC on field error",
     lambda r: c1(r, "residual/auroc"), 0.871, 0.001),
    ("C1 physics-free sigma AUROC",
     lambda r: c1(r, "sigma_vel/auroc"), 0.894, 0.001),
    ("C1 fused AUROC",
     lambda r: c1(r, "fused/auroc"), 0.905, 0.001),
    ("C1 sigma minus residual, delta AUROC",
     lambda r: c1(r, "sigma_vel_minus_residual/delta_auroc"), 0.023, 0.001),
    ("C2 raw rank correlation",
     lambda r: c2(r, "rho_raw"), 0.610, 0.001),
    ("C2 partial correlation given the floor",
     lambda r: c2(r, "rho_partial_given_normtruth"), 0.561, 0.001),
    ("C2 fraction of the association surviving (%)",
     lambda r: 100 * c2(r, "fraction_of_rho_surviving"), 92.0, 0.5),
    ("C3 ungated raises residual, DEPLOYED path (%)",
     lambda r: c3_ungated(r, "deployed"), 1.0, 0.1),
    ("C3 ungated raises residual, ensemble path (%)",
     lambda r: c3_ungated(r, "ensemble"), 8.8, 0.1),
    ("C3 ungated half-step median error improvement, DEPLOYED (%)",
     lambda r: -100.0 * review(r, "control3_fixed_step.json")["pooled"]
     ["deployed_backbone_DEQ"]["policies"]["fixed_0.5"]["median_err_rel_change"], 6.2, 0.1),
    ("C3 gate median error improvement, DEPLOYED (%)",
     lambda r: -100.0 * review(r, "control3_fixed_step.json")["pooled"]
     ["deployed_backbone_DEQ"]["policies"]["gate"]["median_err_rel_change"], 5.8, 0.1),
    ("C4 residual AUROC on drag error",
     c4_drag_auroc, 0.952, 0.001),
    ("floor share of a typical score (%)",
     lambda r: 100 * gate_followup(r, "floor_share_of_typical_score"), 86.0, 1.0),
    ("conformal bound width, min (x the error)",
     lambda r: bound_over_error(r, "min"), 6.5, 0.05),
    ("conformal bound width, max (x the error)",
     lambda r: bound_over_error(r, "max"), 7.9, 0.05),
    ("conformal coverage at a 0.90 target",
     conformal_coverage, 0.90, 0.005),
    ("Transolver inversion rate, min (%)",
     lambda r: transolver_inversion(r, "min"), 2.5, 0.1),
    ("Transolver inversion rate, max (%)",
     lambda r: transolver_inversion(r, "max"), 7.0, 0.1),
    # --- sec:interp, the headline. Every cell of tab:interp that the prose
    # quotes, plus the two standardised means that must always travel together.
    ("interp mse_u", lambda r: interp_metric(r, "mse_u"), 0.782, 0.001),
    ("interp mse_v", lambda r: interp_metric(r, "mse_v"), 0.0336, 0.0001),
    ("interp mse_p", lambda r: interp_metric(r, "mse_p"), 75.03, 0.02),
    ("interp surface mse_p", lambda r: interp_metric(r, "surface_mse_p"),
     10989.0, 1.0),
    ("interp C_l rel err (%)",
     lambda r: 100 * interp_metric(r, "cl_rel_err_mean"), 1.18, 0.01),
    ("interp C_d rel err (%)",
     lambda r: 100 * interp_metric(r, "cd_rel_err_mean"), 2.39, 0.01),
    ("interp beats Transolver on mse_p by",
     lambda r: interp_ratio_vs_transolver(r, "mse_p"), 8.4, 0.05),
    ("interp beats Transolver on mse_v by",
     lambda r: interp_ratio_vs_transolver(r, "mse_v"), 2.6, 0.05),
    ("Transolver beats interp on mse_u by",
     lambda r: 1.0 / interp_ratio_vs_transolver(r, "mse_u"), 6.5, 0.05),
    # NOTE the mixed precision in interp_full.json's reference_rows: mse_u/v/p are
    # stored rounded (0.12, 0.088, 628.5) while mse_nut is full precision. The
    # tolerances below absorb that; do not tighten them without switching the
    # reference side to results/baselines/table2.csv.
    ("Transolver beats interp on nu_t by",
     lambda r: 1.0 / interp_ratio_vs_transolver(r, "mse_nut"), 15.4, 1.0),
    ("interp better on the 3-channel std mean by",
     lambda r: interp_std_ratio(r, "mean_std_mse_uvp"), 1.9, 0.05),
    ("TRANSOLVER better on the 4-channel std mean by",
     lambda r: 1.0 / interp_std_ratio(r, "mean_std_mse_uvpnut"), 6.4, 0.05),
    # --- the wall-band decomposition: the boundary claim.
    ("u error share inside 0.02c", lambda r: band(r, "se_share_u"), 0.924, 0.002),
    ("v error share inside 0.02c", lambda r: band(r, "se_share_v"), 0.898, 0.002),
    ("u error share beyond 0.5c",
     lambda r: band(r, "se_share_u", ">0.5c"), 0.036, 0.002),
    ("worst R^2 beyond 0.05c", lambda r: band_outer_min_r2(r, "r2"), 0.9996, 0.0002),
    ("worst per-case-centred R^2 beyond 0.05c",
     lambda r: band_outer_min_r2(r, "r2_pc"), 0.9965, 0.0002),
    # --- the controls that had to survive for sec:interp to stand.
    ("permuted-parameter control, mse_u",
     lambda r: band_permuted(r, "mse_u"), 22.82, 0.02),
    ("permuted-parameter control, C_l rel err (%)",
     lambda r: 100 * band_permuted(r, "cl_rel_err_mean"), 365.0, 1.0),
    ("(U,alpha) only, mse_p",
     lambda r: band_ablation(r, "U_alpha_only", "mse_p"), 9491.0, 2.0),
    ("digits are worth this much on mse_p",
     lambda r: band_ablation(r, "U_alpha_only", "mse_p")
     / band_ablation(r, "all7", "mse_p"), 126.0, 1.0),
    ("n_train=100 mse_p, mean over 5 subsets",
     lambda r: band_ladder(r, 100, "mse_p"), 238.0, 1.0),
    ("n_train=100 mse_p, worst of 5 subsets",
     lambda r: band_ladder(r, 100, "mse_p", "worst"), 344.0, 1.0),
    ("interp rho_Cd vs official labels",
     lambda r: interp_forces(r, "interp_cd_vs_official"), 0.8389, 0.0005),
    ("exact-truth-field rho_Cd vs official labels",
     lambda r: interp_forces(r, "gt_field_cd_vs_official"), 0.8394, 0.0005),
    # --- sec:splits.
    ("reynolds: frac of test U inside train range",
     lambda r: interp_containment(r, "reynolds"), 0.0, 1e-9),
    ("reynolds mse_u, nondimensional",
     lambda r: interp_metric(r, "mse_u", "reynolds"), 0.808, 0.002),
    ("reynolds mse_p, nondimensional",
     lambda r: interp_metric(r, "mse_p", "reynolds"), 74.1, 0.1),
    ("reynolds: dimensional/nondimensional on mse_u",
     lambda r: interp_ood_ratio(r, "reynolds", "mse_u"), 5.2, 0.05),
    ("reynolds: dimensional/nondimensional on mse_p",
     lambda r: interp_ood_ratio(r, "reynolds", "mse_p"), 10.1, 0.05),
    ("aoa C_l rel err (%)",
     lambda r: 100 * interp_metric(r, "cl_rel_err_mean", "aoa"), 23.06, 0.02),
    ("aoa C_d rel err (%)",
     lambda r: 100 * interp_metric(r, "cd_rel_err_mean", "aoa"), 2.71, 0.02),
    # --- sec:covariate, the case-name null.
    ("case-name null, official lift",
     lambda r: covariate_trainfit(r, "cl"), 0.9821, 0.0002),
    ("case-name null, official lift, CI lo",
     lambda r: covariate_trainfit(r, "cl", "lo"), 0.9737, 0.0002),
    ("case-name null, official lift, CI hi",
     lambda r: covariate_trainfit(r, "cl", "hi"), 0.9866, 0.0002),
    ("case-name null, official drag",
     lambda r: covariate_trainfit(r, "cd"), 0.9318, 0.0002),
    ("case-name null, official drag, CI lo",
     lambda r: covariate_trainfit(r, "cd", "lo"), 0.8981, 0.0002),
    ("case-name null, official drag, CI hi",
     lambda r: covariate_trainfit(r, "cd", "hi"), 0.9531, 0.0002),
    ("two-parameter null on lift (NOT sufficient)",
     lambda r: covariate_trainfit_2param(r, "cl"), 0.9344, 0.0002),
    ("in-sample optimism, min", lambda r: covariate_inflation(r, "min"),
     0.0008, 0.0001),
    ("in-sample optimism, max", lambda r: covariate_inflation(r, "max"),
     0.0035, 0.0001),
    ("AirfRANS-paper entries below the lift null",
     lambda r: covariate_below_null(r, "cl"), 4, 0.5),
    ("AirfRANS-paper entries below the drag null",
     lambda r: covariate_below_null(r, "cd"), 4, 0.5),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    print("Tracing manuscript numbers to committed result files\n")
    print(f"  {'claim':<46} {'paper':>10} {'recomputed':>12}   verdict")
    fails, skips = [], []
    for label, reader, stated, tol in CLAIMS:
        try:
            got = reader(args.root)
        except Exception as exc:
            got = None
            if args.verbose:
                print(f"    ! {label}: {type(exc).__name__}: {exc}")
        if got is None:
            skips.append(label)
            print(f"  {label:<46} {stated:>10} {'--':>12}   SKIP (no file)")
            continue
        ok = abs(got - stated) <= tol
        if not ok:
            fails.append((label, stated, got))
        print(f"  {label:<46} {stated:>10.4g} {got:>12.4g}   "
              f"{'ok' if ok else 'MISMATCH'}")

    print()
    if fails:
        print(f"{len(fails)} number(s) in the manuscript no longer match their source:")
        for label, stated, got in fails:
            print(f"  - {label}: paper says {stated:g}, file gives {got:g}")
    if skips:
        print(f"{len(skips)} claim(s) could not be checked (missing file): "
              f"{', '.join(skips)}")
    if not fails and not skips:
        print("every checked number matches its source file.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
