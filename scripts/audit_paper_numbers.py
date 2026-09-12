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


# The next four readers, and transolver_inversion/gate_followup below, have no
# CLAIMS row any more: the 2026-09-11 compaction pass moved the passages that quoted
# them to the companion paper. They are retained, unreferenced, so that paper's
# auditor can reuse them without re-deriving the key paths.

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


def covariate_entries_scored(root, target):
    """Total published entries carrying a number on this coefficient.

    This is the denominator of the manuscript's "four of five" on lift: the four
    AirfRANS-paper baselines plus Transolver, which clears. Transolver reports no
    drag column, so on drag the denominator is four, not five.
    """
    d = review(root, "covariate_null_trainfit.json")
    if d is None:
        return None
    return float(sum(1 for r in d["leaderboard_vs_full_name_null"]
                     if r[target].get("published") is not None))


def covariate_drag_spans_zero(root):
    """Published drag intervals (mean +- published seed std) that contain zero."""
    d = review(root, "covariate_null_trainfit.json")
    if d is None:
        return None
    n = 0
    for r in d["leaderboard_vs_full_name_null"]:
        cd = r["cd"]
        m, s = cd.get("published"), cd.get("published_std")
        if m is None or s is None:
            continue
        if m - s <= 0.0 <= m + s:
            n += 1
    return float(n)


# ---- the derived percentages the abstract and the introduction bold ---------
# Each is recomputed from the same two artifacts the ratio beside it uses, so a
# re-run that moves either side is caught in both forms at once.

def interp_pct_lower(root, field):
    """How much lower the interpolator's error is than Transolver's, in percent."""
    d = interp(root)
    if d is None:
        return None
    ours = d["variants"]["nd"]["test_metrics"][field]
    ref = d["reference_rows"]["transolver_tab_transolver"][field]
    return float(100.0 * (1.0 - ours / ref)) if ref else None


def transolver_table2(root, metric):
    """Read one Transolver cell out of the committed baseline table.

    interp_full.json's reference_rows block carries no force columns, so the
    force ratios must come from results/baselines/table2.csv, which is the file
    tab:interp's Transolver row was transcribed from.
    """
    path = os.path.join(root, "results/baselines/table2.csv")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5 and parts[0] == "transolver" and parts[2] == metric:
                return float(parts[3])
    return None


def interp_force_ratio(root, metric):
    """Transolver / interpolation on a mean relative force error."""
    ref = transolver_table2(root, metric)
    ours = interp_metric(root, metric)
    if ref is None or ours is None or not ours:
        return None
    return float(ref / ours)


def band_outer_cell_frac(root):
    """Percent of cells lying beyond 0.05c -- the domain share the R^2 claim covers."""
    d = load(os.path.join(root,
                          "results/interpolation/interp_band_control_full.json"))
    if d is None:
        return None
    outer = ["0.05-0.15c", "0.15-0.5c", ">0.5c"]
    return float(100.0 * sum(d["A_band_decomposition"][b]["cell_frac"]
                             for b in outer))


# ---- tab:measure, tab:bandratio and the measure-asymmetry paragraphs ---------
# Producer: scripts/measure_asymmetry.py (rules D1-D3 pre-registered in cdf08f5).
# These readers back the claim that the field-MSE ranking in sec:interp is a
# property of the cell weighting, and the r128 representation ceiling that bounds
# every field number in that section.

MA_BANDS = ["0-0.005c", "0.005-0.01c", "0.01-0.02c", "0.02-0.05c",
            "0.05-0.15c", "0.15-0.5c", ">0.5c"]


def ma(root):
    return load(os.path.join(root,
                             "results/interpolation/measure_asymmetry.json"))


def ma_nodes(root):
    return load(os.path.join(root,
                             "results/interpolation/measure_asymmetry_nodes.json"))


def ma_sens(root):
    return load(os.path.join(
        root, "results/interpolation/measure_asymmetry_sensitivity.json"))


def ma_ratio(root, chan, which):
    """R = Transolver MSE / interpolation MSE; R>1 means the interpolator wins."""
    d = ma(root)
    if d is None:
        return None
    return float(d["verdicts"]["D3"]["ratios"][chan][which])


def ma_ratio_inv(root, chan, which):
    """The same comparison stated the way the manuscript states a loss."""
    r = ma_ratio(root, chan, which)
    return None if not r else float(1.0 / r)


def ma_cell(root, weighting, arm, chan):
    """One cell of tab:measure; Transolver is the mean over the three seeds."""
    d = ma(root)
    if d is None:
        return None
    block = d[f"C_case_mean_{weighting}"]
    if arm == "interp":
        return float(block["interp"][chan])
    seeds = sorted(k for k in block if k.startswith("transolver_seed"))
    return float(np.mean([block[k][chan] for k in seeds]))


def ma_std_mean(root, weighting, channels):
    """Standardised-mean ratio on tab:measure's own rows.

    CONVENTION: returns interp / Transolver, so a value ABOVE 1 means Transolver is
    better by that factor. Callers that quote the interpolator's advantage (the
    area-uniform u,v,p mean, 1.99x) must take the reciprocal; the other three rows
    in the claim table quote Transolver's advantage and do not. The four rows share
    this reader, so the label is the only thing that distinguishes them -- do not
    edit one without re-reading this.

    The train-set variances are read from interp_full.json's var_train block --
    the same divisors tab:interp_std uses -- rather than hard-coded here. They are
    common to both arms, so the ratio is like-for-like.
    """
    d = interp(root)
    if d is None or ma(root) is None:
        return None
    var = d["variants"]["nd"]["standardised_metrics"]["var_train"]
    means = {}
    for arm in ("interp", "transolver"):
        means[arm] = float(np.mean(
            [ma_cell(root, weighting, arm, f"mse_{c}") / var[c] for c in channels]))
    return float(means["interp"] / means["transolver"])


def ma_band_ratio(root, chan, band):
    d = ma(root)
    if d is None:
        return None
    return float(d["verdicts"]["D2"]["r_band_interp_over_transolver"][chan]
                 [MA_BANDS.index(band)])


def ma_band_span(root):
    d = ma(root)
    return None if d is None else float(d["verdicts"]["D2"]["S_u_inner_over_outer"])


def ma_band_inversions(root):
    d = ma(root)
    return None if d is None else float(d["verdicts"]["D2"]["n_inversions_u"])


def ma_transolver_se_share(root, chan, band=">0.5c"):
    """Transolver's own share of squared error in a band, seed-mean, in percent."""
    d = ma(root)
    if d is None:
        return None
    blk = d["B_band_decomposition_7"]
    seeds = sorted(k for k in blk if k.startswith("transolver_seed"))
    return float(100.0 * np.mean([blk[k][band][f"se_share_{chan}"] for k in seeds]))


def ma_gap(root, threshold, key):
    """D1: node fraction, area fraction, or their ratio, inside a wall distance."""
    d = ma_nodes(root)
    if d is None:
        return None
    return float(d["D1_gap"][f"inside_{threshold}"][key])


def ma_band_mass(root, measure, band):
    d = ma_sens(root)
    if d is None:
        return None
    return float(d["band_masses"][measure][MA_BANDS.index(band)])


def ma_band_mass_gap(root, band):
    a = ma_band_mass(root, "area_uniform", band)
    n = ma_band_mass(root, "band_true_node_crop", band)
    return None if not a or n is None else float(n / a)


def ma_ceiling(root, band=None):
    """The r128 round-trip error at the native nodes, over Transolver's own, on u.

    band=None pools over the scored crop with the in-crop node counts as weights.
    Read from mse_in_crop and never from mse_full: outside the crop the bilinear
    sampler clamps query points to the domain boundary, so the outer bands of the
    full-cloud block are contaminated.
    """
    d = ma(root)
    if d is None:
        return None
    blk = d["D_node_space"]
    n = np.asarray(blk["n_nodes_in_crop"], dtype=float)
    ceil = np.asarray(blk["mse_in_crop"]["r128_ceiling"]["u"], dtype=float)
    seeds = sorted(k for k in blk["mse_in_crop"] if k.startswith("transolver_seed"))
    model = np.mean([np.asarray(blk["mse_in_crop"][k]["u"], dtype=float)
                     for k in seeds], axis=0)
    if band is None:
        return float(float(ceil @ n) / float(model @ n))
    i = MA_BANDS.index(band)
    return float(ceil[i] / model[i])


def ma_node_mse_u_inner(root):
    """Transolver's own per-node u error in the innermost band, seed-mean."""
    d = ma(root)
    if d is None:
        return None
    blk = d["D_node_space"]["mse_in_crop"]
    seeds = sorted(k for k in blk if k.startswith("transolver_seed"))
    return float(np.mean([blk[k]["u"][0] for k in seeds]))


def ma_ceiling_abs(root, band="0-0.005c"):
    """The r128 round-trip u error at the native nodes inside one band."""
    d = ma(root)
    if d is None:
        return None
    return float(d["D_node_space"]["mse_in_crop"]["r128_ceiling"]["u"]
                 [MA_BANDS.index(band)])


def ma_sens_p(root, which):
    d = ma_sens(root)
    return None if d is None else float(d["p_node_measure_range"][which])


def ma_sens_worst(root, chan, which):
    """Extreme of 1/R on u or v over the five node-measure constructions."""
    d = ma_sens(root)
    if d is None:
        return None
    vals = [1.0 / d["R"][chan][k] for k in d["p_node_measure_range"]["constructions"]]
    return float(min(vals) if which == "min" else max(vals))


# ---- the native-resolution head-to-head (scripts/point_space_headtohead.py) --
# Rules P1-P4 were committed in a54cb75 before any number existed; amendment 1
# (4ba4dba) raised a GATE tolerance only and amendment 2 added the least-squares
# stage as a supplementary diagnostic with no threshold of its own.
#
# ARM DISCIPLINE, stated here because it is the one place this file could silently
# drift. P1 selects the better of the two in-body conventions per channel on the
# POOLED value, and chose "nearfill" on all four. Every native-resolution number
# in the manuscript -- the aggregate, the band ladder and the wall row -- is read
# off that same arm. docs/paper/review/point_space_headtohead.md Sec.4 prints a
# wall row computed from the OTHER arm ("bridge": 4052x on u, 2823x on v); the
# manuscript quotes the nearfill values (4060x, 2834x) that its own P1 selection
# implies, and the rows below check the manuscript against the arm it declares.
PS_BANDS8 = ["wall", "0-0.005c", "0.005-0.01c", "0.01-0.02c", "0.02-0.05c",
             "0.05-0.15c", "0.15-0.5c", ">0.5c"]
PS_BANDS7 = PS_BANDS8[1:]


def ps(root):
    return load(os.path.join(
        root, "results/interpolation/point_space_headtohead.json"))


def ps_ls(root):
    return load(os.path.join(
        root, "results/interpolation/point_space_oracle_ls.json"))


def ps_ratio(root, chan):
    """P1: per-node MSE ratio interp/Transolver on the native cloud."""
    d = ps(root)
    return None if d is None else float(d["P1"]["R_per_channel"][chan])


def ps_std_ratio(root):
    d = ps(root)
    if d is None:
        return None
    return float(d["P1"]["standardised_mean_uvp"]["ratio_interp_over_transolver"])


def ps_pooled(root, arm, chan):
    """Pooled per-node MSE over all 35.8M native nodes, full cloud."""
    d = ps(root)
    return None if d is None else float(d["tables"][arm]["full"][chan]["pooled"])


def ps_memo(root, which, chan):
    """The two memo rows of tab:native, in-crop pooled (clamp-free region)."""
    d = ps(root)
    return None if d is None else float(d["r128_resample"][which][chan]["pooled"])


def ps_memo_cost(root, chan):
    """What the grid protocol cost the INTERPOLATOR: r128-resampled / native."""
    a = ps_memo(root, "interp_r128_resample", chan)
    d = ps(root)
    if a is None or d is None:
        return None
    b = float(d["tables"]["interp_nearfill"]["in_crop"][chan]["pooled"])
    return float(a / b)


def ps_band_u(root, band):
    """P2: the native-node band ratio on u, over BANDS7 (no wall row)."""
    d = ps(root)
    return None if d is None else float(d["P2"]["r_b_u"][PS_BANDS7.index(band)])


def ps_wall_ratio(root, chan):
    """The wall row (sdf = 0), which the seven-band grid cannot represent."""
    d = ps(root)
    if d is None:
        return None
    i = PS_BANDS8.index("wall")
    num = d["tables"]["interp_nearfill"]["full"][chan]["band"][i]
    den = d["tables"]["transolver_mean"]["full"][chan]["band"][i]
    return float(num / den)


def ps_band_ratio(root, chan, band):
    """Native-node band ratio on any channel, same arm as P1 selected."""
    d = ps(root)
    if d is None:
        return None
    i = PS_BANDS8.index(band)
    num = d["tables"]["interp_nearfill"]["full"][chan]["band"][i]
    den = d["tables"]["transolver_mean"]["full"][chan]["band"][i]
    return float(num / den)


def ps_wins(root, chan):
    d = ps(root)
    return None if d is None else float(d["paired_sign_test"][chan]
                                        ["transolver_better_cases"])


def ps_oracle_Q(root):
    """P3: best SINGLE training field, chosen knowing the answer, over the
    surrogate, on u inside 0.005c. 200 cases."""
    d = ps(root)
    return None if d is None else float(d["P3"]["Q_0_0.005c"])


def ps_ls_ratio(root):
    """Amendment 2: the unconstrained least-squares projection of the TRUE field
    onto the span of all 800 transferred training fields -- a genuine lower bound
    for ANY weighting of that family -- over the surrogate, on u inside 0.005c.

    SCOPE, and the manuscript states it inline: n_ls = 3 cases, channel u, the one
    band whose node count (73-82k) far exceeds the 800 free parameters of the
    projection. It does NOT inherit the 200-case sample size of ps_oracle_Q.
    """
    d = ps_ls(root)
    return None if d is None else float(d["P3_LS"]["ratio"])


def ps_ls_ncases(root):
    d = ps_ls(root)
    return None if d is None else float(d["meta"]["n_ls"])


def ps_geom_mismatch(root, key, band="0-0.005c"):
    d = ps(root)
    if d is None:
        return None
    return float(d["diagnostics"][key][PS_BANDS8.index(band)])


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
    # The per-rung "omitted closure" magnitudes, the 84% residual cut, the 24/24
    # residual-fell count and the 0.91 median field error were quoted only by the
    # detailed residual-audit treatment, which the 2026-09-11 compaction pass moved
    # to the companion paper (docs/paper/sections/residual_audit_removed.tex). Their
    # rows are removed here rather than left dangling; the surviving sec:residual
    # quotes the closure term as a FRACTION of the floor, which the
    # decomp_omitted_fraction row below still checks.
    ("residual-chosen iterate worse, from truth",
     lambda r: descent_selection_worse(r, "from_truth"), 24, 0.5),
    ("residual-chosen iterate worse, perturbed",
     lambda r: descent_selection_worse(r, "perturbed_residual"), 24, 0.5),
    ("perturbed cases where descent helped", descent_perturbed_helped, 18, 0.5),
    # sec:residual states this as the range 3.5-4.8%; the reader returns the mean
    # over cases and rungs, which must sit inside it.
    ("omitted term as % of floor (paper: 3.5-4.8)",
     decomp_omitted_fraction, 4.2, 1.0),
    ("largest shift from repairing the operator (%)", decomp_repaired_shift,
     0.18, 0.05),
    # ------------------------------------------------------------------------
    # REMOVED 2026-09-12 by the JOCS rebuild, rather than left dangling:
    #
    #   audit cost / deployed solve (1.13 ms, 3822 ms)   -- sec:cost
    #   C1 x4, C2 x3, C3 x4, C4  (AUROCs, the gate)      -- sec:selective
    #   conformal bound width min/max, coverage           -- sec:conformal
    #
    # All of that material moved to docs/paper/sections/trust_layer_removed.tex
    # for the companion trust-layer paper, which should reinstate these rows.
    # The readers (audit_cost_median, deployed_solve_median, c1, c2, c3_ungated,
    # c4_drag_auroc, bound_over_error, conformal_coverage) are RETAINED and
    # unreferenced, on the same convention as the 2026-09-11 pass, so that paper
    # does not have to re-derive the key paths.
    #
    #   case-name null x6, two-parameter null, in-sample optimism x2,
    #   entries below the lift/drag null, published entries carrying a lift
    #   number, drag intervals spanning zero                -- sec:covariate
    #   reynolds containment / mse_u / mse_p / nd-vs-raw x2,
    #   aoa C_l and C_d relative error                      -- sec:splits
    #
    # That block is the spine of the five-benchmark NullBench paper and must not
    # be journal-published here; it is preserved in
    # docs/paper/sections/covariate_null_removed.tex. Its readers are likewise
    # retained unreferenced.
    # ------------------------------------------------------------------------
    # --- tab:native. The head-to-head at native resolution, no raster in the
    # comparison path. Rules P1-P4 committed in a54cb75 before any number
    # existed. This is the manuscript's lead result, so every cell of the table
    # and every ratio the prose quotes is checked, including the two memo rows
    # that answer "you crippled the baseline".
    ("native: interp/Transolver on mse_u (per node)",
     lambda r: ps_ratio(r, "u"), 144.2, 0.1),
    ("native: interp/Transolver on mse_v (per node)",
     lambda r: ps_ratio(r, "v"), 133.9, 0.1),
    ("native: interp/Transolver on mse_p (per node)",
     lambda r: ps_ratio(r, "p"), 4.85, 0.01),
    ("native: interp/Transolver on mse_nut (per node)",
     lambda r: ps_ratio(r, "nut"), 18.3, 0.05),
    ("native: standardised mean over u,v,p (x against interp)",
     ps_std_ratio, 24.7, 0.05),
    ("native: interp mse_u", lambda r: ps_pooled(r, "interp_nearfill", "u"),
     93.11, 0.01),
    ("native: interp mse_v", lambda r: ps_pooled(r, "interp_nearfill", "v"),
     66.52, 0.01),
    ("native: interp mse_p", lambda r: ps_pooled(r, "interp_nearfill", "p"),
     41740.0, 2.0),
    ("native: Transolver mse_u", lambda r: ps_pooled(r, "transolver_mean", "u"),
     0.6458, 0.0002),
    ("native: Transolver mse_v", lambda r: ps_pooled(r, "transolver_mean", "v"),
     0.4968, 0.0002),
    ("native: Transolver mse_p", lambda r: ps_pooled(r, "transolver_mean", "p"),
     8605.0, 1.0),
    # The in-body convention P1 chose (nearfill) beats the alternative (bridge)
    # on mse_p by 41740 vs 58020, which the manuscript reports as a 28% gift to
    # the interpolator that does not change the verdict.
    ("native: the in-body convention NOT chosen, on mse_p",
     lambda r: ps_pooled(r, "interp_bridge", "p"), 58020.0, 5.0),
    ("native: Transolver wins on u (of 200 cases)",
     lambda r: ps_wins(r, "u"), 200, 0.5),
    ("native: Transolver wins on v (of 200 cases)",
     lambda r: ps_wins(r, "v"), 200, 0.5),
    ("native: Transolver wins on p (of 200 cases)",
     lambda r: ps_wins(r, "p"), 142, 0.5),
    ("native: Transolver wins on nu_t (of 200 cases)",
     lambda r: ps_wins(r, "nut"), 200, 0.5),
    # tab:native memo rows -- the control that kills "you crippled the baseline".
    ("memo: published r128 interp resampled at the nodes, u",
     lambda r: ps_memo(r, "interp_r128_resample", "u"), 400.2, 0.2),
    ("memo: published r128 interp resampled at the nodes, v",
     lambda r: ps_memo(r, "interp_r128_resample", "v"), 189.7, 0.2),
    ("memo: published r128 interp resampled at the nodes, p",
     lambda r: ps_memo(r, "interp_r128_resample", "p"), 3.058e6, 2000.0),
    ("memo: the r128 raster's own round-trip error at the nodes, u",
     lambda r: ps_memo(r, "r128_ceiling", "u"), 281.5, 0.2),
    ("memo: the r128 raster's own round-trip error at the nodes, v",
     lambda r: ps_memo(r, "r128_ceiling", "v"), 272.6, 0.2),
    ("memo: the r128 raster's own round-trip error at the nodes, p",
     lambda r: ps_memo(r, "r128_ceiling", "p"), 3.012e6, 2000.0),
    ("what the grid protocol cost the INTERPOLATOR on u (x)",
     lambda r: ps_memo_cost(r, "u"), 4.1, 0.05),
    ("what the grid protocol cost the INTERPOLATOR on p (x)",
     lambda r: ps_memo_cost(r, "p"), 69.9, 0.5),
    # --- P2: the native band ladder, and the far-field u entry that DISAGREES
    # with the grid ladder. The manuscript reports both verdicts by name.
    ("P2 native: band ratio on u, 0-0.005c",
     lambda r: ps_band_u(r, "0-0.005c"), 173.0, 0.1),
    ("P2 native: band ratio on u, >0.5c (grid says 0.312; they disagree)",
     lambda r: ps_band_u(r, ">0.5c"), 3.79, 0.01),
    ("P2 native: span S on u", lambda r: ps(r)["P2"]["S"], 45.6, 0.1),
    ("P2 native: inversions (2 -> PARTIAL, against CONFIRMED on the grid)",
     lambda r: ps(r)["P2"]["n_inversions"], 2, 0.5),
    ("native: band ratio on v, 0-0.005c",
     lambda r: ps_band_ratio(r, "v", "0-0.005c"), 167.2, 0.2),
    ("native: band ratio on p, 0-0.005c",
     lambda r: ps_band_ratio(r, "p", "0-0.005c"), 5.42, 0.01),
    ("native: interp wins v beyond 0.5c by (x)",
     lambda r: 1.0 / ps_band_ratio(r, "v", ">0.5c"), 17.7, 0.1),
    ("native: interp wins p beyond 0.5c by (x)",
     lambda r: 1.0 / ps_band_ratio(r, "p", ">0.5c"), 265.0, 1.0),
    # The wall row (sdf = 0), which the seven-band grid cannot represent at all.
    # Read on the SAME arm P1 selected -- see the ARM DISCIPLINE note above.
    ("native wall row: u", lambda r: ps_wall_ratio(r, "u"), 4060.0, 3.0),
    ("native wall row: v", lambda r: ps_wall_ratio(r, "v"), 2834.0, 3.0),
    ("native wall row: p", lambda r: ps_wall_ratio(r, "p"), 1.51, 0.01),
    ("native wall row: nu_t", lambda r: ps_wall_ratio(r, "nut"), 382.0, 1.0),
    # --- P3 and amendment 2: the two bounds that close the "then re-tune it"
    # escape. NOTE the sample sizes differ and the manuscript says so inline.
    ("P3 oracle: best SINGLE training field over Transolver, u, 0-0.005c (200 cases)",
     ps_oracle_Q, 343.6, 0.2),
    ("P3-LS: least-squares bound over the span of all 800 fields (3 cases)",
     ps_ls_ratio, 21.3, 0.05),
    ("P3-LS: the sample size that bound is computed on",
     ps_ls_ncases, 3, 0.5),
    # The mechanism, measured rather than argued (sec:interp).
    ("geometry mismatch inside 0.005c (x further than d_t)",
     lambda r: ps_geom_mismatch(r, "geom_mismatch_over_dt"), 7.8, 0.05),
    ("weight mass landing INSIDE a training body, inside 0.005c",
     lambda r: ps_geom_mismatch(r, "w_frac_inside_body"), 0.354, 0.002),
    ("largest weight mass outside any training hull, any band",
     lambda r: max(ps(r)["diagnostics"]["w_frac_outside_hull"]), 0.00094, 0.00002),
    ("node fraction inside 0.005c, full cloud",
     lambda r: ps_geom_mismatch(r, "node_frac_per_band"), 0.4096, 0.0005),
    # --- sec:interp, the grid comparison. Every cell of tab:interp that the prose
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
    # The "88% lower on mse_p" and "62% lower on mse_v" percentages were withdrawn
    # from the abstract, introduction, sec:interp and the conclusion on 2026-09-12:
    # both are area-uniform statements whose SIGN reverses under the dataset's own
    # node measure (tab:measure), and the manuscript now quotes the two weightings
    # as a pair of ratios instead of a single percentage. Rows removed rather than
    # left dangling; interp_pct_lower() is retained because nothing else computes
    # a percentage form and a future claim may need it.
    ("interp C_l error lower than Transolver by",
     lambda r: interp_force_ratio(r, "cl_rel_err_mean"), 4.9, 0.05),
    ("interp C_d error lower than Transolver by",
     lambda r: interp_force_ratio(r, "cd_rel_err_mean"), 3.8, 0.05),
    # --- tab:measure. The ranking is the weighting: the SAME predictions on the
    # SAME cells, re-weighted from area-uniform to native node count. Every cell
    # and every ratio the manuscript quotes from that table is checked here.
    # Convention: R = Transolver / interp, so R > 1 means the interpolator wins.
    ("measure: R_p area-uniform",
     lambda r: ma_ratio(r, "p", "R_area"), 8.39, 0.02),
    ("measure: R_p node-weighted (the pre-registered D3 estimator)",
     lambda r: ma_ratio(r, "p", "R_node"), 0.440, 0.002),
    ("measure: R_v area-uniform", lambda r: ma_ratio(r, "v", "R_area"), 2.97, 0.02),
    ("measure: R_v node-weighted", lambda r: ma_ratio(r, "v", "R_node"), 0.0663, 0.001),
    ("measure: interp worse on mse_u, area-uniform (x)",
     lambda r: ma_ratio_inv(r, "u", "R_area"), 6.2, 0.05),
    ("measure: interp worse on mse_u, node-weighted (x)",
     lambda r: ma_ratio_inv(r, "u", "R_node"), 86.0, 0.5),
    ("measure: interp worse on nu_t, area-uniform (x)",
     lambda r: ma_ratio_inv(r, "nut", "R_area"), 14.4, 0.1),
    ("measure: interp worse on nu_t, node-weighted (x)",
     lambda r: ma_ratio_inv(r, "nut", "R_node"), 9.2, 0.1),
    ("measure: Transolver mse_p, area-uniform (629.6 here vs 628.5 in tab:interp)",
     lambda r: ma_cell(r, "area_uniform", "transolver", "mse_p"), 629.6, 0.2),
    ("measure: Transolver mse_u, area-uniform",
     lambda r: ma_cell(r, "area_uniform", "transolver", "mse_u"), 0.127, 0.001),
    ("measure: Transolver mse_v, area-uniform",
     lambda r: ma_cell(r, "area_uniform", "transolver", "mse_v"), 0.0999, 0.0005),
    ("measure: interp mse_u, node-weighted",
     lambda r: ma_cell(r, "node_weighted", "interp", "mse_u"), 30.7, 0.05),
    ("measure: interp mse_v, node-weighted",
     lambda r: ma_cell(r, "node_weighted", "interp", "mse_v"), 4.26, 0.01),
    ("measure: interp mse_p, node-weighted",
     lambda r: ma_cell(r, "node_weighted", "interp", "mse_p"), 6012.0, 2.0),
    ("measure: Transolver mse_u, node-weighted",
     lambda r: ma_cell(r, "node_weighted", "transolver", "mse_u"), 0.358, 0.001),
    ("measure: Transolver mse_v, node-weighted",
     lambda r: ma_cell(r, "node_weighted", "transolver", "mse_v"), 0.282, 0.001),
    ("measure: Transolver mse_p, node-weighted",
     lambda r: ma_cell(r, "node_weighted", "transolver", "mse_p"), 2646.0, 2.0),
    # The two standardised means recomputed on tab:measure's own rows, against the
    # same var_train divisors tab:interp_std uses. The area-uniform pair differs
    # from tab:interp_std's 1.9x/6.4x only in the provenance of the Transolver row,
    # and the manuscript says so rather than letting a reader find it.
    ("measure: std mean over u,v,p, area-uniform (interp better by)",
     lambda r: 1.0 / ma_std_mean(r, "area_uniform", ("u", "v", "p")), 1.99, 0.02),
    ("measure: std mean over all four, area-uniform (Transolver better by)",
     lambda r: ma_std_mean(r, "area_uniform", ("u", "v", "p", "nut")), 6.1, 0.05),
    ("measure: std mean over u,v,p, node-weighted (Transolver better by)",
     lambda r: ma_std_mean(r, "node_weighted", ("u", "v", "p")), 8.30, 0.02),
    ("measure: std mean over all four, node-weighted (Transolver better by)",
     lambda r: ma_std_mean(r, "node_weighted", ("u", "v", "p", "nut")), 8.85, 0.02),
    # --- the node/area asymmetry that drives the reversal (rule D1).
    ("D1: node fraction inside 0.05c",
     lambda r: ma_gap(r, "0.05c", "node_frac_crop_A2"), 0.6437, 0.0005),
    ("D1: area fraction inside 0.05c",
     lambda r: ma_gap(r, "0.05c", "area_frac_crop_A1"), 0.01249, 0.00005),
    ("D1: node/area gap inside 0.05c",
     lambda r: ma_gap(r, "0.05c", "g_A2_over_A1"), 51.5, 0.1),
    ("D1: node fraction inside 0.02c",
     lambda r: ma_gap(r, "0.02c", "node_frac_crop_A2"), 0.5553, 0.0005),
    ("D1: area fraction inside 0.02c",
     lambda r: ma_gap(r, "0.02c", "area_frac_crop_A1"), 0.00507, 0.00005),
    ("D1: node/area gap inside 0.02c",
     lambda r: ma_gap(r, "0.02c", "g_A2_over_A1"), 109.5, 0.1),
    ("node fraction inside 0.005c",
     lambda r: ma_band_mass(r, "band_true_node_crop", "0-0.005c"), 0.4322, 0.0005),
    ("area fraction inside 0.005c",
     lambda r: ma_band_mass(r, "area_uniform", "0-0.005c"), 0.00138, 0.00002),
    ("node/area gap inside 0.005c",
     lambda r: ma_band_mass_gap(r, "0-0.005c"), 312.0, 1.0),
    ("node/area gap beyond 0.5c (it inverts)",
     lambda r: ma_band_mass_gap(r, ">0.5c"), 0.14, 0.005),
    ("grid-binned node weight in 0-0.005c (vs the true 0.432)",
     lambda r: ma_band_mass(r, "band_grid_binned_node", "0-0.005c"), 0.098, 0.001),
    # --- 4b: "weight by node count" under five discretisations. None of them
    # leaves the interpolator an aggregate win.
    ("4b: lowest R_p over the five node constructions",
     lambda r: ma_sens_p(r, "min"), 0.431, 0.002),
    ("4b: highest R_p over the five node constructions",
     lambda r: ma_sens_p(r, "max"), 1.61, 0.01),
    ("4b: least adverse node construction on mse_u (x against interp)",
     lambda r: ma_sens_worst(r, "u", "min"), 86.0, 0.5),
    ("4b: most adverse node construction on mse_u (x against interp)",
     lambda r: ma_sens_worst(r, "u", "max"), 551.0, 1.0),
    ("4b: least adverse node construction on mse_v (x against interp)",
     lambda r: ma_sens_worst(r, "v", "min"), 6.3, 0.05),
    ("4b: most adverse node construction on mse_v (x against interp)",
     lambda r: ma_sens_worst(r, "v", "max"), 24.0, 0.6),
    # --- tab:bandratio (rule D2). The localisation thesis, measured on BOTH arms.
    ("D2: band ratio on u, 0-0.005c",
     lambda r: ma_band_ratio(r, "u", "0-0.005c"), 906.2, 0.5),
    ("D2: band ratio on u, 0.005-0.01c",
     lambda r: ma_band_ratio(r, "u", "0.005-0.01c"), 74.1, 0.1),
    ("D2: band ratio on u, 0.01-0.02c",
     lambda r: ma_band_ratio(r, "u", "0.01-0.02c"), 30.4, 0.1),
    ("D2: band ratio on u, 0.02-0.05c",
     lambda r: ma_band_ratio(r, "u", "0.02-0.05c"), 5.33, 0.01),
    ("D2: band ratio on u, >0.5c",
     lambda r: ma_band_ratio(r, "u", ">0.5c"), 0.312, 0.002),
    ("D2: span of the u band ratio",
     ma_band_span, 2905.0, 3.0),
    ("D2: inversions in the u band ratio", ma_band_inversions, 0.0, 0.5),
    ("D2: band ratio on p, 0-0.005c (Transolver wins the first band)",
     lambda r: ma_band_ratio(r, "p", "0-0.005c"), 0.589, 0.002),
    ("D2: interp wins p in 0.15-0.5c by (x)",
     lambda r: 1.0 / ma_band_ratio(r, "p", "0.15-0.5c"), 32.0, 0.5),
    ("D2: interp wins p beyond 0.5c by (x)",
     lambda r: 1.0 / ma_band_ratio(r, "p", ">0.5c"), 237.0, 1.0),
    # Seed-mean shares. The tolerance is 0.3 percentage points rather than 0.1
    # because the manuscript quotes one decimal and the three seeds are averaged.
    ("Transolver's own u error share beyond 0.5c (%)",
     lambda r: ma_transolver_se_share(r, "u"), 71.9, 0.3),
    ("Transolver's own p error share beyond 0.5c (%)",
     lambda r: ma_transolver_se_share(r, "p"), 62.8, 0.3),
    # --- Block D: the representation ceiling. This bounds every field number in
    # sec:interp, for BOTH arms, and is the half of the objection that no
    # re-weighting can reach.
    # NOTE the manuscript quotes 413x pooled, not the 418x that
    # docs/paper/review/measure_asymmetry.md Sec.5 prints. 418 = 277.3/0.664, but
    # 0.664 does not reconstruct from that table's own inputs: node-weighting the
    # seed-mean in-crop band MSEs by n_nodes_in_crop gives 0.6708, hence
    # 277.335/0.6708 = 413.4. This reader recomputes both sides from the artifact,
    # so the manuscript tracks the artifact and not the prose summary.
    ("r128 round-trip error over Transolver's, pooled over the crop (x)",
     ma_ceiling, 413.4, 1.0),
    ("r128 round-trip error over Transolver's, inside 0.005c (x)",
     lambda r: ma_ceiling(r, "0-0.005c"), 495.0, 2.0),
    ("Transolver's per-node u error inside 0.005c",
     ma_node_mse_u_inner, 1.16, 0.01),
    ("r128 round-trip u error at the nodes inside 0.005c",
     ma_ceiling_abs, 572.8, 0.2),
    # --- the wall-band decomposition: the boundary claim.
    ("u error share inside 0.02c", lambda r: band(r, "se_share_u"), 0.924, 0.002),
    ("v error share inside 0.02c", lambda r: band(r, "se_share_v"), 0.898, 0.002),
    ("u error share beyond 0.5c",
     lambda r: band(r, "se_share_u", ">0.5c"), 0.036, 0.002),
    ("worst R^2 beyond 0.05c", lambda r: band_outer_min_r2(r, "r2"), 0.9996, 0.0002),
    ("worst per-case-centred R^2 beyond 0.05c",
     lambda r: band_outer_min_r2(r, "r2_pc"), 0.9965, 0.0002),
    # The domain share the R^2>=0.9996 claim actually covers. NOTE these are two
    # different quantities and the manuscript keeps them apart: 98.7% of cells lie
    # beyond 0.05c (where the R^2 claim holds), while the 0-0.02c wall band that
    # carries 92%/90% of the u/v error is 0.5% of cells. The 0.02-0.05c band lies
    # between them and is covered by neither statement. Do not merge them.
    ("cells beyond 0.05c, where R^2 >= 0.9996 (%)",
     band_outer_cell_frac, 98.7, 0.1),
    ("cells inside the 0-0.02c wall band",
     lambda r: band(r, "cell_frac"), 0.005, 0.0005),
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
    # The sec:splits and sec:covariate rows that stood here were removed by the
    # 2026-09-12 JOCS rebuild together with the sections that quoted them; see the
    # block comment above for the full list and where the text is preserved. The
    # two rows immediately above are KEPT: sec:forces still quotes 0.8389/0.8394
    # as a representation statement (the interpolated field and the exact field
    # are indistinguishable to this integrator), which is not a covariate claim.
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
