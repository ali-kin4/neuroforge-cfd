"""Selective prediction / risk-coverage / AUROC study on the trust scores.

Question answered
-----------------
"How good are the trust scores at DECIDING which cases to reject?" The paper's
published evidence is a residual-vs-error Spearman correlation (backbone
residual_error_spearman 0.625 +/- 0.019; ensemble-mean arm 0.6103). Correlation
is not decision quality: this harness converts the same per-case (score, error)
vectors into

* risk-coverage curves (reject the worst-scored k% of cases, report the mean /
  median rel-L2 SPEED error of the RETAINED cases, against an oracle that
  rejects by TRUE error and a random-rejection baseline),
* AUROC / AUPRC for detecting the top-decile (>= 90th percentile) true-error
  cases from the score alone, and
* the Spearman itself (as a faithfulness gate against the published number).

It also PERSISTS the per-case score/error vectors (results/selective/
selective_percase.json) so a later bootstrap study can attach CIs without
recomputing anything.

Arms (all served from cache -- CPU only, no model forward, no GPU)
------------------------------------------------------------------
a. ENSEMBLE-MEAN  : prediction = cached 5-member ensemble mean field
                    (data/cache/w2/ensemble/<case>.npz, 200 AirfRANS 'full'
                    test cases at resolution 128). Scores per case:
                      - residual  : PhysicsChecker(Config().physics).diagnose(
                                    mean_field, case).residual_norm()  -- the
                                    EXACT scalar evaluate_cases correlates,
                      - sigma_vel : fluid-masked mean of the per-cell ensemble
                                    std averaged over velocity channels 0,1
                                    (sigma_p, channel 2, recorded alongside),
                      - fused     : mean of the two scores' ranks (rankdata).
b. CORRECTED      : per DEQ-corrector seed {0,1,2}, prediction = cached
                    deployed Transolver+DEQ corrected field
                    (data/cache/w2/corrected/seed<k>/<case>.npz); score =
                    residual norm on the corrected field.
c. DEEPCFD        : loaded straight from the existing per-case dump
                    results/deepcfd/deepcfd_trust_percase.json (247 OOD cases
                    x 3 seeds; score = residual_norm, error = rel_l2). No
                    recompute.

Error (arms a, b) -- byte-for-byte the evaluate_cases formula
(src/neuroforge/physics/evaluation.py, lines ~216-222):
    m   = gt.mask > 0.5                      (# _fluid_mask prefers ref.mask)
    err = ||(pred.speed() - gt.speed())[m]|| / (||gt.speed()[m]|| + 1e-12)

FlowField construction for cached arrays mirrors run_w2_conformal_corrected's
MeanPredictor: FlowField.from_array(arr, case.domain, mask=encode_case(case)[1],
sdf=encode_case(case)[0]).

Metrics -- no sklearn (not a dependency)
----------------------------------------
AUROC via the Mann-Whitney U rank statistic (scipy.stats.rankdata, tie-aware);
AUPRC as average precision from the score-sorted precision-recall staircase
(tie groups collapsed). Random rejection baseline is a 1000-draw Monte Carlo
(rng seed 0) of the retained mean/median.

Sanity gates (checked, reported, non-fatal)
-------------------------------------------
1. ensemble-arm residual-vs-error Spearman within 0.6103 +/- 0.02 (published
   ensemble_mean_metrics.residual_error_spearman, computed on ALL 200 cases --
   uq_results.json says n_cases: 200).
2. n cases: 200 per ensemble/corrected arm, 247 per DeepCFD seed.
3. DeepCFD per-seed Spearman recomputed from the JSON arrays reproduces the
   file's own residual_error_spearman_recomputed bit-for-bit, and the file's
   run-time residual_error_spearman_saved rounds to 0.718 / 0.658 / 0.934
   (the two differ ~1e-3 because the arrays are float32-serialized).

Run (CPU, minutes):
    .venv/Scripts/python.exe scripts/run_selective_prediction.py

Outputs: results/selective/selective_prediction.json,
         results/selective/selective_percase.json,
         results/figures/fig_risk_coverage.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets BLAS thread caps)
from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.geometry.encode import encode_case
from neuroforge.physics.residuals import PhysicsChecker

# ---- constants -------------------------------------------------------------- #
PUB_ENSEMBLE_SPEARMAN = 0.6103232580814522  # uq_results.json, n_cases=200
SPEARMAN_TOL = 0.02
DEEPCFD_SPEARMAN_TARGETS = [0.718, 0.658, 0.934]  # rounded, seeds 0/1/2
REJECT_FRACTIONS = [0, 5, 10, 20, 30, 50]  # percent
TOP_ERROR_QUANTILE = 0.9  # "top-decile true-error" positives
N_RANDOM_DRAWS = 1000
RANDOM_SEED = 0


def log(msg: str) -> None:
    print(f"[selective] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Metric primitives (scipy-only, no sklearn)
# --------------------------------------------------------------------------- #
def auroc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (tie-aware midranks).

    AUROC = P(score_pos > score_neg) + 0.5 * P(tie)
          = (sum of positive midranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)
    """
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)  # midranks handle ties correctly
    u = float(r[y].sum()) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auprc_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision from the score-sorted PR staircase, tie groups collapsed.

    AP = sum_k (R_k - R_{k-1}) * P_k over descending-score threshold groups.
    """
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool).astype(np.int64)
    n = s.size
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    ap = 0.0
    tp = 0
    seen = 0
    prev_recall = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        tp += int(y_sorted[i:j].sum())
        seen += j - i
        recall = tp / n_pos
        precision = tp / seen
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(np.asarray(a, np.float64), np.asarray(b, np.float64)).statistic)


def risk_coverage(
    scores: np.ndarray, errors: np.ndarray, rng: np.random.Generator
) -> dict:
    """Reject the worst-k% by score (score DESCENDING = least trusted first).

    Returns, per k in REJECT_FRACTIONS: mean/median rel-L2 of RETAINED cases
    for the score, for an oracle that rejects by TRUE error, and for a random-
    rejection Monte Carlo baseline (mean over N_RANDOM_DRAWS draws).
    """
    s = np.asarray(scores, np.float64)
    e = np.asarray(errors, np.float64)
    n = e.size
    order_score = np.argsort(-s, kind="mergesort")   # highest score rejected first
    order_oracle = np.argsort(-e, kind="mergesort")  # highest true error rejected first
    out = {}
    for k in REJECT_FRACTIONS:
        n_rej = int(round(k / 100.0 * n))
        keep_s = order_score[n_rej:]
        keep_o = order_oracle[n_rej:]
        if n_rej == 0:
            rnd_mean, rnd_median = float(e.mean()), float(np.median(e))
        else:
            means = np.empty(N_RANDOM_DRAWS)
            medians = np.empty(N_RANDOM_DRAWS)
            for d in range(N_RANDOM_DRAWS):
                keep_r = rng.permutation(n)[n_rej:]
                means[d] = e[keep_r].mean()
                medians[d] = np.median(e[keep_r])
            rnd_mean, rnd_median = float(means.mean()), float(medians.mean())
        out[str(k)] = {
            "n_rejected": n_rej,
            "n_retained": n - n_rej,
            "score_mean": float(e[keep_s].mean()),
            "score_median": float(np.median(e[keep_s])),
            "oracle_mean": float(e[keep_o].mean()),
            "oracle_median": float(np.median(e[keep_o])),
            "random_mean": rnd_mean,
            "random_median": rnd_median,
        }
    return out


def arm_metrics(scores: np.ndarray, errors: np.ndarray, rng: np.random.Generator) -> dict:
    """Full decision-quality metric set for one (score, error) pairing."""
    e = np.asarray(errors, np.float64)
    thr = float(np.quantile(e, TOP_ERROR_QUANTILE))
    labels = e >= thr
    return {
        "spearman": spearman(scores, e),
        "auroc_top_decile": auroc_mann_whitney(scores, labels),
        "auprc_top_decile": auprc_average_precision(scores, labels),
        "top_decile_threshold_rel_l2": thr,
        "n_positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "risk_coverage": risk_coverage(scores, e, rng),
    }


# --------------------------------------------------------------------------- #
# Per-case score/error extraction from the caches
# --------------------------------------------------------------------------- #
def rel_l2_speed(pred: FlowField, gt: FlowField) -> float:
    """EXACT evaluate_cases formula (evaluation.py ~216-222): fluid-masked rel-L2 speed."""
    mask = gt.mask if gt.mask is not None else pred.mask
    m = np.asarray(mask) > 0.5
    if not m.any():
        m = np.ones(gt.shape, dtype=bool)
    num = float(np.sqrt(np.sum(((pred.speed() - gt.speed())[m]) ** 2)))
    den = float(np.sqrt(np.sum((gt.speed()[m]) ** 2))) + 1e-12
    return num / den


def compute_cached_arms(args, checker) -> tuple[dict, dict, float]:
    """Walk the 200 AirfRANS test cases; score ensemble-mean + corrected fields."""
    log("loading AirfRANS GT grid pairs (full task, test split, res "
        f"{args.resolution}, limit {args.n_val}) ...")
    pairs = load_airfrans(
        root=args.root, task="full", train=False, resolution=args.resolution,
        limit=args.n_val, cache_dir=args.cache_dir, download=False, progress=False,
    )
    ens_dir = os.path.join(args.w2_cache_dir, "ensemble")
    corr_dirs = {s: os.path.join(args.w2_cache_dir, "corrected", s) for s in args.corrector_seeds}

    ens_rows = []                      # ensemble-mean arm per-case rows
    corr_rows = {s: [] for s in corr_dirs}  # corrected arm per-case rows
    t0 = time.time()
    n_missing = 0
    for i, (case, gt) in enumerate(pairs):
        npz_path = os.path.join(ens_dir, f"{case.name}.npz")
        if not os.path.exists(npz_path):
            n_missing += 1
            continue
        d = np.load(npz_path)
        mean, std = d["mean"], d["std"]
        # FlowField built exactly as run_w2_conformal_corrected.MeanPredictor
        stack = encode_case(case)
        sdf = stack[0].astype(DTYPE)
        mask_geo = stack[1].astype(DTYPE)
        pred = FlowField.from_array(mean, case.domain, mask=mask_geo, sdf=sdf,
                                    meta={"source": "ensemble-mean", "case": case.name})
        diag = checker.diagnose(pred, case)
        residual = diag.residual_norm()
        err = rel_l2_speed(pred, gt)
        # sigma scores: fluid-masked (GT mask, same mask as the error) mean of
        # per-cell ensemble std over velocity channels 0,1; channel-2 recorded.
        m = np.asarray(gt.mask) > 0.5
        sigma_vel = float((0.5 * (std[0] + std[1]))[m].mean())
        sigma_p = float(std[2][m].mean())
        ens_rows.append({"name": case.name, "residual": residual,
                         "sigma_vel": sigma_vel, "sigma_p": sigma_p, "rel_l2": err})

        for s, cdir in corr_dirs.items():
            cpath = os.path.join(cdir, f"{case.name}.npz")
            if not os.path.exists(cpath):
                continue
            corrected = np.load(cpath)["corrected"]
            cpred = FlowField.from_array(corrected, case.domain, mask=mask_geo, sdf=sdf,
                                         meta={"source": f"corrected-{s}", "case": case.name})
            cdiag = checker.diagnose(cpred, case)
            corr_rows[s].append({"name": case.name,
                                 "residual": cdiag.residual_norm(),
                                 "rel_l2": rel_l2_speed(cpred, gt)})
        if (i + 1) % 50 == 0:
            log(f"  scored {i + 1}/{len(pairs)} cases "
                f"({(time.time() - t0) / (i + 1):.2f}s/case)")
    dt = time.time() - t0
    log(f"cached-arm scoring done: {len(ens_rows)} ensemble cases, "
        + ", ".join(f"{s}:{len(r)}" for s, r in corr_rows.items())
        + f" corrected cases, {n_missing} missing, {dt:.1f}s total")
    return {"rows": ens_rows}, corr_rows, dt


def load_deepcfd(path: str) -> list[dict]:
    """Load the existing DeepCFD per-case dump (no recompute)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["per_seed"]


# --------------------------------------------------------------------------- #
# Figure (matplotlib, static, light-surface paper figure)
# --------------------------------------------------------------------------- #
# Categorical slots (validated reference palette, light mode) + chrome ink.
C_RESIDUAL = "#2a78d6"   # slot 1 blue  -- residual score
C_SIGMA = "#eb6834"      # slot 2 orange -- ensemble sigma score
C_FUSED = "#1baf7a"      # slot 3 aqua  -- fused (rank-average) score
C_ORACLE = "#52514e"     # secondary ink -- oracle bound (not a series)
C_RANDOM = "#898781"     # muted ink    -- random baseline (not a series)
C_SURFACE = "#fcfcfb"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_INK = "#0b0b0b"
C_MUTED = "#898781"


def make_figure(results: dict, fig_base: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": C_AXIS,
        "axes.labelcolor": C_INK,
        "text.color": C_INK,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "axes.linewidth": 0.8,
        "figure.facecolor": C_SURFACE,
        "axes.facecolor": C_SURFACE,
        "savefig.facecolor": C_SURFACE,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # ---- Panel A: risk-coverage, ensemble arm (mean rel-L2 of retained) ----
    ens = results["arms"]["ensemble_mean"]
    ks = [int(k) for k in REJECT_FRACTIONS]

    def curve(score_key, field):
        rc = ens["scores"][score_key]["risk_coverage"]
        return [rc[str(k)][field] for k in ks]

    rc_res = ens["scores"]["residual"]["risk_coverage"]
    oracle = [rc_res[str(k)]["oracle_mean"] for k in ks]
    random_ = [rc_res[str(k)]["random_mean"] for k in ks]

    ax1.plot(ks, random_, color=C_RANDOM, lw=1.4, ls=":", label="random", zorder=2)
    ax1.plot(ks, oracle, color=C_ORACLE, lw=1.4, ls="--", label="oracle", zorder=2)
    ax1.plot(ks, curve("residual", "score_mean"), color=C_RESIDUAL, lw=2,
             marker="o", ms=4, label="residual", zorder=3)
    ax1.plot(ks, curve("sigma_vel", "score_mean"), color=C_SIGMA, lw=2,
             marker="s", ms=4, label="ensemble $\\sigma$", zorder=3)
    ax1.plot(ks, curve("fused", "score_mean"), color=C_FUSED, lw=2,
             marker="^", ms=4, label="fused (rank avg)", zorder=3)
    ax1.set_xlabel("cases rejected by score (%)")
    ax1.set_ylabel("mean rel-$L_2$ speed error (retained)")
    ax1.set_title("Risk–coverage — ensemble-mean arm (n=200)", fontsize=9.5)
    ax1.set_xticks(ks)
    ax1.grid(axis="y", color=C_GRID, lw=0.6)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")

    # ---- Panel B: AUROC summary bars across arms ----
    bars = [
        ("Ens.\nresidual", ens["scores"]["residual"]["auroc_top_decile"], C_RESIDUAL),
        ("Ens.\n$\\sigma$", ens["scores"]["sigma_vel"]["auroc_top_decile"], C_SIGMA),
        ("Ens.\nfused", ens["scores"]["fused"]["auroc_top_decile"], C_FUSED),
    ]
    for s in ("seed0", "seed1", "seed2"):
        bars.append((f"Corr.\n{s[-1]}",
                     results["arms"][f"corrected_{s}"]["scores"]["residual"]["auroc_top_decile"],
                     C_RESIDUAL))
    for s in (0, 1, 2):
        bars.append((f"DCFD\n{s}",
                     results["arms"][f"deepcfd_seed{s}"]["scores"]["residual"]["auroc_top_decile"],
                     C_RESIDUAL))
    xs = np.arange(len(bars))
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    rects = ax2.bar(xs, vals, width=0.62, color=cols, edgecolor="none", zorder=3)
    # 2px-equivalent surface gap comes from width<1; round the data end slightly
    # via thin bars is not native in mpl -- keep flat ends, label values directly.
    for r, v in zip(rects, vals):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7.5, color=C_INK)
    ax2.axhline(0.5, color=C_MUTED, lw=1.0, ls=":", zorder=2)
    ax2.text(len(bars) - 0.5, 0.505, "chance", ha="right", va="bottom",
             fontsize=7.5, color=C_MUTED)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([b[0] for b in bars], fontsize=7.5)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("AUROC (top-decile error detection)")
    ax2.set_title("Decision quality across arms", fontsize=9.5)
    ax2.grid(axis="y", color=C_GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_base + ".png", dpi=200)
    fig.savefig(fig_base + ".pdf")
    plt.close(fig)
    log(f"wrote {fig_base}.png / .pdf")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Selective-prediction study on trust scores.")
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--w2-cache-dir", default="data/cache/w2")
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--corrector-seeds", nargs="+", default=["seed0", "seed1", "seed2"])
    p.add_argument("--deepcfd-json", default="results/deepcfd/deepcfd_trust_percase.json")
    p.add_argument("--out-dir", default="results/selective")
    p.add_argument("--fig-dir", default="results/figures")
    args = p.parse_args(argv)

    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    checker = PhysicsChecker(Config().physics)

    # ---- per-case vectors ----
    ens, corr, dt_score = compute_cached_arms(args, checker)
    deepcfd = load_deepcfd(args.deepcfd_json)

    # fused score = mean of the two scores' ranks (residual, sigma_vel)
    ens_rows = ens["rows"]
    res_v = np.array([r["residual"] for r in ens_rows])
    sig_v = np.array([r["sigma_vel"] for r in ens_rows])
    err_v = np.array([r["rel_l2"] for r in ens_rows])
    fused_v = 0.5 * (rankdata(res_v) + rankdata(sig_v))
    for r, f in zip(ens_rows, fused_v):
        r["fused"] = float(f)

    rng = np.random.default_rng(RANDOM_SEED)

    # ---- metrics per arm/score ----
    arms: dict = {}
    arms["ensemble_mean"] = {
        "n_cases": len(ens_rows),
        "scores": {
            "residual": arm_metrics(res_v, err_v, rng),
            "sigma_vel": arm_metrics(sig_v, err_v, rng),
            "sigma_p": arm_metrics(np.array([r["sigma_p"] for r in ens_rows]), err_v, rng),
            "fused": arm_metrics(fused_v, err_v, rng),
        },
    }
    for s, rows in corr.items():
        cr = np.array([r["residual"] for r in rows])
        ce = np.array([r["rel_l2"] for r in rows])
        arms[f"corrected_{s}"] = {
            "n_cases": len(rows),
            "scores": {"residual": arm_metrics(cr, ce, rng)},
        }
    deepcfd_gate = []
    for entry in deepcfd:
        s = entry["seed"]
        dr = np.asarray(entry["residual_norms"], np.float64)
        de = np.asarray(entry["rel_l2_errors"], np.float64)
        arms[f"deepcfd_seed{s}"] = {
            "n_cases": int(dr.size),
            "scores": {"residual": arm_metrics(dr, de, rng)},
        }
        rho = spearman(dr, de)
        json_recomputed = entry.get("residual_error_spearman_recomputed")
        json_saved = entry.get("residual_error_spearman_saved")
        deepcfd_gate.append({
            "seed": s,
            "spearman_recomputed_here": rho,
            "spearman_recomputed_in_json": json_recomputed,
            "spearman_saved_in_json": json_saved,
            "target_rounded": DEEPCFD_SPEARMAN_TARGETS[s],
            # exact reproduction of the JSON's own recomputed-from-arrays value
            # (the arrays are float32-serialized, so 'recomputed' differs from
            # the run-time 'saved' headline by ~1e-3; the JSON documents both).
            "matches_json_recomputed_exactly": bool(
                json_recomputed is not None and abs(rho - json_recomputed) < 1e-12
            ),
            # published headline 0.718/0.658/0.934 = the 'saved' values rounded
            "saved_matches_target_3dp": bool(
                json_saved is not None
                and round(json_saved, 3) == DEEPCFD_SPEARMAN_TARGETS[s]
            ),
        })

    # ---- sanity gates ----
    ens_rho = arms["ensemble_mean"]["scores"]["residual"]["spearman"]
    gates = {
        "ensemble_spearman": {
            "measured": ens_rho,
            "published": PUB_ENSEMBLE_SPEARMAN,
            "tolerance": SPEARMAN_TOL,
            "published_computed_on": "all 200 cached cases (uq_results.json "
                                     "ensemble_mean_metrics.n_cases = 200)",
            "passed": bool(abs(ens_rho - PUB_ENSEMBLE_SPEARMAN) <= SPEARMAN_TOL),
        },
        "n_cases": {
            "ensemble": arms["ensemble_mean"]["n_cases"],
            "corrected": {s: arms[f"corrected_{s}"]["n_cases"] for s in corr},
            "deepcfd": {f"seed{e['seed']}": arms[f"deepcfd_seed{e['seed']}"]["n_cases"]
                        for e in deepcfd},
            "passed": bool(
                arms["ensemble_mean"]["n_cases"] == 200
                and all(arms[f"corrected_{s}"]["n_cases"] == 200 for s in corr)
                and all(arms[f"deepcfd_seed{e['seed']}"]["n_cases"] == 247 for e in deepcfd)
            ),
        },
        "deepcfd_spearman": {
            "per_seed": deepcfd_gate,
            "note": "gate = (a) our Spearman from the JSON arrays matches the "
                    "JSON's own residual_error_spearman_recomputed bit-for-bit, "
                    "AND (b) the JSON's residual_error_spearman_saved (run-time "
                    "headline) rounds to the published 0.718/0.658/0.934. The "
                    "recomputed-vs-saved ~1e-3 delta is float32 array "
                    "serialization, documented in the dump itself.",
            "passed": bool(all(
                g["matches_json_recomputed_exactly"] and g["saved_matches_target_3dp"]
                for g in deepcfd_gate)),
        },
    }
    gates["all_passed"] = bool(all(
        g["passed"] for g in (gates["ensemble_spearman"], gates["n_cases"],
                              gates["deepcfd_spearman"])))

    runtime_sec = time.time() - t_start
    meta = {
        "experiment": "selective prediction / risk-coverage / AUROC on trust scores",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": runtime_sec,
        "scoring_pass_sec": dt_score,
        "device": "cpu (all model outputs served from cache; zero forward passes)",
        "sources": {
            "ensemble_cache": os.path.join(args.w2_cache_dir, "ensemble"),
            "corrected_cache": {s: os.path.join(args.w2_cache_dir, "corrected", s)
                                for s in args.corrector_seeds},
            "deepcfd_percase": args.deepcfd_json,
            "gt": f"AirfRANS full task, test split, resolution {args.resolution}, "
                  f"limit {args.n_val}, root {args.root}, cache {args.cache_dir}",
        },
        "formulas": {
            "residual_score": "PhysicsChecker(Config().physics).diagnose(pred, case)"
                              ".residual_norm() -- interior continuity+momentum RMS "
                              "of L2 magnitude, solid+wall-ring zeroed (Diagnostics."
                              "residual_norm, core/types.py ~314). Identical to the "
                              "scalar behind the published residual_error_spearman.",
            "rel_l2_error": "||(pred.speed()-gt.speed())[m]|| / (||gt.speed()[m]||"
                            "+1e-12), m = gt.mask>0.5 -- byte-for-byte evaluation.py "
                            "evaluate_cases lines ~216-222.",
            "sigma_vel_score": "fluid-masked (gt.mask>0.5, same mask as the error) "
                               "mean of 0.5*(std[0]+std[1]); std = frozen 5-member "
                               "ensemble per-cell std (ddof=0) on physical fields "
                               "from data/cache/w2/ensemble. sigma_p = same for "
                               "channel 2 (pressure), recorded alongside.",
            "fused_score": "0.5*(rankdata(residual)+rankdata(sigma_vel)) -- average "
                           "of the two scores' ranks.",
            "flowfield_construction": "FlowField.from_array(cached_arr, case.domain, "
                                      "mask=encode_case(case)[1], sdf=encode_case("
                                      "case)[0]) -- mirrors run_w2_conformal_"
                                      "corrected.MeanPredictor.predict.",
            "risk_coverage": "sort by score DESCENDING (highest score = least "
                             "trusted), reject worst k% (k in "
                             f"{REJECT_FRACTIONS}), report mean+median rel-L2 of "
                             "retained; oracle rejects by TRUE error; random = "
                             f"{N_RANDOM_DRAWS}-draw MC (rng seed {RANDOM_SEED}).",
            "auroc": "Mann-Whitney U from scipy.stats.rankdata midranks: "
                     "(sum pos ranks - n_pos(n_pos+1)/2)/(n_pos*n_neg). Positives "
                     f"= cases with rel_l2 >= {TOP_ERROR_QUANTILE:.0%} quantile "
                     "(top decile) within the arm.",
            "auprc": "average precision over the descending-score PR staircase, "
                     "tie groups collapsed; baseline = prevalence (~0.10).",
        },
    }

    results = {"meta": meta, "gates": gates, "arms": arms}
    out_path = os.path.join(args.out_dir, "selective_prediction.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {out_path}")

    # ---- per-case dump (feeds the later bootstrap study) ----
    percase = {
        "meta": {
            "description": "Per-case trust scores + rel-L2 SPEED errors for the "
                           "selective-prediction study (bootstrap-CI input).",
            "formulas": "see selective_prediction.json meta.formulas",
            "date": meta["date"],
        },
        "ensemble_mean": ens_rows,
        **{f"corrected_{s}": rows for s, rows in corr.items()},
        **{f"deepcfd_seed{e['seed']}": [
            {"index": i, "residual": float(r), "rel_l2": float(l)}
            for i, (r, l) in enumerate(zip(e["residual_norms"], e["rel_l2_errors"]))
        ] for e in deepcfd},
    }
    percase_path = os.path.join(args.out_dir, "selective_percase.json")
    with open(percase_path, "w", encoding="utf-8") as f:
        json.dump(percase, f, indent=2)
    log(f"wrote {percase_path}")

    # ---- figure ----
    make_figure(results, os.path.join(args.fig_dir, "fig_risk_coverage"))

    # ---- console summary ----
    log("================= SUMMARY =================")
    log(f"gates: ensemble_spearman={gates['ensemble_spearman']['passed']} "
        f"(measured {ens_rho:.4f} vs pub {PUB_ENSEMBLE_SPEARMAN:.4f}), "
        f"n_cases={gates['n_cases']['passed']}, "
        f"deepcfd_spearman={gates['deepcfd_spearman']['passed']} -> "
        f"ALL={'PASS' if gates['all_passed'] else 'FAIL'}")
    log("AUROC / AUPRC (top-decile error detection):")
    for arm_name, arm in arms.items():
        for score_name, mtr in arm["scores"].items():
            log(f"  {arm_name:>18s} / {score_name:<9s}: AUROC={mtr['auroc_top_decile']:.3f} "
                f"AUPRC={mtr['auprc_top_decile']:.3f} rho={mtr['spearman']:.3f}")
    log("risk-coverage, ensemble arm, mean rel-L2 retained (score/oracle/random):")
    rc = arms["ensemble_mean"]["scores"]["residual"]["risk_coverage"]
    for k in ("10", "20", "30"):
        v = rc[k]
        log(f"  reject {k:>2s}%: residual {v['score_mean']:.4f} | "
            f"oracle {v['oracle_mean']:.4f} | random {v['random_mean']:.4f}")
    log(f"total runtime {runtime_sec:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
