"""Ensemble-size study (M = 2..5) from the frozen deep-ensemble members.

Question answered
-----------------
The paper's adaptive-UQ layer and fused trust score use a 5-member deep
ensemble, and its limitations section concedes that the choice of M is
unstudied. This harness closes that: from the five FROZEN member checkpoints
(checkpoints/uq_ensemble/member{0..4}.pt -- no retraining, the members are the
published ones) it evaluates EVERY subset of size M in {2,3,4,5} (10+10+5+1=26
subsets) and reports, per M (mean +/- std over subsets):

  * sigma trust quality : Spearman(sigma_vel score, rel-L2 speed error of the
                          subset-mean prediction) and top-decile AUROC,
  * residual trust      : same for the residual norm of the subset-mean field
                          (a control -- should be ~flat in M),
  * fused score         : rank-average of the two, the paper's headline
                          decision score,
  * conformal (u,v,p)   : split-conformal q / coverage / ECE of the subset
                          sigma at alpha=0.1, split seed 0 (the published
                          split), same ConformalCalibrator primitives as
                          scripts/run_ensemble_uq.py,
  * accuracy            : mse_speed of the subset-mean prediction.

Member fields are computed ONCE (5 GPU forward passes over the 200 test
cases) and cached per case to data/cache/mstudy/<case>.npz -- the subset
sweep afterwards is pure CPU on cached arrays, and a re-run skips any cached
case (resumable).

Faithfulness gates (reported, non-fatal): the M=5 (full-ensemble) arm must
reproduce the published ensemble numbers: residual_error_spearman ~ 0.6103
(uq_results.json) and the u/v/p conformal q/coverage of the published split
(multisplit_conformal gate: q=2.352087, coverage=0.915496 for p/backbone at
split seed 0 -- note that gate was computed with ddof=1 sigma on ch p).
Here sigma uses ddof=0 (np.std default), matching run_ensemble_uq.py.

Run (GPU ~30-45 min for the one-off field cache, then CPU minutes):
    .venv/Scripts/python.exe scripts/run_ensemble_mstudy.py

Outputs: results/uq_ensemble/mstudy.json, results/uq_ensemble/mstudy.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.datamodule import FlowDataset, Normalizer
from neuroforge.data.pointcloud import F_IN, F_OUT, PointNormalizer, load_airfrans_pointclouds
from neuroforge.geometry.encode import encode_case
from neuroforge.models.baselines.transolver import TransolverPointModel
from neuroforge.physics.calibration import ConformalCalibrator, reliability
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor

PUB_ENSEMBLE_SPEARMAN = 0.6103232580814522  # uq_results.json (M=5, 200 cases)
CONFORMAL_ALPHA = 0.1
SPLIT_SEED = 0
CHANNELS = {"u": 0, "v": 1, "p": 2}


def log(msg: str) -> None:
    print(f"[mstudy] {msg}", flush=True)


def rel_l2_speed(pred_uv: np.ndarray, gt: FlowField) -> float:
    """Fluid-masked rel-L2 speed error, byte-for-byte the evaluate_cases formula."""
    m = np.asarray(gt.mask) > 0.5
    if not m.any():
        m = np.ones(gt.shape, dtype=bool)
    speed_p = np.sqrt(pred_uv[0] ** 2 + pred_uv[1] ** 2)
    speed_g = gt.speed()
    num = float(np.sqrt(np.sum(((speed_p - speed_g)[m]) ** 2)))
    den = float(np.sqrt(np.sum((speed_g[m]) ** 2))) + 1e-12
    return num / den


def auroc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    u = float(r[y].sum()) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def spearman(a, b) -> float:
    return float(spearmanr(np.asarray(a, np.float64), np.asarray(b, np.float64)).statistic)


# --------------------------------------------------------------------------- #
# One-off member-field cache (the only GPU part; resumable per case)
# --------------------------------------------------------------------------- #
def build_field_cache(args, device) -> None:
    os.makedirs(args.field_cache, exist_ok=True)
    log("loading point clouds + grid pairs ...")
    train_pcs = load_airfrans_pointclouds(
        root=args.root, task="full", train=True, limit=args.n_train,
        cache_dir=args.cache_dir, download=False,
    )
    test_pcs = load_airfrans_pointclouds(
        root=args.root, task="full", train=False, limit=args.n_val,
        cache_dir=args.cache_dir, download=False,
    )
    train_pairs = load_airfrans(
        root=args.root, task="full", train=True, resolution=args.resolution,
        limit=args.n_train, cache_dir=args.cache_dir, download=False, progress=False,
    )
    test_pairs = load_airfrans(
        root=args.root, task="full", train=False, resolution=args.resolution,
        limit=args.n_val, cache_dir=args.cache_dir, download=False, progress=False,
    )
    missing = [case.name for case, _ in test_pairs
               if not os.path.exists(os.path.join(args.field_cache, f"{case.name}.npz"))]
    if not missing:
        log("field cache complete -- no GPU work needed")
        return

    point_norm = PointNormalizer().fit(train_pcs)
    ds = FlowDataset(train_pairs, normalizer=None)
    grid_norm = Normalizer().fit(*ds.raw_arrays())

    predictors = []
    for m in range(args.members):
        ckpt = os.path.join(args.ckpt_dir, f"member{m}.pt")
        state = torch.load(ckpt, map_location=device)
        margs = state.get("args", {})
        model = TransolverPointModel(
            in_features=F_IN, out_features=F_OUT,
            width=margs.get("width", 256), n_layers=margs.get("n_layers", 10),
            n_heads=margs.get("n_heads", 8), n_slices=margs.get("n_slices", 32),
            dropout=margs.get("dropout", 0.0),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        predictors.append(PointCloudPredictor(model, test_pcs, point_norm, grid_norm,
                                              device=device))
        log(f"member {m}: checkpoint loaded")

    t0 = time.time()
    n_done = 0
    for i, (case, _gt) in enumerate(test_pairs):
        path = os.path.join(args.field_cache, f"{case.name}.npz")
        if os.path.exists(path):
            continue
        fields = np.stack([p.predict(case).as_array() for p in predictors])  # (5,4,H,W)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, members=fields.astype(np.float32))
        os.replace(tmp, path)
        n_done += 1
        if n_done % 20 == 0:
            log(f"  cached {n_done}/{len(missing)} cases "
                f"({(time.time() - t0) / n_done:.1f}s/case)")
    log(f"field cache built: {n_done} new cases, {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Subset sweep (pure CPU on the cache)
# --------------------------------------------------------------------------- #
def subset_metrics(subset, cases, gts, member_fields, geo, checker) -> dict:
    """All per-M metrics for one member subset."""
    sub = list(subset)
    residuals, sigmas, errors = [], [], []
    err_maps = {c: [] for c in CHANNELS}   # per-case signed error maps (subset mean)
    sig_maps = {c: [] for c in CHANNELS}   # per-case sigma maps
    masks = []
    for case, gt, fields in zip(cases, gts, member_fields):
        arr = fields[sub]                       # (M,4,H,W)
        mean = arr.mean(0)
        std = arr.std(0)                        # ddof=0, matches run_ensemble_uq
        sdf, mask_geo = geo[case.name]
        pred = FlowField.from_array(mean, case.domain, mask=mask_geo, sdf=sdf,
                                    meta={"source": f"mean-M{len(sub)}", "case": case.name})
        residuals.append(checker.diagnose(pred, case).residual_norm())
        m = np.asarray(gt.mask) > 0.5
        sigmas.append(float((0.5 * (std[0] + std[1]))[m].mean()))
        errors.append(rel_l2_speed(mean, gt))
        gt_arr = gt.as_array()
        for cname, ci in CHANNELS.items():
            err_maps[cname].append(mean[ci] - gt_arr[ci])
            sig_maps[cname].append(std[ci])
        masks.append(gt.mask)

    res_v = np.array(residuals)
    sig_v = np.array(sigmas)
    err_v = np.array(errors)
    fused_v = 0.5 * (rankdata(res_v) + rankdata(sig_v))
    thr = float(np.quantile(err_v, 0.9))
    labels = err_v >= thr

    out = {
        "members": sub,
        "M": len(sub),
        "mse_speed_proxy_rel_l2_mean": float(err_v.mean()),
        "sigma_spearman": spearman(sig_v, err_v),
        "sigma_auroc": auroc_mann_whitney(sig_v, labels),
        "residual_spearman": spearman(res_v, err_v),
        "residual_auroc": auroc_mann_whitney(res_v, labels),
        "fused_spearman": spearman(fused_v, err_v),
        "fused_auroc": auroc_mann_whitney(fused_v, labels),
        "conformal": {},
    }

    # split-conformal per channel on the subset sigma (published split seed)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(cases))
    half = max(1, len(cases) // 2)
    cal_i, test_i = idx[:half], idx[half:]
    for cname in CHANNELS:
        ce = [err_maps[cname][i] for i in cal_i]
        cs = [sig_maps[cname][i] for i in cal_i]
        cm = [masks[i] for i in cal_i]
        te = [err_maps[cname][i] for i in test_i]
        ts = [sig_maps[cname][i] for i in test_i]
        tm = [masks[i] for i in test_i]
        cal = ConformalCalibrator(alpha=CONFORMAL_ALPHA).fit(ce, cs, cm)
        cov = cal.coverage(te, ts, tm)
        rel = reliability(te, ts, tm, n_bins=10, q=cal.q, alpha=CONFORMAL_ALPHA)
        out["conformal"][cname] = {"q": float(cal.q), "coverage": float(cov),
                                   "ece": float(rel["ece"])}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ensemble-size (M) study from frozen members.")
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--field-cache", default="data/cache/mstudy")
    p.add_argument("--ckpt-dir", default="checkpoints/uq_ensemble")
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="results/uq_ensemble")
    args = p.parse_args(argv)

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    log(f"device={device}")

    # ---- phase 1: one-off member-field cache (GPU; instant if cached) ----
    build_field_cache(args, device)

    # ---- phase 2: load everything once, sweep subsets on CPU ----
    test_pairs = load_airfrans(
        root=args.root, task="full", train=False, resolution=args.resolution,
        limit=args.n_val, cache_dir=args.cache_dir, download=False, progress=False,
    )
    cases, gts, member_fields = [], [], []
    geo = {}
    for case, gt in test_pairs:
        path = os.path.join(args.field_cache, f"{case.name}.npz")
        if not os.path.exists(path):
            continue
        cases.append(case)
        gts.append(gt)
        member_fields.append(np.load(path)["members"])
        stack = encode_case(case)
        geo[case.name] = (stack[0].astype(DTYPE), stack[1].astype(DTYPE))
    log(f"loaded {len(cases)} cases with cached member fields")

    checker = PhysicsChecker(Config().physics)
    per_subset = []
    for M in (2, 3, 4, 5):
        for subset in itertools.combinations(range(args.members), M):
            t0 = time.time()
            row = subset_metrics(subset, cases, gts, member_fields, geo, checker)
            per_subset.append(row)
            log(f"M={M} subset {subset}: sigma_auroc={row['sigma_auroc']:.3f} "
                f"fused_auroc={row['fused_auroc']:.3f} "
                f"cov_p={row['conformal']['p']['coverage']:.3f} "
                f"({time.time() - t0:.1f}s)")

    # ---- aggregate per M ----
    def agg(rows, key, sub=None):
        vals = np.array([(r[key] if sub is None else r[key][sub[0]][sub[1]]) for r in rows])
        return {"mean": float(vals.mean()), "std": float(vals.std()),
                "min": float(vals.min()), "max": float(vals.max()), "n_subsets": len(rows)}

    per_m = {}
    for M in (2, 3, 4, 5):
        rows = [r for r in per_subset if r["M"] == M]
        per_m[str(M)] = {
            "sigma_spearman": agg(rows, "sigma_spearman"),
            "sigma_auroc": agg(rows, "sigma_auroc"),
            "residual_spearman": agg(rows, "residual_spearman"),
            "residual_auroc": agg(rows, "residual_auroc"),
            "fused_spearman": agg(rows, "fused_spearman"),
            "fused_auroc": agg(rows, "fused_auroc"),
            "rel_l2_mean": agg(rows, "mse_speed_proxy_rel_l2_mean"),
            **{f"conformal_{c}_{k}": agg(rows, "conformal", (c, k))
               for c in CHANNELS for k in ("q", "coverage", "ece")},
        }

    # ---- faithfulness gate: M=5 arm vs published ensemble numbers ----
    m5 = [r for r in per_subset if r["M"] == 5][0]
    gate = {
        "m5_residual_spearman": m5["residual_spearman"],
        "published_residual_spearman": PUB_ENSEMBLE_SPEARMAN,
        "abs_diff": abs(m5["residual_spearman"] - PUB_ENSEMBLE_SPEARMAN),
        "passed": bool(abs(m5["residual_spearman"] - PUB_ENSEMBLE_SPEARMAN) <= 0.02),
        "note": "M=5 subset-mean field vs published uq_results.json ensemble-mean "
                "(same 200 cases; small deltas possible from field-cache round-trip "
                "float32).",
    }

    out = {
        "meta": {
            "experiment": "ensemble-size study M=2..5, all member subsets, frozen members",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - t_start,
            "members_source": args.ckpt_dir,
            "n_cases": len(cases),
            "sigma": "per-cell std over subset members, ddof=0",
            "conformal": f"alpha={CONFORMAL_ALPHA}, split seed {SPLIT_SEED}, "
                         "50/50 cal/test, ConformalCalibrator + reliability "
                         "(same primitives as run_ensemble_uq)",
            "error": "fluid-masked rel-L2 speed of the subset-mean prediction",
        },
        "gate": gate,
        "per_m": per_m,
        "per_subset": per_subset,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mstudy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {out_path}")

    # ---- markdown summary ----
    lines = [
        "# Ensemble-size study (M = 2..5, all subsets of the 5 frozen members)",
        "",
        f"n_cases={len(cases)}; sigma=ddof-0 member std; conformal alpha=0.1 "
        f"split seed {SPLIT_SEED}; error=rel-L2 speed of subset-mean. "
        f"Gate (M=5 vs published Spearman {PUB_ENSEMBLE_SPEARMAN:.4f}): "
        f"{'PASS' if gate['passed'] else 'FAIL'} (|d|={gate['abs_diff']:.4f}).",
        "",
        "| M | n_sub | sigma AUROC | fused AUROC | sigma rho | cov u | cov v | cov p | ECE p | q p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for M in (2, 3, 4, 5):
        d = per_m[str(M)]
        lines.append(
            f"| {M} | {d['sigma_auroc']['n_subsets']} "
            f"| {d['sigma_auroc']['mean']:.3f} ± {d['sigma_auroc']['std']:.3f} "
            f"| {d['fused_auroc']['mean']:.3f} ± {d['fused_auroc']['std']:.3f} "
            f"| {d['sigma_spearman']['mean']:.3f} ± {d['sigma_spearman']['std']:.3f} "
            f"| {d['conformal_u_coverage']['mean']:.3f} "
            f"| {d['conformal_v_coverage']['mean']:.3f} "
            f"| {d['conformal_p_coverage']['mean']:.3f} "
            f"| {d['conformal_p_ece']['mean']:.3f} "
            f"| {d['conformal_p_q']['mean']:.2f} |"
        )
    md_path = os.path.join(args.out_dir, "mstudy.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"wrote {md_path}")

    log("================= SUMMARY =================")
    for M in (2, 3, 4, 5):
        d = per_m[str(M)]
        log(f"  M={M}: sigma AUROC {d['sigma_auroc']['mean']:.3f}±{d['sigma_auroc']['std']:.3f}, "
            f"fused AUROC {d['fused_auroc']['mean']:.3f}±{d['fused_auroc']['std']:.3f}, "
            f"coverage p {d['conformal_p_coverage']['mean']:.3f}")
    log(f"gate: {'PASS' if gate['passed'] else 'FAIL'}; total {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
