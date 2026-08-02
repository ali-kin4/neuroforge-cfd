"""Force-vs-official recompute with the repaired integrator schemes.

Extends ``recompute_force_vs_official.py`` (whose loaders/caches it reuses):
one forward pass per seed, then ALL force-estimator schemes are applied to
each predicted field:

    legacy  force_coefficients defaults (offset / field_abs / 1.5 cells)
    wall    force_coefficients(p_scheme="extrap", tau_scheme="field_signed",
                               d1_cells=1.0)  -- wall-extrapolated pressure
    cv      force_coefficients_cv             -- control-volume momentum balance

Scheme selection is justified by the GT-field design study
(``scripts/design_force_integrator.py`` -> results/control/integrator_design.json):
on ground-truth fields vs official labels, "wall" cuts the median lift
magnitude error 0.16 -> 0.05 while preserving ranking, and "cv" recovers
lift almost exactly (rho_L ~ 1.00) and cuts the median drag magnitude error
~3.3 -> ~0.23, at the cost of a weaker drag ranking (rho_D 0.61 vs 0.84).

Run (GPU, ~25 min for 3 seeds x 200 cases):
    .venv/Scripts/python.exe scripts/recompute_force_multischeme.py
Writes results/control/force_vs_official_multischeme.json. Resumable: per-seed
per-scheme coefficient caches under results/control/_cache/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_force_vs_official as base  # noqa: E402

from neuroforge.data.airfrans_loader import load_airfrans  # noqa: E402
from neuroforge.data.pointcloud import load_airfrans_pointclouds  # noqa: E402
from neuroforge.physics.evaluation import coefficient_metrics  # noqa: E402
from neuroforge.physics.metrics import (  # noqa: E402
    force_coefficients,
    force_coefficients_cv,
)
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor  # noqa: E402


def log(msg: str) -> None:
    print(f"[fvo-ms] {msg}", flush=True)


SCHEMES = {
    "legacy": lambda f, c: force_coefficients(f, c),
    "wall": lambda f, c: force_coefficients(
        f, c, p_scheme="extrap", tau_scheme="field_signed", d1_cells=1.0),
    "cv": lambda f, c: force_coefficients_cv(f, c),
}


def coeffs_all_schemes(field, case) -> dict[str, dict]:
    return {s: {"cl": float(fn(field, case)["cl"]),
                "cd": float(fn(field, case)["cd"]),
                "name": case.name}
            for s, fn in SCHEMES.items()}


def metrics_vs(coeff_list: list[dict], target_by_name: dict) -> dict:
    tgt = [target_by_name[c["name"]] for c in coeff_list]
    m = coefficient_metrics(coeff_list, tgt)
    return {
        "rho_D": float(m["rho_cd"]),
        "rho_L": float(m["rho_cl"]),
        "cd_rel_err_mean": float(m["cd_rel_err_mean"]),
        "cl_rel_err_mean": float(m["cl_rel_err_mean"]),
        "cd_rel_err_median": float(np.median(
            [abs(c["cd"] - t["cd"]) / max(abs(t["cd"]), 1e-12)
             for c, t in zip(coeff_list, tgt)])),
        "cl_rel_err_median": float(np.median(
            [abs(c["cl"] - t["cl"]) / max(abs(t["cl"]), 1e-12)
             for c, t in zip(coeff_list, tgt)])),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="full")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--ckpt-dir", default="checkpoints/v2_transolver")
    p.add_argument("--out-dir", default="results/control")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-chunk", type=int, default=0)
    a = p.parse_args(argv)

    device = base._resolve_device(a.device)
    eval_chunk = a.eval_chunk if a.eval_chunk and a.eval_chunk > 0 else None
    cache_sub = os.path.join(a.out_dir, "_cache")
    os.makedirs(cache_sub, exist_ok=True)
    log(f"device={device}  seeds={a.seeds}  n_val={a.n_val}")

    data_root = base._resolve_data_root(a.root)
    if data_root is None:
        log("FATAL: AirfRANS data root not found")
        return 2

    test_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=False, limit=a.n_val,
        cache_dir=a.cache_dir, download=False)
    test_pairs = load_airfrans(
        root=a.root, task=a.task, train=False, resolution=a.resolution,
        limit=a.n_val, cache_dir=a.cache_dir, download=False, progress=False)
    names = [case.name for case, _ in test_pairs]

    label_cache = os.path.join(
        cache_sub, f"official_labels_{a.task}_test_n{a.n_val}.json")
    official = base.official_labels(names, data_root, label_cache)
    if any(nm not in official for nm in names):
        log("FATAL: missing official labels")
        return 3

    # ---- GT arm, per scheme (CPU, seed-independent) ----
    gt_cache = os.path.join(
        cache_sub, f"gt_multischeme_{a.task}_test_r{a.resolution}_n{a.n_val}.json")
    if os.path.exists(gt_cache):
        gt_by_scheme = json.load(open(gt_cache, encoding="utf-8"))
        log("gt multischeme: loaded from cache")
    else:
        t0 = time.time()
        gt_by_scheme = {s: [] for s in SCHEMES}
        for case, ref in test_pairs:
            for s, rec in coeffs_all_schemes(ref, case).items():
                gt_by_scheme[s].append(rec)
        json.dump(gt_by_scheme, open(gt_cache, "w", encoding="utf-8"), indent=2)
        log(f"gt multischeme: done ({time.time()-t0:.1f}s)")

    integrator_only = {
        s: metrics_vs(gt_by_scheme[s], official) for s in SCHEMES}
    for s, m in integrator_only.items():
        log(f"GT[{s:6s}] vs official: rho_D={m['rho_D']:+.3f} "
            f"rho_L={m['rho_L']:+.3f} cd_med={m['cd_rel_err_median']:.2f} "
            f"cl_med={m['cl_rel_err_median']:.3f}")

    # ---- per-seed prediction arm ----
    per_seed: dict[str, list[dict]] = {s: [] for s in SCHEMES}
    for seed in a.seeds:
        pred_cache = os.path.join(cache_sub, f"seed{seed}_pred_multischeme.json")
        pred_done = pred_cache + ".done"
        if os.path.exists(pred_done):
            pred_by_scheme = json.load(open(pred_cache, encoding="utf-8"))
            log(f"seed {seed}: loaded pred multischeme from cache")
        else:
            bb = os.path.join(a.ckpt_dir, f"seed{seed}.pt")
            if not os.path.exists(bb):
                log(f"FATAL: missing checkpoint {bb}")
                return 4
            log(f"seed {seed}: loading backbone + predicting ...")
            model, point_norm, grid_norm, _nu = base.load_backbone(bb, device)
            predictor = PointCloudPredictor(
                model, test_pcs, point_norm, grid_norm,
                device=device, chunk=eval_chunk)
            t0 = time.time()
            pred_by_scheme = {s: [] for s in SCHEMES}
            for k, (case, _ref) in enumerate(test_pairs):
                field = predictor.predict(case)
                for s, rec in coeffs_all_schemes(field, case).items():
                    pred_by_scheme[s].append(rec)
                if (k + 1) % 50 == 0:
                    log(f"seed {seed}: {k+1}/{len(test_pairs)} "
                        f"({time.time()-t0:.0f}s)")
            json.dump(pred_by_scheme,
                      open(pred_cache, "w", encoding="utf-8"), indent=2)
            open(pred_done, "w").write("done\n")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            log(f"seed {seed}: predictions done ({time.time()-t0:.0f}s)")

        for s in SCHEMES:
            m = metrics_vs(pred_by_scheme[s], official)
            m["seed"] = int(seed)
            per_seed[s].append(m)
            log(f"seed {seed} [{s:6s}] vs official: rho_D={m['rho_D']:+.3f} "
                f"rho_L={m['rho_L']:+.3f} cd_med={m['cd_rel_err_median']:.2f} "
                f"cl_med={m['cl_rel_err_median']:.3f}")

    def agg(scheme, key):
        vals = [d[key] for d in per_seed[scheme]]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "per_seed": vals}

    aggregate = {
        s: {k: agg(s, k) for k in
            ("rho_D", "rho_L", "cd_rel_err_mean", "cl_rel_err_mean",
             "cd_rel_err_median", "cl_rel_err_median")}
        for s in SCHEMES}

    out = {
        "meta": {
            "purpose": ("Prediction force coefficients vs OFFICIAL AirfRANS "
                        "labels under three integrator schemes (legacy near-"
                        "field offset, wall-extrapolated near-field, control-"
                        "volume far-field). One forward pass per seed."),
            "schemes": {
                "legacy": "force_coefficients defaults (offset/field_abs/1.5)",
                "wall": "p extrap to wall from 1.0/2.0 cells; flow-signed tau",
                "cv": "momentum balance over the outer grid ring",
            },
            "task": a.task, "seeds": a.seeds, "n_val": a.n_val,
            "resolution": a.resolution, "device": str(device),
            "design_study": "results/control/integrator_design.json",
        },
        "integrator_only_gt_vs_official": integrator_only,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    out_path = os.path.join(a.out_dir, "force_vs_official_multischeme.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
