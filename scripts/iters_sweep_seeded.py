"""Seeded replication of the iteration sweep behind ``tab:iters``.

WHAT tab:iters CLAIMS, AND WHY IT IS EXPOSED
--------------------------------------------
``tab:iters`` (docs/paper/body.tex) is the paper's central empirical falsification
of residual-as-objective: as correction iterations rise, the monitored PDE residual
rises monotonically (0.113 -> 0.620) while field error falls (mse_u 3.924 -> 2.287).
Its own caption concedes "Single sweep (one checkpoint, not seeded)". A load-bearing
headline resting on one unseeded checkpoint will not survive review.

WHAT CANNOT BE FIXED, STATED UP FRONT
--------------------------------------
``tab:iters`` was produced by ``scripts/run_sensitivity.py`` on
``checkpoints/certificates_deq.pt`` -- a dropout-FNO backbone with a DEQ corrector,
mse_u ~ 3.92, n_eval = 80. There is exactly ONE such file in the repository and no
seeded siblings (verified: ``find checkpoints -name '*.pt'``). The n=1 dependency
**for that configuration** cannot be removed without retraining that arm. This
script does NOT claim to remove it and its output must not be read as a substitute
for it.

WHAT THIS SCRIPT DOES INSTEAD -- A DIFFERENT, STRONGER CLAIM
-------------------------------------------------------------
Five seeded, reload-complete Transolver + DEQ checkpoints DO exist
(``checkpoints/v2_transolver/seed{0..4}.pt`` and ``seed{k}_corr_with.pt``, the
headline v2 system, mse_u ~ 0.13). Running the identical sweep on those asks a
question the original could not: **is the residual/error dissociation a property of
one checkpoint, or of the method?** That is backbone-agnostic evidence on the SOTA
arm rather than a repair of the FNO arm.

PRE-DECLARED (before running)
-----------------------------
* residual_norm RISES while mse_u FALLS on >= 4/5 seeds
    -> the dissociation is backbone-agnostic and seed-robust. Replace the n=1
       table with this one; tab:iters demotes to a per-checkpoint illustration.
* otherwise
    -> the dissociation is configuration-specific. tab:iters must be demoted and
       the headline must rest on the residual-floor argument instead.
Reported whichever way it lands.

THE DEQ INERTNESS TRAP
----------------------
``run_sensitivity.py`` documents (and this script re-verifies as a control) that the
ENGINE-level ``solve(case, max_iters=k)`` is **inert** for the DEQ branch: the DEQ
path in ``solver/correction_loop.py`` runs its own internal fixed-point loop and
ignores the engine's ``max_iters``. The knob that actually moves is
``corrector.max_iter``. Sweeping the wrong one produces a flat curve that looks
like a null result. The control below asserts the engine knob is inert before any
sweep number is believed.

Usage
-----
    python scripts/iters_sweep_seeded.py --seeds 0 1 2 3 4 --n-eval 80
    python scripts/iters_sweep_seeded.py --seeds 0            # one seed per call
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# MUST precede numpy/torch: sets the BLAS/OMP thread caps.
import neuroforge  # noqa: F401
import numpy as np
import torch

from neuroforge.core.config import Config
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.pointcloud import load_airfrans_pointclouds
from neuroforge.physics.evaluation import evaluate_cases
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the EXACT restore path the acceptance-gate measurement uses, rather than
# writing a second one that could drift from it.
from measure_acceptance_gate import _load_backbone, _load_corrector  # noqa: E402

CKPT_DIR = "checkpoints/v2_transolver"
OUT_JSON = "results/sensitivity/iters_seeded.json"
OUT_CSV = "results/sensitivity/iters_seeded.csv"
ITERS = (0, 1, 3, 5, 10, 15)
# tab:iters used n_eval = 80 on task='full' at resolution 128 (results/sensitivity/iters.json).
N_EVAL = 80
RES = 128


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _nan_mean(xs):
    a = np.asarray(xs, np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def eval_at_iters(engine, pairs, n_iters, checker):
    """Metrics at a given DEQ fixed-point cap.

    ``n_iters == 0`` is backbone-only (corrector bypassed entirely), matching
    ``run_sensitivity.eval_at_iters``. ``n_iters > 0`` sets ``corrector.max_iter``
    -- the knob that is NOT inert -- and runs the full certified loop.
    """
    corrector = engine.corrector
    if n_iters > 0 and corrector is not None:
        corrector.max_iter = int(n_iters)

    # Memoised: evaluate_cases computes per-case residual norms internally but does
    # not return their mean, and we need it. Caching the field means each case is
    # solved ONCE, not twice -- at 5 seeds x 6 caps x 80 cases that halves the run.
    cache = {}
    deq_iters = []

    def predict(case):
        key = case.name
        if key not in cache:
            if n_iters == 0:
                cache[key] = engine.predictor.predict(case)
            else:
                res = engine.solve(case)
                # How many fixed-point steps the DEQ actually took. On the FNO arm
                # this equalled the cap at every k (the DEQ never converged early).
                # If on this arm it saturates at k~2, the k=5/10/15 rows are
                # near-duplicates and a flat tail means "converged", not "the
                # residual plateaus" -- so it is logged, not assumed.
                if res.history and "deq_iters" in res.history[-1]:
                    deq_iters.append(float(res.history[-1]["deq_iters"]))
                cache[key] = res.field
        return cache[key]

    mets = evaluate_cases(predict, pairs, checker=checker)
    res_norms = []
    for case, _gt in pairs:
        try:
            res_norms.append(float(checker.diagnose(predict(case), case).residual_norm()))
        except Exception:
            res_norms.append(float("nan"))
    cache.clear()
    return {
        "n_iters": int(n_iters),
        "mse_u": float(mets.get("mse_u", float("nan"))),
        "mse_v": float(mets.get("mse_v", float("nan"))),
        "mse_p": float(mets.get("mse_p", float("nan"))),
        "surface_mse_p": float(mets.get("surface_mse_p", float("nan"))),
        "residual_norm": _nan_mean(res_norms),
        "residual_error_spearman": float(
            mets.get("residual_error_spearman", float("nan"))),
        # Per-case residuals are kept, not just their mean: a small mean shift over
        # a couple of dozen unpaired cases is not distinguishable from noise, but a
        # PAIRED per-case sign count against k=0 is.
        "residual_norm_per_case": [float(x) for x in res_norms],
        "deq_iters_mean": _nan_mean(deq_iters) if deq_iters else 0.0,
        "deq_iters_frac_at_cap": (
            float(np.mean([d >= n_iters - 1e-9 for d in deq_iters]))
            if deq_iters else None),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--iters", type=int, nargs="*", default=list(ITERS))
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args(argv)

    t_start = time.time()
    device = torch.device(
        "cuda" if (a.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if a.device == "auto" else a.device))
    log(f"device={device} seeds={a.seeds} n_eval={a.n_eval} iters={a.iters}")

    checker = PhysicsChecker(Config().physics)
    log("loading test point clouds + rasterised GT pairs ...")
    test_pcs = load_airfrans_pointclouds(
        root="data", task="full", train=False, limit=a.n_eval,
        cache_dir="data/cache", progress=False)
    test_pairs = load_airfrans(
        root="data", task="full", train=False, resolution=RES, limit=a.n_eval,
        cache_dir="data/cache", progress=False)
    log(f"  {len(test_pcs)} clouds, {len(test_pairs)} pairs")

    # Resume-friendly: keep any seeds already computed in a previous invocation.
    # Resume policy: keep EVERY previously computed seed, including ones this
    # invocation is about to recompute. A recomputed seed replaces its predecessor
    # only once it has actually finished (see the drop below). The earlier version
    # dropped the requested seeds up front, so a crash part-way through a re-run
    # destroyed results that were already on disk.
    per_seed = []
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out, encoding="utf-8"))
            per_seed = list(prev.get("per_seed", []))
            log(f"  resuming; kept {len(per_seed)} previously computed seeds "
                f"({sorted(s['seed'] for s in per_seed)})")
        except Exception:
            per_seed = []

    control = None
    for seed in a.seeds:
        bb = os.path.join(a.ckpt_dir, f"seed{seed}.pt")
        cc = os.path.join(a.ckpt_dir, f"seed{seed}_corr_with.pt")
        if not (os.path.exists(bb) and os.path.exists(cc)):
            log(f"seed {seed}: MISSING checkpoint ({bb} / {cc}) -- skipped")
            continue
        t0 = time.time()
        model, point_norm, grid_norm = _load_backbone(bb, device)
        corrector = _load_corrector(cc)
        pred = PointCloudPredictor(model, test_pcs, point_norm, grid_norm, device=device)
        engine = NeuroForgeEngine(pred, checker, corrector=corrector, config=Config())
        log(f"seed {seed}: loaded backbone + WITH-residual DEQ corrector "
            f"({time.time() - t0:.0f}s)")

        # ---- control: the ENGINE-level max_iters must be inert on the DEQ branch.
        # If this ever came back False, every sweep row below would be measuring a
        # different knob than the one the paper describes.
        if control is None:
            c0 = test_pairs[0][0]
            r_a = engine.solve(c0, max_iters=1).metrics["residual_norm"]
            r_b = engine.solve(c0, max_iters=15).metrics["residual_norm"]
            control = {"engine_maxiters_inert_on_deq": bool(np.isclose(r_a, r_b, atol=1e-9)),
                       "r_at_engine_maxiters_1": float(r_a),
                       "r_at_engine_maxiters_15": float(r_b),
                       "swept_knob": "DEQCorrector.max_iter (internal fixed-point cap)"}
            log(f"control: engine max_iters inert on DEQ? "
                f"{control['engine_maxiters_inert_on_deq']} ({r_a:.6f} vs {r_b:.6f})")

        rows = []
        for k in a.iters:
            tk = time.time()
            row = eval_at_iters(engine, test_pairs, k, checker)
            rows.append(row)
            log(f"seed {seed} iters={k:>2}: mse_u={row['mse_u']:.4g} "
                f"resid={row['residual_norm']:.4g} "
                f"rho={row['residual_error_spearman']:.3f} ({time.time() - tk:.0f}s)")

        base, fin = rows[0], rows[-1]
        best = min(rows, key=lambda r: r["mse_u"])
        # PAIRED sign count: in how many individual cases is the residual higher at
        # the error-minimising cap than at k=0? This is what makes a small mean
        # shift credible; an unpaired 2% difference over 24 cases is not.
        b0 = base.get("residual_norm_per_case") or []
        bb = best.get("residual_norm_per_case") or []
        paired = [(y > x) for x, y in zip(b0, bb) if np.isfinite(x) and np.isfinite(y)]
        n_paired_up = int(sum(paired))
        seed_out = {
            "seed": int(seed), "rows": rows,
            "residual_rises": bool(fin["residual_norm"] > base["residual_norm"]),
            "error_falls": bool(best["mse_u"] < base["mse_u"]),
            "dissociates": bool(fin["residual_norm"] > base["residual_norm"]
                                and best["mse_u"] < base["mse_u"]),
            "residual_monotone_nondecreasing": bool(
                all(rows[i + 1]["residual_norm"] >= rows[i]["residual_norm"] - 1e-12
                    for i in range(len(rows) - 1))),
            "mse_u_argmin_iters": int(best["n_iters"]),
            "paired_residual_up_at_best_k": n_paired_up,
            "paired_n": len(paired),
            "deq_iters_mean_by_k": {str(r["n_iters"]): r["deq_iters_mean"] for r in rows},
            "deq_iters_frac_at_cap_by_k": {
                str(r["n_iters"]): r["deq_iters_frac_at_cap"] for r in rows},
            "residual_norm_start": base["residual_norm"],
            "residual_norm_end": fin["residual_norm"],
            "mse_u_start": base["mse_u"], "mse_u_best": best["mse_u"],
        }
        # Now that the replacement exists, retire any earlier entry for this seed.
        per_seed = [s for s in per_seed if s["seed"] != int(seed)]
        per_seed.append(seed_out)
        log(f"seed {seed}: dissociates={seed_out['dissociates']} "
            f"(resid {base['residual_norm']:.3f}->{fin['residual_norm']:.3f}, "
            f"mse_u {base['mse_u']:.3f}->{best['mse_u']:.3f} @k={best['n_iters']}) "
            f"[{time.time() - t0:.0f}s]")
        _write(a.out, {"artifact": "iters_sweep_seeded", "status": "in-progress",
                       "control": control, "per_seed": per_seed})
        del model, corrector, pred, engine
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_seed.sort(key=lambda s: s["seed"])
    n = len(per_seed)
    agg_rows = []
    for k in a.iters:
        vals = {m: [s_["rows"][i][m] for s_ in per_seed
                    for i, r in enumerate(s_["rows"]) if r["n_iters"] == k]
                for m in ("mse_u", "surface_mse_p", "residual_norm",
                          "residual_error_spearman")}
        agg_rows.append({"n_iters": k, "n_seeds": n,
                         **{f"{m}_mean": _nan_mean(v) for m, v in vals.items()},
                         **{f"{m}_std": (float(np.std(np.asarray(v, np.float64)))
                                         if v else float("nan"))
                            for m, v in vals.items()}})

    n_diss = sum(s_["dissociates"] for s_ in per_seed)
    verdict = ("SEED-ROBUST: the residual/error dissociation reproduces on "
               f"{n_diss}/{n} seeds of the Transolver+DEQ arm"
               if n and n_diss >= max(1, int(np.ceil(0.8 * n))) else
               f"NOT SEED-ROBUST: dissociation on only {n_diss}/{n} seeds")

    out = {
        "artifact": "iters_sweep_seeded",
        "status": "complete",
        "verdict": verdict,
        "metadata": {
            "script": "scripts/iters_sweep_seeded.py",
            "purpose": ("Seeded replication of the tab:iters dissociation on the "
                        "5-seed Transolver+DEQ arm."),
            "what_this_is_NOT": (
                "This does NOT remove the n=1 dependency of tab:iters itself. "
                "tab:iters was produced on checkpoints/certificates_deq.pt (dropout-FNO "
                "backbone, mse_u ~ 3.92); exactly one such file exists and there are no "
                "seeded siblings, so that specific configuration cannot be re-seeded "
                "without retraining. This is a different, stronger claim on a different "
                "(SOTA, headline) backbone."),
            "checkpoints": [f"{a.ckpt_dir}/seed{s}.pt + seed{s}_corr_with.pt"
                            for s in a.seeds],
            "n_eval": a.n_eval, "task": "full", "resolution": RES,
            "iters": list(a.iters),
            "swept_knob": "DEQCorrector.max_iter (engine-level max_iters is inert on DEQ)",
            "residual_norm": ("Diagnostics.residual_norm -- interior continuity+momentum, "
                              "BC term excluded, mean over eval cases"),
            "device": str(device),
        },
        "control": control,
        "aggregate": {"rows": agg_rows, "n_seeds": n, "n_dissociating": n_diss,
                      "seeds_dissociating": [s_["seed"] for s_ in per_seed
                                             if s_["dissociates"]]},
        "per_seed": per_seed,
        "wall_clock_s": time.time() - t_start,
    }
    _write(a.out, out)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["n_iters", "n_seeds", "mse_u_mean", "mse_u_std",
                    "surface_mse_p_mean", "surface_mse_p_std",
                    "residual_norm_mean", "residual_norm_std",
                    "residual_error_spearman_mean", "residual_error_spearman_std"])
        for r in agg_rows:
            w.writerow([r["n_iters"], r["n_seeds"],
                        f"{r['mse_u_mean']:.6e}", f"{r['mse_u_std']:.6e}",
                        f"{r['surface_mse_p_mean']:.6e}", f"{r['surface_mse_p_std']:.6e}",
                        f"{r['residual_norm_mean']:.6e}", f"{r['residual_norm_std']:.6e}",
                        f"{r['residual_error_spearman_mean']:.4f}",
                        f"{r['residual_error_spearman_std']:.4f}"])

    print("\n" + "=" * 78)
    print(f"{'k':>3} {'mse_u':>18} {'residual_norm':>20} {'rho':>16}")
    for r in agg_rows:
        print(f"{r['n_iters']:>3} {r['mse_u_mean']:>10.4f}+-{r['mse_u_std']:<7.4f}"
              f"{r['residual_norm_mean']:>12.4f}+-{r['residual_norm_std']:<7.4f}"
              f"{r['residual_error_spearman_mean']:>9.3f}+-{r['residual_error_spearman_std']:.3f}")
    print(f"\n{verdict}")
    print(f"wrote {a.out} and {OUT_CSV}  [{(time.time() - t_start) / 60:.1f} min]")
    return 0


def _clean(o):
    """Coerce numpy scalars and map non-finite floats to null.

    ``allow_nan=False`` keeps the JSON strict-spec valid, but ``json`` raises on
    NaN rather than consulting ``default=``, so this must happen before the dump.
    """
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    return o


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    # newline="\n": result JSONs are stored LF; writing CRLF breaks the manifest hash.
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_clean(obj), fh, indent=2, allow_nan=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
