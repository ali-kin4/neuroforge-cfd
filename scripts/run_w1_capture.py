"""W1 capture — ONE resumable overnight GPU run that does three things at once.

This harness reuses ``scripts/run_v2.py``'s EXACT training functions (the ones
that produced the published headline in ``results/v2/v2_results.json``) — it does
NOT reimplement training; it imports and calls ``train_transolver``,
``fit_grid_normalizer``, ``train_corrector`` and the shared ``evaluate_cases``.
Per seed (in order) it:

1. Trains the Transolver backbone ONCE and SAVES a reload-complete checkpoint
   (``checkpoints/v2_transolver/seed{m}.pt``: state_dict + backbone hyperparams +
   the point_norm + grid_norm) — these checkpoints did not exist anywhere (a
   reproducibility hole) and are needed for the later cylinder-OOD experiment.
2. Builds the ``PointCloudPredictor`` on that frozen backbone.
3. Trains the WITH-residual DEQ corrector (``train_corrector``), evaluates the
   certified loop with ``evaluate_cases``, saves ``seed{m}_corr_with.pt``.
4. Trains the NULL-residual DEQ corrector (``train_corrector(zero_residual=True)``
   — the residual input is permanently zeroed at train AND eval, see the
   ``_zero_residual`` tag), evaluates, saves ``seed{m}_corr_null.pt``.

It accomplishes three goals simultaneously:
  (a) saves the missing backbone checkpoints,
  (b) reproduces the published headline as a built-in correctness GATE, and
  (c) settles reviewer finding W1 (with-residual vs without-residual corrector).

GATE (built in, not eyeballed)
------------------------------
After all seeds, the mean WITH-corrector deltas (corrected vs backbone) are
compared to the published headline using the RATIO OF MEANS — i.e.
``(mean WITH) / (mean backbone) - 1`` — NOT the mean of per-seed ratios (the two
differ; the published −9 / −21 / −25 % match the ratio of means). Targets:
mse_u −8.7 %, mse_v −20.6 %, mse_p −25.1 %, and backbone ρ_Cl ≈ 0.9992. Each is
PASS within a stated tolerance chosen for 3-seed GPU non-determinism; an overall
``gate_passed`` bool is written. A FAILED gate means the harness is unfaithful ⇒
the W1 verdict is VOID, and the JSON says so loudly (``w1_verdict`` is forced to
``"VOID (gate failed)"``). A failing gate still writes the JSON and exits 0
(the tiny CPU smoke is EXPECTED to fail the gate — that must not abort).

W1 VERDICT (pre-registered)
---------------------------
Per seed, compute the corrected FIELD-MSE GAIN over the backbone
(``gain = backbone_mse - corrected_mse``, higher = better) on ``mse_u`` and
``mse_speed`` for BOTH the WITH and the NULL corrector. The paired statistic is
``d_seed = gain_WITH - gain_NULL`` (per seed, per metric). Pre-registered rule
(applied to ``mse_u`` AND ``mse_speed``):

    Let ``mean_d`` = mean over seeds of ``d_seed`` and
        ``spread`` = population std over seeds of ``d_seed`` (= "max seed std"
                     interpreted as the across-seed std of the paired difference).
    W1 SUPPORTED if   mean_d > spread   AND   d_seed > 0 on >= 2 of 3 seeds,
    for BOTH mse_u and mse_speed.   Else  ⇒  REFRAME.

SUPPORTED means the residual INPUT has architectural value (the corrector does
better when it can see the physics residual). REFRAME means the corrector gets
the same gain trained without it — the residual is not load-bearing as an input.

FAITHFULNESS of the two run_v2.py edits this run depends on
----------------------------------------------------------
* ``train_corrector(..., zero_residual=False)`` default ⇒ published path is
  byte-identical (the only new code is an ``if zero_residual:`` guard never
  entered; ``torch.zeros_like`` draws no RNG so WITH/NULL stay a paired control).
* ``train_transolver(..., ckpt_path=None, done_path=None)`` defaults ⇒ no resume
  scaffolding runs; the training math is unchanged.
* ``correction_loop.py`` gates on ``getattr(corrector, "_zero_residual", False)``
  ⇒ default correctors (no attribute) keep the published DEQ path byte-identical.

RESUMABILITY (survives shutdown / sleep — the PC died mid-run before)
--------------------------------------------------------------------
Mirrors ``scripts/run_mgn.py`` exactly: per-epoch atomic backbone checkpoints
(tmp + os.replace) with OneCycleLR fast-forward on resume, a ``seed{m}.done``
marker that skips a finished backbone instantly, and per-CONDITION corrector
``.done`` markers so completed correctors are skipped too. Re-running the IDENTICAL
command continues from the last saved state. Progress lines use the SAME
``[v2] seed {m} backbone epoch {e}: train_mse=... ({s}s)`` grammar so the existing
dashboard parses them unchanged.

Always import ``neuroforge`` FIRST (BLAS thread cap) — done at the top, before
``run_v2`` (which pulls numpy/torch) is imported.

SMOKE (tiny; also verifies resume — run it twice; the 2nd run skips done work):
    OMP_NUM_THREADS=8 .venv/Scripts/python.exe scripts/run_w1_capture.py \
        --task full --n-train 8 --n-val 4 --epochs 2 --corrector-epochs 2 \
        --n-points 1024 --width 32 --n-layers 2 --n-heads 4 --n-slices 8 \
        --resolution 128 --seeds 0 1 --device cpu \
        --cache-dir data/cache \
        --out-dir results/control_smoke \
        --ckpt-dir checkpoints/v2_transolver_smoke

FULL run (the deciding 3-seed run; pre-registered config — DO NOT change
batch/width/epochs):
    .venv/Scripts/python.exe scripts/run_w1_capture.py \
        --task full --n-train 800 --n-val 200 --epochs 80 --corrector-epochs 20 \
        --n-points 16384 --width 256 --n-layers 10 --n-heads 8 --n-slices 32 \
        --dropout 0.05 --lr 1e-3 --weight-decay 1e-5 --grad-clip 1.0 \
        --corrector-width 32 --corrector-layers 3 --unroll 3 --noise-std 0.1 \
        --resolution 128 --seeds 0 1 2 --device auto \
        --cache-dir data/cache \
        --out-dir results/control --ckpt-dir checkpoints/v2_transolver
"""

from __future__ import annotations

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

# run_v2 lives in scripts/ alongside this file; importing it pulls numpy/torch,
# so neuroforge must already be imported (above) for the thread caps to take.
from run_v2 import (  # noqa: E402
    fit_grid_normalizer,
    train_corrector,
    train_transolver,
)

from neuroforge.core.config import Config  # noqa: E402
from neuroforge.data.airfrans_loader import load_airfrans  # noqa: E402
from neuroforge.data.pointcloud import (  # noqa: E402
    F_IN,
    F_OUT,
    PointNormalizer,
    load_airfrans_pointclouds,
)
from neuroforge.physics.evaluation import evaluate_cases  # noqa: E402
from neuroforge.physics.residuals import PhysicsChecker  # noqa: E402
from neuroforge.solver.engine import NeuroForgeEngine  # noqa: E402
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor  # noqa: E402


# --------------------------------------------------------------------------- #
# Published headline (results/v2/v2_results.json) — the GATE targets.
# Ratio-of-means deltas verified against that file: u -8.7%, v -20.6%, p -25.1%.
# --------------------------------------------------------------------------- #
_PUBLISHED_DELTA = {"mse_u": -0.087, "mse_v": -0.206, "mse_p": -0.251}
_PUBLISHED_RHO_CL = 0.9992
# Tolerances for 3-seed GPU non-determinism (absolute, on the fractional delta).
_GATE_TOL = {"mse_u": 0.06, "mse_v": 0.08, "mse_p": 0.08}
_GATE_RHO_CL_TOL = 0.003

# Metrics reported per variant (same columns as run_v2's table).
_FIELD_METRICS = [
    "mse_u", "mse_v", "mse_p", "mse_nut", "mse_speed", "surface_mse_p",
    "rho_cl", "rho_cd", "cl_rel_err_mean", "cd_rel_err_mean",
    "residual_error_spearman",
]


def log(msg: str) -> None:
    # Same "[v2] ..." grammar the live dashboard already parses (seed index).
    print(f"[v2] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _make_args(a) -> SimpleNamespace:
    """A shim carrying every field train_transolver / train_corrector read.

    train_corrector reads ``args.corrector`` to pick DEQ vs local; we pin it to
    "deq" (the mandated backbone corrector / published headline).
    """
    return SimpleNamespace(
        # backbone
        width=a.width, n_layers=a.n_layers, n_heads=a.n_heads, n_slices=a.n_slices,
        dropout=a.dropout, lr=a.lr, weight_decay=a.weight_decay, grad_clip=a.grad_clip,
        epochs=a.epochs, n_points=a.n_points,
        # corrector
        corrector="deq", corrector_width=a.corrector_width,
        corrector_layers=a.corrector_layers, unroll=a.unroll, noise_std=a.noise_std,
        corrector_epochs=a.corrector_epochs,
    )


def _eval_metrics(mets: dict) -> dict:
    """Keep only the reported field metrics (finite-coerced for JSON)."""
    out = {}
    for k in _FIELD_METRICS:
        v = mets.get(k, float("nan"))
        out[k] = float(v) if v is not None else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Stage 3/4 — train + eval one corrector condition (WITH or NULL), resumable.
# --------------------------------------------------------------------------- #
def run_corrector_condition(
    args, seed, train_pred, test_pred, train_pairs, test_pairs, grid_norm, nu,
    checker, device, ckpt_path, done_path, zero_residual,
):
    """Train a DEQ corrector (WITH or NULL) and evaluate the certified loop.

    Skips instantly if ``done_path`` exists. Saves the corrector state_dict + its
    config to ``ckpt_path`` and writes ``done_path`` on completion. Returns the
    eval metrics dict (loaded from the cached .done payload on a skip).
    """
    cond = "null" if zero_residual else "with"
    if os.path.exists(done_path):
        log(f"seed {seed}: {cond}-corrector already complete — loading cached metrics")
        return json.load(open(done_path, encoding="utf-8"))["metrics"]

    log(f"seed {seed}: training {cond}-residual DEQ corrector "
        f"(zero_residual={zero_residual}) ...")
    corrector, _hist = train_corrector(
        args, seed, train_pred, train_pairs, grid_norm, nu, device,
        zero_residual=zero_residual,
    )

    # Save the corrector (state_dict + config) for later reuse.
    _atomic_save({
        "corrector_state": corrector.state_dict(),
        "corrector_config": {
            "type": "deq", "width": args.corrector_width,
            "n_layers": args.corrector_layers,
        },
        "zero_residual": bool(zero_residual),
        "seed": int(seed),
    }, ckpt_path)

    cfg = Config()
    engine = NeuroForgeEngine(test_pred, checker, corrector=corrector, config=cfg)

    def loop_predict(case, _eng=engine):
        return _eng.solve(case).field

    log(f"seed {seed}: evaluating {cond}-corrector loop ...")
    t0 = time.time()
    mets = evaluate_cases(loop_predict, test_pairs, checker=checker)
    mets = _eval_metrics(mets)
    log(f"seed {seed}: ({cond}) eval {time.time()-t0:.1f}s -> "
        + ", ".join(f"{k}={mets.get(k, float('nan')):.4g}" for k in _FIELD_METRICS))

    with open(done_path, "w", encoding="utf-8") as f:
        json.dump({"seed": int(seed), "condition": cond, "metrics": mets}, f, indent=2)
    return mets


# --------------------------------------------------------------------------- #
# Gate + W1 verdict
# --------------------------------------------------------------------------- #
def _mean(per_seed, key):
    vals = [d[key] for d in per_seed if np.isfinite(d.get(key, np.nan))]
    return float(np.mean(vals)) if vals else float("nan")


def compute_gate(backbone, with_corr):
    """Ratio-of-means WITH-corrector deltas vs the published headline."""
    checks = {}
    all_pass = True
    for k, target in _PUBLISHED_DELTA.items():
        mb, mw = _mean(backbone, k), _mean(with_corr, k)
        delta = (mw / mb - 1.0) if (np.isfinite(mb) and mb != 0) else float("nan")
        ok = np.isfinite(delta) and abs(delta - target) <= _GATE_TOL[k]
        checks[k] = {
            "measured_delta": delta, "published_delta": target,
            "tol": _GATE_TOL[k], "pass": bool(ok),
        }
        all_pass = all_pass and ok
    rho = _mean(backbone, "rho_cl")
    rho_ok = np.isfinite(rho) and abs(rho - _PUBLISHED_RHO_CL) <= _GATE_RHO_CL_TOL
    checks["rho_cl"] = {
        "measured": rho, "published": _PUBLISHED_RHO_CL,
        "tol": _GATE_RHO_CL_TOL, "pass": bool(rho_ok),
    }
    all_pass = all_pass and rho_ok
    return {"checks": checks, "gate_passed": bool(all_pass)}


def compute_w1(backbone, with_corr, null_corr):
    """Pre-registered paired WITH-vs-NULL gain verdict on mse_u and mse_speed."""
    per_metric = {}
    supported_all = True
    n_seeds = min(len(backbone), len(with_corr), len(null_corr))
    for metric in ("mse_u", "mse_speed"):
        d_seed = []
        for i in range(n_seeds):
            gain_with = backbone[i][metric] - with_corr[i][metric]
            gain_null = backbone[i][metric] - null_corr[i][metric]
            d_seed.append(gain_with - gain_null)  # WITH advantage over NULL
        d_arr = np.asarray(d_seed, dtype=float)
        mean_d = float(np.mean(d_arr))
        spread = float(np.std(d_arr))  # population std across seeds
        n_pos = int(np.sum(d_arr > 0))
        supported = (mean_d > spread) and (n_pos >= 2)
        per_metric[metric] = {
            "d_per_seed": d_seed, "mean_d": mean_d, "spread_std": spread,
            "n_seeds_with_beats_null": n_pos, "n_seeds": n_seeds,
            "supported": bool(supported),
        }
        supported_all = supported_all and supported
    verdict = "SUPPORTED" if supported_all else "REFRAME"
    return {"per_metric": per_metric, "verdict": verdict}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="W1 capture: Transolver checkpoints + headline gate + W1 control."
    )
    p.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--epochs", type=int, default=80, help="backbone epochs")
    p.add_argument("--corrector-epochs", type=int, default=20)
    p.add_argument("--n-points", type=int, default=16384)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="results/control")
    p.add_argument("--ckpt-dir", default="checkpoints/v2_transolver")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-chunk", type=int, default=0, help=">0 = OOM-fallback chunked forward")
    # backbone (published headline defaults)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-slices", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # corrector (published headline defaults)
    p.add_argument("--corrector-width", type=int, default=32)
    p.add_argument("--corrector-layers", type=int, default=3)
    p.add_argument("--unroll", type=int, default=3)
    p.add_argument("--noise-std", type=float, default=0.1)
    a = p.parse_args(argv)

    device = _resolve_device(a.device)
    eval_chunk = a.eval_chunk if a.eval_chunk and a.eval_chunk > 0 else None
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    log(f"device={device}  seeds={a.seeds}  ckpt-dir={a.ckpt_dir}  out-dir={a.out_dir}")
    log("RESUMABLE: stop any time (Stop / Ctrl-C / shutdown); re-run the SAME "
        "command to continue from the last saved state.")

    # ---- data: native clouds (backbone) + GT grid pairs (corrector + eval) ----
    log("loading native point clouds (train + test) ...")
    train_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=True, limit=a.n_train,
        cache_dir=a.cache_dir, download=a.download,
    )
    test_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=False, limit=a.n_val,
        cache_dir=a.cache_dir, download=a.download,
    )
    log("loading rasterised GT grid pairs (shared with NeuroForge eval) ...")
    train_pairs = load_airfrans(
        root=a.root, task=a.task, train=True, resolution=a.resolution,
        limit=a.n_train, cache_dir=a.cache_dir, download=a.download, progress=False,
    )
    test_pairs = load_airfrans(
        root=a.root, task=a.task, train=False, resolution=a.resolution,
        limit=a.n_val, cache_dir=a.cache_dir, download=a.download, progress=False,
    )

    point_norm = PointNormalizer().fit(train_pcs)
    grid_norm = fit_grid_normalizer(train_pairs)
    nu = float(train_pairs[0][0].fluid.kinematic_viscosity)
    checker = PhysicsChecker(Config().physics)

    sargs = _make_args(a)

    backbone_per_seed: list[dict] = []
    with_per_seed: list[dict] = []
    null_per_seed: list[dict] = []
    sec_epoch = float("nan")

    for seed in a.seeds:
        log(f"================= seed {seed} =================")
        # Two distinct files on purpose:
        #  * ``resume_ckpt`` is the per-EPOCH resumable training state that
        #    train_transolver writes/reads (keys: model/opt/np_rng/...); it is the
        #    crash-recovery artifact.
        #  * ``bb_ckpt`` is the RELOAD-COMPLETE backbone (weights + hyperparams +
        #    both normalisers) for the later cylinder-OOD run. Keeping them
        #    separate avoids the key collision (model vs model_state) that would
        #    break train_transolver's done-skip path.
        resume_ckpt = os.path.join(a.ckpt_dir, f"seed{seed}_resume.pt")
        bb_ckpt = os.path.join(a.ckpt_dir, f"seed{seed}.pt")
        bb_done = os.path.join(a.ckpt_dir, f"seed{seed}.done")

        # ---- Stage 1: backbone (per-epoch resumable via run_v2.train_transolver) ----
        was_done = os.path.exists(bb_done)
        model, sec_epoch = train_transolver(
            sargs, seed, train_pcs, point_norm, device,
            ckpt_path=resume_ckpt, done_path=bb_done,
        )
        if not was_done:
            # Save a RELOAD-COMPLETE checkpoint (everything the cylinder OOD run
            # needs: weights + backbone hyperparams + both normalisers) and write
            # the .done marker so a re-run skips this backbone instantly.
            _atomic_save({
                "model_state": model.state_dict(),
                "backbone_config": {
                    "in_features": F_IN, "out_features": F_OUT, "width": a.width,
                    "n_layers": a.n_layers, "n_heads": a.n_heads,
                    "n_slices": a.n_slices, "dropout": a.dropout,
                },
                "point_norm": point_norm.state_dict(),
                "grid_norm": grid_norm.state_dict(),
                "seed": int(seed), "epochs": a.epochs, "nu": nu,
                "sec_per_epoch": sec_epoch,
            }, bb_ckpt)
            with open(bb_done, "w", encoding="utf-8") as f:
                f.write(f"seed {seed} backbone complete: {a.epochs} epochs\n")
            log(f"seed {seed}: saved reload-complete backbone checkpoint -> {bb_ckpt}")

        # ---- Stage 2: predictors (test for eval, train for corrector training) ----
        test_pred = PointCloudPredictor(
            model, test_pcs, point_norm, grid_norm, device=device, chunk=eval_chunk
        )
        train_pred = PointCloudPredictor(
            model, train_pcs, point_norm, grid_norm, device=device, chunk=eval_chunk
        )

        # ---- (a) backbone alone ----
        log(f"seed {seed}: evaluating backbone alone ...")
        t0 = time.time()
        mets_a = _eval_metrics(evaluate_cases(test_pred.predict, test_pairs, checker=checker))
        log(f"seed {seed}: (backbone) eval {time.time()-t0:.1f}s -> "
            + ", ".join(f"{k}={mets_a.get(k, float('nan')):.4g}" for k in _FIELD_METRICS))
        backbone_per_seed.append(mets_a)

        # ---- Stage 3: WITH-residual corrector ----
        mets_with = run_corrector_condition(
            sargs, seed, train_pred, test_pred, train_pairs, test_pairs, grid_norm,
            nu, checker, device,
            ckpt_path=os.path.join(a.ckpt_dir, f"seed{seed}_corr_with.pt"),
            done_path=os.path.join(a.ckpt_dir, f"seed{seed}_corr_with.done"),
            zero_residual=False,
        )
        with_per_seed.append(mets_with)

        # ---- Stage 4: NULL-residual corrector ----
        mets_null = run_corrector_condition(
            sargs, seed, train_pred, test_pred, train_pairs, test_pairs, grid_norm,
            nu, checker, device,
            ckpt_path=os.path.join(a.ckpt_dir, f"seed{seed}_corr_null.pt"),
            done_path=os.path.join(a.ckpt_dir, f"seed{seed}_corr_null.done"),
            zero_residual=True,
        )
        null_per_seed.append(mets_null)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ---- incremental persistence after each seed (crash-safe) ----
        _write_results(a, backbone_per_seed, with_per_seed, null_per_seed, sec_epoch, nu=nu)

    # ---- final gate + verdict ----
    gate = compute_gate(backbone_per_seed, with_per_seed)
    w1 = compute_w1(backbone_per_seed, with_per_seed, null_per_seed)
    if not gate["gate_passed"]:
        # A failing gate means the harness is unfaithful -> the W1 verdict is VOID.
        w1["verdict"] = "VOID (gate failed)"
    _write_results(a, backbone_per_seed, with_per_seed, null_per_seed, sec_epoch,
                   gate=gate, w1=w1, nu=nu)

    log("================= SUMMARY =================")
    log(f"GATE passed={gate['gate_passed']}: "
        + ", ".join(f"{k}={'ok' if v['pass'] else 'FAIL'}" for k, v in gate["checks"].items()))
    log(f"W1 verdict: {w1['verdict']}")
    for m, d in w1["per_metric"].items():
        log(f"  {m}: mean_d={d['mean_d']:.4g} spread={d['spread_std']:.4g} "
            f"with>null on {d['n_seeds_with_beats_null']}/{d['n_seeds']} -> "
            f"{'supported' if d['supported'] else 'no'}")
    log(f"artifacts in {a.out_dir}")
    return 0


def _write_results(a, backbone, with_corr, null_corr, sec_epoch, gate=None, w1=None, nu=None):
    out = {
        "meta": {
            "task": a.task, "seeds": a.seeds, "n_train": a.n_train, "n_val": a.n_val,
            "epochs": a.epochs, "corrector_epochs": a.corrector_epochs,
            "width": a.width, "n_layers": a.n_layers, "n_heads": a.n_heads,
            "n_slices": a.n_slices, "dropout": a.dropout, "resolution": a.resolution,
            "corrector": "deq", "corrector_width": a.corrector_width,
            "corrector_layers": a.corrector_layers, "unroll": a.unroll,
            "noise_std": a.noise_std, "nu": nu, "sec_per_epoch": sec_epoch,
            "ckpt_dir": a.ckpt_dir,
        },
        "backbone_per_seed": backbone,
        "with_corrector_per_seed": with_corr,
        "null_corrector_per_seed": null_corr,
        "gate": gate,
        "w1": w1,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "w1_capture.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {a.out_dir}/w1_capture.json "
        f"({len(backbone)} backbone / {len(with_corr)} with / {len(null_corr)} null seeds)")


if __name__ == "__main__":
    raise SystemExit(main())
