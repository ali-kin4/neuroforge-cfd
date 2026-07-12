"""Adaptive UQ via a deep ensemble of Transolver backbones — FULLY RESUMABLE.

Why this run exists
-------------------
v2's conformal trust layer hits target coverage but via a near-*constant* band:
the MC-dropout sigma at dropout=0.05 is near-degenerate on the (near-deterministic)
SOTA Transolver, so the conformal quantile blows up (~1e9) and ECE ~ 0.31. This
run replaces that uncertainty source with a **deep ensemble** (the gold standard):
M independently-trained backbones whose per-cell disagreement is an *adaptive*
sigma (large where the models disagree). We then run the SAME split-conformal
layer on the ensemble sigma and report coverage + ECE — expecting adaptive,
low-ECE calibration that fixes the single most attackable spot in the paper.

It also reports the ensemble-MEAN accuracy (usually better than any single model)
with the shared ``evaluate_cases`` metric, plus ``residual_error_spearman``.

RESUME MODEL (survives shutdown / restart / sleep)
--------------------------------------------------
Each member's full training state (model + optimizer + scheduler + epoch + RNG)
is checkpointed to ``<ckpt-dir>/member{m}.pt`` after EVERY epoch with an atomic
write (tmp + os.replace, so a crash mid-write never corrupts it). On launch the
script:
  * SKIPS members already finished (``member{m}.done`` marker present),
  * RESUMES an in-progress member from its last saved epoch,
  * STARTS fresh members otherwise.

So you can stop ANY time — dashboard Stop, Ctrl-C, closing the terminal, or just
shutting the PC down — and lose at most the current epoch (~80 s). To continue,
re-run the EXACT SAME command; it picks up where it left off. Nothing has to run
day and night.

Always import ``neuroforge`` before numpy/torch (BLAS thread cap) -- done at top.

SMOKE (tiny; also verifies resume — run it twice and watch it resume):
    .venv/Scripts/python.exe scripts/run_ensemble_uq.py --task full \
        --n-train 8 --n-val 4 --epochs 2 --members 2 --n-points 2048 \
        --width 32 --n-layers 2 --n-heads 4 --resolution 128 \
        --cache-dir data/cache --out-dir results/uq_ensemble_smoke \
        --ckpt-dir checkpoints/uq_ensemble_smoke --smoke

FULL run (deep ensemble of 5; resumable across reboots):
    .venv/Scripts/python.exe scripts/run_ensemble_uq.py --task full \
        --n-train 800 --n-val 200 --epochs 80 --members 5 \
        --n-points 16384 --width 256 --n-layers 10 --resolution 128 \
        --cache-dir data/cache --out-dir results/uq_ensemble \
        --ckpt-dir checkpoints/uq_ensemble
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)
from neuroforge.core.config import Config
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.datamodule import FlowDataset, Normalizer
from neuroforge.data.pointcloud import F_IN, F_OUT, PointNormalizer, load_airfrans_pointclouds
from neuroforge.models.baselines.transolver import TransolverPointModel
from neuroforge.physics.evaluation import evaluate_cases
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor


def log(msg: str) -> None:
    # Same "[v2] ..." grammar the live dashboard already parses (seed = member).
    print(f"[v2] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def fit_grid_normalizer(train_pairs) -> Normalizer:
    """Fit a grid Normalizer on the GT grid pairs (field scale for rasterisation)."""
    ds = FlowDataset(train_pairs, normalizer=None)
    inputs, targets = ds.raw_arrays()
    return Normalizer().fit(inputs, targets)


def _atomic_save(obj, path: str) -> None:
    """Write a checkpoint atomically so an interrupted save never corrupts it."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


# --------------------------------------------------------------------------- #
# Train ONE ensemble member, with per-epoch resumable checkpointing
# --------------------------------------------------------------------------- #
def train_member(args, m, train_pcs, point_norm, device, ckpt_dir):
    """Train member ``m`` to completion, resuming from its checkpoint if present.

    Returns the trained model (eval mode). Checkpoints every epoch; writes a
    ``member{m}.done`` marker when finished so a re-run skips it instantly.
    """
    ckpt = os.path.join(ckpt_dir, f"member{m}.pt")
    done = os.path.join(ckpt_dir, f"member{m}.done")

    model = TransolverPointModel(
        in_features=F_IN, out_features=F_OUT, width=args.width, n_layers=args.n_layers,
        n_heads=args.n_heads, n_slices=args.n_slices, dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(len(train_pcs), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1
    )

    # ---- already finished? load weights and return (instant resume) ----
    if os.path.exists(done) and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        log(f"member {m}: already complete ({args.epochs} epochs) — loaded checkpoint")
        return model

    # ---- resume in-progress member, or start fresh ----
    start_epoch = 0
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = int(state["epoch"]) + 1
        # Restore the OneCycle LR-schedule position by FAST-FORWARDING rather than
        # load_state_dict: OneCycleLR pins total_steps at creation, so loading a
        # state saved under a different --epochs raises. Fast-forwarding rebuilds
        # the position against the CURRENT total_steps and is robust to changing
        # --epochs between runs. (start_epoch * batches_per_epoch steps already done.)
        _ff = min(start_epoch * max(len(train_pcs), 1), max(steps, 1))
        for _ in range(_ff):
            sched.step()
        rng = np.random.default_rng()
        try:
            rng.bit_generator.state = state["np_rng"]
        except Exception:
            rng = np.random.default_rng(1000 + m)
        log(f"member {m}: RESUMED from epoch {start_epoch}/{args.epochs}")
    else:
        # Distinct seed per member -> independent ensemble members.
        torch.manual_seed(1000 + m)
        np.random.seed(1000 + m)
        rng = np.random.default_rng(1000 + m)
        log(f"member {m}: {model.num_params():,} params — fresh start")

    n_pts = int(args.n_points)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = rng.permutation(len(train_pcs))
        agg, n = 0.0, 0
        t0 = time.time()
        for idx in order:
            pc = train_pcs[idx]
            mlen = pc.n_points
            sel = rng.integers(0, mlen, size=min(n_pts, mlen)) if mlen > n_pts else np.arange(mlen)
            x = point_norm.transform_in(pc.features[sel])
            y = point_norm.transform_out(pc.targets[sel])
            xb = torch.from_numpy(x).to(device).unsqueeze(0)
            yb = torch.from_numpy(y).to(device).unsqueeze(0)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg += float(loss.detach())
            n += 1
        dt = time.time() - t0
        # Dashboard-parseable line ("seed" = member index).
        log(f"seed {m} backbone epoch {epoch}: train_mse={agg/max(n,1):.4e} ({dt:.1f}s)")

        # ---- checkpoint EVERY epoch (atomic) — the resume guarantee ----
        _atomic_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "np_rng": rng.bit_generator.state,
            "args": vars(args),
        }, ckpt)

    # mark complete
    with open(done, "w", encoding="utf-8") as f:
        f.write(f"member {m} complete: {args.epochs} epochs\n")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Ensemble sigma -> split-conformal coverage + ECE  (the adaptive UQ payoff)
# --------------------------------------------------------------------------- #
def ensemble_conformal(predictors, pairs, alpha, channel, seed) -> dict:
    """Split-conformal coverage of the ENSEMBLE sigma (per-cell std across members).

    For each case: stack the M rasterised member predictions, take per-cell mean
    and std (the adaptive sigma), then fit a ConformalCalibrator on half the cases
    and report empirical coverage + ECE on the other half — same primitives as the
    v2 / H4 conformal, but with ensemble (not MC-dropout) uncertainty.
    """
    from neuroforge.physics.calibration import ConformalCalibrator, reliability

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    half = max(1, len(pairs) // 2)
    cal_pairs = [pairs[i] for i in idx[:half]]
    test_pairs = [pairs[i] for i in idx[half:]] or cal_pairs

    def err_sigma(case_pairs):
        errs, sigs, masks = [], [], []
        for case, gt in case_pairs:
            if not all(p.has_cloud(case.name) for p in predictors):
                continue
            arr = np.stack([p.predict(case).as_array()[channel] for p in predictors])  # (M,H,W)
            mean = arr.mean(0)
            sig = arr.std(0)                      # ADAPTIVE per-cell uncertainty
            errs.append(mean - gt.as_array()[channel])
            sigs.append(sig)
            masks.append(gt.mask)
        return errs, sigs, masks

    ce, cs, cm = err_sigma(cal_pairs)
    te, ts, tm = err_sigma(test_pairs)
    if not ce or not te:
        return {"status": "n/a (no aligned cases)"}
    cal = ConformalCalibrator(alpha=alpha).fit(ce, cs, cm)
    cov = cal.coverage(te, ts, tm)
    rel = reliability(te, ts, tm, n_bins=10, q=cal.q, alpha=alpha)
    return {
        "status": "ok", "uncertainty": "deep-ensemble", "channel": int(channel),
        "alpha": float(alpha), "n_members": len(predictors), "q": float(cal.q),
        "coverage": float(cov), "target_coverage": float(1.0 - alpha),
        "ece": float(rel["ece"]), "n_cal": len(ce), "n_test": len(te),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Adaptive UQ: deep ensemble of Transolver backbones (resumable).")
    p.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    p.add_argument("--members", type=int, default=5, help="ensemble size")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--n-points", type=int, default=16384)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="results/uq_ensemble")
    p.add_argument("--ckpt-dir", default="checkpoints/uq_ensemble")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-chunk", type=int, default=0)
    # backbone (matched-budget defaults ~7.35M params)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-slices", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    # conformal
    p.add_argument("--conformal-alpha", type=float, default=0.1)
    p.add_argument("--conformal-channel", type=int, default=2)
    p.add_argument("--seed-split", type=int, default=0, help="seed for the cal/test split")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)

    device = _resolve_device(a.device)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    log(f"device={device}  members={a.members}  ckpt-dir={a.ckpt_dir}")
    log("RESUMABLE: stop any time (Stop / Ctrl-C / shutdown); re-run the SAME "
        "command to continue from the last saved epoch.")

    eval_chunk = a.eval_chunk if a.eval_chunk and a.eval_chunk > 0 else None

    # ---- data ----
    log("loading native point clouds (train + test) ...")
    train_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=True, limit=a.n_train,
        cache_dir=a.cache_dir, download=a.download,
    )
    test_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=False, limit=a.n_val,
        cache_dir=a.cache_dir, download=a.download,
    )
    log("loading rasterised GT grid pairs (shared eval) ...")
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
    checker = PhysicsChecker(Config().physics)

    # ---- train / resume each member ----
    models = []
    for m in range(a.members):
        log(f"================= member {m}/{a.members - 1} =================")
        models.append(train_member(a, m, train_pcs, point_norm, device, a.ckpt_dir))

    # ---- ensemble-mean accuracy via the shared metric ----
    predictors = [
        PointCloudPredictor(md, test_pcs, point_norm, grid_norm, device=device, chunk=eval_chunk)
        for md in models
    ]

    def ensemble_predict(case):
        # Average the member rasterised fields -> ensemble-mean FlowField.
        fields = [pr.predict(case) for pr in predictors]
        base = fields[0]
        stacked = np.mean([f.as_array() for f in fields], axis=0).astype(base.as_array().dtype)
        return base.with_array(stacked) if hasattr(base, "with_array") else _rebuild_field(base, stacked)

    log("evaluating ensemble-mean accuracy ...")
    t0 = time.time()
    mets = evaluate_cases(ensemble_predict, test_pairs, checker=checker)
    log(f"ensemble-mean eval {time.time()-t0:.1f}s -> "
        + ", ".join(f"{k}={mets.get(k, float('nan')):.4g}"
                    for k in ("mse_u", "mse_v", "mse_p", "rho_cl", "rho_cd",
                              "cl_rel_err_mean", "cd_rel_err_mean", "residual_error_spearman")))

    # ---- ensemble conformal (the adaptive-UQ headline) ----
    log("ensemble conformal coverage (adaptive sigma) ...")
    cov = ensemble_conformal(predictors, test_pairs, a.conformal_alpha,
                             a.conformal_channel, a.seed_split)
    log(f"ensemble conformal -> {cov}")

    # ---- persist ----
    out = {
        "meta": {
            "task": a.task, "members": a.members, "epochs": a.epochs,
            "width": a.width, "n_layers": a.n_layers,
            "n_params": int(models[0].num_params()),
            "uncertainty": "deep-ensemble",
        },
        "ensemble_mean_metrics": mets,
        "ensemble_conformal": cov,
    }
    with open(os.path.join(a.out_dir, "uq_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {a.out_dir}/uq_results.json")
    log(f"artifacts in {a.out_dir}")
    return 0


def _rebuild_field(template, arr):
    """Fallback: rebuild a FlowField from a template + (C,H,W) array."""
    from neuroforge.core.types import FlowField
    return FlowField(u=arr[0], v=arr[1], p=arr[2], nut=arr[3],
                     mask=template.mask, domain=template.domain)


if __name__ == "__main__":
    raise SystemExit(main())
