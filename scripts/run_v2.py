"""NeuroForge v2 — Transolver backbone behind the certified self-correction loop.

PHASE 1 de-risk experiment. v1's grid FNO backbone trails the strong point-cloud
Transolver by ~16-40x on the AirfRANS field MSEs (paper Table 2). v2 swaps the
backbone for Transolver-on-the-native-cloud (rasterised to a grid FlowField via the
byte-identical adapter), then runs the EXISTING certified loop (PhysicsChecker
residuals, learned corrector, trust map, conformal coverage) on that SOTA-quality
prediction — no edits to the frozen grid contract.

The script does three things, in order:

1. **Train the Transolver backbone** on the AirfRANS native point clouds (reusing
   ``run_baselines``' point-cloud training recipe), per seed.
2. **Train a grid corrector on the TRANSOLVER OUTPUTS' residuals** — the corrector
   MUST be retrained for the new backbone, because it learns to map *this*
   backbone's rasterised field + residual toward the truth. Each unroll starts
   from the (frozen) per-case rasterised Transolver prediction, mirroring
   ``Trainer.fit_corrector`` (multi-step unroll + decaying off-manifold noise +
   per-channel-balanced loss).
3. **Evaluate** with the shared ``evaluate_cases`` on AirfRANS, reporting side by
   side: (a) the v2 backbone ALONE, and (b) the v2 backbone + the certified
   corrector/loop. Plus ``residual_error_spearman`` (free, via ``checker=``) and —
   when the backbone has dropout — split-conformal coverage of a point-cloud
   MC-dropout sigma (reusing the ``calibration`` primitives).

The de-risk question: does the certified loop ADD value on a strong backbone, and
is accuracy now competitive with Table 2's Transolver? Note honestly: the loop's
LOCAL-corrector path guarantees a NON-INCREASING physics-residual norm (the
backtracking acceptance test) — NOT a lower GT MSE; the DEQ path applies its
fixed-point correction directly with NO acceptance test. So (b)-vs-(a) MSE can be
flat or worse, especially at small scale. The real answer is the full run.

Always import ``neuroforge`` before numpy/torch (BLAS thread cap) -- done at top.

Smoke (tiny, foreground, ~minutes; uses the cached n8/n4 full-task subsets at
res 128 -- res 64 would trigger a fresh 800-case rasterise):

    .venv/Scripts/python.exe scripts/run_v2.py --task full \
        --n-train 8 --n-val 4 --epochs 2 --corrector-epochs 2 --n-points 2048 \
        --width 64 --n-layers 4 --n-heads 4 --resolution 128 --seeds 0 \
        --corrector both --corrector-width 24 --corrector-layers 2 --unroll 2 \
        --cache-dir data/cache --out-dir results/v2_smoke --smoke

Full run (3 seeds, DEQ corrector + split-conformal coverage). ``--conformal``
adds an MC-dropout pass at eval (needs ``--dropout > 0``) and is the expensive
stage; drop it for a faster accuracy-only run:

    .venv/Scripts/python.exe scripts/run_v2.py --task full \
        --n-train 800 --n-val 200 --epochs 80 --corrector-epochs 20 \
        --n-points 16384 --width 256 --n-layers 10 --resolution 128 \
        --dropout 0.05 --seeds 0 1 2 --corrector deq \
        --conformal --conformal-mc 16 \
        --cache-dir data/cache --out-dir results/v2
"""

from __future__ import annotations

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)

import argparse
import json
import os
import time

import numpy as np
import torch

from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.datamodule import FlowDataset, Normalizer
from neuroforge.data.pointcloud import F_IN, F_OUT, PointNormalizer, load_airfrans_pointclouds
from neuroforge.geometry.encode import encode_case
from neuroforge.models.baselines.transolver import TransolverPointModel
from neuroforge.models.correction import LocalCorrectionNet
from neuroforge.models.deq_corrector import DEQCorrector
from neuroforge.physics.evaluation import evaluate_cases
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor
from neuroforge.train.losses import per_channel_balanced_mse


# --------------------------------------------------------------------------- #
# Reporting columns
# --------------------------------------------------------------------------- #
_FIELD_METRICS = [
    "mse_u", "mse_v", "mse_p", "mse_nut", "mse_speed", "surface_mse_p",
    "rho_cl", "rho_cd", "cl_rel_err_mean", "cd_rel_err_mean",
    "residual_error_spearman",
]


def log(msg: str) -> None:
    print(f"[v2] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# Stage 1 — Transolver backbone (point-cloud training, same recipe as baselines)
# --------------------------------------------------------------------------- #
def train_transolver(args, seed: int, train_pcs, point_norm, device) -> tuple[torch.nn.Module, float]:
    """Train the Transolver point model for one seed. Returns (model, sec/epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TransolverPointModel(
        in_features=F_IN, out_features=F_OUT, width=args.width, n_layers=args.n_layers,
        n_heads=args.n_heads, n_slices=args.n_slices, dropout=args.dropout,
    ).to(device)
    log(f"seed {seed}: Transolver params = {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(len(train_pcs), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1
    )

    rng = np.random.default_rng(seed)
    n_pts = int(args.n_points)
    sec_per_epoch = float("nan")
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_pcs))
        agg, n = 0.0, 0
        t0 = time.time()
        for idx in order:
            pc = train_pcs[idx]
            m = pc.n_points
            sel = rng.integers(0, m, size=min(n_pts, m)) if m > n_pts else np.arange(m)
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
        sec_per_epoch = time.time() - t0
        log(f"seed {seed} backbone epoch {epoch}: train_mse={agg/max(n,1):.4e} ({sec_per_epoch:.1f}s)")
    return model, sec_per_epoch


# --------------------------------------------------------------------------- #
# Grid normaliser fitted on the GT grid pairs (field scale for the corrector)
# --------------------------------------------------------------------------- #
def fit_grid_normalizer(train_pairs) -> Normalizer:
    """Fit a grid :class:`Normalizer` on the GT grid pairs (NOT Transolver out).

    ``std_out`` here is the true field scale the correction loop uses to map a
    normalised correction back to physical units (``delta_phys = delta_norm *
    std_out``); fitting on the GT keeps that scale principled.
    """
    ds = FlowDataset(train_pairs, normalizer=None)
    inputs, targets = ds.raw_arrays()
    return Normalizer().fit(inputs, targets)


# --------------------------------------------------------------------------- #
# Stage 2 — corrector trained ON THE TRANSOLVER OUTPUTS' residuals
# --------------------------------------------------------------------------- #
def _state_residual(field_norm, inp_phys, dx, dy, nu, grid_norm) -> torch.Tensor:
    """Per-sample-standardised 3-channel residual of a NORMALISED grid state.

    Mirrors :meth:`Trainer._normalised_residual`: denormalise the state to
    physical units, run the differentiable residual operator, stack
    ``[continuity, momentum_x, momentum_y]`` and per-sample standardise. Computed
    from the CURRENT state each unroll step so the corrector learns to RESPOND to
    the residual (the residual-driven property), not a per-case-constant input.
    """
    from neuroforge.physics.residuals import physics_residual_torch

    field_phys = grid_norm.denorm_out(field_norm)
    res = physics_residual_torch(field_phys, inp_phys, dx, dy, nu)
    residual = torch.cat([res["continuity"], res["momentum_x"], res["momentum_y"]], dim=1)
    std = residual.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-6)
    return residual / std.unsqueeze(-1)


def train_corrector(args, seed, predictor, train_pairs, grid_norm, nu, device):
    """Train a grid corrector starting each unroll from the rasterised Transolver field.

    Mirrors :meth:`Trainer.fit_corrector` (truncated unroll + decaying off-manifold
    noise + per-channel-balanced loss) but the starting state is the FROZEN
    per-case rasterised Transolver prediction — so the corrector learns to map
    *this* backbone's outputs (and its own iterates) toward the truth.
    """
    torch.manual_seed(seed)
    if args.corrector == "deq":
        corrector = DEQCorrector(width=args.corrector_width, n_layers=args.corrector_layers)
    else:
        corrector = LocalCorrectionNet(width=args.corrector_width, n_layers=args.corrector_layers)
    corrector = corrector.to(device)

    # Pre-compute the frozen rasterised Transolver field + its grid tensors per case.
    log(f"seed {seed}: pre-computing {len(train_pairs)} rasterised backbone fields for corrector ...")
    samples = []
    for case, gt in train_pairs:
        if not predictor.has_cloud(case.name):
            continue
        field0 = predictor.predict(case)  # rasterised Transolver field (physical)
        inp_phys_np = encode_case(case)                    # (N_IN, H, W) physical
        geom_t = torch.from_numpy(
            np.ascontiguousarray(grid_norm.norm_in(inp_phys_np), DTYPE)
        )[None]
        inp_phys_t = torch.from_numpy(np.ascontiguousarray(inp_phys_np, DTYPE))[None]
        target_t = torch.from_numpy(
            np.ascontiguousarray(grid_norm.norm_out(gt.as_array()), DTYPE)
        )[None]
        field0_t = torch.from_numpy(
            np.ascontiguousarray(grid_norm.norm_out(field0.as_array()), DTYPE)
        )[None]
        mask_t = torch.from_numpy(
            np.ascontiguousarray((field0.mask[None] > 0.5).astype(DTYPE), DTYPE)
        )[None]
        samples.append(dict(case=case, field0=field0, field0_t=field0_t,
                            geom_t=geom_t, target_t=target_t, mask_t=mask_t,
                            inp_phys_t=inp_phys_t,
                            dx=float(field0.domain.dx), dy=float(field0.domain.dy)))
    if not samples:
        raise RuntimeError("no train cases had a matching point cloud for corrector training")

    opt = torch.optim.Adam(corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    unroll = max(int(args.unroll), 1)
    noise_std = float(args.noise_std)
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(args.corrector_epochs):
        corrector.train()
        agg, n = 0.0, 0
        for j in rng.permutation(len(samples)):
            s = samples[j]
            field0_t = s["field0_t"].to(device)
            geom_t = s["geom_t"].to(device)
            target_t = s["target_t"].to(device)
            fluid = s["mask_t"].to(device)
            inp_phys_t = s["inp_phys_t"].to(device)
            dx, dy = s["dx"], s["dy"]

            opt.zero_grad(set_to_none=True)
            field_t = field0_t
            step_loss = field0_t.new_tensor(0.0)
            ok = True
            for k in range(unroll):
                if noise_std > 0.0:
                    sigma = noise_std * (1.0 - k / unroll)
                    state_t = (field_t + sigma * torch.randn_like(field_t) * fluid).detach()
                else:
                    state_t = field_t.detach()
                # Residual computed from the CURRENT state each step (mirrors
                # Trainer._normalised_residual) so the corrector learns to RESPOND
                # to the residual, not a per-case-constant input.
                with torch.no_grad():
                    residual_t = _state_residual(state_t, inp_phys_t, dx, dy, nu, grid_norm)
                delta = corrector(field=state_t, residual=residual_t, geom=geom_t)
                target_delta = (target_t - state_t).detach()
                lk, _pc = per_channel_balanced_mse(delta, target_delta, fluid)
                if not torch.isfinite(lk):
                    ok = False
                    break
                step_loss = step_loss + lk
                field_t = (state_t + delta).detach()
            if not ok:
                continue
            loss = step_loss / unroll
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(corrector.parameters(), args.grad_clip)
            opt.step()
            agg += float(loss.detach())
            n += 1
        ep = agg / max(n, 1)
        history.append(ep)
        log(f"seed {seed} corrector epoch {epoch}: loss={ep:.4e}")
    corrector.eval()
    return corrector, history


# --------------------------------------------------------------------------- #
# Point-cloud MC-dropout UQ -> grid sigma (for conformal coverage)
# --------------------------------------------------------------------------- #
def conformal_coverage(predictor, pairs, n_mc, alpha, channel, seed) -> dict:
    """Split-conformal coverage of a point-cloud MC-dropout sigma on a grid channel.

    Runs the Transolver T times with dropout active, rasterises each prediction,
    and takes the per-cell std as sigma; then fits a :class:`ConformalCalibrator`
    on half the cases and reports empirical coverage + ECE on the other half
    (reusing the ``run_certificates`` H4 logic / ``calibration`` primitives).

    Requires backbone ``dropout > 0`` (otherwise sigma is degenerate); returns a
    ``{'status': 'n/a (no dropout)'}`` marker rather than faking a number.
    """
    from neuroforge.physics.calibration import ConformalCalibrator, reliability

    model = predictor.model
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    if not has_dropout:
        return {"status": "n/a (backbone has no dropout; set --dropout > 0)"}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    half = max(1, len(pairs) // 2)
    cal_pairs = [pairs[i] for i in idx[:half]]
    test_pairs = [pairs[i] for i in idx[half:]] or cal_pairs

    def err_sigma(case_pairs):
        errs, sigs, masks = [], [], []
        for case, gt in case_pairs:
            if not predictor.has_cloud(case.name):
                continue
            # MC-dropout: forward T times with dropout ON, rasterise, per-cell std.
            preds = []
            model.train()  # enable dropout layers
            for _ in range(n_mc):
                f = predictor.predict(case).as_array()[channel]
                preds.append(f)
            model.eval()
            arr = np.stack(preds)  # (T, H, W)
            mean = arr.mean(0)
            sig = arr.std(0)
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
        "status": "ok",
        "channel": int(channel),
        "alpha": float(alpha),
        "n_mc": int(n_mc),
        "q": float(cal.q),
        "coverage": float(cov),
        "target_coverage": float(1.0 - alpha),
        "ece": float(rel["ece"]),
        "n_cal": len(ce),
        "n_test": len(te),
    }


# --------------------------------------------------------------------------- #
# Aggregation + table I/O
# --------------------------------------------------------------------------- #
def _agg(dicts: list[dict]) -> dict[str, tuple[float, float]]:
    out = {}
    for m in _FIELD_METRICS:
        vals = [d[m] for d in dicts if m in d and np.isfinite(d.get(m, np.nan))]
        out[m] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
    return out


def _write_table(table: dict, out_dir: str, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    hdr = "| variant | " + " | ".join(_FIELD_METRICS) + " |"
    sep = "|" + "---|" * (len(_FIELD_METRICS) + 1)
    lines = [
        f"### NeuroForge v2 — Transolver backbone on AirfRANS/{meta['task']}, "
        f"{meta['n_seeds']} seed(s) (mean +/- std)",
        "",
        f"_Backbone: Transolver width={meta['width']} L={meta['n_layers']} "
        f"({meta['n_params']:,} params), trained {meta['epochs']} epochs on the "
        f"native point cloud. Corrector: {meta['corrector']}, "
        f"{meta['corrector_epochs']} epochs on the rasterised Transolver outputs' "
        f"residuals. Scored by the SHARED `evaluate_cases` (same rasterisation as "
        f"the GT grid). (a) = backbone alone; (b) = backbone + certified loop. The "
        f"LOCAL loop guarantees non-increasing physics-residual norm (not GT MSE); "
        f"the DEQ loop applies its fixed-point delta directly (no acceptance test)._",
        "", hdr, sep,
    ]
    for variant, mets in table.items():
        cells = [variant]
        for m in _FIELD_METRICS:
            mu, sd = mets[m]
            cells.append("n/a" if not np.isfinite(mu) else f"{mu:.4g}+/-{sd:.2g}")
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "v2_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"wrote {out_dir}/v2_table.md")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="NeuroForge v2: Transolver backbone + certified loop.")
    p.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--epochs", type=int, default=80, help="backbone epochs")
    p.add_argument("--corrector-epochs", type=int, default=20)
    p.add_argument("--n-points", type=int, default=16384, help="points/step for backbone training")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="results/v2")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-chunk", type=int, default=0, help=">0 = OOM-fallback chunked forward")
    # backbone (matched-budget defaults ~7-8M params)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-slices", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0, help=">0 enables MC-dropout conformal")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # corrector
    p.add_argument("--corrector", default="deq", choices=["deq", "local", "both"],
                   help="'both' trains+evals BOTH (local shows the no-harm acceptance, "
                        "deq is the mandated backbone corrector)")
    p.add_argument("--corrector-width", type=int, default=32)
    p.add_argument("--corrector-layers", type=int, default=3)
    p.add_argument("--unroll", type=int, default=3)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--max-iters", type=int, default=5, help="loop iters (local path)")
    # conformal
    p.add_argument("--conformal", action="store_true", help="run split-conformal coverage (needs dropout)")
    p.add_argument("--conformal-mc", type=int, default=16)
    p.add_argument("--conformal-alpha", type=float, default=0.1)
    p.add_argument("--conformal-channel", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="tiny-scale flags for the de-risk smoke")
    a = p.parse_args(argv)

    device = _resolve_device(a.device)
    log(f"device={device}")
    eval_chunk = a.eval_chunk if a.eval_chunk and a.eval_chunk > 0 else None
    corr_types = ["local", "deq"] if a.corrector == "both" else [a.corrector]

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
    # A representative nu for the residual/corrector training (per-case at solve).
    nu = float(train_pairs[0][0].fluid.kinematic_viscosity)
    checker = PhysicsChecker(Config().physics)

    os.makedirs(a.out_dir, exist_ok=True)
    backbone_seed_metrics: list[dict] = []
    loop_seed_metrics: dict[str, list[dict]] = {c: [] for c in corr_types}
    n_params = 0
    sec_epoch = float("nan")
    conformal_out: dict = {}

    for seed in a.seeds:
        log(f"================= seed {seed} =================")
        # --- Stage 1: backbone ---
        model, sec_epoch = train_transolver(a, seed, train_pcs, point_norm, device)
        n_params = model.num_params()

        # Predictors: TEST clouds for eval, TRAIN clouds for corrector training.
        test_pred = PointCloudPredictor(
            model, test_pcs, point_norm, grid_norm, device=device, chunk=eval_chunk
        )
        train_pred = PointCloudPredictor(
            model, train_pcs, point_norm, grid_norm, device=device, chunk=eval_chunk
        )

        # --- (a) backbone alone ---
        log(f"seed {seed}: evaluating (a) backbone alone ...")
        t0 = time.time()
        mets_a = evaluate_cases(test_pred.predict, test_pairs, checker=checker)
        log(f"seed {seed}: (a) eval {time.time()-t0:.1f}s -> "
            + ", ".join(f"{k}={mets_a.get(k, float('nan')):.4g}" for k in _FIELD_METRICS))
        backbone_seed_metrics.append(mets_a)

        # --- Stage 2 + (b) per corrector type ---
        for ctype in corr_types:
            a.corrector = ctype  # train_corrector reads a.corrector
            log(f"seed {seed}: training {ctype} corrector on Transolver residuals ...")
            corrector, _hist = train_corrector(
                a, seed, train_pred, train_pairs, grid_norm, nu, device
            )
            cfg = Config()
            cfg.correction.max_iters = a.max_iters
            engine = NeuroForgeEngine(test_pred, checker, corrector=corrector, config=cfg)

            log(f"seed {seed}: evaluating (b) backbone + {ctype} loop ...")
            t0 = time.time()

            def loop_predict(case, _eng=engine):
                return _eng.solve(case).field

            mets_b = evaluate_cases(loop_predict, test_pairs, checker=checker)
            log(f"seed {seed}: (b/{ctype}) eval {time.time()-t0:.1f}s -> "
                + ", ".join(f"{k}={mets_b.get(k, float('nan')):.4g}" for k in _FIELD_METRICS))
            loop_seed_metrics[ctype].append(mets_b)

            # No-harm / monotone-residual probe on a few cases (local path only).
            if ctype == "local" and seed == a.seeds[0]:
                probe = []
                for case, _gt in test_pairs[: min(4, len(test_pairs))]:
                    res = engine.solve(case)
                    h = res.history
                    if len(h) >= 1:
                        r0 = h[0]["residual_norm"]
                        rN = h[-1]["residual_norm"]
                        probe.append({"case": case.name, "r0": float(r0), "rN": float(rN),
                                      "non_increasing": bool(rN <= r0 + 1e-9),
                                      "n_iters": len(h)})
                conformal_out["no_harm_probe"] = probe
                log(f"no-harm probe (local): {probe}")

        a.corrector = "both" if len(corr_types) > 1 else corr_types[0]

        # --- conformal coverage (optional; needs dropout) ---
        if a.conformal:
            log(f"seed {seed}: split-conformal coverage ...")
            cov = conformal_coverage(
                test_pred, test_pairs, a.conformal_mc, a.conformal_alpha,
                a.conformal_channel, seed,
            )
            conformal_out[f"seed{seed}"] = cov
            log(f"seed {seed}: conformal -> {cov}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Incremental persistence after each seed.
        meta = dict(task=a.task, n_seeds=len(backbone_seed_metrics), width=a.width,
                    n_layers=a.n_layers, epochs=a.epochs,
                    corrector_epochs=a.corrector_epochs,
                    corrector="+".join(corr_types), n_params=n_params)
        table = {"(a) backbone alone": _agg(backbone_seed_metrics)}
        for ctype in corr_types:
            if loop_seed_metrics[ctype]:
                table[f"(b) backbone + {ctype} loop"] = _agg(loop_seed_metrics[ctype])
        _write_table(table, a.out_dir, meta)
        with open(os.path.join(a.out_dir, "v2_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "meta": meta,
                "backbone_per_seed": backbone_seed_metrics,
                "loop_per_seed": loop_seed_metrics,
                "conformal": conformal_out,
                "sec_per_epoch": sec_epoch,
            }, f, indent=2)

    log("================= SUMMARY =================")
    a_agg = _agg(backbone_seed_metrics)
    log(f"(a) backbone alone:    mse_p={a_agg['mse_p'][0]:.4g}, mse_u={a_agg['mse_u'][0]:.4g}, "
        f"rho_cl={a_agg['rho_cl'][0]:.3g}, res_err_spearman={a_agg['residual_error_spearman'][0]:.3g}")
    for ctype in corr_types:
        if loop_seed_metrics[ctype]:
            b_agg = _agg(loop_seed_metrics[ctype])
            log(f"(b) backbone + {ctype}:  mse_p={b_agg['mse_p'][0]:.4g}, mse_u={b_agg['mse_u'][0]:.4g}, "
                f"rho_cl={b_agg['rho_cl'][0]:.3g}")
    log(f"approx {sec_epoch:.1f}s/backbone-epoch at these settings.")
    log(f"artifacts in {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
