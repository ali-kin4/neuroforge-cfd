"""DeepCFD second-dataset experiment: does the physics-residual TRUST SIGNAL
generalise beyond AirfRANS?  (FNO backbone, 3 seeds, FULLY RESUMABLE.)

Why this run exists
-------------------
The surviving NeuroForge contribution is: the per-case Spearman correlation
between the field-mean PDE residual norm and the true rel-L2 speed error is
**positive** -- a backbone-agnostic *trust signal*. So far that is only shown on
AirfRANS (turbulent external aero over airfoils). This run tests it on **DeepCFD**
(Ribeiro et al. 2020): 2-D **laminar incompressible** channel flow over bluff
obstacles -- a genuinely different geometry AND regime. The metric is computed by
the SHARED ``evaluate_cases(..., checker=)`` with the IDENTICAL ``PhysicsChecker``
residual operator (here in its ``include_bc=False`` interior-only mode, the
apples-to-apples cross-dataset verifier -- see the note below).

Backbone choice: **FNO** (grid-native), trained on the structured DeepCFD grid
with plain masked-fluid data MSE (NO physics-loss term -- training against the
residual would suppress the very quantity we correlate and confound the result).

"Same verifier" note
--------------------
``Diagnostics.residual_norm()`` (frozen ``core/types.py``) is ALREADY the
interior-only continuity+momentum norm; ``evaluate_cases`` feeds the spearman
from it. So ``residual_error_spearman`` is the identical scalar function on
AirfRANS and DeepCFD regardless of ``include_bc``. We still pass
``include_bc=False`` so the (inapplicable) immersed-body BC term is never even
computed against the channel walls/inlet/outlet -- it only affects the trust map,
not the headline metric.

MANDATORY OOD split (the binding risk)
--------------------------------------
DeepCFD's pickles carry NO shape-family label, so we split by a COMPUTABLE
geometry band: obstacle pixel area. We TRAIN on the smaller bodies (area below
the ``--ood-percentile`` quantile) and TEST on the **held-out largest** bodies
(area >= that quantile). Holding out the EXTREME band forces the FNO to
extrapolate to the biggest wakes/blockages, producing a RANGE of per-case errors
-- without spread the Spearman is degenerate/NaN. The script reports the per-case
rel-L2 spread (min/median/max) so the spread is visible.

RESUME MODEL (survives shutdown / restart / sleep)
--------------------------------------------------
Each seed's full training state (model + opt + RNG + epoch) is checkpointed to
``<ckpt-dir>/seed{m}.pt`` after EVERY epoch (atomic tmp + os.replace). On launch
the script SKIPS finished seeds (``seed{m}.done`` marker), RESUMES an in-progress
seed from its last epoch (OneCycleLR position rebuilt by FAST-FORWARDING, robust
to a changed --epochs), and STARTS fresh seeds otherwise. Stop ANY time and re-run
the EXACT SAME command to continue.

Always import ``neuroforge`` before numpy/torch (BLAS thread cap) -- done at top.

SMOKE (tiny; also verifies resume -- run it twice and watch it resume). NOTE:
``--smoke`` is a label only; the smoke behaviour comes from the small flags and
the explicit ``_smoke`` out/ckpt dirs you pass -- ALWAYS pass ``_smoke`` dirs so
a smoke never overwrites the real ``results/deepcfd`` / ``checkpoints/deepcfd``:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_deepcfd.py \
        --n-train 16 --n-test 8 --epochs 2 --seeds 1 \
        --width 8 --n-layers 2 --modes 8 --resolution 64 \
        --out-dir results/deepcfd_smoke --ckpt-dir checkpoints/deepcfd_smoke \
        --device cpu --smoke

FULL run (3 seeds, OOD area split, resumable across reboots). At --ood-percentile
75 the area bands are train_pool=734 (area<p75), test_pool=247 (area>=p75, held
out); n_test caps to 247 even if asked for more:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_deepcfd.py \
        --n-train 700 --n-test 247 --epochs 100 --seeds 3 \
        --width 32 --n-layers 4 --modes 24 --resolution 128 \
        --ood-percentile 75 \
        --out-dir results/deepcfd --ckpt-dir checkpoints/deepcfd \
        --device cuda
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
from neuroforge.data.datamodule import FlowDataset, Normalizer, collate_flow
from neuroforge.data.deepcfd_loader import load_deepcfd, obstacle_areas
from neuroforge.models.fno import FNO2d
from neuroforge.physics.evaluation import evaluate_cases
from neuroforge.physics.metrics import field_errors
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import Predictor
from torch.utils.data import DataLoader

_METRICS = [
    "mse_u", "mse_v", "mse_p", "mse_speed", "surface_mse_p",
    "rho_cl", "rho_cd", "residual_error_spearman", "residual_error_pearson",
]


def log(msg: str) -> None:
    # Same "[v2] ..." grammar the live dashboard parses (seed index).
    print(f"[v2] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


# --------------------------------------------------------------------------- #
# OOD split by obstacle area (hold out the largest bodies)
# --------------------------------------------------------------------------- #
def build_ood_split(root, n_train, n_test, ood_percentile, seed):
    """Return (train_idx, test_idx) by obstacle-area band.

    TRAIN = smaller bodies (area < the ``ood_percentile`` quantile); TEST =
    held-out largest bodies (area >= that quantile). ``n_train``/``n_test`` cap
    each band (shuffled with ``seed`` for the cap, but the BAND membership is
    fixed by area so the OOD held-out family is deterministic).
    """
    areas = obstacle_areas(root)
    n = len(areas)
    thresh = np.percentile(areas, ood_percentile)
    train_pool = np.where(areas < thresh)[0]
    test_pool = np.where(areas >= thresh)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    train_idx = train_pool[: min(n_train, len(train_pool))]
    test_idx = test_pool[: min(n_test, len(test_pool))]
    return (
        sorted(int(i) for i in train_idx),
        sorted(int(i) for i in test_idx),
        float(thresh),
        (int(areas.min()), int(np.median(areas)), int(areas.max())),
    )


# --------------------------------------------------------------------------- #
# Train ONE seed (grid FNO, plain masked MSE), per-epoch resumable checkpoint
# --------------------------------------------------------------------------- #
def train_seed(args, m, train_loader, normalizer, device, ckpt_dir):
    """Train seed ``m`` to completion, resuming from its checkpoint if present."""
    ckpt = os.path.join(ckpt_dir, f"seed{m}.pt")
    done = os.path.join(ckpt_dir, f"seed{m}.done")

    model = FNO2d(
        in_channels=Config().model.in_channels,
        out_channels=Config().model.out_channels,
        width=args.width, n_layers=args.n_layers, modes=args.modes,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(len(train_loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1
    )

    # ---- already finished? load weights and return (instant skip) ----
    if os.path.exists(done) and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        log(f"seed {m}: already complete ({args.epochs} epochs) — loaded checkpoint")
        return model

    # ---- resume in-progress seed, or start fresh ----
    start_epoch = 0
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = int(state["epoch"]) + 1
        # Rebuild OneCycle position by FAST-FORWARDING (robust to --epochs change).
        _ff = min(start_epoch * max(len(train_loader), 1), max(steps, 1))
        for _ in range(_ff):
            sched.step()
        try:
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        except Exception:
            pass
        log(f"seed {m}: RESUMED from epoch {start_epoch}/{args.epochs}")
    else:
        torch.manual_seed(1000 + m)
        np.random.seed(1000 + m)
        log(f"seed {m}: {model.num_params():,} params — fresh start")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        agg, n = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            fluid = (mask > 0.5).to(inp.dtype)
            n_fluid = fluid.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            pred = model(inp)
            # Plain masked-fluid data MSE (NO physics term -- keep the residual
            # honest for the trust-signal measurement).
            loss = ((pred - target) ** 2 * fluid).sum() / (n_fluid * pred.shape[1])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            agg += float(loss.detach())
            n += 1
        dt = time.time() - t0
        log(f"seed {m} backbone epoch {epoch}: train_mse={agg/max(n,1):.4e} ({dt:.1f}s)")

        _atomic_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
            "args": vars(args),
        }, ckpt)

    with open(done, "w", encoding="utf-8") as f:
        f.write(f"seed {m} complete: {args.epochs} epochs\n")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Per-case rel-L2 speed-error spread (the degeneracy read)
# --------------------------------------------------------------------------- #
def per_case_speed_error(predict_fn, pairs) -> list[float]:
    """Masked rel-L2 speed error per case (same formula as evaluate_cases)."""
    out = []
    for case, ref in pairs:
        pred = predict_fn(case)
        mask = ref.mask if ref.mask is not None else pred.mask
        m = (np.asarray(mask) > 0.5) if mask is not None else np.ones(ref.shape, bool)
        if not m.any():
            m = np.ones(ref.shape, bool)
        num = float(np.sqrt(np.sum(((pred.speed() - ref.speed())[m]) ** 2)))
        den = float(np.sqrt(np.sum((ref.speed()[m]) ** 2))) + 1e-12
        out.append(num / den)
    return out


def _agg(per_seed: list[dict]) -> dict[str, dict]:
    out = {}
    for k in _METRICS:
        vals = [d[k] for d in per_seed if k in d and np.isfinite(d.get(k, np.nan))]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}
        else:
            out[k] = {"mean": float("nan"), "std": float("nan")}
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DeepCFD trust-signal experiment (FNO, resumable, 3 seeds).")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..seeds-1)")
    p.add_argument("--n-train", type=int, default=700)
    p.add_argument("--n-test", type=int, default=281)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--root", default="data/deepcfd")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="results/deepcfd")
    p.add_argument("--ckpt-dir", default="checkpoints/deepcfd")
    p.add_argument("--device", default="auto")
    p.add_argument("--ood-percentile", type=float, default=75.0,
                   help="train on bodies with area < this area-percentile; test "
                        "on the held-out largest bodies (>= percentile).")
    p.add_argument("--split-seed", type=int, default=0,
                   help="seed for the within-band shuffle (band membership is "
                        "fixed by area; this only affects the cap subset).")
    # backbone
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--modes", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)

    device = _resolve_device(a.device)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    log(f"device={device}  seeds={a.seeds}  width={a.width}  n_layers={a.n_layers}  "
        f"modes={a.modes}  res={a.resolution}  ckpt-dir={a.ckpt_dir}")
    log("RESUMABLE: stop any time (Stop / Ctrl-C / shutdown); re-run the SAME "
        "command to continue from the last saved epoch.")

    # ---- OOD area split ----
    train_idx, test_idx, thresh, area_stats = build_ood_split(
        a.root, a.n_train, a.n_test, a.ood_percentile, a.split_seed
    )
    log(f"OOD split (hold out largest bodies): area threshold={thresh:.0f}px "
        f"(area min/med/max over all 981 = {area_stats}); "
        f"n_train={len(train_idx)} (small/medium), n_test={len(test_idx)} (large, held-out)")

    # ---- data ----
    log("building DeepCFD train pairs (resampled to grid) ...")
    train_pairs = load_deepcfd(
        root=a.root, resolution=a.resolution, indices=train_idx,
        cache_dir=a.cache_dir, progress=False,
    )
    log("building DeepCFD test pairs ...")
    test_pairs = load_deepcfd(
        root=a.root, resolution=a.resolution, indices=test_idx,
        cache_dir=a.cache_dir, progress=False,
    )

    # ---- normaliser fitted on train; datasets + loader ----
    train_ds = FlowDataset(train_pairs, normalizer=None)
    xi, yi = train_ds.raw_arrays()
    normalizer = Normalizer().fit(xi, yi)
    train_ds.normalizer = normalizer
    train_loader = DataLoader(
        train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_flow, drop_last=False,
    )

    # Interior-only verifier (apples-to-apples cross-dataset trust signal).
    checker = PhysicsChecker(Config().physics, include_bc=False)

    n_params = FNO2d(
        in_channels=Config().model.in_channels, out_channels=Config().model.out_channels,
        width=a.width, n_layers=a.n_layers, modes=a.modes,
    ).num_params()
    log(f"FNO params = {n_params:,}")

    per_seed: list[dict] = []
    spread_per_seed: list[dict] = []
    for m in range(a.seeds):
        log(f"================= seed {m}/{a.seeds - 1} =================")
        model = train_seed(a, m, train_loader, normalizer, device, a.ckpt_dir)

        predictor = Predictor(model, normalizer, device=str(device))
        t0 = time.time()
        mets = evaluate_cases(predictor.predict, test_pairs, checker=checker)
        # Per-case rel-L2 speed-error spread (degeneracy read).
        errs = per_case_speed_error(predictor.predict, test_pairs)
        errs_arr = np.asarray(errs, np.float64)
        spread = {
            "min": float(errs_arr.min()), "median": float(np.median(errs_arr)),
            "max": float(errs_arr.max()), "mean": float(errs_arr.mean()),
            "std": float(errs_arr.std()), "n": int(errs_arr.size),
        }
        log(f"seed {m} eval {time.time()-t0:.1f}s -> "
            + ", ".join(f"{k}={mets.get(k, float('nan')):.4g}" for k in _METRICS))
        log(f"seed {m} rel-L2 speed-error spread: min={spread['min']:.3f} "
            f"median={spread['median']:.3f} max={spread['max']:.3f} "
            f"(mean={spread['mean']:.3f} std={spread['std']:.3f})")
        per_seed.append(mets)
        spread_per_seed.append(spread)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Incremental persistence after EACH seed (crash-safe).
        out = {
            "meta": {
                "model": "fno", "dataset": "deepcfd", "regime": "2D laminar incompressible",
                "seeds": a.seeds, "epochs": a.epochs, "width": a.width,
                "n_layers": a.n_layers, "modes": a.modes, "n_params": int(n_params),
                "n_train": len(train_idx), "n_test": len(test_idx),
                "resolution": a.resolution, "nu": 1.0e-4, "u_inf": 0.1,
                "domain_m": [0.26, 0.12], "include_bc": False,
                "ood_split": {
                    "type": "obstacle-area band (hold out largest bodies)",
                    "ood_percentile": a.ood_percentile,
                    "area_threshold_px": thresh,
                    "area_min_med_max_all": list(area_stats),
                },
            },
            "per_seed": [
                {"seed": i, "metrics": d, "rel_l2_speed_spread": s}
                for i, (d, s) in enumerate(zip(per_seed, spread_per_seed))
            ],
            "aggregate": _agg(per_seed),
        }
        with open(os.path.join(a.out_dir, "deepcfd_results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log(f"wrote {a.out_dir}/deepcfd_results.json ({len(per_seed)}/{a.seeds} seeds)")

    agg = _agg(per_seed)
    log("FINAL (mean +/- std across seeds): "
        + ", ".join(f"{k}={agg[k]['mean']:.4g}+/-{agg[k]['std']:.2g}" for k in _METRICS))
    rs = agg["residual_error_spearman"]
    log(f"==> HEADLINE residual_error_spearman (interior-only) = "
        f"{rs['mean']:.3f} +/- {rs['std']:.3f}  "
        f"(positive => trust signal generalises to DeepCFD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
