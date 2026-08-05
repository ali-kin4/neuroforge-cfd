"""Audit-driven retraining pilot -- Option A of docs/protocols/audit_loop_pilot.md.

Pre-registered design (protocol registered 2026-08-03, BEFORE any run; success
criteria fixed there and mirrored here -- do not tune after seeing results):

  Base pool  : AirfRANS `full` train split (800 cases used throughout the
               paper). Per seed s, a seeded permutation takes n_base=500 as
               the BASE training set; the remaining 300 are the ACQUISITION
               POOL.
  Base model : Transolver backbone, the exact w1/v2 recipe (7.35M params,
               80 epochs, width 256, 10 layers, dropout 0.05), trained on the
               500 base cases.
  Scoring    : the pool is scored with AUDIT INPUTS ONLY -- the base model's
               prediction, its physics residual, and its MC-dropout sigma.
               Pool ground truth is NEVER read during scoring; it is revealed
               only for the 100 acquired cases at retrain time.
                 residual score : PhysicsChecker(...).diagnose(pred).residual_norm()
                                  on the rasterised base-model prediction
                 sigma score    : mean point-level std of predicted speed over
                                  K=8 stochastic (dropout-on) forward passes
                 trust (fused)  : 0.5*(rank(residual) + rank(sigma))
  Arms (identical budget, +100 acquired, retrain FROM SCRATCH on 600,
        same epochs/hyperparameters/init seed -- arms differ only in data):
    A random-100    (baseline)
    B sigma-top-100 (UQ-acquisition baseline)
    C trust-top-100 (the paper's fused audit score -- the tested arm)
    D none          (control: the base-500 model itself, no retrain)
  Metrics    : primary  = test-set (200 held-out) mse_speed
               secondary = mse_u/v/p, residual_error_spearman
  Verdict (fixed in the protocol):
    POSITIVE : C beats A on the primary metric on both seeds, and C >= B on
               both (or beats B on mean with sign-consistency).
    PARTIAL  : C beats A but not B -> "trust matches UQ acquisition".
    NEGATIVE : C fails to beat A on both seeds -> scoped negative.

  Budget gate: run seed 0 first (--seeds 0, 4 trainings, ~6-7 h); add seed 1
  only if the single-seed result is promising (protocol's cost gate).

RESUMABLE: every training checkpoints per epoch (atomic) to
checkpoints/audit_pilot/seed{s}_{arm}.pt with a .done marker; pool scores are
cached to results/audit_pilot/seed{s}_pool_scores.json; re-running the same
command skips/resumes. Stop any time.

Run (seed-0 gate):
    .venv/Scripts/python.exe scripts/run_audit_loop_pilot.py --seeds 0
Then, only if promising:
    .venv/Scripts/python.exe scripts/run_audit_loop_pilot.py --seeds 0 1

Output: results/audit_pilot/audit_pilot.json (accumulates across seeds)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)
import numpy as np
import torch
from scipy.stats import rankdata

from neuroforge.core.config import Config
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.datamodule import FlowDataset, Normalizer
from neuroforge.data.pointcloud import F_IN, F_OUT, PointNormalizer, load_airfrans_pointclouds
from neuroforge.models.baselines.transolver import TransolverPointModel
from neuroforge.physics.evaluation import evaluate_cases
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor

N_BASE = 500   # protocol value; --n-base overrides for smoke tests ONLY
N_ACQ = 100    # protocol value; --n-acq overrides for smoke tests ONLY
K_DROPOUT = 8
ARMS = ("A_random", "B_sigma", "C_trust", "D_none")


def log(msg: str) -> None:
    print(f"[pilot] {msg}", flush=True)


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Training (w1/v2 recipe, per-epoch resumable -- adapted from run_ensemble_uq)
# --------------------------------------------------------------------------- #
def train_backbone(args, seed, tag, train_pcs, point_norm, device):
    """Train the w1-recipe Transolver on ``train_pcs``; resume from checkpoint.

    Init/order seeds depend on ``seed`` only (NOT the arm), so retrained arms
    share initialisation and differ only in their training data.
    """
    ckpt = os.path.join(args.ckpt_dir, f"seed{seed}_{tag}.pt")
    done = os.path.join(args.ckpt_dir, f"seed{seed}_{tag}.done")

    model = TransolverPointModel(
        in_features=F_IN, out_features=F_OUT, width=args.width,
        n_layers=args.n_layers, n_heads=args.n_heads, n_slices=args.n_slices,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = args.epochs * max(len(train_pcs), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1
    )

    if os.path.exists(done) and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        log(f"seed {seed} {tag}: already complete -- loaded checkpoint")
        return model

    start_epoch = 0
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = int(state["epoch"]) + 1
        ff = min(start_epoch * max(len(train_pcs), 1), max(steps, 1))
        for _ in range(ff):
            sched.step()
        rng = np.random.default_rng()
        try:
            rng.bit_generator.state = state["np_rng"]
        except Exception:
            rng = np.random.default_rng(500 + seed)
        log(f"seed {seed} {tag}: RESUMED from epoch {start_epoch}/{args.epochs}")
    else:
        torch.manual_seed(500 + seed)
        np.random.seed(500 + seed)
        rng = np.random.default_rng(500 + seed)
        log(f"seed {seed} {tag}: {model.num_params():,} params, "
            f"{len(train_pcs)} cases -- fresh start")

    n_pts = int(args.n_points)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = rng.permutation(len(train_pcs))
        agg, n = 0.0, 0
        t0 = time.time()
        for idx in order:
            pc = train_pcs[idx]
            mlen = pc.n_points
            sel = (rng.integers(0, mlen, size=min(n_pts, mlen))
                   if mlen > n_pts else np.arange(mlen))
            x = point_norm.transform_in(pc.features[sel])
            y = point_norm.transform_out(pc.targets[sel])
            xb = torch.from_numpy(x).to(device).unsqueeze(0)
            yb = torch.from_numpy(y).to(device).unsqueeze(0)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg += float(loss.detach())
            n += 1
        log(f"seed {seed} {tag} epoch {epoch}: train_mse={agg / max(n, 1):.4e} "
            f"({time.time() - t0:.1f}s)")
        _atomic_save({
            "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "np_rng": rng.bit_generator.state,
            "args": vars(args), "tag": tag, "seed": seed,
        }, ckpt)

    with open(done, "w", encoding="utf-8") as f:
        f.write(f"seed {seed} {tag} complete: {args.epochs} epochs, "
                f"{len(train_pcs)} cases\n")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Pool scoring -- audit inputs only (prediction, residual, dropout sigma)
# --------------------------------------------------------------------------- #
def score_pool(args, seed, model, pool_pcs, pool_cases, point_norm, grid_norm,
               device, checker):
    """Score every pool case WITHOUT its ground truth. Cached to JSON."""
    cache = os.path.join(args.out_dir, f"seed{seed}_pool_scores.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            d = json.load(f)
        if len(d["names"]) == len(pool_pcs):
            log(f"seed {seed}: pool scores cached ({len(d['names'])} cases)")
            return d

    predictor = PointCloudPredictor(model, pool_pcs, point_norm, grid_norm,
                                    device=device)
    names, residuals, sigmas = [], [], []
    t0 = time.time()
    for i, (pc, case) in enumerate(zip(pool_pcs, pool_cases)):
        # residual score on the rasterised deterministic prediction
        model.eval()
        pred = predictor.predict(case)
        residuals.append(float(checker.diagnose(pred, case).residual_norm()))

        # MC-dropout sigma at point level: std of predicted speed over K passes
        model.train()  # enables dropout; no grad
        x = point_norm.transform_in(pc.features)
        xb = torch.from_numpy(x).to(device).unsqueeze(0)
        speeds = []
        with torch.no_grad():
            for _ in range(K_DROPOUT):
                out = model(xb).squeeze(0).cpu().numpy()
                out = point_norm.inverse_out(out)
                speeds.append(np.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2))
        model.eval()
        sigmas.append(float(np.std(np.stack(speeds), axis=0, ddof=0).mean()))
        names.append(case.name)
        if (i + 1) % 50 == 0:
            log(f"  seed {seed}: scored {i + 1}/{len(pool_pcs)} pool cases "
                f"({(time.time() - t0) / (i + 1):.1f}s/case)")

    d = {"names": names, "residual": residuals, "sigma": sigmas,
         "note": "audit inputs only; pool GT never read here"}
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    log(f"seed {seed}: pool scoring done ({time.time() - t0:.0f}s) -> {cache}")
    return d


def select_arms(scores: dict, seed: int) -> dict:
    """Acquisition indices (into the pool) per arm, per the protocol."""
    res = np.asarray(scores["residual"], np.float64)
    sig = np.asarray(scores["sigma"], np.float64)
    fused = 0.5 * (rankdata(res) + rankdata(sig))
    n = res.size
    rng = np.random.default_rng(9000 + seed)  # arm-A draw, seeded & recorded
    return {
        "A_random": sorted(rng.permutation(n)[:N_ACQ].tolist()),
        "B_sigma": sorted(np.argsort(-sig, kind="mergesort")[:N_ACQ].tolist()),
        "C_trust": sorted(np.argsort(-fused, kind="mergesort")[:N_ACQ].tolist()),
    }


# --------------------------------------------------------------------------- #
# Verdict -- fixed mapping from the protocol
# --------------------------------------------------------------------------- #
def compute_verdict(per_seed: dict) -> dict:
    seeds = sorted(per_seed.keys())
    c_beats_a = [per_seed[s]["C_trust"]["mse_speed"] < per_seed[s]["A_random"]["mse_speed"]
                 for s in seeds]
    c_ge_b = [per_seed[s]["C_trust"]["mse_speed"] <= per_seed[s]["B_sigma"]["mse_speed"]
              for s in seeds]
    mean_c = float(np.mean([per_seed[s]["C_trust"]["mse_speed"] for s in seeds]))
    mean_b = float(np.mean([per_seed[s]["B_sigma"]["mse_speed"] for s in seeds]))
    if all(c_beats_a) and (all(c_ge_b) or mean_c < mean_b):
        verdict = "POSITIVE"
    elif any(c_beats_a):
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"
    return {
        "verdict": verdict if len(seeds) >= 2 else f"{verdict} (single-seed, gate only)",
        "n_seeds": len(seeds),
        "c_beats_a_per_seed": c_beats_a,
        "c_ge_b_per_seed": c_ge_b,
        "criteria": "POSITIVE: C<A both seeds AND (C<=B both OR mean(C)<mean(B)); "
                    "PARTIAL: C<A but not B; NEGATIVE: C fails vs A on both. "
                    "Pre-registered in docs/protocols/audit_loop_pilot.md.",
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    global N_BASE, N_ACQ
    p = argparse.ArgumentParser(description="Audit-driven retraining pilot (pre-registered).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="results/audit_pilot")
    p.add_argument("--ckpt-dir", default="checkpoints/audit_pilot")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--device", default="auto")
    # w1/v2 backbone recipe (results/control/w1_capture.json meta)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--n-points", type=int, default=16384)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-slices", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-base", type=int, default=N_BASE,
                   help="protocol: 500; override only for smoke tests")
    p.add_argument("--n-acq", type=int, default=N_ACQ,
                   help="protocol: 100; override only for smoke tests")
    args = p.parse_args(argv)
    N_BASE, N_ACQ = args.n_base, args.n_acq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    log(f"device={device}  seeds={args.seeds}  (resumable; re-run same command "
        "to continue)")

    # ---- data (shared across seeds) ----
    log("loading point clouds + grid pairs ...")
    train_pcs = load_airfrans_pointclouds(
        root=args.root, task="full", train=True, limit=args.n_train,
        cache_dir=args.cache_dir, download=False, progress=False,
    )
    test_pcs = load_airfrans_pointclouds(
        root=args.root, task="full", train=False, limit=args.n_val,
        cache_dir=args.cache_dir, download=False, progress=False,
    )
    train_pairs = load_airfrans(
        root=args.root, task="full", train=True, resolution=args.resolution,
        limit=args.n_train, cache_dir=args.cache_dir, download=False, progress=False,
    )
    test_pairs = load_airfrans(
        root=args.root, task="full", train=False, resolution=args.resolution,
        limit=args.n_val, cache_dir=args.cache_dir, download=False, progress=False,
    )
    train_cases = [c for c, _ in train_pairs]
    checker = PhysicsChecker(Config().physics)

    out_path = os.path.join(args.out_dir, "audit_pilot.json")
    results = {"meta": {
        "protocol": "docs/protocols/audit_loop_pilot.md (Option A, registered "
                    "2026-08-03)",
        "n_base": N_BASE, "n_acq": N_ACQ, "k_dropout": K_DROPOUT,
        "recipe": {k: getattr(args, k) for k in
                   ("epochs", "n_points", "width", "n_layers", "n_heads",
                    "n_slices", "dropout", "lr", "weight_decay")},
        "date_started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, "per_seed": {}}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
        results["per_seed"] = prev.get("per_seed", {})

    def evaluate(model, tag, seed) -> dict:
        predictor = PointCloudPredictor(model, test_pcs,
                                        point_norm, grid_norm, device=device)
        t0 = time.time()
        mets = evaluate_cases(predictor.predict, test_pairs, checker=checker)
        keep = {k: float(mets[k]) for k in
                ("mse_u", "mse_v", "mse_p", "mse_speed",
                 "residual_error_spearman") if k in mets}
        log(f"seed {seed} {tag}: eval {time.time() - t0:.0f}s -> "
            + ", ".join(f"{k}={v:.4g}" for k, v in keep.items()))
        return keep

    for seed in args.seeds:
        skey = str(seed)
        log(f"================= seed {seed} =================")
        # seeded base/pool split of the 800 train cases
        rng = np.random.default_rng(7000 + seed)
        perm = rng.permutation(len(train_pcs))
        base_idx = sorted(perm[:N_BASE].tolist())
        pool_idx = sorted(perm[N_BASE:].tolist())
        base_pcs = [train_pcs[i] for i in base_idx]
        pool_pcs = [train_pcs[i] for i in pool_idx]
        pool_cases = [train_cases[i] for i in pool_idx]

        # normalizers are fit on the BASE set only (no pool leakage)
        point_norm = PointNormalizer().fit(base_pcs)
        ds = FlowDataset([train_pairs[i] for i in base_idx], normalizer=None)
        grid_norm = Normalizer().fit(*ds.raw_arrays())

        seed_out = results["per_seed"].get(skey, {})
        seed_out["base_idx_sha"] = hash(tuple(base_idx)) & 0xFFFFFFFF

        # ---- D (control) = base-500 model ----
        base_model = train_backbone(args, seed, "base", base_pcs, point_norm, device)
        if "D_none" not in seed_out:
            seed_out["D_none"] = evaluate(base_model, "D_none(base500)", seed)
            results["per_seed"][skey] = seed_out
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        # ---- score pool with audit inputs only ----
        scores = score_pool(args, seed, base_model, pool_pcs, pool_cases,
                            point_norm, grid_norm, device, checker)
        arms_sel = select_arms(scores, seed)
        seed_out["acquisition"] = {
            arm: {"pool_indices": sel,
                  "case_names": [pool_cases[i].name for i in sel]}
            for arm, sel in arms_sel.items()
        }
        overlap_bc = len(set(arms_sel["B_sigma"]) & set(arms_sel["C_trust"]))
        seed_out["acquisition"]["overlap_B_C"] = overlap_bc
        log(f"seed {seed}: arm B/C acquisition overlap {overlap_bc}/{N_ACQ}")

        # ---- retrain arms from scratch on base + acquired ----
        for arm in ("A_random", "B_sigma", "C_trust"):
            if arm in seed_out and "mse_speed" in seed_out.get(arm, {}):
                log(f"seed {seed} {arm}: already evaluated -- skip")
                continue
            acq_pcs = [pool_pcs[i] for i in arms_sel[arm]]
            arm_model = train_backbone(args, seed, arm, base_pcs + acq_pcs,
                                       point_norm, device)
            seed_out[arm] = evaluate(arm_model, arm, seed)
            results["per_seed"][skey] = seed_out
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            del arm_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        results["per_seed"][skey] = seed_out
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # ---- verdict over completed seeds ----
    complete = {s: v for s, v in results["per_seed"].items()
                if all(a in v and "mse_speed" in v[a] for a in ARMS)}
    if complete:
        results["verdict"] = compute_verdict(complete)
        log(f"VERDICT ({len(complete)} seed(s)): {results['verdict']['verdict']}")
        for s, v in sorted(complete.items()):
            log(f"  seed {s}: " + "  ".join(
                f"{a}={v[a]['mse_speed']:.4f}" for a in ARMS))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
