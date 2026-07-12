"""MeshGraphNet (Pfaff et al. 2021) backbone on AirfRANS — 3 seeds, FULLY RESUMABLE.

Why this run exists
-------------------
This adds the THIRD backbone family to the NeuroForge comparison: a
message-passing graph network (MeshGraphNet), complementing Transolver (attention
/ slice-token) and the FNO (spectral). It is trained on the AirfRANS NATIVE point
clouds and scored with the **same** ``evaluate_cases`` metrics as NeuroForge
(via a rasterise adapter byte-identical to the GT projection), so the comparison
is apples-to-apples. The graph is a fixed-``k`` kNN over the cloud (mesh-free);
the architecture is faithful Encode-Process-Decode (see
``models/baselines/meshgraphnet.py``).

The headline NeuroForge metric is ``residual_error_spearman`` (does a low PDE
residual track low field error — the trust-signal claim), computed per seed with
the shared ``PhysicsChecker``.

RESUME MODEL (survives shutdown / restart / sleep)
--------------------------------------------------
Each seed's full training state (model + optimizer + scheduler-position + epoch +
RNG) is checkpointed to ``<ckpt-dir>/seed{m}.pt`` after EVERY epoch with an
atomic write (tmp + os.replace). On launch the script:
  * SKIPS seeds already finished (``seed{m}.done`` marker present),
  * RESUMES an in-progress seed from its last saved epoch (OneCycleLR position is
    rebuilt by fast-forwarding, NOT load_state_dict, since total_steps is pinned
    at creation and is robust to changing --epochs between runs),
  * STARTS fresh seeds otherwise.
Stop ANY time and re-run the EXACT SAME command to continue.

Always import ``neuroforge`` before numpy/torch (BLAS thread cap) -- done at top.

SMOKE (tiny; also verifies resume — run it twice and watch it resume):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_mgn.py --task full \
        --n-train 8 --n-val 4 --epochs 2 --seeds 2 --n-points 1024 \
        --width 32 --n-layers 3 --k 6 --resolution 128 \
        --cache-dir data/cache --out-dir results/mgn_smoke \
        --ckpt-dir checkpoints/mgn_smoke --device cpu --smoke

FULL run (matched budget 7,351,214 params vs 7,350,420 anchor = 0.01%; 3 seeds;
resumable across reboots). ``--eval-edge-chunk`` (default 500000) keeps the
whole-cloud (~180k pt) eval within 12GB; see its --help and the memory note in
``models/baselines/meshgraphnet.py``::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_mgn.py --task full \
        --n-train 800 --n-val 200 --epochs 80 --seeds 3 \
        --n-points 16384 --width 290 --n-layers 12 --k 8 --resolution 128 \
        --grad-checkpoint --eval-edge-chunk 500000 \
        --cache-dir data/cache --out-dir results/mgn --ckpt-dir checkpoints/mgn

MEMORY (read before the full GPU run)
-------------------------------------
* EVAL: bounded by ``--eval-edge-chunk`` (chunked edge update is EXACT for MGN).
* TRAIN: NOT chunkable (autograd needs the activations). At width=290 / n_points
  16384 a NON-checkpointed pass stores ~4 (E,W) activations per block over 12
  blocks (E ~= 14*N ~= 0.23M) ~= 13GB fp32 before backward transients — MGN is
  edge-centric so its activation memory is several x Transolver's at matched
  params, and Transolver only just fits 12GB. The FULL command therefore passes
  ``--grad-checkpoint``: each GN block is wrapped in ``torch.utils.checkpoint`` so
  only its (h,e) inputs are kept and inner activations are recomputed in backward.
  This is numerically EXACT (fp32, no AMP; outputs+grads unchanged — verified in
  tests/test_meshgraphnet.py) and drops the dominant per-block activation term,
  bringing the estimate to ~3-4GB (fits 12GB with headroom) at ~1.3x compute. This
  script has no auto-halve (it is not the NeuroForge trainer).
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
from neuroforge.core.types import FlowCase, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.pointcloud import F_IN, F_OUT, PointNormalizer, load_airfrans_pointclouds
from neuroforge.data.rasterize import rasterize_point_cloud
from neuroforge.geometry.sdf import signed_distance, solid_mask
from neuroforge.models.baselines.meshgraphnet import (
    MeshGraphNetPointModel,
    edge_features,
    knn_edges,
)
from neuroforge.physics.evaluation import evaluate_cases
from neuroforge.physics.residuals import PhysicsChecker

# Metrics aggregated across seeds (mean +/- std, population ddof=0).
_METRICS = [
    "mse_u", "mse_v", "mse_p", "rho_cl", "rho_cd",
    "cl_rel_err_mean", "cd_rel_err_mean", "residual_error_spearman",
]


def log(msg: str) -> None:
    # Same "[v2] ..." grammar the live dashboard already parses (seed index).
    print(f"[v2] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _atomic_save(obj, path: str) -> None:
    """Write a checkpoint atomically so an interrupted save never corrupts it."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def _length_scale(point_norm: PointNormalizer) -> tuple[torch.Tensor, float]:
    """Edge-length scale from the train PointNormalizer (raw x,y std = std_in[:2])."""
    l_xy = torch.as_tensor(point_norm.std_in[:2], dtype=torch.float32)
    l_norm = float(torch.linalg.norm(l_xy).item()) or 1.0
    return l_xy, l_norm


# --------------------------------------------------------------------------- #
# Train ONE seed, with per-epoch resumable checkpointing
# --------------------------------------------------------------------------- #
def train_seed(args, m, train_pcs, point_norm, device, ckpt_dir):
    """Train seed ``m`` to completion, resuming from its checkpoint if present.

    Returns the trained model (eval mode). Checkpoints every epoch; writes a
    ``seed{m}.done`` marker when finished so a re-run skips it instantly.
    """
    ckpt = os.path.join(ckpt_dir, f"seed{m}.pt")
    done = os.path.join(ckpt_dir, f"seed{m}.done")

    model = MeshGraphNetPointModel(
        in_features=F_IN, out_features=F_OUT, width=args.width, n_layers=args.n_layers,
        edge_in=3, aggr=args.aggr, grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(len(train_pcs), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1
    )

    l_xy, l_norm = _length_scale(point_norm)

    # ---- already finished? load weights and return (instant resume) ----
    if os.path.exists(done) and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        log(f"seed {m}: already complete ({args.epochs} epochs) — loaded checkpoint")
        return model

    # ---- resume in-progress seed, or start fresh ----
    start_epoch = 0
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = int(state["epoch"]) + 1
        # Rebuild the OneCycle position by FAST-FORWARDING (not load_state_dict):
        # OneCycleLR pins total_steps at creation, so loading a state saved under a
        # different --epochs raises; fast-forwarding is robust to changing --epochs.
        _ff = min(start_epoch * max(len(train_pcs), 1), max(steps, 1))
        for _ in range(_ff):
            sched.step()
        rng = np.random.default_rng()
        try:
            rng.bit_generator.state = state["np_rng"]
        except Exception:
            rng = np.random.default_rng(1000 + m)
        log(f"seed {m}: RESUMED from epoch {start_epoch}/{args.epochs}")
    else:
        # Distinct seed per member -> independent seeds.
        torch.manual_seed(1000 + m)
        np.random.seed(1000 + m)
        rng = np.random.default_rng(1000 + m)
        log(f"seed {m}: {model.num_params():,} params — fresh start")

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
            x = point_norm.transform_in(pc.features[sel])              # (n, F_IN) normalised
            y = point_norm.transform_out(pc.targets[sel])             # (n, F_OUT) normalised
            pos = torch.from_numpy(np.asarray(pc.pos[sel], np.float32)).to(device)  # RAW pos
            ei = knn_edges(pos, k=args.k, chunk=args.knn_chunk)
            ea = edge_features(pos, ei, l_xy, l_norm)
            xb = torch.from_numpy(x).to(device).unsqueeze(0)
            yb = torch.from_numpy(y).to(device).unsqueeze(0)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, ei, ea)
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
        # Dashboard-parseable line ("seed" = seed index).
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

    with open(done, "w", encoding="utf-8") as f:
        f.write(f"seed {m} complete: {args.epochs} epochs\n")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Eval closure — MGN-on-cloud + the byte-identical rasterise tail
# --------------------------------------------------------------------------- #
def make_mgn_predict_fn(model, pointclouds, point_norm, device, args):
    """``FlowCase -> FlowField`` predictor: MGN per-point + rasterise (fair eval).

    Builds the kNN graph from the FULL cloud's RAW ``.pos`` (fixed-``k``, chunked),
    forwards the MGN with memory-bounded edge chunks, then rasterises predictions
    onto ``case.domain`` with the SAME projection + solid-zeroing as the GT loader
    and the Transolver adapter (the fairness keystone — replicated verbatim).
    """
    model = model.to(device)
    model.eval()
    by_name = {pc.name: pc for pc in pointclouds}
    l_xy, l_norm = _length_scale(point_norm)
    edge_chunk = args.eval_edge_chunk if args.eval_edge_chunk and args.eval_edge_chunk > 0 else None
    # The whole-cloud eval kNN (cdist over ~180k pts) builds a (chunk, N) distance
    # tensor; with the training chunk (4096) that is ~2.9GB/case, and because test
    # clouds vary in size (158k-203k pts) the CUDA caching allocator fragments and
    # its RESERVED pool grows past 12GB across cases -> driver thrash -> eval slows
    # from ~7s to minutes/case (a 200-case eval stalled for 10h). Fix: a small eval
    # kNN chunk (bounds the temp) AND empty_cache() per case (defragments). Both are
    # numerically exact (kNN is chunk-invariant). Measured: stable ~7s/case, ~23min.
    eval_knn_chunk = max(int(getattr(args, "eval_knn_chunk", 512)), 1)
    on_cuda = device.type == "cuda"
    n_done = [0]

    def predict_fn(case: FlowCase) -> FlowField:
        pc = by_name.get(case.name)
        if pc is None:
            raise KeyError(
                f"no point cloud for case '{case.name}'. Point clouds and GT pairs "
                "must come from the same af.dataset.load order/task/split."
            )
        feats = point_norm.transform_in(pc.features)  # (M, F_IN)
        pos = torch.from_numpy(np.asarray(pc.pos, np.float32)).to(device)
        with torch.no_grad():
            ei = knn_edges(pos, k=args.k, chunk=eval_knn_chunk)
            ea = edge_features(pos, ei, l_xy, l_norm)
            xb = torch.from_numpy(feats).to(device).unsqueeze(0)
            out = model(xb, ei, ea, edge_chunk=edge_chunk).squeeze(0)
            preds = point_norm.inverse_out(out).detach().cpu().numpy().astype(np.float32)  # (M,4) physical
        # Release this case's GPU tensors and DEFRAGMENT before the next (differently
        # sized) cloud, so the reserved pool can't creep past VRAM across cases.
        del pos, ei, ea, xb, out
        if on_cuda:
            torch.cuda.empty_cache()
        n_done[0] += 1
        if n_done[0] % 25 == 0:
            log(f"  eval progress: {n_done[0]} cases scored")

        domain = case.domain
        raster = rasterize_point_cloud(pc.pos, preds, domain, fill=0.0, method="linear")
        sdf = signed_distance(case.geometry, domain)
        mask = solid_mask(case.geometry, domain)
        solid = mask < 0.5
        u = np.where(solid, 0.0, raster[0])
        v = np.where(solid, 0.0, raster[1])
        p = raster[2]
        nut = np.where(solid, 0.0, np.maximum(raster[3], 0.0))
        return FlowField(
            domain=domain, u=u, v=v, p=p, nut=nut, mask=mask, sdf=sdf,
            meta={"source": "meshgraphnet_baseline", "case": case.name},
        )

    return predict_fn


def _agg(per_seed: list[dict]) -> dict[str, dict]:
    """Mean +/- std (population, ddof=0) across seeds for each reported metric."""
    out = {}
    for k in _METRICS:
        vals = [d[k] for d in per_seed if k in d and np.isfinite(d[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}
        else:
            out[k] = {"mean": float("nan"), "std": float("nan")}
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MeshGraphNet backbone on AirfRANS (resumable, 3 seeds).")
    p.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..seeds-1)")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--n-points", type=int, default=16384)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="results/mgn")
    p.add_argument("--ckpt-dir", default="checkpoints/mgn")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="auto")
    # backbone (matched-budget defaults ~7.35M params)
    p.add_argument("--width", type=int, default=290)
    p.add_argument("--n-layers", type=int, default=12, help="message-passing steps")
    p.add_argument("--k", type=int, default=8, help="kNN neighbours per node")
    p.add_argument("--aggr", default="sum", choices=["sum", "mean"])
    p.add_argument("--knn-chunk", type=int, default=4096, help="train kNN query block size")
    p.add_argument("--eval-knn-chunk", type=int, default=512,
                   help="eval kNN query block size for the whole-cloud (~180k pt) "
                        "graph. Small (512) on purpose: bounds the (chunk,N) cdist "
                        "temp so the CUDA allocator can't fragment past VRAM across "
                        "the variable-size test clouds (the 10h-stall fix). Exact "
                        "(kNN is chunk-invariant).")
    p.add_argument("--eval-edge-chunk", type=int, default=500000,
                   help="edge-block size for the whole-cloud eval edge update. The "
                        "default (500k) bounds eval memory to ~one (E,W) tensor: at "
                        "the full width=290 a dense pass over E~2.9M edges holds "
                        "several (E,290) tensors (~3.4GB each) and OOMs a 12GB GPU. "
                        "Chunking is EXACT for MGN (verified chunk==dense to 1e-7). "
                        "Set 0 to force the dense path (small clouds / big GPU only).")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="activation/gradient checkpointing over the processor blocks "
                        "(numerically exact; ~1.3x compute for a large training-memory "
                        "reduction). Recommended for the full GPU run so width=290 / "
                        "n_points=16384 fits 12GB; off by default (smoke / small).")
    p.add_argument("--dropout", type=float, default=0.0)  # accepted for CLI parity; MGN has none
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)

    device = _resolve_device(a.device)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    log(f"device={device}  seeds={a.seeds}  width={a.width}  n_layers={a.n_layers}  k={a.k}  ckpt-dir={a.ckpt_dir}")
    log("RESUMABLE: stop any time (Stop / Ctrl-C / shutdown); re-run the SAME "
        "command to continue from the last saved epoch.")

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
    test_pairs = load_airfrans(
        root=a.root, task=a.task, train=False, resolution=a.resolution,
        limit=a.n_val, cache_dir=a.cache_dir, download=a.download, progress=False,
    )

    # Sanity: names align between point clouds and GT pairs.
    pc_names = {pc.name for pc in test_pcs}
    missing = [c.name for c, _ in test_pairs if c.name not in pc_names]
    if missing:
        raise RuntimeError(
            f"{len(missing)} GT test cases have no matching point cloud "
            f"(e.g. {missing[:3]}). Use the same task/split/limit."
        )

    point_norm = PointNormalizer().fit(train_pcs)
    checker = PhysicsChecker(Config().physics)
    n_params = MeshGraphNetPointModel(
        in_features=F_IN, out_features=F_OUT, width=a.width, n_layers=a.n_layers
    ).num_params()
    log(f"MeshGraphNet params = {n_params:,} (anchor 7,350,420, delta {abs(n_params-7350420)/7350420*100:.2f}%)")

    # ---- train + eval each seed individually ----
    per_seed: list[dict] = []
    for m in range(a.seeds):
        log(f"================= seed {m}/{a.seeds - 1} =================")
        model = train_seed(a, m, train_pcs, point_norm, device, a.ckpt_dir)

        predict_fn = make_mgn_predict_fn(model, test_pcs, point_norm, device, a)
        t0 = time.time()
        mets = evaluate_cases(predict_fn, test_pairs, checker=checker)
        log(f"seed {m} eval {time.time()-t0:.1f}s -> "
            + ", ".join(f"{k}={mets.get(k, float('nan')):.4g}" for k in _METRICS))
        per_seed.append(mets)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Incremental persistence after EACH seed (crash-safe across the long run).
        out = {
            "meta": {
                "model": "meshgraphnet", "task": a.task, "seeds": a.seeds,
                "epochs": a.epochs, "width": a.width, "n_layers": a.n_layers,
                "k": a.k, "aggr": a.aggr, "n_params": int(n_params),
                "n_train": a.n_train, "n_val": a.n_val, "resolution": a.resolution,
            },
            "per_seed": [{"seed": i, "metrics": d} for i, d in enumerate(per_seed)],
            "aggregate": _agg(per_seed),
        }
        with open(os.path.join(a.out_dir, "mgn_results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log(f"wrote {a.out_dir}/mgn_results.json ({len(per_seed)}/{a.seeds} seeds)")

    agg = _agg(per_seed)
    log("FINAL (mean +/- std across seeds): "
        + ", ".join(f"{k}={agg[k]['mean']:.4g}+/-{agg[k]['std']:.2g}" for k in _METRICS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
