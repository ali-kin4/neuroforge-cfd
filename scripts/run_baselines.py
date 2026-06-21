"""Matched-budget baselines for the NeuroForge paper Table 2 (Transolver on AirfRANS).

Trains a Transolver point-cloud surrogate on the AirfRANS NATIVE point clouds and
scores it with the **same** ``evaluate_cases`` metrics as NeuroForge (via the
rasterise adapter), over multiple seeds, writing ``results/baselines/table2.{md,csv}``
with mean +/- std columns.

This is the BUILD; the full run is an overnight job. Smoke it tiny first:

    python scripts/run_baselines.py --model transolver --task full \
        --n-train 8 --n-val 4 --epochs 2 --n-points 4096 --seeds 0 \
        --cache-dir data/cache --out-dir results/baselines_smoke

Full matched-budget run (3 seeds, see baseline_plan.md):

    python scripts/run_baselines.py --model transolver --task full \
        --n-train 800 --n-val 200 --epochs 80 --n-points 16384 --seeds 0 1 2 \
        --cache-dir data/cache --out-dir results/baselines

Always import ``neuroforge`` before numpy/torch (BLAS thread cap) -- done at top.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)
import numpy as np
import torch

from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.pointcloud import (
    F_IN,
    F_OUT,
    PointNormalizer,
    load_airfrans_pointclouds,
)
from neuroforge.models.baselines.transolver import TransolverPointModel
from neuroforge.models.baselines.transolver_adapter import make_predict_fn
from neuroforge.physics.evaluation import evaluate_cases

# Columns reported in the table. residual_error_spearman is intentionally OMITTED
# (it requires the NeuroForge checker; not a property of a plain baseline).
_METRICS = [
    "mse_u", "mse_v", "mse_p", "mse_nut", "surface_mse_p",
    "rho_cl", "rho_cd", "cl_rel_err_mean", "cd_rel_err_mean",
]


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_model(name: str, width: int, n_layers: int, n_heads: int, n_slices: int,
                 dropout: float) -> torch.nn.Module:
    if name == "transolver":
        return TransolverPointModel(
            in_features=F_IN, out_features=F_OUT, width=width, n_layers=n_layers,
            n_heads=n_heads, n_slices=n_slices, dropout=dropout,
        )
    if name == "meshgraphnet":
        # MeshGraphNet needs an edge set (edge_index + edge_attr) per forward, which
        # this script's edge-less grid-style train loop does not build. Fail loudly
        # rather than return a model that crashes later inside model(xb). The full
        # graph-aware train/eval pipeline lives in scripts/run_mgn.py.
        raise ValueError(
            "MeshGraphNet requires a graph (edge_index + edge_attr) per forward, "
            "which run_baselines.py's train loop does not construct. Use "
            "scripts/run_mgn.py for the MeshGraphNet backbone (it builds the kNN "
            "graph and edge features). This branch exists only so --model "
            "meshgraphnet is a recognised, documented choice."
        )
    raise ValueError(
        f"unknown --model '{name}' (implemented: 'transolver', 'meshgraphnet')"
    )


def _train_one_seed(args, seed: int, train_pcs, normalizer, device) -> tuple[torch.nn.Module, float]:
    """Train the point model for one seed. Returns (model, sec_per_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_model(
        args.model, args.width, args.n_layers, args.n_heads, args.n_slices, args.dropout
    ).to(device)
    n_params = model.num_params()
    print(f"[baseline] seed {seed}: {args.model} params = {n_params:,}")

    # Transolver's own recipe: AdamW + OneCycle (see baseline_plan.md section 4).
    if args.optimizer == "adamw-onecycle":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps = args.epochs * max(len(train_pcs), 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1
        )
    else:  # adam-cosine -- mirror NeuroForge exactly if requested
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs * max(len(train_pcs), 1)
        )

    rng = np.random.default_rng(seed)
    n_pts = int(args.n_points)
    sec_per_epoch = float("nan")

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_pcs))
        agg = 0.0
        n = 0
        t0 = time.time()
        for idx in order:
            pc = train_pcs[idx]
            m = pc.n_points
            sel = rng.integers(0, m, size=min(n_pts, m)) if m > n_pts else np.arange(m)
            x = normalizer.transform_in(pc.features[sel])      # (n, F_IN)
            y = normalizer.transform_out(pc.targets[sel])      # (n, F_OUT)
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
        print(f"[baseline] seed {seed} epoch {epoch}: train_mse={agg/max(n,1):.4e} "
              f"({sec_per_epoch:.1f}s)")
    return model, sec_per_epoch


def _evaluate_seed(model, test_pairs, test_pcs, normalizer, device, chunk) -> dict:
    predict_fn = make_predict_fn(model, test_pcs, normalizer, device=device, chunk=chunk)
    return evaluate_cases(predict_fn, test_pairs)


def _agg(dicts: list[dict]) -> dict[str, tuple[float, float]]:
    out = {}
    for m in _METRICS:
        vals = [d[m] for d in dicts if m in d and np.isfinite(d[m])]
        out[m] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
    return out


def _write_table(table: dict, out_dir: str, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Markdown
    hdr = "| model | n_params | " + " | ".join(_METRICS) + " |"
    sep = "|" + "---|" * (len(_METRICS) + 2)
    lines = [
        f"### Table 2 — baselines on AirfRANS/{meta['task']}, "
        f"{meta['n_seeds']} seeds (mean +/- std)",
        "",
        f"_Train: n_train={meta['n_train']}, epochs={meta['epochs']}, "
        f"n_points/step={meta['n_points']}, optimizer={meta['optimizer']}. "
        f"Native point-cloud input; scored by the SHARED `evaluate_cases` after "
        f"identical rasterisation to the GT grid. `residual_error_spearman` is "
        f"n/a for a plain baseline (NeuroForge-checker-only)._",
        "", hdr, sep,
    ]
    for model_name, mets in table.items():
        cells = [model_name, f"{meta['n_params']:,}"]
        for m in _METRICS:
            mu, sd = mets[m]
            cells.append("n/a" if np.isnan(mu) else f"{mu:.4g}+/-{sd:.2g}")
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "table2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    # CSV
    with open(os.path.join(out_dir, "table2.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_params", "metric", "mean", "std"])
        for model_name, mets in table.items():
            for m, (mu, sd) in mets.items():
                w.writerow([model_name, meta["n_params"], m, mu, sd])
    print(f"[baseline] wrote {out_dir}/table2.md and table2.csv")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="NeuroForge Table 2 baselines (Transolver).")
    p.add_argument("--model", default="transolver", choices=["transolver", "meshgraphnet"])
    p.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--n-points", type=int, default=16384, help="points subsampled per train step")
    p.add_argument("--resolution", type=int, default=128, help="grid res for the GT/eval rasterisation")
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="results/baselines")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--eval-chunk", type=int, default=0,
        help="0 (default) = single WHOLE-CLOUD forward at eval (the faithful "
             "Transolver inference; ~3GB GPU at width=256/L=10, ~180k pts). Set "
             ">0 ONLY as an OOM fallback: it partitions the slice attention and is "
             "NOT equivalent to whole-cloud inference.",
    )
    # Matched-budget knobs (defaults ~7-8M params; see baseline_plan.md section 4).
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-slices", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--optimizer", default="adamw-onecycle",
                   choices=["adamw-onecycle", "adam-cosine"])
    a = p.parse_args(argv)

    device = _resolve_device(a.device)
    print(f"[baseline] device={device}")

    # Native point clouds for the model input (train + test).
    print("[baseline] loading native point clouds ...")
    train_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=True, limit=a.n_train,
        cache_dir=a.cache_dir, download=a.download,
    )
    test_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=False, limit=a.n_val,
        cache_dir=a.cache_dir, download=a.download,
    )
    # GT (FlowCase, FlowField) pairs for the SHARED metric -- the SAME cache
    # NeuroForge evaluates against (aligned by case.name).
    print("[baseline] loading rasterised GT pairs (shared with NeuroForge) ...")
    test_pairs = load_airfrans(
        root=a.root, task=a.task, train=False, resolution=a.resolution,
        limit=a.n_val, cache_dir=a.cache_dir, download=a.download,
    )

    # Sanity: names align between point clouds and GT pairs.
    pc_names = {pc.name for pc in test_pcs}
    missing = [c.name for c, _ in test_pairs if c.name not in pc_names]
    if missing:
        raise RuntimeError(
            f"{len(missing)} GT test cases have no matching point cloud "
            f"(e.g. {missing[:3]}). Point clouds and GT pairs must use the same "
            "task/split/limit so af.dataset.load order matches."
        )

    # Normaliser: fit on TRAIN ONLY (no leakage).
    normalizer = PointNormalizer().fit(train_pcs)
    print(f"[baseline] point normaliser fitted on {len(train_pcs)} train clouds")

    per_seed: list[dict] = []
    n_params = 0
    sec_epoch = float("nan")
    for seed in a.seeds:
        model, sec_epoch = _train_one_seed(a, seed, train_pcs, normalizer, device)
        n_params = model.num_params()
        t0 = time.time()
        eval_chunk = a.eval_chunk if a.eval_chunk and a.eval_chunk > 0 else None
        mets = _evaluate_seed(model, test_pairs, test_pcs, normalizer, device, eval_chunk)
        print(f"[baseline] seed {seed}: eval {time.time()-t0:.1f}s -> "
              + ", ".join(f"{k}={mets.get(k, float('nan')):.4g}" for k in _METRICS))
        per_seed.append(mets)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Incremental persistence: rewrite the table (and a per-seed JSON) after
        # EACH seed so a crash in a later seed of the multi-hour run does not lose
        # the completed ones (mirrors benchmarks/ablation.py's checkpointing).
        meta = {
            "task": a.task, "n_seeds": len(per_seed), "n_train": a.n_train,
            "epochs": a.epochs, "n_points": a.n_points, "optimizer": a.optimizer,
            "n_params": n_params,
        }
        _write_table({a.model: _agg(per_seed)}, a.out_dir, meta)
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, f"seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"seed": int(seed), "metrics": mets, "n_params": n_params}, f)
    print(f"[baseline] approx {sec_epoch:.1f}s/epoch at the configured settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
