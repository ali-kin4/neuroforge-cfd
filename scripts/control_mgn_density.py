"""Discriminating control: is MeshGraphNet's weak eval accuracy a TRAIN/EVAL
graph-density confound, or a genuine architecture/calibration weakness?

The finding under test (rigor audit)
------------------------------------
MGN is TRAINED on a 16,384-point subsample of each AirfRANS cloud but EVALUATED
on the FULL ~180k-point cloud (~11x denser). Its kNN edge features are normalised
by the TRAIN length scale (``l_xy = PointNormalizer.std_in[:2]``; see
``models/baselines/meshgraphnet.py``), so at the denser eval the relative edge
displacements shrink and the fixed 12-hop receptive field covers a smaller
physical neighbourhood. MGN's reported accuracy (mse_u=12.75, mse_v=13.82,
mse_p=42070) might therefore degrade purely from the density shift, not the
architecture -- which would make "MGN is architecturally weak" an unsupported
claim. Transolver (global attention) is density-robust, so the comparison would
be unfair. (The trust-signal / Spearman result is confound-proof and is NOT in
question -- only the accuracy framing.)

The experiment (vary ONLY graph density)
----------------------------------------
For each shared test case we run the SAME trained MGN, with the SAME normalizer,
at two input graph densities and DECLARE the verdict rule BEFORE running:

* **MSE@16k (train density)** -- subsample the cloud to 16,384 pts (the training
  density), build the kNN graph on that subset, predict.
* **MSE@180k (deploy density)** -- the full ~180k cloud, the deployed eval.

VERDICT RULE (declared before running):
  * MSE@16k  <<  MSE@180k  -> DENSITY is the driver; do NOT call MGN
    architecturally weak (report the confound honestly).
  * MSE bad at BOTH         -> MAGNITUDE-CALIBRATION problem; density is a
    side-show and the accuracy row is a fair architecture comparison.

Two metrics, both fed by the same two forwards (advisor-hardened)
----------------------------------------------------------------
1. **probe_mse (PRIMARY, confound-free).** Per-point MSE in PHYSICAL units scored
   at the SAME 16,384 point indices for both arms. The 16k arm is a SUBSET of the
   full cloud, so we score ``pred16 vs targets[sel]`` and ``pred_full[sel] vs
   targets[sel]`` -- identical points, identical model, identical normalizer, so
   the ONLY difference is the graph density seen by the message passing. No
   rasterisation enters, so it carries the verdict cleanly in BOTH directions.
2. **grid_mse (BRIDGE to the headline 12.75).** The exact ``per_channel_mse`` on
   the rasterised 128-grid over the fluid mask -- byte-identical to the metric
   that produced the published mse_u=12.75. Reproducing ~12.75 on the 180k arm
   licenses us to cite the headline as the 180k reference. NOTE: the 16k grid arm
   additionally eats a rasterisation-sparsity penalty (16,384 scattered pts onto
   16,384 grid cells ~= 1 pt/cell vs ~11 pt/cell at 180k), so a 16k WIN here is
   conservative but a 16k "tie/loss" here is NOT trustworthy -- which is exactly
   why probe_mse is primary.

Normalizer (compliance + faithfulness)
--------------------------------------
Fit ``PointNormalizer`` on the n8 TRAIN cache (74 MB -- NOT the forbidden 7.5 GB
n800 cache). The x,y spatial statistics that set the edge length scale ``l_xy``
are stable across airfoils, so n8-train reproduces the deployed ``l_xy`` closely
(keeping the 16k arm IN the trained regime) while staying compliant. Whether the
resulting grid_mse is comparable to 12.75 is verified empirically and reported.

Hard constraints honoured
-------------------------
* CPU-first / device-agnostic (``--device``; honours ``CUDA_VISIBLE_DEVICES=""``).
* ``import neuroforge`` BEFORE numpy/torch (BLAS thread cap) -- done at the top.
* Does NOT write under ``results/mgn`` / ``checkpoints/mgn`` (only READS seed0.pt).
* The full-180k CPU forward is slow, so default ``--n-cases`` is small (decisive
  directional is enough); reuse the n4 test PC cache by default. The script ALSO
  supports a rigorous full-GPU run later (``--device cuda --n-cases 200
  --eval-edge-chunk 500000``) writing to a separate results path.

Quick CPU control (decisive directional, ~minutes):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/control_mgn_density.py \
        --device cpu --n-cases 4 --eval-edge-chunk 200000

Rigorous full-GPU run later (all 200 cases, faithful normalizer on n800):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/control_mgn_density.py \
        --device cuda --n-cases 200 --train-cache-n 800 \
        --eval-edge-chunk 500000 --out results/control/mgn_density_gpu.json
"""

from __future__ import annotations

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets BLAS thread caps)

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

from neuroforge.core.config import Config
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.pointcloud import (
    F_IN,
    F_OUT,
    PointNormalizer,
    load_airfrans_pointclouds,
)
from neuroforge.data.rasterize import rasterize_point_cloud
from neuroforge.geometry.sdf import signed_distance, solid_mask
from neuroforge.models.baselines.meshgraphnet import (
    MeshGraphNetPointModel,
    edge_features,
    knn_edges,
)
from neuroforge.physics.evaluation import per_channel_mse

# Published seed-0 full-180k grid reference (results/mgn/mgn_results.json), used
# only as a comparability check for grid_mse@180k -- never as a substitute for a
# measured arm in this control.
_HEADLINE_180K = {"mse_u": 12.749268536630716, "mse_v": 13.823635332233078,
                  "mse_p": 42070.37973022175}

_CHANNELS = ("u", "v", "p", "nut")


def log(msg: str) -> None:
    print(f"[ctrl] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _length_scale(point_norm: PointNormalizer) -> tuple[torch.Tensor, float]:
    """Edge length scale from the train PointNormalizer (raw x,y std = std_in[:2])."""
    l_xy = torch.as_tensor(point_norm.std_in[:2], dtype=torch.float32)
    l_norm = float(torch.linalg.norm(l_xy).item()) or 1.0
    return l_xy, l_norm


def _load_pcs_limit(cache_dir: str, task: str, train: bool, n: int):
    """Load n point clouds, preferring an exact cache, else slicing a bigger one.

    Avoids re-reading the raw AirfRANS dataset (slow) and avoids the forbidden
    7.5 GB n800 cache unless explicitly requested via a matching cache file.
    """
    # Exact cache?
    exact = load_airfrans_pointclouds(
        task=task, train=train, limit=n, cache_dir=cache_dir, progress=False
    ) if os.path.exists(
        os.path.join(cache_dir, f"airfrans_pc_{task}_{'train' if train else 'test'}_n{n}.pkl")
    ) else None
    if exact is not None:
        return exact[:n]
    # Slice the smallest existing cache that is >= n (avoid loading more than needed).
    split = "train" if train else "test"
    candidates = []
    for fn in os.listdir(cache_dir):
        if fn.startswith(f"airfrans_pc_{task}_{split}_n") and fn.endswith(".pkl"):
            try:
                cn = int(fn[len(f"airfrans_pc_{task}_{split}_n"):-4])
            except ValueError:
                continue
            if cn >= n:
                candidates.append((cn, fn))
    if not candidates:
        raise FileNotFoundError(
            f"No point-cloud cache with >= {n} {split} clouds under {cache_dir}. "
            f"Available: {sorted(os.listdir(cache_dir))}. Re-run with a smaller "
            f"--n-cases / --train-cache-n, or generate the cache via run_mgn.py."
        )
    cn, fn = min(candidates)  # smallest sufficient cache
    log(f"slicing {fn} (has {cn}) for {n} {split} clouds")
    with open(os.path.join(cache_dir, fn), "rb") as fh:
        return pickle.load(fh)[:n]


def build_model_from_ckpt(ckpt_path: str, device: torch.device):
    """Reconstruct the trained MGN from its checkpoint (architecture from saved args)."""
    state = torch.load(ckpt_path, map_location=device)
    a = state.get("args", {})
    width = int(a.get("width", 290))
    n_layers = int(a.get("n_layers", 12))
    aggr = str(a.get("aggr", "sum"))
    k = int(a.get("k", 8))
    n_points = int(a.get("n_points", 16384))
    model = MeshGraphNetPointModel(
        in_features=F_IN, out_features=F_OUT, width=width, n_layers=n_layers,
        edge_in=3, aggr=aggr,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    info = {"width": width, "n_layers": n_layers, "aggr": aggr, "k": k,
            "n_points": n_points, "epoch": int(state.get("epoch", -1)),
            "n_params": int(model.num_params())}
    return model, info


@torch.no_grad()
def _forward_cloud(model, pos_np, feats_phys, point_norm, l_xy, l_norm, k,
                   device, edge_chunk, knn_chunk):
    """Forward MGN on a (sub)cloud; return PHYSICAL per-point predictions (M,4)."""
    feats = point_norm.transform_in(feats_phys)               # (M, F_IN) normalised
    pos = torch.from_numpy(np.asarray(pos_np, np.float32)).to(device)
    ei = knn_edges(pos, k=k, chunk=knn_chunk)
    ea = edge_features(pos, ei, l_xy, l_norm)
    xb = torch.from_numpy(feats).to(device).unsqueeze(0)
    out = model(xb, ei, ea, edge_chunk=edge_chunk).squeeze(0)
    preds = point_norm.inverse_out(out).detach().cpu().numpy().astype(np.float32)
    n_edges = int(ei.shape[1])
    del pos, ei, ea, xb, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds, n_edges


def _grid_mse_for_cloud(pc, preds_phys, case, ref_field):
    """Rasterise physical per-point preds to the case grid, MSE over fluid mask.

    Byte-identical projection + solid-zeroing to run_mgn.make_mgn_predict_fn and
    the GT loader, so grid_mse@180k is directly comparable to the headline 12.75.
    """
    from neuroforge.core.types import FlowField

    domain = case.domain
    raster = rasterize_point_cloud(pc, preds_phys, domain, fill=0.0, method="linear")
    sdf = signed_distance(case.geometry, domain)
    mask = solid_mask(case.geometry, domain)
    solid = mask < 0.5
    pred_field = FlowField(
        domain=domain,
        u=np.where(solid, 0.0, raster[0]),
        v=np.where(solid, 0.0, raster[1]),
        p=raster[2],
        nut=np.where(solid, 0.0, np.maximum(raster[3], 0.0)),
        mask=mask, sdf=sdf, meta={"source": "control_mgn_density"},
    )
    raw = per_channel_mse(pred_field, ref_field)  # keys: mse_u, mse_v, mse_p, ...
    # Re-key to bare channel names (u, v, p, nut) so probe and grid dicts share a
    # schema and the same _summ / _verdict / _write code handles both.
    return {c: float(raw.get(f"mse_{c}", float("nan"))) for c in _CHANNELS}


def _summ(per_case: list[dict], key: str) -> dict:
    """Mean of a per-case metric dict's channel entries across cases."""
    out = {}
    chans = per_case[0][key].keys()
    for c in chans:
        vals = [d[key][c] for d in per_case if np.isfinite(d[key].get(c, np.nan))]
        out[c] = float(np.mean(vals)) if vals else float("nan")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/mgn/seed0.pt",
                   help="trained MGN checkpoint (READ-ONLY).")
    p.add_argument("--task", default="full")
    p.add_argument("--n-cases", type=int, default=4,
                   help="shared test cases (small on CPU; decisive directional).")
    p.add_argument("--train-density", type=int, default=16384,
                   help="the training graph density (subsample size for the 16k arm).")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--train-cache-n", type=int, default=8,
                   help="n train clouds to fit the PointNormalizer (8=74MB cache; "
                        "use 800 for the faithful full-GPU run if that cache exists).")
    p.add_argument("--device", default="cpu",
                   help="cpu (default; a GPU training run may be LIVE) or cuda.")
    p.add_argument("--k", type=int, default=None, help="kNN k (default: from ckpt).")
    p.add_argument("--eval-edge-chunk", type=int, default=200000,
                   help="edge-block size for the full-cloud edge update (memory).")
    p.add_argument("--knn-chunk", type=int, default=512,
                   help="kNN query block (small bounds the cdist temp; exact).")
    p.add_argument("--subsample-seed", type=int, default=0,
                   help="seed for the WITHOUT-replacement 16k subsample.")
    p.add_argument("--out", default="results/control/mgn_density_control.json")
    a = p.parse_args(argv)

    device = _resolve_device(a.device)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    log(f"device={device}  n_cases={a.n_cases}  train_density={a.train_density}  "
        f"out={a.out}")
    if device.type == "cuda":
        log("WARNING: --device cuda requested. A GPU MGN training run may be LIVE; "
            "prefer --device cpu unless you know the GPU is free.")

    # ---- model (architecture read from the checkpoint's own saved args) ----
    model, info = build_model_from_ckpt(a.ckpt, device)
    k = int(a.k) if a.k is not None else info["k"]
    log(f"loaded {a.ckpt}: width={info['width']} n_layers={info['n_layers']} "
        f"aggr={info['aggr']} k={k} epoch={info['epoch']} params={info['n_params']:,}")
    if a.train_density != info["n_points"]:
        log(f"NOTE: --train-density {a.train_density} != ckpt n_points "
            f"{info['n_points']}; using {a.train_density} as the 16k arm density.")

    # ---- normalizer (faithful train length scale; compliant cache size) ----
    train_pcs = _load_pcs_limit(a.cache_dir, a.task, True, a.train_cache_n)
    point_norm = PointNormalizer().fit(train_pcs)
    l_xy, l_norm = _length_scale(point_norm)
    log(f"PointNormalizer fit on {len(train_pcs)} train clouds (n{a.train_cache_n} "
        f"cache); l_xy={l_xy.tolist()} l_norm={l_norm:.4f}")
    del train_pcs

    # ---- shared eval set: point clouds + byte-identical rasterised GT pairs ----
    test_pcs = _load_pcs_limit(a.cache_dir, a.task, False, a.n_cases)
    pairs = load_airfrans(
        task=a.task, train=False, resolution=a.resolution, limit=max(a.n_cases, 4),
        cache_dir=a.cache_dir, progress=False,
    )
    ref_by_name = {c.name: (c, f) for c, f in pairs}
    rng = np.random.default_rng(a.subsample_seed)

    edge_chunk = a.eval_edge_chunk if a.eval_edge_chunk and a.eval_edge_chunk > 0 else None
    per_case: list[dict] = []

    for i, pc in enumerate(test_pcs[: a.n_cases]):
        if pc.name not in ref_by_name:
            log(f"skip {pc.name}: no rasterised GT pair (task/split mismatch).")
            continue
        case, ref = ref_by_name[pc.name]
        m = pc.n_points
        nsub = min(int(a.train_density), m)
        sel = rng.choice(m, size=nsub, replace=False)  # WITHOUT replacement, fixed seed
        sel = np.sort(sel)  # stable indexing for the probe

        t0 = time.time()
        # 16k arm: graph + forward on the SUBSET only.
        pred16, e16 = _forward_cloud(
            model, pc.pos[sel], pc.features[sel], point_norm, l_xy, l_norm, k,
            device, edge_chunk, a.knn_chunk)
        t16 = time.time() - t0

        t0 = time.time()
        # 180k arm: graph + forward on the FULL cloud.
        pred_full, efull = _forward_cloud(
            model, pc.pos, pc.features, point_norm, l_xy, l_norm, k,
            device, edge_chunk, a.knn_chunk)
        tfull = time.time() - t0

        tgt = np.asarray(pc.targets, np.float64)

        # --- PRIMARY: per-point physical MSE at the SAME indices (sel) ---
        def _probe(pred, idx):
            d = pred.astype(np.float64) - tgt[idx]
            return {c: float(np.mean(d[:, j] ** 2)) for j, c in enumerate(_CHANNELS)}

        probe16 = _probe(pred16, sel)            # 16k graph, scored on sel
        probe180 = _probe(pred_full[sel], sel)   # 180k graph, scored on SAME sel

        # --- BRIDGE: rasterised 128-grid MSE over the fluid mask ---
        grid16 = _grid_mse_for_cloud(pc.pos[sel], pred16, case, ref)
        grid180 = _grid_mse_for_cloud(pc.pos, pred_full, case, ref)

        per_case.append({
            "name": pc.name, "n_points": int(m), "n_sub": int(nsub),
            "edges_16k": int(e16), "edges_180k": int(efull),
            "probe_16k": probe16, "probe_180k": probe180,
            "grid_16k": grid16, "grid_180k": grid180,
            "t16_s": round(t16, 2), "tfull_s": round(tfull, 2),
        })
        log(f"[{i+1}/{a.n_cases}] {pc.name}  m={m} sub={nsub}  "
            f"probe u/v/p 16k=({probe16['u']:.3g},{probe16['v']:.3g},{probe16['p']:.4g}) "
            f"180k=({probe180['u']:.3g},{probe180['v']:.3g},{probe180['p']:.4g})  "
            f"({t16:.1f}s+{tfull:.1f}s)")

        # Incremental persistence (crash-safe across long CPU forwards).
        _write(a, info, k, point_norm, per_case)

    if not per_case:
        log("no cases scored.")
        return 1

    _write(a, info, k, point_norm, per_case, final=True)
    return 0


def _verdict(probe16: dict, probe180: dict) -> dict:
    """Apply the pre-declared rule on velocity (u,v) -- the well-posed channels.

    Pressure has a known offset issue (mse_p ~ 4e4) and the 6k overshoot note, so
    the directional call rests on velocity; pressure is reported alongside.
    """
    ratio = {c: (probe180[c] / probe16[c] if probe16[c] > 0 else float("inf"))
             for c in _CHANNELS}
    # "density-driven" if the deploy density is materially worse on velocity.
    vel_ratio = float(np.mean([ratio["u"], ratio["v"]]))
    if vel_ratio >= 1.5:
        v = "DENSITY-DRIVEN"
    elif vel_ratio <= 1.15 and vel_ratio >= 0.85:
        v = "MAGNITUDE-CALIBRATION (density not the driver)"
    else:
        v = "MIXED"
    return {"ratio_180k_over_16k": ratio, "velocity_ratio": vel_ratio, "verdict": v}


def _write(a, info, k, point_norm, per_case, final=False):
    probe16 = _summ(per_case, "probe_16k")
    probe180 = _summ(per_case, "probe_180k")
    grid16 = _summ(per_case, "grid_16k")
    grid180 = _summ(per_case, "grid_180k")
    verdict = _verdict(probe16, probe180)
    # Comparability of grid_mse@180k to the published headline 12.75 (keyed mse_*).
    comp = {c: {"control_180k_grid": grid180.get(c),
                "headline": _HEADLINE_180K.get(f"mse_{c}"),
                "rel_diff": (abs(grid180.get(c, float("nan")) - _HEADLINE_180K[f"mse_{c}"])
                             / _HEADLINE_180K[f"mse_{c}"])
                if f"mse_{c}" in _HEADLINE_180K else None}
            for c in ("u", "v", "p")}
    out = {
        "meta": {
            "experiment": "mgn_train_eval_density_control",
            "ckpt": a.ckpt, "task": a.task, "device": a.device,
            "n_cases_scored": len(per_case), "train_density_16k": a.train_density,
            "resolution": a.resolution, "train_cache_n": a.train_cache_n,
            "subsample": "without_replacement", "subsample_seed": a.subsample_seed,
            "k": k, "model": info, "l_xy": point_norm.std_in[:2].tolist(),
            "final": bool(final),
        },
        "summary_probe_PRIMARY": {
            "note": "per-point PHYSICAL MSE at the SAME 16,384 indices; "
                    "confound-free (no rasterisation). PRIMARY verdict metric.",
            "mse_16k": probe16, "mse_180k": probe180,
        },
        "summary_grid_BRIDGE": {
            "note": "rasterised 128-grid MSE over fluid mask; bridge to the "
                    "published mse_u=12.75. 16k grid arm carries a rasterisation-"
                    "sparsity penalty (conservative for a 16k win only).",
            "mse_16k": grid16, "mse_180k": grid180,
            "headline_comparability": comp,
        },
        "verdict": verdict,
        "per_case": per_case,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if final:
        log("================ RESULT ================")
        log(f"PROBE (primary) u/v/p  16k = "
            f"({probe16['u']:.4g}, {probe16['v']:.4g}, {probe16['p']:.5g})")
        log(f"PROBE (primary) u/v/p 180k = "
            f"({probe180['u']:.4g}, {probe180['v']:.4g}, {probe180['p']:.5g})")
        log(f"ratio 180k/16k u/v/p = ({verdict['ratio_180k_over_16k']['u']:.2f}, "
            f"{verdict['ratio_180k_over_16k']['v']:.2f}, "
            f"{verdict['ratio_180k_over_16k']['p']:.2f})  "
            f"velocity_ratio={verdict['velocity_ratio']:.2f}")
        log(f"GRID (bridge) u/v/p   16k = "
            f"({grid16['u']:.4g}, {grid16['v']:.4g}, {grid16['p']:.5g})")
        log(f"GRID (bridge) u/v/p  180k = "
            f"({grid180['u']:.4g}, {grid180['v']:.4g}, {grid180['p']:.5g})  "
            f"[headline 180k: 12.75, 13.82, 42070]")
        log(f"VERDICT: {verdict['verdict']}")
        log(f"wrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
