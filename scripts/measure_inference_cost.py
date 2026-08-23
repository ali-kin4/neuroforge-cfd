"""End-to-end wall-clock cost of the deployed pipeline — what does a prediction cost?

Question answered
-----------------
``scripts/measure_audit_cost.py`` answered *what does the certificate cost once a
field exists?* (median 1.65 ms/case).  It deliberately did **not** time the thing
being audited: the surrogate forward pass, the DEQ correction loop, or the
ensemble.  JCP's aims and scope require a submission to address "efficacy,
robustness, computational complexity, as well as reproducibility"; this script
supplies the complexity half by timing every stage of the *deployed* pipeline on
the same 200 AirfRANS ``full`` test cases at resolution 128 that produce the
paper's headline numbers, on both CPU and GPU.

Stages timed per case (median over ``--repeats`` timed repeats after one untimed
warm-up; CUDA timings synchronise before and after each repeat):

  encode      encode_case(case)                geometry -> SDF/mask input planes
  backbone    PointCloudPredictor.predict()    Transolver forward on the native
                                               cloud + rasterise to the grid
  solve       NeuroForgeEngine.solve()         backbone + Neural Residual
                                               Iteration (DEQ corrector, gated)
  correction  solve - backbone                 marginal cost of the correction
  audit       diagnose + residual_norm         the certificate (cross-check
                                               against measure_audit_cost.py)

The deep ensemble is K statistically independent backbones, so its cost is
exactly K x ``backbone``; ``--ensemble-k`` re-measures that directly (loading K
checkpoints) rather than asserting linearity.

Two scaling studies, both empirical:

  points   backbone cost vs the native cloud size, which varies case to case
           across the real test set -- no synthetic inputs, the spread is the
           experiment (reported as a log-log least-squares exponent).
  grid     audit cost vs grid cells N = ny*nx, on fields resampled to
           r in {64, 96, 128, 192, 256}.  Cost depends only on array extent, so
           resampling is legitimate here; the resampled fields are NOT physically
           meaningful and no accuracy number is computed from them.

Nothing is invented: the classical-solver anchor must be passed on the command
line with its source, exactly as in ``measure_audit_cost.py``.

Run (GPU + CPU, 50 cases, ~5-10 min):
    .venv/Scripts/python.exe scripts/measure_inference_cost.py --n-val 50

Full fleet as reported in the paper:
    .venv/Scripts/python.exe scripts/measure_inference_cost.py ^
        --n-val 200 --repeats 3 --ensemble-k 5 ^
        --classical-solve-sec 1500 ^
        --classical-source "AirfRANS (Bonnet et al., NeurIPS 2022) Table 4: ~25 min per simulation on 16 cores of an AMD Ryzen Threadripper 3960X"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy/torch import

import numpy as np
import torch

from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE, Domain, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.datamodule import Normalizer
from neuroforge.data.pointcloud import (
    F_IN,
    F_OUT,
    PointNormalizer,
    load_airfrans_pointclouds,
)
from neuroforge.geometry.encode import encode_case
from neuroforge.models import build_corrector
from neuroforge.models.baselines.transolver import TransolverPointModel
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor



def log(msg: str) -> None:
    print(f"[inference-cost] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# crash-safe progress
# --------------------------------------------------------------------------- #
class Checkpoint:
    """Append-only per-measurement journal so a killed run resumes where it died.

    Every timed quantity is written as one JSON line ``{"k": key, "v": value}`` and
    fsynced immediately. A restart replays the file and skips any key already
    present, so an interrupted run (power cut, OOM, Ctrl-C) costs at most the case
    in flight. Keys embed the run parameters that change a timing -- device,
    repeats, resolution, seed -- so a run with different settings never reuses
    numbers measured under the old ones.
    """

    def __init__(self, path, enabled=True, stamp=""):
        self.path = path
        self.enabled = enabled
        self.stamp = stamp
        self.done = {}
        if enabled and path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue        # torn final line after a hard kill
                        self.done[row["k"]] = row["v"]
                if self.done:
                    log(f"resuming: {len(self.done)} measurements replayed from "
                        f"{path}")

    def key(self, *parts):
        return "|".join([self.stamp, *(str(x) for x in parts)])

    def run(self, key, fn):
        """Return the journalled value for ``key``, or compute, journal and return."""
        if key in self.done:
            return self.done[key]
        val = fn()
        self.done[key] = val
        if self.enabled and self.path:
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({"k": key, "v": val}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return val


# --------------------------------------------------------------------------- #
# checkpoint restore (identical path to measure_acceptance_gate.py)
# --------------------------------------------------------------------------- #
def _load_backbone(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["backbone_config"]
    model = TransolverPointModel(
        in_features=int(cfg.get("in_features", F_IN)),
        out_features=int(cfg.get("out_features", F_OUT)),
        width=int(cfg["width"]), n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]), n_slices=int(cfg["n_slices"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    pn = state["point_norm"]
    point_norm = PointNormalizer(
        mean_in=np.asarray(pn["mean_in"], DTYPE), std_in=np.asarray(pn["std_in"], DTYPE),
        mean_out=np.asarray(pn["mean_out"], DTYPE), std_out=np.asarray(pn["std_out"], DTYPE),
        eps=float(pn.get("eps", 1e-6)),
    )
    grid_norm = Normalizer.from_state_dict(state["grid_norm"])
    n_params = sum(p.numel() for p in model.parameters())
    return model, point_norm, grid_norm, n_params


def _load_corrector(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    corrector = build_corrector(state["corrector_config"])
    corrector.load_state_dict(state["corrector_state"])
    corrector.eval()
    return corrector


# --------------------------------------------------------------------------- #
# timing helpers
# --------------------------------------------------------------------------- #
def _sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_call(fn, device, repeats: int) -> float:
    """Median wall-clock of ``fn`` in ms, after one untimed warm-up call."""
    fn()                       # warm-up (allocator, cuDNN autotune, page-ins)
    _sync(device)
    samples = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(samples))


def summarise(values, extra=None) -> dict:
    v = sorted(float(x) for x in values)
    if not v:
        return {"n_cases": 0}
    out = {
        "mean_ms": float(np.mean(v)),
        "median_ms": float(np.median(v)),
        "p95_ms": float(np.percentile(v, 95)),
        "max_ms": v[-1],
        "n_cases": len(v),
    }
    if extra:
        out.update(extra)
    return out


def loglog_exponent(x, y) -> dict:
    """Least-squares slope of log(y) on log(x) -- the empirical scaling exponent."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = (x > 0) & (y > 0)
    if keep.sum() < 3:
        return {"exponent": None, "n": int(keep.sum())}
    lx, ly = np.log(x[keep]), np.log(y[keep])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "exponent": float(slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "n": int(keep.sum()),
    }


def resample_field(field: FlowField, r: int) -> FlowField:
    """Nearest-neighbour resample of every plane to ``(r, r)`` -- COST probe only.

    The result is not physically meaningful; it exists so the audit can be timed
    against grid extent.  No accuracy number is ever computed from it.
    """
    ny, nx = field.domain.ny, field.domain.nx
    iy = np.clip((np.arange(r) * ny / r).astype(int), 0, ny - 1)
    ix = np.clip((np.arange(r) * nx / r).astype(int), 0, nx - 1)

    def rs(arr):
        return np.ascontiguousarray(arr[iy, :][:, ix], dtype=DTYPE)

    return FlowField(
        domain=Domain(bounds=field.domain.bounds, nx=r, ny=r),
        u=rs(field.u), v=rs(field.v), p=rs(field.p),
        nut=rs(field.nut) if field.nut is not None else None,
        mask=rs(field.mask) if field.mask is not None else None,
        sdf=rs(field.sdf) if field.sdf is not None else None,
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/Dataset")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--ckpt-dir", default="checkpoints/v2_transolver")
    p.add_argument("--seed", type=int, default=0,
                   help="backbone seed to time (the deployed tab:v2 system)")
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--devices", default="auto",
                   help="'auto' = cuda,cpu when CUDA is present; or e.g. 'cpu'")
    p.add_argument("--ensemble-k", type=int, default=0,
                   help="if >0, directly time a K-member ensemble forward")
    p.add_argument("--grid-scaling", default="64,96,128,192,256",
                   help="resolutions for the audit-vs-grid-extent probe ('' to skip)")
    p.add_argument("--classical-solve-sec", type=float, default=None)
    p.add_argument("--classical-source", default="")
    p.add_argument("--context", default="", help="load context recorded in the output")
    p.add_argument("--out", default="results/control/inference_cost.json")
    p.add_argument("--progress",
                   default="results/control/_cache/inference_cost_progress.jsonl",
                   help="crash-safe journal of per-case timings ( '' to disable)")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing journal and re-measure everything")
    a = p.parse_args(argv)

    if a.devices == "auto":
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    else:
        devices = [d.strip() for d in a.devices.split(",") if d.strip()]

    t_start = time.time()
    # A wall-clock journal is only valid on the machine that produced it, so the
    # stamp fingerprints the hardware and stack as well as the run parameters. A
    # journal copied to another machine (or a different GPU, torch build or thread
    # cap) simply misses on every key and the run re-measures from scratch, rather
    # than silently reporting someone else's timings as yours.
    host = "|".join([
        platform.platform(), platform.processor(), str(os.cpu_count()),
        torch.__version__,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nocuda",
        str(os.environ.get("OMP_NUM_THREADS")),
    ])
    host_id = hashlib.sha256(host.encode("utf-8")).hexdigest()[:12]
    ckpt = Checkpoint(a.progress, enabled=bool(a.progress) and not a.no_resume,
                      stamp=f"{host_id}r{a.resolution}s{a.seed}x{a.repeats}")
    log(f"loading AirfRANS full/test (res {a.resolution}, limit {a.n_val}) ...")
    pairs = load_airfrans(
        root=a.root, task="full", train=False, resolution=a.resolution,
        limit=a.n_val, cache_dir=a.cache_dir, download=False, progress=False,
    )
    log(f"{len(pairs)} case/GT pairs")

    log("loading native test point clouds ...")
    test_pcs = load_airfrans_pointclouds(
        root=a.root, task="full", train=False, limit=a.n_val,
        cache_dir=a.cache_dir, download=False, progress=False,
    )
    n_points = {pc.name: int(np.asarray(pc.features).shape[0]) for pc in test_pcs}

    checker = PhysicsChecker(Config().physics)

    result = {
        "meta": {
            "experiment": "wall-clock cost of the deployed pipeline (per-case, stage-wise)",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "grid": f"{a.resolution}x{a.resolution}",
            "backbone_seed": a.seed,
            "repeats_per_stage": a.repeats,
            "timing": ("median of repeats per case after one untimed warm-up; "
                       "CUDA synchronised around every repeat; "
                       "fleet stats over per-case medians"),
            "load_context": a.context,
            "environment": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda_device": (torch.cuda.get_device_name(0)
                                if torch.cuda.is_available() else None),
                "blas_thread_caps": {
                    k: os.environ.get(k) for k in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                },
            },
        },
        "per_device": {},
    }

    # ---------------- encode + audit (device-independent, numpy/CPU) --------- #
    log("timing encode + audit (CPU numpy path) ...")
    enc_ms, aud_ms = [], []
    for i, (case, _gt) in enumerate(pairs):
        enc_ms.append(ckpt.run(
            ckpt.key("encode", case.name),
            lambda c=case: time_call(lambda: encode_case(c), "cpu", a.repeats)))
        if (i + 1) % 50 == 0:
            log(f"  encode {i + 1}/{len(pairs)}")
    result["encode_ms"] = summarise(enc_ms)

    # ---------------- per-device stages ------------------------------------- #
    for dev in devices:
        log(f"=== device {dev} ===")
        model, point_norm, grid_norm, n_params = _load_backbone(
            os.path.join(a.ckpt_dir, f"seed{a.seed}.pt"), dev)
        pred = PointCloudPredictor(model, test_pcs, point_norm, grid_norm, device=dev)
        corrector = _load_corrector(os.path.join(a.ckpt_dir, f"seed{a.seed}_corr_with.pt"))
        engine = NeuroForgeEngine(pred, checker, corrector=corrector, config=Config())

        back_ms, solve_ms, pts = [], [], []
        first_audit = not aud_ms
        t0 = time.time()
        for i, (case, _gt) in enumerate(pairs):
            if not pred.has_cloud(case.name):
                continue
            back_ms.append(ckpt.run(
                ckpt.key("bb", dev, case.name),
                lambda c=case: time_call(lambda: pred.predict(c), dev, a.repeats)))
            def _solve(c=case, eng=engine):
                return time_call(lambda: eng.solve(c), dev, a.repeats)
            solve_ms.append(ckpt.run(ckpt.key("solve", dev, case.name), _solve))
            pts.append(n_points.get(case.name, 0))

            if first_audit:
                def _audit(c=case):
                    field = pred.predict(c)
                    return time_call(
                        lambda: checker.diagnose(field, c).residual_norm(),
                        "cpu", a.repeats)
                aud_ms.append(ckpt.run(ckpt.key("audit", case.name), _audit))
            if (i + 1) % 25 == 0:
                log(f"  {i + 1}/{len(pairs)} cases ({(time.time() - t0) / (i + 1):.2f}s/case)")

        corr_ms = [s - b for s, b in zip(solve_ms, back_ms)]
        entry = {
            "backbone_ms": summarise(back_ms),
            "solve_ms": summarise(solve_ms),
            "correction_ms": summarise(corr_ms),
            "backbone_params": int(n_params),
            "points_per_case": {
                "min": int(min(pts)) if pts else None,
                "median": float(np.median(pts)) if pts else None,
                "max": int(max(pts)) if pts else None,
            },
            "scaling_vs_points": loglog_exponent(pts, back_ms),
        }

        if a.ensemble_k and a.ensemble_k > 0:
            members = []
            for k in range(a.ensemble_k):
                ck = os.path.join(a.ckpt_dir, f"seed{k}.pt")
                if not os.path.exists(ck):
                    log(f"  ensemble: seed{k}.pt missing, stopping at K={k}")
                    break
                m, pn_k, gn_k, _ = _load_backbone(ck, dev)
                members.append(PointCloudPredictor(m, test_pcs, pn_k, gn_k, device=dev))
            if members:
                ens_ms = []
                for case, _gt in pairs[: min(len(pairs), 25)]:
                    if not all(mm.has_cloud(case.name) for mm in members):
                        continue
                    ens_ms.append(ckpt.run(
                        ckpt.key("ens", dev, len(members), case.name),
                        lambda c=case: time_call(
                            lambda: [mm.predict(c) for mm in members],
                            dev, a.repeats)))
                entry["ensemble_ms"] = summarise(
                    ens_ms, {"k": len(members),
                             "note": "K independent backbones, first 25 cases"})
                if ens_ms and back_ms:
                    entry["ensemble_over_single"] = (
                        float(np.median(ens_ms)) / float(np.median(back_ms)))
            del members

        result["per_device"][dev] = entry
        del engine, pred, model, corrector
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    result["audit_ms"] = summarise(
        aud_ms, {"note": "diagnose + residual_norm on the CPU numpy path; "
                         "cross-check against results/control/audit_cost.json"})

    # ---------------- audit vs grid extent ---------------------------------- #
    if a.grid_scaling.strip():
        log("timing audit vs grid extent ...")
        res_list = [int(r) for r in a.grid_scaling.split(",") if r.strip()]
        dev0 = devices[0]
        model, point_norm, grid_norm, _ = _load_backbone(
            os.path.join(a.ckpt_dir, f"seed{a.seed}.pt"), dev0)
        pred = PointCloudPredictor(model, test_pcs, point_norm, grid_norm, device=dev0)
        probe = [(c, g) for c, g in pairs if pred.has_cloud(c.name)][:10]
        grid_rows = {}
        for r in res_list:
            ms = []
            for case, _gt in probe:
                def _grid(c=case, rr=r):
                    field = resample_field(pred.predict(c), rr)
                    return time_call(
                        lambda: checker.diagnose(field, c).residual_norm(),
                        "cpu", a.repeats)
                ms.append(ckpt.run(ckpt.key("grid", r, case.name), _grid))
            grid_rows[str(r)] = {"cells": r * r, "median_ms": float(np.median(ms)),
                                 "n_cases": len(ms)}
            log(f"  r={r} ({r * r} cells): {grid_rows[str(r)]['median_ms']:.3f} ms")
        result["audit_vs_grid"] = {
            "rows": grid_rows,
            "scaling_vs_cells": loglog_exponent(
                [v["cells"] for v in grid_rows.values()],
                [v["median_ms"] for v in grid_rows.values()]),
            "note": ("fields nearest-neighbour resampled purely to vary array extent; "
                     "not physically meaningful, no accuracy computed from them"),
        }
        del pred, model
        if dev0.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---------------- classical anchor -------------------------------------- #
    if a.classical_solve_sec:
        fastest = min(result["per_device"][d]["solve_ms"]["median_ms"]
                      for d in result["per_device"])
        result["classical_comparison"] = {
            "classical_solve_sec_per_case": a.classical_solve_sec,
            "classical_source": a.classical_source,
            "deployed_solve_median_ms": fastest,
            "speedup_classical_over_deployed": a.classical_solve_sec * 1e3 / fastest,
        }

    result["meta"]["host_fingerprint"] = host_id
    result["meta"]["progress_journal"] = a.progress or None
    result["meta"]["measurements_replayed_from_journal"] = int(
        sum(1 for _ in ckpt.done)) if ckpt.enabled else 0
    result["meta"]["runtime_sec"] = time.time() - t_start
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    log(f"wrote {a.out}  ({result['meta']['runtime_sec']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
