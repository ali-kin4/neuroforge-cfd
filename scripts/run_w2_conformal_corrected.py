"""W2 — split-conformal coverage on the DEPLOYED Transolver + DEQ-corrected field.

Why this run exists (the gap it closes)
---------------------------------------
The paper's published conformal headline (``results/uq_ensemble/uq_results.json``
``ensemble_conformal``: q=2.352, coverage=0.9155, ECE=0.074, ch=2, alpha=0.1)
certifies the ENSEMBLE-MEAN BACKBONE field. For the DEPLOYED, DEQ-corrected field
the paper currently only ARGUES the band "remains conservative" (tex ~909-942),
backed by an n=15 dropout-FNO directional analysis. This harness REPLACES that
argument with a direct n=100 cal / n=100 test MEASUREMENT on the shipped
Transolver+DEQ system, using the EXACT same frozen ensemble sigma and the EXACT
same conformal methodology as the published backbone number — so the corrected
result is apples-to-apples with it (a within-run, frozen-sigma contrast).

Methodology (must match ``run_ensemble_uq.ensemble_conformal`` bit-for-bit)
---------------------------------------------------------------------------
1. For each test case: stack the 5 ensemble member rasterised fields ->
   per-cell MEAN (m) and per-cell STD (sigma = ``arr.std(0)``, ddof=0). These are
   PHYSICAL (denormalised) grid fields straight off ``.predict(case).as_array()``,
   so sigma is NOT rescaled by std_out (that rescale only applies on the
   normalised ``cases_to_error_sigma`` path, which we deliberately do not use).
2. CORRECTED field (c): build a thin MeanPredictor whose ``.predict(case)``
   returns the ensemble-mean FlowField and whose ``.normalizer`` is the grid
   normaliser, wrap it in ``NeuroForgeEngine(mean_pred, checker, corrector)`` and
   take ``engine.solve(case).field`` — the DEPLOYED path (Neural Residual
   Iteration). uq=None so no fallback patch fires; the ONLY change vs the W1
   deploy path is that field0 is the ensemble mean instead of a single backbone.
3. Nonconformity score = |field - truth| / sigma per cell (fluid mask), EXACTLY
   as the existing code does for the backbone field — only the field whose error
   we score changes (m -> c). sigma is the FROZEN ensemble std for both.
4. Split 100 cal / 100 test with ``np.random.default_rng(0).permutation(200)``
   (same split as the published number), fit q on calibration scores, measure
   test coverage + ECE, per channel (u=0, v=1, p=2).

Corrector choice (resolved + documented in the output JSON ``meta``)
-------------------------------------------------------------------
The DEQ correctors (``checkpoints/v2_transolver/seed{0,1,2}_corr_with.pt``) were
trained on the INDIVIDUAL v2_transolver backbones, not on the ensemble mean. The
ensemble mean is the SAME TYPE of input (a Transolver-on-AirfRANS rasterised
field), so applying the corrector to it is the intended "deploy the corrector on
the ensemble prediction" semantics and is defensible. PRIMARY: seed0_corr_with.
``--correctors`` lets you also run seed1/seed2 to show robustness (each is a
separate corrected pass over the same cached ensemble fields).

Three numbers in the output, mapped to claims
---------------------------------------------
* ``backbone_conformal``  — recomputed on THIS split (self-check vs published).
* ``corrected_conformal`` — the proper certificate: q REFIT on corrected
  calibration scores. Because the split is a random in-distribution shuffle,
  exchangeability holds, so test coverage ~= 1-alpha is essentially GUARANTEED;
  the informative content is SHARPNESS (does corrected-q shrink vs backbone-q?).
* ``corrected_conformal_backbone_q`` — the BACKBONE q=2.352 applied to the
  corrected-field errors. This is the direct test of the tex claim that the
  corrected band "remains conservative": coverage >= target => conservative
  CONFIRMED. Uses the same cached corrected errors + sigma.

Faithfulness GATE (built in, not eyeballed)
-------------------------------------------
The harness re-derives the BACKBONE ch-2 conformal on its cal/test split and
checks it reproduces the published q~2.352 / coverage~0.9155. If it cannot, the
corrected number is untrustworthy -> a loud WARNING is logged and ``meta.gate``
is marked failed (does NOT hard-crash; the tiny CPU smoke with 2 members is
EXPECTED to miss the gate and that must not abort).

Resumability (survives shutdown / sleep)
----------------------------------------
The expensive part is the 5-member ensemble forward per case. It is cached to
``<cache>/ensemble/<case>.npz`` (mean+std+mask, 4 channels) with an atomic write
(tmp + os.replace) and a ``.done`` marker, CORRECTOR-INDEPENDENT so it is reused
across all correctors. The corrected field is cached per corrector to
``<cache>/corrected/<corrector_id>/<case>.npz``. Re-running the IDENTICAL command
skips every completed case instantly. The MeanPredictor reads the cached mean so
``engine.solve`` never re-forwards the members.

Always import ``neuroforge`` FIRST (BLAS thread cap) -- done at the top.

SMOKE (tiny CPU; also verifies resume -- run twice, 2nd run skips cached work):
    OMP_NUM_THREADS=8 .venv/Scripts/python.exe scripts/run_w2_conformal_corrected.py --smoke

FULL run (GPU; pre-registered -- DO NOT change n-cal/n-test/alpha/channels):
    .venv/Scripts/python.exe scripts/run_w2_conformal_corrected.py \
        --task full --n-val 200 --members 5 --resolution 128 --n-points 16384 \
        --width 256 --n-layers 10 --n-heads 8 --n-slices 32 \
        --alpha 0.1 --channels 0 1 2 --seed-split 0 --device auto \
        --correctors seed0 \
        --member-ckpt-dir checkpoints/uq_ensemble \
        --corrector-ckpt-dir checkpoints/v2_transolver \
        --cache-dir data/cache \
        --out-dir results/uq_ensemble \
        --w2-cache-dir data/cache/w2
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
from neuroforge.core.types import DTYPE, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.datamodule import FlowDataset, Normalizer
from neuroforge.data.pointcloud import (
    F_IN,
    F_OUT,
    PointNormalizer,
    load_airfrans_pointclouds,
)
from neuroforge.models import build_corrector
from neuroforge.models.baselines.transolver import TransolverPointModel
from neuroforge.physics.calibration import ConformalCalibrator, reliability
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor

# Published backbone headline (results/uq_ensemble/uq_results.json) — gate target.
_PUB_Q = 2.352087337434264
_PUB_COVERAGE = 0.9154961956739466
_GATE_Q_TOL = 0.05
_GATE_COV_TOL = 0.01

_CHANNEL_NAMES = {0: "u", 1: "v", 2: "p", 3: "nut"}


def log(msg: str) -> None:
    # Same "[v2] ..." grammar the live dashboard already parses.
    print(f"[v2] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _atomic_save_npz(path: str, **arrays) -> None:
    """Write an .npz atomically (tmp + os.replace) so a crash never corrupts it."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    # np.savez appends .npz to a name without it; tmp already ends in .npz so it
    # writes exactly `tmp`. os.replace is atomic on the same filesystem.
    os.replace(tmp, path)


def fit_grid_normalizer(train_pairs) -> Normalizer:
    """Fit a grid Normalizer on the GT grid pairs (field scale for rasterisation)."""
    ds = FlowDataset(train_pairs, normalizer=None)
    inputs, targets = ds.raw_arrays()
    return Normalizer().fit(inputs, targets)


# --------------------------------------------------------------------------- #
# A thin predictor that returns the (cached) ensemble-mean field. Drop-in for
# the v1 Predictor surface the engine / correction loop consume: it exposes
# ``predict(case) -> FlowField``, ``.normalizer`` (grid), and ``.device``.
# --------------------------------------------------------------------------- #
class MeanPredictor:
    """Serve the ensemble-mean FlowField so the deployed engine corrects IT.

    ``predict`` reads the per-case mean from the ensemble cache (so ``solve``
    never re-forwards the 5 members). It reconstructs a 4-channel FlowField with
    the case domain + the geometry mask/sdf from ``encode_case`` (exactly the
    channels ``PointCloudPredictor.predict`` attaches), so the correction loop's
    residual / trust machinery runs unchanged.
    """

    def __init__(self, mean_lookup, grid_normalizer, device: torch.device) -> None:
        self._mean = mean_lookup  # name -> (4, H, W) physical mean field
        self.normalizer = grid_normalizer
        self.device = device

    def predict(self, case) -> FlowField:
        from neuroforge.geometry.encode import encode_case

        arr = self._mean[case.name]
        stack = encode_case(case)
        sdf = stack[0].astype(DTYPE)
        mask = stack[1].astype(DTYPE)
        return FlowField.from_array(
            arr, case.domain, mask=mask, sdf=sdf,
            meta={"source": "ensemble-mean", "case": case.name},
        )

    def predict_tensor(self, x):  # parity with Predictor surface; unused here
        raise NotImplementedError("MeanPredictor serves cached fields only.")


# --------------------------------------------------------------------------- #
# Stage 1 — ensemble member forward (the cost), cached, corrector-independent.
# --------------------------------------------------------------------------- #
def _load_members(args, point_norm, grid_norm, test_pcs, device):
    """Load the M trained ensemble members as PointCloudPredictors (eval)."""
    predictors = []
    for m in range(args.members):
        ckpt = os.path.join(args.member_ckpt_dir, f"member{m}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"missing ensemble member checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        margs = state.get("args", {})
        model = TransolverPointModel(
            in_features=F_IN, out_features=F_OUT,
            width=int(margs.get("width", args.width)),
            n_layers=int(margs.get("n_layers", args.n_layers)),
            n_heads=int(margs.get("n_heads", args.n_heads)),
            n_slices=int(margs.get("n_slices", args.n_slices)),
            dropout=float(margs.get("dropout", 0.0)),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        predictors.append(
            PointCloudPredictor(model, test_pcs, point_norm, grid_norm, device=device)
        )
        log(f"loaded ensemble member {m} from {ckpt}")
    return predictors


def ensemble_forward_cached(args, predictors, pairs, ens_cache_dir):
    """Compute (or load) per-case ensemble mean + std (4ch) + mask. Resumable.

    Returns a dict ``name -> {'mean':(4,H,W), 'std':(4,H,W), 'mask':(H,W)}``.
    Caches each case to ``<ens_cache_dir>/<name>.npz`` atomically with a ``.done``
    marker. The 5-member forward is the expensive step -> cached once, reused by
    every corrector.
    """
    os.makedirs(ens_cache_dir, exist_ok=True)
    out = {}
    n_cached = n_computed = 0
    t_fwd_total = 0.0
    for case, _gt in pairs:
        npz = os.path.join(ens_cache_dir, f"{case.name}.npz")
        done = npz + ".done"
        if os.path.exists(done) and os.path.exists(npz):
            d = np.load(npz)
            out[case.name] = {"mean": d["mean"], "std": d["std"], "mask": d["mask"]}
            n_cached += 1
            continue
        if not all(p.has_cloud(case.name) for p in predictors):
            log(f"WARNING: no aligned cloud for {case.name} across all members — skipping")
            continue
        t0 = time.time()
        arr = np.stack([p.predict(case).as_array() for p in predictors])  # (M,4,H,W)
        t_fwd_total += time.time() - t0
        mean = arr.mean(0).astype(DTYPE)
        std = arr.std(0).astype(DTYPE)  # ddof=0, frozen input-conditional sigma
        # mask from the geometry (same source PointCloudPredictor uses).
        from neuroforge.geometry.encode import encode_case
        mask = encode_case(case)[1].astype(DTYPE)
        _atomic_save_npz(npz, mean=mean, std=std, mask=mask)
        with open(done, "w", encoding="utf-8") as f:
            f.write("ok\n")
        out[case.name] = {"mean": mean, "std": std, "mask": mask}
        n_computed += 1
        if n_computed % 10 == 0:
            log(f"ensemble forward: {n_computed} computed, {n_cached} cached "
                f"({t_fwd_total / max(n_computed,1):.2f}s/case forward)")
    sec_per_case = t_fwd_total / max(n_computed, 1) if n_computed else float("nan")
    speed = f"~{sec_per_case:.2f}s/case (5-member forward)" if n_computed else "(all cached)"
    log(f"ensemble forward done: {n_computed} computed, {n_cached} from cache; {speed}")
    return out, sec_per_case


# --------------------------------------------------------------------------- #
# Stage 2 — corrected field via the DEPLOYED engine path, cached per corrector.
# --------------------------------------------------------------------------- #
def _load_corrector(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    corrector = build_corrector(state["corrector_config"])
    corrector.load_state_dict(state["corrector_state"])
    corrector.eval()
    return corrector, state


def corrected_forward_cached(args, corrector_id, corrector_path, ens_cache,
                             pairs, grid_norm, checker, device, corr_cache_dir):
    """Compute (or load) the DEQ-corrected ensemble-mean field per case. Resumable.

    Returns ``name -> (4,H,W)`` corrected field. The engine wraps a MeanPredictor
    that serves the cached ensemble mean, so member forwards are never repeated.
    """
    os.makedirs(corr_cache_dir, exist_ok=True)
    corrector, _ = _load_corrector(corrector_path)
    mean_lookup = {n: ens_cache[n]["mean"] for n in ens_cache}
    mean_pred = MeanPredictor(mean_lookup, grid_norm, device)
    engine = NeuroForgeEngine(mean_pred, checker, corrector=corrector, config=Config())

    out = {}
    n_cached = n_computed = 0
    t_total = 0.0
    for case, _gt in pairs:
        if case.name not in ens_cache:
            continue
        npz = os.path.join(corr_cache_dir, f"{case.name}.npz")
        done = npz + ".done"
        if os.path.exists(done) and os.path.exists(npz):
            out[case.name] = np.load(npz)["corrected"]
            n_cached += 1
            continue
        t0 = time.time()
        field = engine.solve(case).field
        t_total += time.time() - t0
        corrected = field.as_array().astype(DTYPE)
        _atomic_save_npz(npz, corrected=corrected)
        with open(done, "w", encoding="utf-8") as f:
            f.write("ok\n")
        out[case.name] = corrected
        n_computed += 1
    sec_per_case = t_total / max(n_computed, 1) if n_computed else float("nan")
    speed = f"~{sec_per_case:.2f}s/case (DEQ solve)" if n_computed else "(all cached)"
    log(f"corrected[{corrector_id}] done: {n_computed} computed, {n_cached} cached; {speed}")
    return out, sec_per_case


# --------------------------------------------------------------------------- #
# Stage 3 — split-conformal coverage (matches ensemble_conformal exactly).
# --------------------------------------------------------------------------- #
def _split(pairs, seed):
    """Reproduce run_ensemble_uq's split: rng(seed).permutation, first/second half."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    half = max(1, len(pairs) // 2)
    cal = [pairs[i] for i in idx[:half]]
    test = [pairs[i] for i in idx[half:]] or cal
    return cal, test


def _err_sigma(pairs, field_lookup, ens_cache, channel):
    """Per-cell (error, sigma, mask) on `channel`, sigma = frozen ensemble std.

    The mask is ``gt.mask`` (NOT the cached encode-geometry mask) — bit-for-bit
    the source the published ``ensemble_conformal`` uses, so the backbone gate
    reproduces q=2.352 / coverage=0.9155 exactly (they are numerically identical
    on AirfRANS, but we mirror the published source to keep it apples-to-apples).
    """
    errs, sigs, masks = [], [], []
    for case, gt in pairs:
        if case.name not in ens_cache or case.name not in field_lookup:
            continue
        c = ens_cache[case.name]
        errs.append(field_lookup[case.name][channel] - gt.as_array()[channel])
        sigs.append(c["std"][channel])
        masks.append(gt.mask)
    return errs, sigs, masks


def conformal_one(cal_pairs, test_pairs, field_lookup, ens_cache, channel, alpha,
                  fixed_q=None):
    """Fit q on calibration scores (or use fixed_q) and report test coverage+ECE."""
    ce, cs, cm = _err_sigma(cal_pairs, field_lookup, ens_cache, channel)
    te, ts, tm = _err_sigma(test_pairs, field_lookup, ens_cache, channel)
    if not ce or not te:
        return {"status": "n/a (no aligned cases)", "channel": int(channel)}
    if fixed_q is None:
        cal = ConformalCalibrator(alpha=alpha).fit(ce, cs, cm)
        q = cal.q
    else:
        cal = ConformalCalibrator(alpha=alpha)
        cal.q = float(fixed_q)
        cal.fitted = True
        q = float(fixed_q)
    cov = cal.coverage(te, ts, tm)
    rel = reliability(te, ts, tm, n_bins=10, q=cal.q, alpha=alpha)
    return {
        "status": "ok", "channel": int(channel),
        "channel_name": _CHANNEL_NAMES.get(channel, str(channel)),
        "alpha": float(alpha), "q": float(q),
        "coverage": float(cov), "target_coverage": float(1.0 - alpha),
        "ece": float(rel["ece"]), "n_cal": len(ce), "n_test": len(te),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="W2: split-conformal coverage on the deployed Transolver+DEQ field."
    )
    p.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--n-train", type=int, default=800, help="for normaliser fit only")
    p.add_argument("--n-val", type=int, default=200, help="cal+test pool (100/100)")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--n-points", type=int, default=16384)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--w2-cache-dir", default="data/cache/w2")
    p.add_argument("--out-dir", default="results/uq_ensemble")
    p.add_argument("--member-ckpt-dir", default="checkpoints/uq_ensemble")
    p.add_argument("--corrector-ckpt-dir", default="checkpoints/v2_transolver")
    p.add_argument("--correctors", nargs="+", default=["seed0"],
                   help="which corrector(s): seed0 [seed1 seed2]; first is PRIMARY")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="auto")
    # backbone hyperparams (fallbacks; actual values read from each member ckpt)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-slices", type=int, default=32)
    # conformal
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seed-split", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)

    if a.smoke:
        # Tiny CPU end-to-end: 2 members, 8 cases (4 cal / 4 test), cpu, _smoke dirs.
        a.members = min(a.members, 2)
        a.n_val = 8
        a.n_train = 8
        a.device = "cpu"
        a.correctors = a.correctors[:1]
        a.out_dir = a.out_dir.rstrip("/\\") + "_smoke"
        a.w2_cache_dir = a.w2_cache_dir.rstrip("/\\") + "_smoke"
        log("SMOKE MODE: 2 members, 8 cases (4 cal / 4 test), cpu, _smoke dirs.")

    device = _resolve_device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.w2_cache_dir, exist_ok=True)
    log(f"device={device}  members={a.members}  correctors={a.correctors}")
    log("RESUMABLE: stop any time; re-run the SAME command to continue (cached "
        "ensemble forwards + corrected fields are skipped).")

    # ---- data (same loaders / order as run_ensemble_uq) ----
    log("loading point clouds + GT grid pairs ...")
    train_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=True, limit=a.n_train,
        cache_dir=a.cache_dir, download=a.download,
    )
    test_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=False, limit=a.n_val,
        cache_dir=a.cache_dir, download=a.download,
    )
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

    # ---- Stage 1: ensemble forward (cached, corrector-independent) ----
    predictors = _load_members(a, point_norm, grid_norm, test_pcs, device)
    ens_cache_dir = os.path.join(a.w2_cache_dir, "ensemble")
    ens_cache, sec_fwd = ensemble_forward_cached(a, predictors, test_pairs, ens_cache_dir)
    # Free the member backbones; only cached arrays are needed downstream.
    del predictors
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # backbone-mean field lookup (for the self-check) = the cached mean.
    backbone_lookup = {n: ens_cache[n]["mean"] for n in ens_cache}

    cal_pairs, test_pairs_split = _split(test_pairs, a.seed_split)
    log(f"split: {len(cal_pairs)} cal / {len(test_pairs_split)} test "
        f"(seed={a.seed_split})")

    # ---- self-check GATE: backbone ch-2 conformal vs published ----
    bb_ch2 = conformal_one(cal_pairs, test_pairs_split, backbone_lookup,
                           ens_cache, channel=2, alpha=a.alpha)
    gate_q_ok = (bb_ch2.get("status") == "ok"
                 and abs(bb_ch2["q"] - _PUB_Q) <= _GATE_Q_TOL)
    gate_cov_ok = (bb_ch2.get("status") == "ok"
                   and abs(bb_ch2["coverage"] - _PUB_COVERAGE) <= _GATE_COV_TOL)
    gate_passed = bool(gate_q_ok and gate_cov_ok)
    if a.smoke:
        log("SMOKE: gate is EXPECTED to miss (2 members, 8 cases) — not a failure.")
    if not gate_passed and not a.smoke:
        log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log(f"WARNING: FAITHFULNESS GATE FAILED. backbone ch2 q={bb_ch2.get('q')} "
            f"(pub {_PUB_Q:.3f}, tol {_GATE_Q_TOL}), coverage={bb_ch2.get('coverage')} "
            f"(pub {_PUB_COVERAGE:.4f}, tol {_GATE_COV_TOL}).")
        log("The corrected-field number is UNTRUSTWORTHY — investigate before "
            "citing. (Not hard-crashing; results still written for inspection.)")
        log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        log(f"GATE: backbone ch2 q={bb_ch2.get('q'):.4f} coverage={bb_ch2.get('coverage'):.4f} "
            f"-> {'PASS' if gate_passed else 'miss (smoke)'}")

    # ---- backbone conformal on all requested channels (contrast baseline) ----
    backbone_conformal = {
        _CHANNEL_NAMES.get(ch, str(ch)): conformal_one(
            cal_pairs, test_pairs_split, backbone_lookup, ens_cache, ch, a.alpha)
        for ch in a.channels
    }

    # ---- Stage 2+3: per corrector, corrected field + conformal ----
    corrected_results = {}
    corrector_meta = {}
    sec_corr_by_id = {}
    for cid in a.correctors:
        corr_path = os.path.join(a.corrector_ckpt_dir, f"{cid}_corr_with.pt")
        if not os.path.exists(corr_path):
            log(f"WARNING: corrector {corr_path} not found — skipping {cid}")
            continue
        _, cstate = _load_corrector(corr_path)
        corrector_meta[cid] = {
            "path": corr_path,
            "config": cstate.get("corrector_config"),
            "zero_residual": bool(cstate.get("zero_residual", False)),
            "trained_on": "individual v2_transolver backbone (seed-matched)",
        }
        corr_cache_dir = os.path.join(a.w2_cache_dir, "corrected", cid)
        corr_lookup, sec_corr = corrected_forward_cached(
            a, cid, corr_path, ens_cache, test_pairs, grid_norm, checker, device,
            corr_cache_dir)
        sec_corr_by_id[cid] = sec_corr

        # proper certificate: q REFIT on corrected calibration scores
        refit = {
            _CHANNEL_NAMES.get(ch, str(ch)): conformal_one(
                cal_pairs, test_pairs_split, corr_lookup, ens_cache, ch, a.alpha)
            for ch in a.channels
        }
        # "remains conservative" test: BACKBONE q applied to corrected errors
        bbq = {
            _CHANNEL_NAMES.get(ch, str(ch)): conformal_one(
                cal_pairs, test_pairs_split, corr_lookup, ens_cache, ch, a.alpha,
                fixed_q=backbone_conformal[_CHANNEL_NAMES.get(ch, str(ch))].get("q"))
            for ch in a.channels
        }
        corrected_results[cid] = {
            "corrected_conformal_refit_q": refit,
            "corrected_conformal_backbone_q": bbq,
        }
        log(f"[{cid}] corrected refit-q conformal: "
            + ", ".join(f"{k}: q={v.get('q', float('nan')):.3f} "
                        f"cov={v.get('coverage', float('nan')):.3f}"
                        for k, v in refit.items()))

    # ---- verdict (per channel, primary corrector) ----
    target = 1.0 - a.alpha
    primary = a.correctors[0] if a.correctors and a.correctors[0] in corrected_results else None
    verdict = {}
    if primary is not None:
        refit = corrected_results[primary]["corrected_conformal_refit_q"]
        bbq = corrected_results[primary]["corrected_conformal_backbone_q"]
        for ch in a.channels:
            name = _CHANNEL_NAMES.get(ch, str(ch))
            r = refit.get(name, {})
            b = bbq.get(name, {})
            verdict[name] = {
                "refit_q_coverage": r.get("coverage"),
                "refit_q_in_band": bool(r.get("coverage", 0.0) >= target - 1e-9),
                "refit_q": r.get("q"),
                "backbone_q": backbone_conformal.get(name, {}).get("q"),
                "sharper_than_backbone": bool(
                    r.get("q") is not None
                    and backbone_conformal.get(name, {}).get("q") is not None
                    and r.get("q") < backbone_conformal.get(name, {}).get("q")
                ),
                "backbone_q_on_corrected_coverage": b.get("coverage"),
                "remains_conservative": bool(b.get("coverage", 0.0) >= target - 1e-9),
            }

    # ---- timing extrapolation for the lead ----
    n_correctors_full = len(a.correctors)
    timing = {
        "sec_per_case_ensemble_forward": sec_fwd,
        "sec_per_case_deq_solve": sec_corr_by_id,
        "est_full_ensemble_forward_sec": (sec_fwd * a.n_val) if np.isfinite(sec_fwd) else None,
        "note": "ensemble forward is one-time (cached, corrector-independent); "
                "DEQ solve repeats per corrector. Multiply DEQ sec/case by n_val "
                "and by #correctors.",
    }

    # ---- persist ----
    out = {
        "meta": {
            "experiment": "W2 — conformal coverage on deployed Transolver+DEQ field",
            "task": a.task, "members": a.members,
            "member_ckpts": [os.path.join(a.member_ckpt_dir, f"member{m}.pt")
                             for m in range(a.members)],
            "n_cal": len(cal_pairs), "n_test": len(test_pairs_split),
            "alpha": a.alpha, "channels": a.channels, "seed_split": a.seed_split,
            "resolution": a.resolution,
            "uncertainty": "deep-ensemble per-cell std (FROZEN; identical to the "
                           "published backbone sigma)",
            "sigma_note": "sigma = arr.std(0) over members on PHYSICAL fields; NOT "
                          "rescaled by std_out (that applies only on the normalised "
                          "cases_to_error_sigma path, deliberately unused here).",
            "corrector_choice": {
                "primary": a.correctors[0] if a.correctors else None,
                "reasoning": "DEQ correctors were trained on the individual "
                             "v2_transolver backbones; the ensemble mean is the "
                             "SAME TYPE of Transolver-on-AirfRANS field, so applying "
                             "the corrector to it is the intended 'deploy the "
                             "corrector on the ensemble prediction' semantics.",
                "correctors": corrector_meta,
            },
            "method": "deployed engine path: NeuroForgeEngine(MeanPredictor, "
                      "checker, corrector).solve(case).field; uq=None (no fallback). "
                      "Only field0 (ensemble mean) differs from the W1 deploy path.",
            "device": str(device),
            "timing": timing,
        },
        "gate": {
            "published_q": _PUB_Q, "published_coverage": _PUB_COVERAGE,
            "measured_backbone_ch2": bb_ch2,
            "q_tol": _GATE_Q_TOL, "coverage_tol": _GATE_COV_TOL,
            "q_ok": bool(gate_q_ok), "coverage_ok": bool(gate_cov_ok),
            "gate_passed": gate_passed,
            "is_smoke": bool(a.smoke),
        },
        "backbone_conformal": backbone_conformal,
        "corrected_conformal": corrected_results,
        "verdict": verdict,
        "interpretation": {
            "refit_q": "PROPER certificate. Random in-distribution split => "
                       "exchangeability holds => test coverage ~= 1-alpha is "
                       "essentially guaranteed (a VALIDATION, not a surprise). The "
                       "informative content is SHARPNESS: does corrected-q shrink "
                       "vs backbone-q?",
            "backbone_q_on_corrected": "Direct test of the tex claim that the "
                       "corrected band 'remains conservative' under the backbone's "
                       "own multiplier: coverage >= target => CONSERVATIVE confirmed.",
            "exchangeability": "Holds here (random in-distribution case split, no "
                       "OOD). The distribution-free guarantee is legitimate. Do NOT "
                       "extend this coverage claim to OOD splits without flagging "
                       "the exchangeability breach.",
        },
    }
    out_path = os.path.join(a.out_dir, "w2_conformal_corrected.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {out_path}")

    log("================= SUMMARY =================")
    log(f"GATE passed={gate_passed} (smoke={a.smoke})")
    for name, v in verdict.items():
        log(f"  ch {name}: refit_q={v.get('refit_q')} (cov {v.get('refit_q_coverage')}, "
            f"in_band={v.get('refit_q_in_band')}, sharper={v.get('sharper_than_backbone')}); "
            f"backbone_q_on_corrected cov={v.get('backbone_q_on_corrected_coverage')} "
            f"(conservative={v.get('remains_conservative')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
