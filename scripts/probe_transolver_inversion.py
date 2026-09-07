"""Does the prediction-BELOW-truth-floor inversion survive on the SOTA backbone?

The objection this script exists to settle
------------------------------------------
``docs/paper/sections/residual_floor_theorem.tex`` ("Empirical confirmation")
claims a trained backbone's prediction sits BELOW the truth floor
``||R_h(u*)||`` in 160/200 AirfRANS test cases. That number was measured on the
**dropout-FNO** of ``checkpoints/certificates_deq.pt`` -- the paper's WEAK
backbone, which the same paragraph calls "over-smoothed". A reviewer's first
move is therefore:

    "Your prediction has a lower residual than the truth because your model is
     over-smoothed. That is a fact about your weak model, not a fact about
     residual monitors."

If that stands, 160/200 cannot support the thesis. This script recomputes the
inversion statistic on the **Transolver** backbone (``checkpoints/v2_transolver``
-- the paper's SOTA backbone, the one the headline -9/-21/-25% DEQ result runs
on), over the SAME 200-case AirfRANS test split, per seed.

Why a negative here is EXPECTED and is not a loss
-------------------------------------------------
Linearising the monitored operator (paper Eq. rf-decomp):

    ||R_h(u_hat)||^2 ~ ||r*||^2 + 2 r*.Le + ||Le||^2 ,   e = u_hat - u*

Inversion (``||R_h(u_hat)|| < ||r*||``) requires ``r*.Le < -||Le||^2 / 2`` --
the prediction error must be systematically ANTI-ALIGNED with the floor.
Unbiased error of ANY magnitude drives the monitored residual UP. Since r* is
dominated by unresolved near-wall structure, "anti-aligned with r*" is close to
a definition of over-smoothing. So the reviewer's mechanism is a priori
plausible and the discriminating experiment must be allowed to return NO.

Either way the paper's model-free leg is untouched: the uniform-freestream field
scores EXACTLY 0 on 200/200 cases while the truth scores 0.192 -- a constant
field annihilates every finite-difference derivative, so that leg carries
Theorem (i) with no model in it at all.

What is measured (4 fields, one ruler)
--------------------------------------
Per case: (1) ground truth u* -- the floor; (2) uniform freestream; (3) the
Transolver backbone-alone prediction; (4) the Transolver + DEQ **deployed**
field the user actually receives. (3) and (4) are reported as SEPARATE rows and
never conflated -- the DEQ path applies its fixed-point delta with NO acceptance
test (``run_v2.py`` docstring), so it can and does move the monitored residual.

Smoothness diagnostic (the thing the objection is actually about)
-----------------------------------------------------------------
``norm_pred_std`` is reported for direct comparability with the published
dropout-FNO number, but it is the spread ACROSS CASES of a scalar norm -- NOT a
smoothness measure, and it cannot settle a smoothing question. The verdict is
carried by a **banded gradient-energy ratio**: the non-dimensional velocity
gradient energy of the prediction over that of the truth, measured separately in
the near-body and far-field bands. Bands are the PRE-REGISTERED sdf bands
already committed in ``scripts/control_cylinder_nearwall_artifact.py``
(near-body ``0 < sdf <= 0.15``, far-field ``sdf > 1.0``), intersected with fluid
and with the wall ring REMOVED so smoothing is measured exactly where
``diagnose`` actually evaluates the residual. Over-smoothed => ratio << 1.
Caveat stated in the output: the FNO predicts on the grid while Transolver
predicts on the cloud then rasterises, so cross-backbone gradient energy carries
a rasterisation confound; banding localises but does not remove it.

Measurement identity (two gates, both enforced at runtime)
----------------------------------------------------------
The norm is defined LOCALLY (byte-identical maths to
``scripts/probe_residual_floor.py``: ``PhysicsChecker.diagnose`` -> RMS over
fluid cells of ``sqrt(cont^2 + mom_x^2 + mom_y^2)``, BC term EXCLUDED, predicted
field re-wrapped with the TRUTH mask/sdf) rather than imported, because a
concurrent agent is editing that file and an import would be an unguarded race.
Identity is then PROVEN, not assumed:

  GATE A  recompute ``norm_truth`` on all 200 cases and require an exact match
          (names AND values) against the committed ``per_case`` block of
          ``results/certificates/residual_floor_realdata.json``.
  GATE B  re-derive the dropout-FNO's 160/200 inversion count from scratch.

If GATE A fails the two backbones are not on the same ruler and no comparison is
valid. If GATE B fails, that is a finding about the published number.

Cost / device
-------------
The Transolver forward is ~78 s/case on CPU (8 threads) => ~13 h for the
3 seeds x 200 cases matrix, so this stage runs on GPU (inference only, one
whole-cloud forward per case, ~2 GB). ``nvidia-smi`` MUST be re-checked
immediately before launching: never contend with a live training job. The
dropout-FNO stage is a small grid model and stays on CPU. The device actually
used is recorded in the output JSON. Writes ONE JSON; touches nothing under
``results/mgn/``, ``checkpoints/mgn/``.

Always import ``neuroforge`` FIRST (BLAS thread caps).

Run (full, the deciding measurement):
    OMP_NUM_THREADS=8 .venv/Scripts/python.exe scripts/probe_transolver_inversion.py \
        --device cuda --seeds 0 1 2

Smoke (fast, CPU, 3 cases, FNO gate relaxed):
    OMP_NUM_THREADS=8 .venv/Scripts/python.exe scripts/probe_transolver_inversion.py \
        --n 3 --seeds 0 --device cpu --pc-cache data/cache/airfrans_pc_full_test_n3.pkl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)
from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE, FlowField
from neuroforge.physics.residuals import PhysicsChecker, _solid_adjacent_fluid
from neuroforge.solver.correction_loop import neural_residual_iteration
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor

# scripts/ is on the path so we can reuse the committed checkpoint loaders
# (importing is NOT editing; these are the deployed-model loaders the published
# cylinder figure uses, so the backbone we score is the deployed one).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID_CACHE = "data/cache/airfrans_full_test_r128_n200.pkl"
PC_CACHE = "data/cache/airfrans_pc_full_test_n200.pkl"
REFERENCE_JSON = "results/certificates/residual_floor_realdata.json"
FNO_CKPT = "checkpoints/certificates_deq.pt"
TRANSOLVER_DIR = "checkpoints/v2_transolver"
OUT_JSON = "results/certificates/transolver_inversion.json"

# Pre-registered sdf bands, reused verbatim from the committed
# scripts/control_cylinder_nearwall_artifact.py (NOT invented here).
NEAR_DELTA = 0.15   # near-body: 0 < sdf <= 0.15
FAR_DELTA = 1.0     # far-field: sdf > 1.0

# Published dropout-FNO numbers under attack (results/certificates/...json).
PUBLISHED_FNO_INVERSIONS = 160
PUBLISHED_N = 200


def log(msg: str) -> None:
    print(f"[inv] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# The monitored norm -- byte-identical maths to scripts/probe_residual_floor.py
# --------------------------------------------------------------------------- #
def monitored_m2(diag) -> np.ndarray:
    """Per-cell squared monitored residual: continuity + momentum, BC EXCLUDED.

    diag.* are already the SCALED, solid- and wall-ring-zeroed maps from
    diagnose(). bc_violation is deliberately omitted: the monitored objective
    (physics_residual_torch) does not see the no-slip term.
    """
    cont = np.asarray(diag.continuity, dtype=np.float64)
    mx = np.asarray(diag.momentum_x, dtype=np.float64)
    my = np.asarray(diag.momentum_y, dtype=np.float64)
    return cont * cont + mx * mx + my * my


def rms_over_fluid(m2: np.ndarray, fluid: np.ndarray) -> float:
    fb = fluid > 0.5
    if not np.any(fb):
        return 0.0
    return float(np.sqrt(np.mean(m2[fb])))


def norm_of(field, case, truth, checker) -> float:
    """Monitored norm of ``field``, re-wrapped with the TRUTH mask/sdf.

    Re-wrapping is what probe_residual_floor.py does for the prediction, so the
    masking (solid + wall ring) is IDENTICAL across truth / uniform / prediction.
    """
    wrapped = FlowField(
        domain=truth.domain, u=field.u, v=field.v, p=field.p, nut=field.nut,
        mask=truth.mask, sdf=truth.sdf, meta={"kind": "scored"},
    )
    diag = checker.diagnose(wrapped, case)
    return rms_over_fluid(monitored_m2(diag), np.asarray(truth.mask, dtype=np.float64))


def uniform_field(case, truth) -> FlowField:
    """Constant freestream on the same grid, reusing truth mask/sdf."""
    u_in, v_in = case.bc.inlet_vector()
    shape = truth.shape
    return FlowField(
        domain=truth.domain,
        u=np.full(shape, np.float32(u_in), dtype=np.float32),
        v=np.full(shape, np.float32(v_in), dtype=np.float32),
        p=np.zeros(shape, dtype=np.float32),
        nut=np.zeros(shape, dtype=np.float32),
        mask=truth.mask, sdf=truth.sdf, meta={"kind": "uniform"},
    )


def field_finite(field) -> bool:
    for a in (field.u, field.v, field.p):
        if a is None or not np.all(np.isfinite(a)):
            return False
    return True


# --------------------------------------------------------------------------- #
# Smoothness: banded non-dimensional velocity gradient energy
# --------------------------------------------------------------------------- #
def band_masks(truth) -> dict[str, np.ndarray]:
    """Pre-registered sdf bands, fluid-only, wall ring REMOVED.

    The wall ring is dropped because diagnose() zeroes the residual there, so
    including it would measure smoothing where the monitor is blind.
    """
    fluid = np.asarray(truth.mask, dtype=DTYPE)
    fb = fluid > 0.5
    ring = _solid_adjacent_fluid(fluid)
    sdf = np.asarray(truth.sdf, dtype=np.float64)
    scored = fb & (~ring)
    return {
        "near_body": scored & (sdf > 0.0) & (sdf <= NEAR_DELTA),
        "far_field": scored & (sdf > FAR_DELTA),
        "all_fluid": scored,
    }


def grad_energy(field, case, truth, sel: np.ndarray) -> float:
    """Non-dimensional velocity gradient energy averaged over ``sel``.

    G = < (|grad u|^2 + |grad v|^2) > * (L / u_inf)^2  -- dimensionless, so it is
    comparable across cases and Reynolds numbers. Over-smoothed fields have
    systematically SMALLER G than the truth.
    """
    if not np.any(sel):
        return float("nan")
    dx = float(truth.domain.dx)
    dy = float(truth.domain.dy)
    u_inf = max(float(case.bc.u_inf), 1e-9)
    length = max(float(case.reference_length()), 1e-9)
    tot = np.zeros(truth.shape, dtype=np.float64)
    for comp in (field.u, field.v):
        a = np.asarray(comp, dtype=np.float64)
        gy, gx = np.gradient(a, dy, dx)
        tot += gx * gx + gy * gy
    return float(np.mean(tot[sel]) * (length / u_inf) ** 2)


# --------------------------------------------------------------------------- #
# Exact two-sided sign test (no scipy dependency)
# --------------------------------------------------------------------------- #
def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for k successes in n trials at p=0.5."""
    if n <= 0:
        return float("nan")
    probs = [math.comb(n, i) for i in range(n + 1)]
    total = float(sum(probs))
    obs = probs[k]
    tail = sum(p for p in probs if p <= obs * (1.0 + 1e-12))
    return float(min(1.0, tail / total))


# --------------------------------------------------------------------------- #
# Aggregation helper
# --------------------------------------------------------------------------- #
def summarise(norms: list[float], truths: list[float], label: str) -> dict:
    a = np.asarray(norms, dtype=np.float64)
    t = np.asarray(truths, dtype=np.float64)
    below = a < t
    k = int(below.sum())
    n = int(a.size)
    return {
        "label": label,
        "n_cases": n,
        "norm_mean": float(a.mean()),
        "norm_std": float(a.std()),
        "norm_median": float(np.median(a)),
        "n_below_truth_floor": k,
        "frac_below_truth_floor": k / n if n else float("nan"),
        "median_signed_gap_pred_minus_truth": float(np.median(a - t)),
        "mean_signed_gap_pred_minus_truth": float(np.mean(a - t)),
        "sign_test_p_two_sided": sign_test_p(k, n),
    }


def summarise_grad(ratios: dict[str, list[float]]) -> dict:
    out = {}
    for band, vals in ratios.items():
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
        out[band] = {
            "grad_energy_ratio_mean": float(v.mean()) if v.size else float("nan"),
            "grad_energy_ratio_median": float(np.median(v)) if v.size else float("nan"),
            "n": int(v.size),
        }
    return out


# --------------------------------------------------------------------------- #
# Stage: truth floor + uniform + GATE A
# --------------------------------------------------------------------------- #
def stage_truth(pairs, n, checker):
    per_case, truth_norms = [], []
    for i in range(n):
        case, truth = pairs[i]
        if not field_finite(truth):
            continue
        fluid = np.asarray(truth.mask, dtype=np.float64)
        diag_t = checker.diagnose(truth, case)
        r_truth = rms_over_fluid(monitored_m2(diag_t), fluid)
        r_unif = norm_of(uniform_field(case, truth), case, truth, checker)
        per_case.append({"index": i, "name": case.name,
                         "norm_truth": r_truth, "norm_uniform": r_unif})
        truth_norms.append(r_truth)
    return per_case, truth_norms


def gate_a(per_case) -> dict:
    """Require EXACT reproduction of the committed truth-floor per_case block."""
    if not os.path.exists(REFERENCE_JSON):
        return {"status": "reference JSON missing", "passed": False}
    ref = json.load(open(REFERENCE_JSON, encoding="utf-8"))["per_case"]
    ref_by_name = {r["name"]: r for r in ref}
    max_abs, n_matched, name_ok = 0.0, 0, True
    for rec in per_case:
        r = ref_by_name.get(rec["name"])
        if r is None:
            name_ok = False
            continue
        max_abs = max(max_abs, abs(r["norm_truth"] - rec["norm_truth"]))
        n_matched += 1
    # Order equality too, so the split alignment itself is proven.
    order_ok = all(
        ref[i]["name"] == per_case[i]["name"] for i in range(min(len(ref), len(per_case)))
    )
    passed = name_ok and order_ok and n_matched == len(per_case) and max_abs < 1e-9
    return {
        "status": "recomputed norm_truth vs committed residual_floor_realdata.json",
        "n_matched": n_matched, "max_abs_diff": max_abs,
        "names_all_found": name_ok, "order_identical": order_ok,
        "tolerance": 1e-9, "passed": bool(passed),
    }


# --------------------------------------------------------------------------- #
# Stage: dropout-FNO (GATE B) -- CPU, grid backbone
# --------------------------------------------------------------------------- #
def stage_fno(pairs, per_case, checker, n):
    import neuroforge as nf

    t0 = time.time()
    model = nf.NeuroForge.load(FNO_CKPT, device="cpu")
    predictor = model.predictor
    log(f"loaded dropout-FNO in {time.time() - t0:.1f}s")

    norms, truths = [], []
    ratios = {"near_body": [], "far_field": [], "all_fluid": []}
    t0 = time.time()
    for rec in per_case[:n]:
        case, truth = pairs[rec["index"]]
        pred = predictor.predict(case)
        norms.append(norm_of(pred, case, truth, checker))
        truths.append(rec["norm_truth"])
        bands = band_masks(truth)
        for band, sel in bands.items():
            gt = grad_energy(truth, case, truth, sel)
            gp = grad_energy(pred, case, truth, sel)
            ratios[band].append(gp / gt if gt > 0 else float("nan"))
    log(f"dropout-FNO: {len(norms)} cases in {time.time() - t0:.1f}s")

    summary = summarise(norms, truths, "dropout_FNO_backbone")
    summary["smoothness"] = summarise_grad(ratios)
    return summary, norms


# --------------------------------------------------------------------------- #
# Stage: Transolver backbone + deployed DEQ field
# --------------------------------------------------------------------------- #
def stage_transolver(pairs, per_case, checker, seeds, device, pc_cache, n):
    from make_cylinder_ood_figure import load_backbone, load_corrector

    log(f"loading point clouds {pc_cache} (once, reused across seeds) ...")
    t0 = time.time()
    with open(pc_cache, "rb") as fh:
        pcs = pickle.load(fh)
    log(f"loaded {len(pcs)} clouds in {time.time() - t0:.1f}s")

    cfg = Config()  # deployed defaults -- same as run_w1_capture.py
    out = {}
    for seed in seeds:
        bb = os.path.join(TRANSOLVER_DIR, f"seed{seed}.pt")
        cc = os.path.join(TRANSOLVER_DIR, f"seed{seed}_corr_with.pt")
        if not (os.path.exists(bb) and os.path.exists(cc)):
            log(f"seed {seed}: checkpoints missing -- skipped")
            continue
        model, point_norm, grid_norm, nu, _bcfg = load_backbone(bb, device)
        corrector = load_corrector(cc)
        predictor = PointCloudPredictor(model, pcs, point_norm, grid_norm, device=device)

        bb_norms, deq_norms, truths = [], [], []
        bb_ratios = {"near_body": [], "far_field": [], "all_fluid": []}
        deq_ratios = {"near_body": [], "far_field": [], "all_fluid": []}
        t0 = time.time()
        for j, rec in enumerate(per_case[:n]):
            case, truth = pairs[rec["index"]]
            if not predictor.has_cloud(case.name):
                continue
            # ONE backbone forward: field0 feeds both rows (engine.solve does
            # exactly this -- predict once, then run the loop on field0).
            field0 = predictor.predict(case)
            field_deq, _hist = neural_residual_iteration(
                field0, case, checker, corrector, cfg.correction, predictor, uq=None,
            )
            bb_norms.append(norm_of(field0, case, truth, checker))
            deq_norms.append(norm_of(field_deq, case, truth, checker))
            truths.append(rec["norm_truth"])
            bands = band_masks(truth)
            for band, sel in bands.items():
                gt = grad_energy(truth, case, truth, sel)
                bb_ratios[band].append(
                    grad_energy(field0, case, truth, sel) / gt if gt > 0 else float("nan"))
                deq_ratios[band].append(
                    grad_energy(field_deq, case, truth, sel) / gt if gt > 0 else float("nan"))
            if (j + 1) % 25 == 0:
                log(f"seed {seed}: {j + 1} cases, {time.time() - t0:.0f}s elapsed")
        dt = time.time() - t0
        log(f"seed {seed}: {len(bb_norms)} cases in {dt:.1f}s "
            f"({dt / max(len(bb_norms), 1):.2f}s/case)")

        s_bb = summarise(bb_norms, truths, f"transolver_backbone_seed{seed}")
        s_bb["smoothness"] = summarise_grad(bb_ratios)
        s_deq = summarise(deq_norms, truths, f"transolver_deq_deployed_seed{seed}")
        s_deq["smoothness"] = summarise_grad(deq_ratios)
        out[f"seed{seed}"] = {
            "backbone_alone": s_bb,
            "deq_deployed": s_deq,
            "seconds": dt,
        }
        del model, predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Inversion statistic on the Transolver (SOTA) backbone.")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda")
    p.add_argument("--pc-cache", default=PC_CACHE)
    p.add_argument("--out", default=OUT_JSON)
    p.add_argument("--skip-fno", action="store_true")
    p.add_argument("--skip-transolver", action="store_true")
    a = p.parse_args(argv)

    device = torch.device(a.device if a.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    wall0 = time.time()

    with open(GRID_CACHE, "rb") as fh:
        pairs = pickle.load(fh)
    n = min(a.n, len(pairs))
    checker = PhysicsChecker(Config().physics)

    log(f"truth floor + uniform on {n} cases ...")
    per_case, truth_norms = stage_truth(pairs, n, checker)
    tn = np.asarray(truth_norms, dtype=np.float64)
    un = np.asarray([c["norm_uniform"] for c in per_case], dtype=np.float64)
    gA = gate_a(per_case)
    log(f"GATE A (ruler identity): passed={gA['passed']} "
        f"max_abs_diff={gA['max_abs_diff']:.3e} order_ok={gA['order_identical']}")

    fno_summary, gB = None, None
    if not a.skip_fno:
        log("dropout-FNO stage (CPU) ...")
        fno_summary, _ = stage_fno(pairs, per_case, checker, n)
        k = fno_summary["n_below_truth_floor"]
        gB = {
            "status": "re-derived dropout-FNO inversion count from scratch",
            "recomputed": k, "published": PUBLISHED_FNO_INVERSIONS,
            "published_n": PUBLISHED_N, "n": fno_summary["n_cases"],
            "passed": bool(k == PUBLISHED_FNO_INVERSIONS
                           and fno_summary["n_cases"] == PUBLISHED_N),
        }
        log(f"GATE B (reproduce 160/200): recomputed={k}/{fno_summary['n_cases']} "
            f"passed={gB['passed']}")

    trans = {}
    if not a.skip_transolver:
        log(f"Transolver stage on {device} ...")
        trans = stage_transolver(pairs, per_case, checker, a.seeds, device, a.pc_cache, n)

    # ---- verdict -------------------------------------------------------- #
    bb_counts = [v["backbone_alone"]["n_below_truth_floor"] for v in trans.values()]
    deq_counts = [v["deq_deployed"]["n_below_truth_floor"] for v in trans.values()]
    n_scored = [v["backbone_alone"]["n_cases"] for v in trans.values()]
    if bb_counts:
        frac = np.mean([c / m for c, m in zip(bb_counts, n_scored)])
        verdict = ("INVERSION_SURVIVES" if frac >= 0.5
                   else "INVERSION_DISAPPEARS" if frac <= 0.2
                   else "INVERSION_WEAKENS")
    else:
        verdict = "NOT_RUN"

    wall = time.time() - wall0
    out = {
        "artifact": "transolver_inversion",
        "purpose": (
            "Test whether the prediction-below-truth-floor inversion (160/200, "
            "published on the over-smoothed dropout-FNO) survives on the paper's "
            "SOTA Transolver backbone and on the deployed Transolver+DEQ field."
        ),
        "verdict": verdict,
        "gates": {"gate_a_ruler_identity": gA, "gate_b_reproduce_fno_160": gB},
        "metadata": {
            "dataset": "AirfRANS task=full test split",
            "grid_cache": GRID_CACHE, "pointcloud_cache": a.pc_cache,
            "n_cases": len(per_case),
            "resolution": list(pairs[0][1].shape),
            "device_transolver": str(device),
            "device_fno": "cpu",
            "wall_clock_seconds": wall,
            "transolver_checkpoints": TRANSOLVER_DIR,
            "seeds_scored": list(trans.keys()),
            "seeds_available_not_scored": "seed3, seed4 exist; 0/1/2 are the paper's headline seeds",
            "monitored_residual": "continuity + momentum_x + momentum_y (BC EXCLUDED)",
            "norm": "RMS over fluid cells of sqrt(cont^2+mom_x^2+mom_y^2) on SCALED maps",
            "masking_policy": "solid + solid-adjacent fluid wall ring zeroed (diagnose)",
            "prediction_masking": "predicted fields re-wrapped with TRUTH mask/sdf",
            "deployed_field": (
                "neural_residual_iteration(field0, ..., Config().correction) with the "
                "seed-matched _corr_with DEQ corrector -- identical to "
                "NeuroForgeEngine.solve's path (predict once, then loop)"
            ),
            "smoothness_bands": {
                "near_body": f"0 < sdf <= {NEAR_DELTA}", "far_field": f"sdf > {FAR_DELTA}",
                "provenance": "pre-registered in scripts/control_cylinder_nearwall_artifact.py",
                "wall_ring": "removed (diagnose zeroes the residual there)",
            },
            "smoothness_metric": (
                "G = <|grad u|^2 + |grad v|^2> * (L/u_inf)^2, dimensionless; "
                "reported as the per-case ratio G_pred / G_truth. <1 => smoother "
                "than truth. norm_*_std is ALSO reported for comparability with the "
                "published number but is a spread across cases, NOT a smoothness measure."
            ),
            "smoothness_caveat": (
                "The FNO predicts on the grid; Transolver predicts on the native "
                "cloud and is then rasterised. Cross-backbone gradient-energy "
                "comparison therefore carries a rasterisation confound; banding "
                "localises but does not remove it."
            ),
        },
        "truth_floor": {
            "norm_truth_mean": float(tn.mean()), "norm_truth_std": float(tn.std()),
            "norm_truth_median": float(np.median(tn)),
            "norm_uniform_mean": float(un.mean()), "norm_uniform_std": float(un.std()),
            "n_uniform_lt_truth": int((un < tn).sum()), "n_cases": int(tn.size),
        },
        "dropout_fno": fno_summary,
        "transolver": trans,
        "aggregate": {
            "transolver_backbone_inversions_per_seed": bb_counts,
            "transolver_deq_inversions_per_seed": deq_counts,
            "n_scored_per_seed": n_scored,
            "dropout_fno_inversions": (
                fno_summary["n_below_truth_floor"] if fno_summary else None),
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)

    log("=============== SUMMARY ===============")
    log(f"truth floor ||r*|| mean={tn.mean():.4f} std={tn.std():.4f}; uniform=0 on "
        f"{int((un < tn).sum())}/{int(tn.size)}")
    if fno_summary:
        log(f"dropout-FNO      : {fno_summary['n_below_truth_floor']}/{fno_summary['n_cases']} "
            f"below floor, mean={fno_summary['norm_mean']:.4f} std={fno_summary['norm_std']:.4f}, "
            f"near-body grad ratio={fno_summary['smoothness']['near_body']['grad_energy_ratio_median']:.3f}")
    for sname, v in trans.items():
        b, d = v["backbone_alone"], v["deq_deployed"]
        log(f"transolver {sname} bb : {b['n_below_truth_floor']}/{b['n_cases']} below floor, "
            f"mean={b['norm_mean']:.4f} std={b['norm_std']:.4f}, "
            f"near-body grad ratio={b['smoothness']['near_body']['grad_energy_ratio_median']:.3f}")
        log(f"transolver {sname} deq: {d['n_below_truth_floor']}/{d['n_cases']} below floor, "
            f"mean={d['norm_mean']:.4f}")
    log(f"VERDICT: {verdict}   wall={wall:.0f}s   device={device}")
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
