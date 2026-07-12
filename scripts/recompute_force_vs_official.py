"""Recompute NeuroForge's AirfRANS force-coefficient ranking against the OFFICIAL
AirfRANS labels — replacing a metric that was measured against our own integrator.

WHY THIS EXISTS
---------------
The published ``rho_Cl=0.999 / rho_Cd=0.996`` (``results/v2/v2_results.json``) are
computed by :func:`neuroforge.physics.metrics.force_coefficients` applied to BOTH
the predicted field AND the rasterised GT field, then rank-correlated (see
``evaluate_cases`` -> ``coefficient_metrics``). That is **pred-integral vs
GT-integral through the SAME integrator** (which uses a +1.5-cell outward-normal
sampling offset). It measures the *self-consistency* of our integrator, not
agreement with the official OpenFOAM forces — the shared integrator bias cancels
in a rank correlation, inflating it. AirfRANS's headline result is that drag
ranking ``rho_D ~ 0`` defeats every baseline, so ``rho_Cd=0.996`` is not credible
as stated.

This harness recomputes the ranking against the OFFICIAL labels, obtained from
``airfrans.simulation.Simulation.force_coefficient(reference=True)`` (the
high-fidelity REFERENCE OpenFOAM fields = the official labels), matched per case
by simulation NAME.

THREE FORCE NUMBERS PER CASE
----------------------------
* ``pred_nf``      -- NeuroForge's own ``force_coefficients(pred_field, case)``
                      on the deployed Transolver prediction (the deployed integrator).
* ``gt_nf``        -- NeuroForge's own ``force_coefficients(gt_field, case)`` on the
                      rasterised GROUND-TRUTH field (the OLD comparison's other half).
* ``official_ref`` -- the official label ``Simulation.force_coefficient(reference=True)``.

(``pred_af`` -- airfrans' own integrator on our prediction -- is intentionally
SKIPPED: constructing a Simulation around an arbitrary predicted point cloud is
fiddly, and the ``gt_nf vs official`` correlation below already isolates the
integrator/rasterisation mismatch for free.)

CORRELATIONS REPORTED
---------------------
Per seed, against OFFICIAL labels (the new, credible number):
    rho_D_official = Spearman( pred_nf.cd , official.cd )
    rho_L_official = Spearman( pred_nf.cl , official.cl )
    cd/cl_rel_err_official = mean(|pred_nf - official| / |official|)

Per seed, OLD self-consistency (faithfulness reproduction of ~0.996/0.999):
    rho_D_selfconsistency = Spearman( pred_nf.cd , gt_nf.cd )
    rho_L_selfconsistency = Spearman( pred_nf.cl , gt_nf.cl )

ONCE (seed-independent), isolating integrator vs prediction error:
    rho_D_gt_vs_official = Spearman( gt_nf.cd , official.cd )
    rho_L_gt_vs_official = Spearman( gt_nf.cl , official.cl )
If THIS collapses, the integrator/rasterisation can't recover official ranking
even from perfect (GT) fields — the mismatch is in the measurement, not the model.

FAITHFULNESS BY CONSTRUCTION
----------------------------
``pred_nf`` / ``gt_nf`` / Spearman / rel-err all flow through the EXACT functions
that produced the v2 headline (``force_coefficients``, ``coefficient_metrics``,
``_spearman``). So the self-consistency column reproduces ~0.999/0.996 by
construction on the full set. The smoke only has to prove plumbing + that official
labels load and align.

PRE-COMMIT FRAMING: we will report whatever ``rho_D/rho_L`` vs official comes back,
including a collapse toward 0. (A strongly NEGATIVE ``rho_L`` vs official would be
a lift sign-convention difference -- NF wind-axis rotation vs AirfRANS -- not
genuine anti-ranking; reported raw with a note. rho is scale-invariant, so a large
``cd_rel_err`` alongside a healthy rho is a real integrator-bias finding.)

RESUMABILITY
------------
Per-seed ``pred_nf`` results cached to JSON (``<out>/_cache/seed{m}_prednf.json``)
with a ``.done`` marker; the seed-independent official-label dict and ``gt_nf`` are
cached too. Re-running the SAME command skips finished work. Append-log to stdout.

``import neuroforge`` FIRST (BLAS thread caps) — done at the top before
numpy/torch.

SMOKE (tiny; CPU; proves the full path + official-label loading + alignment):
    OMP_NUM_THREADS=8 .venv/Scripts/python.exe scripts/recompute_force_vs_official.py \
        --smoke --device cpu --n-val 6 --seeds 0 \
        --cache-dir data/cache --ckpt-dir checkpoints/v2_transolver \
        --out-dir results/control_smoke

FULL run (the deciding number — run on GPU; append-log):
    .venv/Scripts/python.exe scripts/recompute_force_vs_official.py \
        --task full --n-val 200 --seeds 0 1 2 --device auto \
        --resolution 128 --cache-dir data/cache \
        --ckpt-dir checkpoints/v2_transolver \
        --out-dir results/control \
        >> results/control/force_vs_official.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)
from neuroforge.core.types import DTYPE
from neuroforge.data.airfrans_loader import (  # noqa: E402
    AIRFRANS_NU,
    _resolve_data_root,
    load_airfrans,
)
from neuroforge.data.datamodule import Normalizer  # noqa: E402
from neuroforge.data.pointcloud import (  # noqa: E402
    PointNormalizer,
    load_airfrans_pointclouds,
)

# The EXACT functions that produced the v2 headline — reused verbatim so the
# self-consistency reproduction is faithful by construction.
from neuroforge.models.baselines.transolver import TransolverPointModel  # noqa: E402
from neuroforge.physics.evaluation import _spearman, coefficient_metrics  # noqa: E402
from neuroforge.physics.metrics import force_coefficients  # noqa: E402
from neuroforge.solver.pointcloud_predictor import PointCloudPredictor  # noqa: E402


def log(msg: str) -> None:
    print(f"[fvo] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# Backbone reload (identical pattern to make_cylinder_ood_figure.load_backbone).
# --------------------------------------------------------------------------- #
def load_backbone(ckpt_path: str, device: torch.device):
    """Load the deployed Transolver backbone + both normalisers from seed{m}.pt."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["backbone_config"]
    model = TransolverPointModel(
        in_features=int(cfg["in_features"]),
        out_features=int(cfg["out_features"]),
        width=int(cfg["width"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        n_slices=int(cfg["n_slices"]),
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
    nu = float(state.get("nu", AIRFRANS_NU))
    return model, point_norm, grid_norm, nu


# --------------------------------------------------------------------------- #
# Official labels (seed-independent; cached once).
# --------------------------------------------------------------------------- #
def official_labels(names: list[str], data_root: str, cache_path: str | None) -> dict[str, dict]:
    """Map ``name -> {'cl','cd'}`` from the official AirfRANS REFERENCE forces.

    Uses ``Simulation(root, name).force_coefficient(compressible=False,
    reference=True)`` per name (the high-fidelity OpenFOAM reference fields = the
    official labels). Cached to JSON; re-runs reuse it.
    """
    if cache_path and os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
        if all(nm in cached for nm in names):
            log(f"official labels: loaded {len(cached)} from cache {cache_path}")
            return cached

    import airfrans.simulation as afsim  # lazy

    out: dict[str, dict] = {}
    t0 = time.time()
    for i, nm in enumerate(names):
        sim = afsim.Simulation(root=data_root, name=nm)
        (cd, _cdp, _cdv), (cl, _clp, _clv) = sim.force_coefficient(
            compressible=False, reference=True
        )
        out[nm] = {"cl": float(cl), "cd": float(cd)}
        if (i + 1) % 25 == 0:
            log(f"official labels: {i + 1}/{len(names)} ({time.time() - t0:.0f}s)")
    log(f"official labels: computed {len(out)} REFERENCE force coeffs "
        f"({time.time() - t0:.0f}s)")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- #
# NF integrator on a list of fields -> list of {'cl','cd'}.
# --------------------------------------------------------------------------- #
def nf_coeffs_for_pairs(predict_fn, pairs) -> list[dict]:
    """Run ``predict_fn`` per case and apply the deployed ``force_coefficients``."""
    coeffs = []
    for case, _ref in pairs:
        field = predict_fn(case)
        c = force_coefficients(field, case)
        coeffs.append({"cl": float(c["cl"]), "cd": float(c["cd"]), "name": case.name})
    return coeffs


def gt_nf_coeffs(pairs) -> list[dict]:
    """NF integrator on the rasterised GROUND-TRUTH field (seed-independent)."""
    coeffs = []
    for case, ref in pairs:
        c = force_coefficients(ref, case)
        coeffs.append({"cl": float(c["cl"]), "cd": float(c["cd"]), "name": case.name})
    return coeffs


# --------------------------------------------------------------------------- #
# Alignment helpers — dict-by-name, never zip-by-index.
# --------------------------------------------------------------------------- #
def _aligned(coeff_list: list[dict], by_name: dict[str, dict], key: str) -> np.ndarray:
    """Pull ``key`` for each case in ``coeff_list`` from ``by_name`` (by name)."""
    return np.array([by_name[c["name"]][key] for c in coeff_list], np.float64)


def _self_lists(coeff_list: list[dict], key: str) -> np.ndarray:
    return np.array([c[key] for c in coeff_list], np.float64)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Recompute AirfRANS force-coeff ranking vs OFFICIAL labels."
    )
    p.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-val", type=int, default=200, help="cap test cases")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--ckpt-dir", default="checkpoints/v2_transolver")
    p.add_argument("--out-dir", default="results/control")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-chunk", type=int, default=0, help=">0 = chunked forward (OOM fallback)")
    p.add_argument("--download", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU run: few cases, prints a per-case table, writes _smoke JSON")
    a = p.parse_args(argv)

    if a.smoke:
        a.device = "cpu"
        if a.n_val > 8:
            a.n_val = 6
        if a.out_dir == "results/control":
            a.out_dir = "results/control_smoke"

    device = _resolve_device(a.device)
    eval_chunk = a.eval_chunk if a.eval_chunk and a.eval_chunk > 0 else None
    os.makedirs(a.out_dir, exist_ok=True)
    cache_sub = os.path.join(a.out_dir, "_cache")
    os.makedirs(cache_sub, exist_ok=True)
    out_name = "force_vs_official_smoke.json" if a.smoke else "force_vs_official.json"

    log(f"device={device}  smoke={a.smoke}  seeds={a.seeds}  n_val={a.n_val}  "
        f"ckpt-dir={a.ckpt_dir}  out-dir={a.out_dir}")

    data_root = _resolve_data_root(a.root)
    if data_root is None:
        log(f"FATAL: AirfRANS manifest.json not found under '{a.root}' / '{a.root}/Dataset'. "
            "Pass --download or fetch the dataset first.")
        return 2
    log(f"data_root={data_root}")

    # ---- data: native clouds (backbone input) + GT grid pairs (gt_nf + cases) ----
    log("loading native test point clouds ...")
    test_pcs = load_airfrans_pointclouds(
        root=a.root, task=a.task, train=False, limit=a.n_val,
        cache_dir=a.cache_dir, download=a.download,
    )
    log("loading rasterised GT grid pairs (shared with NeuroForge eval) ...")
    test_pairs = load_airfrans(
        root=a.root, task=a.task, train=False, resolution=a.resolution,
        limit=a.n_val, cache_dir=a.cache_dir, download=a.download, progress=False,
    )
    names = [case.name for case, _ in test_pairs]
    log(f"loaded {len(test_pairs)} GT pairs / {len(test_pcs)} clouds")

    # ---- alignment guard: every test case must have a cloud (by name) ----
    cloud_names = {pc.name for pc in test_pcs}
    missing_clouds = [nm for nm in names if nm not in cloud_names]
    if missing_clouds:
        log(f"FATAL: {len(missing_clouds)} test cases have NO point cloud (name mismatch). "
            f"First offenders: {missing_clouds[:5]}")
        log("Aborting (silent misalignment would corrupt the number).")
        return 3

    # ---- official labels (seed-independent; cached) ----
    label_cache = os.path.join(cache_sub, f"official_labels_{a.task}_test_n{a.n_val}.json")
    official = official_labels(names, data_root, label_cache)
    missing_lbl = [nm for nm in names if nm not in official]
    if missing_lbl:
        log(f"FATAL: {len(missing_lbl)} test cases have NO official label (name mismatch). "
            f"First offenders: {missing_lbl[:5]}")
        return 3

    # ---- gt_nf (seed-independent; cached) ----
    gt_cache = os.path.join(cache_sub, f"gt_nf_{a.task}_test_r{a.resolution}_n{a.n_val}.json")
    if os.path.exists(gt_cache):
        gt_nf = json.load(open(gt_cache, encoding="utf-8"))
        log(f"gt_nf: loaded {len(gt_nf)} from cache")
    else:
        log("gt_nf: integrating NeuroForge force_coefficients on GT fields ...")
        t0 = time.time()
        gt_nf = gt_nf_coeffs(test_pairs)
        log(f"gt_nf: done {len(gt_nf)} ({time.time() - t0:.1f}s)")
        with open(gt_cache, "w", encoding="utf-8") as f:
            json.dump(gt_nf, f, indent=2)

    # gt_nf vs official (isolates integrator/rasterisation mismatch; seed-independent)
    gt_cd = _self_lists(gt_nf, "cd")
    gt_cl = _self_lists(gt_nf, "cl")
    off_cd_gt = _aligned(gt_nf, official, "cd")
    off_cl_gt = _aligned(gt_nf, official, "cl")
    rho_D_gt_vs_official = _spearman(gt_cd, off_cd_gt)
    rho_L_gt_vs_official = _spearman(gt_cl, off_cl_gt)
    log(f"gt_nf vs OFFICIAL (integrator-only):  rho_D={rho_D_gt_vs_official:.4f}  "
        f"rho_L={rho_L_gt_vs_official:.4f}")

    # ---- per-seed: pred_nf ----
    per_seed: list[dict] = []
    smoke_table = None
    for seed in a.seeds:
        bb_ckpt = os.path.join(a.ckpt_dir, f"seed{seed}.pt")
        if not os.path.exists(bb_ckpt):
            log(f"FATAL: backbone checkpoint not found: {bb_ckpt}")
            return 4

        prednf_cache = os.path.join(cache_sub, f"seed{seed}_prednf.json")
        prednf_done = os.path.join(cache_sub, f"seed{seed}_prednf.done")
        if os.path.exists(prednf_done):
            pred_nf = json.load(open(prednf_cache, encoding="utf-8"))
            log(f"seed {seed}: pred_nf already complete — loaded {len(pred_nf)} from cache")
        else:
            log(f"seed {seed}: loading backbone {bb_ckpt} ...")
            model, point_norm, grid_norm, _nu = load_backbone(bb_ckpt, device)
            predictor = PointCloudPredictor(
                model, test_pcs, point_norm, grid_norm, device=device, chunk=eval_chunk
            )
            log(f"seed {seed}: predicting + integrating pred_nf over {len(test_pairs)} cases ...")
            t0 = time.time()
            pred_nf = nf_coeffs_for_pairs(predictor.predict, test_pairs)
            log(f"seed {seed}: pred_nf done {len(pred_nf)} ({time.time() - t0:.1f}s)")
            with open(prednf_cache, "w", encoding="utf-8") as f:
                json.dump(pred_nf, f, indent=2)
            with open(prednf_done, "w", encoding="utf-8") as f:
                f.write(f"seed {seed} pred_nf complete: {len(pred_nf)} cases\n")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # --- correlations for this seed ---
        # pred_nf vs OFFICIAL  (the new credible number)
        p_cd = _self_lists(pred_nf, "cd")
        p_cl = _self_lists(pred_nf, "cl")
        off_cd = _aligned(pred_nf, official, "cd")
        off_cl = _aligned(pred_nf, official, "cl")
        off_list = [official[c["name"]] for c in pred_nf]  # aligned dicts for rel-err
        m_off = coefficient_metrics(pred_nf, off_list)

        # pred_nf vs gt_nf  (OLD self-consistency — reproduce ~0.999/0.996)
        gtnf_by_name = {c["name"]: c for c in gt_nf}
        gt_list = [gtnf_by_name[c["name"]] for c in pred_nf]
        m_self = coefficient_metrics(pred_nf, gt_list)

        seed_rec = {
            "seed": int(seed),
            "n_cases": len(pred_nf),
            # vs OFFICIAL (rho_cd == drag == rho_D; rho_cl == lift == rho_L)
            "rho_D_official": float(m_off["rho_cd"]),
            "rho_L_official": float(m_off["rho_cl"]),
            "cd_rel_err_official": float(m_off["cd_rel_err_mean"]),
            "cl_rel_err_official": float(m_off["cl_rel_err_mean"]),
            "cd_rel_err_official_std": float(m_off["cd_rel_err_std"]),
            "cl_rel_err_official_std": float(m_off["cl_rel_err_std"]),
            # OLD self-consistency (pred_nf vs gt_nf)
            "rho_D_selfconsistency": float(m_self["rho_cd"]),
            "rho_L_selfconsistency": float(m_self["rho_cl"]),
        }
        per_seed.append(seed_rec)
        log(f"seed {seed}: vs OFFICIAL  rho_D={seed_rec['rho_D_official']:.4f}  "
            f"rho_L={seed_rec['rho_L_official']:.4f}  "
            f"cd_relerr={seed_rec['cd_rel_err_official']:.3f}  "
            f"cl_relerr={seed_rec['cl_rel_err_official']:.3f}")
        log(f"seed {seed}: self-consistency  rho_D={seed_rec['rho_D_selfconsistency']:.4f}  "
            f"rho_L={seed_rec['rho_L_selfconsistency']:.4f}  (target ~0.996/0.999)")

        # In smoke, build a per-case eyeball table for seed0.
        if a.smoke and smoke_table is None:
            smoke_table = []
            for c in pred_nf:
                nm = c["name"]
                smoke_table.append({
                    "name": nm,
                    "official_cd": official[nm]["cd"], "official_cl": official[nm]["cl"],
                    "gt_nf_cd": gtnf_by_name[nm]["cd"], "gt_nf_cl": gtnf_by_name[nm]["cl"],
                    "pred_nf_cd": c["cd"], "pred_nf_cl": c["cl"],
                })

    # ---- aggregate mean +/- std over seeds ----
    def agg(key):
        vals = [d[key] for d in per_seed if np.isfinite(d.get(key, np.nan))]
        return {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "per_seed": [d.get(key, float("nan")) for d in per_seed],
        }

    aggregate = {
        k: agg(k)
        for k in (
            "rho_D_official", "rho_L_official",
            "cd_rel_err_official", "cl_rel_err_official",
            "rho_D_selfconsistency", "rho_L_selfconsistency",
        )
    }

    out = {
        "meta": {
            "purpose": "Recompute AirfRANS force-coeff RANKING vs OFFICIAL labels, "
                       "replacing the self-consistency metric in results/v2/v2_results.json.",
            "task": a.task, "split": "test", "seeds": a.seeds, "n_val": a.n_val,
            "resolution": a.resolution, "device": str(device), "smoke": bool(a.smoke),
            "ckpt_dir": a.ckpt_dir, "data_root": data_root,
            "what_each_is_correlated_against": {
                "rho_D_official": "Spearman( NF force_coefficients(pred).cd , "
                                  "airfrans Simulation.force_coefficient(reference=True).cd ) "
                                  "[drag, official OpenFOAM reference labels]",
                "rho_L_official": "Spearman( NF force_coefficients(pred).cl , "
                                  "official reference .cl ) [lift, official labels]",
                "cd_rel_err_official": "mean(|pred_nf.cd - official.cd| / |official.cd|)",
                "cl_rel_err_official": "mean(|pred_nf.cl - official.cl| / |official.cl|)",
                "rho_D_selfconsistency": "Spearman( pred_nf.cd , gt_nf.cd ) -- pred-integral vs "
                                         "GT-integral through the SAME NF integrator (the OLD "
                                         "metric; reproduces ~0.996 by construction).",
                "rho_L_selfconsistency": "Spearman( pred_nf.cl , gt_nf.cl ) -- OLD metric "
                                         "(reproduces ~0.999 by construction).",
                "rho_D/L_gt_vs_official": "Spearman( gt_nf , official ) -- seed-independent; "
                                          "isolates integrator/rasterisation mismatch from "
                                          "prediction error. Collapse here => the measurement "
                                          "(not the model) cannot recover official ranking.",
            },
            "pred_af_skipped": "airfrans' own integrator on our prediction skipped; "
                               "gt_nf-vs-official isolates the integrator mismatch for free.",
            "notes": [
                "rho_cd==drag==rho_D ; rho_cl==lift==rho_L (key mapping pinned).",
                "A strongly NEGATIVE rho_L vs official would be a lift sign-convention "
                "difference (NF wind-axis rotation vs AirfRANS), not genuine anti-ranking.",
                "rho is scale-invariant: large cd_rel_err with healthy rho is a real "
                "integrator-bias finding, not a bug.",
            ],
        },
        "integrator_only": {
            "rho_D_gt_vs_official": float(rho_D_gt_vs_official),
            "rho_L_gt_vs_official": float(rho_L_gt_vs_official),
            "n_cases": len(gt_nf),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    if a.smoke and smoke_table is not None:
        out["smoke_per_case_table_seed0"] = smoke_table

    out_path = os.path.join(a.out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {out_path}")

    # ---- summary ----
    log("================= SUMMARY =================")
    log(f"integrator-only (gt_nf vs official):  rho_D={rho_D_gt_vs_official:.4f}  "
        f"rho_L={rho_L_gt_vs_official:.4f}")
    log("vs OFFICIAL (mean+/-std over seeds):  "
        f"rho_D={aggregate['rho_D_official']['mean']:.4f}"
        f"+/-{aggregate['rho_D_official']['std']:.4f}  "
        f"rho_L={aggregate['rho_L_official']['mean']:.4f}"
        f"+/-{aggregate['rho_L_official']['std']:.4f}")
    log("self-consistency (OLD; mean+/-std):  "
        f"rho_D={aggregate['rho_D_selfconsistency']['mean']:.4f}"
        f"+/-{aggregate['rho_D_selfconsistency']['std']:.4f}  "
        f"rho_L={aggregate['rho_L_selfconsistency']['mean']:.4f}"
        f"+/-{aggregate['rho_L_selfconsistency']['std']:.4f}  (target ~0.996/0.999)")

    if a.smoke and smoke_table is not None:
        log("--- per-case eyeball (seed0): name | official(cd,cl) | gt_nf(cd,cl) | pred_nf(cd,cl) ---")
        for r in smoke_table:
            log(f"  {r['name'][:34]:34s}  off=({r['official_cd']:+.4f},{r['official_cl']:+.4f})  "
                f"gt=({r['gt_nf_cd']:+.4f},{r['gt_nf_cl']:+.4f})  "
                f"pred=({r['pred_nf_cd']:+.4f},{r['pred_nf_cl']:+.4f})")
        log("NOTE: Spearman on a handful of cases is meaningless — smoke proves the "
            "pipeline + official-label loading + alignment, NOT the final rho.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
