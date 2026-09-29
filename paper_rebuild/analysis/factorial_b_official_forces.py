"""Factorial cell (b): official-integrator forces for the existing Transolver arms.

Implements PREREG_factorial.md cell (b), unchanged:
  GATE : ground-truth native fields -> airfrans Simulation.force_coefficient
         (reference=False) must reproduce the official labels (reference=True):
         median |dC_D| < 1 count and Spearman rho_D > 0.999 over the 200 test cases.
         If the gate fails, no model force is written and the script exits non-zero.
  ARMS : Transolver seeds 0-4 (checkpoints/v2_transolver/seed{k}.pt), predicted
         native (u, v, p) -> the same integrator. nu_t is not used by the integrator
         (it uses the molecular viscosity at the wall).
  G0   : the point cloud's target velocity must equal Simulation.velocity node for
         node (max abs diff < 1e-6), so predictions are placed on the right nodes.
Outcome: per-case C_D, C_D_p, C_D_v, C_L for GT, official, each seed; the paired
comparison with the frozen reference R (null_reference.py, official split) is done in
factorial_b_compare.py so that this GPU script only produces forces.
Resumable: per-case results are appended to a JSONL file and skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import neuroforge  # noqa: F401
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "paper_rebuild", "results", "factorial_b_forces.jsonl")


def coeffs(sim, reference):
    (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient(compressible=False, reference=reference)
    return {"cd": float(cd), "cdp": float(cdp), "cdv": float(cdv),
            "cl": float(cl), "clp": float(clp), "clv": float(clv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import torch
    from airfrans.simulation import Simulation
    from neuroforge.data.pointcloud import load_airfrans_pointclouds
    from recompute_force_vs_official import load_backbone

    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                       else ("cpu" if a.device == "auto" else a.device))
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {json.loads(l)["name"] for l in fh if l.strip()}
    pcs = load_airfrans_pointclouds(root="data", task="full", train=False,
                                    limit=a.limit, cache_dir="data/cache")
    models = {s: load_backbone(os.path.join(ROOT, "checkpoints", "v2_transolver", f"seed{s}.pt"), dev)[:2]
              for s in a.seeds}
    root = os.path.join(ROOT, "data", "Dataset")
    t0 = time.time()
    with open(OUT, "a") as fh:
        for i, pc in enumerate(pcs):
            if pc.name in done:
                continue
            sim = Simulation(root, pc.name)
            g0 = float(np.max(np.abs(sim.velocity - pc.targets[:, :2])))
            assert g0 < 1e-6, f"G0 failed on {pc.name}: {g0}"
            rec = {"name": pc.name, "g0": g0, "official": coeffs(sim, True),
                   "gt": coeffs(sim, False)}
            for s, (m, pn) in models.items():
                feats = pn.transform_in(pc.features)
                with torch.no_grad():
                    y = m(torch.from_numpy(feats).float().to(dev).unsqueeze(0)).squeeze(0)
                    y = pn.inverse_out(y)
                pr = np.asarray(y.detach().cpu().numpy(), np.float64)
                sim.velocity = pr[:, :2].copy()
                sim.pressure = pr[:, 2:3].copy()
                rec[f"seed{s}"] = coeffs(sim, False)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(pcs)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
