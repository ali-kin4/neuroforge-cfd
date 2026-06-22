"""Does the residual track error PER-CELL (spatially within a field), or only PER-CASE?

Reviewer (domain-expert) W1: the headline residual_error_spearman is a PER-CASE
correlation (field-mean residual vs per-case error across cases), but the paper sells a
PER-CELL spatial trust layer ("flags WHERE"). This measures the spatial (within-field)
residual<->error correlation directly, on the same cases, and compares it to the per-case
metric. CPU-only; uses the certificates_deq.pt backbone (a representative deployed model).

Decision rule (declared up front): if the per-cell spatial correlation is decently
positive (mean >~0.3), the "where" claim is supported -> keep it (with this number). If it
is weak/near-zero, reframe the trust signal as PER-CASE design-triage ("which predictions
to distrust").
"""

import neuroforge  # noqa: F401

import json
import os

import numpy as np
from scipy.stats import spearmanr

from neuroforge.core.config import Config
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine

CKPT = "checkpoints/certificates_deq.pt"
N = 15


def main() -> None:
    cfg = Config()
    engine = NeuroForgeEngine.from_checkpoint(CKPT, config=cfg)
    checker = PhysicsChecker(cfg.physics)
    pairs = load_airfrans(task="full", train=False, resolution=128, limit=N,
                          cache_dir="data/cache", progress=False)
    print(f"loaded engine; {len(pairs)} cases", flush=True)

    pc_sp, pc_pe, case_r, case_e = [], [], [], []
    for case, ref in pairs:
        field = engine.solve(case).field
        diag = checker.diagnose(field, case)
        rmap = np.sqrt(diag.continuity ** 2 + diag.momentum_x ** 2 + diag.momentum_y ** 2)
        emap = np.abs(field.speed() - ref.speed())
        fluid = (np.asarray(ref.mask) > 0.5) if ref.mask is not None else np.ones_like(rmap, bool)
        valid = fluid & (rmap > 0)  # residual is zeroed on solid + wall-ring
        if valid.sum() < 50:
            continue
        r, e = rmap[valid], emap[valid]
        sp = spearmanr(r, e).correlation
        pe = float(np.corrcoef(r, e)[0, 1])
        if np.isfinite(sp):
            pc_sp.append(float(sp))
        if np.isfinite(pe):
            pc_pe.append(pe)
        # per-case scalars (reproduce the headline metric on the same data)
        case_r.append(float(np.sqrt(np.mean(rmap[fluid] ** 2))))
        num = float(np.sqrt(np.sum(emap[fluid] ** 2)))
        den = float(np.sqrt(np.sum(ref.speed()[fluid] ** 2))) + 1e-12
        case_e.append(num / den)

    case_sp = float(spearmanr(case_r, case_e).correlation)
    out = {
        "meta": {"checkpoint": CKPT, "n_cases": len(case_r),
                 "per_cell": "spatial Spearman/Pearson of |residual| vs |speed error| over fluid cells, within each field",
                 "per_case": "Spearman of field-mean residual_norm vs per-case speed rel-L2 (the paper's headline metric)"},
        "per_cell_spearman_mean": float(np.mean(pc_sp)), "per_cell_spearman_std": float(np.std(pc_sp)),
        "per_cell_pearson_mean": float(np.mean(pc_pe)), "per_cell_pearson_std": float(np.std(pc_pe)),
        "per_case_spearman": case_sp,
    }
    print("\nPER-CELL spatial residual<->error correlation (within field):", flush=True)
    print(f"  Spearman {out['per_cell_spearman_mean']:.3f} ± {out['per_cell_spearman_std']:.3f}")
    print(f"  Pearson  {out['per_cell_pearson_mean']:.3f} ± {out['per_cell_pearson_std']:.3f}")
    print(f"PER-CASE residual<->error Spearman (headline metric, n={len(case_r)}): {case_sp:.3f}")
    verdict = ("PER-CELL-SUPPORTED" if out["per_cell_spearman_mean"] >= 0.3
               else "PER-CELL-WEAK-REFRAME-TO-CASE-LEVEL")
    out["verdict"] = verdict
    print("VERDICT:", verdict, flush=True)
    os.makedirs("results/control", exist_ok=True)
    with open("results/control/percell_residual_error.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results/control/percell_residual_error.json", flush=True)


if __name__ == "__main__":
    main()
