"""Discriminating control for the r2-brutal W1 finding: does the residual-as-OBJECTIVE
negative survive when the BC (no-slip) term the engine already computes is INCLUDED?

The paper's negative (tab:iters) and the residual-floor theorem part (i) use the
MONITORED residual `Diagnostics.residual_norm` = sqrt(mean(c^2+x^2+y^2)) -- BC-EXCLUDED.
The engine also computes `bc_violation` (proximity-weighted no-slip, ~3-cell band). This
script logs the BC-INCLUSIVE residual sqrt(residual_norm^2 + mean(bc^2)) alongside the
BC-excluded one and the field error, both (A) on truth vs uniform-freestream and (B)
across the DEQ correction sweep, on the existing certificates_deq.pt checkpoint. CPU-only.

Declared before running:
  - (A) expect uniform >> truth under BC-incl (part (i) reverses; near-trivial).
  - (B) DECISIVE: as iters rise and error FALLS, if BC-incl residual RISES -> negative is
    ROBUST (the floor is the discretisation/closure term, not missing BCs). If BC-incl
    FALLS with error -> negative was an artifact of dropping no-slip.
"""

import neuroforge  # noqa: F401  -- thread caps before numpy/torch

import json
import os

import numpy as np

from neuroforge.core.config import Config
from neuroforge.core.types import FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.physics.evaluation import per_channel_mse
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine

CKPT = "checkpoints/certificates_deq.pt"
N = 10
ITERS = [0, 1, 3, 5, 10, 15]


def main() -> None:
    cfg = Config()
    engine = NeuroForgeEngine.from_checkpoint(CKPT, config=cfg)
    checker = PhysicsChecker(cfg.physics)
    pairs = load_airfrans(task="full", train=False, resolution=128, limit=N,
                          cache_dir="data/cache", progress=False)
    print(f"loaded engine (corrector={type(engine.corrector).__name__}); {len(pairs)} cases", flush=True)

    def norms(field, case):
        d = checker.diagnose(field, case)
        rn = float(d.residual_norm())
        bc2 = float(np.mean(np.asarray(d.bc_violation) ** 2))
        return rn, float(np.sqrt(rn * rn + bc2))

    def uniform_field(case, field):
        u_in, v_in = case.bc.inlet_vector()
        sh = field.u.shape
        return FlowField(domain=field.domain,
                         u=np.full(sh, np.float32(u_in)), v=np.full(sh, np.float32(v_in)),
                         p=np.zeros(sh, np.float32), nut=np.zeros(sh, np.float32),
                         mask=field.mask, sdf=field.sdf, meta={"kind": "uniform"})

    # ---- Check A: truth vs uniform, BC-excl and BC-incl ----
    et, it_, eu, iu = [], [], [], []
    for case, ref in pairs:
        a, b = norms(ref, case)
        c, d = norms(uniform_field(case, ref), case)
        et.append(a); it_.append(b); eu.append(c); iu.append(d)
    A = {
        "truth_bc_excl": float(np.mean(et)), "truth_bc_incl": float(np.mean(it_)),
        "uniform_bc_excl": float(np.mean(eu)), "uniform_bc_incl": float(np.mean(iu)),
        "uniform_lt_truth_bc_excl": bool(np.mean(eu) < np.mean(et)),
        "uniform_lt_truth_bc_incl": bool(np.mean(iu) < np.mean(it_)),
    }
    print("\nCHECK A (truth vs uniform):", flush=True)
    print(f"  truth:   BC-excl {A['truth_bc_excl']:.4f}   BC-incl {A['truth_bc_incl']:.4f}")
    print(f"  uniform: BC-excl {A['uniform_bc_excl']:.4f}   BC-incl {A['uniform_bc_incl']:.4f}")
    print(f"  uniform<truth (BC-excl)? {A['uniform_lt_truth_bc_excl']}   (BC-incl)? {A['uniform_lt_truth_bc_incl']}")

    # ---- Check B: correction sweep ----
    corrector = engine.corrector
    saved = corrector.max_iter if corrector is not None else None
    sweep = []
    print("\nCHECK B (correction sweep):", flush=True)
    try:
        for ni in ITERS:
            if ni > 0 and corrector is not None:
                corrector.max_iter = int(ni)
            excl, incl, err = [], [], []
            for case, ref in pairs:
                field = engine.predictor.predict(case) if ni == 0 else engine.solve(case).field
                rn, bi = norms(field, case)
                excl.append(rn); incl.append(bi)
                err.append(per_channel_mse(field, ref)["mse_u"])
            row = {"n_iters": ni, "mse_u": float(np.mean(err)),
                   "bc_excl": float(np.mean(excl)), "bc_incl": float(np.mean(incl))}
            sweep.append(row)
            print(f"  iters={ni:2d}: error(mse_u)={row['mse_u']:.4f}  "
                  f"BC-excl={row['bc_excl']:.4f}  BC-incl={row['bc_incl']:.4f}", flush=True)
    finally:
        if saved is not None:
            corrector.max_iter = saved

    a, b = sweep[0], sweep[-1]
    d_err = b["mse_u"] - a["mse_u"]
    d_excl = b["bc_excl"] - a["bc_excl"]
    d_incl = b["bc_incl"] - a["bc_incl"]
    verdict = ("ROBUST-NEGATIVE" if (d_err < 0 and d_incl > 0)
               else "ARTIFACT-OF-BC-OMISSION" if (d_err < 0 and d_incl < 0)
               else "MIXED")
    print("\nVERDICT:", verdict, flush=True)
    print(f"  error  {a['mse_u']:.3f} -> {b['mse_u']:.3f}  ({d_err:+.3f})")
    print(f"  BC-excl {a['bc_excl']:.3f} -> {b['bc_excl']:.3f}  ({d_excl:+.3f})")
    print(f"  BC-incl {a['bc_incl']:.3f} -> {b['bc_incl']:.3f}  ({d_incl:+.3f})")

    os.makedirs("results/control", exist_ok=True)
    out = {"meta": {"checkpoint": CKPT, "n_cases": N, "iters": ITERS,
                    "bc_incl_def": "sqrt(residual_norm^2 + mean(bc_violation^2))"},
           "check_a_truth_vs_uniform": A, "check_b_sweep": sweep,
           "verdict": verdict,
           "deltas": {"error": d_err, "bc_excl": d_excl, "bc_incl": d_incl}}
    with open("results/control/bc_inclusive_sweep.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/control/bc_inclusive_sweep.json", flush=True)


if __name__ == "__main__":
    main()
