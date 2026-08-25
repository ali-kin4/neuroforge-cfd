"""Paper-2 gate: is a full-domain classical solve at 128^2 expensive enough to be worth avoiding?

The trust-triggered fallback only earns its keep if the classical work it replaces
actually costs something. If a from-scratch solve of the whole domain finishes in
milliseconds, there is no compute to allocate and the hybrid has nothing to sell --
the experiment has to move to a larger grid or a costlier physics setting before any
of the sweep machinery gets built.

So this measures the one ratio that decides it:

    t_full_classical / t_neural_forward

on the cached 128^2 AirfRANS test cases, with the same BLAS thread cap the rest of
the repo uses. Two classical arms are timed, because they bracket the honest cost:

  cold  -- start from uniform freestream (u=u_in, v=v_in, p=0, nut=0). This is what a
           practitioner gets with no surrogate. Note nu_eff is then laminar only: the
           solver freezes nu_eff from the field it is handed and does not evolve a
           turbulence model, so cold-start is *not* a faithful RANS baseline. It is
           timed as a cost probe, not as an accuracy baseline.
  warm  -- start from the reference field, i.e. the best case a perfect surrogate
           could hand the solver. Generous to the hybrid on purpose.

Writes results/control/fallback_cost_smoke.json. Nothing here is a paper claim; it
is a go/no-go on the premise.

    python scripts/smoke_fallback_cost.py [--n 10] [--max-outer 200]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import platform
import time
from pathlib import Path

import neuroforge  # noqa: F401  -- caps BLAS threads before numpy import
import numpy as np

from neuroforge.core.types import FlowField
from neuroforge.solver.classical import local_incompressible_solve, region_residual_norm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "cache" / "airfrans_full_test_r128_n10.pkl"
DEFAULT_OUT = ROOT / "results" / "control" / "fallback_cost_smoke.json"


def uniform_field(case, ref: FlowField) -> FlowField:
    """A cold start: freestream everywhere, no turbulence, zero pressure."""
    shape = ref.u.shape
    u_in = float(getattr(case.bc, "u_inf", 1.0))
    v_in = float(getattr(case.bc, "v_inf", 0.0))
    return FlowField(
        domain=ref.domain,
        u=np.full(shape, u_in),
        v=np.full(shape, v_in),
        p=np.zeros(shape),
        nut=np.zeros(shape),
        mask=ref.mask,
        sdf=ref.sdf,
    )


def time_full_solve(field: FlowField, case, max_outer: int) -> dict:
    """Wall-clock of a classical solve over every fluid cell."""
    region = np.asarray(field.mask) > 0.5
    t0 = time.perf_counter()
    out = local_incompressible_solve(field, case, region, max_outer=max_outer)
    dt = time.perf_counter() - t0
    meta = (out.meta or {}).get("fallback", {})
    return {
        "seconds": dt,
        "cells_solved": int(region.sum()),
        "accepted": bool(meta.get("ran", False)),
        "residual_before": meta.get("residual_before"),
        "residual_after": meta.get("residual_after"),
    }


def time_neural_forward(cases, fields) -> dict | None:
    """Median wall-clock of one surrogate forward pass, if a checkpoint is available."""
    try:
        import torch

        from neuroforge.solver.engine import NeuroForgeEngine
    except Exception as exc:  # pragma: no cover - environment probe
        return {"available": False, "why": f"import failed: {exc}"}

    try:
        pred = NeuroForgeEngine.pretrained().predictor
    except Exception as exc:
        return {"available": False, "why": f"pretrained() failed: {exc}"}

    times = []
    with torch.no_grad():
        pred.predict(cases[0])  # warm-up, untimed
        for case in cases:
            t0 = time.perf_counter()
            pred.predict(case)
            times.append(time.perf_counter() - t0)
    return {"available": True, "median_seconds": float(np.median(times)),
            "device": "cpu", "n": len(times)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--n", type=int, default=10, help="cases to time")
    ap.add_argument("--max-outer", type=int, default=200,
                    help="SIMPLE outer iterations (solver default is 200)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    items = pickle.load(open(args.cache, "rb"))[: args.n]
    print(f"[smoke] {len(items)} cases from {args.cache.name}, max_outer={args.max_outer}")

    # One untimed solve first: the initial call pays scipy's sparse-solver setup
    # (measured at ~9x the steady-state cost), which would otherwise land entirely
    # on case 0 and inflate the cold median.
    warm_case, warm_ref = items[0]
    time_full_solve(uniform_field(warm_case, warm_ref), warm_case, args.max_outer)

    cold, warm = [], []
    for i, (case, ref) in enumerate(items):
        c = time_full_solve(uniform_field(case, ref), case, args.max_outer)
        w = time_full_solve(ref, case, args.max_outer)
        cold.append(c)
        warm.append(w)
        print(f"  case {i:2d}  cold {c['seconds']:7.3f}s  warm {w['seconds']:7.3f}s  "
              f"cells {c['cells_solved']}")

    neural = time_neural_forward([c for c, _ in items], [f for _, f in items])
    med_cold = float(np.median([r["seconds"] for r in cold]))
    med_warm = float(np.median([r["seconds"] for r in warm]))

    def _ok(rows):
        return sum(1 for r in rows
                   if r["accepted"] and r["residual_after"] is not None
                   and np.isfinite(r["residual_after"]))

    n_ok_cold, n_ok_warm = _ok(cold), _ok(warm)
    n_nan = sum(1 for r in cold + warm
                if r["residual_after"] is None or not np.isfinite(r["residual_after"]))

    if n_ok_cold == 0 and n_ok_warm == 0:
        verdict = ("INCONCLUSIVE -- the full-domain solve DIVERGED on every case "
                   "(0 accepted by the no-harm gate). The timings above are time-to-blow-up, "
                   "not time-to-converge, so they do not answer the cost question. The FULL "
                   "baseline the hybrid must beat cannot currently be run at this scope.")
    elif med_cold >= 1.0:
        verdict = "PREMISE HOLDS -- there is compute worth allocating"
    else:
        verdict = ("PREMISE WEAK -- a full solve is cheap at this grid; move to a larger "
                   "grid or costlier physics before building the sweep")

    out = {
        "verdict": verdict,
        "converged_cold": n_ok_cold,
        "converged_warm": n_ok_warm,
        "nonfinite_residuals": n_nan,
        "meta": {
            "question": "is a full-domain classical solve at 128^2 costly enough for a "
                        "trust-triggered local fallback to be worth building?",
            "cache": args.cache.name,
            "n_cases": len(items),
            "max_outer": args.max_outer,
            "timing": "single wall-clock per case (no repeats); medians over cases",
            "caveat": "cold start freezes nu_eff at the laminar value -- a cost probe, "
                      "not a faithful RANS baseline",
            "caveat_neural": "neural_forward times the bundled CPU demo FNO via "
                             "pretrained(), including per-call geometry encoding. It is "
                             "orders of magnitude slower than the deployed Transolver on "
                             "GPU -- do NOT quote the ratios below as the hybrid's speedup",
            "caveat_cold_residual": "a uniform freestream field has EXACTLY zero residual "
                                    "for this operator (residual-floor theorem, leg i), so "
                                    "the no-harm gate can never accept a cold-start solve: "
                                    "it is comparing against a perfect zero",
            "environment": {
                "platform": platform.platform(),
                "blas_thread_caps": {
                    k: os.environ.get(k)
                    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                },
            },
        },
        "full_solve_cold": {"median_seconds": med_cold, "per_case": cold},
        "full_solve_warm": {"median_seconds": med_warm, "per_case": warm},
        "neural_forward": neural,
        "audit_cost_reference_seconds": 0.0017,
    }
    if neural and neural.get("available"):
        out["ratio_cold_over_neural"] = med_cold / neural["median_seconds"]
        out["ratio_warm_over_neural"] = med_warm / neural["median_seconds"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, "w", encoding="utf8", newline="\n").write(json.dumps(out, indent=2))

    print(f"\n[smoke] median full solve: cold {med_cold:.3f}s   warm {med_warm:.3f}s")
    if neural and neural.get("available"):
        print(f"[smoke] neural forward  : {neural['median_seconds']*1000:.1f} ms  "
              f"-> full/neural = {out['ratio_cold_over_neural']:.0f}x (cold), "
              f"{out['ratio_warm_over_neural']:.0f}x (warm)")
    else:
        print(f"[smoke] neural forward  : unavailable ({neural.get('why')})")
    print(f"[smoke] wrote {args.out}")

    print(f"[smoke] converged (no-harm accepted): "
          f"cold {n_ok_cold}/{len(cold)}, warm {n_ok_warm}/{len(warm)}")
    print(f"[smoke] {verdict}")


if __name__ == "__main__":
    main()
