"""Paper-2 probe: how large a patch can the local classical solver actually handle?

The full-domain solve diverges on every case (``scripts/smoke_fallback_cost.py``), so
the classical arm of the planned coverage sweep only exists over the range of patch
sizes the no-harm gate still accepts. That range is what this measures, and it sets
the right-hand edge of every Pareto curve in the paper.

For each cached deployed-ensemble prediction, cells are ranked by per-cell residual
magnitude -- the trust signal the deployed system actually uses -- and the top
fraction ``c`` is handed to :func:`local_incompressible_solve`. Recorded per (case, c):

  accepted        did the no-harm gate keep the patch (residual reduced)?
  box_fraction    the solver works on the *bounding box* of the selected cells plus a
                  halo, so scattered high-residual cells can blow a small ``c`` up into
                  a nearly full-domain solve. This is the number that decides whether
                  "matched cell budget" is even a meaningful control in the sweep.
  d_rel_l2        change in true velocity rel-L2 error against ground truth
                  (negative = the patch helped)

Writes results/control/patch_acceptance.json.

    python scripts/probe_patch_acceptance.py [--n 20]
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
from neuroforge.physics.residuals import continuity_residual, momentum_residual
from neuroforge.solver.classical import local_incompressible_solve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "cache" / "airfrans_full_test_r128_n200.pkl"
DEFAULT_PRED = ROOT / "data" / "cache" / "w2" / "ensemble"
DEFAULT_OUT = ROOT / "results" / "control" / "patch_acceptance.json"

COVERAGES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
HALO = 2


def _jsonable(o):
    """numpy scalars leak out of the residual/metric paths; json cannot take them."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def residual_magnitude(field: FlowField, fluid) -> np.ndarray:
    """Per-cell ``sqrt(r_cont^2 + r_x^2 + r_y^2)`` -- the trust signal, unnormalised."""
    rc = continuity_residual(field)
    rx, ry = momentum_residual(field, fluid)
    return np.sqrt(rc.astype(np.float64) ** 2 + rx.astype(np.float64) ** 2
                   + ry.astype(np.float64) ** 2)


def rel_l2(pred: FlowField, truth: FlowField, fluid_mask: np.ndarray) -> float:
    """Velocity rel-L2 over fluid cells."""
    du = (pred.u - truth.u)[fluid_mask]
    dv = (pred.v - truth.v)[fluid_mask]
    num = np.sqrt(float(np.sum(du ** 2 + dv ** 2)))
    den = np.sqrt(float(np.sum(truth.u[fluid_mask] ** 2 + truth.v[fluid_mask] ** 2)))
    return num / max(den, 1e-12)


def box_fraction(region: np.ndarray, halo: int) -> float:
    """Share of the grid the solver actually touches: bounding box of ``region`` + halo."""
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return 0.0
    ny, nx = region.shape
    y0, y1 = max(0, ys.min() - halo), min(ny, ys.max() + halo + 1)
    x0, x1 = max(0, xs.min() - halo), min(nx, xs.max() + halo + 1)
    return float((y1 - y0) * (x1 - x0)) / float(ny * nx)


def oracle_bc_field(pred: FlowField, truth: FlowField, region: np.ndarray) -> FlowField:
    """Prediction inside the region, ground truth everywhere else.

    The solver reads its Dirichlet values off the *box border* of the field it is
    handed, so this hands it exact boundary data while leaving the cells it is meant
    to fix untouched. It separates two explanations of a failed patch that the main
    probe cannot tell apart: a local solver too crude to help at all, versus a solver
    poisoned by boundary values taken from an imperfect prediction. If error still
    rises here, the solver is the problem.
    """
    keep = np.asarray(region, dtype=bool)
    def blend(a, b):
        return np.where(keep, a, b)
    return FlowField(
        domain=pred.domain,
        u=blend(pred.u, truth.u), v=blend(pred.v, truth.v),
        p=blend(pred.p, truth.p), nut=blend(pred.nut, truth.nut),
        mask=pred.mask, sdf=pred.sdf,
        meta={"source": "oracle_bc", "case": pred.meta.get("case")},
    )


def load_prediction(pred_dir: Path, name: str, truth: FlowField) -> FlowField | None:
    f = pred_dir / f"{name}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    m = d["mean"]
    return FlowField(domain=truth.domain, u=m[0], v=m[1], p=m[2], nut=m[3],
                     mask=d["mask"], sdf=truth.sdf,
                     meta={"source": "ensemble_mean", "case": name})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-outer", type=int, default=200)
    ap.add_argument("--oracle-bc", action="store_true",
                    help="control: give the solver exact boundary data from ground truth")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    items = pickle.load(open(args.cache, "rb"))
    rows, used, skipped = [], 0, 0

    for case, truth in items:
        if used >= args.n:
            break
        pred = load_prediction(args.pred, case.name, truth)
        if pred is None:
            skipped += 1
            continue
        used += 1

        fluid = np.asarray(pred.mask) > 0.5
        score = residual_magnitude(pred, case.fluid)
        score = np.where(fluid, score, -np.inf)
        order = np.argsort(score, axis=None)[::-1]
        n_fluid = int(fluid.sum())
        err0 = rel_l2(pred, truth, fluid)

        for c in COVERAGES:
            k = max(1, int(round(c * n_fluid)))
            region = np.zeros(fluid.shape, dtype=bool)
            region.flat[order[:k]] = True
            bf = box_fraction(region, HALO)

            src = oracle_bc_field(pred, truth, region) if args.oracle_bc else pred
            base_err = rel_l2(src, truth, fluid)
            t0 = time.perf_counter()
            out = local_incompressible_solve(src, case, region, max_outer=args.max_outer,
                                             halo=HALO)
            dt = time.perf_counter() - t0
            meta = (out.meta or {}).get("fallback", {})
            after = meta.get("residual_after")
            accepted = bool(meta.get("ran", False) and after is not None
                           and np.isfinite(after))
            err1 = rel_l2(out, truth, fluid)

            rows.append({
                "case": case.name, "coverage": c, "cells": k,
                "box_fraction": bf, "seconds": dt, "accepted": accepted,
                "residual_before": meta.get("residual_before"),
                "residual_after": after if after is None or np.isfinite(after) else None,
                "rel_l2_before": base_err, "rel_l2_after": err1, "d_rel_l2": err1 - base_err,
            })
        print(f"  {case.name[:44]:44s} "
              + " ".join("%s%d" % ("+" if r["accepted"] else ".", int(r["coverage"] * 100))
                         for r in rows[-len(COVERAGES):]))

    by_c = {}
    for c in COVERAGES:
        sel = [r for r in rows if r["coverage"] == c]
        acc = [r for r in sel if r["accepted"]]
        by_c[str(c)] = {
            "n": len(sel),
            "accepted": len(acc),
            "acceptance_rate": len(acc) / max(len(sel), 1),
            "median_box_fraction": float(np.median([r["box_fraction"] for r in sel])) if sel else None,
            "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
            "median_d_rel_l2_when_accepted": (
                float(np.median([r["d_rel_l2"] for r in acc])) if acc else None),
            "helped_when_accepted": sum(1 for r in acc if r["d_rel_l2"] < 0),
        }

    out = {
        "meta": {
            "question": "over what patch sizes does the local classical solver survive the "
                        "no-harm gate, and does an accepted patch lower true error?",
            "prediction_source": "cached 5-member deep-ensemble mean (data/cache/w2/ensemble)",
            "selection": "top-c fluid cells by per-cell residual magnitude (the trust signal)",
            "oracle_bc": args.oracle_bc,
            "halo": HALO, "max_outer": args.max_outer,
            "n_cases": used, "cases_without_cached_prediction": skipped,
            "note": "box_fraction is the share of the grid actually solved: the solver takes "
                    "the bounding box of the selected cells plus a halo, so scattered "
                    "selections cost far more than their cell count suggests",
            "environment": {
                "platform": platform.platform(),
                "blas_thread_caps": {k: os.environ.get(k) for k in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                      "MKL_NUM_THREADS")},
            },
        },
        "by_coverage": by_c,
        "per_case": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, "w", encoding="utf8", newline="\n").write(
        json.dumps(out, indent=2, default=_jsonable))

    print(f"\n  {'c':>6} {'accepted':>10} {'box frac':>10} {'sec':>8} {'d_relL2':>10}")
    for c in COVERAGES:
        s = by_c[str(c)]
        d = s["median_d_rel_l2_when_accepted"]
        print(f"  {c:6.3f} {s['accepted']:4d}/{s['n']:<5d} {s['median_box_fraction']:10.3f} "
              f"{s['median_seconds']:8.2f} {('%+.4f' % d) if d is not None else '       --':>10}")
    print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
