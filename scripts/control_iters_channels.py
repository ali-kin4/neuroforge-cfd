"""CONTROL 6 -- recover the channels tab:iters omits.

The objection
-------------
``tab:iters`` reports only ``mse_u`` and surface ``mse_p`` across the correction
sweep, while ``tab:indist`` documents this corrector family inflating ``mse_v``
by +129% and volume ``mse_p`` by +57% on the same backbone.  If the omitted
channels rise along the sweep, the table and the figure built on it invert.

Why this is recoverable (and was not "lost")
--------------------------------------------
``run_sensitivity.py`` calls ``per_channel_mse(field, ref)``, which COMPUTES
``mse_u, mse_v, mse_p, mse_speed, mse_nut`` at every iteration setting, and then
keeps only ``pm["mse_u"]`` (line ~184) before aggregating.  The other channels
were computed and dropped on write, not never computed.  The sweep's checkpoint
(``checkpoints/certificates_deq.pt``) IS retained, so the full channel set is
recoverable by re-running the same sweep and keeping the whole dict.  This
script does exactly that and changes nothing else: same checkpoint, same
``ITER_GRID``, same ``n_eval``, same resolution, same metric functions.

A reproduction gate is built in: if the recovered ``mse_u`` and
``surface_mse_p`` do not match the committed ``results/sensitivity/iters.json``
within tolerance, the run reports the mismatch rather than quietly publishing a
different sweep.

Usage
-----
    PYTHONPATH=src .venv/Scripts/python.exe scripts/control_iters_channels.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/control_iters_channels.py --n-eval 8   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets BLAS thread caps)
import numpy as np

from neuroforge.core.config import Config
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.physics.evaluation import per_channel_mse, surface_pressure_mse
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine

CKPT = "checkpoints/certificates_deq.pt"
REF = "results/sensitivity/iters.json"
CHANNELS = ("mse_u", "mse_v", "mse_p", "mse_speed", "mse_nut")


def log(m):
    print(f"[iters_channels] {m}", flush=True)


def _nan_mean(xs):
    a = np.asarray(xs, np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _spearman(a, b):
    from neuroforge.physics.evaluation import _spearman as sp
    return sp(a, b)


def eval_at_iters(engine, pairs, n_iters, checker):
    """Same body as run_sensitivity.eval_at_iters, but keeps ALL channels."""
    corrector = engine.corrector
    saved = getattr(corrector, "max_iter", None) if corrector is not None else None
    acc = {c: [] for c in CHANNELS}
    surf, res_norms, errs, deq_iters = [], [], [], []
    try:
        for case, ref in pairs:
            if n_iters > 0 and corrector is not None:
                corrector.max_iter = int(n_iters)
            if n_iters == 0:
                field = engine.predictor.predict(case)
                rn = float(checker.diagnose(field, case).residual_norm())
            else:
                r = engine.solve(case)
                field = r.field
                rn = float(r.metrics.get("residual_norm", float("nan")))
                if r.history and "deq_iters" in r.history[-1]:
                    deq_iters.append(float(r.history[-1]["deq_iters"]))
            pm = per_channel_mse(field, ref)
            for c in CHANNELS:
                acc[c].append(pm.get(c, float("nan")))
            surf.append(surface_pressure_mse(field, ref, case))
            res_norms.append(rn)
            m = (np.asarray(ref.mask) > 0.5) if ref.mask is not None else None
            ps, rs = field.speed(), ref.speed()
            if m is not None:
                num = float(np.sqrt(np.sum(((ps - rs)[m]) ** 2)))
                den = float(np.sqrt(np.sum((rs[m]) ** 2))) + 1e-12
            else:
                num = float(np.sqrt(np.sum((ps - rs) ** 2)))
                den = float(np.sqrt(np.sum(rs ** 2))) + 1e-12
            errs.append(num / den)
    finally:
        if saved is not None:
            corrector.max_iter = saved

    row = {"n_iters": n_iters}
    for c in CHANNELS:
        row[c] = _nan_mean(acc[c])
    row["surface_mse_p"] = _nan_mean(surf)
    row["residual_norm"] = _nan_mean(res_norms)
    row["residual_error_spearman"] = _spearman(res_norms, errs)
    row["deq_iters_mean"] = _nan_mean(deq_iters) if deq_iters else 0.0
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iter-grid", type=int, nargs="+", default=[0, 1, 3, 5, 10, 15])
    p.add_argument("--n-eval", type=int, default=80)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--task", default="full")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tol", type=float, default=0.02,
                   help="relative tolerance for the mse_u / surface_mse_p reproduction gate")
    p.add_argument("--out", default="results/review/control6_iters_full_channels.json")
    a = p.parse_args(argv)

    t0 = time.time()
    cfg = Config()
    cfg.train.device = a.device
    engine = NeuroForgeEngine.from_checkpoint(CKPT, config=cfg)
    checker = PhysicsChecker(cfg.physics)
    log(f"engine on {engine.predictor.device}, corrector={type(engine.corrector).__name__}")

    pairs = load_airfrans(root="data", task=a.task, train=False, resolution=a.resolution,
                          limit=a.n_eval, cache_dir="data/cache", download=False,
                          progress=False)
    log(f"{len(pairs)} test pairs")

    rows = []
    for k in a.iter_grid:
        tk = time.time()
        r = eval_at_iters(engine, pairs, k, checker)
        rows.append(r)
        log("iters=%2d: " % k + " ".join(f"{c}={r[c]:.4g}" for c in CHANNELS)
            + f" surf_p={r['surface_mse_p']:.4g} res={r['residual_norm']:.4g}"
            + f" rho={r['residual_error_spearman']:.3f} ({time.time() - tk:.0f}s)")

    # reproduction gate against the committed sweep
    gate = {"ref": REF, "checked": [], "pass": None}
    if os.path.exists(REF):
        ref = {r["n_iters"]: r for r in json.load(open(REF))["rows"]}
        ok = True
        for r in rows:
            rr = ref.get(r["n_iters"])
            if not rr:
                continue
            for k in ("mse_u", "surface_mse_p", "residual_norm"):
                rel = abs(r[k] - rr[k]) / max(abs(rr[k]), 1e-30)
                gate["checked"].append({"n_iters": r["n_iters"], "metric": k,
                                        "recovered": r[k], "committed": rr[k],
                                        "rel_diff": rel, "within_tol": bool(rel <= a.tol)})
                ok = ok and rel <= a.tol
        gate["pass"] = bool(ok)
        gate["note"] = ("committed sweep used n_eval=80; a smaller --n-eval will fail this "
                        "gate legitimately (different case subset), which is why the gate "
                        "reports rather than aborts.")

    # direction of travel per channel, from the mse_u minimum onward
    iu = int(np.argmin([r["mse_u"] for r in rows]))
    direction = {}
    for c in list(CHANNELS) + ["surface_mse_p", "residual_norm"]:
        v = [r[c] for r in rows]
        if not np.isfinite(v).all():
            direction[c] = {"available": False}
            continue
        direction[c] = {
            "available": True,
            "values": v,
            "at_iter0": v[0],
            "min_value": float(np.nanmin(v)),
            "argmin_iter": rows[int(np.nanargmin(v))]["n_iters"],
            "at_last_iter": v[-1],
            "pct_change_0_to_last": float(100 * (v[-1] - v[0]) / abs(v[0])) if v[0] else None,
            "pct_change_min_to_last": float(100 * (v[-1] - np.nanmin(v)) / abs(np.nanmin(v)))
            if np.nanmin(v) else None,
            "rises_after_mse_u_minimum": bool(v[-1] > v[iu]),
            "monotone_decreasing": bool(all(v[i + 1] <= v[i] for i in range(len(v) - 1))),
            "monotone_increasing": bool(all(v[i + 1] >= v[i] for i in range(len(v) - 1))),
        }

    out = {
        "artifact": "control6_iters_full_channels",
        "question": ("Do the channels tab:iters omits (mse_v, volume mse_p, mse_nut) rise "
                     "along the correction sweep? tab:indist documents this corrector "
                     "family inflating mse_v +129% and mse_p +57%."),
        "provenance": (
            "run_sensitivity.py already CALLS per_channel_mse, which computes mse_u, "
            "mse_v, mse_p, mse_speed and mse_nut, then keeps only pm['mse_u'] before "
            "aggregating (scripts/run_sensitivity.py ~line 184). The other channels were "
            "computed and dropped on write. This script re-runs the identical sweep on "
            "the identical retained checkpoint and keeps the whole dict."),
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_s": time.time() - t0,
            "ckpt": CKPT, "device": str(engine.predictor.device),
            "n_eval": len(pairs), "task": a.task, "resolution": a.resolution,
            "iter_grid": a.iter_grid,
            "swept_knob": "DEQCorrector.max_iter (internal fixed-point cap)",
        },
        "reproduction_gate": gate,
        "rows": rows,
        "direction_of_travel": direction,
        "mse_u_argmin_iter": rows[iu]["n_iters"],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {a.out} ({time.time() - t0:.0f}s); gate pass={gate['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
