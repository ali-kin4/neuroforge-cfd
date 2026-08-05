"""Wall-clock cost of the audit itself — what does the certificate cost?

Question answered
-----------------
The paper claims the audit (residual maps + trust map + case-level trust score
+ ensemble sigma + conformal band) turns a bare surrogate prediction into an
accept/reject decision with a calibrated error band. A deployment reviewer's
first question is: *what does that audit cost per case?* This script measures
it, component by component, on the same 200 AirfRANS 'full' test cases at
resolution 128 the paper's headline numbers use.

Components timed per case (median over --repeats timed repeats, after one
untimed warm-up evaluation):

  encode      encode_case(case)             geometry -> SDF/mask input planes
              (prediction-side prep, reported for context; NOT audit cost)
  diagnose    PhysicsChecker(Config().physics).diagnose(pred, case)
              full audit: continuity + momentum + BC residual maps,
              non-dimensionalisation, trust map, scalar summary
  score       Diagnostics.residual_norm()   the case-level trust scalar behind
              every Spearman/AUROC headline in the paper
  sigma       fluid-masked mean of the cached 5-member ensemble per-cell std
              (velocity channels) -- the UQ score fused with the residual
  band        per-cell conformal interval mean +/- q*sigma for u, v, p
              (q = published backbone split-conformal quantiles)

Amortised over the whole 200-case fleet (not per-case): the rank-average
fusion of residual and sigma scores (two rankdata calls).

The audit total reported is  diagnose + score + sigma + band  -- everything
that happens AFTER the surrogate emits a field. The comparison anchor is a
classical steady-RANS solve of the same case (--classical-solve-sec, pass a
verified per-case figure with its source; nothing is invented here).

Caveat recorded in the output: timings taken while other compute runs on the
box are contaminated by contention. The script records observed load context
(process count is not measured; pass --context "..." to describe it) and the
recommendation is to re-run on an idle machine for the paper number.

Run (CPU-only, all fields served from cache, ~2-4 min):
    .venv/Scripts/python.exe scripts/measure_audit_cost.py
    .venv/Scripts/python.exe scripts/measure_audit_cost.py ^
        --classical-solve-sec 1500 --classical-source "AirfRANS ..." ^
        --context "idle machine"

Output: results/control/audit_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets BLAS thread caps)
import numpy as np
from scipy.stats import rankdata

from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.geometry.encode import encode_case
from neuroforge.physics.residuals import PhysicsChecker

# Published backbone split-conformal quantiles (results/uq_ensemble, split
# seed 0; the multi-split study varies these by ~3% -- immaterial for timing).
CONFORMAL_Q = {"u": 2.325, "v": 2.551, "p": 2.261}


def log(msg: str) -> None:
    print(f"[audit-cost] {msg}", flush=True)


def time_component(fn, repeats: int) -> float:
    """Median wall-clock seconds of ``fn()`` over ``repeats`` timed runs."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples))


def summarise(values: list[float]) -> dict:
    """Fleet-level stats (ms) for a list of per-case medians (seconds)."""
    a = np.asarray(values, np.float64) * 1e3  # -> ms
    return {
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
        "n_cases": int(a.size),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Measure per-case audit wall-clock cost.")
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--w2-cache-dir", default="data/cache/w2")
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5,
                   help="timed repeats per component per case (median taken)")
    p.add_argument("--classical-solve-sec", type=float, default=None,
                   help="verified per-case classical steady-RANS solve time, "
                        "seconds; ratios reported only if provided")
    p.add_argument("--classical-source", default=None,
                   help="citation/source string for --classical-solve-sec")
    p.add_argument("--context", default=None,
                   help="free-text description of machine load during timing")
    p.add_argument("--out", default="results/control/audit_cost.json")
    args = p.parse_args(argv)

    t_start = time.time()
    checker = PhysicsChecker(Config().physics)

    log(f"loading AirfRANS GT pairs (full/test, res {args.resolution}, "
        f"limit {args.n_val}) ...")
    pairs = load_airfrans(
        root=args.root, task="full", train=False, resolution=args.resolution,
        limit=args.n_val, cache_dir=args.cache_dir, download=False, progress=False,
    )
    ens_dir = os.path.join(args.w2_cache_dir, "ensemble")

    cols: dict[str, list[float]] = {
        "encode": [], "diagnose": [], "score": [], "sigma": [], "band": [],
    }
    n_missing = 0
    t_loop = time.time()
    for i, (case, _gt) in enumerate(pairs):
        npz_path = os.path.join(ens_dir, f"{case.name}.npz")
        if not os.path.exists(npz_path):
            n_missing += 1
            continue
        d = np.load(npz_path)
        mean, std = d["mean"], d["std"]

        # --- prediction-side prep (context only, excluded from audit total) ---
        # Timed once, not --repeats times: at ~1s/case it would dominate the
        # measurement loop while contributing nothing to the audit total.
        stack = encode_case(case)  # warm-up (geometry caches, allocator)
        t_encode = time_component(lambda: encode_case(case), 1)
        sdf = stack[0].astype(DTYPE)
        mask_geo = stack[1].astype(DTYPE)
        pred = FlowField.from_array(mean, case.domain, mask=mask_geo, sdf=sdf,
                                    meta={"source": "ensemble-mean", "case": case.name})
        fluid = np.asarray(mask_geo) > 0.5

        # --- audit components (one untimed warm-up each, then timed median) ---
        diag = checker.diagnose(pred, case)  # warm-up
        t_diagnose = time_component(lambda: checker.diagnose(pred, case), args.repeats)
        diag.residual_norm()
        t_score = time_component(lambda: diag.residual_norm(), args.repeats)

        def sigma_score():
            return float((0.5 * (std[0] + std[1]))[fluid].mean())

        sigma_score()
        t_sigma = time_component(sigma_score, args.repeats)

        def conformal_band():
            # Per-cell 90% band, mean +/- q*sigma, channels u, v, p.
            lo_u, hi_u = mean[0] - CONFORMAL_Q["u"] * std[0], mean[0] + CONFORMAL_Q["u"] * std[0]
            lo_v, hi_v = mean[1] - CONFORMAL_Q["v"] * std[1], mean[1] + CONFORMAL_Q["v"] * std[1]
            lo_p, hi_p = mean[2] - CONFORMAL_Q["p"] * std[2], mean[2] + CONFORMAL_Q["p"] * std[2]
            return lo_u, hi_u, lo_v, hi_v, lo_p, hi_p

        conformal_band()
        t_band = time_component(conformal_band, args.repeats)

        cols["encode"].append(t_encode)
        cols["diagnose"].append(t_diagnose)
        cols["score"].append(t_score)
        cols["sigma"].append(t_sigma)
        cols["band"].append(t_band)

        if (i + 1) % 50 == 0:
            log(f"  timed {i + 1}/{len(pairs)} cases "
                f"({(time.time() - t_loop) / (i + 1):.2f}s/case)")

    n_cases = len(cols["diagnose"])
    if n_cases == 0:
        log("no cached cases found -- nothing to time")
        return 1

    # --- fleet-amortised fusion (two rankdata calls over n_cases scores) ---
    res_scores = np.random.default_rng(0).random(n_cases)  # timing-only stand-in
    sig_scores = np.random.default_rng(1).random(n_cases)

    def fuse():
        return 0.5 * (rankdata(res_scores) + rankdata(sig_scores))

    fuse()
    t_fuse_total = time_component(fuse, max(args.repeats, 20))

    per_component = {k: summarise(v) for k, v in cols.items()}
    audit_total = [
        cols["diagnose"][j] + cols["score"][j] + cols["sigma"][j] + cols["band"][j]
        for j in range(n_cases)
    ]
    total_stats = summarise(audit_total)
    total_stats["includes"] = "diagnose + score + sigma + band (per case); "
    total_stats["includes"] += (
        f"fusion adds {t_fuse_total * 1e3 / n_cases:.4f} ms/case amortised"
    )

    comparison = None
    if args.classical_solve_sec is not None:
        ratio_median = (total_stats["median_ms"] / 1e3) / args.classical_solve_sec
        comparison = {
            "classical_solve_sec_per_case": args.classical_solve_sec,
            "classical_source": args.classical_source,
            "audit_over_classical_median": ratio_median,
            "classical_over_audit_median": 1.0 / ratio_median,
        }

    out = {
        "meta": {
            "experiment": "wall-clock cost of the audit (per-case, component-wise)",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - t_start,
            "repeats_per_component": args.repeats,
            "timing": "median of repeats per case after one untimed warm-up; "
                      "fleet stats over per-case medians",
            "grid": f"{args.resolution}x{args.resolution}",
            "prediction_source": os.path.join(args.w2_cache_dir, "ensemble")
                                 + " (cached 5-member ensemble mean/std; zero "
                                   "forward passes -- audit cost only)",
            "conformal_q": CONFORMAL_Q,
            "environment": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "numpy": np.__version__,
                "blas_thread_caps": {
                    k: os.environ.get(k)
                    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                              "MKL_NUM_THREADS")
                },
            },
            "load_context": args.context or "not recorded -- re-run with "
                                            "--context on an idle machine for "
                                            "the paper number",
            "n_missing": n_missing,
        },
        "per_component_ms": per_component,
        "audit_total_ms": total_stats,
        "fusion_fleet_ms": {
            "total_ms_for_n_cases": t_fuse_total * 1e3,
            "per_case_amortised_ms": t_fuse_total * 1e3 / n_cases,
            "n_cases": n_cases,
        },
        "classical_comparison": comparison,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")

    log("================= SUMMARY =================")
    for k in ("encode", "diagnose", "score", "sigma", "band"):
        s = per_component[k]
        log(f"  {k:>8s}: median {s['median_ms']:8.3f} ms  "
            f"(mean {s['mean_ms']:.3f}, p95 {s['p95_ms']:.3f})")
    log(f"  AUDIT TOTAL (diagnose+score+sigma+band): "
        f"median {total_stats['median_ms']:.3f} ms/case")
    if comparison:
        log(f"  classical solve {args.classical_solve_sec:.0f}s/case -> audit is "
            f"1/{comparison['classical_over_audit_median']:.0f} of a solve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
