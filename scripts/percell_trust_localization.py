"""Per-cell trust localization study: can the physics-residual map say WHERE
a prediction is wrong, once properly post-processed?

Question answered
-----------------
The published control (results/control/percell_residual_error.json, OLD FNO
checkpoint, n=15) found per-cell Spearman(|residual|, |speed error|) of
0.22 +/- 0.06 -- weak. This harness re-asks the question on the DEPLOYED-quality
fields (the cached 5-member Transolver ensemble mean, 200 AirfRANS 'full' test
cases at resolution 128) and evaluates post-processing variants that could
legitimately improve localization. Pre-committed to reporting whichever way it
lands: upgrade OR scoped honest negative.

Data (all cached; CPU-only, zero model forwards)
------------------------------------------------
* prediction: data/cache/w2/ensemble/<case>.npz  ('mean' (4,H,W)); FlowField
  built exactly as run_w2_conformal_corrected.MeanPredictor.predict --
  FlowField.from_array(mean, case.domain, mask=encode_case(case)[1],
  sdf=encode_case(case)[0]).
* ground truth: load_airfrans(root='data', task='full', train=False,
  resolution=128, limit=200, cache_dir='data/cache').
* residual map: PhysicsChecker(Config().physics).diagnose(pred, case) ->
  r = sqrt(continuity^2 + momentum_x^2 + momentum_y^2) per cell (solid and
  wall-ring already zeroed by diagnose, residuals.py ~326-348).
* error map: e = |pred.speed() - gt.speed()|; fluid mask m = gt.mask > 0.5.

Variants (per case, fluid cells only; aggregated over all 200 cases)
--------------------------------------------------------------------
V0  baseline        raw r vs raw e (published-style number, now at n=200).
V1  smoothing       mask-aware Gaussian smoothing of BOTH r and e,
                    sigma in {1,2,4} cells: smooth(x*m)/smooth(m) so solid
                    zeros do not bleed into the fluid.
V2  patch pooling   non-overlapping k x k patches, k in {4,8,16}; patch score
                    = mean r over fluid cells in patch, patch error = mean e;
                    patches with <25% fluid cells dropped. Extra metric:
                    per-case Mann-Whitney AUROC for detecting top-decile-error
                    patches (labels: patch_e >= 90th pct within the case).
V3  dyn-p norm      r / (0.5*speed_pred^2 + eps) vs e, eps = 5% of the case's
                    fluid-mean dynamic pressure (stagnation-blowup guard).
V4  rank transform  per-case rank-normalize r and e, then Pearson -- must
                    reproduce V0's Spearman exactly (sanity identity).

Metrics per variant: (a) mean +/- std over cases of per-case Spearman;
(b) pooled Pearson (all fluid cells / patches of all cases, running-sum
accumulator); (c) V2 only: mean +/- std over cases of patch AUROC.

Sanity gate: V0 per-case Spearman mean in [0.1, 0.4] (the published 0.22 was a
different model/backbone -- a different value inside the band is fine; the
exact value is reported either way).

Run (CPU, single process, ~10 min):
    .venv/Scripts/python.exe scripts/percell_trust_localization.py

Outputs: results/control/percell_localization.json
         results/control/percell_localization.md
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, rankdata, spearmanr

import neuroforge  # noqa: F401  -- MUST precede numpy/torch heavy work (BLAS caps)
from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.geometry.encode import encode_case
from neuroforge.physics.residuals import PhysicsChecker

SIGMAS = [1, 2, 4]
PATCH_KS = [4, 8, 16]
MIN_FLUID_FRAC = 0.25          # V2: drop patches with <25% fluid cells
TOP_ERROR_QUANTILE = 0.9       # V2 AUROC positives: top-decile-error patches
EPS_DYNP_FRAC = 0.05           # V3: eps = 5% of case fluid-mean dynamic pressure
GATE_BAND = (0.1, 0.4)         # plausibility band for the V0 per-case Spearman
PUBLISHED = {"per_cell_spearman_mean": 0.2196, "per_cell_spearman_std": 0.0602,
             "n_cases": 15, "checkpoint": "OLD FNO (checkpoints/certificates_deq.pt)"}


def log(msg: str) -> None:
    print(f"[percell] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def auroc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (tie-aware midranks).

    Identical helper to scripts/run_selective_prediction.py.
    """
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    u = float(r[y].sum()) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


class PooledPearson:
    """Streaming Pearson over all cells/patches of all cases (no big concat)."""

    def __init__(self) -> None:
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.n += x.size
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())
        self.sxy += float((x * y).sum())

    def value(self) -> float:
        if self.n < 2:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx ** 2 / self.n
        vy = self.syy - self.sy ** 2 / self.n
        if vx <= 0 or vy <= 0:
            return float("nan")
        return float(cov / np.sqrt(vx * vy))


# --------------------------------------------------------------------------- #
# Post-processing primitives
# --------------------------------------------------------------------------- #
def masked_smooth(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Mask-aware Gaussian smoothing: smooth(x*m)/smooth(m), no solid bleed."""
    num = gaussian_filter(np.asarray(x, np.float64) * m, sigma=sigma)
    den = gaussian_filter(np.asarray(m, np.float64), sigma=sigma)
    out = np.zeros_like(num)
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out


def patch_pool(r: np.ndarray, e: np.ndarray, m: np.ndarray, k: int):
    """Non-overlapping k x k patches; fluid-mean score/error; drop low-fluid.

    Returns (patch_r, patch_e) 1-D arrays over kept patches (fluid frac >= 25%).
    Trailing rows/cols not divisible by k are cropped (128 divides evenly for
    all k in {4,8,16}, so nothing is cropped in practice).
    """
    H, W = r.shape
    Hk, Wk = (H // k) * k, (W // k) * k
    def blocks(a):
        return a[:Hk, :Wk].reshape(Hk // k, k, Wk // k, k).swapaxes(1, 2)
    mb = blocks(m.astype(np.float64))
    fluid_count = mb.sum(axis=(2, 3))
    frac = fluid_count / (k * k)
    keep = frac >= MIN_FLUID_FRAC
    if not keep.any():
        return np.empty(0), np.empty(0)
    rb = blocks(np.asarray(r, np.float64) * m)
    eb = blocks(np.asarray(e, np.float64) * m)
    pr = rb.sum(axis=(2, 3))[keep] / fluid_count[keep]
    pe = eb.sum(axis=(2, 3))[keep] / fluid_count[keep]
    return pr, pe


# --------------------------------------------------------------------------- #
# Main pass
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Per-cell trust localization study.")
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--ens-dir", default=os.path.join("data", "cache", "w2", "ensemble"))
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--out-dir", default=os.path.join("results", "control"))
    args = p.parse_args(argv)

    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    checker = PhysicsChecker(Config().physics)

    log(f"loading AirfRANS GT pairs (full/test, res {args.resolution}, "
        f"limit {args.n_val}) ...")
    pairs = load_airfrans(
        root=args.root, task="full", train=False, resolution=args.resolution,
        limit=args.n_val, cache_dir=args.cache_dir, download=False, progress=False,
    )

    # Variant registry: name -> per-case spearman list; pooled Pearson accum.
    variant_names = (["V0_raw"]
                     + [f"V1_smooth_s{s}" for s in SIGMAS]
                     + [f"V2_patch_k{k}" for k in PATCH_KS]
                     + ["V3_dynp_norm", "V4_rank"])
    percase_rho = {v: [] for v in variant_names}
    pooled = {v: PooledPearson() for v in variant_names}
    patch_auroc = {f"V2_patch_k{k}": [] for k in PATCH_KS}
    patch_counts = {f"V2_patch_k{k}": [] for k in PATCH_KS}
    v4_max_dev = 0.0            # max |V4 pearson-of-ranks - V0 spearman| identity
    zero_r_fracs = []           # fraction of fluid cells with exactly-zero r
    names = []
    n_missing = 0
    t_diag = 0.0

    for i, (case, gt) in enumerate(pairs):
        npz_path = os.path.join(args.ens_dir, f"{case.name}.npz")
        if not os.path.exists(npz_path):
            n_missing += 1
            continue
        mean = np.load(npz_path)["mean"]
        stack = encode_case(case)
        pred = FlowField.from_array(
            mean, case.domain, mask=stack[1].astype(DTYPE),
            sdf=stack[0].astype(DTYPE),
            meta={"source": "ensemble-mean", "case": case.name})

        t0 = time.time()
        diag = checker.diagnose(pred, case)
        t_diag += time.time() - t0
        r = np.sqrt(diag.continuity ** 2 + diag.momentum_x ** 2
                    + diag.momentum_y ** 2).astype(np.float64)
        e = np.abs(pred.speed().astype(np.float64) - gt.speed().astype(np.float64))
        m = np.asarray(gt.mask) > 0.5
        mf = m.astype(np.float64)
        names.append(case.name)
        zero_r_fracs.append(float((r[m] == 0.0).mean()))

        # ---- V0 raw ----
        rho0 = safe_spearman(r[m], e[m])
        percase_rho["V0_raw"].append(rho0)
        pooled["V0_raw"].update(r[m], e[m])

        # ---- V1 mask-aware smoothing ----
        for s in SIGMAS:
            rs = masked_smooth(r, mf, s)
            es = masked_smooth(e, mf, s)
            v = f"V1_smooth_s{s}"
            percase_rho[v].append(safe_spearman(rs[m], es[m]))
            pooled[v].update(rs[m], es[m])

        # ---- V2 patch pooling (+ top-decile-error patch AUROC) ----
        for k in PATCH_KS:
            pr, pe = patch_pool(r, e, mf, k)
            v = f"V2_patch_k{k}"
            patch_counts[v].append(int(pr.size))
            if pr.size >= 3:
                percase_rho[v].append(safe_spearman(pr, pe))
                pooled[v].update(pr, pe)
                thr = np.quantile(pe, TOP_ERROR_QUANTILE)
                patch_auroc[v].append(auroc_mann_whitney(pr, pe >= thr))
            else:
                percase_rho[v].append(float("nan"))
                patch_auroc[v].append(float("nan"))

        # ---- V3 dynamic-pressure normalization ----
        q = 0.5 * pred.speed().astype(np.float64) ** 2
        eps = EPS_DYNP_FRAC * float(q[m].mean())
        rn = r / (q + eps)
        percase_rho["V3_dynp_norm"].append(safe_spearman(rn[m], e[m]))
        pooled["V3_dynp_norm"].update(rn[m], e[m])

        # ---- V4 rank transform (sanity identity with V0) ----
        rr = rankdata(r[m])
        er = rankdata(e[m])
        rho4 = float(pearsonr(rr, er).statistic)
        percase_rho["V4_rank"].append(rho4)
        pooled["V4_rank"].update(rr, er)
        if np.isfinite(rho0) and np.isfinite(rho4):
            v4_max_dev = max(v4_max_dev, abs(rho4 - rho0))

        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(pairs)} cases "
                f"({(time.time() - t_start) / (i + 1):.2f}s/case, "
                f"diagnose {t_diag / (i + 1):.2f}s/case)")

    n_cases = len(names)
    log(f"pass done: {n_cases} cases, {n_missing} missing npz, "
        f"{time.time() - t_start:.1f}s so far")

    # ---- aggregate ----
    def agg(v):
        a = np.asarray(percase_rho[v], np.float64)
        ok = np.isfinite(a)
        out = {
            "percase_spearman_mean": float(np.nanmean(a)) if ok.any() else None,
            "percase_spearman_std": float(np.nanstd(a)) if ok.any() else None,
            "pooled_pearson": pooled[v].value(),
            "n_cases_valid": int(ok.sum()),
        }
        if v in patch_auroc:
            au = np.asarray(patch_auroc[v], np.float64)
            out["patch_auroc_top_decile_mean"] = float(np.nanmean(au))
            out["patch_auroc_top_decile_std"] = float(np.nanstd(au))
            pc = np.asarray(patch_counts[v], np.float64)
            out["patches_per_case_mean"] = float(pc.mean())
        # paired improvement over V0 (same cases)
        if v != "V0_raw":
            base = np.asarray(percase_rho["V0_raw"], np.float64)
            both = np.isfinite(a) & np.isfinite(base)
            if both.any():
                d = a[both] - base[both]
                out["delta_vs_v0_mean"] = float(d.mean())
                out["delta_vs_v0_std"] = float(d.std())
                out["frac_cases_improved_vs_v0"] = float((d > 0).mean())
        return out

    variants = {v: agg(v) for v in variant_names}

    v0_mean = variants["V0_raw"]["percase_spearman_mean"]
    gate = {
        "v0_percase_spearman_mean": v0_mean,
        "plausible_band": list(GATE_BAND),
        "passed": bool(GATE_BAND[0] <= v0_mean <= GATE_BAND[1]),
        "published_reference": PUBLISHED,
        "note": "published 0.22 was an OLD FNO checkpoint at n=15; this run is "
                "the deployed 5-member Transolver ensemble mean at n=200 -- a "
                "different value is expected and fine.",
        "v4_identity_max_abs_dev_from_v0": v4_max_dev,
    }

    # ---- best variant (per-case Spearman at its own scale) ----
    best = max(variant_names,
               key=lambda v: (variants[v]["percase_spearman_mean"]
                              if variants[v]["percase_spearman_mean"] is not None
                              else -np.inf))

    runtime_sec = time.time() - t_start
    results = {
        "meta": {
            "experiment": "per-cell trust localization: residual map vs |speed "
                          "error| map, post-processing variants",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": runtime_sec,
            "diagnose_sec_per_case": t_diag / max(n_cases, 1),
            "device": "cpu (predictions served from cache; zero model forwards)",
            "n_cases": n_cases,
            "n_missing": n_missing,
            "sources": {
                "prediction": args.ens_dir + " (5-member Transolver ensemble "
                              "mean, deployed quality)",
                "gt": f"AirfRANS full/test, res {args.resolution}, "
                      f"limit {args.n_val}",
                "published_control": "results/control/percell_residual_error"
                                     ".json (OLD FNO, n=15)",
            },
            "formulas": {
                "residual_map": "diag = PhysicsChecker(Config().physics)."
                                "diagnose(pred, case); r = sqrt(continuity^2 + "
                                "momentum_x^2 + momentum_y^2). diagnose zeroes "
                                "r inside the solid AND on the wall-ring "
                                "(fluid cells adjacent to the solid, "
                                "residuals.py ~326-348).",
                "error_map": "e = |pred.speed() - gt.speed()|; fluid mask "
                             "m = gt.mask > 0.5 (same mask as evaluate_cases).",
                "flowfield_construction": "FlowField.from_array(mean, "
                                          "case.domain, mask=encode_case(case)"
                                          "[1], sdf=encode_case(case)[0]) -- "
                                          "mirrors MeanPredictor.predict.",
                "V1_smoothing": "mask-aware: gaussian_filter(x*m,sigma)/"
                                "gaussian_filter(m,sigma), evaluated on fluid "
                                "cells; sigma in " + str(SIGMAS) + " cells; "
                                "BOTH r and e smoothed.",
                "V2_patch": f"non-overlapping k x k patches, k in {PATCH_KS}; "
                            "patch score/error = mean over fluid cells in "
                            f"patch; patches with fluid frac < {MIN_FLUID_FRAC}"
                            " dropped; Spearman over patches per case; AUROC "
                            "(Mann-Whitney, tie-aware midranks, same helper as "
                            "run_selective_prediction.py) for patches with "
                            f"patch_e >= {TOP_ERROR_QUANTILE:.0%} quantile "
                            "within the case, averaged over cases.",
                "V3_dynp": "r / (0.5*speed_pred^2 + eps), eps = "
                           f"{EPS_DYNP_FRAC} * fluid-mean(0.5*speed_pred^2) "
                           "per case (stagnation guard); speed_pred (not GT) "
                           "so the score stays deployment-computable.",
                "V4_rank": "Pearson of per-case rankdata(r), rankdata(e) -- "
                           "mathematically identical to V0 Spearman; used as "
                           "an implementation sanity identity.",
                "pooled_pearson": "streaming Pearson over all fluid cells "
                                  "(V2: all kept patches) of all cases.",
            },
            "zero_residual_fluid_frac_mean": float(np.mean(zero_r_fracs)),
            "zero_residual_note": "fraction of fluid (gt.mask) cells whose r "
                                  "is exactly 0 (wall-ring zeroing + geometry-"
                                  "mask mismatch); these enter the correlation "
                                  "as ties, as in the published control.",
        },
        "sanity_gate": gate,
        "variants": variants,
        "best_variant": best,
        "percase": {
            "names": names,
            "spearman": {v: [float(x) for x in percase_rho[v]]
                         for v in variant_names},
            "patch_auroc": {v: [float(x) for x in patch_auroc[v]]
                            for v in patch_auroc},
        },
    }

    out_json = os.path.join(args.out_dir, "percell_localization.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {out_json}")

    # ---- markdown summary ----
    lines = [
        "# Per-cell trust localization (deployed ensemble mean, n=200)",
        "",
        f"Date: {results['meta']['date']}  |  runtime {runtime_sec:.0f}s (CPU, "
        "cached predictions)",
        "",
        "Can the per-cell physics-residual map localize WHERE the prediction "
        "is wrong? Published control (OLD FNO, n=15): per-cell Spearman "
        f"{PUBLISHED['per_cell_spearman_mean']:.2f} +/- "
        f"{PUBLISHED['per_cell_spearman_std']:.2f}.",
        "",
        "| Variant | Scale | per-case Spearman (mean +/- std) | pooled Pearson"
        " | delta vs V0 | % cases improved | patch AUROC (top-decile err) |",
        "|---|---|---|---|---|---|---|",
    ]
    scale_of = lambda v: ("patch" if v.startswith("V2") else "cell")
    for v in variant_names:
        d = variants[v]
        delta = (f"{d['delta_vs_v0_mean']:+.3f}" if "delta_vs_v0_mean" in d
                 else "--")
        frac = (f"{100 * d['frac_cases_improved_vs_v0']:.0f}%"
                if "frac_cases_improved_vs_v0" in d else "--")
        au = (f"{d['patch_auroc_top_decile_mean']:.3f} +/- "
              f"{d['patch_auroc_top_decile_std']:.3f}"
              if "patch_auroc_top_decile_mean" in d else "--")
        lines.append(
            f"| {v} | {scale_of(v)} | {d['percase_spearman_mean']:.3f} +/- "
            f"{d['percase_spearman_std']:.3f} | {d['pooled_pearson']:.3f} | "
            f"{delta} | {frac} | {au} |")
    lines += [
        "",
        f"Sanity gate: V0 per-case Spearman = {v0_mean:.3f}, plausible band "
        f"{GATE_BAND} -> {'PASS' if gate['passed'] else 'FAIL'}. "
        f"V4 rank-identity max deviation from V0: {v4_max_dev:.2e}.",
        "",
    ]
    # Verdict logic: meaningful = mean per-case Spearman gain over V0 >= 0.1
    # AND >= 80% of cases improved (paired). Patch variants judged at their
    # own scale (localization coarsened to k-cell patches).
    verdict_bits = []
    for v in variant_names:
        d = variants[v]
        if v == "V0_raw" or "delta_vs_v0_mean" not in d:
            continue
        if (d["delta_vs_v0_mean"] >= 0.1
                and d["frac_cases_improved_vs_v0"] >= 0.8):
            verdict_bits.append(
                f"{v} ({scale_of(v)} scale): {d['percase_spearman_mean']:.3f} "
                f"vs V0 {v0_mean:.3f} ({d['delta_vs_v0_mean']:+.3f}, "
                f"{100 * d['frac_cases_improved_vs_v0']:.0f}% of cases)")
    if verdict_bits:
        verdict = ("VERDICT: post-processing meaningfully improves "
                   "localization -- " + "; ".join(verdict_bits) + ".")
    else:
        verdict = ("VERDICT: honest negative -- no post-processing variant "
                   "meaningfully improves per-cell localization (all deltas "
                   f"< +0.1 mean Spearman vs V0 = {v0_mean:.3f}); the residual "
                   "map remains a weak WHERE-signal at cell scale on deployed-"
                   "quality fields at n=200.")
    lines += [verdict, ""]
    out_md = os.path.join(args.out_dir, "percell_localization.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"wrote {out_md}")

    # ---- console summary ----
    log("================= VARIANT TABLE =================")
    for v in variant_names:
        d = variants[v]
        extra = ""
        if "patch_auroc_top_decile_mean" in d:
            extra = (f"  AUROC={d['patch_auroc_top_decile_mean']:.3f}"
                     f"+/-{d['patch_auroc_top_decile_std']:.3f}"
                     f"  patches/case={d['patches_per_case_mean']:.0f}")
        if "delta_vs_v0_mean" in d:
            extra += (f"  d_vs_V0={d['delta_vs_v0_mean']:+.3f}"
                      f" improved={100 * d['frac_cases_improved_vs_v0']:.0f}%")
        log(f"  {v:<16s} rho={d['percase_spearman_mean']:.3f}"
            f"+/-{d['percase_spearman_std']:.3f}"
            f"  pooledPearson={d['pooled_pearson']:.3f}{extra}")
    log(verdict)
    log(f"total runtime {runtime_sec:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
