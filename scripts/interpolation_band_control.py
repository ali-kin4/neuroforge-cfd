"""Controls that try to BREAK the parameter-interpolation baseline.

``scripts/parameter_interpolation_baseline.py`` found that interpolating the
AirfRANS flow field from the case parameters alone beats a matched-budget
Transolver on ``mse_v`` (2.6x) and ``mse_p`` (8.4x) under the paper's own
``evaluate_cases`` protocol. Before that number can be reported, four attacks
have to be answered. Each is stated here with what its outcome would PROVE,
BEFORE it is run.

A. WALL-DISTANCE DECOMPOSITION.
   Attack: "volume MSE over the 3c x 3c crop is dominated by near-freestream
   cells that any smooth interpolant gets right; the surrogate's advantage lives
   in the boundary layer and wake, which your aggregate hides."
   Test: decompose MSE and R^2 = 1 - MSE/Var by signed distance to the wall.
     * If near-wall R^2 is high (>= ~0.9), the interpolator reproduces near-wall
       structure and the objection fails.
     * If near-wall R^2 collapses while the far field carries the aggregate,
       the objection SUCCEEDS as stated about the interpolator -- but then it is
       simultaneously a finding about the BENCHMARK METRIC: AirfRANS volume MSE
       is dominated by cells a parameter interpolator already predicts.
   Either way the boundary gets characterised; neither outcome is spun.

B. TRAINING-SET-SIZE LADDER.
   How many cases does parameter interpolation need to reach the reported level?
   Reported for n_train in {25, 50, 100, 200, 400, 800}. This is the
   data-efficiency statement a reviewer will ask for and it cannot be gamed:
   the config is the one already CV-selected on the FULL train split.

C. PERMUTED-PARAMETER NEGATIVE CONTROL.
   Attack: "your result is an artifact of the metric / rasterisation / masking,
   not of parameter information."
   Test: permute the TEST feature vectors so every case is predicted from
   another case's parameters, everything else identical.
     * If the error stays low, the metric is degenerate and the whole result is
       void.
     * If the error blows up to the train-mean floor, the signal is genuinely
       carried by the parameters.

D. FEATURE ABLATION.
   ``[U, alpha]`` only, vs the full 7-feature vector. Isolates how much of the
   field is recoverable from the two flow parameters alone versus the airfoil
   shape digits.

CPU-only, no training, no GPU. Reads the config selected by (and the cached
pairs used by) the baseline script.

    .venv/Scripts/python.exe scripts/interpolation_band_control.py --task full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import neuroforge  # noqa: F401  -- MUST precede numpy (BLAS thread caps)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parameter_interpolation_baseline import (  # noqa: E402
    REFERENCE_ROWS,
    build_stack,
    build_weights,
    case_features,
    evaluate_cases,
    load_pairs,
    make_predict_fn,
    redimensionalise,
)

# Wall-distance bands in chord units (sdf > 0 in the fluid).
BANDS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 0.50), (0.50, 1e9)]
BAND_NAMES = ["0-0.02c", "0.02-0.05c", "0.05-0.15c", "0.15-0.5c", ">0.5c"]


def band_decomposition(pred_phys, test_pairs, hw):
    """MSE / R^2 / squared-error share per wall-distance band, pooled over cases."""
    H, W = test_pairs[0][1].u.shape
    chans = ("u", "v", "p")
    se = {b: {c: 0.0 for c in chans} for b in BAND_NAMES}
    n = {b: 0 for b in BAND_NAMES}
    s1 = {b: {c: 0.0 for c in chans} for b in BAND_NAMES}
    s2 = {b: {c: 0.0 for c in chans} for b in BAND_NAMES}
    for i, (case, ref) in enumerate(test_pairs):
        fluid = np.asarray(ref.mask).ravel() > 0.5
        d = np.asarray(ref.sdf, np.float64).ravel()
        row = pred_phys[i]
        gt = {"u": np.asarray(ref.u, np.float64).ravel(),
              "v": np.asarray(ref.v, np.float64).ravel(),
              "p": np.asarray(ref.p, np.float64).ravel()}
        pr = {"u": row[0:hw], "v": row[hw:2 * hw], "p": row[2 * hw:3 * hw]}
        for (lo, hi), bname in zip(BANDS, BAND_NAMES):
            m = fluid & (d > lo) & (d <= hi)
            if not m.any():
                continue
            n[bname] += int(m.sum())
            for c in chans:
                g = gt[c][m]
                e = pr[c][m] - g
                se[bname][c] += float((e**2).sum())
                s1[bname][c] += float(g.sum())
                s2[bname][c] += float((g**2).sum())
    out = {}
    tot_se = {c: sum(se[b][c] for b in BAND_NAMES) for c in chans}
    for b in BAND_NAMES:
        if n[b] == 0:
            continue
        row = {"n_cells": n[b], "cell_frac": n[b] / max(sum(n.values()), 1)}
        for c in chans:
            mse = se[b][c] / n[b]
            var = s2[b][c] / n[b] - (s1[b][c] / n[b]) ** 2
            row[f"mse_{c}"] = mse
            row[f"var_{c}"] = var
            row[f"r2_{c}"] = 1.0 - mse / var if var > 0 else float("nan")
            row[f"se_share_{c}"] = se[b][c] / tot_se[c] if tot_se[c] > 0 else float("nan")
        out[b] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="full")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--in-json", default=None)
    ap.add_argument("--out-dir", default="results/interpolation")
    ap.add_argument("--rep", default="nd")
    ap.add_argument("--fill", default="nearest")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    t_start = time.time()
    in_json = a.in_json or os.path.join(a.out_dir, f"interp_{a.task}.json")
    base = json.load(open(in_json, encoding="utf-8"))
    cfg = base["variants"][a.rep]["cv_selected_cfg_raw"]
    print(f"[band] using CV-selected config from {in_json}: {cfg}", flush=True)

    train_pairs = load_pairs(a.cache_dir, a.task, True, a.resolution, a.n_train)
    test_pairs = load_pairs(a.cache_dir, a.task, False, a.resolution, a.n_test)
    names_tr = [c.name for c, _ in train_pairs]
    names_te = [c.name for c, _ in test_pairs]
    H, W = train_pairs[0][1].u.shape
    hw = H * W
    Xtr = np.stack([case_features(n) for n in names_tr])
    Xte = np.stack([case_features(n) for n in names_te])
    Ytr, Mtr, _ = build_stack(train_pairs, a.rep, a.fill)
    print(f"[band] stacks ready {Ytr.shape}", flush=True)

    def run(Xq, active=None, Xt=None):
        Wm = build_weights(cfg, Xt if Xt is not None else Xtr, Xq, active=active)
        pred_nd = Wm @ Ytr
        return redimensionalise(pred_nd, names_te, a.rep, hw)

    def score(pred_phys):
        pfn = make_predict_fn(pred_phys, names_te, (H, W))
        g = evaluate_cases(pfn, test_pairs)
        assert g["n_cases"] == len(test_pairs), g["n_cases"]
        return {k: g[k] for k in ("mse_u", "mse_v", "mse_p", "surface_mse_p",
                                  "rho_cl", "rho_cd", "cl_rel_err_mean",
                                  "cd_rel_err_mean", "n_cases")}

    out = {"task": a.task, "rep": a.rep, "fill": a.fill, "cfg": cfg,
           "reference_rows": REFERENCE_ROWS, "n_train": len(train_pairs),
           "n_test": len(test_pairs)}

    # ---- A. wall-distance decomposition -----------------------------------
    t0 = time.time()
    pred_full = run(Xte)
    out["A_band_decomposition"] = band_decomposition(pred_full, test_pairs, hw)
    out["A_headline_metrics"] = score(pred_full)
    print(f"[band] A done in {time.time() - t0:.1f}s", flush=True)
    for b, r in out["A_band_decomposition"].items():
        print(f"  {b:12s} cells {r['cell_frac']:6.3f}  R2 u/v/p "
              f"{r['r2_u']:.4f}/{r['r2_v']:.4f}/{r['r2_p']:.4f}  "
              f"SEshare u/v/p {r['se_share_u']:.3f}/{r['se_share_v']:.3f}/"
              f"{r['se_share_p']:.3f}", flush=True)

    # ---- B. training-set-size ladder --------------------------------------
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(names_tr))
    ladder = {}
    for nt in (25, 50, 100, 200, 400, len(names_tr)):
        if nt > len(names_tr):
            continue
        active = np.zeros(len(names_tr), bool)
        active[perm[:nt]] = True
        ladder[str(nt)] = score(run(Xte, active=active))
        print(f"  [n_train={nt:4d}] u={ladder[str(nt)]['mse_u']:.4g} "
              f"v={ladder[str(nt)]['mse_v']:.4g} p={ladder[str(nt)]['mse_p']:.5g} "
              f"surf_p={ladder[str(nt)]['surface_mse_p']:.5g}", flush=True)
    out["B_train_size_ladder"] = ladder

    # ---- C. permuted-parameter negative control ---------------------------
    pidx = np.random.default_rng(1234).permutation(len(names_te))
    while np.any(pidx == np.arange(len(names_te))):       # derangement
        pidx = np.random.default_rng(int(rng.integers(1 << 30))).permutation(len(names_te))
    out["C_permuted_parameters"] = score(run(Xte[pidx]))
    print(f"  [permuted] u={out['C_permuted_parameters']['mse_u']:.4g} "
          f"v={out['C_permuted_parameters']['mse_v']:.4g} "
          f"p={out['C_permuted_parameters']['mse_p']:.5g}", flush=True)

    # ---- D. feature ablation ----------------------------------------------
    abl = {}
    subsets = {"U_alpha_only": [0, 1], "U_alpha_thickness": [0, 1, 2],
               "no_U": [1, 2, 3, 4, 5, 6], "all7": list(range(7))}
    for nm, cols in subsets.items():
        cols = np.array(cols)
        # ``_standardise`` applies w_U to COLUMN 0; that column is only U when U
        # is in the subset. Otherwise fall back to an unweighted metric.
        acfg = dict(cfg)
        if cols[0] != 0:
            acfg["w_U"] = 1.0
        Wm = build_weights(acfg, Xtr[:, cols], Xte[:, cols])
        abl[nm] = score(redimensionalise(Wm @ Ytr, names_te, a.rep, hw))
        print(f"  [feat {nm:18s}] u={abl[nm]['mse_u']:.4g} v={abl[nm]['mse_v']:.4g} "
              f"p={abl[nm]['mse_p']:.5g}", flush=True)
    out["D_feature_ablation"] = abl

    out["wallclock_sec"] = time.time() - t_start
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"interp_band_control_{a.task}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"[band] wrote {path} ({out['wallclock_sec']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
