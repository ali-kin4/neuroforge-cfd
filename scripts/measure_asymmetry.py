"""The measure-asymmetry audit: is the 8.4x a finding, or a choice of weighting?

THE OBJECTION (``docs/paper/review/r2_round2.md`` section 1, scored fatal)
--------------------------------------------------------------------------
    The interpolator's hyperparameters are selected by 5-fold CV on the
    AREA-UNIFORM r128 grid MSE (``interpolation_baseline.md:86-89``), a measure
    that is 98.7% far field by the paper's own band table. The Transolver
    minimises PER-POINT MSE under AirfRANS's own WALL-CLUSTERED node measure
    (``run_baselines.py:121-129``). Rasterising the cloud prediction at the
    cloud's own nodes (``transolver_adapter.py:93-104``) makes the grid metric a
    linear RE-WEIGHTING of per-node errors that upweights the sparse far field.
    So one arm is tuned on the scoreboard and the other is not, and the paper
    never measures the size of the gap. It also never decomposes the
    SURROGATE's error by wall distance: the localisation thesis
    (``body.tex:546-550``) is measured on the baseline arm only.

This script answers all three parts, zero training, inference only.

    A. NODE MEASURE vs AREA MEASURE.  Node fractions of the native AirfRANS
       clouds in physical wall bands, against the area (cell) fractions of the
       same bands on the r128 crop, on the same 200 test cases. Three measures
       are reported and they are NOT interchangeable:
         (A1) area-uniform over the 3c x 3c crop  -- the paper's scoring measure;
         (A2) node-uniform RESTRICTED TO THE CROP -- the apples-to-apples
              re-weighting of the same scored region, and the weights used in C;
         (A3) node-uniform over the FULL cloud    -- the measure Transolver's
              training loss actually integrates, and the literal answer to the
              reviewer's question 1.
    B. BAND DECOMPOSITION OF BOTH ARMS.  Transolver's squared error and the
       interpolator's squared error over identical physical wall bands, same
       cases, same rasterisation, same fluid mask, same accumulator; then the
       per-band ratio r_b = MSE_interp(b) / MSE_transolver(b), which IS the
       localisation claim and has never been computed.
    C. RE-SCORING UNDER THE NODE MEASURE.  The same grid errors for both arms,
       re-weighted cell-by-cell by the number of native cloud nodes the cell
       contains. This changes ONLY the weights: same fields, same truth, same
       rasterisation, both arms. It is therefore the clean discriminator for a
       measure objection. Its limitation is stated and not hidden: node
       weighting fixes the WEIGHTING, not the r128 REPRESENTATION ceiling.
    D. THE REPRESENTATION CEILING (diagnostic, no threshold).  Per-node squared
       error of the Transolver prediction against the native targets, by band,
       against the per-node squared error of the RASTERISED GROUND TRUTH
       resampled at those same nodes. The second is the error floor the r128
       scoring protocol imposes on any method. It says what the grid metric
       cannot see, which is the half of the objection that re-weighting cannot
       reach.

PRE-REGISTERED DECISION RULES (written before any number was produced)
----------------------------------------------------------------------
D1 -- SIZE OF THE MEASURE GAP.  Let
       g(band) = node_fraction(band, measure A2) / area_fraction(band, A1).
     Read on the union of bands inside 0.05c:
       g >= 10   -> the reviewer's mechanism is LARGE; the paper must print the
                    factor and scope the 8.4x explicitly;
       2 <= g<10 -> REAL BUT MODERATE; print the factor, scope in one sentence;
       g < 2     -> the mechanism DISSOLVES; the two measures are close and the
                    objection is answered by reporting g.
     Reported for 0.02c as well, and under A3 alongside A2.

D2 -- IS THE LOCALISATION A PROPERTY OF BOTH ARMS?  Let
       r_b = MSE_interp(b) / MSE_transolver(b), pooled over cells, on u,
     over the seven bands 0-0.005 / 0.005-0.01 / 0.01-0.02 / 0.02-0.05 /
     0.05-0.15 / 0.15-0.5 / >0.5 c. Let S = r_innermost / r_outermost.
       S >= 10 and r_b decreasing outward (allowing one inversion)
                 -> CONFIRMED ON BOTH ARMS: the surrogate's advantage is
                    near-wall and the interpolator's advantage is far field;
                    the thesis is measured on both arms.
       S < 3     -> INTERPOLATOR-ONLY: the localisation claim describes where
                    the interpolator's error lives and nothing about where the
                    surrogate earns its keep. ``body.tex:546-550`` must be
                    restated as the reviewer's item 6 demands.
       otherwise -> PARTIAL.
     Reported for v, p and nut; the verdict is read on u, the channel the
     localisation claim is stated about.

D3 -- DOES THE HEADLINE SURVIVE THE NODE MEASURE?  Let
       R_p = mse_p(interp) / mse_p(Transolver),
     computed on the identical grid errors, once with area-uniform weights
     (this must reproduce ~8.4) and once with node-count weights (C).
       R_p_node >= 4          -> SURVIVES. Report the node-measure number as
                                 scope; the headline stands.
       1.5 <= R_p_node < 4    -> SURVIVES SCOPED. The node-measure ratio becomes
                                 co-primary and must appear beside the 8.4x
                                 everywhere the 8.4x appears.
       1.0 <  R_p_node < 1.5  -> WITHDRAW THE HEADLINE NUMBER. Report the range
                                 across measures instead of a single factor.
       R_p_node <= 1.0        -> ARTIFACT. The 8.4x is a measure choice. Say so
                                 plainly, in the abstract, and withdraw it.
     The verdict is read on p (the headline channel). u, v and nut are reported
     and may not be substituted for p.

D4 -- REPRESENTATION CEILING (diagnostic; no threshold, no verdict). Reported
     because a re-weighting cannot answer the representation half of the
     objection and pretending otherwise would be the selective disclosure the
     review names twice.

GATES -- the run aborts if any fails
------------------------------------
G1  The interpolator arm reproduces ``results/interpolation/interp_full.json``
    (mse_u 0.7816026776 / mse_v 0.03361523305 / mse_p 75.03145038) to < 1e-9
    relative. Same CV-selected config, read from that JSON, never re-selected.
G2  Each Transolver seed reproduces ``results/v2/v2_results.json``
    ``backbone_per_seed`` mse_u/mse_v/mse_p to < 2e-2 relative (CPU/GPU float
    reassociation only; the checkpoint is the deployed backbone).
G3  The five-band decomposition of the interpolator arm reproduces
    ``results/interpolation/interp_band_control_full.json`` section A
    (cell_frac, r2_*, r2_pc_*, se_share_*) to < 1e-6 relative -- i.e. this
    script's accumulator is the published one.
G4  For every arm, channel and band grid: sum_b n_b * MSE_b == pooled total SE,
    to < 1e-10 relative. The band table must reconstruct the aggregate.
G5  The reused Delaunay triangulation (one per case, shared across the three
    seeds) reproduces ``rasterize_point_cloud`` EXACTLY on 3 cases, and the
    primed geometry cache reproduces ``signed_distance``/``solid_mask`` exactly
    on 3 cases (the ladder's precedent, same assertion).
G6  Cloud sdf (AirfRANS column 4) against the grid ``signed_distance`` sampled
    at the same node positions: the distribution and the fraction of nodes whose
    BAND ASSIGNMENT agrees are reported. Not a pass/fail gate, a disclosure.

AMENDMENT 2 (recorded, D3's rule and threshold untouched). ``--stage sensitivity``
reads the two committed artifacts and re-derives the headline ratio under FOUR
discretisations of the node measure, because "weight by node count" admits more
than one construction and reporting only the one that was pre-registered would
leave a discoverable:
  (i)   cell-level node counts, per-case then case-mean  -- the PRE-REGISTERED
        estimator D3 is read on, and the finest one available: it is exactly
        "sample the grid error field at each node, average over nodes";
  (ii)  the same, pooled over all cells and cases;
  (iii) band-level, using the grid-binned node mass per band -- coarser: it
        assumes the error is uniform within a band;
  (iv)  band-level, using the TRUE node mass per band from block A -- coarser
        still, and internally inconsistent (it pairs node masses with cell MSEs
        drawn from a different spatial set), reported because it is the
        construction a reader would try by hand from the two tables.
The verdict stays on (i). The range across all four is the number the paper
should quote.

AMENDMENT 1 (recorded, threshold untouched). G6 originally reported only the
seven-band assignment agreement. That statistic is dominated by disagreements
at the 0.005c/0.01c interior edges, which D1 never uses: D1 is read on the
CUMULATIVE fractions inside 0.02c and inside 0.05c. The two cumulative
agreements are therefore added, plus the node fractions inside those two
thresholds computed from the GRID sdf instead of the cloud sdf -- i.e. D1's
headline numbers recomputed under the other distance function. No rule and no
threshold is changed; a diagnostic is added that makes D1 harder to attack.

WHAT THIS CANNOT DO, STATED UP FRONT
------------------------------------
It does not score either arm at the native nodes. Sampling the interpolator's
r128 grid at wall nodes would change the MEASURE and the REPRESENTATION at the
same time and would penalise it for a reason that has nothing to do with the
objection. Block C changes the weights only. Block D reports the size of the
representation gap that remains.

The manuscript's claim that a point-space head-to-head needs a
``PointNormalizer`` the checkpoints do not store is FALSE:
``checkpoints/v2_transolver/seed{m}.pt`` carries ``point_norm`` (mean_in,
std_in, mean_out, std_out, eps) and ``grid_norm``. That correction is part of
the deliverable.

RUN
    .venv/Scripts/python.exe scripts/measure_asymmetry.py --stage nodes
    .venv/Scripts/python.exe scripts/measure_asymmetry.py --stage full --device auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (BLAS thread caps)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parameter_interpolation_baseline import (  # noqa: E402
    _GEOM_CACHE,
    build_stack,
    build_weights,
    case_features,
    load_pairs,
    make_predict_fn as interp_predict_fn,
    redimensionalise,
)

from neuroforge.physics.evaluation import per_channel_mse  # noqa: E402
from neuroforge.physics.metrics import _bilinear_sample  # noqa: E402

INF = 1e9
BANDS5 = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 0.50), (0.50, INF)]
NAMES5 = ["0-0.02c", "0.02-0.05c", "0.05-0.15c", "0.15-0.5c", ">0.5c"]
BANDS7 = [(0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05),
          (0.05, 0.15), (0.15, 0.50), (0.50, INF)]
NAMES7 = ["0-0.005c", "0.005-0.01c", "0.01-0.02c", "0.02-0.05c",
          "0.05-0.15c", "0.15-0.5c", ">0.5c"]
CHANS = ("u", "v", "p", "nut")


def log(msg: str) -> None:
    print(f"[asym] {msg}", flush=True)


def band_index(d: np.ndarray, valid: np.ndarray, bands) -> np.ndarray:
    """Index into ``bands`` per entry; -1 where invalid or in no band."""
    out = np.full(d.shape, -1, np.int64)
    for k, (lo, hi) in enumerate(bands):
        out[valid & (d > lo) & (d <= hi)] = k
    return out


class GridAcc:
    """Pooled-cell accumulators over a band grid, for one arm."""

    def __init__(self, nb: int) -> None:
        self.nb = nb
        self.n = np.zeros(nb)                       # fluid cells
        self.w = np.zeros(nb)                       # node weight
        self.se = {c: np.zeros(nb) for c in CHANS}  # area-uniform SE
        self.wse = {c: np.zeros(nb) for c in CHANS}  # node-weighted SE
        self.s1 = {c: np.zeros(nb) for c in CHANS}  # sum(gt)
        self.s2 = {c: np.zeros(nb) for c in CHANS}  # sum(gt^2)
        self.ssw = {c: np.zeros(nb) for c in CHANS}  # per-case-centred SS

    def add_case(self, bidx, keep, gt, pr, wts) -> None:
        nb = self.nb
        b = bidx[keep]
        w = wts[keep]
        self.n += np.bincount(b, minlength=nb)
        self.w += np.bincount(b, weights=w, minlength=nb)
        for c in CHANS:
            g = gt[c][keep]
            e2 = (pr[c][keep] - g) ** 2
            self.se[c] += np.bincount(b, weights=e2, minlength=nb)
            self.wse[c] += np.bincount(b, weights=w * e2, minlength=nb)
            s1 = np.bincount(b, weights=g, minlength=nb)
            s2 = np.bincount(b, weights=g * g, minlength=nb)
            cnt = np.bincount(b, minlength=nb)
            self.s1[c] += s1
            self.s2[c] += s2
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
            self.ssw[c] += s2 - cnt * mu**2

    def table(self, names) -> dict:
        tot_n = float(self.n.sum())
        out = {}
        tot_se = {c: float(self.se[c].sum()) for c in CHANS}
        for k, nm in enumerate(names):
            if self.n[k] <= 0:
                continue
            n = float(self.n[k])
            row = {"n_cells": int(n), "cell_frac": n / tot_n,
                   "node_weight": float(self.w[k]),
                   "node_weight_frac": float(self.w[k] / max(self.w.sum(), 1e-30))}
            for c in CHANS:
                mse = self.se[c][k] / n
                var = self.s2[c][k] / n - (self.s1[c][k] / n) ** 2
                var_pc = self.ssw[c][k] / n
                row[f"mse_{c}"] = float(mse)
                row[f"mse_node_{c}"] = float(self.wse[c][k] / self.w[k]) if self.w[k] > 0 else float("nan")
                row[f"var_{c}"] = float(var)
                row[f"var_pc_{c}"] = float(var_pc)
                row[f"r2_{c}"] = float(1.0 - mse / var) if var > 0 else float("nan")
                row[f"r2_pc_{c}"] = float(1.0 - mse / var_pc) if var_pc > 0 else float("nan")
                row[f"se_share_{c}"] = float(self.se[c][k] / tot_se[c]) if tot_se[c] > 0 else float("nan")
                row[f"se_node_share_{c}"] = (
                    float(self.wse[c][k] / sum(self.wse[c])) if sum(self.wse[c]) > 0 else float("nan")
                )
            out[nm] = row
        return out

    def pooled(self) -> dict:
        n = float(self.n.sum())
        w = float(self.w.sum())
        return {
            **{f"mse_{c}": float(self.se[c].sum() / n) for c in CHANS},
            **{f"mse_node_{c}": float(self.wse[c].sum() / w) for c in CHANS},
            "n_cells": int(n), "node_weight": w,
        }


def gate_g4(acc: GridAcc, label: str, tol: float = 1e-10) -> dict:
    """sum_b n_b * MSE_b == pooled SE, per channel."""
    rep = {}
    for c in CHANS:
        lhs = float((acc.n * (acc.se[c] / np.maximum(acc.n, 1e-30))).sum())
        rhs = float(acc.se[c].sum())
        rel = abs(lhs - rhs) / max(abs(rhs), 1e-300)
        rep[c] = rel
        if not (rel < tol):
            raise AssertionError(f"G4 FAILED [{label}/{c}]: {lhs} vs {rhs} (rel {rel:.3e})")
    return rep


def node_counts_per_cell(pos: np.ndarray, domain) -> tuple[np.ndarray, np.ndarray]:
    """Node count per grid cell (nearest-node binning) + in-crop mask."""
    xmin, xmax, ymin, ymax = domain.bounds
    nx, ny = domain.nx, domain.ny
    x, y = pos[:, 0].astype(np.float64), pos[:, 1].astype(np.float64)
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    ix = np.clip(np.round((x - xmin) / (xmax - xmin) * (nx - 1)), 0, nx - 1).astype(np.int64)
    iy = np.clip(np.round((y - ymin) / (ymax - ymin) * (ny - 1)), 0, ny - 1).astype(np.int64)
    flat = iy * nx + ix
    counts = np.bincount(flat[inside], minlength=nx * ny).astype(np.float64)
    return counts, inside


# --------------------------------------------------------------------------- #
# Stage A: the node measure (pure numpy, no torch, stands alone)
# --------------------------------------------------------------------------- #
def stage_nodes(a) -> dict:
    from neuroforge.data.pointcloud import load_airfrans_pointclouds

    t0 = time.time()
    test_pairs = load_pairs(a.cache_dir, a.task, False, a.resolution, a.n_test)
    log(f"loaded {len(test_pairs)} rasterised GT pairs ({time.time()-t0:.1f}s)")
    t0 = time.time()
    pcs = load_airfrans_pointclouds(root=a.root, task=a.task, train=False,
                                    limit=a.n_test, cache_dir=a.cache_dir)
    by_name = {pc.name: pc for pc in pcs}
    log(f"loaded {len(pcs)} native clouds ({time.time()-t0:.1f}s)")

    nb5, nb7 = len(BANDS5), len(BANDS7)
    area5, area7 = np.zeros(nb5), np.zeros(nb7)
    node_crop5, node_crop7 = np.zeros(nb5), np.zeros(nb7)
    node_full5, node_full7 = np.zeros(nb5), np.zeros(nb7)
    n_nodes_total = 0
    n_nodes_crop = 0
    n_nodes_wall = 0          # sdf <= 0 (on the body): in no band
    n_nodes_solidcell = 0     # in-crop node landing in a non-fluid cell
    sdf_absdiff = []
    band_agree = 0
    band_total = 0
    thr_agree = {0.02: 0, 0.05: 0}
    thr_grid_in = {0.02: 0, 0.05: 0}
    thr_cloud_in = {0.02: 0, 0.05: 0}
    per_case = []

    for case, ref in test_pairs:
        pc = by_name[case.name]
        fluid = np.asarray(ref.mask).ravel() > 0.5
        d = np.asarray(ref.sdf, np.float64).ravel()
        b5 = band_index(d, fluid, BANDS5)
        b7 = band_index(d, fluid, BANDS7)
        area5 += np.bincount(b5[b5 >= 0], minlength=nb5)
        area7 += np.bincount(b7[b7 >= 0], minlength=nb7)

        s = np.asarray(pc.features[:, 4], np.float64)   # AirfRANS distance to airfoil
        pos = np.asarray(pc.pos, np.float64)
        counts, inside = node_counts_per_cell(pos, case.domain)
        n_nodes_total += s.size
        n_nodes_crop += int(inside.sum())
        n_nodes_wall += int((s <= 0).sum())
        n_nodes_solidcell += int(counts[~fluid].sum())

        ok = np.ones(s.shape, bool)
        nb_full5 = band_index(s, ok, BANDS5)
        nb_full7 = band_index(s, ok, BANDS7)
        node_full5 += np.bincount(nb_full5[nb_full5 >= 0], minlength=nb5)
        node_full7 += np.bincount(nb_full7[nb_full7 >= 0], minlength=nb7)
        nc5 = band_index(s, inside, BANDS5)
        nc7 = band_index(s, inside, BANDS7)
        node_crop5 += np.bincount(nc5[nc5 >= 0], minlength=nb5)
        node_crop7 += np.bincount(nc7[nc7 >= 0], minlength=nb7)

        # G6: cloud sdf vs grid signed_distance at the same points (in-crop only)
        gs = _bilinear_sample(np.asarray(ref.sdf, np.float64), case.domain, pos[inside])
        sc = s[inside]
        sdf_absdiff.append(np.abs(gs - sc))
        bg = band_index(gs, np.ones(gs.shape, bool), BANDS7)
        bc = band_index(sc, np.ones(sc.shape, bool), BANDS7)
        band_agree += int((bg == bc).sum())
        band_total += int(bg.size)
        for e in (0.02, 0.05):
            thr_agree[e] += int(((gs <= e) == (sc <= e)).sum())
            thr_grid_in[e] += int((gs <= e).sum())
            thr_cloud_in[e] += int((sc <= e).sum())

        per_case.append({
            "name": case.name, "n_nodes": int(s.size), "n_nodes_crop": int(inside.sum()),
            "node_frac_lt_002": float((s <= 0.02).mean()),
            "node_frac_lt_005": float((s <= 0.05).mean()),
            "node_frac_crop_lt_002": float(((s <= 0.02) & inside).sum() / max(inside.sum(), 1)),
            "node_frac_crop_lt_005": float(((s <= 0.05) & inside).sum() / max(inside.sum(), 1)),
        })

    diffs = np.concatenate(sdf_absdiff)

    def frac(v):
        return (v / max(v.sum(), 1e-30)).tolist()

    out = {
        "n_cases": len(test_pairs),
        "bands5": NAMES5, "bands7": NAMES7,
        "A1_area_frac_crop_5": frac(area5), "A1_area_frac_crop_7": frac(area7),
        "A2_node_frac_crop_5": frac(node_crop5), "A2_node_frac_crop_7": frac(node_crop7),
        "A3_node_frac_full_5": frac(node_full5), "A3_node_frac_full_7": frac(node_full7),
        "counts": {
            "area_cells_5": area5.tolist(), "area_cells_7": area7.tolist(),
            "nodes_crop_5": node_crop5.tolist(), "nodes_crop_7": node_crop7.tolist(),
            "nodes_full_5": node_full5.tolist(), "nodes_full_7": node_full7.tolist(),
            "n_nodes_total": int(n_nodes_total), "n_nodes_in_crop": int(n_nodes_crop),
            "n_nodes_on_wall_sdf_le_0": int(n_nodes_wall),
            "n_nodes_in_crop_landing_in_solid_cell": int(n_nodes_solidcell),
        },
        "G6_sdf_agreement": {
            "n_nodes": int(diffs.size),
            "mean_abs_diff_c": float(diffs.mean()),
            "median_abs_diff_c": float(np.median(diffs)),
            "p99_abs_diff_c": float(np.percentile(diffs, 99)),
            "max_abs_diff_c": float(diffs.max()),
            "band7_assignment_agreement": float(band_agree / max(band_total, 1)),
            "threshold_agreement_0.02c": float(thr_agree[0.02] / max(band_total, 1)),
            "threshold_agreement_0.05c": float(thr_agree[0.05] / max(band_total, 1)),
            "node_frac_crop_inside_0.02c_by_GRID_sdf": float(thr_grid_in[0.02] / max(band_total, 1)),
            "node_frac_crop_inside_0.02c_by_CLOUD_sdf": float(thr_cloud_in[0.02] / max(band_total, 1)),
            "node_frac_crop_inside_0.05c_by_GRID_sdf": float(thr_grid_in[0.05] / max(band_total, 1)),
            "node_frac_crop_inside_0.05c_by_CLOUD_sdf": float(thr_cloud_in[0.05] / max(band_total, 1)),
        },
        "per_case": per_case,
    }

    # D1: the gap factor g, on the cumulative bands the reviewer named.
    def cum(fr, names, edge):
        k = names.index(edge)
        return float(sum(fr[: k + 1]))

    gaps = {}
    for edge, nm in ((0.02, "0.02-0.05c"), (0.05, "0.05-0.15c")):
        i5 = NAMES5.index(nm) - 1
        a1 = float(sum(out["A1_area_frac_crop_5"][: i5 + 1]))
        a2 = float(sum(out["A2_node_frac_crop_5"][: i5 + 1]))
        a3 = float(sum(out["A3_node_frac_full_5"][: i5 + 1]))
        gaps[f"inside_{edge}c"] = {
            "area_frac_crop_A1": a1, "node_frac_crop_A2": a2, "node_frac_full_A3": a3,
            "g_A2_over_A1": a2 / a1 if a1 > 0 else float("nan"),
            "g_A3_over_A1": a3 / a1 if a1 > 0 else float("nan"),
        }
    g = gaps["inside_0.05c"]["g_A2_over_A1"]
    out["D1_gap"] = gaps
    out["D1_verdict"] = ("LARGE" if g >= 10 else "REAL BUT MODERATE" if g >= 2 else "DISSOLVES")
    out["D1_rule"] = "g = node_frac(A2)/area_frac(A1) inside 0.05c; >=10 LARGE, >=2 MODERATE, <2 DISSOLVES"
    return out


# --------------------------------------------------------------------------- #
# Stage B/C/D: both arms on the grid + Transolver at the nodes
# --------------------------------------------------------------------------- #
def stage_full(a) -> dict:
    import torch

    from neuroforge.data.pointcloud import load_airfrans_pointclouds
    from neuroforge.data.rasterize import rasterize_point_cloud
    from neuroforge.geometry.sdf import signed_distance, solid_mask
    from recompute_force_vs_official import load_backbone

    device = torch.device(a.device if a.device not in ("auto", "") else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"device={device}")

    # ---- data --------------------------------------------------------------
    t0 = time.time()
    test_pairs = load_pairs(a.cache_dir, a.task, False, a.resolution, a.n_test)
    train_pairs = load_pairs(a.cache_dir, a.task, True, a.resolution, a.n_train)
    log(f"grid pairs: {len(train_pairs)} train / {len(test_pairs)} test ({time.time()-t0:.1f}s)")
    names_tr = [c.name for c, _ in train_pairs]
    names_te = [c.name for c, _ in test_pairs]
    assert not (set(names_tr) & set(names_te)), "train/test name overlap"
    H, W = test_pairs[0][1].u.shape
    hw = H * W

    # ---- arm 1: the committed interpolator (config read, never re-selected) --
    in_json = os.path.join(a.out_dir, f"interp_{a.task}.json")
    base = json.load(open(in_json, encoding="utf-8"))
    cfg = base["variants"][a.rep]["cv_selected_cfg_raw"]
    log(f"interpolator config from {in_json}: {cfg}")
    t0 = time.time()
    Xtr = np.stack([case_features(n) for n in names_tr])
    Xte = np.stack([case_features(n) for n in names_te])
    Ytr, _Mtr, _ = build_stack(train_pairs, a.rep, a.fill)
    Wm = build_weights(cfg, Xtr, Xte)
    pred_interp = redimensionalise(Wm @ Ytr, names_te, a.rep, hw)
    log(f"interpolator predictions ready {pred_interp.shape} ({time.time()-t0:.1f}s)")
    del Ytr

    # ---- prime the geometry cache from the cached GT arrays, verify bitwise --
    for case, ref in test_pairs:
        _GEOM_CACHE[case.name] = (np.asarray(ref.sdf), np.asarray(ref.mask))
    for case, ref in test_pairs[:3]:
        s2 = signed_distance(case.geometry, case.domain)
        m2 = solid_mask(case.geometry, case.domain)
        assert np.array_equal(s2, np.asarray(ref.sdf)) and s2.dtype == np.asarray(ref.sdf).dtype, \
            f"G5 geometry cache mismatch (sdf) on {case.name}"
        assert np.array_equal(m2, np.asarray(ref.mask)), f"G5 geometry cache mismatch (mask) on {case.name}"
    log("G5a geometry cache primed and verified bitwise on 3 cases")
    ifn = interp_predict_fn(pred_interp, names_te, (H, W))

    # ---- arm 2: the deployed Transolver backbones ---------------------------
    seeds = list(a.seeds)
    models = {}
    for s in seeds:
        ck = os.path.join(a.ckpt_dir, f"seed{s}.pt")
        m, pn, _gn, _nu = load_backbone(ck, device)
        models[s] = (m, pn)
        log(f"loaded backbone {ck}")

    t0 = time.time()
    pcs = load_airfrans_pointclouds(root=a.root, task=a.task, train=False,
                                    limit=a.n_test, cache_dir=a.cache_dir)
    by_name = {pc.name: pc for pc in pcs}
    log(f"loaded {len(pcs)} native clouds ({time.time()-t0:.1f}s)")
    missing = [n for n in names_te if n not in by_name]
    assert not missing, f"{len(missing)} test cases without a cloud: {missing[:3]}"

    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    arms = ["interp"] + [f"transolver_seed{s}" for s in seeds]
    acc5 = {k: GridAcc(len(BANDS5)) for k in arms}
    acc7 = {k: GridAcc(len(BANDS7)) for k in arms}
    case_metrics = {k: [] for k in arms}          # area-uniform per-case MSE
    case_metrics_node = {k: [] for k in arms}     # node-weighted per-case MSE
    nb7 = len(BANDS7)
    node_se = {f"transolver_seed{s}": {c: np.zeros(nb7) for c in CHANS} for s in seeds}
    node_se["r128_ceiling"] = {c: np.zeros(nb7) for c in CHANS}
    node_n = np.zeros(nb7)
    node_n_crop = np.zeros(nb7)
    node_se_crop = {f"transolver_seed{s}": {c: np.zeros(nb7) for c in CHANS} for s in seeds}
    node_se_crop["r128_ceiling"] = {c: np.zeros(nb7) for c in CHANS}

    t_start = time.time()
    for ci, (case, ref) in enumerate(test_pairs):
        pc = by_name[case.name]
        pos = np.asarray(pc.pos, np.float64)
        fluid = np.asarray(ref.mask).ravel() > 0.5
        d = np.asarray(ref.sdf, np.float64).ravel()
        b5 = band_index(d, fluid, BANDS5)
        b7 = band_index(d, fluid, BANDS7)
        counts, inside = node_counts_per_cell(pos, case.domain)
        counts = np.where(fluid, counts, 0.0)      # node weights live on fluid cells
        gt = {c: np.asarray(getattr(ref, c), np.float64).ravel() for c in CHANS}

        # --- Transolver: per-node predictions, then ONE Delaunay for all seeds
        tri = Delaunay(pos)
        node_pred = {}
        for s in seeds:
            m, pn = models[s]
            feats = pn.transform_in(pc.features)
            with torch.no_grad():
                y = m(torch.from_numpy(feats).to(device).unsqueeze(0)).squeeze(0)
                y = pn.inverse_out(y)
            node_pred[s] = y.detach().cpu().numpy().astype(np.float32)

        preds = {"interp": {c: np.asarray(getattr(ifn(case), c), np.float64).ravel() for c in CHANS}}
        for s in seeds:
            interp_nd = LinearNDInterpolator(tri, np.asarray(node_pred[s], np.float64),
                                             fill_value=0.0)
            X, Y = case.domain.grid()
            xi = np.stack([X.ravel(), Y.ravel()], axis=1)
            raster = np.asarray(interp_nd(xi), np.float64)
            nanrows = np.isnan(raster).any(axis=1)
            if nanrows.any():
                from scipy.interpolate import NearestNDInterpolator
                raster[nanrows] = NearestNDInterpolator(pos, np.asarray(node_pred[s], np.float64))(xi[nanrows])
            raster = raster.reshape(H, W, 4).transpose(2, 0, 1).astype(np.float32)
            solid = np.asarray(ref.mask) < 0.5
            preds[f"transolver_seed{s}"] = {
                "u": np.where(solid, 0.0, raster[0]).astype(np.float64).ravel(),
                "v": np.where(solid, 0.0, raster[1]).astype(np.float64).ravel(),
                "p": raster[2].astype(np.float64).ravel(),
                "nut": np.where(solid, 0.0, np.maximum(raster[3], 0.0)).astype(np.float64).ravel(),
            }
            # G5b: the reused triangulation must equal the library path exactly.
            if ci < 3:
                ref_raster = rasterize_point_cloud(pc.pos, node_pred[s], case.domain,
                                                   fill=0.0, method="linear")
                assert np.array_equal(ref_raster, raster), \
                    f"G5 raster mismatch on {case.name} seed{s}"

        for arm in arms:
            acc5[arm].add_case(b5, b5 >= 0, gt, preds[arm], counts)
            acc7[arm].add_case(b7, b7 >= 0, gt, preds[arm], counts)
            mm = {f"mse_{c}": float(np.mean((preds[arm][c][fluid] - gt[c][fluid]) ** 2)) for c in CHANS}
            case_metrics[arm].append(mm)
            wsum = counts[fluid].sum()
            case_metrics_node[arm].append({
                f"mse_{c}": float(np.sum(counts[fluid] * (preds[arm][c][fluid] - gt[c][fluid]) ** 2) / wsum)
                for c in CHANS
            })

        # --- D: node space (Transolver's own measure) + the r128 ceiling
        s_cloud = np.asarray(pc.features[:, 4], np.float64)
        nb = band_index(s_cloud, np.ones(s_cloud.shape, bool), BANDS7)
        keep = nb >= 0
        node_n += np.bincount(nb[keep], minlength=nb7)
        keep_c = keep & inside
        node_n_crop += np.bincount(nb[keep_c], minlength=nb7)
        tgt = np.asarray(pc.targets, np.float64)
        ceil = np.stack([_bilinear_sample(np.asarray(getattr(ref, c), np.float64),
                                          case.domain, pos) for c in CHANS], axis=1)
        for j, c in enumerate(CHANS):
            e2 = (ceil[:, j] - tgt[:, j]) ** 2
            node_se["r128_ceiling"][c] += np.bincount(nb[keep], weights=e2[keep], minlength=nb7)
            node_se_crop["r128_ceiling"][c] += np.bincount(nb[keep_c], weights=e2[keep_c], minlength=nb7)
            for s in seeds:
                e2s = (np.asarray(node_pred[s], np.float64)[:, j] - tgt[:, j]) ** 2
                node_se[f"transolver_seed{s}"][c] += np.bincount(nb[keep], weights=e2s[keep], minlength=nb7)
                node_se_crop[f"transolver_seed{s}"][c] += np.bincount(nb[keep_c], weights=e2s[keep_c], minlength=nb7)

        if (ci + 1) % 10 == 0:
            el = time.time() - t_start
            log(f"case {ci+1}/{len(test_pairs)}  {el:.0f}s  ({el/(ci+1):.2f}s/case)")

    # ---- gates -------------------------------------------------------------
    gates = {}
    ref_interp = {"mse_u": 0.7816026776, "mse_v": 0.03361523305, "mse_p": 75.03145038}
    got = {k: float(np.mean([m[k] for m in case_metrics["interp"]])) for k in ("mse_u", "mse_v", "mse_p")}
    g1 = {k: abs(got[k] - ref_interp[k]) / ref_interp[k] for k in ref_interp}
    gates["G1_interp_vs_published"] = {"got": got, "published": ref_interp, "rel": g1}
    for k, v in g1.items():
        assert v < 1e-9, f"G1 FAILED on {k}: {got[k]} vs {ref_interp[k]} (rel {v:.3e})"
    log(f"G1 PASS  interpolator reproduces interp_full.json: {got}")

    v2 = json.load(open("results/v2/v2_results.json", encoding="utf-8"))["backbone_per_seed"]
    g2 = {}
    for i, s in enumerate(seeds):
        arm = f"transolver_seed{s}"
        gotm = {k: float(np.mean([m[k] for m in case_metrics[arm]])) for k in ("mse_u", "mse_v", "mse_p")}
        pub = {k: float(v2[s][k]) for k in ("mse_u", "mse_v", "mse_p")}
        rel = {k: abs(gotm[k] - pub[k]) / pub[k] for k in pub}
        g2[arm] = {"got": gotm, "published": pub, "rel": rel}
        for k, v in rel.items():
            assert v < 2e-2, f"G2 FAILED {arm} {k}: {gotm[k]} vs {pub[k]} (rel {v:.3e})"
        log(f"G2 PASS  {arm} reproduces v2_results backbone_per_seed[{s}]: {gotm} (rel {rel})")
    gates["G2_transolver_vs_published"] = g2

    pub_bands = json.load(open(os.path.join(a.out_dir, f"interp_band_control_{a.task}.json"),
                               encoding="utf-8"))["A_band_decomposition"]
    tab5 = acc5["interp"].table(NAMES5)
    g3 = {}
    for bn, prow in pub_bands.items():
        for k, pv in prow.items():
            if k in ("n_cells",) or k not in tab5[bn]:
                continue
            gv = tab5[bn][k]
            rel = abs(gv - pv) / max(abs(pv), 1e-30)
            g3[f"{bn}/{k}"] = rel
            assert rel < 1e-6, f"G3 FAILED {bn}/{k}: {gv} vs {pv} (rel {rel:.3e})"
    gates["G3_band_accumulator_vs_published"] = {"max_rel": max(g3.values()), "n_checked": len(g3)}
    log(f"G3 PASS  band accumulator reproduces interp_band_control_full.json "
        f"({len(g3)} entries, max rel {max(g3.values()):.2e})")

    gates["G4_band_reconstructs_total"] = {
        f"{arm}/{grid}": gate_g4(acc, f"{arm}/{grid}")
        for grid, accs in (("5", acc5), ("7", acc7)) for arm, acc in accs.items()
    }
    log("G4 PASS  band tables reconstruct the pooled total for every arm")
    gates["G5"] = "geometry cache bitwise on 3 cases; reused Delaunay == rasterize_point_cloud on 3 cases x n_seeds"

    # ---- assemble ----------------------------------------------------------
    def case_mean(dicts):
        return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}

    out = {
        "meta": {
            "task": a.task, "resolution": a.resolution, "n_train": len(train_pairs),
            "n_test": len(test_pairs), "seeds": seeds, "rep": a.rep, "fill": a.fill,
            "cfg": cfg, "ckpt_dir": a.ckpt_dir, "device": str(device),
            "wallclock_sec": time.time() - t_start,
        },
        "gates": gates,
        "bands5": NAMES5, "bands7": NAMES7,
        "B_band_decomposition_5": {arm: acc5[arm].table(NAMES5) for arm in arms},
        "B_band_decomposition_7": {arm: acc7[arm].table(NAMES7) for arm in arms},
        "C_case_mean_area_uniform": {arm: case_mean(case_metrics[arm]) for arm in arms},
        "C_case_mean_node_weighted": {arm: case_mean(case_metrics_node[arm]) for arm in arms},
        "C_pooled": {arm: acc7[arm].pooled() for arm in arms},
        "D_node_space": {
            "bands": NAMES7,
            "n_nodes_full": node_n.tolist(),
            "n_nodes_in_crop": node_n_crop.tolist(),
            "mse_full": {k: {c: (node_se[k][c] / np.maximum(node_n, 1e-30)).tolist() for c in CHANS}
                         for k in node_se},
            "mse_in_crop": {k: {c: (node_se_crop[k][c] / np.maximum(node_n_crop, 1e-30)).tolist() for c in CHANS}
                            for k in node_se_crop},
        },
        "per_case_area_uniform": {arm: case_metrics[arm] for arm in arms},
        "per_case_node_weighted": {arm: case_metrics_node[arm] for arm in arms},
    }
    return out


def verdicts(full: dict) -> dict:
    """Apply D2 and D3 exactly as pre-registered."""
    arms = [k for k in full["B_band_decomposition_7"] if k.startswith("transolver")]
    t7 = full["B_band_decomposition_7"]
    out = {}

    # D2 -- per-band ratio, seed-mean Transolver band MSE
    ratios = {}
    for c in CHANS:
        rb = []
        for bn in NAMES7:
            mi = t7["interp"][bn][f"mse_{c}"]
            mt = float(np.mean([t7[arm][bn][f"mse_{c}"] for arm in arms]))
            rb.append(mi / mt if mt > 0 else float("nan"))
        ratios[c] = rb
    ru = np.array(ratios["u"], float)
    S = float(ru[0] / ru[-1])
    inversions = int(np.sum(np.diff(ru) > 0))
    if S >= 10 and inversions <= 1:
        v2 = "CONFIRMED ON BOTH ARMS"
    elif S < 3:
        v2 = "INTERPOLATOR-ONLY"
    else:
        v2 = "PARTIAL"
    out["D2"] = {"r_band_interp_over_transolver": ratios, "S_u_inner_over_outer": S,
                 "n_inversions_u": inversions, "verdict": v2,
                 "rule": "S>=10 and <=1 inversion -> CONFIRMED; S<3 -> INTERPOLATOR-ONLY; else PARTIAL"}

    # D3 -- the headline under both weightings
    au = full["C_case_mean_area_uniform"]
    nw = full["C_case_mean_node_weighted"]
    res = {}
    for c in CHANS:
        ti_a = float(np.mean([au[arm][f"mse_{c}"] for arm in arms]))
        ti_n = float(np.mean([nw[arm][f"mse_{c}"] for arm in arms]))
        res[c] = {
            "interp_area": au["interp"][f"mse_{c}"], "transolver_area": ti_a,
            "R_area": ti_a / au["interp"][f"mse_{c}"],
            "interp_node": nw["interp"][f"mse_{c}"], "transolver_node": ti_n,
            "R_node": ti_n / nw["interp"][f"mse_{c}"],
            "per_seed_R_area": {arm: au[arm][f"mse_{c}"] / au["interp"][f"mse_{c}"] for arm in arms},
            "per_seed_R_node": {arm: nw[arm][f"mse_{c}"] / nw["interp"][f"mse_{c}"] for arm in arms},
        }
    Rp = res["p"]["R_node"]
    v3 = ("SURVIVES" if Rp >= 4 else "SURVIVES SCOPED" if Rp >= 1.5
          else "WITHDRAW THE HEADLINE NUMBER" if Rp > 1.0 else "ARTIFACT")
    out["D3"] = {"ratios": res, "R_p_node": Rp, "verdict": v3,
                 "rule": ">=4 SURVIVES; >=1.5 SURVIVES SCOPED; >1.0 WITHDRAW; <=1.0 ARTIFACT",
                 "note": "R = mse(interp-arm-loses) convention: R>1 means the interpolator is "
                         "BETTER by that factor (Transolver MSE / interpolator MSE)."}
    return out


def stage_sensitivity(a) -> dict:
    """Amendment 2: the headline ratio under four discretisations of the node measure."""
    d = json.load(open(os.path.join(a.out_dir, "measure_asymmetry.json"), encoding="utf-8"))
    n = json.load(open(os.path.join(a.out_dir, "measure_asymmetry_nodes.json"), encoding="utf-8"))
    t = d["B_band_decomposition_7"]
    arms = [k for k in t if k.startswith("transolver")]
    masses = {
        "area_uniform": np.array(n["A1_area_frac_crop_7"]),
        "band_grid_binned_node": np.array([t["interp"][b]["node_weight_frac"] for b in NAMES7]),
        "band_true_node_crop": np.array(n["A2_node_frac_crop_7"]),
        "band_true_node_full_cloud": np.array(n["A3_node_frac_full_7"]),
    }
    out = {"bands": NAMES7, "band_masses": {k: v.tolist() for k, v in masses.items()},
           "R_convention": "R = MSE(Transolver, seed-mean) / MSE(interpolation); "
                           "R>1 means the interpolator is better by that factor",
           "R": {}}
    for c in CHANS:
        mi = np.array([t["interp"][b][f"mse_{c}"] for b in NAMES7])
        mt = np.array([np.mean([t[arm][b][f"mse_{c}"] for arm in arms]) for b in NAMES7])
        row = {k: float((mt @ w) / (mi @ w)) for k, w in masses.items()}
        row["cell_level_node_case_mean_PREREGISTERED"] = float(
            np.mean([d["C_case_mean_node_weighted"][arm][f"mse_{c}"] for arm in arms])
            / d["C_case_mean_node_weighted"]["interp"][f"mse_{c}"])
        row["cell_level_node_pooled"] = float(
            np.mean([d["C_pooled"][arm][f"mse_node_{c}"] for arm in arms])
            / d["C_pooled"]["interp"][f"mse_node_{c}"])
        row["cell_level_area_case_mean"] = float(
            np.mean([d["C_case_mean_area_uniform"][arm][f"mse_{c}"] for arm in arms])
            / d["C_case_mean_area_uniform"]["interp"][f"mse_{c}"])
        out["R"][c] = row
    node_keys = [k for k in out["R"]["p"] if k != "area_uniform" and "area" not in k]
    vals = [out["R"]["p"][k] for k in node_keys]
    out["p_node_measure_range"] = {"min": float(min(vals)), "max": float(max(vals)),
                                   "constructions": node_keys}
    out["verdict_note"] = (
        "D3 is read on cell_level_node_case_mean_PREREGISTERED and returns ARTIFACT. "
        "The coarser band-level constructions land higher; across every construction "
        "the p ratio falls from 8.4x to within a factor of ~2 of parity, and u and v "
        "are catastrophic for the interpolator under all of them.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="full", choices=["nodes", "full", "sensitivity"])
    ap.add_argument("--task", default="full")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--root", default="data")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--ckpt-dir", default="checkpoints/v2_transolver")
    ap.add_argument("--out-dir", default="results/interpolation")
    ap.add_argument("--rep", default="nd")
    ap.add_argument("--fill", default="nearest")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args(argv)

    t0 = time.time()
    if a.stage == "nodes":
        out = stage_nodes(a)
        name = a.out_name or "measure_asymmetry_nodes.json"
        log(f"D1 verdict: {out['D1_verdict']}  "
            f"inside 0.05c: area {out['D1_gap']['inside_0.05c']['area_frac_crop_A1']:.4f} "
            f"node(crop) {out['D1_gap']['inside_0.05c']['node_frac_crop_A2']:.4f} "
            f"node(full) {out['D1_gap']['inside_0.05c']['node_frac_full_A3']:.4f} "
            f"g={out['D1_gap']['inside_0.05c']['g_A2_over_A1']:.2f}")
    elif a.stage == "sensitivity":
        out = stage_sensitivity(a)
        name = a.out_name or "measure_asymmetry_sensitivity.json"
        for c in CHANS:
            log(f"R_{c}: " + "  ".join(f"{k}={v:.4f}" for k, v in out["R"][c].items()))
        log(f"p under node-measure constructions: "
            f"[{out['p_node_measure_range']['min']:.3f}, {out['p_node_measure_range']['max']:.3f}] "
            f"against 8.39 area-uniform")
    else:
        out = stage_full(a)
        out["verdicts"] = verdicts(out)
        name = a.out_name or "measure_asymmetry.json"
        log(f"D2 verdict: {out['verdicts']['D2']['verdict']}  "
            f"S_u={out['verdicts']['D2']['S_u_inner_over_outer']:.2f}")
        log(f"D3 verdict: {out['verdicts']['D3']['verdict']}  "
            f"R_p area={out['verdicts']['D3']['ratios']['p']['R_area']:.2f}  "
            f"R_p node={out['verdicts']['D3']['ratios']['p']['R_node']:.2f}")

    out["wallclock_sec"] = time.time() - t0
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    log(f"wrote {path} ({out['wallclock_sec']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
