"""Does the interpolation finding's STRUCTURE survive grid refinement?

``docs/paper/body.tex`` (``sec:limitations``) names this as the paper's single
most valuable follow-up and the obvious attack on ``sec:interp``:

    every interpolation number is scored on the 128^2 rasterised crop; the native
    AirfRANS point clouds resolve the boundary layer far better than 0.0234c
    cells, and our own decomposition says the entire remaining gap lives inside
    the first cell. A reviewer is entitled to ask whether the interpolation
    advantage survives at native resolution.

The full point-space head-to-head needs Transolver inference with a
``PointNormalizer`` fitted on 800 train clouds and the checkpoints do not store
the normaliser. This script answers the *separable half* of that objection:
**is the localisation of the interpolator's error a rasterisation artifact?**

It re-runs the parameter-interpolation baseline at 128^2, 256^2 and 512^2 on the
SAME split, with the SAME estimator, under the SAME protocol, and reports how the
per-channel errors, the wall-band decomposition and the domain fraction
reproduced without learning move under refinement.

===========================================================================
PRE-REGISTRATION -- written and committed BEFORE the ladder was run
===========================================================================

**What this CANNOT settle, stated first.** There is no Transolver row at 256^2 or
512^2. This ladder therefore does NOT answer "does the interpolation *advantage*
survive at native resolution" -- that remains a limitation and the point-space
head-to-head remains the follow-up. What it can settle is the narrower, and
separately attackable, claim that the error localisation (``92%`` of ``u`` error
inside ``0.02c``, ``R^2 >= 0.9996`` beyond ``0.05c``, ``99.5%`` of the domain
reproduced without learning) is an artifact of a grid whose cells (``0.0234c``)
are *wider than the band the claim is about*. Any verdict below is worded for
that narrower claim and must not be quoted for the wider one.

**Everything except resolution is frozen.**
  * Cases: identical name lists, in identical order, at every rung (asserted).
  * Estimator: the config CV-selected on the 128^2 run and published in
    ``results/interpolation/interp_full.json`` -- ``krr sigma=1.0 lam=1e-3
    w_U=0.25``, representation ``nd``, fill ``nearest``. It is NOT re-selected
    per rung.
  * The weight matrix ``W`` is a function of the case NAMES only
    (``build_weights`` never sees a field), so it is bit-identical at every rung.
    The script hashes ``W`` per rung and asserts equality -- the strongest
    possible "same estimator" guarantee.
  * CV *is* re-run per rung as a SECONDARY stability check (it costs seconds: it
    scores a 2048-cell subsample). Its selection is reported, never used.

**Protocol identity is gated on reproducing the published 128^2 row.** The 128
rung must return ``mse_u = 0.7816026775816814``, ``mse_v = 0.03361523305371703``,
``mse_p = 75.03145038384078``, ``surface_mse_p = 10988.622207526085`` and the
published band table (``se_share_u = 0.9238``, ``r2_u = 0.83952``,
``cell_frac = 0.0051`` in ``0-0.02c``) to 1e-9 relative on the aggregates, and to
the precision each band figure is PUBLISHED to (``cell_frac`` is quoted to two
significant figures, the rest to five). If it does not, the run ABORTS: every
256/512 number would be unanchored. This is the first row of the report.
[Amended after the first 128-only dry run, which is in the git log: the four
aggregate metrics and the band ``R^2``/SE shares matched at relative error
``0.00e+00``--``5e-06``, but the ``cell_frac`` check was written against the
two-significant-figure printout ``0.0051`` at a five-figure tolerance and failed
on the reproduced ``0.00506991``, which rounds to ``0.0051``. The tolerance, not
the protocol, was wrong; no threshold in the decision rule was touched.]

**One permitted optimisation, and why it is exact.** ``make_predict_fn`` rebuilds
each case's ``(sdf, mask)`` with ``signed_distance``/``solid_mask`` from
``case.geometry``; at 512^2 that is 25.7 s per case (86 min for 200 cases) and
dominates everything else. ``airfrans_loader._sim_to_pair`` builds the cached
``FlowField.sdf``/``.mask`` by calling *those same two functions on those same
two arguments*, so the cached arrays are the same object-by-value. The script
pre-populates ``parameter_interpolation_baseline._GEOM_CACHE`` from the cached
fields and PROVES the substitution is bit-identical by recomputing
``signed_distance``/``solid_mask`` from scratch on ``--geom-verify`` cases at
every rung and asserting exact array equality. The cache is cleared between
rungs (its key is the case name, which carries no resolution).

**Bands are fixed in PHYSICAL units.** ``scripts/floor_resolution_ladder.py``
learned this the hard way: a band defined in cells is physically thinner on a
finer grid, so a cell-defined band compares different regions across levels and
can manufacture a trend out of nothing. Every band edge here is in chord units of
the signed distance, identical at every rung. The band grid is imported verbatim
from ``scripts/interpolation_band_control.py``. Cell fraction is reported per
band per rung as the convergence witness: on a uniform crop cell fraction IS area
fraction, so the ``0-0.02c`` band's geometric area fraction (~0.0046 for a
perimeter of ~2.05c in a 3x3c crop minus the body) is the value it must approach;
its 128^2 value of 0.0051 is inflated because the band is then sub-cell.

**The interpolation floor -- the ladder's own limit (h/s).** AirfRANS truth is a
body-fitted unstructured cloud. Refining the raster past the source cloud's local
spacing ``s`` samples a piecewise-linear interpolant, not a solution.
``floor_resolution_ladder.py`` measured ``s`` on the same test split:
``s_far = 0.004595c``, ``s_wall = 4.43e-5c``. So

    h/s_far   = 5.10 / 2.55 / 1.27   at 128 / 256 / 512
    h/s_wall  =  529 /  265 /  132   at 128 / 256 / 512

The caveat binds ONLY in the far field, and only at 512, where ``h/s_far`` is
1.27. It never binds where this finding lives: the raster is 132x coarser than
the source mesh in the near-wall band even at 512^2. Consequence, pre-declared:
**the outer-band absolute-MSE trend (the second arm of the verdict) is read on
the 128->256 pair**, where both rungs have ``h/s_far >= 2.5``; the 512 value is
reported and FLAGGED. As a de-confounder rather than a caveat, outer-band stats
are additionally reported restricted to cells holding ``>= --min-points`` source
points (``floor_resolution_ladder.oversampled_cells``), where the rasteriser is
averaging rather than interpolating.

**The domain-fraction metric, pinned to the sentence the paper already prints.**
``sec:interp`` says the interpolator "reproduces every channel at
``R^2 >= 0.9996``" beyond ``0.05c`` and that this is ``99.5%`` of the domain. That
``99.5%`` is ``1 - 0.0051``, the complement of the ``0-0.02c`` band, i.e. the
cumulative cell fraction of every band beyond ``0.02c``; at 128 every such band
clears POOLED ``R^2 = 0.99``. So the headline definition is

    F_band(tau) := the cumulative fluid-cell fraction of the outermost run of
                   bands, every one of which has min over {u,v,p} of POOLED R^2
                   at or above tau,

with **tau = 0.99** as the headline, which must reproduce ``0.995`` at 128^2
(checked, and part of the protocol gate). Reported at tau in
{0.9, 0.99, 0.999, 0.9996}, under BOTH the pooled ``R^2`` and the strict
per-case-centred ``R^2_pc``, plus a finer band grid so the cut is better
localised than the paper's five bands allow. Cell fraction = area fraction here
because the crop is uniform, so the number is comparable across rungs; that is
stated rather than assumed.

**R^2 alone cannot settle this, so the verdict reads a triple.** R^2's
denominator moves between rungs -- the near-wall truth's variance grows as the
layer resolves -- so R^2 can improve while absolute MSE grows. The aggregate
``mse_u`` is in fact EXPECTED to rise with refinement: 512^2 exposes near-wall
structure that had no representation at 128^2. That is localisation *sharpening*,
not dissolving. The decision therefore runs on:

    S_c(res)     := SE share of channel c in the 0-0.02c band          (c = u, v)
    M_out(res,c) := absolute MSE of channel c over cells with sdf > 0.05c
    F(res)       := F_band(0.99) under pooled R^2

**DECISION RULE.**
  HOLDS      := S_u >= 0.85 AND S_v >= 0.85 at EVERY rung
                AND M_out(256,c) <= 2 * M_out(128,c) for every c in {u,v,p}
                AND F >= 0.99 at every rung.
  SHARPENS   := HOLDS, and additionally S_u, S_v and F are non-decreasing in res.
  DISSOLVES  := S_u < 0.70 at 512^2
                OR M_out(256,c) > 2 * M_out(128,c) for any c
                OR F < 0.98 at any rung.
  PARTIAL    := anything else; the failing component is named.

Thresholds, justified before seeing the numbers: 0.85 is "still the overwhelming
majority" against the published 0.92/0.90; 0.70 is "no longer the dominant term";
the 2x factor on outer MSE is the same factor
``parameter_interpolation_baseline.py`` already pre-registered in its P3 verdict;
0.99/0.98 bracket the published 0.995.

**What each outcome licenses.** HOLDS/SHARPENS: the "you only won because of a
coarse raster" objection fails *as an explanation of the localisation*, the
limitations paragraph loses its strongest sub-claim, and the paper gains a
defensive table. DISSOLVES: the localisation is partly a rasterisation effect,
``sec:interp``'s band paragraph must be scoped to the deployed resolution, and
the paper says so before submission. Both are reported plainly.

**ADDED AFTER THE 128/256 STAGE, in response to what those two rungs showed.**
These are ADDITIONAL diagnostics; no threshold and no clause of the decision rule
above was altered or re-tuned, and the 128^2 gate is untouched.
  (a) ``surface_pressure_block``. ``surface_mse_p`` moved 10989 -> 93105 from
      128^2 to 256^2. That is an ABSOLUTE number in the one channel whose
      resolvable structure changes most with ``h``: at 128^2 the wall band is
      sub-cell, so the bilinear sampler smooths the TRUTH as well as the
      prediction. The variance of the sampled ground truth is therefore reported
      at every rung next to the MSE, together with their ratio, so "the
      interpolator got worse" can be told apart from "the metric started seeing
      structure". The ratio is the resolution-comparable statistic; the absolute
      MSE is the one the paper's tables print.
  (b) ``sel_sdf_mean`` and ``frac_of_outer_selected`` on the oversampled-cell
      control. That restriction is NOT composition-neutral: the source cloud is
      densest near the body, so as ``h`` shrinks the qualifying cells retreat
      toward the wall (362160 -> 267304 cells from 128 to 256). Its MSE trend
      therefore mixes resolution with region and must not be read as the
      far-field trend; the pre-registered outer-MSE arm uses ``outer_mse_all``,
      as written above, and these two fields make the confound measurable
      instead of arguable.

**Forces are not scored.** ``evaluate_cases`` is called (so aggregation, masking
and the ``n_cases`` assert are the shared code path), which includes
``force_coefficients``; ``rho_cl``/``rho_cd`` are recorded but are NOT part of
any pre-registered reading here, because the question is about where field error
lives.

===========================================================================
RUNNING  (CPU-only, no GPU, no training)
===========================================================================
    .venv/Scripts/python.exe scripts/interpolation_resolution_ladder.py --levels 128
    .venv/Scripts/python.exe scripts/interpolation_resolution_ladder.py \
        --levels 128 256 512

Needs ``data/cache/airfrans_full_{train_n800,test_n200}_r{128,256,512}.pkl``.
Writes ``results/interpolation/interp_resolution_ladder.json``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time

import neuroforge  # noqa: F401  -- MUST precede numpy (BLAS thread caps)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interpolation_band_control as ibc  # noqa: E402
import parameter_interpolation_baseline as pib  # noqa: E402
from floor_resolution_ladder import oversampled_cells, source_positions  # noqa: E402
from interpolation_band_control import band_decomposition  # noqa: E402
from parameter_interpolation_baseline import (  # noqa: E402
    build_stack,
    build_weights,
    case_features,
    cfg_name,
    cv_select,
    evaluate_cases,
    load_pairs,
    make_predict_fn,
    redimensionalise,
)

CROP_WIDTH = 3.0

# Source-cloud nearest-neighbour spacing on this split, measured by
# scripts/floor_resolution_ladder.py (results/floor_resolution_ladder.json).
SOURCE_SPACING = {"far_field": 0.004594571575883354, "near_wall": 4.4308898806786156e-05}

# The published 128^2 row that gates protocol identity (results/interpolation/
# interp_full.json and interp_band_control_full.json; docs/paper/body.tex
# tab:interp, tab:interp_bands).
GATE_METRICS = {
    "mse_u": 0.7816026775816814,
    "mse_v": 0.03361523305371703,
    "mse_p": 75.03145038384078,
    "surface_mse_p": 10988.622207526085,
}
# Each is compared at the precision it is PUBLISHED to: ``cell_frac`` is quoted
# to two significant figures (0.0051 in interp_band_control_full.json's printout,
# 0.005 in tab:interp_bands), the rest to five.
GATE_BAND_0_002 = {"cell_frac": (0.0051, 1e-2), "r2_u": (0.83952, 1e-4),
                   "se_share_u": (0.9238, 1e-4), "se_share_v": (0.8979, 1e-4)}
GATE_F_BAND_099 = 0.995

# Cumulative-region edges for the finer domain-fraction curve (chord units).
FINE_EDGES = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.25, 0.50]

TAUS = [0.9, 0.99, 0.999, 0.9996]
OUTER_EDGE = 0.05          # M_out is scored on sdf > OUTER_EDGE


# ---------------------------------------------------------------------------
# Geometry short-circuit (exact; proven per rung -- see the pre-registration)
# ---------------------------------------------------------------------------
def prime_geom_cache(pairs, n_verify: int) -> dict:
    """Fill ``pib._GEOM_CACHE`` from the cached fields and PROVE it is exact.

    ``_sim_to_pair`` set ``field.sdf = signed_distance(geom, domain)`` and
    ``field.mask = solid_mask(geom, domain)``. ``pib._geometry`` recomputes those
    same two calls. Substituting the stored arrays is therefore a memoisation,
    not an approximation -- but it is only worth doing if it is checked, so this
    recomputes from scratch on the first ``n_verify`` cases and asserts exact
    array equality (bitwise, via ``array_equal`` on identical dtypes).
    """
    from neuroforge.geometry.sdf import signed_distance, solid_mask

    pib._GEOM_CACHE.clear()
    checked = []
    for i, (case, fld) in enumerate(pairs):
        sdf = np.asarray(fld.sdf)
        mask = np.asarray(fld.mask)
        if i < n_verify:
            ref_sdf = signed_distance(case.geometry, case.domain)
            ref_mask = solid_mask(case.geometry, case.domain)
            assert ref_sdf.dtype == sdf.dtype, (ref_sdf.dtype, sdf.dtype)
            assert ref_mask.dtype == mask.dtype, (ref_mask.dtype, mask.dtype)
            assert np.array_equal(ref_sdf, sdf), f"sdf mismatch on {case.name}"
            assert np.array_equal(ref_mask, mask), f"mask mismatch on {case.name}"
            checked.append(case.name)
        pib._GEOM_CACHE[case.name] = (sdf, mask)
    return {"n_verified_bitwise": len(checked), "verified_cases": checked}


# ---------------------------------------------------------------------------
# Region statistics -- the SAME estimator as the paper's band table
# ---------------------------------------------------------------------------
def region_decomposition(pred_phys, test_pairs, hw, edges_lo, edges_hi, names):
    """Run ``band_decomposition`` over an arbitrary region list.

    ``band_decomposition`` reads its band list from module globals, so they are
    swapped and restored around the call. This is deliberate REUSE: the R^2,
    R^2_pc and MSE arithmetic is byte-for-byte the code that produced
    ``tab:interp_bands``, which is what makes the 128^2 gate meaningful.
    """
    old_b, old_n = ibc.BANDS, ibc.BAND_NAMES
    try:
        ibc.BANDS = list(zip(edges_lo, edges_hi))
        ibc.BAND_NAMES = list(names)
        return band_decomposition(pred_phys, test_pairs, hw)
    finally:
        ibc.BANDS, ibc.BAND_NAMES = old_b, old_n


def f_band(bands: dict, order: list[str], fracs: list[float], tau: float,
           key: str) -> dict:
    """``F_band(tau)``: cumulative cell fraction of the outermost run of bands
    all of which have ``min over {u,v,p} R^2 >= tau``.

    ``order`` is inner-to-outer. Scanning from the outside in, the cut is the
    innermost band from which every band outward clears ``tau``.
    """
    ok = []
    for b in order:
        r = bands.get(b)
        if r is None:
            ok.append(False)
            continue
        vals = [r[f"{key}_{c}"] for c in ("u", "v", "p")]
        ok.append(bool(np.nanmin(vals) >= tau))
    cut = len(order)
    for i in range(len(order) - 1, -1, -1):
        if ok[i]:
            cut = i
        else:
            break
    frac = float(sum(fracs[i] for i in range(cut, len(order))))
    return {"tau": tau, "cut_band_index": cut,
            "cut_band": order[cut] if cut < len(order) else None,
            "domain_fraction": frac,
            "per_band_clears_tau": {b: bool(o) for b, o in zip(order, ok)}}


def outer_mse(pred_phys, test_pairs, hw, edge: float, over_masks=None) -> dict:
    """Absolute MSE over cells with ``sdf > edge``, optionally restricted to
    cells the rasteriser AVERAGED (>= min_points source points).

    The wall-distance summary of the SELECTED set is returned alongside, because
    the oversampled restriction is NOT composition-neutral across rungs: the
    source cloud is densest near the body, so as ``h`` shrinks the cells that
    still hold >= min_points points retreat toward the wall. Comparing that
    subset's MSE across rungs therefore mixes a resolution effect with a region
    change, and the numbers must be read with ``sel_sdf_mean`` in view.
    """
    chans = ("u", "v", "p")
    se = {c: 0.0 for c in chans}
    n = 0
    n_fluid_outer = 0
    sdf_sum = 0.0
    for i, (case, ref) in enumerate(test_pairs):
        fluid = np.asarray(ref.mask).ravel() > 0.5
        d = np.asarray(ref.sdf, np.float64).ravel()
        base = fluid & (d > edge)
        n_fluid_outer += int(base.sum())
        m = base
        if over_masks is not None:
            om = over_masks.get(case.name)
            if om is None:
                continue
            m = base & om.ravel()
        if not m.any():
            continue
        row = pred_phys[i]
        gt = {"u": np.asarray(ref.u, np.float64).ravel(),
              "v": np.asarray(ref.v, np.float64).ravel(),
              "p": np.asarray(ref.p, np.float64).ravel()}
        pr = {"u": row[0:hw], "v": row[hw:2 * hw], "p": row[2 * hw:3 * hw]}
        n += int(m.sum())
        sdf_sum += float(d[m].sum())
        for c in chans:
            se[c] += float(((pr[c][m] - gt[c][m]) ** 2).sum())
    if n == 0:
        return {"n_cells": 0}
    return {"n_cells": n, "n_fluid_outer": n_fluid_outer,
            "frac_of_outer_selected": n / max(n_fluid_outer, 1),
            "sel_sdf_mean": sdf_sum / n,
            **{f"mse_{c}": se[c] / n for c in chans}}


def surface_pressure_block(pred_phys, test_pairs, hw) -> dict:
    """Surface pressure MSE AND the variance of the sampled ground truth.

    ``surface_mse_p`` is an absolute number in a channel whose resolvable
    structure changes with ``h``: at 128^2 the wall band is sub-cell, so the
    bilinear sampler smooths BOTH prediction and truth. A bare MSE comparison
    across rungs therefore cannot separate "the interpolator got worse" from
    "the metric started seeing structure". The same MSE divided by the variance
    of the SAMPLED TRUTH is the resolution-comparable version, and both are
    reported together. Pooled across all cases' surface points.
    """
    from neuroforge.physics.evaluation import _surface_points_normals
    from neuroforge.physics.metrics import _bilinear_sample

    se = 0.0
    s1 = 0.0
    s2 = 0.0
    n = 0
    per_case = []
    H = test_pairs[0][1].u.shape[0]
    for i, (case, ref) in enumerate(test_pairs):
        pts, _ = _surface_points_normals(case)
        if pts.shape[0] < 2:
            continue
        p_pred = _bilinear_sample(
            pred_phys[i][2 * hw:3 * hw].reshape(H, -1).astype(np.float32),
            ref.domain, pts)
        p_ref = _bilinear_sample(np.asarray(ref.p), ref.domain, pts)
        e = np.asarray(p_pred, np.float64) - np.asarray(p_ref, np.float64)
        g = np.asarray(p_ref, np.float64)
        se += float((e ** 2).sum())
        s1 += float(g.sum())
        s2 += float((g ** 2).sum())
        n += int(g.size)
        per_case.append(float(np.mean(e ** 2)))
    if n == 0:
        return {"n_points": 0}
    mse = se / n
    var = s2 / n - (s1 / n) ** 2
    return {"n_points": n, "pooled_mse": mse, "gt_variance": var,
            "standardised": mse / var if var > 0 else float("nan"),
            "case_mean_mse": float(np.mean(per_case))}


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def check_gate(agg: dict, bands: dict, f099: float, rtol: float) -> dict:
    """Protocol identity against the published 128^2 row. Aborts the run on
    failure: unanchored 256/512 numbers are worse than no numbers."""
    rows = []
    ok = True
    for k, want in GATE_METRICS.items():
        got = float(agg[k])
        rel = abs(got - want) / abs(want)
        rows.append({"key": k, "published": want, "reproduced": got, "rel": rel,
                     "pass": bool(rel <= rtol)})
        ok &= rel <= rtol
    b = bands["0-0.02c"]
    for k, (want, tol) in GATE_BAND_0_002.items():
        got = float(b[k])
        rel = abs(got - want) / abs(want)
        good = bool(rel <= tol)
        rows.append({"key": f"band_0-0.02c.{k}", "published": want, "tol": tol,
                     "reproduced": got, "rel": rel, "pass": good})
        ok &= good
    rel_f = abs(f099 - GATE_F_BAND_099) / GATE_F_BAND_099
    rows.append({"key": "F_band(0.99)", "published": GATE_F_BAND_099,
                 "reproduced": f099, "rel": rel_f, "pass": bool(rel_f <= 1e-3)})
    ok &= rel_f <= 1e-3
    return {"pass": bool(ok), "rows": rows}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def decide(per_level: dict) -> dict:
    levels = sorted(per_level)
    S_u = [per_level[r]["bands"]["0-0.02c"]["se_share_u"] for r in levels]
    S_v = [per_level[r]["bands"]["0-0.02c"]["se_share_v"] for r in levels]
    F = [per_level[r]["domain_fraction"]["pooled"]["0.99"]["domain_fraction"]
         for r in levels]
    out = {"levels": levels, "S_u": S_u, "S_v": S_v, "F_band_0.99": F,
           "failures": []}

    share_ok = all(s >= 0.85 for s in S_u) and all(s >= 0.85 for s in S_v)
    if not share_ok:
        out["failures"].append("SE share in 0-0.02c fell below 0.85 at some rung")
    f_ok = all(f >= 0.99 for f in F)
    if not f_ok:
        out["failures"].append("F_band(0.99) fell below 0.99 at some rung")

    growth = None
    if 128 in per_level and 256 in per_level:
        a = per_level[128]["outer_mse_all"]
        b = per_level[256]["outer_mse_all"]
        growth = {c: b[f"mse_{c}"] / a[f"mse_{c}"] for c in ("u", "v", "p")}
        if any(g > 2.0 for g in growth.values()):
            out["failures"].append(
                f"outer-band MSE grew >2x from 128 to 256: {growth}")
    out["outer_mse_growth_128_to_256"] = growth

    dissolves = (
        (512 in per_level
         and per_level[512]["bands"]["0-0.02c"]["se_share_u"] < 0.70)
        or (growth is not None and any(g > 2.0 for g in growth.values()))
        or any(f < 0.98 for f in F)
    )
    holds = share_ok and f_ok and (growth is None
                                   or all(g <= 2.0 for g in growth.values()))
    if dissolves:
        v = "LOCALISATION DISSOLVES"
    elif holds:
        mono = (all(S_u[i] <= S_u[i + 1] + 1e-12 for i in range(len(S_u) - 1))
                and all(S_v[i] <= S_v[i + 1] + 1e-12 for i in range(len(S_v) - 1))
                and all(F[i] <= F[i + 1] + 1e-12 for i in range(len(F) - 1)))
        v = "LOCALISATION SHARPENS" if mono else "LOCALISATION HOLDS"
    else:
        v = "PARTIAL"
    out["verdict"] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--levels", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--task", default="full")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--rep", default="nd")
    ap.add_argument("--fill", default="nearest")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--cfg-json", default="results/interpolation/interp_full.json")
    ap.add_argument("--out-dir", default="results/interpolation")
    ap.add_argument("--out-name", default="interp_resolution_ladder.json")
    ap.add_argument("--geom-verify", type=int, default=3,
                    help="cases per rung whose sdf/mask are recomputed from "
                         "scratch and asserted bit-identical to the cached ones")
    ap.add_argument("--gate-rtol", type=float, default=1e-9)
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the 128 protocol gate (diagnostics only)")
    ap.add_argument("--min-points", type=int, default=8,
                    help="source points a raster cell needs to count as averaged")
    ap.add_argument("--no-oversampled", action="store_true")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cv-cells", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = json.load(open(a.cfg_json, encoding="utf-8"))["variants"][a.rep][
        "cv_selected_cfg_raw"]
    print(f"[ladder] frozen estimator from {a.cfg_json}: {cfg_name(cfg)} {cfg}",
          flush=True)
    print(f"[ladder] levels {a.levels} | rep={a.rep} fill={a.fill} | "
          f"bands fixed in CHORD units", flush=True)

    # Source clouds, once: resolution-independent, used for the h/s de-confounder.
    positions = None
    if not a.no_oversampled:
        t0 = time.time()
        try:
            positions = source_positions("data", a.task, a.n_test)
            print(f"[ladder] source clouds for {len(positions)} cases in "
                  f"{time.time() - t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"[ladder] ! source clouds unavailable ({type(exc).__name__}: "
                  f"{exc}); the oversampled-cell control is skipped", flush=True)
            positions = None

    fine_lo = FINE_EDGES
    fine_hi = FINE_EDGES[1:] + [1e9]
    fine_names = [f"{lo:g}-{hi:g}c" if hi < 1e8 else f">{lo:g}c"
                  for lo, hi in zip(fine_lo, fine_hi)]

    out = {
        "task": a.task, "levels": a.levels, "rep": a.rep, "fill": a.fill,
        "n_train": a.n_train, "n_test": a.n_test,
        "frozen_cfg": cfg, "frozen_cfg_name": cfg_name(cfg),
        "cfg_source": a.cfg_json,
        "bands_chord": {"paper": list(zip(ibc.BAND_NAMES,
                                          [list(b) for b in ibc.BANDS])),
                        "fine": list(zip(fine_names,
                                         [[lo, hi] for lo, hi in
                                          zip(fine_lo, fine_hi)]))},
        "source_spacing": SOURCE_SPACING,
        "outer_edge_chord": OUTER_EDGE,
        "taus": TAUS,
        "per_level": {},
    }

    names_ref = None
    w_hash_ref = None
    for res in a.levels:
        t_rung = time.time()
        print(f"\n[ladder] ===== {res}^2 =====", flush=True)
        train_pairs = load_pairs(a.cache_dir, a.task, True, res, a.n_train)
        test_pairs = load_pairs(a.cache_dir, a.task, False, res, a.n_test)
        names_tr = [c.name for c, _ in train_pairs]
        names_te = [c.name for c, _ in test_pairs]
        if names_ref is None:
            names_ref = (names_tr, names_te)
        else:
            assert names_tr == names_ref[0], f"train case list differs at {res}"
            assert names_te == names_ref[1], f"test case list differs at {res}"
        assert not (set(names_tr) & set(names_te)), "train/test name overlap"
        H, W_ = test_pairs[0][1].u.shape
        assert (H, W_) == (res, res), (H, W_, res)
        hw = H * W_
        h = CROP_WIDTH / res

        Xtr = np.stack([case_features(n) for n in names_tr])
        Xte = np.stack([case_features(n) for n in names_te])

        t0 = time.time()
        Ytr, Mtr, _ = build_stack(train_pairs, a.rep, a.fill)
        t_stack = time.time() - t0
        print(f"[ladder] train stack {Ytr.shape} in {t_stack:.0f}s", flush=True)

        Wm = build_weights(cfg, Xtr, Xte)
        w_hash = hashlib.sha256(np.ascontiguousarray(Wm).tobytes()).hexdigest()
        if w_hash_ref is None:
            w_hash_ref = w_hash
        assert w_hash == w_hash_ref, (
            "the weight matrix changed between rungs; the estimator is not frozen")
        print(f"[ladder] W sha256 {w_hash[:16]} (identical across rungs)",
              flush=True)

        t0 = time.time()
        pred_phys = redimensionalise(Wm @ Ytr, names_te, a.rep, hw)
        t_gemm = time.time() - t0

        # Secondary: CV stability at this rung (reported, never used).
        t0 = time.time()
        best_here, cv_table = cv_select(Xtr, Ytr, Mtr, names_tr, a.rep, hw,
                                        a.n_folds, a.cv_cells, a.seed,
                                        verbose=False)
        t_cv = time.time() - t0
        print(f"[ladder] CV at this rung would select {cfg_name(best_here)} "
              f"(S={cv_table[0]['S']:.5f}); NOT used  [{t_cv:.0f}s]", flush=True)

        del Ytr, Mtr, train_pairs
        gc.collect()

        geom_proof = prime_geom_cache(test_pairs, a.geom_verify)
        print(f"[ladder] geometry memo primed; {geom_proof['n_verified_bitwise']} "
              f"cases re-derived from scratch and bit-identical", flush=True)

        t0 = time.time()
        pfn = make_predict_fn(pred_phys, names_te, (H, W_))
        agg = evaluate_cases(pfn, test_pairs)
        t_eval = time.time() - t0
        assert agg["n_cases"] == len(test_pairs), agg["n_cases"]

        t0 = time.time()
        bands = band_decomposition(pred_phys, test_pairs, hw)
        fine = region_decomposition(pred_phys, test_pairs, hw, fine_lo, fine_hi,
                                    fine_names)
        t_band = time.time() - t0

        band_order = list(ibc.BAND_NAMES)
        band_fracs = [bands[b]["cell_frac"] for b in band_order]
        fine_order = [n for n in fine_names if n in fine]
        fine_fracs = [fine[n]["cell_frac"] for n in fine_order]
        dom = {"pooled": {}, "per_case_centred": {},
               "pooled_fine": {}, "per_case_centred_fine": {}}
        for tau in TAUS:
            dom["pooled"][str(tau)] = f_band(bands, band_order, band_fracs, tau, "r2")
            dom["per_case_centred"][str(tau)] = f_band(bands, band_order,
                                                       band_fracs, tau, "r2_pc")
            dom["pooled_fine"][str(tau)] = f_band(fine, fine_order, fine_fracs,
                                                  tau, "r2")
            dom["per_case_centred_fine"][str(tau)] = f_band(fine, fine_order,
                                                            fine_fracs, tau, "r2_pc")

        over_masks = None
        over_stats = {"available": False}
        if positions:
            over_masks = {}
            n_over, n_tot = 0, 0
            for case, ref in test_pairs:
                pz = positions.get(case.name)
                if pz is None:
                    continue
                om = oversampled_cells(ref, pz[0], a.min_points)
                over_masks[case.name] = om
                n_over += int(om.sum())
                n_tot += om.size
            over_stats = {"available": True, "min_points": a.min_points,
                          "frac_cells_oversampled": n_over / max(n_tot, 1),
                          "n_cases": len(over_masks)}

        row = {
            "resolution": res, "h_chord": h,
            "h_over_s_far": h / SOURCE_SPACING["far_field"],
            "h_over_s_wall": h / SOURCE_SPACING["near_wall"],
            "h_over_s_far_flag": "AT/PAST LIMIT" if h < SOURCE_SPACING["far_field"]
            else ("NEAR LIMIT" if h < 2.0 * SOURCE_SPACING["far_field"] else "OK"),
            "w_sha256": w_hash,
            "geometry_memo_proof": geom_proof,
            "metrics": {k: agg[k] for k in sorted(agg)},
            "bands": bands,
            "fine_regions": fine,
            "domain_fraction": dom,
            "outer_mse_all": outer_mse(pred_phys, test_pairs, hw, OUTER_EDGE),
            "outer_mse_oversampled": (
                outer_mse(pred_phys, test_pairs, hw, OUTER_EDGE, over_masks)
                if over_masks else {"n_cells": 0}),
            "surface_pressure": surface_pressure_block(pred_phys, test_pairs, hw),
            "oversampled": over_stats,
            "cv_would_select": cfg_name(best_here),
            "cv_top5": cv_table[:5],
            "timing_sec": {"stack": t_stack, "gemm": t_gemm, "cv": t_cv,
                           "evaluate_cases": t_eval, "bands": t_band,
                           "rung_total": time.time() - t_rung},
        }
        out["per_level"][str(res)] = row

        print(f"[ladder] {res}^2 h={h:.5f} h/s_far={row['h_over_s_far']:.2f} "
              f"({row['h_over_s_far_flag']}) h/s_wall={row['h_over_s_wall']:.0f}",
              flush=True)
        print(f"[ladder] mse_u={agg['mse_u']:.5g} mse_v={agg['mse_v']:.5g} "
              f"mse_p={agg['mse_p']:.6g} surf_p={agg['surface_mse_p']:.6g} "
              f"mse_nut={agg.get('mse_nut', float('nan')):.4g}", flush=True)
        for b in band_order:
            r = bands[b]
            print(f"    {b:12s} frac={r['cell_frac']:.4f} "
                  f"R2 {r['r2_u']:.5f}/{r['r2_v']:.5f}/{r['r2_p']:.5f} "
                  f"R2pc {r['r2_pc_u']:.4f}/{r['r2_pc_v']:.4f}/{r['r2_pc_p']:.4f} "
                  f"share {r['se_share_u']:.4f}/{r['se_share_v']:.4f}/"
                  f"{r['se_share_p']:.4f}", flush=True)
        print(f"    F_band(0.99) pooled = "
              f"{dom['pooled']['0.99']['domain_fraction']:.4f}  "
              f"(cut {dom['pooled']['0.99']['cut_band']})", flush=True)
        om_all = row["outer_mse_all"]
        print(f"    outer (sdf>{OUTER_EDGE}c) mse u/v/p = {om_all['mse_u']:.5g}/"
              f"{om_all['mse_v']:.5g}/{om_all['mse_p']:.6g}", flush=True)
        if row["outer_mse_oversampled"]["n_cells"]:
            oo = row["outer_mse_oversampled"]
            print(f"    outer, >= {a.min_points} src pts/cell "
                  f"({oo['n_cells']} cells, {oo['frac_of_outer_selected']:.3f} of "
                  f"outer, mean sdf {oo['sel_sdf_mean']:.4f}c) u/v/p = "
                  f"{oo['mse_u']:.5g}/{oo['mse_v']:.5g}/{oo['mse_p']:.6g}",
                  flush=True)
        sp = row["surface_pressure"]
        print(f"    surface p: pooled MSE {sp['pooled_mse']:.6g}  GT var "
              f"{sp['gt_variance']:.6g}  standardised {sp['standardised']:.6g}",
              flush=True)

        if res == 128 and not a.no_gate:
            gate = check_gate(agg, bands,
                              dom["pooled"]["0.99"]["domain_fraction"],
                              a.gate_rtol)
            out["protocol_gate_128"] = gate
            for r in gate["rows"]:
                print(f"    [gate] {r['key']:22s} published {r['published']:.10g} "
                      f"reproduced {r['reproduced']:.10g} rel {r['rel']:.2e} "
                      f"{'PASS' if r['pass'] else 'FAIL'}", flush=True)
            if not gate["pass"]:
                out["aborted"] = "128^2 protocol gate FAILED"
                path = os.path.join(a.out_dir, a.out_name)
                with open(path, "w", encoding="utf-8", newline="\n") as fh:
                    json.dump(out, fh, indent=2)
                    fh.write("\n")
                print(f"[ladder] ABORT: protocol gate failed; wrote {path}",
                      flush=True)
                return 1
            print("[ladder] protocol gate PASSED -- the 128^2 row is the "
                  "published one", flush=True)

        del pred_phys, test_pairs, pfn, over_masks
        gc.collect()

    lv = {int(k): v for k, v in out["per_level"].items()}
    out["decision"] = decide(lv)
    print(f"\n[ladder] VERDICT: {out['decision']['verdict']}", flush=True)
    for f in out["decision"]["failures"]:
        print(f"    failed component: {f}", flush=True)
    print(f"    S_u {out['decision']['S_u']}", flush=True)
    print(f"    S_v {out['decision']['S_v']}", flush=True)
    print(f"    F   {out['decision']['F_band_0.99']}", flush=True)
    print(f"    outer MSE growth 128->256 "
          f"{out['decision']['outer_mse_growth_128_to_256']}", flush=True)

    out["wallclock_sec"] = time.time() - t_start
    path = os.path.join(a.out_dir, a.out_name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"[ladder] wrote {path} ({out['wallclock_sec']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
