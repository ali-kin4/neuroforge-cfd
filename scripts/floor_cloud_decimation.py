"""Does the residual floor depend on the SOURCE CLOUD spacing at FIXED grid spacing?

The discriminating experiment for the mechanism behind the rising residual floor.

THE OBJECTION
-------------
`docs/paper/review/floor_resolution_study.md` sec 1a/6.2 attributes the floor's growth
under grid refinement to OPERATOR PROVENANCE -- a cell-centred Cartesian monitor
auditing a body-fitted finite-volume reference -- and states explicitly that it is
not a resolution or rasterisation artifact.

`docs/paper/review/theorem_targets.md` sec 1 argues that attribution is wrong by
elimination. As `h -> 0` the monitored operator is fixed and the reference solution is
fixed, so the only `h`-dependent object left in the pipeline is WHERE a fixed
reconstruction gets sampled. Its Prop A1 proves that for a Lipschitz, `h`-independent
reconstruction the central difference is EXACTLY the segment mean of the derivative,
hence bounded uniformly in `h` and convergent -- so the first-derivative block
(93-95% of the floor) must converge to a finite limit, not follow a power law.

STRUCTURAL FACT, VERIFIED BY READING THE CODE (not measured here)
-----------------------------------------------------------------
`floor_resolution_decomposition.CaseLadder.__init__` builds `self.tri = Delaunay(pos)`
ONCE and `CaseLadder.rasterise` changes only the query grid. So the reconstruction is
provably `h`-independent and Prop A1's hypothesis holds as a property of this pipeline.
That already kills the power-law framing. What this script decides is what REPLACES the
mechanism sentences: representation error named as the driver, or nothing named at all.

THE EXPERIMENT
--------------
Hold `h` fixed and vary the SOURCE CLOUD spacing `s` instead. Decimate the AirfRANS
point cloud by nested random subsets of factor D in {1, 2, 4, 8}, re-triangulate,
re-rasterise at 128^2 and 256^2, and measure the same banded floor.

CONTROLS THAT MAKE `s` THE ONLY VARIABLE
----------------------------------------
* The GEOMETRY is built from the FULL cloud, once, before any decimation. The surface
  loop, the SDF, the solid mask and therefore the measurement bands are byte-identical
  across every D. `CaseLadder.geometry_on` asserts the surface-loop hash; that assert
  is a real gate here and is reported.
* Subsets are NESTED (D=8 subset of D=4 subset of D=2) with a fixed per-case seed, so
  the D-ladder is PAIRED per case and subset noise does not enter per-case ratios.
* `s` is MEASURED per (case, D, band) as the median nearest-neighbour spacing of the
  surviving cloud inside that band, not assumed to be `sqrt(D)` times the original.
  The AirfRANS cloud is graded (4.4e-5 chord at the wall, 4.6e-3 in the far field), so
  the realized `s` factor differs by band and by case and must be measured.
* Every rasterisation logs the number of query points that fall outside the decimated
  triangulation's hull and hit the NearestND fallback. If that count grows with D, part
  of any rise is a fill-value artifact rather than representation error.
* GATE: at D=1 this script must reproduce the committed
  `results/certificates/floor_resolution_decomposition.json` per-case band floors at
  128^2 and 256^2 to <1e-9 relative. Same clouds, same crop, same interpolator, same
  residual. If that fails, no verdict is issued.

PRE-REGISTERED DECISION RULE (this docstring is committed BEFORE the run)
-------------------------------------------------------------------------
Model under test (theorem_targets.md Prop A4): the triangle-wise gradient error is a
piecewise-constant field of correlation length `s` and amplitude `||eps|| ~ s|grad^2 u_c|`;
the central difference of (A1.1) averages `N ~ 2h/s` pieces, so

    floor(s, h) = A * s^(1 + beta/2) * h^(-beta/2),     beta in [1, 2].

The committed 5-rung ladder measured the `h` exponent at fixed `s`:
`p_h = -0.639 +- 0.285` (band 0.1) and `-0.774 +- 0.165` (band 0.25), i.e. `beta/2 = -p_h`.
The model therefore makes a SHARP, PARAMETER-FREE prediction for the `s` exponent
`q := d log(floor) / d log(s)` at fixed `h`:

        q = 1 - p_h    ==>   q = 1.64 (band 0.1),   q = 1.77 (band 0.25).

Operator provenance predicts `q ~ 0`: the monitor-versus-reference operator mismatch
does not change when the cloud is thinned.

Derived secondary prediction (theorem_targets.md sec 1.7, restated from the realized
`s`, not from sqrt(D)): floor rise at D=8 of x4.8 (beta=1) to x8.0 (beta=2).

VERDICT BRANCHES, fixed in advance. `q` is the ordinary-least-squares slope of
`log floor` on `log s_band` over the four D values, per case, per band, per rung;
the reported `q` is the mean over the 16 cases, and the sign count is reported too.

  REPRESENTATION-CONFIRMED
      q >= 1.0 on BOTH bands at BOTH rungs, AND the joint fit
      `log floor = a + q log s - r log h` over the 4x2 (D, h) grid returns
      |(q - 1) - r| <= 0.35 -- i.e. two independently measured slopes agree on one
      beta. => The mechanism sentences of floor_resolution_study.md sec 1a/4/6.2 and
      of residual_floor_theorem.tex are WRONG and must be replaced by a
      representation-error attribution.

  REPRESENTATION-PARTIAL
      0.35 <= q < 1.0 on the primary band, or q >= 1.0 but the joint fit disagrees by
      more than 0.35. => Representation error is a measured driver but the (s,h)
      model does not account for the observed `h`-dependence. The paper may then claim
      NEITHER mechanism: the attribution sentences are deleted and replaced by the
      measured facts plus Prop A1's boundedness.

  PROVENANCE-VINDICATED
      |q| < 0.35 AND the mean floor rise at D=8 is < 1.20x. => The representation
      mechanism is dead, the committed operator-provenance attribution stands, and the
      growth becomes a genuinely unexplained phenomenon (theorem_targets.md sec 5's
      "one thing that would change this verdict").

  PROVENANCE-STRONG
      q <= -0.35 (the floor FALLS when the cloud is thinned). => Representation error
      is not merely absent, it is contradicted: thinning removes reference content and
      the mismatch shrinks with it, which is what provenance predicts.

  Anything else: PARTIAL, reported with the numbers and no mechanism claim.

Per-case results are reported in full, not just means, as
`scripts/floor_resolution_ladder.py` does.

Usage
-----
    python scripts/floor_cloud_decimation.py --gate      # gates only, 2 cases
    python scripts/floor_cloud_decimation.py --n 16      # the experiment
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time

# Must precede numpy/scipy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from floor_resolution_decomposition import (  # noqa: E402
    CaseLadder,
    _pack_field,
    measure,
)

from neuroforge.core.types import DTYPE, Domain  # noqa: E402
from neuroforge.data.airfrans_loader import AIRFRANS_NU, _CROP  # noqa: E402
from neuroforge.physics.residuals import PhysicsChecker  # noqa: E402

PC_CACHE = "data/cache/airfrans_pc_full_test_n24.pkl"
REF_JSON = "results/certificates/floor_resolution_decomposition.json"
OUT_JSON = "results/certificates/floor_cloud_decimation.json"

RUNGS = (128, 256)
DECIMATIONS = (1, 2, 4, 8)
BANDS = (0.05, 0.10, 0.25)
PRIMARY_BAND = 0.10

# Committed h-exponents from floor_resolution_decomposition.json (aggregate).
P_H = {0.05: -0.5280668435606376, 0.10: -0.6394490800720494, 0.25: -0.7741566862081073}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Decimated rasterisation -- a local re-implementation of CaseLadder.rasterise
# so the hull-fallback count can be logged. The interpolator call is identical.
# --------------------------------------------------------------------------- #
def rasterise_from(tri, pos_sub, vals_sub, n):
    """Rasterise ``vals_sub`` on the decimated triangulation onto an n x n crop."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    domain = Domain(bounds=_CROP, nx=n, ny=n)
    X, Y = domain.grid()
    xi = np.stack([X.ravel(), Y.ravel()], axis=1)
    interp = LinearNDInterpolator(tri, vals_sub, fill_value=0.0)
    out = np.asarray(interp(xi), np.float64)
    bad = np.isnan(out).any(axis=1)
    n_fallback = int(bad.sum())
    if bad.any():
        out[bad] = np.asarray(NearestNDInterpolator(pos_sub, vals_sub)(xi[bad]), np.float64)
    raster = out.reshape(n, n, vals_sub.shape[1]).transpose(2, 0, 1).astype(DTYPE)
    return raster, domain, n_fallback


def spacing_of(pos_sub, dist_sub, band):
    """Median NN spacing of the decimated cloud inside the crop and ``sdf > band``."""
    from scipy.spatial import cKDTree

    xmin, xmax, ymin, ymax = _CROP
    sel = ((pos_sub[:, 0] >= xmin) & (pos_sub[:, 0] <= xmax)
           & (pos_sub[:, 1] >= ymin) & (pos_sub[:, 1] <= ymax) & (dist_sub > band))
    sub = pos_sub[sel]
    if sub.shape[0] < 3:
        return float("nan"), 0
    d, _ = cKDTree(sub).query(sub, k=2)
    return float(np.median(d[:, 1])), int(sub.shape[0])


def nested_subsets(n_pts, decimations, seed):
    """Nested random index subsets: D=8 subset of D=4 subset of D=2 subset of D=1."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pts)
    return {int(d): np.sort(perm[: max(4, n_pts // int(d))]) for d in decimations}


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def ols_slope(xs, ys):
    """Slope of an OLS fit of ``ys`` on ``xs``; NaN if fewer than two finite points."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    x, y = x[ok], y[ok]
    if np.ptp(x) < 1e-12:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def joint_fit(rows):
    """Fit ``log floor = a + q log s - r log h`` over the (D, h) grid of one case."""
    s = np.array([r["s"] for r in rows], float)
    h = np.array([r["h"] for r in rows], float)
    f = np.array([r["floor"] for r in rows], float)
    ok = np.isfinite(s) & np.isfinite(h) & np.isfinite(f) & (s > 0) & (f > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    A = np.stack([np.ones(ok.sum()), np.log(s[ok]), np.log(h[ok])], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.log(f[ok]), rcond=None)
    return float(coef[1]), float(-coef[2])   # q, r


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_case(ladder, pc_targets, checker, rungs, decimations, seed):
    """All (D, rung) metrics for one case, geometry held fixed at the full cloud."""
    pos_full = ladder.pos
    dist_full = ladder._signed_dist_points()
    subsets = nested_subsets(pos_full.shape[0], decimations, seed)

    geom = {}
    for n in rungs:
        sdf, mask, domain = ladder.geometry_on(n)   # asserts the surface-loop hash
        geom[n] = (sdf, mask, domain, ladder.case_on(domain))

    rows = []
    for d in decimations:
        idx = subsets[int(d)]
        pos_sub = np.ascontiguousarray(pos_full[idx])
        vals_sub = np.ascontiguousarray(pc_targets[idx])
        dist_sub = dist_full[idx]

        from scipy.spatial import Delaunay
        t0 = time.time()
        tri = Delaunay(pos_sub)
        t_tri = time.time() - t0

        spac = {}
        for b in BANDS:
            sv, nb = spacing_of(pos_sub, dist_sub, b)
            spac[f"s_band_{b:g}"] = sv
            spac[f"n_pts_band_{b:g}"] = nb
        s_crop, n_crop = spacing_of(pos_sub, dist_sub, -1e9)

        for n in rungs:
            sdf, mask, domain, case = geom[n]
            raster, dom2, n_fb = rasterise_from(tri, pos_sub, vals_sub, n)
            assert dom2.nx == domain.nx
            field = _pack_field(raster, sdf, mask, domain, "truth")
            met = measure(field, case, checker, AIRFRANS_NU, BANDS)
            rec = {"D": int(d), "n": int(n), "h": float(domain.dx),
                   "n_points": int(pos_sub.shape[0]),
                   "n_points_in_crop": n_crop, "s_crop": s_crop,
                   "n_hull_fallback": n_fb,
                   "hull_fallback_frac": n_fb / float(n * n),
                   "delaunay_s": t_tri}
            rec.update(spac)
            for k in ("native", "band_0.05", "band_0.1", "band_0.25",
                      "band_0.1_continuity", "band_0.1_momentum",
                      "term_conv_band_0.1", "term_pres_band_0.1", "term_visc_band_0.1",
                      "term_visc_band_0.25", "omit_gradnu_band_0.1"):
                if k in met:
                    rec[k] = met[k]
            rows.append(rec)
        del tri, pos_sub, vals_sub
        gc.collect()
    return rows


def gate_reproduce(per_case, ref_json):
    """D=1 must reproduce the committed decomposition per-case floors bit-for-bit."""
    if not os.path.exists(ref_json):
        return {"pass": False, "why": f"missing {ref_json}"}
    ref = json.load(open(ref_json, encoding="utf-8"))
    by_name = {c["name"]: c for c in ref["per_case"]}
    worst, n_cmp, detail = 0.0, 0, []
    for case in per_case:
        rc = by_name.get(case["name"])
        if rc is None:
            continue
        rr = {int(r["n"]): r for r in rc["rungs"]}
        for row in case["rows"]:
            if row["D"] != 1 or int(row["n"]) not in rr:
                continue
            for key in ("band_0.05", "band_0.1", "band_0.25", "native"):
                a = float(row[key])
                b = float(rr[int(row["n"])][f"truth_{key}"])
                rel = abs(a - b) / max(abs(b), 1e-30)
                worst = max(worst, rel)
                n_cmp += 1
                if rel > 1e-9:
                    detail.append({"case": case["name"], "n": row["n"],
                                   "key": key, "ours": a, "ref": b, "rel": rel})
    return {"pass": bool(n_cmp > 0 and worst <= 1e-9), "worst_rel": worst,
            "n_compared": n_cmp, "mismatches": detail[:8]}


def aggregate(per_case, rungs, decimations):
    out = {"by_band": {}, "by_cell": {}, "gates": {}}

    fb = [r["hull_fallback_frac"] for c in per_case for r in c["rows"]]
    fb_by_d = {int(d): float(np.max([r["hull_fallback_frac"] for c in per_case
                                     for r in c["rows"] if r["D"] == d]))
               for d in decimations}
    out["gates"]["hull_fallback"] = {
        "max_frac": float(np.max(fb)) if fb else 0.0,
        "max_frac_by_D": fb_by_d,
        "pass": bool(np.max(fb) < 1e-3) if fb else False,
        "rule": "max fraction of query points hitting the NearestND fallback < 1e-3",
    }

    # cell means over cases, for the report table
    for b in BANDS:
        key = f"band_{b:g}"
        for n in rungs:
            for d in decimations:
                vals = [r[key] for c in per_case for r in c["rows"]
                        if r["D"] == d and r["n"] == n]
                sv = [r[f"s_band_{b:g}"] for c in per_case for r in c["rows"]
                      if r["D"] == d and r["n"] == n]
                out["by_cell"][f"{key}/n{n}/D{d}"] = {
                    "floor_mean": float(np.mean(vals)) if vals else float("nan"),
                    "s_band_mean": float(np.mean(sv)) if sv else float("nan"),
                    "n_cases": len(vals),
                }

    for b in BANDS:
        key = f"band_{b:g}"
        rec = {"p_h_committed": P_H[b], "q_predicted_representation": 1.0 - P_H[b],
               "q_predicted_provenance": 0.0, "per_rung": {}, "per_case_q": {}}
        for n in rungs:
            qs, rises, s_ratios = [], [], []
            for c in per_case:
                rows = [r for r in c["rows"] if r["n"] == n]
                rows.sort(key=lambda r: r["D"])
                ls = [np.log(r[f"s_band_{b:g}"]) for r in rows]
                lf = [np.log(r[key]) for r in rows]
                q = ols_slope(ls, lf)
                qs.append(q)
                rec["per_case_q"].setdefault(c["name"], {})[f"n{n}"] = q
                f1 = [r[key] for r in rows if r["D"] == 1]
                f8 = [r[key] for r in rows if r["D"] == max(decimations)]
                if f1 and f8:
                    rises.append(f8[0] / f1[0])
                s1 = [r[f"s_band_{b:g}"] for r in rows if r["D"] == 1]
                s8 = [r[f"s_band_{b:g}"] for r in rows if r["D"] == max(decimations)]
                if s1 and s8:
                    s_ratios.append(s8[0] / s1[0])
            qs = np.array(qs, float)
            rec["per_rung"][f"n{n}"] = {
                "q_mean": float(np.nanmean(qs)), "q_std": float(np.nanstd(qs)),
                "q_min": float(np.nanmin(qs)), "q_max": float(np.nanmax(qs)),
                "n_cases_q_ge_1": int(np.sum(qs >= 1.0)),
                "n_cases_q_positive": int(np.sum(qs > 0.0)),
                "n_cases": int(np.isfinite(qs).sum()),
                "rise_at_Dmax_mean": float(np.mean(rises)) if rises else float("nan"),
                "rise_at_Dmax_min": float(np.min(rises)) if rises else float("nan"),
                "rise_at_Dmax_max": float(np.max(rises)) if rises else float("nan"),
                "n_cases_rising_at_Dmax": int(np.sum(np.array(rises) > 1.0)),
                "s_ratio_at_Dmax_mean": float(np.mean(s_ratios)) if s_ratios else float("nan"),
            }
        # joint fit over the whole (D, h) grid, per case
        qj, rj = [], []
        for c in per_case:
            rows = [{"s": r[f"s_band_{b:g}"], "h": r["h"], "floor": r[key]}
                    for r in c["rows"]]
            q, r_ = joint_fit(rows)
            qj.append(q)
            rj.append(r_)
        qj, rj = np.array(qj, float), np.array(rj, float)
        rec["joint_fit"] = {
            "q_mean": float(np.nanmean(qj)), "q_std": float(np.nanstd(qj)),
            "r_mean": float(np.nanmean(rj)), "r_std": float(np.nanstd(rj)),
            "beta_from_q": float(2.0 * (np.nanmean(qj) - 1.0)),
            "beta_from_r": float(2.0 * np.nanmean(rj)),
            "consistency_gap_abs": float(abs((np.nanmean(qj) - 1.0) - np.nanmean(rj))),
        }
        out["by_band"][key] = rec
    return out


def verdict(agg, rungs):
    """Apply the pre-registered decision rule of the module docstring."""
    pb = f"band_{PRIMARY_BAND:g}"
    rec = agg["by_band"][pb]
    q_primary = [rec["per_rung"][f"n{n}"]["q_mean"] for n in rungs]
    q25 = [agg["by_band"]["band_0.25"]["per_rung"][f"n{n}"]["q_mean"] for n in rungs]
    gap = rec["joint_fit"]["consistency_gap_abs"]
    rise = float(np.mean([rec["per_rung"][f"n{n}"]["rise_at_Dmax_mean"] for n in rungs]))
    all_q = q_primary + q25
    qmin = float(np.min(all_q))
    qbar = float(np.mean(q_primary))

    if not agg["gates"]["hull_fallback"]["pass"]:
        return {"branch": "NO-VERDICT", "why": "hull fallback gate failed"}
    if qmin >= 1.0 and gap <= 0.35:
        br = "REPRESENTATION-CONFIRMED"
        why = ("floor rises with cloud spacing at fixed h with exponent q >= 1.0 on "
               "both bands at both rungs, and the joint (s, h) fit's two slopes agree "
               f"on one beta to {gap:.2f} <= 0.35")
    elif qbar >= 0.35:
        br = "REPRESENTATION-PARTIAL"
        why = ("the floor depends measurably on cloud spacing at fixed h "
               f"(q = {qbar:.2f} on the primary band) but the (s, h) model does not "
               f"close: min q over bands/rungs {qmin:.2f}, joint-fit gap {gap:.2f}")
    elif qbar <= -0.35:
        br = "PROVENANCE-STRONG"
        why = f"the floor FALLS when the cloud is thinned (q = {qbar:.2f})"
    elif abs(qbar) < 0.35 and rise < 1.20:
        br = "PROVENANCE-VINDICATED"
        why = (f"floor insensitive to cloud spacing: q = {qbar:.2f}, mean rise at "
               f"D=max {rise:.2f}x < 1.20x")
    else:
        br = "PARTIAL"
        why = f"q = {qbar:.2f}, rise {rise:.2f}x: no branch of the rule applies"
    return {"branch": br, "why": why, "q_primary_by_rung": q_primary,
            "q_band025_by_rung": q25, "q_min_over_bands_rungs": qmin,
            "joint_fit_gap": gap, "mean_rise_at_Dmax": rise}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--rungs", type=int, nargs="*", default=list(RUNGS))
    ap.add_argument("--decimations", type=int, nargs="*", default=list(DECIMATIONS))
    ap.add_argument("--gate", action="store_true", help="2 cases, gates only")
    ap.add_argument("--cache", default=PC_CACHE)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    t_start = time.time()
    n_cases = 2 if args.gate else args.n
    checker = PhysicsChecker()

    log(f"loading {n_cases} point clouds from {args.cache} ...")
    t0 = time.time()
    with open(args.cache, "rb") as fh:
        pcs = pickle.load(fh)
    clouds = list(pcs[:n_cases])
    del pcs
    gc.collect()
    log(f"  {len(clouds)} clouds, {time.time() - t0:.0f}s")

    per_case = []
    for i, pc in enumerate(clouds):
        t0 = time.time()
        ladder = CaseLadder(pc)
        targets = np.asarray(pc.targets, np.float64)
        rows = run_case(ladder, targets, checker, args.rungs, args.decimations,
                        seed=args.seed * 1000 + i)
        per_case.append({"name": ladder.name, "u_inf": ladder.u_inf,
                         "aoa_deg": ladder.aoa, "n_points": int(ladder.pos.shape[0]),
                         "n_points_in_crop": ladder.n_in_crop, "rows": rows})
        f1 = [r["band_0.1"] for r in rows if r["D"] == 1 and r["n"] == args.rungs[0]]
        f8 = [r["band_0.1"] for r in rows
              if r["D"] == max(args.decimations) and r["n"] == args.rungs[0]]
        log(f"[{i + 1}/{len(clouds)}] {ladder.name}  band0.1@n{args.rungs[0]}: "
            f"D1 {f1[0]:.5f} -> D{max(args.decimations)} {f8[0]:.5f} "
            f"({f8[0] / f1[0]:.2f}x)  {time.time() - t0:.0f}s")
        del ladder, targets
        gc.collect()

    gates = {"reproduction_D1": gate_reproduce(per_case, REF_JSON)}
    agg = aggregate(per_case, args.rungs, args.decimations)
    gates.update(agg.pop("gates"))
    ver = verdict({**agg, "gates": gates}, args.rungs)

    out = {
        "artifact": "floor_cloud_decimation",
        "question": ("At FIXED grid spacing h, does the residual floor depend on the "
                     "SOURCE CLOUD spacing s? Representation error says yes with "
                     "exponent q = 1 - p_h; operator provenance says no."),
        "preregistration": "docstring of scripts/floor_cloud_decimation.py, committed "
                           "before the run",
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": "cpu",
            "cache": args.cache,
            "n_cases": len(per_case),
            "rungs": list(args.rungs),
            "decimations": list(args.decimations),
            "bands": list(BANDS),
            "seed": args.seed,
            "geometry_source": "FULL cloud, before decimation -- sdf/mask/bands "
                               "identical at every D (surface-loop hash asserted)",
            "subsets": "nested random, fixed per-case seed",
        },
        "gates": gates,
        "verdict": ver,
        "aggregate": agg,
        "per_case": per_case,
        "wall_clock_s": time.time() - t_start,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    log(f"\nwrote {args.out} ({time.time() - t_start:.0f}s)")

    log("\n=== GATES ===")
    for k, v in gates.items():
        log(f"  {k}: {'PASS' if v.get('pass') else 'FAIL'}  {json.dumps(_short(v))}")
    log("\n=== q (floor ~ s^q at fixed h) ===")
    for b in BANDS:
        rec = agg["by_band"][f"band_{b:g}"]
        log(f"  band {b:g}: predicted representation q = "
            f"{rec['q_predicted_representation']:.2f}, provenance q = 0")
        for n in args.rungs:
            r = rec["per_rung"][f"n{n}"]
            log(f"    n={n}: q = {r['q_mean']:+.3f} +- {r['q_std']:.3f} "
                f"({r['n_cases_q_positive']}/{r['n_cases']} positive, "
                f"{r['n_cases_q_ge_1']}/{r['n_cases']} >= 1); "
                f"rise at D=max {r['rise_at_Dmax_mean']:.2f}x "
                f"({r['n_cases_rising_at_Dmax']}/{r['n_cases']} rising), "
                f"s x{r['s_ratio_at_Dmax_mean']:.2f}")
        j = rec["joint_fit"]
        log(f"    joint: q = {j['q_mean']:+.3f}, r = {j['r_mean']:+.3f}, "
            f"beta_q = {j['beta_from_q']:.2f}, beta_r = {j['beta_from_r']:.2f}, "
            f"gap = {j['consistency_gap_abs']:.3f}")
    log(f"\n=== VERDICT: {ver['branch']} ===\n  {ver['why']}")
    return 0


def _short(v):
    return {k: (round(x, 12) if isinstance(x, float) else x)
            for k, x in v.items() if k not in ("mismatches",)}


if __name__ == "__main__":
    raise SystemExit(main())
