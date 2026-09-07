"""Does the residual floor plateau or decay under grid refinement?

The blocking question for the paper-1 reframe. Theorem ``thm:residual-floor``
rests on (H2), the floor ``r* := R_h(u*) != 0``. The theorem file itself
attributes part of the floor to the ``128^2`` grid under-resolving the boundary
layer. If ``||r*|| -> 0`` as ``h -> 0`` the floor is a discretisation artifact;
if it plateaus, the claim is about the *operator* and not the grid, which is a
far stronger and far less desk-rejectable paper.

Read the pre-registration below before the numbers. It was committed before the
run -- see this file's first commit in the git log.

===========================================================================
PRE-REGISTRATION -- written and committed BEFORE the ladder was run
===========================================================================

**What is NOT at risk, and why the reframe plan was wrong about it.**
Leg (i) of the theorem -- ``R_h(u_inf) = 0`` exactly -- is not a rasterisation
artifact and cannot become one. A spatially constant field has identically zero
finite differences at every ``h``. What is grid-dependent is the *magnitude* of
``||r*||``: how much better than the truth the uniform field scores.

**H-A (the floor).** ``||r*||`` over the comparable-cell set defined below.
  * DECAY   := a fit ``||r*|| ~ C h^p`` over the comparable range with
    ``p >= 1.0`` AND the fit extrapolating to within noise of zero.
  * PLATEAU := ``p < 0.5``, or a clear asymptote at a nonzero level.
  * ``0.5 <= p < 1.0`` is reported as PARTIAL and read as decay slower than
    first order, which still leaves a floor at the scale the loop operates at.

**H-B (the omitted closure term).** ``grad(nu_t) . grad(u)`` is a *model*
omission, not a truncation error. Predicted BEFORE running: at ``128^2`` the
sub-cell boundary layer smears ``grad(nu_t)``, so this term is *under*-estimated
there; it should **grow** with refinement and then flatten, not decay. If it
instead decays like the truncation terms, that prediction was wrong and is
reported as wrong.

**H-C (leg (A) of the lambda-sweep -- the real exposure).** The closed-form "no
crossover weight" argument needs ``r_bc^2(u*) > r_bc^2(u_inf)``, which currently
holds (0.0073 vs 0.0053) only because, in the theorem file's own words, "on a
128^2 grid the boundary layer is sub-cell, so the rasterised truth itself
carries near-wall velocity". Predicted: resolving the layer lowers
``r_bc^2(u*)``. If it falls below ``r_bc^2(u_inf)`` at any level then the sign
flips at a finite ``lambda_c = ||r*||^2 / (r_bc^2(u_inf) - r_bc^2(u*))``, and
the paper's "for every lambda >= 0" sentence must be weakened to "for every
lambda < lambda_c". Reported whichever way it lands.

**Mask policy, pre-declared.** ``diagnose()`` zeroes the solid and a *one-cell*
fluid ring, so refining the grid physically *unmasks* the steepest near-wall
cells and can manufacture a plateau out of nothing. The comparable-cell set is
therefore fixed in PHYSICAL units at the coarsest level:

    D_REF = 1.5 * h_128 = 1.5 * 3.0 / 128 = 0.035156 chord

Every level excludes fluid cells closer to the wall than that, so all levels are
scored over the same physical region. The default-mask curve is recorded too, as
the exhibit for why it was not used.

**Two bands for the no-slip term.** The framework's ``_surface_proximity_weight``
decays over ``3 * min(dx, dy)`` -- a band that *shrinks* with refinement, so the
paper's own ``r_bc`` is measured over a physically thinner region at each level.
Both are reported: the framework band (what the lambda-sweep actually used) and
a band fixed at the ``128^2`` physical width (the honest comparison).

**Interpolation floor (the ladder's own limit).** AirfRANS truth lives on a
body-fitted unstructured mesh. Refining the raster past the source mesh's local
spacing does not sample a better solution, it differentiates a piecewise-linear
reconstruction, whose second derivatives are distributional -- so the residual
can *rise* again. The median nearest-neighbour spacing of the source points is
measured near-wall and far-field, and any level whose ``h`` falls below it is
flagged and EXCLUDED from the exponent fit.

**Paired.** The same cases at every level; per-case ladders and sign counts are
reported, not only means, because thin and thick sections resolve differently
and a mean can hide a bimodal split.
===========================================================================

Usage
-----
    python scripts/floor_resolution_ladder.py --levels 128 256 --n-cases 24
    python scripts/floor_resolution_ladder.py --levels 128 256 512 --n-cases 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Must precede numpy: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.core.types import FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.physics.operators import gradient
from neuroforge.physics.residuals import PhysicsChecker, _solid_adjacent_fluid

# Pre-declared in the registration above. 3.0 is the crop width (_CROP).
H_COARSE = 3.0 / 128.0
D_REF = 1.5 * H_COARSE
PROX_BAND_COARSE = 3.0 * H_COARSE     # the 128^2 physical width of the no-slip band


def uniform_like(case, field: FlowField) -> FlowField:
    """Constant freestream on the same grid, reusing the truth's mask and sdf.

    Constant *through* the body as well as around it: the whole point of leg (i)
    is that this field has identically zero finite differences everywhere.
    """
    u_in, v_in = case.bc.inlet_vector()
    return FlowField(
        domain=field.domain,
        u=np.full(field.shape, np.float32(u_in), dtype=np.float32),
        v=np.full(field.shape, np.float32(v_in), dtype=np.float32),
        p=np.zeros(field.shape, dtype=np.float32),
        nut=np.zeros(field.shape, dtype=np.float32),
        mask=field.mask, sdf=field.sdf,
    )


def comparable_cells(field: FlowField) -> np.ndarray:
    """Fluid cells at least ``D_REF`` from the wall, at every resolution.

    The one-cell ring ``diagnose()`` zeroes is physically thinner on a finer
    grid; scoring on that set would compare different regions across levels.
    """
    mask32 = np.asarray(field.mask, dtype=np.float32)
    fluid = np.asarray(field.mask, dtype=float) > 0.5
    sdf = np.asarray(field.sdf, dtype=float)
    return fluid & (sdf >= D_REF) & ~_solid_adjacent_fluid(mask32)


def default_cells(field: FlowField) -> np.ndarray:
    """What ``diagnose()`` actually scores: fluid minus the one-cell wall ring."""
    mask32 = np.asarray(field.mask, dtype=np.float32)
    fluid = np.asarray(field.mask, dtype=float) > 0.5
    return fluid & ~_solid_adjacent_fluid(mask32)


def rms(values, sel: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(v[sel] ** 2))) if np.any(sel) else float("nan")


def monitored_rms(diag, sel: np.ndarray) -> float:
    """RMS of sqrt(cont^2 + mom_x^2 + mom_y^2): the monitored norm, bc excluded."""
    if not np.any(sel):
        return float("nan")
    c = np.asarray(diag.continuity, dtype=np.float64)
    x = np.asarray(diag.momentum_x, dtype=np.float64)
    y = np.asarray(diag.momentum_y, dtype=np.float64)
    return float(np.sqrt(np.mean((c * c + x * x + y * y)[sel])))


def omitted_closure_rms(case, field: FlowField, sel: np.ndarray) -> float:
    """RMS of the dropped term ``grad(nu_t) . grad(u)``, on the momentum scale.

    The monitored operator uses ``nu_eff * lap(u)`` in place of the correct
    ``div(nu_eff grad(u)) = nu_eff lap(u) + grad(nu_eff) . grad(u)``. Only
    ``nu_t`` varies in space, so the omission is ``grad(nu_t) . grad(u)``. It is
    a MODEL omission and should not vanish with h -- hypothesis H-B.
    """
    if not np.any(sel):
        return float("nan")
    dx, dy = field.domain.dx, field.domain.dy
    nut_x, nut_y = gradient(np.asarray(field.nut, dtype=np.float64), dx, dy)
    ux, uy = gradient(np.asarray(field.u, dtype=np.float64), dx, dy)
    vx, vy = gradient(np.asarray(field.v, dtype=np.float64), dx, dy)
    tx = nut_x * ux + nut_y * uy
    ty = nut_x * vx + nut_y * vy
    scale = float(case.bc.u_inf) ** 2 / max(float(case.reference_length()), 1e-9)
    return float(np.sqrt(np.mean((tx * tx + ty * ty)[sel]))) / max(scale, 1e-30)


def bc_terms(diag, field: FlowField):
    """``r_bc^2`` under the framework's h-dependent band and a fixed physical band.

    ``diag.bc_violation`` is already proximity-weighted over ``3*min(dx,dy)``,
    which shrinks as the grid refines. The fixed band re-weights it to the
    ``128^2`` physical width so the levels are comparable.
    """
    fluid = np.asarray(field.mask, dtype=float) > 0.5
    if not np.any(fluid):
        return float("nan"), float("nan")
    bc = np.asarray(diag.bc_violation, dtype=np.float64)
    framework = float(np.mean(bc[fluid] ** 2))
    sdf = np.abs(np.asarray(field.sdf, dtype=np.float64))
    dx, dy = field.domain.dx, field.domain.dy
    w_fw = np.exp(-sdf / max(3.0 * min(dx, dy), 1e-30))
    w_fx = np.exp(-sdf / PROX_BAND_COARSE)
    # Undo the framework weight where it is resolvable, apply the fixed one.
    ratio = np.where(w_fw > 1e-12, w_fx / np.where(w_fw > 1e-12, w_fw, 1.0), np.nan)
    fixed = float(np.nanmean((bc * ratio)[fluid] ** 2))
    return framework, fixed


def source_spacing(root: str, task: str, n_cases: int):
    """Median nearest-neighbour spacing of the AirfRANS point cloud, in chords.

    Near-wall and far-field. A raster level finer than this is resolving the
    interpolant rather than the flow, and is excluded from the exponent fit.
    """
    try:
        from scipy.spatial import cKDTree

        from neuroforge.data.airfrans_loader import (_require_airfrans,
                                                     _resolve_data_root)
        af = _require_airfrans()
        data_root = _resolve_data_root(root)
        if data_root is None:
            return {"error": "manifest.json not found"}
        dataset, names = af.dataset.load(root=data_root, task=task, train=False)
        near, far = [], []
        for i in range(min(n_cases, len(names))):
            d = np.asarray(dataset[i], dtype=np.float64)
            pos, sd = d[:, 0:2], np.abs(d[:, 4])
            keep = ((pos[:, 0] > -1.0) & (pos[:, 0] < 2.0)
                    & (pos[:, 1] > -1.5) & (pos[:, 1] < 1.5))
            pos, sd = pos[keep], sd[keep]
            if len(pos) < 10:
                continue
            dist, _ = cKDTree(pos).query(pos, k=2)
            nn = dist[:, 1]
            if np.any(sd < 0.02):
                near.append(float(np.median(nn[sd < 0.02])))
            if np.any(sd > 0.5):
                far.append(float(np.median(nn[sd > 0.5])))
        del dataset, names
        return {"near_wall": float(np.median(near)) if near else None,
                "far_field": float(np.median(far)) if far else None,
                "n_cases": len(near)}
    except Exception as exc:  # scipy/airfrans absent, or the split will not load
        return {"error": f"{type(exc).__name__}: {exc}"}


def fit_exponent(hs, vals):
    """Least-squares ``p`` in ``v ~ C h^p``."""
    hs, vals = np.asarray(hs, float), np.asarray(vals, float)
    ok = np.isfinite(hs) & np.isfinite(vals) & (vals > 0)
    if ok.sum() < 2:
        return None
    p, logc = np.polyfit(np.log(hs[ok]), np.log(vals[ok]), 1)
    return {"p": float(p), "C": float(np.exp(logc)), "n_points": int(ok.sum())}


def verdict_for(p: float | None) -> str:
    """The pre-registered reading of the fitted exponent. No post-hoc latitude."""
    if p is None:
        return "UNDECIDED (too few levels)"
    if p >= 1.0:
        return "DECAY"
    if p < 0.5:
        return "PLATEAU"
    return "PARTIAL"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="residual-floor resolution ladder")
    ap.add_argument("--levels", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--n-cases", type=int, default=24)
    ap.add_argument("--task", default="full")
    ap.add_argument("--root", default="data")
    ap.add_argument("--cache-dir", default=os.path.join("data", "cache"))
    ap.add_argument("--out",
                    default=os.path.join("results", "floor_resolution_ladder.json"))
    args = ap.parse_args(argv)

    print(f"floor ladder | levels {args.levels} | {args.n_cases} cases | "
          f"D_REF {D_REF:.6f} chord (= 1.5 x h_128)\n")

    checker = PhysicsChecker()
    per_level: dict[int, list[dict]] = {}
    names_ref = None
    for res in args.levels:
        t0 = time.time()
        pairs = load_airfrans(task=args.task, train=False, resolution=res,
                              limit=args.n_cases, root=args.root,
                              cache_dir=args.cache_dir, progress=False)
        names = [c.name for c, _ in pairs]
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            print(f"  ! level {res} case order differs; pairing would be wrong")
            return 1
        rows = []
        for case, truth in pairs:
            sel, dflt = comparable_cells(truth), default_cells(truth)
            unif = uniform_like(case, truth)
            d_t, d_u = checker.diagnose(truth, case), checker.diagnose(unif, case)
            bc_t_fw, bc_t_fx = bc_terms(d_t, truth)
            bc_u_fw, bc_u_fx = bc_terms(d_u, unif)
            rows.append({
                "case": case.name,
                "n_comparable": int(sel.sum()),
                "floor": monitored_rms(d_t, sel),
                "floor_default_mask": monitored_rms(d_t, dflt),
                "uniform": monitored_rms(d_u, sel),
                "cont": rms(d_t.continuity, sel),
                "mom": float(np.hypot(rms(d_t.momentum_x, sel),
                                      rms(d_t.momentum_y, sel))),
                "omitted_closure": omitted_closure_rms(case, truth, sel),
                "bc2_truth_framework": bc_t_fw, "bc2_uniform_framework": bc_u_fw,
                "bc2_truth_fixed": bc_t_fx, "bc2_uniform_fixed": bc_u_fx,
            })
        per_level[res] = rows

        def avg(key, rs=rows):
            return float(np.nanmean([r[key] for r in rs]))

        print(f"  {res:>4}^2  h={3.0 / res:.5f}  floor {avg('floor'):.4f} "
              f"(default mask {avg('floor_default_mask'):.4f})  uniform "
              f"{avg('uniform'):.2e}  omitted {avg('omitted_closure'):.4f}  "
              f"[{time.time() - t0:.0f}s]")

    print("\nsource-mesh spacing (a raster level below this resolves the "
          "interpolant, not the flow):")
    spacing = source_spacing(args.root, args.task, min(8, args.n_cases))
    print(f"  {spacing}")

    levels = sorted(per_level)
    hs = [3.0 / r for r in levels]

    def mean_over(key, res):
        return float(np.nanmean([r[key] for r in per_level[res]]))

    print(f"\n{'level':>7} {'h':>9} {'floor':>9} {'omitted':>9} "
          f"{'bc2 truth':>11} {'bc2 unif':>10} {'bc2 T fix':>10} {'bc2 U fix':>10}")
    for i, res in enumerate(levels):
        print(f"{res:>7} {hs[i]:>9.5f} {mean_over('floor', res):>9.4f} "
              f"{mean_over('omitted_closure', res):>9.4f} "
              f"{mean_over('bc2_truth_framework', res):>11.5f} "
              f"{mean_over('bc2_uniform_framework', res):>10.5f} "
              f"{mean_over('bc2_truth_fixed', res):>10.5f} "
              f"{mean_over('bc2_uniform_fixed', res):>10.5f}")

    # Levels finer than the source mesh near the wall are excluded from the fit.
    near = (spacing or {}).get("near_wall")
    usable = [r for r in levels if not (near and 3.0 / r < near)]
    excluded = [r for r in levels if r not in usable]
    if excluded:
        print(f"\nexcluded from the fit (h below the source spacing "
              f"{near:.5f}): {excluded}")

    fits = {}
    print("\nfitted exponents  v ~ C h^p   (pre-registered: p>=1 DECAY, "
          "p<0.5 PLATEAU, else PARTIAL)")
    for key in ("floor", "omitted_closure", "floor_default_mask"):
        f = fit_exponent([3.0 / r for r in usable],
                         [mean_over(key, r) for r in usable])
        fits[key] = f
        if f:
            fits[key]["verdict"] = verdict_for(f["p"])
            print(f"  {key:>20}  p = {f['p']:+.3f}  over {f['n_points']} levels"
                  f"   -> {fits[key]['verdict']}")
        else:
            print(f"  {key:>20}  not enough points")

    # Per-case sign counts: does every case move the same way, or is it bimodal?
    if len(usable) >= 2:
        lo, hi = usable[0], usable[-1]
        by_case = {r["case"]: r["floor"] for r in per_level[hi]}
        deltas = [by_case[r["case"]] - r["floor"] for r in per_level[lo]
                  if r["case"] in by_case]
        fell = int(sum(d < 0 for d in deltas))
        print(f"\nper-case, {lo}^2 -> {hi}^2: the floor fell in {fell}/{len(deltas)} "
              f"cases, rose in {len(deltas) - fell}")

    # H-C: does leg (A) of the lambda-sweep survive at every level?
    print("\nH-C, leg (A) of the lambda-sweep -- needs bc2(truth) > bc2(uniform):")
    crossover = {}
    for res in levels:
        for band in ("framework", "fixed"):
            t = mean_over(f"bc2_truth_{band}", res)
            u = mean_over(f"bc2_uniform_{band}", res)
            floor = mean_over("floor", res)
            holds = bool(t > u)
            lam = None if holds else floor ** 2 / max(u - t, 1e-30)
            crossover[f"{res}_{band}"] = {"truth": t, "uniform": u, "holds": holds,
                                          "lambda_c": lam}
            print(f"  {res:>4}^2 {band:>10} band: truth {t:.5f} vs uniform "
                  f"{u:.5f} -> " + ("HOLDS for all lambda" if holds
                                    else f"FLIPS; lambda_c = {lam:.1f}"))

    payload = {"levels": levels, "levels_used_in_fit": usable, "n_cases": args.n_cases,
               "task": args.task, "d_ref": D_REF,
               "prox_band_coarse": PROX_BAND_COARSE, "source_spacing": spacing,
               "fits": fits, "crossover": crossover,
               "per_level": {str(k): v for k, v in per_level.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
