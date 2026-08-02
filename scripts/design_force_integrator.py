"""Design study: repair the force-integrator magnitude bias (CPU-only).

Motivation
----------
``results/control/force_vs_official.json`` showed that the deployed
``force_coefficients`` integrator recovers force *ranking* against official
AirfRANS labels (rho_D ~ 0.84, rho_L ~ 0.88) but is badly biased in
*magnitude* (cd_rel_err ~ 11x). Two candidate mechanisms, both testable on
GROUND-TRUTH fields alone (no model, no GPU):

(a) Pressure is sampled 1.5 cells off the wall (bilinear stencil must stay in
    the fluid), where the pressure has relaxed off its wall value. Drag is a
    small fore/aft difference of large pressure contributions, so the bias
    does not cancel.
(b) The viscous term uses the *magnitude* field ``wall_shear_stress`` (which
    contains ``abs(du_t/dn)``) applied along the CCW traversal tangent. The
    local flow direction relative to the traversal tangent flips between the
    upper and lower surfaces, so signed viscous drag partially cancels
    instead of accumulating.

This script grid-searches integrator variants on the 200 rasterised GT test
fields against the cached official labels and reports rank correlation +
relative magnitude error for each variant. It changes nothing in the
deployed code; the winner is then implemented behind an opt-in flag.

Variants
--------
p_scheme:   offset  -- p sampled at d1 cells off the wall (legacy)
            extrap  -- linear extrapolation to the wall from d1 and 2*d1
tau_scheme: field_abs    -- legacy: |tau| field bilinear at d1, along +t_hat
            field_signed -- same magnitude, signed by the local tangential
                            velocity direction
            noslip       -- one-sided no-slip difference:
                            tau_kin = nu_eff(d1) * u_t(d1) / d1  (signed)
d1_cells:   1.0 / 1.5 / 2.0

Run
---
.venv/Scripts/python.exe scripts/design_force_integrator.py
Writes results/control/integrator_design.json and prints a ranked table.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets thread caps)
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.physics.evaluation import _spearman
from neuroforge.physics.metrics import (
    _EPS,
    _bilinear_sample,
    _surface_points_normals,
    wall_shear_stress,
)

LABEL_CACHE = os.path.join("results", "control", "_cache",
                           "official_labels_full_test_n200.json")
OUT_PATH = os.path.join("results", "control", "integrator_design.json")


def variant_coefficients(field, case, p_scheme: str, tau_scheme: str,
                         d1_cells: float) -> dict[str, float]:
    """Parameterised re-implementation of ``force_coefficients``.

    Mirrors the deployed segment/normal/tangent construction exactly; only
    the pressure sampling and the viscous term differ per variant.
    """
    pts, nrm = _surface_points_normals(case)
    n = pts.shape[0]
    if n < 3:
        return {"cl": 0.0, "cd": 0.0}

    p_next = np.roll(pts, -1, axis=0)
    seg = p_next - pts
    ds = np.sqrt((seg ** 2).sum(axis=1))
    ds = np.maximum(ds, _EPS)
    mid = 0.5 * (pts + p_next)
    nrm_next = np.roll(nrm, -1, axis=0)
    nmid = 0.5 * (nrm + nrm_next)
    nmid /= np.maximum(np.sqrt((nmid ** 2).sum(axis=1, keepdims=True)), _EPS)
    tmid = seg / ds[:, None]

    cell = 0.5 * (field.domain.dx + field.domain.dy)
    d1 = d1_cells * cell
    s1 = mid + d1 * nmid

    # ---- pressure ----
    if p_scheme == "offset":
        p_s = _bilinear_sample(field.p, field.domain, s1)
    elif p_scheme == "extrap":
        s2 = mid + (2.0 * d1) * nmid
        p1 = _bilinear_sample(field.p, field.domain, s1)
        p2 = _bilinear_sample(field.p, field.domain, s2)
        p_s = 2.0 * p1 - p2  # linear extrapolation to the wall
    else:
        raise ValueError(p_scheme)

    # ---- viscous ----
    rho = float(case.fluid.density)
    u_s = _bilinear_sample(field.u, field.domain, s1)
    v_s = _bilinear_sample(field.v, field.domain, s1)
    ut_s = u_s * tmid[:, 0] + v_s * tmid[:, 1]  # signed tangential velocity

    if tau_scheme == "field_abs":
        tau_s = _bilinear_sample(wall_shear_stress(field, case),
                                 field.domain, s1)
        tau_kin = tau_s / max(rho, _EPS)                      # legacy: always +t_hat
    elif tau_scheme == "field_signed":
        tau_s = _bilinear_sample(wall_shear_stress(field, case),
                                 field.domain, s1)
        tau_kin = np.sign(ut_s) * tau_s / max(rho, _EPS)      # along local flow
    elif tau_scheme == "noslip":
        nu = float(case.fluid.kinematic_viscosity)
        if field.nut is not None:
            nut_s = np.clip(
                _bilinear_sample(field.nut, field.domain, s1), 0.0, None)
        else:
            nut_s = 0.0
        nu_eff = nu + nut_s
        tau_kin = nu_eff * ut_s / max(d1, _EPS)               # signed one-sided FD
    else:
        raise ValueError(tau_scheme)

    fx = (-p_s * nmid[:, 0] + tau_kin * tmid[:, 0]) * ds
    fy = (-p_s * nmid[:, 1] + tau_kin * tmid[:, 1]) * ds
    Fx, Fy = float(fx.sum()), float(fy.sum())

    a = np.deg2rad(float(case.bc.aoa_deg))
    ca, sa = np.cos(a), np.sin(a)
    drag = Fx * ca + Fy * sa
    lift = -Fx * sa + Fy * ca

    u_inf = float(case.bc.u_inf)
    chord = max(case.reference_length(), _EPS)
    denom = max(0.5 * u_inf * u_inf * chord, _EPS)
    return {"cl": lift / denom, "cd": drag / denom}


def cv_coefficients(field, case) -> dict[str, float]:
    """Control-volume (momentum-theorem) force estimate on the domain boundary.

    F_body = -oint_S [ u (u . n) + (p - p_inf) n ] dS   (kinematic, steady,
    incompressible; viscous stress on the outer boundary neglected).
    Classical far-field alternative to surface integration for under-resolved
    near-wall grids. Uses the outermost grid ring; trapezoid rule per edge.
    """
    u = np.asarray(field.u, dtype=np.float64)
    v = np.asarray(field.v, dtype=np.float64)
    p = np.asarray(field.p, dtype=np.float64)
    dx, dy = field.domain.dx, field.domain.dy

    a = np.deg2rad(float(case.bc.aoa_deg))
    ca, sa = np.cos(a), np.sin(a)
    u_inf = float(case.bc.u_inf)
    # p_inf: mean pressure over the inflow (left) edge, robust gauge choice.
    p_inf = float(p[:, 0].mean())

    def edge_flux(us, vs, ps, nxe, nye, dl):
        # momentum flux + pressure term along one edge (arrays over the edge)
        un = us * nxe + vs * nye
        fx = us * un + (ps - p_inf) * nxe
        fy = vs * un + (ps - p_inf) * nye
        return -np.trapezoid(fx, dx=dl), -np.trapezoid(fy, dx=dl)

    Fx = Fy = 0.0
    # left edge (n = -x), right edge (n = +x)
    for j, nxe in ((0, -1.0), (-1, 1.0)):
        gx, gy = edge_flux(u[:, j], v[:, j], p[:, j], nxe, 0.0, dy)
        Fx += gx
        Fy += gy
    # bottom edge (n = -y), top edge (n = +y)
    for i, nye in ((0, -1.0), (-1, 1.0)):
        gx, gy = edge_flux(u[i, :], v[i, :], p[i, :], 0.0, nye, dx)
        Fx += gx
        Fy += gy

    drag = Fx * ca + Fy * sa
    lift = -Fx * sa + Fy * ca
    chord = max(case.reference_length(), _EPS)
    denom = max(0.5 * u_inf * u_inf * chord, _EPS)
    return {"cl": lift / denom, "cd": drag / denom}


def score(coeffs: list[dict], official: dict, names: list[str]) -> dict:
    cd = np.array([c["cd"] for c in coeffs])
    cl = np.array([c["cl"] for c in coeffs])
    ocd = np.array([official[nm]["cd"] for nm in names])
    ocl = np.array([official[nm]["cl"] for nm in names])
    cd_rel = np.abs(cd - ocd) / np.maximum(np.abs(ocd), 1e-12)
    cl_rel = np.abs(cl - ocl) / np.maximum(np.abs(ocl), 1e-12)
    return {
        "rho_D": _spearman(cd.tolist(), ocd.tolist()),
        "rho_L": _spearman(cl.tolist(), ocl.tolist()),
        "cd_rel_err_mean": float(cd_rel.mean()),
        "cd_rel_err_median": float(np.median(cd_rel)),
        "cl_rel_err_mean": float(cl_rel.mean()),
        "cl_rel_err_median": float(np.median(cl_rel)),
    }


def main() -> int:
    t0 = time.time()
    official = json.load(open(LABEL_CACHE, encoding="utf-8"))
    pairs = load_airfrans(root="data", task="full", train=False,
                          resolution=128, limit=200, cache_dir="data/cache",
                          progress=False)
    names = [case.name for case, _ in pairs]
    missing = [nm for nm in names if nm not in official]
    if missing:
        print(f"FATAL: {len(missing)} cases missing official labels")
        return 2
    print(f"[design] {len(pairs)} GT pairs loaded ({time.time()-t0:.0f}s)")

    results = []
    for p_scheme in ("offset", "extrap"):
        for tau_scheme in ("field_abs", "field_signed", "noslip"):
            for d1 in (1.0, 1.5, 2.0):
                coeffs = [variant_coefficients(ref, case, p_scheme,
                                               tau_scheme, d1)
                          for case, ref in pairs]
                s = score(coeffs, official, names)
                rec = {"p_scheme": p_scheme, "tau_scheme": tau_scheme,
                       "d1_cells": d1, **s}
                results.append(rec)
                print(f"[design] p={p_scheme:6s} tau={tau_scheme:12s} "
                      f"d1={d1:.1f}  rho_D={s['rho_D']:+.3f} "
                      f"rho_L={s['rho_L']:+.3f}  "
                      f"cd_rel med={s['cd_rel_err_median']:7.2f}  "
                      f"cl_rel med={s['cl_rel_err_median']:6.2f}")

    # Control-volume (far-field) variant.
    cv_coeffs = [cv_coefficients(ref, case) for case, ref in pairs]
    s = score(cv_coeffs, official, names)
    rec = {"p_scheme": "control_volume", "tau_scheme": "control_volume",
           "d1_cells": 0.0, **s}
    results.append(rec)
    print(f"[design] p={'cv':6s} tau={'cv':12s} d1=  -  "
          f"rho_D={s['rho_D']:+.3f} rho_L={s['rho_L']:+.3f}  "
          f"cd_rel med={s['cd_rel_err_median']:7.2f}  "
          f"cl_rel med={s['cl_rel_err_median']:6.2f}")

    # Rank: primarily by cd magnitude error among variants that keep ranking.
    ranked = sorted(
        results,
        key=lambda r: (r["rho_D"] < 0.80, r["cd_rel_err_median"]),
    )
    legacy = next(r for r in results
                  if r["p_scheme"] == "offset"
                  and r["tau_scheme"] == "field_abs"
                  and r["d1_cells"] == 1.5)
    out = {
        "meta": {
            "what": "GT-field integrator variants vs official AirfRANS labels",
            "n_cases": len(pairs),
            "note": ("All numbers are ground-truth-field integrations; "
                     "seed-independent. Legacy variant = deployed "
                     "force_coefficients (offset/field_abs/1.5)."),
        },
        "legacy": legacy,
        "ranked": ranked,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    best = ranked[0]
    print(f"[design] BEST: p={best['p_scheme']} tau={best['tau_scheme']} "
          f"d1={best['d1_cells']}  rho_D={best['rho_D']:+.3f} "
          f"cd_rel med={best['cd_rel_err_median']:.2f} "
          f"(legacy {legacy['cd_rel_err_median']:.2f})")
    print(f"[design] wrote {OUT_PATH} ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
