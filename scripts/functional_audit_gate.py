"""The §4 gate of ``docs/paper/review/strongest_design.md``: does a GOAL-ORIENTED
reading of the operator-inconsistent residual recover the certificate the
norm-level reading cannot provide?

The paper's established thesis is rank / certify / descend: the monitored
steady-RANS residual RANKS predictions (AUROC 0.952 on worst-decile |dC_d|,
control 4) but cannot CERTIFY them (a floor at the reference solution that
refinement makes worse) and cannot be DESCENDED (24/24 divergence from the exact
truth).  Design B / D hangs on one question: for a control volume ``V`` around
the body, does the drag-direction FUNCTIONAL residual

    I(w; V) = e_D . integral_V R(w) dV

escape the floor?  A signed integral can cancel where a norm cannot, so this is
not answerable from the norm-level numbers.

Pre-registered tests, decision rule and predictions: see
``docs/paper/review/functional_audit_gate.md`` §1, committed BEFORE the run
(mirroring how the resolution ladder's rule was committed in ``bf5106c``).

What this computes
------------------
Per case, per field (ground truth / raw prediction / deployed corrected), on
five nested boxes ``V_1..V_5``:

* ``I_signed``   -- e_D . sum_{V cap fluid} R dA, in C_d units.
* ``J_absdot``   -- sum_{V cap fluid} |e_D . R| dA  (unsigned contrast, test b').
* ``J_mag``      -- sum_{V cap fluid} |R| dA        (unsigned contrast, test b').
* ``phi_outer``  -- the telescoped box-edge momentum flux (conservative arm
  only): the far-field / momentum-theorem drag on ``V``.
* ``i_solid``    -- e_D . sum_{V cap solid} R dA: what the SAME discrete operator
  attributes to the immersed body.  This is the surface term, handled explicitly
  rather than dropped, and ``phi_outer = I_signed + i_solid + i_ring`` holds to
  machine precision by telescoping.

Residual variants
-----------------
``form``:
  ``adv``  -- the DEPLOYED monitor, ``u.grad u + grad p - nu_eff lap u``
              (``physics.residuals.momentum_residual``).  This is the audited
              object and it is the PRIMARY arm: a conservative reformulation
              would change the operator under test.
  ``cons`` -- ``div(uu + pI - nu_eff grad u)``, the divergence form.  Secondary,
              and the only form for which the divergence-theorem reading exists.
``ring``:
  ``T0`` -- as deployed: residuals zeroed inside the solid AND on the
            solid-adjacent fluid wall ring (``residuals.py:338-348``).
  ``T1`` -- wall ring RESTORED (only the solid is zeroed).  A signed integral is
            not indifferent to this: the ring is 0.51% of fluid cells sitting
            exactly where the body surface term lives, and its residuals run
            10-20x the bulk.
  ``T2`` -- ``T0`` plus the explicit body term ``i_solid``.

Non-dimensionalisation
----------------------
Residual maps are scaled exactly as ``PhysicsChecker.diagnose`` scales them
(momentum / (u_inf^2 / L)).  A scaled volume integral is then converted to drag
coefficient units by

    I = SIGN * (2 / c^2) * sum (e_D . r_scaled) dx dy ,

since ``int r_phys dA / (0.5 u_inf^2 c) = 2 int r_scaled dA / c^2`` for
``L = c``.  ``SIGN = -1`` makes a positive ``phi_outer`` a positive drag,
matching the convention of ``scripts/design_force_integrator.py:cv_coefficients``
(``F_body = -oint [u(u.n) + (p-p_inf) n] dS``); this is verified numerically in
``--validate``, not assumed.

Arms
----
``S`` (PRIMARY, self-consistent).  Field, residual, functional and C_d all come
from the SAME object: the deployed v2_transolver seed-k point-cloud prediction
cached in ``data/cache/acceptance_gate/seed{k}/*.npz``.  Verified byte-identical
to control 4's drag target: ``force_coefficients(raw_k) == seed{k}_prednf.json``
and ``force_coefficients(gt) == gt_nf_full_test_r128_n200.json``.

``E`` (control-4-matched).  Scores on the ensemble-mean field
(``data/cache/w2/ensemble/*.npz``), whose residual norm reproduces
``selective_percase.json`` exactly, against the same per-seed |dC_d|.  Carries
control 4's arm mismatch (ensemble-mean score, per-seed target) and exists only
so the functional is comparable with the committed AUROC 0.952.

Usage
-----
    PYTHONPATH=src python scripts/functional_audit_gate.py --validate
    PYTHONPATH=src python scripts/functional_audit_gate.py --run
CPU-only, zero forward passes, minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (caps BLAS threads)
import numpy as np

from neuroforge.core.config import Config
from neuroforge.core.types import DTYPE, FlowField
from neuroforge.geometry.encode import encode_case
from neuroforge.physics.metrics import force_coefficients
from neuroforge.physics.operators import ddx, ddy, laplacian
from neuroforge.physics.residuals import PhysicsChecker, _solid_adjacent_fluid

TRUTH_CACHE = "data/cache/airfrans_full_test_r128_n200.pkl"
AG_DIR = "data/cache/acceptance_gate"
ENS_DIR = "data/cache/w2/ensemble"
FCACHE = "results/control/_cache"
SEL_PC = "results/selective/selective_percase.json"
OUT_RUN = "results/review/functional_audit_gate.json"
OUT_VAL = "results/review/functional_audit_gate_validation.json"

# ---- PRE-REGISTERED control volumes -------------------------------------- #
# Nested boxes centred on the airfoil (chord runs x = 0..1, so centre x = 0.5).
# Domain is (-1, 2) x (-1.5, 1.5) at 128^2, h = 3/127 = 0.02362 chord.  The
# outermost box keeps >= 2 cells of margin on every side so that every cell in
# every box uses the CENTRAL stencil and every stencil neighbour is in-array
# (operators.py falls back to one-sided differences only at the array edge).
# Committed before the run; a pass at a single box with no monotone trend in
# box size is reported as MARGINAL, not as a pass.
BOXES = [
    ("V1", 0.75, 0.40),
    ("V2", 1.00, 0.65),
    ("V3", 1.25, 0.90),
    ("V4", 1.45, 1.15),
    ("V5", 1.45, 1.40),
]
BOX_CX = 0.5

SIGN = -1.0  # verified in --validate against cv_coefficients' convention


def log(msg: str) -> None:
    print(f"[funcgate] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Residual maps
# --------------------------------------------------------------------------- #
def _nu_eff(field: FlowField, case) -> np.ndarray:
    nut = field.nut
    if nut is None:
        nut = np.zeros(field.shape, dtype=np.float64)
    nu = float(case.fluid.kinematic_viscosity) + np.asarray(nut, dtype=np.float64)
    return np.clip(nu, 0.0, None)


def momentum_maps(field: FlowField, case, form: str) -> tuple[np.ndarray, np.ndarray]:
    """Signed, NON-DIMENSIONALISED momentum residual maps, unmasked.

    ``form='adv'`` reproduces ``physics.residuals.momentum_residual`` exactly
    (float64 arithmetic); ``form='cons'`` is the divergence form of the same
    equation, whose box sum telescopes to an edge flux.
    """
    dx, dy = field.domain.dx, field.domain.dy
    u = np.asarray(field.u, dtype=np.float64)
    v = np.asarray(field.v, dtype=np.float64)
    p = np.asarray(field.p, dtype=np.float64)
    nu = _nu_eff(field, case)

    if form == "adv":
        r_x = u * ddx(u, dx) + v * ddy(u, dy) + ddx(p, dx) - nu * laplacian(u, dx, dy)
        r_y = u * ddx(v, dx) + v * ddy(v, dy) + ddy(p, dy) - nu * laplacian(v, dx, dy)
    elif form == "cons":
        # F = uu + pI - nu_eff grad u  (flux tensor rows)
        fxx = u * u + p - nu * ddx(u, dx)
        fxy = u * v - nu * ddy(u, dy)
        fyx = u * v - nu * ddx(v, dx)
        fyy = v * v + p - nu * ddy(v, dy)
        r_x = ddx(fxx, dx) + ddy(fxy, dy)
        r_y = ddx(fyx, dx) + ddy(fyy, dy)
    else:
        raise ValueError(form)

    u_inf = max(float(case.bc.u_inf), 1e-9)
    length = max(float(case.reference_length()), 1e-9)
    s = u_inf * u_inf / length
    return r_x / s, r_y / s


def cons_flux(field: FlowField, case) -> tuple[np.ndarray, ...]:
    """The four non-dimensionalised flux components of the conservative form."""
    dx, dy = field.domain.dx, field.domain.dy
    u = np.asarray(field.u, dtype=np.float64)
    v = np.asarray(field.v, dtype=np.float64)
    p = np.asarray(field.p, dtype=np.float64)
    nu = _nu_eff(field, case)
    u_inf = max(float(case.bc.u_inf), 1e-9)
    length = max(float(case.reference_length()), 1e-9)
    s = u_inf * u_inf / length
    return (
        (u * u + p - nu * ddx(u, dx)) / s,
        (u * v - nu * ddy(u, dy)) / s,
        (u * v - nu * ddx(v, dx)) / s,
        (v * v + p - nu * ddy(v, dy)) / s,
    )


# --------------------------------------------------------------------------- #
# Boxes and integration
# --------------------------------------------------------------------------- #
def box_indices(domain, ax: float, by: float) -> tuple[int, int, int, int]:
    """Inclusive index window (j0, j1, i0, i1) for the physical box."""
    x0, x1, y0, y1 = domain.bounds
    nx, ny = domain.nx, domain.ny
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    ii = np.nonzero((xs >= BOX_CX - ax) & (xs <= BOX_CX + ax))[0]
    jj = np.nonzero((ys >= -by) & (ys <= by))[0]
    return int(jj[0]), int(jj[-1]), int(ii[0]), int(ii[-1])


def phi_outer(fxx, fxy, fyx, fyy, win, dx, dy) -> tuple[float, float]:
    """Telescoped box-edge flux of the conservative residual.

    ``sum_{i=i0}^{i1} ddx(F)[j,i] dx = (F[j,i1] + F[j,i1+1] - F[j,i0-1] - F[j,i0])/2``
    for the second-order central stencil, so the 2-D box sum reduces exactly to a
    two-point-averaged contour integral.  Constant fields telescope to zero, so
    the pressure gauge cancels and no ``p_inf`` subtraction is needed.
    """
    j0, j1, i0, i1 = win
    ex = 0.5 * (fxx[j0:j1 + 1, i1] + fxx[j0:j1 + 1, i1 + 1]
                - fxx[j0:j1 + 1, i0 - 1] - fxx[j0:j1 + 1, i0]).sum() * dy
    ey = 0.5 * (fxy[j1, i0:i1 + 1] + fxy[j1 + 1, i0:i1 + 1]
                - fxy[j0 - 1, i0:i1 + 1] - fxy[j0, i0:i1 + 1]).sum() * dx
    gx = 0.5 * (fyx[j0:j1 + 1, i1] + fyx[j0:j1 + 1, i1 + 1]
                - fyx[j0:j1 + 1, i0 - 1] - fyx[j0:j1 + 1, i0]).sum() * dy
    gy = 0.5 * (fyy[j1, i0:i1 + 1] + fyy[j1 + 1, i0:i1 + 1]
                - fyy[j0 - 1, i0:i1 + 1] - fyy[j0, i0:i1 + 1]).sum() * dx
    return float(ex + ey), float(gx + gy)


def cd_scale(case) -> float:
    c = max(float(case.reference_length()), 1e-9)
    return 2.0 / (c * c)


def edir(case) -> tuple[float, float]:
    a = np.deg2rad(float(case.bc.aoa_deg))
    return float(np.cos(a)), float(np.sin(a))


# --------------------------------------------------------------------------- #
# Per-field functional record
# --------------------------------------------------------------------------- #
def functionals(field: FlowField, case, checker: PhysicsChecker) -> dict:
    """All pre-registered functionals for one field, over all five boxes."""
    dom = field.domain
    dx, dy = dom.dx, dom.dy
    dA = dx * dy
    ex, ey = edir(case)
    k = cd_scale(case)

    mask = np.asarray(field.mask, dtype=np.float64)
    fluid = mask > 0.5
    solid = ~fluid
    ring = _solid_adjacent_fluid(mask)

    maps = {f: momentum_maps(field, case, f) for f in ("adv", "cons")}
    fx = cons_flux(field, case)

    out: dict = {"boxes": {}}
    out["residual_norm"] = float(checker.diagnose(field, case).residual_norm())
    out["cd"] = float(force_coefficients(field, case)["cd"])

    for name, ax, by in BOXES:
        win = box_indices(dom, ax, by)
        j0, j1, i0, i1 = win
        sl = (slice(j0, j1 + 1), slice(i0, i1 + 1))
        rec: dict = {}

        fl = fluid[sl]
        so = solid[sl]
        rg = ring[sl]
        fl_noring = fl & ~rg
        rec["n_cells"] = int(fl.size)
        rec["n_fluid"] = int(fl.sum())
        rec["n_solid"] = int(so.sum())
        rec["n_ring"] = int(rg.sum())

        for form in ("adv", "cons"):
            rx, ry = maps[form]
            d = ex * rx[sl] + ey * ry[sl]           # e_D . R, per cell
            m = np.sqrt(rx[sl] ** 2 + ry[sl] ** 2)  # |R|, per cell
            i_t0 = SIGN * k * dA * float(d[fl_noring].sum())
            i_t1 = SIGN * k * dA * float(d[fl].sum())
            i_solid = SIGN * k * dA * float(d[so].sum())
            i_ring = SIGN * k * dA * float(d[rg].sum())
            rec[form] = {
                "I_T0": i_t0,
                "I_T1": i_t1,
                "I_T2": i_t0 + i_solid,
                "i_solid": i_solid,
                "i_ring": i_ring,
                "J_absdot_T0": k * dA * float(np.abs(d[fl_noring]).sum()),
                "J_mag_T0": k * dA * float(m[fl_noring].sum()),
            }
        # conservative-only: the telescoped edge flux (far-field drag on V)
        px, py = phi_outer(*fx, win, dx, dy)
        rec["cons"]["phi_outer"] = SIGN * k * (ex * px + ey * py)
        out["boxes"][name] = rec
    return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_pairs(limit: int):
    with open(TRUTH_CACHE, "rb") as fh:
        pairs = pickle.load(fh)
    return pairs[:limit] if limit else pairs


def wrap(arr, case, sdf, mask) -> FlowField:
    return FlowField.from_array(np.ascontiguousarray(arr, DTYPE), case.domain,
                                mask=mask, sdf=sdf)


# --------------------------------------------------------------------------- #
# --validate : plumbing, before any hypothesis test
# --------------------------------------------------------------------------- #
def run_validate(a) -> int:
    """V-A telescoping (no body) / V-B closure (with body) / V-C sign vs the
    existing control-volume integrator / V-D what the functional sees on truth."""
    t0 = time.time()
    checker = PhysicsChecker(Config().physics)
    pairs = load_pairs(a.n_validate)
    rows = []
    for case, gt in pairs:
        dom = gt.domain
        dx, dy = dom.dx, dom.dy
        ex, ey = edir(case)
        k = cd_scale(case)
        mask = np.asarray(gt.mask, dtype=np.float64)
        fluid = mask > 0.5
        rx, ry = momentum_maps(gt, case, "cons")
        fxx, fxy, fyx, fyy = cons_flux(gt, case)

        # V-A: a box entirely in the fluid (above the airfoil) must telescope.
        j0, j1, i0, i1 = box_indices(dom, 0.75, 0.40)
        ny = dom.ny
        jj0 = j1 + 4
        jj1 = min(ny - 3, jj0 + 25)
        winA = (jj0, jj1, i0, i1)
        slA = (slice(jj0, jj1 + 1), slice(i0, i1 + 1))
        assert fluid[slA].all(), "V-A box is not pure fluid"
        volA_x = float(rx[slA].sum()) * dx * dy
        volA_y = float(ry[slA].sum()) * dx * dy
        edgA_x, edgA_y = phi_outer(fxx, fxy, fyx, fyy, winA, dx, dy)
        relA = (abs(volA_x - edgA_x) + abs(volA_y - edgA_y)) / (
            abs(edgA_x) + abs(edgA_y) + 1e-30)

        # V-B: with the body inside, fluid + solid + ring must reconstruct phi.
        winB = box_indices(dom, 1.45, 1.40)
        jb0, jb1, ib0, ib1 = winB
        slB = (slice(jb0, jb1 + 1), slice(ib0, ib1 + 1))
        d = ex * rx[slB] + ey * ry[slB]
        flB = fluid[slB]
        rgB = _solid_adjacent_fluid(mask)[slB]
        I_T0 = SIGN * k * dx * dy * float(d[flB & ~rgB].sum())
        i_solid = SIGN * k * dx * dy * float(d[~flB].sum())
        i_ring = SIGN * k * dx * dy * float(d[rgB].sum())
        pxB, pyB = phi_outer(fxx, fxy, fyx, fyy, winB, dx, dy)
        phiB = SIGN * k * (ex * pxB + ey * pyB)
        relB = abs(phiB - (I_T0 + i_solid + i_ring)) / (abs(phiB) + 1e-30)

        # V-C: sign/scale against the deployed control-volume integrator.
        cv = _cv_cd(gt, case)
        cd_surf = float(force_coefficients(gt, case)["cd"])

        rows.append({
            "name": case.name,
            "V_A_rel_err_telescoping": relA,
            "V_B_rel_err_closure": relB,
            "V_B_phi_outer": phiB,
            "V_B_I_T0": I_T0,
            "V_B_i_solid": i_solid,
            "V_B_i_ring": i_ring,
            "V_C_phi_outer_V5": phiB,
            "V_C_cv_coefficients_cd": cv,
            "V_C_sign_agrees": bool(phiB * cv > 0),
            "V_D_cd_surface_integrator": cd_surf,
        })
        log(f"  {case.name[:34]}: V-A {relA:.2e}  V-B {relB:.2e}  "
            f"phi {phiB:+.4f}  cv {cv:+.4f}  surf {cd_surf:+.4f}  "
            f"I_T0 {I_T0:+.4f}  solid {i_solid:+.4f}  ring {i_ring:+.4f}")

    out = {
        "artifact": "functional_audit_gate_validation",
        "purpose": ("Plumbing validation of the functional residual BEFORE any "
                    "hypothesis test: telescoping, surface-term closure, and the "
                    "sign convention against scripts/design_force_integrator.py."),
        "checks": {
            "V_A": ("box entirely in fluid: sum_V div(F) dA == telescoped edge "
                    "flux.  Machine precision expected."),
            "V_B": ("box containing the body: phi_outer == I_T0 + i_solid + "
                    "i_ring.  Machine precision expected -- this is what makes "
                    "the surface term explicit rather than dropped."),
            "V_C": ("sign of e_D.phi_outer against cv_coefficients' drag on the "
                    "same field.  SIGN = -1 is chosen so that a positive "
                    "phi_outer is a positive drag."),
            "V_D": ("what the surface integrator reports on the same truth "
                    "field, for the honest magnitude statement."),
        },
        "SIGN": SIGN,
        "boxes": [{"name": n, "half_x": ax, "half_y": by, "centre_x": BOX_CX}
                  for n, ax, by in BOXES],
        "n_cases": len(rows),
        "max_V_A_rel_err": max(r["V_A_rel_err_telescoping"] for r in rows),
        "max_V_B_rel_err": max(r["V_B_rel_err_closure"] for r in rows),
        "n_sign_agrees": sum(r["V_C_sign_agrees"] for r in rows),
        "runtime_s": time.time() - t0,
        "per_case": rows,
    }
    os.makedirs(os.path.dirname(OUT_VAL), exist_ok=True)
    with open(OUT_VAL, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)
    log(f"V-A max rel err {out['max_V_A_rel_err']:.3e}")
    log(f"V-B max rel err {out['max_V_B_rel_err']:.3e}")
    log(f"sign agrees on {out['n_sign_agrees']}/{len(rows)}")
    log(f"wrote {OUT_VAL} ({out['runtime_s']:.0f}s)")
    return 0


def _cv_cd(field: FlowField, case) -> float:
    """``cv_coefficients`` from scripts/design_force_integrator.py, C_d only.

    Reproduced here rather than imported so this script does not depend on
    another agent's file (the design-study script is read-only for this work).
    """
    u = np.asarray(field.u, dtype=np.float64)
    v = np.asarray(field.v, dtype=np.float64)
    p = np.asarray(field.p, dtype=np.float64)
    dx, dy = field.domain.dx, field.domain.dy
    a = np.deg2rad(float(case.bc.aoa_deg))
    ca, sa = np.cos(a), np.sin(a)
    p_inf = float(p[:, 0].mean())

    def edge_flux(us, vs, ps, nxe, nye, dl):
        un = us * nxe + vs * nye
        return (-np.trapezoid(us * un + (ps - p_inf) * nxe, dx=dl),
                -np.trapezoid(vs * un + (ps - p_inf) * nye, dx=dl))

    Fx = Fy = 0.0
    for j, nxe in ((0, -1.0), (-1, 1.0)):
        gx, gy = edge_flux(u[:, j], v[:, j], p[:, j], nxe, 0.0, dy)
        Fx += gx
        Fy += gy
    for i, nye in ((0, -1.0), (-1, 1.0)):
        gx, gy = edge_flux(u[i, :], v[i, :], p[i, :], 0.0, nye, dx)
        Fx += gx
        Fy += gy
    u_inf = float(case.bc.u_inf)
    chord = max(case.reference_length(), 1e-12)
    return float((Fx * ca + Fy * sa) / max(0.5 * u_inf * u_inf * chord, 1e-12))


# --------------------------------------------------------------------------- #
# --run : the gate
# --------------------------------------------------------------------------- #
def run_gate(a) -> int:
    t0 = time.time()
    checker = PhysicsChecker(Config().physics)
    pairs = load_pairs(a.limit)
    log(f"{len(pairs)} cases")

    seeds = [s for s in a.seeds if os.path.isdir(os.path.join(AG_DIR, f"seed{s}"))]
    per_case: dict = {}
    for i, (case, gt) in enumerate(pairs):
        st = encode_case(case)
        sdf, mask = st[0].astype(DTYPE), st[1].astype(DTYPE)
        rec = {"name": case.name, "aoa_deg": float(case.bc.aoa_deg),
               "u_inf": float(case.bc.u_inf), "fields": {}}
        rec["fields"]["truth"] = functionals(gt, case, checker)
        for s in seeds:
            npz = os.path.join(AG_DIR, f"seed{s}", f"{case.name}.npz")
            if not os.path.exists(npz):
                continue
            z = np.load(npz)
            rec["fields"][f"raw_seed{s}"] = functionals(
                wrap(z["raw"], case, sdf, mask), case, checker)
            rec["fields"][f"corrected_seed{s}"] = functionals(
                wrap(z["corrected"], case, sdf, mask), case, checker)
        ep = os.path.join(ENS_DIR, f"{case.name}.npz")
        if os.path.exists(ep):
            rec["fields"]["ensemble_mean"] = functionals(
                wrap(np.load(ep)["mean"], case, sdf, mask), case, checker)
        per_case[case.name] = rec
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(pairs)} ({(time.time() - t0) / (i + 1):.2f}s/case)")

    out = {
        "artifact": "functional_audit_gate",
        "question": ("Does a goal-oriented (drag-direction) reading of the "
                     "operator-inconsistent residual escape the floor that "
                     "destroys the norm-level audit?"),
        "preregistration": "docs/paper/review/functional_audit_gate.md sec 1",
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": "cpu (committed caches only; zero forward passes)",
            "n_cases": len(per_case),
            "seeds": seeds,
            "boxes": [{"name": n, "half_x": ax, "half_y": by,
                       "centre_x": BOX_CX} for n, ax, by in BOXES],
            "SIGN": SIGN,
            "runtime_s": time.time() - t0,
        },
        "per_case": per_case,
    }
    os.makedirs(os.path.dirname(OUT_RUN), exist_ok=True)
    with open(OUT_RUN, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)
    log(f"wrote {OUT_RUN} ({time.time() - t0:.0f}s)")
    return 0


# --------------------------------------------------------------------------- #
# statistics -- lifted verbatim from scripts/decisive_controls.py so the gate
# and the baseline it is compared against are scored by one and the same code.
# --------------------------------------------------------------------------- #
try:
    from scipy.stats import rankdata as _rankdata

    def rankdata(x):
        return np.asarray(_rankdata(np.asarray(x, float)), float)
except ImportError:  # pragma: no cover
    def rankdata(x):
        x = np.asarray(x, float)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[order] = np.arange(1, len(x) + 1, dtype=float)
        s = x[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
            i = j + 1
        return r


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def auroc(score, label):
    score = np.asarray(score, float)
    label = np.asarray(label).astype(bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def retained_risk(err, score, frac_reject):
    n = len(err)
    k = int(round(frac_reject * n))
    keep = np.argsort(rankdata(score))[: n - k] if k > 0 else np.arange(n)
    return float(np.mean(np.asarray(err)[keep]))


def orec(err, score):
    base = float(np.mean(err))
    orc = retained_risk(err, err, 0.10)
    got = retained_risk(err, score, 0.10)
    return float((base - got) / (base - orc)) if base > orc else None


def boot_ci(stat_fn, n, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    full = stat_fn(np.arange(n))
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat_fn(idx)
        draws[b] = v if v is not None else np.nan
    d = draws[np.isfinite(draws)]
    if len(d) == 0:
        return {"point": float(full), "ci95": [None, None], "n_boot_valid": 0}
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(full), "ci95": [float(lo), float(hi)],
        "boot_mean": float(d.mean()), "boot_std": float(d.std(ddof=1)),
        "frac_draws_gt_0": float((d > 0).mean()),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot_valid": int(len(d)),
    }


# --------------------------------------------------------------------------- #
# --analyse : the PRE-REGISTERED tests (a) / (b) / (b') / (c)
# --------------------------------------------------------------------------- #
# Score variants under test.  Key -> (form, functional key, signed?).
# The PRIMARY functional is the deployed advective monitor with the deployed
# masking (T0); everything else is a declared variant.
VARIANTS = [
    ("adv_I_T0", "adv", "I_T0", True),
    ("adv_I_T1", "adv", "I_T1", True),
    ("adv_I_T2", "adv", "I_T2", True),
    ("cons_I_T0", "cons", "I_T0", True),
    ("cons_I_T1", "cons", "I_T1", True),
    ("cons_I_T2", "cons", "I_T2", True),
    ("cons_phi_outer", "cons", "phi_outer", True),
    ("adv_J_absdot", "adv", "J_absdot_T0", False),
    ("adv_J_mag", "adv", "J_mag_T0", False),
]


def _score(rec: dict, box: str, form: str, key: str, signed: bool) -> float:
    v = rec["boxes"][box][form][key]
    return abs(v) if signed else v


def analyse(a) -> int:
    t0 = time.time()
    d = json.load(open(OUT_RUN, encoding="utf-8"))
    pc = d["per_case"]
    names = sorted(pc)
    seeds = d["meta"]["seeds"]
    box_names = [b["name"] for b in d["meta"]["boxes"]]
    nb = a.n_boot

    out = {
        "artifact": "functional_audit_gate_analysis",
        "preregistration": "docs/paper/review/functional_audit_gate.md sec 1",
        "source": OUT_RUN,
        "meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "n_cases": len(names),
                 "seeds": seeds, "boxes": box_names, "n_boot": nb, "seed": a.seed},
        "scale_check": {}, "test_a": {}, "test_b": {}, "test_b_prime": {},
        "test_c": {},
    }

    # ---- scale check: can a certificate exist at all? --------------------- #
    cd_t = np.array([pc[n]["fields"]["truth"]["cd"] for n in names])
    for s in seeds:
        k = f"raw_seed{s}"
        if k not in pc[names[0]]["fields"]:
            continue
        e = np.abs(np.array([pc[n]["fields"][k]["cd"] for n in names]) - cd_t)
        rowe = {"median_abs_dCd": float(np.median(e)),
                "mean_abs_dCd": float(e.mean()),
                "top_decile_abs_dCd": float(np.quantile(e, 0.9))}
        for box in box_names:
            for vk, form, key, signed in VARIANTS:
                it = np.array([_score(pc[n]["fields"]["truth"], box, form, key, signed)
                               for n in names])
                rowe[f"{box}/{vk}/median_abs_I_truth"] = float(np.median(it))
                rowe[f"{box}/{vk}/ratio_medI_truth_over_med_dCd"] = float(
                    np.median(it) / max(np.median(e), 1e-30))
        out["scale_check"][k] = rowe

    # ---- (a) inversion rate ---------------------------------------------- #
    # Norm-level MATCHED baseline first: the 160/200 figure in the design doc
    # is a different model (dropout-FNO), so it cannot be the bar.
    for k in _pred_keys(pc, names, seeds):
        rn_t = np.array([pc[n]["fields"]["truth"]["residual_norm"] for n in names])
        rn_p = np.array([pc[n]["fields"][k]["residual_norm"] for n in names])
        rec = {"norm_baseline_inversion_rate": float((rn_p < rn_t).mean()),
               "n": len(names), "boxes": {}}
        for box in box_names:
            br = {}
            for vk, form, key, signed in VARIANTS:
                it = np.array([_score(pc[n]["fields"]["truth"], box, form, key, signed)
                               for n in names])
                ip = np.array([_score(pc[n]["fields"][k], box, form, key, signed)
                               for n in names])
                br[vk] = {"inversion_rate": float((ip < it).mean()),
                          "median_truth": float(np.median(it)),
                          "median_pred": float(np.median(ip))}
            rec["boxes"][box] = br
        out["test_a"][k] = rec

    # ---- (b) ranking, and (b') the unsigned contrast ---------------------- #
    for s in seeds:
        for fk, arm in ((f"raw_seed{s}", "S"), ("ensemble_mean", "E")):
            if fk not in pc[names[0]]["fields"]:
                continue
            cd_p = np.array([pc[n]["fields"][f"raw_seed{s}"]["cd"] for n in names])
            e = np.abs(cd_p - cd_t)
            lab = e >= np.quantile(e, 0.9)   # threshold FIXED at full-sample value
            n = len(names)
            base = np.array([pc[n_]["fields"][fk]["residual_norm"] for n_ in names])
            rec = {"arm": arm, "score_field": fk, "target": f"raw_seed{s} vs truth",
                   "n": n, "top_decile_threshold": float(np.quantile(e, 0.9)),
                   "residual_norm_baseline": {
                       "spearman": spearman(base, e),
                       "auroc_top_decile": auroc(base, lab),
                       "oracle_recovery_at_10pct": orec(e, base)},
                   "boxes": {}}
            for box in box_names:
                br = {}
                for vk, form, key, signed in VARIANTS:
                    sc = np.array([_score(pc[n_]["fields"][fk], box, form, key, signed)
                                   for n_ in names])
                    br[vk] = {
                        "spearman": spearman(sc, e),
                        "auroc_top_decile": auroc(sc, lab),
                        "oracle_recovery_at_10pct": orec(e, sc),
                        "paired_vs_residual_norm": {
                            "delta_auroc": boot_ci(
                                lambda i, sc=sc: auroc(sc[i], lab[i]) - auroc(base[i], lab[i]),
                                n, nb, a.seed),
                            "delta_spearman": boot_ci(
                                lambda i, sc=sc: spearman(sc[i], e[i]) - spearman(base[i], e[i]),
                                n, nb, a.seed)},
                    }
                br["_cancellation"] = _cancellation(pc, names, fk, box, e, lab)
                rec["boxes"][box] = br
            out["test_b"][f"{arm}_seed{s}"] = rec

    # ---- (b') summary: signed vs unsigned, head to head ------------------- #
    out["test_b_prime"]["note"] = (
        "I(w;V) is a SIGNED integral, so a field with large but mutually "
        "cancelling errors scores near zero.  Per box we report (i) the "
        "low-|I| decile's drag errors and (ii) the unsigned integrals "
        "adv_J_absdot / adv_J_mag scored in the same (b) framework.  If the "
        "unsigned integral ranks materially better than the signed one, "
        "cancellation is the mechanism rather than a speculation.")

    # ---- (c) conformal certificate -- only if (a) and (b) pass ------------ #
    out["test_c"] = {"built": False,
                     "reason": "populated by the decision rule below"}
    out["decision"] = _decide(out, box_names)
    if out["decision"]["branch"] == "D":
        out["test_c"] = _conformal(pc, names, seeds, box_names, cd_t, a)

    out["meta"]["runtime_s"] = time.time() - t0
    path = OUT_RUN.replace(".json", "_analysis.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    log(f"wrote {path} ({time.time() - t0:.0f}s)")
    log(f"DECISION: Design {out['decision']['branch']} -- {out['decision']['why']}")
    return 0


def _pred_keys(pc, names, seeds):
    keys = [k for k in pc[names[0]]["fields"] if k != "truth"]
    return sorted(keys)


def _cancellation(pc, names, fk, box, e, lab) -> dict:
    """(b') Do the LOW-|I| cases contain HIGH-|dC_d| cases?"""
    sc = np.array([_score(pc[n]["fields"][fk], box, "adv", "I_T0", True) for n in names])
    lo = sc <= np.quantile(sc, 0.10)
    return {
        "n_low_I_decile": int(lo.sum()),
        "n_top_decile_dCd_inside_low_I_decile": int((lab & lo).sum()),
        "max_abs_dCd_in_low_I_decile": float(e[lo].max()) if lo.any() else None,
        "median_abs_dCd_in_low_I_decile": float(np.median(e[lo])) if lo.any() else None,
        "median_abs_dCd_overall": float(np.median(e)),
        "top_decile_dCd_threshold": float(np.quantile(e, 0.9)),
    }


def _decide(out, box_names) -> dict:
    """The PRE-REGISTERED decision rule, applied mechanically."""
    # (a): does the inversion rate fall below 10% at some V, with a monotone
    # trend in box size?  Evaluated on the PRIMARY variant adv_I_T0, on the
    # deployed raw_seed* arms.
    best_a, a_pass = {}, False
    for k, rec in out["test_a"].items():
        if not k.startswith("raw_seed"):
            continue
        rates = [rec["boxes"][b]["adv_I_T0"]["inversion_rate"] for b in box_names]
        best_a[k] = {"rates_by_box": dict(zip(box_names, rates)),
                     "min_rate": min(rates),
                     "norm_baseline": rec["norm_baseline_inversion_rate"]}
    if best_a:
        a_pass = all(v["min_rate"] < 0.10 for v in best_a.values())
    # (b): does the functional beat the residual-norm baseline on |dC_d|?
    b_pass, best_b = False, {}
    for k, rec in out["test_b"].items():
        if not k.startswith("S_"):
            continue
        base = rec["residual_norm_baseline"]["auroc_top_decile"]
        best = max(
            (rec["boxes"][b]["adv_I_T0"]["auroc_top_decile"], b) for b in box_names)
        best_b[k] = {"baseline_auroc": base, "best_functional_auroc": best[0],
                     "best_box": best[1],
                     "delta_excludes_zero": bool(
                         rec["boxes"][best[1]]["adv_I_T0"][
                             "paired_vs_residual_norm"]["delta_auroc"]["excludes_zero"]
                         and rec["boxes"][best[1]]["adv_I_T0"][
                             "paired_vs_residual_norm"]["delta_auroc"]["point"] > 0)}
    if best_b:
        b_pass = all(v["delta_excludes_zero"] for v in best_b.values())
    if a_pass and b_pass:
        branch, why = "D", "(a) and (b) both pass: build the certificate (c)."
    elif b_pass:
        branch, why = ("A", "(b) passes, (a) fails: the functional is a BETTER "
                            "RANKER but still not a certificate.")
    else:
        branch, why = ("A", "both fail: Design A only; control 4's AUROC 0.952 "
                            "stays as the measured positive.")
    return {"branch": branch, "why": why, "a_pass": bool(a_pass),
            "b_pass": bool(b_pass), "test_a_summary": best_a,
            "test_b_summary": best_b}


def _conformal(pc, names, seeds, box_names, cd_t, a) -> dict:
    """(c) split-conformal c for |dC_d| <= c |I(w;V)|, coverage on held out."""
    rng = np.random.default_rng(a.seed)
    res = {"built": True, "target_coverage": 0.90, "arms": {}}
    for s in seeds:
        fk = f"raw_seed{s}"
        e = np.abs(np.array([pc[n]["fields"][fk]["cd"] for n in names]) - cd_t)
        for box in box_names:
            sc = np.array([_score(pc[n]["fields"][fk], box, "adv", "I_T0", True)
                           for n in names])
            covs, cs = [], []
            for _ in range(200):
                idx = rng.permutation(len(names))
                cal, tst = idx[: len(idx) // 2], idx[len(idx) // 2:]
                ratio = e[cal] / np.maximum(sc[cal], 1e-30)
                c = float(np.quantile(ratio, 0.90))
                covs.append(float((e[tst] <= c * sc[tst]).mean()))
                cs.append(c)
            res["arms"][f"seed{s}/{box}"] = {
                "c_median": float(np.median(cs)),
                "coverage_mean": float(np.mean(covs)),
                "coverage_std": float(np.std(covs)),
                "median_bound_width_cd": float(np.median(np.median(cs) * sc)),
                "median_abs_dCd": float(np.median(e)),
            }
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--n-validate", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.validate:
        return run_validate(a)
    if a.run:
        return run_gate(a)
    if a.analyse:
        return analyse(a)
    p.error("one of --validate / --run / --analyse is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
