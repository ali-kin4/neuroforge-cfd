"""Is the residual floor a rasterisation artifact? Decomposition + artifact controls.

Companion to ``scripts/floor_resolution_ladder.py`` (the pre-registered ladder).
That script asks *whether* ``||R_h(u*)||`` decays under refinement. This one asks
*why*, and closes the three artifact explanations a CFD reviewer will reach for.
The two are independent implementations of the same measurement and are meant to
be read together: agreement between them is a replication, not a duplication.

The objection being killed
--------------------------
The residual-floor theorem (docs/paper/sections/residual_floor_theorem.tex) rests
on (H2): the monitored discrete steady-RANS operator R_h is NONZERO at the
dataset's ground truth (mean 0.192 over 200 AirfRANS test cases at 128^2), while a
uniform freestream gives EXACTLY zero. The theorem file itself attributes part of
the floor to the grid -- "on a 128^2 grid the boundary layer is sub-cell, so the
rasterised truth itself carries near-wall velocity". A reviewer will therefore say
the whole negative result is an artifact of an under-resolved grid. If the floor
decays to zero, leg (i) is a grid artifact and cannot be the paper's headline.

The SAME AirfRANS point clouds are re-rasterised from the raw cloud at every rung
(one Delaunay triangulation per case, reused across rungs, so h is provably the
only thing that varies) and the floor is measured and decomposed at each.

WHAT IS MEASURED AT EVERY RUNG
------------------------------
native      RMS over all fluid cells of sqrt(c^2 + mx^2 + my^2) -- byte-identical
            to scripts/probe_residual_floor.py, i.e. the published number.
band_b      the same RMS restricted to fluid cells with sdf > b chord, never on the
            monitor's zeroed wall ring. THIS IS THE PRIMARY METRIC. The native one
            is NOT resolution-comparable: `diagnose` zeroes a ONE-CELL wall ring
            whose physical thickness shrinks with h, so the averaging set changes
            between rungs and refinement silently admits more of the steepest
            near-wall region. A band is a fixed physical region at every rung.

THE THREE ARTIFACT CONTROLS (this is what makes the answer defensible)
---------------------------------------------------------------------
1. PIPELINE NULL.  A manufactured potential flow (uniform + doublet + point vortex,
   all singularities inside the body) with Bernoulli pressure p = -|u|^2/2. Since
   u = grad(phi): lap(u) = 0 and u.grad(u) = grad(|u|^2/2), so the continuum
   residual is EXACTLY zero for ANY nu_t.
     mms_analytic  evaluated directly on the grid -> PURE TRUNCATION; must show
                   order p ~ 2. This is the measurement gate: if it fails, no
                   verdict can be issued.
     mms_raster    sampled at the case's OWN point-cloud positions and pushed
                   through the IDENTICAL LinearNDInterpolator pipeline -> the
                   pipeline null. Same cloud, same triangulation, same interpolator,
                   same query grid; only the field values differ. Any rise HERE is
                   the machinery, not the physics.
   Caveat, stated so it is not over-read: the MMS field is smooth at the cloud
   scale, so this is a null for the pipeline acting on a RESOLVABLE field. It does
   not by itself prove the pipeline is innocent on a field with sub-cell content.
   Control 2 covers that.
2. SMOOTHNESS CLASS.  Past the cloud spacing we would be twice-differencing a C0
   piecewise-linear interpolant, whose kinks sit at fixed physical triangle edges,
   so a second difference across one scales like ds/h. Re-rasterising with
   CloughTocher (C1, same triangulation) discriminates: if the floor is kink-driven,
   C1 moves it a lot; if it is field content, C1 barely moves it.
3. CLOUD RESOLUTION.  Every rung logs h / s_local, with s_local the median
   nearest-neighbour spacing of the source cloud INSIDE the measurement band. Rungs
   with h < s_local are flagged interpolant-dominated and EXCLUDED from the order fit.

THE MECHANISTIC DECOMPOSITION
-----------------------------
M_gradnu    ||2 dx(nu_eff) dx(u) + dy(nu_eff)(dy(u) + dx(v))|| and its y twin --
            the term the monitored operator DROPS. The incompressible eddy-viscosity
            stress divergence is div(nu_eff (grad u + grad u^T)); the code keeps only
            nu_eff*lap(u). A MODEL omission: it cannot vanish with h.
M_divterm   nu_eff * grad(div U), the remaining piece of the same divergence. It
            would vanish for an exactly divergence-free field, so on real data it
            measures how far the rasterised reference is from discretely
            divergence-free under OUR stencil. Reported separately rather than
            assumed away -- the discrete field is not divergence free.
repairedgn  the floor with the MODEL omission repaired (gradnu added back).
repaired    the floor with BOTH pieces added back. What remains is a LOWER BOUND on
            the reference-operator mismatch: AirfRANS is Spalart-Allmaras on
            finite-volume fluxes on a body-fitted mesh, so its solution never
            satisfies our cell-centred Cartesian operator at ANY h. Lower bound, not
            an exhaustive accounting -- SA source terms, FV flux reconstruction and
            mesh non-orthogonality are not modelled here.
term_*      the momentum residual split into convective, pressure-gradient and
            viscous parts, so the rise (or fall) can be attributed to a term rather
            than merely plotted.

THE NO-SLIP TERM (leg (A) of the lambda-sweep)
----------------------------------------------
bc_violation is TWO additive terms: a proximity-weighted no-slip band AND the
far-field mismatch on the outer one-cell ring. The uniform field has EXACTLY zero
far-field mismatch by construction, while the rasterised truth need not. They are
split here, because the theorem attributes the 0.0073-vs-0.0053 ordering entirely
to the sub-cell boundary layer, and that attribution has to be checked rather than
assumed. Whenever the ordering flips we report the crossover weight
    lambda* = ||R_h(u*)||^2 / (bc2_uniform - bc2_truth)
so a technical break (lambda* ~ 1e6) is not confused with a material one.

PRE-DECLARED DECISION RULE (fixed before the ladder was run)
------------------------------------------------------------
Fit log||r*|| vs log h over trustworthy rungs only, on the primary band metric:
  DECAY   p >= 1.5 AND finest rung <= 30% of the 128^2 value
          -> leg (i) is a grid artifact; demote it to a remark.
  PLATEAU |p| < 0.5, OR the curve flattens with finest rung >= 60% of 128^2
          -> the general claim stands.
  PARTIAL otherwise; report the Richardson h->0 limit.
A negative p (the floor RISING with refinement) falls under PLATEAU by this rule --
it is the opposite of the decay that would sink the paper -- but it is reported as
a rise, with its own mechanism, not laundered into the word "plateau".
This is a WITHIN-CASE convergence study: every case is its own ladder, every
per-case ladder is reported, plus the count of cases plateauing vs decaying.

Usage
-----
    python scripts/floor_resolution_decomposition.py --gate     # gates only
    python scripts/floor_resolution_decomposition.py --n 16     # full ladder
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

from neuroforge.core.types import (
    DTYPE,
    BoundaryConditions,
    Domain,
    FlowCase,
    FlowField,
    FluidProperties,
    Geometry,
)
# Private imports are deliberate: reusing the loader's OWN crop and surface-loop
# ordering is what makes the 128^2 rung REPRODUCE the published per-case numbers
# rather than merely resemble them. See gate 1.
from neuroforge.data.airfrans_loader import AIRFRANS_NU, _CROP, _order_surface_loop
from neuroforge.geometry.sdf import signed_distance, solid_mask
from neuroforge.physics.operators import ddx, ddy
from neuroforge.physics.residuals import PhysicsChecker, _solid_adjacent_fluid

PC_CACHE = "data/cache/airfrans_pc_full_test_n100.pkl"
REF_JSON = "results/certificates/residual_floor_realdata.json"
OUT_JSON = "results/certificates/floor_resolution_decomposition.json"

RUNGS = (128, 181, 256, 362, 512)
BANDS = (0.05, 0.10, 0.25)
CUBIC_RUNGS = (128, 256, 512)


# --------------------------------------------------------------------------- #
# Manufactured solution: an EXACT steady-NS field for any nu_eff
# --------------------------------------------------------------------------- #


def mms_velocity(x, y, u_vec, z0, m, k):
    """Potential flow: uniform + doublet + point vortex, all singular at ``z0``.

    Complex potential ``W(z) = (Ux - i Uy) z + m/(z-z0) + i k log(z-z0)`` gives
    ``dW/dz = u - i v``. Because ``W`` is analytic, ``u`` is divergence-free AND
    irrotational, so ``lap(u) = 0`` and ``u.grad(u) = grad(|u|^2/2)``. With the
    Bernoulli pressure below the continuum residual is identically zero for ANY
    viscosity field -- which is exactly why it isolates truncation.
    """
    z = np.asarray(x, np.float64) + 1j * np.asarray(y, np.float64)
    d = z - z0
    d = np.where(np.abs(d) < 1e-9, 1e-9 + 0j, d)
    w = (u_vec[0] - 1j * u_vec[1]) - m / (d * d) + 1j * k / d
    return np.real(w), -np.imag(w)


def mms_pressure(u, v, u_vec):
    """Bernoulli (kinematic) pressure, zero in the undisturbed freestream."""
    q2 = float(u_vec[0] ** 2 + u_vec[1] ** 2)
    return 0.5 * q2 - 0.5 * (u * u + v * v)


def mms_nut(x, y, z0, amp, scale):
    """A smooth, strictly positive eddy viscosity used only for the unit test."""
    r2 = (np.asarray(x, np.float64) - z0.real) ** 2 + (np.asarray(y, np.float64) - z0.imag) ** 2
    return amp * np.exp(-r2 / (scale * scale))


# --------------------------------------------------------------------------- #
# The omitted stress-divergence terms
# --------------------------------------------------------------------------- #


def omitted_terms(field, nu_lam):
    r"""The two pieces of ``div(nu_eff (grad u + grad u^T))`` the monitor drops.

    The code's momentum residual keeps only ``nu_eff * lap(u)``. Expanding the full
    incompressible eddy-viscosity stress divergence, the x-component is

        nu_eff*lap(u) + nu_eff*d/dx(div U) + [2 dxnu*dxu + dynu*(dyu + dxv)]
                        \_____ divterm ____/  \________ gradnu ____________/

    Incompressibility is NOT used to drop ``divterm``: the discrete field does not
    satisfy it. Returns ``(gx, gy, dxt, dyt)`` -- the gradnu and divterm parts.
    """
    dx, dy = field.domain.dx, field.domain.dy
    u = np.asarray(field.u, np.float64)
    v = np.asarray(field.v, np.float64)
    nut = np.asarray(field.nut if field.nut is not None else 0.0, np.float64)
    nu_eff = np.clip(float(nu_lam) + nut, 0.0, None)

    dudx, dudy = ddx(u, dx), ddy(u, dy)
    dvdx, dvdy = ddx(v, dx), ddy(v, dy)
    dnx, dny = ddx(nu_eff, dx), ddy(nu_eff, dy)

    gx = 2.0 * dnx * dudx + dny * (dudy + dvdx)
    gy = dnx * (dvdx + dudy) + 2.0 * dny * dvdy

    div = dudx + dvdy
    dxt = nu_eff * ddx(div, dx)
    dyt = nu_eff * ddy(div, dy)
    return gx, gy, dxt, dyt


# --------------------------------------------------------------------------- #
# The no-slip / far-field split of bc_violation
# --------------------------------------------------------------------------- #


def bc_split(field, case):
    """Split :func:`bc_violation` into its no-slip and far-field-ring parts.

    Mirrors residuals.bc_violation exactly, but returns the two additive
    contributions separately. The uniform field has an identically zero far-field
    part by construction, so this is what decides whether leg (A) of the
    lambda-sweep is really about a sub-cell boundary layer or about the crop's
    outer boundary.
    """
    from neuroforge.physics.residuals import _surface_proximity_weight

    ny, nx = field.shape
    u = np.asarray(field.u, DTYPE)
    v = np.asarray(field.v, DTYPE)
    speed = np.sqrt(u * u + v * v)
    fluid = np.asarray(field.mask if field.mask is not None else 1.0, DTYPE)

    prox = _surface_proximity_weight(field).astype(np.float64)
    noslip = prox * speed.astype(np.float64) * (fluid > 0.5)
    wall_layer = _solid_adjacent_fluid(fluid)
    noslip[wall_layer] = np.maximum(noslip[wall_layer], speed[wall_layer].astype(np.float64))

    u_in, v_in = case.bc.inlet_vector()
    dev = np.sqrt((u - DTYPE(u_in)) ** 2 + (v - DTYPE(v_in)) ** 2).astype(np.float64)
    border = np.zeros((ny, nx), dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border &= fluid > 0.5
    far = np.zeros((ny, nx), np.float64)
    far[border] = dev[border]
    return noslip, far


# --------------------------------------------------------------------------- #
# Per-case ladder
# --------------------------------------------------------------------------- #


class CaseLadder:
    """One AirfRANS simulation, rasterisable at any resolution.

    The Delaunay triangulation, the surface loop and the boundary conditions are
    built ONCE and reused at every rung, so h is provably the only thing varying.
    """

    def __init__(self, pc):
        self.name = pc.name
        self.pos = np.asarray(pc.pos, np.float64)
        self.targets = np.asarray(pc.targets, np.float64)  # u, v, p, nut
        feats = np.asarray(pc.features, np.float64)
        u_in = feats[:, 2:4]
        normals = feats[:, 5:7]

        uin = u_in[np.isfinite(u_in).all(axis=1)]
        vec = uin.mean(axis=0) if uin.size else np.array([1.0, 0.0])
        if not np.all(np.isfinite(vec)) or float(np.hypot(*vec)) < 1e-9:
            vec = np.array([1.0, 0.0])
        self.u_vec = vec
        self.u_inf = float(np.hypot(*vec))
        self.aoa = float(np.degrees(np.arctan2(vec[1], vec[0])))
        self.reynolds = float(self.u_inf * 1.0 / AIRFRANS_NU)

        on_wall = np.linalg.norm(normals, axis=1) > 1e-8
        wall_pts = self.pos[on_wall]
        if wall_pts.shape[0] < 4:
            order = np.argsort(feats[:, 4])[: max(64, self.pos.shape[0] // 50)]
            wall_pts = self.pos[order]
        self.geom = Geometry(name=self.name, surface_points=_order_surface_loop(wall_pts))
        self._loop_hash = hash(self.geom.surface_points.tobytes())

        from scipy.spatial import Delaunay

        self.tri = Delaunay(self.pos)

        # MMS singularity at the body centroid: guaranteed inside, hence masked.
        c = self.geom.surface_points.astype(np.float64).mean(axis=0)
        self.z0 = complex(c[0], c[1])
        self.m = 5e-4 * self.u_inf
        self.k = 1e-2 * self.u_inf

        xmin, xmax, ymin, ymax = _CROP
        inside = ((self.pos[:, 0] >= xmin) & (self.pos[:, 0] <= xmax)
                  & (self.pos[:, 1] >= ymin) & (self.pos[:, 1] <= ymax))
        self.n_in_crop = int(inside.sum())

        from scipy.spatial import cKDTree

        sub = self.pos[inside]
        if sub.shape[0] > 2:
            d, _ = cKDTree(sub).query(sub, k=2)
            self.s_crop = float(np.median(d[:, 1]))
        else:
            self.s_crop = float("nan")
        self._band_spacing: dict[float, float] = {}

    def spacing_in_band(self, band):
        """Median nearest-neighbour spacing of the source cloud within ``sdf > band``.

        This is the ladder's own resolution limit: once ``h`` drops below it we are
        differentiating a reconstruction rather than sampling a better solution.
        """
        if band in self._band_spacing:
            return self._band_spacing[band]
        xmin, xmax, ymin, ymax = _CROP
        d_wall = self._signed_dist_points()
        sel = ((self.pos[:, 0] >= xmin) & (self.pos[:, 0] <= xmax)
               & (self.pos[:, 1] >= ymin) & (self.pos[:, 1] <= ymax) & (d_wall > band))
        sub = self.pos[sel]
        if sub.shape[0] < 3:
            val = float("nan")
        else:
            from scipy.spatial import cKDTree

            d, _ = cKDTree(sub).query(sub, k=2)
            val = float(np.median(d[:, 1]))
        self._band_spacing[band] = val
        return val

    def _signed_dist_points(self):
        if not hasattr(self, "_sdp"):
            from neuroforge.geometry.sdf import _closed_loop, _unsigned_distance

            self._sdp = _unsigned_distance(self.pos, _closed_loop(self.geom))
        return self._sdp

    def rasterise(self, n, method="linear", values=None):
        """Rasterise onto an ``n x n`` crop with the loader's exact pipeline."""
        from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

        domain = Domain(bounds=_CROP, nx=n, ny=n)
        X, Y = domain.grid()
        xi = np.stack([X.ravel(), Y.ravel()], axis=1)
        vals = self.targets if values is None else values
        if method == "cubic":
            interp = CloughTocher2DInterpolator(self.tri, vals, fill_value=0.0)
        else:
            interp = LinearNDInterpolator(self.tri, vals, fill_value=0.0)
        out = np.asarray(interp(xi), np.float64)
        bad = np.isnan(out).any(axis=1)
        if bad.any():
            from scipy.interpolate import NearestNDInterpolator

            out[bad] = np.asarray(NearestNDInterpolator(self.pos, vals)(xi[bad]), np.float64)
        return out.reshape(n, n, vals.shape[1]).transpose(2, 0, 1).astype(DTYPE), domain

    def geometry_on(self, n):
        domain = Domain(bounds=_CROP, nx=n, ny=n)
        assert hash(self.geom.surface_points.tobytes()) == self._loop_hash, \
            "surface loop mutated between rungs -- h is not the only variable"
        return signed_distance(self.geom, domain), solid_mask(self.geom, domain), domain

    def case_on(self, domain):
        return FlowCase(
            geometry=self.geom,
            bc=BoundaryConditions(u_inf=self.u_inf, aoa_deg=self.aoa, reynolds=self.reynolds),
            fluid=FluidProperties(density=1.0, kinematic_viscosity=AIRFRANS_NU),
            domain=domain, name=self.name,
        )


def _pack_field(raster, sdf, mask, domain, kind):
    solid = mask < 0.5
    return FlowField(
        domain=domain,
        u=np.where(solid, 0.0, raster[0]).astype(DTYPE),
        v=np.where(solid, 0.0, raster[1]).astype(DTYPE),
        p=np.asarray(raster[2], DTYPE),
        nut=np.where(solid, 0.0, np.maximum(raster[3], 0.0)).astype(DTYPE),
        mask=mask, sdf=sdf, meta={"kind": kind},
    )


def _rms(m2, sel):
    return float(np.sqrt(np.mean(m2[sel]))) if np.any(sel) else float("nan")


def measure(field, case, checker, nu_lam, bands):
    """All rung metrics for one field: native + banded floor, term split, bc split."""
    diag = checker.diagnose(field, case)
    cont = np.asarray(diag.continuity, np.float64)
    mx = np.asarray(diag.momentum_x, np.float64)
    my = np.asarray(diag.momentum_y, np.float64)
    m2 = cont * cont + mx * mx + my * my

    fluid = np.asarray(field.mask, np.float64) > 0.5
    sdf = np.asarray(field.sdf, np.float64)
    ring = _solid_adjacent_fluid(np.asarray(field.mask, DTYPE))

    out = {"native": _rms(m2, fluid),
           "native_continuity": _rms(cont * cont, fluid),
           "native_momentum": _rms(mx * mx + my * my, fluid)}
    for b in bands:
        sel = fluid & (sdf > b) & (~ring)
        out[f"band_{b:g}"] = _rms(m2, sel)
        out[f"band_{b:g}_n"] = int(sel.sum())
        out[f"band_{b:g}_continuity"] = _rms(cont * cont, sel)
        out[f"band_{b:g}_momentum"] = _rms(mx * mx + my * my, sel)

    # --- which TERM drives the floor? -------------------------------------- #
    # The momentum residual is convective + pressure - viscous. Splitting it says
    # whether refinement exposes genuine field content (convective/pressure) or
    # amplifies small-scale ripple through the second derivative (viscous).
    dxg, dyg = field.domain.dx, field.domain.dy
    uu = np.asarray(field.u, np.float64)
    vv = np.asarray(field.v, np.float64)
    pp = np.asarray(field.p, np.float64)
    nut_ = np.asarray(field.nut if field.nut is not None else 0.0, np.float64)
    nu_e = np.clip(float(nu_lam) + nut_, 0.0, None)
    m_scale = (float(case.bc.u_inf) ** 2) / max(float(case.reference_length()), 1e-9)
    dudx_, dudy_ = ddx(uu, dxg), ddy(uu, dyg)
    dvdx_, dvdy_ = ddx(vv, dxg), ddy(vv, dyg)
    lap_u = ddx(ddx(uu, dxg), dxg) + ddy(ddy(uu, dyg), dyg)
    lap_v = ddx(ddx(vv, dxg), dxg) + ddy(ddy(vv, dyg), dyg)
    terms = {
        "conv": (uu * dudx_ + vv * dudy_, uu * dvdx_ + vv * dvdy_),
        "pres": (ddx(pp, dxg), ddy(pp, dyg)),
        "visc": (nu_e * lap_u, nu_e * lap_v),
    }
    for tname, (tx, ty) in terms.items():
        t2 = (tx / m_scale) ** 2 + (ty / m_scale) ** 2
        for b in bands:
            out[f"term_{tname}_band_{b:g}"] = _rms(t2, fluid & (sdf > b) & (~ring))

    # --- omitted stress-divergence terms, scaled and masked like the monitor ---
    gx, gy, dxt, dyt = omitted_terms(field, nu_lam)
    scale = m_scale
    zero = (~fluid) | ring
    for name, (ax, ay) in (("gradnu", (gx, gy)), ("divterm", (dxt, dyt))):
        a, b_ = ax.copy() / scale, ay.copy() / scale
        a[zero] = 0.0
        b_[zero] = 0.0
        t2 = a * a + b_ * b_
        out[f"omit_{name}_native"] = _rms(t2, fluid)
        for b in bands:
            out[f"omit_{name}_band_{b:g}"] = _rms(t2, fluid & (sdf > b) & (~ring))

    # --- repaired operator ------------------------------------------------- #
    # Two variants, because they mean different things. `repairedgn` repairs the
    # MODEL omission (a wrong operator). `repaired` also adds nu_eff*grad(div U),
    # which vanishes for an exactly divergence-free field, so on real data that
    # piece measures how far the rasterised reference is from discretely
    # divergence-free under OUR stencil -- a reference-operator artefact.
    for tag, (ax, ay) in (("repaired", (gx + dxt, gy + dyt)),
                          ("repairedgn", (gx, gy))):
        rx = mx - ax / scale
        ry = my - ay / scale
        rx[zero] = 0.0
        ry[zero] = 0.0
        r2 = cont * cont + rx * rx + ry * ry
        out[f"{tag}_native"] = _rms(r2, fluid)
        for b in bands:
            out[f"{tag}_band_{b:g}"] = _rms(r2, fluid & (sdf > b) & (~ring))

    # --- bc term, split ---------------------------------------------------- #
    noslip, far = bc_split(field, case)
    solid = ~fluid
    for arr in (noslip, far):
        arr[solid] = 0.0
        arr[ring] = 0.0
    u_inf = max(float(case.bc.u_inf), 1e-9)
    noslip /= u_inf
    far /= u_inf
    total = np.asarray(diag.bc_violation, np.float64)
    out["bc2"] = float(np.mean(total ** 2))          # matches bc_weight_sweep.py
    out["bc2_noslip"] = float(np.mean(noslip ** 2))
    out["bc2_farfield"] = float(np.mean(far ** 2))
    return out


def uniform_raster(ladder, n):
    r = np.zeros((4, n, n), np.float64)
    r[0] = ladder.u_vec[0]
    r[1] = ladder.u_vec[1]
    return r


# --------------------------------------------------------------------------- #
# Order fitting
# --------------------------------------------------------------------------- #


def fit_order(hs, vals):
    """Least-squares slope p of log(val) vs log(h). Returns (p, r2) or (nan, nan)."""
    h = np.asarray(hs, np.float64)
    v = np.asarray(vals, np.float64)
    ok = np.isfinite(h) & np.isfinite(v) & (v > 0) & (h > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log(h[ok]), np.log(v[ok])
    p, c = np.polyfit(lx, ly, 1)
    pred = p * lx + c
    ss = float(np.sum((ly - pred) ** 2))
    tot = float(np.sum((ly - ly.mean()) ** 2))
    return float(p), (1.0 - ss / tot if tot > 0 else float("nan"))


def classify(hs, vals):
    """Apply the pre-declared decision rule to one series (coarse -> fine)."""
    v = np.asarray(vals, np.float64)
    if not np.all(np.isfinite(v)) or v.size < 3 or v[0] <= 0:
        return "UNDECIDED", float("nan"), float("nan")
    p, _ = fit_order(hs, v)
    ratio = float(v[-1] / v[0])
    if np.isfinite(p) and p >= 1.5 and ratio <= 0.30:
        return "DECAY", p, ratio
    if (np.isfinite(p) and abs(p) < 0.5) or ratio >= 0.60:
        return "PLATEAU", p, ratio
    return "PARTIAL", p, ratio


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def gate_reproduce(ladders, checker, ref_json):
    """Gate 1. The 128^2 rung must match the PUBLISHED per-case numbers.

    This is the check a critic would run first: if the re-rasterisation does not
    reproduce results/certificates/residual_floor_realdata.json, the ladder is
    measuring something else and nothing downstream is trustworthy.
    """
    if not os.path.exists(ref_json):
        return {"status": "reference missing", "path": ref_json}
    with open(ref_json) as fh:
        ref = json.load(fh)
    by_name = {c["name"]: c for c in ref["per_case"]}
    rows = []
    for lad in ladders:
        sdf, mask, domain = lad.geometry_on(128)
        r, _ = lad.rasterise(128)
        f = _pack_field(r, sdf, mask, domain, "truth")
        case = lad.case_on(domain)
        got = measure(f, case, checker, AIRFRANS_NU, BANDS)["native"]
        exp = by_name.get(lad.name, {}).get("norm_truth")
        rel = abs(got - exp) / abs(exp) if exp else float("nan")
        rows.append({"name": lad.name, "published": exp, "reproduced": got, "rel_err": rel})
    worst = max((r["rel_err"] for r in rows if np.isfinite(r["rel_err"])), default=float("nan"))
    return {"rows": rows, "max_rel_err": worst, "pass": bool(worst < 5e-3)}


def gate_mms(ladder, checker, rungs):
    """Gate 2. Pure truncation on the MMS field must show order ~2."""
    rows = []
    for n in rungs:
        sdf, mask, domain = ladder.geometry_on(n)
        X, Y = domain.grid()
        u, v = mms_velocity(X, Y, ladder.u_vec, ladder.z0, ladder.m, ladder.k)
        r = np.stack([u, v, mms_pressure(u, v, ladder.u_vec), np.zeros_like(u)])
        f = _pack_field(r, sdf, mask, domain, "mms_analytic")
        case = ladder.case_on(domain)
        m = measure(f, case, checker, AIRFRANS_NU, BANDS)
        rows.append({"n": n, "h": domain.dx,
                     **{k: val for k, val in m.items() if k.startswith(("native", "band_0"))}})
    return rows


def gate_repaired(ladder, checker):
    """Gate 3. Unit-test the repaired operator against closed-form facts.

    With u = grad(phi) the code's operator never sees nu_t (it multiplies lap(u)=0),
    so a 50x nu_t must leave the monitored norm essentially untouched; a constant
    nu_eff must make the omitted term EXACTLY zero; and a varying one must make it
    fire. Together these prove `omitted_terms` is the term the monitor drops.
    """
    n = 256
    sdf, mask, domain = ladder.geometry_on(n)
    X, Y = domain.grid()
    u, v = mms_velocity(X, Y, ladder.u_vec, ladder.z0, ladder.m, ladder.k)
    p = mms_pressure(u, v, ladder.u_vec)
    case = ladder.case_on(domain)

    zero_nut = _pack_field(np.stack([u, v, p, np.zeros_like(u)]), sdf, mask, domain, "mms0")
    nut = mms_nut(X, Y, ladder.z0, amp=50.0 * AIRFRANS_NU, scale=0.4)
    with_nut = _pack_field(np.stack([u, v, p, nut]), sdf, mask, domain, "mms_nut")

    a = measure(zero_nut, case, checker, AIRFRANS_NU, BANDS)
    b = measure(with_nut, case, checker, AIRFRANS_NU, BANDS)

    # Independent algebraic identity. For potential flow dy(u) == dx(v) in the
    # CONTINUUM, so the coded expansion must satisfy
    #     gx = 2 dxnu dxu + dynu (dyu + dxv)  ->  2 (dxnu dxu + dynu dyu)
    # up to the O(h^2) gap between the two discrete cross-derivatives. A magnitude
    # test cannot catch a mis-transcribed stress divergence, but a CONVERGENCE test
    # can: with the wrong terms (e.g. dudy + dvdy) the gap does not vanish with h.
    def _identity_err(nn):
        sd, mk, dom = ladder.geometry_on(nn)
        Xg, Yg = dom.grid()
        uu, vv = mms_velocity(Xg, Yg, ladder.u_vec, ladder.z0, ladder.m, ladder.k)
        pp = mms_pressure(uu, vv, ladder.u_vec)
        nt = mms_nut(Xg, Yg, ladder.z0, amp=50.0 * AIRFRANS_NU, scale=0.4)
        fl = _pack_field(np.stack([uu, vv, pp, nt]), sd, mk, dom, "id")
        gxx, gyy, _, _ = omitted_terms(fl, AIRFRANS_NU)
        u2 = np.asarray(fl.u, np.float64)
        v2 = np.asarray(fl.v, np.float64)
        ne = AIRFRANS_NU + np.asarray(fl.nut, np.float64)
        hx, hy = dom.dx, dom.dy
        rgx = 2.0 * (ddx(ne, hx) * ddx(u2, hx) + ddy(ne, hy) * ddy(u2, hy))
        rgy = 2.0 * (ddx(ne, hx) * ddx(v2, hx) + ddy(ne, hy) * ddy(v2, hy))
        ins = np.zeros_like(u2, dtype=bool)
        ins[2:-2, 2:-2] = True
        ins &= np.asarray(fl.sdf, np.float64) > 0.10
        den = max(float(np.max(np.abs(rgx[ins]))), 1e-30)
        err = float(np.max(np.abs(gxx[ins] - rgx[ins]) + np.abs(gyy[ins] - rgy[ins]))) / den
        return err, dom.dx

    e1, h1 = _identity_err(256)
    e2, h2 = _identity_err(512)
    id_order = float(np.log(e1 / e2) / np.log(h1 / h2)) if e2 > 0 else float("inf")

    rel = abs(a["native"] - b["native"]) / max(a["native"], 1e-30)
    return {
        "native_zero_nut": a["native"], "native_with_nut": b["native"],
        "native_rel_change_from_nut": rel,
        "native_blind_to_nut": bool(rel < 1e-4),
        "omit_gradnu_zero_nut": a["omit_gradnu_native"],
        "omit_gradnu_exactly_zero_for_constant_nu": bool(a["omit_gradnu_native"] == 0.0),
        "omit_gradnu_with_nut": b["omit_gradnu_native"],
        "omit_gradnu_fires_for_varying_nu": bool(b["omit_gradnu_native"] > 0.0),
        "stress_divergence_identity_rel_err_256": e1,
        "stress_divergence_identity_rel_err_512": e2,
        "stress_divergence_identity_order": id_order,
        "stress_divergence_identity_ok": bool(id_order > 1.6),
        "repaired_with_nut": b["repaired_native"],
        "repairedgn_with_nut": b["repairedgn_native"],
        "pass": bool(rel < 1e-4 and a["omit_gradnu_native"] == 0.0
                     and b["omit_gradnu_native"] > 0.0 and id_order > 1.6),
    }


# --------------------------------------------------------------------------- #


def load_pointclouds(n_cases):
    """First ``n_cases`` clouds of the test split, in af.dataset.load order.

    Taking them from the FRONT is what makes index i align with case i of
    ``airfrans_full_test_r128_n200.pkl`` and hence with the published per-case
    numbers -- gate 1 depends on it. The rest of the cache is dropped immediately.
    """
    with open(PC_CACHE, "rb") as fh:
        pcs = pickle.load(fh)
    sub = list(pcs[:n_cases])
    del pcs
    gc.collect()
    return sub


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16, help="number of AirfRANS cases")
    ap.add_argument("--rungs", type=int, nargs="*", default=list(RUNGS))
    ap.add_argument("--gate", action="store_true", help="run gates only, on 2 cases")
    ap.add_argument("--no-cubic", dest="cubic", action="store_false", default=True)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)

    t_start = time.time()
    checker = PhysicsChecker()
    n_cases = 2 if args.gate else args.n
    print(f"loading {n_cases} point clouds from {PC_CACHE} ...", flush=True)
    t0 = time.time()
    clouds = load_pointclouds(n_cases)
    print(f"  {len(clouds)} clouds, {time.time() - t0:.0f}s", flush=True)

    # Gates run on the first two cases; the ladder rebuilds each case lazily so
    # only one Delaunay triangulation is resident at a time.
    ladders = [CaseLadder(pc) for pc in clouds[:2]]

    gates = {}
    print("\n[gate 1/3] reproduce the published 128^2 floor ...", flush=True)
    gates["reproduction"] = gate_reproduce(ladders, checker, REF_JSON)
    for r in gates["reproduction"].get("rows", []):
        print(f"   {r['name']:<28} published={r['published']:.6f} "
              f"reproduced={r['reproduced']:.6f} rel={r['rel_err']:.2e}", flush=True)
    print(f"   PASS={gates['reproduction'].get('pass')}", flush=True)

    print("\n[gate 2/3] MMS analytic truncation order (must be ~2) ...", flush=True)
    grows = gate_mms(ladders[0], checker, args.rungs)
    hs = [r["h"] for r in grows]
    p_nat, _ = fit_order(hs, [r["native"] for r in grows])
    p_bnd, _ = fit_order(hs, [r[f"band_{BANDS[1]:g}"] for r in grows])
    gates["mms_order"] = {"rows": grows, "p_native": p_nat,
                          f"p_band_{BANDS[1]:g}": p_bnd,
                          "pass": bool(np.isfinite(p_bnd) and 1.6 <= p_bnd <= 2.4)}
    for r in grows:
        print(f"   N={r['n']:>4} h={r['h']:.5f}  native={r['native']:.3e}  "
              f"band0.1={r[f'band_{BANDS[1]:g}']:.3e}", flush=True)
    print(f"   order: native p={p_nat:.2f}  band0.1 p={p_bnd:.2f}  "
          f"PASS={gates['mms_order']['pass']}", flush=True)

    print("\n[gate 3/3] repaired-operator unit test ...", flush=True)
    gates["repaired_unit_test"] = gate_repaired(ladders[0], checker)
    for k, v in gates["repaired_unit_test"].items():
        print(f"   {k}: {v}", flush=True)

    all_pass = all(g.get("pass") for g in gates.values() if isinstance(g, dict) and "pass" in g)
    print(f"\nGATES {'ALL PASS' if all_pass else 'FAILED -- no verdict can be issued'}",
          flush=True)

    if args.gate:
        _write(args.out.replace(".json", "_gates.json"),
               {"artifact": "floor_resolution_decomposition_gates", "gates": gates,
                "all_pass": all_pass, "wall_clock_s": time.time() - t_start})
        return 0 if all_pass else 1

    del ladders
    gc.collect()

    # ----------------------------------------------------------------- ladder --
    print(f"\n[ladder] {len(clouds)} cases x {len(args.rungs)} rungs", flush=True)
    per_case = []
    for ci, pc in enumerate(clouds):
        t_case = time.time()
        lad = CaseLadder(pc)
        rungs_out = []
        for n in args.rungs:
            t0 = time.time()
            sdf, mask, domain = lad.geometry_on(n)
            case = lad.case_on(domain)
            r, _ = lad.rasterise(n)
            truth = _pack_field(r, sdf, mask, domain, "truth")
            mt = measure(truth, case, checker, AIRFRANS_NU, BANDS)

            uf = _pack_field(uniform_raster(lad, n), sdf, mask, domain, "uniform")
            mu = measure(uf, case, checker, AIRFRANS_NU, BANDS)

            # MMS through the SAME point cloud + triangulation + interpolator.
            pu, pv = mms_velocity(lad.pos[:, 0], lad.pos[:, 1], lad.u_vec,
                                  lad.z0, lad.m, lad.k)
            pvals = np.stack([pu, pv, mms_pressure(pu, pv, lad.u_vec),
                              np.zeros_like(pu)], axis=1)
            rm, _ = lad.rasterise(n, values=pvals)
            mm = measure(_pack_field(rm, sdf, mask, domain, "mms_raster"),
                         case, checker, AIRFRANS_NU, BANDS)

            X, Y = domain.grid()
            au, av = mms_velocity(X, Y, lad.u_vec, lad.z0, lad.m, lad.k)
            ra = np.stack([au, av, mms_pressure(au, av, lad.u_vec), np.zeros_like(au)])
            ma = measure(_pack_field(ra, sdf, mask, domain, "mms_analytic"),
                         case, checker, AIRFRANS_NU, BANDS)

            row = {"n": n, "h": domain.dx, "seconds": time.time() - t0}
            row.update({f"truth_{k}": v for k, v in mt.items()})
            row.update({f"uniform_{k}": v for k, v in mu.items()})
            row["mms_raster_native"] = mm["native"]
            row["mms_analytic_native"] = ma["native"]
            for b in BANDS:
                row[f"mms_raster_band_{b:g}"] = mm[f"band_{b:g}"]
                row[f"mms_analytic_band_{b:g}"] = ma[f"band_{b:g}"]

            s_loc = lad.spacing_in_band(BANDS[1])
            row["s_local_band"] = s_loc
            row["h_over_s_local"] = domain.dx / s_loc if s_loc > 0 else float("nan")
            row["interpolant_dominated"] = bool(domain.dx < s_loc)

            # leg (A): does the truth-vs-uniform bc ordering survive refinement?
            dbc = mu["bc2"] - mt["bc2"]
            row["bc2_uniform_minus_truth"] = dbc
            row["legA_holds"] = bool(dbc < 0)   # uniform bc2 < truth bc2 -> (A) holds
            row["lambda_star"] = (mt["native"] ** 2 / dbc) if dbc > 0 else None

            if args.cubic and n in CUBIC_RUNGS:
                rc, _ = lad.rasterise(n, method="cubic")
                mc = measure(_pack_field(rc, sdf, mask, domain, "truth_cubic"),
                             case, checker, AIRFRANS_NU, BANDS)
                row["truth_cubic_native"] = mc["native"]
                for b in BANDS:
                    row[f"truth_cubic_band_{b:g}"] = mc[f"band_{b:g}"]

            rungs_out.append(row)
            print(f"   [{ci + 1}/{len(clouds)}] {lad.name[:26]:<26} N={n:>4} "
                  f"floor={mt['native']:.4f} band0.1={mt[f'band_{BANDS[1]:g}']:.4f} "
                  f"repaired={mt[f'repaired_band_{BANDS[1]:g}']:.4f} "
                  f"mmsR={mm[f'band_{BANDS[1]:g}']:.2e} "
                  f"h/s={row['h_over_s_local']:.2f} ({row['seconds']:.0f}s)", flush=True)

        trust = [r for r in rungs_out if not r["interpolant_dominated"]]
        hs_t = [r["h"] for r in trust]
        entry = {"name": lad.name, "u_inf": lad.u_inf, "aoa_deg": lad.aoa,
                 "n_points_in_crop": lad.n_in_crop, "s_cloud_crop": lad.s_crop,
                 "rungs": rungs_out, "n_trustworthy_rungs": len(trust), "verdicts": {}}
        for key in ([f"band_{b:g}" for b in BANDS] + ["native"]):
            v, p, ratio = classify(hs_t, [r[f"truth_{key}"] for r in trust])
            entry["verdicts"][key] = {"verdict": v, "order_p": p, "fine_over_coarse": ratio}
        for key in (f"truth_repaired_band_{BANDS[1]:g}",
                    f"truth_repairedgn_band_{BANDS[1]:g}",
                    f"truth_omit_gradnu_band_{BANDS[1]:g}",
                    f"mms_raster_band_{BANDS[1]:g}",
                    f"mms_analytic_band_{BANDS[1]:g}"):
            v, p, ratio = classify(hs_t, [r[key] for r in trust])
            entry["verdicts"][key.replace("truth_", "")] = {
                "verdict": v, "order_p": p, "fine_over_coarse": ratio}
        per_case.append(entry)
        name_ = lad.name
        vb = entry["verdicts"][f"band_{BANDS[1]:g}"]
        del lad
        gc.collect()
        print(f"   -> {name_[:26]:<26} band0.1 {vb['verdict']} (p={vb['order_p']:.2f}) "
              f"[{time.time() - t_case:.0f}s]", flush=True)
        _write(args.out, {"artifact": "floor_resolution_decomposition",
                          "status": "in-progress", "gates": gates, "per_case": per_case})

    out = {
        "artifact": "floor_resolution_decomposition",
        "status": "complete",
        "purpose": ("Decide whether the residual floor ||R_h(u*)|| decays with grid "
                    "refinement (a rasterisation artifact) or does not, and attribute "
                    "the result to a mechanism rather than a curve."),
        "metadata": {
            "script": "scripts/floor_resolution_decomposition.py",
            "companion": "scripts/floor_resolution_ladder.py (pre-registered ladder)",
            "point_cloud_cache": PC_CACHE,
            "reference_128_json": REF_JSON,
            "n_cases": len(per_case),
            "rungs": list(args.rungs),
            "bands_chord": list(BANDS),
            "primary_metric": f"band_{BANDS[1]:g} (fixed physical region, resolution-comparable)",
            "native_metric_caveat": (
                "`native` averages over ALL fluid cells, but `diagnose` zeroes a "
                "ONE-CELL wall ring whose physical thickness shrinks with h. The "
                "averaging set therefore changes between rungs and `native` is NOT "
                "resolution-comparable; it is reported only for continuity with the "
                "published 128^2 number."),
            "repaired_caveat": (
                "`repaired` adds back div(nu_eff(grad u + grad u^T)) minus "
                "nu_eff*lap(u). What remains is a LOWER BOUND on the reference-operator "
                "mismatch: AirfRANS is Spalart-Allmaras on finite-volume fluxes on a "
                "body-fitted mesh, and the SA source terms, FV flux reconstruction and "
                "mesh non-orthogonality are not modelled here."),
            "mms_null_caveat": (
                "The MMS field is smooth at the cloud scale, so mms_raster is a null "
                "for the pipeline acting on a RESOLVABLE field. The C1 (cubic) "
                "re-rasterisation is the control for sub-cell content."),
            "decision_rule": (
                "DECAY: p>=1.5 and finest/coarsest<=0.30. PLATEAU: |p|<0.5 or "
                "finest/coarsest>=0.60. Otherwise PARTIAL. Fitted on trustworthy rungs "
                "only (h >= median cloud spacing in the band). A negative p (floor "
                "RISING) classifies as PLATEAU by this rule and is reported as a rise."),
        },
        "gates": gates,
        "aggregate": _aggregate(per_case),
        "per_case": per_case,
        "wall_clock_s": time.time() - t_start,
    }
    _write(args.out, out)
    _report(out)
    return 0


def _aggregate(per_case):
    if not per_case:
        return {}
    agg = {}
    for k in list(per_case[0]["verdicts"].keys()):
        vs = [c["verdicts"][k]["verdict"] for c in per_case]
        ps = [c["verdicts"][k]["order_p"] for c in per_case
              if np.isfinite(c["verdicts"][k]["order_p"])]
        rs = [c["verdicts"][k]["fine_over_coarse"] for c in per_case
              if np.isfinite(c["verdicts"][k]["fine_over_coarse"])]
        agg[k] = {
            "n_plateau": vs.count("PLATEAU"), "n_decay": vs.count("DECAY"),
            "n_partial": vs.count("PARTIAL"), "n_cases": len(vs),
            "n_rising": int(sum(1 for c in per_case
                                if c["verdicts"][k]["order_p"] < 0)),
            "order_p_mean": float(np.mean(ps)) if ps else float("nan"),
            "order_p_std": float(np.std(ps)) if ps else float("nan"),
            "fine_over_coarse_mean": float(np.mean(rs)) if rs else float("nan"),
        }
    rungs = sorted({r["n"] for c in per_case for r in c["rungs"]})
    by_rung = {}
    for n in rungs:
        rows = [r for c in per_case for r in c["rungs"] if r["n"] == n]

        def mn(key):
            v = [r[key] for r in rows
                 if isinstance(r.get(key), (int, float)) and np.isfinite(r.get(key))]
            return float(np.mean(v)) if v else None

        lam = [r["lambda_star"] for r in rows if r.get("lambda_star") is not None]
        by_rung[str(n)] = {
            "h": rows[0]["h"], "n_cases": len(rows),
            "truth_native": mn("truth_native"),
            "truth_band_0.05": mn("truth_band_0.05"),
            "truth_band_0.1": mn("truth_band_0.1"),
            "truth_band_0.25": mn("truth_band_0.25"),
            "truth_repaired_band_0.1": mn("truth_repaired_band_0.1"),
            "truth_repairedgn_band_0.1": mn("truth_repairedgn_band_0.1"),
            "truth_omit_gradnu_band_0.1": mn("truth_omit_gradnu_band_0.1"),
            "truth_omit_divterm_band_0.1": mn("truth_omit_divterm_band_0.1"),
            "truth_band_0.1_continuity": mn("truth_band_0.1_continuity"),
            "truth_band_0.1_momentum": mn("truth_band_0.1_momentum"),
            "truth_term_conv_band_0.1": mn("truth_term_conv_band_0.1"),
            "truth_term_pres_band_0.1": mn("truth_term_pres_band_0.1"),
            "truth_term_visc_band_0.1": mn("truth_term_visc_band_0.1"),
            "mms_analytic_band_0.1": mn("mms_analytic_band_0.1"),
            "mms_raster_band_0.1": mn("mms_raster_band_0.1"),
            "truth_cubic_band_0.1": mn("truth_cubic_band_0.1"),
            "truth_cubic_native": mn("truth_cubic_native"),
            "truth_bc2": mn("truth_bc2"), "uniform_bc2": mn("uniform_bc2"),
            "truth_bc2_noslip": mn("truth_bc2_noslip"),
            "truth_bc2_farfield": mn("truth_bc2_farfield"),
            "uniform_bc2_noslip": mn("uniform_bc2_noslip"),
            "uniform_bc2_farfield": mn("uniform_bc2_farfield"),
            "uniform_native": mn("uniform_native"),
            "n_legA_holds": int(sum(bool(r["legA_holds"]) for r in rows)),
            "n_interpolant_dominated": int(sum(bool(r["interpolant_dominated"]) for r in rows)),
            "lambda_star_median": float(np.median(lam)) if lam else None,
            "h_over_s_local": mn("h_over_s_local"),
        }
    agg["by_rung"] = by_rung
    return agg


def _report(out):
    a = out["aggregate"]
    print("\n" + "=" * 96)
    print("RESOLUTION LADDER (means over cases; band0.1 = primary, fixed-region metric)")
    print("=" * 96)
    print(f"{'N':>5} {'h':>8} {'native':>9} {'band0.1':>9} {'cubicC1':>9} {'repaird':>9} "
          f"{'gradnu':>9} {'MMSanal':>9} {'MMSrast':>9} {'h/s':>6} {'tru_bc2':>9} {'unf_bc2':>9} {'legA':>5}")
    for n, r in a["by_rung"].items():
        def f(k, w=9, p=5):
            v = r.get(k)
            return f"{v:>{w}.{p}f}" if v is not None else " " * (w - 2) + "--"
        print(f"{n:>5} {r['h']:>8.5f} {f('truth_native')} {f('truth_band_0.1')} "
              f"{f('truth_cubic_band_0.1')} {f('truth_repaired_band_0.1')} "
              f"{f('truth_omit_gradnu_band_0.1')} {f('mms_analytic_band_0.1')} "
              f"{f('mms_raster_band_0.1')} {r['h_over_s_local']:>6.2f} "
              f"{f('truth_bc2')} {f('uniform_bc2')} "
              f"{r['n_legA_holds']:>3}/{r['n_cases']}")
    print("\nper-case verdicts (n_rising = cases whose fitted order is negative):")
    for k, v in a.items():
        if k == "by_rung":
            continue
        print(f"  {k:<26} PLATEAU={v['n_plateau']:>2}/{v['n_cases']} DECAY={v['n_decay']:>2} "
              f"PARTIAL={v['n_partial']:>2} rising={v['n_rising']:>2}  "
              f"p={v['order_p_mean']:>6.2f}+-{v['order_p_std']:.2f}  "
              f"fine/coarse={v['fine_over_coarse_mean']:.3f}")
    print(f"\nwall clock {out['wall_clock_s'] / 60:.1f} min")


def _clean(o):
    """Coerce numpy scalars and map non-finite floats to null.

    ``allow_nan=False`` gives strict, spec-valid JSON (no Infinity/NaN tokens), but
    ``json`` raises on NaN rather than consulting ``default=``, so this has to
    happen before the dump.
    """
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    # newline="\n" so the JSON is LF on Windows and its stored hash stays stable.
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_clean(obj), fh, indent=2, allow_nan=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
