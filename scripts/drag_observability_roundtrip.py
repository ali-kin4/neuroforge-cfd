"""Does the RASTER or OUR QUADRATURE own the drag-observability loss on AirfRANS?

The blocking control (``whitespace.md`` A0) for the "lift is observable, drag is
not" finding. ``results/control/force_vs_official.json`` reports
``rho_D_gt_vs_official = 0.8394`` -- our ``force_coefficients`` on a 128^2 raster
of the EXACT ground-truth field, rank-correlated against the official AirfRANS
labels. That number conflates two mechanisms:

  1. **rasterisation loss** -- information destroyed by resampling a body-fitted
     unstructured RANS solution onto a 128^2 Cartesian crop;
  2. **integrator-formulation difference** -- our surface quadrature versus the
     one AirfRANS itself uses.

The existing 18-variant sweep (``scripts/design_force_integrator.py``) controls
our own design choices, but every variant is still OUR quadrature on a raster,
so it cannot separate these. ``force_vs_official.json`` admits the gap in its own
metadata: ``"pred_af_skipped": "airfrans' own integrator on our prediction
skipped"``. This script runs the missing arm.

===========================================================================
THE TRAP, AND THE DESIGN THAT AVOIDS IT
===========================================================================
Calling ``Simulation.force_coefficient(reference=True)`` on the untouched native
cloud is NOT a control: it reads ``internal.point_data['p']`` and ``['U']``
directly, so it returns the label by construction and gives a trivial rho = 1.0.

The control must go through a RASTER ROUND TRIP so that rasterisation loss is
present while the integrator formulation is held at AirfRANS's own:

    native cloud  --rasterise 128^2-->  raster  --scatter back to mesh nodes-->
    overwrite Simulation.velocity / .pressure  -->  force_coefficient(reference=False)

``force_coefficient(reference=False)`` reads the MUTABLE attributes
``self.velocity`` (M,2) and ``self.pressure`` (M,1) and integrates them with
AirfRANS's own formulation (VTK ``compute_derivative`` for wall shear on the
body-fitted mesh, ``reorganize`` onto the airfoil polyline, ``ptc``, cell-Length
quadrature). Rasterisation loss is then the ONLY variable.

Alignment is exact rather than approximate: ``airfrans.dataset.load`` builds its
per-case array by concatenating ``Simulation.position / .velocity / .pressure /
.nu_t``, so the point set our rasteriser consumes IS ``sim.position``, node for
node. No matching is required.

===========================================================================
PRE-REGISTRATION -- written and committed BEFORE the run
===========================================================================

**P0. The claim under test.** ``whitespace.md`` G1 states that on a 128^2 raster
of AirfRANS no near-wall scheme recovers the official drag ranking beyond
rho_D ~ 0.84, and reads that as a ceiling imposed by the representation.

**P1. Primary statistic.** ``rho_D_rt`` := Spearman(round-trip cd via AirfRANS's
own integrator, official cd), n = 200, ``task='full'`` test split, at 128^2.

**P2. Decision rule -- FOUR branches, declared in advance.** Read against our own
0.8394, not against 1.0:

  * **R-MATCH** ``|rho_D_rt - 0.839| <= 0.05``: the raster owns the loss and our
    quadrature adds none. Drag is not recoverable from a 128^2 representation by
    ANY formulation, including the dataset's own. G1's framing is clean.
  * **Q-OWNS** ``rho_D_rt >= 0.95``: the information survives the raster and our
    integrator destroys it. The observability claim COLLAPSES; the paper must
    stop leaning on rho_D = 0.84 and retreat to "our integrator's ceiling".
  * **R-WORSE** ``rho_D_rt < 0.79``: the raster owns the loss AND 0.839 is not a
    ceiling -- faithful integration of the same rasterised information does
    WORSE than our scheme, so our 0.839 cannot be a surface integral of the
    rasterised field and must be carrying non-force information. The
    "ceiling" language must be withdrawn even though the representational claim
    survives and strengthens.
  * **SPLIT** ``0.79 <= rho_D_rt < 0.95`` and not R-MATCH: both mechanisms
    contribute; report the decomposition and claim neither exclusively.

  Predicted before running (recorded so it can be scored): **R-WORSE**. Our
  integrator samples at a +1.5-cell outward-normal offset = 0.035 chord at
  128^2, roughly 7x the viscous sublayer scale, so it cannot be computing wall
  shear; and a single-case probe gives round-trip cdv = 8.3e-5 against a label
  cdv = 8.36e-3, i.e. viscous drag annihilated to 1% of its value.

**P3. The covariate control, which gates the CLAIM rather than the run.** cd is
largely a smooth function of case covariates, and the simulation NAME hands them
over for free (``Simulation.reset`` parses ``inlet_velocity = name.split('_')[2]``,
``angle_of_attack = name.split('_')[3]``). A marginal Spearman against official
cd therefore measures case identity plus force information, mixed. We report:

  * ``rho_cov`` := Spearman(OLS fit of official cd on (U, alpha, alpha^2),
    official cd) -- how far a trivial 3-parameter regression on the case label
    gets WITHOUT ever touching a flow field;
  * ``rho_partial`` := Spearman-partial of each arm against official cd,
    controlling for (U, alpha) -- how much of each arm's number is force
    information rather than case identity.

  **Pre-registered reading.** If ``rho_cov`` >= the arm's marginal rho, then that
  arm's marginal rho is NOT evidence of drag observability and the sentence "no
  scheme recovers drag beyond rho_D ~ 0.84" must be withdrawn as stated, because
  a reviewer can reproduce the covariate regression in ten minutes. Geometry
  (the NACA digits, also in the name) is deliberately NOT controlled for, so
  ``rho_cov`` is a LOWER bound on what case identity alone buys.

**P4. Mechanism, and why the lift/drag asymmetry should follow.** Wall shear is
set by the viscous sublayer. At Re ~ 2e6, y+ ~ 5 is y/c ~ 6e-5 (c_f ~ 0.003,
u_tau/U ~ 0.04); on a 3-chord crop with h = 3/N that needs N >~ 5e4 to difference
faithfully. Surface pressure varies on the CHORD scale and carries no sublayer
requirement. Predicted: cdv (viscous drag) is annihilated by the round trip at
every reachable N, cdp and cl survive far better. Reported per component --
``force_coefficient`` already returns ``((cd,cdp,cdv),(cl,clp,clv))``.

  Consistency check on that scale argument: it independently PREDICTS the source
  mesh's measured near-wall spacing of 4.4e-5 chord (AirfRANS resolves to
  y+ ~ 3-4). If the two disagree by more than ~3x, the argument is reported as
  not supported.

**P5. The ladder, and its honest limit.** The round trip is a CONSISTENT
interpolation: at the mesh nodes it converges to the identity as h -> 0. So
"drag is not observable from a raster" can NEVER be a resolution-independent
claim, and this script does not attempt one. The claim under test is a
quantified RESOLUTION REQUIREMENT: at which N does each component become
recoverable? Levels 128 / 256 / 512. A monotone climb toward 1.0 that is still
far from it at 512^2, with the extrapolated N for recovery far beyond anything
deployed, is the reportable result.

**P6. Gates (each must pass or the run is void).**
  * ``identity``: an arm that overwrites the attributes with the reference values
    themselves must reproduce the label to machine precision.
  * per-arm negative gates: in ``rt_p_only`` (velocity left at reference) cdv must
    equal the label's cdv exactly; in ``rt_u_only`` (pressure left at reference)
    cdp must equal the label's cdp exactly. These catch a ``reference=True`` slip
    that the global identity gate would not.
  * ``outside_crop``: mesh nodes outside the 3x3-chord crop are refilled with
    reference values (the raster does not cover them). Perturbing all nodes with
    sdf > 0.5 chord must leave the force bit-identical, proving the refill policy
    cannot move the answer.

===========================================================================
ARMS
===========================================================================
  ``identity``     nothing round-tripped                    -> gate
  ``rt_pipeline``  u,v,p round-tripped, WITH the loader's solid-mask zeroing of
                   u,v (exactly what ``load_airfrans`` feeds the model)
  ``rt_raw``       u,v,p round-tripped, bare raster, no mask policy
  ``rt_p_only``    p round-tripped, velocity kept at reference
  ``rt_u_only``    u,v round-tripped, pressure kept at reference

CPU-only by construction (no torch, no GPU). Resumable per case.

    .venv/Scripts/python.exe scripts/drag_observability_roundtrip.py \
        --levels 128 256 512 --n-cases 200
"""

from __future__ import annotations

import argparse
import json
import os
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/scipy (sets BLAS thread caps)

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay
from scipy.stats import rankdata, spearmanr

from neuroforge.core.types import Domain

CROP = (-1.0, 2.0, -1.5, 1.5)
ARMS = ("rt_pipeline", "rt_raw", "rt_p_only", "rt_u_only")
COMPONENTS = ("cd", "cdp", "cdv", "cl", "clp", "clv")


def log(msg: str) -> None:
    print(f"[dobs] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
def build_raster(tri: Delaunay, vals: np.ndarray, dom: Domain) -> np.ndarray:
    """Rasterise ``vals`` on the cloud behind ``tri`` onto ``dom``.

    Numerically identical to ``neuroforge.data.rasterize.rasterize_point_cloud``
    with ``method='linear'``: that function builds exactly this ``Delaunay`` +
    ``LinearNDInterpolator(fill_value=fill)`` pair. We build the triangulation
    ONCE per case and reuse it across levels and arms; equivalence to the
    pipeline function is asserted on the first case (``gates.rasteriser_match``).
    """
    X, Y = dom.grid()
    xi = np.stack([X.ravel(), Y.ravel()], axis=1)
    out = np.asarray(LinearNDInterpolator(tri, vals, fill_value=0.0)(xi), np.float64)
    ny, nx = dom.shape
    return out.reshape(ny, nx, vals.shape[1]).transpose(2, 0, 1)


def scatter_back(raster: np.ndarray, dom: Domain, pos: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a ``(K,ny,nx)`` raster back onto the mesh nodes."""
    x, y = dom.axes()
    q = np.column_stack([pos[:, 1], pos[:, 0]])
    out = np.empty((pos.shape[0], raster.shape[0]), np.float64)
    for k in range(raster.shape[0]):
        gi = RegularGridInterpolator(
            (y.astype(np.float64), x.astype(np.float64)), raster[k].astype(np.float64),
            method="linear", bounds_error=False, fill_value=None,
        )
        out[:, k] = gi(q)
    return out


def inside_crop(pos: np.ndarray) -> np.ndarray:
    xmin, xmax, ymin, ymax = CROP
    return ((pos[:, 0] >= xmin) & (pos[:, 0] <= xmax)
            & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax))


def solid_mask_grid(sim, dom: Domain) -> np.ndarray:
    """The loader's solid mask, rebuilt from the on-wall points of this sim.

    Mirrors ``airfrans_loader._sim_to_pair``: reconstruct the surface loop from
    nodes with a nonzero normal, then ``solid_mask`` on the crop.
    """
    from neuroforge.core.types import Geometry
    from neuroforge.data.airfrans_loader import _order_surface_loop
    from neuroforge.geometry.sdf import solid_mask

    normals = sim.normals
    on_wall = np.linalg.norm(normals, axis=1) > 1e-8
    wall_pts = sim.position[on_wall]
    loop = _order_surface_loop(wall_pts)
    return solid_mask(Geometry(name=sim.name, surface_points=loop), dom)


def coeffs(sim, velocity: np.ndarray, pressure: np.ndarray) -> dict:
    """Set the mutable attributes and integrate with AirfRANS's own formulation."""
    sim.velocity = np.ascontiguousarray(velocity, np.float64)
    sim.pressure = np.ascontiguousarray(pressure, np.float64).reshape(-1, 1)
    (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient(
        compressible=False, reference=False
    )
    return {"cd": float(cd), "cdp": float(cdp), "cdv": float(cdv),
            "cl": float(cl), "clp": float(clp), "clv": float(clv)}


def run_case(sim, levels, mask_cache: dict, want_gates: bool) -> dict:
    pos = sim.position.copy()
    vel0 = sim.velocity.copy()
    prs0 = sim.pressure.copy()

    (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient(
        compressible=False, reference=True
    )
    label = {"cd": float(cd), "cdp": float(cdp), "cdv": float(cdv),
             "cl": float(cl), "clp": float(clp), "clv": float(clv)}

    rec: dict = {"name": sim.name, "label": label, "levels": {}}

    # Gate: identity -- attributes set to the reference values must reproduce it.
    if want_gates:
        ident = coeffs(sim, vel0, prs0)
        rec["gate_identity_abs_err"] = max(
            abs(ident[k] - label[k]) for k in COMPONENTS
        )
        # Gate: far-field refill cannot move the force.
        far = np.asarray(sim.sdf).ravel() > 0.5
        vpert = vel0.copy()
        vpert[far] += 3.7
        ppert = prs0.copy()
        ppert[far] += 11.3
        pert = coeffs(sim, vpert, ppert)
        rec["gate_farfield_abs_err"] = max(
            abs(pert[k] - label[k]) for k in COMPONENTS
        )

    tri = Delaunay(pos)
    ins = inside_crop(pos)
    vals = np.concatenate([vel0, prs0], axis=1)  # u, v, p

    for N in levels:
        dom = Domain(bounds=CROP, nx=int(N), ny=int(N))
        raster = build_raster(tri, vals, dom)
        back = scatter_back(raster, dom, pos)
        # Nodes the raster does not cover keep their reference values.
        back[~ins, 0] = vel0[~ins, 0]
        back[~ins, 1] = vel0[~ins, 1]
        back[~ins, 2] = prs0[~ins, 0]

        # The pipeline's solid-mask policy: u, v zeroed inside the body; p raw.
        key = (sim.name, int(N))
        if key not in mask_cache:
            mask_cache[key] = solid_mask_grid(sim, dom)
        solid = mask_cache[key] < 0.5
        rmask = raster.copy()
        rmask[0] = np.where(solid, 0.0, rmask[0])
        rmask[1] = np.where(solid, 0.0, rmask[1])
        back_m = scatter_back(rmask, dom, pos)
        back_m[~ins, 0] = vel0[~ins, 0]
        back_m[~ins, 1] = vel0[~ins, 1]
        back_m[~ins, 2] = prs0[~ins, 0]

        out = {
            "rt_raw": coeffs(sim, back[:, :2], back[:, 2]),
            "rt_pipeline": coeffs(sim, back_m[:, :2], back_m[:, 2]),
            "rt_p_only": coeffs(sim, vel0, back[:, 2]),
            "rt_u_only": coeffs(sim, back[:, :2], prs0),
        }
        # Per-arm negative gates: the component that was NOT round-tripped must
        # come back exactly. Catches a reference=True slip the identity gate misses.
        out["gate_p_only_cdv_err"] = abs(out["rt_p_only"]["cdv"] - label["cdv"])
        out["gate_u_only_cdp_err"] = abs(out["rt_u_only"]["cdp"] - label["cdp"])
        rec["levels"][str(int(N))] = out

    sim.velocity = vel0
    sim.pressure = prs0
    return rec


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def partial_spearman(x: np.ndarray, y: np.ndarray, covars: list[np.ndarray]) -> float:
    """Spearman of x vs y with ``covars`` linearly removed from the RANKS."""
    n = len(x)
    design = np.column_stack([np.ones(n)] + [rankdata(c) for c in covars])

    def resid(v):
        rv = rankdata(v)
        beta, *_ = np.linalg.lstsq(design, rv, rcond=None)
        return rv - design @ beta

    rx, ry = resid(x), resid(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(records: list[dict], levels) -> dict:
    names = [r["name"] for r in records]
    U = np.array([float(n.split("_")[2]) for n in names])
    A = np.array([float(n.split("_")[3]) for n in names])
    lab = {k: np.array([r["label"][k] for r in records]) for k in COMPONENTS}

    # P3 covariate baseline: a trivial regression on the case NAME, no flow field.
    cov = {}
    for tag, cols in (("U_a", [U, A]), ("U_a_a2", [U, A, A ** 2])):
        X = np.column_stack([np.ones(len(U))] + cols)
        for k in ("cd", "cl"):
            beta, *_ = np.linalg.lstsq(X, lab[k], rcond=None)
            cov[f"rho_cov_{tag}_{k}"] = float(spearmanr(X @ beta, lab[k]).statistic)
    cov["rho_alpha_vs_cd"] = float(spearmanr(A, lab["cd"]).statistic)
    cov["rho_absalpha_vs_cd"] = float(spearmanr(np.abs(A), lab["cd"]).statistic)
    cov["rho_alpha_vs_cl"] = float(spearmanr(A, lab["cl"]).statistic)

    per_level = {}
    for N in levels:
        s = str(int(N))
        arms = {}
        for arm in ARMS:
            v = {k: np.array([r["levels"][s][arm][k] for r in records])
                 for k in COMPONENTS}
            e = {}
            for k in COMPONENTS:
                e[f"rho_{k}"] = float(spearmanr(v[k], lab[k]).statistic)
                denom = np.abs(lab[k])
                ok = denom > 1e-12
                e[f"median_rel_err_{k}"] = float(
                    np.median(np.abs(v[k][ok] - lab[k][ok]) / denom[ok])
                )
            e["rho_cd_partial_U_a"] = partial_spearman(v["cd"], lab["cd"], [U, A])
            e["rho_cl_partial_U_a"] = partial_spearman(v["cl"], lab["cl"], [U, A])
            e["cdv_ratio_median"] = float(np.median(v["cdv"] / lab["cdv"]))
            e["cdp_ratio_median"] = float(np.median(v["cdp"] / lab["cdp"]))
            arms[arm] = e
        arms["gates"] = {
            "max_p_only_cdv_err": max(r["levels"][s]["gate_p_only_cdv_err"]
                                      for r in records),
            "max_u_only_cdp_err": max(r["levels"][s]["gate_u_only_cdp_err"]
                                      for r in records),
        }
        per_level[s] = arms

    # P2 verdict on the primary statistic.
    rho = per_level[str(int(levels[0]))]["rt_pipeline"]["rho_cd"]
    if abs(rho - 0.8394) <= 0.05:
        verdict = "R-MATCH: the raster owns the loss; our quadrature adds none."
    elif rho >= 0.95:
        verdict = "Q-OWNS: information survives the raster; OUR INTEGRATOR owns the loss. Claim collapses."
    elif rho < 0.79:
        verdict = ("R-WORSE: the raster owns the loss AND 0.839 is not a ceiling -- "
                   "faithful integration of the same rasterised field does worse.")
    else:
        verdict = "SPLIT: both mechanisms contribute."

    # Label-side facts. The rank structure matters as much as the magnitude
    # share: cdv is the bigger PART of cd but cdp carries almost all of its
    # RANKING, so drag ranking is recoverable from surface pressure in
    # principle -- which is what stops us over-reading the DD-RNO comparison.
    share = {
        "median_cdv_over_cd": float(np.median(lab["cdv"] / lab["cd"])),
        "median_cdp_over_cd": float(np.median(lab["cdp"] / lab["cd"])),
        "median_clp_over_cl": float(np.median(lab["clp"] / lab["cl"])),
        "rho_label_cdp_vs_cd": float(spearmanr(lab["cdp"], lab["cd"]).statistic),
        "rho_label_cdv_vs_cd": float(spearmanr(lab["cdv"], lab["cd"]).statistic),
        "rho_label_cdp_vs_cdv": float(spearmanr(lab["cdp"], lab["cdv"]).statistic),
        "_note": "rho_label_cdp_vs_cd is the ceiling a perfect surface-pressure "
                 "integrator could reach on TOTAL drag ranking without any "
                 "viscous information at all.",
    }

    # P5 resolution requirement: extrapolate the ladder.
    Ns = np.array([float(N) for N in levels])
    extrap = {}
    if len(Ns) >= 2:
        ratio = np.array([per_level[str(int(N))]["rt_pipeline"]["cdv_ratio_median"]
                          for N in levels])
        sl, ic = np.polyfit(np.log(Ns), np.log(ratio), 1)
        extrap["cdv_ratio_power_in_N"] = float(sl)
        extrap["N_for_cdv_ratio_1"] = float(np.exp((0.0 - ic) / sl))
        rcd = np.array([per_level[str(int(N))]["rt_pipeline"]["rho_cd"] for N in levels])
        sl2, ic2 = np.polyfit(np.log(Ns), rcd, 1)
        target = share["rho_label_cdp_vs_cd"]
        extrap["N_for_rho_cd_reaching_cdp_ceiling"] = float(np.exp((target - ic2) / sl2))
        extrap["_note"] = ("Log-linear extrapolations well beyond the measured range; "
                           "reported as an order of magnitude, not a prediction.")
    # Independent scale argument (P4): y+ = 5 on a 3-chord crop.
    NU = 1.56e-5
    Uarr = U
    ut_over_U = float(np.sqrt(0.003 / 2))
    y5 = 5.0 * (NU / (Uarr * 1.0)) / ut_over_U
    extrap["yplus5_chord_min"] = float(y5.min())
    extrap["yplus5_chord_max"] = float(y5.max())
    extrap["N_for_yplus5_min"] = float(3.0 / y5.max())
    extrap["N_for_yplus5_max"] = float(3.0 / y5.min())

    return {"covariate_control": cov, "per_level": per_level,
            "verdict_primary_128": verdict, "rho_D_rt_128": rho,
            "label_composition": share, "resolution_requirement": extrap}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="raster round-trip drag observability")
    ap.add_argument("--levels", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--n-cases", type=int, default=200)
    ap.add_argument("--root", default="data")
    ap.add_argument("--labels",
                    default="results/control/_cache/official_labels_full_test_n200.json")
    ap.add_argument("--out", default="results/control/drag_observability_roundtrip.json")
    ap.add_argument("--cache", default="results/control/_cache/roundtrip_percase.jsonl")
    a = ap.parse_args(argv)

    import airfrans.simulation as afsim
    from neuroforge.data.airfrans_loader import _resolve_data_root

    root = _resolve_data_root(a.root)
    if root is None:
        log(f"FATAL: no AirfRANS manifest under {a.root}")
        return 2

    names = list(json.load(open(a.labels, encoding="utf-8")).keys())[: a.n_cases]
    log(f"root={root}  n_cases={len(names)}  levels={a.levels}")

    done: dict[str, dict] = {}
    if os.path.exists(a.cache):
        for line in open(a.cache, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if all(str(int(N)) in r.get("levels", {}) for N in a.levels):
                    done[r["name"]] = r
        log(f"resumed {len(done)} cases from {a.cache}")

    os.makedirs(os.path.dirname(a.cache), exist_ok=True)
    mask_cache: dict = {}
    t0 = time.time()
    fh = open(a.cache, "a", encoding="utf-8")
    for i, nm in enumerate(names):
        if nm in done:
            continue
        sim = afsim.Simulation(root=root, name=nm)
        rec = run_case(sim, a.levels, mask_cache, want_gates=(len(done) < 5))
        done[nm] = rec
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        mask_cache.clear()
        del sim
        if (i + 1) % 10 == 0:
            log(f"{i + 1}/{len(names)}  ({time.time() - t0:.0f}s)")
    fh.close()

    records = [done[nm] for nm in names if nm in done]
    log(f"analysing {len(records)} cases")
    res = analyse(records, a.levels)

    gates = {
        "identity_max_abs_err": max(
            (r["gate_identity_abs_err"] for r in records if "gate_identity_abs_err" in r),
            default=float("nan")),
        "farfield_refill_max_abs_err": max(
            (r["gate_farfield_abs_err"] for r in records if "gate_farfield_abs_err" in r),
            default=float("nan")),
    }
    out = {
        "meta": {
            "purpose": "Separate rasterisation loss from integrator formulation on "
                       "AirfRANS drag/lift: raster round trip scored with AirfRANS's "
                       "OWN force integrator (whitespace.md A0).",
            "n_cases": len(records), "levels": a.levels, "crop": CROP,
            "arms": list(ARMS), "device": "cpu",
            "integrator": "airfrans.Simulation.force_coefficient(compressible=False, "
                          "reference=False) on overwritten .velocity/.pressure",
            "prereg": "See module docstring P0-P6; committed before the run.",
        },
        "gates": gates,
        **res,
        "wall_clock_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {a.out}")
    log(f"VERDICT: {res['verdict_primary_128']}")
    log(f"rho_D_rt(128, rt_pipeline) = {res['rho_D_rt_128']:.4f}")
    log(f"covariate baseline rho_cov(U,a,a^2 -> cd) = "
        f"{res['covariate_control']['rho_cov_U_a_a2_cd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
