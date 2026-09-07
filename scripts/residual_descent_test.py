"""Explicit gradient descent on the physics-residual objective J = 1/2 ||R_h(u)||^2.

WHY THIS SCRIPT EXISTS
----------------------
The paper claims (abstract, ``sec:iters``, ``thm:residual-floor`` leg (iv)) that the
monitored steady-RANS residual is a poor *correction objective*: "reducing it does not
reduce field error". The evidence cited for that claim (``tab:iters`` /
``results/sensitivity/iters.json``) sweeps ``DEQCorrector.max_iter`` -- the internal
fixed-point cap of a corrector trained by **supervised regression toward ground truth**.
That is an anti-correlation along one trajectory; it is *not* residual minimisation.
Nothing in ``scripts/`` ever minimised ``J``.

This script runs the missing experiment: honest, explicit gradient descent
``u <- u - eta * grad_u J(u)`` in field space, measuring the TRUE field error against
ground truth at every step.

THE OPERATOR UNDER TEST (this is the load-bearing detail)
---------------------------------------------------------
``physics.residuals.physics_residual_torch`` is the *training-loss* twin: it is raw,
dimensional, and masks only the solid. The scalar the paper actually monitors is
``Diagnostics.residual_norm()``, which additionally (a) zeroes the solid-adjacent fluid
ring ``_solid_adjacent_fluid`` and (b) non-dimensionalises continuity by ``u_inf/L`` and
momentum by ``u_inf^2/L``. They are NOT the same functional.

We therefore build a differentiable replica of the **monitored** operator out of the
framework's own backend-agnostic ``operators.ddx/ddy/laplacian`` (which duck-type onto
torch, so gradients flow) and *verify* it reproduces ``Diagnostics.residual_norm()`` to
~1e-8 relative on every case before descending (``--verify``, run automatically). The
raw training-loss twin ``physics_residual_torch`` is available as a secondary objective
via ``--objective raw`` so the result can be shown not to hinge on the choice.

    J(u) = 1/2 * mean_cells( rc^2 + rx^2 + ry^2 )   ==>   residual_norm = sqrt(2 J)

ARMS
----
start field:
  ``truth``       ground-truth field u* (sharpest test of theorem leg (ii): if u* is not
                  a stationary point of J, descent from truth must move AWAY from truth)
  ``transolver``  deployed SOTA backbone, seeds 0/1/2, read from the zero-forward-pass
                  cache ``data/cache/acceptance_gate/seed{k}/*.npz`` (key ``raw``)
  ``fno``         dropout-FNO of ``checkpoints/certificates_deq.pt`` -- the same backbone
                  family as ``tab:iters``, so the numbers are comparable

constraint mode:
  ``free``  unconstrained descent on every cell. NOTE: theorem leg (i) says the uniform
            freestream field is an exact global minimum of J, so this mode is partly a
            demonstration of the ill-posedness of a BC-free residual. Reported, but not
            the headline.
  ``bc``    THE LOAD-BEARING ARM. Dirichlet data pinned at ground truth on a 2-cell outer
            border ring (far field) and on the near-wall ring (no-slip band); the solid is
            frozen. Gradient is projected to zero on all frozen cells. A 2-cell freeze is
            required because grad J carries a 4th-order stencil (double Laplacian), so a
            1-cell freeze would leave the second ring free to drift.
            If descent still raises the error HERE, the paper's claim survives the
            "you just rediscovered that a residual without BCs is ill-posed" objection.

variables:
  ``uvp``   descend on (u, v, p), freeze nut (primary: nut is a closure variable)
  ``uvpn``  descend on all four channels

step rule:
  ``armijo``  backtracking line search on J itself (per case), c1=1e-4 -- guarantees a
              monotone J decrease, so "you picked a bad learning rate" cannot be raised:
              the objective chose the step.
  ``fixed``   fixed eta from a log-spaced grid, for the eta-sensitivity table.

PRE-REGISTERED READINGS (named before the run)
----------------------------------------------
* If BC-constrained descent from the deployed Transolver field lowers rel-L2 on a
  majority of cases by a substantial fraction of the supervised DEQ gain, the claim
  "poor correction objective" is FALSIFIED as stated and must become a regime claim.
* If J falls monotonically while error is U-shaped, the residual supplies no stopping
  signal, so the objective is unusable in deployment even where it briefly helps. This
  is measured per case as argmin_k J vs argmin_k error.
* Exchange rate d(error)/d(residual_norm) is reported because it is step-size invariant
  to first order and therefore comparable across arms and etas.

Artifacts
---------
    results/residual_descent/descent_<arm>_<mode>_<vars>_<rule>.json   per-case + aggregate
    results/residual_descent/eta_sensitivity.json
    results/residual_descent/summary.json
    results/residual_descent/verification.json

Run (CPU-only; ~20 min total on 8 threads)::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=src \
        .venv/Scripts/python.exe scripts/residual_descent_test.py --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets BLAS thread caps)
import numpy as np
import torch

from neuroforge.core.types import DTYPE, FlowField
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.physics.operators import ddx, ddy, laplacian
from neuroforge.physics.residuals import PhysicsChecker, _solid_adjacent_fluid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "residual_descent")
FNO_CACHE = os.path.join(ROOT, "data", "cache", "residual_descent_fno")
GATE_CACHE = os.path.join(ROOT, "data", "cache", "acceptance_gate")

F64 = torch.float64


def log(msg: str) -> None:
    print(f"[residual_descent] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# The differentiable MONITORED residual (exact replica of Diagnostics.residual_norm)
# --------------------------------------------------------------------------- #
def monitored_residual(y, keep, dx, dy, nu, cont_scale, mom_scale):
    """Differentiable monitored residual maps.

    Parameters
    ----------
    y : torch.Tensor ``(B, 4, H, W)`` physical fields ``(u, v, p, nut)``.
    keep : torch.Tensor ``(B, 1, H, W)`` 1.0 on cells the monitor counts
        (fluid minus the solid-adjacent ring), 0.0 elsewhere.
    dx, dy : float grid spacings (identical for all AirfRANS 128^2 cases).
    nu : float laminar kinematic viscosity.
    cont_scale, mom_scale : torch.Tensor ``(B, 1, 1, 1)``
        ``u_inf / L`` and ``u_inf^2 / L`` -- the non-dimensionalisation the monitor applies.

    Returns
    -------
    tuple of three ``(B, 1, H, W)`` tensors ``(cont, r_x, r_y)``, non-dimensionalised
    and masked exactly as ``PhysicsChecker.diagnose`` does.
    """
    u, v, p, nut = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    # numpy path uses np.clip(nu + nut, 0, None); mirror it (NOT the min=nu clamp of
    # physics_residual_torch -- that is the training-loss twin, a different function).
    nu_eff = torch.clamp(nu + nut, min=0.0)
    cont = ddx(u, dx) + ddy(v, dy)
    r_x = u * ddx(u, dx) + v * ddy(u, dy) + ddx(p, dx) - nu_eff * laplacian(u, dx, dy)
    r_y = u * ddx(v, dx) + v * ddy(v, dy) + ddy(p, dy) - nu_eff * laplacian(v, dx, dy)
    return (cont * keep / cont_scale, r_x * keep / mom_scale, r_y * keep / mom_scale)


def raw_residual(y, keep, dx, dy, nu, cont_scale, mom_scale):
    """Secondary objective: the raw training-loss twin ``physics_residual_torch``.

    Same call signature so the two are interchangeable. Masks the solid only (``keep`` is
    replaced by the plain fluid mask by the caller) and applies no non-dimensionalisation.
    """
    from neuroforge.physics.residuals import physics_residual_torch

    inp = torch.zeros(y.shape[0], 7, y.shape[2], y.shape[3], dtype=y.dtype, device=y.device)
    inp[:, 1:2] = keep
    res = physics_residual_torch(y, inp, dx, dy, nu)
    return res["continuity"], res["momentum_x"], res["momentum_y"]


def objective(y, keep, dx, dy, nu, cont_scale, mom_scale, fn):
    """Per-sample ``J = 1/2 * mean_cells(rc^2 + rx^2 + ry^2)``; returns ``(B,)``."""
    c, rx, ry = fn(y, keep, dx, dy, nu, cont_scale, mom_scale)
    return 0.5 * (c ** 2 + rx ** 2 + ry ** 2).mean(dim=(1, 2, 3))


# --------------------------------------------------------------------------- #
# Batched error metrics (match physics.evaluation exactly)
# --------------------------------------------------------------------------- #
def metrics(y, gt, fluid):
    """Per-sample error metrics over the fluid volume. All tensors ``(B, 4|1, H, W)``.

    Returns a dict of ``(B,)`` tensors. ``mse_p_gauge`` subtracts the fluid-mean pressure
    offset first: a uniform p shift is an exact null mode of the monitored residual
    (ker L), so raw ``mse_p`` drift is not attributable to the objective.
    """
    n = fluid.sum(dim=(1, 2, 3)).clamp_min(1.0)

    def mse(a, b):
        return (((a - b) ** 2) * fluid).sum(dim=(1, 2, 3)) / n

    u, v, p, nut = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    gu, gv, gp, gnut = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3], gt[:, 3:4]
    dp = ((p * fluid).sum(dim=(1, 2, 3)) - (gp * fluid).sum(dim=(1, 2, 3))) / n
    sp = torch.sqrt(u ** 2 + v ** 2)
    sg = torch.sqrt(gu ** 2 + gv ** 2)
    num = torch.sqrt((((sp - sg) ** 2) * fluid).sum(dim=(1, 2, 3)))
    den = torch.sqrt(((sg ** 2) * fluid).sum(dim=(1, 2, 3))) + 1e-12
    return {
        "mse_u": mse(u, gu),
        "mse_v": mse(v, gv),
        "mse_p": mse(p, gp),
        "mse_p_gauge": mse(p - dp.view(-1, 1, 1, 1), gp),
        "mse_nut": mse(nut, gnut),
        "mse_speed": mse(sp, sg),
        "rel_l2_speed": num / den,
    }


# --------------------------------------------------------------------------- #
# Batch assembly
# --------------------------------------------------------------------------- #
def dilate(m: np.ndarray, k: int) -> np.ndarray:
    """``k``-fold 4-connected dilation of a boolean map."""
    out = m.copy()
    for _ in range(k):
        nxt = out.copy()
        nxt[:-1, :] |= out[1:, :]
        nxt[1:, :] |= out[:-1, :]
        nxt[:, :-1] |= out[:, 1:]
        nxt[:, 1:] |= out[:, :-1]
        out = nxt
    return out


def build_batch(pairs, starts, mode: str):
    """Assemble the batched tensors for one arm.

    ``starts`` is a list of ``(4, H, W)`` float arrays aligned with ``pairs``.
    Returns ``(Y0, GT, KEEP, FLUID, FREE, cont_scale, mom_scale)``.
    """
    B = len(pairs)
    H, W = pairs[0][1].shape
    Y0 = np.zeros((B, 4, H, W), np.float64)
    GT = np.zeros((B, 4, H, W), np.float64)
    KEEP = np.zeros((B, 1, H, W), np.float64)
    FLUID = np.zeros((B, 1, H, W), np.float64)
    FREE = np.zeros((B, 1, H, W), np.float64)
    cs = np.zeros((B, 1, 1, 1), np.float64)
    ms = np.zeros((B, 1, 1, 1), np.float64)

    for i, ((case, gt), s0) in enumerate(zip(pairs, starts, strict=True)):
        mask = np.asarray(gt.mask if gt.mask is not None else np.ones(gt.shape, DTYPE))
        fluid = mask > 0.5
        ring = _solid_adjacent_fluid(mask.astype(DTYPE))     # monitor zeroes these
        keep = fluid & ~ring
        GT[i] = gt.as_array().astype(np.float64)
        Y0[i] = np.asarray(s0, np.float64)
        KEEP[i, 0] = keep
        FLUID[i, 0] = fluid
        if mode == "free":
            free = np.ones_like(fluid)
        else:
            # Freeze: solid, a 2-cell near-wall band, and a 2-cell outer border ring.
            frozen = dilate(~fluid, 2)
            frozen[:2, :] = True
            frozen[-2:, :] = True
            frozen[:, :2] = True
            frozen[:, -2:] = True
            free = ~frozen
            # Hand the optimiser the exact Dirichlet data on the frozen cells.
            Y0[i][:, frozen] = GT[i][:, frozen]
        FREE[i, 0] = free
        u_inf = max(float(case.bc.u_inf), 1e-9)
        L = max(float(case.reference_length()), 1e-9)
        cs[i] = u_inf / L
        ms[i] = u_inf * u_inf / L

    t = lambda a: torch.from_numpy(a)  # noqa: E731
    return t(Y0), t(GT), t(KEEP), t(FLUID), t(FREE), t(cs), t(ms)


# --------------------------------------------------------------------------- #
# The descent
# --------------------------------------------------------------------------- #
LOG_STEPS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]


def descend(Y0, GT, KEEP, FLUID, FREE, cs, ms, *, dx, dy, nu, n_steps, rule, eta,
            var_channels, objective_fn):
    """Run the descent for one arm; returns per-case records.

    ``rule='armijo'``: per-sample backtracking line search on J (c1=1e-4), guaranteeing
    a monotone J decrease. ``rule='fixed'``: plain ``u <- u - eta*grad J``.
    """
    B = Y0.shape[0]
    chan = torch.zeros(1, 4, 1, 1, dtype=F64)
    for c in var_channels:
        chan[0, c, 0, 0] = 1.0
    proj = FREE * chan                       # (B,4,H,W) updatable-entry mask

    Y = Y0.clone()
    J0 = objective(Y, KEEP, dx, dy, nu, cs, ms, objective_fn).detach()
    m0 = {k: v.clone() for k, v in metrics(Y, GT, FLUID).items()}

    traj = {k: [] for k in ("step", "J", "rn", "mse_u", "mse_v", "mse_p",
                            "mse_p_gauge", "mse_nut", "rel_l2_speed")}
    best_J = J0.clone()
    best_J_step = torch.zeros(B, dtype=torch.long)
    best_err = m0["rel_l2_speed"].clone()
    best_err_step = torch.zeros(B, dtype=torch.long)
    ever_better = torch.zeros(B, dtype=torch.bool)
    # matched-J bookkeeping: error the first time J/J0 drops below each ratio.
    # J/J0 = 0.25 <=> the monitored residual_norm is exactly HALVED (rn = sqrt(2J)).
    ratios = [0.5, 0.25, 0.1, 0.01]
    matched = {r: {"hit": torch.zeros(B, dtype=torch.bool),
                   "err": torch.full((B,), float("nan"), dtype=F64),
                   "mse_u": torch.full((B,), float("nan"), dtype=F64),
                   "step": torch.full((B,), float("nan"), dtype=F64)} for r in ratios}
    # Armijo starts from a deliberately huge trial step and halves down (up to 60 times,
    # i.e. 10^8 .. 10^-10) so the search, not the operator, sets the scale on step 1.
    step_t = torch.full((B, 1, 1, 1), 1e8 if rule == "armijo" else float(eta), dtype=F64)
    stationary = torch.zeros(B, dtype=torch.bool)   # Armijo found no admissible step
    dead = torch.zeros(B, dtype=torch.bool)         # fixed-eta run went non-finite

    def record(k, J, m):
        traj["step"].append(int(k))
        traj["J"].append(J.tolist())
        traj["rn"].append(torch.sqrt(2 * J.clamp_min(0)).tolist())
        for key in ("mse_u", "mse_v", "mse_p", "mse_p_gauge", "mse_nut", "rel_l2_speed"):
            traj[key].append(m[key].tolist())

    record(0, J0, m0)

    for k in range(1, n_steps + 1):
        Yv = Y.detach().requires_grad_(True)
        J = objective(Yv, KEEP, dx, dy, nu, cs, ms, objective_fn)
        g, = torch.autograd.grad(J.sum(), Yv)
        g = g * proj
        Jc = J.detach()
        gn2 = (g ** 2).sum(dim=(1, 2, 3))

        if rule == "armijo":
            t = (step_t * 2.0).clamp(max=1e12)
            ok = torch.zeros(B, dtype=torch.bool)
            Ynew = Y.clone()
            for _ in range(60):
                cand = Y - t * g
                Jn = objective(cand, KEEP, dx, dy, nu, cs, ms, objective_fn).detach()
                acc = (Jn <= Jc - 1e-4 * t.view(-1) * gn2) & torch.isfinite(Jn)
                take = acc & ~ok
                if bool(take.any()):
                    Ynew[take] = cand[take]
                    ok |= acc
                if bool(ok.all()):
                    break
                t = torch.where(ok.view(-1, 1, 1, 1), t, t * 0.5)
            step_t = t
            Y = Ynew
            stationary |= ~ok        # no admissible step => at a stationary point of J
        else:
            cand = Y - float(eta) * g
            bad = ~torch.isfinite(cand).flatten(1).all(dim=1)
            dead |= bad
            keepold = (bad | dead).view(-1, 1, 1, 1)
            Y = torch.where(keepold, Y, cand)

        Jk = objective(Y, KEEP, dx, dy, nu, cs, ms, objective_fn).detach()
        mk = metrics(Y, GT, FLUID)
        imp = Jk < best_J
        best_J = torch.where(imp, Jk, best_J)
        best_J_step = torch.where(imp, torch.full_like(best_J_step, k), best_J_step)
        impe = mk["rel_l2_speed"] < best_err
        best_err = torch.where(impe, mk["rel_l2_speed"], best_err)
        best_err_step = torch.where(impe, torch.full_like(best_err_step, k), best_err_step)
        ever_better |= mk["rel_l2_speed"] < m0["rel_l2_speed"]
        for r in ratios:
            new = (~matched[r]["hit"]) & (Jk <= r * J0) & torch.isfinite(Jk)
            matched[r]["hit"] |= new
            matched[r]["err"] = torch.where(new, mk["rel_l2_speed"], matched[r]["err"])
            matched[r]["mse_u"] = torch.where(new, mk["mse_u"], matched[r]["mse_u"])
            matched[r]["step"] = torch.where(
                new, torch.full_like(matched[r]["step"], float(k)), matched[r]["step"])
        if k in LOG_STEPS:
            record(k, Jk, mk)

    if n_steps not in LOG_STEPS:
        record(n_steps, Jk, mk)

    JF, mF = Jk, mk
    out = []
    for i in range(B):
        rec = {
            "J0": float(J0[i]), "J_final": float(JF[i]),
            "J_ratio": float(JF[i] / max(float(J0[i]), 1e-300)),
            "rn0": float(np.sqrt(2 * float(J0[i]))),
            "rn_final": float(np.sqrt(2 * max(float(JF[i]), 0.0))),
            "argmin_J_step": int(best_J_step[i]),
            "argmin_err_step": int(best_err_step[i]),
            "err_ever_below_start": bool(ever_better[i]),
            "best_err": float(best_err[i]),
            "armijo_stationary": bool(stationary[i]),
            "diverged_nonfinite": bool(dead[i]),
        }
        for key in ("mse_u", "mse_v", "mse_p", "mse_p_gauge", "mse_nut", "rel_l2_speed"):
            rec[f"{key}_0"] = float(m0[key][i])
            rec[f"{key}_final"] = float(mF[key][i])
        for r in (0.5, 0.25, 0.1, 0.01):
            rec[f"err_at_J{r}"] = (float(matched[r]["err"][i])
                                   if bool(matched[r]["hit"][i]) else None)
            rec[f"mse_u_at_J{r}"] = (float(matched[r]["mse_u"][i])
                                     if bool(matched[r]["hit"][i]) else None)
            rec[f"step_at_J{r}"] = (float(matched[r]["step"][i])
                                    if bool(matched[r]["hit"][i]) else None)
        d_rn = rec["rn_final"] - rec["rn0"]
        d_err = rec["rel_l2_speed_final"] - rec["rel_l2_speed_0"]
        rec["exchange_rate"] = (d_err / d_rn) if abs(d_rn) > 1e-12 else None
        out.append(rec)
    return out, traj


# --------------------------------------------------------------------------- #
# Start-field providers
# --------------------------------------------------------------------------- #
def starts_truth(pairs):
    return [gt.as_array().astype(np.float64) for _, gt in pairs]


def starts_transolver(pairs, seed):
    d = os.path.join(GATE_CACHE, f"seed{seed}")
    out, keep = [], []
    for case, gt in pairs:
        f = os.path.join(d, f"{case.name}.npz")
        if not os.path.exists(f):
            continue
        out.append(np.load(f)["raw"].astype(np.float64))
        keep.append((case, gt))
    return keep, out


def starts_fno(pairs, ckpt, device="cpu"):
    """Dropout-FNO predictions from checkpoints/certificates_deq.pt, cached to npz."""
    os.makedirs(FNO_CACHE, exist_ok=True)
    missing = [c.name for c, _ in pairs
               if not os.path.exists(os.path.join(FNO_CACHE, f"{c.name}.npz"))]
    if missing:
        from neuroforge.core.config import Config
        from neuroforge.solver.engine import NeuroForgeEngine

        cfg = Config()
        cfg.train.device = device
        engine = NeuroForgeEngine.from_checkpoint(ckpt, config=cfg)
        log(f"FNO cache: {len(missing)} forwards on {device}")
        t0 = time.time()
        for j, (case, _) in enumerate(pairs):
            f = os.path.join(FNO_CACHE, f"{case.name}.npz")
            if os.path.exists(f):
                continue
            with torch.no_grad():
                arr = engine.predictor.predict(case).as_array().astype(DTYPE)
            np.savez_compressed(f, raw=arr)
            if (j + 1) % 40 == 0:
                log(f"  FNO {j + 1}/{len(pairs)} ({time.time() - t0:.0f}s)")
    return [np.load(os.path.join(FNO_CACHE, f"{c.name}.npz"))["raw"].astype(np.float64)
            for c, _ in pairs]


# --------------------------------------------------------------------------- #
# Verification (run before anything else)
# --------------------------------------------------------------------------- #
def verify(pairs, dx, dy, nu, n=25):
    """Confirm the differentiable monitored J reproduces Diagnostics.residual_norm()."""
    checker = PhysicsChecker()
    sub = pairs[:n]
    Y, GT, KEEP, FLUID, FREE, cs, ms = build_batch(sub, starts_truth(sub), "free")
    J = objective(Y, KEEP, dx, dy, nu, cs, ms, monitored_residual)
    rn_t = torch.sqrt(2 * J).numpy()
    rn_np = np.array([checker.diagnose(gt, case).residual_norm() for case, gt in sub])
    rel = np.abs(rn_t - rn_np) / np.maximum(rn_np, 1e-30)
    # theorem leg (i): uniform freestream field is an exact zero of J
    U0 = torch.zeros_like(Y)
    for i, (case, _) in enumerate(sub):
        ui, vi = case.bc.inlet_vector()
        U0[i, 0] = float(ui)
        U0[i, 1] = float(vi)
    Ju = objective(U0, KEEP, dx, dy, nu, cs, ms, monitored_residual)
    out = {
        "n_cases": len(sub),
        "max_rel_err_vs_Diagnostics_residual_norm": float(rel.max()),
        "mean_rel_err": float(rel.mean()),
        "truth_residual_norm_mean": float(rn_np.mean()),
        "truth_residual_norm_median": float(np.median(rn_np)),
        "uniform_field_residual_norm_max": float(torch.sqrt(2 * Ju).max()),
        "H2_holds_truth_residual_nonzero": bool((rn_np > 1e-6).all()),
    }
    log(f"verify: max rel err vs monitored scalar = {out['max_rel_err_vs_Diagnostics_residual_norm']:.2e}; "
        f"uniform-field ||R||_max = {out['uniform_field_residual_norm_max']:.2e}; "
        f"||r*|| mean = {out['truth_residual_norm_mean']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _wilcoxon(d):
    try:
        from scipy.stats import wilcoxon
        d = np.asarray([x for x in d if np.isfinite(x) and x != 0.0])
        if d.size < 5:
            return None
        return float(wilcoxon(d).pvalue)
    except Exception:
        return None


def _pct(x1, x0):
    """Median relative change in %, or ``None`` when the baseline is ~0 (truth arm)."""
    x0 = np.asarray(x0, np.float64)
    if not np.isfinite(x0).all() or float(np.median(x0)) < 1e-9:
        return None
    d = (np.asarray(x1, np.float64) - x0) / x0
    d = d[np.isfinite(d)]
    return float(np.median(d) * 100) if d.size else None


def aggregate(recs):
    a = lambda k: np.array([r[k] for r in recs], np.float64)  # noqa: E731
    e0, e1 = a("rel_l2_speed_0"), a("rel_l2_speed_final")
    u0, u1 = a("mse_u_0"), a("mse_u_final")
    rn0, rn1 = a("rn0"), a("rn_final")
    out = {
        "n": len(recs),
        "J_ratio_median": float(np.median(a("J_ratio"))),
        "residual_norm_mean_0": float(rn0.mean()),
        "residual_norm_mean_final": float(rn1.mean()),
        "frac_residual_reduced": float(np.mean(rn1 < rn0)),
        "rel_l2_mean_0": float(e0.mean()),
        "rel_l2_mean_final": float(e1.mean()),
        "rel_l2_median_0": float(np.median(e0)),
        "rel_l2_median_final": float(np.median(e1)),
        "rel_l2_median_pct_change": _pct(e1, e0),
        "frac_error_increased": float(np.mean(e1 > e0)),
        "frac_error_decreased": float(np.mean(e1 < e0)),
        "frac_error_ever_below_start": float(np.mean(a("err_ever_below_start"))),
        "mse_u_mean_0": float(u0.mean()),
        "mse_u_mean_final": float(u1.mean()),
        "mse_u_median_pct_change": _pct(u1, u0),
        "mse_v_mean_0": float(a("mse_v_0").mean()),
        "mse_v_mean_final": float(a("mse_v_final").mean()),
        "mse_p_mean_0": float(a("mse_p_0").mean()),
        "mse_p_mean_final": float(a("mse_p_final").mean()),
        "mse_p_gauge_mean_0": float(a("mse_p_gauge_0").mean()),
        "mse_p_gauge_mean_final": float(a("mse_p_gauge_final").mean()),
        "mse_nut_mean_0": float(a("mse_nut_0").mean()),
        "mse_nut_mean_final": float(a("mse_nut_final").mean()),
        "wilcoxon_p_rel_l2": _wilcoxon(e1 - e0),
        "frac_argmin_J_ne_argmin_err": float(np.mean(a("argmin_J_step") != a("argmin_err_step"))),
        "argmin_err_step_median": float(np.median(a("argmin_err_step"))),
        "argmin_J_step_median": float(np.median(a("argmin_J_step"))),
        "frac_armijo_stationary": float(np.mean(a("armijo_stationary"))),
        "frac_diverged_nonfinite": float(np.mean(a("diverged_nonfinite"))),
    }
    ex = np.array([r["exchange_rate"] for r in recs if r["exchange_rate"] is not None])
    out["exchange_rate_median"] = float(np.median(ex)) if ex.size else None
    for r in (0.5, 0.25, 0.1, 0.01):
        hit = np.array([x[f"err_at_J{r}"] is not None for x in recs])
        v = np.array([x[f"err_at_J{r}"] for x in recs if x[f"err_at_J{r}"] is not None])
        vu = np.array([x[f"mse_u_at_J{r}"] for x in recs if x[f"err_at_J{r}"] is not None])
        out[f"frac_reached_J{r}"] = float(hit.mean())
        out[f"rel_l2_median_at_J{r}"] = float(np.median(v)) if v.size else None
        out[f"mse_u_mean_at_J{r}"] = float(vu.mean()) if vu.size else None
        out[f"rel_l2_median_pct_change_at_J{r}"] = _pct(v, e0[hit]) if v.size else None
        out[f"mse_u_median_pct_change_at_J{r}"] = _pct(vu, u0[hit]) if vu.size else None
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cases", type=int, default=200)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--n-steps", type=int, default=300)
    p.add_argument("--stage", default="all", choices=["verify", "eta", "main", "all"])
    p.add_argument("--eta-cases", type=int, default=40)
    p.add_argument("--eta-steps", type=int, default=150)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--fno-ckpt", default=os.path.join("checkpoints", "certificates_deq.pt"))
    p.add_argument("--objective", default="monitored", choices=["monitored", "raw", "both"])
    p.add_argument("--out-dir", default=RESULTS)
    a = p.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(0)
    t_start = time.time()

    pairs = load_airfrans(root="data", task="full", train=False, resolution=a.resolution,
                          limit=a.n_cases, cache_dir="data/cache", download=False,
                          progress=False)
    log(f"loaded {len(pairs)} AirfRANS 'full' test cases at res {a.resolution}")
    dx = float(pairs[0][0].domain.dx)
    dy = float(pairs[0][0].domain.dy)
    nu = float(pairs[0][0].fluid.kinematic_viscosity)
    assert len({(round(c.domain.dx, 12), round(c.domain.dy, 12)) for c, _ in pairs}) == 1

    ver = verify(pairs, dx, dy, nu)
    with open(os.path.join(a.out_dir, "verification.json"), "w") as f:
        json.dump(ver, f, indent=2)
    if ver["max_rel_err_vs_Diagnostics_residual_norm"] > 1e-5:
        raise SystemExit("differentiable monitored residual does NOT match the monitor")
    if a.stage == "verify":
        return 0

    # ---- assemble the arms ------------------------------------------------ #
    arms = {}
    arms["truth"] = (pairs, starts_truth(pairs))
    for s in a.seeds:
        kp, st = starts_transolver(pairs, s)
        if st:
            arms[f"transolver_seed{s}"] = (kp, st)
    if os.path.exists(os.path.join(ROOT, a.fno_ckpt)):
        arms["fno"] = (pairs, starts_fno(pairs, os.path.join(ROOT, a.fno_ckpt)))
    log("arms: " + ", ".join(f"{k}(n={len(v[0])})" for k, v in arms.items()))

    objs = {"monitored": monitored_residual, "raw": raw_residual}
    obj_list = ["monitored", "raw"] if a.objective == "both" else [a.objective]

    def run(arm, mode, varset, rule, eta, n_steps, obj_name, subset=None):
        prs, sts = arms[arm]
        if subset is not None:
            prs, sts = prs[:subset], sts[:subset]
        chans = (0, 1, 2) if varset == "uvp" else (0, 1, 2, 3)
        recs, traj = [], None
        for i in range(0, len(prs), a.batch):
            pb, sb = prs[i:i + a.batch], sts[i:i + a.batch]
            Y, GT, KEEP, FLUID, FREE, cs, ms = build_batch(pb, sb, mode)
            if obj_name == "raw":
                KEEP = FLUID          # the raw twin masks the solid only
            r, tr = descend(Y, GT, KEEP, FLUID, FREE, cs, ms, dx=dx, dy=dy, nu=nu,
                            n_steps=n_steps, rule=rule, eta=eta, var_channels=chans,
                            objective_fn=objs[obj_name])
            for j, rec in enumerate(r):
                rec["name"] = pb[j][0].name
            recs += r
            if traj is None:
                traj = tr
        return recs, traj

    summary = {"meta": {
        "n_cases": len(pairs), "resolution": a.resolution, "n_steps": a.n_steps,
        "device": "cpu", "dx": dx, "dy": dy, "nu": nu,
        "torch_threads": torch.get_num_threads(),
        "objective": "J = 0.5*mean(rc^2+rx^2+ry^2) of the MONITORED (non-dimensionalised, "
                     "wall-ring-zeroed) residual; residual_norm = sqrt(2J) == "
                     "Diagnostics.residual_norm()",
        "verification": ver,
    }, "runs": {}}

    # ---- stage: eta sensitivity ------------------------------------------ #
    if a.stage in ("eta", "all"):
        etas = [10.0 ** k for k in range(-4, 8)]
        eta_out = {}
        for arm in ("truth", "transolver_seed0", "fno"):
            if arm not in arms:
                continue
            for mode in ("free", "bc"):
                rows = []
                for eta in etas:
                    t0 = time.time()
                    recs, _ = run(arm, mode, "uvp", "fixed", eta, a.eta_steps,
                                  "monitored", subset=a.eta_cases)
                    ag = aggregate(recs)
                    ag["eta"] = eta
                    rows.append(ag)
                    pc = ag["rel_l2_median_pct_change"]
                    log(f"eta {arm}/{mode} eta={eta:.0e}: J_ratio={ag['J_ratio_median']:.3g} "
                        f"rel_l2 {ag['rel_l2_median_0']:.5f}->{ag['rel_l2_median_final']:.5f} "
                        f"({'n/a' if pc is None else f'{pc:+.1f}%'}) "
                        f"frac_err_up={ag['frac_error_increased']:.2f} ({time.time()-t0:.0f}s)")
                eta_out[f"{arm}|{mode}"] = rows
        with open(os.path.join(a.out_dir, "eta_sensitivity.json"), "w") as f:
            json.dump({"n_cases": a.eta_cases, "n_steps": a.eta_steps,
                       "etas": etas, "rows": eta_out}, f, indent=2)

    # ---- stage: main (Armijo, full n) ------------------------------------- #
    if a.stage in ("main", "all"):
        combos = []
        for arm in arms:
            for mode in ("bc", "free"):
                combos.append((arm, mode, "uvp"))
        combos.append(("truth", "bc", "uvpn"))
        combos.append(("transolver_seed0", "bc", "uvpn"))
        for obj_name in obj_list:
            for arm, mode, varset in combos:
                tag = f"{arm}_{mode}_{varset}_armijo" + ("" if obj_name == "monitored"
                                                         else f"_{obj_name}")
                t0 = time.time()
                recs, traj = run(arm, mode, varset, "armijo", 1e-6, a.n_steps, obj_name)
                ag = aggregate(recs)
                ag["wall_s"] = time.time() - t0
                summary["runs"][tag] = ag
                with open(os.path.join(a.out_dir, f"descent_{tag}.json"), "w") as f:
                    json.dump({"tag": tag, "arm": arm, "mode": mode, "vars": varset,
                               "rule": "armijo", "objective": obj_name,
                               "n_steps": a.n_steps, "aggregate": ag,
                               "trajectory_batch0": traj, "per_case": recs}, f, indent=2)
                pc = ag["rel_l2_median_pct_change"]
                log(f"MAIN {tag}: n={ag['n']} J_ratio_med={ag['J_ratio_median']:.3g} "
                    f"resid {ag['residual_norm_mean_0']:.4f}->{ag['residual_norm_mean_final']:.4f} "
                    f"| rel_l2 {ag['rel_l2_median_0']:.5f}->{ag['rel_l2_median_final']:.5f} "
                    f"({'n/a' if pc is None else f'{pc:+.1f}%'}) "
                    f"frac_err_up={ag['frac_error_increased']:.3f} "
                    f"mse_u {ag['mse_u_mean_0']:.3f}->{ag['mse_u_mean_final']:.3f} "
                    f"({time.time()-t0:.0f}s)")

    summary["meta"]["wall_s_total"] = time.time() - t_start
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"done in {time.time() - t_start:.0f}s -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
