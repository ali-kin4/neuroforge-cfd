"""The baseline nobody publishes: predict the AirfRANS flow field from the CASE
PARAMETERS alone, by interpolation in parameter space. No neural network, no
flow-field learning, no geometry encoder.

===========================================================================
WHY THIS EXISTS
===========================================================================
AirfRANS cases are indexed by a tiny parameter vector that the simulation NAME
hands over for free::

    airFoil2D_SST_<inlet_velocity>_<angle_of_attack>_<NACA digits...>

``airfrans.simulation.Simulation.reset()`` itself parses fields 2 and 3 as U and
alpha; fields 4.. are the NACA 4-/5-digit camber+thickness digits, which define
the airfoil EXACTLY (``airfrans.naca_generator``). ``scripts/drag_covariate_control.py``
already showed that a three-parameter regression on ``(U, alpha, alpha^2)`` ranks
the OFFICIAL drag label at ``rho = 0.874`` -- beating every field-based integrator
we tried, including on exact ground-truth fields.

If the *scalar* is that recoverable from the case label, how much of the *field*
is? Every AirfRANS surrogate paper reports learned-model MSE without ever
reporting what a parameter-space interpolator scores under the same protocol.
This script supplies that missing row.

===========================================================================
WHAT IT DOES
===========================================================================
Features (analytic, from the NAME only -- the interpolator never sees a test
flow field, and sees STRICTLY LESS input than the surrogates, whose 7-channel
spec already contains ``sdf, mask, x, y, u_in, v_in, log_re``):

    f = [ U, alpha_deg, t_max, alpha_eff, y_c(0.25), y_c(0.50), y_c(0.75) ]

with ``t_max`` the NACA thickness fraction, ``y_c(.)`` the exact camber line
(reimplemented from ``airfrans.naca_generator.camber_line``, validated against
it in ``--selftest``), and ``alpha_eff = alpha - alpha_0L`` the thin-airfoil
effective incidence (``alpha_0L`` from the exact camber slope). Features are
standardised on TRAIN ONLY; the U feature carries a swept weight ``w_U``.

Field representations:
  * ``raw`` -- interpolate (u, v, p, nut) cell-wise.
  * ``nd``  -- interpolate the NONDIMENSIONAL perturbation
        u_hat = (u - U cos a)/U,  v_hat = (v - U sin a)/U,
        p_hat = p/U^2,            nut_hat = nut/(U c)
    and redimensionalise with the TEST case's own (U, alpha). This is the
    honest headline representation: ``mse_p`` scales as U^4 across the split, so
    a raw average over neighbours at different U is dominated by scale mismatch.

Solid-region handling (this is the difference between a baseline and a strawman):
``airfrans_loader._sim_to_pair`` zeroes u/v/nut inside the body but leaves p as
the Delaunay bridge interpolation. A test cell that is FLUID in the test airfoil
but SOLID in a thicker neighbour would otherwise receive a hard ``u=v=0`` --
exactly the annulus where ``surface_pressure_mse`` bilinear-samples. Default
``--fill nearest`` extends each TRAIN field's u/v/nut into its own solid by
nearest-fluid propagation before interpolation; ``--fill none`` is kept as the
naive ablation.

Estimators (all reduce to a weight matrix ``W`` over train cases, so prediction
is one GEMM):
  * ``mean``    -- train-set mean field (floor).
  * ``freestream`` -- u = U cos a, v = U sin a, p = 0 (zero-training floor).
  * ``nn``      -- nearest neighbour in standardised parameter space.
  * ``knn``     -- inverse-distance-weighted k-NN (k swept).
  * ``locallin``-- distance-weighted LOCAL LINEAR regression on the features
                   over the k nearest neighbours (k swept).
  * ``krr``     -- global Gaussian-kernel ridge regression / RBF interpolation
                   (bandwidth and ridge swept).

===========================================================================
PRE-REGISTERED PROTOCOL AND DECISION RULE  (committed BEFORE the run)
===========================================================================
P1. Scoring is the IDENTICAL path used for the surrogates:
    ``neuroforge.physics.evaluation.evaluate_cases`` on the SAME cached
    rasterised ``(FlowCase, FlowField)`` pairs at resolution 128 that
    ``scripts/run_baselines.py`` scores Transolver on. Same crop, same mask,
    same surface sampler, same force integrator.

P2. MODEL SELECTION IS TRAIN-ONLY. Hyperparameters (estimator, k, bandwidth,
    ridge, w_U) are chosen by 5-fold cross-validation WITHIN the train split,
    on a fixed random subsample of cells, using the pre-registered scalar

        S = mean over channels {u, v, p} of  MSE_channel / Var_train(channel)

    (per-channel variance normalisation: a raw sum lets ``mse_p`` dominate by
    four orders of magnitude). The selected config is applied to the test split
    ONCE. The full test sweep is also written out, for transparency, but the
    headline row is the CV-selected one.

P3. VERDICT on ``full`` (200 test cases), against the paper's own rows
    -- Transolver ``mse_u`` 0.120 / ``mse_v`` 0.088 / ``mse_p`` 628.5
    -- our grid backbone ``mse_u`` 3.479 / ``mse_v`` 0.385 / ``mse_p`` 2444.8:

      COMPETITIVE      : the CV-selected interpolator is within 2x of Transolver
                         on ALL THREE volume channels (u, v, p).
      MIXED            : within 2x of Transolver on at least one volume channel
                         but not all three; OR beats the grid-backbone row on
                         all three while >2x worse than Transolver on all three.
      CLEARLY BEATEN   : >2x worse than Transolver on all three volume channels
                         AND does not beat the grid-backbone row on all three.

    Secondary, reported regardless: does interpolation beat the GRID BACKBONE
    row (a surrogate this paper publishes)? That is a concrete finding
    independent of how the Transolver comparison lands.

P4. FORCES. ``rho_cl``/``rho_cd`` from ``evaluate_cases`` are field-to-field
    self-consistency (pred-integral vs GT-integral through the same
    integrator), matching the paper's tables. Against the OFFICIAL AirfRANS
    labels we additionally report the PARTIAL Spearman controlling for
    ``(U, alpha)`` exactly as ``scripts/drag_covariate_control.py`` does, since
    a raw correlation on this benchmark is uninterpretable.

P5. OOD. The same CV-selected-per-task procedure is run on ``reynolds`` and
    ``aoa``. Pre-registered expectation: interpolation degrades MORE than a
    learned surrogate on regime-disjoint extrapolation. If it does NOT degrade,
    that is a finding about those splits, and is reported as such.

P6. LEAKAGE / DUPLICATE CHECKS, reported whatever they show:
    (a) train/test case-name overlap must be 0;
    (b) the distribution of nearest-neighbour parameter distance for each test
        case, and the per-case error-vs-distance rank correlation;
    (c) the headline metrics recomputed with the closest 10% of test cases
        EXCLUDED (if a handful of near-duplicates carry the aggregate, this
        moves);
    (d) ``n_cases`` from ``evaluate_cases`` asserted == the test-set size
        (``coefficient_metrics`` silently drops cases whose force integration
        raises).

===========================================================================
RUNNING  (CPU-only, no GPU, no training)
===========================================================================
    .venv/Scripts/python.exe scripts/parameter_interpolation_baseline.py --selftest
    .venv/Scripts/python.exe scripts/parameter_interpolation_baseline.py --task full
    .venv/Scripts/python.exe scripts/parameter_interpolation_baseline.py --task reynolds
    .venv/Scripts/python.exe scripts/parameter_interpolation_baseline.py --task aoa

Writes ``results/interpolation/interp_<task>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import neuroforge  # noqa: F401  -- MUST precede numpy/scipy (BLAS thread caps)

import numpy as np

from neuroforge.core.types import FlowField
from neuroforge.geometry.sdf import signed_distance, solid_mask
from neuroforge.physics.evaluation import (
    coefficient_metrics,
    evaluate_cases,
    per_channel_mse,
    surface_pressure_mse,
)
from neuroforge.physics.metrics import force_coefficients

CHORD = 1.0
CHANNELS = ("u", "v", "p", "nut")

# The paper's own rows, for the pre-registered verdict (docs/paper/body.tex,
# tab:transolver / tab:v2). Volume MSE in physical units on the r128 crop.
REFERENCE_ROWS = {
    "transolver_tab_transolver": {"mse_u": 0.120, "mse_v": 0.088, "mse_p": 628.5,
                                  "surface_mse_p": 9110.0},
    "transolver_tab_v2_backbone": {"mse_u": 0.133, "mse_v": 0.096, "mse_p": 644.0,
                                   "surface_mse_p": 9843.0},
    "grid_backbone_tab_indist": {"mse_u": 3.479, "mse_v": 0.385, "mse_p": 2444.8,
                                 "surface_mse_p": 548823.0},
}


# ---------------------------------------------------------------------------
# 1. Case parameters, analytic from the simulation NAME
# ---------------------------------------------------------------------------
def _camber_line(cam: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NACA 4-/5-digit camber line and slope.

    Reimplementation of ``airfrans.naca_generator.camber_line`` (kept local so
    this script has no pyvista dependency); ``--selftest`` asserts agreement
    with the library to 1e-12 on every case in the dataset manifest.
    """
    x = np.asarray(x, np.float64)
    y_c = np.zeros_like(x)
    dy_c = np.zeros_like(x)
    if len(cam) == 2:                                    # 4-digit
        m = cam[0] / 100.0
        p = cam[1] / 10.0
        if p == 0:
            return y_c, -2 * m * x
        if p == 1:
            return y_c, 2 * m * (1 - x)
        m1 = x < p
        m2 = ~m1
        y_c[m1] = (m / p**2) * (2 * p * x[m1] - x[m1] ** 2)
        dy_c[m1] = (2 * m / p**2) * (p - x[m1])
        y_c[m2] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x[m2] - x[m2] ** 2)
        dy_c[m2] = (2 * m / (1 - p) ** 2) * (p - x[m2])
        return y_c, dy_c
    if len(cam) == 3:                                    # 5-digit
        ll, p, q = cam
        c_l, x_f = 3 / 20 * ll, p / 20
        old_m = np.array(0.5)
        while True:
            f = old_m * (1 - np.sqrt(old_m / 3)) - x_f
            df = 1 - 3 * np.sqrt(old_m / 3) / 2
            new_m = np.max(np.array([old_m - f / df, 0]))
            if np.abs(old_m - new_m) <= 1e-15:
                break
            old_m = new_m
        m = old_m
        r = (3 * m - 7 * m**2 + 8 * m**3 - 4 * m**4) / np.sqrt(m * (1 - m)) - 3 / 2 * (
            1 - 2 * m
        ) * (np.pi / 2 - np.arcsin(1 - 2 * m))
        k_1 = c_l / r
        m1 = x <= m
        m2 = ~m1
        if q == 0:
            y_c[m1] = k_1 * (x[m1] ** 3 - 3 * m * x[m1] ** 2 + m**2 * (3 - m) * x[m1])
            dy_c[m1] = k_1 * (3 * x[m1] ** 2 - 6 * m * x[m1] + m**2 * (3 - m))
            y_c[m2] = k_1 * m**3 * (1 - x[m2])
            dy_c[m2] = -k_1 * m**3 * np.ones_like(dy_c[m2])
        elif q == 1:
            k = (3 * (m - x_f) ** 2 - m**3) / (1 - m) ** 3
            y_c[m1] = k_1 * ((x[m1] - m) ** 3 - k * (1 - m) ** 3 * x[m1]
                             - m**3 * x[m1] + m**3)
            dy_c[m1] = k_1 * (3 * (x[m1] - m) ** 2 - k * (1 - m) ** 3 - m**3)
            y_c[m2] = k_1 * (k * (x[m2] - m) ** 3 - k * (1 - m) ** 3 * x[m2]
                             - m**3 * x[m2] + m**3)
            dy_c[m2] = k_1 * (3 * k * (x[m2] - m) ** 2 - k * (1 - m) ** 3 - m**3)
        else:
            raise ValueError(f"unexpected 5-digit reflex flag q={q}")
        return y_c, dy_c
    raise ValueError(f"expected 2 or 3 camber digits, got {len(cam)}")


def _alpha_zero_lift_deg(cam: np.ndarray) -> float:
    """Thin-airfoil zero-lift angle (deg) from the exact camber slope."""
    th = np.linspace(1e-6, np.pi - 1e-6, 512)
    x = 0.5 * (1 - np.cos(th))
    _, dy = _camber_line(cam, x)
    return float(np.degrees(-(1.0 / np.pi) * np.trapezoid(dy * (np.cos(th) - 1.0), th)))


def case_features(name: str) -> np.ndarray:
    """``[U, alpha_deg, t_max, alpha_eff_deg, y_c(.25), y_c(.50), y_c(.75)]``."""
    f = name.split("_")
    U = float(f[2])
    alpha = float(f[3])
    g = [float(v) for v in f[4:]]
    t = g[-1] / 100.0
    cam = np.array(g[:-1], np.float64)
    y_c, _ = _camber_line(cam, np.array([0.25, 0.50, 0.75]))
    a0 = _alpha_zero_lift_deg(cam)
    return np.array([U, alpha, t, alpha - a0, y_c[0], y_c[1], y_c[2]], np.float64)


def case_U_alpha(name: str) -> tuple[float, float]:
    f = name.split("_")
    return float(f[2]), float(f[3])


# ---------------------------------------------------------------------------
# 2. Field stacks
# ---------------------------------------------------------------------------
def _nearest_fluid_fill(a: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """Extend ``a`` into ``~fluid`` by nearest-fluid-cell propagation."""
    if fluid.all():
        return a
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
    except Exception:  # pragma: no cover
        return a
    _, idx = distance_transform_edt(~fluid, return_distances=True, return_indices=True)
    return a[tuple(idx)]


def build_stack(pairs, rep: str, fill: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(Y (n, 4*H*W) float32, masks (n, H*W) bool, names)`` in representation ``rep``."""
    names = [c.name for c, _ in pairs]
    n = len(pairs)
    H, W = pairs[0][1].u.shape
    Y = np.empty((n, 4 * H * W), np.float32)
    M = np.empty((n, H * W), bool)
    for i, (case, fld) in enumerate(pairs):
        fluid = np.asarray(fld.mask) > 0.5
        U, a = case_U_alpha(case.name)
        ar = np.radians(a)
        u = np.asarray(fld.u, np.float64)
        v = np.asarray(fld.v, np.float64)
        p = np.asarray(fld.p, np.float64)
        nut = np.asarray(fld.nut, np.float64)
        if fill == "nearest":
            u = _nearest_fluid_fill(u, fluid)
            v = _nearest_fluid_fill(v, fluid)
            nut = _nearest_fluid_fill(nut, fluid)
        if rep == "nd":
            u = (u - U * np.cos(ar)) / U
            v = (v - U * np.sin(ar)) / U
            p = p / U**2
            nut = nut / (U * CHORD)
        elif rep != "raw":
            raise ValueError(rep)
        Y[i] = np.concatenate([u.ravel(), v.ravel(), p.ravel(), nut.ravel()]).astype(np.float32)
        M[i] = fluid.ravel()
    return Y, M, names


def redimensionalise(pred: np.ndarray, names: list[str], rep: str, hw: int) -> np.ndarray:
    """Invert the ``nd`` transform using each QUERY case's own (U, alpha)."""
    if rep == "raw":
        return pred
    out = np.array(pred, np.float64, copy=True)
    for i, nm in enumerate(names):
        U, a = case_U_alpha(nm)
        ar = np.radians(a)
        out[i, 0 * hw:1 * hw] = out[i, 0 * hw:1 * hw] * U + U * np.cos(ar)
        out[i, 1 * hw:2 * hw] = out[i, 1 * hw:2 * hw] * U + U * np.sin(ar)
        out[i, 2 * hw:3 * hw] = out[i, 2 * hw:3 * hw] * U**2
        out[i, 3 * hw:4 * hw] = out[i, 3 * hw:4 * hw] * (U * CHORD)
    return out


# ---------------------------------------------------------------------------
# 3. Estimators -> weight matrix over train cases
# ---------------------------------------------------------------------------
def _standardise(Xtr: np.ndarray, X: np.ndarray, w_U: float) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    w = np.ones(Xtr.shape[1])
    w[0] = w_U                              # feature 0 is U (Reynolds proxy)
    return ((Xtr - mu) / sd) * w, ((X - mu) / sd) * w


def _dists(Zq: np.ndarray, Ztr: np.ndarray) -> np.ndarray:
    d2 = (
        (Zq**2).sum(1)[:, None] + (Ztr**2).sum(1)[None, :] - 2.0 * Zq @ Ztr.T
    )
    return np.sqrt(np.maximum(d2, 0.0))


def build_weights(cfg: dict, Xtr: np.ndarray, Xq: np.ndarray,
                  active: np.ndarray | None = None) -> np.ndarray:
    """``W (n_q, n_train)`` such that ``pred = W @ Y``.

    ``active`` masks which train rows may be used (CV fold exclusion).
    """
    n_tr = Xtr.shape[0]
    if active is None:
        active = np.ones(n_tr, bool)
    Ztr, Zq = _standardise(Xtr[active], Xq, cfg.get("w_U", 1.0))
    idx = np.nonzero(active)[0]
    m = cfg["method"]
    nq = Xq.shape[0]
    Wfull = np.zeros((nq, n_tr), np.float32)

    if m == "mean":
        Wfull[:, idx] = 1.0 / len(idx)
        return Wfull

    D = _dists(Zq, Ztr)

    if m in ("nn", "knn"):
        k = 1 if m == "nn" else int(cfg["k"])
        k = min(k, len(idx))
        pw = float(cfg.get("power", 2.0))
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(D, part, axis=1)
        wk = 1.0 / (dk**pw + 1e-9)
        wk /= wk.sum(1, keepdims=True)
        for i in range(nq):
            Wfull[i, idx[part[i]]] = wk[i].astype(np.float32)
        return Wfull

    if m == "locallin":
        k = min(int(cfg["k"]), len(idx))
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        ridge = float(cfg.get("ridge", 1e-6))
        F = Ztr.shape[1]
        for i in range(nq):
            sel = part[i]
            dk = D[i, sel]
            h = np.median(dk) + 1e-12
            om = np.exp(-0.5 * (dk / h) ** 2)             # tricube-like gaussian
            A = np.column_stack([np.ones(k), Ztr[sel] - Zq[i]])
            AtW = A.T * om
            G = AtW @ A + ridge * np.eye(F + 1)
            try:
                coef = np.linalg.solve(G, AtW)             # (F+1, k)
            except np.linalg.LinAlgError:                  # pragma: no cover
                coef = np.linalg.lstsq(G, AtW, rcond=None)[0]
            Wfull[i, idx[sel]] = coef[0].astype(np.float32)
        return Wfull

    if m == "krr":
        Dtr = _dists(Ztr, Ztr)
        med = float(np.median(Dtr[Dtr > 0])) if (Dtr > 0).any() else 1.0
        sig = float(cfg["sigma"]) * med
        K = np.exp(-0.5 * (Dtr / sig) ** 2)
        lam = float(cfg["lam"])
        A = K + lam * np.eye(len(idx))
        Kq = np.exp(-0.5 * (D / sig) ** 2)
        try:
            Wq = np.linalg.solve(A, Kq.T).T
        except np.linalg.LinAlgError:                      # pragma: no cover
            Wq = np.linalg.lstsq(A, Kq.T, rcond=None)[0].T
        # KRR on centred targets: pred = mean + Wq @ (Y - mean)
        Wfull[:, idx] = (Wq + (1.0 - Wq.sum(1, keepdims=True)) / len(idx)).astype(np.float32)
        return Wfull

    raise ValueError(f"unknown method {m}")


def freestream_stack(names: list[str], hw: int, rep: str) -> np.ndarray:
    """Zero-training floor: uniform freestream (and p=0) everywhere."""
    out = np.zeros((len(names), 4 * hw), np.float64)
    if rep == "raw":
        for i, nm in enumerate(names):
            U, a = case_U_alpha(nm)
            out[i, 0 * hw:1 * hw] = U * np.cos(np.radians(a))
            out[i, 1 * hw:2 * hw] = U * np.sin(np.radians(a))
    return out  # in 'nd' the freestream IS zero


# ---------------------------------------------------------------------------
# 4. Cross-validated model selection (TRAIN ONLY)
# ---------------------------------------------------------------------------
def config_grid(rep: str) -> list[dict]:
    cfgs: list[dict] = []
    w_Us = [0.25, 1.0] if rep == "nd" else [1.0, 0.25]
    for w_U in w_Us:
        cfgs.append({"method": "nn", "w_U": w_U})
        for k in (2, 4, 8, 16, 32, 64):
            cfgs.append({"method": "knn", "k": k, "power": 2.0, "w_U": w_U})
        for k in (16, 32, 64, 128):
            cfgs.append({"method": "locallin", "k": k, "ridge": 1e-4, "w_U": w_U})
        for sigma in (0.5, 1.0, 2.0):
            for lam in (1e-3, 1e-1):
                cfgs.append({"method": "krr", "sigma": sigma, "lam": lam, "w_U": w_U})
    cfgs.append({"method": "mean", "w_U": 1.0})
    return cfgs


def cfg_name(c: dict) -> str:
    m = c["method"]
    if m == "knn":
        return f"knn_k{c['k']}_p{int(c['power'])}_wU{c['w_U']}"
    if m == "locallin":
        return f"locallin_k{c['k']}_wU{c['w_U']}"
    if m == "krr":
        return f"krr_s{c['sigma']}_l{c['lam']}_wU{c['w_U']}"
    return f"{m}_wU{c['w_U']}"


def cv_select(Xtr, Ytr, Mtr, names_tr, rep, hw, n_folds, n_cells, seed, verbose=True):
    """5-fold CV inside the train split. Returns (best_cfg, table)."""
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    # Fixed cell subsample, per channel, restricted to cells fluid in >=95% of cases.
    common = Mtr.mean(0) > 0.95
    pool = np.nonzero(common)[0]
    sub = rng.choice(pool, size=min(n_cells, pool.size), replace=False)
    cols = np.concatenate([sub + ch * hw for ch in range(4)])
    Ys = np.ascontiguousarray(Ytr[:, cols])
    # Per-channel train variance of the REDIMENSIONALISED field (physical units),
    # so the selection scalar is comparable across representations.
    Yphys = redimensionalise(Ytr[:, cols], names_tr, rep, sub.size)
    var = np.array([np.var(Yphys[:, ch * sub.size:(ch + 1) * sub.size]) for ch in range(4)])
    var[var < 1e-30] = 1.0

    folds = np.array_split(rng.permutation(n), n_folds)
    cfgs = config_grid(rep)
    table = []
    for ci, cfg in enumerate(cfgs):
        se = np.zeros(4)
        cnt = 0
        for fold in folds:
            active = np.ones(n, bool)
            active[fold] = False
            W = build_weights(cfg, Xtr, Xtr[fold], active=active)
            pred = W @ Ys
            pred = redimensionalise(pred, [names_tr[i] for i in fold], rep, sub.size)
            ref = Yphys[fold]
            d = (pred - ref) ** 2
            for ch in range(4):
                se[ch] += d[:, ch * sub.size:(ch + 1) * sub.size].sum()
            cnt += len(fold) * sub.size
        mse = se / cnt
        S = float(np.mean(mse[:3] / var[:3]))              # u, v, p  (pre-registered)
        table.append({"cfg": cfg_name(cfg), "S": S,
                      "cv_mse_u": float(mse[0]), "cv_mse_v": float(mse[1]),
                      "cv_mse_p": float(mse[2]), "cv_mse_nut": float(mse[3])})
        if verbose and (ci % 10 == 0 or ci == len(cfgs) - 1):
            print(f"  [cv] {ci + 1}/{len(cfgs)} {cfg_name(cfg):28s} S={S:.5f}", flush=True)
    order = np.argsort([t["S"] for t in table])
    table = [table[i] for i in order]
    best = cfgs[[cfg_name(c) for c in cfgs].index(table[0]["cfg"])]
    return best, table


# ---------------------------------------------------------------------------
# 5. Test-set scoring through the SHARED evaluate_cases
# ---------------------------------------------------------------------------
def make_predict_fn(pred_phys: np.ndarray, names: list[str], shape, domainless_pairs):
    """``FlowCase -> FlowField``, geometry-derived mask/sdf exactly as the
    Transolver adapter builds them (``signed_distance`` / ``solid_mask`` from
    ``case.geometry``), never from the ground-truth field object."""
    H, W = shape
    hw = H * W
    by_name = {nm: i for i, nm in enumerate(names)}
    cache: dict[str, FlowField] = {}

    def predict_fn(case):
        if case.name in cache:
            return cache[case.name]
        i = by_name[case.name]
        row = pred_phys[i]
        u = row[0:hw].reshape(H, W)
        v = row[hw:2 * hw].reshape(H, W)
        p = row[2 * hw:3 * hw].reshape(H, W)
        nut = row[3 * hw:4 * hw].reshape(H, W)
        sdf = signed_distance(case.geometry, case.domain)
        mask = solid_mask(case.geometry, case.domain)
        solid = mask < 0.5
        fld = FlowField(
            domain=case.domain,
            u=np.where(solid, 0.0, u).astype(np.float32),
            v=np.where(solid, 0.0, v).astype(np.float32),
            p=p.astype(np.float32),
            nut=np.where(solid, 0.0, np.maximum(nut, 0.0)).astype(np.float32),
            mask=mask, sdf=sdf,
            meta={"source": "parameter_interpolation", "case": case.name},
        )
        cache[case.name] = fld
        return fld

    return predict_fn


def per_case_metrics(predict_fn, pairs):
    rows = []
    pred_c, ref_c = [], []
    for case, ref in pairs:
        pr = predict_fn(case)
        m = per_channel_mse(pr, ref)
        m["surface_mse_p"] = surface_pressure_mse(pr, ref, case)
        m["name"] = case.name
        try:
            pc = force_coefficients(pr, case)
            rc = force_coefficients(ref, case)
            pred_c.append(pc)
            ref_c.append(rc)
            m["pred_cl"], m["pred_cd"] = pc["cl"], pc["cd"]
            m["gt_cl"], m["gt_cd"] = rc["cl"], rc["cd"]
        except Exception:
            pass
        rows.append(m)
    return rows, pred_c, ref_c


# ---------------------------------------------------------------------------
# 6. Partial Spearman vs official labels (mirrors drag_covariate_control.py)
# ---------------------------------------------------------------------------
def partial_spearman(x, y, covars) -> float:
    from scipy.stats import rankdata

    n = len(x)
    design = np.column_stack([np.ones(n)] + [rankdata(c) for c in covars])

    def resid(v):
        rv = rankdata(v)
        beta, *_ = np.linalg.lstsq(design, rv, rcond=None)
        return rv - design @ beta

    rx, ry = resid(x), resid(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def official_force_block(rows, names):
    path = os.path.join("results", "control", "_cache", "official_labels_full_test_n200.json")
    if not os.path.exists(path):
        return {"available": False, "reason": f"missing {path}"}
    from scipy.stats import spearmanr

    labels = json.load(open(path, encoding="utf-8"))
    keep = [r for r in rows if r["name"] in labels and "pred_cd" in r]
    if len(keep) < 10:
        return {"available": False, "reason": f"only {len(keep)} matched cases"}
    nm = [r["name"] for r in keep]
    U = np.array([case_U_alpha(n)[0] for n in nm])
    A = np.array([case_U_alpha(n)[1] for n in nm])
    off_cd = np.array([labels[n]["cd"] for n in nm])
    off_cl = np.array([labels[n]["cl"] for n in nm])
    pcd = np.array([r["pred_cd"] for r in keep])
    pcl = np.array([r["pred_cl"] for r in keep])
    gcd = np.array([r["gt_cd"] for r in keep])
    gcl = np.array([r["gt_cl"] for r in keep])

    def blk(p, o):
        return {"rho_marginal": float(spearmanr(p, o).statistic),
                "rho_partial_U_alpha": partial_spearman(p, o, [U, A])}

    # The covariate reference: OLS of the official label on (U, alpha, alpha^2).
    def fit_rho(cols, target):
        X = np.column_stack([np.ones(len(target))] + cols)
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        return float(spearmanr(X @ beta, target).statistic)

    return {
        "available": True, "n": len(keep),
        "interp_cd_vs_official": blk(pcd, off_cd),
        "interp_cl_vs_official": blk(pcl, off_cl),
        "gt_field_cd_vs_official": blk(gcd, off_cd),
        "gt_field_cl_vs_official": blk(gcl, off_cl),
        "covariate_only_rho_cd": fit_rho([U, A, A**2], off_cd),
        "covariate_only_rho_cl": fit_rho([U, A, A**2], off_cl),
    }


# ---------------------------------------------------------------------------
# 7. Verdict
# ---------------------------------------------------------------------------
def verdict(metrics: dict) -> dict:
    tr = REFERENCE_ROWS["transolver_tab_transolver"]
    gb = REFERENCE_ROWS["grid_backbone_tab_indist"]
    ch = ["mse_u", "mse_v", "mse_p"]
    ratio_tr = {c: metrics[c] / tr[c] for c in ch}
    ratio_gb = {c: metrics[c] / gb[c] for c in ch}
    within2_tr = {c: ratio_tr[c] <= 2.0 for c in ch}
    beats_gb = all(ratio_gb[c] < 1.0 for c in ch)
    if all(within2_tr.values()):
        v = "COMPETITIVE"
    elif any(within2_tr.values()) or beats_gb:
        v = "MIXED"
    else:
        v = "CLEARLY BEATEN"
    return {"verdict": v, "ratio_vs_transolver": ratio_tr,
            "ratio_vs_grid_backbone": ratio_gb,
            "within_2x_transolver": within2_tr, "beats_grid_backbone_row": beats_gb}


# ---------------------------------------------------------------------------
# 8. Self-test
# ---------------------------------------------------------------------------
def selftest() -> int:
    m = json.load(open(os.path.join("data", "Dataset", "manifest.json"), encoding="utf-8"))
    names = sorted({n for v in m.values() for n in v})
    print(f"[selftest] {len(names)} unique simulation names")
    try:
        from airfrans.naca_generator import camber_line as ref_camber  # type: ignore
    except Exception as exc:
        print(f"[selftest] airfrans unavailable ({exc}); skipping camber cross-check")
        ref_camber = None
    x = np.linspace(0.0, 1.0, 97)
    worst = 0.0
    for nm in names:
        g = [float(v) for v in nm.split("_")[4:]]
        cam = np.array(g[:-1], np.float64)
        y, dy = _camber_line(cam, x.copy())
        if ref_camber is not None:
            ry, rdy = ref_camber(cam, x.copy())
            worst = max(worst, float(np.max(np.abs(y - ry))), float(np.max(np.abs(dy - rdy))))
    if ref_camber is not None:
        print(f"[selftest] max |camber - airfrans.camber_line| over all cases = {worst:.3e}")
        assert worst < 1e-12, "local camber_line disagrees with airfrans"
    # Feature sanity: thickness feature must equal the last name field / 100.
    for nm in names[:200]:
        f = case_features(nm)
        assert abs(f[2] - float(nm.split("_")[-1]) / 100.0) < 1e-12
    print("[selftest] features OK")
    return 0


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def load_pairs(cache_dir, task, train, resolution, limit):
    path = os.path.join(cache_dir, f"airfrans_{task}_{'train' if train else 'test'}"
                                   f"_r{resolution}_n{limit}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"rasterised cache {path} not found. Build it with "
            f"neuroforge.data.airfrans_loader.load_airfrans(task='{task}', "
            f"train={train}, resolution={resolution}, limit={limit}, cache_dir='{cache_dir}')"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="full", choices=["full", "reynolds", "aoa"])
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="results/interpolation")
    ap.add_argument("--reps", nargs="+", default=["nd", "raw"])
    ap.add_argument("--fill", default="nearest", choices=["nearest", "none"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cv-cells", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-test-sweep", action="store_true",
                    help="score EVERY config on the test split (transparency only; "
                         "the headline is the CV-selected config)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[interp] task={a.task} res={a.resolution} fill={a.fill}", flush=True)

    t0 = time.time()
    train_pairs = load_pairs(a.cache_dir, a.task, True, a.resolution, a.n_train)
    test_pairs = load_pairs(a.cache_dir, a.task, False, a.resolution, a.n_test)
    t_load = time.time() - t0
    print(f"[interp] loaded {len(train_pairs)} train / {len(test_pairs)} test "
          f"pairs in {t_load:.1f}s", flush=True)

    names_tr = [c.name for c, _ in train_pairs]
    names_te = [c.name for c, _ in test_pairs]
    overlap = sorted(set(names_tr) & set(names_te))
    H, W = train_pairs[0][1].u.shape
    hw = H * W

    Xtr = np.stack([case_features(n) for n in names_tr])
    Xte = np.stack([case_features(n) for n in names_te])

    out: dict = {
        "task": a.task, "resolution": a.resolution, "fill": a.fill,
        "n_train": len(train_pairs), "n_test": len(test_pairs),
        "feature_names": ["U", "alpha_deg", "t_max", "alpha_eff_deg",
                          "y_c_0.25", "y_c_0.50", "y_c_0.75"],
        "reference_rows": REFERENCE_ROWS,
        "preregistered_selection_scalar":
            "mean over {u,v,p} of MSE_channel / Var_train(channel), 5-fold CV on TRAIN only",
        "leakage": {"train_test_name_overlap": len(overlap),
                    "overlapping_names": overlap[:10]},
        "variants": {},
    }
    assert len(overlap) == 0, f"train/test name overlap: {overlap[:5]}"

    for rep in a.reps:
        print(f"[interp] === representation '{rep}' ===", flush=True)
        t0 = time.time()
        Ytr, Mtr, _ = build_stack(train_pairs, rep, a.fill)
        t_stack = time.time() - t0
        print(f"[interp] train stack {Ytr.shape} built in {t_stack:.1f}s", flush=True)

        t0 = time.time()
        best, cv_table = cv_select(Xtr, Ytr, Mtr, names_tr, rep, hw,
                                   a.n_folds, a.cv_cells, a.seed)
        t_cv = time.time() - t0
        print(f"[interp] CV selected {cfg_name(best)} (S={cv_table[0]['S']:.5f}) "
              f"in {t_cv:.1f}s", flush=True)

        # ---- test-set scoring of the CV-selected config, through evaluate_cases
        def score(cfg: dict) -> tuple[dict, list, list, list, np.ndarray]:
            if cfg["method"] == "freestream":
                pred_nd = freestream_stack(names_te, hw, rep)
            else:
                Wm = build_weights(cfg, Xtr, Xte)
                pred_nd = Wm @ Ytr
            pred_phys = redimensionalise(pred_nd, names_te, rep, hw)
            pfn = make_predict_fn(pred_phys, names_te, (H, W), test_pairs)
            agg = evaluate_cases(pfn, test_pairs)
            rows, pc, rc = per_case_metrics(pfn, test_pairs)
            return agg, rows, pc, rc, pred_phys

        t0 = time.time()
        agg, rows, pc, rc, _ = score(best)
        t_score = time.time() - t0
        assert agg.get("n_cases") == len(test_pairs), (
            f"evaluate_cases scored only {agg.get('n_cases')} of {len(test_pairs)} "
            "cases -- force integration silently dropped cases"
        )

        # ---- leakage / near-duplicate diagnostics
        Ztr, Zte = _standardise(Xtr, Xte, best.get("w_U", 1.0))
        D = _dists(Zte, Ztr)
        nnd = D.min(1)
        err = np.array([r["mse_u"] + r["mse_v"] for r in rows])
        from scipy.stats import spearmanr
        cut = float(np.quantile(nnd, 0.10))
        far = nnd > cut
        excl = {k: float(np.mean([r[k] for r, f in zip(rows, far) if f]))
                for k in ("mse_u", "mse_v", "mse_p", "surface_mse_p")}

        variant = {
            "cv_selected_cfg": cfg_name(best),
            "cv_selected_cfg_raw": best,
            "cv_table_top10": cv_table[:10],
            "test_metrics": agg,
            "verdict": verdict(agg) if a.task == "full" else None,
            "timing_sec": {"build_stack": t_stack, "cv": t_cv, "score": t_score},
            "nn_distance": {
                "min": float(nnd.min()), "p10": float(np.quantile(nnd, 0.1)),
                "median": float(np.median(nnd)), "max": float(nnd.max()),
                "error_vs_nn_distance_spearman": float(spearmanr(nnd, err).statistic),
            },
            "excluding_closest_10pct": {"n": int(far.sum()), **excl},
            "official_forces": official_force_block(rows, names_te) if a.task == "full" else None,
        }

        # ---- floors + a small ladder, same protocol
        ladder = {}
        for cfg in ({"method": "freestream"}, {"method": "mean", "w_U": 1.0},
                    {"method": "nn", "w_U": best.get("w_U", 1.0)},
                    {"method": "knn", "k": 8, "power": 2.0, "w_U": best.get("w_U", 1.0)},
                    {"method": "knn", "k": 32, "power": 2.0, "w_U": best.get("w_U", 1.0)}):
            nm = cfg["method"] if cfg["method"] in ("freestream", "mean") else cfg_name(cfg)
            g, _, _, _, _ = score(cfg)
            ladder[nm] = {k: g[k] for k in ("mse_u", "mse_v", "mse_p", "surface_mse_p",
                                            "rho_cl", "rho_cd", "n_cases")}
            print(f"  [ladder] {nm:24s} u={g['mse_u']:.4g} v={g['mse_v']:.4g} "
                  f"p={g['mse_p']:.5g}", flush=True)
        variant["ladder"] = ladder

        if a.full_test_sweep:
            sweep = {}
            for cfg in config_grid(rep):
                g, _, _, _, _ = score(cfg)
                sweep[cfg_name(cfg)] = {k: g[k] for k in ("mse_u", "mse_v", "mse_p",
                                                          "surface_mse_p")}
            variant["full_test_sweep"] = sweep

        variant["per_case"] = [
            {"name": r["name"], "nn_dist": float(nnd[i]),
             "mse_u": r["mse_u"], "mse_v": r["mse_v"], "mse_p": r["mse_p"],
             "surface_mse_p": r["surface_mse_p"]}
            for i, r in enumerate(rows)
        ]
        out["variants"][rep] = variant
        print(f"[interp] {rep}: mse_u={agg['mse_u']:.4g} mse_v={agg['mse_v']:.4g} "
              f"mse_p={agg['mse_p']:.5g} surf_p={agg['surface_mse_p']:.5g} "
              f"rho_cl={agg['rho_cl']:.4f} rho_cd={agg['rho_cd']:.4f}", flush=True)
        del Ytr, Mtr

    out["wallclock_sec"] = time.time() - t_start
    path = os.path.join(a.out_dir, f"interp_{a.task}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"[interp] wrote {path}  ({out['wallclock_sec']:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
