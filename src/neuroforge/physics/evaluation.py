"""Community-standard evaluation for airfoil-flow surrogates (AirfRANS protocol).

The relative-L2 used elsewhere is ill-conditioned on near-zero-mean channels
(cross-stream velocity ``v`` has tiny L2 norm, so ``||pred-ref|| / ||ref||``
explodes). The AirfRANS benchmark (Bonnet et al., NeurIPS 2022) instead reports:

* **per-channel MSE** over the fluid volume (well-conditioned regardless of mean),
* **surface-restricted MSE** of pressure on the body (the design-relevant field),
* mean/std **relative error on the force coefficients** ``C_L``/``C_D``, and
* **Spearman rank correlation** ``rho_L``/``rho_D`` of predicted-vs-true
  coefficients across the test set — the single most important metric for
  early-design exploration, where only the *ranking* of candidate geometries
  must be right.

This module also provides the **residual-vs-error correlation** diagnostic — the
make-or-break experiment for the "physics-residual as a trust signal" claim:
if a low PDE residual does not track low field error, the trust map is not a
correctness signal. Use :func:`evaluate_cases` to compute all of the above over
a list of ``(FlowCase, ground_truth FlowField)`` pairs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from neuroforge.core.types import DTYPE, FlowCase, FlowField

from .calibration import (
    ConformalCalibrator,
    cases_to_error_sigma,
    reliability,
)
from .metrics import _bilinear_sample, _surface_points_normals, force_coefficients

__all__ = [
    "per_channel_mse",
    "surface_pressure_mse",
    "coefficient_metrics",
    "residual_error_correlation",
    "evaluate_cases",
    "evaluate_calibration",
]


def _fluid_mask(ref: FlowField, pred: FlowField) -> np.ndarray:
    mask = ref.mask if ref.mask is not None else pred.mask
    if mask is None:
        mask = np.ones(ref.shape, dtype=DTYPE)
    m = np.asarray(mask) > 0.5
    return m if m.any() else np.ones(ref.shape, dtype=bool)


def per_channel_mse(pred: FlowField, ref: FlowField) -> dict[str, float]:
    """Mean-squared error per field (u, v, p, nut) over the fluid volume.

    Well-conditioned for every channel (unlike relative-L2). ``speed`` MSE on the
    velocity magnitude is also reported.
    """
    m = _fluid_mask(ref, pred)

    def mse(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, np.float64)[m]
        b = np.asarray(b, np.float64)[m]
        return float(np.mean((a - b) ** 2)) if a.size else 0.0

    out = {
        "mse_u": mse(pred.u, ref.u),
        "mse_v": mse(pred.v, ref.v),
        "mse_p": mse(pred.p, ref.p),
        "mse_speed": mse(pred.speed(), ref.speed()),
    }
    if pred.nut is not None and ref.nut is not None:
        out["mse_nut"] = mse(pred.nut, ref.nut)
    return out


def surface_pressure_mse(pred: FlowField, ref: FlowField, case: FlowCase) -> float:
    """MSE of pressure sampled on the body surface (the design-relevant field)."""
    pts, _ = _surface_points_normals(case)
    if pts.shape[0] < 2:
        return 0.0
    p_pred = _bilinear_sample(pred.p, pred.domain, pts)
    p_ref = _bilinear_sample(ref.p, ref.domain, pts)
    return float(np.mean((p_pred - p_ref) ** 2))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, via scipy if available, else a numpy fallback."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    try:
        from scipy.stats import spearmanr  # type: ignore

        return float(spearmanr(a, b).statistic)
    except Exception:  # pragma: no cover - numpy fallback
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra**2).sum() * (rb**2).sum())
        return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def coefficient_metrics(
    pred_coeffs: list[dict[str, float]], ref_coeffs: list[dict[str, float]]
) -> dict[str, float]:
    """Rank/relative-error metrics on the force coefficients across a test set.

    Parameters
    ----------
    pred_coeffs, ref_coeffs : list of dict
        Per-case ``{'cl','cd',...}`` for prediction and reference, same order.

    Returns
    -------
    dict
        ``rho_cl``/``rho_cd`` (Spearman rank correlation across the set),
        ``cl_rel_err_mean``/``_std`` and ``cd_rel_err_mean``/``_std`` (relative
        error magnitudes), and ``cl_mae``/``cd_mae``.
    """
    if not pred_coeffs:
        return {}
    cl_p = np.array([c.get("cl", np.nan) for c in pred_coeffs], np.float64)
    cd_p = np.array([c.get("cd", np.nan) for c in pred_coeffs], np.float64)
    cl_r = np.array([c.get("cl", np.nan) for c in ref_coeffs], np.float64)
    cd_r = np.array([c.get("cd", np.nan) for c in ref_coeffs], np.float64)

    def rel(p, r):
        d = np.abs(p - r) / (np.abs(r) + 1e-12)
        d = d[np.isfinite(d)]
        return (float(d.mean()), float(d.std())) if d.size else (float("nan"), float("nan"))

    cl_rel = rel(cl_p, cl_r)
    cd_rel = rel(cd_p, cd_r)
    return {
        "rho_cl": _spearman(cl_p, cl_r),
        "rho_cd": _spearman(cd_p, cd_r),
        "cl_rel_err_mean": cl_rel[0], "cl_rel_err_std": cl_rel[1],
        "cd_rel_err_mean": cd_rel[0], "cd_rel_err_std": cd_rel[1],
        "cl_mae": float(np.nanmean(np.abs(cl_p - cl_r))),
        "cd_mae": float(np.nanmean(np.abs(cd_p - cd_r))),
        "n_cases": int(len(pred_coeffs)),
    }


def residual_error_correlation(
    residual_norms: list[float], field_errors: list[float]
) -> dict[str, float]:
    """Correlation between per-case PDE residual and true field error.

    This is the central validity check for using the residual as a trust signal:
    only if residual and error are positively correlated does "low residual"
    mean "trustworthy". Returns Spearman and Pearson coefficients.
    """
    r = np.asarray(residual_norms, np.float64)
    e = np.asarray(field_errors, np.float64)
    ok = np.isfinite(r) & np.isfinite(e)
    r, e = r[ok], e[ok]
    out = {"spearman": _spearman(r, e), "n": int(r.size)}
    if r.size >= 2 and r.std() > 0 and e.std() > 0:
        out["pearson"] = float(np.corrcoef(r, e)[0, 1])
    else:
        out["pearson"] = float("nan")
    return out


def evaluate_cases(
    predict_fn: Callable[[FlowCase], FlowField],
    pairs: list[tuple[FlowCase, FlowField]],
    checker=None,
) -> dict[str, float]:
    """Full AirfRANS-style evaluation over ``(case, ground_truth)`` pairs.

    Parameters
    ----------
    predict_fn : callable
        Maps a :class:`FlowCase` to a predicted :class:`FlowField` (e.g.
        ``Predictor.predict``).
    pairs : list of (FlowCase, FlowField)
        Cases paired with their ground-truth fields.
    checker : PhysicsChecker, optional
        If given, the per-case residual norm is recorded and correlated against
        the per-case field error (the residual-as-trust-signal diagnostic).

    Returns
    -------
    dict
        Aggregated metrics: mean per-channel MSE (``mse_u/v/p/speed``), mean
        ``surface_mse_p``, the coefficient metrics (``rho_cl``/``rho_cd``,
        relative errors), and — if ``checker`` is given — ``residual_error_*``
        correlation.
    """
    if not pairs:
        return {}
    vol: dict[str, list[float]] = {}
    surf: list[float] = []
    pred_coeffs: list[dict] = []
    ref_coeffs: list[dict] = []
    res_norms: list[float] = []
    errs: list[float] = []

    for case, ref in pairs:
        pred = predict_fn(case)
        for k, v in per_channel_mse(pred, ref).items():
            vol.setdefault(k, []).append(v)
        surf.append(surface_pressure_mse(pred, ref, case))
        try:
            pred_coeffs.append(force_coefficients(pred, case))
            ref_coeffs.append(force_coefficients(ref, case))
        except Exception:  # pragma: no cover - degenerate geometry
            pass
        if checker is not None:
            diag = checker.diagnose(pred, case)
            res_norms.append(diag.residual_norm())
            m = _fluid_mask(ref, pred)
            num = float(np.sqrt(np.sum(((pred.speed() - ref.speed())[m]) ** 2)))
            den = float(np.sqrt(np.sum((ref.speed()[m]) ** 2))) + 1e-12
            errs.append(num / den)

    out: dict[str, float] = {k: float(np.mean(v)) for k, v in vol.items()}
    out["surface_mse_p"] = float(np.mean(surf)) if surf else float("nan")
    if pred_coeffs:
        out.update(coefficient_metrics(pred_coeffs, ref_coeffs))
    if checker is not None and res_norms:
        rec = residual_error_correlation(res_norms, errs)
        out["residual_error_spearman"] = rec["spearman"]
        out["residual_error_pearson"] = rec["pearson"]
    return out


def evaluate_calibration(
    predict_fn: Callable[[FlowCase], FlowField],
    uq,
    pairs: list[tuple[FlowCase, FlowField]],
    alpha: float = 0.1,
    channel: int = 2,
    n_bins: int = 10,
) -> dict[str, float]:
    """Split-conformal calibration quality of the UQ trust map on held-out cases.

    Splits ``pairs`` in half: fits a :class:`ConformalCalibrator` on the first
    half (calibration set) and reports the **empirical coverage** and **Expected
    Calibration Error** of the calibrated band ``q·σ`` on the second half (test
    set) — the reviewer-facing reliability numbers, not just the multiplier.

    Parameters
    ----------
    predict_fn : callable
        ``FlowCase -> FlowField`` (e.g. ``Predictor.predict``).
    uq : object
        UQ estimator with ``predict_with_uncertainty(x) -> (mean, std)``.
    pairs : list of (FlowCase, FlowField)
        Cases with ground-truth fields; the first half calibrates, the rest test.
    alpha : float
        Target miscoverage; the conformal guarantee is ``1 − alpha`` coverage.
    channel : int
        Output channel to assess (0=u, 1=v, 2=p, 3=nut; default 2 = pressure).
    n_bins : int
        Number of nominal levels swept for the ECE.

    Returns
    -------
    dict
        ``{q, coverage, ece, target_coverage, n_cal, n_test}``. ``coverage`` and
        ``ece`` are computed on the held-out test split with the fitted ``q``;
        ``target_coverage`` is ``1 − alpha``. Empty/degenerate splits → NaNs.
    """
    target = 1.0 - float(alpha)
    n = len(pairs)
    if n < 2:
        return {
            "q": float("nan"),
            "coverage": float("nan"),
            "ece": float("nan"),
            "target_coverage": target,
            "n_cal": 0,
            "n_test": 0,
        }
    split = n // 2
    cal_pairs, test_pairs = pairs[:split], pairs[split:]

    ce, cs, cm = cases_to_error_sigma(predict_fn, uq, cal_pairs, channel)
    te, ts, tm = cases_to_error_sigma(predict_fn, uq, test_pairs, channel)

    cal = ConformalCalibrator(alpha=alpha).fit(ce, cs, cm)
    cov = cal.coverage(te, ts, tm)
    rel = reliability(te, ts, tm, n_bins=n_bins, q=cal.q, alpha=alpha)
    return {
        "q": float(cal.q),
        "coverage": float(cov),
        "ece": float(rel["ece"]),
        "target_coverage": target,
        "n_cal": len(cal_pairs),
        "n_test": len(test_pairs),
    }
