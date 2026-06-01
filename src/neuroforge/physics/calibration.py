"""Split-conformal calibration of predictive uncertainty (coverage guarantees).

Raw deep-ensemble / MC-dropout standard deviations are *not* calibrated — they
are typically over- or under-dispersed, so a threshold on them has no statistical
meaning, and a trust map built from them cannot claim "X% of high-error cells are
flagged". This module fixes that with **split-conformal prediction**.

Given a held-out calibration set, for every (fluid) cell we form the
nonconformity score ``s = |y_pred − y_true| / σ_pred`` and take the
finite-sample-corrected ``(1−α)`` empirical quantile ``q``. The calibrated band
``q · σ_pred`` then satisfies the distribution-free marginal coverage guarantee

    P( |y_pred − y_true| ≤ q · σ_pred )  ≥  1 − α

on exchangeable data. ``q`` is a single multiplier learned once on the
calibration set; ``calibrate`` applies it, and ``coverage`` empirically verifies
it on fresh data. This follows the conformal-for-operators line (e.g. UQNO, Ma
et al., ICLR 2024, arXiv:2402.01960).
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import DTYPE

__all__ = [
    "ConformalCalibrator",
    "calibrate_from_cases",
    "cases_to_error_sigma",
    "reliability",
    "expected_calibration_error",
]


class ConformalCalibrator:
    """Learns a single conformal multiplier ``q`` for predictive uncertainty.

    Parameters
    ----------
    alpha : float
        Target miscoverage; the guarantee is ``1 − alpha`` coverage (e.g.
        ``alpha=0.1`` → 90%).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.q: float = 1.0
        self.fitted: bool = False

    @staticmethod
    def _scores(error: np.ndarray, sigma: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        err = np.abs(np.asarray(error, np.float64))
        sig = np.abs(np.asarray(sigma, np.float64)) + 1e-8
        s = err / sig
        if mask is not None:
            s = s[np.asarray(mask) > 0.5]
        return s.ravel()

    def fit(
        self,
        errors: list[np.ndarray],
        sigmas: list[np.ndarray],
        masks: list[np.ndarray] | None = None,
    ) -> ConformalCalibrator:
        """Fit ``q`` from per-cell error and uncertainty maps over a calib set."""
        masks = masks or [None] * len(errors)
        scores = np.concatenate(
            [self._scores(e, s, m) for e, s, m in zip(errors, sigmas, masks, strict=False)]
        )
        scores = scores[np.isfinite(scores)]
        n = scores.size
        if n == 0:
            self.q = 1.0
            self.fitted = True
            return self
        # Finite-sample-corrected conformal quantile level.
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q = float(np.quantile(scores, level))
        self.fitted = True
        return self

    def calibrate(self, sigma: np.ndarray) -> np.ndarray:
        """Return the calibrated uncertainty band ``q · sigma``."""
        return (self.q * np.asarray(sigma, DTYPE)).astype(DTYPE)

    def coverage(
        self,
        errors: list[np.ndarray],
        sigmas: list[np.ndarray],
        masks: list[np.ndarray] | None = None,
    ) -> float:
        """Empirical coverage ``P(|error| <= q·sigma)`` on fresh data (target 1−alpha)."""
        masks = masks or [None] * len(errors)
        covered = 0
        total = 0
        for e, s, m in zip(errors, sigmas, masks, strict=False):
            err = np.abs(np.asarray(e, np.float64))
            band = self.q * (np.abs(np.asarray(s, np.float64)) + 1e-8)
            if m is not None:
                sel = np.asarray(m) > 0.5
                err, band = err[sel], band[sel]
            covered += int(np.sum(err <= band))
            total += int(err.size)
        return float(covered / max(total, 1))

    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "q": self.q, "fitted": self.fitted}

    @classmethod
    def from_state_dict(cls, d: dict) -> ConformalCalibrator:
        c = cls(alpha=float(d.get("alpha", 0.1)))
        c.q = float(d.get("q", 1.0))
        c.fitted = bool(d.get("fitted", True))
        return c


def _gather_scores(
    errors: list[np.ndarray],
    sigmas: list[np.ndarray],
    masks: list[np.ndarray] | None,
) -> np.ndarray:
    """Flatten per-cell nonconformity scores ``s = |error| / sigma`` (fluid cells).

    NaN-safe: non-finite scores (e.g. from NaN errors/sigmas) are dropped.
    """
    masks = masks or [None] * len(errors)
    parts = [
        ConformalCalibrator._scores(e, s, m)
        for e, s, m in zip(errors, sigmas, masks, strict=False)
    ]
    scores = np.concatenate(parts) if parts else np.empty(0, np.float64)
    return scores[np.isfinite(scores)]


def _half_normal_quantile(p: np.ndarray) -> np.ndarray:
    """Quantile of the half-normal ``|N(0,1)|`` at probabilities ``p`` in [0,1).

    For a Gaussian error with std σ, ``P(|err| <= z * σ) = p`` when ``z`` is this
    value, i.e. ``z = sqrt(2) * erfinv(p)``. Pure-numpy via ``erfinv`` if SciPy
    is present, else a rational approximation of the inverse error function.
    """
    p = np.clip(np.asarray(p, np.float64), 0.0, 1.0 - 1e-12)
    try:
        from scipy.special import erfinv  # type: ignore
    except Exception:  # pragma: no cover - numpy-only fallback
        def erfinv(y: np.ndarray) -> np.ndarray:
            # Winitzki approximation; good to ~1e-3, adequate for binning.
            y = np.asarray(y, np.float64)
            a = 0.147
            ln = np.log(np.maximum(1.0 - y * y, 1e-300))
            t = 2.0 / (np.pi * a) + ln / 2.0
            return np.sign(y) * np.sqrt(np.sqrt(t * t - ln / a) - t)

    return np.sqrt(2.0) * erfinv(p)


def reliability(
    errors: list[np.ndarray],
    sigmas: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    n_bins: int = 10,
    q: float = 1.0,
    alpha: float | None = None,
) -> dict:
    """Reliability curve and Expected Calibration Error for an uncertainty map.

    Sweeps a set of nominal coverage levels ``p`` and, for each, forms the
    calibrated band ``m(p) * sigma`` and measures *empirical* coverage as the
    fraction of (fluid) cells with ``|error| <= m(p) * sigma``. A perfectly
    calibrated uncertainty traces the diagonal ``empirical == nominal``; an
    over-/under-dispersed σ bows away from it, and the mean absolute gap over the
    bins is the **Expected Calibration Error** (ECE).

    The per-level multiplier is ``m(p) = scale * z(p)`` where
    ``z(p) = sqrt(2)*erfinv(p)`` is the half-normal quantile (the band that a
    well-calibrated *Gaussian* σ needs for coverage ``p``) and ``scale`` rescales
    σ to its true dispersion:

    * ``alpha is None`` (default): ``scale = q``. With ``q=1`` this assesses the
      **raw** σ against the ideal Gaussian band (a mis-scaled σ ⇒ large ECE).
    * ``alpha`` given with a fitted conformal ``q``: ``scale = q / z(1-alpha)``,
      so at the conformal level ``p = 1-alpha`` the band is exactly ``q·σ`` and
      the whole curve reflects the **calibrated** band (small ECE if σ is only
      mis-*scaled*, which conformal corrects).

    Parameters
    ----------
    errors, sigmas : list of np.ndarray
        Per-cell signed/abs error and σ maps (errors are abs'd internally).
    masks : list of np.ndarray, optional
        Per-cell fluid masks (``>0.5`` kept); ``None`` uses all cells.
    n_bins : int
        Number of nominal levels swept in ``(0, 1)`` (bin centres).
    q : float
        Conformal multiplier (default 1).
    alpha : float, optional
        Target miscoverage the ``q`` was fitted for; anchors the calibrated band
        at level ``1-alpha`` (see above). ``None`` ⇒ raw Gaussian assessment.

    Returns
    -------
    dict
        ``nominal`` / ``empirical`` (np.ndarray coverage levels), ``gaps``
        (``|nominal - empirical|``), ``ece`` (mean gap scalar), ``n`` (cell
        count). NaN-safe: non-finite cells are dropped; empty input → NaN ECE.
    """
    n_bins = max(1, int(n_bins))
    nominal = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    # Per-cell ratio |error| / sigma; cell is covered at level p iff ratio<=m(p).
    ratios = _gather_scores(errors, sigmas, masks)
    if ratios.size == 0:
        nan = np.full(n_bins, np.nan)
        return {
            "nominal": nominal,
            "empirical": nan,
            "gaps": nan,
            "ece": float("nan"),
            "n": 0,
        }
    if alpha is None:
        scale = float(q)
    else:
        z_anchor = float(_half_normal_quantile(np.array([1.0 - float(alpha)]))[0])
        scale = float(q) / max(z_anchor, 1e-12)
    bands = scale * _half_normal_quantile(nominal)
    empirical = np.array([float(np.mean(ratios <= b)) for b in bands], np.float64)
    gaps = np.abs(nominal - empirical)
    return {
        "nominal": nominal,
        "empirical": empirical,
        "gaps": gaps,
        "ece": float(np.mean(gaps)),
        "n": int(ratios.size),
    }


def expected_calibration_error(
    errors: list[np.ndarray],
    sigmas: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    n_bins: int = 10,
    q: float = 1.0,
    alpha: float | None = None,
) -> float:
    """Scalar ECE: mean ``|nominal - empirical|`` coverage over the swept bins.

    Thin wrapper over :func:`reliability` returning just the ``ece`` scalar.
    """
    return reliability(errors, sigmas, masks, n_bins=n_bins, q=q, alpha=alpha)["ece"]


def cases_to_error_sigma(
    predict_fn,
    uq,
    pairs,
    channel: int = 2,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Extract per-cell ``(error, sigma, mask)`` maps for a list of cases.

    For each ``(case, ground_truth)`` pair, predicts the field and the per-cell
    uncertainty (from ``uq.predict_with_uncertainty`` on the encoded input) on
    the chosen output ``channel`` (default 2 = pressure), returning the signed
    error, σ (in physical units), and fluid mask maps. Shared by
    :func:`calibrate_from_cases` and the calibration-quality reporting in
    ``evaluation``.
    """
    import torch

    from neuroforge.geometry.encode import encode_case

    errors: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    normalizer = getattr(predict_fn.__self__, "normalizer", None)
    for case, truth in pairs:
        pred = predict_fn(case)
        x = encode_case(case)
        if normalizer is not None:
            x = normalizer.norm_in(x)
        xt = torch.from_numpy(np.ascontiguousarray(x, DTYPE))[None]
        _, std = uq.predict_with_uncertainty(xt)
        sig = std[0, channel].cpu().numpy()
        if normalizer is not None:  # std in normalised space -> physical units
            sig = sig * float(np.asarray(normalizer.std_out)[channel])
        err = pred.as_array()[channel] - truth.as_array()[channel]
        errors.append(err)
        sigmas.append(sig)
        masks.append(truth.mask)
    return errors, sigmas, masks


def calibrate_from_cases(
    predict_fn,
    uq,
    pairs,
    alpha: float = 0.1,
    channel: int = 2,
) -> ConformalCalibrator:
    """Fit a :class:`ConformalCalibrator` from a calibration set.

    For each ``(case, ground_truth)`` pair, predicts the field and the per-cell
    uncertainty (from ``uq.predict_with_uncertainty`` on the encoded input), and
    calibrates on the chosen output ``channel`` (default 2 = pressure).

    Parameters
    ----------
    predict_fn : callable
        ``FlowCase -> FlowField`` (e.g. ``Predictor.predict``).
    uq : object
        UQ estimator with ``predict_with_uncertainty(x) -> (mean, std)``.
    pairs : list of (FlowCase, FlowField)
        Calibration cases with ground-truth fields.
    alpha : float
        Target miscoverage (coverage ``1 − alpha``).
    channel : int
        Output channel to calibrate (0=u, 1=v, 2=p, 3=nut).
    """
    errors, sigmas, masks = cases_to_error_sigma(predict_fn, uq, pairs, channel)
    return ConformalCalibrator(alpha=alpha).fit(errors, sigmas, masks)
