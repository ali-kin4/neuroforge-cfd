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

__all__ = ["ConformalCalibrator", "calibrate_from_cases"]


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
    return ConformalCalibrator(alpha=alpha).fit(errors, sigmas, masks)
