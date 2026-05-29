"""Trust maps: turn residual magnitude + uncertainty into a per-cell trust score.

The trust map is the engine's *self-assessment*. It fuses two error signals —
the physics residual magnitude (how badly the PDE is violated) and the model's
predictive uncertainty — into a single trust value in ``[0, 1]`` (1 = reliable)
and a discrete traffic-light class (2 green / 1 yellow / 0 red) used by the
correction loop to decide where to intervene.

Each input is mapped to roughly ``[0, 1]``. By default the normalisation scale
is the 95th percentile over the fluid region (robust to outliers); callers that
know a *physical* reference scale (e.g. an advective residual scale ``U^2/L``)
can pass it explicitly via ``residual_scale`` / ``uncertainty_scale`` so the
thresholds in :class:`~neuroforge.core.config.PhysicsConfig` have an absolute
meaning. A small absolute floor guards the degenerate case where the residual
is uniformly tiny: such a field reads as ~0 error (green) rather than saturating
to 1 — a self-relative percentile alone cannot tell "uniformly small" from
"uniformly large", so the floor supplies that missing absolute reference.
"""

from __future__ import annotations

import numpy as np

from neuroforge.core.config import PhysicsConfig
from neuroforge.core.types import DTYPE

_EPS = 1e-12
# Cells whose magnitude is below this fraction of the field's own maximum are
# treated as negligible (normalised error ~0). Keeps a near-converged field
# green instead of letting percentile self-scaling saturate it to red.
_REL_FLOOR = 1e-3


def _robust_normalise(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
    percentile: float = 95.0,
    scale: float | None = None,
) -> np.ndarray:
    """Scale ``arr`` to ~``[0, 1]`` and clip.

    Parameters
    ----------
    arr : numpy.ndarray
        Non-negative magnitude map ``(ny, nx)``.
    mask : numpy.ndarray, optional
        ``1`` in the fluid, ``0`` in the solid. If given, the percentile scale
        is computed from fluid cells only.
    percentile : float, default 95
        Percentile used as the normalisation scale when ``scale`` is ``None``.
    scale : float, optional
        Explicit normalisation denominator (a physical reference). When given it
        overrides the percentile estimate.

    Returns
    -------
    numpy.ndarray
        ``float32`` map clipped to ``[0, 1]``.

    Notes
    -----
    With an explicit ``scale`` the mapping is absolute: ``clip(|arr| / scale)``
    (floored at ``_REL_FLOOR * global_max``). Without a scale it is robust and
    *relative*: a baseline (median) is subtracted and the result divided by the
    spread ``(p95 - median)``, so a spatially uniform field maps to ~0 (only
    hotspots above the typical level are flagged) rather than saturating to ~1.
    """
    a = np.abs(np.asarray(arr, dtype=np.float64))
    if a.size == 0:
        return np.zeros_like(a, dtype=DTYPE)

    global_max = float(np.max(a))
    if not np.isfinite(global_max) or global_max < _EPS:
        # Field is identically ~0 -> zero error everywhere.
        return np.zeros_like(a, dtype=DTYPE)

    if scale is not None and np.isfinite(float(scale)) and float(scale) >= _EPS:
        # Absolute physical reference: a small residual maps to a small error
        # regardless of the field's own spread. This is the path used by
        # PhysicsChecker.diagnose, so the trust thresholds are meaningful.
        denom = max(float(scale), _REL_FLOOR * global_max)
        return np.clip(a / denom, 0.0, 1.0).astype(DTYPE)

    # No absolute scale: fall back to robust *relative* scaling that highlights
    # spatial hotspots. We subtract a robust baseline (the median over the
    # fluid) and divide by the spread (p95 - median), so a spatially uniform
    # field (no hotspots, nothing to distrust) maps to ~0 (high trust) instead
    # of saturating to ~1 (all red). Only deviations above the typical level
    # are flagged.
    if mask is not None:
        m = np.asarray(mask) > 0.5
        sample = a[m] if np.any(m) else a.ravel()
    else:
        sample = a.ravel()
    if sample.size == 0:
        sample = a.ravel()
    baseline = float(np.median(sample))
    hi = float(np.percentile(sample, percentile))
    spread = hi - baseline
    if not np.isfinite(spread) or spread < _REL_FLOOR * global_max:
        # Effectively uniform -> no spatial structure to distrust.
        return np.zeros_like(a, dtype=DTYPE)
    return np.clip((a - baseline) / spread, 0.0, 1.0).astype(DTYPE)


def trust_map(
    residual_mag: np.ndarray,
    uncertainty: np.ndarray,
    cfg: PhysicsConfig,
    mask: np.ndarray | None = None,
    residual_scale: float | None = None,
    uncertainty_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine residual magnitude and uncertainty into a trust map.

    Each input is normalised to ~``[0, 1]`` (by an explicit physical scale if
    provided, else its 95th percentile in the fluid, with an absolute floor),
    then combined as ``combine = w_r * r_norm + w_u * u_norm`` (weights
    renormalised to sum to 1). Trust is ``1 - clip(combine, 0, 1)``. The class
    is ``2`` (green) where ``combine < cfg.trust_green_below``, ``0`` (red) where
    ``combine > cfg.trust_red_above``, else ``1`` (yellow).

    Parameters
    ----------
    residual_mag : numpy.ndarray
        Per-cell physics residual magnitude ``(ny, nx)`` (non-negative).
    uncertainty : numpy.ndarray
        Per-cell predictive uncertainty ``(ny, nx)`` (non-negative). Pass zeros
        if no uncertainty estimate is available.
    cfg : PhysicsConfig
        Supplies ``trust_residual_weight``, ``trust_uncertainty_weight``,
        ``trust_green_below`` and ``trust_red_above``.
    mask : numpy.ndarray, optional
        Fluid mask (``1`` fluid, ``0`` solid) scoping the percentile estimate.
    residual_scale, uncertainty_scale : float, optional
        Explicit normalisation denominators (physical reference scales). When
        given they override the percentile estimate for the respective input.

    Returns
    -------
    tuple of numpy.ndarray
        ``(trust, trust_class)`` both ``(ny, nx)``. ``trust`` is ``float32`` in
        ``[0, 1]``; ``trust_class`` is ``float32`` holding integer codes
        ``{0, 1, 2}``.
    """
    residual_mag = np.asarray(residual_mag, dtype=DTYPE)
    uncertainty = np.asarray(uncertainty, dtype=DTYPE)
    if uncertainty.shape != residual_mag.shape:
        # Be forgiving: broadcast a scalar / mismatched uncertainty to zeros.
        uncertainty = np.zeros_like(residual_mag)

    r_norm = _robust_normalise(residual_mag, mask=mask, scale=residual_scale)
    u_norm = _robust_normalise(uncertainty, mask=mask, scale=uncertainty_scale)

    w_r = float(cfg.trust_residual_weight)
    w_u = float(cfg.trust_uncertainty_weight)
    w_sum = w_r + w_u
    if w_sum < _EPS:
        w_r, w_u, w_sum = 0.5, 0.5, 1.0
    w_r /= w_sum
    w_u /= w_sum

    combine = np.clip(w_r * r_norm + w_u * u_norm, 0.0, 1.0).astype(DTYPE)
    trust = (1.0 - combine).astype(DTYPE)

    # Traffic-light classes. Start everything yellow, then paint green / red.
    trust_class = np.ones_like(combine, dtype=DTYPE)
    trust_class[combine < float(cfg.trust_green_below)] = 2.0
    trust_class[combine > float(cfg.trust_red_above)] = 0.0

    return trust, trust_class


__all__ = ["trust_map"]
