"""Tests for split-conformal uncertainty calibration (coverage guarantee)."""

from __future__ import annotations

import numpy as np

from neuroforge.physics.calibration import ConformalCalibrator


def _make(n, true_std, rng):
    errs = [rng.normal(0, true_std, (32, 32)).astype("float32") for _ in range(n)]
    sigs = [np.ones((32, 32), "float32") for _ in range(n)]
    masks = [np.ones((32, 32), "float32") for _ in range(n)]
    return errs, sigs, masks


def test_conformal_coverage_is_close_to_target():
    """A mis-scaled sigma is corrected so empirical coverage ~= 1 - alpha."""
    rng = np.random.default_rng(0)
    ce, cs, cm = _make(50, true_std=2.5, rng=rng)
    te, ts, tm = _make(50, true_std=2.5, rng=rng)
    cal = ConformalCalibrator(alpha=0.1).fit(ce, cs, cm)
    cov = cal.coverage(te, ts, tm)
    assert 0.85 < cov < 0.95, f"coverage {cov} far from target 0.90"
    # q recovers the ~90% quantile of |N(0,2.5)| with sigma=1 (~1.645*2.5).
    assert 3.5 < cal.q < 4.7


def test_conformal_state_roundtrip():
    rng = np.random.default_rng(1)
    e, s, m = _make(10, 1.0, rng)
    cal = ConformalCalibrator(alpha=0.2).fit(e, s, m)
    cal2 = ConformalCalibrator.from_state_dict(cal.state_dict())
    assert abs(cal2.q - cal.q) < 1e-9 and cal2.alpha == cal.alpha


def test_calibrate_applies_multiplier():
    cal = ConformalCalibrator()
    cal.q = 2.0
    sig = np.ones((4, 4), "float32")
    assert np.allclose(cal.calibrate(sig), 2.0)
