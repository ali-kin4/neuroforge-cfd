"""Tests for split-conformal uncertainty calibration (coverage guarantee)."""

from __future__ import annotations

import types

import numpy as np
import torch

from neuroforge.core.types import DTYPE, Domain, FlowCase, FlowField
from neuroforge.physics.calibration import (
    ConformalCalibrator,
    cases_to_error_sigma,
    corrected_predict_fn,
)


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


# --------------------------------------------------------------------------- #
# cases_to_error_sigma: raw-default vs corrected-field path (the deployed-output
# fix). These pin (a) the default normalizer resolution from predict_fn.__self__
# is byte-identical, (b) the new normalizer= override + corrected_predict_fn
# route the DEPLOYED corrected field through the same scoring code, and (c) sigma
# is field-independent (same for raw & corrected), so only the error map moves.
# --------------------------------------------------------------------------- #


def _tiny_case():
    return FlowCase.from_airfoil(
        "naca0012", aoa=2, reynolds=1e6, u_inf=10.0, resolution=24
    )


class _IdentityNorm:
    """Identity normaliser: norm_in is a no-op and std_out == 1 (sigma unscaled)."""

    def __init__(self, n_out=4):
        self.std_out = np.ones(n_out, np.float64)

    def norm_in(self, x):
        return np.asarray(x, DTYPE)


class _ConstUQ:
    """UQ stub: per-cell std is a fixed constant on every channel/cell."""

    def __init__(self, value, shape):
        self.value = float(value)
        self.shape = shape

    def predict_with_uncertainty(self, x):
        n_out = 4
        std = torch.full((1, n_out, *self.shape), self.value, dtype=torch.float32)
        return None, std


def _const_field(case, value, n_out=4):
    h, w = case.domain.ny, case.domain.nx
    arr = np.full((n_out, h, w), float(value), DTYPE)
    return FlowField.from_array(arr, case.domain, mask=np.ones((h, w), DTYPE))


def test_cases_to_error_sigma_default_resolves_normalizer_from_self():
    """Bound-method predict_fn: normalizer comes from predict_fn.__self__ (legacy)."""
    case = _tiny_case()
    h, w = case.domain.ny, case.domain.nx
    truth = _const_field(case, 0.0)

    class _RawPredictor:
        normalizer = _IdentityNorm()

        def predict(self, c):  # constant field => error == predicted value
            return _const_field(c, 3.0)

    pred = _RawPredictor()
    uq = _ConstUQ(2.0, (h, w))
    errs, sigs, masks = cases_to_error_sigma(pred.predict, uq, [(case, truth)], channel=2)
    # error = 3.0 - 0.0 = 3.0 everywhere; sigma = 2.0 * std_out(==1) = 2.0.
    assert np.allclose(errs[0], 3.0)
    assert np.allclose(sigs[0], 2.0)
    assert masks[0].shape == (h, w)


def test_cases_to_error_sigma_corrected_field_via_helper():
    """corrected_predict_fn + explicit normalizer scores the DEPLOYED solve().field.

    The corrected field differs from raw, sigma is identical (input-conditional),
    so the nonconformity scores change only through the error map -- exactly the
    behaviour the deployed-output certificate needs.
    """
    case = _tiny_case()
    h, w = case.domain.ny, case.domain.nx
    truth = _const_field(case, 0.0)
    norm = _IdentityNorm()

    class _RawPredictor:
        normalizer = norm

        def predict(self, c):
            return _const_field(c, 3.0)

    class _Result:
        def __init__(self, field):
            self.field = field

    class _Engine:
        predictor = _RawPredictor()

        def solve(self, c):  # corrector pulls the field closer to truth: 3.0 -> 1.0
            return _Result(_const_field(c, 1.0))

    engine = _Engine()
    uq = _ConstUQ(2.0, (h, w))

    corr_fn = corrected_predict_fn(engine)
    assert isinstance(corr_fn, types.FunctionType)
    # corrected field == engine.solve(case).field
    assert np.allclose(corr_fn(case).as_array(), 1.0)

    # Must pass normalizer= explicitly (closure has no __self__).
    ce, cs, cm = cases_to_error_sigma(
        corr_fn, uq, [(case, truth)], channel=2, normalizer=norm
    )
    re, rs, _ = cases_to_error_sigma(
        engine.predictor.predict, uq, [(case, truth)], channel=2
    )
    # sigma identical (field-independent); error map differs (corrected closer).
    assert np.allclose(cs[0], rs[0])
    assert np.allclose(ce[0], 1.0) and np.allclose(re[0], 3.0)
    assert np.mean(np.abs(ce[0])) < np.mean(np.abs(re[0]))
