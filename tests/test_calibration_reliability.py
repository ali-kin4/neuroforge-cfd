"""Tests for calibration-quality reporting: reliability curve and ECE.

These cover :func:`reliability` / :func:`expected_calibration_error` (the
reviewer-facing reliability diagram + Expected Calibration Error) and the
end-to-end :func:`evaluate_calibration` split-conformal report.
"""

from __future__ import annotations

import numpy as np

from neuroforge.physics.calibration import (
    ConformalCalibrator,
    expected_calibration_error,
    reliability,
)
from neuroforge.physics.evaluation import evaluate_calibration


def _make(n, true_std, sigma_says, rng, shape=(32, 32)):
    """n maps of N(0, true_std) error with a (possibly mis-scaled) constant σ."""
    errs = [rng.normal(0, true_std, shape).astype("float32") for _ in range(n)]
    sigs = [np.full(shape, sigma_says, "float32") for _ in range(n)]
    masks = [np.ones(shape, "float32") for _ in range(n)]
    return errs, sigs, masks


# --------------------------------------------------------------------------- #
# reliability / ECE unit behaviour
# --------------------------------------------------------------------------- #
def test_reliability_curve_shape_and_keys():
    rng = np.random.default_rng(0)
    e, s, m = _make(20, true_std=1.0, sigma_says=1.0, rng=rng)
    rel = reliability(e, s, m, n_bins=10)
    for k in ("nominal", "empirical", "gaps", "ece", "n"):
        assert k in rel
    assert rel["nominal"].shape == (10,)
    assert rel["empirical"].shape == (10,)
    # empirical coverage is monotone non-decreasing in the nominal level
    assert np.all(np.diff(rel["empirical"]) >= -1e-9)
    assert 0.0 <= rel["ece"] <= 1.0
    assert rel["n"] == 20 * 32 * 32


def test_well_calibrated_gaussian_has_small_ece():
    """A correctly-scaled Gaussian σ traces the diagonal → tiny ECE."""
    rng = np.random.default_rng(1)
    e, s, m = _make(60, true_std=2.5, sigma_says=2.5, rng=rng)
    ece = expected_calibration_error(e, s, m)
    assert ece < 0.05, f"well-calibrated ECE unexpectedly large: {ece}"


def test_overconfident_sigma_has_large_ece():
    """An overconfident σ (says 0.3, true 2.5) is badly mis-calibrated raw."""
    rng = np.random.default_rng(2)
    e, s, m = _make(60, true_std=2.5, sigma_says=0.3, rng=rng)
    ece = expected_calibration_error(e, s, m)
    assert ece > 0.2, f"overconfident ECE unexpectedly small: {ece}"


def test_conformal_calibration_reduces_ece_on_holdout():
    """Mis-scaled σ (says 1, true 2.5): conformal calibration → coverage ~0.9,
    small ECE on held-out data, and a larger *pre*-calibration ECE."""
    rng = np.random.default_rng(3)
    ce, cs, cm = _make(80, true_std=2.5, sigma_says=1.0, rng=rng)
    te, ts, tm = _make(80, true_std=2.5, sigma_says=1.0, rng=rng)

    cal = ConformalCalibrator(alpha=0.1).fit(ce, cs, cm)
    cov = cal.coverage(te, ts, tm)
    ece_raw = expected_calibration_error(te, ts, tm, q=1.0)          # raw σ
    ece_cal = expected_calibration_error(te, ts, tm, q=cal.q, alpha=0.1)

    assert 0.85 < cov < 0.95, f"coverage {cov} far from 0.90"
    assert ece_cal < 0.1, f"calibrated ECE {ece_cal} not small"
    assert ece_raw > ece_cal, "calibration did not improve ECE"


def test_reliability_nan_safe():
    """Non-finite cells are dropped; an all-NaN / empty input yields NaN ECE."""
    rng = np.random.default_rng(4)
    e, s, m = _make(5, true_std=1.0, sigma_says=1.0, rng=rng)
    e[0][0, 0] = np.nan          # poison one cell
    s[1][1, 1] = np.inf
    rel = reliability(e, s, m)
    assert np.isfinite(rel["ece"])           # survives the poisoned cells
    assert rel["n"] < 5 * 32 * 32            # poisoned cells were dropped

    empty = reliability([], [], [])
    assert np.isnan(empty["ece"]) and empty["n"] == 0

    allnan = reliability(
        [np.full((4, 4), np.nan, "float32")],
        [np.ones((4, 4), "float32")],
    )
    assert np.isnan(allnan["ece"]) and allnan["n"] == 0


def test_reliability_respects_mask():
    """Only masked-in (fluid) cells contribute to the curve."""
    rng = np.random.default_rng(5)
    err = rng.normal(0, 2.5, (16, 16)).astype("float32")
    sig = np.ones((16, 16), "float32")
    mask = np.zeros((16, 16), "float32")
    mask[:8] = 1.0  # only half the cells are fluid
    rel = reliability([err], [sig], [mask])
    assert rel["n"] == 8 * 16


# --------------------------------------------------------------------------- #
# end-to-end evaluate_calibration
# --------------------------------------------------------------------------- #
def _identity_normalizer(n_in, n_out):
    from neuroforge.data.datamodule import Normalizer

    return Normalizer(
        mean_in=np.zeros(n_in, "float32"),
        std_in=np.ones(n_in, "float32"),
        mean_out=np.zeros(n_out, "float32"),
        std_out=np.ones(n_out, "float32"),
    )


def test_evaluate_calibration_report():
    """evaluate_calibration runs end-to-end and returns a sane report dict."""
    from neuroforge.core.types import N_IN, N_OUT
    from neuroforge.data.synthetic import SyntheticRANS
    from neuroforge.models.base import build_model
    from neuroforge.models.ensemble import DeepEnsemble
    from neuroforge.solver.engine import Predictor

    gen = SyntheticRANS(resolution=24, seed=0)
    pairs = [(gen.sample_case(i), gen.solve(gen.sample_case(i))) for i in range(6)]

    members = [build_model("fno", width=8, n_layers=2, modes=6) for _ in range(2)]
    uq = DeepEnsemble(members)
    predictor = Predictor(members[0], _identity_normalizer(N_IN, N_OUT), device="cpu")

    rep = evaluate_calibration(predictor.predict, uq, pairs, alpha=0.1, channel=2)
    for k in ("q", "coverage", "ece", "target_coverage", "n_cal", "n_test"):
        assert k in rep
    assert rep["target_coverage"] == 0.9
    assert rep["n_cal"] == 3 and rep["n_test"] == 3
    assert np.isfinite(rep["q"]) and rep["q"] > 0
    assert 0.0 <= rep["coverage"] <= 1.0
    assert 0.0 <= rep["ece"] <= 1.0


def test_evaluate_calibration_too_few_pairs():
    rep = evaluate_calibration(lambda c: c, object(), [], alpha=0.1)
    assert np.isnan(rep["q"]) and rep["target_coverage"] == 0.9
