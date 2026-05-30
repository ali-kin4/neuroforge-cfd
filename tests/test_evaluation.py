"""Tests for the AirfRANS-protocol evaluation metrics."""

from __future__ import annotations

import numpy as np

from neuroforge.core.types import FlowField
from neuroforge.data.synthetic import SyntheticRANS
from neuroforge.physics.evaluation import (
    coefficient_metrics,
    evaluate_cases,
    per_channel_mse,
    residual_error_correlation,
)
from neuroforge.physics.residuals import PhysicsChecker


def test_per_channel_mse_zero_for_identical():
    gen = SyntheticRANS(32, seed=0)
    case = gen.sample_case(0)
    f = gen.solve(case)
    mse = per_channel_mse(f, f)
    assert all(v < 1e-10 for v in mse.values())


def test_coefficient_metrics_rank():
    # Perfectly rank-preserving predictions -> Spearman 1.
    ref = [{"cl": c, "cd": 0.01 * c} for c in (0.1, 0.3, 0.5, 0.7)]
    pred = [{"cl": c["cl"] * 1.1 + 0.01, "cd": c["cd"] * 0.9} for c in ref]
    m = coefficient_metrics(pred, ref)
    assert abs(m["rho_cl"] - 1.0) < 1e-9
    assert abs(m["rho_cd"] - 1.0) < 1e-9
    assert m["cd_rel_err_mean"] >= 0.0
    assert m["n_cases"] == 4


def test_residual_error_correlation_monotone():
    r = [1.0, 2.0, 3.0, 4.0]
    e = [0.1, 0.2, 0.35, 0.5]  # increasing with r
    c = residual_error_correlation(r, e)
    assert c["spearman"] > 0.9
    assert c["n"] == 4


def test_evaluate_cases_keys_and_identity():
    gen = SyntheticRANS(32, seed=2)
    pairs = [(gen.sample_case(i), gen.solve(gen.sample_case(i))) for i in range(3)]

    rng = np.random.default_rng(0)

    def noisy(case):
        f = gen.solve(case)
        s = 0.05 * float(np.abs(f.u).max() + 1e-6)
        return FlowField(
            domain=f.domain,
            u=(f.u + rng.normal(0, s, f.u.shape)).astype("float32"),
            v=(f.v + rng.normal(0, s, f.v.shape)).astype("float32"),
            p=f.p, nut=f.nut, mask=f.mask, sdf=f.sdf,
        )

    out = evaluate_cases(noisy, pairs, checker=PhysicsChecker())
    for k in ("mse_u", "mse_v", "mse_p", "surface_mse_p", "rho_cl", "rho_cd",
              "residual_error_spearman"):
        assert k in out
    assert out["mse_u"] > 0.0

    # identity predictor -> ~0 volume MSE
    ident = evaluate_cases(lambda c: gen.solve(c), pairs)
    assert ident["mse_u"] < 1e-8
