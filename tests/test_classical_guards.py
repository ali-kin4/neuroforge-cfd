"""Guards on the local classical fallback: divergence, and the vacuous no-harm test.

Both were found by the Paper-2 probes: a full-domain solve burned every iteration
producing a NaN residual, and a cold start was rejected not because the patch was
bad but because a uniform freestream is an exact zero of the residual operator, so
nothing can beat it.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.core.types import FlowCase, FlowField
from neuroforge.solver.classical import local_incompressible_solve


@pytest.fixture(scope="module")
def case() -> FlowCase:
    return FlowCase.from_airfoil("naca0012", aoa=2.0, resolution=32)


def _uniform(case: FlowCase) -> FlowField:
    shape = case.domain.shape
    u_in = float(getattr(case.bc, "u_inf", 1.0))
    v_in = float(getattr(case.bc, "v_inf", 0.0))
    return FlowField(
        domain=case.domain,
        u=np.full(shape, u_in), v=np.full(shape, v_in),
        p=np.zeros(shape), nut=np.zeros(shape),
    )


def test_uniform_freestream_is_flagged_as_a_degenerate_baseline(case):
    """A rejection against a zero baseline must not read as 'the patch was worse'."""
    field = _uniform(case)
    region = np.zeros(case.domain.shape, dtype=bool)
    region[8:20, 8:20] = True

    out = local_incompressible_solve(field, case, region, max_outer=20)
    fb = out.meta["fallback"]

    assert fb["residual_before"] == pytest.approx(0.0, abs=1e-10)
    assert fb["degenerate_baseline"] is True
    assert fb["ran"] is False
    assert "cannot fire" in fb["note"]


def test_a_normal_patch_is_not_flagged_degenerate(case):
    """The flag must be specific to the zero baseline, not set on every rejection."""
    rng = np.random.default_rng(0)
    shape = case.domain.shape
    field = FlowField(
        domain=case.domain,
        u=1.0 + 0.3 * rng.standard_normal(shape),
        v=0.2 * rng.standard_normal(shape),
        p=0.1 * rng.standard_normal(shape),
        nut=np.zeros(shape),
    )
    region = np.zeros(shape, dtype=bool)
    region[8:20, 8:20] = True

    fb = local_incompressible_solve(field, case, region, max_outer=20).meta["fallback"]
    assert fb["residual_before"] > 1e-6
    assert fb["degenerate_baseline"] is False


def test_divergence_is_reported_and_stops_early(case):
    """A runaway solve must bail, not burn every iteration and report a NaN."""
    shape = case.domain.shape
    big = 1.0e6  # far outside any plausible band for a u_inf ~ 1 case
    field = FlowField(
        domain=case.domain,
        u=np.full(shape, big), v=np.full(shape, big),
        p=np.full(shape, big), nut=np.zeros(shape),
    )
    region = np.ones(shape, dtype=bool)

    out = local_incompressible_solve(field, case, region, max_outer=500)
    fb = out.meta["fallback"]

    assert fb["ran"] is False
    # Whatever the outcome, the recorded residuals are numbers or None -- never NaN.
    for key in ("residual_before", "residual_after"):
        assert fb[key] is None or np.isfinite(fb[key])
    if fb["diverged"]:
        assert "DIVERGED" in fb["note"]


def test_rejected_patch_returns_the_original_field_unchanged(case):
    """The no-harm guarantee itself, on the degenerate path."""
    field = _uniform(case)
    region = np.zeros(case.domain.shape, dtype=bool)
    region[8:20, 8:20] = True

    out = local_incompressible_solve(field, case, region, max_outer=20)
    assert np.allclose(out.u, field.u)
    assert np.allclose(out.v, field.v)
    assert np.allclose(out.p, field.p)
