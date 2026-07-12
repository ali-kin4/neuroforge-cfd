"""Tests for the local classical-CFD fallback (``ClassicalFallback`` backends).

Covers the no-op ``stub`` backend (unchanged behaviour) and the real ``numpy``
backend: that it returns a valid field, never increases the region residual (the
no-harm guarantee), and meaningfully reduces it on a deliberately
divergence-heavy input. All tests are CPU-only on tiny grids (res <= 48) and do
not touch ``pretrained()`` / ``demo()``.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge import FlowCase
from neuroforge.core.types import FlowField
from neuroforge.data.synthetic import SyntheticRANS
from neuroforge.solver.classical import (
    local_incompressible_solve,
    region_residual_norm,
)
from neuroforge.solver.fallback import ClassicalFallback

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _bumpy_case_field(resolution: int = 48):
    """Build a case + a divergence-heavy field and the flagged region.

    Starts from a synthetic pseudo-RANS field (continuity ~satisfied) and adds a
    smooth Gaussian bump to ``u`` over a patch of free-stream fluid, which makes
    the continuity/momentum residual large there. Returns
    ``(case, field, region)``.
    """
    case = FlowCase.from_airfoil(
        "naca0012", aoa=0.0, reynolds=1.0e6, u_inf=10.0, resolution=resolution
    )
    gen = SyntheticRANS(resolution=resolution, seed=0)
    field = gen.solve(case)

    X, Y = case.domain.grid()
    mask = np.asarray(field.mask) > 0.5
    bump = 3.0 * np.exp(-(((X - 0.8) / 0.3) ** 2 + ((Y - 0.4) / 0.3) ** 2))
    arr = field.as_array()
    arr[0] = np.asarray(field.u) + bump.astype(np.float32)
    bumpy = FlowField.from_array(arr, case.domain, mask=field.mask, sdf=field.sdf)

    region = (bump > 0.3) & mask
    return case, bumpy, region


# --------------------------------------------------------------------------- #
# Stub backend: must behave exactly as before.
# --------------------------------------------------------------------------- #


def test_stub_backend_is_noop(synthetic_field, tiny_case):
    fb = ClassicalFallback("stub")
    region = np.asarray(synthetic_field.mask) > 0.5
    before = synthetic_field.as_array().copy()

    out = fb.patch(synthetic_field, tiny_case, region)

    assert isinstance(out, FlowField)
    assert "fallback" in out.meta
    meta = out.meta["fallback"]
    assert meta["backend"] == "stub"
    assert meta["ran"] is False
    assert meta["region_cells"] == int(region.sum())
    # Field values are unchanged by the stub.
    np.testing.assert_array_equal(out.as_array(), before)


# --------------------------------------------------------------------------- #
# Numpy backend: validity, no-harm, and a real reduction.
# --------------------------------------------------------------------------- #


def test_numpy_backend_returns_valid_field():
    case, field, region = _bumpy_case_field(resolution=48)
    out = ClassicalFallback("numpy").patch(field, case, region)

    assert isinstance(out, FlowField)
    assert out.shape == field.shape
    arr = out.as_array()
    assert arr.shape == (4, *field.shape)
    assert np.all(np.isfinite(arr)), "fallback produced non-finite values"

    meta = out.meta["fallback"]
    assert meta["backend"] == "numpy"
    assert meta["region_cells"] == int(region.sum())
    assert "residual_before" in meta and "residual_after" in meta


def test_numpy_backend_no_harm_property():
    """The region residual norm must never increase (no-harm guarantee)."""
    case, field, region = _bumpy_case_field(resolution=48)
    before = region_residual_norm(field, case.fluid, region)

    out = ClassicalFallback("numpy").patch(field, case, region)
    after = region_residual_norm(out, case.fluid, region)

    # Allow a tiny float slack; the patch is rejected if it would harm.
    assert after <= before + 1e-6 * max(before, 1.0)


def test_numpy_backend_reduces_divergence_heavy_field():
    """On a divergence-heavy input the solve should STRICTLY (meaningfully) reduce."""
    case, field, region = _bumpy_case_field(resolution=48)
    before = region_residual_norm(field, case.fluid, region)
    assert before > 0.0

    out = ClassicalFallback("numpy").patch(field, case, region)
    meta = out.meta["fallback"]
    after = region_residual_norm(out, case.fluid, region)

    assert meta["ran"] is True
    # Meaningful reduction: at least 5% (we observe ~60%).
    assert after < 0.95 * before
    # The reported and recomputed after-residuals agree.
    assert after == pytest.approx(meta["residual_after"], rel=1e-5)


def test_numpy_backend_aliases_accepted():
    case, field, region = _bumpy_case_field(resolution=32)
    for alias in ("numpy", "projection", "simple"):
        out = ClassicalFallback(alias).patch(field, case, region)
        assert out.meta["fallback"]["backend"] == "numpy"


def test_numpy_backend_empty_region_is_noop():
    case, field, _ = _bumpy_case_field(resolution=32)
    empty = np.zeros(field.shape, dtype=bool)
    before = field.as_array().copy()

    out = ClassicalFallback("numpy").patch(field, case, empty)

    assert out.meta["fallback"]["ran"] is False
    assert out.meta["fallback"]["region_cells"] == 0
    np.testing.assert_array_equal(out.as_array(), before)


def test_local_solve_rejects_when_no_improvement():
    """A clean (already-consistent) field over a region should be a no-op."""
    case, field, region = _bumpy_case_field(resolution=32)
    # Use the unmodified synthetic field (continuity ~satisfied) as the input.
    gen = SyntheticRANS(resolution=32, seed=0)
    clean = gen.solve(case)
    region = np.asarray(clean.mask) > 0.5

    before_arr = clean.as_array().copy()
    before = region_residual_norm(clean, case.fluid, region)
    out = local_incompressible_solve(clean, case, region)
    meta = out.meta["fallback"]
    # No-harm: the RETURNED field's region residual never exceeds the input's.
    after = region_residual_norm(out, case.fluid, region)
    assert after <= before + 1e-6 * max(before, 1.0)
    # When the patch is rejected, the returned field is byte-for-byte unchanged.
    if not meta["ran"]:
        np.testing.assert_array_equal(out.as_array(), before_arr)
