"""Tests for the warm-start acceptance gate.

The gate's whole value is that it may not cheat. Two ways it could: by looking at
the cold run it is supposed to be an alternative to, and by choosing its
threshold on the case it is scored on. The first is structural -- ``features``
is handed one trace and nothing else -- and the second is what leave-one-case-out
is for. What is tested here is the feature extraction those rest on.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.certificate import features


def test_a_probe_shorter_than_the_run_is_refused():
    assert features(np.array([1e-2, 1e-3, 1e-4]), k=10) is None


def test_the_level_is_the_residual_at_the_probe_not_at_the_end():
    res = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    assert features(res, k=2)["level"] == pytest.approx(-4.0)
    assert features(res, k=4)["level"] == pytest.approx(-6.0)


def test_the_drop_is_measured_from_the_seed_own_start():
    """A seed that starts low and stalls must not look like one that fell far."""
    fell = np.array([1e-1, 1e-2, 1e-3, 1e-4])
    stalled = np.array([1e-4, 1e-4, 1e-4, 1e-4])
    assert features(fell, k=3)["drop"] == pytest.approx(-3.0)
    assert features(stalled, k=3)["drop"] == pytest.approx(0.0)
    # Same level, opposite trajectory: the level alone cannot tell them apart.
    assert features(fell, k=3)["level"] == features(stalled, k=3)["level"]


def test_infinities_from_padding_are_ignored():
    res = np.array([np.inf, 1e-2, 1e-3, 1e-4])
    got = features(res, k=3)
    assert got["level"] == pytest.approx(-4.0)
    assert got["drop"] == pytest.approx(-2.0)


def test_a_trace_of_nothing_usable_returns_none():
    assert features(np.array([np.inf, np.inf, np.inf]), k=2) is None
    assert features(np.array([0.0, 0.0, 0.0]), k=2) is None
