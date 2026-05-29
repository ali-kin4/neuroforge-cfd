"""Visualization tests: plot helpers run without error; report writes HTML."""

from __future__ import annotations

import os

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neuroforge.core.types import SolveResult  # noqa: E402
from neuroforge.physics.residuals import PhysicsChecker  # noqa: E402
from neuroforge.viz import plots  # noqa: E402
from neuroforge.viz.report import build_report  # noqa: E402


@pytest.fixture(scope="module")
def solve_result():
    """Build a SolveResult manually from a synthetic field + diagnostics."""
    from neuroforge import FlowCase
    from neuroforge.data.synthetic import SyntheticRANS

    case = FlowCase.from_airfoil("naca2412", aoa=4.0, reynolds=3e6,
                                 u_inf=30.0, resolution=32)
    field = SyntheticRANS(resolution=32, seed=0).solve(case)
    diag = PhysicsChecker().diagnose(field, case)
    history = [
        {"iter": 0, "residual_norm": 1.0, "max_uncertainty": 0.0, "trust_mean": 0.8},
        {"iter": 1, "residual_norm": 0.5, "max_uncertainty": 0.0, "trust_mean": 0.9},
    ]
    return SolveResult(case=case, field=field, diagnostics=diag,
                       metrics={"cl": 0.4, "cd": 0.02}, history=history)


def test_plot_field(solve_result):
    ax = plots.plot_field(solve_result.field, key="speed")
    assert ax is not None
    plt.close("all")


def test_plot_residual_and_trust_and_uncertainty(solve_result):
    diag = solve_result.diagnostics
    assert plots.plot_residual(diag, key="continuity") is not None
    assert plots.plot_trust(diag) is not None
    assert plots.plot_uncertainty(diag) is not None
    plt.close("all")


def test_plot_cp(solve_result):
    ax = plots.plot_cp(solve_result.field, solve_result.case)
    assert ax is not None
    plt.close("all")


def test_plot_convergence(solve_result):
    ax = plots.plot_convergence(solve_result.history)
    assert ax is not None
    plt.close("all")


def test_overview_figure(solve_result):
    fig = plots.overview_figure(solve_result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_report_writes_html(solve_result, tmp_path):
    out = os.path.join(tmp_path, "report.html")
    written = build_report(solve_result, out)
    assert written == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1024  # > 1KB
    # sibling PNG is written too.
    assert os.path.exists(out + ".png")
