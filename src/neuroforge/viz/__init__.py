"""Visualisation layer: matplotlib plots + HTML report generation.

Importing this package selects the headless ``Agg`` matplotlib backend (in
:mod:`neuroforge.viz.plots`) so all plotting is safe on a server.
"""

from __future__ import annotations

from .plots import (
    overview_figure,
    plot_convergence,
    plot_cp,
    plot_field,
    plot_residual,
    plot_trust,
    plot_uncertainty,
)
from .report import build_report

__all__ = [
    "plot_field",
    "plot_residual",
    "plot_trust",
    "plot_uncertainty",
    "plot_cp",
    "plot_convergence",
    "overview_figure",
    "build_report",
]
