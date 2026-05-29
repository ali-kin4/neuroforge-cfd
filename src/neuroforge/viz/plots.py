"""Matplotlib plotting for NeuroForge CFD.

Every plot helper accepts an optional ``ax`` (creating one on a fresh figure if
``None``) and returns the :class:`matplotlib.axes.Axes` it drew on, except
:func:`overview_figure` which returns a whole :class:`matplotlib.figure.Figure`.
Extra keyword arguments are forwarded to the underlying matplotlib call so the
caller can tune ``cmap``, ``vmin``/``vmax`` etc.

A non-interactive (``Agg``) backend is selected at import time so the module is
safe to use headless / on a server.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless, before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from neuroforge.core.types import Diagnostics, FlowCase, FlowField  # noqa: E402

__all__ = [
    "plot_field",
    "plot_residual",
    "plot_trust",
    "plot_uncertainty",
    "plot_cp",
    "plot_convergence",
    "overview_figure",
]

# Body outline drawn in a neutral colour over field plots.
_BODY_COLOR = "0.15"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ensure_ax(ax: "plt.Axes | None") -> "plt.Axes":
    """Return ``ax`` or a new Axes on a fresh figure."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.0, 4.0))
    return ax


def _extent(domain) -> tuple[float, float, float, float]:
    """matplotlib ``extent`` (left, right, bottom, top) for a Domain."""
    xmin, xmax, ymin, ymax = domain.bounds
    return (float(xmin), float(xmax), float(ymin), float(ymax))


def _overlay_body(ax: "plt.Axes", domain, mask: "np.ndarray | None") -> None:
    """Draw the solid-body boundary as a contour of the fluid mask."""
    if mask is None:
        return
    mask = np.asarray(mask, dtype=float)
    if not np.any(mask < 0.5):  # no solid present
        return
    xmin, xmax, ymin, ymax = _extent(domain)
    ny, nx = mask.shape
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    try:
        ax.contour(X, Y, mask, levels=[0.5], colors=_BODY_COLOR, linewidths=1.2)
        ax.contourf(
            X, Y, (mask < 0.5).astype(float), levels=[0.5, 1.5],
            colors=[_BODY_COLOR], alpha=0.85,
        )
    except Exception:  # pragma: no cover - degenerate mask
        pass


def _masked(arr: np.ndarray, mask: "np.ndarray | None") -> np.ndarray:
    """Mask out solid cells (mask==0) so they don't dominate the colour scale."""
    arr = np.asarray(arr, dtype=float)
    if mask is None:
        return arr
    m = np.asarray(mask) < 0.5
    return np.ma.array(arr, mask=m)


def _show(
    ax: "plt.Axes", data: np.ndarray, domain, *, cmap: str, title: str,
    cbar_label: str, mask: "np.ndarray | None" = None, **kw: Any,
):
    """Shared pcolormesh-style renderer with colorbar + body overlay."""
    data = _masked(data, mask)
    extent = _extent(domain)
    kw.pop("ax", None)
    im = ax.imshow(
        data, origin="lower", extent=extent, aspect="auto",
        cmap=kw.pop("cmap", cmap), **kw,
    )
    _overlay_body(ax, domain, mask)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    return ax


# --------------------------------------------------------------------------- #
# Field plots
# --------------------------------------------------------------------------- #


def plot_field(field: FlowField, key: str = "speed", ax: "plt.Axes | None" = None, **kw: Any):
    """Plot a solution field on the domain.

    Parameters
    ----------
    field : FlowField
        The flow solution.
    key : {'u', 'v', 'p', 'speed', 'nut'}
        Which scalar to draw.
    ax : matplotlib.axes.Axes, optional
        Target Axes; a new one is created if ``None``.
    **kw
        Forwarded to ``imshow`` (e.g. ``cmap``, ``vmin``, ``vmax``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    key = key.lower()
    cmaps = {
        "u": "RdBu_r", "v": "RdBu_r", "p": "coolwarm",
        "speed": "viridis", "nut": "magma",
    }
    labels = {
        "u": "u", "v": "v", "p": "p (kinematic)",
        "speed": "|U|", "nut": "nu_t",
    }
    if key == "speed":
        data = field.speed()
    elif key in ("u", "v", "p", "nut"):
        data = getattr(field, key)
        if data is None:
            data = np.zeros(field.shape, dtype=float)
    else:
        raise ValueError(f"unknown field key {key!r}; expected u|v|p|speed|nut")
    return _show(
        ax, data, field.domain, cmap=cmaps.get(key, "viridis"),
        title=f"{labels.get(key, key)} field", cbar_label=labels.get(key, key),
        mask=field.mask, **kw,
    )


def plot_residual(diag: Diagnostics, key: str = "continuity", ax: "plt.Axes | None" = None, **kw: Any):
    """Plot a physics-residual map from :class:`Diagnostics`.

    Parameters
    ----------
    diag : Diagnostics
    key : {'continuity', 'momentum_x', 'momentum_y', 'bc_violation'}
    ax : matplotlib.axes.Axes, optional
    **kw
        Forwarded to ``imshow``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    key = key.lower()
    valid = {"continuity", "momentum_x", "momentum_y", "bc_violation"}
    if key not in valid:
        raise ValueError(f"unknown residual key {key!r}; expected one of {sorted(valid)}")
    data = np.asarray(getattr(diag, key), dtype=float)

    # continuity / momentum are signed -> diverging map centred on zero.
    # bc_violation is a non-negative magnitude -> sequential map.
    if key == "bc_violation":
        cmap = kw.pop("cmap", "inferno")
        return _show(
            ax, data, _domain_from_diag(diag, data.shape), cmap=cmap,
            title=f"{key} residual", cbar_label="|violation|", **kw,
        )
    cmap = kw.pop("cmap", "RdBu_r")
    vmax = float(np.nanmax(np.abs(data))) if data.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    kw.setdefault("vmin", -vmax)
    kw.setdefault("vmax", vmax)
    return _show(
        ax, data, _domain_from_diag(diag, data.shape), cmap=cmap,
        title=f"{key} residual", cbar_label="residual", **kw,
    )


def _domain_from_diag(diag: Diagnostics, shape: tuple[int, int]):
    """Build a placeholder Domain matching a diagnostics map's shape.

    Diagnostics maps carry no Domain; for standalone residual/trust plots we use
    a unit-extent domain so ``imshow`` has a sensible extent. When these maps are
    drawn inside :func:`overview_figure` the real domain extent is used instead.
    """
    from neuroforge.core.types import Domain

    ny, nx = shape
    return Domain(bounds=(0.0, 1.0, 0.0, 1.0), nx=nx, ny=ny)


def plot_trust(diag: Diagnostics, ax: "plt.Axes | None" = None, **kw: Any):
    """Discrete green/yellow/red trust-class heatmap with a legend.

    Parameters
    ----------
    diag : Diagnostics
        Uses ``diag.trust_class`` (2=green, 1=yellow, 0=red).
    ax : matplotlib.axes.Axes, optional
    **kw
        Forwarded to ``imshow``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.patches as mpatches

    ax = _ensure_ax(ax)
    tc = np.asarray(diag.trust_class, dtype=float)
    domain = _domain_from_diag(diag, tc.shape)

    colors = ["#d62728", "#f0c419", "#2ca02c"]  # 0 red, 1 yellow, 2 green
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    kw.pop("cmap", None)
    kw.pop("vmin", None)
    kw.pop("vmax", None)
    im = ax.imshow(
        tc, origin="lower", extent=_extent(domain), aspect="auto",
        cmap=cmap, norm=norm, **kw,
    )
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("trust map")
    handles = [
        mpatches.Patch(color=colors[2], label="green (reliable)"),
        mpatches.Patch(color=colors[1], label="yellow (acceptable)"),
        mpatches.Patch(color=colors[0], label="red (correct)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)
    return ax


def plot_uncertainty(diag: Diagnostics, ax: "plt.Axes | None" = None, **kw: Any):
    """Plot the predictive-uncertainty map.

    Parameters
    ----------
    diag : Diagnostics
        Uses ``diag.uncertainty``.
    ax : matplotlib.axes.Axes, optional
    **kw
        Forwarded to ``imshow``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    data = np.asarray(diag.uncertainty, dtype=float)
    cmap = kw.pop("cmap", "plasma")
    return _show(
        ax, data, _domain_from_diag(diag, data.shape), cmap=cmap,
        title="uncertainty", cbar_label="std", **kw,
    )


# --------------------------------------------------------------------------- #
# Surface pressure (Cp)
# --------------------------------------------------------------------------- #


def plot_cp(
    field: FlowField, case: FlowCase, ax: "plt.Axes | None" = None,
    ref: "FlowField | None" = None, **kw: Any,
):
    """Surface pressure coefficient ``Cp`` versus ``x/c`` along the airfoil.

    The Cp field is sampled at the body surface points; the conventional
    inverted y-axis (suction up) is applied. If ``ref`` (another FlowField) is
    given its Cp is overlaid for comparison.

    Parameters
    ----------
    field : FlowField
    case : FlowCase
        Supplies the surface geometry and freestream.
    ax : matplotlib.axes.Axes, optional
    ref : FlowField, optional
        Reference field overlaid as a dashed line.
    **kw
        Forwarded to ``plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from neuroforge.physics.metrics import (  # lazy: physics owned elsewhere
        _bilinear_sample,
        pressure_coefficient,
    )

    ax = _ensure_ax(ax)
    pts = np.asarray(case.geometry.surface_points, dtype=float).reshape(-1, 2)
    xmin, xmax, _, _ = case.geometry.bounding_box()
    chord = max(xmax - xmin, 1e-12)
    xc = (pts[:, 0] - xmin) / chord

    cp_field = pressure_coefficient(field, case)
    cp_surf = _bilinear_sample(cp_field, field.domain, pts)

    # Split upper / lower surface by sign of y relative to the camber midline so
    # the curve reads as the usual two-branch Cp plot.
    label = kw.pop("label", "prediction")
    ax.scatter(xc, cp_surf, s=6, color="#1f77b4", label=label, **kw)

    if ref is not None:
        cp_ref_field = pressure_coefficient(ref, case)
        cp_ref = _bilinear_sample(cp_ref_field, ref.domain, pts)
        ax.scatter(xc, cp_ref, s=6, color="#ff7f0e", marker="x", label="reference")

    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title("surface pressure coefficient")
    ax.axhline(0.0, color="0.7", lw=0.6)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()  # suction (negative Cp) plotted upward
    ax.legend(fontsize=7, loc="best")
    return ax


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #


def plot_convergence(history: "list[dict[str, float]]", ax: "plt.Axes | None" = None, **kw: Any):
    """Residual norm versus iteration (semilogy).

    Parameters
    ----------
    history : list of dict
        Per-iteration records with at least ``'residual_norm'`` (and optionally
        ``'iter'``, ``'max_uncertainty'``, ``'trust_mean'``).
    ax : matplotlib.axes.Axes, optional
    **kw
        Forwarded to ``plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    history = list(history or [])
    if not history:
        ax.text(0.5, 0.5, "no convergence history", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("convergence")
        return ax

    iters = [h.get("iter", i) for i, h in enumerate(history)]
    res = [float(h.get("residual_norm", np.nan)) for h in history]
    res_pos = [max(r, 1e-30) for r in res]  # guard for semilogy
    kw.setdefault("marker", "o")
    ax.semilogy(iters, res_pos, color="#d62728", label="residual norm", **kw)

    # Overlay trust_mean on a twin axis if present (linear 0..1).
    if any("trust_mean" in h for h in history):
        ax2 = ax.twinx()
        tm = [float(h.get("trust_mean", np.nan)) for h in history]
        ax2.plot(iters, tm, color="#2ca02c", marker="s", lw=1.0, label="trust mean")
        ax2.set_ylabel("trust mean", color="#2ca02c")
        ax2.set_ylim(0.0, 1.05)
        ax2.tick_params(axis="y", labelcolor="#2ca02c")

    ax.set_xlabel("iteration")
    ax.set_ylabel("residual norm", color="#d62728")
    ax.set_title("convergence")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    return ax


# --------------------------------------------------------------------------- #
# Overview figure
# --------------------------------------------------------------------------- #


def overview_figure(result) -> "plt.Figure":
    """Assemble a 2x3 multi-panel summary figure for a :class:`SolveResult`.

    Panels: speed field, pressure field, continuity residual, uncertainty,
    trust map, convergence. The residual/trust/uncertainty panels are drawn on
    the real case domain extent (rather than the unit placeholder) so all field
    panels line up. A suptitle reports the case name and key metrics.

    Parameters
    ----------
    result : SolveResult

    Returns
    -------
    matplotlib.figure.Figure
    """
    field = result.field
    diag = result.diagnostics
    case = result.case
    domain = field.domain

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0))

    # Row 1: speed, pressure, continuity residual.
    plot_field(field, key="speed", ax=axes[0, 0])
    plot_field(field, key="p", ax=axes[0, 1])

    # Continuity residual on the real domain extent.
    _residual_panel(axes[0, 2], diag.continuity, domain, field.mask,
                    title="continuity residual")

    # Row 2: uncertainty, trust, convergence.
    _residual_panel(axes[1, 0], diag.uncertainty, domain, field.mask,
                    title="uncertainty", cmap="plasma", diverging=False,
                    cbar_label="std")
    _trust_panel(axes[1, 1], diag.trust_class, domain)
    plot_convergence(result.history, ax=axes[1, 2])

    metrics = result.summary()
    bits = []
    for k in ("cl", "cd", "cm", "residual_norm", "n_iters"):
        if k in metrics:
            bits.append(f"{k}={metrics[k]:.4g}")
    suptitle = case.name + ("  |  " + "  ".join(bits) if bits else "")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _residual_panel(ax, data, domain, mask, *, title, cmap="RdBu_r",
                    diverging=True, cbar_label="residual"):
    """Internal: draw a residual/uncertainty map on the real domain extent."""
    data = _masked(data, mask)
    kwargs: dict[str, Any] = {}
    if diverging:
        raw = np.asarray(data)
        vmax = float(np.nanmax(np.abs(raw))) if raw.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        kwargs.update(vmin=-vmax, vmax=vmax)
    im = ax.imshow(data, origin="lower", extent=_extent(domain),
                   aspect="auto", cmap=cmap, **kwargs)
    _overlay_body(ax, domain, mask)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    return ax


def _trust_panel(ax, trust_class, domain):
    """Internal: trust-class panel on the real domain extent (with legend)."""
    import matplotlib.patches as mpatches

    tc = np.asarray(trust_class, dtype=float)
    colors = ["#d62728", "#f0c419", "#2ca02c"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(tc, origin="lower", extent=_extent(domain), aspect="auto",
              cmap=cmap, norm=norm)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("trust map")
    handles = [
        mpatches.Patch(color=colors[2], label="green"),
        mpatches.Patch(color=colors[1], label="yellow"),
        mpatches.Patch(color=colors[0], label="red"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)
    return ax
