"""Render the geometry and both body-fitted meshes to a figure.

Reconstructs the node positions the way ``blockMesh`` does -- transfinite between
the ring curves with the same geometric grading the dictionary specifies -- so
what is drawn is the mesh that gets solved, not a sketch of it.
"""

from __future__ import annotations

import argparse
import os

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

from neuroforge.solver import cgrid as cg, ogrid as og

INK = "#0b0b0b"
INK_2 = "#52514e"
MESH = "#c2c0b8"
BLUE = "#2a78d6"      # the mesh / the wall-normal distribution
ORANGE = "#eb6834"    # the wake cut, and the surrogate's cell
AQUA = "#1baf7a"      # the boundary layer
SURFACE = "#fcfcfb"

# Every panel that draws a mesh is equal-aspect, so its data window is sized to
# the subplot box or the panel floats in whitespace.
BOX = 1.19


def _grade_fracs(n_cells: int, ratio: float) -> np.ndarray:
    """Normalised node positions along an edge under blockMesh `simpleGrading`."""
    if n_cells < 1:
        return np.array([0.0, 1.0])
    r = ratio ** (1.0 / max(n_cells - 1, 1)) if ratio != 1.0 else 1.0
    sizes = r ** np.arange(n_cells) if r != 1.0 else np.ones(n_cells)
    s = np.concatenate([[0.0], np.cumsum(sizes)])
    return s / s[-1]


def _stack(rings, n_cells, ratios):
    """(n_i, n_nodes, 2) node array from ring curves + per-layer grading."""
    cols = [rings[0]]
    for k, (n, ratio) in enumerate(zip(n_cells, ratios)):
        a, b = rings[k], rings[k + 1]
        for t in _grade_fracs(n, ratio)[1:]:
            cols.append(a + t * (b - a))
    return np.stack(cols, axis=1)


def _gradings(spec):
    g_in = og.expansion_ratio(spec.offset, spec.first_cell, spec.n_inner)
    growth = g_in ** (1.0 / max(spec.n_inner - 1, 1))
    last = spec.first_cell * growth ** max(spec.n_inner - 1, 0)
    g_out = og.expansion_ratio(spec.far_radius - spec.offset, last, spec.n_outer)
    return g_in, g_out


def cgrid_nodes(code="naca0012", spec=None):
    spec = spec or cg.CGridSpec()
    inner, nw, ns = cg.inner_curve(code, spec)
    off = cg.offset_open(inner, spec.offset, spec.n_smooth,
                         smooth_range=(nw - 1 - spec.smooth_pad, nw + ns - 2 + spec.smooth_pad))
    far = cg.outer_curve(spec, nw, ns)
    g_in, g_out = _gradings(spec)
    return _stack([inner, off, far], [spec.n_inner, spec.n_outer], [g_in, g_out]), spec, nw, ns


def ogrid_nodes(code="naca0012", spec=None):
    spec = spec or og.OGridSpec()
    loop = og.airfoil_loop(code, n_surface=spec.n_surface, n_te=spec.n_te)
    off = og.offset_curve(loop, spec.offset, spec.n_smooth)
    far = og.far_field_circle(off, spec.far_radius, spec.centre)
    g_in, g_out = _gradings(spec)
    rings = [np.vstack([c, c[:1]]) for c in (loop, off, far)]
    return _stack(rings, [spec.n_inner, spec.n_outer], [g_in, g_out]), spec


def draw_mesh(ax, nodes, every_i=4, every_j=3, lw=0.35, color=MESH):
    ni, nj, _ = nodes.shape
    for i in list(range(0, ni, every_i)) + [ni - 1]:
        ax.plot(nodes[i, :, 0], nodes[i, :, 1], color=color, lw=lw, zorder=1)
    for j in list(range(0, nj, every_j)) + [nj - 1]:
        ax.plot(nodes[:, j, 0], nodes[:, j, 1], color=color, lw=lw, zorder=1)


def frame(ax, num, title, sub, xlim, ylim):
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(f"{num} · {title}", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=6)
    # Subtitle inside the axes: at 1.0 it collides with the panel above.
    ax.text(0.012, 0.972, sub, transform=ax.transAxes, color=INK_2,
            fontsize=8.5, va="top", ha="left", zorder=8, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.82, pad=2.0))


def box_ylim(xlim, centre=0.0):
    half = (xlim[1] - xlim[0]) / BOX / 2
    return (centre - half, centre + half)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("results", "mesh_structure.png"))
    ap.add_argument("--airfoil", default="naca0012")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cn, cspec, nw, ns = cgrid_nodes(args.airfoil)
    on, ospec = ogrid_nodes(args.airfoil)
    surf = cn[nw - 1: nw + ns - 1, 0, :]
    cut_lo = cn[:nw, 0, :]

    fig = plt.figure(figsize=(16.5, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.06,
                          left=0.015, right=0.988, top=0.885, bottom=0.055)

    # 1 ── the geometry itself ------------------------------------------------ #
    xl = (-0.14, 1.20); yl = box_ylim(xl)
    ax = fig.add_subplot(gs[0, 0])
    ax.fill(surf[:, 0], surf[:, 1], color=INK, zorder=3)
    ax.plot([0, 1], [0, 0], color=INK_2, lw=0.9, ls=(0, (7, 5)), zorder=4)
    ax.annotate("sharp trailing edge —\nthe wake cut springs from it",
                xy=(1.002, 0.0), xytext=(0.60, 0.20), fontsize=9, color=ORANGE,
                ha="center", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.annotate("leading edge, radius 0.016 chord", xy=(0.002, 0.0),
                xytext=(0.30, -0.20), fontsize=9, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.0))
    ax.annotate("", xy=(0, -0.085), xytext=(1, -0.085),
                arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1.0))
    ax.text(0.5, -0.125, "chord = 1", ha="center", fontsize=9, color=INK_2)
    ax.text(0.32, 0.085, "max thickness 12%", ha="center", fontsize=9, color=INK_2)
    frame(ax, "1", f"Geometry — {args.airfoil.upper()}",
          f"{ns} surface points, clustered at both edges", xl, yl)

    # 2 ── full C-grid domain -------------------------------------------------- #
    xl = (-22.5, 23.0); yl = box_ylim(xl)
    ax = fig.add_subplot(gs[0, 1])
    draw_mesh(ax, cn, every_i=6, every_j=6, lw=0.3)
    ax.plot(cn[:, -1, 0], cn[:, -1, 1], color=BLUE, lw=2.0, zorder=4)
    x_out = cspec.centre[0] + cspec.wake_length
    ax.plot([x_out, x_out], [-cspec.far_radius, cspec.far_radius],
            color=INK_2, lw=1.6, zorder=4)
    ax.fill(surf[:, 0], surf[:, 1], color=INK, zorder=5)
    ax.text(-20.5, 4.0, "far field\nfreestream\n20 chords", fontsize=9,
            color=BLUE, ha="left", va="center", fontweight="bold", zorder=8,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.82, pad=2.0))
    ax.text(x_out - 1.0, -14.0, "outlet", fontsize=9, color=INK_2,
            ha="right", va="center", fontweight="bold", zorder=8, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.82, pad=2.0))
    frame(ax, "2", "C-grid, full domain",
          f"{cspec.n_cells:,} cells — the C wraps the nose,\nthe wake runs to the outlet", xl, yl)

    # 3 ── near field ---------------------------------------------------------- #
    xl = (-0.85, 3.15); yl = box_ylim(xl, centre=0.05)
    ax = fig.add_subplot(gs[0, 2])
    draw_mesh(ax, cn, every_i=2, every_j=2, lw=0.3)
    ax.plot(cut_lo[:, 0], cut_lo[:, 1], color=ORANGE, lw=2.0, zorder=4)
    ax.fill(surf[:, 0], surf[:, 1], color=INK, zorder=5)
    ax.annotate("wake cut: two coincident sheets that\nshare vertices — so no stitchMesh",
                xy=(2.0, 0.0), xytext=(1.55, -0.95), fontsize=9, color=ORANGE,
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    frame(ax, "3", "C-grid, near field",
          "a streamwise-aligned wake block —\nwhat the O-grid lacks", xl, yl)

    # 4 ── trailing edge ------------------------------------------------------- #
    xl = (0.918, 1.142); yl = box_ylim(xl, centre=0.0)
    ax = fig.add_subplot(gs[1, 0])
    draw_mesh(ax, cn, every_i=1, every_j=1, lw=0.25)
    ax.plot(cut_lo[:, 0], cut_lo[:, 1], color=ORANGE, lw=1.8, zorder=4)
    ax.fill(surf[:, 0], surf[:, 1], color=INK, zorder=5)
    ax.text(1.055, 0.058, "cells above and below the cut\nare separate; the cut itself\nis an internal face",
            fontsize=9, color=ORANGE, ha="center", va="center", fontweight="bold",
            zorder=8, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.82, pad=2.0))
    frame(ax, "4", "Trailing edge, ×9",
          "the cusp where the cut begins", xl, yl)

    # 5 ── wall-normal distribution -------------------------------------------- #
    ax = fig.add_subplot(gs[1, 1])
    mid = nw - 1 + ns // 2
    d = np.linalg.norm(cn[mid, :, :] - cn[mid, 0, :], axis=1)
    d[0] = d[1] / 3.0
    idx = np.arange(len(d))
    bl, sur = 0.019, 3.0 / 127

    ax.fill_between(idx, 1e-7, bl, color=AQUA, alpha=0.10, zorder=1)
    ax.plot(idx, d, color=BLUE, lw=2.0, zorder=4)
    ax.scatter(idx[::5], d[::5], s=11, color=BLUE, zorder=5)

    ax.axhline(bl, color=AQUA, lw=1.6, zorder=3)
    ax.text(2, bl * 1.5, "boundary layer  0.019 chord", color=AQUA,
            fontsize=9.5, fontweight="bold", va="bottom")
    ax.axhline(sur, color=ORANGE, lw=1.6, ls=(0, (6, 3)), zorder=3)
    ax.text(len(d) - 2, sur * 0.62, "ONE cell of the 128² surrogate grid  0.0236 chord",
            color=ORANGE, fontsize=9.5, fontweight="bold", ha="right", va="top")
    ax.axhline(1e-5, color=INK_2, lw=1.0, ls=":", zorder=3)
    ax.text(2, 1.4e-5, "first cell  1e-5 chord  (y⁺ ≈ 1)", color=INK_2,
            fontsize=9, va="bottom")

    ax.set_yscale("log"); ax.set_ylim(4e-6, 60); ax.set_xlim(0, len(d) - 1)
    ax.set_xlabel("cell index outward from the wall", color=INK_2, fontsize=9)
    ax.set_ylabel("distance from wall  [chord]", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MESH)
    ax.set_facecolor(SURFACE)
    ax.set_title("5 · Wall-normal spacing — why a finer surrogate won't help",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=6)
    ax.text(0.012, 0.972,
            "the mesh spends 50 of its 100 cells below the\n"
            "surrogate's single cell — that is where SIMPLE works",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, va="top")

    # 6 ── O-grid, for contrast ------------------------------------------------ #
    xl = (-0.85, 3.15); yl = box_ylim(xl, centre=0.05)
    ax = fig.add_subplot(gs[1, 2])
    draw_mesh(ax, on, every_i=2, every_j=2, lw=0.3)
    ax.fill(on[:, 0, 0], on[:, 0, 1], color=INK, zorder=5)
    ax.annotate("radial lines fan out — the wake is\nresolved no better than the far field",
                xy=(2.5, 0.55), xytext=(1.75, -0.95), fontsize=9, color=INK_2,
                ha="center", arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.1))
    frame(ax, "6", "O-grid, same view (superseded)",
          f"{ospec.n_cells:,} cells — blunt trailing edge,\nno wake block", xl, yl)

    fig.suptitle("NeuroForge · OpenFOAM body-fitted meshes for the Paper-2 warm-start study",
                 color=INK, fontsize=15, fontweight="bold", x=0.015, ha="left", y=0.975)
    fig.text(0.015, 0.933,
             "Reconstructed from the same ring curves and geometric grading blockMesh is given — "
             "this is the mesh that gets solved, not a sketch of it.",
             color=INK_2, fontsize=9.5, ha="left")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
