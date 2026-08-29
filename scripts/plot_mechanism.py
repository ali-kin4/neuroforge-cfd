"""The paper's central figure: representation -> wall gradient -> drag.

Three panels, one argument, every number from a committed ``results/*.json``.

**A.** What each seed does to the first-cell velocity gradient, the quantity
viscous drag integrates. The bar to look at is not the cold start's 2851% but the
*projections of the exact answer* at 1695% and 1890%: a 16,384-value grid,
Cartesian or wall-fitted, has no station 4e-6 chords off the wall, so projecting
even a perfect field through one removes barely half of a cold start's error. The
same network prediction reads 54% evaluated at the cell centres and 1583%
resampled -- which is the controlled version of the whole claim, and the pair the
eye should land on.

**B.** The chain closed. Wall-gradient error against drag-convergence saving,
one point per arm. Arms cluster into the ones that kept the gradient and the ones
that lost it, and the split in saving follows.

**C.** The 2x2. Mesh-native evaluation and boundary-layer-only handover are each
necessary and neither is sufficient; only the corner with both is positive.

Usage
-----
    python scripts/plot_mechanism.py
    python scripts/plot_mechanism.py --out docs/paper/fig_mechanism.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#85837c"
GRID, SURFACE = "#e4e2dc", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, BAD = "#1baf7a", "#d92c2c"

# label, colour, and whether the seed reached the solver without a round-trip
SEEDS = [
    ("cold", "cold start\n(uniform freestream)", INK_3, False),
    ("cartesian_128", "exact answer through\na 128$^2$ Cartesian grid", BAD, False),
    ("fitted_bl", "exact answer through\na 256$\\times$64 wall-fitted grid", ORANGE, False),
    ("nf_bl_proj", "NeuroForge prediction\nthrough the same grid", ORANGE, False),
    ("nf_bl", "NeuroForge prediction\nat the cell centres", GOOD, True),
    ("oracle_mesh", "the exact answer\nitself", BLUE, True),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gradient", default=os.path.join("results", "seed_gradient.json"))
    ap.add_argument("--depth", default=os.path.join("results", "depth_reanalysis.json"))
    ap.add_argument("--out", default=os.path.join("results", "mechanism.png"))
    ap.add_argument("--row", default="Cd@0.01",
                    help="which force row to plot; only readable rows should be used")
    args = ap.parse_args(argv)

    for path in (args.gradient, args.depth):
        if not os.path.isfile(path):
            print(f"missing {path} -- run the diagnostic and the re-analysis first")
            return 1
    with open(args.gradient, encoding="utf-8") as fh:
        grad = json.load(fh)["summary"]
    with open(args.depth, encoding="utf-8") as fh:
        depth = json.load(fh)["by_force"]

    row = depth.get(args.row)
    if row is None:
        print(f"{args.row} not in {args.depth}; have: {', '.join(sorted(depth))}")
        return 1
    if not row.get("readable", True):
        print(f"warning: {args.row} is marked unreadable "
              f"(settled arms disagree by {100 * row.get('settled_spread', 0):.2f}%)")

    def saving(arm):
        entry = row.get(arm)
        return None if not entry else entry.get("saving")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_3)
        ax.tick_params(colors=INK_2, labelsize=9)

    # --- A: what each representation does to the wall gradient ---------------
    ax = axes[0]
    present = [(k, lab, col) for k, lab, col, _ in SEEDS if k in grad]
    values = [100 * grad[k]["grad_err"] for k, _, _ in present]
    y = np.arange(len(present))
    ax.barh(y, values, color=[c for _, _, c in present], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in present], fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("error in the first-cell wall gradient  $du_t/dy$  (%)",
                  fontsize=9.5, color=INK_2)
    ax.set_title("A.  A 16k grid has no station at the wall",
                 fontsize=11, color=INK, loc="left", pad=12)
    for yi, v in zip(y, values):
        ax.text(max(v, 0.6) * 1.25, yi, f"{v:.0f}%" if v >= 1 else f"{v:.1f}%",
                va="center", fontsize=8.5, color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1e4)

    # --- B: the chain -- gradient fidelity against what it bought ------------
    ax = axes[1]
    points = []
    for key, label, colour, native in SEEDS:
        if key not in grad or key == "cold":
            continue
        s = saving(key)
        if s is None:
            continue
        points.append((max(100 * grad[key]["grad_err"], 0.5), 100 * s,
                       label.replace("\n", " "), colour, native))
    for x, y, label, colour, native in points:
        ax.scatter([x], [y], s=150, color=colour, zorder=3,
                   marker="o" if native else "s",
                   edgecolor=SURFACE, linewidth=1.5)
    # The arms separate cleanly in y and pile up in one decade of x, so labels go
    # sideways, not above and below: three stacked labels in the same decade is
    # what the first draft produced. Points on the right are labelled leftwards
    # and vice versa, so no label leaves the axes.
    xs = [p[0] for p in points]
    midpoint = np.sqrt(min(xs) * max(xs))
    for x, y, label, _colour, _native in points:
        right = x < midpoint
        ax.annotate(label, (x, y), fontsize=8, color=INK_2,
                    xytext=(14 if right else -14, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center")
    lo, hi = min(p[1] for p in points), max(p[1] for p in points)
    pad = 0.14 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0.25, 1.2e4)
    ax.axhline(0, color=INK_3, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("error in the first-cell wall gradient (%)",
                  fontsize=9.5, color=INK_2)
    ax.set_ylabel(f"iteration saving on {args.row.replace('@0.01', '@1%')} (%)",
                  fontsize=9.5, color=INK_2)
    ax.set_title("B.  Keep the gradient, keep the saving",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    # Top right. The bottom corners both hold a point or its label; above the
    # zero line and right of the mesh-native cluster is the empty quadrant.
    ax.text(0.99, 0.97, "circles: evaluated at the cell centres\n"
                        "squares: resampled through a grid",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=INK_3)

    # --- C: the 2x2 ----------------------------------------------------------
    ax = axes[2]
    cells = [["fitted_256x64", "fitted_bl"], ["nf_mesh", "nf_bl"]]
    grid_vals = np.full((2, 2), np.nan)
    for i, rowk in enumerate(cells):
        for j, key in enumerate(rowk):
            s = saving(key)
            if s is not None:
                grid_vals[i, j] = 100 * s
    # Scaled to +/-100% and clipped. On the true range one arm at -568% pushes
    # everything else into the middle of the ramp, so -173% reads warmer than
    # -206% and the +34% barely registers as green -- the colour would then say
    # the opposite of the numbers. Clipping makes the sign legible; the printed
    # value is always the real one.
    limit = 100.0
    ax.imshow(np.clip(grid_vals, -limit, limit), cmap="RdYlGn",
              vmin=-limit, vmax=limit, aspect="auto")
    for i in range(2):
        for j in range(2):
            v = grid_vals[i, j]
            ax.text(j, i, "--" if not np.isfinite(v) else f"{v:+.0f}%",
                    ha="center", va="center", fontsize=17,
                    color=SURFACE if np.isfinite(v) and abs(v) > 55 else INK,
                    fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["whole field", "boundary layer only"],
                                              fontsize=9, color=INK_2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["resampled\nto a 16k grid",
                                               "mesh-native"],
                                              fontsize=9, color=INK_2)
    ax.set_title("C.  Both necessary, neither sufficient",
                 fontsize=11, color=INK, loc="left", pad=12)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.text(0.5, -0.19, "colour clipped at $\\pm$100%; the printed value is the "
                        "measured one", transform=ax.transAxes, ha="center",
            fontsize=7.8, color=INK_3)

    n = row.get("cold_n")
    fig.suptitle(
        f"A surrogate must be evaluable at the solver's own cell centres    "
        f"(Re 3$\\times$10$^6$, {n} cases, cold = {row.get('cold_mean', 0):.0f} "
        f"iterations, oracle control "
        f"{100 * (saving('oracle_mesh') or 0):+.0f}%)",
        fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
