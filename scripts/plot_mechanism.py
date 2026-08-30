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

**C.** The two contrasts the study actually controlled, sharing the arm ``nf_bl``
and moving one variable each: representation (cell centres vs resampled) and
region (boundary layer vs whole field). This replaced a 2x2 on 2026-08-30 whose
top row was oracle arms and whose bottom row was network arms -- the cell that
would have completed it was never run, and four cells from two populations read
as a design when they are a gap.

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
    # Split on position along the *drawn* log axis, not on the geometric mean of
    # the data: four of the five arms sit in one decade near the right-hand end,
    # so the geometric mean lands among them and sends the cell-centre arm's
    # label leftwards into the y-axis.
    x_lo, x_hi = 0.25, 1.2e4
    def frac(v):
        return np.log10(v / x_lo) / np.log10(x_hi / x_lo)
    for x, y, label, _colour, _native in points:
        right = frac(x) < 0.6
        ax.annotate(label, (x, y), fontsize=8, color=INK_2,
                    xytext=(14 if right else -14, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center")
    lo, hi = min(p[1] for p in points), max(p[1] for p in points)
    pad = 0.14 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(x_lo, x_hi)
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

    # --- C: the two controlled contrasts -------------------------------------
    # This panel used to be a 2x2. It was retired on 2026-08-30: its top row was
    # oracle arms and its bottom row network arms, and the cell that would have
    # made it a real 2x2 -- a network prediction of the whole field, resampled --
    # was never run. Four cells drawn from two populations read as a design when
    # they are a gap. What replaces it is what the study actually controlled:
    # two contrasts that share the arm nf_bl and move one variable each.
    ax = axes[2]
    contrasts = [
        ("representation",
         [("at the cell\ncentres", "nf_bl", GOOD),
          ("resampled\nthrough 256x64", "nf_bl_proj", ORANGE)]),
        ("region",
         [("boundary\nlayer only", "nf_bl", GOOD),
          ("the whole\nfield", "nf_mesh", BAD)]),
    ]
    xs, heights, colours, labels, groups = [], [], [], [], []
    x = 0.0
    for gi, (gname, pair) in enumerate(contrasts):
        groups.append((x + 0.5, gname))
        for lab, key, col in pair:
            s = saving(key)
            xs.append(x)
            heights.append(100 * s if s is not None else np.nan)
            colours.append(col)
            labels.append(lab)
            x += 1.0
        x += 0.7
    floor_v = min(v for v in heights if np.isfinite(v))
    span = 150 - floor_v * 1.30
    ax.axhline(0, color=INK_2, lw=1.0, zorder=4)
    ax.bar(xs, heights, width=0.82, color=colours, zorder=3)
    for xi, v, lab in zip(xs, heights, labels):
        up = v >= 0
        # Value *and* label both beyond the bar's free end, stacked, so each
        # bar owns one column of text. Splitting them across the zero line put
        # a negative bar's label at the same height as its neighbour's value,
        # which read as if the label belonged to the wrong bar.
        ax.text(xi, v + (0.025 if up else -0.025) * span, f"{v:+.0f}%",
                ha="center", va="bottom" if up else "top",
                fontsize=13, fontweight="bold", color=INK, zorder=5)
        ax.text(xi, v + (0.085 if up else -0.085) * span, lab,
                ha="center", va="bottom" if up else "top",
                fontsize=8, color=INK_2, zorder=5)
    for xc, gname in groups:
        ax.text(xc - 0.5, -0.055, gname, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color=INK_3)
    ax.set_ylim(floor_v * 1.30, 235)
    ax.set_xlim(-0.7, x - 0.9)
    ax.set_xticks([])
    ax.set_ylabel(f"iteration saving on {args.row.replace('@0.01', '@1%')} (%)",
                  fontsize=9, color=INK_2)
    ax.set_title("C.  Two controlled contrasts, one variable each",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.text(0.5, -0.135, "each pair is the same network and the same prediction, "
                         "one variable apart",
            transform=ax.transAxes, ha="center", fontsize=7.8, color=INK_3)

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
