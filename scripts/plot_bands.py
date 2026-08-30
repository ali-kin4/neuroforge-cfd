"""Figure 2: the saving is a function of the band, and only one quantity is flat.

The paper's headline is +33.9% on total drag at a 1% convergence band. One band
tighter it is -4.2%. That is not a footnote to hide -- it is the single most
likely thing a reviewer will find, and it has a clean answer: **viscous drag is
monotone across every band we can read and total drag is not**, while the
converged-field oracle is monotone on *both*. So the instability belongs to the
seed, not to the measurement, and `Cd_v` -- which is 60-84% of the drag here --
is the quantity that behaves like a convergence rate.

Monotone stability across bands *is* the evidence that a number measures a rate
rather than where a wandering curve happens to cross a line (``PLANS.md`` 3.3,
3.4). This figure is that argument, drawn.

Bands the readability rule rejects are drawn hollow and greyed: the rule says
the settled arms must agree to within half the band, and at 0.2% they do not.
They are shown rather than deleted, because a curve with its last point removed
would look more stable than the measurement is.

Usage
-----
    python scripts/plot_bands.py --depth results/depth_repr3_nowake.json
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
BLUE, GOOD, BAD = "#2a78d6", "#1baf7a", "#d92c2c"

BANDS = (0.01, 0.005, 0.002)
PANELS = [("Cd", "total drag  $C_d$"), ("Cd_v", "viscous drag  $C_{d,v}$")]
ARMS = [("nf_bl", "NeuroForge, mesh-native,\nboundary layer only", GOOD, "o"),
        ("oracle_mesh", "the exact converged field\n(control)", BLUE, "s")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", default=os.path.join("results",
                                                    "depth_repr3_nowake.json"))
    ap.add_argument("--out", default=os.path.join("results", "bands.png"))
    args = ap.parse_args(argv)

    if not os.path.isfile(args.depth):
        print(f"missing {args.depth} -- run scripts/reanalyse_depth.py first")
        return 1
    with open(args.depth, encoding="utf-8") as fh:
        blob = json.load(fh)
    depth = blob["by_force"]

    def cell(metric, band, arm):
        row = depth.get(f"{metric}@{band:g}")
        if row is None:
            return None, True
        entry = row.get(arm) or {}
        s = entry.get("saving")
        return (None if s is None else 100 * s), bool(row.get("readable", True))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(BANDS))

    for ax, (metric, nice) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_3)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.axhline(0, color=INK_3, lw=1, ls="--", zorder=1)

        for arm, label, colour, marker in ARMS:
            ys, readable = [], []
            for b in BANDS:
                v, ok = cell(metric, b, arm)
                ys.append(np.nan if v is None else v)
                readable.append(ok)
            ax.plot(x, ys, color=colour, lw=2.0, zorder=3,
                    label=label if metric == "Cd" else None)
            for xi, yi, ok in zip(x, ys, readable):
                if not np.isfinite(yi):
                    continue
                # Hollow = the readability rule rejected this band. Plotted, not
                # deleted: dropping it would flatter the curve.
                ax.scatter([xi], [yi], s=95, zorder=4, marker=marker,
                           color=colour if ok else SURFACE,
                           edgecolor=colour if ok else INK_3,
                           linewidth=1.6 if ok else 1.4)
                ax.annotate(f"{yi:+.1f}%", (xi, yi), fontsize=8.5,
                            color=INK if ok else INK_3,
                            xytext=(0, 11 if yi >= 0 else -15),
                            textcoords="offset points", ha="center")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{100 * b:g}%" for b in BANDS], fontsize=9.5,
                           color=INK_2)
        ax.set_xlim(-0.35, len(BANDS) - 0.65)
        ax.set_xlabel("convergence band (tighter $\\rightarrow$)",
                      fontsize=9.5, color=INK_2)
        ax.set_title(nice, fontsize=11, color=INK, loc="left", pad=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # Headroom for the value labels, which sit 11-15 points off each marker: the
    # first draft put -38.5% on top of the "0.2%" tick.
    everything = [v for metric, _ in PANELS for b in BANDS for arm, *_ in ARMS
                  if (v := cell(metric, b, arm)[0]) is not None]
    lo, hi = min(everything), max(everything)
    axes[0].set_ylim(lo - 0.16 * (hi - lo), hi + 0.14 * (hi - lo))

    axes[0].set_ylabel("iteration saving against a cold start (%)",
                       fontsize=9.5, color=INK_2)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8.5,
                   labelcolor=INK_2)
    axes[1].text(0.99, 0.02, "hollow marker: band rejected by the\n"
                             "readability rule (settled arms disagree\n"
                             "by more than half the band)",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=7.8, color=INK_3)

    n = (depth.get("Cd@0.01") or {}).get("cold_n")
    fig.suptitle("Viscous drag is a rate; total drag is a rate only at 1%"
                 f"    (Re 3$\\times$10$^6$, {n} cases)",
                 fontsize=11.5, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
