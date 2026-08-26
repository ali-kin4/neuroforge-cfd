"""Plot the warm-start crossover: iteration saving against delta / h.

Reads ``results/reynolds_crossover.json``. The x axis is the ratio that controls
the effect -- boundary-layer thickness over surrogate cell size -- rather than
Reynolds number, so the curve reads as a criterion that transfers to other
resolutions instead of a fact about the five Reynolds numbers measured.
"""

from __future__ import annotations

import argparse
import json
import os

# Must precede numpy/torch: see the threading note in neuroforge/__init__.py.
import neuroforge  # noqa: F401
import numpy as np

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#85837c"
GRID = "#e2e0da"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BAD = "#d92c2c"
SURFACE = "#fcfcfb"

ARMS = [
    ("oracle_mesh", BLUE, "exact, at mesh resolution", "the control"),
    ("oracle_128", ORANGE, "exact, through the 128² grid", "what a surrogate could at best do"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join("results", "reynolds_crossover.json"))
    ap.add_argument("--out", default=os.path.join("results", "reynolds_crossover.png"))
    ap.add_argument("--threshold", default="1e-03")
    args = ap.parse_args(argv)

    with open(args.data, encoding="utf-8") as fh:
        blob = json.load(fh)
    by_re = (blob.get("summary") or {}).get("by_reynolds") or {}
    if not by_re:
        print(f"{args.data} has no summary yet -- is the sweep still running?")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted(((v["delta_over_h"], re, v) for re, v in by_re.items()))
    x = np.array([p[0] for p in pts])
    labels = [p[1] for p in pts]
    k = args.threshold

    fig, ax = plt.subplots(figsize=(9.6, 5.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhspan(-320, 0, color=BAD, alpha=0.055, zorder=0)
    ax.axhline(0, color=INK_3, lw=1.2, zorder=2)
    ax.text(x.min(), -6, "below this line a warm start is WORSE than starting cold",
            color=BAD, fontsize=9, va="top", fontweight="bold")
    ax.axvline(1.0, color=INK_3, lw=1.0, ls=(0, (5, 4)), zorder=2)
    ax.text(1.0, 96, "  layer thinner\n  than one cell", color=INK_2, fontsize=9,
            va="top", ha="left")

    for key, color, label, note in ARMS:
        ys, xs = [], []
        for xv, _re, v in pts:
            s = v.get(f"{key}@{k}")
            if s is not None:
                xs.append(xv); ys.append(100 * s)
        if not xs:
            continue
        ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=7, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.6, label=label)
        ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9.5, fontweight="bold", va="center")

    for xv, re, _v in pts:
        ax.annotate(f"Re {float(re):.0e}".replace("e+0", "e"), xy=(xv, 0), xytext=(0, -20),
                    textcoords="offset points", ha="center", color=INK_3, fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{v:.2f}" for v in x])
    ax.minorticks_off()
    ax.set_xlabel("δ / h   —   boundary-layer thickness ÷ surrogate cell size",
                  color=INK_2, fontsize=10.5)
    ax.set_ylabel(f"iteration saving vs a cold start  [%]  (residual {k})",
                  color=INK_2, fontsize=10.5)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_xlim(x.min() * 0.82, x.max() * 1.55)

    ax.set_title("Where a 128² warm start stops paying",
                 color=INK, fontsize=14, fontweight="bold", loc="left", pad=16)
    fig.text(0.008, 0.925,
             "Same C-grid, same schemes, same budget at every point. The arm under test is the "
             "exact answer\ndegraded only in resolution — so its saving is an upper bound on any "
             "surrogate trained on that grid.",
             color=INK_2, fontsize=9.5, ha="left", va="top")
    fig.subplots_adjust(top=0.78, left=0.085, right=0.985, bottom=0.16)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
