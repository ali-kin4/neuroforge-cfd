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
GRID = "#e4e2dc"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, BAD = "#1baf7a", "#d92c2c"
SURFACE = "#fcfcfb"

ARMS = [
    ("oracle_mesh", BLUE, "exact, at mesh resolution"),
    ("oracle_128", ORANGE, "exact, through the 128² grid"),
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

    k = args.threshold
    pts = sorted(((v["delta_over_h"], re, v) for re, v in by_re.items()))
    x = np.array([p[0] for p in pts])

    fig, ax = plt.subplots(figsize=(10.2, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    curves = {}
    for key, color, label in ARMS:
        xs = [xv for xv, _r, v in pts if v.get(f"{key}@{k}") is not None]
        ys = [100 * v[f"{key}@{k}"] for _x, _r, v in pts if v.get(f"{key}@{k}") is not None]
        curves[key] = (xs, ys)

    lo = min(min(y) for _x, y in curves.values())
    hi = max(max(y) for _x, y in curves.values())
    pad = 0.16 * (hi - lo)
    ymin, ymax = lo - pad, hi + pad

    # Where the arm under test changes sign -- the answer the figure exists for.
    cx, cy = curves["oracle_128"]
    cross = None
    for i in range(len(cx) - 1):
        if (cy[i] < 0) != (cy[i + 1] < 0):
            t = -cy[i] / (cy[i + 1] - cy[i])
            cross = cx[i] * (cx[i + 1] / cx[i]) ** t
    if cross:
        ax.axvspan(cross, x.max() * 2, color=GOOD, alpha=0.055, zorder=0)
        ax.axvline(cross, color=GOOD, lw=1.5, zorder=3)
        # Mid-height, in the band the two curves leave empty; at the top it
        # collides with the mesh arm's value label.
        ax.text(cross * 1.05, ymin + 0.55 * (ymax - ymin),
                f"crossover\nδ/h ≈ {cross:.1f}", color=GOOD, fontsize=11,
                fontweight="bold", va="center", ha="left",
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=3))

    ax.axhline(0, color=INK_3, lw=1.3, zorder=2)
    ax.text(x.min() * 0.86, -3, "worse than a cold start", color=BAD, fontsize=9.5,
            va="top", ha="left", style="italic")

    for key, color, label in ARMS:
        xs, ys = curves[key]
        ax.plot(xs, ys, "-o", color=color, lw=2.4, ms=8, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=1.8)
        for xv, yv in zip(xs, ys):
            ax.annotate(f"{yv:+.0f}%", xy=(xv, yv), xytext=(0, -15 if yv < 0 else 11),
                        textcoords="offset points", ha="center", color=color,
                        fontsize=9, fontweight="bold")

    # A legend in the empty mid-left region: end-of-line labels collide with the
    # per-point value labels, and the value labels are the ones worth keeping.
    for i, (key, color, label) in enumerate(ARMS):
        ax.annotate(label, xy=(0.028, 0.60 - i * 0.075), xycoords="axes fraction",
                    color=color, fontsize=10.5, fontweight="bold", va="center",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2.5))
        ax.annotate("", xy=(0.021, 0.60 - i * 0.075), xycoords="axes fraction",
                    xytext=(-11, 0), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=color, lw=3.2))

    for xv, re, _v in pts:
        ax.annotate(f"Re {float(re):.0e}".replace("e+0", "e"), xy=(xv, ymin),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    color=INK_3, fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{v:.2f}" for v in x])
    ax.minorticks_off()
    ax.set_xlabel("δ / h    boundary-layer thickness ÷ surrogate cell size",
                  color=INK_2, fontsize=11)
    ax.set_ylabel(f"iteration saving vs a cold start  [%]", color=INK_2, fontsize=11)
    ax.tick_params(colors=INK_2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_xlim(x.min() * 0.80, x.max() * 1.28)
    ax.set_ylim(ymin, ymax)

    fig.suptitle("A surrogate warm start pays only while the boundary layer spans ~2 cells",
                 color=INK, fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.905,
             f"Same C-grid, schemes and budget at every point; saving measured at residual {k}. "
             f"The arm under test is the exact\nanswer degraded only in resolution, so its saving "
             f"is an upper bound on any surrogate trained on that grid.",
             color=INK_2, fontsize=10, ha="left", va="top")
    fig.subplots_adjust(top=0.80, left=0.075, right=0.985, bottom=0.115)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")
    if cross:
        print(f"crossover at delta/h = {cross:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
