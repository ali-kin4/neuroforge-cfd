"""Figure: what a representation does to the near-wall state, and what that does not explain.

Three panels, in the order the argument runs.

**(A) The closed form bounds the measurement.** Predicted first-cell gradient
overestimate against the measured one, at five first-station heights spanning a
factor of fifty. Nothing is fitted. The identity line is drawn: every point sits
*above* it, which is the claim -- an upper bound, never optimistic. Two of the
five rows are drawn hollow: there the representation's first station sits inside
the mesh's first ring, `clustered_seed` populates it from that ring by
nearest-neighbour donor and maps it straight back, and the round trip is a
structural no-op. Those rows cannot exercise the clipping the closed form
models, so the bound is claimed over the three filled points only.

**(B) Placement decides it, budget does not.** The measured overestimate against
the first station's height, one line per value budget. The budget is cut 4x
(16,384 -> 4,096 values) and the three lines lie exactly on top of one another,
while moving the station along the axis moves the damage by 12.7x. Measured at
the live stations rather than only at the finest one: at `first = 5e-6` the
round trip is the no-op of panel A, so a budget test there compares two no-ops
and could not have shown an effect if one existed.

**(C) And none of it predicts the solve.** Measured gradient error against
convergence saving, one point per arm. The mesh-native seed and a projection
twenty-three times worse on the gradient land on the same viscous drag; the
repaired seeds move the gradient by a factor of twenty-three and convergence by
under a point. This panel is the paper's negative result, drawn.

Usage
-----
    python scripts/plot_placement.py --out results/placement.png
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
BLUE, GOOD, BAD, WARM = "#2a78d6", "#1baf7a", "#d92c2c", "#e08a1e"

# Arms of panel C: (label, values held, colour).
PANEL_C = {
    "nf_bl":          ("mesh-native", None, GOOD),
    "nf_proj":        ("projected", 256 * 64, BAD),
    "nf_proj_fix":    ("+ wall-law repair", 256 * 64, WARM),
    "nf_proj_smooth": ("+ repair, smoothed", 256 * 64, BLUE),
}


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def saving(depth, arm, key):
    entry = (depth.get("by_force") or {}).get(key) if depth else None
    value = (entry or {}).get(arm)
    if isinstance(value, dict):
        return value.get("saving")
    return value if isinstance(value, (int, float)) else None


def gradient_errors(path):
    """arm -> mean first-cell gradient overestimate, from a seed diagnostic."""
    data = load(path)
    if not data:
        return {}
    out: dict[str, list[float]] = {}
    for row in data.get("rows", []):
        ref = row.get("converged_mean_gradient")
        if not ref:
            continue
        for arm, entry in row.get("arms", {}).items():
            g = entry.get("mean_gradient")
            if g:
                out.setdefault(arm, []).append(g / ref)
    return {a: float(np.mean(v)) for a, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation",
                    default=os.path.join("results", "closed_form_validation.json"))
    ap.add_argument("--gradient-repair",
                    default=os.path.join("results", "seed_gradient_repair.json"))
    ap.add_argument("--depth-repair", default=os.path.join("results", "depth_repair.json"))
    ap.add_argument("--coeff", default="Cd_v")
    ap.add_argument("--band", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join("results", "placement.png"))
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val = load(args.validation)
    if not val:
        print(f"missing {args.validation}")
        return 1
    rows = sorted(val["rows"], key=lambda r: r["first"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_3)

    # ---------------------------------------------------------------- panel A
    ax = axes[0]
    pred = np.array([r["predicted"] for r in rows])
    meas = np.array([r["measured"] for r in rows])
    lim = [0.8 * min(pred.min(), meas.min()), 1.3 * max(pred.max(), meas.max())]
    ax.plot(lim, lim, color=INK_3, lw=1.0, ls="--", zorder=1, label="identity")
    live = [r for r in rows if not r.get("degenerate")]
    for r in rows:
        dead = r.get("degenerate")
        ax.scatter([r["measured"]], [r["predicted"]], s=70,
                   color="white" if dead else BLUE, edgecolor=INK_3 if dead else "white",
                   lw=1.1 if dead else 0.9, zorder=3)
        ax.annotate(f"  {r['first']:.0e}", (r["measured"], r["predicted"]),
                    fontsize=8, color=INK_3, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured overestimate", color=INK_2)
    ax.set_ylabel("predicted  $u^+(y_1^+)/u^+(y_c^+)$", color=INK_2)
    ax.set_title("A  the closed form bounds it, and never flatters",
                 color=INK, loc="left", fontsize=11, fontweight="bold")
    lp = np.array([r["predicted"] for r in live])
    lm = np.array([r["measured"] for r in live])
    ax.text(0.04, 0.96,
            f"over-predicts by {np.min(lp / lm):.1f}-{np.max(lp / lm):.1f}x\n"
            f"n = {len(live)} live rows; hollow = round\ntrip is a no-op there",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="top")

    # ---------------------------------------------------------------- panel B
    ax = axes[1]
    budgets = sorted((rows[0].get("values_by_n_n") or {}).items(),
                     key=lambda kv: -kv[1])
    styles = [(BLUE, "o", 8.0, 1.8), (WARM, "s", 6.0, 1.2), (GOOD, "^", 4.5, 0.9)]
    for (key, values), (colour, marker, size, lw) in zip(budgets, styles):
        first = [r["first"] for r in rows]
        meas = [r["measured_by_n_n"][key] for r in rows]
        ax.plot(first, meas, marker=marker, ms=size, lw=lw, color=colour,
                mec="white", mew=0.8, zorder=3, label=f"{values:,} values")
    ax.axhline(1.0, color=INK_3, lw=1.0, ls=":", zorder=2)
    ax.text(rows[0]["first"], 1.06, "  gradient preserved", color=INK_3, fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("first station of the representation  [chord]", color=INK_2)
    ax.set_ylabel("measured gradient overestimate", color=INK_2)
    ax.set_title("B  placement decides; budget does not",
                 color=INK, loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_2)
    if budgets:
        ratio = max(v for _, v in budgets) / min(v for _, v in budgets)
        ax.text(0.97, 0.06, f"budget cut {ratio:.0f}x; curves coincide",
                transform=ax.transAxes, fontsize=8.5, color=INK_2, ha="right")

    # ---------------------------------------------------------------- panel C
    ax = axes[2]
    grads = gradient_errors(args.gradient_repair)
    depth = load(args.depth_repair)
    key = f"{args.coeff}@{args.band:g}"
    points = []
    for arm, (label, values, colour) in PANEL_C.items():
        g, sv = grads.get(arm), saving(depth, arm, key)
        if g is not None and sv is not None:
            points.append((g, 100 * sv, label, colour))
    # Label above or below alternately, ordered along x, so the two repair arms
    # -- which land on top of each other by design -- stay readable.
    points.sort()
    for k, (g, y, label, colour) in enumerate(points):
        ax.scatter([g], [y], s=110, color=colour, edgecolor="white", lw=0.9, zorder=3)
        ax.annotate(label, (g, y), xytext=(0, 13 if k % 2 else -20),
                    textcoords="offset points", fontsize=9, color=INK_2,
                    ha="center")
    plotted = len(points)
    ax.axhline(0, color=INK_3, lw=1.0, zorder=2)
    ax.set_xscale("log")
    ax.margins(x=0.28, y=0.22)
    if points:
        span = max(p[0] for p in points) / min(p[0] for p in points)
        ax.text(0.5, 0.04, f"{span:.0f}x in gradient error, "
                           f"{max(p[1] for p in points) - min(p[1] for p in points):.1f} "
                           "points in convergence",
                transform=ax.transAxes, fontsize=8.5, color=INK_2, ha="center")
    ax.set_xlabel("measured first-cell gradient overestimate", color=INK_2)
    ax.set_ylabel(f"{args.coeff} saving at {100 * args.band:g}% band  [%]", color=INK_2)
    ax.set_title("C  and it does not predict the solve", color=INK, loc="left",
                 fontsize=11, fontweight="bold")
    if not plotted:
        ax.text(0.5, 0.5, "repair tree not scored yet", transform=ax.transAxes,
                ha="center", color=INK_3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}  (A n={len(rows)}, C n={plotted})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
