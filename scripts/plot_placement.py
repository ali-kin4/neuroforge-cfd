"""Figure: what a representation does to the near-wall state, and what that does not explain.

Three panels, in the order the argument runs.

**(A) The closed form bounds the measurement.** Predicted first-cell gradient
overestimate against the measured one, at five first-station heights spanning a
factor of fifty. Nothing is fitted. The identity line is drawn: every point sits
*above* it, which is the claim -- an upper bound, never optimistic.

**(B) Placement decides it, budget does not.** The same measured overestimate
against the first station's `y+`, with marker area proportional to the number of
values the representation holds. The panel makes its own argument: marker area
varies by 32x while the points fall on a curve that depends only on where the
first station sits.

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
    ax.scatter(meas, pred, s=70, color=BLUE, edgecolor="white", lw=0.9, zorder=3)
    for r in rows:
        ax.annotate(f"  {r['first']:.0e}", (r["measured"], r["predicted"]),
                    fontsize=8, color=INK_3, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured overestimate", color=INK_2)
    ax.set_ylabel("predicted  $u^+(y_1^+)/u^+(y_c^+)$", color=INK_2)
    ax.set_title("A  the closed form bounds it, and never flatters",
                 color=INK, loc="left", fontsize=11, fontweight="bold")
    ax.text(0.05, 0.93, "every point above identity\n"
                        f"ratio {np.mean(pred / meas):.2f}, n = {len(rows)}",
            transform=ax.transAxes, fontsize=9, color=INK_2, va="top")

    # ---------------------------------------------------------------- panel B
    ax = axes[1]
    for r in rows:
        first = r["first"]
        values = 256 * 32 if abs(first - 5e-6) < 1e-12 else 256 * 64
        ax.scatter([first], [r["measured"]], s=24 + 300 * values / 16384.0,
                   color=BLUE, alpha=0.8, edgecolor="white", lw=0.9, zorder=3)
    ax.axhline(1.0, color=GOOD, lw=1.2, ls=":", zorder=2)
    ax.text(rows[0]["first"], 1.05, "  gradient preserved", color=GOOD, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("first station of the representation  [chord]", color=INK_2)
    ax.set_ylabel("measured gradient overestimate", color=INK_2)
    ax.set_title("B  placement decides; marker area is the value budget",
                 color=INK, loc="left", fontsize=11, fontweight="bold")

    # ---------------------------------------------------------------- panel C
    ax = axes[2]
    grads = gradient_errors(args.gradient_repair)
    depth = load(args.depth_repair)
    key = f"{args.coeff}@{args.band:g}"
    plotted = 0
    for arm, (label, values, colour) in PANEL_C.items():
        g, sv = grads.get(arm), saving(depth, arm, key)
        if g is None or sv is None:
            continue
        ax.scatter([g], [100 * sv], s=110, color=colour, edgecolor="white",
                   lw=0.9, zorder=3)
        ax.annotate("  " + label, (g, 100 * sv), fontsize=9, color=INK_2,
                    va="center")
        plotted += 1
    ax.axhline(0, color=INK_3, lw=1.0, zorder=2)
    ax.set_xscale("log")
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
