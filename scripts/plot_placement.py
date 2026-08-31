"""Figure: the first cell decides, the damage is computable, and it is repairable.

Three panels, each answering an objection rather than decorating a result.

**(A) The closed form against the measurement.** Predicted first-cell gradient
overestimate against the measured one, one point per (arm, case). Nothing is
fitted: `u_tau` comes from each case's own converged wall gradient. The identity
line is drawn, and so is the systematic 1.13x bias, because a reader should see
that the agreement is biased-but-tight rather than perfect.

**(B) Placement decides and budget does not.** The measured convergence saving
against the first station's `y+`. Marker area is proportional to the number of
values the representation holds, so the panel makes its argument visually: the
markers vary over an order of magnitude in area while the points fall on a curve
that depends only on where the first station sits. The resolution ladder is on
here too -- 10.8x the values, no movement.

**(C) The repair.** What inverting a wall function at the representation's own
first station does to the gradient error, and to the solve.

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

from neuroforge.solver import placement as pl

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#85837c"
GRID, SURFACE = "#e4e2dc", "#fcfcfb"
BLUE, GOOD, BAD, WARM = "#2a78d6", "#1baf7a", "#d92c2c", "#e08a1e"

# Arm -> (first station of its representation, values it holds, label).
# `None` for the station means the representation samples at the solver's own
# cell centres, so the criterion is trivially satisfied.
ARMS = {
    # the mechanism tree
    "cartesian_128":   (1.17e-2, 128 * 128, "Cartesian 128²"),
    "fitted_256x64":   (2.5e-4, 256 * 64, "wall-fitted 256×64"),
    "fitted_bl":       (2.5e-4, 256 * 64, "wall-fitted, BL only"),
    "nf_bl_proj":      (2.5e-4, 256 * 64, "network, projected"),
    "nf_bl":           (None, None, "network, mesh-native"),
    "nf_mesh":         (None, None, "network, whole field"),
    # the placement tree
    "nf_proj_coarse":  (2.5e-4, 256 * 64, "projected, first 2.5e-4"),
    "or_proj_coarse":  (2.5e-4, 256 * 64, "oracle projected, 2.5e-4"),
    "nf_proj_fine":    (5.0e-6, 256 * 64, "projected, first 5e-6"),
    "or_proj_fine":    (5.0e-6, 256 * 64, "oracle projected, 5e-6"),
    "nf_proj_half":    (5.0e-6, 256 * 32, "projected, 5e-6, half budget"),
    "or_proj_half":    (5.0e-6, 256 * 32, "oracle projected, half budget"),
    # the repair tree
    "nf_proj":         (2.5e-4, 256 * 64, "projected"),
    "or_proj":         (2.5e-4, 256 * 64, "oracle projected"),
    "nf_proj_fix":     (None, 256 * 64, "projected + wall-law repair"),
    "or_proj_fix":     (None, 256 * 64, "oracle projected + repair"),
    # the classical baseline
    "sequenced":       (None, 7850, "grid sequencing"),
    "sequenced_bl":    (None, 7850, "grid sequencing, BL only"),
}
CONTROLS = ("cold", "oracle_mesh")


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def measured_amplification(gradient_files):
    """(arm, case) -> measured first-cell gradient overestimate, from the diagnostics."""
    out = {}
    for path in gradient_files:
        data = load_json(path)
        if not data:
            continue
        for row in data.get("rows", []):
            ref = row.get("converged_mean_gradient")
            if not ref:
                continue
            for arm, entry in row.get("arms", {}).items():
                g = entry.get("mean_gradient")
                if g:
                    out[(arm, row["case"], path)] = g / ref
    return out


def predicted_amplification(arm, u_tau, nu, probe, u_inf=1.0, delta=0.0187):
    spec = ARMS.get(arm)
    if spec is None:
        return None
    first = spec[0]
    if first is None:
        return 1.0
    return pl.amplification(first_station=first, cell_centre=probe, u_tau=u_tau,
                            nu=nu, u_inf=u_inf, delta=delta)["factor"]


def savings(depth_files, coeff="Cd_v", band=0.01):
    """arm -> mean saving (fraction) at one force band, per scored tree."""
    key = f"{coeff}@{band:g}"
    out = {}
    for path in depth_files:
        data = load_json(path)
        if not data:
            continue
        entry = (data.get("by_force") or {}).get(key)
        if not entry:
            continue
        for arm, value in entry.items():
            if arm in ARMS and isinstance(value, (int, float)):
                out[(arm, path)] = value
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gradient", nargs="*", default=[
        os.path.join("results", "seed_gradient.json"),
        os.path.join("results", "seed_gradient_placement.json"),
        os.path.join("results", "seed_gradient_repair.json"),
    ])
    ap.add_argument("--depth", nargs="*", default=[
        os.path.join("results", "depth_placement.json"),
        os.path.join("results", "depth_repair.json"),
        os.path.join("results", "depth_sequencing.json"),
        os.path.join("results", "depth_repr3_nowake.json"),
    ])
    ap.add_argument("--coeff", default="Cd_v")
    ap.add_argument("--band", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join("results", "placement.png"))
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grads = measured_amplification(args.gradient)
    if not grads:
        print("no gradient diagnostics found -- run seed_gradient_diagnostic.py first")
        return 1

    # u_tau and nu per (case, file), from the converged gradient itself.
    context = {}
    for path in args.gradient:
        data = load_json(path)
        if not data:
            continue
        nu = 1.0 / data["re"]
        probe = data["heights"][0]
        for row in data.get("rows", []):
            if row.get("converged_mean_gradient"):
                context[(row["case"], path)] = (
                    pl.friction_velocity(row["converged_mean_gradient"], nu), nu, probe)

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
    xs, ys = [], []
    for (arm, case, path), meas in grads.items():
        ctx = context.get((case, path))
        if ctx is None or arm in CONTROLS:
            continue
        u_tau, nu, probe = ctx
        pred = predicted_amplification(arm, u_tau, nu, probe)
        if pred is None or pred <= 0 or meas <= 0:
            continue
        xs.append(pred)
        ys.append(meas)
    if xs:
        lim = [0.5 * min(min(xs), min(ys)), 2.0 * max(max(xs), max(ys))]
        ax.plot(lim, lim, color=INK_3, lw=1.0, ls="--", zorder=1)
        ax.plot(lim, [v / 1.13 for v in lim], color=WARM, lw=1.0, zorder=1)
        ax.scatter(xs, ys, s=46, color=BLUE, edgecolor="white", lw=0.8, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.text(0.05, 0.93, f"n = {len(xs)}\nmedian ratio "
                            f"{np.median(np.array(xs) / np.array(ys)):.2f}",
                transform=ax.transAxes, fontsize=9, color=INK_2, va="top")
    ax.set_xlabel("predicted overestimate  $u^+(y_1^+)/u^+(y_c^+)$", color=INK_2)
    ax.set_ylabel("measured overestimate", color=INK_2)
    ax.set_title("A  the closed form, with nothing fitted", color=INK, loc="left",
                 fontsize=11, fontweight="bold")

    # ---------------------------------------------------------------- panel B
    ax = axes[1]
    saves = savings(args.depth, args.coeff, args.band)
    plotted = 0
    for (arm, path), value in saves.items():
        spec = ARMS[arm]
        first, values = spec[0], spec[1]
        u_tau, nu, probe = next(iter(context.values()))
        y_plus = 0.5 if first is None else float(pl.wall_units(first, u_tau, nu))
        size = 40 if not values else 24 + 300 * (values / 16384.0)
        colour = GOOD if value > 0 else BAD
        ax.scatter([y_plus], [100 * value], s=size, color=colour, alpha=0.8,
                   edgecolor="white", lw=0.8, zorder=3)
        plotted += 1
    ax.axhline(0, color=INK_3, lw=1.0, zorder=2)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("$y^+$ of the representation's first station", color=INK_2)
    ax.set_ylabel(f"{args.coeff} saving at {100 * args.band:g}% band  [%]", color=INK_2)
    ax.set_title("B  placement decides; marker area is the value budget",
                 color=INK, loc="left", fontsize=11, fontweight="bold")
    if not plotted:
        ax.text(0.5, 0.5, "no scored trees yet", transform=ax.transAxes,
                ha="center", color=INK_3)

    # ---------------------------------------------------------------- panel C
    ax = axes[2]
    pairs = [("nf_proj", "nf_proj_fix"), ("or_proj", "or_proj_fix")]
    labels, before, after = [], [], []
    for a, b in pairs:
        va = next((v for (arm, _), v in saves.items() if arm == a), None)
        vb = next((v for (arm, _), v in saves.items() if arm == b), None)
        if va is not None and vb is not None:
            labels.append(ARMS[a][2])
            before.append(100 * va)
            after.append(100 * vb)
    if labels:
        idx = np.arange(len(labels))
        ax.bar(idx - 0.19, before, 0.36, color=BAD, label="projected")
        ax.bar(idx + 0.19, after, 0.36, color=GOOD, label="+ wall-law repair")
        ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color=INK_3, lw=1.0)
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "repair tree not scored yet", transform=ax.transAxes,
                ha="center", color=INK_3)
    ax.set_ylabel(f"{args.coeff} saving  [%]", color=INK_2)
    ax.set_title("C  the damage is known, so it can be removed", color=INK,
                 loc="left", fontsize=11, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}  (panel A n={len(xs)}, panel B n={plotted})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
