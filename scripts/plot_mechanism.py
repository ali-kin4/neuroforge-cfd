"""The paper's central figure: what each property of a warm start is worth.

Two panels, one argument, every number from ``results/decomposition*.json``
(written by ``scripts/decompose.py``).

**A. The levels.** Convergence saving on ``Cd_v@1%`` for five ways of handing the
solver a field, with paired bootstrap intervals. The pair the eye should land on
is the exact converged field read at the solver's own cell centres against the
*same* field stored on a 128^2 raster: 93.6% against 3.4%, an interval spanning
zero. The body-fitted grid of identical budget sits with the mesh-native arms,
which is what makes this a statement about placement rather than about grids.

**B. The contrasts.** The same data as differences, each moving exactly one
property, paired within case before averaging. A raster representation costs
almost everything, accuracy costs more than region, and a body-fitted
representation costs nothing measurable -- drawn hollow and labelled, because its
interval spans zero and a null must not be coloured like an effect.

This replaced a figure whose middle panel was captioned "keep the gradient, keep
the saving" -- a causal chain sections 5.2.1, 5.5 and 6.7 refute -- and whose
gradient numbers came from a diagnostic run before the ``clustered_seed``
wall-distance defect was fixed.

Usage
-----
    python scripts/plot_mechanism.py
    python scripts/plot_mechanism.py --decomposition results/decomposition_n5.json
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
GOOD, BAD, WARM = "#1baf7a", "#d92c2c", "#e08a1e"

# arm -> (label, sub-label, colour)
LEVELS = [
    ("oracle_mesh", "the exact answer itself", "whole field, cell centres", BLUE),
    ("or_proj_coarse", "through a 256$\\times$64 body-fitted grid",
     "16,384 values, boundary layer", GOOD),
    ("oracle_bl", "the exact answer, boundary layer only",
     "whole physics, cell centres", GOOD),
    ("nf_bl", "the surrogate's prediction", "boundary layer, cell centres", WARM),
    ("cartesian_128", "through a 128$^2$ Cartesian raster",
     "16,384 values, whole field", BAD),
]

CONTRASTS = [
    ("representation (raster)", BAD),
    ("accuracy", WARM),
    ("region", ORANGE),
    ("representation (body-fitted)", GOOD),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decomposition",
                    default=os.path.join("results", "decomposition.json"))
    ap.add_argument("--fallback",
                    default=os.path.join("results", "decomposition_n5.json"),
                    help="merged in for any arm or contrast the main file lacks")
    ap.add_argument("--out", default=os.path.join("results", "mechanism.png"))
    args = ap.parse_args(argv)

    def load(path):
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    main_data, fall = load(args.decomposition), load(args.fallback)
    if main_data is None and fall is None:
        print(f"missing {args.decomposition} -- run scripts/decompose.py first")
        return 1
    if main_data is None:
        main_data, fall = fall, None

    def pick(section, key):
        """Prefer the main file; fall back, and say which was used."""
        got = (main_data.get(section) or {}).get(key)
        if got:
            return got, main_data["n_cases"]
        if fall:
            got = (fall.get(section) or {}).get(key)
            if got:
                return got, fall["n_cases"]
        return None, None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_3)

    # ---------------------------------------------------------------- panel A
    ax = axes[0]
    ax.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    rows, y = [], 0
    for arm, label, sub, colour in LEVELS:
        entry, n = pick("levels", arm)
        if entry is None:
            continue
        rows.append((y, entry, label, sub, colour, n))
        y += 1
    for y, e, label, sub, colour, n in rows:
        m = 100 * e["mean"]
        lo, hi = 100 * e["ci"][0], 100 * e["ci"][1]
        ax.barh(y, m, height=0.62, color=colour, zorder=3)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.4, zorder=4,
                solid_capstyle="butt")
        ax.text(m + (2.0 if m >= 0 else -2.0), y,
                f"{m:+.1f}%   {e['wins']}/{e['n']}",
                va="center", ha="left" if m >= 0 else "right",
                fontsize=9.5, color=INK, fontweight="bold", zorder=5)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([f"{r[2]}\n{r[3]}" for r in rows], fontsize=9, color=INK_2)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_3, lw=1.0, zorder=2)
    ax.set_xlabel("convergence saving on  $C_{d,v}$@1%   [%]", color=INK_2)
    ax.set_title("A  the same field, five ways of handing it over",
                 color=INK, loc="left", fontsize=11.5, fontweight="bold")
    ax.set_xlim(min(-12, min(100 * r[1]["ci"][0] for r in rows) - 10), 118)

    # ---------------------------------------------------------------- panel B
    ax = axes[1]
    ax.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    bars = []
    for k, (name, colour) in enumerate(CONTRASTS):
        entry, n = pick("contrasts", name)
        if entry is None:
            continue
        bars.append((len(bars), entry, name, colour))
    for y, e, name, colour in bars:
        m = 100 * e["mean"]
        lo, hi = 100 * e["ci"][0], 100 * e["ci"][1]
        # A contrast whose interval spans zero is a null, and must not be
        # coloured like an effect. Drawn hollow, in ink rather than in the
        # dimension's colour, and labelled as such.
        null = lo <= 0.0 <= hi
        ax.barh(y, m, height=0.58, color="white" if null else colour,
                edgecolor=INK_3 if null else "none",
                hatch="///" if null else None, lw=1.0, zorder=3)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.4, zorder=4,
                solid_capstyle="butt")
        label = f"{m:+.1f}" + ("   null (CI spans 0)" if null else "")
        ax.text(m + (1.6 if m >= 0 else -1.6), y, label,
                va="center", ha="left" if m >= 0 else "right",
                fontsize=10, color=INK_2 if null else INK,
                fontweight="normal" if null else "bold", zorder=5)
    ax.set_yticks([b[0] for b in bars])
    ax.set_yticklabels([b[2].replace(" (", "\n(") for b in bars], fontsize=9.5,
                       color=INK_2)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_3, lw=1.0, zorder=2)
    ax.set_xlabel("points of  $C_{d,v}$@1%  saving, paired within case", color=INK_2)
    ax.set_title("B  each bar moves one variable",
                 color=INK, loc="left", fontsize=11.5, fontweight="bold")
    span = max(abs(100 * b[1]["mean"]) for b in bars) if bars else 100
    ax.set_xlim(-1.35 * span, 0.75 * span)

    n_main = main_data["n_cases"]
    note = (f"$C_{{d,v}}$@1%, the row the readability rule of section 4 admits. "
            f"Bars are means over {n_main} cases with paired 95% bootstrap "
            f"intervals; panel A also gives cases won.")
    if fall and any(pick(sec, k)[1] != n_main
                    for sec, keys in (("levels", [a for a, *_ in LEVELS]),
                                      ("contrasts", [c for c, _ in CONTRASTS]))
                    for k in keys if pick(sec, k)[0]):
        note += (f"  Arms absent from the {n_main}-case corpus are shown from the "
                 f"{fall['n_cases']}-case mechanism tree.")
    fig.text(0.008, 0.015, note, fontsize=8.2, color=INK_3)

    fig.suptitle("What a warm start is worth, decomposed into properties that "
                 "can be set independently",
                 fontsize=13, color=INK, y=0.985, x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0.045, 1, 0.945])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}  (A n={len(rows)} arms, B n={len(bars)} contrasts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
