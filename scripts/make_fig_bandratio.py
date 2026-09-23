"""Generate the band-ratio figure for the interpolation-vs-Transolver localisation
thesis (paper's D2/P2 result) -- the one figure the trust-layer removal left the
manuscript without.

Every number plotted comes straight from the two committed pre-registered
artifacts; nothing is retyped:

  * results/interpolation/measure_asymmetry.json
      -- ``verdicts.D2.r_band_interp_over_transolver.u``: the 7-band ratio of
         interpolation error to Transolver error, measured on the r128 raster
         (area-uniform cells). No surface/wall row exists on this measure --
         a raster cannot represent sdf=0.
      -- ``B_band_decomposition_7.interp[band].cell_frac`` and
         ``D_node_space.n_nodes_in_crop`` (normalised): the cell-fraction /
         node-fraction mismatch behind ``tab:bandratio``'s own two weight
         columns.
  * results/interpolation/point_space_headtohead.json
      -- ``P2.r_b_u``: the same 7-band ratio, re-measured directly on the
         35.8M native AirfRANS cloud nodes, no raster anywhere.
      -- ``tables.interp_nearfill.full.u.band[0]`` /
         ``tables.transolver_mean.full.u.band[0]``: the wall (sdf=0) row that
         only the native measure can see (201,444 surface nodes).

Both readers are copied field-for-field from scripts/audit_paper_numbers.py's
``ma_band_ratio`` / ``ps_band_u`` / ``ps_wall_ratio`` helpers, and this script
cross-checks the tables-based wall/band reading against P2 at runtime so the
figure cannot silently drift from what the audit checks.

Convention note (deliberate, matches tab:bandratio and the audit script): the
grid ladder reads the ``B_band_decomposition_7`` / ``D_node_space`` (in-crop)
tree; the native ladder reads ``P2`` for the 7 non-wall bands and the
``tables[...]/tables[...]`` division (P1's selected "nearfill" arm) for the
wall row. Two different fields inside one artifact family, used the same way
the manuscript's own audit reads them.

Run:
    .venv/Scripts/python.exe scripts/make_fig_bandratio.py

CPU-only, plotting only. Reads results/interpolation/*.json, writes
results/figures/fig_bandratio.{pdf,png}.
"""

# --- thread caps: import neuroforge FIRST (sets OPENBLAS/OMP/MKL=1) ----------
import matplotlib

import neuroforge  # noqa: F401  (side effect: caps BLAS threads before numpy)

matplotlib.use("Agg")  # non-interactive backend

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

MA_PATH = os.path.join(RES, "interpolation", "measure_asymmetry.json")
PS_PATH = os.path.join(RES, "interpolation", "point_space_headtohead.json")

BANDS7 = ["0-0.005c", "0.005-0.01c", "0.01-0.02c", "0.02-0.05c",
          "0.05-0.15c", "0.15-0.5c", ">0.5c"]
BAND_LABELS7 = ["0-\n0.005c", "0.005-\n0.01c", "0.01-\n0.02c", "0.02-\n0.05c",
                "0.05-\n0.15c", "0.15-\n0.5c", ">0.5c"]

# Okabe-Ito colorblind-safe palette (matches scripts/make_figures.py).
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,   # embed as real (Type 1/TrueType-compatible) fonts,
    "ps.fonttype": 42,    # not Type 3 -- some journal toolchains reject Type 3.
})


def load(path):
    with open(path) as f:
        return json.load(f)


# --- readers, mirroring scripts/audit_paper_numbers.py exactly --------------
def grid_ratio_u(ma):
    """D2: interp-MSE / Transolver-MSE per band, area-uniform raster, channel u."""
    vals = ma["verdicts"]["D2"]["r_band_interp_over_transolver"]["u"]
    assert len(vals) == 7
    return np.asarray(vals, dtype=float)


def grid_verdict(ma):
    return ma["verdicts"]["D2"]["verdict"]


def native_ratio_u_bands(ps):
    """P2: same ratio, native cloud, 7 non-wall bands."""
    assert ps["P2"]["bands"] == BANDS7
    return np.asarray(ps["P2"]["r_b_u"], dtype=float)


def native_wall_ratio_u(ps):
    """P1's selected 'nearfill' arm, wall (sdf=0) row -- no raster equivalent."""
    num = ps["tables"]["interp_nearfill"]["full"]["u"]["band"][0]
    den = ps["tables"]["transolver_mean"]["full"]["u"]["band"][0]
    return float(num) / float(den)


def native_verdict(ps):
    return ps["P2"]["verdict"]


def cell_node_fracs(ma):
    """The tab:bandratio weight columns: cell_frac (raster) vs node_frac
    (native nodes falling in the same band, in-crop convention -- the same
    n_nodes_in_crop the D_node_space block reports)."""
    dec = ma["B_band_decomposition_7"]["interp"]
    cell_frac = np.asarray([dec[b]["cell_frac"] for b in BANDS7], dtype=float)
    n_nodes = np.asarray(ma["D_node_space"]["n_nodes_in_crop"], dtype=float)
    assert ma["D_node_space"]["bands"] == BANDS7
    node_frac = n_nodes / n_nodes.sum()
    return cell_frac, node_frac


def cross_check(ma, ps):
    """The two ways of reading the native 7-band ratio (P2 directly, vs dividing
    the pooled per-band 'tables' -- the arm the figure's wall row also reads)
    must agree to within float noise, or the figure and the audit script have
    silently started reading different things."""
    a = native_ratio_u_bands(ps)
    b = np.asarray([
        ps["tables"]["interp_nearfill"]["full"]["u"]["band"][i + 1]
        / ps["tables"]["transolver_mean"]["full"]["u"]["band"][i + 1]
        for i in range(7)
    ])
    rel = np.abs(a - b) / b
    assert rel.max() < 1e-2, f"P2 vs tables mismatch: {rel.max():.4f}"


def make_fig(ma, ps):
    cross_check(ma, ps)

    g = grid_ratio_u(ma)                       # 7 values, x = 1..7
    n7 = native_ratio_u_bands(ps)               # 7 values, x = 1..7
    n_wall = native_wall_ratio_u(ps)            # 1 value,  x = 0
    cell_frac, node_frac = cell_node_fracs(ma)

    x_grid = np.arange(1, 8)
    x_native = np.arange(0, 8)
    n_native_full = np.concatenate([[n_wall], n7])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08},
    )

    # --- Panel A: the ratio ladder --------------------------------------
    ax1.set_yscale("log")
    ax1.axhline(1.0, color=CB["black"], lw=0.9, zorder=1)

    ax1.plot(x_native, n_native_full, color=CB["vermillion"], marker="s",
              ms=5.5, mfc="white", mew=1.4, ls="--", lw=1.6, zorder=3,
              label="native cloud nodes")
    ax1.plot(x_grid, g, color=CB["blue"], marker="o", ms=5.5, ls="-",
              lw=1.8, zorder=4, label=r"$r128$ raster")

    # Wall point gets its own marker style + annotation: the raster cannot
    # represent sdf=0, so only the native measure has a point there.
    ax1.plot([0], [n_wall], color=CB["vermillion"], marker="*", ms=13,
              mec="black", mew=0.6, zorder=5)
    ax1.annotate("surface (sdf=0):\nno raster equivalent",
                 xy=(0, n_wall), xytext=(0.3, 9.0e3),
                 fontsize=7.4, color=CB["vermillion"], ha="left", va="center",
                 arrowprops={"arrowstyle": "-", "color": CB["vermillion"],
                              "lw": 0.8, "shrinkA": 2, "shrinkB": 6})

    ax1.set_ylabel(r"error ratio, channel $u$" "\n" r"(interpolator MSE / Transolver MSE)")
    ax1.set_ylim(2e-3, 2e4)
    ax1.text(2.0, 1.35, r"surrogate better $\uparrow$", fontsize=8, ha="center",
              va="bottom", color=CB["black"])
    ax1.text(2.0, 0.74, r"interpolator better $\downarrow$", fontsize=8, ha="center",
              va="top", color=CB["black"])
    ax1.legend(loc="upper right", fontsize=8.3, handlelength=2.4,
               bbox_to_anchor=(1.0, 0.98))

    # --- Panel B: what each band is worth in cells vs. native nodes -----
    width = 0.36
    ax2.set_yscale("log")
    ax2.bar(x_grid - width / 2, cell_frac, width=width, color=CB["blue"],
             edgecolor="black", linewidth=0.5, label="raster cell fraction")
    ax2.bar(x_grid + width / 2, node_frac, width=width, color=CB["vermillion"],
             edgecolor="black", linewidth=0.5, hatch="///", label="native node fraction")
    ax2.set_ylabel("fraction of\ndomain (log)")
    ax2.set_ylim(5e-4, 8.0)
    ax2.legend(loc="upper center", fontsize=8, handlelength=1.6, ncol=2)

    ax2.set_xticks(x_native)
    ax2.set_xticklabels(["wall"] + BAND_LABELS7, fontsize=7.6)
    ax2.set_xlabel("distance from the wall (chord units)")
    ax2.set_xlim(-0.6, 7.6)

    return fig


def save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"{name}.{ext}")
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(path, bbox_inches="tight", metadata=meta)
        print(f"  wrote {path}")
    plt.close(fig)


def main():
    ma = load(MA_PATH)
    ps = load(PS_PATH)
    fig = make_fig(ma, ps)
    save(fig, "fig_bandratio")


if __name__ == "__main__":
    main()
