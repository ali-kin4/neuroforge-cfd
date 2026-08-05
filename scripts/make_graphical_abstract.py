"""Graphical abstract — the "audit card" in one wide figure.

Four panels, left to right, telling the paper's whole story at editor-triage
speed (this is the artefact Elsevier shows next to the title):

  1 PREDICT    ensemble-mean speed field for one AirfRANS test case
               (surrogate output, airfoil silhouette masked)
  2 AUDIT      the same case's dimensionless physics-residual magnitude map --
               the surrogate checked against the governing equations, no
               ground truth involved
  3 CALIBRATE  fleet view: per-case residual trust score vs true rel-L2 speed
               error over all 200 test cases (the showcased case highlighted,
               top-decile band shaded), Spearman + AUROC annotated
  4 DECIDE     risk-coverage curve (fused score vs oracle vs random) with the
               10%-rejection operating point marked, plus the certificate
               facts: conformal coverage at the 0.90 target and (if measured,
               results/control/audit_cost.json) the audit's per-case cost

Data sources (all committed results -- nothing recomputed except the two
residual/speed MAPS for the showcased case, which are rebuilt from the cached
ensemble-mean field exactly as scripts/run_selective_prediction.py does):
  results/selective/selective_percase.json      per-case scores + errors
  results/selective/selective_prediction.json   risk-coverage + AUROC numbers
  data/cache/w2/ensemble/<case>.npz             mean field for panels 1-2
  results/uq_ensemble/multisplit_conformal.md   coverage numbers (hardcoded
                                                below with source comment)
  results/control/audit_cost.json               optional cost line

Showcased case: the case whose rel-L2 speed error is closest to the fleet
median -- deliberately typical, not cherry-picked (recorded in the output
meta printed to stdout).

Elsevier size spec: minimum 531 (h) x 1328 (w) px at 300 dpi. We emit
13.0 x 5.0 inches at 300 dpi = 3900 x 1500 px (ratio preserved).

Run (CPU, seconds):
    .venv/Scripts/python.exe scripts/make_graphical_abstract.py

Outputs: results/figures/graphical_abstract.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import os

import neuroforge  # noqa: F401  -- MUST precede numpy/torch (sets BLAS thread caps)
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from neuroforge.core.config import Config  # noqa: E402
from neuroforge.core.types import DTYPE, FlowField  # noqa: E402
from neuroforge.data.airfrans_loader import load_airfrans  # noqa: E402
from neuroforge.geometry.encode import encode_case  # noqa: E402
from neuroforge.physics.residuals import PhysicsChecker  # noqa: E402

# Palette (matches run_selective_prediction.py / repo figure conventions).
C_RESIDUAL = "#2a78d6"
C_SIGMA = "#eb6834"
C_FUSED = "#1baf7a"
C_ORACLE = "#52514e"
C_RANDOM = "#898781"
C_SURFACE = "#fcfcfb"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_INK = "#0b0b0b"
C_MUTED = "#898781"
C_BAD = "#c8442c"

# Multi-split conformal coverage across all arms/channels at the 0.90 target
# (results/uq_ensemble/multisplit_conformal.md, 20 cal/test re-draws).
COVERAGE_RANGE = "0.895–0.902"


def log(msg: str) -> None:
    print(f"[abstract] {msg}", flush=True)


def pick_showcase(rows: list[dict]) -> dict:
    """Case whose rel-L2 error is closest to the fleet median (typical case)."""
    errs = np.array([r["rel_l2"] for r in rows])
    med = float(np.median(errs))
    return rows[int(np.argmin(np.abs(errs - med)))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the graphical-abstract figure.")
    p.add_argument("--root", default="data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--w2-cache-dir", default="data/cache/w2")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--percase-json", default="results/selective/selective_percase.json")
    p.add_argument("--selective-json", default="results/selective/selective_prediction.json")
    p.add_argument("--audit-cost-json", default="results/control/audit_cost.json")
    p.add_argument("--out-base", default="results/figures/graphical_abstract")
    args = p.parse_args(argv)

    with open(args.percase_json, encoding="utf-8") as f:
        percase = json.load(f)
    with open(args.selective_json, encoding="utf-8") as f:
        selective = json.load(f)
    rows = percase["ensemble_mean"]

    showcase = pick_showcase(rows)
    log(f"showcase case (median-error, not cherry-picked): {showcase['name']} "
        f"(rel_l2 {showcase['rel_l2']:.4f}, residual {showcase['residual']:.4f})")

    # ---- rebuild the showcased case's fields exactly as the selective study ----
    pairs = load_airfrans(
        root=args.root, task="full", train=False, resolution=args.resolution,
        limit=200, cache_dir=args.cache_dir, download=False, progress=False,
    )
    case = next(c for c, _gt in pairs if c.name == showcase["name"])
    d = np.load(os.path.join(args.w2_cache_dir, "ensemble", f"{case.name}.npz"))
    mean = d["mean"]
    stack = encode_case(case)
    sdf = stack[0].astype(DTYPE)
    mask_geo = stack[1].astype(DTYPE)
    pred = FlowField.from_array(mean, case.domain, mask=mask_geo, sdf=sdf,
                               meta={"source": "ensemble-mean", "case": case.name})
    checker = PhysicsChecker(Config().physics)
    diag = checker.diagnose(pred, case)
    residual_mag = np.sqrt(
        diag.continuity.astype(np.float64) ** 2
        + diag.momentum_x.astype(np.float64) ** 2
        + diag.momentum_y.astype(np.float64) ** 2
        + diag.bc_violation.astype(np.float64) ** 2
    )
    speed = pred.speed()
    solid = np.asarray(mask_geo) <= 0.5

    # ---- figure scaffold ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": C_AXIS,
        "axes.labelcolor": C_INK,
        "text.color": C_INK,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "axes.linewidth": 0.8,
        "figure.facecolor": C_SURFACE,
        "axes.facecolor": C_SURFACE,
        "savefig.facecolor": C_SURFACE,
    })
    fig = plt.figure(figsize=(13.0, 5.0))
    gs = fig.add_gridspec(
        1, 4, left=0.035, right=0.985, top=0.80, bottom=0.13, wspace=0.52,
    )
    fig.suptitle(
        "Self-auditing neural CFD: the surrogate checks its own prediction "
        "against the governing physics — and decides",
        fontsize=14.5, y=0.955, fontweight="bold",
    )

    step_titles = [
        "1 · Predict", "2 · Audit", "3 · Calibrate", "4 · Decide",
    ]
    step_subs = [
        "surrogate speed field",
        "physics-residual map (no ground truth)",
        "trust score ranks true error (200 cases)",
        "reject the least-trusted → near-oracle",
    ]

    # ---- panel 1: prediction ----
    ax1 = fig.add_subplot(gs[0, 0])
    speed_show = np.ma.masked_where(solid, speed)
    im1 = ax1.imshow(speed_show, origin="lower", cmap="viridis",
                     interpolation="nearest")
    ax1.imshow(np.ma.masked_where(~solid, np.zeros_like(speed)), origin="lower",
               cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.set_label("|U|  (m/s)", fontsize=8)
    cb1.ax.tick_params(labelsize=7)

    # ---- panel 2: residual audit ----
    ax2 = fig.add_subplot(gs[0, 1])
    res_show = np.ma.masked_where(solid, residual_mag)
    vmax = float(np.percentile(residual_mag[~solid], 99))
    im2 = ax2.imshow(res_show, origin="lower", cmap="magma", vmin=0.0,
                     vmax=max(vmax, 1e-9), interpolation="nearest")
    ax2.imshow(np.ma.masked_where(~solid, np.zeros_like(speed)), origin="lower",
               cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cb2.set_label("dimensionless residual", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # ---- panel 3: fleet calibration scatter ----
    ax3 = fig.add_subplot(gs[0, 2])
    res_v = np.array([r["residual"] for r in rows])
    err_v = np.array([r["rel_l2"] for r in rows])
    thr = float(np.quantile(err_v, 0.9))
    ax3.axhspan(thr, err_v.max() * 1.15, color=C_BAD, alpha=0.08, zorder=1)
    ax3.scatter(res_v, err_v, s=14, color=C_RESIDUAL, alpha=0.55,
                edgecolors="none", zorder=3)
    ax3.scatter([showcase["residual"]], [showcase["rel_l2"]], s=70,
                facecolors="none", edgecolors=C_INK, linewidths=1.4, zorder=4)
    ax3.annotate("panels 1–2", (showcase["residual"], showcase["rel_l2"]),
                 textcoords="offset points", xytext=(8, -12), fontsize=7.5,
                 color=C_INK)
    ens_scores = selective["arms"]["ensemble_mean"]["scores"]
    rho = ens_scores["residual"]["spearman"]
    auroc_fused = ens_scores["fused"]["auroc_top_decile"]
    ax3.text(0.03, 0.97,
             f"Spearman ρ = {rho:.2f}\nfused AUROC = {auroc_fused:.2f}",
             transform=ax3.transAxes, va="top", ha="left", fontsize=9)
    ax3.text(0.985, 0.985, "worst decile", transform=ax3.transAxes, va="top",
             ha="right", fontsize=7.5, color=C_BAD)
    ax3.set_xlabel("physics-residual trust score", fontsize=9)
    ax3.set_ylabel("true rel-$L_2$ speed error", fontsize=9)
    ax3.grid(color=C_GRID, lw=0.6)
    ax3.set_axisbelow(True)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(labelsize=8)

    # ---- panel 4: decision (risk-coverage + certificate facts) ----
    ax4 = fig.add_subplot(gs[0, 3])
    ks = [0, 5, 10, 20, 30, 50]
    rc_fused = ens_scores["fused"]["risk_coverage"]
    rc_res = ens_scores["residual"]["risk_coverage"]
    oracle = [rc_res[str(k)]["oracle_mean"] for k in ks]
    random_ = [rc_res[str(k)]["random_mean"] for k in ks]
    fused = [rc_fused[str(k)]["score_mean"] for k in ks]
    ax4.plot(ks, random_, color=C_RANDOM, lw=1.4, ls=":", label="random", zorder=2)
    ax4.plot(ks, oracle, color=C_ORACLE, lw=1.4, ls="--", label="oracle", zorder=2)
    ax4.plot(ks, fused, color=C_FUSED, lw=2.2, marker="^", ms=4.5,
             label="fused audit score", zorder=3)
    k10 = rc_fused["10"]
    ax4.scatter([10], [k10["score_mean"]], s=60, facecolors="none",
                edgecolors=C_INK, linewidths=1.4, zorder=4)
    ax4.annotate("reject 10%:\n~91% of oracle", (10, k10["score_mean"]),
                 textcoords="offset points", xytext=(14, -26), fontsize=7.5)
    ax4.set_xlabel("cases rejected by audit (%)", fontsize=9)
    ax4.set_ylabel("mean error of retained cases", fontsize=9)
    ax4.set_xticks(ks)
    ax4.grid(axis="y", color=C_GRID, lw=0.6)
    ax4.set_axisbelow(True)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.tick_params(labelsize=8)
    ax4.legend(frameon=False, fontsize=7.5, loc="lower left")

    cert_lines = [f"conformal coverage {COVERAGE_RANGE} @ 0.90 target"]
    if os.path.exists(args.audit_cost_json):
        with open(args.audit_cost_json, encoding="utf-8") as f:
            cost = json.load(f)
        ms = cost["audit_total_ms"]["median_ms"]
        cert_lines.append(f"audit cost ≈ {ms:.0f} ms/case (CPU)")
    ax4.text(0.985, 0.985, "\n".join(cert_lines), transform=ax4.transAxes,
             va="top", ha="right", fontsize=7.5, color=C_INK,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor=C_AXIS, lw=0.8))

    # ---- step headers + connecting arrows ----
    axes = [ax1, ax2, ax3, ax4]
    for ax, title, sub in zip(axes, step_titles, step_subs):
        pos = ax.get_position()
        cx = 0.5 * (pos.x0 + pos.x1)
        fig.text(cx, 0.875, title, ha="center", va="bottom", fontsize=12,
                 fontweight="bold")
        fig.text(cx, 0.845, sub, ha="center", va="bottom", fontsize=8.5,
                 color=C_MUTED)
    for a, b in zip(axes[:-1], axes[1:]):
        pa, pb = a.get_position(), b.get_position()
        arrow = FancyArrowPatch(
            (pa.x1 + 0.004, 0.885), (pb.x0 - 0.004, 0.885),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
            color=C_MUTED, lw=1.2, shrinkA=0, shrinkB=0,
        )
        fig.add_artist(arrow)

    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    fig.savefig(args.out_base + ".png", dpi=300)
    fig.savefig(args.out_base + ".pdf")
    plt.close(fig)
    log(f"wrote {args.out_base}.png / .pdf (3900x1500 px @300dpi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
