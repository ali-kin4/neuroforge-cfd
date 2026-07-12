"""Generate publication-grade figures for the NeuroForge CFD paper.

All numbers come from the committed result files under ``results/`` — nothing
is hand-typed. Outputs PDF + PNG to ``results/figures/``.

Run:
    .venv/Scripts/python.exe scripts/make_figures.py

CPU-only, plotting only. Does not touch results/v2 or any frozen file.
"""

# --- thread caps: import neuroforge FIRST (sets OPENBLAS/OMP/MKL=1) ----------
import matplotlib

import neuroforge  # noqa: F401  (side effect: caps BLAS threads before numpy)

matplotlib.use("Agg")  # non-interactive backend

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# --- paths ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

# --- style ------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette.
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
})

# Short, human labels for the four NeuroForge arms.
ARM_LABEL = {
    "backbone": "Backbone\n(+phys loss)",
    "backbone (no physics loss)": "Backbone\n(no phys)",
    "backbone + local corrector": "+ Local\ncorrector",
    "backbone + DEQ corrector": "+ DEQ\ncorrector",
}
ARM_ORDER = [
    "backbone (no physics loss)",
    "backbone",
    "backbone + local corrector",
    "backbone + DEQ corrector",
]
ARM_COLOR = {
    "backbone (no physics loss)": CB["grey"],
    "backbone": CB["sky"],
    "backbone + local corrector": CB["orange"],
    "backbone + DEQ corrector": CB["blue"],
}

# Metric metadata: pretty label (with unit) and direction.
#   "lower" -> lower is better;  "rho" -> rho->1 is better.
METRIC = {
    "mse_u":  ("MSE $u$  [m$^2$/s$^2$]", "lower"),
    "mse_v":  ("MSE $v$  [m$^2$/s$^2$]", "lower"),
    "mse_p":  ("MSE $p$  [m$^4$/s$^4$]", "lower"),
    "surface_mse_p": ("Surface MSE $p$  [m$^4$/s$^4$]", "lower"),
    "rho_cl": (r"$\rho(C_l)$", "rho"),
    "rho_cd": (r"$\rho(C_d)$", "rho"),
    "residual_error_spearman": (r"$\rho$(residual, error)", "rho"),
}


def dir_tag(metric):
    return "lower = better" if METRIC[metric][1] == "lower" else r"$\rho\to1$ = better"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")


# --- loaders ----------------------------------------------------------------
def load_ablation_indist():
    """results/full_research/in_dist/ablation.csv -> {arm: {metric: (mean,std)}}."""
    path = os.path.join(RES, "full_research", "in_dist", "ablation.csv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["arm"], {})[row["metric"]] = (
                float(row["mean"]), float(row["std"]))
    return out


def load_ablation_ood():
    """ablation_ood.csv -> {task: {arm: {metric: (mean,std)}}} for reynolds/aoa."""
    path = os.path.join(RES, "full_research", "ood", "ablation_ood.csv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = out.setdefault(row["ood_task"], {})
            t.setdefault(row["arm"], {})[row["metric"]] = (
                float(row["mean"]), float(row["std"]))
    return out


def load_transolver():
    """results/baselines/table2.csv -> {metric: (mean,std)}, n_params."""
    path = os.path.join(RES, "baselines", "table2.csv")
    out, n_params = {}, None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n_params = int(row["n_params"])
            out[row["metric"]] = (float(row["mean"]), float(row["std"]))
    return out, n_params


def load_iters():
    path = os.path.join(RES, "sensitivity", "iters.csv")
    cols = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        names = rd.fieldnames
        for n in names:
            cols[n] = []
        for row in rd:
            for n in names:
                cols[n].append(float(row[n]))
    return {k: np.array(v) for k, v in cols.items()}


def load_json(*parts):
    with open(os.path.join(RES, *parts)) as f:
        return json.load(f)


# ============================================================================
# Figure 1 — In-distribution ablation (Table 1)
# ============================================================================
def fig1_ablation(ab):
    metrics = ["mse_u", "surface_mse_p", "rho_cd", "residual_error_spearman"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    arms = ARM_ORDER
    x = np.arange(len(arms))
    colors = [ARM_COLOR[a] for a in arms]
    labels = [ARM_LABEL[a] for a in arms]

    for ax, m in zip(axes, metrics):
        means = np.array([ab[a][m][0] for a in arms])
        stds = np.array([ab[a][m][1] for a in arms])
        ax.bar(x, means, yerr=stds, color=colors, capsize=4,
               edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{METRIC[m][0]}\n({dir_tag(m)})", fontsize=10)
        if METRIC[m][1] == "rho":
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color=CB["vermillion"], ls=":", lw=1)
        if m == "surface_mse_p":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.suptitle("In-distribution ablation of NeuroForge arms (mean $\\pm$ std over seeds)",
                 fontsize=13)
    save(fig, "fig1_ablation_indist")


# ============================================================================
# Figure 2 — OOD generalisation gap: DEQ arm vs backbone (Tables 1 + 3)
# ============================================================================
def fig2_ood_gap(ab, ood):
    metrics = ["mse_u", "surface_mse_p", "residual_error_spearman"]
    tasks = ["in_dist", "reynolds", "aoa"]
    task_label = {"in_dist": "In-dist", "reynolds": "Reynolds OOD", "aoa": "AoA OOD"}
    arms = ["backbone", "backbone + DEQ corrector"]
    arm_label = {"backbone": "Backbone", "backbone + DEQ corrector": "+ DEQ corrector"}
    arm_color = {"backbone": CB["sky"], "backbone + DEQ corrector": CB["blue"]}

    def get(task, arm, m):
        if task == "in_dist":
            return ab[arm][m]
        return ood[task][arm][m]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(tasks))
    w = 0.36
    for ax, m in zip(axes, metrics):
        for i, arm in enumerate(arms):
            means = np.array([get(t, arm, m)[0] for t in tasks])
            stds = np.array([get(t, arm, m)[1] for t in tasks])
            ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3,
                   color=arm_color[arm], edgecolor="black", linewidth=0.5,
                   label=arm_label[arm], error_kw={"elinewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([task_label[t] for t in tasks], fontsize=9)
        ax.set_title(f"{METRIC[m][0]}\n({dir_tag(m)})", fontsize=10)
        if METRIC[m][1] == "rho":
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color=CB["vermillion"], ls=":", lw=1)
        if m == "surface_mse_p":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(loc="upper left")
    fig.suptitle("Out-of-distribution gap: DEQ corrector shrinks the shift penalty",
                 fontsize=13)
    save(fig, "fig2_ood_gap")


# ============================================================================
# Figure 3 — NeuroForge (DEQ) vs Transolver (Tables 1 + 2). HONEST: baseline wins.
# ============================================================================
def fig3_vs_transolver(ab, trans):
    mse_metrics = ["mse_u", "mse_v", "mse_p", "surface_mse_p"]
    rho_metrics = ["rho_cl", "rho_cd"]
    nf = ab["backbone + DEQ corrector"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12, 4.2),
        gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: MSE metrics on a log scale.
    x = np.arange(len(mse_metrics))
    w = 0.38
    nf_m = np.array([nf[m][0] for m in mse_metrics])
    nf_s = np.array([nf[m][1] for m in mse_metrics])
    tr_m = np.array([trans[m][0] for m in mse_metrics])
    tr_s = np.array([trans[m][1] for m in mse_metrics])
    axL.bar(x - w / 2, nf_m, w, yerr=nf_s, capsize=3, color=CB["blue"],
            edgecolor="black", linewidth=0.5, label="NeuroForge (+DEQ)",
            error_kw={"elinewidth": 1})
    axL.bar(x + w / 2, tr_m, w, yerr=tr_s, capsize=3, color=CB["vermillion"],
            edgecolor="black", linewidth=0.5, label="Transolver (baseline)",
            error_kw={"elinewidth": 1})
    axL.set_yscale("log")
    axL.set_xticks(x)
    axL.set_xticklabels([METRIC[m][0] for m in mse_metrics], fontsize=8.5)
    axL.set_ylabel("MSE (log scale) — lower = better")
    axL.set_title("Field / surface error (log scale)")
    axL.legend(loc="upper left")
    # annotate the gap factor above each NeuroForge bar
    for xi, (a, b) in zip(x, zip(nf_m, tr_m)):
        axL.text(xi, max(a, b) * 1.4, f"{a / b:.0f}$\\times$",
                 ha="center", va="bottom", fontsize=8, color=CB["vermillion"])

    # Right: rho metrics on linear, zoomed near 1.
    xr = np.arange(len(rho_metrics))
    nfr = np.array([nf[m][0] for m in rho_metrics])
    nfr_s = np.array([nf[m][1] for m in rho_metrics])
    trr = np.array([trans[m][0] for m in rho_metrics])
    trr_s = np.array([trans[m][1] for m in rho_metrics])
    axR.bar(xr - w / 2, nfr, w, yerr=nfr_s, capsize=3, color=CB["blue"],
            edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1})
    axR.bar(xr + w / 2, trr, w, yerr=trr_s, capsize=3, color=CB["vermillion"],
            edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1})
    axR.set_xticks(xr)
    axR.set_xticklabels([METRIC[m][0] for m in rho_metrics])
    axR.set_ylim(0.85, 1.01)
    axR.axhline(1.0, color="black", ls=":", lw=1)
    axR.set_title(r"Force correlation ($\rho\to1$)")
    axR.set_ylabel(r"Spearman $\rho$")

    fig.suptitle("Honest baseline comparison: Transolver outperforms NeuroForge on accuracy",
                 fontsize=13)
    save(fig, "fig3_vs_transolver")


# ============================================================================
# Figure 4 — Sensitivity to correction iterations (iters.csv)
# ============================================================================
def fig4_iters(it):
    n = it["n_iters"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # Panel A: accuracy metrics (mse_u, surface_mse_p) on twin axes.
    axb = ax1.twinx()
    l1, = ax1.plot(n, it["mse_u"], "-o", color=CB["blue"], label="MSE $u$ (lower=better)")
    l2, = axb.plot(n, it["surface_mse_p"], "-s", color=CB["orange"],
                   label="Surface MSE $p$ (lower=better)")
    ax1.set_xlabel("# correction iterations")
    ax1.set_ylabel("MSE $u$  [m$^2$/s$^2$]", color=CB["blue"])
    axb.set_ylabel("Surface MSE $p$  [m$^4$/s$^4$]", color=CB["orange"])
    axb.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.tick_params(axis="y", labelcolor=CB["blue"])
    axb.tick_params(axis="y", labelcolor=CB["orange"])
    axb.grid(False)
    ax1.set_title("Accuracy: fast drop then plateau")
    ax1.legend(handles=[l1, l2], loc="upper right")

    # Panel B: the residual-up / error-down nuance.
    axc = ax2.twinx()
    l3, = ax2.plot(n, it["residual_norm"], "-^", color=CB["vermillion"],
                   label="Residual norm (rises)")
    l4, = axc.plot(n, it["residual_error_spearman"], "-D", color=CB["green"],
                   label=r"$\rho$(residual, error) ($\rho\to1$)")
    ax2.set_xlabel("# correction iterations")
    ax2.set_ylabel("Residual norm", color=CB["vermillion"])
    axc.set_ylabel(r"$\rho$(residual, error)", color=CB["green"])
    ax2.tick_params(axis="y", labelcolor=CB["vermillion"])
    axc.tick_params(axis="y", labelcolor=CB["green"])
    axc.set_ylim(0, 1.0)
    axc.grid(False)
    ax2.set_title("Residual rises while its diagnostic value improves")
    ax2.legend(handles=[l3, l4], loc="lower right")

    fig.suptitle("Sensitivity to number of correction (DEQ) iterations", fontsize=13)
    save(fig, "fig4_sensitivity_iters")


# ============================================================================
# Figure 5 — OOD conformal coverage (sensitivity ood_coverage.json)
# ============================================================================
def fig5_ood_coverage(cov):
    target = cov["target"]
    channels = ["u", "v", "p"]
    tasks = ["full", "reynolds", "aoa"]
    task_label = {"full": "In-dist (calib)", "reynolds": "Reynolds OOD", "aoa": "AoA OOD"}
    task_color = {"full": CB["grey"], "reynolds": CB["sky"], "aoa": CB["orange"]}

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(channels))
    w = 0.26
    for i, t in enumerate(tasks):
        vals = [cov["per_channel"][c]["coverage"][t] for c in channels]
        ax.bar(x + (i - 1) * w, vals, w, color=task_color[t],
               edgecolor="black", linewidth=0.5, label=task_label[t])
    ax.axhline(target, color=CB["vermillion"], ls="--", lw=1.5,
               label=f"target = {target:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"${c}$" for c in channels])
    ax.set_xlabel("Field channel")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Conformal coverage holds at the 90% target under distribution shift\n"
                 "(calibrated in-distribution, $\\alpha=0.1$)")
    ax.legend(loc="lower right", ncol=2)
    save(fig, "fig5_ood_coverage")


# ============================================================================
# Figure 6 — Reliability diagram + certified contraction envelope
# ============================================================================
def fig6_reliability_contraction(h4, h5):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # --- Left: reliability diagram (fully regenerable from h4 JSON) ----------
    rel = h4["reliability"]
    cm = {"u": CB["blue"], "v": CB["orange"], "p": CB["green"]}
    axL.plot([0, 1], [0, 1], color="black", ls=":", lw=1.2, label="ideal")
    for c in ("u", "v", "p"):
        nom = np.array(rel[c]["nominal"])
        emp = np.array(rel[c]["empirical"])
        ece = rel[c]["ece"]
        axL.plot(nom, emp, "-o", color=cm[c], ms=4,
                 label=f"${c}$  (ECE={ece:.3f})")
    axL.set_xlabel("Nominal coverage")
    axL.set_ylabel("Empirical coverage")
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.set_aspect("equal")
    axL.set_title(f"Calibration reliability (n_test={h4['n_test']}, "
                  f"MC={h4['n_mc']})")
    axL.legend(loc="upper left")

    # --- Right: certified contraction envelope (from h5 scalar kappas) -------
    # No per-iteration residual data exists in the JSON; we plot the certified
    # geometric-decay envelope r_n = r0 * kappa^n for the median / p90 / max
    # contraction factors, which IS what the certificate bounds.
    budget = int(h5["n_iter_budget"])
    nvec = np.arange(0, budget + 1)
    r0 = 1.0  # normalised initial residual
    kappas = [
        (h5["contraction_factor_median"], CB["blue"], "median"),
        (h5["contraction_factor_p90"], CB["orange"], "p90"),
        (h5["contraction_factor_max"], CB["vermillion"], "max (worst case)"),
    ]
    for k, col, lab in kappas:
        axR.semilogy(nvec, r0 * k ** nvec, "-", color=col, lw=1.8,
                     label=f"$\\kappa$={k:.3f} ({lab})")
    # design bound reference
    kd = h5["kappa_design"]
    axR.semilogy(nvec, r0 * kd ** nvec, "--", color="black", lw=1.2,
                 label=f"design bound $\\kappa$={kd:.2f}")
    # certified iters-to-1e-5 marker
    it5 = h5["iters_to_1e-5_median"]
    axR.axvline(it5, color=CB["green"], ls=":", lw=1.3)
    axR.axhline(1e-5, color=CB["grey"], ls=":", lw=1.0)
    axR.annotate(f"median {it5:.0f} iters\nto $10^{{-5}}$",
                 xy=(it5, 1e-5), xytext=(it5 - 22, 3e-4),
                 fontsize=8.5, color=CB["green"],
                 arrowprops=dict(arrowstyle="->", color=CB["green"], lw=1))
    axR.set_xlabel("Fixed-point iteration $n$")
    axR.set_ylabel("Normalised residual $r_n / r_0$")
    axR.set_ylim(1e-7, 2)
    axR.set_title(f"Certified contraction envelope\n(n_cases={h5['n_cases']}, "
                  f"all reached $10^{{-5}}$)")
    axR.legend(loc="upper right", fontsize=8)

    fig.suptitle("Uncertainty calibration and certified residual contraction (H4 / H5)",
                 fontsize=13)
    save(fig, "fig6_reliability_contraction")


# ============================================================================
def main():
    print("Loading committed results ...")
    ab = load_ablation_indist()
    ood = load_ablation_ood()
    trans, n_params = load_transolver()
    it = load_iters()
    h4 = load_json("certificates", "h4_coverage.json")
    h5 = load_json("certificates", "h5_contraction.json")
    ood_cov = load_json("sensitivity", "ood_coverage.json")

    print("Rendering figures ...")
    fig1_ablation(ab)
    fig2_ood_gap(ab, ood)
    fig3_vs_transolver(ab, trans)
    fig4_iters(it)
    fig5_ood_coverage(ood_cov)
    fig6_reliability_contraction(h4, h5)
    print(f"Done. Figures written to {OUT}")


if __name__ == "__main__":
    main()
