"""Generate the DeepCFD residual-vs-error trust-signal scatter figure.

Produces:
  results/figures/fig_deepcfd_trust.pdf
  results/figures/fig_deepcfd_trust.png
  results/deepcfd/deepcfd_trust_percase.json

Faithfulness: reuses run_deepcfd.py's EXACT pipeline —
  - same OOD split (build_ood_split with defaults: ood_percentile=75, split_seed=0)
  - same load_deepcfd at resolution=128
  - same Normalizer fitted on train pairs
  - same PhysicsChecker(Config().physics, include_bc=False)
  - same per-case residual via checker.diagnose(pred, case).residual_norm()
  - same per-case rel-L2 speed error formula as in evaluate_cases/_fluid_mask
  - same Spearman via residual_error_correlation from neuroforge.physics.evaluation

Cross-checks each seed's recomputed Spearman vs saved deepcfd_results.json;
prints a MISMATCH/PASS verdict.

CPU-only. Always import neuroforge before numpy/torch.
"""
# MUST be first (caps BLAS threads before numpy/torch)
import neuroforge  # noqa: F401

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from neuroforge.core.config import Config
from neuroforge.data.datamodule import FlowDataset, Normalizer, collate_flow
from neuroforge.data.deepcfd_loader import load_deepcfd, obstacle_areas
from neuroforge.models.fno import FNO2d
from neuroforge.physics.evaluation import residual_error_correlation, _spearman
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import Predictor
from torch.utils.data import DataLoader

# Identical to run_deepcfd.build_ood_split (copy so this script is self-contained
# and provably uses the same logic even if run_deepcfd.py drifts).
def build_ood_split(root, n_train, n_test, ood_percentile=75.0, seed=0):
    areas = obstacle_areas(root)
    thresh = np.percentile(areas, ood_percentile)
    train_pool = np.where(areas < thresh)[0]
    test_pool = np.where(areas >= thresh)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    train_idx = train_pool[: min(n_train, len(train_pool))]
    test_idx = test_pool[: min(n_test, len(test_pool))]
    return (
        sorted(int(i) for i in train_idx),
        sorted(int(i) for i in test_idx),
        float(thresh),
    )


def _fluid_mask(ref, pred):
    """Identical to evaluation.py's _fluid_mask."""
    from neuroforge.core.types import DTYPE
    mask = ref.mask if ref.mask is not None else pred.mask
    if mask is None:
        mask = np.ones(ref.shape, dtype=DTYPE)
    m = np.asarray(mask) > 0.5
    return m if m.any() else np.ones(ref.shape, dtype=bool)


def per_case_pairs(predict_fn, test_pairs, checker):
    """Return (res_norms, rel_l2_errors) lists per case.

    Uses the identical formula as evaluate_cases in neuroforge.physics.evaluation.
    """
    res_norms = []
    rel_l2_errs = []
    for case, ref in test_pairs:
        pred = predict_fn(case)
        # residual norm
        diag = checker.diagnose(pred, case)
        res_norms.append(float(diag.residual_norm()))
        # rel-L2 speed error (same formula as evaluate_cases)
        m = _fluid_mask(ref, pred)
        num = float(np.sqrt(np.sum(((pred.speed() - ref.speed())[m]) ** 2)))
        den = float(np.sqrt(np.sum((ref.speed()[m]) ** 2))) + 1e-12
        rel_l2_errs.append(num / den)
    return res_norms, rel_l2_errs


def main():
    # ---- paths ----------------------------------------------------------------
    data_root = os.path.join(ROOT, "data", "deepcfd")
    cache_dir = os.path.join(ROOT, "data", "cache")
    ckpt_dir = os.path.join(ROOT, "checkpoints", "deepcfd")
    out_dir = os.path.join(ROOT, "results", "deepcfd")
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- load saved results for cross-check -----------------------------------
    saved_json = os.path.join(out_dir, "deepcfd_results.json")
    with open(saved_json, encoding="utf-8") as f:
        saved = json.load(f)
    saved_rho = {s["seed"]: s["metrics"]["residual_error_spearman"]
                 for s in saved["per_seed"]}
    saved_agg = saved["aggregate"]["residual_error_spearman"]
    print(f"Saved aggregate rho = {saved_agg['mean']:.4f} +/- {saved_agg['std']:.4f}")
    print(f"Saved per-seed rho: {saved_rho}")

    # ---- OOD split (MUST match run_deepcfd defaults exactly) ------------------
    N_TRAIN, N_TEST, OOD_PCT, SPLIT_SEED = 700, 247, 75.0, 0
    RESOLUTION = 128
    WIDTH, N_LAYERS, MODES = 32, 4, 24

    print(f"\nBuilding OOD split: n_train={N_TRAIN}, n_test={N_TEST}, "
          f"ood_percentile={OOD_PCT}, split_seed={SPLIT_SEED}")
    train_idx, test_idx, thresh = build_ood_split(
        data_root, N_TRAIN, N_TEST, OOD_PCT, SPLIT_SEED
    )
    print(f"  area threshold = {thresh:.0f} px, "
          f"n_train={len(train_idx)}, n_test={len(test_idx)}")

    # Sanity assertion (catches data-folder drift)
    if len(test_idx) != 247:
        print(f"FATAL: expected 247 test cases, got {len(test_idx)}", file=sys.stderr)
        sys.exit(1)
    if abs(thresh - 600.0) > 50:
        print(f"WARNING: area threshold {thresh:.0f} px deviates from expected ~600 px",
              file=sys.stderr)

    # ---- load data ------------------------------------------------------------
    print("Loading DeepCFD train pairs (for normalizer) ...")
    train_pairs = load_deepcfd(
        root=data_root, resolution=RESOLUTION, indices=train_idx,
        cache_dir=cache_dir, progress=False,
    )
    print("Loading DeepCFD test pairs ...")
    test_pairs = load_deepcfd(
        root=data_root, resolution=RESOLUTION, indices=test_idx,
        cache_dir=cache_dir, progress=False,
    )

    # ---- fit normalizer on identical train set --------------------------------
    train_ds = FlowDataset(train_pairs, normalizer=None)
    xi, yi = train_ds.raw_arrays()
    normalizer = Normalizer().fit(xi, yi)
    print(f"Normalizer fitted on {len(train_pairs)} train cases.")

    # ---- physics checker (interior-only, identical to run_deepcfd) -----------
    checker = PhysicsChecker(Config().physics, include_bc=False)

    device = torch.device("cpu")

    # ---- per-seed inference ---------------------------------------------------
    all_res_norms = []   # list of lists (one per seed)
    all_rel_l2 = []
    all_seed_rho = []
    mismatch_flag = False

    for m in range(3):
        ckpt_path = os.path.join(ckpt_dir, f"seed{m}.pt")
        done_path = os.path.join(ckpt_dir, f"seed{m}.done")
        if not (os.path.exists(ckpt_path) and os.path.exists(done_path)):
            print(f"FATAL: checkpoint for seed {m} missing or incomplete "
                  f"({ckpt_path})", file=sys.stderr)
            sys.exit(1)

        print(f"\n--- Seed {m} ---")
        model = FNO2d(
            in_channels=Config().model.in_channels,
            out_channels=Config().model.out_channels,
            width=WIDTH, n_layers=N_LAYERS, modes=MODES,
        ).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        print(f"  Loaded checkpoint (epoch {state.get('epoch', '?')})")

        predictor = Predictor(model, normalizer, device="cpu")

        print(f"  Running per-case residual + error for {len(test_pairs)} cases ...")
        res_norms, rel_l2_errs = per_case_pairs(predictor.predict, test_pairs, checker)

        rho = _spearman(np.array(res_norms), np.array(rel_l2_errs))
        print(f"  Recomputed Spearman rho = {rho:.6f}")
        print(f"  Saved     Spearman rho = {saved_rho[m]:.6f}")
        delta = abs(rho - saved_rho[m])
        if delta > 0.05:
            print(f"  MISMATCH: |delta| = {delta:.4f} > 0.05 — STOP", file=sys.stderr)
            mismatch_flag = True
        else:
            print(f"  PASS: |delta| = {delta:.6f} (within tolerance)")

        all_res_norms.append(res_norms)
        all_rel_l2.append(rel_l2_errs)
        all_seed_rho.append(rho)

        del model

    if mismatch_flag:
        print("\nOne or more seeds have a gross Spearman mismatch. "
              "Check normalizer/split/checker/checkpoint. Aborting figure.", file=sys.stderr)
        sys.exit(2)

    # ---- dump per-case data ---------------------------------------------------
    percase_data = {
        "meta": {
            "description": "Per-case (residual_norm, rel_l2_error) for DeepCFD OOD test set",
            "n_seeds": 3,
            "n_cases_per_seed": 247,
            "checker": "PhysicsChecker(include_bc=False)",
            "ood_split": {
                "ood_percentile": OOD_PCT,
                "split_seed": SPLIT_SEED,
                "area_threshold_px": float(thresh),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            },
        },
        "per_seed": [
            {
                "seed": m,
                "residual_norms": all_res_norms[m],
                "rel_l2_errors": all_rel_l2[m],
                "residual_error_spearman_recomputed": float(all_seed_rho[m]),
                "residual_error_spearman_saved": saved_rho[m],
            }
            for m in range(3)
        ],
        "aggregate_spearman_recomputed": {
            "mean": float(np.mean(all_seed_rho)),
            "std": float(np.std(all_seed_rho)),
        },
        "aggregate_spearman_saved": saved_agg,
    }
    percase_path = os.path.join(out_dir, "deepcfd_trust_percase.json")
    with open(percase_path, "w", encoding="utf-8") as f:
        json.dump(percase_data, f, indent=2)
    print(f"\nDumped per-case data to {percase_path}")

    # ---- figure ---------------------------------------------------------------
    # Style matching existing figures (make_figures.py)
    CB = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "vermillion": "#D55E00",
        "purple": "#CC79A7",
        "sky": "#56B4E9",
        "grey": "#999999",
    }
    SEED_COLORS = [CB["blue"], CB["orange"], CB["green"]]
    SEED_MARKERS = ["o", "s", "^"]

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

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Pool all points to decide on axis scaling
    all_x = np.concatenate([np.array(r) for r in all_res_norms])
    all_y = np.concatenate([np.array(e) for e in all_rel_l2])

    # Use log x-axis if residuals span > 1 decade
    x_ratio = all_x.max() / (all_x.min() + 1e-30)
    use_log_x = x_ratio > 10.0
    print(f"\nResidual range: [{all_x.min():.3e}, {all_x.max():.3e}] "
          f"(ratio={x_ratio:.1f}) -> log_x={use_log_x}")

    for m in range(3):
        xv = np.array(all_res_norms[m])
        yv = np.array(all_rel_l2[m])
        ax.scatter(xv, yv, color=SEED_COLORS[m], marker=SEED_MARKERS[m],
                   s=20, alpha=0.55, linewidths=0,
                   label=f"seed {m}  ($\\rho$={all_seed_rho[m]:.3f})")

    # Pooled OLS trend line in log-x space (guide to the eye; monotone by construction).
    # Fitting log(x) vs y gives a straight line on the log-x axis.
    log_x = np.log(all_x)
    slope, intercept = np.polyfit(log_x, all_y, 1)
    x_fit = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 200)
    y_fit = slope * np.log(x_fit) + intercept
    ax.plot(x_fit, y_fit, color="black", ls="--", lw=1.3,
            alpha=0.65, zorder=5, label="pooled OLS trend (log-x)")

    if use_log_x:
        ax.set_xscale("log")

    # Annotate aggregate rho
    agg_mean = float(np.mean(all_seed_rho))
    agg_std = float(np.std(all_seed_rho))
    n_total = sum(len(r) for r in all_res_norms)
    ax.text(0.97, 0.07,
            f"$\\rho_{{\\rm agg}}$ = {agg_mean:.2f} $\\pm$ {agg_std:.2f}\n"
            f"(n={n_total // 3} per seed, 3 seeds)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel("Field-mean PDE residual norm  (interior continuity + momentum)")
    ax.set_ylabel("Rel-L2 speed error  $\\|\\hat{u}-u\\|_2 / \\|u\\|_2$  (lower = better)")
    ax.set_title("DeepCFD trust signal: PDE residual tracks prediction error\n"
                 "(OOD test set: largest bodies, area $\\geq$ 75th pctile; CPU inference)")
    ax.legend(loc="upper left", markerscale=1.4)

    # Save
    pdf_path = os.path.join(fig_dir, "fig_deepcfd_trust.pdf")
    png_path = os.path.join(fig_dir, "fig_deepcfd_trust.png")
    for ext, path in [("pdf", pdf_path), ("png", png_path)]:
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)

    # ---- final report ---------------------------------------------------------
    print("\n=== FAITHFULNESS CHECK SUMMARY ===")
    for m in range(3):
        d = abs(all_seed_rho[m] - saved_rho[m])
        status = "PASS" if d <= 0.05 else "MISMATCH"
        print(f"  Seed {m}: recomputed={all_seed_rho[m]:.6f}  "
              f"saved={saved_rho[m]:.6f}  |delta|={d:.6f}  [{status}]")
    print(f"  Aggregate (recomputed): mean={agg_mean:.4f}  std={agg_std:.4f}")
    print(f"  Aggregate (saved):      mean={saved_agg['mean']:.4f}  "
          f"std={saved_agg['std']:.4f}")
    print(f"\nFigure: {pdf_path}")
    print(f"        {png_path}")
    print(f"Data:   {percase_path}")


if __name__ == "__main__":
    main()
