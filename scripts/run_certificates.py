"""Certificate experiments H4 (split-conformal coverage) and H5 (DEQ contraction).

Two "certificate" experiments that the paper currently marks as *pending*:

* **H5 — empirical contraction factor of the DEQ corrector.** We train a
  contractive Deep-Equilibrium corrector on real AirfRANS (``task='full'``,
  cached), then *measure* the contraction of the learned fixed-point operator
  ``T_theta`` from scratch on real cases: starting from a random init we iterate
  ``delta_{k+1} = T_theta(delta_k)`` and record the distance-to-fixed-point
  ratio ``||delta_k - delta*|| / ||delta_{k-1} - delta*||`` per step. The
  fixed point ``delta*`` is obtained first at high precision. H5 is *falsified*
  if the measured ratio is >= 1.

  Note: we iterate the **pure operator** ``T_theta`` (not the damped forward map
  ``F = (1-beta) I + beta T_theta`` that ``solve()`` uses internally), because
  the paper's claim in Sec 3.1b is about ``delta_{k+1}=T_theta(delta_k)``. Both
  maps share the same fixed point, so this is the faithful certificate for the
  stated rate.

* **H4 — split-conformal coverage on real AirfRANS.** Using MC-dropout as the
  per-cell predictive sigma, we hold out a calibration split, fit the
  finite-sample-corrected (1-alpha) conformal multiplier ``q`` per channel
  (u, v, p) at alpha=0.1, and verify empirical coverage on a *separate* test
  split. We also sweep alpha (coverage-vs-alpha) and draw a reliability diagram.

Artifacts:
    results/certificates/h5_contraction.json
    results/certificates/h4_coverage.json
    results/certificates/h4_coverage.csv
    results/certificates/figures/h5_contraction_curve.png
    results/certificates/figures/h4_reliability.png
    results/certificates/figures/h4_coverage_vs_alpha.png
    checkpoints/certificates_deq.pt   (gitignored)

Run:
    .venv/Scripts/python.exe scripts/run_certificates.py
"""

from __future__ import annotations

# IMPORTANT: import neuroforge before numpy/torch (thread caps, see CLAUDE.md).
import neuroforge as nf  # noqa: F401

import json
import os
import time

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neuroforge.core.types import DTYPE  # noqa: E402
from neuroforge.geometry.encode import encode_case  # noqa: E402

# ---------------------------------------------------------------------------- #
# Paths & config
# ---------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "certificates")
FIGS = os.path.join(RESULTS, "figures")
CKPT = os.path.join(ROOT, "checkpoints", "certificates_deq.pt")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)

# Training budget (a *moderate* budget — this demonstrates the structural
# guarantee, not a SOTA model). All numbers are reported in the artifacts.
CONFIG = dict(
    backbone="fno",
    width=48,
    n_layers=4,
    modes=20,
    dropout=0.05,          # enables MC-dropout for H4
    corrector="deq",       # contractive DEQ corrector for H5
    epochs=40,
    corrector_epochs=15,
    lr=8e-4,
    resolution=128,
    batch_size=8,
    seed=0,
    # data
    task="full",
    n_train=800,
    n_val=200,
)


def log(msg: str) -> None:
    print(f"[certificates] {msg}", flush=True)


# ---------------------------------------------------------------------------- #
# Train (or load)
# ---------------------------------------------------------------------------- #
def get_model() -> "nf.NeuroForge":
    """Train the dropout-FNO + DEQ corrector on cached AirfRANS, or load it."""
    if os.path.exists(CKPT) and os.environ.get("CERT_RETRAIN", "0") != "1":
        log(f"loading cached checkpoint {CKPT}")
        model = nf.NeuroForge.load(CKPT, device="cuda" if torch.cuda.is_available() else "cpu")
        # the loaded model lacks _data_kw; restore so val_data() works
        model._data_kw = dict(
            source="airfrans", task=CONFIG["task"], root="data",
            n_train=CONFIG["n_train"], n_val=CONFIG["n_val"],
            resolution=CONFIG["resolution"], cache_dir="data/cache", download=False,
        )
        return model

    log("training dropout-FNO backbone + DEQ corrector on AirfRANS (cache) ...")
    t0 = time.time()
    model = nf.NeuroForge(
        backbone=CONFIG["backbone"], width=CONFIG["width"], n_layers=CONFIG["n_layers"],
        modes=CONFIG["modes"], dropout=CONFIG["dropout"], corrector=CONFIG["corrector"],
        epochs=CONFIG["epochs"], corrector_epochs=CONFIG["corrector_epochs"],
        lr=CONFIG["lr"], resolution=CONFIG["resolution"], batch_size=CONFIG["batch_size"],
        device="auto", seed=CONFIG["seed"],
    )
    model.fit(
        "airfrans", task=CONFIG["task"], n_train=CONFIG["n_train"], n_val=CONFIG["n_val"],
        root="data", cache_dir="data/cache", download=False, out=CKPT, verbose=True,
    )
    log(f"training done in {time.time() - t0:.0f}s; val metrics={model.metrics_}")
    return model


# ---------------------------------------------------------------------------- #
# H5 — empirical contraction of the DEQ operator T_theta
# ---------------------------------------------------------------------------- #
def _corrector_cond(model, case, field) -> torch.Tensor:
    """Build the (1, cond_ch, H, W) conditioning tensor c = [field, residual, geom]
    in the same normalised space the corrector was trained in."""
    from neuroforge.solver.correction_loop import _residual_tensor

    normalizer = model.predictor.normalizer
    field_t = torch.from_numpy(
        np.ascontiguousarray(normalizer.norm_out(field.as_array()), DTYPE)
    )[None]
    residual_t = _residual_tensor(field, case, model.predictor)
    geom_t = torch.from_numpy(
        np.ascontiguousarray(normalizer.norm_in(encode_case(case)), DTYPE)
    )[None]
    return torch.cat([field_t, residual_t, geom_t], dim=1)


def measure_contraction(model, pairs, n_cases=24, max_iter_star=200, tol_star=1e-9,
                        n_iter=60, seed=0) -> dict:
    """Measure the empirical contraction factor of T_theta on real cases.

    For each case we:
      1. compute the high-precision fixed point delta* (damped solve, tight tol),
      2. re-iterate the PURE operator delta_{k+1}=T_theta(delta_k) from a random
         init, recording r_k = ||delta_k - delta*|| / ||delta_{k-1} - delta*||.
    The reported contraction factor is the median per-step ratio over the early
    (pre-floor) iterations across cases. Iterations-to-1e-5 is read from the
    relative distance ||delta_k - delta*|| / ||delta_0 - delta*||.
    """
    corrector = model.engine.corrector
    assert type(corrector).__name__ == "DEQCorrector", "need a DEQ corrector"
    device = next(corrector.parameters()).device
    corrector.eval()
    kappa = float(corrector.lipschitz)
    g = torch.Generator(device="cpu").manual_seed(seed)

    per_case_ratios = []     # list of arrays r_k
    per_case_reldist = []    # list of arrays ||delta_k-delta*||/||delta_0-delta*||
    iters_to_1e5 = []
    fp_residuals = []        # ||T(delta*)-delta*||/||delta*|| sanity
    used = 0

    for case, _truth in pairs:
        if used >= n_cases:
            break
        # one-shot backbone field as the conditioning anchor
        field = model.predictor.predict(case)
        cond = _corrector_cond(model, case, field).to(device)

        with torch.no_grad():
            # --- high-precision fixed point via the damped forward solve ---
            b, _, h, w = cond.shape
            delta = cond.new_zeros(b, corrector.out_channels, h, w)
            for _ in range(max_iter_star):
                t = corrector._operator(delta, cond)
                nxt = (1.0 - corrector.damping) * delta + corrector.damping * t
                step = float((nxt - delta).norm())
                rel = step / (float(delta.norm()) + 1e-12)
                delta = nxt
                if rel < tol_star:
                    break
            delta_star = delta
            # fixed-point sanity on the PURE operator: T(delta*) ~ delta*
            t_star = corrector._operator(delta_star, cond)
            fp_res = float((t_star - delta_star).norm() / (delta_star.norm() + 1e-12))
            if not np.isfinite(fp_res):
                continue
            fp_residuals.append(fp_res)

            # --- iterate the PURE operator from a random init ---
            d = torch.randn(delta_star.shape, generator=g).to(device, dtype=delta_star.dtype)
            d0_dist = float((d - delta_star).norm())
            if not np.isfinite(d0_dist) or d0_dist < 1e-12:
                continue
            dists = [d0_dist]
            for _ in range(n_iter):
                d = corrector._operator(d, cond)
                dist = float((d - delta_star).norm())
                if not np.isfinite(dist):
                    break
                dists.append(dist)
            dists = np.asarray(dists, np.float64)

        rel = dists / dists[0]
        ratios = dists[1:] / np.maximum(dists[:-1], 1e-30)
        per_case_ratios.append(ratios)
        per_case_reldist.append(rel)

        # iterations to reach rel distance 1e-5 (None -> not reached within budget)
        below = np.where(rel <= 1e-5)[0]
        iters_to_1e5.append(int(below[0]) if below.size else -1)
        used += 1

    # Aggregate. Use only ratios from the geometric regime: distances well above
    # the fixed-point floor (rel dist > 100x the tightest achieved), to avoid the
    # ratio polluting near machine/solve precision.
    early_ratios = []
    for ratios, rel in zip(per_case_ratios, per_case_reldist):
        floor = max(rel.min(), 1e-12)
        keep = rel[:-1] > 100.0 * floor   # ratios r_k correspond to step k
        if keep.any():
            early_ratios.append(ratios[keep])
    early = np.concatenate(early_ratios) if early_ratios else np.concatenate(per_case_ratios)
    early = early[np.isfinite(early)]

    # mean convergence curve (rel distance) padded to common length
    maxlen = max(len(r) for r in per_case_reldist)
    grid = np.full((len(per_case_reldist), maxlen), np.nan)
    for i, rel in enumerate(per_case_reldist):
        grid[i, : len(rel)] = rel
    mean_curve = np.nanmedian(grid, axis=0)

    reached = [i for i in iters_to_1e5 if i >= 0]
    return {
        "n_cases": used,
        "kappa_design": kappa,
        "contraction_factor_median": float(np.median(early)),
        "contraction_factor_mean": float(np.mean(early)),
        "contraction_factor_p90": float(np.percentile(early, 90)),
        "contraction_factor_max": float(np.max(early)),
        "fp_residual_median": float(np.median(fp_residuals)),
        "iters_to_1e-5_median": (float(np.median(reached)) if reached else None),
        "iters_to_1e-5_fraction_reached": len(reached) / max(used, 1),
        "n_iter_budget": n_iter,
        "_mean_curve": mean_curve.tolist(),
        "_per_case_reldist": [r.tolist() for r in per_case_reldist],
    }


def plot_h5(h5: dict) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: per-case convergence curves + median
    for rel in h5["_per_case_reldist"]:
        ax[0].semilogy(range(len(rel)), rel, color="0.8", lw=0.8)
    mc = np.asarray(h5["_mean_curve"])
    ax[0].semilogy(range(len(mc)), mc, color="C0", lw=2.2, label="median over cases")
    k = h5["kappa_design"]
    cf = h5["contraction_factor_median"]
    xs = np.arange(len(mc))
    ax[0].semilogy(xs, k ** xs, "k--", lw=1.2, label=f"kappa^k (design kappa={k:.2f})")
    ax[0].semilogy(xs, cf ** xs, "C3:", lw=1.4, label=f"measured rate^k ({cf:.3f})")
    ax[0].axhline(1e-5, color="0.5", lw=0.8, ls=":")
    ax[0].set_xlabel("iteration k")
    ax[0].set_ylabel(r"$\||\delta_k-\delta^*\||\,/\,\||\delta_0-\delta^*\||$")
    ax[0].set_title("H5: DEQ fixed-point convergence (pure $T_\\theta$)")
    ax[0].legend(fontsize=8)
    ax[0].set_ylim(1e-7, 2)

    # right: per-step ratio distribution
    ratios = []
    for rel in h5["_per_case_reldist"]:
        rel = np.asarray(rel)
        r = rel[1:] / np.maximum(rel[:-1], 1e-30)
        floor = max(rel.min(), 1e-12)
        r = r[rel[:-1] > 100.0 * floor]
        ratios.append(r)
    allr = np.concatenate(ratios) if ratios else np.array([cf])
    allr = allr[np.isfinite(allr)]
    ax[1].hist(allr, bins=30, color="C0", alpha=0.8)
    ax[1].axvline(cf, color="C3", lw=2, label=f"median = {cf:.3f}")
    ax[1].axvline(1.0, color="k", lw=1.2, ls="--", label="contraction threshold (1.0)")
    ax[1].set_xlabel(r"per-step ratio $\||\delta_{k+1}-\delta^*\||/\||\delta_k-\delta^*\||$")
    ax[1].set_ylabel("count")
    ax[1].set_title("H5: measured contraction ratios")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGS, "h5_contraction_curve.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- #
# H4 — split-conformal coverage (per channel) on real AirfRANS
# ---------------------------------------------------------------------------- #
def run_h4(model, calib_pairs, test_pairs, alpha=0.1, n_mc=16) -> dict:
    from neuroforge.models.ensemble import MCDropoutUQ
    from neuroforge.physics.calibration import (
        ConformalCalibrator,
        cases_to_error_sigma,
        reliability,
    )

    base = MCDropoutUQ(model._model)

    class _UQ:  # bind n_samples for the calibrator's fixed-arity call
        def predict_with_uncertainty(self, x):
            return base.predict_with_uncertainty(x, n_samples=n_mc)

    uq = _UQ()
    predict_fn = model.predictor.predict  # bound method -> __self__.normalizer resolves
    channels = {0: "u", 1: "v", 2: "p"}

    out = {"alpha": alpha, "n_mc": n_mc, "n_calib": len(calib_pairs),
           "n_test": len(test_pairs), "per_channel": {}}
    alphas = [0.30, 0.20, 0.15, 0.10, 0.05]
    cov_vs_alpha = {name: [] for name in channels.values()}
    reliab = {}

    for ci, name in channels.items():
        log(f"H4 channel {name}: extracting calib error/sigma ...")
        ce, cs, cm = cases_to_error_sigma(predict_fn, uq, calib_pairs, channel=ci)
        log(f"H4 channel {name}: extracting test error/sigma ...")
        te, ts, tm = cases_to_error_sigma(predict_fn, uq, test_pairs, channel=ci)

        cal = ConformalCalibrator(alpha=alpha).fit(ce, cs, cm)
        cov = cal.coverage(te, ts, tm)
        # raw (uncalibrated) coverage at the same nominal band q=z(1-alpha)
        z = float(__import__("scipy.special", fromlist=["erfinv"]).erfinv(1 - alpha) * np.sqrt(2)) \
            if _has_scipy() else 1.6448536269514722
        raw_cal = ConformalCalibrator(alpha=alpha)
        raw_cal.q, raw_cal.fitted = z, True
        cov_raw = raw_cal.coverage(te, ts, tm)

        out["per_channel"][name] = {
            "q": float(cal.q),
            "coverage_calibrated": float(cov),
            "coverage_raw_gaussian": float(cov_raw),
            "target": 1 - alpha,
            "sigma_mean": float(np.mean([np.mean(np.abs(s)) for s in cs])),
        }
        log(f"H4 channel {name}: q={cal.q:.3f} coverage={cov:.4f} (target {1-alpha:.2f}) "
            f"raw={cov_raw:.4f}")

        # coverage vs alpha (refit q at each alpha)
        for a in alphas:
            c = ConformalCalibrator(alpha=a).fit(ce, cs, cm)
            cov_vs_alpha[name].append(float(c.coverage(te, ts, tm)))

        # reliability (calibrated band anchored at 1-alpha)
        rel = reliability(te, ts, tm, n_bins=10, q=cal.q, alpha=alpha)
        reliab[name] = {"nominal": rel["nominal"].tolist(),
                        "empirical": rel["empirical"].tolist(),
                        "ece": rel["ece"]}

    out["coverage_vs_alpha"] = {"alphas": alphas, **cov_vs_alpha}
    out["reliability"] = reliab
    return out


def _has_scipy() -> bool:
    try:
        import scipy.special  # noqa: F401
        return True
    except Exception:
        return False


def plot_h4(h4: dict) -> tuple[str, str]:
    # reliability diagram
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name in ("u", "v", "p"):
        r = h4["reliability"][name]
        ax.plot(r["nominal"], r["empirical"], "-o", ms=4,
                label=f"{name} (ECE={r['ece']:.3f})")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("H4: reliability (conformal-calibrated band)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    p1 = os.path.join(FIGS, "h4_reliability.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # coverage vs alpha
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    alphas = np.asarray(h4["coverage_vs_alpha"]["alphas"])
    target = 1 - alphas
    ax.plot(alphas, target, "k--", lw=1, label="target 1-alpha")
    for name in ("u", "v", "p"):
        ax.plot(alphas, h4["coverage_vs_alpha"][name], "-o", ms=4, label=name)
    ax.set_xlabel("alpha")
    ax.set_ylabel("empirical coverage on test")
    ax.set_title("H4: coverage vs alpha (per channel)")
    ax.legend(fontsize=9)
    ax.invert_xaxis()
    fig.tight_layout()
    p2 = os.path.join(FIGS, "h4_coverage_vs_alpha.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    return p1, p2


def write_h4_csv(h4: dict) -> str:
    import csv
    path = os.path.join(RESULTS, "h4_coverage.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "alpha", "target", "q", "coverage_calibrated",
                    "coverage_raw_gaussian", "sigma_mean"])
        for name, d in h4["per_channel"].items():
            w.writerow([name, h4["alpha"], d["target"], f"{d['q']:.6f}",
                        f"{d['coverage_calibrated']:.6f}",
                        f"{d['coverage_raw_gaussian']:.6f}", f"{d['sigma_mean']:.6e}"])
    return path


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    model = get_model()

    # ----- load real AirfRANS test split, split into calib / test for H4 -----
    from neuroforge.data.airfrans_loader import load_airfrans

    log("loading AirfRANS test split from cache ...")
    test_all = load_airfrans(
        root="data", task=CONFIG["task"], train=False, resolution=CONFIG["resolution"],
        limit=CONFIG["n_val"], cache_dir="data/cache", download=False, progress=False,
    )
    log(f"loaded {len(test_all)} test pairs")
    rng = np.random.default_rng(CONFIG["seed"])
    idx = rng.permutation(len(test_all))
    half = len(test_all) // 2
    calib_pairs = [test_all[i] for i in idx[:half]]
    test_pairs = [test_all[i] for i in idx[half:]]

    # ============================== H5 ==============================
    log("=== H5: measuring DEQ contraction ===")
    h5 = measure_contraction(model, test_all, n_cases=24, seed=CONFIG["seed"])
    h5_fig = plot_h5(h5)
    h5_save = {k: v for k, v in h5.items() if not k.startswith("_")}
    h5_save["config"] = CONFIG
    h5_save["figure"] = os.path.relpath(h5_fig, ROOT).replace("\\", "/")
    with open(os.path.join(RESULTS, "h5_contraction.json"), "w") as f:
        json.dump(h5_save, f, indent=2)
    log(f"H5: contraction(median)={h5['contraction_factor_median']:.4f} "
        f"max={h5['contraction_factor_max']:.4f} "
        f"iters_to_1e-5(median)={h5_save['iters_to_1e-5_median']}")

    # ============================== H4 ==============================
    log("=== H4: split-conformal coverage ===")
    h4 = run_h4(model, calib_pairs, test_pairs, alpha=0.1, n_mc=16)
    h4_rel, h4_cva = plot_h4(h4)
    h4["config"] = CONFIG
    h4["figures"] = {
        "reliability": os.path.relpath(h4_rel, ROOT).replace("\\", "/"),
        "coverage_vs_alpha": os.path.relpath(h4_cva, ROOT).replace("\\", "/"),
    }
    with open(os.path.join(RESULTS, "h4_coverage.json"), "w") as f:
        json.dump(h4, f, indent=2)
    csv_path = write_h4_csv(h4)

    # ----- summary -----
    log("================= SUMMARY =================")
    log(f"H5 contraction factor (median) = {h5['contraction_factor_median']:.4f} "
        f"(<1 => PASS)  design kappa={h5['kappa_design']}")
    for name, d in h4["per_channel"].items():
        log(f"H4 {name}: coverage={d['coverage_calibrated']:.4f} (target 0.90), q={d['q']:.3f}")
    log(f"artifacts in {RESULTS}")
    log(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
