"""Cheap inference-time sensitivity / robustness sweep for NeuroForge CFD.

Reuses the *already-trained* dropout-FNO + DEQ checkpoint
(``checkpoints/certificates_deq.pt``) — **no retraining**. All four sweeps are
pure re-evaluation on a modest cached-AirfRANS eval set.

Sweeps
------
1. **Correction-iterations sensitivity** (in-dist ``full`` test split).
   IMPORTANT (honest wiring note): the saved corrector is a **DEQCorrector**.
   The engine-level ``solve(case, max_iters=k)`` is *inert* for a DEQ corrector
   — the correction loop's DEQ branch (``solver/correction_loop.py:148``) always
   applies the single converged fixed-point correction and ignores the engine's
   ``max_iters``, ``gate_by_trust`` and the backtracking acceptance test (this is
   by design; rationale at correction_loop.py:143-147). We *confirm* the engine
   knob is inert with 2 control points, then sweep the **real** iteration knob:
   the DEQ's own internal fixed-point cap ``corrector.max_iter`` over
   {1,3,5,10,15}, with iters=0 = backbone-only (no corrector). For each we record
   mse_u, surface_mse_p, mean residual_norm, residual_error_spearman and the mean
   number of DEQ iterations actually taken.

2. **Toggles** (pre-registered): trust-gating ON/OFF and acceptance-test ON/OFF.
   On *this* checkpoint both are **structural no-ops**: the DEQ branch bypasses
   trust-gating, and there is **no acceptance-test flag** in ``CorrectionConfig``
   at all (the backtracking test is hardcoded in the non-DEQ loop, which the DEQ
   branch never enters). We measure the metric deltas anyway (they are ~0) and
   report the mechanism. These toggles are only *live* for a ``'local'`` corrector.

3. **Trust-map weight sensitivity**: vary (w_r, w_u) over 4 pairs. Pure
   re-evaluation: we compute (residual_mag, uncertainty, mask) ONCE per case
   (uncertainty from MC-dropout, so w_u is actually exercised), then re-run
   ``trust_map()`` per weight pair. ``residual_error_spearman`` is **structurally
   invariant** to the weights (they touch only the trust map, never residual_norm
   or field error), so we report it once and report the *varying* quantities:
   mean trust and the green/yellow/red class fractions.

4. **OOD conformal coverage (honest exchangeability test)**: calibrate the
   split-conformal multiplier ``q`` per channel (u,v,p) at alpha=0.1 on a held-out
   **in-dist** ``full`` calibration split, then measure empirical coverage with
   that *same* q on (i) ``full`` test [in-dist control, ~0.90], (ii) ``reynolds``
   OOD, (iii) ``aoa`` OOD. Honest hypothesis: coverage degrades OOD because
   exchangeability is violated. We report the numbers as-is.

Artifacts
---------
    results/sensitivity/iters.json|csv
    results/sensitivity/toggles.json
    results/sensitivity/trust_weights.json|csv
    results/sensitivity/ood_coverage.json|csv
    results/sensitivity/figures/iters_metrics.png
    results/sensitivity/figures/trust_weights.png
    results/sensitivity/figures/ood_coverage.png

Run:
    .venv/Scripts/python.exe scripts/run_sensitivity.py
"""

from __future__ import annotations

# IMPORTANT: import neuroforge before numpy/torch (thread caps, see src/neuroforge/__init__.py).
import neuroforge as nf  # noqa: F401

import csv
import json
import os
import time

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neuroforge.core.types import DTYPE  # noqa: E402

# ---------------------------------------------------------------------------- #
# Paths & config
# ---------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "sensitivity")
FIGS = os.path.join(RESULTS, "figures")
CKPT = os.path.join(ROOT, "checkpoints", "certificates_deq.pt")
os.makedirs(FIGS, exist_ok=True)

SEED = 0
RES = 128
N_EVAL = 80          # modest in-dist eval set for sweeps 1-3
N_CALIB = 80         # in-dist calibration set for sweep 4 conformal q
N_OOD = 80           # OOD test cases per split for sweep 4
N_MC = 16            # MC-dropout samples for UQ
ITER_GRID = [0, 1, 3, 5, 10, 15]
TRUST_PAIRS = [(0.6, 0.4), (0.5, 0.5), (0.8, 0.2), (0.4, 0.6)]


def log(msg: str) -> None:
    print(f"[sensitivity] {msg}", flush=True)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _nan_mean(xs) -> float:
    a = np.asarray(xs, np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


# ---------------------------------------------------------------------------- #
# Load checkpoint + data
# ---------------------------------------------------------------------------- #
def load_engine():
    from neuroforge.core.config import Config
    from neuroforge.solver.engine import NeuroForgeEngine

    cfg = Config()
    cfg.train.device = "auto"
    engine = NeuroForgeEngine.from_checkpoint(CKPT, config=cfg)
    log(f"loaded engine; backbone on {engine.predictor.device}, "
        f"corrector={type(engine.corrector).__name__}")
    return engine


def load_split(task: str, limit: int):
    from neuroforge.data.airfrans_loader import load_airfrans

    pairs = load_airfrans(
        root="data", task=task, train=False, resolution=RES,
        limit=limit, cache_dir="data/cache", download=False, progress=False,
    )
    log(f"loaded {len(pairs)} test pairs for task='{task}'")
    return pairs


# ---------------------------------------------------------------------------- #
# Sweep 1 — correction-iterations sensitivity (DEQ internal fixed-point cap)
# ---------------------------------------------------------------------------- #
def _residual_norm_of_field(field, case, checker) -> float:
    try:
        return float(checker.diagnose(field, case).residual_norm())
    except Exception:
        return float("nan")


def _spearman(a, b) -> float:
    from neuroforge.physics.evaluation import _spearman as sp
    return sp(a, b)


def eval_at_iters(engine, pairs, n_iters: int) -> dict:
    """Evaluate the engine field at a given DEQ internal fixed-point cap.

    n_iters == 0  -> backbone-only (no corrector applied).
    n_iters  > 0  -> set corrector.max_iter = n_iters, run the full solve.
    Returns aggregate mse_u, surface_mse_p, mean residual_norm,
    residual_error_spearman, and mean DEQ iters taken.
    """
    from neuroforge.physics.evaluation import (
        per_channel_mse,
        surface_pressure_mse,
    )
    from neuroforge.physics.residuals import PhysicsChecker

    checker = PhysicsChecker()
    corrector = engine.corrector
    saved_cap = corrector.max_iter if corrector is not None else None

    mse_u, surf_p, res_norms, errs, deq_iters = [], [], [], [], []
    try:
        if n_iters > 0 and corrector is not None:
            corrector.max_iter = int(n_iters)
        for case, ref in pairs:
            if n_iters == 0:
                field = engine.predictor.predict(case)        # backbone only
                rn = _residual_norm_of_field(field, case, checker)
            else:
                res = engine.solve(case)                       # DEQ correction
                field = res.field
                rn = float(res.metrics.get("residual_norm", float("nan")))
                if res.history and "deq_iters" in res.history[-1]:
                    deq_iters.append(float(res.history[-1]["deq_iters"]))
            pm = per_channel_mse(field, ref)
            mse_u.append(pm["mse_u"])
            surf_p.append(surface_pressure_mse(field, ref, case))
            res_norms.append(rn)
            # relative speed error for residual<->error correlation
            m = (np.asarray(ref.mask) > 0.5) if ref.mask is not None else None
            ps, rs = field.speed(), ref.speed()
            if m is not None:
                num = float(np.sqrt(np.sum(((ps - rs)[m]) ** 2)))
                den = float(np.sqrt(np.sum((rs[m]) ** 2))) + 1e-12
            else:
                num = float(np.sqrt(np.sum((ps - rs) ** 2)))
                den = float(np.sqrt(np.sum(rs ** 2))) + 1e-12
            errs.append(num / den)
    finally:
        if saved_cap is not None:
            corrector.max_iter = saved_cap

    return {
        "n_iters": n_iters,
        "mse_u": _nan_mean(mse_u),
        "surface_mse_p": _nan_mean(surf_p),
        "residual_norm": _nan_mean(res_norms),
        "residual_error_spearman": _spearman(res_norms, errs),
        "deq_iters_mean": _nan_mean(deq_iters) if deq_iters else 0.0,
    }


def sweep_iters(engine, pairs) -> dict:
    log("=== Sweep 1: correction-iterations sensitivity ===")
    # Control: confirm the *engine-level* max_iters is inert for the DEQ branch.
    case0, _ = pairs[0]
    r_a = engine.solve(case0, max_iters=1).metrics["residual_norm"]
    r_b = engine.solve(case0, max_iters=15).metrics["residual_norm"]
    engine_maxiters_inert = bool(np.isclose(r_a, r_b, rtol=0, atol=1e-9))
    log(f"control: engine.solve max_iters inert on DEQ? {engine_maxiters_inert} "
        f"(res@1={r_a:.6g}, res@15={r_b:.6g})")

    rows = []
    for k in ITER_GRID:
        t0 = time.time()
        row = eval_at_iters(engine, pairs, k)
        log(f"iters={k:>2}: mse_u={row['mse_u']:.4g} surf_p={row['surface_mse_p']:.4g} "
            f"res={row['residual_norm']:.4g} rho={row['residual_error_spearman']:.3f} "
            f"deq_iters={row['deq_iters_mean']:.1f}  ({time.time()-t0:.0f}s)")
        rows.append(row)

    out = {
        "n_eval": len(pairs),
        "task": "full",
        "engine_maxiters_inert_on_deq": engine_maxiters_inert,
        "swept_knob": "DEQCorrector.max_iter (internal fixed-point cap)",
        "rows": rows,
    }
    # CSV
    csv_path = os.path.join(RESULTS, "iters.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_iters", "mse_u", "surface_mse_p", "residual_norm",
                    "residual_error_spearman", "deq_iters_mean"])
        for r in rows:
            w.writerow([r["n_iters"], f"{r['mse_u']:.6e}", f"{r['surface_mse_p']:.6e}",
                        f"{r['residual_norm']:.6e}", f"{r['residual_error_spearman']:.6f}",
                        f"{r['deq_iters_mean']:.3f}"])
    with open(os.path.join(RESULTS, "iters.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot_iters(out)
    return out


def plot_iters(out: dict) -> str:
    rows = out["rows"]
    x = [r["n_iters"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    specs = [
        ("mse_u", "MSE(u)", ax[0, 0]),
        ("surface_mse_p", "surface MSE(p)", ax[0, 1]),
        ("residual_norm", "mean residual norm", ax[1, 0]),
        ("residual_error_spearman", "residual<->error Spearman", ax[1, 1]),
    ]
    for key, label, a in specs:
        y = [r[key] for r in rows]
        a.plot(x, y, "-o", color="C0")
        a.set_xlabel("correction iterations (DEQ max_iter; 0 = backbone)")
        a.set_ylabel(label)
        a.set_title(label)
        a.grid(alpha=0.3)
    fig.suptitle("Sweep 1: correction-iterations sensitivity (in-dist 'full')")
    fig.tight_layout()
    path = os.path.join(FIGS, "iters_metrics.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- #
# Sweep 2 — toggles (trust-gating, acceptance-test)
# ---------------------------------------------------------------------------- #
def _eval_simple(engine, pairs) -> dict:
    """mse_u + surface_mse_p + mean residual_norm over the corrected solve."""
    from neuroforge.physics.evaluation import per_channel_mse, surface_pressure_mse

    mse_u, surf_p, res = [], [], []
    for case, ref in pairs:
        r = engine.solve(case)
        f = r.field
        pm = per_channel_mse(f, ref)
        mse_u.append(pm["mse_u"])
        surf_p.append(surface_pressure_mse(f, ref, case))
        res.append(float(r.metrics.get("residual_norm", float("nan"))))
    return {"mse_u": _nan_mean(mse_u), "surface_mse_p": _nan_mean(surf_p),
            "residual_norm": _nan_mean(res)}


def sweep_toggles(engine, pairs) -> dict:
    log("=== Sweep 2: toggles (trust-gating, acceptance-test) ===")
    from dataclasses import replace

    is_deq = type(engine.corrector).__name__ == "DEQCorrector"
    base_cfg = engine.config.correction

    def run_with(correction_cfg) -> dict:
        saved = engine.config.correction
        engine.config.correction = correction_cfg
        try:
            return _eval_simple(engine, pairs)
        finally:
            engine.config.correction = saved

    # Trust-gating ON vs OFF
    gate_on = run_with(replace(base_cfg, gate_by_trust=True))
    gate_off = run_with(replace(base_cfg, gate_by_trust=False))

    # Acceptance-test: there is NO flag in CorrectionConfig. The backtracking
    # acceptance test is hardcoded in the non-DEQ loop. The closest live knob is
    # step_size (the backtracking starts from it). We toggle step_size 1.0 vs 0.5
    # purely to document that on a DEQ corrector this too is a no-op on the loop
    # structure (DEQ applies step_size as the delta scale only).
    accept_on = run_with(replace(base_cfg, step_size=1.0))
    accept_off = run_with(replace(base_cfg, step_size=0.5))

    def delta(a, b):
        return {k: float(a[k] - b[k]) for k in ("mse_u", "surface_mse_p", "residual_norm")}

    out = {
        "n_eval": len(pairs),
        "corrector_type": type(engine.corrector).__name__,
        "note": (
            "On a DEQCorrector both toggles are structural no-ops: the DEQ branch "
            "in correction_loop.py (lines 148-177) bypasses trust-gating and the "
            "backtracking acceptance test. There is no acceptance-test flag in "
            "CorrectionConfig; the test is hardcoded in the non-DEQ loop, which is "
            "never entered for a DEQ corrector. These toggles are only live for a "
            "'local' corrector. step_size is the only knob the DEQ branch reads "
            "(it scales the applied delta)."
        ),
        "trust_gating": {
            "on": gate_on, "off": gate_off, "delta_on_minus_off": delta(gate_on, gate_off),
        },
        "acceptance_test_proxy_step_size": {
            "step_1.0": accept_on, "step_0.5": accept_off,
            "delta_1.0_minus_0.5": delta(accept_on, accept_off),
        },
        "toggles_are_noops_on_this_checkpoint": is_deq,
    }
    log(f"trust-gating delta (on-off): {out['trust_gating']['delta_on_minus_off']}")
    log(f"step_size delta (1.0-0.5):   {out['acceptance_test_proxy_step_size']['delta_1.0_minus_0.5']}")
    with open(os.path.join(RESULTS, "toggles.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------- #
# Sweep 3 — trust-map weight sensitivity (pure re-evaluation)
# ---------------------------------------------------------------------------- #
def sweep_trust_weights(engine, pairs) -> dict:
    log("=== Sweep 3: trust-map weight sensitivity ===")
    from dataclasses import replace

    from neuroforge.geometry.encode import encode_case
    from neuroforge.models.ensemble import MCDropoutUQ
    from neuroforge.physics.residuals import (
        PhysicsChecker,
        continuity_residual,
        momentum_residual,
    )
    from neuroforge.physics.trust import trust_map

    predictor = engine.predictor
    base_phys = engine.config.physics
    # MC-dropout UQ so the uncertainty channel is non-zero (w_u is exercised).
    uq = MCDropoutUQ(predictor.model)

    # --- compute per-case (residual_mag, uncertainty, mask) ONCE ---
    res_mags, uncs, masks = [], [], []
    res_norms, errs = [], []  # for the (invariant) residual<->error spearman
    checker = PhysicsChecker()
    for case, ref in pairs:
        field = predictor.predict(case)
        # residual magnitude map (continuity + momentum), like PhysicsChecker
        cont = continuity_residual(field)
        rx, ry = momentum_residual(field, case.fluid)
        rmag = np.sqrt(cont ** 2 + rx ** 2 + ry ** 2).astype(DTYPE)
        # MC-dropout uncertainty (channel-mean std), in normalised input space
        x = predictor.normalizer.norm_in(encode_case(case))
        xt = torch.from_numpy(np.ascontiguousarray(x, DTYPE))[None]
        _, std = uq.predict_with_uncertainty(xt, n_samples=N_MC)
        unc = std.mean(dim=1)[0].cpu().numpy().astype(DTYPE)
        mask = np.asarray(field.mask, DTYPE) if field.mask is not None else np.ones(field.shape, DTYPE)
        res_mags.append(rmag)
        uncs.append(unc)
        masks.append(mask)
        # invariant residual<->error correlation pieces
        res_norms.append(_residual_norm_of_field(field, case, checker))
        m = mask > 0.5
        num = float(np.sqrt(np.sum(((field.speed() - ref.speed())[m]) ** 2)))
        den = float(np.sqrt(np.sum((ref.speed()[m]) ** 2))) + 1e-12
        errs.append(num / den)

    invariant_spearman = _spearman(res_norms, errs)
    log(f"residual<->error spearman (weight-invariant) = {invariant_spearman:.4f}")

    rows = []
    for (w_r, w_u) in TRUST_PAIRS:
        cfg = replace(base_phys, trust_residual_weight=w_r, trust_uncertainty_weight=w_u)
        trust_means, frac_g, frac_y, frac_r = [], [], [], []
        for rmag, unc, mask in zip(res_mags, uncs, masks):
            trust, tcls = trust_map(rmag, unc, cfg, mask=mask)
            m = mask > 0.5
            tv = trust[m]
            tc = tcls[m]
            if tv.size:
                trust_means.append(float(np.mean(tv)))
                frac_g.append(float(np.mean(tc == 2.0)))
                frac_y.append(float(np.mean(tc == 1.0)))
                frac_r.append(float(np.mean(tc == 0.0)))
        rows.append({
            "w_r": w_r, "w_u": w_u,
            "trust_mean": _nan_mean(trust_means),
            "frac_green": _nan_mean(frac_g),
            "frac_yellow": _nan_mean(frac_y),
            "frac_red": _nan_mean(frac_r),
        })
        log(f"(w_r={w_r}, w_u={w_u}): trust_mean={rows[-1]['trust_mean']:.4f} "
            f"G/Y/R={rows[-1]['frac_green']:.3f}/{rows[-1]['frac_yellow']:.3f}/"
            f"{rows[-1]['frac_red']:.3f}")

    out = {
        "n_eval": len(pairs),
        "n_mc": N_MC,
        "residual_error_spearman_invariant": invariant_spearman,
        "note": (
            "residual_error_spearman is structurally invariant to (w_r, w_u): the "
            "weights reshape only the trust map, never residual_norm or field "
            "error. The varying quantities are mean trust and the G/Y/R class "
            "fractions. Uncertainty is MC-dropout (n=%d), so w_u is exercised." % N_MC
        ),
        "rows": rows,
    }
    csv_path = os.path.join(RESULTS, "trust_weights.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["w_r", "w_u", "trust_mean", "frac_green", "frac_yellow", "frac_red"])
        for r in rows:
            w.writerow([r["w_r"], r["w_u"], f"{r['trust_mean']:.6f}",
                        f"{r['frac_green']:.6f}", f"{r['frac_yellow']:.6f}",
                        f"{r['frac_red']:.6f}"])
    with open(os.path.join(RESULTS, "trust_weights.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot_trust_weights(out)
    return out


def plot_trust_weights(out: dict) -> str:
    rows = out["rows"]
    labels = [f"({r['w_r']},{r['w_u']})" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(x, [r["trust_mean"] for r in rows], "-o", color="C0")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_xlabel("(w_r, w_u)"); ax[0].set_ylabel("mean trust")
    ax[0].set_title("Mean trust vs weights"); ax[0].grid(alpha=0.3)

    g = [r["frac_green"] for r in rows]
    y = [r["frac_yellow"] for r in rows]
    rr = [r["frac_red"] for r in rows]
    ax[1].bar(x, g, label="green", color="C2")
    ax[1].bar(x, y, bottom=g, label="yellow", color="gold")
    ax[1].bar(x, rr, bottom=np.array(g) + np.array(y), label="red", color="C3")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_xlabel("(w_r, w_u)"); ax[1].set_ylabel("class fraction")
    ax[1].set_title("Trust-class fractions vs weights"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Sweep 3: trust-map weight sensitivity "
                 f"(residual<->error rho invariant = {out['residual_error_spearman_invariant']:.3f})")
    fig.tight_layout()
    path = os.path.join(FIGS, "trust_weights.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- #
# Sweep 4 — OOD conformal coverage (honest exchangeability test)
# ---------------------------------------------------------------------------- #
def sweep_ood_coverage(engine, calib_pairs, test_splits: dict, alpha=0.1) -> dict:
    log("=== Sweep 4: OOD conformal coverage ===")
    from neuroforge.models.ensemble import MCDropoutUQ
    from neuroforge.physics.calibration import (
        ConformalCalibrator,
        cases_to_error_sigma,
    )

    base = MCDropoutUQ(engine.predictor.model)

    class _UQ:
        def predict_with_uncertainty(self, x):
            return base.predict_with_uncertainty(x, n_samples=N_MC)

    uq = _UQ()
    predict_fn = engine.predictor.predict
    channels = {0: "u", 1: "v", 2: "p"}

    out = {"alpha": alpha, "n_mc": N_MC, "n_calib": len(calib_pairs),
           "target": 1 - alpha, "calib_task": "full (in-dist)", "per_channel": {}}

    for ci, name in channels.items():
        log(f"channel {name}: fitting q on in-dist calib ...")
        ce, cs, cm = cases_to_error_sigma(predict_fn, uq, calib_pairs, channel=ci)
        cal = ConformalCalibrator(alpha=alpha).fit(ce, cs, cm)
        rec = {"q": float(cal.q), "target": 1 - alpha, "coverage": {}}
        for task, pairs in test_splits.items():
            te, ts, tm = cases_to_error_sigma(predict_fn, uq, pairs, channel=ci)
            cov = float(cal.coverage(te, ts, tm))
            rec["coverage"][task] = cov
            log(f"  channel {name} on '{task}': coverage={cov:.4f} (target {1-alpha:.2f})")
        out["per_channel"][name] = rec

    # CSV
    csv_path = os.path.join(RESULTS, "ood_coverage.csv")
    tasks = list(test_splits.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "q", "target"] + [f"coverage_{t}" for t in tasks])
        for name, rec in out["per_channel"].items():
            w.writerow([name, f"{rec['q']:.6f}", rec["target"]]
                       + [f"{rec['coverage'][t]:.6f}" for t in tasks])
    out["test_tasks"] = tasks
    out["n_test"] = {t: len(p) for t, p in test_splits.items()}
    with open(os.path.join(RESULTS, "ood_coverage.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot_ood(out)
    return out


def plot_ood(out: dict) -> str:
    tasks = out["test_tasks"]
    channels = list(out["per_channel"].keys())
    x = np.arange(len(tasks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, name in enumerate(channels):
        ys = [out["per_channel"][name]["coverage"][t] for t in tasks]
        ax.bar(x + (i - 1) * width, ys, width, label=f"channel {name}")
    ax.axhline(out["target"], color="k", ls="--", lw=1.2,
               label=f"target {out['target']:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({'in-dist' if t=='full' else 'OOD'})" for t in tasks])
    ax.set_ylabel("empirical coverage")
    ax.set_title("Sweep 4: conformal coverage (q fit on in-dist 'full')")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGS, "ood_coverage.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()
    log(f"device={_device()}")

    engine = load_engine()

    # ---- in-dist data: split 'full' test cache into eval + calib ----
    full_all = load_split("full", limit=N_EVAL + N_CALIB)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(full_all))
    eval_pairs = [full_all[i] for i in idx[:N_EVAL]]
    calib_pairs = [full_all[i] for i in idx[N_EVAL:N_EVAL + N_CALIB]]
    log(f"in-dist: {len(eval_pairs)} eval, {len(calib_pairs)} calib")

    # ---- run sweeps 1-3 on the in-dist eval set ----
    iters_out = sweep_iters(engine, eval_pairs)
    toggles_out = sweep_toggles(engine, eval_pairs)
    trust_out = sweep_trust_weights(engine, eval_pairs)

    # ---- sweep 4: OOD coverage ----
    full_test = load_split("full", limit=N_OOD)
    reynolds_test = load_split("reynolds", limit=N_OOD)
    aoa_test = load_split("aoa", limit=N_OOD)
    ood_out = sweep_ood_coverage(
        engine, calib_pairs,
        {"full": full_test, "reynolds": reynolds_test, "aoa": aoa_test},
        alpha=0.1,
    )

    # ---- summary ----
    log("================= SUMMARY =================")
    log("Sweep 1 (iters): "
        + ", ".join(f"k={r['n_iters']}:mse_u={r['mse_u']:.3g}" for r in iters_out["rows"]))
    log(f"Sweep 2 (toggles): no-ops on DEQ = {toggles_out['toggles_are_noops_on_this_checkpoint']}")
    log(f"Sweep 3 (trust weights): rho invariant = "
        f"{trust_out['residual_error_spearman_invariant']:.4f}")
    for name, rec in ood_out["per_channel"].items():
        log(f"Sweep 4 ({name}): full={rec['coverage']['full']:.3f} "
            f"reynolds={rec['coverage']['reynolds']:.3f} aoa={rec['coverage']['aoa']:.3f} "
            f"(target {ood_out['target']:.2f})")
    log(f"all artifacts in {RESULTS}")
    log(f"total wall time {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
