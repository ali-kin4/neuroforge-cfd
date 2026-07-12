"""DECISIVE real-data probe for the 'residual is a poor correction objective' theorem.

The monitored / differentiable steady-RANS residual (physics_residual_torch) is
continuity + momentum ONLY -- it omits the no-slip BC term. Its global minimiser
is therefore the trivial uniform-freestream field, not the viscous truth. This
script demonstrates that on REAL AirfRANS test fields by computing the monitored
discrete residual norm ||R_h|| (same non-dim scaling + solid + wall-ring masking
the verifier uses) on:

    (1) ground-truth field u*        -- the 'floor'
    (2) uniform-freestream field     -- expect ~0
    (3) [optional] a DEQ-FNO prediction u0_hat

We reuse PhysicsChecker.diagnose() so the masking (solid + wall-ring zeroing,
residuals.py:297-319) and per-equation non-dim scaling (residuals.py:326-331)
match the module / synthetic probe EXACTLY. The norm uses only the continuity
+ momentum components (NOT bc) because that is what the monitored objective sees.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

import neuroforge  # noqa: F401  -- caps BLAS/OMP threads BEFORE numpy/torch
from neuroforge.core.types import FlowField
from neuroforge.physics.residuals import PhysicsChecker

CACHE = "data/cache/airfrans_full_test_r128_n200.pkl"
OUT_JSON = "results/certificates/residual_floor_realdata.json"


def monitored_norm(diag) -> float:
    """RMS over fluid cells of the monitored residual (continuity + momentum only).

    diag.continuity / momentum_x / momentum_y are already the SCALED, solid- and
    wall-ring-zeroed maps produced by diagnose(). We deliberately exclude
    diag.bc_violation: the monitored objective omits the no-slip term.
    """
    cont = np.asarray(diag.continuity, dtype=np.float64)
    mx = np.asarray(diag.momentum_x, dtype=np.float64)
    my = np.asarray(diag.momentum_y, dtype=np.float64)
    m2 = cont * cont + mx * mx + my * my  # per-cell squared monitored magnitude
    return m2


def rms_over_fluid(m2: np.ndarray, fluid: np.ndarray) -> float:
    fb = fluid > 0.5
    if not np.any(fb):
        return 0.0
    return float(np.sqrt(np.mean(m2[fb])))


def _component_rms(diag, fluid: np.ndarray) -> tuple[float, float]:
    """Separate RMS of (continuity) and (momentum magnitude) over fluid cells.

    Used to show WHICH term dominates the truth floor: momentum dominance points
    at the omitted no-slip + closure (grad-nut) terms rather than pure
    continuity discretization.
    """
    fb = fluid > 0.5
    if not np.any(fb):
        return 0.0, 0.0
    cont = np.asarray(diag.continuity, dtype=np.float64)
    mx = np.asarray(diag.momentum_x, dtype=np.float64)
    my = np.asarray(diag.momentum_y, dtype=np.float64)
    r_cont = float(np.sqrt(np.mean(cont[fb] ** 2)))
    r_mom = float(np.sqrt(np.mean(mx[fb] ** 2 + my[fb] ** 2)))
    return r_cont, r_mom


def uniform_field(case, field) -> FlowField:
    """Constant freestream field on the same grid, reusing truth mask/sdf."""
    u_in, v_in = case.bc.inlet_vector()
    shape = field.shape
    u = np.full(shape, np.float32(u_in), dtype=np.float32)
    v = np.full(shape, np.float32(v_in), dtype=np.float32)
    p = np.zeros(shape, dtype=np.float32)
    nut = np.zeros(shape, dtype=np.float32)
    return FlowField(
        domain=field.domain, u=u, v=v, p=p, nut=nut,
        mask=field.mask, sdf=field.sdf, meta={"kind": "uniform"},
    )


def field_finite(field) -> bool:
    for a in (field.u, field.v, field.p):
        if a is None or not np.all(np.isfinite(a)):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of cases to use")
    ap.add_argument("--one", action="store_true", help="validate on a single case")
    ap.add_argument("--pred", action="store_true",
                    help="also compute the DEQ-FNO raw prediction residual (optional)")
    args = ap.parse_args()

    with open(CACHE, "rb") as fh:
        pairs = pickle.load(fh)
    n_total = len(pairs)
    n = min(args.n if not args.one else 1, n_total)

    predictor = None
    if args.pred:
        import time

        import neuroforge as nf
        t0 = time.time()
        model = nf.NeuroForge.load("checkpoints/certificates_deq.pt", device="cpu")
        predictor = model.predictor
        print(f"loaded DEQ-FNO checkpoint in {time.time() - t0:.1f}s")

    checker = PhysicsChecker()
    per_case = []
    skipped = []
    pred_norms = []

    for i in range(n):
        case, field = pairs[i]
        if not field_finite(field):
            skipped.append({"index": i, "name": case.name, "reason": "non-finite truth"})
            continue
        fluid = np.asarray(field.mask, dtype=np.float64)

        diag_truth = checker.diagnose(field, case)
        r_truth = rms_over_fluid(monitored_norm(diag_truth), fluid)
        r_cont_truth, r_mom_truth = _component_rms(diag_truth, fluid)

        uf = uniform_field(case, field)
        diag_unif = checker.diagnose(uf, case)
        r_unif = rms_over_fluid(monitored_norm(diag_unif), fluid)

        rec = {
            "index": i,
            "name": case.name,
            "u_inf": float(case.bc.u_inf),
            "aoa_deg": float(case.bc.aoa_deg),
            "reynolds": float(case.bc.reynolds),
            "ref_length": float(case.reference_length()),
            "norm_truth": r_truth,
            "norm_truth_continuity": r_cont_truth,
            "norm_truth_momentum": r_mom_truth,
            "norm_uniform": r_unif,
            "uniform_lt_truth": bool(r_unif < r_truth),
            # Ratio is undefined when the uniform norm is exactly 0; use null
            # (the boolean + 60/60 count carry the claim). null keeps the JSON
            # strict-spec valid (no non-standard Infinity token).
            "ratio_truth_over_uniform": (float(r_truth / r_unif) if r_unif > 0 else None),
        }

        if predictor is not None:
            pred_field = predictor.predict(case)  # raw FNO mean, physical units
            # diagnose with the predicted field but the TRUTH mask/sdf so masking
            # is identical across all three fields.
            pf = FlowField(
                domain=field.domain, u=pred_field.u, v=pred_field.v,
                p=pred_field.p, nut=pred_field.nut,
                mask=field.mask, sdf=field.sdf, meta={"kind": "deq_fno_pred"},
            )
            diag_pred = checker.diagnose(pf, case)
            r_pred = rms_over_fluid(monitored_norm(diag_pred), fluid)
            rec["norm_pred"] = r_pred
            rec["pred_lt_truth"] = bool(r_pred < r_truth)
            pred_norms.append(r_pred)

        per_case.append(rec)

        if args.one:
            print(f"[ONE] {case.name}")
            print(f"  ||R_h(truth)||   = {r_truth:.6e}")
            print(f"  ||R_h(uniform)|| = {r_unif:.6e}")
            print(f"  ratio truth/uniform = {r_truth / r_unif if r_unif>0 else float('inf'):.3e}")
            print(f"  uniform < truth? {r_unif < r_truth}")
            return

    rt = np.array([c["norm_truth"] for c in per_case], dtype=np.float64)
    ru = np.array([c["norm_uniform"] for c in per_case], dtype=np.float64)
    n_uniform_lt_truth = int(sum(c["uniform_lt_truth"] for c in per_case))

    rc = np.array([c["norm_truth_continuity"] for c in per_case], dtype=np.float64)
    rm = np.array([c["norm_truth_momentum"] for c in per_case], dtype=np.float64)

    # Ratio truth/uniform: when every uniform norm is exactly 0 (constant field
    # trivially satisfies continuity+momentum), the ratio is unbounded/undefined.
    # Report null (None) in that case to keep the JSON strict-spec valid.
    ratios = rt / np.where(ru > 0, ru, np.nan)
    mean_ratio = None if np.all(~np.isfinite(ratios)) else float(np.nanmean(ratios))

    verdict = (
        "SUPPORTS"
        if (rt.mean() > 10 * ru.mean() and n_uniform_lt_truth == len(per_case) and rt.mean() > 1e-3)
        else "MIXED_OR_WEAKENS"
    )

    have_pred = len(pred_norms) > 0
    rp = np.array(pred_norms, dtype=np.float64) if have_pred else None
    n_pred_lt_truth = (
        int(sum(c.get("pred_lt_truth", False) for c in per_case)) if have_pred else 0
    )

    out = {
        "artifact": "residual_floor_realdata",
        "purpose": (
            "Real-data demonstration that the monitored steady-RANS residual "
            "(continuity + momentum, no no-slip term) prefers the trivial "
            "uniform-freestream field over the viscous ground truth."
        ),
        "verdict": verdict,
        "metadata": {
            "dataset": "AirfRANS task=full test split",
            "cache": CACHE,
            "n_cases": len(per_case),
            "n_skipped": len(skipped),
            "resolution": list(pairs[0][1].shape),
            "fields_compared": (
                ["ground_truth", "uniform_freestream", "deq_fno_prediction"]
                if have_pred else ["ground_truth", "uniform_freestream"]
            ),
            "prediction_model": (
                "checkpoints/certificates_deq.pt (dropout-FNO backbone, raw mean prediction; "
                "DEQ corrector NOT applied)" if have_pred else None
            ),
            "monitored_residual": "continuity + momentum_x + momentum_y (BC term EXCLUDED, matching physics_residual_torch)",
            "norm": "RMS over fluid cells of sqrt(cont^2 + mom_x^2 + mom_y^2) on the SCALED residual maps",
            "non_dim_scaling": "continuity / (u_inf/L), momentum / (u_inf^2/L) -- per residuals.py:326-331 via PhysicsChecker.diagnose",
            "masking_policy": "residuals zeroed inside solid AND on the solid-adjacent fluid wall ring (residuals.py:297-319)",
            "nu_eff": "nu_laminar + nut, clamped >= 0 (truth nut from data; uniform nut=0)",
            "uniform_field": "constant (u,v)=inlet_vector, p=0, nut=0; reuses truth mask & sdf",
            "caveat": (
                "The truth floor ||R_h(u*)|| > 0 is multi-causal: (1) the monitored "
                "objective OMITS the no-slip BC penalty [the headline]; (2) it uses "
                "nu_eff*lap(u) and omits the spatially-varying-nut closure term "
                "grad(nut).grad(u), so it is nonzero on the exact continuum field; "
                "(3) coarse-grid (128^2) discretization/interpolation of an "
                "unresolvable Re~1e6 boundary layer. NONE of these rescue the "
                "objective: uniform=0 holds on the SAME grid regardless of all "
                "three, so apples-to-apples on the deployed grid the monitored "
                "residual still globally prefers the wrong (uniform) field over the "
                "viscous truth. The uniform field is an EXACT zero of "
                "continuity+momentum (all derivatives of a constant field vanish), "
                "independent of grid, Reynolds number, or closure."
            ),
        },
        "aggregate": {
            "norm_truth_mean": float(rt.mean()),
            "norm_truth_std": float(rt.std()),
            "norm_truth_median": float(np.median(rt)),
            "norm_uniform_mean": float(ru.mean()),
            "norm_uniform_std": float(ru.std()),
            "norm_uniform_median": float(np.median(ru)),
            "mean_ratio_truth_over_uniform": mean_ratio,
            "norm_truth_continuity_mean": float(rc.mean()),
            "norm_truth_momentum_mean": float(rm.mean()),
            "truth_momentum_over_continuity": (
                float(rm.mean() / rc.mean()) if rc.mean() > 0 else None
            ),
            "n_uniform_lt_truth": n_uniform_lt_truth,
            "frac_uniform_lt_truth": n_uniform_lt_truth / len(per_case),
            **(
                {
                    "norm_pred_mean": float(rp.mean()),
                    "norm_pred_std": float(rp.std()),
                    "norm_pred_median": float(np.median(rp)),
                    "n_pred_lt_truth": n_pred_lt_truth,
                    "frac_pred_lt_truth": n_pred_lt_truth / len(rp),
                } if have_pred else {}
            ),
        },
        "per_case": per_case,
        "skipped": skipped,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        # allow_nan=False enforces strict, spec-valid JSON (no Infinity/NaN tokens).
        json.dump(out, fh, indent=2, allow_nan=False)

    print(f"VERDICT: {verdict}")
    print(f"n_cases={len(per_case)} skipped={len(skipped)}")
    print(f"||R_h(truth)||   mean={rt.mean():.4e} std={rt.std():.4e} median={np.median(rt):.4e}")
    print(f"||R_h(uniform)|| mean={ru.mean():.4e} std={ru.std():.4e} median={np.median(ru):.4e}")
    print(f"uniform < truth: {n_uniform_lt_truth}/{len(per_case)}")
    _mr = "undefined (all uniform norms == 0)" if mean_ratio is None else f"{mean_ratio:.3e}"
    print(f"mean ratio truth/uniform = {_mr}")
    print(f"truth momentum/continuity = {rm.mean()/rc.mean():.2f}" if rc.mean() > 0 else "")
    if have_pred:
        print(f"||R_h(pred)||    mean={rp.mean():.4e} std={rp.std():.4e} median={np.median(rp):.4e}")
        print(f"pred < truth: {n_pred_lt_truth}/{len(rp)}")
    print(f"wrote {OUT_JSON}")

    _make_figure(rt, ru, rp if have_pred else None)


def _make_figure(rt, ru, rp) -> None:
    """Bar chart of mean monitored residual: truth vs uniform vs (optional) pred."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"skipping figure (matplotlib unavailable: {exc!r})")
        return

    labels = ["ground\ntruth $u^*$", "uniform\nfreestream"]
    means = [rt.mean(), ru.mean()]
    errs = [rt.std(), ru.std()]
    colors = ["#2c7fb8", "#d95f02"]
    if rp is not None:
        labels.append("DEQ-FNO\nprediction $\\hat u_0$")
        means.append(rp.mean())
        errs.append(rp.std())
        colors.append("#7570b3")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=errs, capsize=5, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"monitored residual $\|R_h\|$  (non-dim, RMS over fluid)")
    ax.set_title("Real AirfRANS: the monitored residual prefers\nwrong fields over the viscous truth")
    for x, m in zip(xs, means):
        ax.text(x, m + max(means) * 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.15)
    fig.tight_layout()
    out_fig = "results/figures/residual_floor.png"
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
