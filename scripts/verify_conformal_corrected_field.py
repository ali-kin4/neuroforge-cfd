"""RE-VERIFY (light, CPU): certify the DEPLOYED (DEQ-corrected) field, via the
shipped calibration code path.

Closes the finding "the conformal certificate is calibrated on the RAW backbone,
not the CORRECTED field NeuroForge deploys". The earlier probe
(scripts/probe_conformal_after_deq.py) computed corrected errors *inline*; this
script instead routes EVERY number through the public, committed API added to
``neuroforge.physics.calibration``:

  * raw certificate     : cases_to_error_sigma(predictor.predict, ...)        [default path, unchanged]
  * corrected certificate: cases_to_error_sigma(corrected_predict_fn(engine),
                            ..., normalizer=engine.predictor.normalizer)       [NEW deployed-field path]

so the evidence and the shipped code are ONE artifact. We report, per channel:

  1. raw_certificate        : fit q on raw cal-err; coverage/ECE on raw test-err.
  2. frozen_q_on_corrected  : keep the raw-fit q; coverage/ECE on CORRECTED test-err
                              -> "does the cert issued for the raw field still hold
                                 once we ship the corrected field?" (the objection).
  3. recalibrated_corrected : fit q on CORRECTED cal-err; coverage/ECE on corrected
                              test-err -> "does calibrating on the deployed field
                                 restore the guarantee?" (the fix).

Lead metric is COVERAGE at (1-alpha) (that is the guarantee); ECE is secondary.

sigma is input-conditional (a function of the encoded case, not the predicted
field), so it is identical for raw vs corrected by construction -- only the error
map changes. That is exactly why a contraction guarantee on the residual cannot
imply coverage survival: it does not control |error|/sigma.

CPU only; set CUDA_VISIBLE_DEVICES="" before launch. import neuroforge first.
Does NOT touch results/mgn | checkpoints/mgn | mgn_run.log.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import neuroforge as nf  # noqa: F401  -- MUST precede numpy/torch (thread caps)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "certificates_deq.pt")
OUT = os.path.join(ROOT, "results", "certificates", "verify_conformal_corrected_field.json")

N_CASES = 30        # small set; split half cal / half test
N_MC = 12           # MC-dropout samples for sigma
ALPHA = 0.1
RES = 128
CHANNELS = {0: "u", 1: "v", 2: "p"}


def log(m: str) -> None:
    print(f"[verify] {m}", flush=True)


def main() -> None:
    from neuroforge.data.airfrans_loader import load_airfrans
    from neuroforge.models.ensemble import MCDropoutUQ
    from neuroforge.physics.calibration import (
        ConformalCalibrator,
        cases_to_error_sigma,
        corrected_predict_fn,
        reliability,
    )

    torch.manual_seed(0)
    np.random.seed(0)

    log(f"loading checkpoint {CKPT} on cpu ...")
    model = nf.NeuroForge.load(CKPT, device="cpu")
    corr_type = type(model.engine.corrector).__name__ if model.engine.corrector else "none"
    log(f"corrector = {corr_type}")
    assert corr_type == "DEQCorrector", "need the DEQ corrector for the deployed field"

    # Fixed-arity UQ wrapper: cases_to_error_sigma calls predict_with_uncertainty(x)
    # with no n_samples, so bind N_MC here. sigma is identical for raw/corrected.
    base = MCDropoutUQ(model._model)

    class _UQ:
        def predict_with_uncertainty(self, x):
            return base.predict_with_uncertainty(x, n_samples=N_MC)

    uq = _UQ()

    log("loading cached AirfRANS full test pairs (r128) ...")
    pool = load_airfrans(root="data", task="full", train=False, resolution=RES,
                         limit=200, cache_dir="data/cache", download=False,
                         progress=False)
    rng = np.random.default_rng(0)
    sel = rng.permutation(len(pool))[:N_CASES]
    pairs = [pool[i] for i in sel]
    half = len(pairs) // 2
    cal_pairs, test_pairs = pairs[:half], pairs[half:]
    log(f"using {len(pairs)} cases  ->  {len(cal_pairs)} cal / {len(test_pairs)} test")

    # ---- the two predict functions: raw backbone vs DEPLOYED corrected field ----
    raw_predict = model.predictor.predict                  # bound method (legacy default)
    corr_predict = corrected_predict_fn(model.engine)      # NEW: engine.solve(case).field
    norm = model.predictor.normalizer

    # Defensive: confirm the corrector actually changes the field (else probe is void).
    c0 = test_pairs[0][0]
    d = float(np.mean(np.abs(corr_predict(c0).as_array() - raw_predict(c0).as_array())))
    log(f"mean|corrected - raw| on a probe case = {d:.4e} (must be > 0)")
    assert d > 0, "corrector left the field unchanged -- nothing to certify differently"

    results = {
        "meta": {
            "checkpoint": "checkpoints/certificates_deq.pt",
            "backbone": "dropout-FNO (real AirfRANS, res128)",
            "corrector": corr_type,
            "code_path": "neuroforge.physics.calibration.cases_to_error_sigma + "
                         "corrected_predict_fn (the SHIPPED calibration API)",
            "n_cal": len(cal_pairs), "n_test": len(test_pairs),
            "n_mc": N_MC, "alpha": ALPHA, "target_coverage": 1 - ALPHA,
            "mean_abs_delta_probe_case": d,
            "note": "sigma is input-conditional (function of case, NOT field); identical "
                    "for raw and corrected by construction. Lead metric: coverage@(1-alpha).",
        },
        "per_channel": {},
    }

    for ci, name in CHANNELS.items():
        t0 = time.time()
        # RAW path (default, normalizer resolved from predict_fn.__self__).
        cal_raw_e, cal_raw_s, cal_raw_m = cases_to_error_sigma(raw_predict, uq, cal_pairs, channel=ci)
        te_raw_e, te_raw_s, te_raw_m = cases_to_error_sigma(raw_predict, uq, test_pairs, channel=ci)
        # CORRECTED path (NEW; closure -> pass normalizer= explicitly).
        cal_cor_e, cal_cor_s, cal_cor_m = cases_to_error_sigma(
            corr_predict, uq, cal_pairs, channel=ci, normalizer=norm)
        te_cor_e, te_cor_s, te_cor_m = cases_to_error_sigma(
            corr_predict, uq, test_pairs, channel=ci, normalizer=norm)

        # (1) raw certificate on raw field.
        raw_cal = ConformalCalibrator(alpha=ALPHA).fit(cal_raw_e, cal_raw_s, cal_raw_m)
        cov_raw = raw_cal.coverage(te_raw_e, te_raw_s, te_raw_m)
        ece_raw = reliability(te_raw_e, te_raw_s, te_raw_m, q=raw_cal.q, alpha=ALPHA)["ece"]

        # (2) frozen raw-q applied to the DEPLOYED corrected field (the objection).
        cov_frozen = raw_cal.coverage(te_cor_e, te_cor_s, te_cor_m)
        ece_frozen = reliability(te_cor_e, te_cor_s, te_cor_m, q=raw_cal.q, alpha=ALPHA)["ece"]

        # (3) recalibrate q on the corrected field (the fix).
        cor_cal = ConformalCalibrator(alpha=ALPHA).fit(cal_cor_e, cal_cor_s, cal_cor_m)
        cov_refit = cor_cal.coverage(te_cor_e, te_cor_s, te_cor_m)
        ece_refit = reliability(te_cor_e, te_cor_s, te_cor_m, q=cor_cal.q, alpha=ALPHA)["ece"]

        mae_raw = float(np.mean([np.mean(np.abs(e[np.asarray(m) > 0.5]))
                                 for e, m in zip(te_raw_e, te_raw_m)]))
        mae_cor = float(np.mean([np.mean(np.abs(e[np.asarray(m) > 0.5]))
                                 for e, m in zip(te_cor_e, te_cor_m)]))

        results["per_channel"][name] = {
            "field_mae_raw": mae_raw,
            "field_mae_corrected": mae_cor,
            "corrector_reduced_error": bool(mae_cor < mae_raw),
            "raw_certificate": {"q": raw_cal.q, "coverage": cov_raw, "ece": ece_raw},
            "frozen_q_on_corrected": {"q": raw_cal.q, "coverage": cov_frozen, "ece": ece_frozen},
            "recalibrated_corrected": {"q": cor_cal.q, "coverage": cov_refit, "ece": ece_refit},
        }
        log(f"[{name}] mae {mae_raw:.3f}->{mae_cor:.3f} | "
            f"raw cov={cov_raw:.3f} | frozen-q-on-corrected cov={cov_frozen:.3f} "
            f"ece={ece_frozen:.3f} | recalibrated cov={cov_refit:.3f} ece={ece_refit:.3f} "
            f"({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
