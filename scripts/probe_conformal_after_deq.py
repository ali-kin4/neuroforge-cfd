"""LIGHT CPU feasibility probe: does split-conformal coverage SURVIVE the DEQ corrector?

Loads the cached real-AirfRANS dropout-FNO backbone + DEQ corrector
(checkpoints/certificates_deq.pt), and on a SMALL set of cached test cases asks:
after the DEQ correction modifies the predicted field, does the conformal band
q*sigma still hold (1-alpha) coverage and stay calibrated (ECE)?

Key structural fact (and the thing any theorem must confront): sigma here is
input-conditional -- uq.predict_with_uncertainty(encoded_input) is a function of
the CASE, not the predicted field. The corrector changes the error map but NOT
sigma. So we compute sigma ONCE per case and reuse it for raw vs corrected.

We report a TRIAD per channel (so "survives" is unambiguous):
  1. raw certificate : fit q on (cal raw-err, sigma); cover (test raw-err, sigma).
     (sanity: should ~reproduce results/certificates/h4_coverage.json)
  2. frozen-q transfer: keep raw-fit q; cover (test CORRECTED-err, sigma).
     -> the literal "does the cert I issued for the raw field still hold post-correction".
  3. refit-q         : fit q on (cal corrected-err, sigma); cover (test corr-err, sigma).
     -> is the corrected field still CALIBRATABLE with the same machinery.
Plus ECE (reliability) for raw & corrected, and sharpness (mean band q*sigma vs mean|err|).

CPU only; set CUDA_VISIBLE_DEVICES="" before launch. import neuroforge first.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import neuroforge as nf  # noqa: F401  -- MUST precede numpy/torch (thread caps)
from neuroforge.core.types import DTYPE
from neuroforge.geometry.encode import encode_case

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "certificates_deq.pt")
OUT = os.path.join(ROOT, "results", "certificates", "probe_conformal_after_deq.json")

N_CASES = 40        # total small set; random split ~20 cal / 20 test
N_MC = 12           # MC-dropout samples for sigma (one pass per case)
ALPHA = 0.1
RES = 128
CHANNELS = {0: "u", 1: "v", 2: "p"}


def log(m: str) -> None:
    print(f"[probe] {m}", flush=True)


def sigma_for_case(uq, predictor, case, n_mc):
    """Per-channel MC-dropout sigma in PHYSICAL units, (N_OUT, H, W). Once per case."""
    x = encode_case(case)
    xn = predictor.normalizer.norm_in(x)
    xt = torch.from_numpy(np.ascontiguousarray(xn, DTYPE))[None]
    _, std = uq.predict_with_uncertainty(xt, n_samples=n_mc)
    sig = std[0].cpu().numpy()  # (N_OUT, H, W) normalised
    std_out = np.asarray(predictor.normalizer.std_out).reshape(-1, 1, 1)
    return (sig * std_out).astype(np.float64)


def main() -> None:
    from neuroforge.data.airfrans_loader import load_airfrans
    from neuroforge.models.ensemble import MCDropoutUQ
    from neuroforge.physics.calibration import ConformalCalibrator, reliability

    torch.manual_seed(0)
    np.random.seed(0)

    log(f"loading checkpoint {CKPT} on cpu ...")
    model = nf.NeuroForge.load(CKPT, device="cpu")
    _dropout_types = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d,
                      torch.nn.AlphaDropout)
    has_dropout = any(isinstance(m, _dropout_types) for m in model._model.modules())
    corr_type = type(model.engine.corrector).__name__ if model.engine.corrector else "none"
    log(f"backbone dropout present={has_dropout}; corrector={corr_type}")
    assert has_dropout, "need MC-dropout backbone for sigma"
    assert corr_type == "DEQCorrector", "need DEQ corrector"

    log("loading cached AirfRANS test pairs (full r128) ...")
    # Load a larger pool then take a RANDOM permutation subset: AirfRANS cases are
    # ordered, so first/last are not exchangeable (biases the conformal q). A random
    # subset restores exchangeability for the small-set baseline.
    pool = load_airfrans(root="data", task="full", train=False, resolution=RES,
                         limit=200, cache_dir="data/cache", download=False,
                         progress=False)
    rng = np.random.default_rng(0)
    sel = rng.permutation(len(pool))[:N_CASES]
    pairs = [pool[i] for i in sel]
    log(f"using {len(pairs)} cases (random subset of {len(pool)})")

    uq = MCDropoutUQ(model._model)
    predict = model.predictor.predict

    # ---- per-case: sigma (once), raw field, DEQ-corrected field ----
    per_case = []
    delta_mags = []
    t0 = time.time()
    for i, (case, gt) in enumerate(pairs):
        sig = sigma_for_case(uq, model.predictor, case, N_MC)      # (C,H,W)
        raw = predict(case).as_array()                            # deterministic eval mean
        corrected = model.engine.solve(case).field.as_array()      # DEQ fixed point applied
        truth = gt.as_array()
        mask = gt.mask
        per_case.append(dict(sig=sig, raw=raw, corr=corrected, truth=truth, mask=mask))
        # defensive: did the correction actually change the field (fluid cells)?
        fl = np.asarray(mask) > 0.5
        dm = float(np.mean(np.abs((corrected - raw)[:, fl])))
        delta_mags.append(dm)
        log(f"case {i+1}/{len(pairs)} {case.name}: mean|corr-raw|(fluid)={dm:.4e} "
            f"({time.time()-t0:.1f}s)")
    mean_delta = float(np.mean(delta_mags))
    log(f"mean |corrected - raw| over cases = {mean_delta:.4e} (must be >0 for a valid probe)")

    half = len(per_case) // 2
    cal, test = per_case[:half], per_case[half:]

    def err_lists(subset, key, ch):
        errs = [c[key][ch] - c["truth"][ch] for c in subset]
        sigs = [c["sig"][ch] for c in subset]
        masks = [c["mask"] for c in subset]
        return errs, sigs, masks

    def sharpness(errs, sigs, masks, q):
        bands, abserr = [], []
        for e, s, m in zip(errs, sigs, masks):
            sel = np.asarray(m) > 0.5
            bands.append(q * (np.abs(s[sel]) + 1e-8))
            abserr.append(np.abs(e[sel]))
        bands = np.concatenate(bands); abserr = np.concatenate(abserr)
        return float(np.mean(bands)), float(np.mean(abserr))

    def per_case_coverage(errs, sigs, masks, q):
        """Coverage computed per case then averaged (case = exchangeable unit)."""
        covs = []
        for e, s, m in zip(errs, sigs, masks):
            sel = np.asarray(m) > 0.5
            band = q * (np.abs(s[sel]) + 1e-8)
            covs.append(float(np.mean(np.abs(e[sel]) <= band)))
        return float(np.mean(covs)), float(np.std(covs))

    results = {
        "meta": {
            "checkpoint": "checkpoints/certificates_deq.pt",
            "backbone": "dropout-FNO (real AirfRANS, res128)",
            "corrector": corr_type,
            "n_cases_total": len(per_case), "n_cal": half, "n_test": len(test),
            "n_mc": N_MC, "alpha": ALPHA, "target_coverage": 1 - ALPHA,
            "mean_abs_delta_fluid": mean_delta,
            "note": "sigma is input-conditional (function of case, NOT field); "
                    "identical for raw and corrected by construction.",
        },
        "per_channel": {},
    }

    for ch, name in CHANNELS.items():
        # calibration-split errors
        cal_raw_e, cal_raw_s, cal_raw_m = err_lists(cal, "raw", ch)
        cal_cor_e, cal_cor_s, cal_cor_m = err_lists(cal, "corr", ch)
        # test-split errors
        te_raw_e, te_raw_s, te_raw_m = err_lists(test, "raw", ch)
        te_cor_e, te_cor_s, te_cor_m = err_lists(test, "corr", ch)

        # (1) raw certificate
        cal_raw = ConformalCalibrator(alpha=ALPHA).fit(cal_raw_e, cal_raw_s, cal_raw_m)
        q_raw = cal_raw.q
        cov_raw = cal_raw.coverage(te_raw_e, te_raw_s, te_raw_m)
        ece_raw = reliability(te_raw_e, te_raw_s, te_raw_m, n_bins=10,
                              q=q_raw, alpha=ALPHA)["ece"]
        band_raw, abserr_raw = sharpness(te_raw_e, te_raw_s, te_raw_m, q_raw)

        # (2) frozen-q transfer: raw-fit q applied to corrected error
        cov_frozen = cal_raw.coverage(te_cor_e, te_cor_s, te_cor_m)
        ece_frozen = reliability(te_cor_e, te_cor_s, te_cor_m, n_bins=10,
                                 q=q_raw, alpha=ALPHA)["ece"]
        band_frz, abserr_cor = sharpness(te_cor_e, te_cor_s, te_cor_m, q_raw)

        # (3) refit-q on corrected
        cal_cor = ConformalCalibrator(alpha=ALPHA).fit(cal_cor_e, cal_cor_s, cal_cor_m)
        q_cor = cal_cor.q
        cov_refit = cal_cor.coverage(te_cor_e, te_cor_s, te_cor_m)
        ece_refit = reliability(te_cor_e, te_cor_s, te_cor_m, n_bins=10,
                                q=q_cor, alpha=ALPHA)["ece"]
        band_refit, _ = sharpness(te_cor_e, te_cor_s, te_cor_m, q_cor)

        pc_raw, pc_raw_sd = per_case_coverage(te_raw_e, te_raw_s, te_raw_m, q_raw)
        pc_frz, pc_frz_sd = per_case_coverage(te_cor_e, te_cor_s, te_cor_m, q_raw)
        pc_ref, pc_ref_sd = per_case_coverage(te_cor_e, te_cor_s, te_cor_m, q_cor)

        results["per_channel"][name] = {
            "field_improved_by_corrector": bool(abserr_cor < abserr_raw),
            "raw_certificate": {
                "q": q_raw, "coverage": cov_raw, "ece": ece_raw,
                "mean_band": band_raw, "mean_abs_err": abserr_raw,
                "per_case_coverage_mean": pc_raw, "per_case_coverage_std": pc_raw_sd},
            "frozen_q_transfer": {
                "q": q_raw, "coverage": cov_frozen, "ece": ece_frozen,
                "mean_band": band_frz, "mean_abs_err": abserr_cor,
                "per_case_coverage_mean": pc_frz, "per_case_coverage_std": pc_frz_sd},
            "refit_q_corrected": {
                "q": q_cor, "coverage": cov_refit, "ece": ece_refit,
                "mean_band": band_refit, "mean_abs_err": abserr_cor,
                "per_case_coverage_mean": pc_ref, "per_case_coverage_std": pc_ref_sd},
        }
        impr = "IMPROVED" if abserr_cor < abserr_raw else "DEGRADED"
        log(f"[{name}] field {impr} ({abserr_raw:.3f}->{abserr_cor:.3f}) | "
            f"raw cov={cov_raw:.3f} ece={ece_raw:.3f} q={q_raw:.3f} | "
            f"frozen-q-on-corr cov={cov_frozen:.3f} ece={ece_frozen:.3f} | "
            f"refit cov={cov_refit:.3f} ece={ece_refit:.3f} q={q_cor:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
