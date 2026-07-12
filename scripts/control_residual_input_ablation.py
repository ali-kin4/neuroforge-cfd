"""Discriminating control for reviewer finding W1: is the physics residual a
*used, load-bearing INPUT* to the deployed DEQ corrector, or a decorative channel
the corrector ignores?

The paper makes a three-role claim for the residual; the second role is "residual
as corrector INPUT" (the DEQ corrector concatenates a 3-channel residual map into
its conditioning -- see deq_corrector.py:154 and correction_loop.py:154-159). A
hostile reviewer says this is asserted, not shown: the corrector could zero-weight
the residual channels and behave identically. This is the CHEAP occlusion test that
settles it on the DEPLOYED checkpoint (no retraining). CPU-only.

INJECTION POINT (surgical, params/architecture identical across arms): we wrap
``corrector.forward`` and replace its ``residual`` argument before the fixed-point
solve. Nothing else changes -- same weights, same field & geom conditioning.

THREE ARMS (paired per case, only the residual input toggled):
  (A) NORMAL    -- real residual map (as deployed).
  (B) ZEROED    -- residual channels set to 0 (occlude the input entirely).
  (C) MISMATCH  -- residual map from a DIFFERENT case (same shape/scale), i.e. a
                   plausible-but-WRONG residual. This defangs the obvious rebuttal
                   to (B): "zero is out-of-distribution, so degradation only shows
                   OOD-sensitivity, not that the residual carries correctness signal."
                   If the corrector truly USES residual CONTENT, C (wrong content,
                   in-distribution scale) should also differ from A.

PRIMARY METRIC -- the corrector's GAIN on field MSE (the deployed claim is the
-9/-21/-25% field-MSE improvement, NOT residual-norm reduction; our own thesis says
the DEQ barely targets the monitored residual, so residual-norm change is secondary):
    gain_X = MSE(backbone) - MSE(corrected under arm X)     [per channel, per case]
We report mse_u and mse_speed. Retained-gain fraction:
    frac_B = gain_B / gain_A ,  frac_C = gain_C / gain_A   (on mean gains).

SANITY GATE (checked & printed FIRST): the corrector must actually improve MSE on
these N cases under arm A. If gain_A <= ~0, the occlusion test cannot discriminate
anything -- we say so and stop interpreting (widen N / pick the channel that moves).

================================ DECISION RULE (PRE-REGISTERED, before running) ============================
Let frac_B, frac_C be the fraction of the corrector's field-MSE gain that survives
occlusion (B) / mismatch (C), on the channel that the corrector actually moves
(primary: mse_u; corroborate with mse_speed).

  SUPPORTED  (residual is a used, load-bearing input):
     occlusion erases a substantial share of the gain AND wrong content hurts too:
       frac_B < 0.50  AND  frac_C < 0.80
     i.e. zeroing the residual destroys >half the corrector's benefit, and even a
     plausible-but-wrong residual measurably degrades it (so it is not mere OOD).
     Justification for 0.50: a channel carrying <half the measured benefit is hard
     to call "decorative"; 0.80 for C is laxer because mismatch is a weaker
     perturbation than full occlusion.

  REFRAME    (residual input is decorative / not demonstrated):
     correction is ~unchanged with the residual occluded:
       frac_B > 0.85  (and arms A/B/C field MSE within noise; paired test n.s.)
     -> the paper must drop or soften the "residual as INPUT" sub-claim.

  AMBIGUOUS  (anything in between, or sanity gate fails): the cheap occlusion does
     not settle it -> the from-scratch RETRAIN-WITHOUT-RESIDUAL harness
     (scripts/retrain_without_residual.py, GPU-long, NOT run here) is needed.

We additionally report (descriptive mechanism only, NOT a verdict driver): the
correction-magnitude change ||delta_B - delta_A|| / ||delta_A|| and the residual-norm
reduction per arm. A big delta change alone does not prove useful use (could be OOD
garbage), so it does not drive the verdict.
=============================================================================================================
"""

import json
import os

import numpy as np
import torch

import neuroforge  # noqa: F401  -- thread caps before numpy/torch
from neuroforge.core.config import Config
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.physics.evaluation import per_channel_mse
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine

CKPT = "checkpoints/certificates_deq.pt"
N = 20


def _wilcoxon_or_sign(diffs: np.ndarray) -> dict:
    """Paired test that occlusion changes per-case error. Wilcoxon if scipy, else a
    sign test. ``diffs`` = (arm error) - (arm-A error); positive => occlusion worse."""
    d = np.asarray(diffs, np.float64)
    d = d[np.isfinite(d)]
    n_worse = int(np.sum(d > 0))
    n_total = int(np.sum(d != 0))
    out = {"n_worse_than_A": n_worse, "n_nonzero": n_total, "frac_worse": (n_worse / n_total) if n_total else float("nan")}
    try:
        from scipy.stats import wilcoxon

        if n_total >= 1 and not np.allclose(d, 0):
            w = wilcoxon(d)
            out["wilcoxon_p"] = float(w.pvalue)
    except Exception:
        pass
    return out


def main() -> None:
    cfg = Config()
    engine = NeuroForgeEngine.from_checkpoint(CKPT, config=cfg)
    corrector = engine.corrector
    assert corrector is not None, "checkpoint has no corrector"
    assert type(corrector).__name__ == "DEQCorrector", f"expected DEQCorrector, got {type(corrector).__name__}"
    pairs = load_airfrans(task="full", train=False, resolution=128, limit=N,
                          cache_dir="data/cache", progress=False)
    print(f"loaded engine (corrector={type(corrector).__name__}); {len(pairs)} cases", flush=True)

    orig_forward = corrector.forward
    captured = {}  # arm-tagged: last delta tensor for magnitude diagnostics

    # --- occlusion modes via a wrapped forward; only the `residual` arg changes ---
    state = {"mode": "A", "mismatch": None}  # mismatch: a (1,3,H,W) tensor to substitute

    def wrapped_forward(field, residual, geom):
        if state["mode"] == "B":
            residual = torch.zeros_like(residual)
        elif state["mode"] == "C":
            mm = state["mismatch"]
            # Substitute a different case's residual; match this case's shape exactly.
            if mm is not None and mm.shape == residual.shape:
                residual = mm
            else:
                # Fallback: shuffle the spatial content (still real-scale, wrong place).
                residual = residual.flip(dims=(-1, -2))
        delta = orig_forward(field=field, residual=residual, geom=geom)
        captured["delta"] = delta.detach().clone()
        return delta

    corrector.forward = wrapped_forward  # type: ignore[assignment]

    checker = PhysicsChecker(cfg.physics)

    # Precompute each case's normalised residual tensor (for the MISMATCH arm we feed
    # the NEXT case's residual). We compute it the same way the loop does.
    from neuroforge.solver.correction_loop import _residual_tensor

    backbone_fields = []
    residual_tensors = []
    for case, ref in pairs:
        f0 = engine.predictor.predict(case)
        backbone_fields.append(f0)
        residual_tensors.append(_residual_tensor(f0, case, engine.predictor))

    def run_arm(mode: str) -> dict:
        per = {"mse_u": [], "mse_speed": [], "mse_p": [],
               "res_norm_corr": [], "delta_norm": [], "delta_rel_to_A": []}
        corrected_fields = []
        for i, (case, ref) in enumerate(pairs):
            state["mode"] = mode
            # mismatch residual = a different case's residual (cyclic neighbour), padded/cropped to shape
            if mode == "C":
                j = (i + 1) % len(pairs)
                mm = residual_tensors[j]
                if mm.shape == residual_tensors[i].shape:
                    state["mismatch"] = mm
                else:
                    state["mismatch"] = None  # wrapped_forward falls back to spatial flip
            res = engine.solve(case)
            field = res.field
            corrected_fields.append(field)
            m = per_channel_mse(field, ref)
            per["mse_u"].append(m["mse_u"])
            per["mse_speed"].append(m["mse_speed"])
            per["mse_p"].append(m["mse_p"])
            per["res_norm_corr"].append(float(res.metrics["residual_norm"]))
            per["delta_norm"].append(float(captured["delta"].norm()))
        return {"per": per, "fields": corrected_fields}

    state["mode"] = "A"

    # ---- Backbone (no correction) baseline, per case ----
    bb = {"mse_u": [], "mse_speed": [], "mse_p": [], "res_norm": []}
    for f0, (case, ref) in zip(backbone_fields, pairs):
        m = per_channel_mse(f0, ref)
        bb["mse_u"].append(m["mse_u"])
        bb["mse_speed"].append(m["mse_speed"])
        bb["mse_p"].append(m["mse_p"])
        bb["res_norm"].append(float(checker.diagnose(f0, case).residual_norm()))

    # ---- Run the three arms ----
    A = run_arm("A")
    B = run_arm("B")
    C = run_arm("C")

    corrector.forward = orig_forward  # type: ignore[assignment]

    bb_u = np.array(bb["mse_u"]); bb_sp = np.array(bb["mse_speed"]); bb_p = np.array(bb["mse_p"])

    def arr(d, k):
        return np.array(d["per"][k], np.float64)

    # ---- Sanity gate: does the corrector improve MSE under arm A? ----
    gainA_u = float(np.mean(bb_u - arr(A, "mse_u")))
    gainA_sp = float(np.mean(bb_sp - arr(A, "mse_speed")))
    gainA_p = float(np.mean(bb_p - arr(A, "mse_p")))
    print("\nSANITY GATE (corrector gain under arm A; positive = corrector helps):", flush=True)
    print(f"  mse_u   : backbone {bb_u.mean():.5f} -> corrected {arr(A,'mse_u').mean():.5f}   gain {gainA_u:+.5f}")
    print(f"  mse_speed: backbone {bb_sp.mean():.5f} -> corrected {arr(A,'mse_speed').mean():.5f}   gain {gainA_sp:+.5f}")
    print(f"  mse_p   : backbone {bb_p.mean():.5f} -> corrected {arr(A,'mse_p').mean():.5f}   gain {gainA_p:+.5f}")

    # Choose the primary channel as the one with the largest (positive) corrector gain.
    gains = {"mse_u": gainA_u, "mse_speed": gainA_sp, "mse_p": gainA_p}
    primary = max(gains, key=lambda k: gains[k])
    sanity_ok = gains[primary] > 0
    print(f"  primary channel (largest corrector gain): {primary}   gate_ok={sanity_ok}", flush=True)

    # ---- Gain-fraction framing on the primary channel ----
    bb_prim = {"mse_u": bb_u, "mse_speed": bb_sp, "mse_p": bb_p}[primary]
    gA = bb_prim - arr(A, primary)   # per-case gain, arm A
    gB = bb_prim - arr(B, primary)
    gC = bb_prim - arr(C, primary)
    meanA, meanB, meanC = float(gA.mean()), float(gB.mean()), float(gC.mean())
    frac_B = meanB / meanA if meanA != 0 else float("nan")
    frac_C = meanC / meanA if meanA != 0 else float("nan")

    # Paired tests: does occlusion (B) / mismatch (C) raise per-case error vs A?
    diffB = arr(B, primary) - arr(A, primary)   # >0 => B worse
    diffC = arr(C, primary) - arr(A, primary)
    testB = _wilcoxon_or_sign(diffB)
    testC = _wilcoxon_or_sign(diffC)

    # Descriptive mechanism: correction-magnitude change.
    dA = arr(A, "delta_norm"); dB = arr(B, "delta_norm"); dC = arr(C, "delta_norm")
    delta_change_B = float(np.mean(np.abs(dB - dA) / (np.abs(dA) + 1e-12)))
    delta_change_C = float(np.mean(np.abs(dC - dA) / (np.abs(dA) + 1e-12)))

    print(f"\nGAIN-FRACTION (primary={primary}):", flush=True)
    print(f"  gain A (real residual)   = {meanA:+.5f}")
    print(f"  gain B (zeroed)          = {meanB:+.5f}   frac_B = {frac_B:.2f}")
    print(f"  gain C (mismatch)        = {meanC:+.5f}   frac_C = {frac_C:.2f}")
    print(f"  paired B vs A: {testB}")
    print(f"  paired C vs A: {testC}")
    print(f"  |delta| rel change  B={delta_change_B:.3f}  C={delta_change_C:.3f}  (descriptive only)")

    # ---- Verdict ----
    if not sanity_ok:
        verdict = "AMBIGUOUS-SANITY-GATE-FAILED"
    elif (frac_B < 0.50) and (frac_C < 0.80):
        verdict = "SUPPORTED"
    elif frac_B > 0.85:
        verdict = "REFRAME"
    else:
        verdict = "AMBIGUOUS"

    print("\nVERDICT:", verdict, flush=True)

    def summ(label, d):
        return {
            "mse_u": float(arr(d, "mse_u").mean()),
            "mse_speed": float(arr(d, "mse_speed").mean()),
            "mse_p": float(arr(d, "mse_p").mean()),
            "res_norm_corr": float(arr(d, "res_norm_corr").mean()),
            "delta_norm": float(arr(d, "delta_norm").mean()),
        }

    out = {
        "meta": {
            "checkpoint": CKPT, "n_cases": len(pairs), "corrector": type(corrector).__name__,
            "injection": "wrapped corrector.forward; only residual arg toggled",
            "arms": {"A": "real residual", "B": "zeroed residual", "C": "mismatch (neighbour case residual)"},
            "primary_channel": primary,
        },
        "backbone": {"mse_u": float(bb_u.mean()), "mse_speed": float(bb_sp.mean()),
                     "mse_p": float(bb_p.mean()), "res_norm": float(np.mean(bb["res_norm"]))},
        "arm_A": summ("A", A), "arm_B": summ("B", B), "arm_C": summ("C", C),
        "sanity_gate": {"gain_mse_u": gainA_u, "gain_mse_speed": gainA_sp, "gain_mse_p": gainA_p,
                        "primary_channel": primary, "ok": bool(sanity_ok)},
        "gain_fraction": {"primary": primary, "gain_A": meanA, "gain_B": meanB, "gain_C": meanC,
                          "frac_B": frac_B, "frac_C": frac_C},
        "paired_tests": {"B_vs_A": testB, "C_vs_A": testC},
        "delta_magnitude_change": {"B": delta_change_B, "C": delta_change_C},
        "decision_rule": "SUPPORTED iff frac_B<0.50 and frac_C<0.80; REFRAME iff frac_B>0.85; else AMBIGUOUS",
        "verdict": verdict,
        "note": "Occlusion is a PROXY on the deployed checkpoint. The clean settle is the "
                "from-scratch retrain-without-residual harness (scripts/retrain_without_residual.py, GPU-long, NOT run here).",
    }
    os.makedirs("results/control", exist_ok=True)
    with open("results/control/residual_input_ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/control/residual_input_ablation.json", flush=True)


if __name__ == "__main__":
    main()
