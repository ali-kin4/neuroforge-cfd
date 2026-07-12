"""RIGOROUS follow-up for reviewer finding W1 (the ARCHITECTURAL-VALUE question).

*** GPU-LONG. DO NOT RUN ON CPU / DO NOT RUN FROM THE AGENT LOOP. ***
*** Launch this only on the workstation GPU as a top-level job. It trains ***
*** 2 conditions x 3 seeds = 6 DEQ correctors from scratch on real AirfRANS. ***

WHY THIS EXISTS
---------------
The cheap occlusion test (scripts/control_residual_input_ablation.py) is a PROXY:
it perturbs the residual INPUT of the *already-trained* deployed corrector. It
showed (on certificates_deq.pt, FNO backbone, task=full, N=20):
  - mismatched residual HURTS (frac_C=0.31, p=0.003)  -> channel is READ, not decorative
  - ZEROED residual does NOT hurt and in fact HELPS (frac_B=1.80, 17/20, p=0.001)
    -> the real residual is NOT a useful input *on this checkpoint*.
But zeroing is OOD for weights trained to expect a residual; it is NOT a fair
stand-in for a corrector TRAINED FROM SCRATCH without the residual. The deep
question -- does conditioning on the residual give the corrector any architectural
benefit it could not get from field+geom alone? -- can ONLY be settled by retraining.

THE EXPERIMENT (clean, controlled)
----------------------------------
Identical architecture, data, optimiser, schedule, seeds. The ONLY difference is
whether the 3 residual channels carry the real residual or are permanently ZEROED
*during training AND inference* (so the corrector never learns to rely on them):
  - condition WITH : residual = real normalised residual (deployed setting)
  - condition NULL : residual = 0 everywhere (the residual input is removed in
                     effect; weights on those channels get no signal and the
                     corrector must do the job from field+geom alone)
We also cover the HEADLINE backbone: the -9/-21/-25% field-MSE numbers are on a
TRANSOLVER (3 seeds), whereas certificates_deq.pt is an FNO. Set BACKBONE below to
run the occlusion's backbone (fno) and/or the headline one (transformer).

PRE-REGISTERED DECISION RULE
----------------------------
Compare mean test field-MSE gain (backbone - corrected) WITH vs NULL, paired by
seed, on mse_u / mse_speed (u/v-dominated; mse_p is out of W1 scope and noted
separately).
  - If WITH beats NULL by a margin that exceeds the seed-to-seed std (e.g.
    gain_WITH - gain_NULL > 1 sigma across seeds, same sign on >=2/3 seeds) ->
    the residual INPUT has architectural value -> W1 SUPPORTED, restore the claim.
  - If NULL matches or beats WITH (gains within seed noise, or NULL >= WITH) ->
    the residual input gives no benefit when the corrector is trained without it
    -> W1 REFRAME stands: the residual is a strong trust SIGNAL and (we have shown
    separately) a poor correction OBJECTIVE, but NOT a useful corrector INPUT.

OUTPUT: results/control/retrain_without_residual.json
"""

import json
import os

import numpy as np
import torch

import neuroforge  # noqa: F401  -- thread caps before numpy/torch
from neuroforge.core.config import Config, DataConfig
from neuroforge.data.airfrans_loader import load_airfrans
from neuroforge.data.datamodule import build_dataloaders
from neuroforge.models import build_corrector
from neuroforge.models.base import build_model
from neuroforge.physics.evaluation import per_channel_mse
from neuroforge.physics.residuals import PhysicsChecker
from neuroforge.solver.engine import NeuroForgeEngine, Predictor
from neuroforge.train.trainer import Trainer

# ----------------------------- config knobs -----------------------------------
BACKBONE = "fno"          # "fno" (matches certificates_deq.pt) or "transformer" (headline)
SEEDS = [0, 1, 2]
N_TEST = 50
CORR_EPOCHS = 30          # match the production corrector schedule
DEVICE = "cuda"           # GPU job; do NOT set to cpu here
# Backbone configs (mirror the deployed/headline checkpoints).
BACKBONE_CFG = {
    "fno": dict(name="fno", width=48, n_layers=4, modes=20),
    "transformer": dict(name="transformer", width=128, n_layers=4, n_heads=4, n_slices=32),
}
CORR_CFG = {"type": "deq", "width": 32, "n_layers": 3, "lipschitz": 0.9,
            "damping": 0.5, "max_iter": 25, "tol": 1e-4}


def _zero_residual_patch():
    """Context-managerless monkeypatch: force every residual the trainer/inference
    builds to ZERO, so the corrector is trained & deployed without residual content.
    Returns a restore() callable. Patches BOTH the training residual and the
    inference-loop residual so the NULL condition is consistent end-to-end."""
    import neuroforge.solver.correction_loop as cl

    orig_tr = Trainer._normalised_residual
    orig_cl = cl._residual_tensor

    def zero_tr(self, state, inp_phys, dx, dy, nu):
        r = orig_tr(self, state, inp_phys, dx, dy, nu)
        return torch.zeros_like(r)

    def zero_cl(field, case, predictor):
        r = orig_cl(field, case, predictor)
        return torch.zeros_like(r)

    Trainer._normalised_residual = zero_tr
    cl._residual_tensor = zero_cl

    def restore():
        Trainer._normalised_residual = orig_tr
        cl._residual_tensor = orig_cl

    return restore


def _train_one(cfg, train_loader, val_loader, normalizer, seed):
    """Train one backbone + corrector. The residual-zeroing patch (if any) is
    applied/removed by the CALLER (``main``), once per condition -- so this fn is
    patch-agnostic and never leaks state."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(**BACKBONE_CFG[BACKBONE])
    trainer = Trainer(model, cfg, normalizer, nu=1.5e-5)
    trainer.fit(train_loader, val_loader)
    corrector = build_corrector(CORR_CFG)
    trainer.fit_corrector(train_loader, corrector, epochs=CORR_EPOCHS)
    return model, normalizer, corrector


def _eval(model, normalizer, corrector, cfg, pairs):
    """Evaluate field-MSE gain. Like ``_train_one``, assumes the caller has the
    correct residual-patch state active for this condition."""
    predictor = Predictor(model, normalizer, device=DEVICE)
    checker = PhysicsChecker(cfg.physics)
    engine = NeuroForgeEngine(predictor, checker, corrector=corrector, config=cfg)
    bb_u, bb_sp, cu, csp = [], [], [], []
    for case, ref in pairs:
        f0 = predictor.predict(case)
        mb = per_channel_mse(f0, ref)
        bb_u.append(mb["mse_u"]); bb_sp.append(mb["mse_speed"])
        fc = engine.solve(case).field
        mc = per_channel_mse(fc, ref)
        cu.append(mc["mse_u"]); csp.append(mc["mse_speed"])
    return {
        "gain_u": float(np.mean(bb_u) - np.mean(cu)),
        "gain_speed": float(np.mean(bb_sp) - np.mean(csp)),
        "backbone_u": float(np.mean(bb_u)), "corrected_u": float(np.mean(cu)),
        "backbone_speed": float(np.mean(bb_sp)), "corrected_speed": float(np.mean(csp)),
    }


def main() -> None:
    cfg = Config()
    cfg.train.device = DEVICE
    cfg.data = DataConfig(source="airfrans", task="full", resolution=128, batch_size=4)
    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)
    pairs = load_airfrans(task="full", train=False, resolution=128, limit=N_TEST,
                          cache_dir="data/cache", progress=True)

    results = {"WITH": [], "NULL": []}
    for cond, zero in (("WITH", False), ("NULL", True)):
        # Patch ONCE per condition (train + eval share the same residual state),
        # and always restore in finally -> no cross-condition leakage.
        restore = _zero_residual_patch() if zero else (lambda: None)
        try:
            for seed in SEEDS:
                model, norm, corr = _train_one(cfg, train_loader, val_loader, normalizer, seed)
                ev = _eval(model, norm, corr, cfg, pairs)
                ev["seed"] = seed
                results[cond].append(ev)
                print(f"[{cond} seed={seed}] gain_speed={ev['gain_speed']:+.5f} gain_u={ev['gain_u']:+.5f}", flush=True)
        finally:
            restore()

    def agg(rows, k):
        v = np.array([r[k] for r in rows], np.float64)
        return float(v.mean()), float(v.std())

    gw_m, gw_s = agg(results["WITH"], "gain_speed")
    gn_m, gn_s = agg(results["NULL"], "gain_speed")
    margin = gw_m - gn_m
    n_with_better = int(np.sum(
        np.array([r["gain_speed"] for r in results["WITH"]])
        > np.array([r["gain_speed"] for r in results["NULL"]])
    ))
    verdict = ("SUPPORTED" if (margin > max(gw_s, gn_s) and n_with_better >= 2)
               else "REFRAME")

    out = {
        "meta": {"backbone": BACKBONE, "seeds": SEEDS, "n_test": N_TEST,
                 "corr_epochs": CORR_EPOCHS, "corrector": CORR_CFG},
        "WITH": results["WITH"], "NULL": results["NULL"],
        "summary": {"gain_speed_WITH": [gw_m, gw_s], "gain_speed_NULL": [gn_m, gn_s],
                    "margin_WITH_minus_NULL": margin, "n_seeds_WITH_better": n_with_better},
        "verdict": verdict,
    }
    os.makedirs("results/control", exist_ok=True)
    with open("results/control/retrain_without_residual.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}  (margin={margin:+.5f}, WITH better on {n_with_better}/3 seeds)")
    print("wrote results/control/retrain_without_residual.json")


if __name__ == "__main__":
    raise SystemExit(
        "retrain_without_residual.py is a GPU-LONG job. Comment out this guard and "
        "launch deliberately on the workstation GPU (see module docstring). Do NOT run on CPU."
    )
    main()  # unreachable by design -- the guard above must be removed to run
