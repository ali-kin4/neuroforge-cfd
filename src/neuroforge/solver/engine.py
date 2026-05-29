"""The product surface: :class:`Predictor`, :class:`NeuroForgeEngine`, ``demo``.

* :class:`Predictor` bridges a trained torch backbone + :class:`Normalizer` to
  the numpy :class:`FlowField` world.
* :class:`NeuroForgeEngine` composes a predictor, a physics checker, an optional
  correction network (Neural Residual Iteration) and an optional uncertainty
  estimator + classical fallback into the self-correcting solve.
* :func:`demo` is the one-call end-to-end entry point.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from neuroforge.core.config import Config
from neuroforge.core.types import (
    DTYPE,
    FlowCase,
    FlowField,
    SolveResult,
)
from neuroforge.geometry.encode import encode_case

from .correction_loop import neural_residual_iteration
from .fallback import ClassicalFallback

__all__ = ["Predictor", "NeuroForgeEngine", "neural_residual_iteration", "ClassicalFallback", "demo"]

_DEMO_CKPT = os.path.join("checkpoints", "demo.pt")
# A small checkpoint shipped inside the package so pretrained() is instant on
# first use (no training needed). Built by scripts/build_demo_checkpoint.py.
_BUNDLED_CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "demo.pt")


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #


class Predictor:
    """Run a trained backbone on a :class:`FlowCase` and return a :class:`FlowField`.

    Parameters
    ----------
    model : NeuralSolver
        Trained backbone.
    normalizer : Normalizer
        Fitted normaliser matching the model's training.
    device : str, optional
        Torch device string; defaults to ``'cpu'``.
    """

    def __init__(self, model, normalizer, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.normalizer = normalizer

    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Raw model output in normalised space for a normalised input ``x``."""
        x = x.to(self.device)
        return self.model.predict(x)

    def predict(self, case: FlowCase) -> FlowField:
        """Predict the flow field for ``case`` (denormalised, with mask/sdf)."""
        stack = encode_case(case)                       # (7, H, W) physical
        xn = self.normalizer.norm_in(stack)
        xt = torch.from_numpy(np.ascontiguousarray(xn, DTYPE))[None]
        with torch.no_grad():
            yn = self.predict_tensor(xt)
        y = self.normalizer.denorm_out(yn)[0].cpu().numpy().astype(DTYPE)

        # mask / sdf come straight from the encoded geometry channels.
        sdf = stack[0]
        mask = stack[1]
        return FlowField.from_array(
            y, case.domain, mask=mask.astype(DTYPE), sdf=sdf.astype(DTYPE),
            meta={"source": "neural", "case": case.name},
        )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class NeuroForgeEngine:
    """The self-correcting geometry-native AI CFD engine.

    Parameters
    ----------
    predictor : Predictor
        The trained backbone bridge.
    checker : PhysicsChecker
        The physics verifier.
    corrector : CorrectionNetwork, optional
        Learned residual corrector for Neural Residual Iteration.
    uq : optional
        Uncertainty estimator (``predict_with_uncertainty``).
    config : Config, optional
        Engine configuration (correction + physics + fallback settings).
    """

    def __init__(
        self,
        predictor: Predictor,
        checker,
        corrector=None,
        uq=None,
        config: Config | None = None,
    ) -> None:
        self.predictor = predictor
        self.checker = checker
        self.corrector = corrector
        self.uq = uq
        self.config = config if config is not None else Config()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(cls, path: str, config: Config | None = None) -> "NeuroForgeEngine":
        """Build an engine from a Trainer-style checkpoint.

        Rebuilds the backbone + normaliser (via :meth:`Trainer.load`), the
        :class:`PhysicsChecker`, and the :class:`LocalCorrectionNet` if a
        corrector was saved.
        """
        from neuroforge.physics.residuals import PhysicsChecker
        from neuroforge.train.trainer import Trainer

        cfg = config if config is not None else Config()
        model, normalizer, model_cfg = Trainer.load(path, map_location="cpu")
        cfg.model = model_cfg
        predictor = Predictor(model, normalizer, device="cpu")
        checker = PhysicsChecker(cfg.physics)

        corrector = None
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("corrector_state") is not None:
            from neuroforge.models.correction import LocalCorrectionNet

            cc = ckpt.get("corrector_config") or {}
            corrector = LocalCorrectionNet(
                width=int(cc.get("width", 24)),
                n_layers=int(cc.get("n_layers", 3)),
            )
            corrector.load_state_dict(ckpt["corrector_state"])
            corrector.eval()

        return cls(predictor, checker, corrector=corrector, uq=None, config=cfg)

    # cache for pretrained()
    _pretrained_cache: "NeuroForgeEngine | None" = None

    @classmethod
    def pretrained(cls) -> "NeuroForgeEngine":
        """Load the bundled demo checkpoint, training a tiny one if absent.

        The trained checkpoint is cached on disk (``checkpoints/demo.pt``) and in
        memory, so training happens at most once.
        """
        if cls._pretrained_cache is not None:
            return cls._pretrained_cache
        # Prefer a bundled checkpoint (instant), then a locally cached one,
        # else train a tiny model once and cache it to disk.
        for cand in (_BUNDLED_CKPT, _DEMO_CKPT):
            if os.path.exists(cand):
                engine = cls.from_checkpoint(cand)
                cls._pretrained_cache = engine
                return engine
        cls._train_demo_checkpoint(_DEMO_CKPT)
        engine = cls.from_checkpoint(_DEMO_CKPT)
        cls._pretrained_cache = engine
        return engine

    @staticmethod
    def _train_demo_checkpoint(path: str) -> None:
        """Train a tiny FNO + corrector on synthetic data and save to ``path``.

        Sized for CPU: a width-16 / 2-layer FNO on a small synthetic set. The
        wall time is dominated by the (one-time, then cached) synthetic
        data-generation, which costs ~15-25 s per case regardless of resolution
        (it is set by the surface-panel solve, not the grid), so the case count
        is kept small. Trained once and cached to ``checkpoints/demo.pt``.
        """
        from neuroforge.core.config import DataConfig
        from neuroforge.data.datamodule import build_dataloaders
        from neuroforge.models.base import build_model
        from neuroforge.models.correction import LocalCorrectionNet
        from neuroforge.train.trainer import Trainer

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        cfg = Config()
        cfg.data = DataConfig(
            source="synthetic", resolution=40, n_train=6, n_val=2, batch_size=3
        )
        cfg.train.epochs = 6
        cfg.train.lr = 1e-3
        cfg.train.log_every = 0  # quiet

        train_loader, val_loader, normalizer = build_dataloaders(cfg.data)
        model = build_model("fno", width=16, n_layers=2, modes=12)
        # A representative nu for the loss (overwritten per-case at solve time).
        trainer = Trainer(model, cfg, normalizer, nu=1.5e-5)
        trainer.fit(train_loader, val_loader)

        corrector = LocalCorrectionNet(width=16, n_layers=2)
        trainer.fit_corrector(train_loader, corrector, epochs=3)

        trainer.save(path, corrector=corrector)

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #

    def solve(self, case: FlowCase, max_iters: int | None = None) -> SolveResult:
        """Predict, self-correct, optionally patch, and compute metrics.

        Returns
        -------
        SolveResult
            With the final field, diagnostics, the per-iteration history, and a
            ``metrics`` dict including ``cl``/``cd``/``cm``, ``residual_norm`` and
            ``n_iters``.
        """
        cfg = self.config
        corr_cfg = cfg.correction
        if max_iters is not None:
            # Shallow override without mutating the shared config.
            from dataclasses import replace

            corr_cfg = replace(corr_cfg, max_iters=int(max_iters))

        field0 = self.predictor.predict(case)

        field, history = neural_residual_iteration(
            field0, case, self.checker, self.corrector, corr_cfg,
            self.predictor, uq=self.uq,
        )

        # Final diagnostics (re-diagnose to also capture uncertainty if any).
        unc = None
        if self.uq is not None:
            from .correction_loop import _diag_uncertainty

            unc = _diag_uncertainty(self.uq, self.predictor, case)
        diag = self.checker.diagnose(field, case, uncertainty=unc)

        # Optional uncertainty-gated classical fallback patch (stub).
        max_unc = float(np.max(diag.uncertainty)) if diag.uncertainty.size else 0.0
        if corr_cfg.fallback_enabled and max_unc > corr_cfg.fallback_uncertainty_threshold:
            region = (np.asarray(diag.trust, dtype=DTYPE) < 0.5) & (
                np.asarray(field.mask, dtype=DTYPE) > 0.5
            )
            fb = ClassicalFallback(backend=corr_cfg.fallback_solver)
            field = fb.patch(field, case, region)

        metrics = self._metrics(field, case, diag, history)

        return SolveResult(
            case=case,
            field=field,
            diagnostics=diag,
            metrics=metrics,
            history=history,
            meta={"n_iters": len(history)},
        )

    @staticmethod
    def _metrics(field, case, diag, history) -> dict:
        """Assemble the scalar metrics dict (forces + residual + field summary)."""
        from neuroforge.physics.metrics import force_coefficients

        try:
            forces = force_coefficients(field, case)
        except Exception:
            forces = {"cl": float("nan"), "cd": float("nan"), "cm": float("nan")}

        speed = field.speed()
        metrics = {
            "cl": float(forces.get("cl", float("nan"))),
            "cd": float(forces.get("cd", float("nan"))),
            "cm": float(forces.get("cm", float("nan"))),
            "residual_norm": float(diag.residual_norm()),
            "n_iters": float(len(history)),
            "speed_max": float(np.max(speed)) if speed.size else 0.0,
            "speed_mean": float(np.mean(speed)) if speed.size else 0.0,
            "p_min": float(np.min(field.p)) if field.p.size else 0.0,
            "p_max": float(np.max(field.p)) if field.p.size else 0.0,
            "trust_mean": float(np.mean(diag.trust)) if diag.trust.size else 1.0,
        }
        if history:
            metrics["residual_norm_initial"] = float(history[0]["residual_norm"])
        return metrics


# --------------------------------------------------------------------------- #
# One-call demo
# --------------------------------------------------------------------------- #


def demo() -> SolveResult:
    """End-to-end demo: build the pretrained engine and solve a NACA case."""
    engine = NeuroForgeEngine.pretrained()
    case = FlowCase.from_airfoil(
        "naca2412", aoa=5, reynolds=3e6, u_inf=30.0, resolution=64
    )
    return engine.solve(case)
