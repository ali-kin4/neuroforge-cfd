"""High-level, scikit-learn / Keras-style API: the :class:`NeuroForge` estimator.

One object, clean progression — hyper-parameters in the constructor, data in
``fit``, everything else a one-liner::

    import neuroforge as nf

    model = nf.NeuroForge(backbone="fno", width=64, modes=24, corrector="deq")
    model.fit("airfrans", task="full", n_train=400, n_val=120)   # -> self (chainable)

    print(model.evaluate())                       # AirfRANS metrics: rho_Cd, MSE, ...
    field  = model.predict(case)                  # fast one-shot backbone field
    result = model.solve(case)                    # full self-correcting solve + diagnostics
    model.calibrate(alpha=0.1)                     # conformal-calibrated trust (coverage guarantee)
    model.save("model.pt")
    model = nf.NeuroForge.load("model.pt")

It wraps the config + dataloaders + trainer + engine that you can still use
directly; this is just the ergonomic front door.
"""

from __future__ import annotations

from typing import Any

from neuroforge.core.config import Config, DataConfig, ModelConfig
from neuroforge.core.types import FlowCase, FlowField, SolveResult

__all__ = ["NeuroForge"]


class NeuroForge:
    """A trainable, self-correcting AI-CFD model with a clean estimator API.

    Parameters
    ----------
    backbone : str
        Neural-operator backbone: ``'fno'`` | ``'geo_fno'`` | ``'transformer'`` |
        ``'unet'`` | ``'deeponet'``.
    width, n_layers, modes : int
        Backbone size (``modes`` applies to the FNO family).
    dropout : float
        Dropout probability (``>0`` enables MC-dropout uncertainty / calibration).
    corrector : str
        Correction operator: ``'deq'`` (contractive Deep-Equilibrium, convergence
        guarantee) | ``'local'`` (feed-forward) | ``'none'``.
    epochs, corrector_epochs : int
        Training epochs for the backbone and the corrector.
    lr, physics_weight, bc_weight : float
        Optimiser learning rate and physics / BC loss weights.
    resolution, batch_size : int
        Grid resolution and batch size.
    amp : bool
        CUDA mixed precision (no-op on CPU).
    device : str
        ``'auto'`` | ``'cpu'`` | ``'cuda'``.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        backbone: str = "fno",
        *,
        width: int = 48,
        n_layers: int = 4,
        modes: int = 20,
        dropout: float = 0.0,
        corrector: str = "deq",
        epochs: int = 100,
        corrector_epochs: int = 20,
        lr: float = 8e-4,
        physics_weight: float = 0.1,
        bc_weight: float = 0.1,
        resolution: int = 128,
        batch_size: int = 8,
        amp: bool = True,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        cfg = Config(seed=seed)
        cfg.model = ModelConfig(
            name=backbone, width=width, n_layers=n_layers, modes=modes, dropout=dropout
        )
        cfg.train.epochs = epochs
        cfg.train.lr = lr
        cfg.train.physics_weight = physics_weight
        cfg.train.bc_weight = bc_weight
        cfg.train.amp = amp
        cfg.train.device = device
        cfg.data.resolution = resolution
        cfg.data.batch_size = batch_size
        cfg.data.seed = seed
        self.config = cfg
        self.corrector_type = corrector
        self.corrector_epochs = 0 if corrector == "none" else corrector_epochs

        # Populated by fit().
        self.engine = None
        self.predictor = None
        self.metrics_: dict[str, float] = {}
        self.history_: dict[str, Any] = {}
        self.calibrator = None
        self._data_kw: dict[str, Any] = {}
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        data: str = "synthetic",
        *,
        task: str = "scarce",
        n_train: int = 200,
        n_val: int = 100,
        root: str = "data",
        cache_dir: str | None = None,
        download: bool = True,
        epochs: int | None = None,
        out: str | None = None,
        verbose: bool = True,
    ) -> NeuroForge:
        """Train the backbone (+corrector) on ``data`` and return ``self``.

        ``data`` is ``'synthetic'`` (zero-download smoke substrate) or
        ``'airfrans'`` (downloaded/cached on demand).
        """
        from neuroforge.physics.residuals import PhysicsChecker
        from neuroforge.solver.engine import NeuroForgeEngine, Predictor
        from neuroforge.train.recipes import train_recipe

        cfg = self.config
        cfg.data = DataConfig(
            source=data, root=root, task=task, resolution=cfg.data.resolution,
            n_train=n_train, n_val=n_val, batch_size=cfg.data.batch_size,
            cache_dir=cache_dir, download=download, seed=cfg.seed,
        )
        if epochs is not None:
            cfg.train.epochs = int(epochs)
        self._data_kw = dict(
            source=data, task=task, root=root, n_train=n_train, n_val=n_val,
            resolution=cfg.data.resolution, cache_dir=cache_dir, download=download,
        )

        result = train_recipe(
            cfg, corrector_type=self.corrector_type,
            corrector_epochs=self.corrector_epochs, download=download,
            out=out, verbose=verbose,
        )
        device = result["device"]
        self.predictor = Predictor(result["model"], result["normalizer"], device=device)
        self.engine = NeuroForgeEngine(
            self.predictor, PhysicsChecker(cfg.physics),
            corrector=result["corrector"], config=cfg,
        )
        self.metrics_ = dict(result["val_errors"])
        self.history_ = {
            "train": result["history"], "corrector": result["corrector_history"]
        }
        self._model = result["model"]
        self._normalizer = result["normalizer"]
        self._nu = float(getattr(result["trainer"], "nu", 1.5e-5))
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, case: FlowCase) -> FlowField:
        """Fast one-shot backbone prediction (no correction)."""
        self._check_fitted()
        return self.predictor.predict(case)

    def solve(self, case: FlowCase, max_iters: int | None = None) -> SolveResult:
        """Full self-correcting solve: predict → check → correct → diagnostics."""
        self._check_fitted()
        return self.engine.solve(case, max_iters=max_iters)

    # ------------------------------------------------------------------ #
    # Evaluation & calibration
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        pairs: list[tuple[FlowCase, FlowField]] | None = None,
        *,
        corrected: bool = False,
        limit: int | None = None,
    ) -> dict[str, float]:
        """AirfRANS-protocol metrics (per-channel MSE, surface MSE, ρ_Cl/ρ_Cd,
        residual↔error correlation) on held-out data.

        ``corrected=False`` evaluates the raw backbone; ``True`` evaluates the
        full self-correcting solve. ``pairs`` defaults to the validation split.
        """
        self._check_fitted()
        from neuroforge.physics.evaluation import evaluate_cases
        from neuroforge.physics.residuals import PhysicsChecker

        if pairs is None:
            pairs = self._load_val(limit)

        if corrected:
            def predict_fn(c):
                return self.solve(c).field
        else:
            predict_fn = self.predict
        return evaluate_cases(predict_fn, pairs, checker=PhysicsChecker())

    def ablate_corrector(
        self, pairs: list[tuple[FlowCase, FlowField]] | None = None, limit: int | None = None
    ) -> dict[str, dict[str, float]]:
        """Corrector on/off ablation: returns ``{'backbone': {...}, 'corrected': {...}}``."""
        if pairs is None:
            pairs = self._load_val(limit)
        return {
            "backbone": self.evaluate(pairs, corrected=False),
            "corrected": self.evaluate(pairs, corrected=True),
        }

    def val_data(self, limit: int | None = None) -> list[tuple[FlowCase, FlowField]]:
        """The validation ``(case, ground_truth)`` pairs (for plotting / metrics)."""
        self._check_fitted()
        return self._load_val(limit)

    def calibrate(
        self,
        pairs: list[tuple[FlowCase, FlowField]] | None = None,
        *,
        alpha: float = 0.1,
        channel: int = 2,
        n_samples: int = 8,
        limit: int | None = None,
    ):
        """Fit conformal trust calibration (coverage guarantee). Needs dropout>0.

        Returns the fitted :class:`ConformalCalibrator` (also stored on the
        model) or ``None`` if no usable uncertainty is available.
        """
        self._check_fitted()
        from neuroforge.models.ensemble import MCDropoutUQ
        from neuroforge.physics.calibration import calibrate_from_cases

        if pairs is None:
            pairs = self._load_val(limit)
        if float(self.config.model.dropout) <= 0:
            print("[calibrate] model has dropout=0; set dropout>0 (e.g. 0.05) to "
                  "enable MC-dropout uncertainty, then re-fit. Skipping.")
            return None

        base = MCDropoutUQ(self._model)

        class _UQ:  # bind n_samples for the calibrator's fixed-arity call
            def predict_with_uncertainty(self, x):
                return base.predict_with_uncertainty(x, n_samples=n_samples)

        self.calibrator = calibrate_from_cases(
            self.predictor.predict, _UQ(), pairs, alpha=alpha, channel=channel
        )
        return self.calibrator

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> NeuroForge:
        """Save the trained model + corrector + normaliser to ``path``."""
        self._check_fitted()
        from neuroforge.train.trainer import Trainer

        trainer = Trainer(self._model, self.config, self._normalizer, nu=self._nu)
        trainer.save(path, corrector=self.engine.corrector)
        return self

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> NeuroForge:
        """Load a model saved by :meth:`save` (or any Trainer checkpoint)."""
        from neuroforge.solver.engine import NeuroForgeEngine

        obj = cls.__new__(cls)
        engine = NeuroForgeEngine.from_checkpoint(path)
        obj.engine = engine
        obj.predictor = engine.predictor
        obj.config = engine.config
        obj.corrector_type = type(engine.corrector).__name__.lower() if engine.corrector else "none"
        obj.corrector_epochs = 0
        obj.metrics_ = {}
        obj.history_ = {}
        obj.calibrator = None
        obj._data_kw = {}
        obj._model = engine.predictor.model
        obj._normalizer = engine.predictor.normalizer
        obj._nu = 1.5e-5
        obj._fitted = True
        return obj

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("NeuroForge is not fitted yet — call .fit(...) first.")

    def _load_val(self, limit: int | None = None):
        kw = self._data_kw
        if not kw:
            raise RuntimeError("No validation data available; pass `pairs=` explicitly.")
        n_val = limit if limit is not None else kw["n_val"]
        if kw["source"] == "airfrans":
            from neuroforge.data.airfrans_loader import load_airfrans

            return load_airfrans(
                root=kw["root"], task=kw["task"], train=False,
                resolution=kw["resolution"], limit=n_val, cache_dir=kw["cache_dir"],
                download=kw["download"],
            )
        from neuroforge.data.synthetic import SyntheticRANS

        gen = SyntheticRANS(resolution=kw["resolution"], seed=self.config.seed)
        start = kw["n_train"]
        return [(gen.sample_case(i), gen.solve(gen.sample_case(i)))
                for i in range(start, start + n_val)]

    def __repr__(self) -> str:
        m = self.config.model
        state = "fitted" if self._fitted else "unfitted"
        return (f"NeuroForge(backbone='{m.name}', width={m.width}, modes={m.modes}, "
                f"corrector='{self.corrector_type}', {state})")
