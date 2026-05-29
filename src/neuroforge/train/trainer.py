"""Training loops for the backbone and the second-stage correction network.

:class:`Trainer` owns the optimisation of a :class:`NeuralSolver` backbone with
the :class:`~neuroforge.train.losses.CompositeLoss` (data + physics + BC), plus a
second-stage routine, :meth:`Trainer.fit_corrector`, that freezes the backbone
and trains a :class:`~neuroforge.models.correction.LocalCorrectionNet` to predict
the residual-driven correction toward the target.

Checkpoints are plain ``torch.save`` dicts whose schema is shared verbatim with
:meth:`NeuroForgeEngine.from_checkpoint` (see :meth:`Trainer.save`).
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from neuroforge.core.config import Config, ModelConfig
from neuroforge.data.datamodule import Normalizer
from neuroforge.models.base import NeuralSolver, build_model
from neuroforge.models.base import CorrectionNetwork

from .losses import CompositeLoss
from .schedule import WarmupCosineScheduler

__all__ = ["Trainer"]


def _model_kwargs(model_cfg: ModelConfig) -> dict:
    """Build the kwargs accepted by ``build_model`` for this backbone.

    Only the universally-accepted hyper-parameters plus the per-family extras
    that the target class actually takes are passed, so e.g. ``modes`` is not
    forwarded to a UNet (which would reject it). Selection is by the model's own
    ``__init__`` signature, so it stays correct as new backbones are added.
    """
    import inspect

    from neuroforge.models.base import MODEL_REGISTRY

    kwargs = {
        "in_channels": model_cfg.in_channels,
        "out_channels": model_cfg.out_channels,
        "width": model_cfg.width,
        "n_layers": model_cfg.n_layers,
        "dropout": model_cfg.dropout,
    }
    extras = {
        "modes": model_cfg.modes,
        "n_heads": model_cfg.n_heads,
        "n_slices": model_cfg.n_slices,
        "activation": model_cfg.activation,
    }
    cls = MODEL_REGISTRY.get(model_cfg.name.lower())
    if cls is not None:
        try:
            params = set(inspect.signature(cls.__init__).parameters)
        except (TypeError, ValueError):
            params = set()
        for k, v in extras.items():
            if k in params:
                kwargs[k] = v
    return kwargs


def _resolve_device(name: str) -> torch.device:
    """Map a config device string to a concrete torch device (CPU-first)."""
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    """Train a backbone (and optionally a corrector) on flow-field data.

    Parameters
    ----------
    model : NeuralSolver
        The backbone to optimise.
    cfg : Config
        Full configuration (training hyper-parameters under ``cfg.train``).
    normalizer : Normalizer
        Fitted normaliser; stored in the checkpoint and used by the loss.
    nu : float, optional
        Laminar kinematic viscosity for the physics loss term.
    """

    def __init__(
        self,
        model: NeuralSolver,
        cfg: Config,
        normalizer: Normalizer,
        nu: float = 1.5e-5,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.normalizer = normalizer
        self.nu = float(nu)
        self.device = _resolve_device(cfg.train.device)
        self.model.to(self.device)
        self.loss_fn = CompositeLoss(cfg, normalizer, nu=self.nu)

    # ------------------------------------------------------------------ #
    # Stage 1: backbone training
    # ------------------------------------------------------------------ #

    def fit(self, train_loader, val_loader) -> dict:
        """Train the backbone for ``cfg.train.epochs`` epochs.

        Returns
        -------
        dict
            History with per-epoch ``train_loss`` / ``val_loss`` lists and the
            per-term breakdown (``train_data``/``train_physics``/``train_bc``).
        """
        tc = self.cfg.train
        opt = torch.optim.Adam(
            self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay
        )
        total_steps = max(len(train_loader) * max(tc.epochs, 1), 1)
        sched = WarmupCosineScheduler(opt, total_steps, tc.lr, warmup_frac=0.05)

        # Mixed precision (CUDA only). bfloat16 on Ampere+ (no scaler needed),
        # otherwise float16 with gradient scaling. The FNO spectral conv forces
        # float32 internally, so the FFT/complex math stays safe under autocast.
        use_amp = bool(tc.amp) and self.device.type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if (use_amp and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(use_amp and amp_dtype == torch.float16)
        )
        if use_amp:
            print(f"[train] AMP enabled (dtype={'bfloat16' if amp_dtype == torch.bfloat16 else 'float16'})")

        history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_data": [], "train_physics": [], "train_bc": [],
        }

        global_step = 0
        for epoch in range(tc.epochs):
            self.model.train()
            agg = {"loss": 0.0, "data": 0.0, "physics": 0.0, "bc": 0.0}
            n_batches = 0
            for batch in train_loader:
                inp = batch["input"].to(self.device)
                target = batch["target"].to(self.device)
                mask = batch["mask"].to(self.device)

                sched.step()
                opt.zero_grad(set_to_none=True)
                # Autocast only the (expensive) forward; the physics-aware loss
                # is computed in float32 for numerical stability.
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                    pred = self.model(inp)
                loss, parts = self.loss_fn(pred.float(), target, inp, mask)

                if not torch.isfinite(loss):
                    # Skip a pathological batch rather than poisoning the weights.
                    global_step += 1
                    continue

                scaler.scale(loss).backward()
                if tc.grad_clip and tc.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), tc.grad_clip
                    )
                scaler.step(opt)
                scaler.update()

                for k in agg:
                    agg[k] += parts[k]
                n_batches += 1
                global_step += 1
                if tc.log_every and global_step % tc.log_every == 0:
                    print(
                        f"[train] epoch {epoch} step {global_step} "
                        f"loss={parts['loss']:.4e} data={parts['data']:.4e} "
                        f"phys={parts['physics']:.4e} bc={parts['bc']:.4e}"
                    )

            n_batches = max(n_batches, 1)
            train_loss = agg["loss"] / n_batches
            val_loss = self._evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_data"].append(agg["data"] / n_batches)
            history["train_physics"].append(agg["physics"] / n_batches)
            history["train_bc"].append(agg["bc"] / n_batches)
            print(
                f"[epoch {epoch}] train_loss={train_loss:.4e} "
                f"val_loss={val_loss:.4e}"
            )

        return history

    @torch.no_grad()
    def _evaluate(self, val_loader) -> float:
        """Mean composite loss over the validation loader."""
        if val_loader is None:
            return float("nan")
        self.model.eval()
        total = 0.0
        n = 0
        for batch in val_loader:
            inp = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            mask = batch["mask"].to(self.device)
            pred = self.model(inp)
            loss, _ = self.loss_fn(pred, target, inp, mask)
            if torch.isfinite(loss):
                total += float(loss.detach())
                n += 1
        return total / max(n, 1)

    # ------------------------------------------------------------------ #
    # Stage 2: corrector training (residual-driven)
    # ------------------------------------------------------------------ #

    def fit_corrector(
        self,
        train_loader,
        corrector: CorrectionNetwork,
        epochs: int | None = None,
    ) -> dict:
        """Train ``corrector`` to predict the correction toward the target.

        The (frozen, eval-mode) backbone produces a normalised prediction; we
        denormalise it, compute the 3-channel physics residual map on the
        physical field, renormalise the residual, and train the corrector

            ``corrector(field=pred_norm, residual=residual_norm, geom=inp) -> delta``

        to match ``target_delta = target_norm - pred_norm`` under a masked MSE.
        The backbone weights are not updated.

        Returns
        -------
        dict
            History with a per-epoch ``corrector_loss`` list.
        """
        from neuroforge.physics.residuals import physics_residual_torch

        epochs = int(epochs) if epochs is not None else self.cfg.train.epochs
        corrector = corrector.to(self.device)
        loss_fn = self.loss_fn  # for dx/dy + nu
        dx, dy, nu = loss_fn.dx, loss_fn.dy, self.nu

        opt = torch.optim.Adam(
            corrector.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )

        # Freeze the backbone.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Residual normalisation scale: physical advective scale U^2/L is data-
        # dependent; a robust per-batch std keeps the corrector input ~O(1).
        history: dict[str, list] = {"corrector_loss": []}

        for epoch in range(epochs):
            corrector.train()
            agg = 0.0
            n = 0
            for batch in train_loader:
                inp = batch["input"].to(self.device)
                target = batch["target"].to(self.device)
                mask = batch["mask"].to(self.device)
                fluid = (mask > 0.5).to(inp.dtype)

                with torch.no_grad():
                    pred_norm = self.model(inp)
                    pred_phys = self.normalizer.denorm_out(pred_norm)
                    inp_phys = self.normalizer.denorm_in(inp)
                    res = physics_residual_torch(pred_phys, inp_phys, dx, dy, nu)
                    residual = torch.cat(
                        [res["continuity"], res["momentum_x"], res["momentum_y"]],
                        dim=1,
                    )
                    # Per-channel standardisation of the residual maps (stable input).
                    std = residual.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-6)
                    residual_norm = residual / std.unsqueeze(-1)
                    target_delta = (target - pred_norm).detach()
                    pred_norm_d = pred_norm.detach()

                opt.zero_grad(set_to_none=True)
                delta = corrector(field=pred_norm_d, residual=residual_norm, geom=inp)
                diff2 = (delta - target_delta) ** 2
                n_fluid = fluid.sum().clamp_min(1.0)
                loss = (diff2 * fluid).sum() / (n_fluid * delta.shape[1])

                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if self.cfg.train.grad_clip and self.cfg.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        corrector.parameters(), self.cfg.train.grad_clip
                    )
                opt.step()
                agg += float(loss.detach())
                n += 1

            ep_loss = agg / max(n, 1)
            history["corrector_loss"].append(ep_loss)
            print(f"[corrector epoch {epoch}] loss={ep_loss:.4e}")

        # Re-enable grads on the backbone (caller may keep training it).
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._last_corrector = corrector
        return history

    # ------------------------------------------------------------------ #
    # Checkpoint I/O
    # ------------------------------------------------------------------ #

    def save(self, path: str, corrector: CorrectionNetwork | None = None) -> None:
        """Save the model (+ optional corrector), normalizer and config to ``path``.

        The checkpoint schema is shared verbatim with
        :meth:`NeuroForgeEngine.from_checkpoint`.
        """
        import neuroforge

        corrector = corrector if corrector is not None else getattr(
            self, "_last_corrector", None
        )
        model_cfg = self._infer_model_config()

        corr_state = None
        corr_cfg = None
        if corrector is not None:
            corr_state = corrector.state_dict()
            corr_cfg = {
                "width": int(getattr(corrector, "width", 24)),
                "n_layers": int(getattr(corrector, "n_layers", 3)),
            }

        ckpt = {
            "model_state": self.model.state_dict(),
            "model_config": asdict(model_cfg),
            "normalizer": self.normalizer.state_dict(),
            "nu": float(self.nu),
            "corrector_state": corr_state,
            "corrector_config": corr_cfg,
            "neuroforge_version": neuroforge.__version__,
        }
        torch.save(ckpt, path)

    def _infer_model_config(self) -> ModelConfig:
        """Reconstruct a ModelConfig describing the current backbone."""
        m = self.model
        name = getattr(m, "registry_name", self.cfg.model.name)
        base = self.cfg.model
        dp = getattr(m, "dropout_p", None)
        dropout = float(dp) if isinstance(dp, (int, float)) else float(base.dropout)
        return ModelConfig(
            name=name,
            in_channels=int(getattr(m, "in_channels", base.in_channels)),
            out_channels=int(getattr(m, "out_channels", base.out_channels)),
            width=int(getattr(m, "width", base.width)),
            n_layers=int(getattr(m, "n_layers", base.n_layers)),
            modes=int(getattr(m, "modes", base.modes)),
            n_heads=int(getattr(m, "n_heads", base.n_heads)),
            n_slices=int(getattr(m, "n_slices", base.n_slices)),
            dropout=dropout,
            # `activation` is stored on the model as an nn.Module, not the string
            # name, so it cannot be recovered from the backbone; keep the config's.
            activation=base.activation,
        )

    @staticmethod
    def load(
        path: str, map_location: str = "cpu"
    ) -> tuple[NeuralSolver, Normalizer, ModelConfig]:
        """Rebuild ``(model, normalizer, ModelConfig)`` from a checkpoint.

        Mirrors the save schema; the corrector (if present) is *not* returned
        here — :meth:`NeuroForgeEngine.from_checkpoint` rebuilds it.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        mc = ckpt["model_config"]
        model_cfg = ModelConfig(**mc)
        model = build_model(model_cfg.name, **_model_kwargs(model_cfg))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
        return model, normalizer, model_cfg
