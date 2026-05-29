"""NeuroForge CFD — explicit synthetic train -> solve -> report walkthrough.

Unlike ``quickstart.py`` (which uses ``NeuroForgeEngine.pretrained()``), this
example spells out every step of the pipeline on the zero-download *synthetic*
dataset: build dataloaders, train a small model with the Trainer, assemble the
self-correcting engine around it, solve a case, and write a report. It is the
fastest fully-local end-to-end path and is CPU-friendly.

Run it with::

    python examples/demo_synthetic.py
"""

from __future__ import annotations


def main() -> None:
    # All heavy imports are local so this module imports even on a partially
    # built source tree (the solver/trainer are built by another agent).
    from neuroforge import FlowCase
    from neuroforge.core.config import Config
    from neuroforge.data.datamodule import build_dataloaders
    from neuroforge.models.base import build_model
    from neuroforge.solver.engine import NeuroForgeEngine, Predictor
    from neuroforge.physics.residuals import PhysicsChecker
    from neuroforge.train.trainer import Trainer

    # 1) Configure a tiny, CPU-friendly run on synthetic data.
    cfg = Config()
    cfg.data.source = "synthetic"
    cfg.data.resolution = 64
    cfg.data.n_train = 16
    cfg.data.n_val = 4
    cfg.data.batch_size = 4
    cfg.model.name = "fno"
    cfg.model.width = 16
    cfg.model.modes = 8
    cfg.train.epochs = 2  # just enough to demonstrate the loop

    # 2) Build dataloaders + a fitted Normalizer from the synthetic generator.
    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)

    # 3) Build and train the backbone.
    model = build_model(
        cfg.model.name,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        width=cfg.model.width,
        n_layers=cfg.model.n_layers,
        modes=cfg.model.modes,
    )
    trainer = Trainer(model, cfg, normalizer, nu=1.5e-5)
    history = trainer.fit(train_loader, val_loader)
    print("training history:", history)

    # 4) Assemble the self-correcting engine around the trained model.
    predictor = Predictor(model, normalizer, device="cpu")
    engine = NeuroForgeEngine(predictor, PhysicsChecker(cfg.physics), config=cfg)

    # 5) Solve a fresh case and save a report.
    case = FlowCase.from_airfoil("naca2412", aoa=4, reynolds=2e6,
                                 u_inf=30.0, resolution=cfg.data.resolution)
    result = engine.solve(case)
    print("metrics:", result.summary())
    print("report:", result.save_report("reports/demo_synthetic.html"))


if __name__ == "__main__":
    main()
