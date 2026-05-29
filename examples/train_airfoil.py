"""NeuroForge CFD — training on the AirfRANS dataset.

This example mirrors ``demo_synthetic.py`` but uses the real **AirfRANS**
dataset of RANS simulations over airfoils. AirfRANS is an optional dependency
and the dataset is a sizeable download, so both are guarded: the script prints
clear instructions and exits gracefully if ``airfrans`` is not installed.

Prerequisites
-------------
1. Install the data extras::

       pip install 'neuroforge-cfd[data]'      # pulls in `airfrans`

2. The dataset is downloaded automatically on first use via
   ``neuroforge.data.airfrans_loader.download_airfrans`` (you can also point the
   loader at an existing copy via ``DataConfig.root``).

Run it with::

    python examples/train_airfoil.py
"""

from __future__ import annotations


def _airfrans_available() -> bool:
    """True if the optional ``airfrans`` package can be imported."""
    import importlib.util

    return importlib.util.find_spec("airfrans") is not None


def main() -> None:
    if not _airfrans_available():
        print(
            "The 'airfrans' package is not installed.\n"
            "Install it with:  pip install 'neuroforge-cfd[data]'\n"
            "(or: pip install airfrans), then re-run this example."
        )
        return

    # Heavy imports are local so the module imports without the training stack.
    from neuroforge.core.config import Config, DataConfig
    from neuroforge.data.datamodule import build_dataloaders
    from neuroforge.models.base import build_model
    from neuroforge.physics.metrics import field_errors
    from neuroforge.train.trainer import Trainer

    # Optional explicit download step (build_dataloaders will also fetch lazily):
    #
    #     from neuroforge.data.airfrans_loader import download_airfrans
    #     download_airfrans(root="data")

    # 1) Configure the AirfRANS data source. The 'scarce' task is the smallest.
    cfg = Config()
    cfg.data = DataConfig(
        source="airfrans",
        root="data",
        task="scarce",
        resolution=128,
        batch_size=2,
    )
    cfg.model.name = "transformer"   # Transolver-style physics-attention backbone
    cfg.train.epochs = 10

    # 2) Build dataloaders (rasterises the AirfRANS point clouds onto the grid).
    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)

    # 3) Build and train the model.
    model = build_model(
        cfg.model.name,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        width=cfg.model.width,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        n_slices=cfg.model.n_slices,
    )
    trainer = Trainer(model, cfg, normalizer)
    history = trainer.fit(train_loader, val_loader)
    trainer.save("checkpoints/airfrans.pt")
    print("training history:", history)

    # 4) Evaluate: relative-L2 field errors on the first validation sample.
    #    (Here we simply show how field_errors compares a prediction to a
    #    reference field; wire it to your own held-out cases as needed.)
    print("Trained. Use field_errors(pred, ref) to evaluate against ground truth.")
    _ = field_errors  # referenced for the reader; evaluation loop omitted


if __name__ == "__main__":
    main()
