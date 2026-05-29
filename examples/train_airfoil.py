"""NeuroForge CFD — training on the real AirfRANS dataset.

Uses the high-level :func:`neuroforge.train.train_recipe` (the same code path as
``scripts/train_airfrans.py`` and the Colab notebook). AirfRANS is an optional
dependency and a sizeable download, so both are guarded: the script prints clear
instructions and exits gracefully if ``airfrans`` is not installed.

Prerequisites
-------------
1. Install the data extras::

       pip install 'neuroforge-cfd[data]'      # pulls in `airfrans`

2. The dataset downloads automatically on first use (``download=True`` below),
   or point ``DataConfig.root`` at an existing copy.

Run it with::

    python examples/train_airfoil.py

For a serious run prefer the CLI script (more knobs, caching, GPU auto-detect)::

    python scripts/train_airfrans.py --download --task scarce --epochs 80 --out checkpoints/airfrans.pt
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

    from neuroforge.core.config import Config, DataConfig, ModelConfig
    from neuroforge.train import train_recipe

    # Configure a small AirfRANS run (the 'scarce' task is the smallest split).
    cfg = Config()
    cfg.data = DataConfig(
        source="airfrans", root="data", task="scarce", resolution=128,
        n_train=100, n_val=50, batch_size=4, cache_dir="data/cache",
    )
    cfg.model = ModelConfig(name="fno", width=48, modes=20, n_layers=4)
    cfg.train.epochs = 40
    cfg.train.device = "auto"          # CUDA if available, else CPU

    # Download (if needed) -> dataloaders -> train backbone + corrector -> evaluate.
    result = train_recipe(
        cfg, download=True, corrector_epochs=10, out="checkpoints/airfrans.pt",
    )

    ve = result["val_errors"]
    print("\nValidation relative-L2 errors:",
          {k: round(ve[k], 3) for k in ("u", "v", "p", "speed") if k in ve})
    print("Checkpoint saved to:", result["checkpoint"])
    print("Build an engine with: NeuroForgeEngine.from_checkpoint('checkpoints/airfrans.pt')")


if __name__ == "__main__":
    main()
