"""Train NeuroForge CFD on the AirfRANS dataset.

Thin CLI wrapper around ``neuroforge.train.train_recipe``. GPU-aware
(``--device auto`` picks CUDA on Colab). Rasterised data is cached so repeated
runs are fast.

Examples
--------
Quick run on the small 'scarce' split::

    PYTHONPATH=src python scripts/train_airfrans.py --download \
        --task scarce --n-train 200 --n-val 100 --epochs 50 --out checkpoints/airfrans.pt

Full split, bigger FNO::

    python scripts/train_airfrans.py --task full --n-train 800 --n-val 200 \
        --model fno --width 64 --modes 24 --layers 5 --epochs 150 --batch-size 8 \
        --cache-dir data/cache --out checkpoints/airfrans_full.pt
"""

from __future__ import annotations

import argparse

import neuroforge  # noqa: F401  (caps math-library threads on import)
from neuroforge.core.config import Config, DataConfig, ModelConfig
from neuroforge.train import train_recipe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NeuroForge CFD on AirfRANS.")
    # data
    p.add_argument("--source", default="airfrans", choices=["airfrans", "synthetic"])
    p.add_argument("--root", default="data", help="dataset root directory")
    p.add_argument("--task", default="scarce", choices=["full", "scarce", "reynolds", "aoa"])
    p.add_argument("--download", action="store_true", help="download AirfRANS if missing")
    p.add_argument("--res", type=int, default=128, help="rasterisation resolution")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cache-dir", default=None, help="cache rasterised pairs here")
    # model
    p.add_argument("--model", default="fno", choices=["fno", "geo_fno", "transformer", "unet", "deeponet"])
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    # train
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--physics-weight", type=float, default=0.1)
    p.add_argument("--bc-weight", type=float, default=0.1)
    p.add_argument("--corrector-epochs", type=int, default=20)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default="checkpoints/airfrans.pt")
    return p


def config_from_args(args) -> Config:
    cfg = Config()
    cfg.data = DataConfig(
        source=args.source, root=args.root, task=args.task, resolution=args.res,
        n_train=args.n_train, n_val=args.n_val, batch_size=args.batch_size,
        num_workers=args.workers, cache_dir=args.cache_dir,
    )
    cfg.model = ModelConfig(
        name=args.model, width=args.width, n_layers=args.layers,
        modes=args.modes, dropout=args.dropout,
    )
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.physics_weight = args.physics_weight
    cfg.train.bc_weight = args.bc_weight
    cfg.train.device = args.device
    cfg.train.log_every = args.log_every
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    result = train_recipe(
        cfg, download=args.download, corrector_epochs=args.corrector_epochs, out=args.out,
    )
    print("\n=== summary ===")
    print(f"device        : {result['device']}")
    print(f"params        : {result['n_params']:,}")
    print(f"train / val   : {result['n_train']} / {result['n_val']}")
    ve = result["val_errors"]
    print("val rel-L2    : " + ", ".join(f"{k}={ve[k]:.3f}" for k in ("u", "v", "p", "speed") if k in ve))
    print(f"infer         : {ve.get('infer_ms_per_sample', 0):.2f} ms/sample")
    print(f"checkpoint    : {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
