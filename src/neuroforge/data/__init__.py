"""Data layer: synthetic generator, AirfRANS loader, rasterisation, datamodule."""

from .datamodule import (
    FlowDataset,
    Normalizer,
    build_dataloaders,
    collate_flow,
)
from .rasterize import rasterize_point_cloud
from .synthetic import SyntheticRANS

__all__ = [
    "SyntheticRANS",
    "rasterize_point_cloud",
    "Normalizer",
    "FlowDataset",
    "collate_flow",
    "build_dataloaders",
]


def load_airfrans(*args, **kwargs):
    """Lazy proxy to :func:`neuroforge.data.airfrans_loader.load_airfrans`."""
    from .airfrans_loader import load_airfrans as _f

    return _f(*args, **kwargs)


def download_airfrans(*args, **kwargs):
    """Lazy proxy to :func:`neuroforge.data.airfrans_loader.download_airfrans`."""
    from .airfrans_loader import download_airfrans as _f

    return _f(*args, **kwargs)
