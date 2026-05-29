"""Compare neural backbones on synthetic CFD data.

Trains each registered backbone (fno / unet / deeponet / transformer) on a small
synthetic dataset, then evaluates on the held-out validation set:

* relative L2 field errors (via :func:`neuroforge.physics.metrics.field_errors`),
* inference time per case,
* parameter count, and
* mean continuity residual.

Prints a ``rich`` table and returns a dict keyed by model name. Sized to run in
well under ~2 minutes on CPU (tiny grids, few cases, few epochs).
"""

from __future__ import annotations

import os
import sys
import time

# Ensure `import neuroforge` works when run as a script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

import neuroforge  # noqa: E402,F401  (caps BLAS threads on import)

_MODELS = ("fno", "unet", "deeponet", "transformer")


def _default_cfg() -> dict:
    return {
        "models": _MODELS,
        "resolution": 32,
        "n_train": 8,
        "n_val": 4,
        "batch_size": 4,
        "epochs": 4,
        "width": 16,
        "n_layers": 3,
        "seed": 0,
    }


def _seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_benchmarks(cfg: dict | None = None) -> dict:
    """Train + evaluate each backbone on synthetic data; return a results dict.

    Parameters
    ----------
    cfg : dict, optional
        Overrides for the default benchmark settings (keys: ``models``,
        ``resolution``, ``n_train``, ``n_val``, ``batch_size``, ``epochs``,
        ``width``, ``n_layers``, ``seed``).

    Returns
    -------
    dict
        ``{model_name: {rel_l2_u, rel_l2_v, rel_l2_p, rel_l2_speed,
        infer_ms_per_case, n_params, continuity_residual_mean}}``.
    """
    import torch

    from neuroforge.core.config import Config, DataConfig
    from neuroforge.data.datamodule import build_dataloaders
    from neuroforge.models.base import build_model
    from neuroforge.physics.metrics import field_errors
    from neuroforge.physics.residuals import continuity_residual
    from neuroforge.train.trainer import Trainer

    settings = _default_cfg()
    if cfg:
        settings.update(cfg)

    _seed(settings["seed"])

    base = Config()
    base.data = DataConfig(
        source="synthetic",
        resolution=settings["resolution"],
        n_train=settings["n_train"],
        n_val=settings["n_val"],
        batch_size=settings["batch_size"],
        seed=settings["seed"],
    )
    base.train.epochs = settings["epochs"]
    base.train.log_every = 0

    # Build the data once (shared across all models for a fair comparison).
    train_loader, val_loader, normalizer = build_dataloaders(base.data)

    # Validation reference (case, physical ground-truth field) pairs. We
    # regenerate them from SyntheticRANS using the same indices build_dataloaders
    # uses for the validation split (n_train .. n_train + n_val).
    from neuroforge.data.synthetic import SyntheticRANS

    gen = SyntheticRANS(resolution=settings["resolution"], seed=settings["seed"])
    val_pairs = [
        (c := gen.sample_case(i), gen.solve(c))
        for i in range(settings["n_train"], settings["n_train"] + settings["n_val"])
    ]

    results: dict[str, dict] = {}

    for name in settings["models"]:
        _seed(settings["seed"])
        model = build_model(
            name, width=settings["width"], n_layers=settings["n_layers"]
        )
        trainer = Trainer(model, base, normalizer, nu=1.5e-5)
        trainer.fit(train_loader, val_loader)
        model.eval()

        from neuroforge.solver.engine import Predictor

        predictor = Predictor(model, normalizer, device="cpu")

        errs = {"u": [], "v": [], "p": [], "speed": []}
        cont = []
        t0 = time.perf_counter()
        for case, ref in val_pairs:
            pred = predictor.predict(case)
            e = field_errors(pred, ref)
            for k in errs:
                errs[k].append(e[k])
            mask = np.asarray(pred.mask) > 0.5
            r = continuity_residual(pred)
            cont.append(float(np.mean(np.abs(r[mask]))) if mask.any() else float("nan"))
        elapsed = time.perf_counter() - t0
        n_cases = max(len(val_pairs), 1)

        results[name] = {
            "rel_l2_u": float(np.mean(errs["u"])),
            "rel_l2_v": float(np.mean(errs["v"])),
            "rel_l2_p": float(np.mean(errs["p"])),
            "rel_l2_speed": float(np.mean(errs["speed"])),
            "infer_ms_per_case": float(1000.0 * elapsed / n_cases),
            "n_params": int(model.num_params()),
            "continuity_residual_mean": float(np.mean(cont)),
        }

    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    """Render the benchmark results as a rich table (fallback: plain text)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:  # pragma: no cover
        print("model            relL2_u  relL2_p  ms/case   params   cont_res")
        for name, r in results.items():
            print(f"{name:<15} {r['rel_l2_u']:.4f}  {r['rel_l2_p']:.4f}  "
                  f"{r['infer_ms_per_case']:.2f}  {r['n_params']}  "
                  f"{r['continuity_residual_mean']:.4e}")
        return

    table = Table(title="NeuroForge CFD — backbone benchmark (synthetic)",
                  header_style="bold cyan")
    table.add_column("model")
    table.add_column("relL2 u", justify="right")
    table.add_column("relL2 v", justify="right")
    table.add_column("relL2 p", justify="right")
    table.add_column("relL2 speed", justify="right")
    table.add_column("ms/case", justify="right")
    table.add_column("params", justify="right")
    table.add_column("cont. resid.", justify="right")
    for name in sorted(results):
        r = results[name]
        table.add_row(
            name,
            f"{r['rel_l2_u']:.4f}",
            f"{r['rel_l2_v']:.4f}",
            f"{r['rel_l2_p']:.4f}",
            f"{r['rel_l2_speed']:.4f}",
            f"{r['infer_ms_per_case']:.2f}",
            f"{r['n_params']:,}",
            f"{r['continuity_residual_mean']:.3e}",
        )
    Console().print(table)


if __name__ == "__main__":
    run_benchmarks()
