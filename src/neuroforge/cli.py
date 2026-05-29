"""Command-line interface for NeuroForge CFD.

``neuroforge <subcommand>`` (entry point ``neuroforge.cli:main``). Heavy
dependencies (the solver engine, trainer, dataloaders, benchmarks) are imported
*lazily* inside each subcommand so that ``neuroforge info`` works even while the
engine is still being built, and so importing this module is cheap.

Subcommands
-----------
info        Print version, available models, and the torch device.
demo        Run the bundled end-to-end demo and write a report.
train       Train a model from a YAML config and save a checkpoint.
predict     Solve a single airfoil case from a checkpoint.
benchmark   Run the model comparison benchmark suite.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

__all__ = ["main", "build_parser"]


# --------------------------------------------------------------------------- #
# Pretty printing (rich)
# --------------------------------------------------------------------------- #


def _console():
    """Return a rich Console (lazy import); falls back to None if unavailable."""
    try:
        from rich.console import Console

        return Console()
    except Exception:  # pragma: no cover - rich is a hard dep but be safe
        return None


def _print(msg: str) -> None:
    con = _console()
    if con is not None:
        con.print(msg)
    else:  # pragma: no cover
        print(_strip_markup(msg))


def _strip_markup(s: str) -> str:  # pragma: no cover - fallback only
    import re

    return re.sub(r"\[/?[^\]]*\]", "", s)


def _metrics_table(title: str, metrics: dict[str, Any]) -> None:
    """Render a metrics dict as a rich table (or plain text)."""
    con = _console()
    if con is None:  # pragma: no cover
        print(title)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return
    from rich.table import Table

    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k in sorted(metrics):
        v = metrics[k]
        try:
            sval = f"{float(v):.5g}"
        except (TypeError, ValueError):
            sval = str(v)
        table.add_row(str(k), sval)
    con.print(table)


# --------------------------------------------------------------------------- #
# torch device helper
# --------------------------------------------------------------------------- #


def _torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        return "cpu"
    except Exception:  # pragma: no cover
        return "cpu (torch unavailable)"


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #


def cmd_info(args: argparse.Namespace) -> int:
    """Print version, available models, and the torch device."""
    from neuroforge import __version__

    models: list[str] = []
    try:
        import neuroforge.models  # noqa: F401 - registers backbones on import
        from neuroforge.models.base import available_models

        models = available_models()
    except Exception as exc:  # pragma: no cover - partial tree
        _print(f"[yellow]could not list models: {exc}[/yellow]")

    _print(f"[bold]NeuroForge CFD[/bold] v{__version__}")
    _print(f"torch device : [green]{_torch_device()}[/green]")
    _print(
        "available models : "
        + (", ".join(models) if models else "[yellow](none registered)[/yellow]")
    )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the end-to-end demo, print metrics, and write a report."""
    try:
        from neuroforge.solver.engine import demo  # lazy: engine built in parallel
    except Exception as exc:
        _print(f"[red]demo unavailable — the solver engine is not ready:[/red] {exc}")
        return 2

    _print("[cyan]running demo (this may train a tiny model on first run)...[/cyan]")
    result = demo()
    _metrics_table(f"demo metrics — {result.case.name}", result.summary())

    report_path = args.report
    if report_path is None:
        import os

        out_dir = args.out or "reports"
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "demo.html")
    written = result.save_report(report_path)
    _print(f"[green]report written:[/green] {written}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model from a YAML config and save a checkpoint."""
    try:
        from neuroforge.core.config import Config
        from neuroforge.data.datamodule import build_dataloaders
        from neuroforge.models.base import build_model
        from neuroforge.train.trainer import Trainer
    except Exception as exc:
        _print(f"[red]training stack unavailable:[/red] {exc}")
        return 2

    cfg = Config.from_yaml(args.config)
    _print(f"[cyan]building dataloaders (source={cfg.data.source})...[/cyan]")
    train_loader, val_loader, normalizer = build_dataloaders(cfg.data)

    _print(f"[cyan]building model '{cfg.model.name}'...[/cyan]")
    model = build_model(
        cfg.model.name,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        width=cfg.model.width,
        n_layers=cfg.model.n_layers,
        modes=cfg.model.modes,
        dropout=cfg.model.dropout,
    )

    nu = 1.5e-5
    trainer = Trainer(model, cfg, normalizer, nu=nu)
    _print(f"[cyan]training for {cfg.train.epochs} epochs...[/cyan]")
    history = trainer.fit(train_loader, val_loader)

    out = args.out
    if out is None:
        import os

        os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
        out = os.path.join(cfg.train.ckpt_dir, f"{cfg.model.name}.pt")
    trainer.save(out)
    _print(f"[green]checkpoint saved:[/green] {out}")
    if isinstance(history, dict) and history:
        last = {k: (v[-1] if isinstance(v, (list, tuple)) and v else v)
                for k, v in history.items()}
        _metrics_table("final training metrics", last)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Solve a single airfoil case from a checkpoint and print metrics."""
    try:
        from neuroforge.core.types import FlowCase
        from neuroforge.solver.engine import NeuroForgeEngine
    except Exception as exc:
        _print(f"[red]predict unavailable — the solver engine is not ready:[/red] {exc}")
        return 2

    _print(f"[cyan]loading engine from {args.ckpt}...[/cyan]")
    engine = NeuroForgeEngine.from_checkpoint(args.ckpt)

    case = FlowCase.from_airfoil(
        airfoil=args.airfoil,
        aoa=args.aoa,
        reynolds=args.re,
        u_inf=args.u,
        resolution=args.res,
    )
    _print(f"[cyan]solving {case.name}...[/cyan]")
    result = engine.solve(case)
    _metrics_table(f"prediction — {case.name}", result.summary())

    if args.report:
        written = result.save_report(args.report)
        _print(f"[green]report written:[/green] {written}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the model comparison benchmark suite (repo-root ``benchmarks/``)."""
    try:
        from benchmarks.run_benchmarks import run_benchmarks  # lazy, built by QA agent
    except Exception as exc:
        _print(
            "[yellow]benchmark suite not available.[/yellow] "
            "Expected `benchmarks/run_benchmarks.py` with a `run_benchmarks()` "
            f"function at the repo root. ({exc})"
        )
        return 2

    _print("[cyan]running benchmarks...[/cyan]")
    results = run_benchmarks()
    if isinstance(results, dict):
        # Flatten a per-model dict of dicts into one table per model, else print.
        for name, vals in results.items():
            if isinstance(vals, dict):
                _metrics_table(f"benchmark — {name}", vals)
            else:
                _print(f"{name}: {vals}")
    if args.out:
        import json

        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        _print(f"[green]results written:[/green] {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="neuroforge",
        description="NeuroForge CFD — self-correcting geometry-native AI CFD.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_info = sub.add_parser("info", help="show version, models, and device")
    p_info.set_defaults(func=cmd_info)

    p_demo = sub.add_parser("demo", help="run the end-to-end demo + write a report")
    p_demo.add_argument("--report", default=None, help="report HTML path")
    p_demo.add_argument("--out", default=None, help="output directory for the report")
    p_demo.set_defaults(func=cmd_demo)

    p_train = sub.add_parser("train", help="train a model from a YAML config")
    p_train.add_argument("--config", required=True, help="path to config YAML")
    p_train.add_argument("--out", default=None, help="checkpoint output path")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="solve one airfoil case from a checkpoint")
    p_pred.add_argument("--ckpt", required=True, help="checkpoint path")
    p_pred.add_argument("--airfoil", default="naca2412", help="NACA code")
    p_pred.add_argument("--aoa", type=float, default=5.0, help="angle of attack (deg)")
    p_pred.add_argument("--re", type=float, default=3.0e6, help="Reynolds number")
    p_pred.add_argument("--u", type=float, default=30.0, help="freestream speed")
    p_pred.add_argument("--res", type=int, default=128, help="grid resolution")
    p_pred.add_argument("--report", default=None, help="optional report HTML path")
    p_pred.set_defaults(func=cmd_predict)

    p_bench = sub.add_parser("benchmark", help="run the model comparison benchmarks")
    p_bench.add_argument("--out", default=None, help="optional JSON results path")
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: "list[str] | None" = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        _print("[yellow]interrupted[/yellow]")
        return 130
    except Exception as exc:  # surface a clean error, non-zero exit
        _print(f"[red]error:[/red] {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
