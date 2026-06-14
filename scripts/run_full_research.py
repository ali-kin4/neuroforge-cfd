"""One command to reproduce 100% of the pre-registered NeuroForge research.

Runs the full protocol from `docs/EXPERIMENTS.md` end to end and writes every
results table the paper needs:

  Stage 1  in-distribution ablation  (AirfRANS `full`, 4 arms x N seeds)
           -> <out>/in_dist/ablation.md + ablation.csv   (Table 1)
  Stage 2  out-of-distribution ablation (`reynolds` + `aoa` splits)
           -> <out>/ood/ablation_ood.md + ablation_ood.csv   (Table 3)

It is a thin orchestrator over `benchmarks/ablation.py` (the committed harness),
adding the things you want for a long unattended run on a workstation:

  * an environment MANIFEST.json (GPU / torch / CUDA / git commit / args) so the
    numbers are traceable,
  * per-stage wall-clock timing in SUMMARY.md,
  * **stage-level resume** — re-running skips any stage whose output already
    exists, so a crashed or interrupted job continues where it stopped
    (use --force to recompute).

Presets
-------
  --preset full   AirfRANS, 800/200, 80 epochs, seeds 0 1 2     (the deciding run)
  --preset lite   AirfRANS, 400/120, 80 epochs, seeds 0 1 2     (~half the cost)
  --preset smoke  synthetic, 24/8, 4 epochs, 1 seed, res 32     (validate the
                  pipeline on any machine in a couple of minutes — CPU is fine)

Examples
--------
    # validate the whole pipeline first (fast, no download):
    PYTHONPATH=src python scripts/run_full_research.py --preset smoke

    # the real overnight run on the desktop GPU:
    PYTHONPATH=src python scripts/run_full_research.py --preset full \
        --cache-dir data/cache --out-dir results/full_research

Any preset value can be overridden with an explicit flag (e.g. add
``--seeds 0 1 2 3 4`` for tighter error bars, or ``--no-ood`` to skip Stage 2).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import neuroforge  # noqa: F401  (caps math-library threads on import; before torch)

# `benchmarks/` lives at the repo root (a sibling of this `scripts/` dir); add the
# root to sys.path so `from benchmarks.ablation import ...` works regardless of CWD.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_PRESETS = {
    "full": dict(
        source="airfrans", task="full", n_train=800, n_val=200,
        epochs=80, corrector_epochs=20, seeds=[0, 1, 2], resolution=128,
        width=48, modes=20, layers=4, batch_size=16,
    ),
    "lite": dict(
        source="airfrans", task="full", n_train=400, n_val=120,
        epochs=80, corrector_epochs=20, seeds=[0, 1, 2], resolution=128,
        width=48, modes=20, layers=4, batch_size=16,
    ),
    "smoke": dict(
        source="synthetic", task="scarce", n_train=24, n_val=8,
        epochs=4, corrector_epochs=2, seeds=[0], resolution=32,
        width=16, modes=10, layers=3, batch_size=4,
    ),
}


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def _manifest(args, out_dir: str) -> dict:
    import torch

    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else None
    vram = (torch.cuda.get_device_properties(0).total_memory / 1e9) if cuda else None
    m = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_toolkit": torch.version.cuda if cuda else None,
        "gpu": gpu,
        "vram_gb": round(vram, 1) if vram else None,
        "cpu_count": os.cpu_count(),
        "config": {
            "source": args.source, "task": args.task,
            "n_train": args.n_train, "n_val": args.n_val,
            "epochs": args.epochs, "corrector_epochs": args.corrector_epochs,
            "seeds": args.seeds, "resolution": args.resolution,
            "width": args.width, "modes": args.modes, "layers": args.layers,
            "batch_size": args.batch_size, "device": args.device,
            "ood": not args.no_ood, "ood_tasks": args.ood_tasks,
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    return m


def _done(path: str, force: bool) -> bool:
    """A stage is already complete if its CSV exists (and we are not forcing)."""
    return (not force) and os.path.exists(path)


def _fmt(seconds: float) -> str:
    h = seconds / 3600.0
    return f"{h:.2f} h" if h >= 1 else f"{seconds / 60.0:.1f} min"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reproduce the full NeuroForge protocol.")
    p.add_argument("--preset", choices=list(_PRESETS), default="full")
    # All of these default to the preset's value (resolved after parse).
    p.add_argument("--source", choices=["airfrans", "synthetic"])
    p.add_argument("--task", default=None)
    p.add_argument("--n-train", type=int)
    p.add_argument("--n-val", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--corrector-epochs", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--resolution", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--modes", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--cache-dir", default="data/cache", help="cache rasterised AirfRANS pairs")
    p.add_argument("--out-dir", default="results/full_research")
    p.add_argument("--ood-tasks", nargs="+", default=["reynolds", "aoa"])
    p.add_argument("--no-ood", action="store_true", help="skip Stage 2 (OOD)")
    p.add_argument("--force", action="store_true", help="recompute completed stages")
    args = p.parse_args(argv)

    # Resolve unspecified knobs from the chosen preset.
    pre = _PRESETS[args.preset]
    for k, v in pre.items():
        flag = k if k != "layers" else "layers"
        if getattr(args, flag, None) in (None, []):
            setattr(args, flag, v)

    from benchmarks.ablation import run_ablation, run_ood_ablation

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = _manifest(args, args.out_dir)
    download = args.source == "airfrans"

    print("=" * 64)
    print(f"NeuroForge — full research run   (preset: {args.preset})")
    print("=" * 64)
    dev = "CUDA: " + manifest["gpu"] if manifest["cuda_available"] else "CPU only"
    print(f"  device   : {dev}")
    print(f"  source   : {args.source} / {args.task}   res={args.resolution}")
    print(f"  data     : {args.n_train} train / {args.n_val} val   seeds={args.seeds}")
    print(f"  model    : fno width={args.width} modes={args.modes} layers={args.layers} "
          f"batch={args.batch_size}")
    print(f"  epochs   : {args.epochs} backbone + {args.corrector_epochs} corrector")
    print(f"  out-dir  : {args.out_dir}")
    print(f"  OOD      : {'no' if args.no_ood else ', '.join(args.ood_tasks)}")
    print()

    common = dict(
        n_train=args.n_train, n_val=args.n_val, resolution=args.resolution,
        seeds=tuple(args.seeds), epochs=args.epochs,
        corrector_epochs=args.corrector_epochs, width=args.width,
        modes=args.modes, n_layers=args.layers, batch_size=args.batch_size,
        cache_dir=args.cache_dir if args.source == "airfrans" else None,
        download=download, device=args.device,
    )

    timings: dict[str, float] = {}

    # ----- Stage 1: in-distribution ablation (Table 1) -------------------- #
    in_dir = os.path.join(args.out_dir, "in_dist")
    in_csv = os.path.join(in_dir, "ablation.csv")
    if _done(in_csv, args.force):
        print(f"[stage 1] in-distribution ablation already done -> {in_csv} (skip)")
    else:
        print("[stage 1] in-distribution ablation ...")
        t0 = time.time()
        run_ablation(args.source, task=args.task, out_dir=in_dir, **common)
        timings["in_dist"] = time.time() - t0
        print(f"[stage 1] done in {_fmt(timings['in_dist'])}")

    # ----- Stage 2: out-of-distribution ablation (Table 3) ---------------- #
    if not args.no_ood:
        ood_dir = os.path.join(args.out_dir, "ood")
        ood_csv = os.path.join(ood_dir, "ablation_ood.csv")
        if _done(ood_csv, args.force):
            print(f"[stage 2] OOD ablation already done -> {ood_csv} (skip)")
        elif args.source != "airfrans":
            print("[stage 2] OOD ablation needs source=airfrans - skipping for "
                  f"source={args.source}.")
        else:
            print("[stage 2] out-of-distribution ablation ...")
            t0 = time.time()
            run_ood_ablation(
                args.source, ood_tasks=tuple(args.ood_tasks),
                out_dir=ood_dir, **common,
            )
            timings["ood"] = time.time() - t0
            print(f"[stage 2] done in {_fmt(timings['ood'])}")

    _write_summary(args, manifest, timings)
    print("\n[done] all stages complete. See "
          f"{os.path.join(args.out_dir, 'SUMMARY.md')}")
    return 0


def _write_summary(args, manifest: dict, timings: dict) -> None:
    total = sum(timings.values())
    lines = [
        "# NeuroForge — full research run summary",
        "",
        f"- generated: {manifest['timestamp_utc']}",
        f"- git commit: `{manifest['git_commit']}`",
        f"- device: {manifest['gpu'] or 'CPU'}"
        + (f" ({manifest['vram_gb']} GB)" if manifest['vram_gb'] else ""),
        f"- torch {manifest['torch']}, CUDA {manifest['cuda_toolkit']}",
        f"- preset: `{args.preset}`  ({args.source}/{args.task}, "
        f"{args.n_train}/{args.n_val}, seeds {args.seeds})",
        "",
        "## Stage timings (this invocation; skipped stages omitted)",
        "",
        "| stage | wall-clock |",
        "|---|---|",
    ]
    label = {"in_dist": "in-distribution ablation (Table 1)",
             "ood": "out-of-distribution ablation (Table 3)"}
    for k, v in timings.items():
        lines.append(f"| {label.get(k, k)} | {_fmt(v)} |")
    if timings:
        lines.append(f"| **total (this run)** | **{_fmt(total)}** |")
    else:
        lines.append("| _(all stages already complete — nothing re-run)_ | — |")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `in_dist/ablation.md` / `.csv` — Table 1 (arms x metrics, mean±std)",
        "- `ood/ablation_ood.md` / `.csv` — Table 3 (in-dist -> OOD gap)",
        "- `MANIFEST.json` — environment + exact config for reproducibility",
        "",
        "Re-running this script skips completed stages; pass `--force` to recompute.",
    ]
    with open(os.path.join(args.out_dir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
