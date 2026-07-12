"""GPU sanity check + wall-clock estimator for the full research run.

Run this FIRST on a new machine (e.g. the desktop with the RTX 4070 Ti) before
committing to an overnight job. It:

1. prints an environment manifest (torch / CUDA / GPU / VRAM / CPU / git commit),
2. micro-benchmarks one real forward+backward+optim step of the full-scale
   backbone at the target batch size (AMP on CUDA), reporting ms/step, and
3. extrapolates that measured step time to the *whole* pre-registered protocol
   (`scripts/run_full_research.py`: in-distribution + 2 OOD splits x N seeds x
   4 arms), so you get a calibrated ETA instead of a guess.

It downloads nothing and trains nothing for real — the micro-benchmark runs on a
random in-memory batch, so it finishes in seconds on either CPU or GPU.

    PYTHONPATH=src python scripts/gpu_sanity.py
    PYTHONPATH=src python scripts/gpu_sanity.py --batch-size 16 --seeds 3

The ETA is a *training-compute* estimate (the dominant cost). One-time AirfRANS
download + rasterisation of the 1000-case dataset is reported separately as a
fixed overhead, since it is cached and paid only once.
"""

from __future__ import annotations

import argparse
import math
import platform
import subprocess
import time

import neuroforge  # noqa: F401  (caps math-library threads on import; before torch)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def print_environment() -> object:
    """Print and return the runtime/hardware manifest."""
    import torch

    cuda = torch.cuda.is_available()
    print("=" * 64)
    print("ENVIRONMENT")
    print("=" * 64)
    print(f"  platform      : {platform.platform()}")
    print(f"  python        : {platform.python_version()}")
    print(f"  torch         : {torch.__version__}")
    print(f"  git commit    : {_git_commit()}")
    print(f"  cpu count     : {os_cpu_count()}")
    print(f"  CUDA available: {cuda}")
    if cuda:
        dev = torch.cuda.get_device_properties(0)
        print(f"  GPU           : {dev.name}")
        print(f"  VRAM          : {dev.total_memory / 1e9:.1f} GB")
        print(f"  capability    : sm_{dev.major}{dev.minor}")
        print(f"  CUDA toolkit  : {torch.version.cuda}")
    else:
        print("  GPU           : (none detected - will run on CPU)")
        print("  NOTE          : the full protocol is impractical on CPU; this")
        print("                  machine is best used only for smoke-testing.")
    return torch.device("cuda" if cuda else "cpu")


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def benchmark_step(
    device,
    *,
    resolution: int,
    batch_size: int,
    width: int,
    modes: int,
    n_layers: int,
    amp: bool,
    n_warmup: int = 3,
    n_iters: int = 12,
) -> float:
    """Time one backbone train step (fwd+bwd+opt) on a random batch; ms/step."""
    import torch

    from neuroforge.core.types import N_IN, N_OUT
    from neuroforge.models.base import build_model

    model = build_model(
        "fno", in_channels=N_IN, out_channels=N_OUT,
        width=width, modes=modes, n_layers=n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(batch_size, N_IN, resolution, resolution, device=device)
    y = torch.randn(batch_size, N_OUT, resolution, resolution, device=device)

    use_amp = amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def one_step() -> None:
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = (model(x) - y).pow(2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            opt.step()

    for _ in range(n_warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iters):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    params = model.num_params()
    print("=" * 64)
    print("MICRO-BENCHMARK  (full-scale FNO backbone, random batch)")
    print("=" * 64)
    print(f"  config        : width={width} modes={modes} layers={n_layers} "
          f"res={resolution} batch={batch_size}")
    print(f"  params        : {params:,}")
    print(f"  AMP           : {'on (' + str(amp_dtype).split('.')[-1] + ')' if use_amp else 'off'}")
    ms = 1000.0 * elapsed / n_iters
    print(f"  measured      : {ms:.1f} ms / train step  ({n_iters} iters)")
    return ms


def estimate_eta(
    ms_per_step: float,
    *,
    n_train: int,
    batch_size: int,
    epochs: int,
    corrector_epochs: int,
    seeds: int,
    include_ood: bool,
) -> None:
    """Extrapolate measured step time to the whole research protocol."""
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))

    # Per seed, run_ablation trains 3 backbones (deq-arm, local-arm, no-physics
    # arm) plus 2 correctors (deq unroll=1, local unroll=3). The corrector net is
    # smaller/cheaper than the backbone, so counting its unrolled steps at the
    # backbone step-cost is a deliberate slight over-estimate (head-room).
    backbone_steps = 3 * epochs * steps_per_epoch
    corrector_steps = corrector_epochs * steps_per_epoch * (1 + 3)
    steps_per_seed = backbone_steps + corrector_steps

    n_tasks = 1 + (2 if include_ood else 0)  # in-dist + {reynolds, aoa}
    total_steps = n_tasks * seeds * steps_per_seed

    train_s = total_steps * ms_per_step / 1000.0
    # Field evaluation + per-arm overhead is small but non-zero; pad 20%.
    lo = train_s * 1.15
    hi = train_s * 1.45

    print("=" * 64)
    print("ETA  - full pre-registered protocol")
    print("=" * 64)
    print(f"  splits        : {'in-dist + reynolds + aoa' if include_ood else 'in-dist only'} "
          f"({n_tasks} task-run(s))")
    print(f"  seeds         : {seeds}")
    print(f"  per fit       : {epochs} backbone + {corrector_epochs} corrector epochs")
    print(f"  steps/epoch   : {steps_per_epoch}  (n_train={n_train} / batch={batch_size})")
    print(f"  total steps   : ~{total_steps:,}")
    print(f"  estimated wall: {_fmt(lo)} - {_fmt(hi)}  (training compute)")
    print()
    print("  one-time, cached on THIS machine (paid once, not per seed):")
    print("    AirfRANS download+unzip (~9 GB, multi-GB) + rasterise 1000 cases")
    print("    -> ~20-60 min the first run; cached under data/ and skipped after")
    print()
    print("  NOTE: calibrated to THIS machine's measured ms/step. Re-run after")
    print("        changing --batch-size / model size to re-estimate.")


def _fmt(seconds: float) -> str:
    h = seconds / 3600.0
    if h >= 1:
        return f"{h:.1f} h"
    return f"{seconds / 60.0:.0f} min"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU sanity check + full-run ETA.")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--width", type=int, default=48, help="ablation-arm backbone width")
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    # protocol shape (must match run_full_research.py defaults)
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--corrector-epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--no-ood", action="store_true", help="estimate in-distribution only")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = print_environment()
    ms = benchmark_step(
        device,
        resolution=args.resolution, batch_size=args.batch_size,
        width=args.width, modes=args.modes, n_layers=args.layers,
        amp=not args.no_amp,
    )
    estimate_eta(
        ms,
        n_train=args.n_train, batch_size=args.batch_size,
        epochs=args.epochs, corrector_epochs=args.corrector_epochs,
        seeds=args.seeds, include_ood=not args.no_ood,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
