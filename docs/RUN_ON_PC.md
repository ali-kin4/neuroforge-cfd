# Running NeuroForge on a desktop GPU (RTX 4070 Ti / i7-13700 / 64 GB)

This repo is CPU-first *so it always runs*, but it is fully GPU-aware. On a
strong consumer workstation the whole pre-registered research protocol
(`docs/EXPERIMENTS.md`) is an **overnight job**, not a multi-week one. This page
is the start-to-finish runbook for moving off a small server and onto your PC.

## TL;DR

```bash
git clone https://github.com/ali-kin4/neuroforge-cfd.git
cd neuroforge-cfd
python -m venv .venv && .venv\Scripts\activate     # Windows; use source .venv/bin/activate on Linux
pip install -e .

# 1. Check the GPU is seen and get a calibrated ETA (downloads/trains nothing):
python scripts/gpu_sanity.py

# 2. Validate the whole pipeline in ~2 min (synthetic, no download):
python scripts/run_full_research.py --preset smoke

# 3. The real run (downloads AirfRANS once, then trains overnight):
python scripts/run_full_research.py --preset full --cache-dir data/cache
```

> Installing with `pip install -e .` puts `neuroforge` on the path, so you do
> **not** need `PYTHONPATH=src` on the PC. (The `PYTHONPATH=src` form is for
> running from a bare checkout without installing.)

## 0. Install the CUDA build of PyTorch

The server here has the **CPU** wheel (`torch 2.6.0+cpu`) — that will not use
your GPU. On the PC, install a CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA GeForce RTX 4070 Ti
```

`scripts/gpu_sanity.py` prints this for you and warns loudly if CUDA is missing.

## 1. Sanity + ETA (`scripts/gpu_sanity.py`)

Prints an environment manifest, micro-benchmarks one real backbone train step on
your hardware (AMP on), and **extrapolates that measured ms/step to the full
protocol** — so you get a calibrated wall-clock estimate before committing a
machine for the night.

```bash
python scripts/gpu_sanity.py --batch-size 16
```

Re-run it with whatever `--batch-size` / model size you plan to use; the ETA is
calibrated to *your* measured step time, not a guess. (For reference, the CPU
server here measured ~4.5 s/step — the 4070 Ti should be ~30-80 ms/step, i.e.
roughly 50-150x faster.)

## 2. Validate the pipeline (`--preset smoke`)

```bash
python scripts/run_full_research.py --preset smoke
```

Synthetic data, tiny grid, one seed — runs in a couple of minutes on anything
(no download). If this writes `results/full_research/in_dist/ablation.md`, the
whole training/eval/reporting path is healthy and you can trust the real run.

## 3. The full research run (`--preset full`)

```bash
python scripts/run_full_research.py --preset full --cache-dir data/cache
```

This reproduces **100% of the pre-registered protocol** and writes the paper's
tables:

| Stage | What | Output |
|---|---|---|
| 1 | In-distribution ablation — 4 arms x 3 seeds on AirfRANS `full` (800/200) | `results/full_research/in_dist/ablation.{md,csv}` (Table 1) |
| 2 | Out-of-distribution ablation — `reynolds` + `aoa` splits | `results/full_research/ood/ablation_ood.{md,csv}` (Table 3) |

It also writes `MANIFEST.json` (GPU, torch/CUDA versions, git commit, exact
config — for reproducibility) and `SUMMARY.md` (per-stage wall-clock timings).

### Resume / unattended runs

- **Stage-level resume is automatic.** Re-running skips any stage whose output
  CSV already exists, so a crashed or interrupted job continues where it left
  off. Pass `--force` to recompute from scratch.
- The AirfRANS download + rasterisation (the 1000-case dataset, **~9 GB unzipped**)
  is **cached** under `data/` and paid **once on this machine** — across all seeds
  and stages, and across later runs. It is gitignored, so a fresh clone does not
  carry it; the first run fetches it, every run after reuses the local cache.
- On CUDA out-of-memory the training recipe **auto-halves the batch size** down
  to 1 rather than crashing — so an over-ambitious `--batch-size` degrades
  gracefully. (Start at 16 on 12 GB; drop to 8 if you prefer headroom.)

### Presets and overrides

| Preset | Source | n_train / n_val | epochs | seeds | Use |
|---|---|---|---|---|---|
| `full` | AirfRANS | 800 / 200 | 80 | 0 1 2 | the deciding, error-barred run |
| `lite` | AirfRANS | 400 / 120 | 80 | 0 1 2 | ~half the compute, same shape |
| `smoke` | synthetic | 24 / 8 | 4 | 0 | validate the pipeline (CPU OK) |

Any preset value is overridable, e.g. tighter error bars or skip the OOD stage:

```bash
python scripts/run_full_research.py --preset full --seeds 0 1 2 3 4
python scripts/run_full_research.py --preset full --no-ood          # Stage 1 only
python scripts/run_full_research.py --preset full --batch-size 8    # more VRAM headroom
```

## 4. (Optional) one headline checkpoint

If you also want a single high-quality trained model (bigger FNO, 150 epochs)
for the demo/Streamlit app and qualitative figures — separate from the ablation
arms — use the GPU config:

```bash
neuroforge train --config configs/gpu_full.yaml
```

## 5. Push results back to the repo

Results are version-controlled now — there is **no Google Drive**. The
`run_full_research.py` / notebook outputs land in the repo's tracked `results/`
dir (tables, CSVs, `MANIFEST.json`, `SUMMARY.md`, figures), while the large,
regenerable artifacts (the AirfRANS download + rasterised cache under `data/`,
and trained checkpoints under `checkpoints/`) stay **gitignored**.

So the loop is: run locally → results appear in `results/` → push them back:

```bash
python scripts/push_results.py            # stages results/, commits, pushes
python scripts/push_results.py -m "full run, seeds 0-2, RTX 4070 Ti"
python scripts/push_results.py --no-push  # commit locally only
# or by hand:
git add results/ && git commit -m "results: full run" && git push
```

It only commits when `results/` actually changed, and never tries to push the
multi-GB cache or checkpoints (those are ignored). See `results/README.md` for
exactly what is and isn't tracked.

## What does *not* need changing

Nothing structural. The trainer already auto-selects CUDA (`train.device:auto`),
supports AMP (bf16 on your Ada GPU), clears the CUDA cache between stages, and
recovers from OOM. The frozen contracts (`core/types.py`, `core/config.py`,
`models/base.py`, `CONVENTIONS.md`) are untouched. Moving to the PC is a
configuration + orchestration change, not a rewrite.
