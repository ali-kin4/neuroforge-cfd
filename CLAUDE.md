# CLAUDE.md — NeuroForge CFD

Guidance for working in this repository.

## What this is

`neuroforge-cfd` — a self-correcting, geometry-native AI CFD engine. Given an
airfoil/body geometry + boundary conditions it predicts the flow field with a
neural operator, **checks its own physics residuals**, estimates uncertainty,
and runs **Neural Residual Iteration** (a learned, residual-driven correction
loop with a backtracking acceptance test that guarantees the residual norm is
monotone non-increasing), optionally falling back to local classical CFD only
where uncertainty is high. First application: 2-D external aerodynamics.

The package is **CPU-first** and runs end-to-end with zero downloads via a
synthetic pseudo-RANS data generator; it also loads the real AirfRANS dataset.

## Layout

```
src/neuroforge/
  core/      frozen contracts: types.py (FlowCase/FlowField/Diagnostics/SolveResult,
             the fixed 7-in/4-out channel spec), config.py, models/base.py (ABCs+registry)
  geometry/  NACA gen, SDF/masks, network encoding
  data/      synthetic generator, AirfRANS loader, rasterise, datamodule (Normalizer/loaders)
  models/    fno, geo_fno, transformer (Transolver-style), unet, deeponet, correction, ensemble (UQ)
  physics/   operators, residuals (+ differentiable torch residuals), metrics, trust map
  solver/    engine (Predictor, NeuroForgeEngine, demo), correction_loop, fallback
  train/     trainer, losses (data+physics+BC), schedule
  viz/       plots, report (HTML)
  cli.py     `neuroforge` CLI ;  app/ Streamlit UI ;  assets/demo.pt bundled checkpoint
tests/  benchmarks/  docs/paper/  examples/  scripts/
```

`CONVENTIONS.md` is the cross-module interface contract — read it before editing.
`core/types.py`, `core/config.py`, `models/base.py`, `CONVENTIONS.md` are frozen.

## ⚠️ Performance gotcha (important)

The prebuilt numpy/torch wheels ship OpenBLAS/MKL with `DYNAMIC_ARCH,
MAX_THREADS=24`. On low-core machines this **oversubscribes catastrophically** —
a 200×200 solve took 24 s and a 32×32 FNO FFT took 0.15 s (→13 s/training step).
`neuroforge/__init__.py` caps `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/
`MKL_NUM_THREADS` to 1 (via `setdefault`, before numpy/torch import) which fixes
it (solve 0.002 s, FFT 0.003 s, step 0.35 s). **Always `import neuroforge`
before heavy numpy/torch work.** Power users override with e.g.
`OMP_NUM_THREADS=8` before import.

## Commands

```bash
PYTHONPATH=src python -m pytest -q          # 54 fast tests (~30s); slow tests need -m slow
PYTHONPATH=src python -m neuroforge.cli demo
PYTHONPATH=src python benchmarks/run_benchmarks.py
PYTHONPATH=src python scripts/build_demo_checkpoint.py   # rebuild bundled assets/demo.pt
pip install -e .                            # then `neuroforge demo`
```

Tests/CI: keep grids small (res 24–32, n_train ≤ 4); `SyntheticRANS.solve` is
~0.9 s/case. Don't call `pretrained()`/`demo()` in fast tests (they train ~40s
unless `assets/demo.pt` exists) — use the `trained_engine` fixture.

## Conventions

Fields are `(ny,nx)` float32; network tensors `(B,C,H,W)`, channel-first.
Input channels (7): `sdf, mask, x, y, u_in, v_in, log_re`. Output (4): `u, v, p, nut`.
Pressure is kinematic (p/ρ); residuals run on physical (denormalised) fields with
`nu_eff = nu + nut`. Models are device-agnostic; register via `@register_model`.
