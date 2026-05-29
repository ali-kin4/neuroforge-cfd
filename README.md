<div align="center">

# 🌀 NeuroForge CFD

**A self-correcting, geometry-native AI CFD engine.**

*Predict flow fields directly from geometry + boundary conditions — then verify the
physics, estimate uncertainty, and iteratively self-correct, calling classical CFD
only where it's actually needed.*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ali-kin4/neuroforge-cfd/blob/main/notebooks/NeuroForge_CFD_Colab.ipynb)

</div>

---

## Why NeuroForge is different

Most AI-for-CFD systems are **one-shot surrogates**: geometry goes in, a flow
field comes out, and you have no idea whether to trust it. NeuroForge treats AI
prediction like a *solver*, not a *guess*:

> **Predict → check physics residuals → estimate uncertainty → correct the
> unreliable regions → repeat.** Fall back to classical CFD *only* where the
> model says it can't be trusted.

This closed loop — which we call **Neural Residual Iteration** — is the core
contribution. A classical solver iterates until its residuals drop; NeuroForge
does the same, but with a learned, residual-conditioned correction operator
instead of a linear solve, over *unseen* geometries.

```
   CAD / STL / airfoil              ┌─────────────────────────────────────────┐
   + boundary conditions  ───▶      │  geometry-native encoding (SDF + masks)  │
                                     └────────────────────┬────────────────────┘
                                                          ▼
                          ┌───────────────────────────────────────────────┐
                          │   neural operator backbone (FNO / Transolver)  │  ◀─┐
                          └───────────────────────┬───────────────────────┘    │
                                                  ▼                             │
                          ┌───────────────────────────────────────────────┐    │
                          │  physics check:  ∇·u,  momentum, BC violation  │    │  Neural
                          │  uncertainty (ensemble / MC-dropout)           │    │ Residual
                          │  ➜  TRUST MAP  (🟢 reliable │🟡│ 🔴 fix)        │    │Iteration
                          └───────────────────────┬───────────────────────┘    │
                                                  ▼                             │
                          ┌───────────────────────────────────────────────┐    │
                          │  local correction net  (trust-gated Δ-field)  │ ───┘
                          └───────────────────────┬───────────────────────┘
                                                  ▼   (only if still red)
                          ┌───────────────────────────────────────────────┐
                          │  optional classical CFD patch (OpenFOAM/SU2)   │
                          └───────────────────────┬───────────────────────┘
                                                  ▼
        flow field · pressure · wall shear · Cl/Cd · uncertainty & residual maps
```

## Outputs

pressure & velocity fields · wall shear stress · lift/drag coefficients · flow
separation indicators · **uncertainty heatmap** · **physics-residual map** ·
**trust map** · per-iteration convergence history.

## Install

```bash
git clone <this-repo> && cd CFD
pip install -e .            # core (numpy, scipy, torch, matplotlib)
pip install -e ".[all]"     # + airfrans data, pyvista, streamlit app, dev tools
```

## 60-second quickstart (no GPU, no downloads)

```python
import neuroforge as nf

# A NACA 2412 at 5° AoA, Re = 3e6.
case = nf.FlowCase.from_airfoil("naca2412", aoa=5, reynolds=3e6, u_inf=30.0, resolution=128)

# A ready-to-go engine: loads the bundled demo checkpoint (instant);
# trains a tiny model on synthetic data only if no checkpoint is found.
engine = nf.NeuroForgeEngine.pretrained()

result = engine.solve(case)                 # predict → check → self-correct
print(result.summary())                     # {'cl':..., 'cd':..., 'residual_norm':..., ...}
result.save_report("report.html")           # field + residual + trust + Cp + convergence
```

Or from the command line:

```bash
neuroforge demo                              # end-to-end, writes a report
neuroforge predict --airfoil naca0012 --aoa 8 --re 1e6 --report out.html
neuroforge benchmark                         # FNO vs U-Net vs DeepONet vs Transolver-lite
neuroforge info
```

## First application: external aerodynamics of 2-D airfoils

We start narrow on purpose. Benchmarks/datasets:

| Dataset | Use |
|---|---|
| **synthetic** (bundled) | runs instantly, zero downloads — potential-flow + viscous-wake pseudo-RANS |
| [**AirfRANS**](https://airfrans.readthedocs.io) | 1000 incompressible RANS sims, NACA 4/5-digit, Re 2–6M, AoA −5°→15° |

Then: arbitrary 2-D bluff bodies → 3-D vehicle-like geometries (AhmedML,
DrivAerML) in later stages — see [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Train on real data (AirfRANS) — locally or on a GPU

The bundled demo model is tiny (trained on synthetic data); real accuracy comes
from training on AirfRANS. One command (GPU auto-detected, rasterised data cached):

```bash
pip install -e ".[data]"
python scripts/train_airfrans.py --download --task scarce \
    --n-train 200 --n-val 100 --epochs 80 --model fno --width 48 --modes 20 \
    --cache-dir data/cache --out checkpoints/airfrans.pt
# scale up:  --task full --n-train 800 --n-val 200 --width 64 --modes 24 --layers 5 --epochs 150
```

Or in Python via the reusable recipe (used by the CLI and the notebook too):

```python
from neuroforge.core.config import Config, DataConfig, ModelConfig
from neuroforge.train import train_recipe

cfg = Config()
cfg.data  = DataConfig(source="airfrans", task="scarce", resolution=128,
                       n_train=200, n_val=100, cache_dir="data/cache")
cfg.model = ModelConfig(name="fno", width=48, modes=20, n_layers=4)
cfg.train.epochs, cfg.train.device = 80, "auto"      # "auto" -> CUDA if available
result = train_recipe(cfg, download=True, corrector_epochs=20, out="checkpoints/airfrans.pt")
print(result["val_errors"])                          # rel-L2 for u/v/p/speed
```

### ▶️ Colab Pro (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ali-kin4/neuroforge-cfd/blob/main/notebooks/NeuroForge_CFD_Colab.ipynb)

[`notebooks/NeuroForge_CFD_Colab.ipynb`](notebooks/NeuroForge_CFD_Colab.ipynb) is a
ready-to-run notebook: it installs the package, (optionally) mounts Drive for
caching, downloads AirfRANS, trains FNO + corrector on the GPU, evaluates field
errors and Cl/Cd, runs the self-correcting solver, and plots prediction vs CFD.
Click the badge, set a **GPU** runtime (*Runtime → Change runtime type*), and run
top to bottom — the clone URL is already wired to this repo.

## Docs

- [`docs/paper/neuroforge_cfd.md`](docs/paper/neuroforge_cfd.md) — research-paper
  draft (method, related work, Neural Residual Iteration, experimental protocol).
- [`docs/architecture.md`](docs/architecture.md) — engineer-facing architecture:
  module map, data contracts, `solve()` control flow, extension points.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — the 3-stage plan and explicit non-goals.

## Project layout

```
src/neuroforge/
  core/        # frozen data contracts: FlowCase, FlowField, Diagnostics, Config
  geometry/    # NACA/STL → SDF, masks, network encoding
  data/        # synthetic pseudo-RANS generator + AirfRANS loader + datamodule
  models/      # FNO, Geo-FNO, physics-attention transformer, U-Net/DeepONet baselines,
               #   local correction net, deep-ensemble / MC-dropout UQ
  physics/     # differentiable residuals (continuity/momentum/BC), metrics, trust map
  solver/      # NeuroForgeEngine + Neural Residual Iteration + classical fallback
  train/       # physics-informed training loop & composite loss
  viz/         # field / residual / trust / Cp / convergence plots + HTML report
  cli.py       # `neuroforge` CLI ;  app/  Streamlit UI
docs/paper/    # research paper draft
```

## The thesis

> NeuroForge CFD introduces an AI-first, self-correcting CFD workflow that replaces
> full-domain iterative simulation during early design by combining geometry-aware
> neural operators, physics-residual validation, uncertainty estimation, and local
> adaptive correction.

**AI-first CFD with physics-verified confidence** — not "we replaced CFD."

## Citing / related work

Builds on ideas from DoMINO (NVIDIA, arXiv:2501.13350), Transolver / Transolver++
(ICML 2024 / arXiv:2502.02414), Geo-FNO (arXiv:2207.05209), AirfRANS (NeurIPS 2022),
residual-based error correctors (arXiv:2306.12047), and calibration-aware UQ for
neural PDE surrogates. See [`docs/paper/neuroforge_cfd.md`](docs/paper/neuroforge_cfd.md).

## License

MIT © 2026 Kasra (Ali) Ghanavati
