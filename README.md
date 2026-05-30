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

This closed loop — which we call **Neural Residual Iteration** — is the organising
idea. A classical solver iterates until its residuals drop; NeuroForge applies a
learned, residual-conditioned correction operator with a backtracking acceptance
test, over *unseen* geometries.

> ### ⚠️ Research status & honest scope (read this)
> NeuroForge is an **early-stage, open research scaffold**, not a validated solver.
> Three things to be clear about up front:
> - **A low PDE residual does *not* prove correctness.** A smooth near-freestream
>   field can have *zero* residual yet be completely wrong. So the trust map is a
>   physics-residual **consistency monitor** (necessary, not sufficient) — *not*
>   "physics-verified confidence." Whether the residual actually tracks error is an
>   **empirical question** you can now measure directly
>   (`physics.evaluation.residual_error_correlation`); treat the loop as unproven
>   until that correlation is shown on real data.
> - **The mechanism is not new in isolation.** Learned solver-correction with
>   convergence guarantees (Hsieh et al., ICLR 2019), learned fixed points for
>   steady PDEs (FNO-DEQ, NeurIPS 2023), iterative refiners (PDE-Refiner), and
>   residual-corrector operators all predate it. NeuroForge's contribution is the
>   *integrated, open, reproducible engine* — predict → verify → UQ → correct →
>   fall back — not a new operator.
> - **Resolution limits matter.** A uniform Cartesian grid cannot resolve a
>   Re ≈ 10⁶ boundary layer (≈ sub-cell at 128²), so wall quantities (Cf, and
>   Cl/Cd from shear) are approximate; body-fitted / point-cloud backbones are on
>   the roadmap. No large-scale accuracy results are claimed yet.
>
> **Update — principled core.** The engine now ships a **contractive
> Deep-Equilibrium corrector** (`corrector_type='deq'`): the correction is the
> fixed point of a spectrally-normalised (Lipschitz < 1) operator, so it has a
> *real* Banach convergence guarantee (measured contraction ≈ 0.5, converges to
> 10⁻⁵ in ~15 iters), trained with Jacobian-Free Backprop — plus **split-conformal
> UQ calibration** giving the trust map a coverage guarantee. See
> [`docs/paper`](docs/paper/neuroforge_cfd.md) §3.1b–c.

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
| **synthetic** (bundled) | runs instantly, zero downloads — *potential flow + algebraic boundary layer*; a plumbing **smoke-test substrate only**, not a momentum-consistent RANS solution (do not use for accuracy claims) |
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

Or in Python with the clean **`NeuroForge` estimator API** (scikit-learn/Keras
style — hyper-parameters in the constructor, data in `fit`, everything else one line):

```python
import neuroforge as nf

model = nf.NeuroForge(backbone="fno", width=48, modes=20, corrector="deq", epochs=80)
model.fit("airfrans", task="scarce", n_train=200, n_val=100, cache_dir="data/cache")

print(model.evaluate())            # AirfRANS metrics: rho_Cd, rho_Cl, per-channel MSE
print(model.ablate_corrector())    # does the corrector improve accuracy (not just residual)?
model.calibrate(alpha=0.1)         # conformal-calibrated trust (90% coverage guarantee)

field  = model.predict(case)       # fast one-shot backbone field
result = model.solve(case)         # full self-correcting solve + diagnostics
model.save("model.pt")             # ... nf.NeuroForge.load("model.pt")
```

(The lower-level `Config` + `train_recipe` + `NeuroForgeEngine` are still there if
you want full control; `NeuroForge` is just the ergonomic front door.)

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
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — the 3-stage product plan and non-goals.
- [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) — the **multi-paper research
  roadmap**: aerodynamics (3-D, compressible, foundation) and an honest
  **PEM fuel-cell** multiphysics track, with datasets, methods, and venues.

## Project layout

```
src/neuroforge/
  core/        # frozen data contracts: FlowCase, FlowField, Diagnostics, Config
  geometry/    # NACA/STL → SDF, masks, network encoding
  data/        # synthetic potential-flow smoke generator + AirfRANS loader + datamodule
  models/      # FNO, Geo-FNO, physics-attention transformer, U-Net/DeepONet baselines,
               #   local correction net, deep-ensemble / MC-dropout UQ
  physics/     # differentiable residuals (continuity/momentum/BC), metrics, trust map
  solver/      # NeuroForgeEngine + Neural Residual Iteration + classical fallback
  train/       # physics-informed training loop & composite loss
  viz/         # field / residual / trust / Cp / convergence plots + HTML report
  cli.py       # `neuroforge` CLI ;  app/  Streamlit UI
docs/paper/    # research paper draft
```

## The thesis (scoped honestly)

> NeuroForge CFD explores an AI-first, self-correcting CFD workflow that aims to be
> a **ranking-preserving surrogate for early-design exploration** — combining
> geometry-aware neural operators, physics-residual *monitoring*, uncertainty
> estimation, and local adaptive correction. The goal is not to "replace CFD" but
> to predict fast and flag where to trust the prediction.

The design-facing success metric is therefore **rank correlation of Cl/Cd across
candidate geometries** (Spearman ρ — see `physics.evaluation`), the quantity early
design actually needs, *not* a claim of replacing the solver. Whether the
self-correction loop improves end accuracy is an open, measurable question
(corrector-on-vs-off ablation + residual↔error correlation).

## Citing / related work

The **closest prior art** — and the right baselines to beat — are *learned PDE
solvers with convergence guarantees* (Hsieh et al., ICLR 2019, arXiv:1906.01200),
*deep-equilibrium / learned fixed points for steady PDEs* (FNO-DEQ, NeurIPS 2023,
arXiv:2312.00234), *iterative refiners* (PDE-Refiner, arXiv:2308.05732), and
*residual-corrector operators* (arXiv:2306.12047). NeuroForge is positioned as the
*integrated, open engine* around these ideas, not a new operator. Backbones/data
build on FNO/Geo-FNO (arXiv:2010.08895 / 2207.05209), Transolver (ICML 2024,
arXiv:2402.02366), DoMINO (arXiv:2501.13350), AirfRANS (NeurIPS 2022,
arXiv:2212.07564), and conformal UQ for operators (UQNO, ICLR 2024,
arXiv:2402.01960). See [`docs/paper/neuroforge_cfd.md`](docs/paper/neuroforge_cfd.md).

## License

MIT © 2026 Kasra (Ali) Ghanavati
