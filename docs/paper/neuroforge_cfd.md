# NeuroForge CFD: A Self-Correcting Geometry-Native Neural Solver for Fast Flow Prediction over Unseen Engineering Geometries

**Kasra (Ali) Ghanavati**

*Preprint draft — v0.1 (2026). Corresponding author: kasraghanavati@icloud.com*

---

## Abstract

Machine-learning surrogates for computational fluid dynamics (CFD) now predict
steady flow fields over engineering geometries orders of magnitude faster than
classical solvers. Yet the dominant paradigm is a **one-shot surrogate**:
geometry and boundary conditions go in, a flow field comes out, and the user is
given no principled way to know whether to trust it — especially on geometries
outside the training distribution, exactly where early-stage design exploration
lives. We present **NeuroForge CFD**, an open-source Python package that reframes
neural CFD prediction as a *solver loop* rather than a *single guess*. NeuroForge
predicts a flow field with a geometry-native neural operator, **verifies** it
against the discretised steady incompressible RANS residuals, **estimates** its
own predictive uncertainty, and then runs **Neural Residual Iteration**: a
learned, residual-conditioned correction operator that is applied under a
backtracking acceptance test guaranteeing the physics-residual norm is monotone
non-increasing across accepted iterations — the same convergence discipline a
classical solver enforces, but with a learned update over unseen geometries.
A trust map fuses residual and uncertainty into a traffic-light reliability
field that gates an (interface-level) classical-CFD fallback so expensive solves
are invoked only where the model admits it cannot be trusted. The novel
contribution is not any single block but the **closed self-correction loop plus
uncertainty-gated fallback as an integrated, reproducible engine**. We describe
the implemented method and package, a zero-download synthetic Hess–Smith
source-panel pseudo-RANS data generator that makes the whole pipeline
reproducible without any external dataset, and an AirfRANS loader. We lay out the
full experimental protocol on the AirfRANS benchmark and report bundled
synthetic *smoke* results from the benchmark harness; these use deliberately tiny
CPU-sized models and are illustrative of the scaffold, not state-of-the-art
claims. Large-scale training and validation are explicitly future work.

**Keywords:** neural operators, surrogate CFD, physics-informed learning,
uncertainty quantification, self-correction, airfoil aerodynamics.

---

## 1. Introduction

Classical CFD resolves the governing PDEs by iterating a discretised system until
its residuals fall below a tolerance. It is accurate and trusted, but slow:
meshing and a converged steady RANS solve over a new geometry can take from
minutes to hours, which throttles the rapid design-space exploration that early
engineering most needs. Neural surrogates promise the opposite trade — millisecond
inference — and a vigorous literature now maps geometry plus boundary conditions
directly to flow fields with neural operators and attention models.

The problem is **trust**. A trained surrogate is a one-shot function: it emits a
field with no built-in self-assessment, no convergence signal, and no recovery
path when it extrapolates. On a geometry far from the training set — the common
case in design — the user cannot distinguish a faithful prediction from a
confident hallucination. This is the product and research gap NeuroForge targets.

> **Research question.** *Can an AI-first CFD workflow predict flow fields over
> unseen engineering geometries, verify those predictions against the governing
> physics, quantify its own uncertainty, and iteratively self-correct the
> unreliable regions — invoking a classical solver only where it is genuinely
> needed — so that the result is fast yet trustworthy?*

> **Thesis (one sentence).** *NeuroForge CFD introduces an AI-first,
> self-correcting CFD workflow that replaces full-domain iterative simulation
> during early design by combining geometry-aware neural operators,
> physics-residual validation, uncertainty estimation, and local adaptive
> correction.*

The framing is deliberately **AI-first CFD with physics-verified confidence**,
not "we replaced CFD." The classical solver remains the ground truth; NeuroForge
amortises the common case and falls back to the classical solver precisely where
its own diagnostics say it must.

Our contributions are:

1. **Neural Residual Iteration** — a learned, residual-driven fixed-point
   iteration with a *backtracking acceptance test* that makes the physics-residual
   norm provably non-increasing across accepted steps (Section 3.4). To our
   knowledge this acceptance-gated, residual-conditioned learned update, embedded
   in an end-to-end prediction engine, is new as an integrated mechanism.
2. **An integrated trust-and-fallback engine** — a physics checker and an
   uncertainty estimator fused into a per-cell trust map that both *gates* the
   correction and *triggers* a classical-CFD patch on the flagged region only.
3. **A reproducible, CPU-first open-source package** (`neuroforge-cfd`) with a
   frozen I/O contract, multiple interchangeable backbones, a zero-download
   synthetic pseudo-RANS data generator, an AirfRANS loader, and a benchmark
   harness — so every claim in this paper runs on a laptop without a GPU or a
   dataset download.

We are explicit throughout about what is **implemented** versus **planned**, and
we make no SOTA accuracy claims pending full-scale training (Sections 5–6).

---

## 2. Related Work

**Neural operators (FNO / Geo-FNO).** Fourier Neural Operators learn a
resolution-agnostic mapping between function spaces by parameterising a global
convolution in the spectral domain (Li et al., 2021). Geo-FNO (Li et al., 2022;
arXiv:2207.05209) extends this to irregular geometries by learning a deformation
that maps the physical domain to a latent uniform grid where the FFT is valid.
These are strong *one-shot* operators; NeuroForge implements both as backbones
but treats the operator as the *first guess* inside a verify-and-correct loop
rather than as the final answer.

**Point-cloud iterative operators (DoMINO, DrivAerML).** DoMINO (NVIDIA;
arXiv:2501.13350) is a decomposable, multi-scale, point-cloud operator for
external automotive aerodynamics that predicts surface and volume fields and is
evaluated on large industrial datasets such as DrivAerML. It demonstrates that
geometry-native, locality-aware operators scale to complex 3-D bodies.
NeuroForge shares the geometry-native ambition but differs in mechanism: its
iteration is *physics-residual-driven with an acceptance test*, not an
architectural multi-scale decomposition, and its present scope is 2-D structured
grids.

**Physics-attention transformers (Transolver, Transolver++).** Transolver (Wu et
al., ICML 2024; arXiv:2402.02366) replaces quadratic token-to-token attention
with attention over a small set of learnable *physics slices*, giving
linear-in-points cost and strong accuracy on PDE benchmarks; Transolver++
(arXiv:2502.02414) scales this to massive meshes with improved slice
representations and parallelism. NeuroForge includes a Transolver-style
physics-attention backbone (Section 4) as one interchangeable predictor; the
contribution here is the surrounding loop, not the attention mechanism.

**Physics-informed neural networks (PINNs).** PINNs (Raissi et al., 2019) embed
the PDE residual in the loss to fit a single solution (or a parameterised family)
by collocation. They are powerful for inverse problems and individual solves but
typically require re-optimisation per case and can struggle with stiff,
high-Reynolds turbulent RANS — so they are *not* NeuroForge's primary engine. We
borrow their key idea, the differentiable residual, and use it twice: as a
training-loss regulariser (Section 4) and, decisively, as the *runtime verifier*
and correction signal (Section 3) — moving the residual from training-time only
to an inference-time control loop.

**Residual-based error correctors.** A line of work learns to predict and correct
the *error* of a coarse or surrogate solution (e.g. arXiv:2306.12047 on learned
error correction for PDE surrogates). NeuroForge's correction net is in this
family — it is trained to predict the correction *toward truth* conditioned on the
current physics residual — but is distinguished by being applied iteratively under
a residual-monotone acceptance test inside the engine, rather than as a single
post-hoc additive pass.

**Physics-adaptive and iterative refinement (PAR-DeepONet, PDE-Refiner).**
Physics-adaptive refinement methods (PAR-DeepONet) and iterative-refinement
generative schemes such as PDE-Refiner (Lippe et al., 2023) improve a prediction
through successive passes — PDE-Refiner via a diffusion-style multi-step denoising
that restores high-frequency content. NeuroForge's iteration is conceptually
adjacent but mechanistically different: each step is *deterministic*,
*residual-conditioned*, and *accepted only if it does not increase the physics
residual*, giving an explicit, monotone convergence signal rather than a fixed
number of refinement passes.

**Uncertainty quantification for neural PDE surrogates.** Deep ensembles
(Lakshminarayanan et al., 2017) and Monte-Carlo dropout (Gal & Ghahramani, 2016)
are standard epistemic-uncertainty estimators, and recent work studies
calibration-aware UQ specifically for neural PDE/CFD surrogates (e.g.
arXiv:2503.03178 and related calibration studies). NeuroForge implements both a
deep ensemble and MC-dropout and, novelly for this setting, *fuses* the resulting
uncertainty with the physics residual into a single trust field that **acts** —
gating corrections and triggering fallback — rather than merely being reported.

**Benchmarks (AirfRANS, AhmedML, DrivAerNet++).** AirfRANS (Bonnet et al.,
NeurIPS 2022) provides ~1000 incompressible steady RANS simulations over NACA
4/5-digit airfoils (Re ≈ 2–6×10⁶, AoA −5°→15°) as point clouds with `full`,
`scarce`, `reynolds`, and `aoa` splits designed to probe generalisation; AhmedML
and DrivAerNet++ extend the trend to 3-D bluff and vehicle bodies. NeuroForge
adopts AirfRANS as its primary benchmark (Section 5) and its splits as the
generalisation protocol; AhmedML / DrivAerNet++ are roadmap targets.

### 2.1 Positioning

| Approach | Core idea | Self-verifies physics at inference? | Estimates uncertainty? | Iterative self-correction? | Classical fallback? |
|---|---|---|---|---|---|
| FNO / Geo-FNO | spectral neural operator | no | no | no | no |
| DoMINO (point-cloud) | multi-scale geometry-native operator | no | no | no | no |
| Transolver / ++ | physics-slice linear attention | no | no | no | no |
| PINNs | residual in training loss, per-case fit | implicit (training) | no (typically) | optimisation, not closed-loop | no |
| Residual error correctors | learn & add the error field | partial | no | usually one-shot | no |
| PDE-Refiner / PAR | multi-step refinement | no | no | fixed-step refinement | no |
| Deep ensembles / MC-dropout | UQ for surrogates | no | yes | no | no |
| **NeuroForge CFD** | **predict → verify → UQ → residual-monotone correct → gated fallback** | **yes (runtime)** | **yes (ensemble / MC-dropout)** | **yes (acceptance-gated)** | **yes (trust-gated, interface)** |

The novelty is the **integration**: a closed self-correction loop with a
guaranteed monotone residual, fused with uncertainty into an acting trust map
that gates a classical fallback — packaged as one engine. Each individual block
draws on prior art, cited above and academically honestly attributed.

---

## 3. Method

### 3.1 Pipeline overview

```
CAD / STL / airfoil + BCs
        │
        ▼  geometry-native encoding (SDF + solid mask + coords + freestream + log Re)
   x ∈ ℝ^{7×H×W}
        │
        ▼  neural-operator / transformer backbone  f_θ
   ŷ ∈ ℝ^{4×H×W}  (u, v, p, ν_t)        ◀───────────────┐
        │                                               │
        ▼  physics residual checker  R(·)               │  Neural
   continuity, momentum_x, momentum_y, BC violation     │  Residual
        │                                               │  Iteration
        ▼  uncertainty  σ(·)  (deep ensemble / MC-dropout)
        │                                               │
        ▼  trust map  T = g(‖R‖, σ)  → {green, yellow, red}
        │                                               │
        ▼  local correction net  c_φ(field, residual, geom) → Δ
            apply Δ under backtracking acceptance test ─┘
        │   (residual norm provably non-increasing)
        ▼  (only if a region stays red / high-σ)
   uncertainty-gated classical CFD patch  (interface)
        │
        ▼  flow field · Cp · wall shear · Cl/Cd · uncertainty map · residual map · trust map · convergence history
```

### 3.2 Geometry-native encoding and governing equations

A `FlowCase` (geometry + boundary conditions + fluid + domain) is encoded into a
fixed channel-first stack `x ∈ ℝ^{7×H×W}` in the frozen `INPUT_CHANNELS` order
`(sdf, mask, x, y, u_in, v_in, log_re)`: the signed distance to the body surface
(negative inside the solid), a fluid/solid mask, normalised cell coordinates, the
freestream velocity components broadcast over the grid, and $\log_{10}\mathrm{Re}$.
The backbone outputs `OUTPUT_CHANNELS` $(u, v, p, \nu_t)$, where $p$ is the
**kinematic** pressure $p/\rho$.

The verifier evaluates the steady, incompressible, 2-D RANS equations in
primitive form on the **physical (denormalised)** fields, with effective
viscosity $\nu_{\mathrm{eff}} = \nu + \nu_t$ (laminar plus turbulent eddy
viscosity), evaluated pointwise:

$$
\text{continuity:}\quad r_c = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y},
$$

$$
\text{x-momentum:}\quad r_x = u\,\frac{\partial u}{\partial x} + v\,\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu_{\mathrm{eff}}\,\nabla^2 u,
$$

$$
\text{y-momentum:}\quad r_y = u\,\frac{\partial v}{\partial x} + v\,\frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu_{\mathrm{eff}}\,\nabla^2 v.
$$

Because $p$ is kinematic, no explicit density appears. Derivatives use central
finite differences with one-sided stencils at the borders; residuals are zeroed
inside the solid (the equations do not hold there). A boundary-condition
violation map $r_{bc}$ adds two penalties: a **no-slip** term penalising velocity
magnitude $|U|$ in the thin fluid band adjacent to the wall (weighted by a
proximity factor $\exp(-|\mathrm{sdf}|/\ell)$ with $\ell\!\approx\!3$ cells, and
emphasised on the discrete solid→fluid transition cells where $|U|$ should vanish),
and a **far-field** term penalising the deviation
$\lVert(u-u_\infty,\,v-v_\infty)\rVert$ on the outer one-cell border ring.

### 3.3 Trust map: fusing residual and uncertainty

The combined PDE-residual magnitude
$\rho_{\mathrm{res}} = \sqrt{r_c^2 + r_x^2 + r_y^2 + r_{bc}^2}$ and a per-cell
predictive uncertainty $\sigma$ are each mapped to $[0,1]$. NeuroForge uses an
**absolute physical reference scale** when available — the advective/continuity
scale $s = U_\infty^2/L + U_\infty/L$ from the case (with $L$ the chord) — so the
trust thresholds have physical meaning rather than being self-relative; absent a
scale it falls back to a robust 95th-percentile normalisation over the fluid with
a small absolute floor so a *uniformly tiny* residual reads as high trust rather
than saturating. The two normalised signals are combined,

$$
e = \mathrm{clip}\!\big(w_r\, \hat{r} + w_u\, \hat{\sigma},\; 0,\; 1\big), \qquad
T = 1 - e,
$$

with default weights $w_r=0.6$, $w_u=0.4$ (renormalised to sum to one). The
traffic-light class is **green** (2) where $e < 0.15$, **red** (0) where
$e > 0.45$, else **yellow** (1). Uncertainty $\sigma$ is the channel-mean standard
deviation produced by a deep ensemble (disagreement across independently
parameterised members) or by MC-dropout (spread over stochastic forward passes
with dropout kept active); when no estimator is attached, $\sigma = 0$ and trust
is driven by the residual alone.

### 3.4 Neural Residual Iteration (the core contribution)

Let $y_k$ be the field at iteration $k$, with diagnostics yielding a scalar
residual norm $N(y_k) = \sqrt{\overline{r_c^2 + r_x^2 + r_y^2}}$ (root-mean-square
over the grid). A learned local correction operator $c_\phi$ — a small residual
CNN — predicts an additive correction conditioned on the *current* field, its
*current* 3-channel physics-residual map, and the geometry channels:

$$
\Delta_k = c_\phi\big(\text{field}=y_k,\; \text{residual}=R(y_k),\; \text{geom}=x\big).
$$

The correction is computed in the normaliser's space (matching $c_\phi$'s
training) and the additive delta is mapped to physical units by the output
standard deviation only — the affine mean offset cancels for a *delta*. With
trust gating enabled, $\Delta_k$ is scaled by $(1 - T)$ so it acts mainly where
the field is least trustworthy, and it is always zeroed inside the solid.

The decisive ingredient is a **backtracking acceptance test**. A candidate
$y_{k+1} = y_k + s\,\Delta_k$ is accepted only if it does not increase the
residual norm:

$$
N(y_k + s\,\Delta_k) \le N(y_k) + \varepsilon .
$$

If the full step ($s = s_0$) fails, $s$ is halved up to four times; if no step is
accepted, the loop stops. Consequently:

$$
N(y_0) \ge N(y_1) \ge \dots \ge N(y_K),
$$

i.e. **the residual norm is monotone non-increasing across accepted iterations.**
This is the same convergence discipline a classical solver enforces when it
iterates until residuals drop — but here the update is a *learned,
residual-conditioned* operator rather than a linear solve, and it operates over
*unseen* geometries. The loop also halts on a residual tolerance, a stalled
relative improvement, or the iteration cap.

The correction net is trained (with the backbone frozen) to predict the
*correction toward truth* conditioned on the residual: the frozen backbone emits
$\hat{y}$ in normalised space; the physics residual is computed on the
denormalised field and renormalised; and $c_\phi$ is fit under a masked MSE so
that

$$
c_\phi\big(\hat{y}_{\mathrm{norm}},\, R_{\mathrm{norm}},\, x\big) \;\approx\; y^{\star}_{\mathrm{norm}} - \hat{y}_{\mathrm{norm}},
$$

with $y^\star$ the ground-truth field. The final convolution is initialised near
zero, so an untrained corrector produces a negligible correction and the first
iteration is stable.

### 3.5 Uncertainty-gated classical fallback

After the loop, if the maximum uncertainty exceeds a configurable threshold, the
region where trust $< 0.5$ (and the cell is in the fluid) is handed to a
`ClassicalFallback`. The intent is to extract the flagged sub-region, run a local
steady incompressible solve (e.g. `simpleFoam`) initialised from the engine's
prediction, and blend the result back across the trust gradient. In the present
release the fallback is an **interface with a `stub` backend** that reports what
*would* run (and the region size) without invoking an external solver; the
`openfoam` and `su2` backends raise `NotImplementedError` with setup guidance.
This keeps the engine importable and runnable with nothing installed while fixing
the integration seam.

---

## 4. Implementation

NeuroForge is an open-source, **CPU-first**, pure-Python package
(`neuroforge-cfd`, Python ≥ 3.10, NumPy/SciPy/PyTorch/Matplotlib). Importing the
package performs no heavy work and caps BLAS/OMP/MKL thread counts to one by
default (Section 4.3). The codebase is organised around a **frozen I/O contract**
in `core/` that every other module imports through stable signatures.

**Module map.**

- `core/` — frozen data contracts (`FlowCase`, `FlowField`, `Diagnostics`,
  `SolveResult`, `Domain`, `Geometry`) and the `Config` schema. NumPy-only, never
  imports torch.
- `geometry/` — NACA 4/5-digit airfoils and `.dat` loaders, signed-distance and
  solid-mask rasterisation, surface normals, STL/OBJ stubs, and `encode_case`
  (which builds the 7-channel network input).
- `data/` — the synthetic pseudo-RANS generator, the AirfRANS loader, a
  point-cloud rasteriser, and a `datamodule` with a per-channel `Normalizer` and
  a `FlowDataset` bridge to training.
- `models/` — `FNO2d`, `GeoFNO`, a Transolver-style `PhysicsTransformer`, `UNet`
  and grid-to-grid `DeepONet` baselines, the `LocalCorrectionNet`, and the
  `DeepEnsemble` / `MCDropoutUQ` uncertainty wrappers, all registered via a
  string-keyed model registry.
- `physics/` — backend-agnostic differential operators, the `PhysicsChecker`
  (residuals + diagnostics), `trust_map`, force/Cp/field-error metrics, and the
  differentiable `physics_residual_torch` used in the loss.
- `solver/` — `Predictor`, `NeuroForgeEngine`, `neural_residual_iteration`, and
  `ClassicalFallback`.
- `train/` — the `CompositeLoss` and `Trainer` (backbone + corrector).
- `viz/`, `cli.py`, `app/` — plotting, the HTML report, the `neuroforge` CLI, and
  a Streamlit UI.

**The fixed I/O channel contract.** Inputs are always the 7 channels
`(sdf, mask, x, y, u_in, v_in, log_re)`; outputs are always the 4 channels
`(u, v, p, ν_t)`. Structured fields are `(ny, nx)` `float32`; network tensors are
channel-first `(B, C, ny, nx)`. Pressure is kinematic, and physics residuals are
always computed on physical (denormalised) fields with $\nu_{\mathrm{eff}} =
\nu + \nu_t$. Fixing this contract is what makes the backbones interchangeable and
the engine backbone-agnostic.

**Backbones.** `FNO2d` is a faithful spectral FNO: a $1\times1$ lifting
convolution, several Fourier layers (each `rfft2` → low-mode complex channel
mixing → `irfft2`, in parallel with a $1\times1$ residual path and a
nonlinearity), and a two-layer projection head; the kept-mode count is clamped to
what each grid resolution can provide. `GeoFNO` adds a small CNN that reads the
geometry channels and produces a per-cell gate/bias conditioning the lifted
features before the spectral core. `PhysicsTransformer` implements Transolver-style
physics attention: each grid token is softly assigned to one of `n_slices`
learnable slices, full multi-head attention runs only among the (few) slice
tokens — cost linear in the number of grid points — and the attended tokens are
scattered back. `UNet` and `DeepONet` (a branch CNN over the input stack and a
coordinate trunk MLP) are baselines.

**Training loss.** The `CompositeLoss` sums (i) a masked data MSE over fluid cells
in normalised space, (ii) a physics term — the squared steady-RANS residuals from
`physics_residual_torch` on the *denormalised* fields, made fully differentiable
so gradients flow to the prediction (the term is skipped if it goes non-finite),
and (iii) a no-slip BC term penalising velocity magnitude inside/at the solid. The
correction net is trained separately with the backbone frozen, as in Section 3.4.

**Synthetic Hess–Smith pseudo-RANS generator (zero-download reproducibility).**
`SyntheticRANS` produces RANS-like 2-D airfoil fields *without a real solver*, so
the entire pipeline runs with no dataset download. It superposes: (1) a
**potential core** — uniform freestream plus a **Hess–Smith source-panel**
distribution on the body (point sources at panel midpoints, solved so the body is
near-impermeable, with a small Tikhonov regulariser) and a single bound vortex at
the quarter-chord enforcing the Kutta condition via thin-airfoil
$\Gamma = -\tfrac{1}{2}U c\, C_l$; (2) a **viscous correction** — a near-wall
no-slip ramp $1 - \exp(-(d/\delta)^{1.4})$ and a Gaussian wake deficit, with
boundary-layer thickness $\delta \sim c/\sqrt{\mathrm{Re}}$ floored to a few
cells; (3) **kinematic Bernoulli pressure**
$p \approx \tfrac{1}{2}(U_\infty^2 - |u|^2)$; and (4) a mixing-length-style $\nu_t$
concentrated in the boundary layer and wake. Singular kernels are desingularised
at $\sim$1.5 cell diagonals (Rosenhead–Moore), overspeed is clipped, and a light
binomial smoothing yields RANS-like fields with small discrete divergence. The
fields are physically plausible and continuity-respecting, but **analytic** — they
are a reproducibility and smoke-test substrate, not a substitute for solver data.

**AirfRANS loader.** `load_airfrans` lazily imports the `airfrans` package, reads
each simulation's `(M, 11)` point cloud, reconstructs the airfoil loop from the
on-wall points, rasterises the target $(u, v, p, \nu_t)$ onto a structured crop
around the body (`scipy.griddata`), and returns `(FlowCase, FlowField)` pairs for
the `full`, `scarce`, `reynolds`, and `aoa` splits (air, $\nu = 1.56\times10^{-5}$,
unit chord).

---

## 5. Experiments: protocol and preliminary results

### 5.1 Planned evaluation on AirfRANS

The primary evaluation is on **AirfRANS** across its four splits, chosen to probe
generalisation directly:

- **full** — large train set, in-distribution test;
- **scarce** — few training simulations (data efficiency);
- **reynolds** — train and test on disjoint Reynolds ranges (extrapolation in Re);
- **aoa** — train and test on disjoint angle-of-attack ranges (extrapolation in
  AoA), the closest proxy for unseen operating geometry/conditions.

**Baselines.** U-Net, FNO, DeepONet, and a Transolver-style physics-attention
transformer — all implemented in-package and trained identically — plus the
NeuroForge engine (a backbone wrapped in Neural Residual Iteration with the trust
map and optional UQ).

**Metrics.**

- field relative-$L_2$ error per channel and on speed (fluid-masked);
- force-coefficient errors $\Delta C_l$, $\Delta C_d$ (and $C_m$);
- continuity / momentum residual norms (physics fidelity);
- wall-BC violation (no-slip);
- inference time per case;
- **unseen-geometry generalisation** — the gap between in-distribution and the
  `reynolds`/`aoa` splits, and the *reduction* of that gap attributable to the
  correction loop;
- correction-loop diagnostics — per-iteration residual-norm history (verifying
  monotonicity), trust-class fractions, and how often the fallback is triggered.

Large-scale training and the full AirfRANS comparison are **future work**; we
present the harness and the protocol as the reproducible scaffold.

### 5.2 Bundled synthetic smoke results (illustrative, not SOTA)

The package ships `benchmarks/run_benchmarks.py`, which trains each backbone on a
*tiny* synthetic dataset and evaluates it on a held-out synthetic split. These
runs exist to exercise the scaffold end-to-end on CPU in minutes; the models are
intentionally minuscule (width 16, 3 layers, 4 epochs, $32\times32$ grid, 8 train
/ 4 val cases). **They are illustrative of the harness, not accuracy claims.**

Representative output from one run:

| backbone | relL2 $u$ | relL2 $p$ | relL2 speed | params | infer ms/case | mean cont. resid. |
|---|---:|---:|---:|---:|---:|---:|
| FNO | 0.265 | 0.848 | 0.265 | 394,836 | ~200 | 2.8×10⁻² |
| U-Net | 0.263 | 0.936 | 0.264 | 484,068 | ~22 | 5.3 |
| DeepONet | 0.264 | 0.851 | 0.266 | 11,972 | ~20 | 1.2×10⁻¹ |
| Transformer | 0.293 | 1.01 | 0.287 | 8,532 | ~360 | 1.1 |

Two patterns are already visible and worth noting precisely. First, the
**spectral / global-operator models (FNO, DeepONet) produce far smaller discrete
continuity residuals** than the local U-Net at this tiny scale, illustrating
exactly the physics signal the engine's verifier and correction loop consume.
Second, absolute field errors are large here — expected for 4-epoch,
width-16 models on a handful of cases — which is precisely why we make no SOTA
claims from these numbers. The `relL2 v` column is large because synthetic
cross-stream velocities are near zero, so a relative norm is ill-conditioned;
this is a known artefact of the illustrative setup, not a model failure.

A single end-to-end `neuroforge demo` solve (NACA 2412, AoA 5°, Re 3×10⁶, on a
$64\times64$ grid with the bundled tiny FNO + corrector) returns a coherent field,
Cp/force coefficients, and the per-iteration history; with the tiny demo corrector
the loop accepts no residual-reducing step beyond the initial diagnosis on this
case (the near-zero-initialised corrector emits a negligible delta), which is the
*correct, safe behaviour* of the acceptance test — it never accepts a step that
fails the monotonicity guarantee. Demonstrating substantive residual reduction is
a function of training the corrector at scale, which is future work.

---

## 6. Limitations

- **2-D first.** The implemented engine operates on structured 2-D grids and
  external airfoil aerodynamics. Arbitrary 2-D bluff bodies and 3-D geometries are
  on the roadmap (`docs/ROADMAP.md`).
- **Synthetic data is analytic.** The zero-download generator is a
  superposition of potential flow, an algebraic boundary layer, and a wake model.
  It is RANS-*like* and continuity-respecting by construction, but it is not solver
  ground truth; quantitative accuracy must be established on AirfRANS.
- **Classical fallback is a stub interface.** The trust-gated fallback fixes the
  integration seam and reports what would run; the OpenFOAM/SU2 backends are not
  yet implemented.
- **Results pending full training.** All accuracy numbers in this draft are
  illustrative smoke results from tiny CPU models. Backbone- and engine-level
  accuracy, the size of the correction-loop improvement, and calibration of the
  trust map remain to be measured at scale on AirfRANS.
- **Monotonicity ≠ accuracy.** The acceptance test guarantees the *physics
  residual norm* does not increase; it does not by itself guarantee convergence to
  the true field, and a poorly trained corrector can simply make no progress (it
  cannot do harm). Establishing that lower residual tracks lower field error
  empirically is part of the planned evaluation.

---

## 7. Conclusion

NeuroForge CFD reframes neural CFD from a one-shot surrogate into a *solver-like*,
self-correcting engine: predict with a geometry-native operator, verify against
the steady-RANS residuals, estimate uncertainty, and iterate a learned,
residual-conditioned correction under a backtracking acceptance test that makes
the residual norm monotone non-increasing — falling back to a classical solver
only where a fused trust map says the prediction cannot be trusted. The
contribution is the *integration* of these ideas into one reproducible,
CPU-first, open-source engine with a frozen I/O contract, multiple backbones, a
zero-download synthetic data generator, an AirfRANS loader, and a benchmark
harness. We are explicit that quantitative accuracy is pending full-scale
training on AirfRANS; what this work establishes is a credible, runnable, and
academically honest *architecture for trustworthy AI-first CFD*. See the
[README](../../README.md), the engineer-facing
[architecture document](../architecture.md), and the
[roadmap](../ROADMAP.md) for the implementation status and the staged plan.

---

## References

1. Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A.
   Anandkumar. *Fourier Neural Operator for Parametric Partial Differential
   Equations.* ICLR 2021. arXiv:2010.08895.
2. Z. Li, D. Z. Huang, B. Liu, A. Anandkumar. *Fourier Neural Operator with
   Learned Deformations for PDEs on General Geometries (Geo-FNO).* 2022.
   arXiv:2207.05209.
3. R. Ranade et al. (NVIDIA). *DoMINO: A Decomposable Multi-scale Iterative Neural
   Operator for external aerodynamics.* 2025. arXiv:2501.13350.
4. H. Wu, H. Luo, H. Wang, J. Wang, M. Long. *Transolver: A Fast Transformer
   Solver for PDEs on General Geometries.* ICML 2024. arXiv:2402.02366.
5. H. Luo, H. Wu, et al. *Transolver++: An Accurate Neural Solver for PDEs on
   Million-Scale Geometries.* 2025. arXiv:2502.02414.
6. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-Informed Neural Networks:
   A Deep Learning Framework for Solving Forward and Inverse Problems Involving
   Nonlinear PDEs.* Journal of Computational Physics, 2019.
7. *Learned residual error correction for PDE / numerical surrogates.* 2023.
   arXiv:2306.12047.
8. P. Lippe, B. Veeling, P. Perdikaris, R. Turner, J. Brandstetter.
   *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers.*
   NeurIPS 2023. arXiv:2308.05732.
9. B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and Scalable Predictive
   Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. arXiv:1612.01474.
10. Y. Gal, Z. Ghahramani. *Dropout as a Bayesian Approximation: Representing
    Model Uncertainty in Deep Learning (MC-Dropout).* ICML 2016. arXiv:1506.02142.
11. *Calibration-aware uncertainty quantification for neural PDE / CFD surrogates.*
    2025. arXiv:2503.03178 (and related: arXiv:2602.11090, arXiv:2603.11052).
12. F. Bonnet, J. Mazari, P. Cinnella, P. Gallinari. *AirfRANS: High-Fidelity
    Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged
    Navier–Stokes Solutions.* NeurIPS 2022 Datasets & Benchmarks. arXiv:2212.07564.
13. *AhmedML: A High-Fidelity Dataset for ML in Automotive Aerodynamics (Ahmed
    body).* 2024.
14. M. Elrefaie et al. *DrivAerNet++ / DrivAerML: Large-Scale Datasets for Data-
    Driven Automotive Aerodynamics.* 2024.

*Note on citation completeness: arXiv identifiers marked for the
calibration-aware UQ entry and the residual-corrector entry should be verified
against the published versions before submission; venues/IDs for the dataset
papers (AhmedML, DrivAerML) are given to the best available reference at time of
writing.*
