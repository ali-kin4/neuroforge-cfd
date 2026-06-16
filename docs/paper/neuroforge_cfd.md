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
source-panel potential-flow smoke-test data generator that makes the whole pipeline
reproducible without any external dataset, and an AirfRANS loader. We lay out the
full experimental protocol on the AirfRANS benchmark and report bundled
synthetic *smoke* results from the benchmark harness; these use deliberately tiny
CPU-sized models and are illustrative of the scaffold, not state-of-the-art
claims. On the in-distribution AirfRANS `full` split (3 seeds), the DEQ
correction loop improves surface-pressure fidelity (−34% surface MSE),
drag-ranking versus its own backbone (ρ_Cd 0.895→0.923), and the residual↔error
trust signal (0.40→0.83), at a *measured cost* to volume cross-stream and
pressure accuracy (mse_v +129%, mse_p +57%); the feed-forward corrector does not
improve accuracy. We therefore position the engine as a trust-signal-bearing
surrogate with an honest accuracy trade-off, not as a uniform accuracy improver.
Out-of-distribution generalisation and large-scale/external-baseline validation
are explicitly future work.

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

> **Thesis (one sentence).** *NeuroForge CFD explores an AI-first, self-correcting
> CFD workflow intended as a **ranking-preserving surrogate for early-design
> exploration**, combining geometry-aware neural operators, physics-residual
> *monitoring*, uncertainty estimation, and local adaptive correction.*

> **Honest scope & relationship to prior work (please read).** This is an
> early-stage research scaffold, not a validated solver. (1) *Residual ≠
> correctness*: a low PDE residual is **necessary but not sufficient** — a smooth
> near-freestream field can have *zero* residual yet be entirely wrong, so the
> trust map is a physics-residual **consistency monitor**, not "verified
> confidence"; whether the residual tracks error is an empirical question we now
> measure (residual↔error correlation, §Experiments). (2) *The mechanism is not
> new in isolation*: learned solver-correction with convergence guarantees
> (Hsieh et al., ICLR 2019), learned fixed points for steady PDEs (FNO-DEQ,
> NeurIPS 2023), iterative refiners (PDE-Refiner, NeurIPS 2023), and
> residual-corrector operators (Jha, CMAME 2024) all predate it; NeuroForge's
> contribution is the *integrated, open, reproducible engine*, not a new operator.
> (3) *Resolution limits*: a uniform Cartesian grid cannot resolve a Re ≈ 10⁶
> boundary layer (sub-cell at 128²), so wall quantities are approximate.

The framing is **AI-first CFD with physics-residual-consistency monitoring** (not
"physics-verified confidence"). The classical solver remains the ground truth; NeuroForge
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
   synthetic potential-flow smoke-test data generator, an AirfRANS loader, and a benchmark
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
*calibrated* UQ for operator learning, notably conformal prediction (UQNO, Ma et
al., ICLR 2024, arXiv:2402.01960). NeuroForge currently implements a deep ensemble
and MC-dropout and *fuses* the resulting (presently **uncalibrated**) uncertainty
with the physics residual into a single trust field that **acts** — gating
corrections and triggering fallback. We flag that this fusion is hand-weighted and
not yet calibrated; conformal calibration of the trust threshold (giving coverage
guarantees) is the right next step (see Limitations).

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

### 3.1b Principled corrector: a contractive Deep-Equilibrium fixed point

The backtracking acceptance test only guarantees a *non-increasing residual*,
which is satisfiable by the identity and need not track truth. We therefore also
provide a **principled corrector with a genuine convergence guarantee**, which is
the recommended mechanism. The correction `δ` is defined as the fixed point of a
learned operator

    δ* = T_θ(δ*; c),    T_θ(δ; c) = κ · g_θ([δ, c]),    c = [ŷ, r(ŷ), geom],

where `g_θ` is a CNN whose layers are **spectrally normalised** (each ≤ 1‑Lipschitz)
and `κ < 1`. Then `T_θ` is a `κ`‑contraction in `δ`, so by the **Banach fixed‑point
theorem** the equilibrium exists, is unique, and the iteration `δ_{k+1}=T_θ(δ_k)`
converges geometrically: `‖δ_k − δ*‖ ≤ κ^k ‖δ_0 − δ*‖`. This is a Deep‑Equilibrium
model (Bai et al., 2019); we control the Lipschitz constant via spectral
normalisation (Winston & Kolter, 2020) and train it with **Jacobian‑Free
Backpropagation** (Fung et al., 2022) — the fixed point is found under `no_grad`
and a single extra operator application carries the gradient (O(1) memory). The
operator is trained on *data* (it targets correctness, with the residual as an
informative input feature, rather than minimising a residual that need not
coincide with truth), and at inference the converged `δ*` is applied directly.
A contraction factor `< 1` holds by construction (spectral norm + `κ < 1`); the
*empirical* contraction factor and convergence curve (H5) are pending the
contraction-measurement run on AirfRANS and are not yet measured — we therefore
do not report a specific empirical contraction number here. This converts the "self‑correcting loop" from
integration glue with a vacuous property into a corrector with a *real* fixed‑point
convergence guarantee — addressing the closest prior art (Hsieh et al., 2019;
FNO‑DEQ, 2023) on its own terms.

### 3.1c Calibrated trust via split‑conformal prediction

Raw ensemble / MC‑dropout standard deviations are uncalibrated, so a threshold on
them carries no guarantee. We add **split‑conformal calibration**: on a held‑out
set we compute nonconformity scores `s = |ŷ − y| / σ` and take the
finite‑sample‑corrected `(1−α)` quantile `q`; the band `q·σ` then satisfies the
distribution‑free coverage guarantee `P(|ŷ − y| ≤ q·σ) ≥ 1−α` on exchangeable
data (cf. UQNO, Ma et al., 2024). The trust map can then be thresholded with a
known coverage level rather than a hand‑picked constant; `coverage()` verifies it
empirically. To date this has been demonstrated only on a *synthetic* check (a
deliberately mis‑scaled `σ` is corrected toward the 90 % target); empirical
per‑channel coverage at α=0.1 on real AirfRANS (H4) is **pending** the
calibration run and is not yet measured.

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
- `data/` — the synthetic potential-flow smoke-test generator, the AirfRANS loader, a
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

**Synthetic Hess–Smith potential-flow smoke-test generator (zero-download reproducibility).**
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

### 5.3 In-distribution AirfRANS ablation (Stage 1, 3 seeds, mean ± std)

The pre-registered ablation harness (`benchmarks/ablation.py`, protocol in
`docs/EXPERIMENTS.md`) has been run on **real AirfRANS** for the in-distribution
`full` split (800 train / 200 test, 80 epochs) over
**3 seeds (0, 1, 2)**, reported here as
**mean ± std**. This is the Stage-1 (in-distribution) result; the out-of-distribution
`reynolds`/`aoa` runs (the generalisation claim) are still in progress and are
reported as protocol only (§5.1, §6). These numbers establish the engine's
*in-distribution trade-off profile*; they are not a SOTA claim and do not yet
include the matched-budget external baselines (Transolver/GINO/MeshGraphNet),
which are future work. Metrics follow the AirfRANS community protocol
(`evaluate_cases`): per-channel volume MSE (lower is better; well-conditioned,
*not* relative-$L_2$), surface pressure MSE on the body, Spearman rank correlation
of the force coefficients $\rho_{C_l},\rho_{C_d}$ (closer to 1 is better — what
early-design ranking needs), and `residual_error_spearman` (> 0 means low residual
tracks low error).

| arm | mse_u | mse_v | mse_p | surface_mse_p | $\rho_{C_l}$ | $\rho_{C_d}$ | resid↔err ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| backbone | 3.479±0.105 | 0.385±0.012 | 2444.8±120.7 | 548823±27946 | 0.9868±0.0005 | 0.895±0.013 | 0.397±0.003 |
| backbone (no physics loss) | **1.995±0.042** | **0.323±0.008** | **1963.5±55.5** | 1123649±104822 | **0.9920±0.0002** | **0.945±0.008** | 0.605±0.016 |
| backbone + local corrector | 3.482±0.058 | 0.398±0.007 | 2469.4±71.0 | 541534±20453 | 0.9854±0.0017 | 0.888±0.012 | 0.389±0.031 |
| **backbone + DEQ corrector** | 3.457±0.811 | 0.880±0.064 | 3832.7±331.3 | **361681±12273** | 0.9856±0.0017 | 0.923±0.014 | **0.827±0.002** |

*Table: in-distribution AirfRANS ablation, 3 seeds, mean ± std (std is the
**population** std, `ddof=0`; recomputing with sample std `ddof=1` widens the bars
by ≈ 1.22× at n=3). Lower MSE is better; $\rho$ closer to 1 is better; resid↔err ρ
> 0 supports the trust signal. Bold marks the best arm per column. Effects called
"robust" below are sign-consistent across all 3 seeds with large effect sizes,
not n=3 noise. The eddy-viscosity channel `mse_nut` is omitted because it is
near-degenerate at this scale (≈ 5×10⁻⁸ for every arm with no arm-to-arm signal;
see §6).*

**Reading against the pre-registered hypotheses (honestly).**

- **H1 (the corrector improves accuracy).** *Partially supported, and only when
  rescoped from the pre-registered volume field MSE to surface/ranking metrics —
  a post-hoc rescope we flag explicitly.* Over 3 seeds the DEQ corrector improves
  surface-pressure MSE by **34 %** (548823 → 361681, 3/3 seeds) and drag ranking
  $\rho_{C_d}$ from **0.895 → 0.923** (3/3) versus its own backbone, but
  **regresses** volume `mse_v` (0.385 → 0.880, +129 %) and `mse_p`
  (2445 → 3833, +57 %) on all 3 seeds, and $\rho_{C_l}$ is flat
  (0.987 → 0.986). The feed-forward `local` corrector does **not** beat the
  backbone on any volume MSE channel and is slightly worse on $\rho_{C_d}$, so it
  fails H1 as pre-registered. We do **not** claim the corrector improves accuracy
  in aggregate. (The single-seed preliminary figures previously reported here —
  $\rho_{C_l}$ 0.924 → 0.958, surface MSE −25 % — were artifacts of an
  undertrained run; the $\rho_{C_l}$ improvement in particular does **not**
  replicate and is retracted, see below.)
- **H2 (the physics residual is a valid trust signal).** *Supported, robustly.*
  The residual↔error Spearman correlation is positive for every arm on every seed
  and **roughly doubles under the DEQ corrector, 0.397 → 0.827** (3/3 seeds, std
  ≈ 0.002) — i.e. a lower physics residual tracks lower field error far more
  reliably with the corrector engaged, and the effect is nowhere near the ≤ 0
  falsification threshold. Consistent with our framing, this is evidence that the
  residual is an informative **consistency monitor**, not "verified confidence";
  it does not by itself certify correctness.
- **H3 (DEQ corrector ≥ feed-forward corrector).** *Supported on the
  design-relevant axis only — the two correctors trade off and are not uniformly
  ordered.* The contractive DEQ fixed point beats the feed-forward `local`
  corrector 3/3 on surface-pressure MSE, $\rho_{C_d}$ (0.923 vs 0.888), and the
  residual↔error correlation (0.827 vs 0.389), but is 3/3 **worse** on volume
  `mse_v` (0.880 vs 0.398) and `mse_p` (3833 vs 2469), and tied on `mse_u` /
  $\rho_{C_l}$. H3 holds on the design metrics; on volume it fails.

**The ρ_Cl retraction.** The preliminary single-seed run reported
$\rho_{C_l}$ rising 0.924 → 0.958 under the corrector. This does **not** replicate:
in the 3-seed run the backbone already sits at 0.987 and the DEQ corrector is
0.986 (flat-to-slightly-worse, sign-inconsistent across seeds). The single-seed
0.924 → 0.958 figure was an artifact of the undertrained preliminary run and is
**retracted**.

**Robust trade-off #1 — a volume-accuracy regression (asset, not bug).** The DEQ
corrector **regresses** the volume cross-stream component `mse_v` (0.385 → 0.880,
+129 %, 3/3 seeds) and the volume pressure `mse_p` (2445 → 3833, +57 %, 3/3),
while improving surface-pressure MSE (−34 %), drag ranking, and the trust signal.
Every DEQ seed sits above every backbone seed (non-overlapping distributions),
so this is a genuine, replicated trade-off — the corrector optimises surface and
ranking fidelity at a measured cost to volume fields, not small-sample noise. This
does **not** contradict the §3.4 no-harm guarantee, which is *residual-norm-scoped*
and applies to the backtracking acceptance loop; the DEQ `δ*` is applied directly
(§3.1b) without that acceptance test, consistent with our "monotone residual ≠
accuracy" caveat.

**Robust trade-off #2 — the physics loss itself trades volume/force accuracy for
surface fidelity.** Removing the physics loss (`backbone (no physics loss)`)
**improves** `mse_u`, `mse_v`, `mse_p`, $\rho_{C_l}$, $\rho_{C_d}$ and the
residual↔error correlation — all 3/3 — but **doubles** surface-pressure MSE
(549k → 1124k, 3/3). The physics term's *one* benefit is surface-pressure fidelity,
bought at a uniform cost to volume accuracy and force ranking. Crucially, the
physics-loss-free backbone is the **best force-ranking model in the table**
($\rho_{C_d}$ **0.945** > DEQ 0.923; $\rho_{C_l}$ **0.992**, best of all arms),
with no correction loop at all. We therefore do **not** claim the correction loop
delivers best-in-class force ranking; the loop's value is the **trust signal** and
**surface-pressure fidelity**, while a plain physics-loss-free backbone ranks
forces better. Volume MSE and the trust signal are distinct axes, exactly the
trade-off the pre-registered protocol asks us to report rather than hide.

**Status.** Every number here comes from the committed, reproducible harness
(`benchmarks/ablation.py`, run per `docs/EXPERIMENTS.md`) over the pre-registered
3 seeds. H1 as literally pre-registered (volume field MSE + $\rho_{C_d}$) is **not**
supported; rescoped to surface-pressure + $\rho_{C_d}$ for the DEQ arm only it is
supported with robust 3/3 effects, while the feed-forward corrector fails it. H2 is
robustly supported. H3 holds on the design axis and fails on volume. H4 (conformal
coverage) and H5 (empirical contraction) are **pending** — no coverage or
contraction artifact exists in the Stage-1 data (§3.1b, §3.1c). The
out-of-distribution generalisation result that the paper's thesis ultimately rests
on (the `reynolds`/`aoa` splits) is **forthcoming** and is not asserted here.

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
- **Results are Stage-1 (in-distribution) only.** The §5.2 synthetic numbers are
  illustrative smoke results from tiny CPU models. The §5.3 AirfRANS numbers are a
  real 3-seed in-distribution ablation that establishes the engine's trade-off
  profile, but they are *not* a SOTA claim, carry no matched-budget external
  baseline, and do not bear on generalisation. Out-of-distribution accuracy (the
  `reynolds`/`aoa` splits), trust-map calibration (H4), and the empirical
  contraction factor (H5) remain to be measured.
- **Monotonicity ≠ accuracy.** The acceptance test guarantees the *physics
  residual norm* does not increase; it does not by itself guarantee convergence to
  the true field, and a poorly trained corrector can simply make no progress (it
  cannot do harm). Establishing that lower residual tracks lower field error
  empirically is part of the planned evaluation.
- **Eddy-viscosity channel is near-degenerate.** The $\nu_t$ output channel is
  effectively unlearned at this scale (per-arm volume MSE ≈ 5×10⁻⁸ with no
  arm-to-arm signal); since the residuals use $\nu_{\mathrm{eff}} = \nu + \nu_t$,
  the physics term currently leans on the laminar viscosity. A trainable $\nu_t$
  target is future work.
- **Accuracy is a measured trade-off, not a uniform improvement.** The Stage-1
  in-distribution ablation (§5.3) shows the DEQ correction loop improves
  surface-pressure fidelity, drag ranking versus its own backbone, and the trust
  signal, but *regresses* volume cross-stream and pressure accuracy; the
  feed-forward corrector helps on nothing; and a physics-loss-free backbone ranks
  forces best. The conformal coverage (H4) and empirical contraction factor (H5)
  are not yet measured on AirfRANS, and the out-of-distribution generalisation
  claim is pending Stage 2.

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
harness. The Stage-1 in-distribution AirfRANS ablation (3 seeds) shows the
correction loop is best understood as a **trust-signal-bearing surrogate with an
honest accuracy trade-off** — it strengthens the residual↔error trust signal
(0.40→0.83) and surface-pressure fidelity (−34 %) at a measured cost to volume
accuracy, and it does not deliver best-in-class force ranking. The certified
self-correction framing — the residual-monotone no-harm guarantee, and conformal
coverage of the trust threshold as the calibrated contribution — is the durable
claim; out-of-distribution generalisation, calibrated coverage on real AirfRANS,
and the empirical contraction factor are pending and explicitly future work. What
this work establishes is a credible, runnable, and academically honest
*architecture for trustworthy AI-first CFD*. See the
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
11. Calibrated UQ & closest learned-solver prior art: Z. Ma et al. *Calibrated
    UQ for Operator Learning via Conformal Prediction (UQNO).* ICLR 2024.
    arXiv:2402.01960. — J.-T. Hsieh et al. *Learning Neural PDE Solvers with
    Convergence Guarantees.* ICLR 2019. arXiv:1906.01200. — T. Marwah et al.
    *Deep Equilibrium Based Neural Operators for Steady-State PDEs (FNO-DEQ).*
    NeurIPS 2023. arXiv:2312.00234. — PDE-Refiner: P. Lippe et al. NeurIPS 2023.
    arXiv:2308.05732.
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
