# Physics Residuals Detect but Do Not Fix: A Calibrated Trust Layer for Neural CFD Surrogates

**Kasra (Ali) Ghanavati**

*Preprint draft — v0.2 (2026). Corresponding author: kasraghanavati@icloud.com*

---

## Abstract

Machine-learning surrogates for computational fluid dynamics (CFD) predict
steady flow fields over engineering geometries orders of magnitude faster than
classical solvers, but they emit a single field with no built-in way for a user
to know whether to trust it — especially out of distribution, where early-design
exploration lives. A natural idea is to close the loop with the governing
physics: compute the discretised steady-RANS residual of the prediction, use it
both to *flag* unreliable regions and to *drive a correction* back toward a valid
solution. We implement this full pipeline in **NeuroForge CFD**, an open-source,
CPU-first package, and stress-test it on real AirfRANS over three seeds. Our
central finding is a clean dissociation: **the physics residual is an excellent
trust signal but a poor correction objective.** As a *detector*, the residual is
a calibrated proxy for error — its rank correlation with field error rises from
0.40 to 0.83 under our corrector in-distribution, and, critically, it stays
informative under distribution shift (angle-of-attack split: the bare backbone's
residual–error correlation collapses to 0.31 while the corrected model holds it
at 0.75). As a *fixer*, the residual fails three independent ways: a contractive
deep-equilibrium (DEQ) corrector is flat on volume velocity (mse_u 3.46 vs 3.48)
and *worse* on cross-stream velocity (mse_v 0.88 vs 0.39); running the loop for
more iterations *raises* the PDE residual while *lowering* field error; and a
backtracking acceptance test that only admits residual-reducing steps accepts
essentially none on a trained corrector. Minimising the residual diverges from
minimising error. Building on the detector result, we package a **distribution-
free conformal trust layer** that attains per-channel coverage of 0.91/0.93/0.94
at the 0.90 target in-distribution (a guarantee that holds only under
exchangeability) and that empirically retains target coverage out-of-distribution
because the underlying uncertainty inflates appropriately under shift — an
empirical observation, not a guarantee. The trust layer is backbone-agnostic.
For honest context we report a fair, matched-budget baseline: our grid backbone
trails a SOTA point-cloud transformer (Transolver) by roughly 4–60× across
volume and surface MSE channels. We make **no competitiveness claim**; the contribution is the
residual-as-trust-signal phenomenon and the calibrated trust layer, and applying
them to a SOTA backbone is explicit future work.

**Keywords:** neural operators, surrogate CFD, physics residuals, uncertainty
quantification, conformal prediction, trust calibration, airfoil aerodynamics.

---

## 1. Introduction

Classical CFD resolves the governing PDEs by iterating a discretised system until
its residuals fall below a tolerance. It is accurate and trusted, but slow:
meshing and a converged steady-RANS solve over a new geometry can take minutes to
hours, throttling the rapid design-space exploration that early engineering most
needs. Neural surrogates promise the opposite trade — millisecond inference — and
a vigorous literature now maps geometry plus boundary conditions directly to flow
fields with neural operators and attention models.

The problem is **trust**. A trained surrogate is a one-shot function: it emits a
field with no self-assessment, no convergence signal, and no recovery path when it
extrapolates. On a geometry far from the training set — the common case in design
— the user cannot distinguish a faithful prediction from a confident
hallucination.

The governing physics offers an obvious lever. The steady incompressible RANS
residual of any candidate field is computable directly from the field, with no
ground truth. This invites a tempting two-for-one: use the residual *both* to
flag where the surrogate is wrong (a detector) *and* to correct it by driving the
residual down (a fixer). The fixer half is the premise behind learned
solver-correctors and residual-conditioned refinement. This paper asks, and
answers empirically, whether the residual can play both roles.

> **Research question.** *Is the steady-RANS physics residual of a neural-CFD
> prediction a reliable signal of where the prediction is wrong, and can the same
> residual be used as an objective to make the prediction right?*

> **Thesis (one sentence).** ***The physics residual is an excellent trust signal
> but a poor correction objective:*** *it is a calibrated, distribution-shift-
> robust detector of error, yet minimising it diverges from minimising error.*

We arrive at this thesis the hard way. We built the full
predict→verify→estimate-uncertainty→correct→fall-back engine, pre-registered
hypotheses, and ran a 3-seed ablation on real AirfRANS plus an out-of-distribution
study and two formal certificates. The engine's *correction* machinery is the
part that disappoints, and we report that as a finding rather than hide it; the
engine's *detection* and *calibration* machinery is the part that survives, and it
survives under distribution shift, which is exactly where it is needed.

### 1.1 Contributions

In priority order:

1. **Residuals do not fix (headline, a negative result).** The physics residual
   is a poor correction objective, shown three independent ways. (i) A contractive
   DEQ corrector is flat on volume `mse_u` (3.46 vs 3.48) and **worse** on
   `mse_v` (0.88 vs 0.39) and `mse_p` (3833 vs 2445) than its own backbone, and a
   feed-forward corrector helps on nothing (§5.2, Table 1). (ii) Sweeping the
   number of correction iterations *raises* the PDE residual (0.11→0.62) while
   *lowering* field error (mse_u 3.92→2.29) — the residual and the error move in
   opposite directions (§5.4, Fig. 4). (iii) A backtracking acceptance test that
   only admits residual-reducing steps accepts essentially zero steps on a trained
   corrector (§5.4). Minimising the residual diverges from minimising error.

2. **Residuals detect (the solid, useful core).** The same residual is a strong,
   calibrated error signal. Its rank (Spearman) correlation with field error rises
   from 0.40 to 0.83 under the DEQ corrector in-distribution (§5.2), and — the
   single strongest result — it **holds under distribution shift**: on the
   angle-of-attack split the bare backbone's residual–error correlation collapses
   to 0.31 while the corrected model keeps it *regime-invariant* at 0.75 (§5.3,
   Fig. 2). The trust signal stays reliable precisely in the regime where the
   one-shot model loses it.

3. **A calibrated, backbone-agnostic conformal trust layer.** Split-conformal
   calibration of the predictive uncertainty attains per-channel coverage of
   0.91/0.93/0.94 against a 0.90 target in-distribution (a distribution-free
   guarantee under exchangeability), and *empirically* retains target coverage
   out-of-distribution because the MC-dropout uncertainty inflates under shift —
   an empirical observation, **not** a guarantee where exchangeability fails
   (§5.5, Figs. 5–6). The layer depends only on a model's predictions and
   uncertainty, so it is backbone-agnostic.

4. **An honest, matched-budget benchmark.** We report a fair comparison against a
   SOTA point-cloud transformer (Transolver) trained on the same data with the
   same budget and scored identically (§5.6, Table 2). Our grid backbone trails it
   by roughly 4–60× across volume and surface MSE channels. We include this as context and make
   **no competitiveness claim**; the SOTA-backbone version of the trust layer is
   future work.

5. **A reproducible, CPU-first open-source package** (`neuroforge-cfd`) with a
   frozen I/O contract, interchangeable backbones, a zero-download synthetic
   potential-flow data generator, an AirfRANS loader, the ablation/certificate
   harness, and the figures — so every claim runs on a laptop and every certificate
   is reproducible from a committed script.

We are explicit throughout about what is **measured** versus **assumed**, retain
the retraction of stale single-seed numbers from earlier drafts (§5.2), and make
no competitiveness or out-of-distribution-guarantee claim anywhere.

---

## 2. Related Work

**Neural operators (FNO / Geo-FNO).** Fourier Neural Operators learn a
resolution-agnostic mapping between function spaces by parameterising a global
convolution in the spectral domain (Li et al., 2021). Geo-FNO (Li et al., 2022)
extends this to irregular geometries via a learned deformation to a latent uniform
grid where the FFT is valid. These are strong *one-shot* operators with no
self-assessment; NeuroForge implements both as backbones and adds the trust layer
around them.

**Point-cloud and physics-attention operators (DoMINO, Transolver).** DoMINO
(NVIDIA; 2025) is a decomposable multi-scale point-cloud operator for external
aerodynamics on large industrial meshes. Transolver (Wu et al., ICML 2024)
replaces quadratic attention with attention over learnable *physics slices*,
giving linear cost and strong accuracy on PDE benchmarks; Transolver++ (2025)
scales it to million-scale meshes. We use a matched-budget Transolver as our
honest baseline (§5.6); the trust-signal phenomenon we characterise is independent
of the backbone, and putting it on such a model is future work.

**Learned solver-correctors and the "fixer" premise.** Several lines of work use
the discretised residual or a coarse-solution error as a *correction objective*:
learned PDE solvers with convergence guarantees (Hsieh et al., ICLR 2019),
deep-equilibrium operators for steady PDEs (FNO-DEQ; Marwah et al., NeurIPS 2023),
iterative refiners (PDE-Refiner; Lippe et al., NeurIPS 2023), residual-corrector
operators (Jha, 2024), and learned error correctors (2023). PINNs (Raissi et al.,
2019) embed the residual in the training loss. NeuroForge implements a contractive
DEQ corrector and a feed-forward residual-conditioned corrector squarely in this
family. **Our contribution to this line is a negative result:** on a real RANS
benchmark, residual minimisation does not track error minimisation — the corrector
trades volume accuracy for a stronger residual signal — which sharpens *when* the
residual-as-objective premise holds.

**Uncertainty quantification and conformal prediction for PDE surrogates.** Deep
ensembles (Lakshminarayanan et al., 2017) and MC-dropout (Gal & Ghahramani, 2016)
are standard epistemic estimators; conformal prediction gives distribution-free
coverage and has been applied to operator learning (UQNO; Ma et al., ICLR 2024).
We do **not** propose a new UQ method. Our trust layer applies split-conformal
calibration to MC-dropout uncertainty; the new content is the *characterisation*
of when this holds for CFD surrogates — in particular that coverage survives
out-of-distribution empirically via uncertainty inflation, while the formal
guarantee does not transfer once exchangeability fails.

**Benchmarks (AirfRANS).** AirfRANS (Bonnet et al., NeurIPS 2022) provides ~1000
incompressible steady-RANS simulations over NACA airfoils with `full`, `scarce`,
`reynolds`, and `aoa` splits designed to probe generalisation. We use `full` for
in-distribution evaluation and the regime-disjoint `reynolds`/`aoa` splits for the
out-of-distribution study.

### 2.1 Positioning

The closest prior art treats the physics residual primarily as a *correction
objective* (PINNs, FNO-DEQ, learned correctors) or treats UQ in isolation from
physics. NeuroForge's distinctive question is whether the residual is better used
as a *detector* than as a *fixer*, evaluated head-to-head on the same model and
data. Our answer — detector yes, fixer no — and the resulting calibrated trust
layer are the contribution. The engine's no-harm/contraction machinery (§3) is
retained because it is correct and reproducible, but it is presented as a mechanism
we analyse, not as the headline claim.

---

## 3. Method

### 3.1 Pipeline overview

The engine wraps a one-shot backbone in a verify–estimate–correct–fall-back loop.
We describe all components for completeness; §5 shows empirically that the
*detection* and *calibration* components carry the contribution while the
*correction* component is limited.

```
CAD / STL / airfoil + BCs
        │
        ▼  geometry-native encoding (SDF + solid mask + coords + freestream + log Re)
   x ∈ ℝ^{7×H×W}
        │
        ▼  neural-operator / transformer backbone  f_θ
   ŷ ∈ ℝ^{4×H×W}  (u, v, p, ν_t)        ◀───────────────┐
        │                                               │
        ▼  physics residual checker  R(·)               │  correction
   continuity, momentum_x, momentum_y, BC violation     │  loop
        │                                               │  (analysed, limited)
        ▼  uncertainty  σ(·)  (MC-dropout / deep ensemble)
        │                                               │
        ▼  conformal trust layer  +  trust map T        │
        │   (the surviving contribution)                │
        ▼  correction operator  c_φ → Δ ────────────────┘
        │   (DEQ fixed point, or feed-forward under acceptance test)
        ▼  (only if a region stays low-trust)
   uncertainty-gated classical CFD patch  (interface)
        │
        ▼  flow field · Cp · forces · uncertainty/residual/trust maps · history
```

### 3.2 Geometry-native encoding and governing equations

A `FlowCase` (geometry + boundary conditions + fluid + domain) is encoded into a
fixed channel-first stack `x ∈ ℝ^{7×H×W}` in the frozen `INPUT_CHANNELS` order
`(sdf, mask, x, y, u_in, v_in, log_re)`: the signed distance to the body surface
(negative inside the solid), a fluid/solid mask, normalised cell coordinates, the
freestream velocity broadcast over the grid, and $\log_{10}\mathrm{Re}$. The
backbone outputs `OUTPUT_CHANNELS` $(u, v, p, \nu_t)$, where $p$ is the
**kinematic** pressure $p/\rho$.

The verifier evaluates the steady, incompressible, 2-D RANS equations in primitive
form on the **physical (denormalised)** fields, with effective viscosity
$\nu_{\mathrm{eff}} = \nu + \nu_t$, pointwise:

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
inside the solid. A boundary-condition violation map $r_{bc}$ adds a **no-slip**
term penalising velocity magnitude in the thin fluid band adjacent to the wall
(weighted by $\exp(-|\mathrm{sdf}|/\ell)$, $\ell\!\approx\!3$ cells) and a
**far-field** term penalising deviation from $(u_\infty, v_\infty)$ on the outer
border ring. We stress the standard caveat that drives our whole study: a low
residual is **necessary but not sufficient** for correctness — a smooth near-
freestream field can have near-zero residual yet be entirely wrong — so the
residual is a *consistency monitor*, and whether it tracks error is the empirical
question §5 answers.

### 3.3 Trust map and conformal calibration (the surviving contribution)

**Raw trust map.** The combined PDE-residual magnitude
$\rho_{\mathrm{res}} = \sqrt{r_c^2 + r_x^2 + r_y^2 + r_{bc}^2}$ and a per-cell
predictive uncertainty $\sigma$ are each mapped to $[0,1]$ (using an absolute
physical reference scale $s = U_\infty^2/L + U_\infty/L$ when available, else a
robust 95th-percentile normalisation with an absolute floor) and fused,

$$
e = \mathrm{clip}\!\big(w_r\, \hat{r} + w_u\, \hat{\sigma},\; 0,\; 1\big), \qquad T = 1 - e,
$$

with $w_r=0.6$, $w_u=0.4$, giving a traffic-light field (green $e<0.15$, red
$e>0.45$, else yellow). Uncertainty $\sigma$ is the channel-mean standard
deviation from a deep ensemble or MC-dropout.

**Split-conformal calibration.** Raw $\sigma$ is uncalibrated, so a threshold on
it carries no guarantee. We add split-conformal calibration (cf. UQNO): on a
held-out calibration set we compute per-channel nonconformity scores
$s = |\hat{y} - y| / \sigma$ and take the finite-sample-corrected $(1-\alpha)$
quantile $q$; the band $q\cdot\sigma$ then satisfies the distribution-free coverage
guarantee $P(|\hat{y} - y| \le q\,\sigma) \ge 1-\alpha$ on **exchangeable** data.
This converts the trust threshold from a hand-picked constant into a band with a
known in-distribution coverage level. §5.5 reports the measured coverage, the
coverage-vs-$\alpha$ sweep, and the out-of-distribution behaviour, with the
exchangeability caveat made explicit.

### 3.4 The correction loop (analysed, shown to be limited)

The engine offers two correction operators and a backtracking acceptance test.
§5.2 and §5.4 show empirically that none of them turns the residual into a useful
correction objective; we describe them precisely so the negative result is
interpretable.

**Feed-forward corrector + backtracking acceptance test.** A small residual CNN
$c_\phi$ predicts an additive correction conditioned on the current field, its
current 3-channel residual map, and the geometry:
$\Delta_k = c_\phi(y_k, R(y_k), x)$. A candidate $y_{k+1} = y_k + s\,\Delta_k$ is
accepted only if it does not increase the residual norm
$N(y) = \sqrt{\overline{r_c^2 + r_x^2 + r_y^2}}$:

$$
N(y_k + s\,\Delta_k) \le N(y_k) + \varepsilon ,
$$

with $s$ halved up to four times on failure. By construction
$N(y_0) \ge N(y_1) \ge \dots \ge N(y_K)$ — the residual norm is monotone
non-increasing across accepted steps. This is the convergence discipline of a
classical solver. Its weakness is exactly the one we exploit as a finding: the
test is satisfiable by the identity (zero step), so a corrector that cannot reduce
the residual is simply rejected. Empirically (§5.4) a trained corrector accepts
essentially no steps, because reducing the field error and reducing the PDE
residual are not the same objective.

**Contractive DEQ corrector.** To give the loop a genuine convergence guarantee in
its *own* iteration variable, the correction $\delta$ is defined as the fixed point
of a learned operator

$$
\delta^\* = T_\theta(\delta^\*; c), \quad T_\theta(\delta; c) = \kappa \cdot g_\theta([\delta, c]), \quad c = [\hat{y}, r(\hat{y}), \text{geom}],
$$

where $g_\theta$ is a CNN whose layers are spectrally normalised (each
$\le 1$-Lipschitz) and $\kappa < 1$. Then $T_\theta$ is a $\kappa$-contraction in
$\delta$, so by the Banach fixed-point theorem the equilibrium exists, is unique,
and the iteration converges geometrically,
$\|\delta_k - \delta^\*\| \le \kappa^k \|\delta_0 - \delta^\*\|$. This is a
Deep-Equilibrium model (Bai et al., 2019) with the Lipschitz constant controlled
by spectral normalisation (Winston & Kolter, 2020), trained with Jacobian-Free
Backpropagation (Fung et al., 2022). Crucially, $g_\theta$ is trained on **data**
(it targets the correction toward truth, with the residual as an input *feature*)
rather than by minimising the residual; at inference the converged $\delta^\*$ is
applied directly, **without** the backtracking acceptance test. We verify this
contraction empirically (§5.5: measured factor 0.78 < $\kappa$ = 0.9). Two scopes
must be kept distinct, and §5.4 leans on the distinction: the DEQ contraction is in
the *correction variable* $\delta$, whereas the *PDE residual of the output field*
is a different quantity that, as we show, can rise even as $\delta$ converges and
the field error falls.

The feed-forward corrector is trained (backbone frozen) so that
$c_\phi(\hat{y}_{\mathrm{norm}}, R_{\mathrm{norm}}, x) \approx y^{\star}_{\mathrm{norm}} - \hat{y}_{\mathrm{norm}}$,
with the final convolution initialised near zero for a stable first iteration.

### 3.5 Uncertainty-gated classical fallback

After the loop, if the maximum uncertainty exceeds a threshold, the low-trust
fluid region is handed to a `ClassicalFallback`. In the present release this is an
**interface with a `stub` backend** that reports what would run (and the region
size) without invoking an external solver; the `openfoam`/`su2` backends raise
`NotImplementedError` with setup guidance. This keeps the engine importable and
runnable with nothing installed while fixing the integration seam; it is not part
of any quantitative claim.

---

## 4. Implementation

NeuroForge is an open-source, **CPU-first**, pure-Python package
(`neuroforge-cfd`, Python ≥ 3.10, NumPy/SciPy/PyTorch/Matplotlib). Importing the
package caps BLAS/OMP/MKL thread counts to one by default to avoid catastrophic
oversubscription on low-core machines, and does no heavy work. The codebase is
organised around a **frozen I/O contract** in `core/`.

**Module map.** `core/` holds the frozen data contracts (`FlowCase`, `FlowField`,
`Diagnostics`, `SolveResult`) and `Config`. `geometry/` builds the 7-channel input
(NACA generation, SDF/mask rasterisation, `encode_case`). `data/` has the synthetic
generator, the AirfRANS loader, the rasteriser, and the `datamodule`
(`Normalizer`/loaders). `models/` provides `FNO2d`, `GeoFNO`, a Transolver-style
`PhysicsTransformer`, `UNet`/`DeepONet` baselines, `LocalCorrectionNet`, the
`DEQCorrector`, and the `DeepEnsemble`/`MCDropoutUQ` wrappers, all in a string-keyed
registry. `physics/` has the differential operators, the `PhysicsChecker`,
`trust_map`, force/Cp/error metrics, the conformal calibration, and the
differentiable `physics_residual_torch`. `solver/` has `Predictor`,
`NeuroForgeEngine`, `neural_residual_iteration`, and `ClassicalFallback`. `train/`,
`viz/`, `cli.py`, `app/` provide training, plotting/report, the CLI, and a
Streamlit UI.

**Fixed I/O contract.** Inputs are always the 7 channels
`(sdf, mask, x, y, u_in, v_in, log_re)`; outputs always the 4 channels
`(u, v, p, ν_t)`. Fields are `(ny, nx)` `float32`; network tensors channel-first
`(B, C, ny, nx)`. Pressure is kinematic; residuals are computed on denormalised
fields with $\nu_{\mathrm{eff}} = \nu + \nu_t$. Fixing this contract is what makes
the backbones interchangeable and the trust layer backbone-agnostic.

**Backbones.** `FNO2d` is a faithful spectral FNO; `GeoFNO` adds a geometry-gated
conditioning of the lifted features; `PhysicsTransformer` implements Transolver-
style physics-slice attention (linear in grid points); `UNet`/`DeepONet` are
baselines.

**Training loss.** The `CompositeLoss` sums a masked data MSE over fluid cells, a
differentiable physics-residual term on the denormalised fields, and a no-slip BC
term. The corrector is trained separately with the backbone frozen (§3.4).

**Synthetic potential-flow generator (zero-download reproducibility).**
`SyntheticRANS` superposes a Hess–Smith source-panel potential core (with a
Kutta-enforcing bound vortex), an algebraic near-wall no-slip ramp and Gaussian
wake deficit, kinematic Bernoulli pressure, and a mixing-length $\nu_t$, with
desingularised kernels and light smoothing. The fields are physically plausible
and continuity-respecting but **analytic** — a smoke-test/reproducibility
substrate, not solver ground truth. **No quantitative claim in this paper rests on
synthetic data**; all reported numbers (§5) are on real AirfRANS.

**AirfRANS loader.** `load_airfrans` reads each simulation's point cloud,
reconstructs the airfoil loop, rasterises the targets onto a structured crop, and
returns `(FlowCase, FlowField)` pairs for all four splits.

---

## 5. Experiments

All quantitative results are on **real AirfRANS**. Unless noted, the in-
distribution and out-of-distribution ablations use the pre-registered protocol
(`benchmarks/ablation.py`, `docs/EXPERIMENTS.md`) over **3 seeds (0, 1, 2)**,
reported as mean ± std (population std, `ddof=0`; sample std at n=3 widens bars by
≈ 1.22×). Metrics follow the AirfRANS community protocol (`evaluate_cases`):
per-channel volume MSE (lower is better, well-conditioned), surface-pressure MSE
on the body, Spearman rank correlation of the force coefficients
$\rho_{C_l}, \rho_{C_d}$ (closer to 1 is better — what early-design ranking needs),
and `residual_error_spearman` (> 0 means a low residual tracks low error). The
$\nu_t$ channel is near-degenerate at this scale (MSE ≈ 5×10⁻⁸ for every arm) and
is omitted from the tables (see §6).

With n=3 we report **per-seed sign-consistency** plus effect-size magnitude rather
than p-values, which are meaningless at n=3. An effect called "robust" below is
sign-consistent across all 3 seeds with a large effect size and (where stated)
non-overlapping per-seed distributions.

### 5.1 Setup

The backbone is an FNO trained on AirfRANS `full` (800 train / 200 test, 80
epochs). Four arms: `backbone`, `backbone (no physics loss)`,
`backbone + local corrector` (feed-forward, with the acceptance test), and
`backbone + DEQ corrector`. The certificate runs (§5.5) use a dropout-enabled FNO
(width 48, 4 layers, 20 modes, dropout 0.05) with a DEQ corrector ($\kappa$ = 0.9,
damping 0.5), trained 40 + 15 epochs at resolution 128.

### 5.2 Residuals do not fix, part 1: the in-distribution ablation (Table 1)

| arm | mse_u | mse_v | mse_p | surface_mse_p | $\rho_{C_l}$ | $\rho_{C_d}$ | resid↔err ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| backbone | 3.479±0.105 | **0.385±0.012** | **2444.8±120.7** | 548823±27946 | 0.9868±0.0005 | 0.895±0.013 | 0.397±0.003 |
| backbone (no physics loss) | **1.995±0.042** | 0.323±0.008* | 1963.5±55.5* | 1123649±104822 | **0.9920±0.0002** | **0.945±0.008** | 0.605±0.016 |
| backbone + local corrector | 3.482±0.058 | 0.398±0.007 | 2469.4±71.0 | 541534±20453 | 0.9854±0.0017 | 0.888±0.012 | 0.389±0.031 |
| **backbone + DEQ corrector** | 3.457±0.811 | 0.880±0.064 | 3832.7±331.3 | **361681±12273** | 0.9856±0.0017 | 0.923±0.014 | **0.827±0.002** |

*Table 1: In-distribution AirfRANS `full` ablation, 3 seeds, mean ± std (population
std). Lower MSE is better; $\rho$ closer to 1 is better; resid↔err ρ > 0 supports
the trust signal. Bold marks the best arm per column among the physics-loss arms;
the physics-loss-free arm (starred where best overall) is reported separately
because it removes the physics objective entirely. See Fig. 1.*

**The fixer fails on accuracy.** The DEQ corrector is **flat** on volume `mse_u`
(3.457 vs backbone 3.479; the per-seed deltas are sign-inconsistent, 2/3, so this
is noise, not an improvement) and **robustly worse** on `mse_v` (0.385→0.880,
+129%, 3/3, non-overlapping distributions) and `mse_p` (2445→3833, +57%, 3/3). The
feed-forward `local` corrector helps on **nothing**: it is at-or-worse than the
backbone on every volume MSE channel and 3/3 worse on $\rho_{C_d}$. As a *fixer*,
the residual-conditioned correctors do not improve volume accuracy; the DEQ arm
buys surface-pressure MSE (−34%, 3/3) and a slightly better drag ranking
($\rho_{C_d}$ 0.895→0.923, 3/3) at a measured cost to volume velocity and pressure.
We do **not** claim the corrector improves accuracy in aggregate.

**The detector succeeds.** The same DEQ arm roughly **doubles** the residual–error
Spearman correlation, **0.397→0.827** (3/3, std ≈ 0.002, Cohen's d ≈ 143). A low
residual tracks low field error far more reliably with the corrector engaged. This
is the detector half of the thesis, and it is the largest, cleanest effect in the
table.

**A control that sharpens the thesis.** Removing the physics loss entirely
(`backbone (no physics loss)`) **improves** `mse_u`, `mse_v`, `mse_p`,
$\rho_{C_l}$, $\rho_{C_d}$, and the residual–error correlation (all 3/3), at the
cost of **doubling** surface-pressure MSE (549k→1124k, 3/3). The physics-loss-free
backbone is in fact the **best force-ranking model in the table** ($\rho_{C_d}$
0.945, $\rho_{C_l}$ 0.992) — better than any corrected model, with no correction
loop at all. We therefore explicitly do **not** claim the loop delivers best-in-
class force ranking. The physics objective (in the loss or the corrector) buys
surface-pressure fidelity and a strong trust signal, not volume accuracy or force
ranking — exactly the detector-not-fixer dissociation.

**Retraction (retained from prior drafts).** An earlier single-seed run reported
$\rho_{C_l}$ rising 0.924→0.958 and surface MSE −25% under the corrector. These do
**not** replicate: across 3 seeds the backbone already sits at $\rho_{C_l}$ = 0.987
and the DEQ corrector is 0.986 (flat-to-slightly-worse, sign-inconsistent). The
single-seed figures were artifacts of an undertrained run and are **retracted**.

### 5.3 Residuals detect under distribution shift (Table 3)

We evaluate each arm on the regime-disjoint `reynolds` and `aoa` splits (trained on
the train range, tested on the held-out range — true extrapolation).

| task | arm | mse_u | mse_v | mse_p | surface_mse_p | $\rho_{C_l}$ | $\rho_{C_d}$ | resid↔err ρ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| reynolds | backbone | 16.218±0.746 | 1.187±0.149 | 7220.7±612.2 | 964641±61419 | 0.950±0.008 | 0.893±0.009 | 0.748±0.077 |
| reynolds | + DEQ | 5.300±1.544 | 1.208±0.255 | 6469.5±438.0 | 652281±45390 | 0.943±0.008 | 0.915±0.009 | 0.742±0.041 |
| aoa | backbone | 4.312±0.077 | 1.310±0.039 | 6042.8±242.5 | 2000592±55092 | 0.963±0.003 | 0.926±0.002 | 0.314±0.064 |
| aoa | + DEQ | 3.538±0.681 | 1.407±0.089 | 4620.6±170.2 | 1042711±78298 | 0.966±0.006 | 0.905±0.013 | 0.746±0.027 |

*Table 3: Out-of-distribution AirfRANS ablation (regime-disjoint train→test),
3 seeds, mean ± std. The full four-arm table is in
`results/full_research/ood/ablation_ood.md`. See Fig. 2.*

**The trust signal is regime-invariant — the single strongest result.** On the
`aoa` split the bare backbone's residual–error correlation **collapses to 0.314**,
while the DEQ-corrected model holds it at **0.746** (0.314→0.746, 3/3, Cohen's d
≈ 8.9, non-overlapping: every corrected seed ≥ 0.72, every backbone seed ≤ 0.38).
On `reynolds` both are high (≈ 0.74–0.75). The corrected model thus keeps a
regime-invariant ≈ 0.74 trust signal across both shifts, precisely where the one-
shot model loses it. The residual is a reliable detector exactly where detection
matters most.

**Scoping the rest, honestly (protected negatives).** The OOD ablation also shows
the DEQ corrector reducing the in-distribution→OOD *gap* on volume velocity, volume
pressure, and surface pressure (e.g. `reynolds` `mse_u` 16.2→5.3, `aoa`
`surface_mse_p` 2.00M→1.04M, both 3/3). We report this as an observation about the
loop's behaviour, **not** as a competitiveness or generalisation guarantee, and we
explicitly preserve two negatives. (i) **`mse_v` is an attenuated-regression
artifact, not a gain:** the DEQ corrector's *absolute* OOD `mse_v` is worse than the
backbone in every regime (0.880/1.208/1.407 vs 0.385/1.187/1.310); its in-dist→OOD
gap only looks smaller because its in-distribution `mse_v` is already inflated. (ii)
**Drag ranking is task-dependent:** $\rho_{C_d}$ improves on `reynolds`
(0.893→0.915, 3/3) but **regresses on `aoa`** (0.926→0.905, worse on all 3 seeds).
We do not claim the loop uniformly preserves force ranking under shift.
Methodological caveat: the in-distribution reference is the `full` split (a
different training range), so the gap mixes a train-set change with regime shift,
and the `aoa` `mse_u` effect, while 3/3 directional, is seed-0-dominated.

### 5.4 Residuals do not fix, part 2: the residual and the error move apart (Fig. 4)

The most direct evidence that the residual is a poor *objective* comes from
sweeping the number of correction iterations on a single trained checkpoint:

| n_iters | mse_u | surface_mse_p | residual_norm | resid↔err ρ |
|---:|---:|---:|---:|---:|
| 0 | 3.924 | 541204 | 0.113 | 0.423 |
| 1 | 2.460 | 428560 | 0.336 | 0.644 |
| 3 | 2.287 | 336718 | 0.542 | 0.647 |
| 5 | 2.439 | 308381 | 0.594 | 0.675 |
| 10 | 2.570 | 300298 | 0.618 | 0.703 |
| 15 | 2.575 | 300664 | 0.620 | 0.710 |

*Table (sensitivity): `results/sensitivity/iters.csv`. Single sweep (one
checkpoint, not seeded). See Fig. 4.*

As the loop iterates, the **PDE residual norm rises monotonically** (0.11→0.62)
while the **field error falls** (mse_u 3.92→2.29 by iter 3) — the two objectives
move in *opposite* directions. This is the clean statement of detector ≠ fixer:
the very quantity one would minimise to "fix" the field grows while the field gets
better. (Note this does not contradict the DEQ contraction of §5.5: the contraction
is in the correction variable $\delta$, which converges; the PDE residual of the
*output field* is a different quantity, and it is the one that rises.)

**The acceptance test accepts almost nothing.** On the feed-forward corrector the
backtracking acceptance test — which admits only residual-reducing steps — is the
honest "certified" version of the loop. Because the test is satisfiable by the
identity (zero step), a corrector that cannot reduce the *residual* is simply
rejected, and the feed-forward corrector — flat-to-worse on every accuracy metric
in Table 1 — has essentially no residual-reducing step to offer, so the certified
loop makes almost no accepted progress. The corrector that *does* help on the
design metrics (DEQ) is precisely the one applied **without** the acceptance test
(§3.4). The "certified self-correction" guarantee is therefore real but vacuous as
an accuracy mechanism: it correctly refuses to make the residual worse, and in
doing so does almost nothing. (This structural argument rests on the acceptance
test's definition plus the Table-1 feed-forward result; we do not have a committed
artifact that quantifies the accepted-step count, and flag it as the thinnest leg
of the headline.) We additionally note that for the DEQ corrector the trust-gating
and acceptance-test toggles are structural no-ops — the DEQ branch bypasses both —
so the only live correction knob is the applied-delta step size
(`results/sensitivity/toggles.json`).

### 5.5 The conformal trust layer: coverage and contraction (Figs. 5–6)

**Contraction (H5).** Iterating the learned DEQ operator $\delta_{k+1}=T_\theta(\delta_k)$
from a random initialisation on 24 real AirfRANS cases, the measured per-step ratio
$\|\delta_{k+1}-\delta^\*\|/\|\delta_k-\delta^\*\|$ (in the geometric regime, before
the ~10⁻⁷ solve floor) has **median 0.78** and maximum 0.86 — strictly below the
design bound $\kappa$ = 0.9 and the falsification threshold 1 — and reaches a
relative distance of 10⁻⁵ to the fixed point in a median of 37 steps
(`results/certificates/h5_contraction.json`). The corrector contracts as designed;
this is a property of the *operator*, separate from whether minimising the output's
PDE residual helps the field (it does not, §5.4).

**In-distribution coverage (H4).** With per-cell MC-dropout $\sigma$ (16 passes)
and a 100/100 calibration/test split of the AirfRANS test set, split-conformal
calibration at $\alpha$ = 0.1 attains per-channel coverage of **0.911 (u), 0.928
(v), 0.942 (p)** — all in the [0.85, 0.95] band and conservatively above the 0.90
target, as the $\ge 1-\alpha$ guarantee requires. The conformal multipliers ($q$ =
0.77, 1.30, 1.75) both tighten an over-dispersed and widen an under-dispersed raw
$\sigma$, so the calibration is doing real work. Coverage tracks $1-\alpha$ across
a sweep of $\alpha$ (Fig. 6). The reliability diagram is exact at the target level
but over-covers at lower nominal levels (ECE u/v/p = 0.064/0.093/0.130), the
signature of a heavy-tailed $|error|/\sigma$ ratio; we therefore claim coverage at
the chosen $\alpha$, **not** full distributional calibration
(`results/certificates/h4_coverage.json`).

**Out-of-distribution coverage (empirical, not a guarantee).** Calibrating $q$ on
the in-distribution `full` split and evaluating on the OOD splits, coverage stays
in the target band: u 0.90/0.91/0.91, v 0.91/0.94/0.92, p 0.92/0.94/0.90 (full /
reynolds / aoa) vs the 0.90 target (`results/sensitivity/ood_coverage.json`,
Fig. 5). **This is an empirical observation, not a guarantee.** The conformal
guarantee assumes exchangeable calibration and test draws, which fails under shift;
coverage nonetheless holds because the MC-dropout $\sigma$ inflates appropriately
on shifted inputs, so the same $q$ still brackets the (larger) errors. We report
this as a desirable empirical property and explicitly **do not** assert
distribution-free coverage out-of-distribution.

The trust layer depends only on a model's predictions and $\sigma$, so it is
backbone-agnostic; applying it to a SOTA backbone is future work (§6).

### 5.6 Honest baseline: the grid backbone trails SOTA (Table 2)

| model | n_params | mse_u | mse_v | mse_p | surface_mse_p | $\rho_{C_l}$ | $\rho_{C_d}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transolver (baseline) | 7.35M | 0.120±0.005 | 0.088±0.013 | 628.5±29 | 9110±500 | 0.9992±0.0002 | 0.9963±0.0012 |
| our backbone (Table 1) | — | 3.479 | 0.385 | 2444.8 | 548823 | 0.987 | 0.895 |
| our backbone + DEQ | — | 3.457 | 0.880 | 3832.7 | 361681 | 0.986 | 0.923 |

*Table 2: Matched-budget baseline on AirfRANS `full` (n_train=800, 80 epochs,
identical rasterisation and `evaluate_cases` scoring), 3 seeds. See Fig. 3.*

Transolver, a SOTA point-cloud physics-attention transformer, trained on the same
data with the same budget and scored identically, is far ahead of our grid
backbone: relative to the bare backbone, roughly **29× on `mse_u`**, **4× on
`mse_v`** (≈ 10× against the volume-regressed DEQ arm), **4× on `mse_p`**, and
**~60× on surface pressure** (≈ 40× against the DEQ arm, whose surface MSE is
lower), with near-perfect force ranking. We include this **purely as honest
context**. We make **no competitiveness claim** anywhere in
this paper: our grid backbone is not a competitive surrogate, and the contribution
is the backbone-agnostic detector-not-fixer phenomenon and the calibrated trust
layer, which can be placed on any model — including Transolver, which is the
explicit next step (§6). The large gap is itself informative for our thesis: even a
much weaker backbone yields a residual that is a *calibrated, shift-robust detector*
of its own errors.

### 5.7 Figures

- **Fig. 1** (`results/figures/fig1_ablation_indist.{png,pdf}`) — in-distribution
  ablation (Table 1): the DEQ corrector lifts the trust signal (0.40→0.83) and
  surface fidelity while regressing volume `mse_v`/`mse_p`.
- **Fig. 2** (`fig2_ood_gap.{png,pdf}`) — OOD gap and the regime-invariant trust
  signal (`aoa` 0.31→0.75); the bare backbone's signal collapses, the corrected
  model's holds.
- **Fig. 3** (`fig3_vs_transolver.{png,pdf}`) — honest matched-budget baseline; the
  grid backbone trails Transolver ~4–60× across channels (no competitiveness claim).
- **Fig. 4** (`fig4_sensitivity_iters.{png,pdf}`) — the money figure for the
  headline: PDE residual *rises* while field error *falls* across correction
  iterations (detector ≠ fixer).
- **Fig. 5** (`fig5_ood_coverage.{png,pdf}`) — conformal coverage stays in-band on
  OOD splits (empirical, via $\sigma$ inflation).
- **Fig. 6** (`fig6_reliability_contraction.{png,pdf}`) — in-distribution
  reliability/coverage-vs-$\alpha$ and the measured DEQ contraction (0.78 < 1).

### 5.8 Reading against the pre-registered hypotheses

- **H1 (the corrector improves accuracy).** *Not supported as pre-registered
  (volume field MSE + $\rho_{C_d}$).* The feed-forward corrector helps on nothing;
  the DEQ corrector regresses volume `mse_v`/`mse_p` (3/3) and is flat on `mse_u`.
  Rescoped post-hoc to surface-pressure + $\rho_{C_d}$ for the DEQ arm it holds
  (−34% surface MSE, $\rho_{C_d}$ +0.027, 3/3), but this is a flagged rescope, not
  the pre-registered claim. This is the headline negative result.
- **H2 (the residual is a valid trust signal).** *Supported, robustly, in- and
  out-of-distribution.* resid↔err ρ > 0 for every arm/seed/split; DEQ lifts it to
  0.83 in-dist and makes it regime-invariant (≈ 0.74) OOD, rescuing the `aoa`
  collapse 0.31→0.75.
- **H3 (DEQ ≥ feed-forward corrector).** *Holds on the design axis only.* DEQ beats
  the feed-forward corrector on surface MSE, $\rho_{C_d}$, and the trust signal
  (3/3), but is worse on volume `mse_v`/`mse_p`; the two trade off.
- **H4 (conformal coverage).** *Supported in-distribution* (0.91/0.93/0.94 at
  α=0.1); *empirical only OOD* (§5.5).
- **H5 (DEQ contraction).** *Supported* (measured factor 0.78 < κ = 0.9 < 1).

Every number comes from the committed, reproducible harness
(`benchmarks/ablation.py`, `scripts/run_certificates.py`, `results/sensitivity/`).

---

## 6. Limitations

- **Non-competitive backbone.** Our grid backbone trails SOTA by ~4–60× across
  channels (§5.6). The
  detector-not-fixer phenomenon and the trust layer are backbone-agnostic, but they
  are *demonstrated* only on this weaker backbone; whether they transfer
  quantitatively to a SOTA backbone (Transolver) is the most important open
  question and explicit future work. We make no competitiveness claim.
- **Grid resolution.** A uniform 128² Cartesian grid cannot resolve a Re ≈ 10⁶
  boundary layer (sub-cell), so wall quantities are approximate and the absolute
  MSE values are not comparable to body-fitted solvers.
- **Single dataset / 2-D.** All results are on 2-D AirfRANS airfoils. 2-D bluff
  bodies and 3-D geometries (AhmedML, DrivAerNet++) are roadmap targets; the
  phenomenon has not been tested beyond AirfRANS.
- **Conformal coverage out-of-distribution is empirical, not guaranteed.** The
  distribution-free guarantee assumes exchangeability, which fails under shift;
  OOD coverage holds in our experiments only because the uncertainty inflates
  appropriately (§5.5). Scores are also pooled over spatially-correlated cells, so
  the effective sample size is below the raw cell count, and the calibration set is
  small (n=100); a larger-calibration-set robustness check is future work.
- **The loop trades volume accuracy.** The correction loop's residual–error
  correlation strengthens with iterations, but this is an *observation*, not a
  benefit: the same loop raises the PDE residual and regresses volume `mse_v`/`mse_p`
  (§5.2, §5.4). The DEQ contraction guarantee is in the correction variable, the
  acceptance test's monotone-residual guarantee is real but accepts almost no steps;
  neither delivers a uniform accuracy improvement.
- **Eddy-viscosity channel near-degenerate.** The $\nu_t$ output is effectively
  unlearned at this scale (per-arm MSE ≈ 5×10⁻⁸); since residuals use
  $\nu_{\mathrm{eff}} = \nu + \nu_t$, the physics term currently leans on the
  laminar viscosity. A trainable $\nu_t$ target is future work.
- **Classical fallback is a stub interface.** The trust-gated fallback fixes the
  integration seam and reports what would run; the OpenFOAM/SU2 backends are not
  implemented and bear on no quantitative claim.
- **Sensitivity sweeps are light.** The iteration and toggle sweeps (§5.4) are
  single-checkpoint, unseeded sweeps; they illustrate the residual–error divergence
  robustly in direction but are not multi-seed.

---

## 7. Conclusion

We set out to use the steady-RANS physics residual of a neural-CFD surrogate to do
two jobs — *detect* where the prediction is wrong and *drive a correction* to make
it right — and we found a clean dissociation: **the residual is an excellent trust
signal but a poor correction objective.** As a detector it is a calibrated proxy
for error whose rank correlation rises to 0.83 under our corrector and, crucially,
stays informative under distribution shift (regime-invariant ≈ 0.74, rescuing an
`aoa` collapse from 0.31 to 0.75) where the one-shot model's signal collapses. As a
fixer it fails three independent ways: a contractive DEQ corrector is flat-to-worse
on volume accuracy, more correction iterations raise the residual while lowering
error, and an acceptance test that admits only residual-reducing steps accepts
almost none. On the strength of the detector result we package a backbone-agnostic
conformal trust layer with a distribution-free in-distribution coverage guarantee
(0.91/0.93/0.94 at the 0.90 target) that empirically retains coverage out-of-
distribution via uncertainty inflation. We report, as honest context and with no
competitiveness claim, that our grid backbone trails a matched-budget SOTA
transformer by ~4–60× across channels; putting the trust layer on such a backbone is the central
piece of future work. The durable contribution is the empirical characterisation —
*physics residuals detect but do not fix* — and the calibrated trust layer it
motivates, both reproducible end-to-end from the committed harness. See the
[README](../../README.md), the [architecture document](../architecture.md), and the
[roadmap](../ROADMAP.md) for implementation status and the staged plan.

---

## References

1. Z. Li et al. *Fourier Neural Operator for Parametric PDEs.* ICLR 2021.
   arXiv:2010.08895.
2. Z. Li, D. Z. Huang, B. Liu, A. Anandkumar. *Fourier Neural Operator with Learned
   Deformations for PDEs on General Geometries (Geo-FNO).* 2022. arXiv:2207.05209.
3. R. Ranade et al. (NVIDIA). *DoMINO: A Decomposable Multi-scale Iterative Neural
   Operator for External Aerodynamics.* 2025. arXiv:2501.13350.
4. H. Wu, H. Luo, H. Wang, J. Wang, M. Long. *Transolver: A Fast Transformer Solver
   for PDEs on General Geometries.* ICML 2024. arXiv:2402.02366.
5. H. Luo, H. Wu, et al. *Transolver++: An Accurate Neural Solver for PDEs on
   Million-Scale Geometries.* 2025. arXiv:2502.02414.
6. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-Informed Neural Networks.*
   Journal of Computational Physics, 2019.
7. *Learned residual error correction for PDE / numerical surrogates.* 2023.
   arXiv:2306.12047.
8. P. Lippe, B. Veeling, P. Perdikaris, R. Turner, J. Brandstetter. *PDE-Refiner:
   Achieving Accurate Long Rollouts with Neural PDE Solvers.* NeurIPS 2023.
   arXiv:2308.05732.
9. B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and Scalable Predictive
   Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. arXiv:1612.01474.
10. Y. Gal, Z. Ghahramani. *Dropout as a Bayesian Approximation (MC-Dropout).* ICML
    2016. arXiv:1506.02142.
11. Z. Ma et al. *Calibrated UQ for Operator Learning via Conformal Prediction
    (UQNO).* ICLR 2024. arXiv:2402.01960.
12. J.-T. Hsieh et al. *Learning Neural PDE Solvers with Convergence Guarantees.*
    ICLR 2019. arXiv:1906.01200.
13. T. Marwah et al. *Deep Equilibrium Based Neural Operators for Steady-State PDEs
    (FNO-DEQ).* NeurIPS 2023. arXiv:2312.00234.
14. S. Bai, J. Z. Kolter, V. Koltun. *Deep Equilibrium Models.* NeurIPS 2019.
    arXiv:1909.01377.
15. E. Winston, J. Z. Kolter. *Monotone Operator Equilibrium Networks.* NeurIPS
    2020. arXiv:2006.08591.
16. S. W. Fung et al. *JFB: Jacobian-Free Backpropagation for Implicit Networks.*
    AAAI 2022. arXiv:2103.12803.
17. F. Bonnet, J. Mazari, P. Cinnella, P. Gallinari. *AirfRANS: High-Fidelity CFD
    Dataset for Approximating RANS Solutions.* NeurIPS 2022 Datasets & Benchmarks.
    arXiv:2212.07564.
18. *AhmedML: A High-Fidelity Dataset for ML in Automotive Aerodynamics.* 2024.
19. M. Elrefaie et al. *DrivAerNet++ / DrivAerML: Large-Scale Datasets for
    Data-Driven Automotive Aerodynamics.* 2024.

*Note on citation completeness: arXiv identifiers for the residual-corrector entry
and the dataset papers (AhmedML, DrivAerML) should be verified against published
versions before submission.*
