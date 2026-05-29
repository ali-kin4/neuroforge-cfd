# NeuroForge CFD — Roadmap

A staged plan from research prototype to product. Each item is tagged
**[done]** (implemented in the package today), **[partial]** (interface or a
reduced version exists), or **[planned]** (not yet built). See the
[README](../README.md), the [architecture doc](architecture.md), and the
[paper draft](paper/neuroforge_cfd.md) for context.

The guiding principle is unchanged across all stages: **AI-first CFD with
physics-verified confidence** — predict fast, verify the physics, quantify
uncertainty, self-correct, and fall back to a classical solver only where needed.

---

## Stage 1 — Research prototype: 2-D airfoils *(current)*

Goal: prove the self-correcting, geometry-native loop on external airfoil
aerodynamics, end-to-end and reproducibly on CPU.

- **[done]** Frozen data contracts (`FlowCase`, `FlowField`, `Diagnostics`,
  `SolveResult`) and the fixed 7-in / 4-out channel I/O spec.
- **[done]** Geometry-native encoding: NACA 4/5-digit airfoils + `.dat` loader,
  signed distance, solid mask, surface normals, `encode_case`.
- **[done]** Interchangeable backbones via a registry: FNO, Geo-FNO,
  Transolver-style physics-attention transformer, U-Net and DeepONet baselines.
- **[done]** Physics verifier: continuity / momentum / BC residuals on physical
  fields with ν_eff = ν + ν_t, plus the differentiable `physics_residual_torch`.
- **[done]** Uncertainty: deep ensembles and MC-dropout estimators.
- **[done]** Trust map fusing residual + uncertainty into a traffic-light field.
- **[done]** **Neural Residual Iteration** — the residual-conditioned correction
  loop with the backtracking acceptance test (monotone residual norm).
- **[done]** `NeuroForgeEngine.solve` orchestration, `pretrained()` /
  `from_checkpoint`, and the one-call `demo()`.
- **[done]** Composite physics-informed training loss + `Trainer` (backbone and
  corrector).
- **[done]** Zero-download synthetic Hess–Smith source-panel pseudo-RANS data
  generator.
- **[done]** AirfRANS loader (`full` / `scarce` / `reynolds` / `aoa` splits) with
  point-cloud rasterisation.
- **[done]** Visualisation (field / residual / trust / Cp / convergence), HTML
  report, `neuroforge` CLI, and a Streamlit app.
- **[done]** Benchmark harness comparing backbones on synthetic data.
- **[partial]** Classical fallback — interface + `stub` backend implemented;
  OpenFOAM/SU2 backends raise `NotImplementedError` with guidance.
- **[planned]** Full-scale training and the complete AirfRANS evaluation
  (accuracy, force coefficients, generalisation gap, correction-loop gain). All
  current accuracy numbers are illustrative smoke results, **not** SOTA claims.

---

## Stage 2 — Engineering prototype: arbitrary 2-D + simple 3-D *(planned)*

Goal: move beyond airfoils to general bodies and a real validation data path.

- **[planned]** Arbitrary 2-D bluff bodies (cylinders, plates, multi-element and
  user-drawn sections).
- **[planned]** Simple 3-D bluff bodies (e.g. Ahmed-body class geometries).
- **[partial → planned]** STL/OBJ import — loaders exist as stubs in
  `geometry/io.py` (`NotImplementedError("planned for v0.2")`); implement the 2-D
  slice/projection and 3-D path.
- **[planned]** PyVista integration for 3-D meshing, slicing, and visualisation.
- **[planned]** Streamlit app extended for 2-D/3-D geometry upload and inspection
  (the 2-D airfoil app already exists from Stage 1).
- **[planned]** OpenFOAM / SU2 **dataset generator** — produce solver ground truth
  to train and validate on real (not synthetic) RANS data.
- **[planned]** Train backbones + correction loop on solver-generated data and
  report quantitative accuracy.

---

## Stage 3 — Product: drag-and-drop CAD-to-prediction *(planned)*

Goal: a usable tool for early-design engineers.

- **[planned]** Drag-and-drop CAD/STL upload.
- **[planned]** Boundary-condition GUI (set freestream, AoA, Reynolds, fluid).
- **[planned]** Instant prediction with the trust/uncertainty overlay surfaced as
  first-class **reliability warnings**.
- **[planned]** Local correction on demand (run Neural Residual Iteration on the
  flagged regions interactively).
- **[planned]** VTK / ParaView export of fields, residuals, and trust maps.
- **[planned]** Optional **OpenFOAM validation** — one-click classical
  cross-check / fallback on regions the engine flags as untrusted (the
  `ClassicalFallback` interface from Stage 1 is the seam this plugs into).

---

## Non-goals (what NOT to do)

These are explicit boundaries to keep the project honest and focused:

- **Do not claim to replace classical CFD.** NeuroForge accelerates the common
  case and *defers to* the classical solver where its own diagnostics say it must.
  The framing is "AI-first CFD with physics-verified confidence," not "we replaced
  CFD."
- **Do not present illustrative/synthetic results as state-of-the-art.** The
  bundled synthetic data is analytic, and the smoke-test benchmark uses tiny CPU
  models. Quantitative accuracy claims wait for full training on real solver data
  (AirfRANS / OpenFOAM / SU2).
- **Do not chase unsteady, compressible, multiphysics, or transient simulation
  early.** Stage 1 is steady, incompressible, single-phase, 2-D. Scope creep here
  would dilute the core self-correction contribution.
- **Do not build a bespoke classical solver.** Reuse OpenFOAM / SU2 for ground
  truth and fallback; NeuroForge owns the AI + verification + correction loop, not
  a new finite-volume code.
- **Do not break the frozen I/O contract** (`core/types.py`, `core/config.py`,
  `models/base.py`). Backbones, data sources, and fallbacks plug into the fixed
  7-in / 4-out interface; extend via the registry and the documented extension
  points, do not redefine the contract.
- **Do not require a GPU or a dataset download to run the package.** CPU-first and
  zero-download reproducibility (the synthetic generator) are load-bearing design
  constraints, not conveniences.
