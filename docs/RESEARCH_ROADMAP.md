# NeuroForge — Multi-Paper Research Roadmap

> **Principle.** Narrow first, go deep, then expand. One *correct, well-evaluated*
> contribution per paper beats one broad-but-shallow paper at every venue, and a
> *series* of focused papers + a growing open tool is how a research program
> accumulates real impact and citations. "Fully general — any shape, any fluid,
> all equations" is a **multi-year frontier approached incrementally**, never a
> single deliverable.

The unifying asset across every paper below is the **NeuroForge framework** —
`predict → check physics residuals → estimate (calibrated) uncertainty →
self-correct (contractive DEQ) → fall back to a classical solver only where
needed` — which is **domain-agnostic**. Each new domain/physics ships inside the
same `NeuroForge` API and open-source package, so usage compounds.

---

## Track A — Aerodynamics (the current core)

### Paper 1 (current) — *Self-correcting geometry-native RANS surrogate for airfoils*
- **Scope:** 2-D, incompressible, steady RANS; external aerodynamics; NACA airfoils.
- **Contribution:** the **contractive Deep-Equilibrium corrector** (Banach
  convergence guarantee) + **conformal-calibrated trust map** + the integrated
  predict/verify/correct/fallback engine.
- **Data:** AirfRANS (NeurIPS 2022).
- **Must-have results:** corrector-on/off ablation showing it improves ρ_Cd / field
  MSE (not just residual); residual↔error correlation; calibrated coverage;
  honest baselines (the *real* Transolver / GINO, MeshGraphNet).
- **Realistic venue:** ML-for-science workshop (NeurIPS/ICML) or CMAME / JCP /
  *Physics of Fluids* short paper. Top main-track only if the DEQ clearly wins.
- **Status:** built; pending the AirfRANS training run + ablation.

### Paper 2 — *3-D external aerodynamics + arbitrary boundary conditions*
- **Scope:** 3-D vehicle/wing geometries; multiple inlets/outlets, symmetry,
  moving walls encoded as extra input channels.
- **Method:** point-cloud / body-fitted backbone (the boundary layer needs
  near-wall resolution a uniform grid can't give); DEQ corrector in 3-D.
- **Data:** AhmedML, DrivAerML / DrivAerNet++.
- **Difficulty:** medium-high (3-D data + GPU + 3-D operator).

### Paper 3 — *Compressible & non-isothermal flow with a learned turbulence closure*
- **Scope:** transonic airfoils/wings (shocks), heat transfer; **energy equation**;
  a real turbulence-closure residual (solve the SA/k-ω transport PDE, not a frozen ν_t).
- **Method:** add temperature/density fields + equation of state; shock-aware loss.
- **Data:** UniFoil (transonic/transitional), or generated compressible RANS.
- **Difficulty:** hard (shocks + coupled energy + closure).

### Paper 4 — *A PDE foundation surrogate for external aerodynamics*
- **Scope:** one model across geometries / Reynolds / Mach / BCs — the "broad
  generalization" goal (reachable *partially*, not as literal universality).
- **Method:** large-scale pretraining (cf. Poseidon, DoMINO), in-context/zero-shot
  adaptation, the self-correction loop as the OOD safety net.
- **Difficulty:** very hard (foundation-model scale + compute).

---

## Track B — Electrochemical multiphysics: **PEM fuel cells** (ambitious new application)

**Why.** Massive real-world demand (clean-energy transition); PEMFC CFD is slow
and specialised (ANSYS Fluent / COMSOL have dedicated modules); ML surrogates for
fuel cells are an emerging, high-impact area. The NeuroForge framework
(geometry-native + physics-residual verification + calibrated trust + correction)
**transfers conceptually** — this would be *"NeuroForge for fuel cells."*

**Honest reality.** A PEMFC is **far harder than airfoil RANS** — not one PDE but
a strongly-coupled, nonlinear, **multiphysics, multiscale, multi-domain** system.
"Full PEMFC, all geometries, 3-D, all equations" is a **multi-year program (several
papers / a thesis-scale effort)**, not an extension of the airfoil work, and not a
single paper. It is genuinely exciting and genuinely large.

### The governing equations a *full* PEMFC model couples
1. **Mass** (continuity) of the gas mixture.
2. **Momentum** — Navier–Stokes in the flow channels; Darcy/Brinkman in the porous
   gas-diffusion / catalyst layers.
3. **Species transport** — convection–diffusion for each of H₂, O₂, N₂, H₂O(vapour)
   (Fickian / Stefan–Maxwell), with reaction sources at the catalyst layers.
4. **Charge conservation ×2** — electronic potential Φ_s (solid phase, Ohm's law)
   and ionic potential Φ_m (membrane/ionomer, proton transport).
5. **Electrochemical kinetics** — Butler–Volmer at the anode (HOR) and cathode (ORR)
   catalyst layers; couples species, potentials and overpotentials.
6. **Membrane water transport** — water content λ, electro-osmotic drag + back-
   diffusion (Springer model); conductivity σ(λ, T).
7. **Two-phase water** — liquid saturation in the GDL, capillary transport, phase
   change (condensation/evaporation).
8. **Energy** — temperature with reversible/irreversible reaction heat, ohmic
   heating and phase-change latent heat; conjugate across solid + fluid + membrane.
9. **Closures** — ideal gas, Bruggeman effective properties, capillary
   pressure–saturation curves, etc.

### Why it's hard (set expectations)
- **Strong nonlinear coupling** across ~8 fields and 4 physical domains
  (channel / GDL / catalyst layer / membrane).
- **Multiscale**: membrane ~10 µm, channels ~mm, cell/stack ~cm.
- **Data scarcity**: each high-fidelity 3-D PEMFC sim is expensive; there is no
  "AirfRANS of fuel cells" — a dataset must be generated or sourced.
- The physics-residual loss must encode *all* coupled equations — a major effort.

### Phased plan (each phase ≈ one paper)
- **B1 — 2-D single-channel, reduced physics.** Species + charge (Φ_s, Φ_m) +
  Butler–Volmer at fixed geometry; vary operating conditions (stoichiometry, RH,
  current). Surrogate for **current-density distribution + polarization curve**.
  *This is the credible entry point.*
- **B2 — full 2-D coupled multiphysics.** Add membrane water transport, two-phase
  liquid water, and the energy equation (non-isothermal). Validate against a
  classical solver.
- **B3 — 3-D single cell, varying geometry.** Channel/rib/GDL geometry as input
  (serpentine vs parallel vs interdigitated flow fields) — *geometry generalization*.
- **B4 — stack-level / broad geometry.** Multi-cell, manifolds; the "all geometries"
  goal approached incrementally.

### Data strategy (the bottleneck)
- Generate with **open solvers** — OpenFOAM + `openFuelCell2`, or a custom
  finite-volume PEMFC model — sweeping geometry + operating conditions.
- Or **collaborate** with a fuel-cell modelling group / use published datasets.
- Start small (B1, fixed geometry) so a usable dataset is cheap.

### Method transfer from NeuroForge
- Geometry-native encoding → SDF/point-cloud of the channel/GDL/CL/membrane stack.
- Multi-field output (each species, potentials, λ, saturation, T).
- Physics-residual checker → the coupled PEMFC residuals (built incrementally).
- DEQ corrector + conformal trust → same machinery, new physics.
- **Venues:** *J. Power Sources*, *J. Electrochemical Society*, *Applied Energy*,
  *Int. J. Hydrogen Energy*, plus ML-for-science workshops. Strong real-world pull.

---

## Track C — The software package (cross-cutting, compounding)

- The `NeuroForge` estimator API + engine + tooling is the unifying deliverable;
  every paper above ships inside it.
- **Separate, very achievable publication:** a **JOSS / software paper** for the
  open package once it has real users — software citations compound over time.
- Visualization toward ANSYS-grade (streamlines, vectors, **VTK/ParaView export**,
  interactive 3-D) makes the tool genuinely usable in industry.

---

## Summary

| Paper | Domain | Key new physics / capability | Data | Difficulty |
|---|---|---|---|---|
| **A1** (now) | Airfoil RANS | DEQ corrector + calibrated trust | AirfRANS | Medium |
| **A2** | 3-D aero + BCs | 3-D, arbitrary boundary conditions | DrivAerML | Med-High |
| **A3** | Compressible | energy eq., shocks, real closure | UniFoil / gen. | Hard |
| **A4** | Foundation aero | broad cross-condition generalization | large-scale | Very hard |
| **B1** | PEMFC 2-D | species + charge + Butler–Volmer | generated | Hard |
| **B2** | PEMFC multiphysics | water + two-phase + energy | generated | Very hard |
| **B3** | PEMFC 3-D | 3-D + geometry generalization | generated | Very hard |
| **B4** | PEMFC stack | all geometries / stack scale | generated | Frontier |
| **C** | Software | the open `NeuroForge` tool (JOSS) | — | Low-Med |

## Non-goals (discipline — do *not* attempt early)
Combustion, multiphase free-surface, general transient turbulence, "one model for
every PDE", and "full PEMFC all-at-once". Each is a trap that turns a publishable
result into an unfinished demo. Expand only after the prior phase is *proven*.

**The unlock for the entire program is Paper A1's empirical result.** Prove the
narrow case works (the AirfRANS run + ablation), and every track above becomes a
natural, fundable extension of a credible foundation.
