# NeuroForge — TODO (future work)

Parked future ideas + the concrete next steps, so the repo stays focused on the
current state. Ordered by horizon. (Full rationale: `docs/RESEARCH_ROADMAP.md`.)

## 🔓 The unlock (do this first — everything depends on it)
- [ ] Run the AirfRANS training on Colab with `corrector='deq'` (full task, L4/A100).
- [ ] Report `model.ablate_corrector()` — does the corrector improve **ρ_Cd / field MSE**, not just residual?
- [ ] Report `residual_error_spearman` — does low residual actually track low error?
- [ ] Calibrate (`model.calibrate`) and report empirical coverage vs the 90% target.
- [ ] Write up Paper A1 if the ablation is positive.

## 🎯 Near-term improvements (raise quality/credibility of A1)
- [ ] **Point-cloud / body-fitted backbone** (resolve the Re≈10⁶ boundary layer — the biggest accuracy lever; why MARIO/Geo-FNO win on AirfRANS).
- [ ] **Honest baselines**: the *real* Transolver, GINO, MeshGraphNet on the native point cloud (not the toy strawmen).
- [ ] **OOD experiments**: AirfRANS `reynolds` / `aoa` splits; report the in-dist→OOD gap and whether the corrector reduces it.
- [ ] **Multi-seed** runs + error bars; pin dataset/version + split manifest.
- [ ] Real **turbulence-closure residual** (solve the SA transport PDE, not a frozen ν_t).
- [ ] `nut` **log/asinh normalization** (huge dynamic range on real data — minor).

## 🎨 Visualization toward ANSYS-grade
- [ ] **Streamline + vector/quiver plots** (matplotlib `streamplot`/`quiver`).
- [ ] **VTK / `.vti` export** → open in **ParaView** (free, interactive contours/streamlines/3-D).
- [ ] Optional PyVista interactive 3-D viewer.

## 🧩 Engine / product polish
- [ ] Implement the **OpenFOAM / SU2 classical fallback** (currently a stub).
- [ ] STL / STEP geometry import (mesh-free CAD → SDF).
- [ ] Streamlit app: wire to the clean `NeuroForge` API.

## 📚 Research roadmap (future papers — see RESEARCH_ROADMAP.md)
### Track A — Aerodynamics
- [ ] **A2** — 3-D external aero (DrivAerML) + arbitrary BCs.
- [ ] **A3** — compressible + **energy equation** (transonic, heat transfer).
- [ ] **A4** — PDE **foundation surrogate** (broad cross-condition generalization).
### Track B — PEM fuel cells (multi-year; own track)
- [ ] **B1** — 2-D single-channel, reduced physics (species + charge + Butler–Volmer) → polarization curve / current-density. *(entry point)*
- [ ] **B2** — full 2-D multiphysics (membrane water + two-phase + energy).
- [ ] **B3** — 3-D single cell, geometry generalization (serpentine/parallel/interdigitated).
- [ ] **B4** — stack-level / all geometries.
- [ ] Data: generate via OpenFOAM + `openFuelCell2` or collaborate (the bottleneck).
### Track C — Software
- [ ] **JOSS** software paper for the open `NeuroForge` package (once it has users).

## 🚫 Non-goals (stay disciplined — do NOT attempt early)
Combustion · multiphase free-surface · general transient turbulence · "one model
for every PDE" · "full PEMFC all-at-once". Expand only after the prior phase is *proven*.
