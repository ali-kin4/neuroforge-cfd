# GOALS — what this project is for

**Living document. Update whenever a goal is reached, dropped, or reframed.**
Companion: `docs/PLANS.md` (what next), this file (why, and what has been won).

Last updated: **2026-08-28**

---

## The ultimate goal

A **self-correcting, geometry-native AI CFD engine** that predicts a flow field,
checks its own physics, knows when not to be trusted, and hands the untrustworthy
cases to a classical solver it has already warm-started — buying classical
accuracy at a fraction of classical cost.

Everything below is a step toward that, or a measured limit on it.

---

## Hard constraints (do not violate)

- **No pay-to-publish.** Subscription/hybrid venues only; take the subscription
  licence. (See memory: `no-apc-venues-only`.)
- **Frozen contracts**: `core/types.py`, `core/config.py`, `models/base.py`,
  `CONVENTIONS.md`. The 7-in/4-out channel spec is load-bearing for Paper 1 and
  must not move while it is under review.
- **Paper 1 states `ClassicalFallback`'s `openfoam`/`su2` backends are
  unimplemented.** That was true at submission and must stay true on `main`.

---

## Goal ladder

### ✅ Reached

| Goal | Evidence |
|---|---|
| A working CPU-first neural CFD engine with physics residuals, UQ and a correction loop | Paper 1, submitted to JCP 2026-08-25 |
| A calibrated trust signal with conformal coverage | Paper 1 |
| The residual-floor theorem | Paper 1 |
| **A real classical solver in the loop** — OpenFOAM v2606 driven from WSL2, body-fitted meshes we generate ourselves, reaching AirfRANS Reynolds | `solver/{openfoam,ogrid,cgrid}.py`, ~340 tests |
| **A measurement rig that cannot fool itself** — every experiment carries an oracle control that must pass before any other arm is read | six experiments, control +68% to +98% in all of them |

### ▶ In progress

| Goal | State |
|---|---|
| **Paper 2**: warm-started classical fallback with real cost accounting | Six experiments done. Positive at moderate Reynolds. At Re 3e6 the wall-fitted result is **not yet established** — it was read off a stalled residual, and the stall has since been traced to under-relaxation and fixed. Re-measuring on a converging solve, scored on Cd/Cl. See `PLANS.md` §4. |

### ○ Queued

- Reliability-benchmark release ("submit AirfRANS predictions, receive an audit
  card"). No such benchmark exists in the field as of 2026-08.
- Audit-driven active learning at scale (`docs/protocols/audit_loop_pilot.md`).
- Fuel-cell track: 3-D PEMFC data via OpenFOAM + **openFuelCell2** (research-grade;
  pins to a specific ESI release, so check version compatibility first).
- 3-D external aero (DrivAerML); compressible + energy equation.

---

## Novel outcomes worth claiming

Written as claims, with the evidence that backs each. **Report the convergence
depth alongside any iteration saving** — the sign changes with it.

### 1. Warm-starting RANS from a surrogate works, and there is a criterion for when
Measured, controlled, reproducible: **+14% at Re 1e4** and **+8% at Re 1e3** (at
residual 1e-3); **+47%** in the Re-1e4 pilot from a neighbouring-case start. The
oracle control passes in every case (+50% to +93%).

Those numbers are the *corrected* ones — the first pass read +58%, +14% and
+69.7% through a parser that took pressure from a third of the way back through
the run (see `PLANS.md` §3.1). The Re-1e3 claim mostly did not survive it, and
the low-Re runs also carry residual floors of 1.1e-4 to 3.5e-4, which puts the
1e-3 threshold only 3–9× above the floor. **Re-measure on the relaxed settings
before this goes in a paper.** The Re-1e4 result is the solid one.

### 2. It is the **representation**, not the resolution — ⚠️ UNDER RE-MEASUREMENT
The intended headline. At **equal output budget** (16,384 values — exactly a 128²
grid), a wall-fitted (arclength, wall-distance) grid appeared to save **+13% to
+30%** of iterations at Re 3e6 where a uniform Cartesian grid of the *same size*
costs 22–33%. Projections and coverage were matched (69.5% vs 67.7% of cells), so
the only variable is **where the points sit**.

**Do not claim this yet.** Those savings were read at residual thresholds sitting
1.9×, 1.3× and 0.9× above the residual floor, where the curve had already
flattened; the same arm reads +15%, +31% and +13% at three adjacent thresholds,
and +10%, +49%, −15% across the three cases. The floor has since been traced to
over-aggressive under-relaxation (0.9 with SIMPLEC) rather than the mesh, and
dropping it to 0.7 takes the residual to 1.9e-6 and still falling. The re-run on
a solve that actually converges — scored on **Cd/Cl convergence**, not on a
residual threshold — is `PLANS.md` §4 move (1).

If it survives, it is the positive, actionable result, and it converts the frozen
7-in/4-out Cartesian spec from a limitation into the finding: *a surrogate
intended to accelerate a solver should predict on a wall-fitted grid.* If it does
not, outcome 4 hardens and the paper is a criterion paper.

### 3. A falsified criterion, and the right one
The intuitive parameter — boundary-layer thickness over cell size, `delta/h` —
looks convincing on a Reynolds sweep (clean sign change at `delta/h = 2.0`) and
is **wrong**: refining the grid at fixed Reynolds does not reproduce it. Across
the sweep `delta/h` moves 5× while the viscous ratio `y(y+=30)/h` collapses
**1660×**. Testing a criterion on two independent axes, and having it fail on
one, is a stronger methodological result than asserting it on one axis.

### 4. Negative results that save the field time
At Re 3e6 with a **uniform Cartesian** surrogate, warm-starting fails and cannot
be fixed by: training (the exact answer fails identically), better projection
(mask-aware round-trip is identical), post-hoc boundary-layer reconstruction, or
resolution (128→421² flat; one cell across the inner layer would need N ≈ 11,800,
28× beyond what AirfRANS contains).

### 5. A methodological result about measuring warm starts
Iteration savings are the currency of this literature, and they are reported
against a residual threshold. Two faults in that practice showed up here, both
of which flip signs rather than shift magnitudes:

- **The threshold has to be far above the residual floor.** Below a few times the
  floor, an iteration count records where a flat curve crosses a line. Report the
  threshold as a *multiple of the floor*, or score the forces instead — Cd and Cl
  settle and stay settled whatever the residual does.
- **The floor itself is often an artifact.** Here it was under-relaxation at 0.9,
  not the 218,987-aspect-ratio cells the mesh carries. It was diagnosable in
  minutes and moved the floor by more than 6×. A negative warm-start result taken
  against an artificial floor is not a result about warm starts.

Both are cheap to check and neither appears in the papers we surveyed.

### 6. Engineering contributions, reusable
- A **stitch-free C-grid wake cut**: emit the two cut sheets with the *same
  vertex ids* at `j = 0` and blockMesh joins them itself — no `stitchMesh`, no
  degenerate trailing-edge vertex pair.
- A convergence metric that works when `residualControl` never fires
  (`iterations_to_threshold` + `residual_floor`), and a force-based one
  (`iterations_to_force_band`) for when even that is on the floor.

---

## Venue thinking

- Paper 1 → **JCP**, submitted 2026-08-25. TMLR fallback. CMAME desk-rejected it
  as not-new-methodology; do not resubmit there.
- Paper 2 → **CMAME** is the natural home *if* framed as new computational
  methodology (the representation criterion + the warm-start protocol), which is
  exactly what CMAME said Paper 1 lacked. JCP alternative.
- A pure negative result invites rejection. Frame as a **criterion plus a
  positive regime**, which is what the evidence actually supports.
