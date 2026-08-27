# GOALS — what this project is for

**Living document. Update whenever a goal is reached, dropped, or reframed.**
Companion: `docs/PLANS.md` (what next), this file (why, and what has been won).

Last updated: **2026-08-27**

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
| **Paper 2**: warm-started classical fallback with real cost accounting | Six experiments done. Positive at moderate Reynolds; positive at Re 3e6 *only* with a wall-fitted representation, and that needs a longer budget to firm up. See `PLANS.md` §4. |

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
Measured, controlled, reproducible: **+58% at Re 1e3, +14% at Re 1e4** (at
residual 1e-3); **69.7%** in the Re-1e4 pilot from a neighbouring-case start.
The oracle control passes in every case.

### 2. It is the **representation**, not the resolution — and this is the headline
At **equal output budget** (16,384 values — exactly a 128² grid), a wall-fitted
(arclength, wall-distance) grid saves **+13% to +30%** of iterations at Re 3e6
scored at convergence depth, while a uniform Cartesian grid of the *same size*
costs 22–33%. Projections and coverage matched (69.5% vs 67.7% of cells), so the
only variable is **where the points sit**.

This is the positive, actionable result, and it converts the frozen 7-in/4-out
Cartesian spec from a limitation into the finding: *a surrogate intended to
accelerate a solver should predict on a wall-fitted grid.*

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

### 5. Engineering contributions, reusable
- A **stitch-free C-grid wake cut**: emit the two cut sheets with the *same
  vertex ids* at `j = 0` and blockMesh joins them itself — no `stitchMesh`, no
  degenerate trailing-edge vertex pair.
- A convergence metric that works when `residualControl` never fires
  (`iterations_to_threshold` + `residual_floor`), which on a staircase mesh it
  does not.

---

## Venue thinking

- Paper 1 → **JCP**, submitted 2026-08-25. TMLR fallback. CMAME desk-rejected it
  as not-new-methodology; do not resubmit there.
- Paper 2 → **CMAME** is the natural home *if* framed as new computational
  methodology (the representation criterion + the warm-start protocol), which is
  exactly what CMAME said Paper 1 lacked. JCP alternative.
- A pure negative result invites rejection. Frame as a **criterion plus a
  positive regime**, which is what the evidence actually supports.
