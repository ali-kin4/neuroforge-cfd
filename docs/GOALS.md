# GOALS — what this project is for

**Living document. Update whenever a goal is reached, dropped, or reframed.**
Companion: `docs/PLANS.md` (what next), this file (why, and what has been won).

Last updated: **2026-08-30**

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
| **A real classical solver in the loop** — OpenFOAM v2606 driven from WSL2, body-fitted meshes we generate ourselves, at AirfRANS Reynolds | `solver/{openfoam,ogrid,cgrid}.py`, ~360 tests |
| **A measurement rig that cannot fool itself** — every experiment carries an oracle control that must pass before any other arm is read, and every force band declares its own readability | twelve experiments, control +68% to +99.9% in all of them; `solver/scoring.py` |
| **A trained NeuroForge model that accelerates a production RANS solver, at n=13** | **+18.4% on Cd_v@1% [+12.4, +25.3], 13/13 cases, p = 0.0002**, Re 3e6, oracle control +93.6%, Cartesian negative control null (+3.4%, p = 0.27). On the 5-case mechanism study, +33.9% on Cd@1%. |

### ▶ In progress

| Goal | State |
|---|---|
| **Paper 2**: mesh-native, boundary-layer-only warm starts, with a no-harm certificate | Mechanism measured **and controlled**, guarantee cross-validated, **generality closed at n=13** (all 13 admitted, 13/13 wins, p = 0.0002), **full draft written** (`docs/paper2/DRAFT.md`). **One gap left: wall-clock, at n=0 for the recommended arm** (`PLANS.md` §3.8) — Phase C. The pre-registered +30% bar is **missed at +18.4%** and stays where it is; see `PLANS.md` §0. |

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
depth alongside any saving** — the sign changes with it — and **only quote a row
the readability check passed** (`PLANS.md` §3.3).

### 1. A surrogate must be evaluable at the solver's own cell centres
*The paper's central claim, and the mechanism behind every other result here.*

The first-cell tangential velocity gradient is what viscous drag integrates. Six
cases at Re 3e6, measuring each seed as the solver received it:

| seed | wall-gradient error |
|---|---:|
| cold start | 2851% |
| Cartesian 128² projection of **the exact answer** | 1695% |
| wall-fitted 256×64 projection of **the exact answer** | 1890% |
| **trained NeuroForge, queried at the cell centres** | **54%** |

A 16,384-value grid — Cartesian *or* wall-fitted — has no station 5e-6 chords off
the wall, so it puts near-freestream velocity at the first cell centre and
overestimates the wall shear by a factor of 20. Projecting the *exact answer*
through one removes only a third to a half of a cold start's error in the
quantity that decides drag. Pointwise evaluation on the mesh removes 98%.

The point-cloud literature argues mesh-native evaluation is better *for
prediction accuracy*. This says something stronger and from the other side: it is
the difference between a warm start that pays and one that costs six times more.

**Status (2026-08-30): controlled, and the confound is gone.**
`scripts/mesh_native_probe.py` landed: `nf_bl` vs `nf_bl_proj` is the same
network, the same prediction, the same region, differing only in whether the
field was resampled — +33.9% vs −58.8% on total drag. The old oracle-vs-network
comparison (`fitted_bl` vs `nf_bl`) is no longer load-bearing anywhere and the
table that used it is retired (§2 below). The abstract is written; the draft is
`docs/paper2/DRAFT.md`.

### 2. Both conditions are necessary; neither is sufficient
Cd@1%, five cases, cold = 805 iterations, oracle control +92.1%. Two controlled
contrasts sharing the common arm `nf_bl`, each changing exactly one variable
(**the old 2×2 here is retired — it was stale by ~12 points and mixed oracle and
network arms in the same table; see `PLANS.md` §3.2**):

| axis | arms compared | held fixed | result |
|---|---|---|---|
| **representation** | `nf_bl` vs `nf_bl_proj` | network, BL only, one prediction | **+33.9%** vs **−58.8%** |
| **region** | `nf_bl` vs `nf_mesh` | network, mesh-native | **+33.9%** vs **<−568.3%** |

Mesh-native evaluation preserves the wall gradient; restricting to the boundary
layer avoids handing over an outer field the model extrapolates badly. Only the
arm with both is positive — and it is positive on **total drag**, the quantity
every other arm in the study failed at. Resampling the *exact converged field*
is −172.6% (wall-fitted) and −548.4% (Cartesian), so this is not an accuracy
effect.

### 3. Viscous drag is the quantity that pays, and it is 60–84% of the drag
Three wall-fitted seed constructions, three bands, 5/5 cases, monotone, no sign
flip: `fitted_bl` +41.7%/+31.7%/+26.4%, `fitted_256x64` +37.2%/+29.4%/+24.5%,
`nf_bl` +14.6%/+13.7%/+11.0%, against Cartesian's +10.0%/+12.4%/**−37.9%**.

Stability across bands *is* the evidence this is a rate measurement rather than
a flat curve crossing a line. **Now quantified: `nf_bl` on total drag reads
+33.9% / −4.2% / −38.5% across the same three bands** — the advantage is real at
1% and gone by 0.5% — while `oracle_mesh` is monotone on both, so the
instability belongs to the seed and not to the measurement.

**Decision (2026-08-30):** the §0 bar is not restated around Cd_v, because
+14.6% would fail an item we pre-registered at +30%. Cd@1% stays the headline
and is **never quoted without its band curve**; Cd_v is reported alongside as
the stability evidence. Both rows, always, in that order (`PLANS.md` §3.3).

### 4. A no-harm certificate makes a mixed result deployable
A 25-iteration probe on the residual, with the threshold calibrated on other
cases and applied to a held-out one, converts a population of 70 seeds averaging
**−163.6%** with a **−1169.6%** tail into one averaging **+1.5%** whose worst
single seed is **−5.8%**, admitting **none of the 46 harmful seeds**. Worst case
is bounded at (1 + K/N) × cold by construction. Longer probes are monotonically
worse.

**Claim it as insurance, not as a mean saving.** The mean is small and it moved
when the arm set grew (+3.7% → +1.5% on drag, +1.9% → −8.2% on lift; see
`PLANS.md` §3.6). What did not move is the pair that matters: the tail collapses
and no harmful seed is admitted. **And the gate fails on lift** — its gated worst
equals its ungated worst — which the paper states rather than omits.

This is what makes the paper practical rather than observational, and it is the
property [PCGBandit](https://arxiv.org/html/2509.08765) sells as "never worse
than the default" — supplied here for seeds rather than preconditioners.

### 5. Two falsified predictions, both of them ours
Stronger evidence than the results they replaced, and both stay in the paper.

- **`δ/h` is not the criterion.** It looks convincing on a Reynolds sweep (clean
  sign change at δ/h = 2.0) and fails on the grid axis: refining at fixed
  Reynolds does not reproduce it. Across the sweep δ/h moves 5× while the viscous
  ratio `y(y⁺=30)/h` collapses **1660×**.
- **"Seed the field the solver is slow at" is wrong.** The convergence
  decomposition is real — cold RANS takes ~1850 iterations on pressure drag
  against ~700 on viscous drag — and the inference from it is false. A
  pressure-only seed is *inert* (+0.1% everywhere), potential flow is inert on
  drag (+0.7%), and the composite of the two is negative (−320%). SIMPLE
  recomputes pressure from continuity given the velocity, so a pressure seed
  inconsistent with `U` is overwritten within a few iterations. The winner hands
  over velocity and eddy viscosity inside the boundary layer and **no pressure at
  all**.

### 6. Negative results that save the field time
At Re 3e6 with a **uniform Cartesian** surrogate, warm-starting fails and cannot
be fixed by: training (the exact answer fails identically), better projection
(mask-aware round-trip is identical), post-hoc boundary-layer reconstruction, or
resolution (128→421² flat; one cell across the inner layer would need N ≈ 11,800,
28× beyond what AirfRANS contains).

### 7. A methodological result about measuring warm starts
Six rules, each of which changed a **sign** on this project's own data, each in
`solver/scoring.py` with a test naming the mistake. The two that generalise
furthest beyond CFD:

- **The threshold has to be far above the residual floor**, and **the floor
  itself is often an artifact** — here under-relaxation at 0.9, not the
  218,987-aspect-ratio cells, diagnosable in minutes and worth 30× in floor
  depth. A negative warm-start result taken against an artificial floor is not a
  result about warm starts.
- **Only arms that have stopped moving may define the reference they are all
  scored against.** One diverged arm condemned an entire table here, and the
  blunt version of the rule hid a headline that was real and inflated one that
  was not (+41.8% at Cd@0.5%, withdrawn; on a clean reference over the declared
  arm set that row is readable and reads **−4.2%**). The same rule later made
  Cd@0.5% readable again at a 0.232% spread against a 0.25% limit — a thin
  margin that the 13-case sweep will re-decide (`PLANS.md` §3.3).

Neither appears in the papers we surveyed.

### 8. Engineering contributions, reusable
- A **stitch-free C-grid wake cut**: emit the two cut sheets with the *same*
  vertex ids at `j = 0` and blockMesh joins them itself — no `stitchMesh`, no
  degenerate trailing-edge vertex pair.
- A convergence metric that works when `residualControl` never fires
  (`iterations_to_threshold` + `residual_floor`), a force-based one
  (`iterations_to_force_band`) for when even that is on the floor, and a
  readability test that says which of the two may be quoted.

### 9. Warm-starting works at moderate Reynolds — with a caveat
**+14.4% at Re 1e4** (residual 1e-3), **+47.3%** in the Re-1e4 pilot from a
neighbouring case, oracle control passing throughout. The Re 1e3 claim mostly did
not survive the parser fix (+58% → +8.1%), and those runs carry residual floors
of 1.1e-4 to 3.5e-4, putting the 1e-3 threshold only 3–9× above the floor.
**Re-measure on the relaxed settings before quoting** (`PLANS.md` Phase B5). The
Re-1e4 result is the solid one.

---

## Venue thinking

- Paper 1 → **JCP**, submitted 2026-08-25. TMLR fallback. CMAME desk-rejected it
  as not-new-methodology; do not resubmit there.
- Paper 2 → **CMAME**. It is new computational methodology — a criterion for
  surrogate architecture derived from a solver-side measurement, a recipe, and a
  certificate — which is exactly what CMAME said Paper 1 lacked. JCP alternative.
- The paper is now **positive-led**: a trained model accelerating a production
  solver on total drag, with a guarantee. The negatives support it rather than
  carrying it, which is the framing that survives review.
