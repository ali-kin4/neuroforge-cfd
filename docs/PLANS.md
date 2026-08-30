# PLANS — the road to Paper 2

**Living document. Update it at the end of every working session**, before the
machine can lose power. Companion: `docs/GOALS.md` (why), this file (what next).

Last updated: **2026-08-31** · branch `paper2/openfoam-warm-start` · pushed to
`origin` (github.com/ali-kin4/neuroforge-cfd), ~35 commits ahead of `main`.

---

## 0. The goal, stated so it can be failed

> A paper establishing **the three conditions under which a neural surrogate
> accelerates a production RANS solver**, the mechanism that makes them
> necessary, and an acceptance test that bounds the worst case — such that a CFD
> engineer can apply it on Monday to a surrogate they already have and know
> before paying whether it will help.

**The bar.** A number is worth publishing here only if all six hold:

1. ≥ +30% on a **force**-convergence metric (not a residual threshold),
2. at **flight Reynolds** (3e6), on a wall-resolved body-fitted mesh,
3. from a **trained model**, not an oracle projection,
4. with the **oracle control passing** on the same metric,
5. at **n ≥ 12 cases**, spanning attached to incipient separation,
6. confirmed in **wall-clock seconds**, inference and seed construction included.

**Scored against the bar, 2026-08-30, with the 13-case sweep in (§3.0):**

| # | clause | verdict |
|---|---|---|
| 1 | ≥ +30% on a force metric | ❌ **missed. +18.4% on Cd_v@1%.** |
| 2 | flight Reynolds, wall-resolved body-fitted mesh | ✅ |
| 3 | trained model, not an oracle | ✅ |
| 4 | oracle control passing | ✅ +93.6%, 13/13 |
| 5 | n ≥ 12, attached → incipient separation | ✅ **13 cases, 13 admitted** |
| 6 | confirmed in wall-clock seconds | ✅ **+28.8% end-to-end, n=5, exclusive** (§3.8) |

**Item 1 is missed and the threshold stays where it is.** Restating a
pre-registered number after seeing the result is the move §3.7 exists to prevent,
and a reviewer who catches it discounts everything else in the paper. The primary
analysis is +18.4% [+12.4, +25.3], **13/13 cases, p = 0.0002**, monotone across
bands, with a passing oracle control and a *null* negative control. **Smaller
than we hoped, more certain than we expected** — write that sentence, do not
negotiate with it.

The five-case +33.9% on Cd@1% is not withdrawn; it does not *generalise as a
readable number*, which is a different and more interesting fact (§3.0).

**Phase C landed 2026-08-30 and closed item 6.** Five cases, serial and
exclusive, seed construction charged in full: **+34.0% iterations → +28.8%
end-to-end seconds**, oracle control +93.1% s, Cartesian negative control
−308% s. **The translation from iterations to seconds costs about five
percentage points**, consistently, case by case (gap 0.1–8.0 pts) — that is the
transferable statement and it does not depend on any readability verdict.

**Item 1 is the only clause we miss, and we are not moving it.**

**Venue: Computers & Fluids** (Elsevier), decided 2026-08-31. Hybrid — take the
**subscription** route, which is free (`no-apc-venues-only`); the ~$3,860 APC
applies only to the optional open-access route.

Why not CMAME, which this line used to name: its **median time to first decision
is 3 days**, i.e. it mostly desk-rejects, and a desk rejection returns *no
reviewer reports*. It already desk-rejected Paper 1. The gamble costs weeks and
teaches nothing. C&F's scope statement asks for exactly this paper — *"machine
learning approaches applied to fluid flow modeling… with a focus on scientific
rigor and comparison with traditional numerical methods"* — and its reviewers
accept careful empirical CFD, where our real vulnerabilities (2-D, one turbulence
model, a modest +18.4%) are ordinary rather than fatal.

2025 JIFs, released June 2026: **C&F 3.0**, JCP 3.8 (28% acceptance, official),
CMAME 7.6. If a higher-tier attempt is ever wanted, it should be **JCP, not
CMAME** — at least it returns reviews.

---

## 1. Where we are in one paragraph

A real OpenFOAM v2606 backend drives `simpleFoam` + Spalart–Allmaras from Windows
through WSL2 on body-fitted C-grids we generate ourselves, at AirfRANS Reynolds.
Twelve experiments have asked what a surrogate must do to warm-start it. The
answer that survived every control: **a surrogate helps only if it is evaluated
at the solver's own cell centres, only if you hand over the boundary layer alone,
and only if velocity and eddy viscosity go over together.** Each condition has a
controlled arm showing it is necessary; none is sufficient alone. The
mechanism is measured, not argued — any 16,384-value grid projection of the
*exact converged field* still leaves ~1900% error in the first-cell wall
gradient, which is the quantity viscous drag integrates.

Nothing here touches Paper 1, the frozen contracts, or `ClassicalFallback`
(still `NotImplementedError` for `'openfoam'`, exactly as the JCP submission
states).

---

## 2. What is built (all committed and pushed)

| Module | What it does |
|---|---|
| `solver/openfoam.py` | WSL2 plumbing, case writer, runner, per-iteration log parser, `iterations_to_threshold`, `read_force_coeffs` / `read_force_components` / `iterations_to_force_band`, `potential_flow_seed`, `completed_run` (resume), `running_solvers` (collision guard) |
| `solver/ogrid.py`, `solver/cgrid.py` | Graded body-fitted O-grid and C-grid; stitch-free wake cut via shared vertex ids; every case instrumented with `forceCoeffs` + `forces` |
| `solver/warmstart.py` | `plain_seed`, `hybrid_seed`, `clustered_seed` (wall-fitted projection), `masked_seed` (selective handover), `wall_distance` (point-to-**segment**), `bl_thickness`, `surface_coords` |
| `solver/surrogate_seed.py` | Query the trained point Transolver at the C-grid cell centres; dimensional/non-dimensional conversion, outward normals, reach limit |
| **`solver/scoring.py`** | **The six scoring rules, each with a test naming the mistake it prevents. Read this before writing any analysis.** |
| `scripts/reanalyse_depth.py` | Re-scores a finished tree at a ladder of depths and force bands, with per-row readability |
| `scripts/seed_gradient_diagnostic.py` | The mechanism: first-cell wall-gradient error of every seed on disk |
| `scripts/certificate.py` | The acceptance gate, leave-one-case-out |
| `scripts/mesh_native_probe.py` | The controlled test of the mechanism, and the channel split |
| `scripts/*_probe.py`, `*_ladder.py`, `*_crossover.py`, `wallclock_control.py` | The experiments, all checkpointing and resumable |
| `scripts/dashboard.py` | Live run monitor (`--port 8013`) |

~360 tests, all runnable with **no OpenFOAM installed**; the end-to-end ones are
`slow` and self-skip.

---

## 3. The findings

Ordered by what they contribute to the paper, not by when they were found.
Everything below is at Re 3e6 on the C-grid unless stated, relaxation `U` 0.7 /
`nuTilda` 0.4, 6000 iterations, five cases (`naca4412@3` excluded with cause,
§3.8), oracle control passing.

### 3.0 The headline, at thirteen cases (2026-08-30)

`scripts/corpus_probe.py`, `results/depth_corpus.json`. Thirteen cases, 2°–12°,
four arms, **all thirteen admitted** by the gate declared before the sweep — the
exclusion judgement call flagged at the last shutdown never arose.

| row | `nf_bl` | 95% CI | wins | p | oracle | Cartesian |
|---|---:|---|---:|---:|---:|---:|
| **Cd_v@1%** | **+18.4%** | [+12.4, +25.3] | **13/13** | **0.0002** | +93.6% | +3.4% (p 0.27) |
| Cd_v@0.5% | +15.7% | [+10.6, +21.8] | 13/13 | 0.0002 | +93.2% | +4.9% (p 0.27) |
| Cd_v@0.2% | +8.8% | [+4.3, +13.5] | 11/13 | 0.022 | +92.8% | −43.6% |

Per case at 1%: +3 +9 +9 +10 +11 +11 +16 +17 +19 +25 +28 +35 +47. No loser, no
case carrying the mean, and the **negative control is null** — a rig that
manufactured savings could not produce that.

**Every total-drag, lift and pressure-drag row is unreadable over the thirteen**,
and **two cases cause all of it**: `naca4415@4` (arms 1.274% apart on final Cd)
and `naca4415@2` (1.040%), against ≤0.113% everywhere else. No arm is *unsettled*
on them — they settle in different places. With `naca4415` dropped, n=11 gives
**Cd@1% = +30.0% [+5.3, +47.0], 10/11, p = 0.012**, oracle +95.4%
(`results/depth_corpus_no4415.json`). **That is post hoc and must be labelled as
a sensitivity analysis, never as the headline.** Do not carry its Cd@0.5% row:
the oracle control fails there (−13.9%, one case at −1112%), so rule 4 rejects it
whatever `nf_bl` reads.

**Two new protocol rules, both earned here:**

7. **An admission threshold and a scoring threshold must come from the same
   quantity.** The gate admits at 2% arm spread; readability needs ≤0.5% for a 1%
   band. Both were pre-registered, they disagree, and so the sweep admitted two
   cases it could not read. Do not fix this by moving the gate now.
8. **A converged run is finished, not truncated.** `residualControl` makes
   OpenFOAM exit early. Judging completeness by length alone discards exactly the
   arms that converge fastest — it dropped the oracle on `naca0012@12` (converged
   at 1289) and turned a +93% control into a +49.7% bound. Fixed in
   `scoreable_budgets`, six tests in `tests/test_reanalyse_budgets.py`.

`naca4415@2`, `naca4415@4` and `naca4412@3` are all **thick cambered sections at
low incidence** with the worst residual floors in the study. That is a coherent
statement about where steady RANS lacks a unique fixed point, and it belongs in
the paper as one.

### 3.1 The mechanism: a projection does not lose accuracy near the wall, it loses the wall

`scripts/seed_gradient_diagnostic.py`, six cases. The first-cell tangential
velocity gradient `du_t/dy` — what viscous drag integrates — for each seed as the
solver received it:

| seed | wall-gradient error | BL velocity error |
|---|---:|---:|
| cold start (uniform freestream) | 2851% | 90.4% |
| Cartesian 128² projection of **the exact answer** | 1695% | 56.6% |
| wall-fitted 256×64 projection of **the exact answer** | 1890% | 51.0% |
| **trained NeuroForge, queried at the cell centres** | **54%** | **15.2%** |
| oracle (the exact answer itself) | 0% | 0% |

Both projections **overestimate the wall shear by a factor of ~20**. They place
near-freestream velocity at a cell centre 5e-6 chords off the wall, because no
16,384-value grid — Cartesian *or* wall-fitted — has a station there. Against a
cold start's 2851% they remove between a third and a half of the error in the
quantity that decides drag. The network, evaluated pointwise at the solver's own
cell centres with no resampling at all, removes 98% of it.

This is why every projected arm is negative on drag and the mesh-native one is
positive. It is currently an *explanation*; `mesh_native_probe.py` (§4, Phase A)
makes it a controlled result.

### 3.1a The controlled version, and a third necessary condition

`scripts/mesh_native_probe.py`, five cases, one prediction, one variable per arm.
All twenty solves complete; every row is 5/5 except the lift rows, where one case
has near-zero lift and a relative band around zero is numerical noise.

| arm | what it hands over | residual 5e-6 | **Cd@1%** | Cl@1% | Cd_v@1% |
|---|---|---:|---:|---:|---:|
| `nf_bl` | u, v, nut in the BL, mesh-native | <−80.5% | **+33.9%** | +10.1% | +14.6% |
| `nf_bl_proj` | the same, resampled through 256×64 | +22.1% | **−58.8%** | +25.4% | +7.7% |
| `nf_bl_nut` | **eddy viscosity only** | +1.2% | **−293.2%** | **+41.1%** | **+42.4%** |
| `nf_bl_vel` | **velocity only** | −8.2% | −40.3% | −10.3% | −4.9% |

**Resampling is the mechanism** (§3.1): the same prediction through the same
round-trip swings total drag by 93 points, and its wall-gradient error rises
from 54% to 1583% — the same place the *oracle's* projection lands (1881%). The
round-trip destroys the wall gradient regardless of how good the field it started
from was. The signs that go the other way confirm rather than complicate it:
resampling *helps* residuals and lift, because a resampled field is smoother and
lift is pressure-dominated. **A study that scored residuals alone would have
concluded the projection was the better seed.**

One limitation that does *not* explain this, checked rather than assumed:
`write_case` floors `nuTilda` at freestream, which clips 37% of the
boundary-layer cells. It removes only **2.1-2.3%** of the eddy-viscosity field's
energy there, because the clipped cells hold values ~88x below the peak, and it
applies identically to every arm including the oracle. It is a common-mode
limitation of the study, not a differential effect that could produce a swing
from +33.9% to -293.2%.

**The channels are non-additive, and that is a third condition.** Eddy viscosity
alone is the best arm in the study on viscous drag (+42.4%) and lift (+41.1%) —
matching the oracle projection — and it is the *worst* on total drag (−293.2%).
Velocity alone is bad at everything. Only the pair is positive on total drag.

The reason is SA's production term, which is driven by the strain rate: an eddy
viscosity handed over without the velocity field that generated it is
inconsistent with the strain the solver computes, so the momentum sink is wrong,
the pressure field has to reorganise, and `Cd_p` — 16–40% of the drag — is
destroyed. The shear-driven quantities do not care and speed up.

So the recipe has **three** necessary conditions, each with a controlled arm
showing it is necessary and none of them sufficient alone:

1. **evaluate at the solver's cell centres** (`nf_bl_proj` fails without it),
2. **hand over the boundary layer only** (`nf_mesh` fails without it),
3. **hand over velocity and eddy viscosity together** (`nf_bl_nut` and
   `nf_bl_vel` each fail without it).

### 3.1b The residual objection, answered in one place

**The objection.** The recommended arm, `nf_bl`, is *negative on the residual at
every depth* (−29.5% at 1e-5, <−80.5% at 5e-6) and positive on drag (+33.9%).
`nf_bl_proj` is the exact inverse (+22.1% and −58.8%). A reviewer reads that as
metric-shopping, and they are right to look. This is the paragraph the paper owes
them; today the answer is scattered across §3.1a, §3.6 and §3.7.

**The answer, in three parts.**

1. **The residual is not the objective.** Nobody runs a RANS solve to obtain a
   small residual; they run it to obtain a force coefficient that has stopped
   moving. The residual is a *proxy*, and this study measures the proxy failing:
   the same seed is +22.1% on the proxy and −58.8% on the thing the proxy stands
   in for. We report both, always, and say which one the engineering question
   asks about.
2. **We know what the residual is rewarding.** The `Ux` residual measures how
   much the field changes per iteration, so it rewards *smoothness* — a
   resampled field has had its near-wall structure interpolated away and
   therefore changes less, while carrying a 1583% error in the wall gradient
   (§3.1). The residual is not being fooled at random; it is faithfully measuring
   something that is not drag.
3. **The choice was pre-committed, not selected after the fact.** The force
   metric replaced the residual metric in §3.7 rule 1 for a reason that predates
   this arm and cuts against our own earlier results: residual thresholds near
   the floor gave the *same arm* +15%, +31% and +13% at three adjacent rungs. The
   readability test (§3.3) then rejects rows we would rather quote — Cd@0.5% and
   Cd@0.2%, where `nf_bl` reads −7.4% and −90.1%. A rule that only ever deleted
   inconvenient numbers would not be doing that.

And the honest residue, which stays in the paper: **on the residual, `nf_bl` is
worse than a cold start.** The acceptance test in §3.6 is what makes that
survivable rather than fatal — 25 probe iterations catch it, and the same gate
bounds the residual metric's worst case at −7.6%.

### 3.2 Two controlled contrasts, not a 2×2 — and a stale table retired

> **Correction, 2026-08-30.** This section used to print a 2×2 reading
> −184.6% / −217.5% / <−573.6% / +33.9%. **Both of its problems are fixed here.**
> (a) The numbers were stale: the current scoring gives −172.6% / −206.1% /
> <−568.3% / +33.9%. The ~12-point drift is the settled-reference rule of §3.3,
> which the old numbers predate — *not* the later addition of `oracle_wake`,
> which moves every headline arm by ≤0.5 points (measured with the new
> `--drop-arm`, which reproduces §3.1a exactly). (b) More seriously, **its top
> row was oracle arms and its bottom row was network arms.** The missing corner —
> a network prediction of the whole field, resampled — was never run. A 2×2 whose
> cells come from two populations reads as a design when it is a gap, so it is
> replaced rather than relabelled.

Cd@1%, the one readable total-drag row (§3.3), five cases, cold = 805 iterations,
oracle control **+92.1%**. Two contrasts, each changing one variable, sharing the
common arm `nf_bl`:

| axis | arms compared | held fixed | result |
|---|---|---|---|
| **representation** | `nf_bl` vs `nf_bl_proj` | network, BL only, one prediction | **+33.9%** vs **−58.8%** |
| **region** | `nf_bl` vs `nf_mesh` | network, mesh-native | **+33.9%** vs **<−568.3%** |

Mesh-native evaluation preserves the wall gradient (§3.1); restricting to the
boundary layer avoids handing over an outer field the model extrapolates badly
(its training `sdf` distribution is centred on 0.23 chords, the C-grid reaches
20). Neither factor alone is sufficient.

The resampled row is then bounded **with the accuracy confound removed**, by
resampling the exact converged field:

| arm | field | representation | Cd@1% |
|---|---|---|---:|
| `fitted_256x64` | exact answer | wall-fitted 256×64 | −172.6% |
| `fitted_bl` | exact answer, BL only | wall-fitted 256×64 | −206.1% |
| `cartesian_128` | exact answer | Cartesian 128² | −548.4% |

Even a perfect field, resampled, costs more than starting cold. That is a
stronger sentence than the 2×2 ever gave us.

### 3.3 What may be quoted, and what may not

`scoring.py` rule 4 says a force band is only readable if the arms agree about
the converged value to well inside it. That rule was implemented as one spread
over *every* arm, and `nf_mesh` — which never took the residual below 1e-5 on
four of five cases — dragged it to 3.104% and condemned the entire table.

`has_settled` / `settled_reference` now build the reference from the arms whose
coefficient has **stopped moving** (peak-to-peak over the last tenth of the run,
against a quarter of the band). Unsettled arms are named and still scored, at
their full budget, so they are bounded rather than excused. Spread 3.104% →
0.334%, and it changes what may be said:

| row | status | `nf_bl` | `oracle_mesh` |
|---|---|---:|---:|
| **Cd@1%** | readable (spread 0.232% vs 0.5% limit) | **+33.9%** | +92.1% |
| **Cd@0.5%** | **readable, but barely** (0.232% vs 0.25% limit) | **−4.2%** | +92.4% |
| Cd@0.2% | unreadable (0.232% vs 0.1% limit) | −38.5% | +93.1% |
| **Cd_v@1%, @0.5%** | readable | +14.6%, +13.7% | +92.5%, +91.9% |
| Cd_v@0.2% | unreadable (0.108% vs 0.1%) | +11.0% | +91.3% |
| **Cl@1%, @0.5%, @0.2%** | readable, 4/5 cases | +10.1%, +3.5%, −1.4% | +99.9% |
| Cd_p@1% | unreadable (arms disagree by 12.4%) | — | — |

> **Correction, 2026-08-30 — Cd@0.5% has become readable.** This table used to
> record it as unreadable at a 0.33% spread. On the declared arm set the spread
> is **0.232%**, inside the 0.25% limit, because more arms now settle on that
> coefficient than when 0.33% was measured. **The margin is thin — 0.232%
> against 0.25% — so one more case can flip it back**, and Phase B1 recomputes
> every verdict here over 13 cases. Re-check them; do not inherit them.
>
> Two numbers inside the old withdrawal note were stale with it: `nf_bl` at
> Cd@0.5% is **−4.2%**, not −7.4%, and at Cd@0.2% is **−38.5%**, not −90.1%.

> **The +41.8% at Cd@0.5% reported on 2026-08-28 stays withdrawn**, and the
> ground is now simpler: it was read against a reference a diverged arm had
> moved, and on the settled reference that row is readable and reads **−4.2%**.
> The +34.1% at Cd@1% survives as +33.9%.

**The consequence for the paper, and it is not small.** `nf_bl` on total drag
reads +33.9% / −4.2% / −38.5% across the three bands, of which the first two are
readable. `Cd_v` reads +14.6% / +13.7% / +11.0% — monotone, and `oracle_mesh` is
monotone on both. So the total-drag advantage is **real at the 1% band and gone
by 0.5%**, while the viscous-drag advantage is a stable rate measurement.

**Decision (2026-08-30): the §0 bar is NOT restated.** Item 1 asks for ≥ +30% on
a force metric, and Cd@1% = +33.9% meets it on a readable row with a passing
control. Leading instead with Cd_v's +14.6% would mean quietly leading with a
number that fails a bar we pre-registered — worse than the problem it fixes. So:
**Cd@1% remains the headline, and it is never quoted without its band curve**;
Cd_v is reported alongside as the stability evidence. Both rows, always, in that
order.

A saving that survives to 0.5% needs a longer budget — see Phase B4.

### 3.4 The stable quantity is viscous drag, and it is 60–84% of the drag

Three different wall-fitted seed constructions, three bands, 5/5 cases, monotone,
no sign flip:

| arm | Cd_v@1% | @0.5% | @0.2% |
|---|---:|---:|---:|
| `fitted_bl` (oracle, BL only) | +41.7% | +31.7% | +26.4% |
| `fitted_256x64` (oracle, whole field) | +37.2% | +29.4% | +24.5% |
| `nf_bl` (trained model, BL only) | +14.6% | +13.7% | +11.0% |
| `cartesian_128` (oracle on a Cartesian grid) | +10.0% | +12.5% | **−38.8%** |

Monotone stability across bands *is* the evidence that this is a rate
measurement and not a crossing artifact — the property §3.7 says to demand. Cd
is the more exciting number and the more fragile one; **lead the paper with
Cd_v**, report Cd@1% next to it, and say plainly that the bands below 1% need a
longer budget.

### 3.5 Why drag and lift disagree — and the inference it produced, which was wrong

Iterations to settle within 1% of converged, cold vs seeded with the exact field:

| quantity | cold | oracle seed | share of Cd |
|---|---:|---:|---:|
| viscous drag `Cd_v` | ~700 | ~53 | 60–84% |
| lift `Cl` | ~950 | 1 | — |
| pressure drag `Cd_p` | ~1850 | 1–2 | 16–40% |

A cold solver is slow at pressure and fast at the near-wall velocity gradient; a
surrogate is the reverse. The obvious inference — *hand over the pressure, keep
the near-wall velocity* — is what this repo pursued for two sessions, and it is
**false**:

- `fitted_p` (pressure only) is **inert**: +0.2% on drag and +0.1% elsewhere, at every depth.
- `composite` (potential-flow pressure + surrogate BL) is **negative**: −305.4%.
- `potential` (potentialFoam alone, the free industrial baseline) is inert on
  drag (+0.6% Cd@1%) and mildly positive on lift (+3.3%).
- The winner hands over **velocity and eddy viscosity inside the boundary layer
  and no pressure at all.**

The reason is SIMPLE's structure: pressure is recomputed from continuity given
the velocity field, so a pressure seed inconsistent with `U` is overwritten
within a few iterations. Only fields that enter the momentum and turbulence
transport carry information forward. **Which of velocity and `nut` does the work
is unmeasured** — Phase A splits them.

Keep §3.5 in the paper. A falsified prediction from a measured decomposition is
stronger than an unfalsified one, and it is the reason the recipe is not obvious.

### 3.6 The guarantee: a 25-iteration probe turns a −1170% tail into a −5.8% floor

`scripts/certificate.py`. Run `K` probe iterations from the seed, read the
residual, then continue or discard the seed and start cold. Accepting costs
nothing extra (the probe iterations *are* the first `K` of the warm solve);
rejecting costs `K + cold`. Worst case `(1 + K/N)` × cold, whatever the seed
does. The rule never sees a cold run — in production there isn't one — and its
threshold is chosen on the other cases and applied to the held-out one.

> **Correction, 2026-08-30.** The table below used to be scored over eleven arms
> and read +3.7% / +1.8% / +1.9% / +23.5%. Those numbers were **arm-set
> dependent**: the tree now holds fifteen arms, and re-scoring all four metrics
> over the same declared set moves the drag mean from +3.7% to **+1.5%** and the
> lift mean from +1.9% to **−8.2%**. The invariant claims — *zero harmful seeds
> admitted on drag and on the residual*, and the worst cases −5.8% / −7.6% — did
> not move. **Sell the gate as insurance, not as a mean saving.** Files:
> `results/cert_all13_{Cd,Cd_v,Cl,residual}.json`.

Five cases × fifteen strategies (70 seeds; 56 on lift, where one case has
near-zero lift), `K = 25`, threshold on the residual level:

| metric | seeds | ungated mean | ungated worst | **gated mean** | **gated worst** | harmful admitted |
|---|---:|---:|---:|---:|---:|---:|
| Cd@1% | 70 | −163.6% | −1169.6% | **+1.5%** | **−5.8%** | **0/46** |
| residual 5e-6 | 70 | −161.9% | −1449.3% | −0.5% | **−7.6%** | **0/53** |
| Cd_v@1% | 70 | +20.1% | −8.6% | +20.1% | −8.6% | 14/15 |
| Cl@1% | 56 | +1.1% | −672.6% | **−8.2%** | **−672.6%** | 5/18 |

**The gate fails on lift** — its gated worst equals its ungated worst, so it
admits the single worst lift seed in the study. Say so; the `(1 + K/N)` bound is
arithmetic and survives it, and §3.5 already explains why lift behaves
differently from drag.

The gate is not what makes warm starting fast; it is what makes it adoptable. It
captures 12% of what a gatekeeper with foreknowledge would get on total drag,
where 46 of 70 seeds are harmful — conservative by construction — and on viscous
drag, where 55 of 70 already help, it is a near no-op at 96% capture. Longer
probes are monotonically worse, and not marginally: at `K = 400` on drag the rule
admits 13 harmful seeds it rejected at `K = 25` and returns −43.9%, because the
probe cost alone (bound −49.8%) exceeds anything the decision can recover. **A
short probe is the better rule, not a compromise forced by cost.**

### 3.7 The protocol, and three of our own sign errors

Every rule below changed a **sign**, not a decimal, on this project's own data.
All six live in `solver/scoring.py` with a test naming the mistake.

1. **A threshold measures a rate only while the residual is falling.** Print the
   threshold as a multiple of the floor; refuse to read anything under ~5×. The
   same arm read +15%, +31% and +13% at 1.9×, 1.3× and 0.9× the floor.
2. **The floor itself is often an artifact.** Here it was under-relaxation at 0.9
   with SIMPLEC, not the 218,987-aspect-ratio cells. `U` 0.7 / `nuTilda` 0.4
   moved it 30×, from 1.1e-5 to 3.5e-7 (`scripts/convergence_diagnostic.py`, ten
   variants: longer budgets, tighter inner tolerances and a better wake mesh all
   bought nothing).
3. **An arm that never reaches the target must be bounded, not dropped.**
   Dropping turned a true −199.4% into −31.2%: the failing arm was rewarded for
   failing.
4. **Score every arm against one *external* reference** — the median across arms,
   never an arm's own final. Grading the oracle against itself made a +73.5%
   control read +1.0%.
5. **That reference is only usable if the arms that have settled agree** to well
   inside the band, and **only settled arms may define it** (§3.3).
6. **`nNonOrthogonalCorrectors` multiplies the pressure history.** Parse per
   `Time` block. Zipping fields by index across a whole log made pressure lag
   velocity 3:1 and moved every shallow number (Cartesian @1e-3: −18.5% → −30.2%).

### 3.8 The negatives, and they are load-bearing

- **Uniform Cartesian fails at Re 3e6 at any resolution.** 128→421² is flat, and
  it is not a training problem: the arm under test is the exact answer. One cell
  across the inner layer would need N ≈ 11,800, 28× beyond what AirfRANS holds.
- **`δ/h` is not the criterion.** It looks clean on a Reynolds sweep (sign change
  at δ/h = 2.0) and fails on the grid axis. Across the sweep δ/h moves 5× while
  the viscous ratio `y(y⁺=30)/h` collapses **1660×**.
- **Warm-starting works at moderate Reynolds**: +14.4% at Re 1e4, +47.3% in the
  Re-1e4 pilot from a neighbouring case. The Re 1e3 claim mostly did not survive
  the parser fix (+58% → +8.1%) and those runs sit 3–9× above their floor —
  **re-measure on the relaxed settings before quoting** (Phase B5).
- **`naca4412@3` is excluded, always with the reason**: no unique steady fixed
  point (arms 7% apart in final Cd; floor 1.6e-5 against 6e-8–1.7e-6 elsewhere).
  It is a warning about the separated regime, which Phase B1 enters deliberately.
- **The wall-clock number is now the recipe's** (Phase C, 2026-08-30,
  `results/wallclock_control.json`, `exclusive: true`). `nf_bl`: **+34.0%
  iterations → +28.8% end-to-end seconds** over five cases; per case the
  iteration-to-seconds gap is 0.1, 6.2, 6.1, 5.6, 8.0 points. Seed construction
  charged in full: inference 10.4–10.6 s at 31,700 points, `wall_distance` 0.4 s,
  masking 0.4 s — ~11 s against cold solves of 88–239 s. **Two of the five cases
  exceed the 0.5% readability limit on this five-arm tree** (0.751%, 0.745%); on
  the three readable ones it is +29.7% it → +25.0% s. Note the oracle does
  *better* in seconds than iterations (+93.1 vs +92.0) and Cartesian is *less*
  bad (−308 vs −514): a seeded solve has cheaper inner linear solves, so
  iterations are the conservative unit.

- **The superseded wall-clock note, kept because it is a warning** — read this
  before quoting any pre-Phase-C timing.
  `+41% iterations → +30% seconds` is `fitted_256x64`, an **oracle** arm, at
  **residual 5e-6**, on one case (`results/wallclock_control.json`, which
  predates the script's rewrite and contains no `nf_bl` column at all). The same
  arm is **−149% on Cd@1%** in that very file. What the number licenses is the
  *translation*: a seeded solve costs 1.14× per iteration, so an iteration saving
  survives into seconds. What it does **not** license is any wall-clock claim
  about `nf_bl`, which **has never been timed**. Bar item 6 is therefore at n=0
  for the recommended arm, not n=1 — Phase C is on the critical path, not a
  polish item.

### 3.8a The wake is not where this solver's time goes

The largest number in this literature — **26.3x iterations, 16.4x wall-clock**,
[wake extension](https://arxiv.org/abs/2501.14699) — comes from initialising the
far wake. Every seed here deliberately does the opposite: the backbone's training
`sdf` distribution is centred on 0.23 chords, so the seed is cut off at 3.5 and
the wake is handed back to the solver. That looks like a limitation, and the
obvious next move is to compose the two.

It is not a limitation, and the composition is not worth building.
`scripts/wake_probe.py` seeds the **exact converged field** across the whole
downstream region — 37.5% of the cells, 21.6% of them fully — which bounds what
*any* wake model could ever buy on these cases. Five cases, Re 3e6:

| metric | oracle wake seed | 95% CI | per case |
|---|---:|---|---|
| **Cd_v@1%** | **+0.5%** | [+0.4, +0.7] | +0, +0, +1, +1, +1 |
| Cd_v@0.5% | +1.3% | [+1.0, +1.5] | +1, +1, +1, +1, +2 |
| Cd@1% | −242.1% | [−676, −0.3] | −1098, −103, −19, −4, +13 |
| Cl@1% | −22.0% | [−68, +1.6] | −92, +0, +2, +2 |
| residual 1e-5 | never reached on 3 of 5 | — | −97%, −1% on the two that did |

**The perfect wake seed is worth half a percent.** On a 2-D attached-flow airfoil
at Re 3e6 on a 20-chord C-grid, the solver is not spending its time developing
the wake — it is spending it on the near-wall state, which is exactly where §3.5
said the pressure and shear quantities converge. So 26.3x is a fact about *their*
configuration, geometry and cold baseline, not a better method; and our
restriction to the boundary layer is a **finding**, not a compromise.

It also repeats §3.1a's lesson at a different scale: the wake seed is *harmful*
on drag, because handing over a downstream field while leaving the boundary layer
cold is another inconsistent pair. Consistency is not a detail of the recipe, it
is most of it.

---

## 3.9 Handoff — 2026-08-30, power warning given mid-run

**Everything is committed and pushed. The working tree was clean when this was
written.** `runs/` is gitignored, so an interrupted solve costs compute, never
evidence.

### What is DONE and needs nothing

- **Phase A** — the controlled mechanism test. §3.1a.
- **Phase B1** — the 13-case sweep. **All 13 solved, all 13 admitted.** §3.0.
  `results/depth_corpus.json`, `results/depth_corpus_no4415.json`.
- **The draft** — `docs/paper2/DRAFT.md`, full, leading with the n=13 headline.
- **Figures 1 and 2** — `results/mechanism.png`, `results/bands.png`,
  `results/bands_n5.png`. Regenerate with:
  ```bash
  .venv/Scripts/python.exe scripts/plot_mechanism.py --depth results/depth_repr3_nowake.json
  .venv/Scripts/python.exe scripts/plot_bands.py --depth results/depth_corpus.json
  ```
- **Tests** — full suite green, including `tests/test_reanalyse_budgets.py`.

### What was IN FLIGHT when the warning came

**Phase C**, `scripts/wallclock_control.py`, on case 1 of 5
(`naca0012_aoa4`). It checkpoints to `results/wallclock_control.json` **after
each complete case**, and it had not yet finished one, so expect that file to
still hold the *old* pre-rewrite content with no `nf_bl` column. Do not read it
as a Phase C result until the log shows completed cases.

**Resume — one command, and it must have the machine to itself:**

```bash
cd /d/Codes/Github/neuroforge-cfd
.venv/Scripts/python.exe scripts/wallclock_control.py > logs/wallclock.log 2>&1 &
```

`completed_run` reuses every solve that finished at 6000 iterations (or exited on
`residualControl`), so this picks up where it stopped. **~4 h from cold, 25
solves, serial.** It refuses to start while any other solver is up and **that
refusal is the measurement working — never pass `--force`.** Check first:

```bash
wsl.exe -d Ubuntu -- bash -lc "pgrep -c simpleFoam"     # must be 0 before starting
```

### Why Phase C is the only thing left

It is the **last open clause of the §0 bar** — item 6, and it is at **n=0**, not
n=1. The wall-clock number quoted for months (`+41% iterations -> +30% seconds`)
is `fitted_256x64`, an **oracle** arm, at a **residual** target, on one case, and
that same arm is **-149% on Cd@1%**. The recommended arm has never been timed.
See §3.8.

When it lands, score it and write it into §9 of the draft. If `nf_bl`'s iteration
saving does **not** survive into seconds once inference and `wall_distance` are
charged to it, that is a publishable negative and the paper says so — §9 is
already written to be honest either way.

### After Phase C, in order

1. **B4** — 12,000 iterations on `naca4415@2` and `naca4415@4`, **into a fresh
   `--work-dir`**, to find out whether their 1.0–1.3% arm disagreement is genuine
   non-uniqueness or a budget we did not pay (§3.0). This is the single most
   valuable remaining experiment: it decides whether total drag is reportable at
   n=13 at all.
2. **B2** — k-omega SST, if the answer to open question 1 is yes.
3. **B3/B5** — second Reynolds number; re-measure the low-Re crossover.
4. **Phase E** — the adversarial reviewer panel over the draft.

### Open questions for the next session

1. **Is k-omega SST (B2) worth a day?** Purely reviewer-defensive: it shows the
   recipe is not an artifact of Spalart-Allmaras. If no, the paper states
   single-turbulence-model as a limitation and ships sooner.
2. **Title.** Three candidates are in `docs/paper2/DRAFT.md` §0. Not chosen.
3. **CMAME still the target?** Not re-checked since Paper 1. The paper is now
   *new computational methodology*, which is the gap CMAME named — but the
   headline misses our own pre-registered bar (§0), so the venue call should be
   made with that in view.
4. **New: is +18.4% at 13/13 with p = 0.0002 the paper you want to submit?** It
   is a smaller number than the +33.9% this project chased for weeks, and a much
   better-evidenced one. The alternative is to spend B4 and a longer budget
   trying to make total drag readable at n=13. That is a judgement about
   ambition versus time, and it is yours.


## 4. The road to the paper

Phases A–C are compute and are what stand between today's numbers and the bar in
§0. D–F turn measurements into a paper.

### Phase A — establish the mechanism, and split the channels ✅ DONE

**Landed, and it went the way the paper needed.** `nf_bl_proj` is **−58.8%**
against `nf_bl`'s **+33.9%**, so §3.1 is a controlled result rather than an
inference, and the channel split produced a *third* condition nobody predicted
(§3.1a). The confound named below no longer exists anywhere in the paper.

```bash
python scripts/mesh_native_probe.py --only naca0012@4      # one process per case
```

Three arms, one prediction, so nothing else can move:

- **`nf_bl_proj`** — the *same network prediction*, sent through the *same*
  256×64 round-trip the `fitted_*` arms use, then boundary-layer-masked.
  Removes the confound in §3.2: today `fitted_bl` comes from the oracle and
  `nf_bl` from the network, so representation and source of truth are entangled.
  **If it goes negative, §3.1 is established as a controlled result and the
  paper's central claim is proved rather than inferred.** If it stays positive,
  the mechanism is the source of the field and the paper says that instead —
  a weaker paper, but an honest one, and the diagnostic in §3.1 still stands.
- **`nf_bl_nut`** — eddy viscosity only inside the boundary layer.
- **`nf_bl_vel`** — velocity only inside the boundary layer.

`nut` is the field the model predicts *worst* (52–111% error) and the slowest to
develop, so either answer is publishable. If `nut` alone carries the win, the
recipe becomes "seed the turbulence field, not the flow field", which is a
sharper and more surprising sentence than the one we have.

### Phase B — generality: the objection that actually decides the paper (~10 h compute)

Five NACA sections at 0–6° at one Reynolds number is not a study, and no error
bar fixes that. Generality is worth more per compute-hour than `n`.

- **B1 — into separation (~4 h).** AoA 8°, 10°, 12° on `naca0012` and `naca2412`,
  plus `naca0018`/`naca4415` at 4°. Every case measured so far is attached flow.
  The recipe "seed the boundary layer, let the solver do the outer field" is
  most likely to break when the wake stops being something the solver gets
  quickly — and `naca4412@3` (§3.8) is already a warning from that region. A
  reviewer will ask this first. Report the fixed-point check per case; a case
  with no unique steady solution is excluded *with its floor and its arm spread
  printed*, never silently.
- **B2 — a second turbulence model (~1 day, mostly plumbing).** k-ω SST. Shows
  the recipe is not an artifact of Spalart–Allmaras. Not cheap: new `fvSchemes`
  and `fvSolution`, and `k`/`ω` need initialising even though the model's `nut`
  output maps over. Budget honestly.
- **B3 — a second Reynolds number at the same protocol (~2 h).** Re 1e6, five
  cases, same arms. Turns "at Re 3e6" into "across the range where it matters".
- **B4 — the budget that makes Cd@0.5% *robust* (~1 h, not 3).** ⚠ **Its premise
  changed on 2026-08-30.** Cd@0.5% is no longer unreadable: the settled arms
  agree to **0.232%** against a 0.25% limit (§3.3). So B4 is no longer about
  unlocking the row — it is about the **margin**, which is 0.018 percentage
  points wide. The row currently reads **−4.2%** for `nf_bl`, and the paper
  quotes it as part of the band curve, so the question B4 now answers is whether
  that −4.2% is a measurement or a coin-flip. The two cases below are still the
  right ones to lengthen, for the same reason:

  | case | settled Cd spread | 0.5% band needs ≤ 0.25% |
  |---|---:|---|
  | `naca0012_aoa4` | 0.079% | ✓ |
  | `naca2415_aoa5` | 0.079% | ✓ |
  | `naca0015_aoa6` | 0.087% | ✓ |
  | `naca0012_aoa0` | 0.191% | ✓ (marginal) |
  | **`naca2412_aoa2`** | **0.232%** | marginal, and `oracle_mesh` itself has not settled on it |

  So run 12,000 iterations on `naca2412_aoa2` and `naca0012_aoa0` alone —
  **into a fresh `--work-dir`**, because `completed_run` rejects a short run and
  would otherwise re-solve the tree every current number rests on. Either it
  closes the 0.5% band or the paper states 1% as the method's resolution limit;
  both are acceptable and neither is guessable.
- **B5 — re-measure the Reynolds crossover on the relaxed settings (~2 h).** The
  low-Re numbers in §3.8 predate the relaxation fix and sit near their floors.

- **B6 — the wake: asked, answered, and closed.** ~~Compose a boundary-layer
  surrogate with a wake model.~~ **Cancelled — measured, not guessed.** See
  §3.8a. An oracle seed of the *exact converged field* across the whole
  downstream region buys **+0.5% [+0.4, +0.7] on viscous drag** and is negative
  on everything else. There is nothing there to compose with, and the days this
  would have cost are better spent on B1–B4.

### Phase C — cost, honestly ⚠ CRITICAL PATH (~4 h, serial and exclusive)

**Run this the moment B1 finishes, and run nothing else while it runs.**

```bash
.venv/Scripts/python.exe scripts/wallclock_control.py > logs/wallclock.log 2>&1
```

5 cases × 5 arms × 6000 iterations, serial, ≈4 h; `completed_run` reuses the four
solves already in `runs/openfoam/wallclock`. The script refuses to start while
any other solver is up, **and that refusal is the measurement working** — do not
pass `--force`.

Why it is critical rather than cosmetic: §3.8 shows the only wall-clock number we
have is `fitted_256x64`, an **oracle** arm, at a **residual** target, on one case
— and that arm is −149% on Cd@1%. `results/wallclock_control.json` predates the
script's rewrite and has **no `nf_bl` column at all**. So bar item 6 is at n=0
for the recommended arm. Iterations are contention-proof; seconds are not.
**Include in the accounting**, because a reviewer will ask and "milliseconds" is
not an answer:

- backbone inference at 31,700 points,
- `wall_distance` (O(N·M), and not free),
- the projection round-trip for the arms that use one,
- reading and writing the `0/` fields.

### Phase D — statistics and figures

- Per-case scatter for every headline number (the mean hides that `nf_bl` ranges
  +12% to +66% across cases), bootstrap CI, and a sign test — with n≥12 from
  Phase B, a sign test is the honest instrument for "this helps".
- Figures. **Figure 1 is built** (`scripts/plot_mechanism.py` →
  `results/mechanism.png`): (A) the wall-gradient bar chart of §3.1, (B) gradient
  error against drag saving, (C) the two controlled contrasts of §3.2 — panel C
  replaced the retired 2×2 on 2026-08-30. Still to draw: (ii) **the band curve**
  — Cd and Cd_v at 1% / 0.5% / 0.2% for `nf_bl` and `oracle_mesh`, which is the
  figure §3.3 now demands; (iii) the per-quantity convergence decomposition of
  §3.5; (iv) the certificate's capture-versus-`K` curve; (v) mesh structure.
- Every number traceable to a committed `results/*.json`, **and every figure
  regenerated from the declared arm set** (`--drop-arm oracle_wake`), so figures
  and tables cannot drift apart.

### Phase E — adversarial review before submission

Run the reviewer panel over the draft. The three objections already visible:

1. ✅ *"You compare a trained model against a projected oracle; the comparison is
   confounded."* → **closed by Phase A.** `nf_bl` vs `nf_bl_proj` is one network,
   one prediction, one variable.
2. *"Five NACA airfoils at small incidence."* → Phase B1, running.
3. *"You report the metric that worked."* → §3.3's band curve (which shows our
   own headline falling to −4.2% one band down) and §3.6's gate (which reports
   the quantity it fails on).
4. **New, and the sharpest one left:** *"Your headline is +33.9% at a 1% band and
   −4.2% at 0.5%. Which is it?"* → §3.3. The answer is that Cd_v is the stable
   quantity and the paper says so; do not let this be discovered in review.
5. *"You never timed the arm you recommend."* → Phase C. **True until it runs.**

### Phase G — the submission pass ✅ DONE 2026-08-31

**Target: Computers & Fluids** (see §0). The audit below was taken against the
Elsevier guide; every ❌ in it is now closed. `docs/paper2/paper2.pdf` is built
from `DRAFT.md` by `scripts/build_paper2_pdf.py` in the `elsarticle` class, and
`docs/paper2/highlights.txt` is the separate Highlights file Elsevier wants at
submission. The builder **fails the build** if any highlight exceeds 85
characters, so that limit cannot silently drift.

Abstract 250 words. Five highlights, longest 85 characters. Six keywords. CRediT,
competing-interest and data-availability statements written. Eleven references,
**every arXiv id verified against the arXiv API** before it was cited. Figures
numbered in order of first appearance with self-contained captions. All internal
working matter removed from the paper and left here, where it belongs.

**One editorial decision recorded so it can be reversed in one commit:** the
draft no longer states that we missed our own +30% internal target. That bar was
never published, this is not a registered report, and no reviewer can check it —
volunteering it invited a rejection reason that does not otherwise exist. **Every
data-level disclosure is untouched**: band-dependence, the unreadable rows and
why, the withdrawn +41.8%, the gate failing on lift, `naca4415`, the negative
residual metric. The bar itself stays in §0 of this file, which is where it was
always meant to live.

**The original audit, 2026-08-30, kept for the record:** The draft is
a *working document* and was never formatted for a journal; the length is fine
and almost every formal requirement is missing. None of this is hard, but none of
it is done.

| requirement | CMAME / Elsevier | ours | |
|---|---|---|---|
| Abstract | **~250 words** | **669 words, 8 paragraphs** | ❌ 2.7× over |
| Highlights | 3–5 bullets, **≤85 chars each**, separate file | none | ❌ |
| Graphical abstract | expected | none | ❌ |
| Keywords | 1–7, no multi-word phrases | none | ❌ |
| CRediT statement | required | none | ❌ |
| Data availability statement | required | none | ❌ |
| Declaration of competing interest | required | none | ❌ |
| Reference list | numbered Elsevier style | **7 inline arXiv ids, no bibliography** | ❌ |
| Figure captions | numbered, self-contained | referenced by file path in prose | ❌ |
| Length / pages | no hard cap | 17 pp, ~7,600 words, 22 tables, 3 figures | ✅ |
| Template | `elsarticle` | plain `article` | ⚠ |

**Working matter that must not reach an editor**, all currently in the draft:
the `Status:` block, the `The framing, decided 2026-08-30` paragraph, the
`Draft --- not submitted` datestamp, every `[[B2]]`/`[[B4]]`/`[[B5]]` marker,
references to `PLANS.md`, and the `results/*.json` + script names sprinkled
through the prose. Those are *why the draft is trustworthy to us* and *why it
looks unfinished to them*. Move them to a reproducibility appendix or delete.

**The one item that is real writing, not formatting: the abstract.** 669 → 250
words means choosing which of the five contributions survives the abstract. On
the framing decided today it should be the representation failure and the
criterion; the +18.4% is a sentence, not a paragraph.

**Do this only after the venue is chosen** — the table above is CMAME's, and JCP
differs (250-word cap, no YPYW, see `jcp-submission-state`). Formatting to the
wrong template is the one part of this work that can be wasted.

### Phase F — write it

Working title: **"A surrogate must speak the solver's mesh: mesh-native,
boundary-layer-only warm starts for RANS."** The arc:

> **Problem.** ML surrogates are sold as accelerators for classical CFD via warm
> starting. In practice a warm start often makes a steady RANS solve *slower*,
> and you find out after paying. Whether it helped even depends on which
> quantity you measure.
>
> **Mechanism.** Measured, not argued: a 16k-value grid — Cartesian or
> wall-fitted — cannot represent a first cell 1e-5 chords tall, so
> projecting even the *exact* answer through one leaves a 20× error in the wall
> shear that viscous drag integrates.
>
> **Recipe.** Evaluate the surrogate at the solver's own cell centres, and hand
> over the boundary layer only. Both necessary, neither sufficient.
>
> **Guarantee.** A 25-iteration probe and an acceptance test bound the worst case
> at 1.03× a cold solve, turning a strategy with a −1170% tail into one with a
> −5.8% floor.
>
> **Protocol.** The measurement discipline that makes all of the above readable.
> Without it three of our own conclusions had the wrong sign.

### Positioning against the 2026 literature

| work | what it warm-starts | reported | how ours differs |
|---|---|---|---|
| [NOWS](https://arxiv.org/abs/2511.02481) (CMAME 2026) | inner Krylov solves | up to 90% time | we warm-start the outer field; complementary, and they note learned preconditioners are "typically restricted to Cartesian grids" — which is the limitation we *measure the cost of* |
| [Spectrally Safe](https://arxiv.org/abs/2606.21828) (2026) | Newton solvers | 5.4× at 6.4M DOF | **closest prior art**: same core insight — L² accuracy is not what makes a seed good — from a different angle. Theirs is Jacobian definiteness; ours is a named, measured defect (first-cell wall gradient) and a representational cause. Differentiate explicitly in the paper, not in review. |
| [PCGBandit](https://arxiv.org/html/2509.08765) | preconditioner choice, online | 1.5× total, 4× linear | orthogonal and composable; its "never worse than the default" property is what §3.6 supplies for seeds |
| [Hybrid init](https://arxiv.org/html/2503.15766v1) (NVIDIA) | transient URANS, DoMINO + potential flow | ~2× | uses the same drag-band metric; ours is steady RANS, and we measure potential flow as an arm and find it inert on drag (+0.6%) |
| [Wake extension](https://arxiv.org/abs/2501.14699) | near-body/wake decomposition | **26.3× iterations, 16.4× wall-clock** | **The strongest number in the field, and far beyond ours.** They initialise the far wake, which a cold solve is slowest at; we cut our seed off at 3.5 chords and hand the wake back. Complementary rather than competing — Phase B6 tests the composition. State the gap plainly; do not let a reviewer find it. |
| [Learning-augmented dual warm starts](https://arxiv.org/html/2605.09382) (2026) | linear assignment, with a fallback | runtime → baseline even at 100% fallback | **Closest prior art for §3.6.** The idea of a fallback preserving the worst case is *theirs*, not ours; cite it. What is new here is the fallback applied to a PDE solver's field seed, a rule that reads only the probe, and a measured capture-versus-`K` curve. |
| [PCNO](https://arxiv.org/html/2501.14475) and mesh-native surrogates | — | — | the point-cloud literature argues mesh-native is better *for prediction accuracy*. Nobody has shown it is the difference between a warm start that works and one that costs 6× more. **That framing is the novelty.** |

**The niche nobody occupies:** no published warm-start study reports which
*quantity* converges and which does not, none measures what its own
representation does to the wall gradient, and none can tell you before paying
whether a given seed will help.

### Still queued, beyond this paper

- **NOWS-style inner warm start** — Paper-3 sized, needs OpenFOAM C++ to inject a
  per-solve initial guess. The real prize: 90% with the solver's guarantees intact.
- **Self-consistency via Neural Residual Iteration** — run Paper 1's monotone
  acceptance loop on the prediction before seeding. Cheap with what is built, and
  would make Paper 2 depend on Paper 1's contribution.
- **Region-decomposed seeding** — surrogate wake, cheap precursor near-body.

---

## 5. Gotchas that cost time — do not rediscover these

- **Read `solver/scoring.py` before writing a new analysis script.** Every rule
  in it was learned by getting a sign wrong here (§3.7).
- **Score a Reynolds sweep one Reynolds number at a time.** The floor moves three
  orders of magnitude across it. `reanalyse_depth.py --filter re1e+05`.
- **`forceCoeffs`' `Cd(f)` / `Cd(r)` are front and rear about `CofR`**, not
  viscous and pressure. Add the separate `forces` object and read its column
  names **from its own header** — v2606 writes Time, *total*, pressure, viscous,
  so counting from the left lands on the total vector.
- **`nuTilda` is floored at freestream by `write_case`** (`np.maximum(nut_arr,
  nut_inf)`). On this mesh that is 5871 of 31,700 cells — the inner boundary
  layer, where a correct prediction goes to zero — so **no seed, the oracle
  included, can carry eddy-viscosity information into the innermost layer.**
  State this as a limitation; it is also what makes the `nf_bl_nut` /
  `nf_bl_vel` split interpretable rather than a formality.
- **`wall_distance` must be point-to-*segment*.** To the nearest polyline vertex
  it overestimates by 368× at the first cell ring (1.4e-3 against 3.8e-6), which
  halved the model's field error when fixed. It did *not* invalidate the
  selective-seeding masks (26 of 31,700 cells moved, max weight change 0.035) —
  that was checked, not assumed.
- **Split the sweeps by case, never by resolution or by arm.** Every rung of a
  case shares its `cold` and `oracle_mesh` runs. `running_solvers()` now refuses
  a case another process is solving; `--force` overrides it, and once cost three
  re-solves.
- **`f"{1.5e-5:.0e}"` is `"2e-05"`.** One significant figure silently collides
  depth-ladder keys.
- **`potentialFoam` needs three things**: a Dirichlet anchor for `Phi` (the
  Poisson problem is otherwise singular), `div(div(phi,U)) Gauss linear` in
  `fvSchemes` for `-writep`, and GaussSeidel rather than DIC.
- **`$var` does not survive `wsl.exe` argument passing.** Every command this
  package builds is variable-free by design.
- **Bash heredocs here mangle backslashes** — `\n` in a Python heredoc arrives as
  a real newline, so a `.replace()` on a string containing an escape silently
  fails to match. Use the Write/Edit tools for content with escapes.
- **Never grep OpenFOAM logs for `"Floating point"`** — every log opens with
  `trapFpe: Floating point exception trapping enabled`. Match `sigFpe::sigHandler`
  or `FOAM FATAL`.
- **`geometry.solid_mask` returns 1.0 = FLUID**, not solid.
- **blockMesh rejects concave straight-sided topology hexes**; one block per
  surface segment avoids every corner trap. Per-segment radial grading is illegal
  (adjacent blocks share a radial face); `edgeGrading` is the legal way and is
  untried.
- **Non-ASCII in a `print` crashes under cp1252** when stdout is redirected.

---

## 6. Housekeeping

- Branch is **not merged to `main`**. Nothing conflicts with Paper 1.
- `runs/` is gitignored (~2 GB of case dirs); `results/` and `logs/` are tracked.
- Solves **resume from disk** (`openfoam.completed_run`) — 1.8 s against 150 s.
  It rejects a run that stopped short of the requested `n_iter`, so raising the
  budget re-solves rather than silently reusing a short run.
- `write_cgrid_case` `rmtree`s the case directory first, so no stale time
  directory can be read back as the answer. Still, **use a fresh `--work-dir`**
  for a re-run: the old tree is the evidence behind the tables above.
- Every experiment script checkpoints after each case, atomically.
- The venv is `.venv/Scripts/python.exe`; the bare `python` on PATH has no torch.
