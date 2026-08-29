# PLANS — the road to Paper 2

**Living document. Update it at the end of every working session**, before the
machine can lose power. Companion: `docs/GOALS.md` (why), this file (what next).

Last updated: **2026-08-29** · branch `paper2/openfoam-warm-start` · pushed to
`origin` (github.com/ali-kin4/neuroforge-cfd) · 29 commits ahead of `main`.

---

## 0. The goal, stated so it can be failed

> A paper establishing **the two conditions under which a neural surrogate
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

Today: 1–4 hold (+33.9% Cd@1%, control +92.1%), 5 is at n=5, 6 is at n=1.
Phases B and C close those two. Nothing else is missing to clear the bar.

**Venue.** CMAME — this is new computational methodology, which is exactly what
CMAME told us Paper 1 lacked. JCP alternative. Subscription licence, no APC
(`no-apc-venues-only`).

---

## 1. Where we are in one paragraph

A real OpenFOAM v2606 backend drives `simpleFoam` + Spalart–Allmaras from Windows
through WSL2 on body-fitted C-grids we generate ourselves, at AirfRANS Reynolds.
Twelve experiments have asked what a surrogate must do to warm-start it. The
answer that survived every control: **a surrogate helps only if it can be
evaluated at the solver's own cell centres, and only if you hand over the
boundary layer alone.** Both conditions are necessary; neither is sufficient. The
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
near-freestream velocity at a cell centre 4e-6 chords off the wall, because no
16,384-value grid — Cartesian *or* wall-fitted — has a station there. Against a
cold start's 2851% they remove between a third and a half of the error in the
quantity that decides drag. The network, evaluated pointwise at the solver's own
cell centres with no resampling at all, removes 98% of it.

This is why every projected arm is negative on drag and the mesh-native one is
positive. It is currently an *explanation*; `mesh_native_probe.py` (§4, Phase A)
makes it a controlled result.

### 3.2 The recipe: a 2×2 in which only one corner works

Cd@1%, the one readable total-drag row (§3.3), five cases, cold = 802 iterations:

|  | whole field | boundary layer only |
|---|---:|---:|
| **resampled to a 16k grid** | −184.6% | −217.5% |
| **mesh-native** | < −573.6% | **+33.9%** |

Both factors necessary, neither sufficient. Mesh-native evaluation preserves the
wall gradient (§3.1); restricting to the boundary layer avoids handing over an
outer field the model extrapolates badly (its training `sdf` distribution is
centred on 0.23 chords, the C-grid reaches 20).

The oracle control reads **+92.1%** on the same row.

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
| **Cd@1%** | readable | **+33.9%** | +92.1% |
| Cd@0.5% | unreadable (settled arms disagree by 0.33%) | −7.4% | +92.6% |
| Cd@0.2% | unreadable | −90.1% | +92.9% |
| **Cd_v@1%, @0.5%** | readable | +14.6%, +13.7% | +92.5%, +91.9% |
| **Cl@1%, @0.5%, @0.2%** | readable | +10.1%, +3.5%, −1.3% | +99.9% |
| Cd_p@1% | unreadable (arms disagree by 12.4%) | — | — |

> **The +41.8% at Cd@0.5% reported on 2026-08-28 is withdrawn.** It was read
> against a reference a diverged arm had moved; on the settled reference it is
> −7.4%. The +34.1% at Cd@1% survives as +33.9%.

Making Cd@0.5% readable needs a longer budget, not a better metric — see Phase B4.

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

- `fitted_p` (pressure only) is **inert**: +0.1% on every metric, at every depth.
- `composite` (potential-flow pressure + surrogate BL) is **negative**: −320.0%.
- `potential` (potentialFoam alone, the free industrial baseline) is inert on
  drag (+0.7% Cd@1%) and mildly positive on lift (+3.3%).
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

Five cases × eleven strategies, `K = 25`, threshold on the residual level:

| metric | ungated mean | worst seed | **gated mean** | **gated worst** | harmful admitted |
|---|---:|---:|---:|---:|---:|
| Cd@1% | −168.9% | −1169.6% | **+3.7%** | **−5.8%** | 0/32 |
| residual 5e-6 | −170.8% | −1449.3% | +1.8% | −7.6% | 0/36 |
| Cl@1% | −21.5% | −671.9% | +1.9% | −100.0% | 1/22 |
| Cd_v@1% | +23.5% | −8.6% | +23.5% | −8.6% | 9/10 |

The gate is not what makes warm starting fast; it is what makes it adoptable. It
captures only 17–24% of what a gatekeeper with foreknowledge would get on the
metrics where most seeds are harmful — conservative by construction — and on
viscous drag, where 40 of 50 seeds already help, it is nearly a no-op at 97%
capture. Longer probes are monotonically worse: by `K = 400` the probe cost alone
(−49.6%) exceeds anything the decision can recover.

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
- **The wall-clock saving is real** but n=1: `+41% iterations → +30% seconds` on
  one case, serial and exclusive. The per-iteration penalty is 1.14×, not the
  1.62× a contended box suggested. Phase C makes this n=5.

---

## 3.9 In flight right now

| tree | what | budget | why |
|---|---|---:|---|
| `runs/openfoam/repr3` | `mesh_native_probe.py`, 5 cases × 3 arms, one process per case | 6000 | Phase A: the controlled test of §3.1 and the channel split of §3.5 |

`logs/mesh_native_*.log`; results land in `results/mesh_native_*.json`. Every case
checkpoints atomically and `completed_run` reuses finished ones, so an interrupted
sweep resumes by re-running the same command.

---

## 4. The road to the paper

Phases A–C are compute and are what stand between today's numbers and the bar in
§0. D–F turn measurements into a paper.

### Phase A — establish the mechanism, and split the channels (running, ~1 h)

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
- **B4 — the budget that makes Cd@0.5% readable (~3 h).** 12,000 iterations on
  the five existing cases, `cold` / `oracle_mesh` / `nf_bl` only. §3.3 says the
  settled arms must agree to 0.25% for a 0.5% band; they are at 0.33%. Either
  this closes it or the paper states 1% as the resolution limit of the method —
  both acceptable, neither guessable.
- **B5 — re-measure the Reynolds crossover on the relaxed settings (~2 h).** The
  low-Re numbers in §3.8 predate the relaxation fix and sit near their floors.

### Phase C — cost, honestly (~2 h, serial and exclusive)

`scripts/wallclock_control.py` at n=5, nothing else running. Iterations are
contention-proof; seconds are not, and the current sentence rests on one case.
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
- Figures: (i) the wall-gradient bar chart of §3.1 — the paper's central image;
  (ii) the 2×2 of §3.2; (iii) the per-quantity convergence decomposition of §3.5;
  (iv) the certificate's capture-versus-`K` curve; (v) mesh structure.
- Every number traceable to a committed `results/*.json`.

### Phase E — adversarial review before submission

Run the reviewer panel over the draft. The three objections already visible:

1. *"You compare a trained model against a projected oracle; the comparison is
   confounded."* → Phase A exists to remove exactly this. Do not submit without it.
2. *"Five NACA airfoils at small incidence."* → Phase B.
3. *"You report the metric that worked."* → §3.3's readability table and §3.6's
   gate, both of which report the metrics that did not.

### Phase F — write it

Working title: **"A surrogate must speak the solver's mesh: mesh-native,
boundary-layer-only warm starts for RANS."** The arc:

> **Problem.** ML surrogates are sold as accelerators for classical CFD via warm
> starting. In practice a warm start often makes a steady RANS solve *slower*,
> and you find out after paying. Whether it helped even depends on which
> quantity you measure.
>
> **Mechanism.** Measured, not argued: a 16k-value grid — Cartesian or
> wall-fitted — cannot represent a first cell 4e-6 chords off the wall, so
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
| [Hybrid init](https://arxiv.org/html/2503.15766v1) (NVIDIA) | transient URANS, DoMINO + potential flow | ~2× | uses the same drag-band metric; ours is steady RANS, and we measure potential flow as an arm and find it inert on drag (+0.7%) |
| [Wake extension](https://arxiv.org/abs/2501.14699) | near-body/wake decomposition | 26.3× iterations | decomposition by region; ours by **field and wall distance**, derived from a measured convergence decomposition |
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
