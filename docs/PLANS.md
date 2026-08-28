# PLANS — next moves (Paper-2 / OpenFOAM track)

**Living document. Update it at the end of every working session**, before the
machine can lose power. Companion: `docs/GOALS.md` (why), this file (what next).

Last updated: **2026-08-28** · branch `paper2/openfoam-warm-start` · pushed to
`origin` (github.com/ali-kin4/neuroforge-cfd) · 25 commits ahead of `main`.

---

## 1. Where we are in one paragraph

A real OpenFOAM v2606 backend drives `simpleFoam` + Spalart–Allmaras from
Windows through WSL2, on body-fitted meshes we generate ourselves (O-grid and
C-grid, both reaching AirfRANS Reynolds). Six experiments asked whether a
NeuroForge prediction can warm-start a RANS solve. Warm-starting **works at
moderate Reynolds**. At Re 3e6 a uniform Cartesian surrogate grid never pays, at
any resolution. Whether a **wall-fitted** surrogate pays at Re 3e6 is **open and
being re-measured** — the +13%/+30% previously claimed was read off a residual
curve that had already stopped moving (§3.1), and the reason it had stopped has
now been found and fixed (§3.2).

Nothing here touches Paper 1, the frozen contracts, or `ClassicalFallback`
(still `NotImplementedError` for `'openfoam'`, exactly as the JCP submission
states).

---

## 2. What is built (all committed and pushed)

| Module | What it does |
|---|---|
| `src/neuroforge/solver/openfoam.py` | WSL2 plumbing, case writer, runner, log/field parsers, uniform Cartesian mesh, `iterations_to_threshold`, `read_force_coeffs` / `iterations_to_force_band`, `completed_run` (resume) |
| `src/neuroforge/solver/ogrid.py` | Graded body-fitted **O-grid**, one block per surface segment |
| `src/neuroforge/solver/cgrid.py` | Body-fitted **C-grid**, stitch-free wake cut via shared vertex ids; instruments every case with `forceCoeffs` |
| `src/neuroforge/solver/warmstart.py` | Seeding strategies: `plain_seed`, `hybrid_seed`, `clustered_seed` (wall-fitted), `bl_thickness`, `surface_coords` |
| `scripts/dashboard.py` + `app/openfoam_dashboard.html` | Live run monitor (residuals, runtime, ETA, grouped runs) |
| `scripts/convergence_diagnostic.py` | **Why the solve stalls.** Ten variants of numerics and mesh on one case |
| `scripts/reanalyse_depth.py` | Re-scores a finished run tree at a ladder of depths, with each depth as a multiple of the residual floor |
| `scripts/*_probe.py`, `*_ladder.py`, `*_crossover.py` | The six experiments, all checkpointing and resumable |
| `scripts/plot_mesh_structure.py`, `plot_crossover.py` | Publication figures |

~350 tests, all runnable with **no OpenFOAM installed**; the end-to-end ones are
`slow` and self-skip.

---

## 3. The findings, in the order they were established

1. **Uniform Cartesian mesh dies above cell Re ≈ 250** (SIGFPE). Body-fitted
   meshes were not optional.
2. **`residualControl` never fires** on the staircase mesh — steady SIMPLE
   stagnates at a nonzero floor. Metric became `iterations_to_threshold`.
3. **Pilot at Re 1e4**: warm-starting from a neighbouring case saves **69.7%**
   (oracle control **95.4%**).
4. **Re 3e6, 128² grid: fails**, and it is *not* a training problem — the arm
   under test is the **exact answer** degraded only in resolution. Mask-aware
   round-trip identical, so not a projection artifact.
5. **Rebuilding the boundary layer does not rescue it. Refining the grid does
   not either** (128→421², flat). `delta/h` does **not** transfer between the
   Reynolds axis and the grid axis — the controlling scale is the viscous length
   `nu/u_tau`, which collapses 1660× across the sweep while `delta/h` moves 5×.
6. **Wall-fitted representation at equal budget (16,384 values)** looked positive
   at deep thresholds. **Now under re-measurement** — see below.

### 3.1 ⚠️ Two metric faults — read this before trusting any earlier number

**Fault A — the pressure history was three times too long.** With
`nNonOrthogonalCorrectors 2`, `simpleFoam` solves pressure three times per SIMPLE
iteration and the log parser appended all three to one list. Index `i` meant
outer iteration `i` for velocity but `i/3` for pressure, and
`iterations_to_threshold` compares fields at a common index. Fixed
(`parse_simple_foam_log` now groups per `Time` block and keeps the first solve of
each field).

Every tree on disk has been re-scored with the corrected parser — no re-solving,
`scripts/reanalyse_depth.py`, results in `results/depth_*.json`. **The negative
results all hold or harden; the positive low-Reynolds ones shrink**, and one of
them mostly disappears:

| claim | first reported | corrected | verdict |
|---|---:|---:|---|
| Pilot Re 1e4, neighbour seed @1e-3 | +69.7% | **+47.3%** | holds, smaller (n=5 of 10 — the others never reach 1e-3) |
| Pilot oracle control | +95.4% | +55.6% | control still passes |
| Crossover Re 1e3 | +58% | **+8.1%** | mostly gone (−3.8% and +20.0%) |
| Crossover Re 1e4 | +14% | +14.4% | holds |
| Re 3e6 uniform 128² @1e-3 | −18.5% | −30.2% | hardens |
| Wall-fitted 256×64 @1e-3 | −44.4% | −70.8% | hardens |
| Hybrid BL reconstruction @1e-3 | −31.5% | −33.9% | hardens |

So the **crossover** narrative — a clean sign change on the Reynolds axis — is
weaker than claimed. The sign change is still there, but it sits between Re 1e4
(+14%) and Re 1e5 (−17%), and Re 1e3 is +8% with one of its two cases negative.
Those low-Re runs also carry high residual floors (1.1e-4 to 3.5e-4), so 1e-3 is
only 3–9× above the floor there: they need re-measuring on the relaxed settings
before the figure goes in a paper.

> **This table clears fault A only.** Re-scoring fixes the parser; it cannot fix
> the solver. Every number above — the negatives included — was still *solved* at
> relaxation 0.9, on the limit cycle. A good seed dropped into a limit cycle
> shows nothing, so **`cartesian_128` may also turn positive at proper
> convergence.** If it does, "it is the representation, not the resolution" is
> dead and the contribution is the methodology instead (§3.2 plus the measurement
> protocol) — a different paper, and not obviously a worse one. Do not write an
> abstract until `repr2` lands.

**Fault B — the positive readings sit on the floor.** Corrected ladder on
`runs/openfoam/repr`, with each depth expressed as a multiple of the cold
residual floor (~1.3e-5):

| depth | × floor | cold it | oracle (control) | uniform 128² | wall-fitted 256×64 |
|---|---:|---:|---:|---:|---:|
| 1e-2 | 627× | 13 | +92.1% | −20.6% | −104.4% |
| 1e-3 | 63× | 35 | +73.6% | −30.2% | −70.8% |
| 1e-4 | 6.3× | 74 | +64.0% | −366.4% | −42.9% |
| 5e-5 | 3.1× | 95 | +65.2% | −140.9% | −79.6% |
| 3e-5 | **1.9×** | 273 | +84.3% | −33.6% | **+15.0%** |
| 2e-5 | **1.3×** | 347 | +86.1% | −21.7% | **+31.2%** |
| 1.5e-5 | **0.9×** | 458 | +88.0% | −9.0% | **+13.5%** |

Every positive number appears only within a factor of two of the floor, and the
same arm reads +15%, +31%, +13% at three adjacent rungs. Per case at 3e-5 the
fitted arm is +10%, +49%, −15%. **That is not a measurement of a convergence
rate; it is where a flat curve happens to cross a line.**

`scripts/reanalyse_depth.py --per-case` prints all of this without re-solving.

### 3.2 The floor was under-relaxation, not the mesh

`scripts/convergence_diagnostic.py`, one case at Re 3e6, cold, matched budget:

| variant | change | result |
|---|---|---|
| shipped | — | stalls at 1.1e-5, never reaches 1e-5 |
| `long` | 4000 iterations | **1.13e-5** — a longer budget buys nothing |
| `tight` | inner relTol p 1e-3, U 1e-2 | 1.16e-5, unchanged |
| `upwind` | first-order convection | 8.8e-6, marginal |
| `wake` | AR 218,987 → 57,389 | not what breaks the stall |
| `relax` | U, nuTilda 0.9 → 0.7 | 1e-5 at iteration 327, on to 1.9e-6, still falling |
| `relax05` | 0.5 | 1.2e-6 — further is not better |
| `relax_wake` | 0.7 + the wake mesh | 1.7e-6 — the mesh adds nothing |
| **`relax_nut`** | **U 0.7, nuTilda 0.4** | **the only variant to reach 1e-6 (iteration 1305); ends at Ux 3.5e-7** |

The oracle arm had already said the budget was not the constraint: seeded with
the converged field, `Ux` reaches 1.2e-5 by iteration 100 and sits there for the
next 700. No budget digs below a level the exact answer rests on.

So the shipped relaxation of 0.9 (with SIMPLEC) was a limit cycle. Dropping U to
0.7 moves the floor down 6×; `nuTilda` then becomes the laggard at ~10× `Ux`, and
relaxing *it* to 0.4 buys another decade — 3.5e-7, thirty times below where this
started. `RELAX_U = 0.7` / `RELAX_NUT = 0.4` are now the defaults, and the forces
settle to match: Cd within 0.01% of converged by iteration 2000, against 0.3% for
uniform 0.7 relaxation.

Also useful to know what did **not** matter: the mesh. The `wake` variant cuts
the worst aspect ratio from 218,987 to 57,389 and changes nothing.

### 3.3 First result on a solve that converges (`repr2`, six cases, 3000 iterations)

Relaxation 0.7/0.4, per-outer-iteration parser, forces instrumented, arms that
never reach a target counted rather than dropped. Cells are
`saving (reached/total)`; `<` marks a bound.

| depth | × floor | cold it | oracle (control) | uniform 128² | wall-fitted 256×64 |
|---|---:|---:|---:|---:|---:|
| 1e-3 | 995× | 74 | +93.1% (6/6) | +11.8% (6/6) | −9.7% (6/6) |
| 1e-4 | 100× | 201 | +93.0% (6/6) | +17.8% (6/6) | +2.9% (6/6) |
| **1e-5** | **10×** | 595 | +87.6% (6/6) | < −199.4% (4/6) | **+6.0% (6/6)** |
| **5e-6** | **5×** | 1054 | +92.4% (6/6) | < −62.5% (4/6) | **+35.8% (6/6)** |
| 1e-6 | 1.0× | 1823 | +92.3% (4/4) | < −62.1% (1/4) | < +21.8% (3/4) |
| Cl @1% | — | 940 | +99.9% (5/5) | −134.2% (5/5) | **+86.0% (5/5)** |
| Cd @1% | — | 750 | **+1.0%** ⚠ | < −386.8% (3/6) | −204.5% (6/6) |

**What is solid.** The wall-fitted representation beats the uniform Cartesian one
at *every* depth and on *both* force coefficients, often by hundreds of percent.
That comparison is the experiment's whole point and it is unambiguous.

**What is not readable yet.** The oracle control **fails on Cd** (+1.0%), so by
the rule this repo has followed throughout, the Cd column must not be read. The
cause is known: the shared reference is the oracle arm's final drag, and at 3000
iterations that is not converged — cold and oracle disagree by up to 0.75% on
`naca4412`, wider than the band being measured. `repr3` re-runs at 6000.

**What is genuinely open.** Wall-fitted *versus a cold start* is mixed: positive
on residuals at depth (+6% to +36%) and on lift (+86%), negative on drag.

### 3.4 Why drag and lift disagree — and what it says to do

Decomposing the coefficients (`forces` function object; `Cd(f)`/`Cd(r)` are front
and rear, **not** viscous and pressure) settles it. Iterations to settle within 1%
of the converged value, `repr3`, 6000-iteration runs:

| quantity | cold start | seeded with the exact field | share of Cd |
|---|---:|---:|---:|
| viscous drag `Cd_v` | ~700 | ~53 | 60–84% |
| lift `Cl` | ~950 | **1** | — |
| pressure drag `Cd_p` | ~1850 | **1–2** | 16–40% |

**A cold solver is slow at pressure and fast at the near-wall velocity
gradient.** A surrogate is the exact reverse: pressure is the one channel that
survives projection intact, and the wall gradient is what it destroys. And the
wall gradient dominates total drag.

So handing over the whole field buys a large gain on the pressure-dominated
quantities and pays for it on the one the solver was going to get right by
itself. That is the +86% / −71% split, and it predicts a fix that needs no better
model: **hand over less.** `warmstart.masked_seed` +
`scripts/selective_seed_probe.py` are built and tested for exactly this.

If this is right it is the paper's headline, because it is a *recipe* rather than
an observation: it applies to any surrogate anyone already has.

**Bonus, and not in any iteration count**: the oracle arm runs at **0.134 s per
iteration against the cold arm's 0.251 s** on the same box. A good initial guess
shortens the inner linear solves too, so the cost saving exceeds the iteration
saving. Needs a serial run to quantify honestly.

---

## 3.9 In flight right now

| tree | what | budget | why |
|---|---|---:|---|
| `runs/openfoam/repr3` | 6 cases × 4 arms, Re 3e6 | 6000 | the decisive experiment; 3000 left the arms disagreeing on final Cd by 1.6%, wider than the band |
| `runs/openfoam/crossover2` | 2 airfoils × 5 Reynolds | 3000 / 6000 | where warm-starting stops paying, on the relaxed settings |
| `runs/openfoam/resladder2` | 3 cases × 5 resolutions | 6000 | "refining the surrogate grid does not help", re-measured |
| — | `scripts/wallclock_control.py` | 6000 | **run last, alone.** Refuses to start while another solve is running |

Restart commands are in each script's docstring; every case checkpoints
atomically and `completed_run` reuses finished ones, so an interrupted sweep
resumes by re-running the same command.

The wall-clock control is not optional and not a footnote. On the shared box the
arms ran at 0.134 / 0.251 / 0.281 / 0.407 s per iteration (oracle / cold /
uniform / fitted). Taken at face value, the fitted arm's **+35.8% iteration
saving at 5e-6 becomes a 4% loss in seconds**, while the oracle arm's +92.4%
stays a 96% win. Whether those ratios are real or an artifact of a dozen solves
competing for memory bandwidth decides which of those two sentences goes in the
paper.

---

## 4. The road to the paper

The target is a paper that is **novel, practical, and positive** — a recipe a CFD
engineer can apply on Monday to a surrogate they already have, with a guarantee
attached. Everything below serves that. Phases 1–3 are compute; 4–6 are what turn
measurements into a defensible paper.

### The paper, in one arc

> **Problem.** ML surrogates are sold as accelerators for classical CFD via warm
> starting. In practice a warm start can make a steady RANS solve *slower*, and
> you only find out after paying for the solve. Whether it helped even depends on
> which quantity you measure.
>
> **Protocol.** A measurement discipline that does not fool itself (§3.1, §3.2,
> `solver/scoring.py`). Without it, three of our own conclusions had the wrong
> sign.
>
> **Diagnosis.** Cold RANS is slow at pressure and fast at near-wall shear;
> a surrogate is exactly the reverse (§3.4). That is *why* the literature's
> warm-start results are mixed.
>
> **Recipe.** Hand over the pressure and outer field; keep the near-wall velocity.
> And put the surrogate's points in the solver's wall-normal coordinates, not on
> a Cartesian grid — same output budget, opposite sign.
>
> **Guarantee.** A cheap acceptance test that bounds the worst case, so the recipe
> can never cost more than a fixed fraction over a cold start.

### Phase 1 — the controlled corpus (running, ~2–3 h)

`repr3`, `resladder2`, `crossover2`. Gives the representation criterion, the
resolution negative, and the Reynolds dependence, all on a converging solve and
all with an oracle control. See §3.9.

### Phase 2 — the recipe (built, runs next, ~2 h)

```bash
python scripts/selective_seed_probe.py --only naca0012@4 --work-dir runs/openfoam/repr3
```
`fitted_p` (pressure only) and `fitted_outer` (velocity handed back inside the
boundary layer) against the same wall-fitted projection. Reuses `cold` and
`oracle_mesh` from `repr3`, so three solves per case, not five.

**Also Phase 2, and a gap in the plan until now: the classical baseline.**
`potentialFoam` ships with OpenFOAM and produces a potential-flow initial field
in seconds, for free, with no model and no training data. It is what industry
actually does, and it is the baseline
[NVIDIA's hybrid initialisation](https://arxiv.org/html/2503.15766v1) blends
*with*. **If a NeuroForge seed does not beat potential flow, the paper has no
practical claim** — and no reviewer will let that pass unasked. Add it as an arm.

### Phase 3 — wall-clock, serially (~1 h, alone)

`scripts/wallclock_control.py`. Iterations are contention-proof; seconds are not.
This decides whether "+35.8% iterations" is also a speed-up (§3.9).

### Phase 4 — a real model, not just the oracle (~2 h)

Every arm so far is the **exact answer** degraded only in representation, which
makes it an upper bound on any surrogate. That is the right way to isolate the
mechanism and the wrong way to end a paper: a reviewer will ask whether a
*trained* model gets anywhere near the bound. Seed from an actual NeuroForge
prediction on the same cases and report the fraction of the oracle's saving that
survives. Include inference cost in the accounting (milliseconds, but state it).

### Phase 5 — the no-harm certificate (mostly analysis)

Run `K` probe iterations from the seed; if the residual has not fallen below the
cold trajectory's envelope by `K`, discard the seed and continue cold. Worst case
is then `(1 + K/N)` times a cold solve — a *guarantee*, not a hope. Two things to
measure from the corpus Phases 1–4 produce: the smallest `K` that separates the
winners from the losers, and the fraction of the available saving the rule
captures. Paper 1's trust map is the natural pre-flight predictor to test
alongside it.

This is the contribution that matches where the field has moved: both
[NOWS](https://arxiv.org/abs/2511.02481) and
[PCGBandit](https://arxiv.org/html/2509.08765) sell *inherited correctness* over
raw speed, and PCGBandit's headline property is "always at least as fast as the
default".

### Phase 6 — write it

Figures: the representation comparison, the per-quantity convergence
decomposition (§3.4), the selective-seeding ablation, the certificate's
capture-versus-cost curve. Tables from `results/*.json` — every number in the
paper traceable to a committed file.

### Positioning against the 2026 literature

| work | what it warm-starts | reported | how ours differs |
|---|---|---|---|
| [NOWS](https://arxiv.org/abs/2511.02481) (CMAME 2026) | inner Krylov solves | up to 90% time | we warm-start the outer field; complementary, and they note learned preconditioners are "typically restricted to Cartesian grids" |
| [Spectrally Safe](https://arxiv.org/abs/2606.21828) (2026) | Newton solvers | 5.4× at 6.4M DOF | same core insight from a different angle — **L² accuracy is not the property that makes a seed good**; theirs is Jacobian definiteness, ours is *which field* is handed over |
| [PCGBandit](https://arxiv.org/html/2509.08765) | preconditioner choice, online | 1.5× total, 4× linear | orthogonal, and composable with ours |
| [Hybrid init](https://arxiv.org/html/2503.15766v1) (NVIDIA) | transient URANS, DoMINO + potential flow | ~2× | uses the same drag-band convergence metric we do; ours is steady RANS, and asks *what to hand over* rather than *what to blend* |
| [Wake extension](https://arxiv.org/abs/2501.14699) | near-body/wake decomposition | 26.3× iterations | decomposition by region; ours is by **field and wall distance**, and derived from a measured convergence-rate decomposition |

The niche nobody occupies: **no published warm-start study reports which
*quantity* converges and which does not, and none can tell you before paying
whether a given seed will help.** That is the gap Phases 2 and 5 fill.

### Still queued, beyond this paper

- **NOWS-style inner warm start** — Paper-3 sized, needs OpenFOAM C++ to inject a
  per-solve initial guess. The real prize: 90% with the solver's guarantees intact.
- **Self-consistency via Neural Residual Iteration** — run Paper 1's monotone
  acceptance loop on the prediction before seeding. Cheap with what is built, and
  would make Paper 2 depend on Paper 1's contribution.
- **Region-decomposed seeding** — the inverse of what we tried: surrogate wake,
  cheap precursor near-body.


## 5. Gotchas that cost time — do not rediscover these

- **The four scoring rules are in `solver/scoring.py`, with a test naming each
  mistake they prevent.** Read that module before writing a new analysis script;
  every one of these was made here, and each changed a sign, not a decimal:
  a threshold only measures a rate while the residual is falling; an arm that
  never reaches the target is bounded rather than dropped; all arms are scored
  against one *external* reference; and that reference is only usable if the arms
  agree to well inside the band.
- **A residual threshold is only a measurement while the residual is falling.**
  Print the threshold as a multiple of the floor and refuse to read anything
  under ~5×. Better: score on the forces.
- **Score a Reynolds sweep one Reynolds number at a time.** The floor moves three
  orders of magnitude across it, so an aggregate "× floor" describes no point in
  the sweep. `reanalyse_depth.py --filter re1e+05`.
- **`forceCoeffs`' `Cd(f)` / `Cd(r)` are front and rear about `CofR`**, not
  viscous and pressure. For that, add the separate `forces` object and read its
  column names from its own header — v2606 writes Time, *total*, pressure,
  viscous, so counting columns from the left lands on the total vector.
- **Split the sweeps by case, never by resolution.** Every rung of a case shares
  its `cold` and `oracle_mesh` runs; two processes splitting resolutions write
  the same directory at once.
- **`nNonOrthogonalCorrectors` multiplies the pressure history.** Group residuals
  per `Time` block; never scan a whole log for `Initial residual` and zip fields
  by index.
- **`f"{1.5e-5:.0e}"` is `"2e-05"`.** One significant figure silently collides
  depth-ladder keys. `reanalyse_depth.key()` uses one decimal.
- **`$var` does not survive `wsl.exe` argument passing.** A probe like
  `for f in ...; do echo "$f"; done` silently returns nothing. Every command this
  package builds is variable-free by design.
- **Bash heredocs here mangle backslashes** — `\\n` in a Python heredoc arrives
  as a real newline. Use the Write/Edit tools for content with escapes.
- **Never grep OpenFOAM logs for `"Floating point"`** — every log opens with
  `trapFpe: Floating point exception trapping enabled`. Match
  `sigFpe::sigHandler` / `FOAM FATAL` instead.
- **`geometry.solid_mask` returns 1.0 = FLUID**, not solid. Reading it the other
  way inverts the mesh cut-out and both guards in `write_case` survive it.
- **blockMesh rejects concave straight-sided topology hexes.** One block per
  surface segment avoids every corner-placement trap.
- **Per-segment radial grading is illegal** — adjacent blocks share a radial face
  and must agree ("Point merge failure ... inconsistent grading"). `edgeGrading`
  is the legal way to vary it along the C, and is untried.
- **Non-ASCII in a `print` crashes under cp1252** when stdout is redirected.
- Dashboard: `python scripts/dashboard.py --port 8013`. It dies with the session;
  restart it, nothing is lost.

---

## 6. Housekeeping

- Branch is **not merged to `main`**. Nothing conflicts with Paper 1.
- `runs/` is gitignored (~2 GB of case dirs); `results/` is tracked.
- Solves **resume from disk** (`openfoam.completed_run`) — an interrupted
  experiment re-reads finished cases instead of re-solving (1.8 s vs 150 s).
  `completed_run` rejects a run that stopped short of the requested `n_iter`, so
  raising the budget re-solves rather than silently reusing a short run.
- `write_cgrid_case` `rmtree`s the case directory first, so no stale time
  directory can be read back as the answer. Still, **use a fresh `--work-dir`**
  for a re-run: the old tree is the evidence behind the tables above.
- Every experiment script checkpoints after each case, atomically.
- The venv is `.venv/Scripts/python.exe`; the bare `python` on PATH has no torch.
