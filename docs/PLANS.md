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
each field). The deep numbers barely moved — `Ux`/`Uy` bind there — but the
shallow ones did.

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
| **`relax`** | **U, nuTilda 0.9 → 0.7** | **1e-5 at iteration 327, on to 1.9e-6, still falling** |
| `tight_relax` | both | 2.3e-6, steepest tail slope (−0.061 dec/100) |

The oracle arm had already said the budget was not the constraint: seeded with
the converged field, `Ux` reaches 1.2e-5 by iteration 100 and sits there for the
next 700. No budget digs below a level the exact answer rests on.

So the shipped relaxation of 0.9 (with SIMPLEC) was a limit cycle. Dropping it to
0.7 moves the floor down by at least 6×, which puts the interesting thresholds
back where an iteration count means something. `nuTilda` becomes the laggard at
~10× `Ux`.

---

## 4. Next moves, ranked

### ▶ NEXT: (1) Re-measure the representation claim on a solve that converges
Three things changed since the numbers in §3.1 were taken, and all three have to
be in place before the claim is worth anything:

- relaxation 0.7, so the residual keeps falling past 1e-6;
- the per-outer-iteration parser, so thresholds mean what they say;
- `forceCoeffs` on every case, so convergence can be scored on **Cd and Cl**
  rather than on a residual — `iterations_to_force_band` returns the first
  iteration after which the coefficient *stays* inside a relative band, measured
  against a shared reference for both arms. This is what the warm-start
  literature reports and it does not care where the residual floor is.

```bash
python scripts/representation_probe.py --re 3e6 --n-iter 2500 \
       --work-dir runs/openfoam/repr2
python scripts/reanalyse_depth.py --root runs/openfoam/repr2 --per-case
```
Then widen `CASES` beyond three and add Reynolds numbers, so the claim is not a
single operating point. Budget ~4–6 h. **This decides whether there is a
positive result to publish.** Report both metrics; if they disagree, the force
metric wins and the disagreement is itself worth a paragraph.

### (2) Region-decomposed seeding — literature says 26×
[arXiv 2501.14699](https://arxiv.org/abs/2501.14699) reports **26.3× fewer
iterations, 16.4× wall-clock** on RANS by splitting near-body / wake / off-body,
taking an *accurate* near-body and letting a CNN predict the **wake**. Our
measurements agree that the outer field survives projection intact. Re-evaluate
`hybrid_seed` on the converging setup, and try the *inverse* of what we tried:
surrogate wake + cheap precursor near-body.

### (3) NOWS-style inner warm start — the real prize, Paper-3 sized
[NOWS](https://arxiv.org/abs/2511.02481) (CMAME 2026) warm-starts the **inner
Krylov solves** every outer iteration rather than the outer field: up to **90%**
time reduction with the solver's convergence guarantees intact. This dissolves
the self-consistency problem — a Krylov method does not care whether the guess is
physically consistent, only that it is close in the right norm. Needs OpenFOAM
C++ to inject a per-solve initial guess. Surveys note learned preconditioners are
*"typically restricted to Cartesian grids"* — which is exactly the gap
NeuroForge's geometry-native design fills.

### (4) Still untried: self-consistency via Neural Residual Iteration
Run Paper 1's NRI (monotone-residual acceptance) on the prediction, then seed
OpenFOAM. Attacks the mechanism the data points at, and would make Paper 2 depend
on Paper 1's contribution. Cheap with what is built.

---

## 5. Gotchas that cost time — do not rediscover these

- **A residual threshold is only a measurement while the residual is falling.**
  Print the threshold as a multiple of the floor and refuse to read anything
  under ~5×. Better: score on the forces.
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
