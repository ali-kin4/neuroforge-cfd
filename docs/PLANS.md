# PLANS — next moves (Paper-2 / OpenFOAM track)

**Living document. Update it at the end of every working session**, before the
machine can lose power. Companion: `docs/GOALS.md` (why), this file (what next).

Last updated: **2026-08-27** · branch `paper2/openfoam-warm-start` · pushed to
`origin` (github.com/ali-kin4/neuroforge-cfd) · 21 commits ahead of `main`.

---

## 1. Where we are in one paragraph

A real OpenFOAM v2606 backend now drives `simpleFoam` + Spalart–Allmaras from
Windows through WSL2, on body-fitted meshes we generate ourselves (O-grid and
C-grid, both reaching AirfRANS Reynolds). Using it, six experiments asked whether
a NeuroForge prediction can warm-start a RANS solve. The answer is **yes at
moderate Reynolds, and — measured at convergence depth — yes at Re 3e6 too, but
only with a wall-fitted output representation**. A uniform Cartesian surrogate
grid never pays at flight Reynolds, at any resolution.

Nothing here touches Paper 1, the frozen contracts, or `ClassicalFallback`
(still `NotImplementedError` for `'openfoam'`, exactly as the JCP submission
states).

---

## 2. What is built (all committed and pushed)

| Module | What it does |
|---|---|
| `src/neuroforge/solver/openfoam.py` | WSL2 plumbing, case writer, runner, log/field parsers, uniform Cartesian mesh, `iterations_to_threshold`, `completed_run` (resume) |
| `src/neuroforge/solver/ogrid.py` | Graded body-fitted **O-grid**, one block per surface segment |
| `src/neuroforge/solver/cgrid.py` | Body-fitted **C-grid**, stitch-free wake cut via shared vertex ids |
| `src/neuroforge/solver/warmstart.py` | Seeding strategies: `plain_seed`, `hybrid_seed`, `clustered_seed` (wall-fitted), `bl_thickness`, `surface_coords` |
| `scripts/dashboard.py` + `app/openfoam_dashboard.html` | Live run monitor (residuals, runtime, ETA, grouped runs) |
| `scripts/*_probe.py`, `*_ladder.py`, `*_crossover.py` | The six experiments, all checkpointing and resumable |
| `scripts/plot_mesh_structure.py`, `plot_crossover.py` | Publication figures |

~340 tests, all runnable with **no OpenFOAM installed**; the end-to-end ones are
`slow` and self-skip.

---

## 3. The findings, in the order they were established

1. **Uniform Cartesian mesh dies above cell Re ≈ 250** (SIGFPE). Body-fitted
   meshes were not optional.
2. **`residualControl` never fires** on the staircase mesh — steady SIMPLE
   stagnates at a nonzero floor. Metric became `iterations_to_threshold`.
3. **Pilot at Re 1e4**: warm-starting from a neighbouring case saves **69.7%**
   (oracle control **95.4%**).
4. **Re 3e6, 128² grid: fails** (−13% to −60%), and it is *not* a training
   problem — the arm under test is the **exact answer** degraded only in
   resolution. Mask-aware round-trip identical, so not a projection artifact.
5. **Rebuilding the boundary layer does not rescue it** (−31.5%). **Refining the
   grid does not either** (128→421², flat). `delta/h` does **not** transfer
   between the Reynolds axis and the grid axis — the controlling scale is the
   viscous length `nu/u_tau`, which collapses 1660× across the sweep while
   `delta/h` moves only 5×.
6. **Wall-fitted representation, equal budget (16,384 values), scored at
   convergence depth: +13% to +30%.** Uniform Cartesian at the same depth still
   costs 22–33%.

### ⚠️ The metric error — read this before trusting any earlier number

Experiments 3–5 reported savings at **residual 1e-3**. A cold start reaches 1e-3
in **36 iterations** and the residual floor (~1.3e-5) after ~350, so 1e-3 is
about **12% of the way to convergence**. The sign of the result changes with
depth:

| depth | cold it | oracle (control) | uniform 128² | wall-fitted 256×64 |
|---|---:|---:|---:|---:|
| 1e-3 | 36 | +73.1% | −18.5% | −44.4% |
| 1e-4 | 85 | +68.5% | −306.4% | −49.5% |
| **3e-5** | 274 | +84.3% | −33.2% | **+13.0%** |
| **2e-5** | 347 | +85.9% | −21.7% | **+30.4%** |

`scripts/reanalyse_depth.py` re-scores any finished run tree at a depth ladder
without re-solving. **Always report the depth alongside a saving.**

---

## 4. Next moves, ranked

### ▶ NEXT: (1) Re-run at proper convergence depth
The +13%/+30% is real but thin: `n` drops to 2 at the deepest rung because the
800-iteration budget truncates, and the spread is wide (+4%, −15%, +49% at 3e-5).
This decides whether there is a positive result to publish.

```bash
python scripts/representation_probe.py --re 3e6 --n-iter 4000
python scripts/reanalyse_depth.py --root runs/openfoam/repr
```
Also widen `CASES` in `representation_probe.py` beyond three, and add a couple of
Reynolds numbers so the claim is not a single operating point. Budget ~3–5 h.

### (2) Region-decomposed seeding — literature says 26×
[arXiv 2501.14699](https://arxiv.org/abs/2501.14699) reports **26.3× fewer
iterations, 16.4× wall-clock** on RANS by splitting near-body / wake / off-body,
taking an *accurate* near-body and letting a CNN predict the **wake**. Our
measurements agree that the outer field survives projection intact. Re-evaluate
`hybrid_seed` at convergence depth, and try the *inverse* of what we tried:
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
- **Per-segment grading is illegal** — adjacent blocks share a radial face and
  must agree ("Point merge failure ... inconsistent grading").
- Dashboard: `python scripts/dashboard.py --port 8013`. It dies with the session;
  restart it, nothing is lost.

---

## 6. Housekeeping

- Branch is **not merged to `main`**. Nothing conflicts with Paper 1.
- `runs/` is gitignored (~2 GB of case dirs); `results/` is tracked.
- Solves **resume from disk** (`openfoam.completed_run`) — an interrupted
  experiment re-reads finished cases instead of re-solving (1.8 s vs 150 s).
- Every experiment script checkpoints after each case, atomically.
