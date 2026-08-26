# Paper 2 roadmap — trust-gated hybrid solving (PLANS ONLY, not started)

Status: **planning document**. Nothing here is implemented or claimed anywhere in
Paper 1. Kept in-repo so the seams Paper 1 already exposes (`ClassicalFallback`,
the conformal trust gate, the triage policy) map cleanly onto the next build.

## Thesis
The audit (Paper 1) tells you *which* predictions not to trust. Paper 2 makes that
decision *actionable*: low-trust cases are handed to a warm-started classical
solver, buying classical accuracy at a fraction of classical cost — a new
computational workflow, not a study of a signal.

## Build plan (est. 2–4 weeks, CPU-dominated)
1. **OpenFOAM backend** — `src/neuroforge/solver/openfoam.py` (WSL2, `simpleFoam`,
   **Spalart–Allmaras**). *Built; awaiting a real run.* Two corrections to the
   original plan:
   - The closure is SA, **not** k-omega SST. The AirfRANS case directories are
     named `airFoil2D_SST_*`, but the released `.vtu`/`.vtp` carry only `U`, `p`,
     `nut` — no `k`, no `omega` — and the paper states a one-equation SA model.
     SST would also be unwarm-startable from our frozen 4-channel `(u,v,p,nut)`
     output spec.
   - The entry point is a **full-case** solve, not a `ClassicalFallback`
     sub-region patch. A region patch is a local BVP with approximate Dirichlet
     data; `scripts/probe_patch_acceptance.py` measured that it never lowers true
     field error (0/140, and 0 again with exact boundary data), and a heavier
     solver inside the box cannot fix wrong data on the box border. A full-case
     solve has no such seam, so the claim is *cost*, not accuracy — which is what
     item 5 below actually asserts. `ClassicalFallback` is therefore untouched
     and still raises `NotImplementedError` for `'openfoam'`.
2. **Warm-start mapping** — *built*. The case mesh is a uniform 2-D block whose
   cell centres coincide with the NeuroForge grid points (block extended by half
   a cell, `nx × ny` cells), with body cells removed by `topoSet` + `subsetMesh`.
   Fields are written before the subset so `subsetMesh` maps them; a `cellId`
   marker field rides along, so the mesh↔grid correspondence is *read back* from
   the mesh rather than assumed. `nuTilda` is seeded as `max(nut, 3ν)`.
   Not a body-fitted C-grid — it is the cheapest mesh on which warm and cold are
   compared on identical footing. A snappyHexMesh path can replace `write_case`
   without touching the runner or the log parser.
3. **Gate policies to compare**: always-solve / never-solve / random-k% /
   ensemble-sigma gate / residual gate / fused (Paper-1 triage score).
4. **Pre-registered metrics**: final field error vs pure-neural and pure-classical;
   iterations-to-convergence (warm vs cold start); wall-clock per case and per
   fleet; cost–accuracy Pareto per gate policy.
5. **Headline target claim**: "audit-gated hybrid attains X% of classical accuracy
   at Y% of classical cost; warm-starting from the surrogate saves Z% iterations."

## Running it

```bash
# 1. install OpenFOAM in WSL2 (needs your sudo password -- run it yourself)
wsl -d Ubuntu -- bash -c 'curl -fsSL https://dl.openfoam.com/add-debian-repo.sh | sudo bash && sudo apt-get update'
wsl -d Ubuntu -- bash -c "apt-cache search 'openfoam.*-default' | tail"   # pick the current tag
wsl -d Ubuntu -- bash -c 'sudo apt-get install -y openfoam2506-default'   # substitute the tag

# 2. verify the host can see it
python scripts/openfoam_warm_start.py --check

# 3. one tiny cold case, to prove the pipeline runs end to end
python scripts/openfoam_warm_start.py --smoke

# 4. the experiment
python scripts/openfoam_warm_start.py --n-cases 5 --resolution 128 --n-iter 3000
```

Use **resolution >= 128**. On the default 3-chord domain a NACA section is 4% of
the domain height, so at res 64 the rasterised body is <= 3 cells thick and the
mesh cut-out is a coarse staircase; `write_case` warns when that happens.

Overrides: `$NEUROFORGE_WSL_DISTRO` picks the distro, `$NEUROFORGE_OPENFOAM_BASHRC`
points at an unusual install. Tests: `tests/test_openfoam.py` (41, all run with no
OpenFOAM installed; the end-to-end one is `slow` and self-skips).

The script asserts that cold and warm land on the same field (L-inf over fluid
cells) before reporting any saving -- if they disagree, the comparison is void.

## Verified on hardware (2026-08-26, OpenFOAM v2606, WSL2 Ubuntu 24.04)

**The pipeline works.** `blockMesh` -> `topoSet` -> `subsetMesh` -> `simpleFoam`
runs end to end on the first attempt. `subsetMesh` reports *"Adding exposed
internal faces to existing patch: airfoil"* — the zero-face patch declared in
`blockMeshDict` is the correct way to give it a target — and at res 64 keeps
4062 of 4096 cells with a 38-face `airfoil` wall patch. At **Re 100 the solve
converges in 89 iterations** to median |U| = 1.04, max 1.17: positive evidence
that the boundary conditions, the block mesh, the body cut-out, the patch
assignment and the `cellId` field round-trip are all correct.

**The uniform mesh has a hard Reynolds ceiling.** Measured envelope
(`naca0012`, aoa 0, budget 1500 iterations):

| res | Re | cell Re | outcome |
|----:|----:|--------:|---------|
| 64  | 1e2 | 5       | converged, 89 it |
| 64  | 1e3 | 48      | stable, not converged |
| 64  | 1e4 | 476     | **SIGFPE** |
| 128 | 1e2 | 2       | converged, 204 it |
| 128 | 1e3 | 24      | stable, not converged |
| 128 | 1e4 | 236     | stable, not converged |
| 128 | 1e5 | 2362    | **SIGFPE** |
| 128 | 3e6 | 70866   | **SIGFPE** |

Ceiling is cell Re ~250. AirfRANS is Re 2e6–6e6, so this mesh is ~300x short of
it. That is not a defect in the writer — *no* uniform mesh does high-Re external
aero; a RANS airfoil needs y+~1, i.e. a first cell ~1e-5 chord. Reaching
AirfRANS Reynolds requires a graded body-fitted C-grid (preferred over
snappyHexMesh for a 2-D section: fewer moving parts, direct control of wall
spacing), which changes the warm-start mapping from exact cell-to-cell to
interpolation onto `writeCellCentres` output.

**`residualControl` is not a usable convergence criterion here.** Steady SIMPLE
stagnates at a nonzero residual floor — at res 128 / Re 1e4, `Ux` sits at
6.2e-4 and `p` at 1.0e-3, *bit-identical from iteration 500 to 1500*. The flag
never fires, so iterations-to-convergence is undefined. The metric is therefore
`openfoam.iterations_to_threshold()`: iterations to drive max(Ux, Uy, p) below a
threshold chosen above the measured `residual_floor()`. A threshold under the
floor returns `None` (a refusal to measure) rather than a fake zero.

### Pilot result: warm-starting works (2026-08-26)

`scripts/openfoam_warmstart_pilot.py`, 10 cases (naca0012 / naca2412 x aoa
0,2,4,6,8), Re 1e4, res 128, 1500-iteration budget, metric = iterations to drive
max(Ux,Uy,p) below 1e-2 (floor was 2.6e-3). Three arms on an identical mesh:

| arm | what it starts from | iterations | mean saving |
|-----|--------------------|-----------:|------------:|
| cold | uniform freestream | 60–269 | — |
| **oracle** | the case's own final field | **1–6** | **95.4%** |
| **neighbour** | a *different* case's field (Δaoa = 2°) | **16–38** | **69.7%** |

The oracle arm is the control: starting from the answer collapses the solve to
1–6 iterations, so the measurement apparatus is sound and the neighbour number
can be read as a result. The neighbour start was wrong by L-inf 0.35–0.53 in
velocity (35–53% of freestream) — considerably worse than a trained surrogate
would be — and still saved ~70% of iterations. **This clears the gate for the
body-fitted-mesh build.**

Caveat, honestly: these runs stagnate rather than converge, so "both arms reach
the same answer" holds only to ~1e-2, and two of ten cases
(`naca0012_aoa2`, `naca2412_aoa8`) ended 2–3e-1 apart. The weakest saving (42%)
is one of those two. Any headline number must apply the agreement gate.

### Body-fitted O-grid: the Reynolds ceiling is gone (2026-08-26)

`src/neuroforge/solver/ogrid.py`. A graded O-grid replaces the Cartesian mesh:
one ring of `n_surface` segments x two radial layers, an inner wall-normal layer
carrying the boundary-layer grading (first cell 1e-5 chord, ~12% growth) and an
outer layer out to a 20-chord far-field circle. 24,000 cells. `checkMesh`: Mesh
OK, max skewness 2.3, aspect ratio 348, non-orthogonality max 76 deg (hence
`nNonOrthogonalCorrectors 2`).

Measured on `naca0012` at aoa 4, 2000-iteration budget:

| Re | result | exec | residual floor | to 1e-2 | to 1e-3 |
|----:|--------|-----:|---------------:|--------:|--------:|
| 1e4 | **converged, 534 it** | 28 s | 6.1e-8 | 12 | 57 |
| 1e5 | **converged, 377 it** | 21 s | 2.0e-7 | 12 | 57 |
| 1e6 | budget cap | 90 s | 8.9e-7 | 12 | 57 |
| 3e6 | budget cap | 259 s | 1.9e-5 | 14 | 54 |

**Re 3e6 runs** -- AirfRANS Reynolds -- with median |U| 0.98 and peak 1.53, the
suction peak an airfoil at 4 deg should have. And `residualControl` now fires:
the floor fell from 2.6e-3 on the staircase mesh to 6e-8, confirming the
stagnation documented above was a mesh artifact (`wallDist meshWave` on a
staircase wall feeds the Spalart-Allmaras production term garbage), not physics.

**Design note -- why one block per surface segment.** `blockMesh` builds a
straight-sided "topology" hex per block from its corner vertices, applies the
curved `edges` afterwards, and rejects any concave topology hex ("zero or
negative pyramid volume"). Every wide-block corner placement fails at the
trailing edge, each differently: evenly spaced corners land mid-TE-face, where
the offset curve turns ~90 deg while the surface barely moves; corners anchored
*on* the TE corners sit on a ~90-deg surface turn, so the quad's interior angle
exceeds 180 deg; and a block straddling the TE has its two corners on opposite
surfaces, so the chord between them passes through the airfoil. All three are
properties of a wide block at a high-curvature feature, so adding blocks only
shrank the violation asymptotically (-2e-3 -> -5e-4 -> -8e-5). One block per
segment removes the failure mode by construction -- the topology quad *is* the
cell -- needs no `edges` entries at all, and is verified directly in numpy by
`_segment_quads_convex` before a dict is ever written.

Tests: `tests/test_ogrid.py` (41, no OpenFOAM required).

### NULL RESULT: a 128^2 surrogate cannot warm-start a Re-3e6 solve (2026-08-26)

`scripts/ogrid_resolution_probe.py`, 3 cases at Re 3e6 on the O-grid, 800-iteration
budget. Three arms, identical mesh and schemes:

| threshold | oracle_mesh (exact) | oracle_128 (same field, via 128^2) |
|-----------|--------------------:|-----------------------------------:|
| 1e-2 | **+93.4%** | **-13.2%** |
| 1e-3 | **+81.9%** | **+3.8%** |
| 1e-4 | **+76.7%** | **-17.2%** |

`oracle_mesh` is the control and it passes decisively: warm-starting from the
case's own converged field at mesh resolution collapses the solve (14 -> 1
iterations at 1e-2). So the measurement is sound.

`oracle_128` is the *same field*, projected onto the 128^2 Cartesian grid and
interpolated back. Nothing changed but the resolution -- and the saving vanishes,
sometimes going negative. **This is not a training problem.** It was the exact
answer; no surrogate trained on that grid can do better.

**Mechanism** (`results/ogrid_resolution_bands.json`): velocity error of the
round-tripped field, binned by distance from the wall (naca0012, aoa 4):

| wall distance | cells | mean \|U\| | rel. error |
|---------------|------:|-----------:|-----------:|
| 0 - 1e-4      | 18    | 0.131 | **440%** |
| 1e-4 - 1e-3   | 417   | 0.162 | **352%** |
| 1e-3 - 1e-2   | 9365  | 0.657 | **44%** |
| 1e-2 - 0.019  | 1536  | 0.993 | 15% |
| 0.019 - 0.1   | 3991  | 1.018 | 1% |
| > 0.1         | 8673  | 0.999 | ~0% |

The outer field survives the round-trip essentially intact. The boundary layer
does not: at Re 3e6 it is ~0.019 chord thick while the 128^2 grid on the 3-chord
crop spans 0.0236 chord per cell, so **the entire boundary layer is thinner
than a single surrogate cell** -- and a single cell cannot represent a 0-to-1
profile at all, so this is not marginal. The O-grid resolves the same layer with
a first cell at 1e-5 chord. The surrogate therefore hands SIMPLE a near-wall state that
is 3-4x wrong precisely where the iterations are spent, while being perfect where
they are not. Being *wrong* near the wall is worse than being uniform: the solver
must first undo it, which is why two of the three thresholds go negative.

**Control: this is resolution, not a projection artifact.** `to_grid` returns no
mask, so the interpolation back onto the mesh can blend across the body boundary.
Re-running the round trip with solid grid points refilled from the nearest
*fluid* point -- so the wall is never crossed -- leaves the two near-wall bands
**identical** (440% and 352%); only 144 of 16384 grid points fall inside the
body, so masking is a wash. Recorded in `results/ogrid_resolution_bands.json`.

**Consequences for the Paper-2 plan.** Item 5's claim ("warm-starting from the
surrogate saves Z% iterations") does not hold at AirfRANS Reynolds with a
Cartesian-grid surrogate, and no amount of training changes that. Three honest
options, in order of cost:

1. **Hybrid seed** (cheap, untested): use the surrogate outside the boundary
   layer and a wall-consistent profile inside it. The band table says the outer
   field is essentially exact, so this is the variant most likely to convert the
   null into a result.
2. **Reframe to a regime where it does work.** The Re-1e4 pilot measured 69.7%
   saving from a neighbouring-case start; the claim survives at moderate
   Reynolds, where the boundary layer is resolved by the surrogate's grid.
3. **Change the surrogate's output representation** so it carries near-wall
   structure (wall-normal coordinates, or predicting on the solver mesh). This is
   a Paper-3-sized change to the frozen 7-in/4-out spec.

Either way this is a publishable negative result about surrogate-warm-started
CFD, and it was obtained *before* training anything.

## Companion: reliability benchmark release
Package the Paper-1 evaluation as a public harness ("submit AirfRANS predictions,
receive an audit card": trust AUROC, risk–coverage, conformal coverage). No
reliability benchmark exists in the field as of 2026-08; accuracy leaderboards do.

## Also queued (either paper)
- Audit-driven active learning at scale (Paper-1 pilot protocol:
  `docs/protocols/audit_loop_pilot.md`).
- Variationally-correct residual monitor (error-equivalent norm; would make the
  Paper-1 floor theorem constructive). High risk, high theory payoff.

## Venue
CMAME first (this is the "new computational methodology" they asked for),
JCP alternative. Cite Paper 1 for the audit machinery.
