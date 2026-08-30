# [TITLE NOT CHOSEN — see §0 below]

**Status:** first full draft, 2026-08-30. Written against `docs/PLANS.md` §3.
Every number here is traceable to a `results/*.json` file and a script; numbers
still pending are marked `[[B1]]` (13-case sweep, running) or `[[C]]` (wall-clock
at n=5). **No number in this file may be changed without changing PLANS.md too.**

**Two decisions are the author's and are deliberately left open:** the title and
the venue. The draft is written venue-neutral (no template, no page budget) so
that either choice costs a formatting pass and nothing else.

---

## 0. Title candidates

1. *A surrogate must speak the solver's mesh: mesh-native, boundary-layer-only
   warm starts for RANS* — the working title; states the finding, slightly long.
2. *Three conditions for a neural surrogate to accelerate a RANS solver* —
   shorter, promises exactly what the paper delivers, loses the mechanism.
3. *The wall gradient is the warm start: why projected neural predictions slow
   RANS solvers down* — leads with the mechanism, sharper, riskier.

---

## Abstract

Neural surrogates for external aerodynamics are usually evaluated as predictors.
We evaluate one as an *initial condition*: we hand a trained surrogate's
prediction to a production RANS solver (OpenFOAM `simpleFoam`, Spalart–Allmaras)
on a wall-resolved body-fitted C-grid at flight Reynolds number, and measure the
iterations the solver still needs before its force coefficients stop moving.

The result is strongly conditional. A trained surrogate accelerates convergence
of total drag by **+33.9%** (viscous drag **+14.6%**, lift **+10.1%**) against a
cold start, with a converged-field oracle control reading +92.1% on the same
measurement — but only when three conditions hold simultaneously, each of which
we show is necessary with a controlled arm, and none of which is sufficient
alone:

1. the prediction is **evaluated at the solver's own cell centres**, never
   resampled through a grid;
2. only the **boundary layer** is handed over, not the whole field;
3. **velocity and eddy viscosity are handed over together**, not separately.

The mechanism is measured rather than argued. The quantity viscous drag
integrates is the first-cell tangential velocity gradient, and the first cell
centre on this mesh sits 4e-6 chords off the wall. No 16,384-value grid —
Cartesian *or* wall-fitted — has a station there, so projecting even the **exact
converged field** through one leaves ~1900% error in that gradient, against 2851%
for a uniform freestream. The same surrogate queried pointwise at the cell
centres leaves 54%. Resampling one identical prediction swings total drag by 93
percentage points.

Because a bad seed can be much worse than no seed, we add an acceptance test: run
K = 25 probe iterations, read one scalar from the residual history, and either
continue or discard the seed and start cold. The rule never observes a cold run,
its threshold is calibrated leave-one-case-out, and it bounds the worst case at
(1 + K/N) x cold by construction. Across 5 cases x 11 seeding strategies it
turns an ungated mean of -168.9% (worst -1169.6%) into **+3.7%, worst -5.8%, with
zero harmful seeds admitted**.

We report three of our own falsified predictions and six measurement rules that
each changed a *sign* on our own data, including one published number we withdraw
here.

---

## 1. Introduction

A neural surrogate that predicts a flow field in 30 ms is not, by itself, a
faster CFD pipeline. Engineering decisions are made on force coefficients with
error bars, and a surrogate whose drag is 5% off does not replace a solve — it
precedes one. The natural role for it is therefore as an **initial condition**:
if the surrogate's field is closer to the answer than a uniform freestream, the
solver should need fewer iterations to get there, and the saving is free of any
accuracy claim, because the converged answer is the solver's, not the network's.

That argument is clean and, as stated, wrong. This paper is about why, and about
what has to be true instead.

Our starting configuration is deliberately unfavourable to shortcuts. The solver
is unmodified OpenFOAM v2606 `simpleFoam` with the Spalart–Allmaras model. The
mesh is a wall-resolved body-fitted C-grid, 31,700 cells, first cell centre
4e-6 chords off the wall, aspect ratios up to 2e5. The Reynolds number is
3e6 — flight, not a demonstration. The surrogate is a Transolver-style point
model trained on AirfRANS, evaluated on airfoils and angles it did not see. The
metric is not a residual threshold: it is the number of iterations before a force
coefficient enters and stays inside a band around its converged value, because
that is the quantity an engineer actually waits for.

Under those conditions the naive experiment fails, and it fails in a way that is
easy to measure and easy to misattribute. Take one surrogate prediction,
restricted to the boundary layer, and hand it to the solver twice: once evaluated
at the solver's own cell centres, once resampled through a wall-fitted 256x64
grid first. The first is **+33.9%** on total drag. The second is **-58.8%**. Same
network, same weights, same case, same prediction — 93 percentage points, and the
only difference is the representation it travelled through.

That is not an accuracy effect, and the cleanest way to see so is to remove
accuracy from the experiment entirely. Resample the **exact converged field** and
seed with that: -172.6% on a wall-fitted 256x64 grid, -548.4% on a Cartesian
128^2 grid. A perfect answer, stored the way neural-operator outputs are normally
stored, is a *worse* initial condition than a uniform freestream.

The failure is specific and it has a location: the wall. Viscous drag is the
integral of the tangential velocity gradient in the first cell off the surface,
and it is 60–84% of the total drag in these cases. A 16,384-value grid, however
it is arranged, has no sample point 4e-6 chords from a wall. Projecting a field
onto such a grid and back therefore does not *degrade* the near-wall state, it
*deletes* it and replaces it with something near-freestream — overestimating wall
shear by a factor of ~20. We show this is true even when the field being
projected is the exact converged answer, which isolates the projection from any
question about surrogate accuracy: the oracle's own projection carries 1890%
error in the wall gradient, and the trained network queried at the cell centres
carries 54%.

### Contributions

1. **Three necessary conditions**, each isolated by a controlled arm that changes
   exactly one variable, on a common prediction (§5). Mesh-native evaluation, the
   boundary layer alone, and the velocity–eddy-viscosity pair.
2. **The mechanism, measured**: first-cell wall-gradient error for every seed
   construction, including oracle projections that remove the accuracy confound
   entirely (§6).
3. **Three falsified predictions** that a reader would otherwise make: seed the
   pressure (inert), seed what the cold solver is slow at (negative), seed the
   wake (worth +0.5% here, against a 26.3x claim elsewhere) (§7).
4. **An acceptance certificate** that bounds the worst case at (1 + K/N) x cold
   using a rule that never sees a cold run (§8).
5. **A measurement protocol** whose six rules each changed a sign on our own
   data, and under which we withdraw one of our own previously reported numbers
   (§4).

### Is this a fact about your network?

No, and that is the reason the oracle arms are in the study rather than in an
appendix. Every statement about representation is reproduced with the **exact
converged field** in place of the prediction: the projection of a perfect answer
carries 1881% wall-gradient error and costs −206% on drag. Nothing about that
depends on which network produced the field, on how it was trained, or on how
accurate it is. The surrogate supplies the *practical* version of the result; the
oracle supplies the version that cannot be explained away by our model being bad.

### Is the cold baseline a straw man?

The industrial alternative to a uniform freestream is a potential-flow
initialisation, and OpenFOAM ships one. We ran it as an arm: `potentialFoam`
alone is worth **+0.6%** on Cd@1% and +3.3% on lift — inert. A uniform freestream
is therefore not a weak baseline being beaten by a strong method; it is what the
strong classical alternative is also worth on this configuration.

### What this paper does not claim

It does not claim the surrogate is accurate — accuracy is a separate question and
is irrelevant here, since the solver converges to its own answer regardless. It
does not claim a speed record: the largest number in this literature is 26.3x,
and §7.3 explains, with a measurement rather than a rebuttal, why that number and
ours are about different regimes. It does not claim generality beyond 2-D
incompressible steady RANS with one turbulence model on NACA 4-digit sections at
one Reynolds number; §10 says so plainly.

---

## 2. Related work

**Neural surrogates for external aerodynamics.** AirfRANS established the
reference dataset and the point-cloud evaluation protocol. Transolver and its
successors, and point-cloud neural operators such as PCNO (arXiv:2501.14475),
predict fields on the native mesh points rather than on a raster. That capability
is usually presented as an accuracy or memory convenience. **We show it is the
difference between a warm start that works and one that is worse than nothing** —
a surrogate stored as a 128^2 image cannot be used this way at all, and the loss
is not recoverable by refining the raster (§7.1).

**Warm-starting linear and nonlinear solvers with learning.** Learned initial
guesses and preconditioners are an established line: NOWS (arXiv:2511.02481)
learns operator-aware warm starts; Spectrally Safe (arXiv:2606.21828) constrains
learned corrections to preserve convergence; PCGBandit (arXiv:2509.08765) selects
solver configurations online. NVIDIA's hybrid initialisation (arXiv:2503.15766)
is the closest in spirit — a network's prediction as a CFD initial field — and
reports gains on configurations where the near-wall state is not the bottleneck.
Our contribution relative to this line is not the idea of seeding; it is (a) the
demonstration that *how the prediction is represented* dominates *how accurate it
is*, and (b) a per-case acceptance test.

**Wake initialisation.** The largest reported acceleration in this literature —
26.3x iterations, 16.4x wall-clock (arXiv:2501.14699) — comes from initialising
the far wake. We deliberately seed the opposite region. §7.3 reports an oracle
experiment bounding what *any* wake model could buy in our configuration: **+0.5%
on viscous drag**. That is a statement about our regime, not a criticism of
theirs, and it is the reason our restriction to the boundary layer is a finding
rather than a compromise.

**Fallbacks and guarantees.** Preserving worst-case behaviour by falling back to
the classical method when a learned component is untrustworthy is not new;
learning-augmented algorithms with dual warm starts (arXiv:2605.09382) apply
exactly this pattern to linear assignment. **We cite it as prior art for the
pattern.** What is ours is the instantiation for a PDE solver's initial field: a
decision rule that reads only a short probe of the solve it is about to commit
to, a leave-one-case-out calibration, and a measured capture-versus-cost curve
showing that longer probes are monotonically worse.

---

## 3. Setup

**Solver.** OpenFOAM v2606 (ESI), `simpleFoam`, steady incompressible SIMPLEC
(`consistent yes`), Spalart–Allmaras. `nNonOrthogonalCorrectors 2`.
Under-relaxation `U` 0.7, `nuTilda` 0.4 — chosen, not defaulted; see §4 rule 2.
Budget 6000 iterations. Every case is instrumented with both the `forceCoeffs`
and `forces` function objects, so total, pressure and viscous drag are available
separately at every iteration.

**Mesh.** Body-fitted C-grid generated by `blockMesh` from a specification we
control (`solver/cgrid.py`): 31,700 cells, 20-chord far field, stitch-free wake
cut via shared vertex ids, first cell centre ~4e-6 chords, wall-normal grading
to y+ < 1. The same mesh is used for every arm of every experiment, so mesh
quality is never a differential effect.

**Cases.** NACA 4-digit sections at Re = 3e6, `u_inf = 1`, kinematic pressure.
The core study is 5 cases (0012@4°, 2412@2°, 0015@6°, 0012@0°, 2415@5°). The
generality sweep is 13 cases spanning 0°–12°, into incipient separation `[[B1]]`.

**Surrogate.** A Transolver-style point model, (B,N,7) -> (B,N,4), trained on
AirfRANS under the conventions of the companion paper (dimensional fields,
nu = 1.56e-5, Re from |u_in|). It is queried directly at the C-grid cell centres —
`solver/surrogate_seed.py` — with no intermediate representation of any kind. Its
training `sdf` distribution is centred on 0.23 chords, which is why seeds are cut
off at 3.5 chords and why the outer field is left cold.

**Arms.** Every experiment carries a **converged-field oracle** as a control. The
oracle is the cold run's own converged solution, re-injected as an initial
condition, and it must post a large positive saving before any other arm in that
experiment is read. Across twelve experiments the control has read +68% to
+99.9%. An experiment whose control fails is a broken measurement, and we say so
rather than reporting its other arms.

---

## 4. How to measure a warm start without fooling yourself

This section is a contribution, not preamble. Each of the six rules below changed
a **sign** on this project's own data. All six are implemented in
`solver/scoring.py`, each with a test that names the mistake it prevents.

1. **A residual threshold measures a convergence rate only while the residual is
   still falling.** Report every threshold as a multiple of the run's residual
   floor and refuse to read anything below ~5x. The same arm read +15%, +31% and
   +13% at 1.9x, 1.3x and 0.9x the floor — noise presented as a trend.
2. **The floor itself is usually an artifact, so find out which one.** Ours was
   under-relaxation at 0.9 with SIMPLEC, not the 2e5-aspect-ratio cells:
   `U` 0.7 / `nuTilda` 0.4 moved the floor 30x, from 1.1e-5 to 3.5e-7. Ten
   other variants — longer budgets, tighter inner tolerances, a better wake mesh
   — bought nothing.
3. **An arm that never reaches the target must be bounded at its full budget, not
   dropped.** Dropping non-finishers turned a true -199.4% into -31.2%: the
   failing arm was rewarded for failing.
4. **Score every arm against one external reference**, never against its own
   final value. Grading the oracle against itself turned a +73.5% control into
   +1.0%.
5. **Only arms whose coefficient has stopped moving may define that reference.**
   Peak-to-peak over the last tenth of the run, against a quarter of the band. A
   single diverged arm previously set the reference and condemned an entire
   table; restricting the reference to settled arms moved the spread from 3.104%
   to 0.334%.
6. **`nNonOrthogonalCorrectors` multiplies the pressure history.** Parse per
   `Time` block; zipping fields by index across a log made pressure lag velocity
   3:1 and moved every shallow number.

One consequence of rule 5 is worth stating even though it is small here. Because
the reference is a median over the arms that have settled, *adding* an arm to a
tree re-scores every arm already in it. **A number must therefore name the arm
set it was computed over**, and ours does (`--drop-arm` declares it). We measured
the sensitivity rather than assuming it away: removing our most recently added
arm moves every headline number by at most 0.5 percentage points.

**The primary metric.** `iterations_to_force_band`: the first iteration after
which a coefficient stays within +/-b of the reference for the rest of the run.
Saving is 1 - warm/cold, bounded at the budget. We report b = 1%, 0.5%, 0.2%
and state which bands are readable.

**Readability.** A band is readable only if the settled arms agree about the
converged value to well inside it (rule 5). This is a public criterion that
deletes our own numbers: at b = 0.5% and 0.2% the total-drag rows are unreadable,
and **Cd@1% is the only readable total-drag row in the core study.**

> **A withdrawal.** A previously recorded +41.8% at Cd@0.5% was read against a
> reference that a diverged arm had moved. On the settled reference it is
> **-7.4%**, and the row is unreadable in any case. The +33.9% at Cd@1% is
> unaffected. We report this because a protocol that only ever deleted
> inconvenient numbers would not be a protocol.

**Statistics.** Savings 1 - warm/cold are left-skewed, so we report percentile
bootstrap 95% CIs (10,000 resamples) rather than t-intervals, plus an exact
two-sided sign test. At n = 5 the smallest attainable p is 0.0625; we state this
wherever n < 6 rather than reporting a non-significant p as if the test could
have succeeded. `[[B1]]` raises n to 13, where the sign test can reach p ~ 0.0002.

---

## 5. Three conditions, each with its own control

`scripts/mesh_native_probe.py`. Five cases, **one prediction**, one variable
changed per arm; all twenty solves complete. Scored over the thirteen-arm
`repr3` set with `oracle_wake` dropped (§4); including it moves no entry below by
more than 0.5 points.

| arm | what it hands over | residual 5e-6 | **Cd@1%** | Cl@1% | Cd_v@1% |
|---|---|---:|---:|---:|---:|
| `nf_bl` | u, v, nut in the BL, mesh-native | < -80.5% | **+33.9%** | +10.1% | +14.6% |
| `nf_bl_proj` | the same, resampled through 256x64 | +22.1% | **-58.8%** | +25.4% | +7.7% |
| `nf_bl_nut` | eddy viscosity only | +1.2% | **-293.2%** | **+41.1%** | **+42.4%** |
| `nf_bl_vel` | velocity only | -8.2% | -40.3% | -10.3% | -4.9% |

### 5.1 Condition 1 — evaluate at the solver's cell centres

`nf_bl` and `nf_bl_proj` differ in exactly one respect: whether the identical
prediction was resampled through a wall-fitted 256x64 grid before being written.
Total drag swings **93 percentage points**, and the seed's first-cell
wall-gradient error rises from 54% to 1583% — landing in the same place as the
matched oracle arm, `fitted_bl`, which is the *exact converged field* through the
same 256x64 round trip and the same boundary-layer mask (1881%). The round trip
destroys the wall gradient regardless of the quality of the field that entered
it: a network prediction and a perfect answer come out of it indistinguishable
where it matters.

The signs that go the other way confirm the mechanism rather than complicating
it. Resampling *helps* the residual (+22.1%) and *helps* lift (+25.4%), because a
resampled field is smoother — it changes less per iteration — and lift is
pressure-dominated. **A study that scored residuals alone would have concluded
the projection was the better seed.**

### 5.2 Condition 2 — hand over the boundary layer only

The region axis is controlled the same way as the resampling axis: both arms are
the same network, both mesh-native, differing only in whether the handover is
masked to the boundary layer. Cd@1%, five cases, cold = 805 iterations, oracle
control **+92.1%**:

| arm (network, mesh-native) | region handed over | Cd@1% |
|---|---|---:|
| `nf_bl` | boundary layer only | **+33.9%** |
| `nf_mesh` | the whole field | **< -568.3%** |

Handing over the outer field fails because the model is extrapolating there — its
training `sdf` distribution is centred on 0.23 chords while the C-grid reaches 20
— and an extrapolated outer field is an inconsistent boundary condition for a
boundary layer the solver is still computing.

**Together with §5.1 this gives two controlled contrasts sharing a common arm**
(`nf_bl`), one per axis, each changing a single variable. Neither factor alone is
sufficient: mesh-native evaluation of the whole field is the worst arm in the
study, and boundary-layer restriction under resampling is still negative.

We deliberately do **not** present these as a 2×2. The fourth corner — a network
prediction of the whole field, resampled — was never run, and the arms that would
fill it come from a different population (oracle rather than network). A table
whose cells mix provenance would read as a design when it is a gap.

**The oracle bound on the resampled row**, which removes the accuracy confound
entirely, is reported separately and is the stronger statement:

| arm | field | representation | Cd@1% |
|---|---|---|---:|
| `fitted_256x64` | exact converged answer | wall-fitted 256x64 | -172.6% |
| `fitted_bl` | exact converged answer, BL only | wall-fitted 256x64 | -206.1% |
| `cartesian_128` | exact converged answer | Cartesian 128^2 | -548.4% |

Even a perfect field, resampled, costs the solver more than starting it cold.

### 5.3 Condition 3 — velocity and eddy viscosity together

This condition was not anticipated; it came out of splitting the channels.

Eddy viscosity alone is simultaneously the **best** arm in the study on viscous
drag (+42.4%) and lift (+41.1%), matching the oracle projection, and the **worst**
on total drag (-293.2%). Velocity alone is bad at everything. Only the pair is
positive on total drag.

The reason is the structure of the Spalart–Allmaras production term, which is
driven by the strain rate. An eddy viscosity handed over without the velocity
field that generated it is inconsistent with the strain the solver computes from
the field it actually has; the momentum sink is therefore wrong, the pressure
field must reorganise, and `Cd_p` — 16–40% of the drag here — is destroyed. The
shear-driven quantities do not care, and speed up.

**Consistency between handed-over channels is not a detail of the recipe. It is
most of it**, and §7.3 finds the same lesson again at a completely different
length scale.

### 5.4 A common-mode limitation, checked rather than assumed

`write_case` floors `nuTilda` at its freestream value, which clips 37% of the
boundary-layer cells. That sounds like it could produce these swings. It cannot:
the clipped cells hold values ~88x below the peak, so the floor removes only
**2.1–2.3%** of the eddy-viscosity field's energy, and it applies identically to
every arm including the oracle. It is a common-mode limitation of the study, not
a differential effect that could move an arm from +33.9% to -293.2%.

### 5.5 The residual objection

`nf_bl` is **negative on the residual at every depth** and positive on drag;
`nf_bl_proj` is the exact inverse. A reviewer is right to look hard at that. Our
answer has three parts.

1. **The residual is not the objective.** Nobody runs a RANS solve to obtain a
   small residual; they run it to obtain a force coefficient that has stopped
   moving. The residual is a proxy, and this study measures the proxy failing:
   one seed is +22.1% on the proxy and -58.8% on what the proxy stands for. We
   report both, always.
2. **We know what the residual is rewarding.** The `Ux` residual measures how
   much the field changes per iteration, so it rewards smoothness. A resampled
   field has had its near-wall structure interpolated away and therefore changes
   less, while carrying 1583% error in the wall gradient. The residual is not
   being fooled at random — it is faithfully measuring something that is not drag.
3. **The choice of metric was pre-committed and cuts against us.** The force
   metric replaced the residual metric for the reason in §4 rule 1, which
   predates this arm; and the readability rule then rejected rows we would rather
   have quoted, including the withdrawal above.

The honest residue stays in the paper: **on the residual, `nf_bl` is worse than a
cold start.** §8 is what makes that survivable — the same 25-iteration gate that
bounds drag bounds the residual metric's worst case at -7.6%.

---

## 6. The mechanism

`scripts/seed_gradient_diagnostic.py`, six cases. For each seed *as the solver
received it*, the error in the first-cell tangential velocity gradient du_t/dy
— the quantity viscous drag integrates — measured along the outward wall normal
(not by nearest neighbour: near-wall cells here are ~2500x wider than they are
tall, so nearest-neighbour sampling would silently sample the wrong cell).

| seed | wall-gradient error | BL velocity error |
|---|---:|---:|
| cold start (uniform freestream) | 2851% | 90.4% |
| Cartesian 128^2 projection of **the exact answer** | 1695% | 56.6% |
| wall-fitted 256x64 projection of **the exact answer** | 1890% | 51.0% |
| **trained surrogate, queried at the cell centres** | **54%** | **15.2%** |
| oracle (the exact answer itself) | 0% | 0% |

Read the middle two rows carefully: **the field being projected is the exact
converged answer.** Any question of surrogate accuracy is removed. Both
projections still overestimate the wall shear by a factor of ~20, because they
place near-freestream velocity at a cell centre 4e-6 chords off the wall — no
16,384-value grid has a station there. Against a cold start they remove between a
third and a half of the error in the quantity that decides drag. The network,
evaluated pointwise at the solver's own cell centres with no resampling at all,
removes 98% of it.

This is not an accuracy story with a smoothness twist. An earlier version of this
work hypothesised that projected seeds were *accurate but rough*, and that the
solver preferred smooth. That hypothesis is dead: the projection is 35x **less**
accurate on the wall gradient, not accurate-but-rough.

**The wall-fitted projection is the important control.** One might assume the
problem is the Cartesian raster, and that a body-fitted 256x64 output layout
would fix it. It does not — 1890% versus 1695%, i.e. slightly worse — because at
equal output budget (16,384 values) a wall-fitted grid buys angular resolution,
not wall-normal resolution, and wall-normal resolution is the entire question.

---

## 7. What does not work, and why that matters

Three predictions a careful reader would make, each falsified by measurement. We
keep them because a recipe with three falsified neighbours is a recipe; one
without them is an anecdote.

### 7.1 Refining the raster does not help

A uniform Cartesian seed fails at Re = 3e6 **at any resolution we can reach**:
128^2 to 421^2 is flat. It is not a training problem — the arm under test is the
exact converged answer. Resolving one cell across the inner layer would need
N ~ 11,800, which is 28x beyond what AirfRANS itself stores.

We also tested the obvious dimensionless criterion. delta/h (boundary-layer
thickness over cell size) looks clean on a Reynolds sweep — a sign change at
delta/h = 2.0 — and then fails on the grid axis. Across that sweep delta/h moves
5x while the viscous ratio y(y+=30)/h collapses **1660x**. The near-wall viscous
scale, not the boundary-layer thickness, is what decides whether a grid
representation can carry a warm start.

Warm-starting at moderate Reynolds does work: +14.4% at Re = 1e4. A previously
recorded Re = 1e3 result mostly did not survive a parser fix (+58% -> +8.1%) and
those runs sit 3–9x above their residual floor; they are excluded pending
re-measurement on the relaxed settings `[[B5]]`.

### 7.2 Seeding what the cold solver is slow at makes it slower

Iterations to settle within 1% of converged, cold versus seeded with the exact
field:

| quantity | cold | oracle seed | share of Cd |
|---|---:|---:|---:|
| viscous drag `Cd_v` | ~700 | ~53 | 60–84% |
| lift `Cl` | ~950 | 1 | — |
| pressure drag `Cd_p` | ~1850 | 1–2 | 16–40% |

A cold solver is slow at pressure and comparatively fast at the near-wall
velocity gradient; a surrogate is the reverse. The obvious inference — hand over
the pressure, keep the near-wall velocity — is what this project pursued for two
working sessions, and it is **false**:

- `fitted_p` (pressure only) is **inert**: +0.1% on every metric at every depth.
- `composite` (potential-flow pressure + surrogate boundary layer) is
  **negative**: -320.0%.
- `potential` (`potentialFoam` alone — the free industrial baseline) is inert on
  drag (+0.7%) and mildly positive on lift (+3.3%).

The reason is SIMPLE's structure: pressure is *recomputed* from continuity given
the velocity field, so a pressure seed inconsistent with `U` is overwritten
within a few iterations. Only fields entering the momentum and turbulence
transport equations carry information forward.

### 7.3 The wake is worth +0.5% here

The largest acceleration in this literature initialises the far wake
(arXiv:2501.14699; 26.3x iterations). Every seed here does the opposite. The
obvious next move is to compose the two — so we bounded the payoff before
building anything.

`scripts/wake_probe.py` seeds the **exact converged field** across the whole
downstream region (37.5% of cells, 21.6% of them fully), which bounds what any
wake model could ever buy on these cases. Five cases, Re = 3e6:

| metric | oracle wake seed | 95% CI | per case |
|---|---:|---|---|
| **Cd_v@1%** | **+0.5%** | [+0.4, +0.7] | +0, +0, +1, +1, +1 |
| Cd_v@0.5% | +1.3% | [+1.0, +1.5] | +1, +1, +1, +1, +2 |
| Cd@1% | -242.1% | [-676, -0.3] | -1098, -103, -19, -4, +13 |
| Cl@1% | -22.0% | [-68, +1.6] | -92, +0, +2, +2 |

**The perfect wake seed is worth half a percent.** On a 2-D attached-flow airfoil
at Re = 3e6 on a 20-chord C-grid, the solver is not spending its time
developing the wake; it is spending it on the near-wall state, exactly where §7.2
located it. The 26.3x figure is therefore a fact about that configuration,
geometry and cold baseline — not a better method — and our restriction to the
boundary layer is a finding rather than a compromise.

Note also that the oracle wake seed is *harmful* on total drag. Handing over a
downstream field while leaving the boundary layer cold is another inconsistent
pair, which is §5.3's lesson at a length scale two orders of magnitude larger.

---

## 8. An acceptance test that bounds the worst case

Warm starting is only adoptable if a bad seed cannot cost more than not seeding.
Ours can: ungated across 5 cases x 11 strategies, the mean is -168.9% and the
worst seed is -1169.6% on Cd@1%.

**The rule.** Run K probe iterations from the seed. Read two scalars from the
residual history — the level log10 r_K and the drop log10 r_K - log10 r_0. Either
continue the solve, or discard the seed and start cold. Accepting costs nothing
extra, because the probe iterations *are* the first K iterations of the warm
solve; rejecting costs K + cold. The worst case is therefore
(1 + K/N_cold) x cold **by construction**, whatever the seed does.

**The rule never observes a cold run.** In production there isn't one — that is
the entire point of warm starting. The threshold is calibrated leave-one-case-out
and applied to the held-out case.

K = 25 (~3% of a cold solve):

| metric | ungated mean | worst seed | **gated mean** | **gated worst** | harmful admitted |
|---|---:|---:|---:|---:|---:|
| Cd@1% | -168.9% | -1169.6% | **+3.7%** | **-5.8%** | 0/32 |
| residual 5e-6 | -170.8% | -1449.3% | +1.8% | -7.6% | 0/36 |
| Cl@1% | -21.5% | -671.9% | +1.9% | -100.0% | 1/22 |
| Cd_v@1% | +23.5% | -8.6% | +23.5% | -8.6% | 9/10 |

**Where the gate fails, stated rather than buried.** On lift it admits one
harmful seed in 22 and its gated worst case is −100%. Lift converges by a
different route — it is pressure-dominated, and §5.1 showed that resampling
*helps* lift while destroying drag — so a probe reading the momentum residual is
weakly informative about it. The gate should be applied per quantity, and on the
quantity a user cares about; we do not claim it is a universal filter. Its
worst-case bound of (1 + K/N) x cold survives regardless, because that bound is
arithmetic, not statistical: it holds for any seed, any metric, and any threshold,
including a threshold that admits every seed.

The gate is not what makes warm starting fast; it is what makes it deployable. It
is conservative by construction, capturing only 17–24% of what a gatekeeper with
foreknowledge would achieve on the metrics where most seeds are harmful, and it
is nearly a no-op (97% capture) on viscous drag, where 40 of 50 seeds already
help. Longer probes are monotonically worse: by K = 400 the probe cost alone
(-49.6%) exceeds anything the decision can recover.

---

## 9. Wall-clock

Iterations are the honest unit for a mechanism, but the claim an engineer cares
about is seconds, and a mesh-native seed costs seconds a projected one does not:
wall-distance computation, surrogate inference, masking, and writing the case.

`scripts/wallclock_control.py` charges each preparation stage to the arms that
need it and reports end-to-end seconds, running serially and refusing to start
while any other solver is up. At n = 1: **+41% iterations -> +30% seconds**, with
a per-iteration penalty of 1.14x (a contended box had suggested 1.62x, which is
why the exclusivity check exists). `[[C]]` extends this to n = 5.

---

## 10. Limitations

- **2-D, incompressible, steady, one turbulence model.** Spalart–Allmaras only.
  Whether the three conditions survive a two-equation model is untested `[[B2]]`.
- **One solver and one mesh family.** OpenFOAM SIMPLEC on a C-grid we generate.
  Nothing here has been tried on an unstructured or commercial solver.
- **One Reynolds number for the headline.** Re = 3e6; the moderate-Re result is
  n = 1 per Reynolds number and one low-Re claim is withdrawn pending
  re-measurement.
- **Bands below 1% need a longer budget.** At b = 0.5% and 0.2% the total-drag
  rows are unreadable at 6000 iterations, and we do not quote them `[[B4]]`.
- **`nuTilda` is floored at freestream** on write, a common-mode limitation
  quantified in §5.4.
- **naca4412@3° is excluded, always with cause**: no unique steady fixed point
  at this budget (arms 7% apart in final Cd; floor 1.6e-5 against 6e-8–1.7e-6
  elsewhere). This is a real warning about the separated regime, which the
  generality sweep enters deliberately.
- **The residual metric is negative for the recommended arm** (§5.5).

---

## 11. Conclusion

A neural surrogate can accelerate a production RANS solver, and the conditions
under which it does are narrow, checkable, and explicable by a single measured
quantity. It must be evaluated at the solver's own cell centres, because the
first-cell wall gradient — which no practical grid representation can carry — is
what viscous drag integrates. It must hand over the boundary layer and nothing
else, because the outer field is where the model extrapolates. And it must hand
over velocity and eddy viscosity together, because a turbulence field
inconsistent with its own strain is worse than no turbulence field at all.

Read as advice rather than as a result, the paper is short: **query your surrogate
where the solver lives, give it only the region your surrogate was trained on,
give it whole physics rather than single channels, and spend 3% of a solve
checking before you commit.**

---

## Appendix A — reproduction

Every table above is regenerated by one command from a checkpointed tree; see
`docs/PLANS.md` §3 for the mapping from table to script and `results/*.json` file.

## Appendix B — the scoring rules as code

`solver/scoring.py`: `has_settled`, `settled_reference`, `bounded_saving`,
`shared_reference`, `reference_spread`, `readable_depth`, `bootstrap_ci`,
`sign_test`, with `MIN_DEPTH_OVER_FLOOR = 5.0` and `MAX_SPREAD_FRACTION = 0.5`.
Each has a test named for the mistake it prevents.
