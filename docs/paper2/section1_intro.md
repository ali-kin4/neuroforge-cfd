## 1. Introduction

A neural surrogate that predicts a flow field in 30 ms is not, by itself, a
faster CFD pipeline. Engineering decisions are made on force coefficients with
error bars, and a surrogate whose drag is 5% off does not replace a solve — it
precedes one. The natural role for it is therefore as an **initial condition**:
if the surrogate's field is closer to the answer than a uniform freestream, the
solver should need fewer iterations to get there, and the saving is free of any
accuracy claim, because the converged answer is the solver's, not the network's.

That argument is clean and, as stated, wrong. This paper is about why, about the
one number that decides it, and about what to do when that number comes out
badly.

Our starting configuration is deliberately unfavourable to shortcuts. The solver
is unmodified OpenFOAM v2606 [11] `simpleFoam` with the Spalart–Allmaras model
[10]. The mesh is a wall-resolved body-fitted C-grid, 31,700 cells, first cell
centre 5·10⁻⁶ chords off the wall, aspect ratios up to 2·10⁵. The Reynolds number
is 3·10⁶ — flight, not a demonstration. The surrogate is a Transolver-style point
model trained on AirfRANS, evaluated on airfoils and angles it did not see. The
metric is not a residual threshold: it is the number of iterations before a force
coefficient enters and stays inside a band around its converged value, because
that is the quantity an engineer actually waits for.

### The failure, with accuracy removed from the experiment

Take the **exact converged flow field** — not a prediction, the answer itself —
store it the way neural-operator outputs are normally stored, and hand it back to
the solver as an initial condition. On a 128² Cartesian raster, total-drag
convergence is **548% slower** than starting from uniform freestream. On an
equal-budget wall-fitted 256×64 grid, **173% slower**. A perfect answer, stored
in the field's standard output format, is a *worse* initial condition than no
initialisation at all.

Because the field being stored is exact, nothing about this depends on how
accurate any network is, how it was trained, or which architecture it uses. The
defect is in the **representation**.

### It is not resolution, and the arithmetic says so

The obvious response is to spend more values. We measured that too — a ladder of
uniform rasters from 128² to 421² of the same exact field — and the saving is
flat and negative throughout (§7.1).

The reason is that the failure is not one of resolution but of **placement**, and
it is arithmetic rather than statistical. When a field is resampled through a
grid, every mesh cell nearer the wall than the grid's first wall-normal station
receives the value belonging to that station; the representation holds no sample
in between. Viscous drag is the integral of the tangential velocity gradient in
the first cell — 60–84% of total drag in these cases — so the reconstructed
gradient is not degraded but *replaced*, and overestimated by

> **G = u⁺(y₁⁺) / u⁺(y_c⁺)**

where `y₁⁺` and `y_c⁺` are the wall-unit positions of the representation's first
station and the mesh's first cell centre. This is the law of the wall and nothing
else; **it contains no fitted parameter**. Against six converged cases it predicts
23.7× where 21.0× is measured, a ratio of 1.13 ± 0.02 (§6.2).

Two things follow immediately. `u⁺` grows *logarithmically*, so a 10.8-fold
increase in stored values moves the predicted damage from 36.6× to 35.3× — which
is why the resolution ladder is flat, and why it is flat by a computable amount.
And the fix is a grading choice rather than a budget: a 64-level geometric stack
whose first station lies inside the first cell needs a growth ratio of 1.214, an
ordinary mesh. **A 512² raster holds 262,144 values and still fails; a wall-fitted
grid of 8,192 values passes** (§6.5).

### Contributions

Ordered by what we think survives, not by what is largest.

1. **A representational failure of the field's standard output format, measured
   with the accuracy confound removed** (§1, §7.1). The exact converged answer,
   stored as a raster, is a catastrophically bad initial condition, and refining
   the raster cannot fix it.
2. **A closed-form criterion for it, with no fitted parameter**, computable from
   a mesh and an output format before any solve is run, and accurate to 13% over
   six cases (§6.2). We ship it as a command-line tool so it can be run on a mesh
   and a format that have nothing to do with this study (§6.4).
3. **The criterion holds across Reynolds number**, checked from Re 10³ to 3·10⁶
   at no compute cost, including its counterintuitive prediction that the same
   representation on the same mesh does *more* damage at lower Reynolds — and a
   measured statement of the regime where it stops being quantitative (§6.6).
4. **A demonstration that the criterion is necessary and not sufficient**, by
   building the repair it implies and watching it fail (§5.5). Because the
   damage is known in closed form it can be removed: inverting a wall function
   at the representation's own first station restores the first-cell gradient
   from 1583% error to 42.5% — better than the mesh-native prediction's 53.7% —
   using only what the representation already carries. **The convergence saving
   does not follow.** A seed with a better wall gradient than the recommended one
   converges 79 points worse. We predicted otherwise, in writing, before running
   it.
5. **A recipe with three necessary conditions**, each isolated by a controlled
   arm changing one variable on one prediction, none sufficient alone
   (§5.2–§5.4), and **generality at n = 13** — +18.4% on viscous drag, 13/13
   cases, p = 0.0002, with a passing oracle control and a null negative control
   (§5.1).
6. **A comparison with the classical warm start** the machine-learning
   initialisation literature does not make: grid sequencing, with its coarse
   solve charged (§5.7). It makes a **better seed than ours** — +75.9% on viscous
   drag against +14.6%, exactly as the criterion predicts for a coarsened
   body-fitted mesh — and still loses, because the coarse solve costs 1486
   fine-equivalent iterations against a cold run of 696. **The learned seed's
   advantage is price, not quality.**
7. **An acceptance certificate** bounding the worst case at (1 + K/N) × cold with
   a rule that never sees a cold run (§8), and **three falsified predictions** a
   reader would otherwise make (§7).

### What this paper does not claim

**It does not claim a speed record, and it is not competing for one.** The
largest number in this literature is 26.3× [8]; ours is +18.4%. §2.1 and §7.3
explain by measurement rather than rebuttal why that number and ours concern
different regimes — an oracle seed of the entire wake buys +0.5% here, and their
own result is reported as conditional on an accurate near-body field supplied
separately. A reader who wants the fastest warm start should read that paper; a
reader who wants to know whether *their* surrogate can warm-start *their* solver,
and to find out before paying for it, should read this one.

It does not claim the surrogate is accurate — accuracy is a separate question and
irrelevant here, since the solver converges to its own answer regardless. And it
does not claim generality of the *demonstration* beyond 2-D incompressible steady
RANS with one turbulence model on NACA 4-digit sections. The **criterion** is more
general than that by construction, since it is a statement about wall units and
sampling positions rather than about any solver or model, but we measured its
consequences in one place and §10 says so plainly.
