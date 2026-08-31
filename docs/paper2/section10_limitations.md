## 10. Limitations

### On the criterion and the closed form

- **The closed form assumes an equilibrium wall profile.** It uses the law of
  the wall with standard smooth-wall constants (κ = 0.41, B = 5.0), unmodified.
  Under strong adverse pressure gradients, separation, roughness, compressibility
  or heat transfer the profile departs from it, and the predicted factor should
  be read as indicative rather than as the 13%-accurate number it is on the
  attached cases measured here.
- **It is quantitative only while the representation's first station lies inside
  the boundary layer.** Above that the velocity has saturated at freestream and
  the expression becomes an upper bound; we report the regime alongside every
  number rather than leaving a reader to infer it.
- **It is necessary, not sufficient** (§7.4). A representation that fails the
  check can be ruled out for free; one that passes still has to satisfy the
  region and channel conditions, and `nf_mesh` is the counterexample the study
  carries — perfect gradient retention, worst arm in the study.
- **The 13% agreement is a systematic over-prediction, not scatter.** Six cases
  give 1.13 ± 0.02. We report the bias rather than absorbing it into a fitted
  coefficient, because a fitted coefficient would make the expression a
  description of these six cases instead of a prediction.

### On the repair

- **It assumes the same equilibrium profile the closed form does**, and it
  assumes the representation's first station carries a usable velocity. Where
  the boundary layer is separating, the inverted `u_τ` is not meaningful and the
  repair has no basis. Every case here is attached to incipiently separated;
  **the repair is untested in separation and we do not claim it there.**
- **It rebuilds `nut` from a damped mixing length**, which is what
  Spalart–Allmaras relaxes to in the log layer but is not what SA solves. This
  is a seed, not a model, and the solver replaces it — but it is an assumption
  and it is the channel §5.4 shows is least forgiving.
- **It repairs a representation; it does not repair a bad prediction.** If the
  surrogate's value at the first station is wrong, the repair faithfully
  propagates that error down to the wall.

### On the study

- **2-D, incompressible, steady, one turbulence model.** Spalart–Allmaras only.
  Whether the three conditions survive a two-equation model is untested, and
  k-ω SST is the obvious next experiment.
- **One solver and one mesh family.** OpenFOAM SIMPLEC on a C-grid we generate.
  Nothing here has been tried on an unstructured or commercial solver.
- **One Reynolds number for the headline** (Re = 3·10⁶). Because the criterion
  is expressed in wall units, a Reynolds sweep is the natural test of it and is
  not done: the closed form predicts the damage *grows* at lower Reynolds on the
  same mesh, since the first cell moves deeper into the linear sublayer while
  the representation's station stays in the log layer. **That prediction is
  stated here and untested.**
- **Bands below 1% need a longer budget.** At 0.5% and 0.2% the total-drag rows
  are unreadable at 6000 iterations on the corpus.
- **`nuTilda` is floored at freestream on write**, a common-mode limitation
  quantified in §5.6 — it removes 2.1–2.3% of the eddy-viscosity field's energy
  in the boundary layer and applies identically to every arm including the
  oracle.
- **Three cases have no unique steady drag at this budget**, and they share a
  shape: `naca4412@3°`, `naca4415@2°`, `naca4415@4°` — thick cambered sections
  at low incidence, carrying the corpus's worst residual floors. We do not know
  whether this is genuine non-uniqueness or a budget we did not pay, and we do
  not claim to.
- **The residual metric is negative for the recommended arm** (§5.6). We report
  it every time we report the force metric.
- **Wall-clock is n = 5 and was measured exclusively; the placement, repair and
  sequencing trees were not.** Those three were run concurrently, so iterations
  from them are sound — iteration counts are contention-proof — and **no
  wall-clock number is quoted from them.**
- **The acceptance test fails on lift** (§8): its gated worst case equals its
  ungated worst case, so it admits the single worst lift seed in the study. The
  `(1 + K/N)` bound is arithmetic and survives that, and §7.2 explains why lift
  behaves differently from drag.
