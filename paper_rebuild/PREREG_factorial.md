# Pre-registration: what an AirfRANS force metric can credit beyond the case parameters

Committed before any cell below is run. Later changes go in `DEVIATIONS.md` with both results.
Unit: case. Intervals: paired BCa case bootstrap, B = 10,000, seed 0. Seeds enter as a random
effect: seed-averaged per case for the primary estimate, per seed for the secondary estimate.

## Outcomes (error scale, primary)
- **Drag:** per-case |C_D − C_D^official| in drag counts (1 count = 1e-4), and log(1 − ρ_D).
- **Lift:** |C_L − C_L^official| and log(1 − ρ_L).
- **Margin of practical equivalence:** 5 drag counts on the mean absolute error. This is a
  convention taken from DPW-VI code-to-code scatter. For log(1 − ρ) the margin is log 1.25.
- ρ itself is reported, but no claim rests on a ρ difference.

## Reference (R)
- The frozen reference from `analysis/null_reference.py`, Q and K families chosen by
  training-CV. It is applied unchanged to every split below.
- On geometry-held-out splits, the family is still chosen by training-CV, never by test.

## Cells

### (a) Reference-side power check (CPU)
The reference is refit on the official split with thickness withheld (R−t), and separately
with camber and camber position withheld (R−c).
- **Question.** Can a model that reads geometry show a positive Δ at all against the full
  reference, which gets the same information exactly? The check measures how much the geometry
  digits carry.
- **Pass (the design has power).** Withholding raises the reference's drag MAE by at least
  5 counts.
- **Fail.** AirfRANS drag is determined by (U, α) almost alone. Geometry-learning claims on
  AirfRANS forces are then untestable, and we state that.
- **Reading.** R−t and R−c become the *constructed positive-control references*. A model that
  reads geometry must beat them by more than the margin, or Δ has no power to detect geometry
  learning.

### (b) Official-integrator forces for existing field models — gated
- **Gate first.** Ground-truth native fields → `airfrans.Simulation.force_coefficient` must
  reproduce the official labels: median |ΔC_D| < 1 count and ρ_D > 0.999 over the 200 test
  cases.
- **If the gate fails,** no model forces are read. The path is debugged, and the failure is
  recorded.
- **Arms.** Transolver seeds 0–4 (`checkpoints/v2_transolver/seed{k}.pt`), predicted native
  fields → the same integrator.
- **Outcome.** Δ_F = error(Transolver, integrated) − error(R), paired per case.

### (c) Readout vs representation (GPU, minutes)
Direct scalar heads (C_D, C_L) are fit on frozen Transolver case embeddings (mean-pooled last
hidden layer). Two heads:
- (c1) embeddings only (geometry + U, α as Transolver sees them);
- (c2) embeddings + the 7 case parameters.

Head family: the same K and Q families as R, training-CV. Seeds 0–4.

**Reading.**
- (c1) ≈ R, while (b) is worse than R: the field-to-force gap is a *readout* effect, not
  missing information.
- (c1) worse than R, and (c2) ≈ R: the representation lacks what the parameters give.

### (d) Published flexible reference
MMGP re-run from its released code on the official split, if the code runs on this machine
within one day of effort. Otherwise it is reported as not reproduced, with the reason.

### (e) Geometry-held-out resplits (GPU, overnight, resumable; runs last)
- Splits from the pooled 1,000 cases:
  - 4-digit→5-digit and 5-digit→4-digit series;
  - top thickness decile held out.
- Transolver is retrained with the `run_v2` recipe, seeds 0–2.
- Cells (b) and (c1) are repeated on each split, and R is refit on each training set.

## Controls
- **Negative.** Training labels are permuted across cases before fitting R and the (c) heads.
  Their errors must match the intercept-only error within the margin. If they do not, the
  pipeline leaks.
- **Positive.** Cell (a)'s R−t and R−c.

## What would refute the thesis on AirfRANS

The thesis: *the official AirfRANS force metric cannot credit geometry or flow learning beyond
the case parameters.* Any of these refutes it:
1. A geometry-input model (b or c1) beats the full reference R by more than the margin on the
   official split.
2. A geometry-input model loses less than R on a geometry-held-out split, by more than the
   margin. That would mean it generalises across geometry better than the parameter map.

If neither occurs and (a) passes, the thesis stands on AirfRANS. If (a) fails, the thesis is
untestable on AirfRANS forces, and the paper reports that instead.
