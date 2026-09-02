## 11. Conclusion

Whether a neural surrogate can accelerate a production RANS solve is decided
before any solve is run, by a property of the surrogate's **output format** that
has nothing to do with its accuracy: whether that format holds a sample inside
the solver's first cell.

The evidence is arithmetic before it is empirical. A resampled field hands every
cell nearer the wall than the representation's first station the value belonging
to that station, so the first-cell wall gradient — which viscous drag integrates,
and which is 60–84% of drag here — is overestimated by `u⁺(y₁⁺)/u⁺(y_c⁺)`. That
expression has no fitted parameter, predicts the measured damage to 13% over six
cases, explains why refining a raster is flat to within a computable amount, and
holds across three and a half decades of Reynolds number, including in the
direction most practitioners would guess wrong: on the same mesh, lower Reynolds
is **worse**.

Read as advice rather than as a result, the paper is short:

> **Check where your surrogate's output puts its first sample.** If nothing lands
> inside the solver's first cell, the seed will misreport the wall gradient by
> roughly `u⁺(y₁⁺)/u⁺(y_c⁺)`, and no amount of extra resolution will fix it — a
> 512² raster holds 262,144 values and still fails, while a wall-fitted grid of
> 8,192 passes. Fix it by grading, by predicting on the solver's own points, or
> by repairing the profile below the first station from a wall function. Then
> give the solver only the region your surrogate is trusted on, hand over whole
> physics rather than single channels, and spend 3% of a solve checking before
> you commit.

**The size of the effect is modest and we say so.** The recommended seed
accelerates viscous-drag convergence by 18.4% across thirteen cases, winning
every one of them (p = 0.0002), with a converged-field control at 93.6% and a
null negative control. That is far short of the 26.3× reported elsewhere for a
different regime, and short of the 11–16× reported for a mesh-native operator on
a comparable configuration. What is durable here is not the number.

What is durable is that the criterion **predicts things it was not derived from**:
the flat resolution ladder, quantitatively; the ordering of the published
warm-start literature by representation and seeded region; the behaviour of grid
sequencing, a classical method with no network in it; and the Reynolds
dependence, on data that already existed. And it is durable that the paper's own
explanations were tested rather than asserted — three predictions a reader would
make are falsified here, one of them our own account of why the boundary-layer
restriction works.

The practical consequence is a change in what a surrogate is optimised for. The
field currently selects output representations for prediction accuracy and
memory. If the surrogate is ever to be handed to a solver, there is a second
criterion, it is cheap to evaluate, and it is not satisfied by any uniform raster
at any resolution a dataset can hold.
