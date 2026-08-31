# Title and abstract — draft of 2026-08-31

⚠ Numbers marked `[PENDING]` are waiting on the three probes launched today.
Nothing here goes into the paper until they land.

## Title

**Chosen criterion:** the title must be provable by *every* section, not just the
headline one. Checked against the current outline:

| section | does it prove "the first cell decides"? |
|---|---|
| §5.2 placement ladder | directly — 8,192 well-placed values against 16,384 badly placed `[PENDING]` |
| §5.5 repair | directly — restore the first cell, recover the saving `[PENDING]` |
| §6 mechanism | directly — the closed form *is* the first-cell gradient |
| §7.1 resolution ladder | yes — 10.8x the values changes nothing because none of them lands in the first cell |
| §7.3 wake | yes — the wake has no first cell, and an oracle wake seed is worth +0.5% |
| §5.7 grid sequencing | yes — a coarsened body-fitted mesh keeps a station there, and the criterion predicts it helps |
| §8 certificate | indirectly — it bounds what happens when the criterion is not checked |

**Leading candidate**

> **The first cell decides the warm start: a closed-form criterion, and a repair,
> for neural initialisation of RANS**

**Alternatives kept for the record**

- *Placement, not resolution: what a neural surrogate must resolve to accelerate
  a RANS solver* — sharper on the negative result, weaker on the repair.
- *Resolve the first cell, or repair it: warm-starting RANS with neural
  surrogates* — punchier, but "or repair it" reads as an afterthought.

The retired title, *"The wall gradient is the warm start: why projected neural
predictions slow a RANS solver down"*, stops proving itself the moment a
correctly-placed projection works, and would be actively misleading if the
repair succeeds.

## Abstract (target ≤ 250 words)

Neural surrogates for external aerodynamics are usually evaluated as predictors.
Used instead as initial conditions for a production RANS solver they are
routinely worse than no initialisation at all, and we show the cause is neither
accuracy nor resolution but **where a representation places its samples**. Store
the exact converged flow field as a 128² Cartesian raster or an equal-budget
wall-fitted grid and hand it back to `simpleFoam`: total-drag convergence is 548%
and 173% slower than from uniform freestream. The reason is arithmetic. Every
mesh cell nearer the wall than the representation's first station receives that
station's velocity, so the first-cell tangential gradient — which viscous drag
integrates — is overestimated by `u⁺(y₁⁺)/u⁺(y_c⁺)`, a ratio fixed by the law of
the wall with no free parameter. Predicted 23.7×, measured 21.0× across six
cases (1.13 ± 0.02). This gives a criterion evaluable before any solve, and it
indicts placement rather than budget: refining a raster 10.8-fold moves the
predicted damage from 36.6× to 35.3×, because `u⁺` grows logarithmically, while
**`[PENDING]` halving the values and moving the first station inside the first
cell recovers the saving**. Because the factor is known it can also be removed:
inverting a wall function at the representation's own first station repairs a
projected seed from 1900% to 37–52% gradient error, against 54% for a mesh-native
prediction `[PENDING solver result]`. Across thirteen cases the resulting seed
accelerates viscous-drag convergence by 18.4% (13/13, p = 0.0002), with a
converged-field control at 93.6% and a null negative control.

**Word count: `[recount before submission]`**

## Highlights (≤ 85 characters each, verified)

1. A perfect flow field, stored on a 128^2 grid, is a worse RANS start than freestream *(83)*
2. Placement beats budget: 8,192 well-placed values against 16,384 badly placed ones *(81)* `[PENDING]`
3. Wall-gradient damage follows in closed form from y+ alone, to within 13 per cent *(80)*
4. Inverting a wall function repairs a projected seed to mesh-native wall accuracy *(79)*
5. Mesh-native boundary-layer seeding: +18.4% viscous-drag convergence, 13/13 cases *(80)*

Sixth candidate, if one of the above is dropped: *The criterion predicts grid
sequencing, a classical method it was not derived from* *(82)*.
