# Acceptance-gate measurement — the residual-as-GATE artifact

Produced by `scripts/measure_acceptance_gate.py`. CPU-only, zero forward passes:
the deployed correction step is recovered as `corrected_cache − ensemble_cache`.
200 AirfRANS `full` test cases × 3 DEQ corrector seeds = **600 gated steps**.
Faithfulness gate: the ensemble-mean residual and rel-L2 reproduce
`results/selective/selective_percase.json` with max abs diff **0.0** (n=200).

## ⚠️ This measurement CONTRADICTS the manuscript

`sec:iters` currently states that the backtracking acceptance test "accepts
essentially zero steps on a trained corrector" and that "the certified loop makes
almost no accepted progress", flagging the absence of this artifact as "the
thinnest leg of the residual-as-objective negative". Measured, the opposite holds
on this path.

## What the gate actually does

The gate is a **throttle, not a filter**. At the full step the residual usually
rises, so the full correction is refused — but backtracking then finds a reduced
step that does lower the residual, and that step is accepted.

| outcome | count | share |
|---|---:|---:|
| accepted (any step) | 592 / 600 | **98.7 %** |
| full correction admitted (step = 1.0) | 120 / 600 | 20.0 % |
| throttled to a partial step | 472 / 600 | 78.7 % |
| rejected outright (all 5 steps refused) | 8 / 600 | 1.3 % |

Median admitted fraction of the correction: **0.50**.

Residual ratio `N(y0 + s·δ) / N(y0)` as the step is backtracked:

| step | median ratio | fraction ≤ 1 (would accept) |
|---:|---:|---:|
| 1.0 | 1.027 | 0.200 |
| 0.5 | 0.975 | 0.890 |
| 0.25 | 0.998 | 0.698 |
| 0.125 | 1.000 | 0.312 |
| 0.0625 | 1.000 | 0.273 |

## The decisive number: the admitted step improves accuracy

For the step the gate actually admits:

- median residual reduction: **−2.07 %**
- median true rel-L2 error change: **−2.58 %**
- true error improves on **440 / 592 = 74.3 %** of accepted steps
  (per seed: 0.877 / 0.785 / 0.569)

So the monotone-residual guarantee is **not vacuous as an accuracy mechanism** on
this path: the certified loop delivers a real, if modest, accuracy gain. Note the
throttling *helps* — the full step improves true error on only 191/600 (32 %) of
cases, while the halved step the gate selects improves it on 74 %.

## Scope — what this does and does not measure

- It applies the **shipped** gate (`_MAX_BACKTRACK`, `_EPS` and the accept rule are
  imported from `neuroforge.solver.correction_loop`, not re-implemented) to the
  **shipped** DEQ correction, at the deployed `step_size = 1.0`.
- The deployed DEQ path *bypasses* this gate by design, so the gate is applied
  counterfactually to the deployed step.
- The correction here is the DEQ step applied to the **ensemble-mean** field (the
  `data/cache/w2` path used by the conformal and selective-prediction studies),
  not to an individual backbone. Since the ensemble mean is already more accurate
  than a DEQ-corrected single backbone (`sec:v2`), the correction has less headroom
  here than in `tab:v2`.
- It is a **single** gated step, not the multi-iteration feed-forward loop of
  `tab:indist`, whose corrector checkpoints were not retained.

What survives unchanged: the residual **rises** at the full step in 80 % of cases
(480/600) while error often falls — the residual/error divergence that underpins
the paper's central dissociation is intact. What does not survive is the stronger
sub-claim that the acceptance gate is an empty mechanism.
