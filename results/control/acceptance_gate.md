# Acceptance-gate measurement — the residual-as-GATE artifact

Produced by `scripts/measure_acceptance_gate.py`. Two independent paths, 200 AirfRANS
`full` test cases each, over 3 seeds = **1200 gated steps** total.
Faithfulness gate: the ensemble-mean residual and rel-L2 reproduce
`results/selective/selective_percase.json` with max abs diff **0.0** (n=200).

| path | what the step is | cost |
|---|---|---|
| **deployed** | DEQ correction on its OWN Transolver backbone (`seed{k}.pt` + `seed{k}_corr_with.pt`) — the `tab:v2` system | GPU forward + DEQ solve, cached per case |
| ensemble-mean | DEQ correction applied to the cached ensemble mean (the `data/cache/w2` path used by the conformal and selective-prediction studies) | CPU only, zero forwards |

## ⚠️ This measurement CONTRADICTS the manuscript

The manuscript states that the backtracking acceptance test "accepts essentially zero
steps on a trained corrector", that "the certified loop makes almost no accepted
progress", and that the guarantee is "real but vacuous as an accuracy mechanism" — and it
flags the absence of this artifact as "the thinnest leg of the residual-as-objective
negative". Measured, the opposite holds on **both** paths, and most strongly on the
deployed one.

## Headline

| | deployed (`tab:v2`) | ensemble-mean |
|---|---:|---:|
| accepted (any step) | **599/600 = 99.8 %** | 592/600 = 98.7 % |
| full correction admitted | **304 = 50.7 %** | 120 = 20.0 % |
| throttled to a partial step | 295 = 49.2 % | 472 = 78.7 % |
| refused outright | 1 = 0.2 % | 8 = 1.3 % |
| residual rose at full step | 296 = 49.3 % | 480 = 80.0 % |
| **accepted step improves true error** | **535/599 = 89.3 %** | 440/592 = 74.3 % |
| **median true rel-L2 change at accepted step** | **−5.82 %** | −2.58 % |

Per seed:

| arm | accepted | full | throttled | improves err | median err change |
|---|---:|---:|---:|---:|---:|
| backbone_seed0 | 200/200 | 112 | 88 | 93.5 % | −5.80 % |
| backbone_seed1 | 200/200 | 100 | 100 | 95.5 % | −7.03 % |
| backbone_seed2 | 199/200 | 92 | 107 | 78.9 % | −4.42 % |
| seed0 (ens.) | 195/200 | 49 | 146 | 87.7 % | −3.74 % |
| seed1 (ens.) | 200/200 | 30 | 170 | 78.5 % | −2.57 % |
| seed2 (ens.) | 197/200 | 41 | 156 | 56.9 % | −0.58 % |

## What the gate actually does

It is a **throttle, not a filter**. Where the full step would raise the residual it is
refused, but backtracking then finds a damped step that lowers it — and that step is
admitted. Residual ratio `N(y0 + s·δ) / N(y0)` as the step is backtracked
(ensemble path, where the effect is clearest):

| step | median ratio | fraction ≤ 1 (would accept) |
|---:|---:|---:|
| 1.0 | 1.027 | 0.200 |
| 0.5 | 0.975 | 0.890 |
| 0.25 | 0.998 | 0.698 |
| 0.125 | 1.000 | 0.312 |
| 0.0625 | 1.000 | 0.273 |

On the deployed path the full step is already near-neutral in residual (median ratio
0.995–1.003 per seed), so the gate admits it outright half the time.

## The decisive result

**The monotone-residual guarantee is not vacuous as an accuracy mechanism.** On the
deployed system the step the gate admits reduces true rel-L2 error on 89.3 % of cases, by
a median 5.82 %. The certified loop delivers a real, cheap accuracy gain while retaining
its provable non-worsening property.

What survives unchanged: the residual and the error are still not the same objective — the
full step raises the residual on half the deployed cases (80 % on the ensemble path) while
error often falls, and `tab:iters` still shows the residual rising across iterations while
error falls. The central trust-signal-vs-correction-objective dissociation is untouched.
What does not survive is the sub-claim that the acceptance gate is an empty mechanism.

## Scope

- The gate under test is the **shipped** gate: the accept rule, `_MAX_BACKTRACK` and
  `_EPS` are imported from `neuroforge.solver.correction_loop`, not re-implemented, and
  the step size is the deployed `CorrectionConfig.step_size = 1.0`.
- The deployed DEQ path *bypasses* this gate by design, so the gate is applied
  counterfactually to the deployed step.
- This is a **single** gated step, not the multi-iteration feed-forward loop of
  `tab:indist`, whose corrector checkpoints were not retained.
