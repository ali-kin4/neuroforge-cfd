# Pre-registered protocol — audit-driven retraining pilot ("the audit closes its own loop")

Status: registered 2026-08-03, BEFORE any run. Success/failure criteria fixed below.
Purpose: test whether the Paper-1 trust score can *improve* the model it audits,
not just reject its outputs. First-in-domain if positive (2026 active-learning
work acquires by UQ, not by physics-residual trust).

## Design
Base pool: AirfRANS `full` train split. Base model: Transolver backbone,
`scripts/run_w1_capture.py` recipe, but trained on a REDUCED seed set of
n_base = 500 cases (leaves an acquisition pool of 300).

Arms (identical budget: +100 acquired cases, retrain from scratch on 600, same
epochs/hyperparameters, seeds 0 and 1):
  A. random-100 from pool (baseline)
  B. sigma-top-100: highest ensemble/dropout uncertainty on pool (UQ baseline,
     analogous to 2026 active-operator-learning)
  C. trust-top-100: highest fused audit score (residual rank + sigma rank) on pool
  D. (control) no acquisition: n_base only

Scoring the pool uses ONLY audit inputs (prediction, residual, sigma) — never
pool ground truth. Pool GT is revealed only for the acquired cases at retrain.

## Pre-registered outcomes
Primary metric: test-set (200 held-out) mse_speed; secondary: mse_u/v/p,
residual_error_spearman, conformal q at 0.90 coverage.
- POSITIVE: C beats A on the primary metric on both seeds, and C >= B on both
  (or beats B on mean with sign-consistency). Report with bootstrap CIs.
- PARTIAL: C beats A but not B -> report as "trust matches UQ acquisition"
  (still a usable equivalence claim; physics score needs no ensemble).
- NEGATIVE: C fails to beat A on both seeds -> report honestly as a scoped
  negative ("case-level trust ranks error but its top cases are not the most
  informative for training"), consistent with the paper's dissociation style.

## Budget
2 seeds x 4 arms x (train 600 or 500 cases, 80 epochs) — but arms share the
base-500 backbone per seed (train once, fine-tune?  NO: retrain from scratch per
arm for cleanliness). ~8 backbone trainings ~ 78.6 s/epoch x 80 x 8 ~ 14 h GPU.
If too long: drop to seeds {0} first (4 trainings, ~7 h overnight), add seed 1
only if the single-seed result is promising (gate before spending).

## Placement
If POSITIVE/PARTIAL and ready before JCP submission: short new subsection in
Paper 1 ("the audit closes its own loop") + limitations update.
If run after submission: revision material / Paper 2 section.
