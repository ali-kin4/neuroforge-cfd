# Suggested Reviewers — JCP (NeuroForge)

JCP requests 3–5 suggested reviewers. Below are conflict-light candidates spanning the paper's
three pillars: neural operators for CFD, the AirfRANS benchmark, and UQ/calibration for PDE
surrogates. **Action for the author:** remove anyone you have co-authored with, share an
institution with, or have a personal conflict with (the journal requires arms-length
reviewers), then enter 3–5 in Editorial Manager with the one-line rationale.

> Note: I have **not** verified current email/affiliation for each — confirm before entering.
> Pick a balance (don't take all from one group). I deliberately avoid suggesting authors of
> the very closest concurrent work where a competitive conflict is plausible.

## Neural operators / ML-for-CFD
1. **Paris Perdikaris** (physics-informed ML, operator learning) — covers the residual-as-physics
   and surrogate-fidelity angle; would scrutinize the dissociation claim rigorously.
2. **George Em Karniadakis** (operator learning, PINNs, scientific ML) — authority on
   residual-based learning and its limits; ideal for the "poor objective" negative + theorem.
3. **Nikola Kovachki** (neural operators, theory) — strong on the operator-theoretic framing and
   the residual-floor theorem's rigor.

## AirfRANS / aerodynamics surrogates
4. **Patrick Gallinari** (AirfRANS co-author, geometric DL for CFD) — deep familiarity with the
   benchmark, its metrics, and the force-ranking subtleties we correct (the famous ρ_D≈0 result).
   *Conflict note: heavily cited; fine as a domain reviewer but author should confirm no direct tie.*

## UQ / conformal prediction for PDE surrogates
5. **Somdatta Goswami** (ANCHOR; residual-based error estimation, operator UQ) — closest to the
   trust-signal contribution; would test the novelty positioning hardest (a rigorous, fair test).
6. **(alternate) A conformal-prediction-for-operators author** (e.g. the UQNO / function-space
   conformal line) — for the calibration certificate and exchangeability scoping.

## Reviewers to AVOID suggesting (perceived conflict / too close)
- Authors of the immediate Transolver-lineage SOTA leaderboard race (LinearNO/PGOT/LRSA, etc.)
  — competitive proximity on the same benchmark.
- Anyone the author has collaborated with.

Recommended final set: **#1, #2, #4, #5** (+ #3 or #6 as the fifth), balancing the three pillars.
