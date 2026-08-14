# Suggested Reviewers — JCP (NeuroForge)

JCP requests suggested reviewers (Editorial Manager asks for a set; 3–5 is standard). Below
are conflict-light candidates spanning the paper's three pillars: neural operators for CFD,
the AirfRANS benchmark, and UQ/calibration for PDE surrogates.

**Conflict screen (2026-07-12, both authors):** none of the candidates below shares an
institution with **Ali Jabbary** (Urmia University) or **Kasra Ghanavati** (University of
Greenwich), and none appears among either author's co-authors (Ali's: Ghasabehi, Shams,
Jafarmadar, Pourmahmoud, Rosen, Abdollahi, Ahmadi, Samanipour). All are arms-length.

> Note: verify current email/affiliation for each in Editorial Manager before entering.
> Pick a balance (don't take all from one group). I deliberately avoid suggesting authors of
> the very closest concurrent work where a competitive conflict is plausible.
> **Judgment flag on #5 (Goswami):** she co-authored ANCHOR (`roy2025anchor`), the closest
> prior art the paper positions against. She knows the area cold (a rigorous reviewer) but has
> a plausible competitive stake in the novelty framing — include only if you want the hardest
> fair test of positioning; otherwise use #6.

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
