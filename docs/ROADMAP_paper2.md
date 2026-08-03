# Paper 2 roadmap — trust-gated hybrid solving (PLANS ONLY, not started)

Status: **planning document**. Nothing here is implemented or claimed anywhere in
Paper 1. Kept in-repo so the seams Paper 1 already exposes (`ClassicalFallback`,
the conformal trust gate, the triage policy) map cleanly onto the next build.

## Thesis
The audit (Paper 1) tells you *which* predictions not to trust. Paper 2 makes that
decision *actionable*: low-trust cases are handed to a warm-started classical
solver, buying classical accuracy at a fraction of classical cost — a new
computational workflow, not a study of a signal.

## Build plan (est. 2–4 weeks, CPU-dominated)
1. **OpenFOAM backend** for `ClassicalFallback` (WSL2, `simpleFoam`, k-omega SST —
   the exact AirfRANS solver). 2-D steady solves are minutes on a 13700K.
2. **Warm-start mapping**: neural prediction (u, v, p, nut on the 128^2 grid) ->
   OpenFOAM initial fields on the case mesh (inverse of the rasteriser).
3. **Gate policies to compare**: always-solve / never-solve / random-k% /
   ensemble-sigma gate / residual gate / fused (Paper-1 triage score).
4. **Pre-registered metrics**: final field error vs pure-neural and pure-classical;
   iterations-to-convergence (warm vs cold start); wall-clock per case and per
   fleet; cost–accuracy Pareto per gate policy.
5. **Headline target claim**: "audit-gated hybrid attains X% of classical accuracy
   at Y% of classical cost; warm-starting from the surrogate saves Z% iterations."

## Companion: reliability benchmark release
Package the Paper-1 evaluation as a public harness ("submit AirfRANS predictions,
receive an audit card": trust AUROC, risk–coverage, conformal coverage). No
reliability benchmark exists in the field as of 2026-08; accuracy leaderboards do.

## Also queued (either paper)
- Audit-driven active learning at scale (Paper-1 pilot protocol:
  `docs/protocols/audit_loop_pilot.md`).
- Variationally-correct residual monitor (error-equivalent norm; would make the
  Paper-1 floor theorem constructive). High risk, high theory payoff.

## Venue
CMAME first (this is the "new computational methodology" they asked for),
JCP alternative. Cite Paper 1 for the audit machinery.
