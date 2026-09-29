# Deviation log

Every change to a protocol after it was committed, with its timestamp, reason, and
both results. Protocols live in the docstrings of the scripts they govern.

| Date (UTC+3:30) | Script / protocol | Change | Reason | Result before | Result after |
|---|---|---|---|---|---|
| 2026-09-29 | null_reference.py (A1) | ADDITION, not an edit: a secondary strong-flexible fit on WindsorML and AhmedML only (ARD-GP, KRR over a wider grid l in 0.1..32, tuned gradient boosting, 5x repeated 10-fold) | A1's grid-edge hits make the reference anti-conservative exactly where it is claimed weak | A1 values stand as recorded | to be reported beside A1, never replacing it |
| 2026-09-29 | all reference analyses | ADDITION: error-scale outcomes, log(1-rho) and drag counts; per-geometry reference area co-primary where published; size-only decomposition | A +/-0.02 margin in rho near 0.999 is vacuous (review) | — | — |
| 2026-09-29 | PREREG_factorial.md cell (a) | IMPLEMENTATION CLARIFICATION (before running): 'withhold thickness' is implemented on an aligned NACA encoding [U, alpha, series5, camber, camber_pos, reflex, thickness]; the full reference R is refit on the same encoding so the comparison is like-for-like | Raw encoding puts thickness in different columns for 4- and 5-digit airfoils | — | — |
| 2026-09-29 | PREREG_factorial.md cell (c) | ADDITION before running: secondary embedding pooled over surface nodes only (primary stays mean over all nodes). Head (c2) uses the 6 raw case parameters of null_reference.py (the prereg's '7 case parameters' counted alpha^2, which family Q adds by squaring) | Forces are surface integrals; parameter-count wording | — | — |

## Disclosed history (earlier analyses, before this log existed)

- 2026-09-11, `scripts/covariate_null.py`: the pre-registered comparator for AirfRANS lift was
  the (U, alpha) feature set. A feature set that adds the NACA digits was adopted about
  17 minutes after registration. It was not labelled a deviation at the time. Both are
  reported: rho_L 0.934 (U, alpha) and 0.982 (with digits).
- `src/neuroforge/nullbench/harmonised.py`: the AirfRANS entry omits alpha^2, while the
  original AirfRANS null includes it. Harmonised drag R^2 is 0.683 without it and 0.895 with
  it. The frozen reference (`null_reference.py`) squares every column on every benchmark, so
  this asymmetry does not recur.
- `docs/paper/review/null_mechanism.md` Sec. 4 attributed WindsorML's weak null to an
  unpublished design parameter. The parameter is published in the per-run files. Restoring it
  leaves the drag null unchanged (0.107 -> 0.105). See `analysis/windsor_completed_metadata.py`.
