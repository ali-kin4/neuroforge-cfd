# Deviation log

Every change to a protocol after it was committed, with its timestamp, reason, and
both results. Protocols live in the docstrings of the scripts they govern.

| Date (UTC+3:30) | Script / protocol | Change | Reason | Result before | Result after |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

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
