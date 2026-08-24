# Cover Letter — Journal of Computational Physics

Dear Editors,

I am pleased to submit **"NeuroForge: Self-Auditing Neural CFD Surrogates with Calibrated
Physics-Residual Trust"** for consideration as a full-length article in the *Journal of
Computational Physics*.

**Why this belongs in JCP.** The manuscript extends a line the journal has recently opened:
Yu, Ho and Wang's conformal-prediction framework for physics-informed networks (*JCP* 561:114979,
2026) and Garg and Chakraborty's deep-ensemble uncertainty quantification for operator
surrogates (*JCP* 534:114012, 2025). Both establish that a learned PDE surrogate can carry a
calibrated statement about its own reliability. Our paper asks the question those results
raise next and answers it in both directions: **precisely which jobs can the discretised
steady-RANS residual do, and which can it provably not do?** We show the residual is a
calibrated, backbone-agnostic *trust signal* yet a poor *correction objective*, and we prove
a floor theorem identifying the operator's undetectable kernel modes — a formal mechanism
for why smallness in residual space cannot by itself certify accuracy in solution space.
That is a statement about a discrete operator, not about a network architecture, and we
believe it is of direct interest to JCP's readership.

**Contributions.**
- A **two-way dissociation** established head-to-head on one benchmark, validated across
  **three architecturally distinct backbones** (Transolver, Geo-FNO, MeshGraphNet) and **two
  datasets** (turbulent-RANS AirfRANS airfoils; laminar DeepCFD bluff bodies), with
  case-level bootstrap confidence intervals throughout.
- A **residual-floor theorem** for the monitored discrete operator, quantifying the
  operator-specific detection limit and the undetectable modes.
- **Near-oracle selective prediction**: AUROC 0.87–0.91 for detecting worst-decile-error
  cases; fused with deep-ensemble spread, AUROC 0.905, recovering ~91% of the oracle's
  achievable error reduction at a 10% rejection budget (residual alone recovers ~67%). A
  **distribution-free split-conformal certificate**, calibrated on the deployed corrected
  field, holds coverage 0.895–0.902 against a 0.90 target across 20 re-draws.
- A **measured cost model** (Section 5.7): the full audit costs 1.13 ms against a 3.82 s
  prediction — **0.03%**, roughly one part in 3,400 — so the certificate is effectively free,
  while the ensemble arm costs 4.86x a single backbone. The trust/accuracy trade-off is
  therefore priced, not asserted.
- A **repaired force-measurement pipeline**: a control-volume integrator recovers official
  AirfRANS lift at Spearman 0.998 (per-seed median magnitude error 3.6–3.9%) and decomposes
  the residual drag error into measurement-limited versus model-limited parts.

**Reproducibility.** Every headline number maps to a committed script and result file via
`docs/REPRODUCE.md`, with a manifest recording seeds, environment and SHA-256 file hashes.
The package is CPU-first, runs end-to-end with zero downloads, and is permanently archived
at Zenodo (DOI 10.5281/zenodo.21277928). The paper is explicit throughout about what is
*measured* versus *assumed* and preserves its negative results rather than hiding them —
including a self-falsifying control on a force-ranking metric that led us to recompute
against official benchmark labels and repair the integrator.

**Suggested reviewers** (all arms-length; no shared institution or prior collaboration with
either author):
1. Souvik Chakraborty — deep-ensemble UQ for operator surrogates (*JCP* 534:114012, 2025)
2. Yangshuai Wang — conformal prediction for PINNs (*JCP* 561:114979, 2026)
3. Paris Perdikaris — physics-informed machine learning and operator learning
4. Nikola Kovachki — neural operator theory
5. Vignesh Gopakumar — conformal prediction for PDE surrogates

The manuscript is original and is not under consideration elsewhere; a preprint is posted to
arXiv (arXiv:2607.10333), consistent with Elsevier's preprint policy. Both authors have
approved this submission, declare no competing interests, and the manuscript includes the
declarations of generative-AI use required by Elsevier policy. We confirm the work complies
with the journal's authorship and research-integrity policies.

Thank you for your consideration.

Sincerely,
Ali Jabbary (corresponding author, on behalf of both authors)
Department of Mechanical Engineering, Urmia University
st_a.jabbary@urmia.ac.ir | https://alijabbary.com

Kasra Ghanavati
School of Computing and Mathematical Sciences, University of Greenwich
kg1111r@gre.ac.uk
