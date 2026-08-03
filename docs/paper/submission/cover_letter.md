# Cover Letter — Journal of Computational Physics

Dear Editors,

I am pleased to submit the manuscript **"NeuroForge: Self-Auditing Neural CFD Surrogates
with Calibrated Physics-Residual Trust"** for consideration as a full-length article in
the *Journal of Computational Physics*.

Machine-learning surrogates for CFD predict steady flow fields orders of magnitude faster
than classical solvers, but they emit a single field with no built-in way for a user to
know whether to trust it — precisely where early-design exploration and
out-of-distribution use live. This work closes that loop with the governing physics and
asks, rigorously, **which jobs the discretised steady-RANS residual can and cannot do.**

The title is a defined, tested claim rather than branding: we call a surrogate
*self-auditing* if, from its own prediction alone, it supplies (i) a validated case-level
trust score, (ii) a calibrated, distribution-free error band, and (iii) an accept/reject
decision with measured near-oracle efficiency — and the paper instantiates, stress-tests,
and formally bounds each component.

**Contributions.**
- A **clean two-way dissociation**, established head-to-head on the same benchmark: the
  physics residual is a calibrated, backbone-agnostic *trust signal* (it tells you *where*
  a prediction is wrong) but a poor *correction objective* (it does not tell you *how* to
  fix it). The trust signal is validated across **three architecturally distinct
  backbones** (Transolver, Geo-FNO, MeshGraphNet) and **two datasets** (turbulent-RANS
  AirfRANS airfoils; laminar DeepCFD bluff bodies), with case-level bootstrap confidence
  intervals throughout.
- The trust signal supports **near-oracle selective prediction**: AUROC 0.87–0.91 for
  detecting worst-decile-error cases, and fused with a deep-ensemble spread, AUROC 0.905 —
  recovering ~93% of the oracle's achievable error reduction at a 10% rejection budget.
- A **residual-floor theorem** for the monitored discrete operator, quantifying the
  operator-specific detection limit and the undetectable kernel modes — a formal mechanism
  for why residual-space smallness cannot by itself certify solution-space accuracy, which
  we connect to current work on residual-based conformal scores and error certification.
- A **distribution-free split-conformal certificate calibrated on the deployed, corrected
  field**, shown robust over 20 calibration/test re-draws (coverage 0.895–0.902 at the
  0.90 target in every arm), with an honest in-/out-of-distribution scoping.
- A **repaired force-measurement pipeline**: a control-volume (far-field momentum-balance)
  integrator recovers official AirfRANS lift from predicted fields at Spearman 0.998 with
  per-seed median magnitude errors of 3.6–3.9%, and decomposes the remaining drag error
  into measurement-limited versus model-limited parts.

**Fit for JCP and reproducibility.** The paper contributes computational methodology at
the physics/ML interface: a self-auditing, calibrated trust layer for neural PDE
surrogates, a formal analysis of the discrete residual as monitor versus objective, and
controlled attribution experiments (including a null-input corrector control) of the kind
the journal's readership expects of a numerical method. It is explicit throughout about
what is *measured* versus *assumed*, and preserves its negative results rather than hiding
them — including a self-falsifying control on a force-ranking metric that led us to
recompute against official benchmark labels and repair the integrator. Every headline
number maps to a committed script and result file via `docs/REPRODUCE.md`, with a manifest
recording seeds, environment, and file hashes; the package is CPU-first, runs end-to-end
with zero downloads, and is permanently archived at Zenodo (DOI 10.5281/zenodo.21277928).

The manuscript is original and is not under consideration by any other journal; a preprint
has been posted to arXiv (arXiv:2607.10333), consistent with Elsevier's preprint policy.
Both authors have approved this submission and declare no competing interests, and the
manuscript includes the declarations of generative-AI use required by Elsevier policy. We
confirm the work complies with the journal's authorship and research-integrity policies.

Thank you for your consideration.

Sincerely,
Ali Jabbary (corresponding author, on behalf of both authors)
Department of Mechanical Engineering, Urmia University
st_a.jabbary@urmia.ac.ir | https://alijabbary.com

Kasra Ghanavati
School of Computing and Mathematical Sciences, University of Greenwich
kg1111r@gre.ac.uk
