# Cover Letter — Computer Methods in Applied Mechanics and Engineering

Dear Editors,

I am pleased to submit the manuscript **"NeuroForge: A Self-Correcting, Geometry-Native
Neural CFD Engine with Calibrated Physics-Residual Trust"** for consideration as a Research
Paper in *Computer Methods in Applied Mechanics and Engineering*.

Machine-learning surrogates for CFD predict steady flow fields orders of magnitude faster than
classical solvers, but they emit a single field with no built-in way for a user to know whether
to trust it — precisely where early-design exploration and out-of-distribution use live. This
work closes that loop with the governing physics and asks, rigorously, **what jobs the
discretised steady-RANS residual can and cannot do.**

**Contributions.**
- A **clean two-way dissociation**: the physics residual is a calibrated, backbone-agnostic
  *trust signal* (it tells you *where* a prediction is wrong) but a poor *correction objective*
  (it does not tell you *how* to fix it). The trust signal is validated across **three
  architecturally distinct backbones** (Transolver, Geo-FNO, MeshGraphNet) and across
  **two datasets** — turbulent-RANS AirfRANS airfoils and laminar DeepCFD bluff bodies —
  so it is not specific to one architecture, geometry, or turbulence model.
- A **residual-floor theorem** quantifying the operator-specific detection limit and the
  undetectable kernel modes — a formal result on what the residual can certify.
- A **distribution-free split-conformal trust certificate calibrated on the deployed,
  corrected field**, with an honest in-distribution / out-of-distribution scoping.
- A learned self-correction loop with a monotone-residual no-harm guarantee.

**Fit for CMAME and reproducibility.** The paper contributes computational methodology for
engineering simulation: a self-auditing trust layer and a certified correction loop that wrap
*any* neural surrogate, evaluated with the rigor the journal's readership expects of a numerical
method. It is explicit throughout about what is *measured* versus *assumed*, and preserves its
negative results rather than hiding them — including a self-falsifying control on a
force-ranking metric, which led us to recompute it against the official benchmark labels and
report the (lower, honest) value. Every headline number maps to a committed script and result
file via `docs/REPRODUCE.md`, with a manifest recording seeds, environment, and file hashes; the
package is CPU-first, runs end-to-end with zero downloads, and is permanently archived at
Zenodo (DOI 10.5281/zenodo.21277928).

The manuscript is original and is not under consideration by any other journal; a preprint has
been posted to arXiv, consistent with Elsevier's preprint policy. The work has a single author
with no competing interests, and the manuscript includes the declarations of generative-AI use
required by Elsevier policy. I confirm the work complies with the journal's authorship and
research-integrity policies.

Thank you for your consideration.

Sincerely,
Ali Jabbary
Department of Mechanical Engineering, Urmia University
st_a.jabbary@urmia.ac.ir | https://alijabbary.com
