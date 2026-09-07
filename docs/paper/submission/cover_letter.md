# Cover Letter — Computers & Fluids

Dear Editors,

I am pleased to submit **"The ground truth fails its own physics check: what a
surrogate-side RANS residual can and cannot certify"** for consideration as an original
research paper in *Computers & Fluids*.

**What the paper reports.** A widely adopted proposal for trusting machine-learning CFD
surrogates is to have the surrogate check itself against the governing equations, by
evaluating a discrete RANS residual on its own prediction. We show that this check is built
on an operator the ground truth itself cannot pass. On 200 of 200 real AirfRANS cases the
monitored residual of the *exact* field is substantially nonzero (mean 0.192), while the
physically wrong uniform-freestream field scores an exact zero — so the objective strictly
prefers a wrong field to the truth. The natural objection is that this is a resolution
artifact, and we test it: refining the grid from 128² to 512² makes the floor **larger**,
in 21 of 24 cases, under a decision rule registered in the repository before the
measurement was run. Restoring the omitted turbulent-stress term moves the floor by less
than 0.2%, and rasterising the same reference data with a higher-order interpolant *raises*
it. The consequence for correction is then measured directly rather than argued: starting
from the exact ground truth, with no neural network in the loop, gradient descent on the
residual cuts it by 84% while driving the field error from zero to the range a trained
surrogate starts in.

**Why this belongs in Computers & Fluids.** The journal's scope names "uncertainty
quantification in fluid flow simulations, reduced-order and surrogate models for fluid
flows", and states that machine-learning papers are welcome provided they show excellent
scientific character. The four specific things the scope asks of such papers map onto this
manuscript as follows.

- *Comparison with traditional numerical reconstruction methods.* This is the axis on which
  we are most exposed and we would rather say so plainly than have it discovered. The paper
  makes no controlled speed-up claim: it states in the text that the ~286× figure is not a
  controlled measurement, and that the OpenFOAM and SU2 verification backends described in
  the software are unimplemented. What the paper does contain is the comparison that bears
  on its actual claim — a direct contrast against *solver-consistent* residual correction
  (Newton–Krylov and related methods), which succeeds on steady CFD precisely because the
  residual being driven is the solver's own. That contrast is the paper's organising result,
  not an aside: it locates the boundary at operator consistency rather than at the problem
  class.
- *Clear presentation of training versus validation cases, with sufficient diversity.* Two
  datasets (turbulent-RANS AirfRANS airfoils, laminar DeepCFD bluff bodies), three backbone
  families, an explicit out-of-distribution regime shift, five seeds on the headline
  backbone, and case-level bootstrap confidence intervals throughout.
- *Physical consistency and theoretical analysis of the model.* This is now the spine of the
  paper rather than an appendix. We show the monitored operator is an inconsistent
  discretisation of RANS, so its floor has a nonzero continuum limit and cannot be removed
  by refinement; and we characterise the operator's kernel, the modes it can neither detect
  nor correct.
- *Limitations as well as merits.* The paper withdraws or narrows seven claims made in
  earlier versions of this work, including one of its own headline numbers. Most pointedly,
  we report an *ungated fixed half-step control that outperforms our own acceptance gate on
  accuracy* (95.8% of cases improved against 89.3%), and conclude that the gate's value is
  the guarantee it provides rather than the accuracy it was credited with. We also report
  that a physics-free uncertainty score matches the physics residual at ranking field error,
  and that the physics wins outright only on drag.

**Reproducibility.** Every headline number maps to a committed script and result file
through `docs/REPRODUCE.md`, with a manifest recording seeds, environment and SHA-256
hashes. The package is CPU-first, runs end to end with no downloads via a synthetic data
generator, and is permanently archived at Zenodo (DOI 10.5281/zenodo.21277928), which also
serves as the deposited research dataset the journal's Option C data policy requires. The
grid-refinement study's decision rule was committed before the study was run, and the commit
is identifiable in the public history.

**Preprint disclosure.** A preprint of an earlier version is posted at arXiv:2607.10333,
consistent with Elsevier's preprint policy. It carries a different title and a substantially
different framing: that version led with a calibrated trust layer. The present manuscript is
a substantial revision built around the grid-refinement and residual-descent measurements
described above, which are new, and several claims in the preprint are explicitly withdrawn
here. An updated preprint reflecting the present version will be posted. I mention this so
that a search on the title does not suggest either a concurrent submission or an unexplained
divergence between the two documents.

**Suggested reviewers** are listed separately. The manuscript is original, is not under
consideration elsewhere, and both authors have approved this submission. We declare no
competing interests and include the declaration of generative-AI use required by Elsevier
policy, together with a CRediT contribution statement. We have selected the **subscription**
publishing route.

Thank you for your consideration.

Sincerely,

Ali Jabbary (corresponding author, on behalf of both authors)
Department of Mechanical Engineering, Urmia University, Urmia, Iran
st_a.jabbary@urmia.ac.ir | https://alijabbary.com | ORCID 0000-0003-0573-6909

Kasra Ghanavati
School of Computing and Mathematical Sciences, University of Greenwich, London, UK
kg1111r@gre.ac.uk | ORCID 0009-0009-0888-3307

---

## Note to self — not part of the letter

**On disclosing the two desk rejections (CMAME 2026-08-02, JCP 2026-09-07):** the letter
above does **not** mention them. That is a deliberate decision, taken on the venue plan's
recommendation: there is no obligation to disclose, prior rejection is not discoverable by
the editor, and volunteering it hands them a pre-authorised reason to decline. What *is*
disclosed is the preprint, because that is discoverable and would otherwise look like a
concurrent submission. Reverse this only if you would rather lead with the revision history.

**On the numbers:** every figure in this letter is checked against the manuscript as built
on 2026-09-07 — 0.192 floor, 21/24, <0.2%, 84%, 95.8% vs 89.3%. The earlier JCP letter
quoted "three architecturally distinct backbones" and an audit cost of "1.13 ms against
3.82 s — 0.03%"; the first is now qualified in the paper (the third backbone is evaluated
off its training density) and so is not claimed here, and the second is stated in the paper
with both its measurements. Do not reinstate either phrasing without re-checking.
