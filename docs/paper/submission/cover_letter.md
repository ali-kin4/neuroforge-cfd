# Cover Letter — Journal of Computational Science

Dear Editors,

I am pleased to submit **"The near-wall barrier on AirfRANS is a coordinate artifact, and
only on velocity"** for consideration as an original research paper in the *Journal of Computational
Science*.

**What the paper reports.** Machine-learning surrogates for simulation are almost universally
scored by resampling their predictions onto a uniform grid. We measure what that resampling
costs, and the measurement changes the answer. On a standard external-aerodynamics benchmark,
kernel interpolation over seven scalars parsed from the case file name — no network, no
flow-field learning at all — gives 8.4x lower volume-pressure error than a Transolver
trained on the same data on an area-uniform 128^2 raster. Re-weighting the *identical* predictions by the
dataset's own node measure takes that ratio to 0.44x, and scoring per node on the native
point cloud with no rasterisation anywhere takes it to 0.21x. One pair of predictions, three
measures in current use, and the ranking reverses.

Run where the data actually lives the interpolator loses on every channel, and inside the
first 0.005 chord the exact least-squares bound over the whole family — what *any* weighting
of the 800 training fields can achieve, with no free parameters remaining — is 25x worse on
all 200 cases. On the published reading, that is where the paper would have stopped: a
surrogate earns the boundary layer and no interpolation reaches it.

**We then measured why, and the explanation dissolved the result.** Inside that band a query
lands 7.8x further from the training case's wall than from its own, and 35% of the kernel's
weight falls *inside* the training airfoil. That is a statement about coordinates, not about
representation, so it makes a prediction: pose the same family in a wall-following frame and
the gap should go. It does. With nothing retuned and the same 800 fields, the identical bound
falls from 25x to **0.0044x, better on 200 of 200 cases**, while the physical arm rescored on
those same nodes still reads 25.2x. The near-wall advantage the published protocol reports is
an artifact of the coordinate system its baseline is posed in.

We are careful about what that does and does not claim, and the manuscript is explicit about
it. The finding rests on the *bound*, which is unanimous and has no tail. The deployed
estimator in the same frame improves by two orders of magnitude but wins only at the median
(0.38x, 152/200 cases) and loses on the case-mean; we therefore claim that no member of this
family is barred from the wall by representation, and **not** that parameter interpolation
beats a trained surrogate there. The frame also has a measurable cost: a wall-normal
coordinate does not exist on 4.98% of near-wall nodes, behind the trailing edge, where an
entire wake fan projects onto a single vertex.

The protocol that reports these numbers cannot see there. The scoring raster's own round-trip
error at the native nodes exceeds the surrogate's per-node error by 413x pooled and 495x near
the wall. We state that symmetrically, because it is not a concession about one arm: it is a
limit on what a 128^2 field metric can establish about *any* method on this benchmark, ours
included, and it is why the head-to-head had to be re-run at native resolution rather than
refined in place.

**Why this belongs in the Journal of Computational Science.** The contribution is an
evaluation result — the object of study is the measurement protocol, not a new architecture —
and this journal has recently and repeatedly published exactly that genre:

- *When simpler models win: a large-scale computational benchmark of lexical and transformer
  NLP pipelines for predicting medication effectiveness* (2026-08-27)
- *Exploring the limitations of transformer models for metocean forecasting* (2026-06-03)
- *Accuracy vs efficiency: benchmarking graph neural networks on edge GPU hardware* (2026-07-30)
- *Benchmarking atom-level explainability against pharmacophore-computed labels in molecular
  machine learning* (2026-07-08)

Two of these are limitations-of-a-model-class papers and two are head-to-head benchmarks; one
is in a geophysical-fluids forecasting domain. We surveyed the adjacent computational-methods
and fluids journals for this pattern and did not find it: the venue whose scope looks closest
on subject matter has published nothing in this genre in seven years. The fit here is to the
journal's demonstrated interest in whether a computational result's reported number means what
it appears to mean, which is precisely the question this paper asks of a widely used benchmark.

**Where the paper is most exposed.** We would rather state this than have it found.

- *The baseline estimator is not new.* The interpolator is kernel ridge regression over a
  seven-dimensional parameter space — response-surface methodology, and decades old. The paper
  says so in those words. It is deliberately not a contribution: its role is to be the
  cheapest defensible thing that could occupy a leaderboard position, and the finding is that
  on one published measure it does, which is a statement about the measure. A reader who wants
  a novel estimator will not find one here.
- *Scope is one benchmark, two dimensions, one learned architecture.* We do not claim the
  ratios transfer. What we argue transfers is the diagnosis — that a wall-clustered
  unstructured mesh scored on a uniform raster puts most of its nodes in a small fraction of
  the scoring area — and that is a property of the discretisation, checkable anywhere, not a
  property of this dataset. The paper states the limitation rather than hedging it.
- *This paper twice refuted its own headline, in public, using rules it had committed in
  advance.* The 8.4x advantage was once the lead claim; the pre-registered rule that governed
  it returned ARTIFACT and the manuscript withdraws it in those words. The near-wall
  representational claim replaced it, and the body-fitted arm — named in an earlier version's
  Limitations as deliberately untested — returned BODY-FITTED-BOUND-OPEN against it. Both
  rules, both thresholds and both verdict strings were committed to a public repository before
  the runs that read them. We regard that history as the strongest evidence we can offer that
  the surviving claims were not selected after the fact, and we would rather the editor learn
  it from us than infer it.

**Reproducibility.** Every headline number maps to a committed script and result file through a
per-claim reproduction map, with a manifest recording seeds, environment and SHA-256 hashes.
The decisive comparison is gated rather than asserted: the body-fitted and physical arms are
computed in one pass from one node array, so the coordinate map is the only difference between
them; the frame is verified injective with zero coordinate collisions and a round-trip exact to
2.1e-13; and the implementation was required to reproduce the frozen earlier artifact at a
worst relative difference of exactly 0.00e+00 before it was permitted to compute a new number.
The software is released as a CPU-first open-source package with a frozen I/O contract and an
AirfRANS loader, and is permanently archived at Zenodo (DOI 10.5281/zenodo.21277928), which
also serves as the deposited dataset the journal's data policy requires. Each decision rule in
the paper — including the one that returned a negative verdict against our own preferred
outcome — was registered in a script docstring and committed to a public repository before the
run it governs, and those commits are identifiable in the history.

**Preprint disclosure.** An earlier preprint of this line of work is posted at arXiv:2607.10333
(currently v3, "NeuroForge: self-auditing neural CFD surrogates with calibrated
physics-residual trust"). I flag the relationship explicitly because the title and framing
differ enough that a search would otherwise raise a fair question. That version led with a
calibrated trust layer built on a physics residual. The present manuscript is not a revision of
it in the ordinary sense: the trust-layer and force-coefficient material has been *removed*,
the native-resolution and measure-dependence experiments described above are new and are now
the paper's spine, and the removed material is intended for a separate paper that will carry
its own preprint. I will replace arXiv:2607.10333 with the present version so that the public
record and this submission agree, and the separate material will be posted under its own
identifier rather than silently duplicated. Neither the present manuscript nor the separate
paper is under consideration at another journal.

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

## Notes to self — not part of the letter

**Rewritten 2026-09-16 for JOCS.** The previous letter was addressed to Computers & Fluids and
described the pre-split paper (residual floor, trust layer, ungated control) — a manuscript
that no longer exists. C&F was withdrawn on an editorial conflict: its ML special-issue
editorial is authored by Ashton, Dwight and Cinnella, three authors of benchmarks this paper
audits. The C&F letter is recoverable from git history if any of its prose is wanted.

**On disclosing the two desk rejections (CMAME 2026-08-02, JCP 2026-09-07):** the letter above
still does **not** mention them, and the reasoning carries over to this third venue unchanged —
there is no obligation to disclose, prior rejection is not discoverable by the editor, and
volunteering it hands them a pre-authorised reason to decline. What *is* disclosed is the
preprint, because it is discoverable and would otherwise look like a concurrent submission.
Reverse this only if you would rather lead with the revision history.

**On the numbers:** every figure in this letter is taken from `abstract.tex` and the
highlights file as built on 2026-09-16, both of which are checked by
`scripts/audit_paper_numbers.py` against their source JSONs. Do not add a number to this letter
that is not already in the abstract — the abstract is audited, free prose is not. Two scope
points are load-bearing and must not be compressed if this letter is shortened: the 344x oracle
is 200 cases while the 21x least-squares bound is a **three-case** probe, and the band
separation is monotone on the grid ladder but **not** at the native nodes.

**On the preprint paragraph — act before filing.** arXiv was confirmed live at v3 on
2026-09-16 by reading the abstract page. A v4 package exists at
`docs/paper/submission/arxiv_v4/` but was **never posted and is now stale** — it was built for
the pre-split residual-floor paper (TMLR build, 43 pp, 10 figures). Do not upload it as-is. The
letter promises a replacement matching the present version; rebuild the package from the
current sources before making good on that, or soften the sentence.

**STILL UNVERIFIED FOR THIS VENUE — do not inherit the C&F answers.** The following were read
at source from the *C&F* guide on 2026-09-07 and have **not** been confirmed for JOCS:
whether editable LaTeX source is required at submission or only at revision; whether Highlights
are required rather than merely encouraged; the declarations-tool .doc/.docx upload step; and
which research-data option applies. ScienceDirect served a CAPTCHA on 2026-09-16 and it was not
bypassed, so the guide could not be re-read. Verified for JOCS as of 2026-09-12: the 250-word
abstract cap, the 1-7 keyword rule, and the absence of any length limit on regular articles.
