# Suggested Reviewers — Computers & Fluids

**Rewritten 2026-09-07 for the reframed paper.** The previous list targeted JCP and a
paper whose headline was a calibrated trust layer, so it was drawn from conformal
prediction and neural-operator theory. The claim now needing assessment is different:
*a monitored discrete operator is inconsistent with the one that generated its labels,
so its residual carries a floor that does not vanish under refinement, and minimising it
moves the field away from the truth.* That is a numerical-analysis judgment before it is
a machine-learning one, and the list is rebalanced accordingly.

**Selection principle.** Assignability over fame. Aim for a mix that can actually
adjudicate both halves: reviewers who can check the discretisation-consistency argument,
and reviewers who can check the surrogate/UQ half. The scope paragraph of *Computers &
Fluids* asks for physical consistency and theoretical analysis, so at least two
reviewers should be able to referee the operator argument on its own terms.

**Conflict screen (re-affirmed 2026-09-07):** none shares an institution with **Ali
Jabbary** (Urmia University) or **Kasra Ghanavati** (University of Greenwich), and none
appears among either author's co-authors (Ali's: Ghasabehi, Shams, Jafarmadar,
Pourmahmoud, Rosen, Abdollahi, Ahmadi, Samanipour). All are arms-length.

> ⚠️ **Verify every affiliation and email in Editorial Manager before entry — do not
> enter from memory.** Also check the current *Computers & Fluids* editorial board
> before submitting: suggesting a sitting editor is a category error, and the board was
> last checked for JCP, not for this journal.

---

## Recommended set

### Numerical analysis of the residual argument (the new core)

1. **A specialist in a-posteriori error estimation for CFD.** The paper's central object
   is an estimator whose calibration point — a zero residual at the exact solution — does
   not exist. The closest classical relatives, all now cited, are defect correction
   (Stetter), multigrid τ-correction (Brandt), data oscillation (Morin, Nochetto and
   Siebert) and least-squares FEM norm-equivalence (Bochev and Gunzburger). A reviewer
   from the goal-oriented/DWR error-estimation community is the right referee for
   whether our consistency-floor theorem is correct and non-vacuous.
   *Candidate to verify against the board:* **Rolf Rannacher** (Heidelberg) or a
   mid-career DWR researcher; we cite Becker and Rannacher (2001). Prefer a mid-career
   name for assignability.

2. **The authors of the positive-side anchor.** **Lei, Tang, Zhang and Chen**
   (arXiv:2608.04400, Newton–Krylov correction of surrogate predictions) sit on the
   *working* side of the boundary this paper draws, and we cite them as such.
   *Judgment call, flagged rather than decided:* they are arms-length and are the best
   qualified to say whether our characterisation of their regime is fair, which is a
   point of genuine risk for us. The counter-argument is that a paper drawing a boundary
   with someone else's result on the favourable side may not want that someone as its
   referee. **Recommend suggesting them**; the characterisation is favourable to their
   work and being wrong about it is the failure mode we most want caught early.

### Surrogate models and UQ for fluids (the retained half)

3. **Vignesh Gopakumar** — conformal prediction for PDE surrogates; uses the residual
   *as* the conformal nonconformity score, which is the nearest prior art to the trust
   layer we retain.
   *Judgment call, flagged:* this is the work that most threatened our novelty claim
   under the old framing. Under the new framing our result constrains his construction
   rather than competing with it, so he is well placed to judge whether that
   constraint is real. He is also the reviewer most likely to reject if it is not.
   **Recommend suggesting**; if the argument does not survive him it does not survive.

4. **Souvik Chakraborty** — deep-ensemble UQ for operator surrogates. Directly qualified
   on the physics-free-versus-physics comparison (control C1), which is the concession
   most likely to be contested in either direction.

5. **Paris Perdikaris** — physics-informed machine learning and operator learning; senior,
   and able to referee the claim that a residual objective fails in this regime against
   the broader PINN literature where residual minimisation is the method.

### Reserve

6. **Nikola Kovachki** — neural operator theory; the right referee if the theorem's
   functional-analytic framing is challenged.
7. **Somdatta Goswami** — operator learning and UQ; mid-career, good assignability.

---

## Do not suggest

- Anyone on the current *Computers & Fluids* editorial board (**check this — the board
  was screened for JCP in August, not for this journal**).
- **George Em Karniadakis**, **Dongbin Xiu**, **Charbel Farhat**, **George Biros** — all
  JCP board members as of 2026-08-24. Their status at C&F is unchecked, but Karniadakis
  in particular is a plausible *handling editor* for this paper at any venue in scope.

## A note on what changed and why

The old list's rationale was "two candidates published this exact genre in JCP within
the last 18 months, which reinforces the scope argument". That logic was venue-specific
and is void here. The replacement logic is that the paper now makes a claim about
discrete operators that a pure ML reviewer cannot check and a pure CFD reviewer can — so
the list must contain at least one of the latter, which the previous list did not.
