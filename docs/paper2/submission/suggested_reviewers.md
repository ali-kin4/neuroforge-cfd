# Suggested reviewers — *Computers & Fluids* (Paper 2)

> ⚠️ **Two verification steps that must happen before any of this is entered.**
>
> 1. **The C&F editorial board was not machine-readable on 2026-08-31** —
>    ScienceDirect returns HTTP 403 to automated fetches. **Open
>    `sciencedirect.com/journal/computers-and-fluids/about/editorial-board` by
>    hand and strike anyone who appears on it.** Suggesting a sitting editor is
>    a category error and it happened to be caught by hand for Paper 1
>    (Karniadakis, an AE at JCP). Assume nothing here is board-screened.
> 2. **Affiliations and email addresses must be looked up at the time of
>    submission**, never entered from this file. People move.

**Selection principle**, carried over from Paper 1 and still right: the goal is
**assignability**, not fame. Editors preferentially pick reviewers already in
the journal's database; very senior people decline or delegate. So the list is
weighted toward mid-career people who have published *this exact genre* in the
last 24 months, plus two seniors for weight.

**Conflict screen (2026-08-31).** None shares an institution with **Ali
Jabbary** (Urmia University) or **Kasra Ghanavati** (University of Greenwich),
and none appears among either author's co-authors (Ali's: Ghasabehi, Shams,
Jafarmadar, Pourmahmoud, Rosen, Abdollahi, Ahmadi, Samanipour). All arms-length.

---

## Recommended set

### 1. Heng Xiao — *closest prior art, and the right person to test the claim*
Senior. Author of **neural operator-based super-fidelity** (arXiv:2312.11842,
*J. Comput. Phys.* 2025), which warm-starts a steady RANS solver with a
mesh-native operator and reports 11–16× to reach a force band. That is the
nearest published result to ours and it is *consistent* with our criterion — his
operator evaluates natively. He is the reviewer most able to say whether our
differentiation from his work is honest.

*Why he is not a conflict:* competing work is not a conflict of interest, and
naming the closest competitor signals we are not hiding from the comparison.

### 2. Philipp Bekemeyer (DLR) — *applied aerodynamics + ML, and audits this field*
Mid-career, and senior author of **"Evaluation of State-of-the-Art Deep Learning
Architectures for Aerodynamical Predictions"** (arXiv:2607.13866, July 2026). He
benchmarks exactly the class of surrogate we warm start with, from an industrial
aerodynamics standpoint. Strong C&F profile; very likely already in the database.

### 3. Paola Cinnella (Sorbonne Université) — *RANS, UQ, and an AirfRANS author*
Senior. Co-author of **AirfRANS** (arXiv:2212.07564), whose own paper reports
that models overestimate near-wall velocities and that this damages the drag
coefficient — the observation our mechanism explains and quantifies. Deep
classical RANS credibility, which is what will decide whether our solver
protocol is judged sound.

### 4. Eric M. Wolf (AFRL) — *the 26.3× wake result, and multi-fidelity warm starts*
Mid-career. Co-author of both the **wake-extension initialisation** paper
(arXiv:2501.14699 — the largest number in this literature) and
**multi-fidelity ML for steady flows** (arXiv:2501.14870), as well as
super-fidelity. Our §7.3 argues by measurement that his 26.3× and our +18.4%
concern different regimes and compose rather than compete; he is the person who
should check that argument.

*If the editor prefers not to take two authors from the super-fidelity group,
this is the one to drop, not #1.*

### 5. Hrvoje Jasak (Wikki / Univ. of Zagreb) — *the OpenFOAM side*
Senior, and the reviewer who would find any error in the SIMPLEC setup,
`nNonOrthogonalCorrectors` handling, `residualControl` exit behaviour, the
C-grid construction, or the force-object bookkeeping — all of which this paper
leans on hard. A hostile-but-fair read from him is worth more than a friendly
one from a machine-learning reviewer.

---

## Deliberately not suggested

- **Anyone from the DD-RNO group** (arXiv:2608.13490, three weeks old at
  submission). Their result is convergent evidence for our condition 2; asking
  them to review invites a citation-trading appearance.
- **Rishikesh Ranade / Kaustubh Tangsali (NVIDIA)** — relevant
  (arXiv:2503.15766) but a corporate group whose product line overlaps the
  application; the appearance is worse than the expertise is worth.
- **Anyone at Urmia or Greenwich.** See the conflict screen.

## Opposed reviewers

None.
