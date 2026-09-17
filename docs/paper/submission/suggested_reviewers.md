# Suggested Reviewers — Journal of Computational Science

**Rewritten 2026-09-17.** The previous list was doubly stale: it targeted *Computers &
Fluids*, and it was built for a paper that no longer exists — one whose headline was an
operator-inconsistent residual floor and a calibrated trust layer. Both of those
sections have been cut. The list below is built for the claim the manuscript now makes.

## What a referee has to be able to check

> Machine-learning surrogates for simulation are scored by resampling onto a uniform
> grid, and that choice decides the ranking. On AirfRANS the same pair of predictions
> gives 8.4x, 0.44x and 0.21x under three measures in current use. At native resolution
> the surrogate wins on every channel, and no weighting of the training set reaches the
> near-wall region in physical coordinates. The scoring raster cannot adjudicate it: its
> own round-trip error at the nodes is far above either arm's.

That is **three different referee competencies**, and the previous list had none of
them:

1. **ML-for-CFD benchmarking and metric design** — is the measure-dependence result a
   real gap in how the field reports, or a restatement that different metrics differ?
2. **Unstructured-mesh discretisation and interpolation error** — is the
   representation-ceiling argument sound, and is the round-trip error the right
   instrument?
3. **Classical parametric surrogates (Kriging / response surfaces / ROM)** — is the
   baseline a fair and competently-tuned member of its family, or a strawman?

At least one reviewer from **each** group. A pure ML reviewer cannot check (2); a pure
numerical-analysis reviewer cannot check (1); neither can check (3).

## Conflict screen — read this before adding anyone

This paper **audits published benchmarks**. Suggesting an author of an audited benchmark
means asking the audited party to referee the audit. That is a category error and it is
the single most likely way to get a hostile, non-arms-length report.

**Excluded on that basis** (do not suggest, do not add later):

| Name | Why excluded |
|---|---|
| Florent Bonnet, Jocelyn Ahmed Mazari, Paola Cinnella, Patrick Gallinari | Authors of **AirfRANS**, the benchmark this paper audits |
| Haixu Wu, Mingsheng Long et al. | Authors of **Transolver**, the surrogate arm |
| Neil Ashton | Author of **DrivAerML** and of the automotive benchmarking framework the paper cites; also co-authored the C&F ML special-issue editorial that caused us to withdraw from that venue |
| Richard Dwight | Same C&F editorial |
| Mohamed Elrefaie et al. | Authors of **DrivAerNet++**, cited as the counter-example to our own scope claim |

**Author conflicts (re-affirmed 2026-09-17):** none of the suggested names shares an
institution with **Ali Jabbary** (Urmia University) or **Kasra Ghanavati** (University of
Greenwich), and none appears among either author's co-authors (Ali's: Ghasabehi, Shams,
Jafarmadar, Pourmahmoud, Rosen, Abdollahi, Ahmadi, Samanipour).

> ⚠️ **Two things must be checked in a browser before entering any of this in Editorial
> Manager.** (a) The **current JOCS editorial board** — suggesting a sitting editor is a
> category error, and the board has never been screened for this journal; the founding
> Editor-in-Chief is Peter Sloot. (b) **Every affiliation and email**, from the person's
> own page, not from memory and not from this file. ScienceDirect served a CAPTCHA to
> automated access on 2026-09-16 and 2026-09-17, so neither has been verified here.

---

## Recommended set

### Group 1 — ML-for-CFD benchmarking and metric design

1. **Rishikesh Ranade** (NVIDIA) — co-author of the PhysicsNeMo-CFD benchmarking
   framework (arXiv:2507.10747), which argues that *"standard metrics such as R² and
   pointwise MSE mask deficiencies that matter for engineering."* That is this paper's
   thesis reached independently and from the automotive side. He is arms-length from
   AirfRANS, which makes him the best-qualified reviewer who is not a conflicted party.
   **Strongest single recommendation.**

2. **Mohammad Amin Nabian** (NVIDIA) — same framework, same competence; use as the
   alternate if Ranade is unavailable, rather than suggesting both and spending two
   slots on one group.

   > Note the shared-affiliation risk: Ranade, Nabian and Tangsali are all NVIDIA and all
   > on the same paper. Suggest **one**.

### Group 2 — discretisation and interpolation error on unstructured meshes

3. **A specialist in interpolation and conservative remapping between unstructured
   meshes.** The representation ceiling is a claim about what a projection destroys, and
   it needs someone who thinks about remap error for a living rather than a
   neural-operator theorist. The mesh-to-mesh transfer / conservative-remap community
   (ALE hydrodynamics, climate model coupling) is the right pool.
   **To fill from the JOCS board's adjacent subject editors when the board is read** —
   this is a competence, not a person, and naming the wrong individual is worse than
   naming the competence to the editor in the cover letter.

### Group 3 — classical parametric surrogates, to judge the baseline

4. **A Kriging / response-surface-methodology researcher in aerodynamic design.** The
   paper's baseline is kernel ridge regression over seven design parameters, and the
   paper concedes in its own words that this is decades-old RSM. The failure mode this
   reviewer catches is the one that matters most to us: *"that is not how anyone would
   actually build a parametric aerodynamic database, so the comparison is unfair."*
   The gradient-enhanced-Kriging and cokriging-for-aerodynamic-functions community is
   the pool; prefer a mid-career name for assignability over a founding figure.

### Group 4 — surrogate UQ and operator learning (retained, lower priority)

5. **Souvik Chakraborty** (IIT Delhi) — operator-surrogate UQ; broad enough to referee
   the head-to-head protocol and the seed reporting. Retained from the previous list
   because he remains qualified under the new framing, unlike most of it.

6. **Somdatta Goswami** (Johns Hopkins) — operator learning; mid-career, good
   assignability, and able to judge whether the matched-budget claim is defensible.

### Reserve

7. **Vignesh Gopakumar** — retained from the previous list *only as reserve*. His
   conformal-prediction-on-the-residual work was the nearest prior art to the **trust
   layer, which has been cut from this paper**. He is no longer the most relevant
   referee, and suggesting him would signal that we think the paper is still about
   residual-based trust. Use only if the editor asks for more names.

---

## Do not suggest

- Everyone in the conflict table above.
- Anyone on the **current JOCS editorial board** — unscreened; check first.
- **George Em Karniadakis**, **Dongbin Xiu**, **Charbel Farhat**, **George Biros** —
  JCP board members as of 2026-08-24, and Karniadakis is a plausible *handling editor*
  for this paper at any venue in scope. Their JOCS status is unchecked.

## What changed, and why

The old list's core was two numerical-analysts who could referee a
consistency-floor theorem, plus a conformal-prediction specialist. The theorem section
and the trust layer are both gone, so that rationale is void — keeping the list would
have told the editor we had not noticed our own paper had changed.

The new list's organising idea is that **no single reviewer can referee this paper.**
The measure-dependence claim, the representation ceiling and the fairness of the
interpolation baseline are three separate judgments drawn from three separate
literatures, and a panel missing any one of them will either wave the paper through or
reject it for the wrong reason. Groups 2 and 3 are deliberately specified as
*competencies to request* rather than named individuals, because naming a plausible-
sounding wrong person is worse than telling the handling editor precisely what expertise
the paper needs.
