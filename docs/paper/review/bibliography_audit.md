# Bibliography audit — `docs/paper/refs.bib`

**Date:** 2026-09-07 · **Branch:** `paper1/reframe-after-jcp` (confirmed via `.git/HEAD`; the
session-start `gitStatus` snapshot saying `paper2/openfoam-warm-start` was stale)
**Scope:** all **43** entries in `refs.bib` (the task brief said "roughly 37"; the true count is 43),
verified against primary sources. Plus Part 2 characterisation checks and Part 3 missing-citation
triage.

---

## HEADLINE VERDICT

**The bibliography is safe to submit after the corrections recorded below, which have been
applied.** No entry was fabricated. No entry failed to exist. Every author list, venue, volume,
page range and year now matches a primary source, with the four explicitly-flagged residual
uncertainties listed at the end.

Three findings were material:

1. **`brandt2011multigrid` carried a DOI that resolved to a completely different book.**
   `10.1137/1.9781611972030` is Pierre Grisvard, *Elliptic Problems in Nonsmooth Domains*
   (SIAM Classics). The correct DOI for Brandt & Livne is `10.1137/1.9781611970753`. This is
   exactly the misattribution class that would be fatal on a third submission, and it was in the
   hand-added block of six classical references, not in the ML block. **Fixed.**
2. **`refs.bib` contained malformed syntax** — a stray `,` + `}` fragment at former lines 355-356,
   debris from the removal of the known-bad `bochev2009lsfem` entry. **Removed.**
3. **Two entries presented published papers as arXiv preprints** — the mirror image of the error
   the brief warned about. Geo-FNO is a JMLR 2023 paper; Gopakumar et al. is ICML 2025.
   DrivAerNet++ is NeurIPS 2024 D&B. **All three fixed.**

**`mukherjee2026certification`, flagged in the brief as unverified and potentially fabricated,
is real**, and its author list matches the entry exactly. It was the single biggest risk in the
file and it cleared.

**Important limitation:** Bash is disabled in this session, so I could **not** run `bibtex`/`biber`
or a LaTeX build. This audit certifies *metadata correctness*, not that the file compiles. The
malformed fragment I removed was of unverified build impact (BibTeX skips junk between entries;
biber is less forgiving). **Someone must run a clean build before submission.** Note also that
`neuroforge_cfd.bbl` and `neuroforge_cfd_elsevier.bbl` are pre-built and are now **stale** with
respect to these corrections — they must be regenerated.

---

## PART 1 — entry-by-entry verification

Status key: **V** = verified as-was · **C** = corrected · **V+** = verified, missing metadata added
· **?** = could not fully verify (flagged)

| # | Key | Status | Primary source consulted | Finding / correction |
|---|-----|--------|--------------------------|----------------------|
| 1 | `li2021fno` | V | `export.arxiv.org/api/query?id_list=2010.08895` | Title, all 7 authors, ICLR 2021 correct. |
| 2 | `li2022geofno` | **C** | arXiv API `journal_ref` + `jmlr.org/papers/v24/23-0064.html` | **Was `journal = {arXiv preprint}`. Actually published: JMLR 24(388):1–26 (2023).** Also `Daniel Z.` → `Daniel Zhengyu`. Year 2022→2023; arXiv id retained in `note`. ⚠ arXiv's `journal_ref` gives pages "18593–18618", which is the **ACM Digital Library aggregated pagination**, not JMLR's own; the canonical JMLR record is 1–26. Use JMLR's. |
| 3 | `ranade2025domino` | V | arXiv API 2501.13350 | 7 authors correct. No `journal_ref` — correctly labelled a preprint. |
| 4 | `wu2024transolver` | **V+** | arXiv API 2402.02366; `proceedings.mlr.press/v235/wu24r.html` | ICML 2024 claim **correct**. Added PMLR v235, pp. 53681–53705. (Spotlight.) |
| 5 | `luo2025transolverpp` | V | arXiv API 2502.02414 | 7 authors correct. No `journal_ref` — preprint label stands. |
| 6 | `raissi2019pinn` | **V+** | Crossref `10.1016/j.jcp.2018.10.045` | JCP 378:686–707 (2019) exact. DOI was missing; added. |
| 7 | `learned2023residualcorrection` | V | arXiv 2306.12047 + Crossref `10.1016/j.cma.2023.116595` | Jha **sole author**, CMAME 419:116595, 2024 — all correct. Key says `2023`, entry says `2024`; cosmetic only, BibTeX uses the field. Left as-is. |
| 8 | `lippe2023pderefiner` | V | arXiv API 2308.05732 | All 5 authors, NeurIPS 2023 correct. |
| 9 | `lakshminarayanan2017ensembles` | V | arXiv API 1612.01474 (comment: "NIPS 2017") | Correct. |
| 10 | `gal2016dropout` | V | arXiv API 1506.02142 (comment: "Published in ICML 2016") | Correct. |
| 11 | `ma2024uqno` | V | TMLR 2024, OpenReview `cGpegxy12T` (carried forward + re-corroborated) | 4-author list **incl. David Pitt is correct for the TMLR version**. ⚠ Note: arXiv v2 (2402.01960) lists only **3** authors (no Pitt), so the `note = {arXiv:2402.01960}` points at a differently-authored version. Entry is right; discrepancy is upstream. |
| 12 | `hsieh2019neuralpde` | V | arXiv 1906.01200; OpenReview `rklaWn0qK7` | ICLR 2019, all 5 authors correct. |
| 13 | `marwah2023fnodeq` | V | arXiv API 2312.00234 (comment: "NeurIPS 2023") | All 6 authors correct. |
| 14 | `bai2019deq` | V | arXiv API 1909.01377 (comment: "NeurIPS 2019 Spotlight Oral") | Correct. |
| 15 | `winston2020monotone` | V | arXiv API 2006.08591 (comment: "NeurIPS 2020") | Correct. |
| 16 | `fung2022jfb` | V | arXiv API 2103.12803 | All 6 authors, AAAI 2022 correct. |
| 17 | `bonnet2022airfrans` | V | NeurIPS 2022 D&B proceedings page (hash `94ab7b23…`) | Correct, incl. **`Mazari, Jocelyn Ahmed`** — this matches the NeurIPS version of record. arXiv's "Ahmed Jocelyn Mazari" is the outlier; **do not "fix" it to match arXiv.** (Official title uses an en-dash "Navier–Stokes"; cosmetic.) |
| 18 | `ahmedml2024` | **C** | arXiv API 2407.20801 | Author names incomplete: → `Maddix, Danielle C.` and `Shabestari, Parisa M.` No `journal_ref`; preprint label correct. |
| 19 | `elrefaie2024drivaernet` | **C** | arXiv API 2406.09624, `journal_ref` | **Was `arXiv preprint`. Actually NeurIPS 2024 Datasets & Benchmarks Track.** Fixed. |
| 20 | `krishnapriyan2021failure` | **V+** | arXiv 2109.01050; NeurIPS 2021 proceedings PDF | All 5 authors, NeurIPS 2021 correct. Added missing `note = {arXiv:2109.01050}`. |
| 21 | `wang2022ntk` | **V+** | Crossref `10.1016/j.jcp.2021.110768` | JCP 449:110768 (2022) exact; authors Wang/Yu/Perdikaris correct. DOI added. |
| 22 | `trefethenbau1997` | **V+** | Crossref `10.1137/1.9780898719574` | SIAM, 1997 confirmed. DOI + address added. ⚠ Crossref gives the second author as **"David Bau, III"**; entry has `Bau, David`. Left unchanged deliberately — the 3-part BibTeX suffix form `Bau, III, David` is a build-risk I cannot test with Bash disabled. Optional refinement, not an error of substance. |
| 23 | `roy2025anchor` | V | arXiv API 2512.19643 | 3 authors, title, 2025-12-22 posting all correct. Preprint — correctly labelled. |
| 24 | `song2026structureaware` | V | arXiv API 2603.11052 (carried forward) | 7 authors correct. **Unrefereed preprint — correctly labelled** `arXiv preprint`. |
| 25 | `ribeiro2020deepcfd` | V | arXiv API 2004.08826 | 4 authors correct. No `journal_ref`; preprint label correct. |
| 26 | `gopakumar2025pre` | **C** | arXiv API 2502.04406, `journal_ref: ICML 2025` | **Was `arXiv preprint`. Actually ICML 2025.** Fixed. All 8 authors correct. |
| 27 | `mukherjee2026certification` | **V** | arXiv API 2603.19165 | **EXISTS.** Title exact; authors Mukherjee / Fitzsimmons / Del Rey Fernández / Liu exact. Unrefereed preprint — correctly labelled. |
| 28 | `huang2025physicscorrect` | V | arXiv API 2507.02227, `journal_ref: Proc. AAAI 2026`, comment "AAAI 2026 Oral" | Entry correct as written (`year=2025` + `note={AAAI 2026}`). ⚠ Now formally an **AAAI 2026 Oral**; upgrading to `@inproceedings{... year=2026}` would be more accurate but would change every rendered "(Huang & Perdikaris, 2025)" in prose other agents are editing. **Flagged, not changed — owner's call.** |
| 29 | `rigotti2026gist` | V | arXiv API 2603.16849 (carried forward) | 3 authors correct. Unrefereed preprint — correctly labelled. |
| 30 | `scherz2026evaluation` | V | arXiv API 2607.13866 (carried forward) | 3 authors correct. Unrefereed preprint — correctly labelled. |
| 31 | `jia2026multigranularity` | V | arXiv API 2607.17297 | 6 authors correct. Unrefereed preprint — correctly labelled. |
| 32 | `yu2026conformalpinn` | V | carried forward (JCP 561:114979, `10.1016/j.jcp.2026.114979`) | Verified earlier this session. Genuinely journal-published. |
| 33 | `garg2025dfuq` | V | carried forward (JCP 534:114012, `10.1016/j.jcp.2025.114012`) | Verified earlier this session. |
| 34 | `hillebrecht2025posteriori` | V | Crossref `10.1109/TNNLS.2023.3335837` | **Exact match**, incl. the early-access/final-issue combination the brief worried about: IEEE TNNLS **36(1):1583–1593, 2025**. No change needed. |
| 35 | `beckerrannacher2001dwr` | V | Crossref `10.1017/S0962492901000010` | Acta Numerica 10:1–102, 2001 exact. |
| 36 | `lei2026newtonkrylov` | V | arXiv API 2608.04400 (carried forward) | 4 authors exact; posted 2026-08-05, matching the entry's note. **Unrefereed preprint — correctly labelled.** |
| 37 | `zhang2026phymgn` | V | arXiv API 2602.14918 (carried forward) | 7 authors exact; posted 2026-02-16, matching the note. Unrefereed preprint — correctly labelled. |
| 38 | `mcgreivy2024weak` | V | Crossref `10.1038/s42256-024-00897-5` | Nature Machine Intelligence **6:1256–1269 (2024)** exact (issue 10). |
| 39 | `brandt2011multigrid` | **C — SERIOUS** | Crossref DOI lookup by title | **DOI `10.1137/1.9781611972030` resolved to Grisvard, *Elliptic Problems in Nonsmooth Domains* — a different book by a different author.** Corrected to `10.1137/1.9781611970753`. Authors/title/publisher/year were right. Added verified `volume = {67}` (Classics in Applied Mathematics no. 67, ISBN 9781611970746). |
| 40 | `stetter1978defect` | V | Crossref `10.1007/BF01432879` | Numer. Math. 29(4):425–443, 1978 exact. |
| 41 | `morin2000data` | V | Crossref `10.1137/S0036142999360044` | SINUM 38(2):466–488, 2000 exact. Matches the brief's expected values. |
| 42 | `bochevgunzburger2009book` | **V+** | Crossref `10.1007/b13382`; Springer AMS series listing | Springer, **2009** confirmed — the brief's worry that "year 2009 looks wrong" applied to the *deleted* `bochev2009lsfem` journal entry, **not** to this book, whose 2009 date is correct. Added verified `volume = {166}`. ⚠ Crossref returns the author order reversed ("Gunzburger, Bochev"); the book itself is **Bochev & Gunzburger**, so the entry is right and Crossref is wrong. |
| 43 | `rhiechow1983` | V | Crossref `10.2514/3.8284` | AIAA J. 21(11):1525–1532, 1983 exact. |
| — | `bochev2009lsfem` | **REMOVED** (already) | — | The known-bad entry is **absent** from the file — a previous session removed it correctly. It left the malformed `,`/`}` debris, which I removed. No citation anywhere in `docs/paper/*.tex` still references this key (verified by grep). |

**Orphan check:** every one of the 43 keys is cited at least once in `body.tex` or
`sections/residual_floor_theorem.tex`. No unused entries, no undefined citations.

---

## PART 2 — characterisation checks

Metadata being right is not enough; these are the claims the repositioning rests on. **All twelve
checked characterisations are accurate.** Two are near-verbatim faithful to the source.

### `lei2026newtonkrylov` — the central boundary argument ✅ CONFIRMED

The manuscript quotes this figure at least three times (`body.tex:136-138`, `260`, `1517-1520`).
The source abstract (arXiv:2608.04400) says, in the authors' own words:

> "the framework lowers the median residual L_2 ratio by **over seven orders of magnitude while
> substantially reducing field and aerodynamic errors**" … "deliver accurate, efficient and
> scalable **steady CFD** across industrial workflows."

- Seven orders of magnitude: **confirmed** (the manuscript's "over seven orders" at :1519 is exact;
  :137 drops "over", which is conservative and fine).
- Residual reduction and error reduction **co-occur**: confirmed — "while" is the source's own
  conjunction, not the manuscript's gloss. The manuscript is reproducing the authors' claim
  faithfully rather than inferring co-occurrence.
- **Steady CFD: confirmed** explicitly.

⚠ **One precision caveat to raise with the prose owner.** The benchmark is "geometries sampled from
actual **transonic** airfoil optimization trajectories" (plus a 3-D flying wing). `body.tex:1520`
says "residual down and error down, **in the same regime we study**." It is the same regime in the
sense that carries the argument (*steady*, which is all the boundary claim needs), but it is
transonic/compressible whereas NeuroForge is incompressible subsonic RANS. Consider softening to
"in the steady regime we study" to remove a cheap reviewer objection. **Not an error — a hardening
opportunity.**

### `zhang2026phymgn` — Appendix E ✅ CONFIRMED, quotation is verbatim

The cite is to Appendix E specifically. Appendix E exists, is titled *"Spatial-Temporal error
distribution of training data"*, and contains both the substance and the exact quoted string:

> "the physics-informed loss cannot be formulated using the absolute PDE residual alone; instead,
> it must be defined relative to the residual present in the ground-truth data"

This matches `body.tex:1541-1543` **word for word**, including the part inside quotation marks.
Appendix E also states the residual is prevented "from vanishing even for the ground-truth
solution." Both the Appendix E pointer and the quotation are correct.

### `mukherjee2026certification` ✅ CONFIRMED (and it exists)

`body.tex:300-302` claims it proves "vanishing residuals guarantee solution convergence only under
additional compactness assumptions." The abstract states:

> "We prove that **when neural approximations lie in a compact subset of the solution space**,
> vanishing residual error guarantees convergence to the true solution."

Exact. The word "compactness" is the source's own.

### `hillebrecht2025posteriori` ✅ CONFIRMED

`body.tex:292-294` claims it bounds a PINN's true error by its residual scaled by an operator
stability constant. Confirmed from the full text (arXiv:2210.03426, the preprint of the TNNLS
paper): Theorem III.1 gives ε(t) = ‖S(t)‖ζ₀ + ∫₀ᵗ‖S(t−s)‖ζ(s)ds — the residual bound ζ weighted by
**semigroup** norms — and Corollary III.2 makes the constant explicit as ε(t) ≤ Me^{ωt}ζ₀ +
∫Me^{ω(t−s)}ζ(s)ds. "Residual scaled by a stability constant of the underlying operator" is an
accurate description. ⚠ Minor: the mechanism is time-evolution/semigroup-based, so it is naturally
a statement about *dynamical* systems; the manuscript does not claim otherwise.

### `gopakumar2025pre` — closest prior art to the trust half ✅ CONFIRMED, both halves

This one had to be exactly right. Both halves check out.

- Residual-as-nonconformity-score: abstract says it "leverages **physics residual errors as
  nonconformity scores**, enabling data-free UQ with marginal and joint coverage guarantees."
- Transfer to solution space left open **as a set-propagation problem** — confirmed from the
  paper's own limitations discussion:
  > "Our method's coverage bounds exist in the **PDE residual space** rather than the Euclidean
  > space of physical variables. Transforming to physical space involves challenging **set
  > propagation** through integral operations, which may require approximations or expensive Monte
  > Carlo sampling."

`body.tex:280-283` — "in *residual* space, with the transfer to solution space left open as a
set-propagation problem" — is an unusually faithful characterisation; "set-propagation" is the
source's own term. The whitespace claim built on it is sound.

### `huang2025physicscorrect` — transient-only ✅ CONFIRMED (with a prose flag)

The scoping depends on this. Confirmed: the three benchmarks are Navier–Stokes, the wave equation
and Kuramoto–Sivashinsky, all framed as "error accumulation during **long-term rollouts**", i.e.
time-dependent. **No steady-state problem appears.** "Up to 100x" error reduction is also confirmed
verbatim. So `body.tex:238` ("up to 100× error reduction on transient benchmarks") and `:255`
("resolved transient PDEs stepped on their training grid") are both correct.

⚠ **Prose problem, `body.tex:1521`** (reporting, not fixing — I may not edit `body.tex`):
> "\citet{huang2025physicscorrect} report a similar success."

This sits immediately after the sentence establishing Lei et al. as *steady* CFD "in the same
regime we study," and immediately before "In **those works** the residual being driven is the
solver's own discrete operator." A reader can easily take "a similar success" as also asserting a
steady result. It does not literally say so, but since the paper's whole boundary argument turns on
Huang being transient and Lei being steady, this sentence blurs the very distinction it is there to
draw. Suggest making it explicit, e.g. "…report a similar success on transient rollouts."

### Remaining Part 2 items — all ✅ CONFIRMED

| Cite | Manuscript claim | Verdict |
|---|---|---|
| `roy2025anchor` | residual-based error estimator detects accumulating error in NO **time-marching**, triggers a classical solver | Exact. Abstract: "adaptively couples it with a classical numerical solver using a physics-informed, residual-based error estimator"; EMA of normalised PDE residual "to detect accumulating error and **trigger corrective solver interventions**". All six PDEs are time-dependent. |
| `jia2026multigranularity` | case-level quantile regression + residual-normalised point-level bands; DrivAerML; nonconformity scales **purely data-driven** | Exact on all four. "conformalized quantile regression constructs calibrated case-level intervals"; "residual-scale estimation and residual-normalized conformal calibration"; DrivAerML; and the "residual" there is a statistical spread estimate, **not** a PDE residual — so "purely data-driven" is right and the contrast the paper draws is legitimate. |
| `song2026structureaware` | epistemic-UQ scheme whose bands are **aligned with localised residual structure** | Exact; "uncertainty bands should align with the localized residual structures" is the source's own phrasing, and it reports "improved residual-uncertainty alignment". |
| `beckerrannacher2001dwr` | residual magnitude is not itself an error estimate; becomes one when weighted by an adjoint solution | Standard and correct characterisation of DWR. Metadata exact. |
| `krishnapriyan2021failure` | ties ill-conditioning to **optimisation difficulty** of PINN losses | Correct — the paper is about failure/trainability of PINN optimisation, not minimiser displacement. The manuscript's careful distinction at `residual_floor_theorem.tex:545-548` is accurate. |
| `wang2022ntk` | spectral bias and loss-term imbalance in PINN **training dynamics** | Correct (NTK analysis of training dynamics). |
| `trefethenbau1997` | forward-vs-backward error gap is classical | Correct; standard textbook support. |
| `rhiechow1983` | (momentum interpolation / checkerboard context) | Metadata exact; the classic Rhie–Chow reference is appropriate for the pressure–velocity coupling point at `residual_floor_theorem.tex:383`. |
| `learned2023residualcorrection` | learned residual-error corrector; "variational problems with certified residuals" | Correct — Jha's corrector is for *nonlinear variational boundary-value problems*, matching `body.tex:256`. |
| `bonnet2022airfrans` | primary benchmark, 2-D incompressible steady RANS over NACA airfoils | Correct. |
| `wu2024transolver` | strong geometry-general transformer backbone | Correct; ICML 2024 venue claim confirmed. |
| `scherz2026evaluation` | identifies Transolver(++) among the strongest aerodynamic surrogates | Supported: "In particular the Bi-Stride Multi-Scale GNN **and Transolver(++) are highlighted as promising** surrogate models for aerodynamical applications." ⚠ Note the benchmark is airfoil surface pressure + a 3-D aircraft, **not AirfRANS**; and Bi-Stride MS-GNN is co-equally highlighted. "Among the strongest" is fair; do not upgrade it to "the strongest". |
| `rigotti2026gist` | newer architectures have advanced the AirfRANS accuracy frontier | Supported: "GIST **sets state-of-the-art on the AirfRANS**, ShapeNet-Car, DrivAerNet, and DrivAerNet++ mesh benchmarks". |

---

## PART 3 — missing citations, ranked by necessity

**First, a correction to the brief:** four of the seven suggested works are **already present and
already cited**, so the whitespace analysis that generated the list is out of date:

- `brandt2011multigrid` — present, cited at `residual_floor_theorem.tex:570` (τ-correction). ✔
- `stetter1978defect` — present, cited at `:569` (defect correction). ✔
- `morin2000data` — present, cited at `:578` (data oscillation). ✔
- `mcgreivy2024weak` — present, cited at `body.tex:1546`. ✔

That leaves three genuinely absent. All three exist and were verified.

### 1. Cao et al. — **GENUINELY NEEDED** (strongest case)

> Lianghao Cao, Thomas O'Leary-Roseberry, Prashant K. Jha, J. Tinsley Oden, Omar Ghattas.
> "Residual-based error correction for neural operator accelerated infinite-dimensional Bayesian
> inverse problems." *Journal of Computational Physics* **486**:112104, 2023.
> DOI `10.1016/j.jcp.2023.112104` · arXiv:2210.03008

Why needed: it is a **residual-based correction of a neural operator** that proves a **quadratic
reduction** of approximation error — i.e. it is a direct, strong, *positive* result in exactly the
family the paper's negative result addresses, and it is not cited. A reviewer who knows it will ask
why. It also strengthens rather than threatens the paper's thesis: Cao et al. correct by solving a
linear variational problem **using the PDE residual with the true forward operator available**,
which is precisely the "operator availability" condition the paper identifies as the boundary
(`residual_floor_theorem.tex:558-561`). It belongs next to `learned2023residualcorrection` in
`body.tex:231-239` — note it shares an author (Jha) with the corrector already cited.

⚠ **Trap if you add it:** arXiv's own `journal_ref` for 2210.03008 is **corrupted** — it displays
"SIAM/ASA JUQ Vol 3, No 1, p.116-145 (2015)", which is actually the ROMES paper's journal ref.
**Do not copy it.** The DOI is right; use the Crossref record (JCP 486:112104, 2023).

### 2. Drohmann & Carlberg (ROMES) — **NEEDED IF the ROM sentence stays as written**

> Martin Drohmann, Kevin Carlberg. "The ROMES Method for Statistical Modeling of Reduced-Order-Model
> Error." *SIAM/ASA Journal on Uncertainty Quantification* **3**(1):116–145, 2015.
> DOI `10.1137/140969841` · arXiv:1405.5170

Why: `residual_floor_theorem.tex:553-558` makes a specific, load-bearing claim about goal-oriented
estimation working "beautifully for **reduced-order models**" because the FOM residual vanishes at
truth by construction — and cites only `beckerrannacher2001dwr`, which is a general FEM paper, not
a ROM paper. The sentence asserts something about ROMs and supports it with a non-ROM citation.
ROMES is the right support: it explicitly "uses **dual-weighted residuals** … as indicators" and
maps error indicators to ROM error, which is exactly the claim. Adding it converts a soft spot into
a precise one. **Recommend adding.**

### 3. DD-RNO — **ADJACENT, optional**

> T. A. Mehta, P. S. Bhati, H. D. Akolekar. "DD-RNO: A Domain-Decomposed Routed Neural Operator for
> Airfoil Flow Prediction." arXiv:2608.13490, 2026 (unrefereed preprint).

Why only adjacent: it is a strong, very recent **AirfRANS** result (claims 17×/12× velocity-MSE
reduction over the best baseline, and a large drag-rank-correlation improvement, ρ 0.250→0.997). It
has **no** self-verification, residual monitoring, UQ or certificate component, so it is not prior
art for any NeuroForge claim. Its only function would be currency on the accuracy frontier — a role
`rigotti2026gist` already fills at `body.tex:227`. **Add only if you want a second frontier
citation; do not pad.** If the paper anywhere implies AirfRANS accuracy leadership, this and GIST
are the two that must both be acknowledged.

**Recommendation: add #1 and #2 (four lines of BibTeX, both fully verified above); treat #3 as
optional.** I have not added them, since they require corresponding prose in `body.tex` /
`residual_floor_theorem.tex`, which other agents own.

---

## Prose problems observed (reported, not fixed — I may not edit these files)

1. **`body.tex:1521`** — "report a similar success" for `huang2025physicscorrect` reads as claiming
   a steady-CFD result in a steady-CFD paragraph. Huang is transient-only. See Part 2. **Highest
   priority of the four.**
2. **`body.tex:1520`** — "in the same regime we study" for Lei et al., which is transonic; true in
   the steady sense the argument needs, but loose. Suggest "in the steady regime we study."
3. **`body.tex:315`** — "**Within this journal** the same combination is being pursued from both the
   PINN and the operator side" introduces `yu2026conformalpinn` and `garg2025dfuq`, both *Journal of
   Computational Physics* papers. This is a **JCP-specific framing left over from the previous
   submission**. On a branch named `paper1/reframe-after-jcp` targeting a different venue, this
   sentence is factually wrong and silently discloses the prior submission target. **Must be
   rewritten** (e.g. "In the same literature…").
4. **`body.tex:225-227`** — "identify Transolver(++) among the strongest" is fair, but the source
   co-equally highlights Bi-Stride Multi-Scale GNN and does not evaluate on AirfRANS. Keep the
   hedge; do not strengthen.

---

## Explicitly flagged residual uncertainties

Per instruction, these are findings, not gaps:

1. **No build verification.** Bash is disabled; I could not run `bibtex`/`biber`/LaTeX. Metadata is
   certified; compilation is not. **The two `.bbl` files are now stale and must be regenerated.**
2. **`ma2024uqno` author-list provenance.** The 4-author TMLR list (incl. David Pitt) is corroborated
   by search results reporting the TMLR record, but **OpenReview itself was unreachable** (bot
   challenge on both `openreview.net` and `api2.openreview.net`); DBLP was also blocked (Anubis).
   The arXiv version has only 3 authors. I consider the entry correct, but the OpenReview page was
   not read directly by me this session.
3. **`luo2025transolverpp` and `ranade2025domino` venue status.** Both are labelled preprints, and
   neither has a `journal_ref` on arXiv. arXiv `journal_ref` is author-maintained and often not
   updated after acceptance, so I cannot rule out that either has since been published. The current
   preprint labelling is *defensible and not an overclaim* (it under-claims if anything), but worth
   a 30-second check by someone who follows those groups.
4. **`trefethenbau1997` second author suffix** — should strictly be "David Bau, III". Left unchanged
   to avoid an untestable BibTeX name-parsing change.

---

## Summary of changes applied to `refs.bib`

| Entry | Change |
|---|---|
| *(file-level)* | Removed malformed `,` / `}` debris left by the `bochev2009lsfem` removal |
| `brandt2011multigrid` | **DOI corrected** `10.1137/1.9781611972030` (wrong book — Grisvard) → `10.1137/1.9781611970753`; added `volume = {67}` |
| `li2022geofno` | preprint → JMLR **24(388):1–26 (2023)** (canonical JMLR pagination, per `jmlr.org/papers/v24/23-0064.html`); `Daniel Z.` → `Daniel Zhengyu` |
| `gopakumar2025pre` | preprint → `@inproceedings` / `booktitle` ICML 2025 |
| `elrefaie2024drivaernet` | preprint → `@inproceedings` / `booktitle` NeurIPS 2024 Datasets & Benchmarks Track |
| `ahmedml2024` | `Maddix, Danielle C.`; `Shabestari, Parisa M.` |
| `bochevgunzburger2009book` | added verified `volume = {166}` |
| `wu2024transolver` | added PMLR v235, pp. 53681–53705 |
| `raissi2019pinn` | added DOI `10.1016/j.jcp.2018.10.045` |
| `wang2022ntk` | added DOI `10.1016/j.jcp.2021.110768` |
| `krishnapriyan2021failure` | added `note = {arXiv:2109.01050}` |
| `trefethenbau1997` | added DOI `10.1137/1.9780898719574`, address |

No entry was deleted. No prose file was touched.

---

## Handoff (Bash was disabled — nothing is staged or committed)

Both changed files are on disk and **unstaged**. On a workstation that loses power without warning,
this needs doing promptly. On branch `paper1/reframe-after-jcp`, stage **by name only** (the brief
forbids `git add -A` / `git add .`):

```
git add docs/paper/refs.bib docs/paper/review/bibliography_audit.md
```

Then **regenerate both `.bbl` files** — `neuroforge_cfd.bbl` and `neuroforge_cfd_elsevier.bbl` are
pre-built and now stale with respect to these corrections — and run one clean LaTeX + BibTeX/biber
pass to confirm the file compiles. Metadata is certified; compilation is not.
