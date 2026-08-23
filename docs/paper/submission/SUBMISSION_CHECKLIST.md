# JCP Submission Checklist — "NeuroForge: Self-Auditing Neural CFD Surrogates with Calibrated Physics-Residual Trust"

Target: **Journal of Computational Physics** (Elsevier).

**Submission portal (verified 2026-08-23):** Editorial Manager's *Submit New
Manuscript* now **redirects** to Elsevier's new service at
<https://submit.elsevier.com/JCOMP>. That service is a separate flow: it says it
cannot be used to continue an EM-started submission, and it emails EM log-in details
*after* you finish. Sign in with **st_a.jabbary@urmia.ac.ir** (the corresponding-author
address and the identity behind EM account `AJabbary-884`) — Chrome autofills
`light.knight32@gmail.com` there, which would file the paper under a second Elsevier
identity. "Try another account" → that email → "Sign in with a one-time link" avoids
both the wrong account and the wrong saved password.

History: desk-rejected at CMAME (CMAME-D-26-03937, 2026-08-02, "no new computational
methodology within scope; suggest ML-oriented journal") under the old engine-framed title.
Repositioned (title/abstract/claims) + hardened (selective prediction, multi-split
conformal, bootstrap CIs, repaired force integrator, 5-seed extension) before JCP.

**Format (guide verified 2026-08-14; portal re-verified 2026-08-23):** the guide makes
*no* mention of "Your Paper Your Way", and says you will be asked for "all relevant
editable source files upon submission or revision". The **submission portal itself is
narrower** and it governs: its on-page instructions read *"If you have written your
manuscript using LaTeX you need to upload a PDF. We will not attempt to extract your
metadata. You do not need to upload your LaTeX source files until revision."* The
required-file dropdown offers only **Manuscript / Declaration of competing interests /
Cover letter**; LaTeX sources appear under *optional* files. So: PDF now, sources at
revision.

**→ Submit `docs/paper/neuroforge_cfd_elsevier.pdf` (elsarticle, A4, 46 pp), NOT the TMLR-styled PDF.**
The manuscript now builds two ways from shared content, so they cannot drift:

| build | wrapper | for |
|---|---|---|
| **elsarticle** | `neuroforge_cfd_elsevier.tex` | **the JCP submission** |
| TMLR-styled | `neuroforge_cfd.tex` | the arXiv/preprint build (TMLR is no longer the fallback — see Fallback) |

Shared by both: `preamble.tex`, `abstract.tex`, `body.tex`, `sections/`, `refs.bib`.
Source files to upload: `neuroforge_cfd_elsevier.tex`, `preamble.tex`, `abstract.tex`,
`body.tex`, `sections/residual_floor_theorem.tex`, `refs.bib`,
`neuroforge_cfd_elsevier.bbl`, and the 9 figure PDFs from `results/figures/`.
Both builds are warning-clean (0 overfull / underfull / undefined references).

**Two build settings that must not be reverted** (both fixed 2026-08-23):

1. `\documentclass[preprint,12pt,a4paper]{elsarticle}` — without `a4paper` the class
   emits **US letter**, which Elsevier does not use.
2. `\bibliographystyle{elsarticle-num-names}` — **not** `elsarticle-num`. The plain
   `-num` style writes bare `\bibitem{key}` with no author data, so every `\citet{...}`
   in the paper rendered as **"(author?) [23]"**. There were 12 of these in Related
   Work, in the PDF that was first uploaded. The TMLR build uses a natbib-compatible
   `.bbl` and never showed them, which is why it went unnoticed. If you regenerate the
   `.bbl`, delete it first and re-run `latexmk` so BibTeX picks up the right style, then
   check: `pdftotext neuroforge_cfd_elsevier.pdf - | grep -c 'author?'` must print `0`.

**Abstract:** JCP requires ≤ 250 words. Counted on the *rendered* text (the way a
copyeditor would), the abstract body is **248–249** (the
`Keywords:` line is not part of it — EM has a separate keywords field). Two constructions
make naive counters read ~252: `$\approx 0.9$` and the em-dash in `it---especially`. If EM
rejects the count, the smallest lossless cuts are "with AUROC $\approx 0.9$" → "with AUROC
0.9" and dropping "clean" from "a clean two-way dissociation".

**Keywords:** 7 supplied (limit is 1–7); none contain "and"/"of", per guide.

## Ready now (in `docs/paper/submission/` + paper back-matter)
- [x] Manuscript PDF — `docs/paper/neuroforge_cfd_elsevier.pdf` (elsarticle, A4, 46 pp, builds clean; the TMLR-styled `neuroforge_cfd.pdf` is the arXiv/preprint build)
- [x] Cover letter — `cover_letter.md` (retargeted to JCP, new title + new results)
- [x] Highlights — `highlights.txt`, **bullets only, nothing else**. It previously also
      carried a title line, the Elsevier rule restated as a note, and a character-count
      appendix; all of that was working scaffolding and none of it may reach the
      journal, so the file is now exactly the five bullets and is upload-ready as-is.
      (Do not add a second `Highlights.txt` alongside it — Windows' filesystem is
      case-insensitive, so the two names are the same file and one silently clobbers
      the other.) Guide limits, kept here now that the appendix is gone: **3–5 bullets,
      each ≤ 85 characters including spaces**, and the word "highlights" must appear in
      the file name. Current lengths: 83 / 81 / 82 / 75 / 73.
- [x] Data & code availability — in paper back-matter (Zenodo DOI 10.5281/zenodo.21277928)
- [x] CRediT author statement — in paper back-matter (two authors)
- [x] Declaration of competing interests — in paper back-matter (none)
- [x] Declarations of generative-AI use — in paper back-matter (Elsevier policy)
- [x] Reproducibility artifacts — committed code, REPRODUCE.md, manifest with hashes
- [x] Suggested reviewers — `suggested_reviewers.md` (re-check conflicts before entry)
- [x] Funding statement — added to back-matter 2026-08-23 ("no specific grant",
      confirmed by the author). The guide lists funding as a required policy item.
- [x] Acknowledgements — added to back-matter 2026-08-23. The guide requires a separate
      section placed *directly before the reference list*; it was missing entirely.
- [x] Computational cost — JCP's aims and scope require "efficacy, robustness,
      computational complexity, as well as reproducibility". Complexity was one
      sentence; it is now a measured subsection (`scripts/measure_inference_cost.py`,
      `results/control/inference_cost.json`).

## Author must do (needs your accounts / decisions)
- [ ] Start the JCP submission in Editorial Manager (jcomp). EM username — **confirm which**:
      this file previously said `Ali_Kin4`, but that is the *arXiv* username; the Editorial
      Manager account on record is `AJabbary-884`. Try the latter first.
- [ ] During the EM flow, **decline the SSRN preprint opt-in**. JCP offers to auto-post the
      manuscript to SSRN at desk-review; accepting mints a *second* preprint DOI alongside
      the existing arXiv one, which is not wanted.
- [ ] Confirm author metadata (both authors, ORCIDs, affiliations; co-author email
      confirmation will go to Kasra).
- [ ] Enter 3–5 suggested reviewers from `suggested_reviewers.md`.
- [ ] Attach graphical abstract — upload **`results/figures/graphical_abstract.pdf`**, NOT
      the .png: Elsevier's preferred types are TIFF, EPS, PDF or MS Office, and PNG is not
      on that list. Spec is 531 x 1328 px (h x w) "or proportionally more", i.e. aspect **2.5009**,
      readable at 5 x 13 cm. Rebuilt 2026-08-23 at 13.0 x 5.2 in = **3900 x 1560 px,
      aspect exactly 2.50** (it was 2.60), panels 1–2 cropped to the body and near wake
      and all type enlarged so the smallest label clears ~7 pt at the 5 cm display size.
      "predict -> audit -> calibrate -> decide" strip built by
      `scripts/make_graphical_abstract.py` from committed numbers only; supersedes the
      old two-roles figure in the private `cmame_submission/` repo.
- [ ] At the JCP licensing step choose the **subscription** route, NOT open access. JCP is
      hybrid; the OA option carries an APC and is not wanted.
- [ ] Final read-through of the PDF.
- [x] Post arXiv v2 (new title/abstract) **before or same-day as** the EM submit — not
      after. **Done** (submitted 2026-08-21, replacement of arXiv:2607.10333). `cover_letter.md:57` points the editor at arXiv:2607.10333; that page still
      shows the *old* title, so an editor clicking it during desk-check lands on a
      differently-titled paper. This journal already desk-rejected the work once.

## At revision stage (only if accepted-with-revisions)
- [x] Reformat to `elsarticle` — **done ahead of submission**, see the format note above.
- [x] Provide editable source files (.tex, figures) — in repo, list above.

## Fallback

**Hard constraint: no pay-to-publish.** No article-processing charges, so fully open-access
venues are out. JCP is hybrid, so the primary target already costs nothing — take the
subscription route, not the OA option, at the licensing step.

**Ruled out:** TMLR (author's call). **MLST** (IOP) — looked ideal on scope, but it is fully
open access at **£2,500 / $3,125** on acceptance, *and* its research papers are "normally not
more than 8500 words" against this manuscript's ~14,700. Both disqualifying.

### If JCP desk-declines → Engineering Applications of Artificial Intelligence (EAAI)

Verified against the live guide for authors, 2026-08-21:

| | |
|---|---|
| Cost | **Hybrid — free via the subscription route** (it "supports open access", it does not require it) |
| Impact factor | **9.0** (CiteScore 11.7) — higher than JCP's 3.9 |
| Length limit | **none** — the only word limit in the entire guide is the 250-word abstract |
| Publisher | Elsevier, Editorial Manager — the existing account works |
| Society | A journal of IFAC (International Federation of Automatic Control) |

Scope asks for "the practical application of AI methods in all branches of engineering", and
requires papers be "validated using public data sets for easy replicability of the research
results" — which this paper satisfies unusually well (AirfRANS + DeepCFD, hash-manifested).
The risk is the flip side: EAAI wants a *real-world engineering application*, and this is a
benchmark-validated methods paper, so lead the cover letter with the aerodynamic-design use
case rather than the dissociation finding.

**Three concrete prep items before submitting there (do not discover these at 2am):**

1. **Double-anonymized review.** EAAI conceals author identity. The title page (with author
   details) and an **anonymized manuscript** (without them) are *separate files*. The
   anonymized file must contain no names, affiliations, or acknowledgements.
2. **The repository links de-anonymize us.** The back-matter cites
   `github.com/ali-kin4/neuroforge-cfd` and the Zenodo DOI; the GitHub URL contains the
   author's username. For review, swap to an anonymized mirror (e.g. anonymous.4open.science)
   or a "withheld for review" note, and restore the real links at acceptance.
3. **Keywords: 7 → 6.** EAAI allows 1–6; the paper currently lists 7. Drop one (suggest
   "trust calibration", already implied by "conformal prediction").

Abstract is fine as-is: EAAI's cap is 250 words and the abstract is 249. Highlights are fine:
3–5 bullets, ≤85 characters — the existing file already complies.

### Third option, if EAAI declines

**Computers & Fluids** (Elsevier, hybrid/free; IF 3.0, CiteScore 5.6). CFD-side audience, so
expect harder scrutiny of the uniform 128^2 grid and the near-wall blindness than an AI
venue would apply. Lower metrics than both above, but scope-safe and costs nothing.
