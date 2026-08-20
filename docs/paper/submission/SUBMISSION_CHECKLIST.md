# JCP Submission Checklist — "NeuroForge: Self-Auditing Neural CFD Surrogates with Calibrated Physics-Residual Trust"

Target: **Journal of Computational Physics** (Elsevier). Submission portal:
Editorial Manager (https://www.editorialmanager.com/jcomp/).

History: desk-rejected at CMAME (CMAME-D-26-03937, 2026-08-02, "no new computational
methodology within scope; suggest ML-oriented journal") under the old engine-framed title.
Repositioned (title/abstract/claims) + hardened (selective prediction, multi-split
conformal, bootstrap CIs, repaired force integrator, 5-seed extension) before JCP.

**Format (verified against the JCP guide for authors, 2026-08-14):** the guide makes *no*
mention of "Your Paper Your Way". It states plainly that "a PDF is not an acceptable source
file" and that you will be asked for "all relevant editable source files upon submission or
revision" — so sources go up at *initial* submission, not just at revision.

**→ Submit `docs/paper/neuroforge_cfd_elsevier.pdf` (elsarticle, 48 pp), NOT the TMLR-styled PDF.**
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

**Abstract:** JCP requires ≤ 250 words. The current abstract body is **249** (the
`Keywords:` line is not part of it — EM has a separate keywords field). Two constructions
make naive counters read ~252: `$\approx 0.9$` and the em-dash in `it---especially`. If EM
rejects the count, the smallest lossless cuts are "with AUROC $\approx 0.9$" → "with AUROC
0.9" and dropping "clean" from "a clean two-way dissociation".

**Keywords:** 7 supplied (limit is 1–7); none contain "and"/"of", per guide.

## Ready now (in `docs/paper/submission/` + paper back-matter)
- [x] Manuscript PDF — `docs/paper/neuroforge_cfd_elsevier.pdf` (elsarticle, 48 pp, builds clean; the TMLR-styled `neuroforge_cfd.pdf` is the arXiv/preprint build)
- [x] Cover letter — `cover_letter.md` (retargeted to JCP, new title + new results)
- [x] Highlights — `highlights.txt` (5 bullets, each ≤85 chars, refreshed)
- [x] Data & code availability — in paper back-matter (Zenodo DOI 10.5281/zenodo.21277928)
- [x] CRediT author statement — in paper back-matter (two authors)
- [x] Declaration of competing interests — in paper back-matter (none)
- [x] Declarations of generative-AI use — in paper back-matter (Elsevier policy)
- [x] Reproducibility artifacts — committed code, REPRODUCE.md, manifest with hashes
- [x] Suggested reviewers — `suggested_reviewers.md` (re-check conflicts before entry)

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
      on that list. Spec is 531 x 1328 px (h x w) "or proportionally more"; the .png form is
      1500 x 3900 (aspect 2.60 vs 2.50 — both dimensions well above the minimum, fine).
      "predict -> audit -> calibrate -> decide" strip built by
      `scripts/make_graphical_abstract.py` from committed numbers only; supersedes the
      old two-roles figure in the private `cmame_submission/` repo.
- [ ] At the JCP licensing step choose the **subscription** route, NOT open access. JCP is
      hybrid; the OA option carries an APC and is not wanted.
- [ ] Final read-through of the PDF.
- [ ] Post arXiv v2 (new title/abstract) **before or same-day as** the EM submit — not
      after. `cover_letter.md:57` points the editor at arXiv:2607.10333; that page still
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
