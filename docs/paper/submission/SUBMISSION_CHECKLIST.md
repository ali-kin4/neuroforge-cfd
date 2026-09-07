# Computers & Fluids Submission Checklist

Target: **Computers & Fluids** (Elsevier), ISSN 0045-7930. Impact factor 3.0,
CiteScore 5.6.

Manuscript: **"The ground truth fails its own physics check: what a surrogate-side
RANS residual can and cannot certify"**.

> **Requirements below were read directly from the journal's own pages on 2026-09-07**
> (`.../computers-and-fluids/publish/guide-for-authors` and `.../publish/open-access-options`).
> Earlier versions of this file worked from search extraction because ScienceDirect
> returns HTTP 403 to automated fetchers; it does not block a real browser. Every row
> marked **[V]** was read in full text, not inferred.

## History — read before writing anything to the editor

| Date | Venue | Outcome |
|---|---|---|
| 2026-08-02 | CMAME (CMAME-D-26-03937) | Desk reject: "no new computational methodology within scope; suggest ML-oriented journal" |
| 2026-08-25 | JCP | Submitted |
| 2026-09-07 | JCP | Desk reject: "originality with respect to other published papers is too questionable". **Will not reconsider — do not appeal or email the editor.** |

Both rejections were desk screens, on the same axis, without review. The paper was
therefore **reframed**, not merely reformatted: see `docs/paper/review/RESOLUTION.md`.
The cover letter deliberately does **not** disclose these; see the note at the foot of
`cover_letter.md` for the reasoning and how to reverse it.

## The one thing most likely to go wrong

**C&F wants editable source AT SUBMISSION. JCP did not.** The JCP portal said *"If you
have written your manuscript using LaTeX you need to upload a PDF… You do not need to
upload your LaTeX source files until revision."* **That rule does not carry over.** The
C&F guide says [V]:

> "We ask you to provide editable source files for your entire submission (including
> figures, tables and text graphics)." … "A PDF is not an acceptable source file."

So upload the LaTeX sources and the figure files, not only the PDF. Do not reuse the
JCP upload plan.

## Required files

| File | Where | Status |
|---|---|---|
| Manuscript source (LaTeX) | `neuroforge_cfd_elsevier.tex` + `preamble.tex`, `abstract.tex`, `body.tex`, `sections/residual_floor_theorem.tex`, `refs.bib`, `.bbl` | ready |
| Manuscript PDF (built from the above) | `docs/paper/neuroforge_cfd_elsevier.pdf` | ready — **elsarticle build, not the TMLR one** |
| Figures, as separate files | `results/figures/*.pdf` | ready (vector PDF; the guide accepts EPS/PDF for vector art) |
| **Highlights** [V] | `docs/paper/submission/highlights.txt` | ready — **3–5 bullets, ≤85 chars each incl. spaces**; ours are 5 at 76–79. Filename must contain "highlights". |
| Cover letter | `docs/paper/submission/cover_letter.md` | ready |
| Suggested reviewers | `docs/paper/submission/suggested_reviewers.md` | ready — **verify board membership at C&F first** |
| **Declaration of competing interests** [V] | — | **AUTHOR STEP, cannot be generated from here.** The guide requires the declarations tool's output uploaded as a **.doc/.docx** at the "attach/upload files" step. Author signatures are not required. Select "I have nothing to declare". |
| Data statement [V] | — | **AUTHOR STEP** at submission. Research data is **Option C**: deposit, cite and link. Satisfied by Zenodo DOI 10.5281/zenodo.21277928, which the paper cites. |

## Verified requirements [V]

| Item | Value | Ours |
|---|---|---|
| **Manuscript length limit** | **None.** No word, page or figure cap anywhere in the guide. | ~14,700 words — **not a problem**, and this was the open risk that ruled out RESS (13,000 cap). |
| Abstract | ≤ **250 words** | 241 |
| Keywords | **1–7**; multi-word keywords joined by "and"/"of" discouraged | 7, none joined that way |
| Highlights | **Required**, 3–5 bullets, ≤85 chars | 5, at 76–79 |
| Peer review | **Single anonymized** | No anonymisation work. Preprint, system name, GitHub and Zenodo links all stay. |
| Article type | Original research paper | Yes |
| Section numbering | Numbered 1, 1.1, 1.1.1; cross-reference by number, not "the text" | elsarticle handles it |
| Appendices | Lettered A, B; equations Eq. (A.1) | n/a |
| References | **No strict format at submission**, but must be internally consistent and complete | BibTeX, consistent |
| Preprint references | Must be marked "preprint" or name the server, with the preprint DOI | check `refs.bib` for arXiv entries |
| CRediT | Required | present |
| AI-use declaration | Required, titled section before the reference list | present |
| Funding statement | Required; use the "no specific grant" sentence if none | **check it is present** |
| Self-archiving embargo | **24 months**; published version may not go on ResearchGate/Academia.edu | Same as JCP. arXiv preprint unaffected. |

## Publishing route — the hard constraint

**Take the SUBSCRIPTION route.** The open-access page says, verbatim [V]:

> Subscription — **"No open access publication fee."**

The APC is **USD 3890** and applies *only* if open access is chosen. The page also
states the choice "will have no effect on the peer review process or acceptance of your
submission". Do not let an acceptance flow silently flip this to open access at the
licensing step — that is exactly where the fee is triggered. See `no-apc-venues-only`.

## Scope fit — the editor reads against this list

The Aims and scope paragraph names four things it wants from an ML paper. Three are now
strengths; one is not, and the cover letter pre-empts it.

| Asked for | Us |
|---|---|
| Comparison with traditional numerical reconstruction methods | **Weakest.** No controlled speed-up claim; OpenFOAM/SU2 backends unimplemented. Pre-empted in the cover letter, and §`sec:regime` is a direct contrast against solver-consistent correction. |
| Training vs validation cases, with diversity | Strong |
| Physical consistency / theoretical analysis | **Strongest** — now the spine of the paper |
| Limitations as well as merits | Strong — seven claims withdrawn or narrowed |

## Before you click submit

- [ ] `python scripts/check_submission.py` passes
- [ ] Both PDFs rebuilt from current source; no undefined references in the log
- [ ] `pytest -q` passes
- [ ] Editorial board of **Computers & Fluids** checked against the suggested-reviewer list
- [ ] Funding statement present
- [ ] Corresponding author signed in as **st_a.jabbary@urmia.ac.ir** — Chrome autofills
      `light.knight32@gmail.com`, which would file the paper under a second Elsevier
      identity. Editorial Manager account is `AJabbary-884`.
- [ ] Subscription route selected at the licensing step
- [ ] arXiv v4 posted or scheduled, so the preprint and the submission do not diverge
      (see `docs/paper/submission/arxiv_v4/`)

## Fallbacks, in order

EAAI → RESS (**13,000-word cap — would require cutting**) → CPC → JOCS or Engineering
with Computers. Rationale and per-venue verification in
`docs/paper/review/venue_plan.md`.
