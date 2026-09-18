# Journal of Computational Science — Submission Checklist

Target: **Journal of Computational Science** (Elsevier), ISSN 1877-7503.

Manuscript: **"Near the wall, AirfRANS measures the coordinate system, not the model"**.

> **Read the verification legend before trusting any row.**
>
> - **[V-JOCS]** — read at source from the *Journal of Computational Science* guide for
>   authors, in a real browser, on the date given. Trustworthy.
> - **[GEN]** — Elsevier-wide policy, journal-independent.
>
> **ALL [V-CF] ROWS ARE CLOSED as of 2026-09-17.** The whole guide was read at source in a
> browser the author cleared the CAPTCHA in, and every rule previously inherited from
> Computers & Fluids has now been confirmed or corrected against the JOCS text. Nothing in
> this file is a hypothesis about JOCS any more. The raw guide is saved at
> `scratchpad/jocs_guide.txt` for the session that read it.

## History — read before writing anything to the editor

| Date | Venue | Outcome |
|---|---|---|
| 2026-08-02 | CMAME (CMAME-D-26-03937) | Desk reject: "no new computational methodology within scope; suggest ML-oriented journal" |
| 2026-08-25 | JCP | Submitted |
| 2026-09-07 | JCP | Desk reject: "originality with respect to other published papers is too questionable". **Will not reconsider — do not appeal or email the editor.** |
| 2026-09-12 | Computers & Fluids | **Withdrawn by us before filing, on conflict.** Its ML special-issue editorial is authored by Ashton, Dwight and Cinnella — authors of benchmarks this paper audits. |
| — | **Journal of Computational Science** | Current target |

Both rejections were desk screens, on the same axis, without review. The paper was therefore
**reframed and then split**, not reformatted: see `docs/paper/review/RESOLUTION.md` and
`jocs_rebuild.md`. The cover letter deliberately does **not** disclose the rejections; see the
notes at the foot of `cover_letter.md` for the reasoning and how to reverse it.

## The two things most likely to go wrong

**1. The source-files rule — RESOLVED, and the safe default was right.** [V-JOCS 2026-09-17],
verbatim:

> "We encourage you use our LaTeX template when preparing a LaTeX submission. You will be
> asked to provide all relevant editable source files **upon submission or revision**."

So JOCS may ask at submission. **Upload the sources with the PDF.** JCP's "not until revision"
rule does not carry over, which is exactly the trap this row existed to catch.

**2. The wrong Elsevier identity.** Chrome autofills `light.knight32@gmail.com`, which would
file the paper under a second Elsevier identity. Sign in as **st_a.jabbary@urmia.ac.ir**;
Editorial Manager account is `AJabbary-884`.

## Required files

| File | Where | Status |
|---|---|---|
| Manuscript source (LaTeX) | `neuroforge_cfd_elsevier.tex` + `preamble.tex`, `abstract.tex`, `body.tex`, `sections/residual_floor_theorem.tex`, `refs.bib`, `.bbl` | ready |
| Manuscript PDF (built from the above) | `docs/paper/neuroforge_cfd_elsevier.pdf` | ready — **elsarticle build, not the TMLR one** |
| Figures, as separate files | `results/figures/fig_bandratio.pdf` | ready (vector PDF) — one figure |
| **Highlights** | `docs/paper/submission/highlights.txt` | ready — 5 bullets at 77–83 chars, filename contains "highlights", separate editable file. **[V-JOCS 2026-09-17]: "3 to 5 bullet points, each a maximum of 85 characters, including spaces" — we comply.** |
| Cover letter | `docs/paper/submission/cover_letter.md` | ready — rewritten for JOCS 2026-09-16, headline updated 2026-09-18 for the coordinate-artifact result |
| Suggested reviewers | `docs/paper/submission/suggested_reviewers.md` | rebuilt for JOCS and for the current paper 2026-09-17. **Two browser checks still open: the JOCS board, and every affiliation/email.** |
| **Declaration of competing interests** | — | **AUTHOR STEP, cannot be generated from here.** [V-JOCS 2026-09-17], verbatim: *"The declarations tool should always be completed"*; authors with none select **"I have nothing to declare"**; *"The resulting Word document … should be uploaded at the 'attach/upload files' step … saved in the .doc/.docx file format. Author signatures are not required."* |
| Data statement | — | **AUTHOR STEP** at submission. **[V-JOCS 2026-09-17]: "you are required to state the availability of any data at submission"** — required, and it is a free-text availability statement in the submission flow, *not* the lettered A/B/C menu the C&F row assumed. Satisfied by Zenodo DOI 10.5281/zenodo.21277928, which the paper cites. |

## Requirements

| Item | Value | Verified | Ours |
|---|---|---|---|
| **Manuscript length limit** | **None for regular articles.** Limits exist only for *Communications* (6 double-spaced pages, ≤20 refs, ≤4 figures/tables) and *Correspondence* (4 pages, ≤15 refs, ≤3). | **[V-JOCS 2026-09-12]** | **15,476 words** — no rule against it, but see the note below |
| Abstract | ≤ **250 words** | **[V-JOCS 2026-09-12]** | **246**, measured by `check_submission.py` |
| Keywords | **1–7** | **[V-JOCS 2026-09-12]** | 7 |
| Highlights | **3–5 bullets, max 85 chars incl. spaces**; separate editable file, "highlights" in filename | **[V-JOCS 2026-09-17]** | 5, at 77–83 |
| Article type | Original research paper (**not** Communication/Correspondence — those carry the caps above) | [V-JOCS] | Yes |
| Peer review | **Single anonymized**; editors screen first, then **a minimum of two reviewers** | **[V-JOCS 2026-09-17]** | No anonymisation work. Preprint, system name, GitHub and Zenodo links all stay. |
| References | No strict format at submission; must be internally consistent | [V-JOCS 2026-09-17] | BibTeX, consistent |
| Preprint policy | **[V-JOCS 2026-09-17]:** *"Sharing preprints, such as on a preprint server, will not count as prior publication."* arXiv:2607.10333 is safe to disclose, and the cover letter does. | **[V-JOCS]** | ✓ refs audited 2026-09-17: 52 entries, every arXiv-only preprint names the server AND carries its 10.48550 DOI; no entry lacks a venue field |
| CRediT | Required | [GEN] | present |
| AI-use declaration | Required, titled section before the reference list | [GEN] | present |
| Funding statement | Required; use the "no specific grant" sentence if none | [GEN] | present — `body.tex` \section*{Funding}, "no specific grant" wording ✓ (checked 2026-09-17) |
| Self-archiving embargo | 24 months (C&F figure, journal-specific — **still unconfirmed for JOCS**, and it only bites after acceptance) | [V-CF] | arXiv preprint unaffected |

**On length.** 15,476 words is long, and it grew during the September rework as controls were
added. There is no rule against it at this venue, but length is an editorial signal even where
it is not a limit, and this paper has been desk-screened twice. That is a judgement call, not a
gate — `check_submission.py` reports the count and does not block on it. If it is ever cut, cut
evidence last: the two desk rejections were about originality, never about length.

## Publishing route — the hard constraint

**Take the SUBSCRIPTION route.** Under Elsevier's hybrid model the subscription route carries
**no publication fee**, and the choice has no effect on peer review or acceptance. The
open-access APC for *this* journal has **not** been read at source — the USD 3890 figure in
earlier revisions of this file is **the Computers & Fluids APC** and must not be quoted for
JOCS. Whatever the number is, do not let an acceptance flow silently flip the licence to open
access at the licensing step: that is exactly where the fee is triggered. See
`no-apc-venues-only`.

## Scope fit — why this venue

Unlike C&F, the case here is **not** a scope-paragraph mapping; it is the published record.
JOCS has run this genre four times in five months (details and queries in
`docs/paper/review/journal_shortlist.md` §2.2–2.4):

| Paper | Date | Genre |
|---|---|---|
| *When simpler models win: a large-scale computational benchmark…* | 2026-08-27 | simple-baseline-beats-deep-model benchmark |
| *Accuracy vs efficiency: benchmarking GNNs on edge GPU hardware* | 2026-07-30 | head-to-head benchmark |
| *Benchmarking atom-level explainability against pharmacophore-computed labels* | 2026-07-08 | benchmark against computed reference |
| *Exploring the limitations of transformer models for metocean forecasting* | 2026-06-03 | limitations of a model class, in fluids |

The control matters as much as the hits: the inherited critique query returns **zero** on JOCS,
and the C&F record shows **zero** papers of this genre in seven years. The previous sweep's
terms were measuring the old headline.

## Before you click submit

- [x] **JOCS guide read at source 2026-09-17; every [V-CF] row closed** (one exception: the self-archiving embargo, which only matters after acceptance)
- [ ] `python scripts/check_submission.py` passes
- [ ] `python scripts/audit_paper_numbers.py` passes with no SKIP rows
- [ ] Both PDFs rebuilt from current source; no undefined references in the log
- [ ] `pytest -q` passes
- [ ] Editorial board of **the Journal of Computational Science** checked against the
      suggested-reviewer list — the current list was built for C&F and is stale
- [x] Funding statement present (2026-09-17)
- [x] `refs.bib` preprint-DOI audit (2026-09-17)
- [ ] Corresponding author signed in as **st_a.jabbary@urmia.ac.ir**, not the autofilled Gmail
- [ ] Subscription route selected at the licensing step
- [ ] arXiv replacement posted or scheduled. **Use `docs/paper/submission/arxiv_v5/`** (built
      2026-09-18 from current sources, 26 pp, builds clean). `arxiv_v4/` is stale and must NOT
      be uploaded. v5 changes the TITLE and ABSTRACT — paste both; see
      `arxiv_v5_metadata.txt`, and untick arXiv's pre-ticked "delete .bbl" box.

## Fallbacks, in order

Engineering with Computers → Advances in Engineering Software → CPC. ASME JVVUQ is the closest
on subject matter but fails a gate; see `docs/paper/review/journal_shortlist.md` §3.
RESS is out (13,000-word cap). C&F is out on the conflict recorded above.
