# Journal of Computational Science — Submission Checklist

Target: **Journal of Computational Science** (Elsevier), ISSN 1877-7503. Article type:
**regular research article** (not Communication or Correspondence, which carry page caps).

Manuscript: **"The near-wall barrier to parameter interpolation on AirfRANS is a coordinate
artifact"**.

> **Verification legend.** **[V-JOCS]** — read at source from the JOCS guide for authors, in a
> real browser, on the date given (the raw guide was saved as `jocs_guide.txt` in the session
> scratchpad on 2026-09-17). **[GEN]** — Elsevier-wide policy. Nothing in this file is a
> hypothesis about JOCS any more.

## History — read before writing anything to the editor

| Date | Venue | Outcome |
|---|---|---|
| 2026-08-02 | CMAME (CMAME-D-26-03937) | Desk reject: "no new computational methodology within scope; suggest ML-oriented journal" |
| 2026-08-25 | JCP | Submitted |
| 2026-09-07 | JCP | Desk reject: "originality with respect to other published papers is too questionable". **Will not reconsider — do not appeal or email the editor.** |
| 2026-09-12 | Computers & Fluids | **Withdrawn by us before filing, on conflict.** Its ML special-issue editorial is authored by Ashton, Dwight and Cinnella — authors of benchmarks this paper audits. |
| — | **Journal of Computational Science** | Current target |

The paper was reframed and then split, not reformatted (`docs/paper/review/RESOLUTION.md`,
`jocs_rebuild.md`). The cover letter deliberately does **not** disclose the rejections; see the
notes at the foot of `cover_letter.md`. Nothing is under consideration anywhere else.

## Blocking — do these first, in this order

1. **Publish release `v1.1.0` so the availability statement is true.** The manuscript and
   the cover letter say the snapshot matching the paper is release `v1.1.0`, archived at
   Zenodo under the concept DOI 10.5281/zenodo.21277928. Until that release exists, the DOI
   resolves to **v1.0.5 (2026-08-20), which archives the old trust-layer paper and none of
   the headline scripts**, and `main` on GitHub is several hundred commits behind the branch
   that has them. A reviewer following either link today finds the wrong paper's code.
   Steps (needs the author's go-ahead; a minted DOI cannot be retracted):
   fast-forward `main` to `paper1/reframe-after-jcp`; tag `v1.1.0`; publish the GitHub
   release (the Zenodo webhook fires on `release` events; `.zenodo.json` already carries
   the new title, keywords and description); then confirm on Zenodo that the concept DOI
   resolves to v1.1.0.
2. **Fill the vitae.** `vitae.md` has `[[FILL]]` markers for each author's position and
   degree and the second author's research interests — facts not recorded anywhere in the
   repository. Fill them, then rebuild the bundle until it prints `READY`.
3. **Declaration of interest via Elsevier's tool.** [V-JOCS 2026-09-17]: *"The declarations
   tool should always be completed."* Select **"I have nothing to declare"**, save as
   .docx, upload at "attach files". Deliberately **not** generated here.

## The upload bundle

Build it with `python scripts/build_jocs_package.py` → `docs/paper/submission/jocs_upload/`.
The script builds the unmodified sources and the flat upload copy side by side. It checks
that the two match in every label, number and page, with a byte-identical bibliography.
It also checks that both builds have zero errors, warnings, overfull/underfull boxes and
undefined references. Then it scans every upload text for private strings. It prints
`READY` only when nothing needs the author.

| Editorial Manager item | File in `jocs_upload/` |
|---|---|
| Manuscript (LaTeX source) | `neuroforge_cfd_elsevier.tex` |
| LaTeX source files | `preamble.tex`, `abstract.tex`, `body.tex`, `refs.bib`, `neuroforge_cfd_elsevier.bbl`, `Figure_1.pdf` (or `LaTeX_sources.zip` if a zip item is offered) |
| Figure | `Figure_1.pdf` (vector PDF, fonts embedded, no Type 3, no title on the artwork) |
| Highlights | `Highlights.docx` |
| Cover letter | `Cover_letter.pdf` (or paste `Cover_letter.txt`) |
| Author biographies | `Author_biographies.docx` — **after** filling `vitae.md` |
| Declaration of interest | from Elsevier's tool (see Blocking 3) |
| Reference PDF | `neuroforge_cfd_elsevier.pdf` — Editorial Manager builds its own from the sources |

**Never upload `cover_letter.md`**: its "Notes to self" discuss the desk rejections. The
packager builds the letter from the text above the `---` rule only, and refuses to build if
that heading moves.

**Data statement** (free text at submission, [V-JOCS 2026-09-17]; research-data Option C).
Paste: *"The code, result files, pre-registered decision rules and per-claim reproduction
map are archived at Zenodo as release v1.1.0 (https://doi.org/10.5281/zenodo.21277928) and
cited in the article. The AirfRANS dataset used is public (Bonnet et al., NeurIPS 2022)."*

## Requirements, and where the manuscript stands

| Item | Rule | Verified | Ours |
|---|---|---|---|
| Length | **None for regular articles**; caps only for Communications (6 pp) and Correspondence (4 pp) | [V-JOCS 2026-09-12] | 45 pp elsarticle preprint; 17,936 words incl. tables and captions (`check_submission.py`) |
| Abstract | ≤ 250 words, no citations | [V-JOCS 2026-09-12] | **245**, no citations |
| Keywords | 1–7, avoid multi-word "and"/"of" | [V-JOCS 2026-09-12] | 7 |
| Highlights | 3–5 bullets, ≤ 85 chars incl. spaces, separate editable file named "highlights" | [V-JOCS 2026-09-17] | 5, at 73–83 chars, `Highlights.docx` |
| Title page | title, authors, affiliations with country, corresponding author's email | [V-JOCS 2026-09-17] | ✓ (street addresses optional; not invented) |
| References | numbered in square brackets in order of first citation; journal names abbreviated per **LTWA**; DOIs where available | [V-JOCS 2026-09-17] | ✓ 35 refs in first-citation order (checked from `.aux` vs `.bbl`); 13 journal titles abbreviated to ISO 4 on 2026-09-23; 0 BibTeX warnings |
| Research data | Option C: deposit, **cite and link** in the article | [V-JOCS 2026-09-17] | ✓ Zenodo archive cited as a `[software]` reference and linked in "Code and data availability" (true once Blocking 1 is done) |
| Figures | separate files named Figure_1…; caption title not on the artwork; vector with embedded fonts | [V-JOCS 2026-09-17] | ✓ one figure, placed on p. 26 beside its table (was drifting to p. 45 until 2026-09-23) |
| Tables | editable, cited, numbered in order, **no vertical rules** | [V-JOCS 2026-09-17] | ✓ booktabs, all cited |
| Appendices | A, B…; Table A.1, Eq. (A.1) | [V-JOCS 2026-09-17] | ✓ the appendix table is **Table A.1** (printed "A.7" until 2026-09-23) |
| CRediT | required | [V-JOCS] | present |
| Generative-AI declaration | titled section before the references | [V-JOCS 2026-09-17] | present |
| Funding | state it, or the "no specific grant" sentence | [V-JOCS 2026-09-17] | present |
| Vitae | ≤ 100 words per author, editable format | [V-JOCS 2026-09-17] | drafted; **markers to fill** (Blocking 2) |
| Declaration of interest | Elsevier tool, .doc/.docx | [V-JOCS 2026-09-17] | **author step** (Blocking 3) |
| Peer review | single anonymized; editor screen, then ≥ 2 reviewers | [V-JOCS 2026-09-17] | no anonymisation needed; preprint and links stay |
| Preprint | *"will not count as prior publication"* | [V-JOCS 2026-09-17] | arXiv:2607.10333 disclosed in the letter |
| Graphical abstract | encouraged, not required | [V-JOCS 2026-09-17] | none |

## Publishing route — the hard constraint

**Take the SUBSCRIPTION route.** It carries no publication fee and has no effect on review.
Do not let an acceptance flow flip the licence to open access at the licensing step; that is
where the fee is triggered. See `no-apc-venues-only`.

## Before you click submit

- [ ] **Blocking 1–3 above are done**, and `build_jocs_package.py` prints `READY`
- [x] `python scripts/check_submission.py` — all mechanical requirements satisfied (2026-09-23)
- [x] `python scripts/audit_paper_numbers.py` — every checked number matches, 0 SKIP (2026-09-23)
- [x] Both PDFs rebuilt from current source: 0 errors, warnings, over/underfull, undefined (2026-09-23)
- [x] `pytest` — fast suite passes, exit 0 (2026-09-23)
- [x] Editorial board screened against the suggested reviewers (2026-09-17; EiC is Valeria Krzhizhanovskaya)
- [ ] Signed in as **st_a.jabbary@urmia.ac.ir** (EM account `AJabbary-884`), **not** the
      Chrome-autofilled `light.knight32@gmail.com`
- [ ] Article type: regular research article
- [ ] Authors entered in the manuscript's order; Kasra Ghanavati will receive an EM email to
      confirm co-authorship. The letter says both authors approved — make that true first.
- [ ] Suggested reviewers entered from `suggested_reviewers.md`, with emails taken from each
      person's institutional page at entry time
- [ ] Data statement pasted (text above)
- [ ] Subscription route selected at the licensing step
- [ ] arXiv replacement: upload `docs/paper/submission/arxiv_v5/` (28 pp, rebuilt 2026-09-23
      with comments stripped and labels checked against the in-place build). It changes the
      title and abstract; paste both from `arxiv_v5_metadata.txt` and untick arXiv's
      pre-ticked "delete .bbl" box. `arxiv_v4/` is stale and must **not** be uploaded.

## Fallbacks, in order

Engineering with Computers → Advances in Engineering Software → CPC. ASME JVVUQ is the closest
on subject matter but fails a gate; see `docs/paper/review/journal_shortlist.md` §3.
RESS is out (13,000-word cap). C&F is out on the conflict recorded above.
