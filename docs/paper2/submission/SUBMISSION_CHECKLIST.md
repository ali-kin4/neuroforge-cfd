# Submission checklist — *Computers & Fluids* (Paper 2)

Venue decided 2026-08-31 (`docs/PLANS.md` §0). Elsevier, hybrid; take the
**subscription** route, which is free — the ~$3,860 APC is the optional
open-access route only, and `no-apc-venues-only` binds.

Status key: ✅ done · ⏳ in progress · ❌ not started · ⚠ needs the author

---

## A. The manuscript

| requirement | status | where |
|---|---|---|
| `elsarticle` class, single-column preprint layout | ✅ | `scripts/build_paper2_pdf.py` → `docs/paper2/paper2.pdf` |
| Abstract ≤ 250 words | ⏳ | rewritten for the placement thesis; **re-count after the rewrite** |
| Highlights: 3–5 bullets, ≤ 85 characters each, separate file | ⏳ | `docs/paper2/highlights.txt`; the builder **fails** if any exceeds 85 |
| Keywords, 1–7, no multi-word phrases where avoidable | ✅ | six |
| Graphical abstract | ❌ | optional at C&F. **Decide: produce or drop.** Phase G's note claimed the audit was fully closed; it was not, and the note is corrected |
| CRediT authorship statement | ✅ | both authors |
| Declaration of competing interest | ✅ | |
| Data availability statement | ⏳ | now names the `paper2-v1` tag; **DOI still to insert** (§C) |
| Numbered Elsevier-style reference list | ⏳ | 11 → ~28 after the literature sweep; every arXiv id verified against the arXiv API 2026-08-31 (`docs/paper2/literature.md` §6) |
| Figure captions numbered, self-contained, in order of first appearance | ⏳ | 2 figures now, 4 planned |
| No internal working matter (`PLANS.md`, `[[B2]]`, `results/*.json` in prose) | ✅ | moved to the appendix / this repo |
| Line numbers + double spacing for review | ❌ | Elsevier's `review` option — add at build time |

## B. Files to upload

| item | status |
|---|---|
| Manuscript PDF | ⏳ |
| Highlights (separate file) | ✅ `docs/paper2/highlights.txt` |
| Cover letter | ⏳ `cover_letter.md` |
| Suggested reviewers (≥ 3, arms-length) | ⏳ `suggested_reviewers.md` — **⚠ affiliations, emails and editorial-board status must be verified before entry, not taken from this file** |
| Declaration of interest form | ⚠ author |
| Source files (`.tex` + figures) if requested at revision | ✅ `docs/paper2/_build/` |

## C. The blocker — data availability

The statement names `https://github.com/ali-kin4/neuroforge-cfd`. **Everything
this paper rests on lives on the branch `paper2/openfoam-warm-start`, which is
~45 commits ahead of `main` and deliberately unmerged.** A reviewer or editor
following that URL lands on `main`, where `solver/openfoam.py`, `cgrid.py`,
`warmstart.py`, `scoring.py`, `placement.py` and every `results/*.json` this
paper cites **do not exist**. Verified 2026-08-31 with `git ls-tree origin/main`.

Do **not** fix this by merging to `main`: Paper 1 is under review at JCP and its
own submission describes the state of `main`.

Fix, in order:

1. Tag the final paper-2 commit — `git tag -a paper2-v1 -m ...` and push the tag.
   A tag URL is stable and does not disturb `main`.
2. ⚠ **Author step:** archive that tag on Zenodo (GitHub → Zenodo integration)
   and put the resulting **DOI** in the data-availability statement, as Paper 1
   did (v1.0.4). A DOI is what an Elsevier editor expects; a branch URL is
   second best.
3. Update the statement to name the tag/DOI explicitly, not the bare repo root.

## D. Before clicking submit

- [ ] Abstract word count re-verified after the final edit
- [ ] Every number in the paper traced to a committed `results/*.json`
- [ ] Every figure regenerated from the declared arm set in one command
- [ ] Test suite green (`PYTHONPATH=src python -m pytest -q`)
- [ ] Adversarial review pass run and its findings closed or conceded in writing
- [ ] Publishing option set to **Subscription** — the order summary must read
      "nothing to pay". This is an initial choice and can be silently flipped by
      an acceptance flow; check it again at acceptance.
- [ ] arXiv preprint updated to match the submitted manuscript
