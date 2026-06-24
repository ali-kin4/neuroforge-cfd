# JCP Submission Checklist — NeuroForge

Target: **Journal of Computational Physics** (Elsevier). Submission portal:
Editorial Manager (https://www.editorialmanager.com/jcomp/).

JCP uses **"Your Paper Your Way"** — the *initial* submission may be a single PDF; you are only
asked to reformat to the Elsevier template (`elsarticle`) at the revision/acceptance stage. So
the current `docs/paper/neuroforge_cfd.pdf` (24 pp) is acceptable as-is for first submission.

## Ready now (drafted, in `docs/paper/submission/` + paper back-matter)
- [x] Manuscript PDF — `docs/paper/neuroforge_cfd.pdf` (builds clean, 24 pp)
- [x] Cover letter — `cover_letter.md`
- [x] Highlights — `highlights.txt` (5 bullets, each ≤85 chars)
- [x] Data & code availability statement — in paper back-matter + repo (REPRODUCE.md, MANIFEST.json)
- [x] CRediT author statement — in paper back-matter (single author)
- [x] Declaration of competing interests — in paper back-matter (none)
- [x] Reproducibility artifacts — committed code, REPRODUCE.md, manifest with hashes

## You must do (need your accounts / decisions — I can't)
- [ ] **Create an Elsevier / Editorial Manager account** and start the JCP submission.
- [ ] **Archive the repo on Zenodo to mint a DOI**, then drop the DOI into the back-matter
      "Code and data availability" (currently "DOI to be assigned on acceptance"). Zenodo +
      GitHub release integration does this in a few clicks; needs your GitHub/Zenodo auth.
- [ ] **Suggested reviewers** (JCP asks for 3–5): pick from the neural-operator / ML-for-CFD
      community — e.g. authors of Transolver, AirfRANS, GINO, conformal-for-operators — avoiding
      anyone with a conflict. (I can draft a list with rationales if you want.)
- [ ] **Confirm author metadata** (affiliation, ORCID if you have one).
- [ ] Decide on optional **graphical abstract** (not required by JCP).
- [ ] Final read-through of the PDF (it's in your viewer).

## At revision stage (only if accepted-with-revisions)
- [ ] Reformat to `elsarticle` (LaTeX source is ready: `neuroforge_cfd.tex` + `refs.bib`).
- [ ] Provide editable source files (.tex, figures) — already in repo.

## Fallback
If JCP declines, the same package submits to **CMAME** (similar requirements) or **TMLR**
(the guaranteed-quality floor; the paper already uses the TMLR template).
