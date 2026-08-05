# JCP Submission Checklist — "NeuroForge: Self-Auditing Neural CFD Surrogates with Calibrated Physics-Residual Trust"

Target: **Journal of Computational Physics** (Elsevier). Submission portal:
Editorial Manager (https://www.editorialmanager.com/jcomp/).

History: desk-rejected at CMAME (CMAME-D-26-03937, 2026-08-02, "no new computational
methodology within scope; suggest ML-oriented journal") under the old engine-framed title.
Repositioned (title/abstract/claims) + hardened (selective prediction, multi-split
conformal, bootstrap CIs, repaired force integrator, 5-seed extension) before JCP.

JCP uses **"Your Paper Your Way"** — the *initial* submission may be a single PDF; you are
only asked to reformat to the Elsevier template (`elsarticle`) at the revision/acceptance
stage. The current `docs/paper/neuroforge_cfd.pdf` is acceptable as-is for first submission.

## Ready now (in `docs/paper/submission/` + paper back-matter)
- [x] Manuscript PDF — `docs/paper/neuroforge_cfd.pdf` (builds clean; new title)
- [x] Cover letter — `cover_letter.md` (retargeted to JCP, new title + new results)
- [x] Highlights — `highlights.txt` (5 bullets, each ≤85 chars, refreshed)
- [x] Data & code availability — in paper back-matter (Zenodo DOI 10.5281/zenodo.21277928)
- [x] CRediT author statement — in paper back-matter (two authors)
- [x] Declaration of competing interests — in paper back-matter (none)
- [x] Declarations of generative-AI use — in paper back-matter (Elsevier policy)
- [x] Reproducibility artifacts — committed code, REPRODUCE.md, manifest with hashes
- [x] Suggested reviewers — `suggested_reviewers.md` (re-check conflicts before entry)

## Author must do (needs your accounts / decisions)
- [ ] Start the JCP submission in Editorial Manager (jcomp); Ali's EM account: Ali_Kin4.
- [ ] Confirm author metadata (both authors, ORCIDs, affiliations; co-author email
      confirmation will go to Kasra).
- [ ] Enter 3–5 suggested reviewers from `suggested_reviewers.md`.
- [ ] Attach graphical abstract — `results/figures/graphical_abstract.png` (3900x1500 px
      @300dpi, meets Elsevier spec; "predict -> audit -> calibrate -> decide" strip built
      by `scripts/make_graphical_abstract.py` from committed numbers only; supersedes the
      old two-roles figure in the private `cmame_submission/` repo).
- [ ] Final read-through of the PDF.
- [ ] After submission: post arXiv v2 (new title/abstract) so the preprint matches.

## At revision stage (only if accepted-with-revisions)
- [ ] Reformat to `elsarticle` (LaTeX source ready: `neuroforge_cfd.tex` + `refs.bib`).
- [ ] Provide editable source files (.tex, figures) — already in repo.

## Fallback
If JCP desk-declines: **TMLR** within the week (the paper already uses the TMLR template;
internal committee scored it a solid TMLR accept). EAAI is the applications-framed
alternative if venue IF is preferred over venue identity.
