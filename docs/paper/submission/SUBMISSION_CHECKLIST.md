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
- [ ] Final read-through of the PDF.
- [ ] Post arXiv v2 (new title/abstract) **before or same-day as** the EM submit — not
      after. `cover_letter.md:57` points the editor at arXiv:2607.10333; that page still
      shows the *old* title, so an editor clicking it during desk-check lands on a
      differently-titled paper. This journal already desk-rejected the work once.

## At revision stage (only if accepted-with-revisions)
- [x] Reformat to `elsarticle` — **done ahead of submission**, see the format note above.
- [x] Provide editable source files (.tex, figures) — in repo, list above.

## Fallback

If JCP desk-declines, go to **Machine Learning: Science and Technology** (MLST, IOP
Publishing; ISSN 2632-2153) — *not* TMLR.

**Why MLST is the right second venue** (checked against the live journal page, 2026-08-20):
its remit is to bridge "the application of machine learning across the sciences with
advances in machine learning methods and theory as motivated by physical insights" — which
is what this paper is. It runs focus collections on *Physics-Informed Machine Learning*,
*AI and Partial Differential Equations*, and *Explainable Machine Learning in Sciences*,
and publishes PDE-surrogate/emulator work. Crucially it does **not** require a novel
numerical scheme — the exact bar CMAME desk-rejected us on, and the residual risk at JCP.
CMAME's own rejection letter advised an "ML-oriented journal"; this is that, with a journal
identity and metrics rather than a community-run venue. Impact factor 4.2, CiteScore 6.7.
Median submission→first decision **5 days** before review, 52 days after — a fast desk
verdict, which matters for a fallback.

**The one thing to settle before submitting there:** MLST is fully open access, so an APC
applies. IOP is running an expanded APC discount/waiver programme for low- and middle-income
countries in 2026. It is applied automatically only when *all* authors are in an eligible
country — which fails here (Ali/Urmia + Kasra/Greenwich) — so use the documented route:
**request a discretionary waiver or discount by email to the journal office at the point of
submission.** IOP members also get 25% off. Settle this before formatting work, not after.

**If the APC cannot be waived**, drop to an Elsevier *hybrid* journal (no mandatory fee,
and the EM account already exists): **Engineering Applications of Artificial Intelligence**
(applications-framed, high IF) or **Computers & Fluids** (CFD-side, lower IF, would scrutinise
the uniform 128^2 grid and near-wall blindness harder). Note: I could not verify either
journal's current guide for authors — ScienceDirect served a CAPTCHA — so confirm scope and
format yourself before committing.
