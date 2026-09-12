# The JOCS rebuild — what was cut, what was added, and what is unverified

**Date: 2026-09-12.** Implements the decision in `docs/paper/review/journal_shortlist.md`
§8: retarget Paper 1 at the **Journal of Computational Science** (IF 4.0, hybrid with a
free subscription route, single anonymized, **12,000-word cap**), and split the
five-benchmark metadata null out into Paper 2 for NeurIPS Evaluations & Datasets 2027.

---

## 0. READ THIS FIRST — the session had no shell, so nothing is built or committed

> **This session's Bash tool was disabled.** I could not run `git`, `pdflatex`, `bibtex`,
> `scripts/audit_paper_numbers.py`, `scripts/check_submission.py`, or a word count. Every
> edit below is **written to disk and durable**; none of it is **verified by a build or a
> script**. Constraints 2, 3 and 4 of the task brief are therefore *not discharged by this
> session*, and I am reporting them as UNRUN rather than asserting a clean build.
>
> The compensating discipline is the one `journal_shortlist.md` used: every change landed
> as its own small edit, so a power cut loses at most one edit rather than the pass.

### 0.1 What the parent agent must run, in order

```
cd docs/paper
pdflatex neuroforge_cfd_elsevier && bibtex neuroforge_cfd_elsevier && pdflatex neuroforge_cfd_elsevier && pdflatex neuroforge_cfd_elsevier
pdflatex neuroforge_cfd            && bibtex neuroforge_cfd            && pdflatex neuroforge_cfd            && pdflatex neuroforge_cfd
cd ../..
./.venv/Scripts/python.exe scripts/audit_paper_numbers.py --verbose
./.venv/Scripts/python.exe scripts/check_submission.py
```

**`bibtex` is required this round.** Citations were removed (§4.4), so the `.bbl` is stale
until bibtex reruns; without it the bibliography will still print entries nothing cites.

### 0.2 What the parent agent must commit, explicitly by path

`git status` shows an untracked `psh_ls.err` at the repo root. **Do not `git add -A`.**

```
git add docs/paper/body.tex docs/paper/abstract.tex docs/paper/preamble.tex
git add docs/paper/neuroforge_cfd.tex docs/paper/neuroforge_cfd_elsevier.tex
git add docs/paper/sections/trust_layer_removed.tex
git add docs/paper/sections/covariate_null_removed.tex
git add docs/paper/submission/highlights.txt
git add scripts/audit_paper_numbers.py scripts/check_submission.py
git add docs/paper/review/jocs_rebuild.md
git add docs/paper/review/point_space_headtohead.md
git add results/interpolation/point_space_oracle_ls.json
```

`point_space_headtohead.md` was already modified in the working tree before this session and
is modified again by it (§4.3): it now carries a recorded correction block in §4(c).

**`results/interpolation/point_space_oracle_ls.json` is currently untracked and the auditor
now reads it.** If it is not committed, two rows SKIP, and a SKIP suppresses the
"every checked number matches its source file" line exactly as a MISMATCH does. This is a
precondition, not a nicety.

---

## 1. The new title

**Chosen:**

> **What a neural flow surrogate buys is near-wall representation, and the scoring measure
> decides the ranking**

Applied to `neuroforge_cfd.tex`, `neuroforge_cfd_elsevier.tex` and the `pdftitle` in
`preamble.tex`. It satisfies both desk-screen constraints from `journal_shortlist.md` §7.1:
"airfoil" does not appear at all, "benchmark" is not the first noun, and the first clause is
a **positive measured result** rather than a complaint about reporting practice. The dataset
name is held back to the abstract's second sentence, where it reads as the instance rather
than the subject.

**Two alternates considered, and what each trades away:**

| Candidate | Trade |
|---|---|
| *Measure dependence and representation limits in the field-error scoring of machine-learning flow surrogates* (`journal_shortlist.md` §8's own proposal) | Maximally neutral and unambiguously in-scope for a generalist EiC, but it buries finding 1 — the positive result and the strongest measurement in the paper — and reads as a methods-critique paper, which §7.1 names as the bounce mode |
| *Near-wall representation, measure dependence, and a representation ceiling in the field-error scoring of flow surrogates* | Most complete; three nouns before the first verb. Accurate, hard to read, and the weakest of the three at thirty seconds |

---

## 2. Word count, before and after

**I could not measure this — no shell.** Here is the honest accounting.

| | lines in `body.tex` | words |
|---|---:|---:|
| **Before** (commit `47dc59d`) | **2110** | **~14,700** — measured figure, `venue_taste.md` §0 |
| **After** (this rebuild) | **1418** | **~9,900, estimated; NOT measured** |

The estimate is a line-count proportion calibrated on the one measured figure the repository
has: 14,700 words / 2110 lines = **6.97 words per line**, applied to 1418 lines. That
calibration is imperfect in a known direction — the surviving text is proportionally denser
prose and proportionally less table scaffolding — so treat the result as a **band of roughly
9,500–11,500 words**. **The honest summary is that the cut is very likely sufficient and is
not yet proven sufficient**: the low end clears the 12,000 cap with room, the high end
clears it by 4%, and nothing here has been counted. Do not report "under 12,000" until
`check_submission.py` has printed it.

**So I made it measurable instead of guessing.** `scripts/check_submission.py` previously
counted only the abstract and printed *"no manuscript length or page limit is set by this
journal's guide"* — true of Computers & Fluids, false of JOCS. It now counts
`abstract.tex + body.tex` under the same `strip_tex` semantics and **fails** above 12,000
words. That is a new go/no-go gate and the parent must run it.

- `abstract.tex`: 246 words before → **~221 after** (hand count; cap 250).
- No evidence was cut to reach the length. The trust layer and the covariate null were cut
  for *publication* reasons (§3, §4), and the length came out of that for free.

---

## 3. What was cut, and where it is preserved

### 3.1 The trust layer → `docs/paper/sections/trust_layer_removed.tex`

Instruction: `r2_round2.md` §2.6. Twelve verbatim blocks, each with its original `body.tex`
line range, following the header convention of the existing `residual_audit_removed.tex`.

| Removed | Labels that died with it |
|---|---|
| Contributions 5 and 6 of six | — |
| Related Work, *"Residual-based, data-free error signals, and conformal UQ"* (replaced by a trimmed 8-line version covering only the error-indicator line `sec:residual` needs) | — |
| Method, *"Trust map, conformal calibration, and the correction operators"* (73 lines) | `sec:correction`, `prop:coverage`, `prop:contraction` |
| `sec:v2`, the trust/corrector half | `sec:v2`, `tab:v2`, `tab:uq` |
| `sec:indist-ablation` — the whole weak grid backbone | `sec:indist-ablation`, `tab:indist`, `fig:fig1` |
| `sec:ood`, incl. the DeepCFD second dataset and the cylinder probe | `sec:ood`, `tab:ood`, `fig:fig2`, `fig:deepcfd` |
| `sec:conformal` | `sec:conformal`, `fig:fig6`, `fig:fig5` |
| `sec:selective` | `sec:selective`, `fig:risk_coverage` |
| `sec:cost` | `sec:cost`, `tab:cost` |
| *"Reading against the pre-registered hypotheses"* (H1–H5, all trust-layer) | — |
| Five trust-layer Limitations bullets | — |
| Appendix `fig:fig3` and `fig:cylinder` | `fig:fig3`, `fig:cylinder` |

**Two consequences worth flagging rather than burying.**

1. **The grid backbone leaves the paper entirely.** `tab:indist` was its only home, so its
   two rows in `tab:interp`, its row in `tab:interp_std` and `fig:fig3` went with it. The
   paper now has exactly two arms, which is what the head-to-head needs.
2. **The manuscript now contains zero figures** — all ten were trust-layer or appendix
   context. It carries 7 tables. That is defensible for a measurement paper and it is what
   the evidence supports, but **the parent should decide consciously** whether to commission
   one figure of the two band ladders side by side (grid vs native). I did not invent one.

### 3.2 The covariate null and `reynolds` split → `docs/paper/sections/covariate_null_removed.tex`

This is Paper 2's spine. Eight verbatim blocks: the two Introduction paragraphs,
contributions 2 and 3, `sec:splits` in full, `sec:covariate` in full including
`tab:covariate`, the covariate half of the drag-integrator paragraph, two Limitations
bullets, and the Conclusion sentences.

**The file records the reason in its own header: NeurIPS does not accept work already
published in a journal, so shipping any of this inside the JOCS submission closes ED 2027
to it.** That is a publication decision, not a length decision.

**One open item from `journal_shortlist.md` §8 is now closed, deliberately.** That document
flagged the `reynolds`-split finding as genuinely ambiguous — AirfRANS-specific (belongs
with the measurement paper) but also a benchmark-protocol defect (strengthens NullBench).
**Decision recorded in the preservation file's header: it goes to Paper 2.** It is a claim
about what a published split tests, which is NullBench's subject; the JOCS paper is about
measure dependence and representation limits, full stop.

### 3.3 What was compressed rather than cut

`sec:v2` is gone as a section, but its two force-integrator paragraphs survive as a short
new subsection **`sec:forces`** — *"What the force columns measure, and what the raster
costs them"*. They stayed because they are a **representation** result on a second quantity
through a different code path (the raster, not the quadrature, owns the drag loss; 68% of
drag is viscous and lives in a sublayer the grid cannot see), which is finding 3 measured
twice. The three sentences inside them that made the *covariate* argument moved to the
Paper 2 file.

---

## 4. What was added, and what it changed

### 4.1 `tab:native` and the native-resolution head-to-head — the new lead

`sec:interp` is restructured to lead with the measurement that has no raster in it:
200 `full` test cases, 35,849,332 nodes, both arms gated against their published rows
(G1 ≤ 1.6e-8 relative, G2 exactly 0.00e+00). Interpolation loses 144× / 134× / 4.85× /
18.3×, 24.7× standardised, 200/200 cases on three channels. Verdict
**NOT-COMPETITIVE-AT-NATIVE**, rule committed in `a54cb75`.

`tab:native` carries the two memo rows `point_space_headtohead.md` §6.3 mandates, which
answer *"you crippled the baseline"*: the grid protocol was costing the **interpolator**
4.1× on `u` and 69.9× on `p`.

### 4.2 The family bound — and a scope correction I am flagging

The oracle (`P3`) and the least-squares projection (amendment 2) are both new to the
manuscript. **The two have different sample sizes and the paper now says so inline:**

- **343.6×** — best *single* training field, chosen knowing the answer. **200 cases**, `u`,
  0–0.005c.
- **21.3×** — unconstrained least-squares projection onto the span of all 800 fields, the
  exact lower bound for *any* weighting of that family. **`meta.n_ls` = 3 cases**, `u`, and
  the one band whose node count (73–82k) far exceeds the 800 free parameters of the
  projection, which is the only band where the quantity is a bound rather than an
  interpolation.

The task brief described the 21.3× without a sample size. I verified it against
`results/interpolation/point_space_oracle_ls.json` and the rule in the script docstring, and
**wrote the n=3 scope into the sentence itself** rather than a footnote, because a reader
who takes 21.3× for a 200-case result has been misled. There is no write-up of this stage in
`point_space_headtohead.md` — that file has no section on amendment 2 — so this is the first
prose statement of the number, and the justification in the manuscript is the
**pre-registered** one from the script docstring (the published weights sum to one, so
redimensionalisation commutes with the combination and the estimator's prediction lies in
the span of the 800 redimensionalised single-field predictions; the least-squares projection
of the truth onto that span is therefore a genuine lower bound for any weighting, convex or
not), not a reconstruction.

**The abstract carries the same scope.** The first draft of the rewritten abstract stated
"an oracle ... allowed the best least-squares combination of all 800 training fields, is
still 21× worse", which let a three-case number sit unmarked beside four 200-case numbers in
the same sentence and merged the two different oracles into one clause. It now leads with
the **344×** figure, which *is* the 200-case number and is the larger one, and names the 21×
as "a three-case probe of the exact least-squares bound over the whole family". A standing
comment at the top of `abstract.tex` records both this and the monotonicity point below, so
a later compression pass cannot quietly undo either.

### 4.2b The abstract's band-separation sentence

Same class of error, caught at the same time. The first draft said the two methods' errors
were "separated **monotonically** across three orders of magnitude" — true of the grid ladder
(`D2` CONFIRMED, 0 inversions) and **false at the native nodes** (`P2` PARTIAL, 2 inversions,
the far-field `u` entry reversing), which is precisely the disagreement §4.4 documents at
length in the body. The abstract now claims the two measures **agree in size** and names the
one channel on which they differ in the far-field sign. The word "monotonically" does not
appear in the abstract.

### 4.3 An arm inconsistency in a committed review document — CORRECTED, ROUTE TO RIGOR-AUDITOR

> **`docs/paper/review/point_space_headtohead.md` §4's wall row is computed on a different
> interpolation arm from the rest of that document.**

`P1` selects the better in-body convention **per channel on the pooled value** and chose
`nearfill` on all four. §4's band table prints a wall row of **4052× on `u` and 2823× on
`v`** — those are the **`bridge`** arm (670.123 / 0.1653776 = 4052.1; 712.788 / 0.252526 =
2822.8). The `nearfill` arm the document's own §0 headline uses gives **4060× and 2834×**
(671.454 / 0.1653776; 715.598 / 0.252526). The `p` wall row it prints (1.50×, "149 200
against 99 227") *is* `nearfill`, so the document mixes arms within one table row set.

**The manuscript quotes the `nearfill` values** — 4060×, 2834×, 1.51×, 382× — because that
is the arm the paper declares it uses, and `P2`'s band ladder uses it too. The auditor
carries an explicit ARM DISCIPLINE comment and four `native wall row` claim rows so the
choice cannot silently drift back.

**Corrected in both places, not one.** `point_space_headtohead.md` §4(c) now carries a
`> Correction ... recorded rather than silently applied (2026-09-12)` block, in the same
pattern that document already uses for its correction to `venue_taste.md` §1.C: it names the
arm, shows both arithmetic derivations, states which value the manuscript quotes and why,
and records that **no pre-registered verdict changes** — the wall band is read by none of
`P1`, `P2`, `P3`, and both arms agree at the surface by three orders of magnitude. Leaving
4052 in a committed review file and 4060 in the manuscript would have been a fresh instance
of the "selective disclosure" pattern `r2_round2.md` §6 names as this project's recurring
failure, discoverable by any reviewer who reads the manifest the paper invites them to read.

### 4.4 The far-field `u` correction, reported as a disagreement rather than a fix

`point_space_headtohead.md` §4b is explicit that the grid claim *"Transolver is a factor of
three less accurate beyond 0.5c on u"* is not supported, **and** that "it was the raster" is
not supported either (the raster accounts for 1.27× of a ~10× swing). The manuscript
therefore:

- reports **both** pre-registered verdicts by name — `D2` **CONFIRMED** on the grid
  (906× → 0.31×, 7 bands, 0 inversions) and `P2` **PARTIAL** at the nodes (S = 45.6, 2
  inversions, interpolator behind in every band on `u`);
- names the two mechanisms that change together (which points inside a band are sampled;
  whether the reference is rasterised) and states that the decomposition separating them
  **has not been run**;
- carries a dedicated Limitations bullet for it.

This is the strongest available framing: the two measures disagree on exactly one entry, for
reasons that are themselves about measure and representation, which is the paper's thesis
applied to its own instrument.

### 4.5 Citations

**All nine lineage citations added yesterday are kept**, and the RSM paragraph *"The
estimator is decades old, and we say so first"* is untouched:
`queipo2005surrogate`, `poliak2018hypothesis`, `gururangan2018artifacts`, `errica2020fair`,
`dacrema2019progress`, `ahlmanneltze2025deep`, `feng2019misleading`, `ethayarajh2022vusable`,
`rasp2024weatherbench2`.

The `feng2019misleading` sentence gained a clause that is now load-bearing and was not
before: at native resolution **our cheap baseline fails**, so the paper's weight rests on
what its failure localises, not on the baseline beating anything.

**Citations no longer used** (entries remain in `refs.bib`; bibtex simply stops printing
them): `ribeiro2020deepcfd`, `bai2019deq`, `winston2020monotone`, `fung2022jfb`,
`lakshminarayanan2017ensembles`, `gal2016dropout`, `ma2024uqno`, `yu2026conformalpinn`,
`garg2025dfuq`, `jia2026multigranularity`, `roy2025anchor`, `song2026structureaware`,
`gopakumar2025pre`. Three UQ/error-indicator keys are **retained**:
`beckerrannacher2001dwr`, `hillebrecht2025posteriori`, `mukherjee2026certification`.

---

## 5. The new outline

| § | Content | Status |
|---|---|---|
| 1 Introduction | 5 paragraphs: near-wall representation + family bound; the weighting; the representation ceiling; the band separation (both verdicts); the residual. Then *The gap this fills* | **rewritten** |
| 1.1 Contributions | **4 items** (was 6) | **rewritten** |
| 2 Related Work | operators; solver-correctors; residual error indicators (trimmed); benchmarks; evaluation standards; simple-baseline lineage; RSM | trimmed |
| 2.1 Positioning | object is a discretisation; positive result is what survives | **rewritten** |
| 3 Method | geometry/channel contract + the monitored RANS operator **only** | trust half deleted |
| 4 Experiments preamble | the **three measures**, each named every time it is used | **rewritten** |
| 4.1 `sec:interp` | baseline → **`tab:native`** → oracle + LS family bound → mechanism → `tab:interp` (grid) → `tab:interp_std` → node-measure sensitivity + `tab:measure` → `tab:bandratio` + native ladder + the disagreement → `tab:interp_bands` → representation ceiling → resolution ladder → 5 adversarial controls → feature/train-size ladders → cost asymmetry | **restructured, lead replaced** |
| 4.2 `sec:forces` | what the force columns measure; the raster (not the quadrature) owns the drag loss | compressed from `sec:v2` |
| 4.3 `sec:residual` | floor at the truth; 200/200 descent; iterate selection; the operator boundary | unchanged but re-pointed |
| 5 Limitations | 6 bullets (was 11) | **rewritten** |
| 6 Conclusion | 4 paragraphs, leading with the positive result | **rewritten** |
| App. A | `tab:airfrans-sota` only | figures removed |

**Exhibits: 7 tables, 0 figures** (was 9 tables + 10 figures).
`tab:native` (new) · `tab:interp` · `tab:interp_std` · `tab:measure` · `tab:bandratio` ·
`tab:interp_bands` · `tab:airfrans-sota`.

---

## 6. Auditor rows added and removed

`scripts/audit_paper_numbers.py`. Every removal carries an in-file comment in the
established style saying what went and where the text is preserved; every removed row's
**reader is retained, unreferenced**, so the companion papers can reuse the key paths.

### 6.1 Added (34 rows) — all new readers over the point-space artifacts

New readers: `ps`, `ps_ls`, `ps_ratio`, `ps_std_ratio`, `ps_pooled`, `ps_memo`,
`ps_memo_cost`, `ps_band_u`, `ps_wall_ratio`, `ps_band_ratio`, `ps_wins`, `ps_oracle_Q`,
`ps_ls_ratio`, `ps_ls_ncases`, `ps_geom_mismatch`.

| Group | Rows |
|---|---|
| `tab:native` aggregate | R on u/v/p/nu_t; standardised 24.7×; the 6 pooled cells; the *unchosen* in-body convention on `p` (58 020) |
| paired sign test | 200/200/142/200 |
| `tab:native` memo rows | 6 cells + the two "what the grid cost the interpolator" ratios (4.1×, 69.9×) |
| P2 native ladder | u at 0–0.005c and >0.5c; S; **n_inversions = 2**; v and p inner bands; the two far-field wins (17.7×, 265×) |
| wall row | u/v/p/nu_t, on the P1-selected arm |
| P3 / P3-LS | 343.6× (200 cases); 21.3× (3 cases); **`n_ls` itself, as its own row** |
| mechanism | 7.8× geometry mismatch; 35.4% mass inside a training body; max outside-hull mass; node fraction inside 0.005c |

`ps_ls_ncases` is deliberately a claim row: if anyone re-runs the stage at a different
`n_ls`, the manuscript's inline scope note becomes wrong and the auditor says so.

### 6.2 Removed (30 rows)

| Rows | Section removed with them |
|---|---|
| audit cost 1.13 ms; deployed solve 3822 ms | `sec:cost` |
| C1 ×4, C2 ×3, C3 ×4, C4 (AUROCs, the gate) | `sec:selective` |
| conformal bound width min/max; conformal coverage | `sec:conformal` |
| case-name null ×6; two-parameter null; in-sample optimism ×2; entries below the lift/drag null; published entries carrying a lift number; drag intervals spanning zero | `sec:covariate` |
| reynolds containment / mse_u / mse_p / nd-vs-raw ×2; aoa C_l and C_d | `sec:splits` |

**Two rows deliberately KEPT** despite `sec:covariate` going: `interp rho_Cd vs official`
(0.8389) and `exact-truth-field rho_Cd vs official` (0.8394). `sec:forces` still quotes them
— as a *representation* statement (the interpolated field and the exact field are
indistinguishable to this integrator), which is not a covariate claim.

### 6.3 `scripts/check_submission.py`

- Venue strings: Computers & Fluids → Journal of Computational Science, with the withdrawal
  reason pointed at `journal_shortlist.md` §4.
- **New `check_manuscript_length`**: counts `abstract.tex + body.tex` under the existing
  `strip_tex` semantics and fails above 12,000 words. The docstring's claim that the venue
  sets no length limit is removed — it was true of C&F and is false of JOCS.
- `highlights.txt` rewritten: bullet 5 (the case-name force-rank null) replaced by the two
  native-resolution results. All five are ≤85 characters by hand count (79, 77, 83, 79, 79);
  the script is the authority.

---

## 7. Build and script status

| Check | Status |
|---|---|
| `pdflatex neuroforge_cfd_elsevier` ×2 | **UNRUN — no shell in this session** |
| `pdflatex neuroforge_cfd` ×2 | **UNRUN** |
| `bibtex` (both) | **UNRUN, and REQUIRED** — 13 citations were dropped |
| 0 overfull / underfull / undefined / errors / font-shape warnings | **UNVERIFIED** |
| `audit_paper_numbers.py` ends "every checked number matches its source file", no SKIP | **UNRUN** — and blocked until `point_space_oracle_ls.json` is committed |
| `check_submission.py` ends "all mechanical requirements satisfied" | **UNRUN** — now includes a 12,000-word gate that has never been executed |

**What I did do to make the build likely to pass on the first attempt:** every label defined
only in removed material was swept out of the surviving text. A grep for `sec:v2`,
`sec:covariate`, `sec:splits`, `sec:conformal`, `sec:selective`, `sec:cost`, `sec:ood`,
`sec:indist-ablation`, `sec:correction`, `tab:v2`, `tab:uq`, `tab:indist`, `tab:ood`,
`tab:cost`, `tab:covariate`, `fig:fig1`, `fig:fig2`, `fig:fig3`, `fig:fig5`, `fig:fig6`,
`fig:deepcfd`, `fig:risk_coverage`, `fig:cylinder`, `prop:coverage`, `prop:contraction`
returns **only the three header comment lines** of `body.tex`. All 63 surviving `\autoref`
targets resolve to a label that still exists. `\begin`/`\end` pairs balance at 46
occurrences (7 table, 7 tabular, 6 adjustbox, 2 itemize, 2 enumerate, 2 align).

**The two residual build risks I cannot exclude:**

1. `preamble.tex` still declares the `theorem`/`proposition` environments, now unused by
   either build. Declaring an unused amsthm environment is a no-op, and both removal files
   need them, so I kept them with a comment. If a `nag`-style check objects, delete the two
   `\newtheorem` lines.
2. The paper has no `\includegraphics` left, so `graphicx` is loaded and unused. Harmless.

---

## 8. Items routed to the rigor-auditor / left for a human

1. **`point_space_headtohead.md` §4 mixed interpolation arms in the wall row** (§4.3 above).
   The manuscript uses the arm `P1` selected, and the review document now carries a recorded
   correction block. **Nothing outstanding — listed so the rigor-auditor sees it.** The one
   judgement to confirm is that quoting `nearfill` throughout is the right call; the
   alternative (quote `bridge` at the wall, because it is marginally kinder to the baseline
   there) would make the band ladder and the aggregate two different arms, which is the
   defect being corrected.
2. **The 21.3× family bound is n=3.** Stated inline in the manuscript. If the parent wants
   it as a headline number at the 200-case scale, the `oracle_ls` stage must be re-run with
   a larger `--n-ls` (it cost 301 s for 3 cases, so 200 cases is ~5.6 h on the same path).
3. **Zero figures.** A conscious decision is needed; I did not fabricate one.
4. **The must-eyeball list in `journal_shortlist.md` §10 is untouched** — in particular the
   69-member JOCS editorial board, which needs a browser Ctrl-F for the eight names before
   filing.
5. **`docs/paper/submission/cover_letter.md` and `suggested_reviewers.md` still target
   Computers & Fluids.** Out of scope for this pass; they must be rewritten before filing,
   using the four-paragraph structure `journal_shortlist.md` §8 drafts.
6. **`docs/PLANS.md` and `docs/GOALS.md` are not updated.** The repository convention is to
   update both at the end of every session.
