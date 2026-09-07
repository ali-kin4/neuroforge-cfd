# Tone pass: apologetic framing out, honest scoping in

Date: 2026-09-07. Files touched: `docs/paper/abstract.tex`, `docs/paper/body.tex`,
`docs/paper/sections/residual_floor_theorem.tex`.

## Build status: UNVERIFIED

Bash was disabled in the session that made these edits. **None of the following were run
and all must be run before anything is committed:**

- `pdflatex` (twice) on `neuroforge_cfd_elsevier.tex` and on `neuroforge_cfd.tex`
- `scripts/audit_paper_numbers.py`
- `scripts/check_submission.py`

Risk controls applied in lieu of a build: no new macros, no changed maths, no deleted
`\label`s or `\autoref` targets, no altered float or list environments, no changed
`\citep`/`\citet` keys, and **no numeral altered**. Every edit is prose inside an
existing paragraph or a `\section`/`\subsection`/`\paragraph` title.

Three numerals were *introduced*, all reused verbatim from elsewhere in the manuscript:
`1.13`\,ms, `3822`\,ms and `≈91%` now appear in the Conclusion, taken from `tab:cost` and
`sec:conformal` (1.13/3822 = 0.03%, consistent with `tab:cost`). **The Conclusion is now a
consumer of `tab:cost`** — worth knowing if that table is ever re-measured. No existing
value was edited, so `audit_paper_numbers.py` cannot drift from this pass.

The two check scripts
are structurally insensitive to this pass: `audit_paper_numbers.py` compares hard-coded
Python constants against committed JSON and never parses the `.tex`; `check_submission.py`
reads only the abstract word count, the keyword block and `submission/highlights.txt`. The
abstract edit is net **two words shorter**, so the 250-word cap has more headroom than
before, not less.

## Page count

**Before: 74 pages** (as reported in the brief). **After: not measurable from this
session** — see build status. Prose reduction is roughly **900 words net** (~1,050 removed,
~150 added), i.e. about 6% of the ~14,700-word manuscript.

That is well short of the 74 → 55 target, and I want to be direct about why rather than
report a number I did not measure. The target implies removing ~3,700 words. Two things
stopped me:

1. **The remaining duplication is in the places where papers are supposed to repeat
   themselves** — abstract, contribution list, positioning table, the hypothesis ledger,
   conclusion. Consolidating those would not shorten the paper so much as gut its
   navigation.
2. **The page count is float-dominated, not prose-dominated.** The manuscript carries 9
   tables, 8 figures and a full-page TikZ pipeline schematic in a single-column 12pt
   elsarticle layout. A 25% prose cut removes perhaps 7–8 pages of the 19 wanted; the rest
   would have to come from tables and figures, which the brief forbids and which I agree
   should not move.

If 55 pages is a hard requirement, the lever is float policy (merging `tab:transolver`
into `tab:v2`, moving `tab:airfrans-sota` and the two OOD/reliability figures to an
appendix), not prose. That is a separate decision and I have not taken it.

## 1. Self-retraction narration removed (20 passages)

The brief counted 16; there were 20. All are gone except one, replaced by a neutral
version note.

### `body.tex`

| Was | Now |
|---|---|
| "Earlier drafts of this work inferred the failure of the residual as an objective from a theorem plus an iteration sweep… as a reviewer would rightly have said" (intro) | Deleted. The paragraph now asserts what the evidence is and what form the claim takes. |
| "We arrived here the hard way, and by discarding earlier claims of our own" | Deleted; replaced by "The evidence is a full … pipeline with pre-registered hypotheses". |
| "Prior drafts of this work inferred the negative from a theorem and an iteration sweep that did not test it. We now test it directly" (contribution 3) | "We test this directly, by minimising the residual with the differentiable operator and no network in the loop." |
| "retain the retraction of stale single-seed numbers from earlier drafts" | Deleted from the measured-vs-assumed sentence. |
| "including the ones that falsified claims we had previously made" ("How to read") | Deleted with the roadmap compression. |
| "That $0.84$ is not a ceiling, **and we withdraw an earlier reading of it as one**" (title) | "That $0.84$ is not a ceiling: a field-free covariate baseline beats it." |
| **"Retraction (retained from prior drafts)"** paragraph | Replaced by the single retained version note (below). Corrected numbers kept verbatim. |
| "This subsection reports an earlier and weaker observation… We keep it because it is what first prompted the question, not because it settles it." | "This subsection reports a weaker, complementary observation … and the four limits that keep it from carrying the claim alone." |
| "not the channel earlier drafts emphasised" | Deleted. |
| "Quoting the $0\to3$ endpoints alone, **as earlier drafts of this paper did**, overstates the effect" | "The $0\to3$ endpoints alone therefore overstate the effect." |
| "Earlier drafts of this paper argued from the acceptance test's definition that it must accept almost nothing… **the structural argument was wrong**" | "The acceptance test is satisfiable by the identity, so its definition suggests it should admit almost nothing. Measured, it does the opposite." |
| "But the accuracy is damping, not the residual test—**and we concede it**" (title) | "The accuracy is damping, not the residual test." |
| "rather than where **earlier drafts of this work drew it**, at steady-versus-transient…" | "The obvious alternatives—steady versus transient, resolved versus under-resolved—are falsified by Lei et al. being steady and successful." |
| "several of the **concessions** we make above" | "the **controls** we report above". |
| "**An earlier draft of this paper attributed** 'AUROC ≈0.9' to the residual; that is the fused score's number and **the attribution was wrong**" | "The gain that survives is the fusion's: the AUROC 0.905 belongs to the fused score, not to the residual alone." Same disambiguation, no confession. |
| "the acceptance gate … **and we no longer claim it is vacuous**" (Limitations) | "The acceptance gate built on the residual is a different mechanism: measured, it admits…" |
| "the resolved-versus-under-resolved lines **earlier drafts of this work drew**" (Conclusion) | "…lines it might be confused with." |

### `sections/residual_floor_theorem.tex`

| Was | Now |
|---|---|
| "state the limitation rather than the closed form **we previously claimed** for every grid" | "This leg is resolution-contingent and we claim no more." |
| "Either way **the claim we withdraw** is the resolution-independent one." | Deleted (the preceding sentence already scopes it). |
| "Our contribution is threefold **and narrower than earlier drafts of this work claimed**" | "Our contribution is threefold." |
| "**We previously advertised** instead an upper bound $\|e_\infty\|\le\|r^\star\|/\sigma_{\min}$… **and we withdraw it**" | Deleted. The substantive reason not to use that bound survives intact in the ranking subsection ("we do not argue it from a two-sided bound: $\sigma_{\min}$ … is not separable from zero numerically"). |
| "…replacing **what was previously an existence claim with two examples**" | Deleted. |
| "**The previous version of this section** offered two examples of null modes. The situation is considerably worse than two examples" | "The undetectable subspace is not a matter of a few isolated null modes. It is large, and its size is provable by counting." |
| "we have **retired the earlier claim** that it followed from the $128^2$ grid under-resolving the boundary layer" | Deleted; the positive statement ("It follows from operator provenance…") carries it. |
| Header comment narrating the 2026-09-07 rewrite ("which existed nowhere in the paper", "contribution (a) no longer advertises…") | Replaced with a structural map of the section. Invisible in the PDF, but the `.tex` ships to arXiv. |

### The one retained apology

`body.tex` §5.3, retitled **"Version note for readers of the preprint"**: the single-seed
$\rho_{C_l}$ $0.924\to0.958$ and surface-MSE $-25\%$ figures appear in earlier public
versions of the preprint, so a silent contradiction between versions would be worse than a
neutral note. Phrased as a version note, not an apology; the word "retracted" is gone; the
corrected 3-seed numbers are unchanged. Wording deliberately says "earlier public versions"
rather than naming a version number, because I could not verify from this session which
arXiv version first carried them — **please check that before committing** and tighten to
"v1" if it is v1.

## 2. Section and paragraph retitles

| Old | New |
|---|---|
| Trust map and conformal calibration **(the surviving contribution)** | Trust map and conformal calibration |
| **Honest baseline: the grid backbone trails SOTA** | Matched-budget baseline |
| Descending the residual: **the experiment the negative claim requires** | Descending the residual directly |
| **Does the physics earn its place?** | Where the physics earns its place: drag |
| The **weak** grid backbone: corrector trades accuracy, lifts the trust signal | The grid backbone: the corrector trades accuracy for trust signal |
| Self-auditing, defined—**and what of it survives** | Self-auditing, defined—and what the physics earns |
| Two implementation facts that bear on later claims, **disclosed here** | Two implementation facts that bear on later claims |
| **A disclosure on** the MeshGraphNet arm | The MeshGraphNet arm is not a like-for-like third point |
| Corrector vs. ensemble **(cost, stated plainly)** | Corrector vs. ensemble: two points on a cost axis |
| **Two deflations of these numbers, which we would rather state than be asked** | Two limits on these force numbers |
| **Scoping the rest, honestly (protected negatives)** | Two negatives the OOD ablation preserves |
| Four limits of this sweep, **stated before its numbers** | Four limits of this sweep |
| But the accuracy is damping, not the residual test—**and we concede it** | The accuracy is damping, not the residual test |
| **The claim we make, and the one we do not** | What holds universally is iterate selection |
| On the **weak** backbone the corrector trades accuracy | On this backbone the corrector trades accuracy |
| **Scope, stated before a reviewer states it for us** (theorem) | Scope of legs (iii) and (iv) |
| **Honest scope**, with a controlled negative | Scope, with a controlled negative |
| What **remains deployable, with its accuracy claim corrected** | What is deployable: a certificate, not an accuracy mechanism |
| Figure 3 caption: "**Honest** matched-budget baseline" | "Matched-budget baseline" |
| Figure 4 caption: "The iteration sweep **that first prompted the question**" | "The iteration sweep" |

"weak" now appears in no section title. It survives in body prose where it is the finding
(the backbone-dependence result needs the backbone to be weak).

## 3. Repetition consolidated — one home each

The four facts the brief identified as appearing three-to-five times:

- **W1 residual-input null.** Home: §5.2 (`sec:v2`), where the ablation is described.
  Related Work now cross-references it in one clause instead of restating the design and
  the wider-stencil caveat. The Limitations entry that duplicated it wholesale was merged
  into the residual-as-objective entry, with every number and the stencil caveat retained.
- **The floor is not a resolution artifact.** Home: §5.7 (`sec:floor_ladder`) plus the
  theorem. Related Work's restatement of the certificate width and the "nor can refinement
  recover it" line was compressed to a cross-reference.
- **The ungated fixed half step.** Home: §5.6 (`sec:iters`). The Positioning paragraph's
  "retained because it is correct and reproducible" hedge was cut; the intro contribution
  and the conclusion each keep one sentence, which is the normal abstract/intro/conclusion
  recapitulation.
- **The physics-free tie.** Home: §5.11 (`sec:selective`). §5.11's closing paragraph, which
  re-derived the audit-cost argument and the whole `tab:uq` result from §5.2 and §5.12, is
  now two cross-referenced sentences — both distinct cost measurements (1.13 ms clean,
  1.65 ms contended) survive.

Not consolidated, to be clear: the **wider-differentiable-stencil caveat** on the W1 null
was *relocated*, not reduced. It appears in Method (where the two stencils are defined), in
§5.2 (attached to the null it qualifies) and in Limitations. That is deliberate — it is a
caveat on a headline attribution and should travel with it.

Two Limitations bullets merged into one ("The residual fails as an objective; the gate does
not"), and the iteration-sweep bullet tightened. No limitation was dropped and no number in
either bullet was lost.

## 4. Opening rebalanced

- **Abstract**: the closing turn "What survives is narrower" became "What the monitor does
  deliver is triage and a certificate". Same two facts, same numbers, same width caveat,
  stated as a result. Two words shorter.
- **Introduction, ¶3**: the freestream-versus-truth contrast now lands in the paragraph
  that sets up the three roles, rather than waiting for the contribution list.
- **"How to read this paper"**: 22 lines → 14, and the rhetorical questions ("Is the floor
  real, or is it our grid?", "Does it keep working?") are gone. Every cross-reference is
  preserved.
- **Conclusion, "What the audit delivers"** (was "What survives, stated at its real
  strength"): now leads with the working audit — 1.13 ms against a 3822 ms solve, and 91%
  of oracle triage recovered at a 10% rejection budget — before the scope qualifiers, which
  are unchanged. Those two numbers were previously absent from the conclusion despite being
  among the paper's strongest positive results.

## 5. Tempted to cut, but KEPT — the overcorrection check

Every item below reads as a concession and is not one. Each is load-bearing scope a
reviewer needs, and I left it intact.

1. **"We explicitly do not claim residual descent always harms"** (contribution 3, §5.8,
   conclusion). This is the boundary of the paper's central negative. Deleting it would be
   overclaiming.
2. **The whole `\paragraph{Four limits of this sweep}`** — all four, with the
   direction-only verdict, the $1.3\times$-vs-$2\times$ bar, the $45/120=38\%$ paired count
   and the $n=1$ dependency. Only the title's flinch went.
3. **"Four controls, because a rising floor is usually a bug."** Kept the title verbatim,
   per the coordinator: it justifies a methodological choice rather than pre-empting a
   reviewer.
4. **The MeshGraphNet density disclosure** — the $\textsc{density-driven}$ verdict, the
   $2.44$ MSE ratio, and the resulting restatement of the range as $0.40$–$0.61$ at matched
   density. This qualifies a headline number and stays in full, in both §5.2 and
   Limitations.
5. **"Two limits on these force numbers"** — the $\rho_L=0.998$ deflation (angle of attack
   is an input channel; Kutta–Joukowski reads the far field) and the statement that viscous
   drag is *unavailable* on this grid. This is the most self-damaging passage in the paper
   and it is completely correct.
6. **"OOD coverage is empirical, not a guarantee"** and the exchangeability failure. Kept
   the paragraph, the title and the bold "This is an empirical observation, not a
   guarantee."
7. **The physics-free tie itself** ($0.894$ vs $0.871$, CI $[-0.035,+0.083]$), everywhere it
   legitimately appears. Reframed once, never softened.
8. **"an ungated fixed half-step does better on accuracy alone (95.8%, median 6.2%)"** —
   the control that beats our own gate. Kept in the intro contribution, §5.6 and the
   conclusion.
9. **The `mse_v` attenuated-regression negative** and the `aoa` drag-ranking regression in
   the OOD section. Retitled, not trimmed.
10. **"We make no competitiveness claim for it"** about the grid backbone, and the
    $\sim4$–$60\times$ gap. The section title lost its flinch; the claim did not.
11. **The wider-differentiable-stencil caveat** on the W1 null ("a live alternative
    explanation we have not excluded"). Moved into the W1 paragraph itself so it travels
    with the result instead of being restated in three places.
12. **The near-wall masking disclosure** — that the monitor discards a mean 46% of the
    squared no-slip penalty, and that it is an unlogged 8-case spot check reported as such.
13. **Every "n=", seed count, per-seed sign count, population-std note, Cohen's d and
    bootstrap CI**, including the awkward ones: the $n=15$ conformal arm, the single
    ensemble training run, the $n=1$ unseeded sweep.
14. **"$\sigma_{\min}$ is not separable from zero numerically, and a bound resting on it
    would be vacuous."** The withdrawal of the old $\|r^\star\|/\sigma_{\min}$ bound went;
    the mathematical reason not to use such a bound stayed, because it is doing real work.
15. **"The durable contribution is a measurement and the boundary it draws, not a
    system."** This reads modest but it is the paper's actual intellectual position, not an
    apology.
16. **The pre-registration ledger's "had it been registered it would be falsified"** for
    the unregistered descent result. That is what pre-registration honesty looks like.

## 6. Deliberately left alone

- **`docs/paper/submission/arxiv_v4/stage/`** still contains "Retraction (retained from
  prior drafts)" and the other old phrasings. That directory is a snapshot of a submitted
  version and must not be rewritten. A `grep` run without gitignore filtering will hit it;
  that is expected, not a missed passage.
- **The "cannot" (31) and "fails" (16) counts from the brief.** Most instances are the
  paper's own assertions — "the floor cannot be refined away", "the residual cannot be
  minimised", "the ground truth fails the check". Shaving them would weaken accurate
  claims. Those counts were a symptom of the framing, and the framing is what this pass
  changed.

## 7. Flagged, not fixed

- **Version-note wording** — verify whether the single-seed figures are in arXiv v1
  specifically before committing (see §1).
- **The 55-page target is not met** and, in my judgement, is not reachable by prose alone
  (§Page count). Decide on float policy separately.
- **`\paragraph{The certificate is calibrated on the deployed, corrected field}`** in
  §5.10 is a single ~750-word paragraph carrying two independent verification arms (the
  $n=15$ dropout-FNO contrast and the production-split Transolver check). It is the
  densest passage left and a candidate for splitting into two paragraphs for readability —
  but every sentence in it carries a measurement, so I did not touch it.
