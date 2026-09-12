# Response to Reviewer 2, round 2 (`docs/paper/review/r2_round2.md`)

Date: 2026-09-12. Branch: `paper1/reframe-after-jcp`.
Scope: the four findings the brief judged fixable from material already in the repository
(review §2.2, §2.3 labelling, §2.5, §2.4). The fatal §1 measure-asymmetry objection is
**not** addressed here — it needs a run, and another agent owns that measurement
(`docs/paper/review/measure_asymmetry.md`).

> **Read §5 first if you are about to merge.** Bash was unavailable in the session that
> made these edits, so **no build was run, no BibTeX was run, neither audit script was
> run, and nothing was committed.** Every edit is on disk. The gates are listed with the
> exact commands in §5 and must be run before this lands.

---

## 1. The missing prior-art inoculation (review §2.2, question 6)

### What changed

**`docs/paper/refs.bib`** — nine entries added (42 → 51), under a comment block naming
their provenance. Metadata table in §4 below.

**`docs/paper/body.tex`, Related Work** — two new paragraphs after
*"Evaluation and reporting standards in ML for PDEs"*, before `\subsection{Positioning}`:

* **"Simple-baseline audits: the lineage both of our reference points belong to."** Opens
  *"Neither control is a new kind of instrument."* Walks the lineage — partial-input
  baselines in NLI \citep{poliak2018hypothesis,gururangan2018artifacts}, structure-agnostic
  baselines in graph classification \citep{errica2020fair} (with the explicit line
  *"substitute geometry for structure and that is the sentence we are testing here"*),
  the recommender-systems audit \citep{dacrema2019progress}, and the Nature Methods genre
  proof point \citep{ahlmanneltze2025deep}. It then states \citet{feng2019misleading}'s
  caveat **and honours its direction**: a cheap baseline that succeeds shows a benchmark is
  cheatable; a cheap baseline that fails licenses nothing. It closes with the one
  structural difference (generative code, not a subset of the model's input) and flags
  AirfRANS as the hybrid case it is — $U$ and $Re$ *are* input channels, the NACA digits
  are not.
* **"The estimator is decades old, and we say so first."** States in the first sentence
  that Gaussian kernel ridge over design variables is Kriging in all but name and that
  response surfaces over design variables are long-standing aerodynamic practice
  \citep{queipo2005surrogate}; disclaims novelty for the estimator; claims it for the
  measurement. Carries the positioning line the brief asked for verbatim in substance:
  *"McGreivy and Hakim showed that the field compares against weak **numerical** solvers;
  we show that it does not compare against a trivial **statistical** baseline at all, and
  we measure how much that conceals."* Ends with WeatherBench 2 as the **positive
  precedent** \citep{rasp2024weatherbench2} — weather forecasting already mandates
  persistence and climatology in its headline scorecard, external-aerodynamics
  benchmarking does not — which converts the recommendation from a complaint into an
  import.

**`docs/paper/body.tex`, `sec:interp`, "The baseline."** — one sentence, so an
aerodynamics reader meets the concession before the result:
*"The estimator is deliberately old---this is response-surface methodology, and Gaussian
kernel ridge over design variables is Kriging in all but name \citep{queipo2005surrogate}
---so the claim we make is for the measurement, not for the method."*

The strings **"response surface"** and **"Kriging"** now appear in `body.tex` (they did
not before).

### Two deliberate departures from `naming_and_positioning.md` §6

1. **The V-usable-information identity is *not* asserted.** §6's drop-in paragraph leans on
   $I_\mathcal{V} = -\tfrac12\log(1-R^2)$. That identity is about $R^2$; **this** paper's
   null reports a **Spearman rank correlation**, not $R^2$, so the identity does not apply
   to the statistic printed here. \citet{ethayarajh2022vusable} is therefore cited as
   *framing only* — "how much usable information a benchmark's own published metadata
   carries about its headline label" — followed by the explicit clause that we answer it
   *"in the benchmark's own currency---volume MSE and Spearman rank---rather than as an
   information-theoretic quantity."*
2. **The "order of magnitude across five benchmarks" clause is dropped.** §6 was drafted
   for the cross-benchmark paper (`null_travels.md`). This manuscript is one benchmark and
   the paragraph says so.

**"POD" is still absent from `body.tex`, on purpose.** The reviewer listed it as a grep
string, but `queipo2005surrogate` is a surrogate-modelling review (polynomial response
surfaces, Kriging, RBFs), not a POD reference, and no verified POD citation is in the
repository. Adding one would have meant citing a paper nobody checked. Flagged rather than
faked; if a POD sentence is wanted, it needs a scout pass first.

---

## 2. The force-error claim relabelled (review §2.3 labelling, question 4)

`physics/evaluation.py:108-147` computes $|pred-ref|/|ref|$ where `ref` is the force
integrated from the **rasterised ground-truth field** through the same grid integrator that
`body.tex` itself calls $\approx11\times$ biased on absolute drag (`sec:v2`) and
structurally unable to see viscous drag. Six sites now say so.

| site | before | after |
|---|---|---|
| `abstract.tex` | "force errors **4.9× lower on lift** and 3.8× lower on drag" | "$4.9\times$/$3.8\times$ **closer agreement** with the ground-truth field's own integrated lift and drag---**not with the official force labels**" |
| `body.tex` intro ¶2 | "force-coefficient errors 4.9× lower on lift and 3.8× lower on drag" | "$4.9\times$/$3.8\times$ **closer agreement** … with the forces integrated from the rasterised ground-truth field through the same grid integrator (… **at $128^2$**)", plus a following sentence naming it a **field-fidelity proxy** and quoting the $\approx11\times$ bias and the unavailable viscous component |
| `tab:interp` caption | "Force columns are mean relative error" | full relabel: "mean relative error against the force integrated from the rasterised *ground-truth field* through the same grid integrator---a field-to-field consistency measure, not agreement with the official AirfRANS labels", plus "these four columns rank arms against one another; they do not measure force accuracy" |
| `sec:interp` "Result, with its counterweight." | "on both force magnitudes" | "on both force **columns**", then: *"Those two columns need their name said out loud, because it is not the name a reader supplies"* — states the $4.9\times$/$3.8\times$ are consistency ratios, not better lift and drag, and adds that against official labels the interpolator's drag rank ($0.8389$) is indistinguishable from the exact truth field's ($0.8394$) |
| `tab:v2` caption | unlabelled $C_l$/$C_d$ columns | "relative error against the ground-truth field's own integrated forces, not against the official labels" |
| Conclusion | "4.9×/3.8× lower lift and drag errors" | "$4.9\times$/$3.8\times$ closer agreement with the lift and drag integrated from the ground-truth field through the same grid integrator---a field-fidelity proxy, not the official labels" |

`tab:airfrans-sota`'s force-error columns were **deliberately left alone**: those are the
published AirfRANS-paper numbers against *official* labels, a different quantity, and
relabelling them would have introduced a new error.

**One tension recorded, not resolved.** Review §2.3 and question 5 ask how the interpolator
can be $1.21\times$ *worse* on surface-pressure MSE and $4.9\times$ better on a lift column
that is a surface integral of that same pressure. `sec:interp` now says exactly that, and
says we have not decomposed which cells drive each. **This is a live item for the
rigor-auditor** — it is a question the paper should eventually answer, not a labelling fix.

---

## 3. The resolution ladder disclosed as PARTIAL (review §2.5, question 7)

Source: `docs/paper/review/interpolation_resolution_ladder.md`,
`results/interpolation/interp_resolution_ladder.json`,
`results/interpolation/interp_resolution_ladder_pfill.json`. The ladder appeared **nowhere**
in `body.tex` before this pass (the "ladder" hits at lines 1335/1483 are the *residual
floor* ladder, a different experiment).

**`sec:interp`, new paragraph after `tab:interp_bands`:** heading *"Under refinement the
localisation holds and the share statistic dilutes: a pre-registered `PARTIAL`."*

* **Verdict first and named:** *"The pre-registered verdict is `PARTIAL` and we report it
  as such."* The rule (share $\ge0.85$ on **both** $u$ and $v$ at every rung) is quoted,
  the shares are given at all three rungs — $0.924/0.898 \to 0.901/0.855 \to
  \mathbf{0.879/0.842}$ — and the miss is named: $v$ at $512^2$. Stated explicitly that the
  $0.85$ was written before any number existed and was not moved.
* **Protocol identity:** estimator frozen (weight matrix is a function of case *names*,
  hashes bit-identically at every rung), bands fixed in chord units, and the $128^2$ rung
  reproduces `tab:interp` at relative error $0.00\mathrm{e}{+}00$, which gates the run.
* **Why the share erodes — the numerator shrank:** inside $0.005c$ ($0.15\%$ of the domain)
  the absolute $u$ error **falls 45%** ($466.7\to257.7$) and its $R^2$ rises
  $0.507\to0.723$; beyond $0.5c$ ($80\%$ of the domain) the absolute $u$ error moves
  $0.6\%$. "A share is a ratio, so when the dominant term halves and the rest holds, every
  other band's share rises without any of them degrading."
* **The stronger result, stated as such:** at $512^2$, **72%** of the $u$ error lies inside
  $0.005c$ — the localisation is *tighter* at fine resolution. Domain fraction
  $0.9949/0.9949/0.9950$; $R^2\ge0.9996$ beyond $0.05c$ at all three rungs; outer MSE flat
  within $1.5\%$ across a $16\times$ cell-count increase.
* **Two scope notes:** $h/s_\mathrm{far}=1.28$ at $512^2$, so the outer-MSE arm was
  pre-declared to be read on the $128\to256$ pair; and **no Transolver row exists at
  $256^2$ or $512^2$**, so the ladder settles the *localisation* and settles nothing about
  the *advantage*.
* **Volunteered extra (review §2.10):** the $R^2\ge0.9996$ claim's binding entry ($p$ in
  $0.05$–$0.15c$) clears at $0.99961$ at every rung, "so that claim rests on a
  fourth-decimal margin at any resolution."

**Also added, from the same artifact:**

* **Force columns under refinement** (ladder §3c/§8.5), as a third paragraph: $C_d$ relative
  error rises $2.4\times$ (median) / $2.6\times$ (90%-trimmed) by $512^2$ **while absolute
  drag error falls** ($0.0085\to0.0060$); the untrimmed $1.84$ is one case whose rasterised
  *reference* drag crosses zero ($+0.146 \to -1.7\times10^{-4}$); $\rho_{C_l}$ unmoved at
  $0.9998$, $\rho_{C_d}$ $0.9991\to0.9811$. The long case-name token is *not* typeset (it
  would be a ~45-character unbreakable `\texttt`); the text says "the case is named in the
  artifact".
* **The fill control along the ladder** (ladder §3b/§8.4), appended to "How we tried to
  break it": the two defensible fills disagree $13.7\times / 5.8\times / 1.7\times$ at
  $128^2/256^2/512^2$, so **we make no claim about surface pressure and grid resolution**
  and we read our own $r128$ surface number as "a number about the sampler and the solid
  fill as much as about the flow." Note this is a concession *against* one of our columns.
  **The ladder document's §8.1 is stale on this point** — it proposes conceding a
  $1.9\times$ surface-pressure degradation, which its own §3b/§8.4 withdraws because the
  `nearest_all` control reverses the sign. The withdrawn version was not written into the
  paper.

**`sec:limitations`**, the "point-space head-to-head" bullet: middle replaced — "We can now
answer half of that", the ladder's stable quantities, the share easing to $0.88/0.84$ with
the missed $0.85$ floor on $v$, the verdict `PARTIAL` twice, and the unchanged statement
that the point-space head-to-head remains the single most valuable follow-up.

**Intro and Conclusion** each carry one clause attaching $128^2$ to the $92\%/90\%$ figures
and naming the `PARTIAL`.

---

## 4. "Four of five below the null" on lift, qualified (review §2.4, question 8)

Gaps recomputed from `tab:covariate` and `published_baselines_verified.md`
(null $0.9821$): MLP $(0.9821-0.913)/0.018 = \mathbf{3.8}$; PointNet
$(0.9821-0.938)/0.023 = \mathbf{1.9}$; GraphSAGE $(0.9821-0.965)/0.011 = \mathbf{1.6}$;
Graph U-Net $(0.9821-0.967)/0.019 = \mathbf{0.8}$. These match the reviewer's table
(1.55, 0.79).

**The pre-registered rule is not abandoned.** Both readings are reported everywhere.

* **`sec:covariate`** — a **fourth** qualification added ("Three qualifications" → "Four"),
  headed *"On lift the margin is decisive for two entries, not four."* It gives all four
  gaps in units of each model's own published seed sd, states that **the two intervals are
  not commensurable** (ours a case-level bootstrap over 200 test cases, theirs a spread
  over five training seeds, neither carrying the other's variance source), and lands on:
  *"the null's point estimate exceeds all four lift entries and separates decisively from
  two of them."* Final sentence scopes it: **none of this applies to drag.**
* **`body.tex` intro ¶3** — "Four of the five published baselines fall below that null on
  lift" replaced by the rule-based statement (all four AirfRANS-paper entries below under
  the registered rule) **plus** the sd reading (3.8 / 1.9 / 1.6 / 0.8) and "we rest nothing
  on the last two". Drag is where the bold now sits.
* **Conclusion** — reordered so the unqualified claim is the **drag** one; lift carries the
  per-entry sd separation.
* **`abstract.tex`** — "four of five published baselines fall below it on lift" →
  "above all four published lift entries---decisively for only two ($3.8$ and $1.9$ seed
  standard deviations)---and above every drag entry, three of four intervals spanning zero".
* **`sec:covariate` line 843** — the bold was removed from "four of the five published
  entries" so the emphasis sits on the qualified claim. The count itself is unchanged and
  still true.
* **`docs/paper/submission/highlights.txt`** — bullet 4 carried the exact overstatement
  ("Four of five published AirfRANS lift ranks fall below a case-name regression"). Because
  a highlight has no room for a qualification, it was **moved to drag**, where no
  qualification is needed: "Every published AirfRANS drag rank falls below a regression on
  the case name" (76 chars). Bullet 3 gained its resolution label: "At 128x128, 92% of
  velocity error sits within 0.02 chord of the wall: 0.5% of cells" (83 chars). Still 5
  bullets, all ≤85.

**Note on the brief's wording.** It asked to "reserve any unqualified 'four of five' for
drag". That exact phrase cannot be used on drag: Transolver reports **no drag column**, so
the drag denominator is **four**, not five (`audit_paper_numbers.py:717-719`, and
`body.tex` already said "every AirfRANS-paper entry"). The unqualified drag claim is
therefore written as *"every AirfRANS-paper entry falls below the null"*, never "four of
five on drag" — same intent, without inventing a new numeric error while fixing one.

---

## 5. New citations, with verified sources

All nine were confirmed in this session at the primary record — Crossref for the three
journal entries, the arXiv abstract pages for the six conference entries — on top of the
`[V]` verification already recorded in `naming_and_positioning.md` §2/§6.

| key | citation | verified how |
|---|---|---|
| `queipo2005surrogate` | Queipo, Haftka, Shyy, Goel, Vaidyanathan, Tucker. *Surrogate-based analysis and optimization.* Prog. Aerospace Sci. **41**(1):1–28, 2005. `10.1016/j.paerosci.2005.02.001` | Crossref `api.crossref.org/works/10.1016/j.paerosci.2005.02.001` — full author list, volume, issue, pages, year all confirmed |
| `poliak2018hypothesis` | Poliak, Naradowsky, Haldar, Rudinger, Van Durme. *Hypothesis Only Baselines in Natural Language Inference.* \*SEM 2018, 180–191. `10.18653/v1/S18-2023` | arXiv `1805.01042` abstract page (title, 5 authors, venue); pages/DOI from `naming_and_positioning.md` `[V]` |
| `gururangan2018artifacts` | Gururangan, Swayamdipta, Levy, Schwartz, Bowman, Smith. *Annotation Artifacts in Natural Language Inference Data.* NAACL 2018, 107–112. `10.18653/v1/N18-2017` | arXiv `1803.02324` abstract page (title, 6 authors, venue); pages/DOI `[V]` |
| `feng2019misleading` | Feng, Wallace, Boyd-Graber. *Misleading Failures of Partial-input Baselines.* ACL 2019, 5533–5538. `10.18653/v1/P19-1554` | arXiv `1905.05778` abstract page; pages/DOI `[V]` |
| `errica2020fair` | Errica, Podda, Bacciu, Micheli. *A Fair Comparison of Graph Neural Networks for Graph Classification.* ICLR 2020 | arXiv `1912.09893` abstract page — title, 4 authors, "published at ICLR 2020" |
| `dacrema2019progress` | Ferrari Dacrema, Cremonesi, Jannach. *Are We Really Making Much Progress?…* RecSys 2019 (Best Long Paper) | arXiv `1907.06902` abstract page — title, 3 authors, RecSys 2019. **Page range and ACM DOI deliberately omitted**: not verified here, and a wrong page range is worse than none |
| `ahlmanneltze2025deep` | Ahlmann-Eltze, Huber, Anders. *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines.* Nature Methods **22**(8):1657–1661, 2025. `10.1038/s41592-025-02772-6` | Crossref — full author list, volume, issue, pages, year (nature.com itself 303s to an IdP) |
| `ethayarajh2022vusable` | Ethayarajh, Choi, Swayamdipta. *Understanding Dataset Difficulty with $\mathcal{V}$-Usable Information.* ICML 2022, PMLR **162**:5988–6008 | arXiv `2110.08420` abstract page (title, 3 authors, ICML 2022 Outstanding Paper); PMLR volume/pages `[V]` |
| `rasp2024weatherbench2` | Rasp et al. (18 authors). *WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models.* JAMES **16**(6):e2023MS004019, 2024. `10.1029/2023MS004019` | arXiv `2308.15560` (full 18-author list, in order) **and** Crossref (journal, volume, issue, article number, year, author count = 18) |

Two BibTeX-hygiene notes, since no `bibtex` run could catch them: every `author` field uses
`First Last and First Last` form (a bare comma-separated surname list misparses as
`Last, First`), and `ethayarajh2022vusable`'s title math is brace-protected as
`{$\mathcal{V}$}-Usable` against case-changing `.bst` styles.

---

## 6. Build and script status — **NOT RUN**

Bash was disabled for the entire session that produced these edits, in this agent and in
subagents. **No LaTeX build, no `bibtex`, no `audit_paper_numbers.py`, no
`check_submission.py`, and no `git` command was executed.** All four gates remain
outstanding and must be run before this is merged.

```
cd docs/paper
pdflatex neuroforge_cfd_elsevier && bibtex neuroforge_cfd_elsevier && pdflatex neuroforge_cfd_elsevier && pdflatex neuroforge_cfd_elsevier
pdflatex neuroforge_cfd          && bibtex neuroforge_cfd          && pdflatex neuroforge_cfd          && pdflatex neuroforge_cfd
cd ../..
./.venv/Scripts/python.exe scripts/audit_paper_numbers.py
./.venv/Scripts/python.exe scripts/check_submission.py
```

**What was done instead of running them, and the residual risk.**

* **Overfull/underfull.** `preamble.tex:46-48` sets `\emergencystretch=3em`,
  `\hbadness=10000`, `\vbadness=10000`, and the Elsevier wrapper adds `\sloppy` with
  `\emergencystretch=4em`. Underfull warnings are suppressed by construction; overfull is
  controlled by `\emergencystretch`. Every addition is **prose only** — no new table, no
  new figure, no new float, no change to any tabular. The longest new unbreakable tokens
  are `\texttt{scripts/interpolation\_resolution\_ladder.py}` and
  `\texttt{results/interpolation/interp\_resolution\_ladder.json}`, both shorter than
  `\texttt{results/interpolation/interp\_band\_control\_full.json}`, which the current
  build already typesets at zero overfull. The 45-character AirfRANS case-name token was
  deliberately kept out for this reason. **Low risk, not verified.**
* **Undefined references.** Nine new `\citep`/`\citet` keys, each matching a new `refs.bib`
  entry by name; all `\autoref` targets used (`sec:v2`, `sec:interp`, `sec:limitations`,
  `tab:interp`, `tab:interp_bands`) already exist. **`bibtex` must be re-run** — the first
  `pdflatex` after this change will report undefined citations until it is.
* **`audit_paper_numbers.py`.** Not modified and **must not need to be**: no audited number
  changed value. The two rows most at risk read
  `covariate_below_null(..., "cl") == 4` and `covariate_entries_scored(..., "cl") == 5` —
  both statements survive verbatim in `sec:covariate`. New ladder numbers were *not* added
  as CLAIMS rows, because the correct JSON key paths into
  `interp_resolution_ladder.json` could not be checked without running the script; adding
  unverified readers would have converted a passing audit into a SKIP or a MISMATCH.
  **Recommended follow-up:** add ladder rows once someone can run it.
* **`check_submission.py`.** The abstract was rewritten under the 250-word cap. Hand count
  under the script's own `strip_tex` semantics (bare numerals count as words; `\%` → "%" is
  dropped; `[0.9737,0.9866]` is one token; `name---no` is one token): **244 words**, six
  under the limit, and the count and its counting rules are recorded in a comment at the
  head of `abstract.tex`.
  Highlights: still 5 bullets; the two rewritten ones measure **83** and **74** characters.
  **Hand-counted, not verified.**

---

## 7. Files changed, and the commit batches

Nothing is staged or committed. All four fixes touch `body.tex`, so a per-fix commit split
needs `git add -p`; the anchor text for each hunk is given below.

**Batch 1 — prior art.** `git add docs/paper/refs.bib` and the `body.tex` hunks at
*"Simple-baseline audits: the lineage"*, *"The estimator is decades old"*, and
*"The estimator is deliberately old"*.

```
Name the prior art this baseline comes from, and cite the audits it follows

refs.bib gains nine entries and Related Work gains two paragraphs. The first
walks the simple-baseline lineage -- partial-input, structure-agnostic,
recommender systems, Nature Methods -- and honours Feng's caveat in the
direction it actually runs. The second says in its first sentence that the
estimator is response-surface methodology and Kriging in all but name, claims
novelty only for the measurement, and closes on WeatherBench 2: weather already
mandates the trivial control we are asking for. sec:interp says it once more
where an aerodynamics reader meets the estimator.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01R3gXz5WT9knGywE2qwCvQg
```

**Batch 2 — force columns.** `abstract.tex` plus the `body.tex` hunks at intro ¶2, the
`tab:interp` caption, *"Result, with its counterweight."*, the `tab:v2` caption, and the
Conclusion.

```
Say what the force columns measure, in the abstract and at every table

The 4.9x/3.8x is agreement with the forces integrated from the ground-truth
field through the same grid integrator this paper calls ~11x biased and blind to
viscous drag. It is a field-to-field consistency measure, not agreement with the
official labels, and it now says so in the abstract, the introduction, both
table captions and the conclusion. The surface-pressure/lift tension is recorded
where a reader meets it, unresolved and marked as such.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01R3gXz5WT9knGywE2qwCvQg
```

**Batch 3 — the ladder.** The `body.tex` hunks at *"Under refinement the localisation
holds"*, the force-column ladder paragraph, the fill-control sentences in *"How we tried to
break it"*, the `sec:limitations` bullet, and the intro/conclusion clauses.

```
Report the interpolation resolution ladder, verdict first, as PARTIAL

The ladder was committed and the manuscript did not mention it. It does now,
with the pre-registered verdict named before any number: PARTIAL, missing only
the 0.85 share floor on v at 512^2. The mechanism is disclosed too -- the
numerator shrank, the innermost absolute error fell 45%, and at 512^2 72% of the
streamwise error sits inside 0.15% of the domain, so the localisation is tighter
than the deployed table could show. The fill control's own ladder concedes that
our r128 surface-pressure number is not fill-robust.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01R3gXz5WT9knGywE2qwCvQg
```

**Batch 4 — the lift margin.** `submission/highlights.txt` plus the `body.tex` hunks at
intro ¶3, `sec:covariate` (both), and the Conclusion.

```
State the lift margin in units of each model's own published spread

The pre-registered rule stands and all four AirfRANS-paper lift entries fall
below the null under it. Against each model's own seed spread the separation is
3.8 and 1.9 standard deviations for MLP and PointNet but 1.6 and 0.8 for
GraphSAGE and Graph U-Net, and the two intervals are not commensurable anyway.
Both readings are now reported, the unqualified claim is reserved for drag, and
the highlights bullet that carried the overstatement is replaced.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01R3gXz5WT9knGywE2qwCvQg
```

---

## 8. Left open, and routed elsewhere

* **§1, the fatal measure asymmetry** — needs the node-fraction measurement and/or a
  point-space head-to-head. Owned by the concurrent agent
  (`docs/paper/review/measure_asymmetry.md`). `body.tex:450-452`'s "the input asymmetry runs
  *against* the baseline" is **untouched** and still needs the objective-asymmetry sentence
  once that measurement lands.
* **§2.3 (experiment half)** — official-label $C_l$/$C_d$ magnitude error for both arms;
  median and trimmed means for the force columns. Labelling done, measurement not.
* **§2.6** — whether the trust-layer sections move to the companion paper. A structural
  decision, not a rewrite; not taken here.
* **§2.8** — the Transolver train/eval density probe. Cheap, not run.
* **§2.9** — contribution 3's unconditional `reynolds` claim. `sec:splits` has the correct
  conditional form; the contributions list, abstract and conclusion still drop the
  condition. **Not in the four, flagged for the rigor-auditor.**
* **§2.10** — partially addressed: the fourth-decimal margin on $R^2\ge0.9996$ is now
  disclosed in `sec:interp`, but the abstract still quotes the pooled $0.9996$ while
  `tab:interp_bands` prints only $R^2_\mathrm{pc}$. Adding the pooled column to that table
  is the clean fix and was not attempted (table edit, unbuildable session).
* **§2.11** — the two Transolver rows and the two differently-defined $\rho_{C_l}$ are still
  unlabelled as such.
