# Compaction pass — the residual audit moves out, the evaluation results carry the paper

Date 2026-09-11. Branch `paper1/reframe-after-jcp`.

---

## ⚠ NOT BUILT / NOT AUDITED / NOT COMMITTED — read this first

**The `Bash` tool was disabled for this session, in this agent and in subagents.** I could
not run `git`, `pdflatex`, `bibtex`, `scripts/audit_paper_numbers.py` or
`scripts/check_submission.py`. Every edit below is **written to disk**; none of it is
built, audited, staged or committed. §8 is the exact command list the next process must
run and §9 is everything I could not verify. This is the same constraint the previous pass
hit (`restructure.md` §7–§8); it has now bitten twice and is worth fixing before the next
paper session.

Two mechanical consequences the next process must handle:

1. **`bibtex` must be re-run**, not just `pdflatex` twice. I added no new `\cite` key, so
   no reference can become undefined — but eight keys are now cited **nowhere**, having
   lived only in the theorem section this pass unhooked: `rhiechow1983`,
   `krishnapriyan2021failure`, `wang2022ntk`, `trefethenbau1997`, `stetter1978defect`,
   `brandt2011multigrid`, `morin2000data`, `bochevgunzburger2009book`. Without a `bibtex`
   re-run the stale `.bbl` will print all eight as uncited bibliography entries. All eight
   are already in `refs.bib`, so the re-run is safe.
2. **`tab:interp` is now ten columns** (§3.2). `adjustbox` guarantees it cannot overfull,
   but it will be scaled down and someone should eyeball its legibility in the PDF. If it
   is too small, the honest fix is to split the force columns back out, **not** to drop a
   column.

---

## 1. Before and after

The brief's original target was pages; the coordinator corrected it mid-pass to **prose
word count**, which is the right signal (`elsarticle` at `preprint,12pt,a4paper` is
single-column 12pt with generous margins, roughly 2.5–3× less dense than the published
two-column format).

| measure | before | after | method |
|---|---:|---:|---|
| `body.tex` prose words | **17,624** | **≈ 11,000** *(est., range 10,750–11,700)* | see below |
| `sections/residual_floor_theorem.tex` | 4,125 | 0 (unhooked) | no longer `\input` |
| `abstract.tex` | 242 | **≈ 243** *(hand-simulated)* | `check_submission.count_words` rule, applied by hand |
| **total prose** | **21,991** | **≈ 11,250** | — |
| `neuroforge_cfd_elsevier.pdf` pages | 81 | **≈ 51** *(est.)* | see below |
| `neuroforge_cfd.pdf` (TMLR) pages | 50 | **≈ 32** *(est.)* | scaled from the same ratio |
| `body.tex` source lines | 2,688 | **1,682** | exact |

**How the word estimate was made, since I could not run a counter.** `body.tex` is 1,682
lines, of which 134 are blank, 10 are the header comment block, 311 sit inside
`table`/`figure` environments and 21 inside `align`/`equation`/`proposition` — leaving
**1,206 prose lines**. Calibrating against the coordinator's own measurement of the
pre-pass file (17,624 words over an equivalently computed 1,906 prose lines) gives **9.25
words per prose line**, hence ≈ 11,150. The range quoted allows 9.0–9.8. **Please re-run
your own counter; treat my figure as an estimate, not a measurement.**

The page estimate scales the 71 text pages of the old 81-page Elsevier build by the prose
ratio (1,206 / 1,906 → ≈ 45 pages) and subtracts ≈ 3.6 pages of deleted float area from
the old 10 float pages (`fig:pipeline` a full page, `tab:positioning` ≈ 0.8,
`fig:fig4` ≈ 0.45, `tab:transolver` ≈ 0.35, `tab:descent200` ≈ 0.4, `tab:floor_ladder`
≈ 0.35, `tab:iters` ≈ 0.3).

### The honest floor, and what binds it

I stopped at ≈ 11,000 rather than 10,000, and the binding constraint is the instruction
*never cut tables of results, seed counts, confidence intervals, controls, or any measured
number.* The manuscript now carries eleven results tables and, by rough count, north of
three hundred distinct measured quantities, most of them with a control or an interval
attached. The specific blocks I judged I could not compress further without losing
evidence:

- `sec:v2`'s force-integrator forensics (≈ 300 words) — fifteen measured numbers
  establishing that our `rho_Cd` is self-consistency and not agreement with official
  labels, plus the covariate-baseline result that the near-field number sits *below*.
- `sec:conformal`'s corrected-field calibration paragraph (≈ 330 words) — three
  independent verifications (n=15 frozen-q contrast, n=20 replication, the deployed
  100/100 three-seed direct test) with per-channel coverages and quantiles.
- `sec:interp`'s five adversarial controls and the feature/train-size ablations (≈ 350
  words) — these are the reason the headline survives review.
- Limitations (≈ 1,000 words) — merged from fourteen items to twelve; every distinct
  limitation is preserved, and several now carry numbers that used to live in the removed
  sections.

If ≈ 11,000 is still too long, the next honest cut is a **decision**, not an edit: drop the
force-integrator forensics and the OOD ablation (`sec:ood`, `tab:ood`) to the companion
paper as well, which would take roughly 900 further words. I did not do that unilaterally
because both are controls.

---

## 2. Exactly what was removed, and where the companion paper picks it up

**Nothing was deleted without preservation.** Two files hold it, neither `\input` by either
build, both left in the repository:

| file | contents | state |
|---|---|---|
| `docs/paper/sections/residual_floor_theorem.tex` | the full theorem section: `thm:consistency-floor`, `thm:residual-floor`, `prop:kernel`, `sec:rf_operator`, all proofs and scope paragraphs (628 lines / 4,125 words) | **untouched**, as instructed; only the `\input` line in `body.tex` was removed |
| `docs/paper/sections/residual_audit_removed.tex` | **new.** Eight verbatim blocks lifted out of `body.tex`, each with its original line range recorded in a comment above it | new file, never compiled |

`residual_audit_removed.tex` contains, in order:

| block | what it is | original `body.tex` lines (at commit `d0ba530`) |
|---|---|---|
| 1 | Related Work, ``Learned solver-correctors and the `fixer' premise'' — full version, including the operator-consistency argument and the **86 % floor-share of a typical score** | 228–268 |
| 2 | Related Work, ``Residual-based, data-free error signals'' — full version, including the a-posteriori / adjoint-weighting argument and the three-part contribution list | 270–305 |
| 3 | `tab:positioning` (the solver-consistent vs. surrogate-side proxy boundary table) and the Positioning paragraph that introduced it | 380–440 |
| 4 | `sec:iters`, ``The iteration sweep'' — the four-limits paragraph, `tab:iters`, the ``As the loop iterates'' paragraph, `fig:fig4`'s caption, and the ``This does not rescue the residual as an objective'' coda | 1551–1615, 1650–1660, 2288–2298 |
| 5 | `sec:floor_ladder`, ``The floor does not refine away'' — `tab:floor_ladder`, the four controls (MMS, closure repair, cubic interpolant, pipeline null) and ``What this leaves'' | 1674–1760 |
| 6 | `sec:descent`, ``Descending the residual directly'' — the 24-case Adam arm, the iterate-selection paragraph, `tab:descent200`, and the `rho`-regime exception | 1762–1857 |
| 7 | `sec:regime`, ``The boundary is the operator, not the problem class'' — including the `zhang2026phymgn` positioning | 1859–1886 |
| 8 | `sec:conformal`, ``What the floor costs the certificate'' — the full version of the floor-subtraction reversal argument | 1964–1986 |

**Labels the companion paper must re-declare** (they no longer exist anywhere in a compiled
file): `sec:floor_ladder`, `sec:descent`, `sec:regime`, `sec:residual_floor`,
`sec:rf_operator`, `tab:floor_ladder`, `tab:descent200`, `tab:iters`, `fig:fig4`,
`tab:positioning`, `thm:consistency-floor`, `thm:residual-floor`, `prop:kernel`,
`sec:baseline`. This list is repeated in the header of `residual_audit_removed.tex`.

**Two scope caveats were nearly lost in compression and have been restored**, because the
sections that used to carry them are gone and both are previously-fixed overclaims:

1. The **MMS control** ($p=2.06$) bounds truncation error and is **not** evidence that the
   monitored operator is consistent with the one that generated the labels---the
   manufactured field is a potential flow, on which the monitored viscous term vanishes
   identically. Without this clause `sec:residual` would assert second-order convergence
   with no counterweight inside a section whose point is operator inconsistency.
   (`whitespace.md` §0a; the `d0ba530` withdrawal.)
2. The **cubic-interpolant control** bears on the second-derivative block---5--7 % of the
   floor---not on the first-derivative block carrying 93--95 % of it
   (`mechanism_decision.md` B2).

Separately, one measured clause that fell out of the `sec:v2` rewrite was restored rather
than silently dropped: *"recovering it needs $N\sim10^5$ cells where lift, set by
chord-scale surface pressure, converges at first order over the same ladder."*

**What the manuscript kept** — `sec:residual`, ≈ 750 words, four paragraphs, no table:
the floor at the truth (mean 0.192, median 0.133, uniform freestream exactly zero on
200/200), the refinement ladder in one sentence (0.0624 → 0.0779 → 0.1067, 21/24,
p = −0.387, and the independent 0/16), the four controls in one sentence each, the
200/200 descent-from-truth result with the n = 200 Armijo numbers, the 24/24
iterate-selection statement, the `rho`-regime exception, and the operator-consistency
boundary. It opens by naming the reviewer objection it answers — *"then use the physics
residual to fix the metric"* — and closes with an explicit pointer to the companion paper
for the theorem and the full evidence.

---

## 3. Everything else that changed

### 3.1 Structural

| change | effect |
|---|---|
| `\input{sections/residual_floor_theorem}` removed | −4,125 words |
| `fig:pipeline` (78-line TikZ schematic) deleted | −1 float page; it depicted the demoted three-roles framing |
| `Method` reduced from 5 subsections to 2 | module map, frozen-contract prose, backbone descriptions, training loss, synthetic generator and the standalone fallback subsection all replaced by one pointer sentence to the released package |
| `Implementation` section deleted entirely | its AI-assisted-development disclosure was **preserved** verbatim, moved to the back matter as its own `\section*` (it is a required declaration, not prose) |
| `sec:baseline` (matched-budget baseline) dissolved | `tab:transolver` merged into `tab:interp`; its prose folded into `sec:interp`'s cost-asymmetry paragraph and `sec:v2`'s "Relation to the corrector result", which is now two sentences inside the corrector paragraph |
| `sec:iters` dissolved | the acceptance-gate measurement (the part that is a result) moved into `sec:selective`; the iteration sweep moved out |
| `\appendix` added | the `Figures` subsection inside Experiments is gone. **Context floats only** move to Appendix A: `tab:airfrans-sota` (published baselines, field context), `fig:fig3` (the grid backbone's gap to Transolver, for which we make no competitiveness claim) and `fig:cylinder` (explicitly qualitative). The six **evidence** figures now sit beside the claims they support: `fig:fig1` in `sec:indist-ablation`, `fig:fig2` and `fig:deepcfd` in `sec:ood`, `fig:fig6` and `fig:fig5` in `sec:conformal`, `fig:risk_coverage` in `sec:selective`. All floats use `[t]`, the document's established specifier. |
| Contributions list | stops restating the four findings the introduction paragraphs already state with their numbers; each item now adds only what the paragraph does not say |
| Conclusion | three long restatements merged into two paragraphs plus the standing recommendation |
| Limitations | fourteen items merged to twelve; every distinct limitation preserved. Merged pairs: {grid resolution, near-wall blindness, viscous-drag unavailability} → one item; {n=24 Adam scope, the frozen-`nu_t` headline arm} → one item; {conformal OOD caveat, MC-dropout non-adaptivity, the M-study} → one item; {residual-as-objective, the gate, the W1 null} → one item. The iteration-sweep limitation was removed with the sweep. |

### 3.2 `tab:interp` is now the single comparison table

It gained `rho_Cl` and `rho_Cd` columns and the `our grid backbone + DEQ` row from
`tab:transolver`, which is deleted. **No number was lost**: Transolver's force-rank seed
standard deviations (0.0002, 0.0012) moved into the caption to keep the ten-column table
from scaling down further. `fig:fig3` now points at `tab:interp`.

### 3.3 The gap statement (end of the introduction)

Added as a `\paragraph{The gap this fills.}`, deliberately scoped, and grounded in
`field_scan_sept2026.md` §1.2(f) and `whitespace.md`:

- Claims only that *no published work reports a parameter-space interpolation baseline for
  AirfRANS field metrics or a covariate null for its force-rank metric.*
- **Does not** say "the baseline nobody publishes." It names DrivAerNet++'s AutoML tabular
  baseline on its 26 design parameters as the partial occupant and states the three things
  that survive it (R²-on-magnitude not rank; the same group's follow-on benchmark drops it;
  nobody audits *published* force numbers against it).
- Quotes the AirfRANS authors' own **page-8 diagnosis** (verified in
  `published_baselines_verified.md`): *"the models have difficulties to predict the wall
  shear stresses as the velocity values at the closest nodes from the geometry are often
  largely overestimated"*, which *"particularly affects the accuracy of the drag
  coefficient"*. This is corroboration from the dataset's own paper and is the strongest
  citation available; it needs no new `refs.bib` entry.

`restructure.md` §9.1's open item stands: **DD-RNO (arXiv:2608.13490) is still not cited.**
Adding it needs a new `refs.bib` entry, which this session could not verify with a build.

---

## 4. The bolded quantities, and the artifact behind each

Bolding density: **five** in the abstract, **four** in the introduction's headline
paragraph, and roughly one per results paragraph thereafter.

| quantity | where | artifact | auditor row |
|---|---|---|---|
| **88 % lower** volume-pressure error (8.4×) | abstract, intro, `sec:interp`, conclusion | `interp_full.json` `variants.nd.test_metrics.mse_p` = 75.03 vs `reference_rows.transolver_tab_transolver.mse_p` = 628.5 | **new**: `interp mse_p lower than Transolver by (%)` = 88.0 |
| 62 % lower cross-flow error (2.6×) | abstract, intro, `sec:interp` | same file, `mse_v` 0.0336 vs 0.088 | **new**: `interp mse_v lower than Transolver by (%)` = 62.0 |
| **4.9× lower on lift**, 3.8× lower on drag | abstract, intro, `sec:interp`, conclusion | `results/baselines/table2.csv` `cl/cd_rel_err_mean` (0.058109 / 0.089898) ÷ `interp_full.json` (0.011804 / 0.023895) | **new**: two rows, tolerance 0.05 = the rounding half-width |
| **92 % / 90 %** of u/v error inside 0.02c | abstract, intro, `sec:interp`, conclusion | `interp_band_control_full.json` `A_band_decomposition["0-0.02c"].se_share_u/v` | existing rows (0.924, 0.898) |
| **0.5 % of cells** | abstract, intro, `sec:interp`, conclusion | same block, `cell_frac` = 0.005070 | **new**: `cells inside the 0-0.02c wall band` = 0.005 |
| **98.7 % of the domain** at R² ≥ 0.9996 | abstract, intro, `sec:interp`, conclusion | same block, `cell_frac` summed over the three bands beyond 0.05c = 0.98751 | **new**: `cells beyond 0.05c, where R^2 >= 0.9996 (%)` = 98.7 |
| **four of five** published baselines below the lift null | abstract, intro, `sec:covariate`, conclusion | `covariate_null_trainfit.json` `leaderboard_vs_full_name_null` — four `BELOW THE NULL`, Transolver `clears` | existing row (4) **plus new**: `published entries carrying a lift number` = 5 |
| three of four published **drag intervals span zero** | intro, `sec:covariate`, conclusion | same block, `published ± published_std`: MLP −0.117±0.256 ✓, PointNet −0.022±0.097 ✓, Graph U-Net −0.138±0.258 ✓, GraphSAGE −0.303±0.124 ✗ | **new**: `published drag intervals that span zero at +-1 seed std` = 3 |
| 200/200, exact zero, 24/24 | intro, `sec:residual`, conclusion | `residual_floor_realdata.json`, `residual_descent.json` | existing rows |

### Two framings from the brief that I did **not** write as given — routed to the rigor-auditor

1. **"99.5 % of the domain is reproduced with no learning at all" is not the same
   measurement as "R² above 0.9996 beyond 0.05 chord."** R² ≥ 0.9996 holds *beyond 0.05c*,
   which is 0.0294 + 0.1583 + 0.7998 = **98.7 %** of cells. 99.5 % is the complement of the
   0–0.02c band, and the 0.02–0.05c band in between (0.7 % of cells) is covered by
   *neither* statement — it sits at R²_pc 0.9986. The manuscript says **98.7 % at
   R² ≥ 0.9996** and, separately, **0.5 % of cells** for the wall band. The auditor now
   carries both, with a comment forbidding the merge.
2. **"The entire learning advantage is confined to 0.5 % of cells" is true for velocity,
   not for pressure.** The wall band's squared-error share is 0.924 (u) and 0.898 (v) but
   **0.535 (p)**. `sec:interp` now carries an explicit scope note saying so. The abstract
   and introduction attach the claim to the u and v channels only.

Neither is a change of evidence; both are corrections to how the evidence was being
phrased, and both should be checked by the rigor-auditor before submission.

---

## 5. Auditor changes (`scripts/audit_paper_numbers.py`)

**Eight rows removed**, each because the passage quoting it left the manuscript. None were
left dangling:

| removed row | why |
|---|---|
| `omitted closure @128^2`, `omitted closure @512^2` | the per-rung magnitudes were quoted only by `sec:floor_ladder`; the surviving `sec:residual` quotes the closure term as a **fraction** of the floor, which `decomp_omitted_fraction` still checks |
| `residual cut, descent from truth (%)` = 84.0 | the 24-case Adam arm's headline moved to the companion |
| `cases where residual fell, from truth` = 24 | same |
| `median field error after descent from truth` = 0.91 | same |
| `floor share of a typical score (%)` = 86.0 | quoted only by the Related Work paragraph now in `residual_audit_removed.tex` block 1 |
| `Transolver inversion rate, min (%)`, `... max (%)` | quoted only inside `residual_floor_theorem.tex` (lines 434, 461) — this row was **already** checking a number the main body never stated |

**Six rows added**, all listed in §4. **Five readers added**
(`covariate_entries_scored`, `covariate_drag_spans_zero`, `interp_pct_lower`,
`transolver_table2`, `interp_force_ratio`, `band_outer_cell_frac`). Every ratio and
percentage is recomputed from the artifact rather than hardcoded, so a re-run that moves
either side is caught in both the ratio form and the percentage form at once.

**Five readers are now unreferenced** (`descent`, `descent_residual_cut`, `descent_wins`,
`transolver_inversion`, `gate_followup`). They are retained deliberately, with a comment
saying why: the companion paper's auditor can reuse them without re-deriving the JSON key
paths.

`results/baselines/table2.csv` is a **new source file** for the auditor, needed because
`interp_full.json`'s `reference_rows` block carries no force columns. It is the file
`tab:interp`'s Transolver row was transcribed from.

---

## 6. Label integrity — checked by hand, since no build was possible

I extracted every `\label{}` and every `\autoref{}` from the new `body.tex` and diffed the
two sets. **Every referenced label is defined.** The 29 labels actually referenced:

`sec:method` `sec:experiments` `sec:interp` `sec:splits` `sec:covariate` `sec:v2`
`sec:indist-ablation` `sec:ood` `sec:conformal` `sec:selective` `sec:cost` `sec:residual`
`sec:limitations` `prop:coverage` `prop:contraction` `tab:interp` `tab:interp_std`
`tab:interp_bands` `tab:covariate` `tab:v2` `tab:uq` `tab:indist` `tab:ood` `tab:cost`
`tab:airfrans-sota` `fig:fig1` `fig:fig2` `fig:fig3` `fig:fig5` `fig:fig6`
`fig:risk_coverage` `fig:deepcfd` `fig:cylinder`.

One label, `sec:correction`, is defined and unreferenced — harmless, and it marks the
subsection the correction operators live in. `prop:contraction`, `tab:cost` and `fig:fig3`
were each unreferenced after the first draft of this pass; I added one reference to each
rather than leave an orphan.

All `\begin`/`\end` pairs balance: 18 float environments (10 `table`, 8 `figure`, counted
as 36 matching `\begin`/`\end` lines), 8 `adjustbox`/`resizebox`, 11 `tabular`, 2
`proposition`, 1 `align`, 1 `equation`, 1 `enumerate`, 1 `itemize`. Column counts were
checked row by row for every table. Eighteen float labels, no duplicates.

### 6.1 The CLAIMS ↔ manuscript cross-check (the auditor cannot do this itself)

`restructure.md` §4 recorded the auditor's blind spot in its own words: it compares JSON to
a hardcoded constant *in the script*, never to the `.tex`. So "every checked number matches
its source file" would pass whether or not the number is still in the paper — exactly the
wrong property after a wholesale 2,688 → 1,671-line rewrite. I therefore grepped every
surviving row's stated value against `body.tex` and `abstract.tex`.

**Result: clean in both directions.** All 81 surviving rows have their value present in the
manuscript, and no removed passage left a row behind. Two pre-existing, benign rounding
mismatches between the script's stated constant and the manuscript's rounding, both
inherited and both inside tolerance:

- `Transolver beats interp on nu_t by` — script constant 15.4, manuscript says "$15\times$".
- `aoa C_l rel err (%)` — script constant 23.06, manuscript says "$23.1\%$".

One row was relabelled rather than removed: `omitted term as % of floor` now reads
`omitted term as % of floor (paper: 3.5-4.8)`, because the surviving `sec:residual` states
that quantity as a range while the reader returns the mean, which must sit inside it.

---

## 7. Files changed

Written this session:

```
docs/paper/body.tex                              rewritten (2,688 -> 1,671 lines)
docs/paper/abstract.tex                          rewritten (~243 words)
docs/paper/sections/residual_audit_removed.tex   NEW (preservation file, never compiled)
docs/paper/submission/highlights.txt             one bullet rescoped to "four of five"
scripts/audit_paper_numbers.py                   8 rows removed, 6 added, 6 readers added
docs/paper/review/compaction.md                  NEW (this file)
```

Untouched, deliberately:

```
docs/paper/sections/residual_floor_theorem.tex   left exactly as it was, now unreferenced
docs/paper/preamble.tex                          pifont/tikz are now unused but harmless
docs/paper/neuroforge_cfd.tex                    no change needed
docs/paper/neuroforge_cfd_elsevier.tex           no change needed
docs/paper/submission/arxiv_v4/stage/            frozen shipped snapshot
```

---

## 8. Commands the next process MUST run — none were run here

```
cd docs/paper
bibtex neuroforge_cfd            # REQUIRED: 8 keys are no longer cited (see the banner)
bibtex neuroforge_cfd_elsevier
pdflatex neuroforge_cfd.tex && pdflatex neuroforge_cfd.tex
pdflatex neuroforge_cfd_elsevier.tex && pdflatex neuroforge_cfd_elsevier.tex
# both must end: 0 overfull, 0 underfull, 0 undefined, 0 errors
cd ../..
./.venv/Scripts/python.exe scripts/audit_paper_numbers.py   # must end "every checked number matches its source file"
./.venv/Scripts/python.exe scripts/check_submission.py      # must end "all mechanical requirements satisfied"
```

If `bibtex` complains, note that `neuroforge_cfd.aux` / `neuroforge_cfd_elsevier.aux` are
stale (they still carry the deleted labels) and should be removed before the first
`pdflatex` so the `.aux` is rebuilt from the new source.

Files to stage, **by name** (never `git add -A`):

```
docs/paper/body.tex
docs/paper/abstract.tex
docs/paper/sections/residual_audit_removed.tex
docs/paper/submission/highlights.txt
docs/paper/review/compaction.md
scripts/audit_paper_numbers.py
```

Suggested split into two commits, so the preservation is separable from the deletion:

1. `Preserve the residual-audit passages before removing them` — `residual_audit_removed.tex` only.
2. `Cut the paper to the evaluation results, and quantify the headline` — everything else.

Both messages must end:

```
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016TFg4pPp42KHpYP9ztng2a
```

---

## 9. Final status of the two builds and the two scripts

| gate | required | actual |
|---|---|---|
| `neuroforge_cfd.tex`, pdflatex ×2 | 0 overfull / 0 underfull / 0 undefined / 0 errors | **NOT RUN** |
| `neuroforge_cfd_elsevier.tex`, pdflatex ×2 | same | **NOT RUN** |
| `scripts/audit_paper_numbers.py` | "every checked number matches its source file" | **NOT RUN** |
| `scripts/check_submission.py` | "all mechanical requirements satisfied" | **NOT RUN** |

What I checked by hand instead, and how far that goes:

- **Undefined references:** §6. Complete set diff, high confidence.
- **Undefined citations:** no new `\cite` key was introduced, so none can be undefined.
  The eight now-uncited keys are a cosmetic problem only, fixed by `bibtex`.
- **Underfull boxes:** `preamble.tex` sets `\hbadness=10000` and `\vbadness=10000`, which
  suppresses them by construction. Unchanged by this pass.
- **Overfull boxes:** the risk sits in `tab:interp` at ten columns. `adjustbox`'s
  `max width=\textwidth` shrinks rather than overflows, so this cannot produce an overfull
  box; it can produce an unreadably small table. Eyeball it.
- **Abstract word count:** hand-simulated `check_submission.count_words` sentence by
  sentence: **≈ 243** against the 250 limit, leaving ≈ 7 words of margin. My count could be
  off by one or two; if it comes back over, the cheapest legitimate cut is
  `"---no network, no flow-field learning---"` → `"with no network"` (−3).
- **Highlights:** five bullets, longest now 83 characters (the rescoped bullet 4), all
  under 85. Counted by hand.
- **New auditor rows:** each was hand-computed against the JSON to inside its stated
  tolerance. The two force-ratio rows use a tolerance of 0.05, which is exactly the
  half-width of the one-decimal rounding the manuscript quotes, so they pass iff the
  manuscript's rounding is right (4.9229 → 4.9; 3.7621 → 3.8).
- **Python syntax:** not executed. The edits are three helper-function insertions, one
  helper deletion, and additions/removals of complete `(label, reader, value, tol)` tuples
  including their trailing commas. No tuple was half-deleted.

---

## 10. Open items this pass did not close

1. **DD-RNO (arXiv:2608.13490) is still uncited** — `restructure.md` §9.1. Needs a
   `refs.bib` entry and a build to verify. It is the strongest available citation for "the
   field knows the confound exists and still does not report the baseline."
2. **`mechanism_decision.md` B7** (the `h/s` appendix sentence) remains unapplied; the
   appendix it targeted is now in the companion paper's file.
3. **`docs/paper/review/interpolation_baseline.md` §8** still contains a ready-to-paste
   paragraph with two wrong numbers ("from 50 training cases", "10794 ± 910"). Neither
   reached the manuscript in either pass, but §8 is written to be pasted. Fix it at source.
4. **No figure exists for findings 1–3.** All eight surviving figures illustrate the
   demoted trust-layer results. The six that are evidence now sit beside their claims in
   the body, but the three lead sections (`sec:interp`, `sec:splits`, `sec:covariate`) run
   figure-free. A wall-band error-share plot and a null-vs-leaderboard scatter would be the
   two highest-value additions; neither exists.
5. **`submission/cover_letter.md` and `suggested_reviewers.md`** still argue the pre-
   restructure thesis and were not touched.
6. **The `Bash`-disabled sessions are now a pattern.** Two consecutive passes have produced
   unbuilt, unaudited, uncommitted work. The paper has not been compiled since before the
   restructure; two large rewrites are stacked on top of the last green build.
