# Restructure: the evaluation results lead, the residual audit becomes support

Date 2026-09-11. Branch `paper1/reframe-after-jcp`.

**READ THIS FIRST — the deliverable is incomplete in one specific, mechanical way.**
The `Bash` tool was disabled for this session, in this agent and in subagents. I could
not run `pdflatex`, `git`, `scripts/audit_paper_numbers.py` or
`scripts/check_submission.py`. Every edit below is written; **none of it is built,
audited or committed.** §7 lists the exact commands the next process must run, and §8
lists everything I could not verify.

---

## 1. The new spine

| # | finding | where it lives now |
|---|---|---|
| 1 | Parameter interpolation is the missing baseline; the surrogate's advantage is one grid cell wide | `sec:interp`, `tab:interp`, `tab:interp_std`, `tab:interp_bands` |
| 2 | The benchmark's declared primary metric is a covariate property of the case index | `sec:covariate`, `tab:covariate` |
| 3 | The `reynolds` split does not test Reynolds generalisation | `sec:splits` |
| 4 | What a surrogate-side physics residual can and cannot do | `sec:residual` (one subsection) + `sec:residual_floor` (the theorem) |

## 2. New outline

```
Title      Neural CFD surrogates earn their keep in the first cell off the wall:
           a parameter-interpolation baseline and a covariate null for AirfRANS
Abstract   rewritten; 239 words by check_submission's own counting rule (limit 250)

1 Introduction
  - opening: the field measures gains only against other learned models
  - Research question: what does learning add over interpolation, and do the
    benchmark's metrics measure it?
  - para: learning earns its keep in the first cell off the wall   [finding 1]
  - para: the force metrics do not measure that difference          [finding 2]
  - para: one of the extrapolation splits does not extrapolate      [finding 3]
  - para: what a surrogate-side physics residual can and cannot do  [finding 4]
  1.1 Contributions  (7 items, reordered: interp -> covariate -> splits ->
      residual -> detector/fusion -> certificate+gate -> package)
  How to read this paper  (rewritten to the new order)

2 Related Work
  ... existing paragraphs ...
  - Benchmarks, and what they compare against        [REWRITTEN + expanded]
  - Evaluation and reporting standards in ML for PDEs [NEW]
  2.1 Positioning                                     [REWRITTEN]

3 Method / 4 Implementation   (unchanged)

5 Experiments
  5.1 Setup
  5.2 Where learning earns its keep: the parameter-interpolation baseline  [NEW]
  5.3 The reynolds split does not test Reynolds generalisation             [NEW]
  5.4 The benchmark's force metrics do not measure that difference         [NEW]
  5.5 v2 (Transolver backbone)      ... + new "Relation to the corrector result"
  5.6 grid backbone / 5.7 OOD / 5.8 iteration sweep
  5.9 What the physics residual can and cannot do          [MERGED, ~45% shorter]
      5.9.1 The floor does not refine away      (was 5.x "...---it grows")
      5.9.2 Descending the residual directly
      5.9.3 The boundary is the operator, not the problem class
  5.10 conformal / 5.11 selective / 5.12 cost / 5.13 matched-budget baseline
  5.14 Figures / 5.15 pre-registered hypotheses
  (theorem section is \input here, unchanged in position)

6 Limitations  (4 new items first) / 7 Conclusion (rewritten, findings 1-3 lead)
```

## 3. What moved, what is new, what was cut

**New prose and tables** (all numbers traced, see §5):
- `sec:interp` — method, `tab:interp` (9 rows incl. floor ladder + permuted-parameter
  negative control), `tab:interp_std` (both standardised means side by side),
  `tab:interp_bands` (wall-distance decomposition with the strict per-case-centred
  R²), plus paragraphs on the five adversarial controls, the feature ablation, the
  train-size ladder, and the cost asymmetry.
- `sec:splits` — containment proof, nd-vs-raw control, `aoa` contrast.
- `sec:covariate` — `tab:covariate` (four nulls with bootstrap CIs, five published
  entries), plus three pre-emptive qualifications (digits are fair / fit is honest /
  no leakage accusation).
- New row in `tab:transolver`: `parameter interpolation (no learning)`.
- New paragraph "Relation to the corrector result" after the AirfRANS positioning
  paragraph, connecting the baseline to the `tab:v2` corrector deltas before a
  reviewer does.
- Related work: "Evaluation and reporting standards in ML for PDEs" paragraph.
- Four new Limitations items, placed first.

**Compressed** — `sec:floor_ladder` + `sec:descent` + `sec:regime` went from three
top-level subsections (~253 lines) to three `\subsubsection`s inside one subsection
(~140 lines). No evidence was deleted: the two tables (`tab:floor_ladder`,
`tab:descent200`), all four controls, the 24/24 and 200/200 counts, the
iterate-selection framing, the `rho`-regime exception and the `lei2026newtonkrylov`
boundary all survive. What was cut is repetition, the per-case ladder narration and
the duplicated `mcgreivy2024weak` genre paragraph (moved into Related Work).

**Cut outright** from the Introduction: the old "The problem is trust" paragraph, the
old three-roles paragraph, the old Thesis sentence, the "Why this is not a statement
about our grid" paragraph and the "Self-auditing, defined" paragraph. Their content
survives, shorter, in the new finding-4 paragraph and in `sec:residual`.

**Labels.** Every label that was referenced still exists. `sec:floor_ladder`,
`sec:descent` and `sec:regime` are now `\subsubsection` anchors (numbered, so
`\autoref` resolves cleanly). New labels: `sec:interp`, `sec:splits`, `sec:covariate`,
`sec:residual`, `tab:interp`, `tab:interp_std`, `tab:interp_bands`, `tab:covariate`.
`sec:regime` is now unreferenced but retained (harmless).

## 4. Withdrawn claims I found still live in the manuscript — REPORTED, and fixed

`docs/paper/review/mechanism_decision.md` §6 lists edits T1–T8 and B1–B8 that withdraw
the operator-provenance attribution and correct the conformal numbers. **Only A1 (the
abstract width) and T3 had been applied.** The rest were still in the files on this
branch, despite commit `d0ba530` ("Withdraw the mechanism claim"). Specifically:

| site | what it still said | status |
|---|---|---|
| `body.tex` intro thesis | "by a margin that grows under grid refinement" | paragraph deleted in the rewrite |
| `body.tex` "Why this is not a statement about our grid" | cubic control sold as "the sharpest control" (B2) | paragraph deleted in the rewrite |
| `body.tex` contribution bullet 2 | "What sets its size is the mismatch..." (B3) | bullet replaced |
| `body.tex` `sec:floor_ladder` title | "---it grows" (B4) | **fixed** |
| `body.tex` MMS raster control (B5) | "The pipeline does not manufacture rising residuals" — the control's fine end reverses and supports the opposite | **fixed**, full reversal now reported |
| `body.tex` "What this leaves" (B6) | single-mechanism attribution | **fixed**, two contributions, neither separated |
| `body.tex` conclusion (B8) | "refinement resolves more of that mismatch, not less" | **fixed** |
| `body.tex` `sec:conformal` | "the floor grows under refinement, so a finer grid widens this certificate" | **fixed** |
| `residual_floor_theorem.tex` l.535 (T1/T2) | "It follows from operator provenance: the floor grows under refinement" | **fixed**, decimation result added |
| `residual_floor_theorem.tex` l.393 (T5/T5b/T6) | `$5.6$--$6.8\times$`, `$1.6\times$`, "refinement cannot recover it, because the floor grows", and the causal clause the floor-subtraction gate reversed | **fixed**; T7 exchangeability caveat added |

This mattered: `abstract.tex` already said `6.5`--`7.9x` while the theorem section said
`5.6`--`6.8x`. **`audit_paper_numbers.py` cannot catch that**, because it compares JSON
to a hardcoded constant in the script, never to the `.tex`. A paper can pass "every
checked number matches its source file" while contradicting itself.

`mechanism_decision.md` B7 (the `h/s` appendix sentence) was **not** applied — it adds
length to a section this pass was demoting. Flagging it as still open.

## 5. Every new number and its source

| claim | source |
|---|---|
| interp `mse_u/v/p` 0.782 / 0.0336 / 75.03, surf 10989, `mse_nut` 8.84e-8, `C_l` 1.18%, `C_d` 2.39% | `results/interpolation/interp_full.json` → `variants.nd.test_metrics` |
| Transolver 0.120 / 0.088 / 628.5 / 9110 / 5.73e-9, `C_l` 5.81%, `C_d` 8.99% | `results/baselines/table2.csv` (verified line by line) |
| ratios 8.4x, 2.6x, 6.5x, 15x | recomputed in the auditor from the two above |
| standardised means 1.16e-3 / 2.22e-3 (uvp) and 1.76e-2 / 2.75e-3 (uvpnut) → 1.9x and 6.4x | `interp_full.json` → `variants.nd.standardised_metrics` |
| floor ladder rows (8-NN, 1-NN, train mean, freestream) | `interp_full.json` → `variants.nd.ladder` |
| permuted-parameter control 22.82 / 14.36 / 129622, `C_l` 365% | `interp_band_control_full.json` → `C_permuted_parameters` |
| band shares 0.924 / 0.898 / 0.535 and R²/R²pc | `interp_band_control_full.json` → `A_band_decomposition` |
| n=100 `mse_p` 238 ± 65, worst-of-5 344; n=50 519 / 735 | `interp_band_control_full.json` → `B_train_size_ladder` |
| feature ablation 9491 / 7507 / 75.0 / 93.4 (126x) | `interp_band_control_full.json` → `D_feature_ablation` |
| nn-distance 0.189 / 0.440; excl. closest 10% → 82.1 | `interp_full.json` → `nn_distance`, `excluding_closest_10pct` |
| p-fill control 75.0 → 154.7, surf 10989 → 153174 | `results/interpolation/interp_full_pfill.json` |
| interp ρ_Cd 0.8389 vs exact-field 0.8394; ρ_Cl 0.8758 vs 0.8757 | `interp_full.json` → `official_forces` |
| `reynolds` U containment 0.0; 0.808 / 74.1; raw 4.194 / 751.1 (5.2x / 10.1x) | `results/interpolation/interp_reynolds.json` |
| `aoa` `C_l` 23.06%, `C_d` 2.71% | `results/interpolation/interp_aoa.json` |
| null: lift 0.9821 [0.9737, 0.9866], drag 0.9318 [0.8981, 0.9531]; (U,α) 0.9344 | `results/review/covariate_null_trainfit.json` |
| in-sample optimism 0.0008–0.0035; 0.0017 lift / 0.0020 drag at full features | `results/review/covariate_null.json` |
| published lift 0.913/0.938/0.965/0.967, drag −0.117/−0.022/−0.303/−0.138, Transolver 0.9978 | `covariate_null_trainfit.json` `leaderboard_vs_full_name_null`, cross-checked against `field_scan_sept2026.md` §1.2 (verified from the NeurIPS PDF) |
| decimation: 16/16, q = +0.55 ± 0.27 | `results/certificates/floor_cloud_decimation.json` |
| floor-subtraction: 7.4x → 11.3x, 3/3; Spearman 0.616/0.592/0.626 → 0.681/0.703/0.601; 17.2x uninformative | `results/review/floor_subtraction_gate.json` |

**41 new rows were added to `scripts/audit_paper_numbers.py`** covering all of the
above except the p-fill control, the nn-distance and the decimation exponent. Ratios
are recomputed inside the readers rather than hardcoded, so a re-run that moves either
side of a ratio is caught.

## 6. Three numbers from the source reports that I did NOT use, and why

1. **"reaches Transolver's `mse_p` from 50 training cases"** (`interpolation_baseline.md`
   §8 ready-made paragraph). §3 and §7 of the same report say **100**: at 50 the mean
   clears 628.5 (519) but one of five subsets does not (735). The paper says 100.
   §8's paragraph should not be pasted verbatim.
2. **"within one std of the `tab:v2` Transolver run, 10794 ± 910"** (surface pressure).
   That figure is from a 3-seed v2 run and is **not** in any committed artifact I can
   reach; `interp_full.json`'s own reference row and `tab:v2` both give **9843**
   (5 seeds). The paper now says plainly "$10989$ against $9110$, $1.21\times$" — the
   interpolator loses on surface pressure, stated without a cushion.
3. **"in-sample inflation is only 0.0015–0.0035"** (task brief). The JSON's four `cd`
   rows include 0.00085 and 0.00097, both below 0.0015. The paper reports the honest
   full range **0.0008–0.0035** and the two headline-feature-set values (0.0017 lift,
   0.0020 drag).

A fourth wording decision: the abstract says "a Transolver trained on the same data",
not "a matched-budget Transolver". "Matched budget" is true in the sense
`tab:transolver` uses it (same data, 80 epochs, same scorer) but a referee reading it
next to `0 params (210 MB)` will read it as a capacity claim and object. The cost
asymmetry is conceded in `sec:interp` and in Limitations; the abstract simply avoids the
word rather than making a claim its concession is forty lines away from. `tab:transolver`'s
caption now glosses the `0 (210MB)` cell in place, because that table sits in a different
section from the paragraph that explains it.

Also **not** written: the report's own title framing, "the baseline nobody publishes."
`field_scan_sept2026.md` §1.2(f) records that as **false as stated** — DrivAerNet++
publishes an AutoML tabular baseline on its 26 design parameters. The manuscript states
the scoped three-part version (R²-on-magnitude not rank; the same group's follow-on
benchmark drops it; nobody audits published numbers against it) in Related Work and in
`sec:covariate`.

## 7. Commands the next process MUST run — none of these were run here

```
cd docs/paper
pdflatex neuroforge_cfd.tex && pdflatex neuroforge_cfd.tex
pdflatex neuroforge_cfd_elsevier.tex && pdflatex neuroforge_cfd_elsevier.tex
# both must end: 0 overfull, 0 underfull, 0 undefined, 0 errors
cd ../..
./.venv/Scripts/python.exe scripts/audit_paper_numbers.py   # must end "every checked number matches its source file"
./.venv/Scripts/python.exe scripts/check_submission.py      # must end "all mechanical requirements satisfied"
```

Files to stage, by name (never `git add -A`):

```
docs/paper/body.tex
docs/paper/abstract.tex
docs/paper/preamble.tex
docs/paper/neuroforge_cfd.tex
docs/paper/neuroforge_cfd_elsevier.tex
docs/paper/sections/residual_floor_theorem.tex
docs/paper/submission/highlights.txt
docs/paper/review/restructure.md
scripts/audit_paper_numbers.py
```

Do **not** stage `docs/paper/*.pdf`, `*.aux`, `*.log`, `*.out`, `*.bbl`, `*.blg`,
`*.spl` unless the project already tracks them and the rebuild is intended.

## 8. What I could not verify

- **Neither build was run.** Specific risks, in descending order:
  - Three new tables. Column counts were checked by hand (8 / 7 / 8 / 3) and every row
    matched. They use `\begin{adjustbox}{max width=\textwidth}`, the preamble's own
    recommended idiom, rather than `\resizebox` (which enlarges a table that already
    fits). The elsarticle measure is the narrower of the two, so if they fit there they
    fit in TMLR.
  - `\subsubsection` is used for the first time in this manuscript. In `article` and
    `elsarticle` it is inside the default `secnumdepth` of 3, so it is numbered and
    `\autoref` resolves to "subsubsection"; hyperref defines
    `\subsubsectionautorefname`. No new package is needed.
  - `<` and `>` in `\texttt{}` were avoided: the preamble does **not** load
    `fontenc[T1]`, so those characters would render as inverted punctuation under OT1.
    The AirfRANS filename is now written without angle brackets.
  - No new `\cite` key was introduced, so the existing `.bbl` files remain complete and
    no `bibtex` run is required for a clean build. See the follow-up below.
- **Neither script was run.** The abstract is **242 words** by hand-simulation of
  `check_submission.count_words` (limit 250); the five highlights are 75–80 characters
  (limit 85). The 41 new auditor rows were each checked by hand against the JSON to a
  tighter margin than the stated tolerance, with one caveat now written into the
  script: `interp_full.json`'s `reference_rows` block stores `mse_u/v/p` **rounded**
  (0.12, 0.088, 628.5) and `mse_nut` at **full precision**, so ratios computed against
  it carry up to 0.15% of rounding slop. The `nu_t` row's tolerance was widened to 1.0
  for that reason. The script itself has not executed —
  in particular the helper `interp_containment` assumes the key path
  `feature_containment.per_feature_frac_test_inside_train_range.U`, which I confirmed
  by reading `interp_reynolds.json` but not by running.
- **Nothing was committed.**

## 9. Follow-ups this pass deliberately left open

1. **DD-RNO is not cited.** `field_scan_sept2026.md` §1.2(d) verifies arXiv:2608.13490
   directly from the PDF and records the strongest available citation for "the field
   knows the confound exists and still does not report the baseline" — they run the
   right partial correlation on a sub-term and never on the headline. Adding it needs a
   new `refs.bib` entry and a `bibtex` run, which this session could not do without
   risking undefined citations in the build. **Add it before submission**, with the
   memory note's constraint intact: the defensible claim is the measurement gap, never
   label leakage.
2. **The point-space head-to-head.** Now stated as the first Limitations item and named
   as the single most valuable follow-up. It needs Transolver inference with the
   `PointNormalizer` fitted on the 800 train point clouds; the checkpoints do not store
   the normaliser.
3. **Matched surrogate rows on `reynolds`/`aoa`.** Without them `sec:splits` can only
   report the interpolator's own degradation, which the section says in its own voice.
4. **The covariate null on the other three benchmarks** (DrivAerNet++, DrivAerML,
   AhmedML) and refit inside the `scarce`/`reynolds`/`aoa` splits. Both scoped out in
   Limitations.
5. **`mechanism_decision.md` B7** (the `h/s` ladder sentence) remains unapplied.
5b. **`docs/paper/review/interpolation_baseline.md` §8 should be corrected or marked
   superseded.** Its ready-to-paste paragraph still says "from $50$ training cases"
   (§3 and §7 of the same report say 100) and still cites "10794 ± 910" for the
   `tab:v2` surface pressure (the committed value is 9843). Neither reached the
   manuscript, but §8 is written to be pasted, and this project's pattern is that a
   later pass pastes from it. Fix it at the source so the error cannot recur.
5c. **Cosmetic, `tab:interp`:** the interpolation row bolds both the model name and four
   cells, which is heavier than `tab:transolver`'s no-bold convention. Left as-is
   because it marks the row a reader is meant to find; trim if the author prefers
   consistency.
6. **Figures still carry the old spine.** `fig:fig1`–`fig:fig6`, `fig:risk_coverage`,
   `fig:deepcfd`, `fig:cylinder` are all residual/trust-layer figures; there is no
   figure for findings 1–3. A wall-band error-share plot and a null-vs-leaderboard
   scatter would be the two highest-value additions and neither exists.
7. **`docs/paper/submission/cover_letter.md` and `suggested_reviewers.md`** still argue
   the old thesis and were not touched.
8. **`docs/paper/submission/arxiv_v4/stage/`** is a frozen shipped snapshot and was
   deliberately left alone; it will be regenerated.
