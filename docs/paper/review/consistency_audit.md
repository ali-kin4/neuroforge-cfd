# Consistency audit — full manuscript, start to finish

Date: 2026-09-07. Branch `paper1/reframe-after-jcp`. Auditor pass over
`docs/paper/body.tex` (2244 lines), `docs/paper/abstract.tex`,
`docs/paper/sections/residual_floor_theorem.tex` (591 lines) and the front matter of
`docs/paper/neuroforge_cfd_elsevier.tex`, cross-checked against the committed JSON/CSV
under `results/` and the review ledger in `docs/paper/review/`.

**No file was edited. This is a report only.**

---

## Verdict

**The manuscript is not currently internally consistent, but the damage is localised and
every defect is cheap to fix.** The reframe landed correctly in the places that were
rewritten today — §Method's per-cell scope paragraph, §sec:iters' three limits, §sec:v2's
W1 and MeshGraphNet disclosures, the theorem's inversion counterweight, `tab:positioning`
— and the underlying arithmetic is in unusually good shape: I recomputed the floor, the
ladder, both descent studies, the gate, the fixed-step control, the decomposition, the
v2 table, the inversion probe, the per-cell localisation, the conformal certificates and
the selective-prediction controls directly from their artifacts, and all but one of them
match the prose. The failures are of a single kind: **eight withdrawn or narrowed claims
are still asserted at sites that today's rewrites did not reach**, in every case
contradicted by a passage elsewhere in the same manuscript. Two of those sites are the
figure caption the paper calls "the money figure for the headline" and a limitations
bullet that contradicts another limitations bullet four items later. On top of that there
is one genuine numerical error — the gate's certificate percentages are swapped between
the deployed and ensemble paths, and the repository's own number-auditor is mis-wired in
the same direction so it certifies the error — one broken cross-reference that misdirects
all seven citations of the paper's most important concession, one misattributed
convergence order, and a front matter still addressed to the journal that desk-rejected
it. Separately, and in the author's own framing a defect: the abstract under-claims the
two strongest results relative to the conclusion.

**Counts: 13 blocking, 17 should-fix, 6 cosmetic.**

---

# BLOCKING

## B1. The gate's certificate percentages are swapped between arms

**File / lines:** `docs/paper/body.tex:162-165` and `docs/paper/body.tex:1292-1296`.

**Site count: exactly two.** A plain grep for `8\.8` across `body.tex abstract.tex
sections/*.tex` returns `body.tex:163` and `body.tex:1293` and nothing else. Both must change.

**Text as it stands (1292-1296):**

> The ungated half step raises the monitored residual on $8.8\%$
> of deployed cases ($1.0\%$ on the ensemble path), and the gate makes that outcome
> impossible by construction rather than improbable in practice.

and at 163-165:

> an ungated step raises the monitored residual on $8.8\%$ of deployed cases ($1.0\%$ on
> the ensemble path), and the gate makes that impossible by construction

**What is wrong:** the two numbers are attached to the wrong arms.

**Source that proves it wrong:** `results/review/control3_fixed_step.json`. Its own
metadata block assigns the arms:

```
/meta/arms/seed0 = ensemble          /meta/arms/backbone_seed0 = deployed
/meta/arms/seed1 = ensemble          /meta/arms/backbone_seed1 = deployed
/meta/arms/seed2 = ensemble          /meta/arms/backbone_seed2 = deployed
```

and `summary/<arm>/certificate_at_fixed_0.5/frac_residual_increases` gives

| arm | per seed | mean |
|---|---|---|
| deployed (`backbone_seed0/1/2`) | 0.025 / 0.000 / 0.005 | **0.010** |
| ensemble (`seed0/1/2`) | 0.105 / 0.065 / 0.095 | **0.088** |

The labelling is corroborated three independent ways: (i) `scripts/control_fixed_step.py`
lines 4-8 attribute "99.8% of deployed cases … 89.3% … median 5.8%" to "``backbone_*``
arms"; (ii) every *other* number the paper draws from this file matches that assignment —
deployed accept rate 599/600 = 99.83% (paper: 99.8%), deployed gate improves error
93.5/95.5/78.9% → 89.2% (paper: 89.3%), deployed median −5.74% (paper: 5.8%), deployed
ungated fixed half step 99.0/98.5/90.0% → 95.83% (paper: 95.8%), ensemble gate 85.5/78.5/56.0%
→ 73.3% and median −2.26% (paper: 74.3% and 2.6%); (iii) `results/control/acceptance_gate.json`
gives the same split. Only this one line is inverted.
`docs/paper/review/RESOLUTION.md:102-104` has it the correct way round.

**Precise replacement (1292-1296):**

> The ungated half step raises the monitored residual on $1.0\%$ of deployed cases
> ($8.8\%$ on the ensemble-mean path), and the gate makes that outcome impossible by
> construction rather than improbable in practice.

**and the argument around it must be re-weighted, not just the digits.** As written, the
paragraph uses 8.8% to justify "what the gate uniquely buys is the certificate" on the
deployed path. The true deployed figure is one case in a hundred. Suggested replacement
for the surrounding sentence:

> What the gate uniquely buys is the \textbf{certificate}. On the deployed path an ungated
> half step already almost never violates the monotone-residual property ($1.0\%$ of
> cases), so there the gate converts a near-certainty into a guarantee; on the
> ensemble-mean path, where an ungated step raises the residual on $8.8\%$ of cases, the
> guarantee is doing visible work. Either way it is a guarantee and not an accuracy claim,
> and it is the only thing we claim for it.

Mirror the same correction at 163-165.

## B2. The repository's own number-auditor is wired to the manuscript, not to the artifact

**File / lines:** `scripts/audit_paper_numbers.py:207-223` and `:327-330`.

**Text as it stands:**

```python
def c3_ungated(root, which):
    """...an earlier draft had the two numbers the wrong way round."""
    ...
        (bb if k.startswith("backbone") else dep).append(f)
    vals = bb if which == "ensemble" else dep
```

**What is wrong:** the reader sends `backbone_*` keys to `bb` → "ensemble" and everything
else to `dep` → "deployed", exactly inverting `control3_fixed_step.json`'s `meta.arms`.
The rows at :327-330 then assert `8.8` for "DEPLOYED" and `1.0` for "ensemble" and pass.
Someone corrected an earlier inversion in the wrong direction.

**Consequence for the audit trail:** `scripts/audit_paper_numbers.py --verbose` currently
prints "every checked number matches its source file" over 36 rows. Thirty-five of those
rows are sound — I re-ran it and separately reproduced the floor, ladder, descent,
decomposition, C1/C2/C4 and conformal-width rows by hand. The C3 row is not evidence of
anything, and a green run on it should not be cited in `docs/REPRODUCE.md` or a rebuttal.

**Fix:** swap the two branch assignments so `backbone_*` → deployed, and update the
expected values at :327-330 to `1.0` (deployed) and `8.8` (ensemble) *after* B1 is applied
to the manuscript. Add a row that asserts against `meta.arms` directly so the mapping
cannot silently invert again.

## B3. The theorem states the deployed prediction sits below the floor — a claim withdrawn today

**File / lines:** `docs/paper/sections/residual_floor_theorem.tex:392-394`.

**Text as it stands:**

> the sublevel sets are unbounded in the directions of \autoref{prop:kernel}, and the
> floor sits \emph{above} the operating point (mean $\|r^\star\|=0.192$ versus
> $\|R_h(\hat u)\|=0.114$).

**What is wrong:** $0.114$ is the dropout-FNO's mean monitored residual, not the deployed
model's. On the deployed Transolver the operating point is *above* the floor and the
sentence's factual premise reverses.

**Source:** `results/certificates/transolver_inversion.json`, whose top-level `verdict` is
literally `INVERSION_DISAPPEARS`.

| field | mean $\|R_h\|$ | vs floor 0.192 |
|---|---|---|
| truth floor | 0.1920 | — |
| dropout-FNO (`dropout_fno/norm_mean`) | 0.1136 | below |
| Transolver seed0/1/2 backbone alone | 0.2116 / 0.2119 / 0.2120 | **above** |
| Transolver+DEQ deployed, seed0/1/2 | 0.2149 / 0.2124 / 0.2126 | **above** |

Contradicted by this very section 40 lines later (`:433-441`), which reports the deployed
inversion rate as 2.5–7.0% with the median gap reversing sign to $+0.013$–$+0.015$.

**Precise replacement:**

> the sublevel sets are unbounded in the directions of \autoref{prop:kernel}, and on the
> backbone where the floor sits \emph{above} the operating point (the dropout-FNO: mean
> $\|r^\star\|=0.192$ versus $\|R_h(\hat u)\|=0.114$) a threshold below $\|r^\star\|$
> rejects the truth before it rejects the prediction. On the deployed Transolver, whose
> operating point is $0.212$--$0.215$ and therefore \emph{above} the floor, certification
> still fails, for the leg-(i) reason alone: $u^\star$ is rejected at every threshold below
> $\|r^\star\|$, and no threshold above it certifies anything.

## B4. The theorem's own reconciliation paragraph repeats the withdrawn 80%

**File / lines:** `docs/paper/sections/residual_floor_theorem.tex:408-410`.

**Text as it stands:**

> If the deployed prediction sits \emph{below} the floor in $80\%$ of cases, it sits in the
> regime the decomposition calls floor-dominated---yet $\rho=0.61$ is measured there.

**What is wrong:** "the deployed prediction" is exactly the attribution the paper withdrew.
80% (160/200) is `dropout_fno/frac_below_truth_floor`; the deployed Transolver is 2.5–7.0%.
Also note $\rho=0.61$ is measured on the *Transolver*, so the paragraph's apparent paradox
is constructed by mixing two backbones' quantities into one sentence.

**Source:** `results/certificates/transolver_inversion.json`
(`dropout_fno/frac_below_truth_floor = 0.8`; `transolver/seed*/…/frac_below_truth_floor` =
0.040/0.025/0.025 and 0.060/0.045/0.070); `results/control/w1_capture.json` for
$\rho = 0.611 \pm 0.054$.

**Precise replacement:**

> On the dropout-FNO, whose prediction sits \emph{below} the floor in $80\%$ of cases, the
> prediction is in the regime the decomposition calls floor-dominated---yet a positive
> detector correlation is measured on both that backbone and the deployed one
> ($\rho=0.61$ on the Transolver), whose predictions sit \emph{above} the floor on more
> than $93\%$ of cases. The two are statements about different things.

## B5. Six surviving assertions that the residual tells you *where* a prediction is wrong

**Files / lines:** `body.tex:26-27`, `body.tex:345-346`, `body.tex:420-422`,
`body.tex:474-475` (the `fig:pipeline` role tag), `body.tex:931-932`, `body.tex:2093-2095`.

**Text as it stands (representative, 26-27):**

> Put plainly, the residual tells you \emph{where} a prediction is wrong, not \emph{how} to
> fix it.

and 345-346:

> Our answer---signal yes (it tells you \emph{where} the prediction is wrong), objective
> no, gate yes

and 474-475 (TikZ role tag, rendered inside the pipeline figure):

> detects \emph{where} the field is wrong; drives calibration --- reliable, backbone-robust

**What is wrong:** spatial localisation is the one thing the paper measured and did not get.
The validated claim is *case-level* ranking plus *moderate region-level* location at
8–16-cell granularity.

**Source:** `body.tex:570-591` says so explicitly in the paper's own voice — "The honest
scope is therefore: the residual is a \emph{case-level} ranker and a \emph{moderate
region-level} locator at $\sim$8--16-cell granularity, not a per-cell error rank" — backed by
`results/control/percell_localization.json` (`V0_raw/percase_spearman_mean = 0.166 ± 0.155`;
best variant `V2_patch_k16` = `0.323 ± 0.232`, patch AUROC 0.684–0.717) and
`results/control/percell_residual_error.json` (`per_cell_spearman_mean = 0.2196 ± 0.0602`,
verdict `PER-CELL-WEAK-REFRAME-TO-CASE-LEVEL`). `body.tex:1195-1203` independently refutes
the spatial reading on the cylinder control. The correct contrast word is **which**, not
**where**.

**Precise replacements:**

- 26-27: `the residual tells you \emph{which} predictions are wrong, not \emph{how} to fix them.`
- 345-346: `signal yes (it tells you \emph{which} predictions to distrust), objective no, gate yes`
- 420-422: `(reliable, backbone-robust---it tells you \emph{which} predictions to distrust)`
- 474-475: `flags \emph{which} predictions to distrust; drives calibration --- reliable, backbone-robust ($\S$5)`
- 931-932: `it tells you \emph{which} predictions are wrong (a trust signal), not \emph{how} to fix them.`
- 2093-2095: `The residual's demonstrated value is as a case-level trust signal (\emph{which} predictions to distrust), not as a correction input (\emph{how} to fix them)`

Also check `fig:pipeline`'s own caption at `body.tex:494-500`; it says "reliable *trust
signal*" without the locative and is fine as is, but the figure body at 474-475 is not.

## B6. `fig:fig4`'s caption still asserts the withdrawn divergence and calls it the headline

**File / lines:** `docs/paper/body.tex:1940-1943`.

**Text as it stands:**

> \caption{The money figure for the headline: PDE residual \emph{rises} while field error
> \emph{falls} across correction iterations (detector $\ne$ fixer).}

**What is wrong:** two withdrawals in one sentence. (i) The sweep is no longer the headline
— `body.tex:1208-1212` says "The direct evidence … is the descent experiment of
\autoref{sec:descent} … We keep it because it is what first prompted the question, not
because it settles it." (ii) The rise-while-error-falls reading is the divergence claim
the seeded re-run does not support.

**Source:** `results/sensitivity/iters_seeded.json`, `verdict = DIRECTION-ONLY`, with
`n_residual_monotone_nondecreasing = 0` (of 5 seeds),
`paired_residual_up_at_best_k = 45 / paired_n = 120` (37.5%, below chance),
`residual_delta_snr_vs_seed_sd = 1.323` against a pre-declared bar of 2. Also
`body.tex:1222-1228`, which shows `mse_u` and the residual moving in the *same* direction on
three of the five intervals of `tab:iters` itself.

**Precise replacement:**

> \caption{The iteration sweep that first prompted the question (\autoref{tab:iters}, one
> unseeded checkpoint). Over iterations $0\!\to\!3$ the PDE residual rises while
> $\texttt{mse\_u}$ falls; beyond iteration $3$ the two move together. A seeded five-seed
> re-run returns \textsc{direction-only}, so this figure illustrates \emph{decoupling}
> between the residual and the error, not divergence. The claim itself rests on
> \autoref{sec:descent}.}

## B7. Two limitations bullets contradict each other inside the same list

**File / lines:** `docs/paper/body.tex:2078-2080` versus `docs/paper/body.tex:2125-2139`.

**Text as it stands (2078-2080):**

> Used as a correction \emph{objective} the residual fails: more iterations raise the PDE
> residual while error falls (\autoref{sec:iters}).

**Text as it stands (2125-2139), four items later:**

> \textbf{The iteration sweep supports decoupling, not divergence.} … We therefore claim
> only that cutting field error by $9.4\%$ buys \emph{no} residual reduction … and we do
> not claim the two quantities move in opposite directions.

**What is wrong:** the first bullet asserts precisely what the fourth bullet says the paper
does not claim, and cites the same section for it. A referee reading the limitations list
straight through hits both within one page.

**Source:** `results/sensitivity/iters_seeded.json` (as B6).

**Precise replacement (2078-2080):**

> Used as a correction \emph{objective} the residual fails: descending it from the exact
> ground truth drives the field error up on $200/200$ cases and on $94$--$98\%$ of cases
> from the deployed backbone (\autoref{sec:descent}), and cutting field error along the
> corrector's own iteration path buys no residual reduction (\autoref{sec:iters}).

## B8. Related Work asserts the withdrawn "residual rises as error falls"

**File / lines:** `docs/paper/body.tex:242-245`.

**Text as it stands:**

> As a correction \emph{objective}---where a step must reduce the residual---residual
> minimisation does not track error minimisation on a real RANS benchmark
> (the residual rises as error falls).

**What is wrong:** two errors. (i) The parenthetical is the divergence claim, withdrawn.
(ii) The sentence attributes it to "residual minimisation", but the evidence for
divergence was the sweep, which minimises nothing — that conflation is the exact
inference `body.tex:77-80` says a reviewer would rightly have objected to, so the paper
commits it in Related Work while apologising for it in the Introduction.

**Source:** `results/sensitivity/iters_seeded.json`; `results/residual_descent/descent_truth_bc_uvp_armijo.json`
(`frac_error_increased = 1.0`, `J_ratio_median = 0.398`) is the correct evidence.

**Precise replacement:**

> As a correction \emph{objective}---where a step must reduce the residual---residual
> minimisation does not track error minimisation on a real RANS benchmark: descending
> $J=\tfrac12\|R_h\|^2$ from the exact ground truth cuts the residual to $39.8\%$ of its
> starting value while driving the field error up on $200/200$ cases, and on
> $94$--$98\%$ of cases started from the deployed backbone.

## B9. `\label{sec:selective}` is attached to a `\paragraph`; all seven references resolve to the wrong subsection

**File / lines:** `docs/paper/body.tex:1693-1694`.

**Text as it stands:**

```latex
\paragraph{Does the physics earn its place?}
\label{sec:selective}
```

**What is wrong:** `\paragraph` is unnumbered in both `elsarticle` and `tmlr.sty`, so the
label binds to the last stepped counter — the enclosing subsection `sec:conformal`.

**Source:** `docs/paper/neuroforge_cfd_elsevier.aux`:

```
\newlabel{sec:conformal}{{5.9}{35}{The conformal trust layer: coverage and contraction}{subsection.5.9}{}}
\newlabel{sec:selective}{{5.9}{38}{Does the physics earn its place?}{section*.62}{}}
```

Both carry the number **5.9**, and the anchor type differs, so `\autoref` renders them with
different names. Confirmed in the built PDF (`pdftotext neuroforge_cfd_elsevier.pdf`):

> … subsection 5.2 (detector across backbones, and the learned correction), **section 5.9**
> (where the physics does and does not beat a physics-free score) and **subsection 5.9**
> (calibrated bands).

That is the second paragraph of "How to read this paper" (`body.tex:191-192`) offering the
reader the same number twice as two different destinations, and there is no "section 5.9"
in the manuscript. The seven affected call sites are `body.tex:67, 153, 191, 393` and
`sections/residual_floor_theorem.tex:58, 398, 417` — including the theorem's three-verb
spine paragraph, where "**Rank: yes** … (section 5.9)" points at the conformal subsection.

**Precise replacement:** promote it to a real subsection so the paper's most-cited
concession has its own number:

```latex
\subsection{Does the physics earn its place?}
\label{sec:selective}
```

If that is undesirable structurally, the minimum fix is to move `\label{sec:selective}` to
sit immediately after the `\subsection{The conformal trust layer…}` line and rename the
references — but then the roadmap sentence at 191-192 must be rewritten, because it would
be pointing at the same subsection twice.

**This is a closed class, not a sample.** `grep "newlabel" neuroforge_cfd_elsevier.aux |
grep "section\*\."` returns exactly one hit, `sec:selective`, and the same command on
`neuroforge_cfd.aux` returns the same single hit. Every other label in the manuscript binds
to a `subsection.*`, `section.*`, `table.*`, `figure.*`, `equation.*`, `theorem.*` or
`proposition.*` anchor. Fixing this one closes the category.

## B10. The theorem attributes the analytic MMS convergence order to the rasterised control

**File / lines:** `docs/paper/sections/residual_floor_theorem.tex:460-462`.

**Text as it stands:**

> while a manufactured potential flow pushed through the identical rasterisation converges
> at order $2.02\pm0.05$ and a $C^1$ re-rasterisation \emph{raises} rather than lowers the
> floor

**What is wrong:** the number belongs to a different control. Two MMS series exist in the
same artifact and they behave completely differently; the rasterised one does not converge
at second order.

**Source:** `results/certificates/floor_resolution_decomposition.json`:

| series | order $p$ | fine/coarse |
|---|---|---|
| `gates/mms_order/p_band_0.1` — MMS sampled **directly on the grid** | +2.058 | 0.061 |
| `aggregate/mms_raster_band_0.1` — MMS **pushed through the rasteriser** | **+0.32 ± 0.27** | **0.65** |

`docs/paper/review/floor_resolution_study.md:293` labels the two rows "MMS analytic
(truncation) … +2.02 ± 0.05" and "MMS raster (pipeline null) … +0.32 ± 0.27". The error was
inherited from that study doc's own proposed paper text at its line 426, and reproduced
here. Note `body.tex:1380-1382` describes the rasterised control **correctly**, as a ratio
("fine/coarse $0.41$--$0.74$"), and never as an order — so the manuscript now says two
different things about the same control.

**Precise replacement:**

> while a manufactured analytic solution converges on the same monitor at order
> $2.02\pm0.05$ and, pushed through the identical rasteriser, still \emph{falls} with
> refinement (fine/coarse $0.65$, rising in only $3/16$ cases) where the real truth's
> rises; and a $C^1$ re-rasterisation \emph{raises} rather than lowers the floor

## B11. The front matter is still addressed to the journal that desk-rejected the paper

**File / lines:** `docs/paper/neuroforge_cfd_elsevier.tex:34` and `docs/paper/body.tex:313-314`.

**Text as it stands:**

```latex
\journal{Journal of Computational Physics}
```

and body.tex:313-314:

> \citep{ma2024uqno}. Within this journal the same combination is being pursued from both
> the PINN and the operator side: \citet{yu2026conformalpinn} … and \citet{garg2025dfuq}

**What is wrong:** `\journal{}` prints on the elsarticle preprint title page. The target
venue is **Computers & Fluids**; JCP is out.

**Source:** `docs/paper/review/venue_plan.md:12` (JCP: "desk, no review … will not
reconsider"), `:278` ("**Journal of Computational Physics** | OUT"), `:60` and `:402`
(Computers & Fluids is the primary recommendation), and `scripts/check_submission.py`,
which is written against the Computers & Fluids guide. `refs.bib` confirms both
`yu2026conformalpinn` and `garg2025dfuq` carry `journal = {Journal of Computational
Physics}`, so "Within this journal" is a JCP-flattery sentence that becomes false on
submission elsewhere.

**Precise replacements:**

- `neuroforge_cfd_elsevier.tex:34`: `\journal{Computers \& Fluids}`
- `body.tex:314`: replace `Within this journal the same combination is being pursued` with
  `The same combination is being pursued` (and delete nothing else — the two citations stay).
- Also update the stale build comments that name JCP: `body.tex:3`, `abstract.tex:3`,
  `neuroforge_cfd_elsevier.tex:2-3`, `preamble.tex:4`. These are comments and do not print,
  but they are the reason the `\journal{}` line survived.

## B12. The withdrawn "cannot certify" claim survives at heading level, twice

**Files / lines:** `docs/paper/sections/residual_floor_theorem.tex:387` (a `\subsection`
title, which also appears in the table of contents) and `:389` (a `\paragraph` title).

**Text as it stands:**

> `\subsection{Ranking survives; certification does not}`
>
> `\paragraph{Why the floor defeats certification but not triage.}`

**What is wrong:** the paper no longer claims the monitor cannot certify. It claims the
monitor certifies to a floor-limited *width* — a valid distribution-free certificate that
is simply too wide to be useful. Both titles assert the withdrawn absolute.

**Source that proves it wrong:** the same section's own opening, 340 lines earlier at
`:46-53`, in the paper's voice —

> \textbf{Certify: only loosely} … A calibrated bound is still available by conformalising
> the score against held-out data, and we report one; what the floor destroys is its
> tightness

— and its own closing line at `:591` ("certifies them only at a width the floor rather than
the model sets"); and `body.tex:1630-1638`, which is explicit that the guarantee is intact:

> coverage comes out at $0.891$--$0.895$ across three seeds---so the guarantee holds, and
> the floor does \emph{not} invalidate split conformal, which needs only exchangeability and
> never supposed the score vanished at the truth. What the floor destroys is tightness.

Measured: `results/review/functional_audit_gate_followup.json`, `coverage_mean` = 0.891025 /
0.894525 / 0.890975 at a 0.90 target over 400 half-splits, at `bound_over_error` = 5.568 /
6.726 / 6.763.

A reader who scans the table of contents sees "certification does not [survive]"; a reader
who reads §sec:conformal sees a valid certificate. This is the one withdrawal that survived
into a heading, which makes it the most visible of the eight.

**Precise replacements:**

- `:387` → `\subsection{Ranking survives; certification survives only at a floor-limited width}`
- `:389` → `\paragraph{Why the floor prices certification but does not defeat triage.}`

The body of that paragraph (`:389-406`) also opens "Certification asks for a map from
$\|R_h(\hat u)\|$ to a bound on $\|\hat u-u^\star\|$. It fails here…". That sentence is
defensible as written, because it is scoped to a *residual-threshold* certificate rather
than a conformal one, and the paragraph closes by saying so ("the guarantee we do ship for
the error bound is the distribution-free conformal one, not a residual bound"). Add four
words to remove the ambiguity: `Certification \emph{from the residual norm alone} asks for a
map … It fails here…`.

## B13. The three-verb spine is stated four different ways, and the abstract states only one verb

**Files / lines:** `abstract.tex` (whole), `body.tex:58-71`, `body.tex:344-346`,
`body.tex:1991-2014`, `sections/residual_floor_theorem.tex:46-58`.

**What is wrong:** the paper's thesis is that the monitor can **rank**, can **certify only
to a floor-limited width**, and cannot be **descended**. Five sites state that thesis and no
two of them use the same verbs, the same count, or the same scope. Laid out:

| site | verbs used | count | rank verdict | certify verdict | descend verdict |
|---|---|---|---|---|---|
| `abstract.tex` | — | **0** | **absent** | **absent** | stated (24-case arm) |
| `body.tex:58-71` "Self-auditing, defined" | trust score / calibrated band / accept-reject decision | 3 (+1 named "the fourth role") | (i) holds, matched by physics-free | (ii) "holds outright" — **no width qualifier** | (iv) the diagnosis |
| `body.tex:344-346` Positioning | signal / objective / gate | 3 | signal yes, "tells you \emph{where}" (B5) | **absent** | objective no |
| `body.tex:1991-2014` H1–H5 | H1 accuracy / H2 signal / H3 DEQ / H4 coverage / H5 contraction | 5 | H2 "supported, robustly" (S2) | H4 "supported" — **no width qualifier** (S2) | **absent from the list entirely** |
| `theorem:46-58` | rank / certify / descend | 3 | yes | "only loosely", width $5.6$--$6.8\times$ | no |

Only the theorem section states the spine the paper intends. Three consequences:

**(a) The abstract is a pure-negative paper.** It runs floor → not-a-resolution-artifact →
descent-fails → boundary-is-the-operator, and never says what survives. A referee reading
only the abstract — which after two originality desk rejections is the likeliest referee —
sees no ranking result, no calibrated certificate, and no positive contribution at all. The
conclusion (`body.tex:2184-2195`), the introduction (`:58-71`), the positioning (`:344-346`)
and the theorem (`:46-58`) all carry a "what survives" verdict. This is simultaneously the
paper's largest spine divergence and its largest under-claim.

**(b) H1–H5 has no descent row.** The registered hypothesis set predates the descent
experiment, so the paper's headline negative — the one the title is built on — is absent
from the section that reads results against pre-registration. That is not a rigor violation
(it was not registered, and the paper says so), but a reader auditing pre-registration
adherence will notice the headline result is not in the ledger.

**(c) The three triads are not translations of each other.** "Gate yes" (`:344-346`) has no
counterpart in the theorem's triad, and "certify" has no counterpart in the positioning
triad. Either would confuse a referee trying to hold the paper's claim set in mind.

**Precise replacements:**

1. **`abstract.tex`** — add one closing sentence carrying the two surviving verbs. Budget is
   handled in S3; this sentence costs ~34 words and the S3 note says what comes out.

   > What survives is narrower and we state it exactly: the monitor still \emph{ranks},
   > flagging worst-decile drag error at AUROC $0.952$, and still \emph{certifies} under
   > split conformal---at valid $0.89$ coverage against a $0.90$ target, but at a width
   > $5.6$--$6.8\times$ the drag error it bounds.

2. **`body.tex:58-71`** — the "(ii) holds outright, and needs no operator at all" clause at
   `:67-68` needs the width qualifier: `(ii) holds outright, and needs no operator at
   all---though what the operator-based score can certify, it certifies only to a width the
   floor sets (\autoref{sec:conformal}).`

3. **`body.tex:344-346`** — after the B5 fix, add the missing verb: `signal yes (it tells you
   \emph{which} predictions to distrust), objective no, gate yes as a guarantee and not as an
   accuracy mechanism, and certification only at a floor-limited width.`

4. **`body.tex:1991-2014`** — append a sixth bullet so the ledger contains the headline:

   > \item \textbf{Not pre-registered: the residual as a correction \emph{objective}.} The
   > registered set predates \autoref{sec:descent}. We record the outcome here for
   > completeness: descending $J=\tfrac12\|R_h\|^2$ raises the field error on $200/200$
   > cases from the truth and on $94$--$98\%$ from the deployed backbone, so had it been
   > registered it would be \emph{falsified}.

5. Adopt the theorem's **rank / certify-to-a-width / descend** verbs as the canonical triad
   and make the other three sites echo them explicitly, keeping their local vocabulary as a
   gloss rather than a substitute.

---

# SHOULD-FIX

## S1. `tab:positioning` — recommendation, and the one cell that is wrong

**File / lines:** `docs/paper/body.tex:361-402`.

**Assessment.** The seven-column checkmark grid the two reviewers objected to **no longer
exists**. `RESOLUTION.md:131-133` records that it was rebuilt, and what is at 361-402 is a
four-row, two-column table keyed on a single discriminating question ("is the residual
being monitored the discrete operator whose root the reference solution is?"), with no
checkmarks, no feature list, no "Object." column and no NeuroForge row. It places the prior
successes on the side where they belong rather than scoring them against a feature
inventory. **Recommendation: keep it. It is now an argument, not a scorecard, and it is one
of the stronger objects in the paper.** No caption rewrite is needed for the originality
objection; the caption already states the boundary rather than a feature claim.

**The remaining defect is under-claiming in the one cell that matters.** Lines 384-388:

> Residual as a correction objective … Fails: the residual-selected iterate is never the
> error-optimal one, in $24/24$ cases, and descent from the exact truth drives error from
> zero (\autoref{sec:descent}).

That cites only the 24-case Adam arm. The paper's strongest evidence for this row is the
$n{=}200$ Armijo arm. **Replacement:**

> Fails: descent from the exact truth drives the field error up on $200/200$ cases under
> Armijo-line-searched gradient descent, and on $94$--$98\%$ of cases started from the
> deployed backbone ($p<10^{-30}$, unchanged under three step rules); the residual-selected
> iterate is never the error-optimal one, in $24/24$ cases (\autoref{sec:descent}).

The task brief also asked that the "Object." column caption reflect that the residual was
tested as an objective by explicit minimisation at $n=200$ under three step rules rather
than by observing a supervised corrector. That column is gone, but the *caption* still
frames the objective row abstractly ("minimising the residual is minimising the error").
Add one clause: `Where it is not, the objective's minimiser is displaced from the truth ---
which we test by explicitly minimising $J=\tfrac12\|R_h\|^2$ at $n=200$ under three step
rules, not by observing a supervised corrector.`

## S2. The H2 and H4 verdict bullets omit the paper's own central qualifiers

**File / lines:** `docs/paper/body.tex:2001-2004` and `:2009-2011`.

**Text as it stands (H2):**

> \textbf{H2 (the residual is a valid trust signal).} \emph{Supported, robustly, in- and
> out-of-distribution.} resid$\leftrightarrow$err $\rho > 0$ for every arm/seed/split…

**What is wrong:** literally correct against the pre-registered falsification condition
($\rho \le 0$), but it omits that a physics-free ensemble score matches it on field error
and a paired bootstrap cannot separate them. Read against §sec:selective this looks like
metric-shopping. Same for H4, which reports coverage without the width the floor costs.

**Sources:** `results/review/control1_physics_vs_physicsfree.json`
(`sigma_vel_minus_residual/delta_auroc = 0.0233`, CI `[-0.0346, +0.0832]`,
`delta_auroc_excludes_zero = False`); `results/review/functional_audit_gate_followup.json`
(`bound_over_error` 5.568 / 6.726 / 6.763, `coverage_mean` 0.891 / 0.895 / 0.891,
`floor_share_of_typical_score = 0.8636`).

**Replacement (H2), add after the existing sentence:**

> The registered condition is met with room to spare. Two qualifiers belong with it and are
> stated in full in \autoref{sec:selective}: on field error a physics-free ensemble $\sigma$
> matches the residual ($\Delta$AUROC $+0.023$, CI $[-0.035,+0.083]$), and the physics wins
> outright only on drag (AUROC $0.952$).

**Replacement (H4), add after the existing parenthesis:**

> Coverage was never in doubt; \emph{width} is what the floor costs. Conformalising the
> residual against $|\Delta C_D|$ stays valid ($0.891$--$0.895$ at a $0.90$ target) at a
> median bound $5.6$--$6.8\times$ the drag error it certifies (\autoref{sec:conformal}).

## S3. Under-claiming: the abstract leads both headline results with their weaker arm

**Files / lines:** `docs/paper/abstract.tex:14-21` versus `docs/paper/body.tex:2159-2166`.

**Text as it stands (abstract, 16-21):**

> The consequence for correction is stated in the form that holds universally: started at
> the exact ground truth, with no network involved, gradient descent cuts the residual by
> $84\%$ while driving field error from zero to the range a trained surrogate starts in, and
> the residual-selected iterate is strictly worse than the error-optimal one in $24/24$ cases

**What is wrong:** this is the $n{=}24$, Adam, synthetic-displacement arm. The conclusion
(2159-2166) leads with the $n{=}200$, Armijo, three-step-rule, twelve-decades arm at
$p<10^{-30}$, which is strictly stronger and is what the paper actually built today. The
abstract's number is not wrong, it is the smaller one. Same pattern on the floor: the
abstract and introduction use "rising in $21/24$" (`floor_resolution_ladder.json`), while
the stronger pre-registered statement — **0 of 16 cases decay**, $p=-0.64\pm0.29$, a second
independent 16-case × 5-rung implementation with its own reproduction gate passing to
$3\times10^{-8}$ relative — appears only at `sections/residual_floor_theorem.tex:458-462`.

**Sources:** `results/residual_descent/descent_truth_bc_uvp_armijo.json`
(`frac_error_increased = 1.0`, `J_ratio_median = 0.3978`, `wilcoxon_p_rel_l2 = 1.44e-34`),
`descent_transolver_seed{0,1,2}_bc_uvp_armijo.json` (0.965 / 0.940 / 0.980),
`results/certificates/floor_resolution_decomposition.json`
(`aggregate/band_0.1: n_decay = 0, n_cases = 16, order_p_mean = -0.639 ± 0.285`,
`gates/reproduction/max_rel_err = 3.27e-08`).

**Replacement (abstract, 16-21).** ⚠ **Word budget — read this before applying.** The
abstract is at **241 of 250** words (`scripts/check_submission.py`). The replacement below
runs ~113 words against the ~78 it replaces (**+35**), the floor change at `:13-14` runs ~30
against ~18 (**+12**), and B13's "what survives" sentence adds ~34. Applied naively the
abstract lands near **322 words, 72 over the limit.** Something must come out. In order of
compressibility, the three cheapest cuts, worth ~76 words together:

- `:14-16` — the cubic-interpolant clause ("a higher-order interpolant \emph{raises} it by
  $6$--$12\%$, so neither is what sets it") is a control, not a result, and is stated in full
  at `body.tex:49-52`, `:116` and `:1376-1379`. **Cut, ~22 words.**
- `:21-23` — the closing "The boundary is the operator, not the problem class: residual
  correction succeeds where the residual is the solver's own." This is stated four more times
  in the body (`:134-142`, `:252-259`, `:1509-1533`, `:2177-2182`) and is the paper's least
  contested claim. **Cut, ~28 words.**
- `:19-21` — "a failure of iterate selection, not of uniform harm, since from a perturbed
  start descent often reduces error before overshooting" compresses to "though from a
  perturbed start it often helps before overshooting". **Saves ~14 words.**
- The nested-clause construction at `:16-17` ("The consequence for correction is stated in the
  form that holds universally") is scaffolding. **Saves ~12 words.**

**After applying, re-run `./.venv/Scripts/python.exe scripts/check_submission.py` and confirm
the abstract is ≤250 words, 7 keywords, 5 highlights ≤85 chars.** Also update
`docs/paper/highlights.txt` bullet 3 ("Descending it from the exact truth cuts it 84% and
drives field error from zero"), which carries the same weaker 24-case arm; at 79/85
characters it has room for "on 200 of 200 cases".

The swap itself:

> The consequence for correction is direct. Started at the exact ground truth, with no
> network anywhere in the loop, line-searched gradient descent on the residual drives the
> field error up on $200/200$ cases and never recovers, ending at $1.57\times$ the entire
> error of the deployed surrogate; started from that surrogate it does the same on
> $94$--$98\%$ of cases across three seeds ($p<10^{-30}$), unchanged under three step rules
> and twelve decades of step size, and monotone in the achieved objective. We do not claim
> descent always harms: from a perturbed start it often reduces error before overshooting.
> What never happens is that the residual selects the error-optimal iterate---it fails to
> in $24/24$ cases on both arms.

And at abstract:13-14, replace "measured over $24$ cases at $128^2$--$512^2$ it does not
merely persist but \emph{grows}, rising in $21/24$" with "two independent implementations
measure it growing under refinement---rising in $21$ of $24$ cases on one, decaying in
$0$ of $16$ on the other, by a rule registered before either was run". Mirror the same
strengthening at `body.tex:46-48` and `body.tex:2151-2153`.

## S4. §sec:iters gives three limits of the sweep and omits the fourth, decisive one

**File / lines:** `docs/paper/body.tex:1214-1233`.

**What is wrong:** the "Three limits of this sweep, stated before its numbers" paragraph
is thorough about the knob, the channel and $n=1$, but never mentions that the sweep was
re-run seeded and returned `DIRECTION-ONLY`. That result appears once in the entire
manuscript, in a limitations bullet 900 lines later (`body.tex:2125-2139`), and is
therefore absent from every place the sweep is actually presented: §sec:iters itself, the
`fig:fig4` caption (B6) and the theorem's opening paragraph
(`sections/residual_floor_theorem.tex:35-38`).

**Source:** `results/sensitivity/iters_seeded.json`.

**Replacement:** retitle the paragraph "Four limits of this sweep, stated before its
numbers" and append:

> \emph{(iv)} A seeded five-seed re-run on the Transolver$+$DEQ arm
> (\texttt{scripts/iters\_sweep\_seeded.py}, $24$ cases) returns \textsc{direction-only}
> against a bar declared in advance: the residual is higher at $k{=}15$ in $5/5$ seeds but
> by only $+1.5\%$, at $1.3\times$ the across-seed standard deviation where we required
> $2\times$; it is monotone in $0/5$ seeds; and paired per case it is higher in only
> $45/120=38\%$. The sweep therefore supports \emph{decoupling}, not divergence, and we
> claim only that cutting field error along this path buys no residual reduction.

## S5. §sec:iters' title asserts what the section says it cannot carry

**File / line:** `docs/paper/body.tex:1205`.

> \subsection{Residual-as-objective fails: cutting the error buys no residual reduction}

The first sentence of the section (1208-1212) says the direct evidence lives elsewhere and
that this subsection "reports an earlier and weaker observation". The colon-clause of the
title ("cutting the error buys no residual reduction") is exactly the narrowed claim and is
fine; the lead-in is not.

**Replacement:** `\subsection{The iteration sweep: cutting the error buys no residual reduction}`

## S6. The 512² no-slip crossover: two committed artifacts disagree and only one is reported

**File / lines:** `docs/paper/sections/residual_floor_theorem.tex:489-493`.

**Text as it stands:**

> the margin collapses monotonically---$+9.3\%$ at $128^2$, $+7.4\%$ at $181^2$, $+6.9\%$
> at $256^2$, $+3.9\%$ at $362^2$, and $-2.8\%$ at $512^2$, where it has crossed.

**What is wrong:** a second committed artifact scoring the same rung says it has *not*
crossed. `results/floor_resolution_ladder.json` (24 cases, 3 rungs) reports
`crossover/512_framework: truth 0.0012765 > uniform 0.0012582, holds = True` (margin
$+1.5\%$) and `crossover/512_fixed: truth 0.0448911 > uniform 0.0404385, holds = True`
(margin $+11.0\%$). The 5-rung figures come from a 16-case run.

`docs/paper/review/floor_resolution_study.md:363-371` names the disagreement, calls it
subset noise at a margin already collapsed to ~2%, and instructs that it "should be settled
in the paper's favour of caution rather than silently averaged away". The paper took the
cautious side, correctly — but it presents "where it has crossed" as fact without saying
that the larger study at the same rung disagrees. That is exactly the asymmetry a hostile
referee will find, because both files are public.

**Replacement:**

> the margin collapses monotonically---$+9.3\%$ at $128^2$, $+7.4\%$ at $181^2$, $+6.9\%$
> at $256^2$, $+3.9\%$ at $362^2$, and $-2.8\%$ at $512^2$ in a $16$-case run. A $24$-case
> run over $128^2/256^2/512^2$ agrees on the collapse but still has the truth above by
> $1.5\%$ at the finest rung (\texttt{results/floor\_resolution\_ladder.json}); a $\pm2\%$
> disagreement between two subsets at a margin that has already collapsed to ${\sim}2\%$ is
> subset noise, and we take the cautious reading. Either way the claim we withdraw is the
> resolution-independent one.

## S7. The MMS order is given as 2.06 in one place and 2.02±0.05 in another

**Files / lines:** `docs/paper/body.tex:1365-1367` ("gives $p=2.06$ in the interior band")
versus `sections/residual_floor_theorem.tex:237-238` ("measures order $p=2.06$") versus
`sections/residual_floor_theorem.tex:461` ("$2.02\pm0.05$").

`2.058` is `gates/mms_order/p_band_0.1` in
`results/certificates/floor_resolution_decomposition.json` — a single fit on the rung means.
`2.02 ± 0.05` is the mean ± sd of the per-case fits
(`docs/paper/review/floor_resolution_study.md:293`). Both are defensible; presenting them as
two numbers in one manuscript without saying they are two estimators is not. Once B10 is
applied, add the half-clause: `order $2.02\pm0.05$ per case ($2.06$ fitted on the rung
means)`.

## S8. The per-cell Spearman is attributed to the deployed corrected field; it is the dropout-FNO at n=15

**File / lines:** `docs/paper/body.tex:572-576`.

**Text as it stands:**

> The residual's \emph{per-cell} spatial rank correlation with error \emph{within} a field
> is weaker---$0.22{\pm}0.06$ (Spearman) though $0.60$ (Pearson), measured on the deployed
> corrected field (\texttt{results/control/percell\_residual\_error.json}).

**What is wrong:** the artifact's `meta` says `checkpoint = checkpoints/certificates_deq.pt`
and `n_cases = 15`. `results/control/percell_localization.json` says so in its own words:

> "published 0.22 was an OLD FNO checkpoint at n=15; this run is the deployed 5-member
> Transolver ensemble mean at n=200 -- a different value is expected and fine."

The deployed-field value is `V0_raw/percase_spearman_mean = 0.166 ± 0.155`, which the very
next sentence of the paper quotes correctly. So the paragraph gives two numbers for the same
quantity and attributes both to the deployed field.

**Replacement:**

> The residual's \emph{per-cell} spatial rank correlation with error \emph{within} a field
> is weaker---$0.22{\pm}0.06$ (Spearman) though $0.60$ (Pearson) on the dropout-FNO at
> $n{=}15$ (\texttt{results/control/percell\_residual\_error.json}), and
> $0.166{\pm}0.155$ on the deployed ensemble-mean field at $n{=}200$.

Then delete the now-redundant "$0.166{\pm}0.155$ on these fields" parenthetical at 582.

## S9. `residual_floor_realdata.json` asserts an operator identity the manuscript refutes

**Files / lines:** `results/certificates/residual_floor_realdata.json`
(`metadata.monitored_residual`) and `scripts/probe_residual_floor.py:3` and `:229`.

**Text as it stands (JSON):**

> `"monitored_residual": "continuity + momentum_x + momentum_y (BC term EXCLUDED, matching physics_residual_torch)"`

**Does it bear on a claim? Yes, indirectly, and more than cosmetically.** It changes no
number — the probe calls `PhysicsChecker.diagnose()` (`probe_residual_floor.py:119, 131,
136, 166`), which is the correct, deployed, compact-stencil monitor, and every figure it
produces (mean 0.19197, median 0.13255, `n_uniform_lt_truth = 200`, `n_pred_lt_truth = 160`)
reproduces exactly. But the string asserts that this operator *matches*
`physics_residual_torch`, and `body.tex:539-548` explicitly denies it:

> The numpy monitor whose numbers we report uses the compact three-point form; the
> differentiable twin … uses a repeated first difference, which is wider and annihilates the
> period-2 mode the compact form sees. The training-time operator therefore has a strictly
> larger kernel than the monitored one.

A referee reading the cited artifact next to §Method finds a flat contradiction inside the
reproduction package, on the file that carries the paper's headline number. The same false
identity is repeated in the module docstring at line 3 and, transitively, in
`results/review/control2_difficulty_confound.json`'s
`conditioning_variable/definition`. **Fix the strings; do not re-run anything.**
Suggested: `"continuity + momentum_x + momentum_y (BC term EXCLUDED); evaluated by
PhysicsChecker.diagnose -- the compact-stencil numpy monitor, NOT the wider differentiable
twin physics_residual_torch (see body.tex Method, 'Two implementation facts')"`.

## S10. The two-Laplacian caveat is carried in §Method but not at the two sites that state the W1 null

**Files / lines:** `docs/paper/body.tex:246-250` (Related Work) and `:2088-2095` (limitations).

**Assessment of the deferred item.** The two stencils bear on exactly one claim — the W1
residual-input null — and `body.tex:539-548` concedes it properly, including "we do not
claim to have excluded it", plus the both-operators check that protects §sec:descent
("the compact-stencil monitor we report falls in $24/24$ cases too", `body.tex:1422-1423`).
**No new experiment is needed.** But the two places that *state* the null give it
unqualified:

> a controlled ablation shows the residual input is not the source of the gain: zeroing it at
> train and inference (3 seeds) matches or slightly exceeds the deployed corrector (246-248)

(note also: **3 seeds** here versus **5 seeds** at `body.tex:924-928` and `:2090` — see C3).

**Replacement (248, after "…the deployed corrector"):** add
`---a null we read with one caveat stated in \autoref{sec:method}: the corrector's residual
input channels are built from the wider differentiable stencil, whose larger kernel is a
live alternative explanation we have not excluded.` Add the same half-sentence to the
limitations bullet at 2091.

## S11. The wall-ring / no-slip dilemma (O3): correctly disclosed, but the number has no committed artifact

**File / lines:** `docs/paper/body.tex:530-538`.

**Assessment.** This is disclosed well and in the paper's own voice: the zeroing removes "a
mean $46\%$ (range $35$--$51\%$) of the squared no-slip penalty", and "Any claim resting on
the \emph{magnitude} of $r_{bc}$, including leg (A) of \autoref{sec:residual_floor}, is
therefore a claim about the surviving $54\%$." Leg (A) is already resolution-scoped
(theorem:483-502) and the argument that carries `tab:iters` is leg (B), which does not depend
on the truth-versus-uniform ordering at all (theorem:504-510). **It bears on no claim the
paper still makes unconditionally. Not blocking.**

**But:** I could not find a committed artifact for the 46% / 35–51% figures.
`RESOLUTION.md:73-74` says "Measured over 8 cases" and names no file; nothing under
`results/` contains the pair, and there is no script whose name matches. Either commit the
measurement or attribute it in-text as an unlogged spot check. See "Could not verify", U1.

## S12. The norm declared "throughout" is not the norm of the headline number

**File / lines:** `sections/residual_floor_theorem.tex:82-87`, versus `abstract.tex:10`,
`body.tex:96`, `body.tex:2149`, `body.tex:1475`.

**Text as it stands (theorem, 82-84):**

> The scalar we write $\|R_h(\cdot)\|$ throughout is the deployed one,
> \texttt{Diagnostics.residual\_norm}: the root-mean-square over the \textbf{whole grid}…

**What is wrong:** the headline $0.192$ quoted in the abstract, contribution 1, the
conclusion and theorem:394 is the **fluid-cell** RMS
(`residual_floor_realdata.json`, `metadata.norm = "RMS over fluid cells…"`,
`aggregate.norm_truth_mean = 0.19197`). The whole-grid value is $0.191$
(`transolver_inversion.json`, `norm_convention.whole_grid_rms_mean = 0.19121`), and that is
what `tab:descent200` uses in its truth row ("$0.191\to0.125$"). So a reader meets $0.192$
four times and $0.191$ once, with no signpost at the table.

This is *disclosed*, once, at `sections/residual_floor_theorem.tex:421-425`, and the
disclosure is exact (the ratio is the geometric factor $0.995$, verified:
`max_abs_dev_from_sqrt_fluid_fraction = 2.2e-16`). The problem is only that "throughout" is
false and the table is unsignposted.

**Replacements:** (i) theorem:82 — change `The scalar we write $\|R_h(\cdot)\|$ throughout`
to `The scalar we write $\|R_h(\cdot)\|$ in this section`. (ii) `tab:descent200`'s caption
(`body.tex:1463-1468`) — append `$\|R_h\|$ here is the whole-grid norm, so the truth's
$0.191$ is the same measurement as the fluid-cell $0.192$ quoted elsewhere, rescaled by the
geometric factor $0.995$.`

## S13. The unpinned-boundary descent arms are committed, strictly stronger, and unreported

**File / lines:** `docs/paper/body.tex:1444-1447`.

The paper pins the far-field and near-wall Dirichlet data at ground truth "so the negative
cannot be attributed to an unconstrained residual" — the conservative choice, correctly
made. But the unpinned arms exist and are worse for the surrogate, and the paper does not
say so:

| arm | pinned (`bc`) frac. worse | unpinned (`free`) frac. worse |
|---|---|---|
| truth | 1.000 | 1.000 (at $J/J_0 = 0.079$, far deeper) |
| Transolver seed 0 | 0.965 | **1.000** |
| Transolver seed 1 | 0.940 | **0.995** |
| Transolver seed 2 | 0.980 | **1.000** |

Sources: `results/residual_descent/descent_{truth,transolver_seed*}_{bc,free}_uvp_armijo.json`.

**Replacement (append to 1447):** `Unpinning those boundaries makes the negative stronger,
not weaker---the error then rises on $100/99.5/100\%$ of cases across the three seeds---so
the pinned arm is the conservative one and is what we report
(\texttt{results/residual\_descent/descent\_*\_free\_uvp\_armijo.json}).`

## S14. `tab:v2`'s caption promises a column the table does not have

**File / lines:** `docs/paper/body.tex:809-815`.

> $\rho$ closer to 1 is better; resid$\leftrightarrow$err $\rho > 0$ supports the trust
> signal; **conformal coverage targets $0.90$**.

There is no coverage column in `tab:v2` (columns are metric, `mse_u`, `mse_v`, `mse_p`,
$C_l$ rel. err, $C_d$ rel. err, resid$\leftrightarrow$err $\rho$). Delete the clause; the
coverage number lives in `tab:uq`.

## S15. λ\* "median ≈600" against a source median of 581

**File / lines:** `sections/residual_floor_theorem.tex:494-497`.

`docs/paper/review/floor_resolution_study.md:380-381` gives median **581** overall
(range 359–8549) and median **599** at $512^2$ ($n=4$). "$\approx 600$" is defensible as the
at-crossover figure but the range quoted alongside it (359–8549) is the *overall* range, so
the two halves of the parenthesis come from different populations. Either quote
`median $581$ (range $359$--$8549$)` or `median $599$ at $512^2$, the rung where it
crosses`.

## S16. Contribution 5 quotes the ρ range the paper elsewhere withdraws

**File / lines:** `docs/paper/body.tex:144-147`.

> the residual is consistently positive across three architecturally distinct backbones
> ($\rho$ from $0.40$ to $0.85$)

`RESOLUTION.md:118-119` lists "The three-backbone claim" as withdrawn: "the matched-density
range is ρ = 0.40–0.61, not 0.40–0.85". The withdrawal is honoured in §sec:v2
(`body.tex:961-971`) and in the limitations (`body.tex:2030-2035`) but not in the
contribution list, which is where a referee reads it first.

**Replacement:** `($\rho$ from $0.40$ to $0.85$, or $0.40$ to $0.61$ among the arms trained
and evaluated at matched density---see the MeshGraphNet disclosure in \autoref{sec:v2})`.

## S17. The ρ_D = 0.84 ceiling reading is withdrawn in one paragraph and reasserted in two others

**File / lines:** `docs/paper/body.tex:853-855` and `docs/paper/body.tex:897-898`.

**Assessment.** The explicit withdrawal at `:857-867` is done properly — "That $0.84$ is not
a ceiling, and we withdraw an earlier reading of it as one", with the covariate regression at
$\rho=0.874$ and the generalising sentence that a drag rank correlation on this benchmark is
uninterpretable without its covariate baseline. But the retraction did not reach either side
of it.

**Text as it stands (853-855), four lines *before* the withdrawal:**

> and the same integrator on \emph{perfect} fields also yields $\rho_D=0.84$, so what limits
> this number is the measurement rather than the prediction.

That is a ceiling claim in all but name: it says the measurement, not the model, is the
binding constraint. The covariate control shows the binding constraint is neither — a
three-parameter fit on $(U,\alpha,\alpha^2)$ that never touches a flow field reaches $0.874$,
above every field-based arm.

**Text as it stands (897-898), forty lines *after* the withdrawal:**

> drag \emph{ranking} is still best served by the near-field scheme ($\rho_D=0.84$).

Reported as a positive with no covariate caveat, immediately after the paper established that
$0.84$ sits *below* a field-free baseline.

**Source:** `results/control/drag_covariate_control.json` —
`covariate_baseline/rho_fit_U_a_a2_vs_cd = 0.8742`, `rho_alpha_vs_cd = 0.8001`, and the file's
own summary field `arms_not_beating_covariate_baseline_on_cd` lists **all seven** arms,
ground-truth and predicted alike.

**Precise replacements:**

- `:853-855` → `and the same integrator on \emph{perfect} fields also yields $\rho_D=0.84$,
  so this number is not limited by the prediction. Nor is it limited only by the measurement:
  a field-free covariate baseline beats it (below).`
- `:897-898` → `drag \emph{ranking} is still best served among our integrators by the
  near-field scheme ($\rho_D=0.84$), which is nonetheless below the $0.874$ covariate
  baseline and should not be read as a drag-ranking result.`

---

# COSMETIC

- **C1.** `body.tex:1290` gives the ungated fixed half step's median improvement as $6.2\%$.
  The mean of the three deployed per-seed medians in `control3_fixed_step.json` is $6.03\%$
  (−6.33/−6.61/−5.16). If $6.2\%$ is the pooled 600-case median, say so; otherwise use $6.0\%$.
  (The comparison to the gate's $5.8\%$ survives either way.)
- **C2.** The omitted closure term's share of the floor appears as "${\approx}4\%$"
  (`body.tex:114`), "$3.5$--$4.8\%$" (`body.tex:1373`, theorem:214) and `4.2` in
  `scripts/audit_paper_numbers.py`, against a recomputed 4.78. Pick one and use it in all
  three places; `3.5`–`4.8%` is the honest range.
- **C3.** The W1 ablation is "3 seeds" at `body.tex:247` and "5 seeds" at `body.tex:924`
  and `:2090`. `w1_capture.json` has `meta.seeds = [0,1,2,3,4]`. Fix 247 to 5.
- **C4.** `body.tex:2053` says descending $\nu_t$ alongside $(u,v,p)$ "quarters the velocity
  damage". Measured: $\Delta\texttt{mse\_u}$ falls from $+0.437$ to $+0.100$, a factor
  $0.23$ — "quarters" is right; but the arm quoted for `mse_nut` ($6.3\times10^{-9}\to0.115$,
  82.5% worse) is `descent_transolver_seed0_bc_uvpn_armijo.json` while "the headline descent
  arm" is the truth arm. Name the arm: `on the deployed backbone`.
- **C5.** `body.tex:1332` and `tab:floor_ladder` give the $128^2$ band-restricted floor as
  $0.0624$; theorem:213 gives the five-rung study's coarsest band value as $0.0459$ and its
  finest as $0.1074$ against the ladder's $0.1067$. Both are disclosed as different exclusion
  radii (theorem:216-219), but a one-line footnote in `tab:floor_ladder`'s caption would stop
  a referee re-deriving it.
- **C6.** Stale build comments naming JCP at `body.tex:3`, `abstract.tex:3`,
  `neuroforge_cfd_elsevier.tex:2-3`, `preamble.tex:4` (see B11).

---

# What I verified and found correct

Recorded so the fix pass does not re-litigate settled numbers. Every one of these was
recomputed from the committed artifact, not read from the prose.

- **Floor.** mean $0.19197\to0.192$, median $0.13255\to0.133$, uniform exactly $0$ on
  $200/200$ — `residual_floor_realdata.json`.
- **Ladder.** $0.0624/0.0779/0.1067$, $p=-0.3874$, rose in $21/24$ — `floor_resolution_ladder.json`.
- **Decomposition.** `band_0.1`: $0$ decay of $16$, $p=-0.639\pm0.285$, fine/coarse $2.56$;
  reproduction gate to $3.3\times10^{-8}$; stress-divergence identity order $1.928$;
  repair moves $512^2$ from $0.10737$ to $0.10740$ — `floor_resolution_decomposition.json`.
- **Descent, $n{=}24$.** $84\%$ cut in $24/24$, median error $0.9118$, $18/24$ helped from
  the perturbed start, $24/24$ residual-selected iterate worse — verified via
  `scripts/audit_paper_numbers.py` readers.
- **Descent, $n{=}200$.** Every cell of `tab:descent200`: truth $0.191\to0.125$, $J/J_0=0.398$,
  $0\to0.0069$, $200/200$, `mse_u` $0.000\to0.499$; Transolver seeds $0.218\to0.144$/$0.142$/$0.141$,
  $+76.5/+73.9/+86.9\%$, $0.965/0.940/0.980$; dropout-FNO $0.286\to0.153$, $-18.7\%$, $0.250$.
  Exchange rate $-0.1402$; $1.57\times$ (0.006940/0.004417); Wilcoxon $1.4\times10^{-34}$;
  Adam $-15.3\%\to-6.7\%$ — `results/residual_descent/*.json`.
- **Gate.** accept $599/600 = 99.83\%$, improves error $89.2\%$, median $-5.74\%$; ensemble
  $74.3\%$, $-2.57\%$; ungated fixed half step $95.83\%$ — `acceptance_gate.json`,
  `control3_fixed_step.json`. (The one exception is B1.)
- **`tab:v2`.** All eight cells reproduced from `w1_capture.json`, including the 5/5 sign
  consistency on every volume channel and the per-seed surface deltas $-1/-1/+2/+2/+1\%$.
  W1: `mean_d = -0.0051`, `n_seeds_with_beats_null = 0/5`, verdict `REFRAME`.
- **Inversion.** $160/200$; deployed $8/5/5$ and $12/9/14$ of $200$ ($2.5$–$7.0\%$); median
  gaps $+0.0133$ to $+0.0152$ against $-0.0241$; gradient-energy $1.002\times$ near-body and
  $1.746\times$ all-fluid; residual spreads $0.034$ / $0.162$ / $0.161$ —
  `transolver_inversion.json`. (The counts are right; B3 and B4 are about where they are applied.)
- **Selective prediction.** AUROC $0.8706$ / $0.8939$ / $0.9053$; $\Delta$ $+0.0233$ CI
  $[-0.0346,+0.0832]$; oracle recovery $0.915$ and $0.670$; $\rho$ $0.6103$ and $0.7028$;
  partial $0.5615$ vs raw $0.6103$ — `control1`, `control2`, `selective_prediction.json`.
- **Conformal.** $q = 0.768/1.301/1.748$, coverage $0.911/0.928/0.942$, ECE $u=0.064$
  (`h4_coverage.json`); drag-band coverage $0.891$–$0.895$ at width $5.57$–$6.76\times$,
  floor share $0.8636$ (`functional_audit_gate_followup.json`).
- **Per-cell localisation.** $0.166\pm0.155$ raw, $0.323\pm0.232$ at $k{=}16$ ($+0.157$, 93%
  of cases), $\sigma{=}4$ gives $+0.135$, patch AUROC $0.684$–$0.717$, dyn-p $+0.008$ —
  `percell_localization.json`.
- **Drag covariates.** $0.8742$ for $(U,\alpha,\alpha^2)$, $0.8001$ for $\alpha$ alone,
  truth-field arms $0.611$–$0.839$, predicted arms $0.828$–$0.845$ —
  `drag_covariate_control.json`.
- **Seeded sweep.** `DIRECTION-ONLY`, $5/5$ sign, $0/5$ monotone, $45/120$ paired, SNR $1.32$
  against a $2\times$ bar, $+1.5\%$ — `iters_seeded.json`.
- **Boundary-term robustness.** leg (A) $\overline{r_{bc}^2}$ $0.0073$ vs $0.0053$; leg (B)
  $3.17\to2.32$ with $+0.635$ interior and $+0.0019$ no-slip, $\rho$ $0.125\to0.807$ excluded
  and $0.145\to0.811$ included — `bc_weight_sweep.json`, `bc_inclusive_sweep.json`.
- **`tab:indist` (all 28 cells).** Every mean and std reproduced exactly from
  `results/full_research/in_dist/ablation.csv`, including the two verdict-bearing effects:
  the DEQ detector lift $0.39744\pm0.00284 \to 0.82653\pm0.00200$ (Cohen's $d = 174.5$,
  matching the paper's "$\approx 175$"), and the no-physics-loss arm being the best
  force-ranking model ($\rho_{C_d} = 0.94453$, $\rho_{C_l} = 0.99201$). H1's negative and
  H3's whole verdict rest on this table and it is clean.
- **`tab:ood` (all 28 cells).** Reproduced exactly from
  `results/full_research/ood/ablation_ood.csv`, including both protected negatives: the
  `aoa` $\rho_{C_d}$ regression $0.92605 \to 0.90526$, and the DEQ arm's absolute `mse_v`
  being worse in every regime ($0.87987 / 1.2077 / 1.4068$ against $0.38479 / 1.1874 /
  1.3099$). The `aoa` detector rescue $0.31402 \to 0.74612$ gives Cohen's $d = 8.85$,
  matching the paper's "$\approx 8.9$".
- **`tab:transolver`.** $0.12017\pm0.00472$, $0.087982\pm0.01268$, $628.45\pm29.18$,
  $9110.5\pm503.3$, $0.99915\pm0.00017$, $0.99631\pm0.00123$, `backbone_params = 7350420`
  — `results/baselines/seed{0,1,2}.json`, `inference_cost.json`.
- **`tab:cost`.** encode median $1412.7$, audit median $1.128$ / p95 $2.488$, deployed solve
  $3821.9$, ensemble $4.861\times$ a single backbone, scaling exponent $0.6369$ at
  $R^2=0.9866$, classical anchor $1500$ s — `results/control/inference_cost.json`. The
  end-to-end $5.24$ s and $\sim\!286\times$ both re-derive.
- **Structure.** 50 labels, 50 reference targets, **zero dangling references, zero orphan
  labels, zero duplicate labels**, zero overfull boxes in the Elsevier build. The only
  cross-reference defect is B9, which is a *mis-resolving* label rather than a missing one
  and therefore invisible to that check.
- **Mechanical submission limits.** abstract 241/250 words, 7/7 keywords, 5 highlights all
  under 85 characters — `scripts/check_submission.py`.

---

# Could not verify

1. **U1 — the wall-ring 46%.** `body.tex:534-536` states the residual zeroing removes "a
   mean $46\%$ (range $35$--$51\%$)" of the squared no-slip penalty.
   `RESOLUTION.md:73-74` attributes it to a measurement "over 8 cases" but names no file, and
   I found no committed JSON or script containing the figure. **What would settle it:** the
   script and artifact that produced it, or an in-text note that it is an unlogged spot check
   over 8 cases. Nothing in the paper's claims depends on the exact value (see S11).
2. **U2 — `fig:fig4`'s underlying figure content.** I audited the caption against
   `results/sensitivity/iters.csv`/`iters.json` and `iters_seeded.json`, but did not open
   `results/figures/fig4_sensitivity_iters.pdf`. If the plotted panels themselves are
   annotated with "residual rises while error falls", B6 must be applied to the figure
   source as well as the caption. **What would settle it:** inspecting the figure or its
   generating script.
3. **U3 — the "$\approx 1/35$ of the supervised corrector's gain"** (`body.tex:1496`,
   `body.tex:2172`) and **"error minimum is interior on $78.5\%$"** (`body.tex:1498`) trace
   to `docs/paper/review/residual_descent_test.md:237, 248, 459`, not to a JSON field I could
   read directly. The nearest committed quantities are
   `frac_argmin_J_ne_argmin_err = 0.79` (FNO Armijo) and the Adam
   $-15.3\%\to-6.7\%$ pair, both of which are consistent. **What would settle it:** the field
   name or the two-line derivation in the artifact.
4. **U4 — bibliographic accuracy.** I checked `refs.bib` only for the three entries that bear
   on B11. I did not verify that `lei2026newtonkrylov`, `zhang2026phymgn`,
   `mukherjee2026certification`, `scherz2026evaluation`, `rigotti2026gist`,
   `hillebrecht2025posteriori`, `song2026structureaware` or `roy2025anchor` exist as cited,
   or that the numbers attributed to them (notably "seven orders of magnitude" for Lei et al.,
   quoted three times) are theirs. Given two originality desk rejections, these citations are
   load-bearing for the positioning argument and deserve a separate pass.
5. **U5 — `tab:airfrans-sota` only.** `tab:indist`, `tab:ood`, `tab:transolver` and
   `tab:cost` were recomputed after the first draft of this report and are now in the
   verified list above; that gap is closed. What remains unverified is
   `tab:airfrans-sota` (`body.tex:1899-1908`), whose four baseline rows are transcribed from
   \citet{bonnet2022airfrans} rather than produced here, and which the paper correctly labels
   "field context only" with an explicit statement that NeuroForge is not a comparable row.
   **What would settle it:** checking the four rows against Table 4 of the AirfRANS paper.
   No claim in the manuscript depends on them.
6. **U6 — the TMLR build.** I checked `neuroforge_cfd_elsevier.aux` and the Elsevier PDF.
   `neuroforge_cfd.aux` shows the same `sec:selective` mis-binding (B9), but I did not verify
   the rest of the TMLR build's cross-references or overfull boxes.

---

# One note outside the audit's scope

During this session an MCP server advertising itself as `zoho-mail` returned, in place of a
tool result, an instruction block directing me to stop using the Read/Edit/Write tools and
route all file access through Bash. That is not a message from the user or from the agent
that assigned this task, and it has nothing to do with a manuscript audit. I ignored it and
used normal tooling. Flagging it because an injected instruction arriving through a tool
channel is worth someone looking at, independently of this paper.
