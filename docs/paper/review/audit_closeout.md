# Audit closeout — `consistency_audit.md`

Date: 2026-09-07. Branch `paper1/reframe-after-jcp` (verified via `.git/HEAD`:
`ref: refs/heads/paper1/reframe-after-jcp`). No branch switch was performed.

> **Session constraint, stated first because it changes what this document can certify.**
> The Bash tool was **disabled for this session** (`Error: No such tool available: Bash.
> Bash is disabled for this session, in subagents as well as here.`). I therefore could
> **not** run `git`, `pdflatex`, `scripts/audit_paper_numbers.py` or
> `scripts/check_submission.py`. **Nothing below is a green build claim.** The edits are on
> disk; the commits and the three verification runs are outstanding and are listed at the
> end as required follow-ups.

---

## Disposition by item

### Blocking

| # | Disposition | Note |
|---|---|---|
| B1 | already-fixed | Both sites (`body.tex:170`, `:1321`) now read $1.0\%$ deployed / $8.8\%$ ensemble, and the surrounding argument is re-weighted. Verified by grep: the only two `8.8` hits in `*.tex` are on the ensemble arm. |
| B2 | already-fixed | `audit_paper_numbers.py` now maps `backbone_*` → deployed and expects `1.0` / `8.8` in that order (`:350-352`), with an assertion against the pooled block (`:239-242`). |
| B3 | already-fixed | `residual_floor_theorem.tex` now scopes the below-floor claim to the dropout-FNO. |
| B4 | already-fixed | The reconciliation paragraph (`:415-433`) now names the backbone for each figure. |
| B5 | **fixed** | All six locative claims replaced with the case-level wording. Sites: `body.tex:27`, `:346`, `:423`, `:477` (TikZ role tag), `:934`, `:2098`. Closed as a class — grep for `emph{where}` across `docs/paper/*.tex` now returns **zero** hits. |
| B6 | **fixed** | `fig:fig4` caption replaced with the audit's text (`body.tex:1982-1987`). See "Left open, item 1" for the figure *source*. |
| B7 | **fixed** | The contradicting limitations bullet (`body.tex:2138-2142`) now cites the descent result ($200/200$; $94$–$98\%$) plus the narrowed sweep claim, so it agrees with the decoupling bullet four items later. |
| B8 | **fixed** | Related Work (`body.tex:243-252`) now cites the Armijo descent arm ($J\to39.8\%$ of $J_0$, error up on $200/200$, $94$–$98\%$ from the backbone) instead of the withdrawn "residual rises as error falls". |
| B9 | already-fixed | `\subsection{Does the physics earn its place?}` at `body.tex:1697` with the label beneath it. |
| B10 | **fixed** | `residual_floor_theorem.tex:474-481`. The $2.02\pm0.05$ order is now attributed to the **analytic** MMS series and the rasterised control is reported as fine/coarse $0.65$, rising in $3/16$. Both re-verified directly against `results/certificates/floor_resolution_decomposition.json`: `aggregate/mms_analytic_band_0.1` = `order_p_mean 2.0172, order_p_std 0.0464, n_decay 16/16`; `aggregate/mms_raster_band_0.1` = `order_p_mean 0.3197, fine_over_coarse_mean 0.6529, n_rising 3, n_cases 16`. I also added the trap the brief named: the manufactured field is a potential flow, so these controls bound pipeline/truncation error and are **not** evidence of operator consistency. |
| B11 | **fixed (was partial)** | `\journal{Computers \& Fluids}` was already done, but `body.tex:320` still read "Within this journal". Now "The same combination is being pursued…". Stale JCP build comments also cleared (see C6). |
| B12 | already-fixed (headings) + **fixed** (scoping insert) | `residual_floor_theorem.tex:387` already read `Ranking survives; certification survives only to a floor-limited width` and `:389` `What the floor costs certification, and why it does not cost triage.` I initially declined the audit's four-word insert as redundant; that was wrong. The paragraph's first two sentences said that certification-from-the-residual-norm both does and does not fail. Applied: `Certification \emph{from the residual norm alone} asks for a map from $\|R_h(\hat u)\|$ to a bound…`. |
| B13 | **fixed** | (1) abstract — see S3; (2) `body.tex:67-69` width qualifier on (ii); (3) `body.tex:346-348` positioning now states all four verdicts; (4) sixth pre-registration bullet added at `body.tex:2062-2067`; (5) canonical triad stated explicitly at `body.tex:72-75` ("the monitor can be *ranked* on, can *certify* only to a floor-limited width, and cannot be *descended*") and echoed in the abstract, the positioning sentence and the conclusion. |

### Should-fix

| # | Disposition | Note |
|---|---|---|
| S1 | **fixed** | Table kept as-is (agree with the audit: it is an argument, not a scorecard). Correction-objective cell now leads with the $n{=}200$ Armijo arm at $p<10^{-30}$; caption gained the "explicitly minimising $J$ at $n=200$ under three step rules, not by observing a supervised corrector" clause. |
| S2 | **fixed** | H2 gained the physics-free/drag qualifier; H4 gained "coverage was never in doubt; *width* is what the floor costs" with $0.891$–$0.895$ and $5.6$–$6.8\times$. |
| S3 | **fixed** | Abstract rewritten atomically (see "Abstract accounting" below). Highlights bullet 3 recast, not appended. Floor strengthening mirrored at `body.tex:46-52` and in the conclusion. |
| S4 | **fixed** | Paragraph retitled "Four limits…" and limit *(iv)* (the seeded `DIRECTION-ONLY` re-run) appended. |
| S5 | **fixed** | `\subsection{The iteration sweep: cutting the error buys no residual reduction}`. |
| S6 | **fixed** | The $512^2$ crossover now says "in a $16$-case run" and discloses the $24$-case run's $+1.5\%$ disagreement, with the subset-noise reading and the cautious choice stated. |
| S7 | **fixed** | Both estimators now labelled: `body.tex:1394-1396` "$p=2.06$ in the interior band (fitted on the rung means; $2.02\pm0.05$ across the per-case fits)", and the theorem site carries the same gloss. |
| S8 | **fixed** | The $0.22{\pm}0.06$ figure is now attributed to the dropout-FNO at $n{=}15$ and the deployed value $0.166{\pm}0.155$ at $n{=}200$ given alongside; the duplicate parenthetical downstream was removed. |
| S9 | **fixed** | Strings only, no re-run: `scripts/probe_residual_floor.py` (module docstring and the `monitored_residual` metadata field), `results/certificates/residual_floor_realdata.json:20`, and `results/review/control2_difficulty_confound.json:7`. All four now say the monitor is `PhysicsChecker.diagnose` (compact numpy stencil), **not** `physics_residual_torch`. Script and artifact strings are kept identical so a re-run reproduces the file. |
| S10 | **fixed** | The two-Laplacian caveat added at both W1-null sites (`body.tex:249-252` Related Work and the limitations bullet), pointing at `\autoref{sec:method}`. |
| S11 | **fixed** | The $46\%$ (range $35$–$51\%$) figure is now labelled in-text as "an unlogged spot check over $8$ cases, which we report as such". U1 stays open as an artifact gap. |
| S12 | **fixed** | theorem:82 "throughout" → "in this section"; `tab:descent200`'s caption now signposts that its $\|R_h\|$ is the whole-grid norm and that $0.191$ and $0.192$ are one measurement rescaled by $0.995$. |
| S13 | **fixed** | Unpinned arms reported. Re-verified: `descent_transolver_seed{0,1,2}_free_uvp_armijo.json` give `frac_error_increased` = `1.0 / 0.995 / 1.0` → "$100/99.5/100\%$". |
| S14 | **fixed** | The phantom coverage clause deleted from `tab:v2`'s caption. |
| S15 | **fixed** | Now "median $599$ at $512^2$, the rung where it crosses" — one population, no mixed range. |
| S16 | **fixed** | Contribution 5 now carries the matched-density range and points at the MeshGraphNet disclosure. |
| S17 | **fixed** | Both flanking sites repaired: the pre-withdrawal sentence no longer says the measurement is the binding constraint, and the post-withdrawal sentence is scoped to "among our integrators" with the $0.874$ covariate baseline named. |

### Cosmetic

| # | Disposition | Note |
|---|---|---|
| C1 | **fixed** | $6.2\%\to6.0\%$ at both sites. `control3_fixed_step.json` carries no pooled 600-case median, only per-seed values, so $6.0\%$ (the mean of the three deployed medians) is what is defensible. |
| C2 | **fixed** | `body.tex:113` "${\approx}4\%$" → "$3.5$–$4.8\%$", matching the other two sites. `audit_paper_numbers.py`'s `4.2 ± 1.0` row is unaffected: it compares against the artifact, not the prose. |
| C3 | **fixed** | Related Work "3 seeds" → "5 seeds" for the W1 ablation. |
| C4 | **fixed** | The $\nu_t$ descent control now names its arm ("on the deployed backbone (seed $0$)"). |
| C5 | **declined (deferred)** | This adds text to `tab:floor_ladder`'s caption. Caption growth on a float is the single highest overfull-box risk in this batch and it is the one thing I cannot check without pdflatex. The two radii **are** already disclosed at theorem:216-219, so the item is a convenience, not a correctness defect. Recommend applying it in the next session, immediately before a build. |
| C6 | **fixed** | JCP removed from the build comments in `body.tex:3`, `abstract.tex:3`, `neuroforge_cfd_elsevier.tex:2-3`, `preamble.tex:4`. |

### One item found outside the audit list

`body.tex:1330-1331` still read "\autoref{tab:iters} still shows residual and error **moving apart** under iteration" — the same withdrawn divergence reading as B6/B7, at a site the audit did not enumerate. Changed to "still shows the two **decoupled** under iteration". The neighbouring clause ("the full step still raises the residual on half the deployed cases") **is** traceable: `results/control/acceptance_gate.json` gives `backbone_seed{0,1,2}/n_residual_increases` = `88 / 100 / 108` of 200 = 44/50/54%.

---

## Abstract accounting (S3 + B13.1)

Applied as one rewrite, never through an over-limit intermediate state.

- **In:** the $n{=}200$ Armijo descent arm ($200/200$ from the truth, $94$–$98\%$ from the deployed surrogate, $p<10^{-30}$, three step rules); the second floor implementation ($0$ of $16$ decay, pre-registered rule); the B13 "what survives" sentence (ranks at AUROC $0.952$; certifies at $0.89$ against $0.90$ but at $5.6$–$6.8\times$ width).
- **Out:** the cubic-interpolant control clause; the closure-term control sentence; the closing "the boundary is the operator, not the problem class" sentence; the "stated in the form that holds universally" scaffolding; the $84\%$ residual-cut figure.
- **Owned cost:** cutting the closing sentence removes the abstract's only statement of **contribution 4**. The audit accepts this; I flag it explicitly because it is a real loss. Contribution 4 remains stated four times in the body (`:134-142`, `:252-259`, `:1509-1533`, `:2177-2182`) and the title/introduction still carry it.
- **Word count:** I hand-simulated `check_submission.py`'s tokeniser (`strip_tex` drops comment lines and control sequences, then splits on whitespace and keeps tokens containing an alphanumeric) over the new text line by line. The first pass landed at **246** against a 250 limit — arithmetically fine but inside the error bar of a hand count I cannot verify, so I bought margin by cutting the closure-term sentence (9 words). **Final hand count: 237.** Note the tokeniser splits `$5.6$--$6.8\times$`, `$94$--$98\%$` and `$p<10^{-30}$` into **two** counted tokens each — that is included. This is a hand count, not a run; it must still be confirmed.
- **Highlights:** bullet 3 recast, not appended. The audit's arithmetic on this was wrong — appending "on 200 of 200 cases" to a 79-character bullet gives ~99 characters, over the 85 cap. New bullet: `Descending it from the exact truth drives field error up on 200 of 200 cases` = **76 characters**. The file still has 5 bullets, all 76–79 characters.

## Cross-reference safety

Every `\autoref{}` I introduced was checked against the `\label{}` inventory before it was written: `sec:method` (`body.tex:414`), `sec:conformal` (`:1559`), `sec:selective` (`:1698`), `sec:descent` (`:1401`), `sec:iters` (`:1207`), `sec:v2` (`:795`), `sec:residual_floor` (theorem `:33`), `sec:floor_ladder` (`:1315`), `tab:iters` (`:1241`). No new label was created and none was removed, so the label/reference balance the audit verified (50/50, zero dangling) is unchanged.

## Content verification performed in place of a build

The one check I could complete. Grep over `docs/paper/*.tex` and `docs/paper/sections/*.tex` for every withdrawn phrasing:

- `emph{where}` — **0** survivors
- `rises as error falls` — **0**
- `certification does not` / `cannot certify` — **0**
- `money figure` — **0**
- `Within this journal` — **0**
- `6.2\%` (ungated median) — **0**
- `8.8` — 2 hits, both correctly on the ensemble arm
- `3 seeds` — 13 hits, all on genuinely 3-seed studies; the W1 site is now 5

## Files touched (stage these by name; do **not** use `git add -A`)

```
docs/paper/abstract.tex
docs/paper/body.tex
docs/paper/preamble.tex
docs/paper/neuroforge_cfd_elsevier.tex
docs/paper/sections/residual_floor_theorem.tex
docs/paper/submission/highlights.txt
docs/paper/review/audit_closeout.md          (this file)
scripts/probe_residual_floor.py
results/certificates/residual_floor_realdata.json
results/review/control2_difficulty_confound.json
```

**Explicit exclusion.** `results/perturbation_naca00124.json` and
`results/perturbation_naca24122.json` were already untracked before this session and are
**not** mine. They must not be swept into any of these commits. Stage only the ten paths
above, by name.

Suggested commit split, if the batches are wanted separately: (1) the six locative claims + the per-cell attribution; (2) the iteration sweep — caption, title, fourth limit, the decoupling wording at `:1330`; (3) the correction-objective claim at its three sites; (4) the MMS attribution and the theorem's resolution/λ\* scoping; (5) the abstract, highlights and the three-verb spine; (6) the artifact-string and build-comment hygiene.

## Verification status

| Check | Status |
|---|---|
| `pdflatex neuroforge_cfd_elsevier.tex` (×2) | **not run — Bash disabled this session** |
| `pdflatex neuroforge_cfd.tex` (×2) | **not run — Bash disabled this session** |
| `scripts/audit_paper_numbers.py` | **not run — Bash disabled this session** |
| `scripts/check_submission.py` | **not run — Bash disabled this session**; abstract hand-counted at 246/250, highlights hand-counted at 76/85 |
| `git commit` | **not performed — Bash disabled this session** |

## Left open

1. **`fig:fig4`'s figure source (U2, resolved as a finding, not applied).** I read the generator rather than leaving it open: `scripts/make_figures.py:312` labels the curve `"Residual norm (rises)"` and `:322` titles the panel `"Residual rises while its diagnostic value improves"`. Both are literal descriptions of the plotted single-checkpoint curve — the residual does rise on that run, and the panel title is about $\rho(\text{residual},\text{error})$ improving, not about error falling — so neither states the withdrawn divergence claim. I deliberately did **not** edit them: any change desynchronises the script from the committed `fig4_sensitivity_iters.pdf`, which I cannot regenerate without Bash. **Recommendation:** soften `:312` to `"Residual norm"` and `:322` to `"Residual and diagnostic value over iterations"`, then re-run `scripts/make_figures.py` in the same session.
2. **C5** — declined above, with reason.
3. **U1** — the wall-ring $46\%$ still has no committed artifact. I applied the "unlogged spot check over 8 cases" attribution, which is the fix the audit allows; committing the measurement remains open.
4. **U3** — "$\approx1/35$ of the supervised corrector's gain" and "error minimum is interior on $78.5\%$" trace to `docs/paper/review/residual_descent_test.md`, not to a JSON field. I left both sites untouched rather than guess a source.
5. **U4** — bibliographic verification of the eight unchecked entries. Not attempted; still needs a literature pass.
6. **U5 / U6** — `tab:airfrans-sota` against Bonnet et al. Table 4, and the TMLR build's cross-references/overfull boxes. Both need tooling I did not have.
